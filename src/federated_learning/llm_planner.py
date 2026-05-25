"""
LLM决策规划器 (LLM Planner)
用于场景C阶段3: 接入LLM API进行联邦学习策略决策

本模块实现:
- LLMClient: LLM API客户端封装（支持DeepSeek API）
- LLMPlanner: LLM决策规划器，构造prompt并解析响应
"""

import json
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests
import threading
from queue import Queue, Empty


class LLMClient:
    """
    LLM API客户端

    封装对LLM API的调用，支持DeepSeek API（兼容OpenAI格式）
    包含重试机制、连接复用和代理自动检测
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        初始化LLM客户端

        Args:
            api_key: API密钥
            model_name: 模型名称（如 deepseek-chat, deepseek-coder）
            base_url: API基础URL
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

        # API端点
        self.chat_endpoint = f"{self.base_url}/chat/completions"

        # 请求头
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 使用 Session 复用连接（TCP连接池）
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # 自动检测系统代理
        proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') \
            or os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        if proxy:
            self.session.proxies = {'http': proxy, 'https': proxy}
            print(f"  [LLMClient] 检测到系统代理: {proxy}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        调用LLM生成响应（含重试机制）

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）
            max_tokens: 最大输出token数
            temperature: 采样温度

        Returns:
            LLM生成的文本响应
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._do_request(request_payload)
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                raise ValueError(f"Unexpected API response format: {result}")
            except Exception as e:
                last_error = e
                error_str = str(e)
                # WinError 10013 特殊提示
                if "10013" in error_str:
                    print(f"  [LLMClient] 第{attempt}次尝试失败: Windows套接字权限被拒绝 (WinError 10013)")
                    print(f"  [LLMClient] 可能原因: 防火墙/杀毒软件拦截了HTTPS请求")
                    print(f"  [LLMClient] 建议: 1) 检查Windows防火墙设置 2) 临时关闭杀毒软件 3) 以管理员权限运行")
                else:
                    print(f"  [LLMClient] 第{attempt}次尝试失败: {e}")

                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # 指数退避: 2s, 4s, 8s
                    print(f"  [LLMClient] {wait_time}秒后重试...")
                    time.sleep(wait_time)

        raise RuntimeError(
            f"LLM API 在{self.max_retries}次尝试后仍然失败: {last_error}"
        )

    def _do_request(self, payload: dict) -> dict:
        """执行单次API请求（线程隔离，防止挂起）"""
        result_queue: Queue = Queue(maxsize=1)

        def _request_worker():
            try:
                response = self.session.post(
                    self.chat_endpoint,
                    json=payload,
                    timeout=(10, self.timeout)
                )
                response.raise_for_status()
                result_queue.put(("ok", response.json()))
            except Exception as e:
                result_queue.put(("err", e))

        worker = threading.Thread(target=_request_worker, daemon=True)
        worker.start()

        hard_timeout_seconds = self.timeout + 5
        worker.join(timeout=hard_timeout_seconds)
        if worker.is_alive():
            raise RuntimeError(
                f"LLM API请求超时 ({hard_timeout_seconds}s)"
            )

        try:
            status, payload_or_error = result_queue.get_nowait()
        except Empty:
            raise RuntimeError("LLM API请求返回空结果")

        if status == "err":
            raise RuntimeError(f"LLM API请求失败: {payload_or_error}")

        return payload_or_error


class LLMPlanner:
    """
    LLM决策规划器
    
    职责:
    - 构造prompt（包含训练历史和候选策略）
    - 调用LLM获取决策
    - 解析LLM响应为可执行的策略参数
    - 记录LLM决策日志
    """
    
    def __init__(
        self,
        config: dict,
        llm_client: LLMClient,
        log_dir: str = "results/logs"
    ):
        """
        初始化LLM规划器
        
        Args:
            config: 配置字典
            llm_client: LLM客户端实例
            log_dir: 日志保存目录
        """
        self.config = config
        self.llm_client = llm_client
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 从配置中读取参数
        scene_c_config = config.get('scene_c', {})
        llm_config = scene_c_config.get('llm', {})
        
        self.recent_rounds = llm_config.get('recent_rounds', 10)  # 增加到10轮历史
        self.max_tokens = llm_config.get('max_tokens', 1024)
        self.temperature = llm_config.get('temperature', 0.8)  # 稍高温度增加多样性
        
        # 候选策略列表
        self.candidate_strategies = self._get_candidate_strategies(scene_c_config)
        
        # 日志文件
        self.decisions_log_path = self.log_dir / "scene_C_llm_decisions.jsonl"
        self.decisions_log_path.write_text("", encoding="utf-8")
        
        # 记录上一轮LLM的决策，用于反馈
        self.last_llm_decision = None
    
    def _get_candidate_strategies(self, scene_c_config: dict) -> List[Dict]:
        """
        从配置中获取候选策略列表
        
        Args:
            scene_c_config: scene_c配置段
            
        Returns:
            候选策略列表
        """
        strategies = scene_c_config.get('strategies', [])
        if not strategies:
            # 默认策略
            strategies = [
                {"name": "size_only", "description": "按样本数加权"},
                {"name": "perf_only", "description": "按性能加权"},
                {"name": "hybrid", "description": "混合加权", "lambda_hybrid": 0.5},
                {"name": "fairness_clip", "description": "公平性约束", "lambda_hybrid": 0.3, "alpha_min": 0.1}
            ]
        return strategies
    
    def build_prompt(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int = 20
    ) -> str:
        """
        构造LLM提示词（增强版）

        Args:
            history_round_metrics: 训练历史指标列表
            current_round: 当前轮次
            num_rounds: 总轮数

        Returns:
            构造好的prompt字符串
        """
        # 获取最近K轮的历史
        recent_history = history_round_metrics[-self.recent_rounds:] if len(history_round_metrics) >= self.recent_rounds else history_round_metrics
        
        # 构造历史数据摘要
        history_summary = []
        for record in recent_history:
            round_info = {
                "round": record["round"] + 1,
                "strategy": record["strategy_name"],
                "global_val_mape": f"{record['global_val']['mape']*100:.2f}%",
                "global_val_rmse": f"${record['global_val']['rmse']:,.0f}",
                "clients": {}
            }
            for cid, metrics in record["client_metrics"].items():
                round_info["clients"][cid] = {
                    "val_mape": f"{metrics['val_mape']*100:.2f}%",
                    "n_samples": metrics["n_samples"]
                }
            history_summary.append(round_info)
        
        # 计算历史最佳成绩和趋势分析
        all_mapes = [r['global_val']['mape'] for r in history_round_metrics]
        best_mape = min(all_mapes)
        best_round = all_mapes.index(best_mape) + 1
        current_mape = all_mapes[-1] if all_mapes else 1.0
        
        # 计算首轮MAPE作为初始基准
        initial_mape = all_mapes[0] if all_mapes else 1.0
        improvement_from_start = initial_mape - current_mape
        
        # 计算最近5轮的改善趋势
        trend_info = ""
        if len(all_mapes) >= 5:
            recent_5 = all_mapes[-5:]
            improvement = recent_5[0] - recent_5[-1]
            if improvement > 0.01:
                trend_info = f"最近5轮MAPE改善了{improvement*100:.2f}%，保持良好势头"
            elif improvement > -0.01:
                trend_info = f"最近5轮MAPE基本持平，需要尝试新策略"
            else:
                trend_info = f"最近5轮MAPE恶化了{-improvement*100:.2f}%，需要调整方向"
        
        # 统计各策略的使用次数和平均效果
        strategy_stats = {}
        for r in history_round_metrics:
            s_name = r['strategy_name']
            if s_name not in strategy_stats:
                strategy_stats[s_name] = {'count': 0, 'mapes': []}
            strategy_stats[s_name]['count'] += 1
            strategy_stats[s_name]['mapes'].append(r['global_val']['mape'])
        
        strategy_analysis = []
        for s_name, stats in strategy_stats.items():
            avg_mape = sum(stats['mapes']) / len(stats['mapes'])
            strategy_analysis.append(f"- {s_name}: 使用{stats['count']}次，平均MAPE={avg_mape*100:.2f}%")
        
        # 构造上一轮决策反馈
        feedback_section = ""
        if self.last_llm_decision and len(history_round_metrics) >= 2:
            last_decision = self.last_llm_decision
            prev_mape = history_round_metrics[-2]['global_val']['mape']
            curr_mape = history_round_metrics[-1]['global_val']['mape']
            mape_change = curr_mape - prev_mape
            
            if mape_change < -0.005:
                effect = f"效果良好，MAPE从{prev_mape*100:.2f}%降至{curr_mape*100:.2f}%（改善{-mape_change*100:.2f}%）"
            elif mape_change > 0.005:
                effect = f"效果不佳，MAPE从{prev_mape*100:.2f}%升至{curr_mape*100:.2f}%（恶化{mape_change*100:.2f}%）"
            else:
                effect = f"效果一般，MAPE基本持平（{prev_mape*100:.2f}% -> {curr_mape*100:.2f}%）"
            
            feedback_section = f"""
## 上一轮决策反馈
- 你上轮选择了策略: {last_decision.get('chosen_strategy_name', 'N/A')}
- lr_scale={last_decision.get('lr_scale', 1.0)}, epoch_delta={last_decision.get('epoch_delta', 0)}
- {effect}
- 你的决策理由是: "{last_decision.get('reasoning', 'N/A')[:100]}..."
请根据这个反馈调整你的下一步决策。
"""
        
        # 构造候选策略说明
        strategies_desc = """策略详细说明和适用场景：

1. **size_only** (标准FedAvg)
   - 原理：权重完全按样本数分配，数据多的客户端贡献大
   - 适用：训练初期、各客户端数据质量相近时
   - 公司A(274样本)约占50%，公司B(107样本)约占20%，公司C(160样本)约占30%

2. **perf_only** (性能驱动)
   - 原理：权重与1/val_mape成正比，验证MAPE低的客户端权重高
   - 适用：各客户端数据质量差异大时，让"专家"主导聚合
   - 风险：可能导致小数据客户端被忽视

3. **hybrid** (混合策略, lambda=0.5)
   - 原理：50%考虑样本数，50%考虑性能
   - 适用：需要平衡数据量和性能贡献时
   - 推荐：中后期使用，兼顾稳定性和性能优化

4. **fairness_clip** (公平性约束, alpha_min=0.1, alpha_max=0.6)
   - 原理：在hybrid基础上限制权重范围[0.1, 0.6]
   - 适用：防止单一客户端主导、需要公平参与时
   - 特点：保护小客户端的贡献不被完全忽视"""
        
        prompt = f"""你是一个联邦学习优化专家。你需要为工程造价预测的联邦学习系统选择最佳的聚合策略。

## 任务背景
这是一个横向联邦学习系统，有3个客户端（公司A、B、C）共同训练一个神经网络模型来预测高速公路工程造价。
目标是最小化全局验证集的MAPE（平均绝对百分比误差），同时尽量保持各客户端的公平性。

## 当前训练状态
- **历史最佳成绩**: MAPE = {best_mape*100:.2f}% (第{best_round}轮)
- **当前MAPE**: {current_mape*100:.2f}%
- **累计改善**: {improvement_from_start*100:+.2f}%（相比第1轮）

{trend_info}
{feedback_section}
## 最近{len(recent_history)}轮训练历史
```json
{json.dumps(history_summary, indent=2, ensure_ascii=False)}
```

## 历史策略效果统计
{chr(10).join(strategy_analysis) if strategy_analysis else "暂无足够数据"}

## 可选的聚合策略
{strategies_desc}

## 当前状态
- 当前轮次: {current_round + 1}
- 总轮数: {num_rounds}
- 剩余轮数: {num_rounds - current_round}

## 决策指导原则
1. 如果当前MAPE仍在稳定下降，保持当前策略继续优化
2. 如果MAPE停滞不前（最近5轮无改善），尝试不同策略
3. 早期（1-10轮）可多尝试不同策略积累经验
4. 中期（11-20轮）选择效果最好的策略深入优化
5. 后期（21-30轮）保持最优策略，微调学习率

## 请你决策
请分析训练历史和上述信息，选择下一轮应该使用的策略。

请严格按照以下JSON格式输出你的决策：
```json
{{
  "chosen_strategy_name": "策略名称(size_only/perf_only/hybrid/fairness_clip)",
  "lr_scale": 学习率缩放因子(0.5-2.0之间的数字，1.0表示不变),
  "epoch_delta": 本地epoch调整量(-5到+5之间的整数，0表示不变),
  "reasoning": "详细解释你的决策理由（包括：当前问题分析、策略选择原因、预期效果）"
}}
```

注意：只输出JSON，不要有其他内容。"""
        
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析LLM响应
        
        Args:
            response: LLM原始响应文本
            
        Returns:
            解析后的决策字典
        """
        # 尝试从响应中提取JSON
        try:
            # 查找JSON块
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                decision = json.loads(json_str)
                
                # 验证必要字段
                required_fields = ["chosen_strategy_name"]
                for field in required_fields:
                    if field not in decision:
                        raise ValueError(f"Missing required field: {field}")
                
                # 设置默认值
                decision.setdefault("lr_scale", 1.0)
                decision.setdefault("epoch_delta", 0)
                decision.setdefault("reasoning", "")
                
                # 验证策略名称有效性
                valid_strategies = [s["name"] for s in self.candidate_strategies]
                if decision["chosen_strategy_name"] not in valid_strategies:
                    print(f"  [Warning] Invalid strategy '{decision['chosen_strategy_name']}', using 'size_only'")
                    decision["chosen_strategy_name"] = "size_only"
                
                # 限制lr_scale范围
                decision["lr_scale"] = max(0.5, min(2.0, float(decision["lr_scale"])))
                
                # 限制epoch_delta范围
                decision["epoch_delta"] = max(-5, min(5, int(decision["epoch_delta"])))

                return decision
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [Warning] Failed to parse LLM response: {e}")
            print(f"  [Warning] Using default strategy: size_only")
            return {
                "chosen_strategy_name": "size_only",
                "lr_scale": 1.0,
                "epoch_delta": 0,
                "reasoning": f"Parse error: {e}"
            }
    
    def choose_strategy(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int = 20
    ) -> Dict[str, Any]:
        """
        调用LLM选择策略

        Args:
            history_round_metrics: 训练历史指标列表
            current_round: 当前轮次
            num_rounds: 总轮数

        Returns:
            决策字典，包含:
            - chosen_strategy_name: 选择的策略名称
            - lr_scale: 学习率缩放因子
            - epoch_delta: 本地epoch调整量
            - reasoning: 决策理由
        """
        # 前4轮使用不同的固定策略进行探索（强制探索阶段）
        if current_round == 0:
            decision = {
                "chosen_strategy_name": "size_only",
                "lr_scale": 1.0,
                "epoch_delta": 0,
                "reasoning": "第1轮：使用size_only(标准FedAvg)建立基线"
            }
            print(f"  [LLM] Round {current_round + 1} (Exploration Phase): strategy={decision['chosen_strategy_name']}, "
                  f"lr_scale={decision['lr_scale']:.2f}, epoch_delta={decision['epoch_delta']}")
            print(f"  [LLM] Reasoning: {decision['reasoning']}")
            self._log_decision(current_round, "", "", decision)
            self.last_llm_decision = decision
            return decision
        
        if current_round == 1:
            decision = {
                "chosen_strategy_name": "perf_only",
                "lr_scale": 1.0,
                "epoch_delta": 0,
                "reasoning": "第2轮：尝试perf_only策略，探索性能驱动聚合的效果"
            }
            print(f"  [LLM] Round {current_round + 1} (Exploration Phase): strategy={decision['chosen_strategy_name']}, "
                  f"lr_scale={decision['lr_scale']:.2f}, epoch_delta={decision['epoch_delta']}")
            print(f"  [LLM] Reasoning: {decision['reasoning']}")
            self._log_decision(current_round, "", "", decision)
            self.last_llm_decision = decision
            return decision
        
        if current_round == 2:
            decision = {
                "chosen_strategy_name": "hybrid",
                "lr_scale": 1.0,
                "epoch_delta": 0,
                "reasoning": "第3轮：尝试hybrid策略，探索混合聚合的效果"
            }
            print(f"  [LLM] Round {current_round + 1} (Exploration Phase): strategy={decision['chosen_strategy_name']}, "
                  f"lr_scale={decision['lr_scale']:.2f}, epoch_delta={decision['epoch_delta']}")
            print(f"  [LLM] Reasoning: {decision['reasoning']}")
            self._log_decision(current_round, "", "", decision)
            self.last_llm_decision = decision
            return decision
        
        if current_round == 3:
            decision = {
                "chosen_strategy_name": "fairness_clip",
                "lr_scale": 1.0,
                "epoch_delta": 0,
                "reasoning": "第4轮：尝试fairness_clip策略，完成所有策略的探索"
            }
            print(f"  [LLM] Round {current_round + 1} (Exploration Phase): strategy={decision['chosen_strategy_name']}, "
                  f"lr_scale={decision['lr_scale']:.2f}, epoch_delta={decision['epoch_delta']}")
            print(f"  [LLM] Reasoning: {decision['reasoning']}")
            self._log_decision(current_round, "", "", decision)
            self.last_llm_decision = decision
            return decision
        
        # 从第5轮开始让LLM做决策
        # 构造prompt
        prompt = self.build_prompt(history_round_metrics, current_round, num_rounds)
        
        # 系统提示词（增强版）
        system_prompt = """你是一个专业的联邦学习优化专家，擅长分析训练数据并做出最优决策。

你的核心目标是：让联邦学习系统的MAPE尽可能降低，持续优化模型性能。

决策原则：
1. 仔细分析历史数据中每种策略的实际效果
2. 关注MAPE的变化趋势，而不仅仅是单轮数值
3. 如果某种策略效果好，可以继续使用；如果效果不好，勇于尝试其他策略
4. 学习率调整：收敛慢时增加(lr_scale>1)，震荡时减小(lr_scale<1)
5. epoch调整：欠拟合时增加，过拟合时减少

请始终以JSON格式输出你的决策，不要包含其他内容。"""
        
        try:
            # 调用LLM
            print(f"  [LLM] Requesting strategy decision for round {current_round + 1}...")
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # 解析响应
            decision = self.parse_response(response)
            
            print(f"  [LLM] Decision: strategy={decision['chosen_strategy_name']}, "
                  f"lr_scale={decision['lr_scale']:.2f}, epoch_delta={decision['epoch_delta']}")
            # 显示完整的reasoning
            reasoning = decision.get('reasoning', '')
            print(f"  [LLM] Reasoning: {reasoning}")
            
            # 记录日志
            self._log_decision(current_round, prompt, response, decision)
            
            # 保存决策用于下一轮反馈
            self.last_llm_decision = decision
            
            return decision
            
        except Exception as e:
            print(f"  [Warning] LLM call failed: {e}")
            # 失败时选择上次成功的策略或默认策略
            fallback_strategy = self.last_llm_decision.get('chosen_strategy_name', 'size_only') if self.last_llm_decision else 'size_only'
            decision = {
                "chosen_strategy_name": fallback_strategy,
                "lr_scale": 1.0,
                "epoch_delta": 0,
                "reasoning": f"LLM调用失败，使用上次策略: {e}"
            }
            self._log_decision(current_round, prompt, str(e), decision)
            self.last_llm_decision = decision
            return decision
    
    def _log_decision(
        self,
        round_idx: int,
        prompt: str,
        response: str,
        decision: Dict[str, Any]
    ):
        """
        记录LLM决策日志
        
        Args:
            round_idx: 轮次索引
            prompt: 发送的prompt
            response: LLM原始响应
            decision: 解析后的决策
        """
        log_entry = {
            "round": round_idx + 1,
            "prompt_length": len(prompt),
            "prompt_excerpt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "raw_response": response[:1000] + "..." if len(response) > 1000 else response,
            "decision": decision
        }
        
        # 追加写入JSONL文件
        with open(self.decisions_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

