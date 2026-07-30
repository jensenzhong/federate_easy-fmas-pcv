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

from src.federated_learning.generated_strategy import (
    compute_coherence_weights,
    compute_robust_prior_weights,
    project_generated_strategy,
)


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
        log_dir: str = "results/logs",
        decisions_log_name: str = "scene_C_llm_decisions.jsonl",
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
        self.decisions_log_path = self.log_dir / decisions_log_name
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
        num_rounds: int = 20,
        decision_context: Optional[Dict[str, Any]] = None
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
        return self._build_enhanced_prompt(
            history_round_metrics=history_round_metrics,
            current_round=current_round,
            num_rounds=num_rounds,
            decision_context=decision_context or {}
        )

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
    
    def _build_enhanced_prompt(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int,
        decision_context: Dict[str, Any]
    ) -> str:
        recent_history = history_round_metrics[-self.recent_rounds:] if len(history_round_metrics) >= self.recent_rounds else history_round_metrics

        history_summary = []
        for record in recent_history:
            clients = {}
            for cid, metrics in record.get("client_metrics", {}).items():
                clients[cid] = {
                    "n_samples": metrics.get("n_samples"),
                    "train_loss": round(float(metrics.get("train_loss", 0.0)), 6),
                    "val_mape": round(float(metrics.get("val_mape", 0.0)), 6),
                    "val_rmse": round(float(metrics.get("val_rmse", 0.0)), 4),
                    "val_mae": round(float(metrics.get("val_mae", 0.0)), 4),
                    "val_mpe": round(float(metrics.get("val_mpe", 0.0)), 6),
                }
            history_summary.append({
                "round": record.get("round", 0) + 1,
                "strategy": record.get("strategy_name"),
                "lr": record.get("lr"),
                "local_epochs": record.get("local_epochs"),
                "aggregation_weights": {
                    cid: round(float(weight), 6)
                    for cid, weight in record.get("aggregation_weights", {}).items()
                },
                "clients": clients,
                "global_val": {
                    "mape": round(float(record.get("global_val", {}).get("mape", 0.0)), 6),
                    "rmse": round(float(record.get("global_val", {}).get("rmse", 0.0)), 4),
                    "mae": round(float(record.get("global_val", {}).get("mae", 0.0)), 4),
                    "mpe": round(float(record.get("global_val", {}).get("mpe", 0.0)), 6),
                    "r2": round(float(record.get("global_val", {}).get("r2", 0.0)), 6),
                }
            })

        all_mapes = [float(r.get("global_val", {}).get("mape", 0.0)) for r in history_round_metrics]
        current_clients = history_round_metrics[-1].get("client_metrics", {}) if history_round_metrics else {}
        client_mapes = [float(m.get("val_mape", 0.0)) for m in current_clients.values()]
        diagnostics = {
            "best_val_mape": round(min(all_mapes), 6) if all_mapes else None,
            "best_round": all_mapes.index(min(all_mapes)) + 1 if all_mapes else None,
            "current_val_mape": round(all_mapes[-1], 6) if all_mapes else None,
            "improvement_from_first_round": round(all_mapes[0] - all_mapes[-1], 6) if all_mapes else None,
            "client_mape_gap": round(max(client_mapes) - min(client_mapes), 6) if client_mapes else None,
            "mpe_bias_direction": self._bias_direction(
                float(history_round_metrics[-1].get("global_val", {}).get("mpe", 0.0))
            ) if history_round_metrics else None,
        }
        for window in (3, 5):
            if len(all_mapes) >= window:
                delta = all_mapes[-1] - all_mapes[-window]
                diagnostics[f"last_{window}_round_delta_mape"] = round(delta, 6)
                diagnostics[f"last_{window}_round_relative_change"] = round(delta / max(all_mapes[-window], 1e-9), 6)

        payload = {
            "privacy_boundary": {
                "allowed_inputs": "anonymous client-level validation metrics, aggregate validation metrics, aggregation weights, and derived diagnostics only",
                "forbidden_inputs": "raw rows, raw feature values, raw labels, sample-level predictions, and held-out final evaluation metrics",
            },
            "round_state": {
                "current_round": current_round + 1,
                "num_rounds": num_rounds,
                "remaining_rounds": num_rounds - current_round,
            },
            "current_client_validation_metrics": decision_context.get("current_client_metrics", {}),
            "validation_diagnostics": diagnostics,
            "recent_round_history": history_summary,
            "marginal_effect_history": self._build_marginal_effect_history(history_round_metrics),
            "candidate_weight_preview": decision_context.get("candidate_weight_preview", {}),
            "candidate_validation_preview": decision_context.get("candidate_validation_preview", {}),
        }

        return f"""You are controlling a horizontal federated learning optimizer.

Use validation evidence only. Do not use or infer from held-out final evaluation results.
The server may receive model parameters and anonymous aggregate validation metrics. It must not receive raw client data.

Strategy mechanisms:
- size_only: client weight = n_samples / sum(n_samples).
- perf_only: client weight is proportional to 1 / client_val_mape, then normalized across clients.
- hybrid: client weight = (1 - lambda_hybrid) * size_weight + lambda_hybrid * perf_weight.
- fairness_clip: first compute hybrid weights, then clip each weight into [alpha_min, alpha_max], then renormalize.

Decision evidence:
```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```

Reasoning requirements:
- Compare candidate strategies using current client validation errors, candidate weights, validation previews, and marginal validation effects.
- Do not conclude that a strategy is better only because it was used in later rounds with lower absolute validation MAPE.
- Do not prefer a strategy because its description sounds safer; use validation evidence.
- You may freely choose strategy and tune lr_scale, epoch_delta, lambda_hybrid, alpha_min, and alpha_max within the allowed ranges.
- If a server-side adaptive optimizer is enabled, choose server_lr_scale only from 0.5, 1.0, or 1.5.

Return only JSON:
```json
{{
  "chosen_strategy_name": "size_only|perf_only|hybrid|fairness_clip",
  "lr_scale": 1.0,
  "server_lr_scale": 1.0,
  "epoch_delta": 0,
  "lambda_hybrid": 0.5,
  "alpha_min": 0.1,
  "alpha_max": 0.6,
  "evidence": ["validation-only evidence used for this decision"],
  "risk": "main validation or stability risk",
  "reasoning": "concise explanation based only on validation evidence"
}}
```"""

    @staticmethod
    def _bias_direction(mpe: float) -> str:
        if mpe < -0.01:
            return "under_prediction"
        if mpe > 0.01:
            return "over_prediction"
        return "near_unbiased"

    def _build_marginal_effect_history(self, history_round_metrics: List[Dict]) -> List[Dict]:
        effects = []
        for idx, record in enumerate(history_round_metrics):
            current_mape = float(record.get("global_val", {}).get("mape", 0.0))
            prev_mape = float(history_round_metrics[idx - 1].get("global_val", {}).get("mape", 0.0)) if idx > 0 else None
            client_mapes = [float(m.get("val_mape", 0.0)) for m in record.get("client_metrics", {}).values()]
            prev_client_mapes = [
                float(m.get("val_mape", 0.0))
                for m in history_round_metrics[idx - 1].get("client_metrics", {}).values()
            ] if idx > 0 else []
            delta = current_mape - prev_mape if prev_mape is not None else None
            prev_gap = max(prev_client_mapes) - min(prev_client_mapes) if prev_client_mapes else None
            curr_gap = max(client_mapes) - min(client_mapes) if client_mapes else None
            effects.append({
                "round": record.get("round", 0) + 1,
                "strategy": record.get("strategy_name"),
                "val_mape_before": round(prev_mape, 6) if prev_mape is not None else None,
                "val_mape_after": round(current_mape, 6),
                "delta_val_mape": round(delta, 6) if delta is not None else None,
                "relative_improvement": round((-delta) / max(prev_mape, 1e-9), 6) if delta is not None else None,
                "new_best": current_mape <= min(float(r.get("global_val", {}).get("mape", 0.0)) for r in history_round_metrics[:idx + 1]),
                "client_gap_before": round(prev_gap, 6) if prev_gap is not None else None,
                "client_gap_after": round(curr_gap, 6) if curr_gap is not None else None,
                "client_gap_delta": round(curr_gap - prev_gap, 6) if curr_gap is not None and prev_gap is not None else None,
            })
        return effects[-self.recent_rounds:]

    @staticmethod
    def _candidate_score(candidate: Dict[str, Any]) -> float:
        score = candidate.get("score")
        if score is None:
            score = candidate.get("validation_metrics", {}).get("mape")
        try:
            return float(score)
        except (TypeError, ValueError):
            return float("inf")

    @classmethod
    def _best_candidate_id(cls, candidate_preview: Dict[str, Any]) -> Optional[str]:
        best_id = None
        best_score = float("inf")
        for candidate_id, candidate in candidate_preview.items():
            score_value = cls._candidate_score(candidate)
            if score_value < best_score:
                best_id = candidate_id
                best_score = score_value
        if best_id is None and candidate_preview:
            best_id = next(iter(candidate_preview))
        return best_id

    @classmethod
    def _near_best_candidate_ids(
        cls,
        candidate_preview: Dict[str, Any],
        score_tolerance: float,
    ) -> List[str]:
        best_candidate_id = cls._best_candidate_id(candidate_preview)
        if best_candidate_id is None:
            return []
        best_score = cls._candidate_score(candidate_preview[best_candidate_id])
        tolerance = max(float(score_tolerance or 0.0), 0.0)
        near = [
            candidate_id for candidate_id, candidate in candidate_preview.items()
            if cls._candidate_score(candidate) <= best_score + tolerance + 1e-12
        ]
        return sorted(near, key=lambda candidate_id: cls._candidate_score(candidate_preview[candidate_id]))

    @staticmethod
    def _rank_values(values: Dict[str, float]) -> Dict[str, float]:
        ordered = sorted(values, key=lambda key: (values[key], key))
        if len(ordered) <= 1:
            return {key: 0.0 for key in ordered}
        return {
            key: rank / (len(ordered) - 1)
            for rank, key in enumerate(ordered)
        }

    @classmethod
    def _balanced_recommended_candidate_id(
        cls,
        candidate_preview: Dict[str, Any],
        score_tolerance: float,
    ) -> Optional[str]:
        near_ids = cls._near_best_candidate_ids(candidate_preview, score_tolerance)
        if not near_ids:
            return cls._best_candidate_id(candidate_preview)
        if len(near_ids) == 1:
            return near_ids[0]

        def metric(candidate_id: str, name: str, default: float = 0.0) -> float:
            candidate = candidate_preview[candidate_id]
            validation = candidate.get("validation_metrics", {})
            if name == "abs_mpe":
                value = validation.get("mpe", validation.get("val_mpe", default))
                return abs(float(value or 0.0))
            if name == "rmse":
                return float(validation.get("rmse", validation.get("val_rmse", default)) or 0.0)
            if name == "gap":
                return float(candidate.get("client_gap", default) or 0.0)
            if name == "norm":
                return float(candidate.get("update_norm", default) or 0.0)
            if name == "score":
                return cls._candidate_score(candidate)
            return default

        rank_score = cls._rank_values({cid: metric(cid, "score") for cid in near_ids})
        rank_rmse = cls._rank_values({cid: metric(cid, "rmse") for cid in near_ids})
        rank_bias = cls._rank_values({cid: metric(cid, "abs_mpe") for cid in near_ids})
        rank_gap = cls._rank_values({cid: metric(cid, "gap") for cid in near_ids})
        rank_norm = cls._rank_values({cid: metric(cid, "norm") for cid in near_ids})

        balanced_scores = {}
        for cid in near_ids:
            balanced_scores[cid] = (
                0.30 * rank_score[cid]
                + 0.25 * rank_rmse[cid]
                + 0.25 * rank_bias[cid]
                + 0.15 * rank_gap[cid]
                + 0.05 * rank_norm[cid]
            )
        return min(balanced_scores, key=lambda cid: (balanced_scores[cid], cls._candidate_score(candidate_preview[cid]), cid))

    @classmethod
    def _sanitize_for_candidate_prompt(cls, value: Any) -> Any:
        forbidden_key_parts = (
            "test",
            "true_value",
            "predicted_value",
            "prediction",
            "sample",
            "target",
            "feature",
            "label",
            "raw",
        )
        if isinstance(value, dict):
            safe = {}
            for key, item in value.items():
                key_str = str(key)
                lowered = key_str.lower()
                if any(part in lowered for part in forbidden_key_parts):
                    continue
                safe[key_str] = cls._sanitize_for_candidate_prompt(item)
            return safe
        if isinstance(value, list):
            return [cls._sanitize_for_candidate_prompt(item) for item in value[:20]]
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, (int, bool)) or value is None:
            return value
        if isinstance(value, str):
            return value[:500]
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return str(value)[:500]

    def build_candidate_prompt(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int,
        candidate_preview: Dict[str, Any],
        score_tolerance: float = 0.002,
    ) -> str:
        recent_history = history_round_metrics[-self.recent_rounds:] if len(history_round_metrics) >= self.recent_rounds else history_round_metrics
        history_summary = []
        for record in recent_history:
            history_summary.append({
                "round": record.get("round", 0) + 1,
                "accepted_candidate_id": record.get("selected_candidate_id"),
                "requested_candidate_id": record.get("requested_candidate_id"),
                "gate_status": record.get("gate_status"),
                "aggregation_weights": self._sanitize_for_candidate_prompt(
                    record.get("aggregation_weights", {})
                ),
                "global_validation": self._sanitize_for_candidate_prompt({
                    "mape": record.get("global_val", {}).get("mape"),
                    "rmse": record.get("global_val", {}).get("rmse"),
                    "mae": record.get("global_val", {}).get("mae"),
                    "mpe": record.get("global_val", {}).get("mpe"),
                    "r2": record.get("global_val", {}).get("r2"),
                }),
                "candidate_score": record.get("candidate_score"),
                "client_level_validation": self._sanitize_for_candidate_prompt(
                    record.get("client_metrics", {})
                ),
            })

        sanitized_candidates = self._sanitize_for_candidate_prompt(candidate_preview)
        best_candidate_id = self._best_candidate_id(sanitized_candidates)
        near_best_candidate_ids = self._near_best_candidate_ids(
            sanitized_candidates,
            score_tolerance=score_tolerance,
        )
        balanced_candidate_id = self._balanced_recommended_candidate_id(
            sanitized_candidates,
            score_tolerance=score_tolerance,
        )
        payload = {
            "privacy_boundary": "Use validation-only aggregate evidence. Do not use private row-level records or final evaluation evidence.",
            "round_state": {
                "current_round": current_round + 1,
                "num_rounds": num_rounds,
                "remaining_rounds": num_rounds - current_round,
            },
            "candidate_selection_rule": {
                "allowed_action": "Select one candidate_id from the provided candidates.",
                "validation_gate": "A deterministic gate may override the choice if validation score or stability is clearly worse.",
                "best_validation_score_candidate": best_candidate_id,
                "score_tolerance": float(score_tolerance),
                "near_best_candidate_ids": near_best_candidate_ids,
                "balanced_recommended_candidate_id": balanced_candidate_id,
            },
            "candidate_preview": sanitized_candidates,
            "recent_round_history": history_summary,
            "marginal_effect_history": self._sanitize_for_candidate_prompt(
                self._build_marginal_effect_history(history_round_metrics)
            ),
        }

        return f"""You are the central policy agent in a multi-agent federated learning controller.

Choose exactly one candidate action from candidate_preview. Each candidate has already been evaluated with validation-only evidence after a FedYogi-TR preview update.

Decision priorities:
1. First restrict attention to near_best_candidate_ids. These candidates are within the allowed validation-score tolerance.
2. Within that near-best set, do not mechanically choose the lowest MAPE candidate. Prefer the balanced_recommended_candidate_id when it improves RMSE, absolute MPE, client gap, or update stability without leaving the near-best set.
3. Only choose outside near_best_candidate_ids if you can justify the extra validation-score risk; the validation gate may reject it.

Decision evidence:
```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```

Return only JSON:
```json
{{
  "selected_candidate_id": "{best_candidate_id}",
  "objective_profile": {{
    "primary": "mape",
    "secondary": ["rmse", "client_gap", "mpe_bias"],
    "risk_tolerance": "conservative|balanced|aggressive"
  }},
  "reasoning": "brief validation-only reason for the selected candidate",
  "risk": "main validation or stability risk"
}}
```"""

    def parse_candidate_response(
        self,
        response: str,
        candidate_preview: Dict[str, Any],
    ) -> Dict[str, Any]:
        best_candidate_id = self._best_candidate_id(candidate_preview)
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No JSON found in response")

            decision = json.loads(response[json_start:json_end])
            selected_candidate_id = str(decision.get("selected_candidate_id", ""))
            if selected_candidate_id not in candidate_preview:
                raise ValueError(f"Unknown candidate id: {selected_candidate_id}")

            objective_profile = decision.get("objective_profile", {})
            if not isinstance(objective_profile, dict):
                objective_profile = {}
            secondary = objective_profile.get("secondary", ["rmse", "client_gap", "mpe_bias"])
            if not isinstance(secondary, list):
                secondary = [str(secondary)]

            return {
                "selected_candidate_id": selected_candidate_id,
                "objective_profile": {
                    "primary": str(objective_profile.get("primary", "mape")),
                    "secondary": [str(item) for item in secondary],
                    "risk_tolerance": str(objective_profile.get("risk_tolerance", "balanced")),
                },
                "reasoning": str(decision.get("reasoning", "")),
                "risk": str(decision.get("risk", "")),
                "fallback_used": False,
                "fallback_candidate_id": None,
            }
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "selected_candidate_id": best_candidate_id,
                "objective_profile": {
                    "primary": "mape",
                    "secondary": ["rmse", "client_gap", "mpe_bias"],
                    "risk_tolerance": "conservative",
                },
                "reasoning": f"Fallback to best validation-score candidate because candidate response could not be used: {e}",
                "risk": f"LLM candidate parse fallback: {e}",
                "fallback_used": True,
                "fallback_candidate_id": best_candidate_id,
            }

    def choose_candidate(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int = 20,
        candidate_preview: Optional[Dict[str, Any]] = None,
        score_tolerance: float = 0.002,
    ) -> Dict[str, Any]:
        candidate_preview = candidate_preview or {}
        sanitized_candidates = self._sanitize_for_candidate_prompt(candidate_preview)
        prompt = self.build_candidate_prompt(
            history_round_metrics=history_round_metrics,
            current_round=current_round,
            num_rounds=num_rounds,
            candidate_preview=sanitized_candidates,
            score_tolerance=score_tolerance,
        )
        system_prompt = (
            "You are a conservative federated learning policy agent. "
            "Select one provided candidate using validation-only aggregate evidence. "
            "Return JSON only."
        )

        try:
            print(f"  [LLM] Requesting candidate decision for round {current_round + 1}...")
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            decision = self.parse_candidate_response(response, sanitized_candidates)
            self._log_decision(current_round, prompt, response, decision)
            self.last_llm_decision = decision
            print(
                f"  [LLM] Candidate decision: requested={decision['selected_candidate_id']}, "
                f"fallback={decision.get('fallback_used', False)}"
            )
            return decision
        except Exception as e:
            decision = self.parse_candidate_response("", sanitized_candidates)
            decision["risk"] = f"LLM candidate call fallback: {e}"
            decision["reasoning"] = f"Fallback to best validation-score candidate because LLM call failed: {e}"
            decision["fallback_used"] = True
            self._log_decision(current_round, prompt, str(e), decision)
            self.last_llm_decision = decision
            print(f"  [Warning] LLM candidate decision failed: {e}")
            return decision

    def build_validation_preview_generative_prompt(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int,
        candidate_preview: Dict[str, Any],
        client_summaries: Optional[Dict[str, Any]] = None,
        coherence_diagnostics: Optional[Dict[str, Any]] = None,
        score_tolerance: float = 0.002,
    ) -> str:
        sanitized_candidates = self._sanitize_for_candidate_prompt(candidate_preview)
        best_candidate_id = self._best_candidate_id(sanitized_candidates)
        near_best_candidate_ids = self._near_best_candidate_ids(
            sanitized_candidates,
            score_tolerance=score_tolerance,
        )
        recent_history = history_round_metrics[-self.recent_rounds:] if len(history_round_metrics) >= self.recent_rounds else history_round_metrics
        history_summary = []
        for record in recent_history:
            history_summary.append(self._sanitize_for_candidate_prompt({
                "round": record.get("round", 0) + 1,
                "accepted_candidate_id": record.get("selected_candidate_id"),
                "requested_candidate_id": record.get("requested_candidate_id"),
                "gate_status": record.get("gate_status"),
                "aggregation_weights": record.get("aggregation_weights", {}),
                "global_validation": record.get("global_val", {}),
                "candidate_score": record.get("candidate_score"),
                "candidate_source": record.get("candidate_source"),
            }))

        payload = {
            "privacy_boundary": "Use client-level aggregate summaries, validation-preview metrics, and update diagnostics only. Do not use test or row-level evidence.",
            "round_state": {
                "current_round": current_round + 1,
                "num_rounds": num_rounds,
                "remaining_rounds": num_rounds - current_round,
            },
            "candidate_rule": {
                "allowed_action": "Select one or more near-best candidates and return convex mixture weights over them.",
                "best_validation_score_candidate": best_candidate_id,
                "score_tolerance": float(score_tolerance),
                "near_best_candidate_ids": near_best_candidate_ids,
                "fallback": "A deterministic gate will fall back to the best validation candidate if the response is invalid or leaves the near-best set.",
            },
            "candidate_preview": sanitized_candidates,
            "client_summaries": self._sanitize_for_candidate_prompt(client_summaries or {}),
            "coherence_diagnostics": self._sanitize_for_candidate_prompt(coherence_diagnostics or {}),
            "recent_round_history": history_summary,
        }

        return f"""You are a federated learning control agent for FedYogi-TR.

Each candidate has already been evaluated by a validation-only preview update. Choose a convex mixture of near-best candidate actions, not raw client data.

Decision priorities:
1. Restrict selected_candidate_ids to near_best_candidate_ids unless the response explicitly accepts fallback risk.
2. Prefer the lowest validation MAPE candidate when the evidence is clear.
3. If another near-best candidate materially improves RMSE, absolute MPE, client gap, or update stability, mix it with the best candidate.
4. Do not use test evidence or row-level records.

Decision evidence:
```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```

Return only JSON:
```json
{{
  "selected_candidate_ids": ["{best_candidate_id}"],
  "mixture_weights": {{"{best_candidate_id}": 1.0}},
  "server_lr_scale": 1.0,
  "decision_type": "validation_improvement|drift_recovery|bias_mitigation|stability|balanced",
  "reasoning": "short validation-only reason",
  "risk": "main validation or stability risk"
}}
```"""

    @staticmethod
    def _normalize_simple_weights(weights: Dict[str, float], client_ids: List[str]) -> Dict[str, float]:
        clipped = {client_id: max(float(weights.get(client_id, 0.0)), 0.0) for client_id in client_ids}
        total = sum(clipped.values())
        if total <= 0 and client_ids:
            return {client_id: 1.0 / len(client_ids) for client_id in client_ids}
        return {client_id: value / total for client_id, value in clipped.items()}

    def _fallback_validation_preview_decision(
        self,
        candidate_preview: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        best_candidate_id = self._best_candidate_id(candidate_preview)
        best = candidate_preview.get(best_candidate_id, {})
        return {
            "selected_candidate_ids": [best_candidate_id] if best_candidate_id else [],
            "mixture_weights": {best_candidate_id: 1.0} if best_candidate_id else {},
            "projected_weights": {
                client_id: float(weight)
                for client_id, weight in best.get("weights", {}).items()
            },
            "server_lr_scale": float(best.get("server_lr_scale", 1.0)),
            "decision_type": "validation_improvement",
            "reasoning": f"Fallback to best validation-preview candidate: {reason}",
            "risk": f"LLM validation-preview fallback: {reason}",
            "fallback_used": True,
            "fallback_candidate_id": best_candidate_id,
        }

    def parse_validation_preview_generative_response(
        self,
        response: str,
        candidate_preview: Dict[str, Any],
        score_tolerance: float = 0.002,
    ) -> Dict[str, Any]:
        try:
            if not candidate_preview:
                raise ValueError("candidate_preview must not be empty")
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No JSON found in response")
            payload = json.loads(response[json_start:json_end])
            selected_ids = payload.get("selected_candidate_ids", [])
            if isinstance(selected_ids, str):
                selected_ids = [selected_ids]
            selected_ids = [str(candidate_id) for candidate_id in selected_ids]
            if not selected_ids:
                raise ValueError("selected_candidate_ids must not be empty")
            for candidate_id in selected_ids:
                if candidate_id not in candidate_preview:
                    raise ValueError(f"Unknown candidate id: {candidate_id}")

            best_id = self._best_candidate_id(candidate_preview)
            best_score = self._candidate_score(candidate_preview[best_id])
            for candidate_id in selected_ids:
                candidate_score = self._candidate_score(candidate_preview[candidate_id])
                if candidate_score > best_score + float(score_tolerance) + 1e-12:
                    raise ValueError(f"Candidate {candidate_id} is outside validation score tolerance")

            raw_mixture = payload.get("mixture_weights", {})
            if not isinstance(raw_mixture, dict):
                raise ValueError("mixture_weights must be an object")
            mixture = {
                candidate_id: max(float(raw_mixture.get(candidate_id, 0.0)), 0.0)
                for candidate_id in selected_ids
            }
            total_mixture = sum(mixture.values())
            if total_mixture <= 0:
                mixture = {candidate_id: 1.0 / len(selected_ids) for candidate_id in selected_ids}
            else:
                mixture = {candidate_id: value / total_mixture for candidate_id, value in mixture.items()}

            first_weights = candidate_preview[selected_ids[0]].get("weights", {})
            client_ids = list(first_weights.keys())
            mixed_weights = {client_id: 0.0 for client_id in client_ids}
            for candidate_id, mix_weight in mixture.items():
                candidate_weights = candidate_preview[candidate_id].get("weights", {})
                if set(candidate_weights.keys()) != set(client_ids):
                    raise ValueError(f"Candidate {candidate_id} has incompatible client ids")
                for client_id in client_ids:
                    mixed_weights[client_id] += mix_weight * float(candidate_weights[client_id])
            mixed_weights = self._normalize_simple_weights(mixed_weights, client_ids)

            decision_type = str(payload.get("decision_type", "balanced"))
            allowed_types = {"validation_improvement", "drift_recovery", "bias_mitigation", "stability", "balanced"}
            if decision_type not in allowed_types:
                decision_type = "balanced"
            return {
                "selected_candidate_ids": selected_ids,
                "mixture_weights": mixture,
                "projected_weights": mixed_weights,
                "server_lr_scale": float(payload.get("server_lr_scale", 1.0)),
                "decision_type": decision_type,
                "reasoning": str(payload.get("reasoning", "")),
                "risk": str(payload.get("risk", "")),
                "fallback_used": False,
                "fallback_candidate_id": None,
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return self._fallback_validation_preview_decision(candidate_preview, str(e))

    def choose_validation_preview_generative_strategy(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int,
        candidate_preview: Dict[str, Any],
        client_summaries: Optional[Dict[str, Any]] = None,
        coherence_diagnostics: Optional[Dict[str, Any]] = None,
        score_tolerance: float = 0.002,
    ) -> Dict[str, Any]:
        sanitized_candidates = self._sanitize_for_candidate_prompt(candidate_preview or {})
        prompt = self.build_validation_preview_generative_prompt(
            history_round_metrics=history_round_metrics,
            current_round=current_round,
            num_rounds=num_rounds,
            candidate_preview=sanitized_candidates,
            client_summaries=client_summaries,
            coherence_diagnostics=coherence_diagnostics,
            score_tolerance=score_tolerance,
        )
        system_prompt = (
            "You are a conservative federated learning control agent. "
            "Choose a validation-preview candidate mixture using aggregate evidence only. "
            "Return JSON only."
        )
        try:
            print(f"  [LLM] Requesting validation-preview generative decision for round {current_round + 1}...")
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            decision = self.parse_validation_preview_generative_response(
                response,
                sanitized_candidates,
                score_tolerance=score_tolerance,
            )
        except Exception as e:
            response = str(e)
            decision = self._fallback_validation_preview_decision(sanitized_candidates, str(e))

        self._log_decision(current_round, prompt, response, decision)
        self.last_llm_decision = decision
        print(
            f"  [LLM] Validation-preview decision: selected={decision.get('selected_candidate_ids')}, "
            f"fallback={decision.get('fallback_used', False)}"
        )
        return decision

    @classmethod
    def _sanitize_for_generative_prompt(cls, value: Any) -> Any:
        forbidden_key_parts = (
            "test",
            "true_value",
            "predicted_value",
            "prediction",
            "target",
            "feature",
            "label",
            "raw",
        )
        if isinstance(value, dict):
            safe = {}
            for key, item in value.items():
                key_str = str(key)
                if any(part in key_str.lower() for part in forbidden_key_parts):
                    continue
                safe[key_str] = cls._sanitize_for_generative_prompt(item)
            return safe
        if isinstance(value, list):
            return [cls._sanitize_for_generative_prompt(item) for item in value[:20]]
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, (int, bool)) or value is None:
            return value
        if isinstance(value, str):
            return value[:500]
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return str(value)[:500]

    def build_generative_weight_prompt(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int,
        client_summaries: Dict[str, Any],
        coherence_diagnostics: Dict[str, Any],
        previous_weights: Optional[Dict[str, float]] = None,
        recent_validation_trend: Optional[List[Dict[str, Any]]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        recent_history = history_round_metrics[-self.recent_rounds:] if len(history_round_metrics) >= self.recent_rounds else history_round_metrics
        validation_trend = recent_validation_trend
        if validation_trend is None:
            validation_trend = [
                {
                    "round": record.get("round", 0) + 1,
                    "mape": record.get("global_val", {}).get("mape"),
                    "rmse": record.get("global_val", {}).get("rmse"),
                    "mae": record.get("global_val", {}).get("mae"),
                    "mpe": record.get("global_val", {}).get("mpe"),
                    "weights": record.get("aggregation_weights", {}),
                }
                for record in recent_history
            ]

        client_ids = list(coherence_diagnostics.keys())
        coherence_prior_weights = compute_coherence_weights(coherence_diagnostics)
        size_prior_weights = {}
        total_size_weight = sum(
            max(float(row.get("sample_size_weight", 0.0)), 0.0)
            for row in coherence_diagnostics.values()
        )
        if total_size_weight > 0:
            size_prior_weights = {
                client_id: max(float(coherence_diagnostics[client_id].get("sample_size_weight", 0.0)), 0.0) / total_size_weight
                for client_id in client_ids
            }
        elif client_ids:
            size_prior_weights = {client_id: 1.0 / len(client_ids) for client_id in client_ids}
        robust_prior_weights = compute_robust_prior_weights(
            coherence_diagnostics,
            coherence_blend=0.02,
        )

        payload = {
            "privacy_boundary": "Use client-level aggregate summaries and update-coherence evidence only.",
            "round_state": {
                "current_round": current_round + 1,
                "num_rounds": num_rounds,
                "remaining_rounds": num_rounds - current_round,
            },
            "client_summaries": self._sanitize_for_generative_prompt(client_summaries),
            "coherence_diagnostics": self._sanitize_for_generative_prompt(coherence_diagnostics),
            "coherence_prior_weights": self._sanitize_for_generative_prompt(coherence_prior_weights),
            "size_prior_weights": self._sanitize_for_generative_prompt(size_prior_weights),
            "robust_prior_weights": self._sanitize_for_generative_prompt(robust_prior_weights),
            "recent_accepted_weights": self._sanitize_for_generative_prompt(previous_weights or {}),
            "recent_validation_trend": self._sanitize_for_generative_prompt(validation_trend),
            "allowed_constraints": self._sanitize_for_generative_prompt(constraints or {
                "weight_sum": 1.0,
                "min_client_weight": 0.05,
                "max_client_weight": 0.80,
                "max_l1_change_from_previous": 0.40,
                "anchor_l1_limit_range": [0.03, 0.30],
                "server_lr_scale_range": [0.5, 1.5],
                "control_actions": [
                    "balanced",
                    "coherence_shift",
                    "global_underfit_recovery",
                    "drift_suppression",
                    "bias_correction",
                ],
                "negative_coherence_cap": "client weight cannot exceed size weight",
                "large_update_norm_cap": "client weight cannot exceed size weight",
            }),
        }

        default_weights = {
            client_id: round(float(robust_prior_weights.get(client_id, 0.0)), 6)
            for client_id in client_ids
        }

        return f"""You are a federated learning control agent for FedYogi-TR.

Generate a federated control action for the next server update. You may control aggregation weights, server_lr_scale, and how far the weights may move from the robust prior.

Decision rules:
1. Treat robust_prior_weights as the default action, not uniform averaging, but do not stay at the default when there is clear underfitting, drift, bias, or coherence asymmetry evidence.
2. Do not return exact uniform weights unless coherence_prior_weights is already near-uniform or you cite concrete conflicting evidence.
3. Use control_action="global_underfit_recovery" with server_lr_scale above 1.0 when all clients have high validation MAPE, mostly negative MPE, positive coherence, and no drift warning. This changes the server step without changing the privacy boundary.
4. Use control_action="coherence_shift" when one or more clients have clearly stronger coherence and normal update norm; then set anchor_l1_limit between 0.12 and 0.25.
5. Use control_action="drift_suppression" when a client has negative coherence or very large update norm; reduce its weight and keep server_lr_scale at or below 1.0.
6. The size prior is the generalization anchor; coherence_prior_weights is directional evidence, not an automatic target.
7. If validation improves only after a late weight shift, treat it as possible validation overfitting; prefer stable coherence evidence over chasing recent validation MAPE alone.
8. Use control_action="balanced" only when there is no actionable signal; if balanced, keep anchor_l1_limit small.
9. Any deviation from robust_prior_weights must name the client-level evidence that justifies it.

Decision evidence:
```json
{json.dumps(payload, indent=2, ensure_ascii=False)}
```

Return only JSON:
```json
{{
  "aggregation_weights": {json.dumps(default_weights, ensure_ascii=False)},
  "server_lr_scale": 1.0,
  "control_action": "balanced|coherence_shift|global_underfit_recovery|drift_suppression|bias_correction",
  "anchor_l1_limit": 0.05,
  "decision_type": "coherence_driven|bias_correction|stability_recovery|balanced",
  "reasoning": "short evidence-based reason",
  "risk": "main risk and mitigation"
}}
```"""

    def parse_generative_weight_response(
        self,
        response: str,
        client_ids: List[str],
    ) -> Dict[str, Any]:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No JSON found in response")
            decision = json.loads(response[json_start:json_end])
            raw_weights = decision.get("aggregation_weights", {})
            if not isinstance(raw_weights, dict):
                raise ValueError("aggregation_weights must be an object")
            expected = set(client_ids)
            provided = set(str(client_id) for client_id in raw_weights.keys())
            if provided != expected:
                raise ValueError(f"Client ids mismatch: expected {sorted(expected)}, got {sorted(provided)}")
            weights = {client_id: max(float(raw_weights[client_id]), 0.0) for client_id in client_ids}
            total = sum(weights.values())
            if total <= 0:
                raise ValueError("aggregation_weights sum must be positive")
            weights = {client_id: value / total for client_id, value in weights.items()}
            decision_type = str(decision.get("decision_type", "balanced"))
            allowed_types = {"coherence_driven", "bias_correction", "stability_recovery", "balanced"}
            if decision_type not in allowed_types:
                decision_type = "balanced"
            control_action = str(decision.get("control_action", "balanced"))
            allowed_actions = {
                "balanced",
                "coherence_shift",
                "global_underfit_recovery",
                "drift_suppression",
                "bias_correction",
            }
            if control_action not in allowed_actions:
                control_action = "balanced"
            server_lr_scale = max(0.5, min(1.5, float(decision.get("server_lr_scale", 1.0))))
            default_anchor_limit = 0.05 if control_action == "balanced" else 0.20
            anchor_l1_limit = max(0.03, min(0.30, float(decision.get("anchor_l1_limit", default_anchor_limit))))
            return {
                "aggregation_weights": weights,
                "server_lr_scale": server_lr_scale,
                "control_action": control_action,
                "anchor_l1_limit": anchor_l1_limit,
                "decision_type": decision_type,
                "reasoning": str(decision.get("reasoning", "")),
                "risk": str(decision.get("risk", "")),
                "fallback_used": False,
                "fallback_source": None,
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return {
                "aggregation_weights": {},
                "server_lr_scale": 1.0,
                "control_action": "balanced",
                "anchor_l1_limit": 0.05,
                "decision_type": "balanced",
                "reasoning": f"Fallback to coherence baseline because generated response could not be used: {e}",
                "risk": f"LLM generative parse fallback: {e}",
                "fallback_used": True,
                "fallback_source": "coherence_baseline",
            }

    @staticmethod
    def _median(values: List[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def _activate_global_underfit_recovery_if_needed(
        self,
        decision: Dict[str, Any],
        coherence_diagnostics: Dict[str, Any],
        current_round: int,
        num_rounds: int,
    ) -> Optional[Dict[str, Any]]:
        if decision.get("fallback_used"):
            return None
        if decision.get("control_action", "balanced") != "balanced":
            return None
        if num_rounds - current_round < 3:
            return None
        rows = list(coherence_diagnostics.values())
        if not rows:
            return None

        mapes = [float(row.get("val_mape", 0.0)) for row in rows]
        mpes = [float(row.get("val_mpe", 0.0)) for row in rows]
        cosines = [float(row.get("cosine_to_mean_update", 0.0)) for row in rows]
        norms = [float(row.get("update_norm", 0.0)) for row in rows]
        median_norm = self._median(norms)
        has_large_norm = median_norm > 0 and any(norm > median_norm * 2.5 for norm in norms)

        high_error = all(mape >= 0.75 for mape in mapes)
        global_underestimate = sum(mpes) / len(mpes) <= -0.25
        coherent_updates = all(cosine >= 0.20 for cosine in cosines)
        no_negative_drift = all(cosine >= -0.05 for cosine in cosines)

        if high_error and global_underestimate and coherent_updates and no_negative_drift and not has_large_norm:
            decision["control_action"] = "global_underfit_recovery"
            decision["decision_type"] = "stability_recovery"
            decision["server_lr_scale"] = max(float(decision.get("server_lr_scale", 1.0)), 1.5)
            decision["anchor_l1_limit"] = max(float(decision.get("anchor_l1_limit", 0.05)), 0.12)
            decision["reasoning"] = (
                f"{decision.get('reasoning', '')} "
                "Execution layer activated global_underfit_recovery because all client-reported "
                "validation MAPE values are high, mean MPE is strongly negative, updates are coherent, "
                "and no large-norm or negative-coherence drift warning is present."
            ).strip()
            return {
                "control_activation": "global_underfit_recovery",
                "mean_val_mape": sum(mapes) / len(mapes),
                "mean_val_mpe": sum(mpes) / len(mpes),
                "min_cosine_to_mean_update": min(cosines),
                "remaining_rounds": num_rounds - current_round,
            }
        return None

    def choose_generated_weights(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int,
        client_summaries: Dict[str, Any],
        coherence_diagnostics: Dict[str, Any],
        previous_weights: Optional[Dict[str, float]] = None,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        weight_l1_change_limit: float = 0.40,
    ) -> Dict[str, Any]:
        client_ids = list(coherence_diagnostics.keys())
        prompt = self.build_generative_weight_prompt(
            history_round_metrics=history_round_metrics,
            current_round=current_round,
            num_rounds=num_rounds,
            client_summaries=client_summaries,
            coherence_diagnostics=coherence_diagnostics,
            previous_weights=previous_weights,
            constraints={
                "weight_sum": 1.0,
                "min_client_weight": min_client_weight,
                "max_client_weight": max_client_weight,
                "max_l1_change_from_previous": weight_l1_change_limit,
                "anchor_l1_limit_range": [0.03, 0.30],
                "server_lr_scale_range": [0.5, 1.5],
                "control_actions": [
                    "balanced",
                    "coherence_shift",
                    "global_underfit_recovery",
                    "drift_suppression",
                    "bias_correction",
                ],
                "negative_coherence_cap": "client weight cannot exceed size weight",
                "large_update_norm_cap": "client weight cannot exceed size weight",
            },
        )
        system_prompt = (
            "You are a conservative federated learning control agent. "
            "Generate aggregation weights from aggregate client summaries and coherence evidence. "
            "Return JSON only."
        )

        raw_response = ""
        try:
            print(f"  [LLM] Requesting generative weights for round {current_round + 1}...")
            raw_response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            decision = self.parse_generative_weight_response(raw_response, client_ids)
        except Exception as e:
            decision = self.parse_generative_weight_response("", client_ids)
            decision["reasoning"] = f"Fallback to coherence baseline because LLM call failed: {e}"
            decision["risk"] = f"LLM generative call fallback: {e}"
            raw_response = str(e)

        control_activation = self._activate_global_underfit_recovery_if_needed(
            decision,
            coherence_diagnostics,
            current_round=current_round,
            num_rounds=num_rounds,
        )

        if decision.get("fallback_used"):
            raw_weights = compute_coherence_weights(
                coherence_diagnostics,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
            )
            projected = project_generated_strategy(
                generated_weights=raw_weights,
                diagnostics=coherence_diagnostics,
                previous_weights=previous_weights,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
                l1_change_limit=weight_l1_change_limit,
            )
            projected.constraint_status["fallback_used"] = True
            projected.constraint_status["anchor_l1_limit_used"] = None
        else:
            raw_weights = decision["aggregation_weights"]
            robust_prior_weights = compute_robust_prior_weights(
                coherence_diagnostics,
                coherence_blend=0.02,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
            )
            anchor_l1_limit = float(decision.get("anchor_l1_limit", 0.05))
            projected = project_generated_strategy(
                generated_weights=raw_weights,
                diagnostics=coherence_diagnostics,
                previous_weights=previous_weights,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
                l1_change_limit=weight_l1_change_limit,
                anchor_weights=robust_prior_weights,
                anchor_l1_limit=anchor_l1_limit,
                decision_type=decision.get("decision_type", "balanced"),
                snap_to_size_l1_threshold=0.02,
            )
            projected.constraint_status["anchor_l1_limit_used"] = anchor_l1_limit

        if control_activation:
            projected.constraint_status.update(control_activation)

        decision["generated_weights_raw"] = raw_weights
        decision["projected_weights"] = projected.weights
        decision["constraint_status"] = projected.constraint_status
        self._log_decision(current_round, prompt, raw_response, decision)
        self.last_llm_decision = decision
        print(
            f"  [LLM] Generative weights ready: fallback={decision.get('fallback_used', False)}, "
            f"type={decision.get('decision_type', 'balanced')}"
        )
        return decision

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
                decision.setdefault("server_lr_scale", decision.get("lr_scale", 1.0))
                decision.setdefault("epoch_delta", 0)
                decision.setdefault("lambda_hybrid", 0.5)
                decision.setdefault("alpha_min", 0.1)
                decision.setdefault("alpha_max", 0.6)
                decision.setdefault("evidence", [])
                decision.setdefault("risk", "")
                decision.setdefault("reasoning", "")
                
                # 验证策略名称有效性
                valid_strategies = [s["name"] for s in self.candidate_strategies]
                if decision["chosen_strategy_name"] not in valid_strategies:
                    print(f"  [Warning] Invalid strategy '{decision['chosen_strategy_name']}', using 'size_only'")
                    decision["chosen_strategy_name"] = "size_only"
                
                # 限制lr_scale范围
                decision["lr_scale"] = max(0.5, min(2.0, float(decision["lr_scale"])))
                allowed_server_lr_scales = [0.5, 1.0, 1.5]
                raw_server_lr_scale = float(decision.get("server_lr_scale", 1.0))
                decision["server_lr_scale"] = min(
                    allowed_server_lr_scales,
                    key=lambda value: abs(value - raw_server_lr_scale),
                )
                
                # 限制epoch_delta范围
                decision["epoch_delta"] = max(-5, min(5, int(decision["epoch_delta"])))
                decision["lambda_hybrid"] = max(0.0, min(1.0, float(decision["lambda_hybrid"])))
                alpha_min = max(0.0, min(1.0, float(decision["alpha_min"])))
                alpha_max = max(0.0, min(1.0, float(decision["alpha_max"])))
                if alpha_min > alpha_max:
                    alpha_min, alpha_max = alpha_max, alpha_min
                decision["alpha_min"] = alpha_min
                decision["alpha_max"] = alpha_max
                if not isinstance(decision["evidence"], list):
                    decision["evidence"] = [str(decision["evidence"])]
                decision["risk"] = str(decision.get("risk", ""))

                return decision
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [Warning] Failed to parse LLM response: {e}")
            print(f"  [Warning] Using default strategy: size_only")
            return {
                "chosen_strategy_name": "size_only",
                "lr_scale": 1.0,
                "server_lr_scale": 1.0,
                "epoch_delta": 0,
                "lambda_hybrid": 0.5,
                "alpha_min": 0.1,
                "alpha_max": 0.6,
                "evidence": [],
                "risk": f"Parse error: {e}",
                "reasoning": f"Parse error: {e}"
            }
    
    def choose_strategy(
        self,
        history_round_metrics: List[Dict],
        current_round: int,
        num_rounds: int = 20,
        decision_context: Optional[Dict[str, Any]] = None
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
        force_initial_exploration = self.config.get("scene_c", {}).get("llm", {}).get(
            "force_initial_exploration", False
        )

        if force_initial_exploration and current_round == 0:
            decision = {
                "chosen_strategy_name": "size_only",
                "lr_scale": 1.0,
                "server_lr_scale": 1.0,
                "epoch_delta": 0,
                "reasoning": "第1轮：使用size_only(标准FedAvg)建立基线"
            }
            print(f"  [LLM] Round {current_round + 1} (Exploration Phase): strategy={decision['chosen_strategy_name']}, "
                  f"lr_scale={decision['lr_scale']:.2f}, epoch_delta={decision['epoch_delta']}")
            print(f"  [LLM] Reasoning: {decision['reasoning']}")
            self._log_decision(current_round, "", "", decision)
            self.last_llm_decision = decision
            return decision
        
        if force_initial_exploration and current_round == 1:
            decision = {
                "chosen_strategy_name": "perf_only",
                "lr_scale": 1.0,
                "server_lr_scale": 1.0,
                "epoch_delta": 0,
                "reasoning": "第2轮：尝试perf_only策略，探索性能驱动聚合的效果"
            }
            print(f"  [LLM] Round {current_round + 1} (Exploration Phase): strategy={decision['chosen_strategy_name']}, "
                  f"lr_scale={decision['lr_scale']:.2f}, epoch_delta={decision['epoch_delta']}")
            print(f"  [LLM] Reasoning: {decision['reasoning']}")
            self._log_decision(current_round, "", "", decision)
            self.last_llm_decision = decision
            return decision
        
        if force_initial_exploration and current_round == 2:
            decision = {
                "chosen_strategy_name": "hybrid",
                "lr_scale": 1.0,
                "server_lr_scale": 1.0,
                "epoch_delta": 0,
                "reasoning": "第3轮：尝试hybrid策略，探索混合聚合的效果"
            }
            print(f"  [LLM] Round {current_round + 1} (Exploration Phase): strategy={decision['chosen_strategy_name']}, "
                  f"lr_scale={decision['lr_scale']:.2f}, epoch_delta={decision['epoch_delta']}")
            print(f"  [LLM] Reasoning: {decision['reasoning']}")
            self._log_decision(current_round, "", "", decision)
            self.last_llm_decision = decision
            return decision
        
        if force_initial_exploration and current_round == 3:
            decision = {
                "chosen_strategy_name": "fairness_clip",
                "lr_scale": 1.0,
                "server_lr_scale": 1.0,
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
        prompt = self.build_prompt(
            history_round_metrics,
            current_round,
            num_rounds,
            decision_context=decision_context or {}
        )
        
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
                "server_lr_scale": 1.0,
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

