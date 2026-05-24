## 场景 C（MAS-FL-LLM）使用说明

### 1. 环境准备
- Python 3.9+
- 安装依赖：`pip install -r requirements.txt`
- LLM Key 配置位置：`configs/config.yaml -> scene_c.llm.providers.<provider>.api_key`
- 更推荐使用环境变量：`scene_c.llm.providers.<provider>.api_key_env`

### 2. 数据要求
确保存在统一数据源：
- `Data/all_Data/Client_Data_Split_Cleaned.csv`

脚本会自动按 `configs/config.yaml -> scene_c.data` 做列映射、清洗和切分。

### 3. 运行方式

```bash
# LLM 决策模式（推荐）
python experiments/scenario_C_llm.py --use_llm --num_rounds 20

# 固定策略模式
python experiments/scenario_C_llm.py --strategy perf_only --num_rounds 20

# 四策略对比（不走LLM）
python experiments/scenario_C_llm.py --compare_strategies --num_rounds 20
```

### 4. 关键参数
- `scene_c.llm.call_every_n_rounds`: 每 N 轮调用一次 LLM（已在训练逻辑中生效）
- `scene_c.llm.timeout`: 单次 LLM 请求超时（秒）
- `scene_c.learning_rate`: 基础学习率
- `scene_c.local_epochs`: 基础本地 epoch

### 5. 输出文件
- `results/models/scenario_C_llm_model.pt`
- `results/scenario_c_results.csv`
- `results/logs/scene_C_round_metrics.csv`
- `results/logs/scene_C_client_metrics.csv`
- `results/logs/scene_C_training_history.json`
- `results/logs/scene_C_llm_decisions.jsonl`

### 6. 说明
- 当前实现包含 LLM 请求硬超时保护，避免单次API阻塞导致整次实验卡死。
- 结论以 `results/scenario_c_results.csv` 和 `results/experiment_ABC_comparison.md` 为准。
