## 新主线联邦自适应方法使用说明

本项目正式主线已经从旧 `Scene C / MAS_ADAPTIVE` 收口到以下三种自适应方法：

- `FEDYOGI`
- `VG_FEDYOGI_TR`
- `MAS_VG_FEDYOGI_TR`

旧 `C` 和 `MAS_ADAPTIVE` 仅保留为历史或附录材料，不再进入默认正式统计、主图和主表。

### 1. 环境准备

- Python 3.9+
- 安装依赖：`pip install -r requirements.txt`
- LLM Key 配置位置：`configs/config.yaml -> scene_c.llm.providers.<provider>.api_key`
- 更推荐使用环境变量：`scene_c.llm.providers.<provider>.api_key_env`

### 2. 数据要求

统一数据源保持不变：

- `Data/all_Data/Client_Data_Split_Cleaned.csv`

正式 `split_seed` 继续固定为 `42`。

### 3. 单次运行示例

FedYogi-TR:

```bash
python experiments/scenario_C_llm.py --num_rounds 20 --strategy size_only --server_optimizer fedyogi --output_prefix fedyogi --method_key FEDYOGI --seed 42
```

Validation-guided FedYogi-TR:

```bash
python experiments/scenario_C_llm.py --num_rounds 20 --strategy size_only --server_optimizer fedyogi --adaptive_mode validation_guided --output_prefix vg_fedyogi_tr --method_key VG_FEDYOGI_TR --seed 42
```

MAS-VG-FedYogi-TR:

```bash
python experiments/scenario_C_llm.py --num_rounds 20 --use_llm --temperature 0 --server_optimizer fedyogi --adaptive_mode mas_validation_guided --llm_score_tolerance 0.003 --output_prefix mas_vg_fedyogi_tr --method_key MAS_VG_FEDYOGI_TR --seed 42
```

### 4. Formal Pilot Freeze

正式 adaptive 主实验前，先冻结 pilot 推荐参数：

```bash
python scripts/run_adaptive_pilot.py --seeds 777 888 --server_lrs 0.2 0.3 0.4 0.5 --max_coordinate_step_ratios 0.75 1.0 --clip_norms none 2.0
```

冻结输出：

- `results/adaptive_pilot/pilot_summary.csv`
- `results/adaptive_pilot/pilot_group_summary.csv`
- `results/adaptive_pilot/pilot_recommendation.csv`

正式 adaptive 多种子运行会强制读取 `pilot_recommendation.csv`。

### 5. 正式多种子运行

```bash
python scripts/run_multi_seed.py --scenarios A A_prime B FEDYOGI VG_FEDYOGI_TR MAS_VG_FEDYOGI_TR --seeds 42 123 456 789 2024
```

### 6. 关键日志与结果文件

`FEDYOGI`

- `results/fedyogi_results.csv`
- `results/fedyogi_predictions.csv`
- `results/logs/fedyogi_round_metrics.csv`

`VG_FEDYOGI_TR`

- `results/vg_fedyogi_tr_results.csv`
- `results/vg_fedyogi_tr_predictions.csv`
- `results/logs/vg_fedyogi_tr_round_metrics.csv`

`MAS_VG_FEDYOGI_TR`

- `results/mas_vg_fedyogi_tr_results.csv`
- `results/mas_vg_fedyogi_tr_predictions.csv`
- `results/logs/mas_vg_fedyogi_tr_round_metrics.csv`
- `results/logs/mas_vg_fedyogi_tr_client_metrics.csv`
- `results/logs/mas_vg_fedyogi_tr_llm_decisions.jsonl`

### 7. 说明

- 正式 adaptive 参数不允许静默退回旧默认值。
- `MAS_VG_FEDYOGI_TR` 的主图日志语义已经改为 candidate request / selection / gate，而不是旧策略切换语义。
- 正式结论以 multi-seed summary 和 matched-seed significance 为准。
