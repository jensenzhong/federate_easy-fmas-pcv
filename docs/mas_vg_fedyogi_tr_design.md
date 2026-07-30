# 多智能体协同自适应联邦学习改造设计

日期：2026-06-09

## 1. 结论先行

当前项目的研究背景是：在不共享原始工程造价数据的前提下，让多个客户端协同训练同一个 ANN 造价预测模型，并比较传统机器学习、传统神经网络、传统联邦学习和多智能体联邦学习的预测效果。

在这个背景下，原先的 `size_only / perf_only / hybrid / fairness_clip` 四策略切换可以作为早期消融，但不适合作为最终核心方法。它的问题是：

- 策略集合是人为枚举的，老师或审稿人会追问为什么只选这四种。
- `perf_only / hybrid / fairness_clip` 更多是启发式规则，不是一个足够强的算法贡献。
- LLM 只能从几个策略名里选，动作空间太窄，不能充分体现“多智能体协同自适应”。
- 当前实验日志显示，LLM 实际只切换了 `size_only` 和 `perf_only`，服务器学习率、local epochs、连续权重参数基本没有发挥作用。

建议把正式方法改为：

> **多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）**

核心思想：

- 不再把核心决策定义为“四个聚合策略之间切换”。
- 每轮在客户端权重空间中生成一组连续候选权重。
- 中心端用 validation-only 证据预演候选动作经过 FedYogi-TR 后的效果。
- LLM 策略智能体只负责给出搜索偏好、风险判断和可解释选择。
- 验证门控智能体保证最终执行的动作必须通过 validation 证据，不允许 LLM 无证据乱调。

这样更符合联邦学习文献中的“自适应聚合权重”“客户端公平性”“服务器端自适应优化”方向，也更适合当前只有 3 个客户端的小样本工程造价场景。

## 2. 为什么原四策略不适合作为正式核心

### 2.1 `size_only` 是合理基线

`size_only` 本质是 FedAvg 的样本量加权，即客户端样本越多，对全局模型贡献越大。这是传统联邦学习最常用的基线，适合作为“传统联邦学习”方法。

### 2.2 其他三个策略只能算启发式

`perf_only`、`hybrid`、`fairness_clip` 的直觉是合理的：

- `perf_only`：验证误差低的客户端权重大。
- `hybrid`：样本量和性能共同决定权重。
- `fairness_clip`：限制单个客户端权重过大或过小。

但这些策略存在方法论问题：

- 没有证明这四个规则覆盖了合理聚合空间。
- 不同数据集上最优权重可能不是这四类规则能表达的。
- LLM 选择的是策略标签，而不是直接优化客户端权重。
- 如果提升不明显，很难证明“多智能体”真正发挥了作用。

因此建议保留它们作为消融：

- `ab-fixed-size`
- `ab-fixed-perf`
- `ab-fixed-hybrid`
- `ab-fixed-fairness-clip`

但正式主方法不再依赖这四个菜单。

## 3. 推荐的正式方法

### 3.1 方法名称

英文内部 key：

- `VG_FEDYOGI_TR`
- `MAS_VG_FEDYOGI_TR`

中文显示名：

- `验证引导自适应联邦学习（VG-FedYogi-TR）`
- `多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）`

其中：

- `VG` = Validation-Guided
- `FedYogi` = 服务器端自适应优化器
- `TR` = Trust Region，服务器更新约束
- `MAS` = Multi-Agent System，多智能体协同

### 3.1.1 与旧方法的关系

当前仓库里已经有：

- `多智能体协同联邦学习`
- `自适应联邦学习（FedYogi-TR）`
- `多智能体协同自适应联邦学习（FedYogi-TR）`

这些结果在新一轮设计中应重新定位：

- `多智能体协同联邦学习`：作为旧 MAS 聚合策略方法。
- `自适应联邦学习（FedYogi-TR）`：作为服务器端自适应优化 baseline。
- `多智能体协同自适应联邦学习（FedYogi-TR）`：作为旧版 LLM 策略菜单方法，从正式主链路移除，不再进入新主表；旧结果可保留在 archive 或历史诊断记录中。

新方法不应覆盖旧结果文件，避免口径混乱。

### 3.2 多智能体分工

本方法中的“多智能体”不应只是每个客户端都接一个大模型。更稳妥的定义是多个功能智能体协同决策：

1. **本地诊断智能体**
   - 每个客户端本地训练。
   - 输出 train loss、local val MAPE/RMSE/MAE/MPE、update norm、validation gap。
   - 不上传原始样本、原始特征、样本级预测。

2. **中心候选生成智能体**
   - 根据客户端诊断生成连续聚合权重候选。
   - 生成不同 `server_lr_scale` 候选。
   - 生成有限的 `epoch_delta` 候选。

3. **验证预演智能体**
   - 对每个候选动作做 validation-only 预演。
   - 预演时克隆当前 FedYogi-TR 状态，不污染真实训练状态。
   - 输出候选的 global validation MAPE/RMSE/MAE/MPE、client gap、update norm、TR clipping 情况。

4. **LLM 策略智能体**
   - 读取候选摘要和历史趋势。
   - 输出目标偏好、风险判断和候选选择理由。
   - 不接触 test set。
   - 不接触原始样本和样本级预测。

5. **验证门控智能体**
   - 如果 LLM 选择的候选不满足 validation 证据，自动回退到验证得分最优候选。
   - 如果候选过度偏向单一客户端或更新不稳定，自动降级为更稳健候选。

### 3.3 每轮候选动作空间

当前只有 3 个客户端，连续权重搜索是可行的，不需要复杂强化学习。

候选权重满足：

```text
w = [w1, w2, w3]
w1 + w2 + w3 = 1
w_i >= min_client_weight
w_i <= max_client_weight
```

建议默认：

```text
min_client_weight = 0.05
max_client_weight = 0.80
weight_grid_step = 0.05 或 0.10
```

为了避免候选过多，正式实验中每轮候选控制在 15-30 个：

- 样本量权重附近的局部候选。
- 当前最佳历史权重附近的局部候选。
- 高误差客户端补偿候选。
- 低 MPE 偏差候选。
- 若干均匀/稳健候选。

同时组合有限服务器控制：

```text
server_lr_scale in {0.5, 1.0, 1.5}
epoch_delta in {-5, 0, +5}
```

第一版建议先让 `epoch_delta` 只影响下一轮，不参与同轮预演，降低复杂度。

### 3.4 候选评分函数

每个候选执行一次虚拟 FedYogi-TR 更新，并在 validation 上评估。

建议评分：

```text
score =
    val_mape
  + lambda_gap * client_mape_gap
  + lambda_bias * abs(global_val_mpe)
  + lambda_update * update_instability_penalty
```

默认：

```text
lambda_gap = 0.05
lambda_bias = 0.02
lambda_update = 0.01
```

正式主指标仍然是 MAPE，因此 `val_mape` 是主导项。RMSE/MAE/MPE 用作辅助稳定性约束。

### 3.5 LLM 的正确作用

LLM 不应该自由输出任意聚合权重，也不应该只选策略名。

建议 LLM 输出：

```json
{
  "objective_profile": {
    "primary": "mape",
    "secondary": ["rmse", "client_gap", "mpe_bias"],
    "risk_tolerance": "conservative|balanced|aggressive"
  },
  "selected_candidate_id": "candidate_012",
  "reasoning": "validation-only evidence for this candidate",
  "risk": "main risk and mitigation"
}
```

最终是否执行由验证门控决定。

这样 LLM 的价值是：

- 解释为什么某个候选在当前轮更合理。
- 在 MAPE 接近时根据 RMSE、MPE、公平性做偏好选择。
- 根据历史趋势调节探索/保守程度。
- 形成可解释训练过程。

不是让 LLM 直接替代优化器。

## 4. 为什么可以使用验证集预演

验证集用于训练过程中的模型选择和策略选择是合理的，和早停、checkpoint 选择、学习率选择属于同一类。

但必须遵守边界：

- test set 不能用于 pilot、候选选择、LLM prompt、早停或调参。
- 所有正式方法使用同一 train/validation/test split。
- 候选数量要有限，避免过度适配 validation。
- 必须新增无 LLM 的 `VG-FedYogi-TR` baseline，区分 validation-guided 本身和 LLM 多智能体的贡献。

因此正式比较应包含：

1. `传统联邦学习（FedAvg）`
2. `多智能体协同联邦学习（旧方法，四策略菜单，可作为历史方法）`
3. `自适应联邦学习（FedYogi-TR）`
4. `验证引导自适应联邦学习（VG-FedYogi-TR，无LLM）`
5. `多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）`

这样老师问“是不是因为多用了验证集才提升”时，可以回答：

> 我们专门设置了无 LLM 的验证引导基线。若 MAS-VG-FedYogi-TR 优于 VG-FedYogi-TR，才说明 LLM 策略智能体有额外贡献；若只优于 FedYogi-TR，则说明主要贡献来自 validation-guided adaptive aggregation。

### 4.1 防止 validation 过拟合的额外约束

验证集预演是合理的，但比普通早停更频繁，因此必须主动限制自由度。

正式实现必须加入：

- **候选预算上限**：每轮最多 30 个候选动作。
- **固定候选生成规则**：候选生成规则在 pilot 后冻结，正式 5 seeds 不再改变。
- **保守候选必含**：每轮候选中必须包含样本量权重、均匀权重、上一轮已接受权重。
- **最小改善阈值**：若最优候选相对保守候选的 validation MAPE 改善小于 `epsilon=0.002`，优先选择更保守的候选。
- **权重变化限制**：相邻两轮客户端权重 L1 变化不超过 `0.40`，除非 validation MAPE 改善超过 `0.01`。
- **单客户端上限**：默认单客户端权重不超过 `0.80`，避免某个客户端实质上主导全局模型。
- **完整日志**：每轮保存所有候选的 validation 指标、score、是否被 gate 接受。

这些约束的目的不是追求理论最优，而是保证方法可复现、可解释、不过度利用验证集。

### 4.2 验证集使用的公平性

为了避免“新方法多用了验证集所以不公平”的质疑，正式比较中要明确区分三层贡献：

1. `FedYogi-TR`
   - 只使用 validation 做 best checkpoint。
   - 不做每轮候选预演。

2. `VG-FedYogi-TR`
   - 使用 validation 做每轮候选预演。
   - 无 LLM。
   - 代表“验证引导自适应聚合”的贡献。

3. `MAS-VG-FedYogi-TR`
   - 使用相同候选和相同 validation 预演。
   - 增加 LLM 策略智能体。
   - 代表“LLM 多智能体策略”的额外贡献。

如果 MAS 方法不能超过 VG baseline，论文不能写 LLM 提升性能，只能写 LLM 提供可解释控制和决策框架。

## 5. 与当前项目背景是否匹配

匹配，原因如下：

### 5.1 客户端数量少，适合权重搜索

项目目前是 3 个客户端。对于 3 个客户端，连续权重空间是二维 simplex，小网格搜索完全可控。相比强化学习或复杂元学习，这个方案更稳、更容易复现。

### 5.2 小样本回归任务需要保守控制

工程造价数据样本量有限，MAPE 对小项目敏感。如果让 LLM 大范围自由调参，很容易过拟合或不稳定。验证门控和 trust-region 可以降低这个风险。

### 5.3 不改变 ANN，避免混入模型结构贡献

继续使用当前 ANN：

```text
CostEstimationMLP(input_dim=10, hidden_dims=[128,128,64,32], dropout=0.1)
```

这样新增效果只来自联邦优化与自适应聚合，而不是模型架构变化。

### 5.4 与隐私边界一致

中心端只接收：

- 客户端模型更新。
- 客户端级 validation 指标。
- 聚合权重和更新范数。

不接收：

- 原始样本。
- 原始特征。
- 原始标签。
- 样本级预测。
- test set 指标。

## 6. 参考依据和设计来源

### 6.1 FedAvg

McMahan et al. 提出的 FedAvg 是传统联邦学习的基础方法，核心是客户端本地训练后由服务器聚合模型更新。它支持把 `size_only` 作为传统基线。

参考：

- McMahan et al., 2017, Communication-Efficient Learning of Deep Networks from Decentralized Data: https://arxiv.org/abs/1602.05629
- Google Research 页面：https://research.google/pubs/communication-efficient-learning-of-deep-networks-from-decentralized-data/

### 6.2 FedOpt / FedYogi

Reddi et al. 提出 Adaptive Federated Optimization，把 Adagrad、Adam、Yogi 等自适应优化思想引入服务器端更新。当前项目采用 FedYogi-TR 是合理方向。

参考：

- Reddi et al., 2021, Adaptive Federated Optimization: https://arxiv.org/abs/2003.00295
- Google Research 页面：https://research.google/pubs/adaptive-federated-optimization/
- ICLR 2021 PDF：https://openreview.net/pdf?id=LkFG3lB13U5

### 6.3 客户端分布、公平性和非固定权重

Agnostic Federated Learning 说明联邦模型可能偏向某些客户端，并将目标分布视为客户端分布的混合。这支持“不要只固定样本量权重”的思想。

参考：

- Mohri et al., 2019, Agnostic Federated Learning: https://proceedings.mlr.press/v97/mohri19a.html

q-FFL/q-FedAvg 关注客户端间性能公平，说明优化平均损失可能损害部分客户端表现。这支持在候选评分中加入 client gap / fairness penalty。

参考：

- Li et al., 2020, Fair Resource Allocation in Federated Learning: https://arxiv.org/abs/1905.10497
- ICLR 页面：https://iclr.cc/virtual/2020/poster/2041

### 6.4 Learnable / adaptive aggregation weights

FedLAW 直接研究联邦学习中的可学习聚合权重，指出传统 FedAvg 的样本量权重不是唯一选择。这是本项目从“四策略菜单”转向“连续权重搜索”的重要依据。

参考：

- Li et al., 2023, Revisiting Weighted Aggregation in Federated Learning with Neural Networks: https://arxiv.org/abs/2302.10911
- Hugging Face paper page: https://huggingface.co/papers/2302.10911

FedDRL 使用强化学习自适应确定客户端影响因子，说明“动态客户端权重”是已有研究方向。但本项目样本量小、客户端少，不建议直接引入深度强化学习，采用 validation-guided 小候选搜索更可控。

参考：

- FedDRL: Deep Reinforcement Learning-based Adaptive Aggregation for Non-IID Data in Federated Learning: https://arxiv.org/abs/2208.02442

## 7. 实验设计

### 7.1 不改变的部分

- 不改变 ANN 架构。
- 不改变数据划分。
- 不改变 train/validation/test。
- 不改变主指标 MAPE。
- 不用 test 做 pilot 或调参。
- 正式 rounds 仍为 20。
- 正式 seeds 仍为 `42, 123, 456, 789, 2024`。

### 7.1.1 数据使用边界

正式实现必须在日志和 CSV 中区分：

- `train`: 只用于本地模型训练。
- `local_validation`: 用于客户端诊断，不上传样本级信息。
- `global_validation`: 用于候选预演、gate、早停和 pilot。
- `global_test`: 只用于最终评估。

`global_test` 的指标不得进入：

- LLM prompt。
- candidate preview。
- pilot recommendation。
- early stopping。
- candidate score。
- server optimizer selection。

这条边界必须写进测试。

### 7.2 新增方法

新增两个正式方法：

1. `VG_FEDYOGI_TR`
   - 无 LLM。
   - 使用连续权重候选搜索。
   - 使用 validation gate 直接选最优候选。
   - 用来衡量 validation-guided adaptive aggregation 本身的贡献。

2. `MAS_VG_FEDYOGI_TR`
   - 使用同一候选集合。
   - LLM 读取候选摘要和历史趋势。
   - LLM 选择候选并解释。
   - validation gate 审核。
   - 用来衡量 LLM 多智能体策略智能体是否带来额外贡献。

### 7.2.1 建议新增诊断方法

为了判断 LLM 是否真的有贡献，建议增加一个不进入主表、只进入附录或诊断表的方法：

3. `MAS_VG_FEDYOGI_TR_no_gate`
   - 使用同一候选集合和 LLM。
   - 不启用 validation gate，只记录 LLM 原始选择。
   - 目的不是追求最好结果，而是诊断 gate 是否在频繁纠正 LLM。

如果 no-gate 明显差，说明 gate 必不可少；如果 no-gate 与 gate 相近，说明 LLM 选择本身较稳定。

### 7.3 Pilot

非正式 pilot：

```text
seeds = 777, 888
server_lr = 0.2, 0.3, 0.4
weight_grid_step = 0.10, 0.05
min_client_weight = 0.05, 0.10
candidate_budget = 15, 30
epsilon = 0.001, 0.002
```

选择标准：

```text
mean_best_val_mape
```

不得根据 test 选择参数。

pilot 冻结后，正式实验必须记录：

- `candidate_budget`
- `weight_grid_step`
- `min_client_weight`
- `max_client_weight`
- `selection_epsilon`
- `weight_l1_change_limit`
- `score_profile`
- `pilot_recommendation_file`

### 7.4 正式结果解释

如果结果为：

- `MAS-VG-FedYogi-TR > VG-FedYogi-TR > FedYogi-TR`：
  - 可以写多智能体策略智能体带来额外贡献。

- `MAS-VG-FedYogi-TR ≈ VG-FedYogi-TR > FedYogi-TR`：
  - 可以写验证引导自适应聚合有效，但 LLM 主要提供解释性和控制框架，性能增益证据不足。

- `VG-FedYogi-TR / MAS-VG-FedYogi-TR` 都不优于 FedYogi-TR：
  - 不能写性能优越，只能作为方法探索或负结果分析。

### 7.5 必须避免的论文表述

无论结果如何，都不应写：

- “全局最优聚合权重”
- “证明 LLM 能显著提升联邦学习”
- “全面优于传统联邦学习”
- “验证集搜索保证泛化”

除非统计结果明确支持，否则只能写：

- “在固定候选预算下，validation-guided candidate selection 带来均值趋势改善。”
- “LLM 策略智能体提供候选选择解释；其性能贡献由 VG baseline 对照评估。”
- “差异未达统计显著时，结论限定为趋势而非显著提升。”

## 8. 代码改造范围

### 8.1 新增模块

建议新增：

```text
src/federated_learning/adaptive_candidates.py
```

职责：

- 生成客户端权重候选。
- 生成 server_lr_scale 候选。
- 计算候选 score。
- 输出候选表。

候选对象至少包含：

```json
{
  "candidate_id": "candidate_012",
  "weights": {"Client 1": 0.45, "Client 2": 0.20, "Client 3": 0.35},
  "server_lr_scale": 1.0,
  "epoch_delta": 0,
  "source": "size_anchor|uniform_anchor|previous_best|error_compensation|local_grid",
  "validation_metrics": {},
  "score": 0.0,
  "gate_status": "accepted|rejected|fallback"
}
```

### 8.2 修改服务器优化器

修改：

```text
src/federated_learning/server_optimizers.py
```

新增：

- `clone_state()`
- `load_optimizer_state()`
- `preview_step(current_state, weighted_average_state, server_lr_scale)`

要求 preview 不改变真实优化器状态。

### 8.3 修改中心智能体

修改：

```text
src/federated_learning/mas_agents.py
```

新增：

- `build_continuous_candidate_preview()`
- `select_candidate_by_validation_gate()`
- `run_training_with_validation_guided_adaptation()`
- `run_training_with_mas_validation_guided_adaptation()`

### 8.4 修改 LLM planner

修改：

```text
src/federated_learning/llm_planner.py
```

从输出策略名改为输出候选 ID：

```json
{
  "selected_candidate_id": "candidate_012",
  "objective_profile": {
    "primary": "mape",
    "secondary": ["rmse", "client_gap", "mpe_bias"]
  },
  "reasoning": "...",
  "risk": "..."
}
```

### 8.5 修改实验入口

修改：

```text
experiments/scenario_C_llm.py
scripts/run_multi_seed.py
scripts/statistical_analysis.py
```

新增 CLI：

```text
--adaptive_mode fixed_strategy|validation_guided|mas_validation_guided
--candidate_budget 30
--weight_grid_step 0.05
--min_client_weight 0.05
--max_client_weight 0.80
--candidate_score_profile mape_primary
```

## 9. 验收标准

### 9.1 单元测试

必须覆盖：

- 候选权重满足 sum=1 和上下界。
- 候选列表必须包含 size anchor、uniform anchor 和 previous accepted anchor。
- 候选数量不得超过 candidate budget。
- candidate preview 不改变真实 FedYogi optimizer 状态。
- validation gate 在 LLM 选择较差候选时会回退。
- validation gate 在改善小于 epsilon 时偏向保守候选。
- 相邻轮权重变化超过限制时必须被 gate 拒绝，除非达到大幅改善阈值。
- test metrics 不进入 candidate preview 和 LLM prompt。
- LLM 输出无效 candidate_id 时会回退。
- 新结果 CSV 包含 candidate metadata。

### 9.2 Smoke test

```text
VG_FEDYOGI_TR seed=42 20 rounds 跑通
MAS_VG_FEDYOGI_TR seed=42 20 rounds 跑通
```

### 9.3 正式验收

- `python -m pytest -q` 通过。
- `results/multi_seed/all_results.csv` 包含新增方法。
- 新增方法各 5 个正式 seeds。
- 统计比较包含：
  - FedYogi-TR vs VG-FedYogi-TR
  - VG-FedYogi-TR vs MAS-VG-FedYogi-TR
  - 多智能体协同联邦学习 vs MAS-VG-FedYogi-TR
  - ANN传统神经网络 vs 两个新增方法
- 每轮候选动作日志存在。
- LLM prompt 日志不含原始样本、样本级预测、test 指标。

## 10. 最终建议

本项目下一步应按以下优先级改：

1. 先实现候选生成、FedYogi-TR preview 和 validation gate。
2. 先跑 `VG-FedYogi-TR`，证明 validation-guided 连续权重搜索能跑通。
3. 再实现 `MAS-VG-FedYogi-TR`，让 LLM 在候选证据上做选择和解释。
4. 四策略菜单保留为消融，不再作为正式核心方法。
5. 先用 pilot 选候选预算、权重步长和 epsilon，只看 validation。
6. 最后跑正式 5 seeds，再判断是否值得写成主要贡献。

这套设计比原来的四策略切换更稳，也更容易回答老师的问题：

- 为什么这样聚合？因为在客户端权重空间中用 validation-only 证据选择，而不是人为指定固定规则。
- 为什么不是最优？不声称全局最优，只声称在有限候选预算和固定实验协议下做自适应选择。
- 为什么用验证集？它只用于训练阶段策略选择，test 完全保留为最终评估。
- 多智能体体现在哪里？本地诊断、中心候选、验证预演、LLM 策略、验证门控各自承担不同角色。
