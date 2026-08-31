# Baseline Fairness & Protocol Audit

> Audit date: 2026-09-01
> Commit: `556120ba03668d49f42563febf02ad4a8c4387a8`
> Machine-readable record: `audits/baseline_fairness_audit.json`

## Decision

**Baseline Fairness Audit PASS，可以进入正式 freeze。**

本次审计没有发现算法实现错误、更新方向错误、数据协议不一致、指标错误、checkpoint 不公平或 Locked Test 泄露。没有修改算法、参数、阈值、数据划分或实验结果，也不需要重跑当前 seed 42 矩阵。

当前 `development_gate.json` 为 `gate_passed=true`。FMAS 三次 repetition 全部通过，要求仅为至少两次通过。

## Seed 42 Results

| Method | Best round | MAPE | RMSE | MAE | R² | Gate |
|---|---:|---:|---:|---:|---:|---|
| FedAvg | 16 | 0.424588 | 1,727,490.89 | 1,149,675.28 | 0.358585 | baseline |
| FedYogi | 19 | 0.777766 | 3,354,404.30 | 2,447,969.12 | -1.418457 | baseline |
| DPCV | 17 | 0.424407 | 1,698,443.97 | 1,133,954.93 | 0.379974 | ablation |
| SA rep1 | 16 | 0.417608 | 1,796,915.41 | 1,186,509.20 | 0.305995 | FAIL: R² only |
| SA rep2 | 16 | 0.410432 | 1,715,671.74 | 1,125,749.75 | 0.367332 | PASS |
| SA rep3 | 17 | 0.398978 | 1,804,722.32 | 1,179,102.89 | 0.299951 | FAIL: R² only |
| FMAS rep1 | 16 | 0.400568 | 1,698,328.05 | 1,106,435.20 | 0.380059 | PASS |
| FMAS rep2 | 16 | 0.419375 | 1,732,950.89 | 1,144,870.41 | 0.354524 | PASS |
| FMAS rep3 | 16 | 0.408194 | 1,709,328.02 | 1,110,741.47 | 0.372002 | PASS |

FMAS 在三次真实 LLM repetition 中均优于选定的严格联邦基线 FedAvg 的 MAPE，并同时满足 RMSE 与 R² guardrail。这里只是 development screening 结果，不构成多种子统计结论。

## FedYogi

FedYogi 公式、更新方向、一阶矩、Yogi 二阶矩、状态保留、learning-rate 应用次数和 checkpoint 恢复均为 PASS。当前参数为 `server_lr=0.0175`、`beta1=0.9`、`beta2=0.99`、`tau=0.001`，未启用 clipping。

异常表现不是代码 bug，也不是数值爆炸。其轨迹在前 13 轮下降很慢，14-19 轮明显改善，最佳 MAPE 为 `0.777766`，第 20 轮轻微退化到 `0.846831`。准确诊断是“20-round horizon 内延迟收敛并在末轮回退”。

参数公平性为 PASS，但论文必须如实披露：学习率校准经历用户批准的 v1、v2、v3 三阶段，只使用 seed 42 Controller Validation，没有使用 Locked Test。不能描述成一次性预注册网格。

## DPCV Mechanism

DPCV 20 轮中选择 FedAvg anchor `17/20`，FedYogi anchor `1/20`，其他 deterministic candidate `2/20`。`server_lr_scale` 始终为 `1.0`，没有触发 clipping。

因此 DPCV 的主要作用是绕开独立 FedYogi 的慢收敛轨迹，先沿 FedAvg 到达有效区域，再进行少量验证引导的局部修正；它不是继续并修复独立 FedYogi 已积累的 moment trajectory。

## R² And Checkpoints

R² 由跨客户端可加总充分统计量重建全局 SSE/SST，不是客户端 R² 平均。Gate 阈值固定为：MAPE 不劣于基线、RMSE 增幅不超过 `5%`、R² 差不低于 `-0.02`，并使用无隐藏容差的包含边界比较。

九个运行统一按 `aggregated_client_val_mape` 选择 checkpoint。MAPE、RMSE、MAE 和 R² 均从同一个 best model state 一次性计算，没有为不同指标选择不同轮次。独立从九个 checkpoint 重算的指标与发布 JSON 精确一致。

旧矩阵中的 FMAS rep1 曾因全局 R² 差 `-0.038625` 失败；该产物已隔离为 invalidated。当前有效 rep1 的 R² 差为 `+0.021473`，已经通过。不存在 client-level R² guardrail。

## Strict Federation

九个有效运行绑定相同 commit、seed、partition、sealed metadata 和 base config。模型、初始化、本地 Adam、客户端学习率 `0.0005`、20 local epochs、batch size 32、全客户端参与、20 rounds、指标与 checkpoint 规则一致。差异仅为预登记的 aggregation/PCV/agent treatment。

完整 provenance 扫描未发现 Locked Test 文件、指标值、文件哈希、原始特征、标签、逐样本预测、残差或行标识进入 agent prompt/response、rounds、validation、completion 或 development gate。所有有效运行均为 `locked_test_unlocked=false`。

## Non-blocking Notes

- SA rep3 的连接失败和获批恢复证据均已保留，完成记录可追溯，不影响可比性。
- Development completion 文件绑定 validation JSON SHA，但未记录 checkpoint SHA。本次独立 checkpoint 重算消除了当前结果疑问；正式 Locked Test 路径本身会绑定 checkpoint SHA。
- 本次审计没有发现需要修复的 bug，因此 `rerun_required=false`。

下一步是正式 freeze；在 freeze 之前仍不得读取 Locked Test，也不应继续围绕 seed 42 调参。
