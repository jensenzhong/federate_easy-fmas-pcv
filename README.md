# Federated Learning for Highway Cost Prediction
数据选取与处理

  本研究所采用的数据来源于美国佛罗里达州交通部（Florida Department of Transportation,
  FDOT）的公路项目数据库，涵盖了2002年至2012年间佛罗里达州7个行政区（District
  1–7）的高速公路路面重铺（pavement resurfacing）工程项目。该数据库初始包含1,249条项目记录，记录了每个工
  程项目从招标到完工的关键信息。选择佛罗里达州公路造价数据作为研究对象，主要基于以下考量：一方面，FDOT数
  据库具有较高的数据完整性和可信度，作为美国第三大州的交通管理机构，其数据采集和管理流程较为规范；另一方
  面，佛罗里达州7个行政区在地理环境、经济发展水平和建设市场竞争格局上存在显著差异，这种天然的区域异质性 
  为联邦学习中非独立同分布（Non-IID）场景的模拟提供了理想的实验条件。

  在数据清洗阶段，本研究对原始数据进行了系统性的筛选和处理。首先，剔除了存在关键字段缺失的记录；其次，对
  异常值（如合同金额为零或负值、工期明显不合理的项目）进行了排除；最后，统一了各字段的数据格式和量纲。经
  过上述清洗流程，最终保留了688条有效项目记录。清洗后的数据集在特征完整性方面表现良好，无缺失值，且各变 
  量的取值范围均在工程实践的合理区间之内。

  为模拟联邦学习中多方协作的实际场景，本研究将7个行政区的数据按照地理邻近性和样本数量均衡性原则划分为3个
  客户端（Client）。具体划分方案如下：客户端1包含第4、6、7区的项目数据，共计233条记录；客户端2包含第2、3
  区的数据，共计235条记录；客户端3包含第1、5区的数据，共计220条记录。这一划分策略的设计遵循了两个核心原 
  则：其一，确保各客户端的样本数量大致均等（220–235条），避免因数据规模严重不平衡导致联邦聚合过程中某一 
  客户端权重过大而主导全局模型的更新方向；其二，每个客户端内部的数据均来源于地理相邻或经济特征相近的行政
  区，使得各客户端的数据分布具有一定的内在一致性，同时客户端之间保留了因区域差异带来的数据异质性。这种划
  分方式真实地反映了实际工程场景中不同施工企业或交通管理部门各自持有本地区历史造价数据、但因商业机密或行
  政管辖限制而无法直接共享原始数据的现实情况。

  数据集包含10个预测特征变量和1个目标变量。预测特征由6个工程项目特征和4个宏观经济指标构成。工程项目特征 
  包括：投标人数（BIDS，取值1–7），反映市场竞争激烈程度；承包商过往表现评级（CPPR，取值44–106分），衡量 
  承包商的历史绩效水平；合同工期（ContDays，取值40–540天），表征项目的时间规模；项目长度（LEN，取值0.06–
  22.49英里），作为工程物理规模的核心度量；车道数量（LANE，取值0–8条），体现工程的横向复杂度；以及天气影
  响天数（WDays，取值2–174天），量化外部气候风险。宏观经济指标包括：消费者价格指数（CPI）、建筑投资支出 
  （ContSpend）、基准贷款利率（PLR）以及滞后两年的生产者价格指数（PPI），分别从通货膨胀水平、市场需求景 
  气度、融资成本和原材料价格变动等维度捕捉宏观经济环境对工程造价的影响。目标变量为合同金额（ContAmnt），
  即最终签订的工程合同价格，取值范围从约12.6万美元到超过1,500万美元，跨度达两个数量级，呈现显著的右偏分 
  布特征。

  在数据预处理方面，本研究采用了隐私保护的联邦标准化方法和非线性目标变换两项关键技术。对于特征标准化，为
  避免在联邦学习框架下集中原始数据，本研究设计了一种基于聚合统计量的联邦StandardScaler：各客户端仅将本地
  数据的特征求和值、平方求和值及样本计数上传至中央服务器，服务器据此计算全局均值和标准差，再将标准化参数
  下发至各客户端执行局部标准化。该方法在数学上等价于集中式StandardScaler的计算结果，同时严格保证了原始数
  据不离开本地。对于目标变量，鉴于合同金额分布的严重右偏性（偏度约2.06），本研究采用了0.25次幂变换（即四
  次方根变换），将目标值从原始的$[1.26 \times 10^5, 1.53 \times 10^7]$区间压缩至约$[18.85,
  57.84]$的范围内，使得模型在训练过程中对不同规模项目的百分比误差给予更为均衡的关注。在模型预测输出后， 
  通过四次方逆变换恢复至原始尺度。这一变换策略在工程造价预测文献中已有相关实践基础，其核心优势在于使损失
  函数在变换空间中对中小型项目和大型项目的预测偏差施加近似等权的惩罚，有效缓解了大型项目因绝对值大而主导
  训练梯度的问题。

  在数据集划分策略上，本研究采用了全局分层抽样与本地二次划分相结合的两级划分方案。首先，在全局层面按照客
  户端标签（Client）进行分层抽样，将688条数据按80%/10%/10%的比例划分为训练集（约550条）、验证集（约68条 
  ）和测试集（约70条），分层抽样确保了每个划分子集中三个客户端的数据比例与总体一致。全局训练集随后按客户
  端归属分配至对应的本地节点，每个客户端在本地进一步按80%/20%的比例划分为本地训练集和本地验证集。本地验 
  证集用于计算各客户端每轮训练后的验证指标（如验证MAPE），这些指标为联邦聚合阶段的权重计算提供依据，尤其
  是在基于性能加权的聚合策略中发挥关键作用。全局验证集和测试集则保留在服务器端，分别用于联邦训练过程中全
  局模型的选择以及最终的性能评估，其中测试集在整个训练过程中保持不可见，以确保评估结果的客观性。        

  ---
  以上共六个段落，分别覆盖了数据来源与选择理由、数据清洗、客户端划分策略、特征与目标变量描述、预处理技术
  、数据集划分方案。行文符合学术论文的段落式叙述风格，避免了列表和表格，适合直接嵌入论文的"数据与方法"章
  节。如需调整篇幅、增减细节或修改措辞，请告诉我。

## Latest Status

- Last full rerun date: 2026-02-26
- Rerun scope: A / A-prime / B / C
- Unified split: train/val/test = 550/68/70 (same protocol for all scenarios)
- Seed: 42

## 1. Experiment Setup

- A: Centralized GBR baseline
- A-prime: Centralized MLP baseline
- B: FedAvg baseline (size_only, 30 rounds)
- C: MAS-FL-LLM (dynamic strategy selection, 20 rounds)

All scenarios use the same cleaned source and split protocol for fair comparison.

## 2. Latest Rerun Results (Test Set)

| Scenario | Method | Test MAPE | Test RMSE | Test MAE | Test MPE |
|---|---|---:|---:|---:|---:|
| A | Centralized GBR | 64.33% | $1,639,790.76 | $1,189,341.47 | 29.67% |
| A-prime | Centralized MLP | 50.58% | **$1,406,208.50** | **$1,018,121.00** | 19.84% |
| B | FedAvg (size_only, 30 rounds) | 44.46% | $1,436,501.60 | $1,026,088.56 | 6.25% |
| C | MAS-FL-LLM (20 rounds) | **42.73%** | $1,789,666.50 | $1,271,675.00 | -23.61% |

## 3. Unified Conclusions

1. By MAPE, C is best in this rerun (42.73%), then B (44.46%).
2. By absolute error (RMSE/MAE), A-prime is best; C is not best.
3. B is currently the most balanced federated baseline (good MAPE with better RMSE/MAE than C).
4. C shows a metric tradeoff: lower MAPE but larger absolute error and stronger underestimation bias.

Do not claim "C fully outperforms B" without metric-specific qualification.

## 4. Current Completeness Check

Completed:
- Single-run rerun for A / A-prime / B / C under unified data split.
- Core outputs regenerated under `results/`.
- In-code consistency fixes applied (metric semantics and Scene C config effect).

Still missing for publication-grade evidence:
- Multi-seed repeated runs with mean ± std / confidence interval.
- Significance testing for B vs C.
- Dedicated ablation table (without LLM, with fixed strategies, call interval sensitivity).

## 5. Reproduction Commands

```bash
pip install -r requirements.txt

# A
python experiments/scenario_A_centralized.py

# A-prime
python experiments/scenario_A_prime.py

# B
python experiments/scenario_B_fedavg.py

# C
python experiments/scenario_C_llm.py --use_llm --num_rounds 20
```

## 6. Key Output Files

- `results/centralized_results.csv`
- `results/centralized_nn_results.csv`
- `results/fedavg_results.csv`
- `results/scenario_c_results.csv`
- `results/experiment_ABC_comparison.md`
- `results/logs/scene_B_round_metrics.csv`
- `results/logs/scene_C_round_metrics.csv`
- `results/logs/scene_C_llm_decisions.jsonl`
