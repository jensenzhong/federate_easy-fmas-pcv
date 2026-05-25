# Project Status

> Last updated: 2026-05-25

## Current State

The project has been re-baselined, cleaned, rerun, and documented. The current experiment source of truth is the 688-row client-labelled dataset:

```text
Data/all_Data/Client_Data_Split_Cleaned.csv
```

The data audit is generated at:

```text
docs/data_audit_report.md
```

## Data Integrity

The canonical data audit checks:

- 688 rows.
- 0 missing cells.
- 0 duplicate rows.
- Client distribution: Client 1 = 233, Client 2 = 235, Client 3 = 220.
- Required raw columns are present.

The historical `Company_A/B/C_train.csv` files are retained as legacy amount-stratified files and should not be used as current client splits.

## Experiment Protocol

| Scenario | Method | Purpose |
|---|---|---|
| A | Centralized GBR | Classical centralized baseline |
| A' | Centralized MLP | Neural centralized baseline |
| B | FedAvg | Federated baseline |
| C | MAS-FL-LLM | Dynamic LLM-guided federated strategy |

## Current Results

Use these files as the current source of truth:

- `results/multi_seed/statistical_summary.csv`
- `results/multi_seed/significance_tests.csv`
- `results/ablation_summary.csv`
- `results/paper_tables.md`
- `results/figures/`

Main multi-seed comparison:

| Scenario | N | MAPE | RMSE | MAE | MPE | R2 |
|---|---:|---:|---:|---:|---:|---:|
| B FedAvg | 5 | 51.63% +/- 5.73% | 1,716,607 +/- 122,275 | 1,242,667 +/- 131,528 | -12.05% +/- 13.05% | 0.3038 +/- 0.1027 |
| C MAS-FL-LLM | 5 | 50.14% +/- 4.66% | 1,699,480 +/- 181,203 | 1,223,061 +/- 162,467 | -11.31% +/- 14.07% | 0.3142 +/- 0.1476 |
| C + bias correction | 5 | 50.20% +/- 4.48% | 1,534,372 +/- 61,640 | 1,091,372 +/- 53,981 | 3.25% +/- 5.56% | 0.4453 +/- 0.0448 |

B vs C paired tests by matching seed:

| Metric | p-value | Effect size | Interpretation |
|---|---:|---:|---|
| MAPE | 0.4166 | 0.318 | Not significant |
| RMSE | 0.8244 | 0.124 | Not significant |
| MAE | 0.7489 | 0.148 | Not significant |

Non-LLM ablation rerun:

| Config | N | MAPE |
|---|---:|---:|
| ab-1 B-baseline | 5 | 51.63% +/- 5.73% |
| ab-2 B+FedProx | 5 | 51.62% +/- 5.73% |
| ab-3 C-fixed-perf | 5 | 51.54% +/- 5.96% |
| ab-4 C-fixed-hybrid | 5 | 51.69% +/- 5.80% |

The LLM ablation rows `ab-5` and `ab-6` were not rerun in the final ablation batch because the external DeepSeek API approval was rejected during that run. The main C multi-seed run did complete with real LLM calls and should be used for current C claims.

## Result Policy

Use this hierarchy for claims:

1. Multi-seed summary if available.
2. Single-seed result only for quick checks or when multi-seed is unavailable.
3. Archived results only as historical references.

Do not state that C fully outperforms B. Compare by metric: MAPE, RMSE, MAE, MPE, and R2.

## Reproduction

```bash
python scripts/audit_project_data.py
python experiments/scenario_A_centralized.py
python experiments/scenario_A_prime.py
python experiments/scenario_B_fedavg.py --seed 42 --num_rounds 20
python experiments/scenario_C_llm.py --use_llm --seed 42 --num_rounds 20
python scripts/run_multi_seed.py --scenarios B C --seeds 42 123 456 789 2024
python scripts/run_ablation.py --seeds 42 123 456 789 2024
python scripts/statistical_analysis.py
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
```

## Verification Status

- A, A', B, and C single-seed outputs regenerated.
- B/C multi-seed summary regenerated for seeds `42, 123, 456, 789, 2024`.
- Non-LLM ablation summary regenerated for seeds `42, 123, 456, 789, 2024`.
- Paper tables and figures regenerated from current outputs.
- Data audit passes the 688-row project contract.
