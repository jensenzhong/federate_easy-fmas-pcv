# Federated Learning for Highway Cost Prediction

This project studies highway construction cost prediction with centralized baselines, FedAvg, and MAS-FL-LLM.

## Current Source of Truth

- Canonical raw data: `Data/all_Data/Client_Data_Split_Cleaned.csv`
- Canonical processed data: `Data/processed/Client_Data_Split_Cleaned_EN.csv`
- Current outputs: `results/`
- Data audit: `docs/data_audit_report.md`
- Historical material: `archive/`

The old `Data/processed/Company_A_train.csv`, `Company_B_train.csv`, and `Company_C_train.csv` files are historical amount-stratified files. They are not the current client split used by the main experiments.

## Data Status

The current dataset has 688 records, no missing cells, no duplicate rows, and three client groups:

| Client | Records |
|---|---:|
| Client 1 | 233 |
| Client 2 | 235 |
| Client 3 | 220 |

Run the audit:

```bash
python scripts/audit_project_data.py
```

## Experiments

| Scenario | Method |
|---|---|
| A | Centralized Gradient Boosting Regressor |
| A' | Centralized MLP |
| B | FedAvg baseline |
| C | MAS-FL-LLM with dynamic strategy selection |

## Current Main Results

Final claims should use the five-seed B/C results in `results/multi_seed/statistical_summary.csv`.

| Scenario | N | MAPE | RMSE | MAE | MPE | R2 |
|---|---:|---:|---:|---:|---:|---:|
| B FedAvg | 5 | 51.63% +/- 5.73% | 1,716,607 +/- 122,275 | 1,242,667 +/- 131,528 | -12.05% +/- 13.05% | 0.3038 +/- 0.1027 |
| C MAS-FL-LLM | 5 | 50.14% +/- 4.66% | 1,699,480 +/- 181,203 | 1,223,061 +/- 162,467 | -11.31% +/- 14.07% | 0.3142 +/- 0.1476 |
| C + bias correction | 5 | 50.20% +/- 4.48% | 1,534,372 +/- 61,640 | 1,091,372 +/- 53,981 | 3.25% +/- 5.56% | 0.4453 +/- 0.0448 |

B vs C paired tests are not statistically significant on MAPE, RMSE, or MAE at `p < 0.05`. Use metric-specific wording instead of saying C comprehensively outperforms B.

Single-seed reproduction:

```bash
python experiments/scenario_A_centralized.py
python experiments/scenario_A_prime.py
python experiments/scenario_B_fedavg.py --seed 42 --num_rounds 20
python experiments/scenario_C_llm.py --use_llm --seed 42 --num_rounds 20
```

Paper-level runs:

```bash
python scripts/run_multi_seed.py --scenarios B C --seeds 42 123 456 789 2024
python scripts/run_ablation.py --seeds 42 123 456 789 2024
python scripts/statistical_analysis.py
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
```

## Result Policy

Single-seed results are fast checks. Final claims should use multi-seed summaries where available.

Do not claim that Scenario C fully outperforms Scenario B without metric-specific qualification. Report MAPE, RMSE, MAE, MPE, and R2 separately.

## Project Layout

```text
configs/       Experiment configuration
Data/          Canonical raw and processed datasets
docs/          Data audit and paper-facing analysis
experiments/   Scenario A, A', B, and C runners
scripts/       Audit, multi-seed, ablation, table, and figure utilities
src/           Data processing, models, utilities, and federated learning code
tests/         Regression and audit tests
archive/       Historical reports and previous rerun backups
results/       Regenerated current outputs
```

See `QUICKSTART.md` for the short reproduction guide and `docs/code_review_report.md` for the code review and reliability fixes.
