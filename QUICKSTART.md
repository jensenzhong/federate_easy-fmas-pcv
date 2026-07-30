# Quickstart

## Data Source

The canonical dataset is:

```text
Data/all_Data/Client_Data_Split_Cleaned.csv
```

It contains 688 rows, no missing cells, no duplicate rows, and three client labels.

## Run Experiments

```bash
pip install -r requirements.txt
python scripts/audit_project_data.py
python experiments/scenario_A_centralized.py
python experiments/scenario_A_prime.py
python experiments/scenario_B_fedavg.py --seed 42 --num_rounds 20
python experiments/scenario_C_llm.py --use_llm --seed 42 --num_rounds 20
```

For paper-level evidence:

```bash
python scripts/run_multi_seed.py --scenarios B C --seeds 42 123 456 789 2024
python scripts/run_ablation.py --seeds 42 123 456 789 2024
python scripts/statistical_analysis.py
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
```

## Current Result Snapshot

Use `results/multi_seed/statistical_summary.csv` for final B/C claims:

| Scenario | N | MAPE | RMSE | MAE | MPE | R2 |
|---|---:|---:|---:|---:|---:|---:|
| B FedAvg | 5 | 51.63% +/- 5.73% | 1,716,607 +/- 122,275 | 1,242,667 +/- 131,528 | -12.05% +/- 13.05% | 0.3038 +/- 0.1027 |
| C MAS-FL-LLM | 5 | 50.14% +/- 4.66% | 1,699,480 +/- 181,203 | 1,223,061 +/- 162,467 | -11.31% +/- 14.07% | 0.3142 +/- 0.1476 |
| C + bias correction | 5 | 50.20% +/- 4.48% | 1,534,372 +/- 61,640 | 1,091,372 +/- 53,981 | 3.25% +/- 5.56% | 0.4453 +/- 0.0448 |

B vs C paired significance tests are saved at `results/multi_seed/significance_tests.csv`; MAPE, RMSE, and MAE are all non-significant at `p < 0.05`.

## Outputs

- `docs/data_audit_report.md`: data quality and data lineage audit.
- `results/*.csv`: current single-run result files.
- `results/multi_seed/statistical_summary.csv`: multi-seed summary.
- `results/multi_seed/significance_tests.csv`: paired B/C significance tests.
- `results/ablation_summary.csv`: ablation summary.
- `results/paper_tables.md`: LaTeX table source.
- `results/figures/`: regenerated paper figures.
- `docs/code_review_report.md`: code review and reliability fixes.

Use multi-seed summaries for final claims whenever available.
