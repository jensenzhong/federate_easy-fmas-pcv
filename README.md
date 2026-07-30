# Federated Learning for Highway Cost Prediction

This project is now closed onto the new mainline experiment family:

- `A`
- `A_prime`
- `B`
- `FEDYOGI`
- `VG_FEDYOGI_TR`
- `MAS_VG_FEDYOGI_TR`

Legacy `C` and `MAS_ADAPTIVE` remain runnable for historical reference, but they are no longer part of the default formal pipeline, default statistics, main tables, or main figures.

## Current Source of Truth

- Canonical raw data: `Data/all_Data/Client_Data_Split_Cleaned.csv`
- Canonical processed data: `Data/processed/Client_Data_Split_Cleaned_EN.csv`
- Formal experiment outputs: `results/`
- Adaptive pilot freeze file: `results/adaptive_pilot/pilot_recommendation.csv`
- Paper-facing summaries:
  - `results/multi_seed/all_results.csv`
  - `results/multi_seed/statistical_summary.csv`
  - `results/multi_seed/significance_tests.csv`
  - `results/ablation_summary.csv`
  - `results/stratified_evaluation.csv`
  - `results/paper_tables.md`
  - `results/figures/`

## Formal Mainline

| Scenario | Method |
|---|---|
| `A` | Centralized GBR |
| `A_prime` | Centralized MLP |
| `B` | FedAvg baseline |
| `FEDYOGI` | FedYogi-TR |
| `VG_FEDYOGI_TR` | Validation-guided FedYogi-TR |
| `MAS_VG_FEDYOGI_TR` | Multi-agent validation-guided FedYogi-TR |

Formal multi-seed comparisons use seeds `42 123 456 789 2024`.

## Recommended Workflow

Run the baseline verification first:

```bash
python -m pytest -q
```

Freeze the adaptive pilot before formal adaptive runs:

```bash
python scripts/run_adaptive_pilot.py --seeds 777 888 --server_lrs 0.2 0.3 0.4 0.5 --max_coordinate_step_ratios 0.75 1.0 --clip_norms none 2.0
```

Then run the formal six-scenario mainline:

```bash
python scripts/run_multi_seed.py --scenarios A A_prime B FEDYOGI VG_FEDYOGI_TR MAS_VG_FEDYOGI_TR --seeds 42 123 456 789 2024
```

Run the new four-line ablation:

```bash
python scripts/run_ablation.py --seeds 42 123 456 789 2024
```

Generate analysis outputs:

```bash
python scripts/statistical_analysis.py
python scripts/stratified_evaluation.py
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
```

## Result Policy

- Final claims should use multi-seed summaries and matched-seed significance tests.
- Adaptive formal runs must read frozen pilot parameters from `results/adaptive_pilot/pilot_recommendation.csv`.
- If the pilot recommendation is missing required fields or does not complete the expected seeds, formal adaptive runs now fail fast instead of silently falling back.
- Corrected metrics are supplementary for the adaptive trio and are reported alongside raw metrics when available.

## Notes

- `MAS_VG_FEDYOGI_TR` is the formal LLM-enabled mainline method.
- Fig.1 to Fig.9 are the current acceptance target for the paper package.
- `fig10` and `fig11` are treated as extra analysis, not part of the mainline acceptance gate.
