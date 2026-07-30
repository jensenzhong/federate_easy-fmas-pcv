# Project Status

> Last updated: 2026-06-18

## Current State

The codebase has been aligned to the new formal mainline:

- `A`
- `A_prime`
- `B`
- `FEDYOGI`
- `VG_FEDYOGI_TR`
- `MAS_VG_FEDYOGI_TR`

Legacy `C` and `MAS_ADAPTIVE` are now historical/manual only.

## Formal Output Contract

The formal release path is expected to produce:

- `results/adaptive_pilot/pilot_recommendation.csv`
- `results/multi_seed/all_results.csv`
- `results/multi_seed/statistical_summary.csv`
- `results/multi_seed/significance_tests.csv`
- `results/ablation_summary.csv`
- `results/stratified_evaluation.csv`
- `results/paper_tables.md`
- `results/figures/fig1_*` through `fig9_*`

## Pipeline Behavior

- `scripts/run_multi_seed.py` defaults to the six formal mainline scenarios.
- Adaptive formal scenarios require frozen pilot parameters and fail fast if `pilot_recommendation.csv` is missing required fields.
- `scripts/run_ablation.py` defaults to the four-line mainline ablation.
- `scripts/stratified_evaluation.py` now loads the new mainline prediction family:
  - `fedavg_predictions.csv`
  - `fedyogi_predictions.csv`
  - `vg_fedyogi_tr_predictions.csv`
  - `mas_vg_fedyogi_tr_predictions.csv`
- `scripts/generate_paper_figures.py` now treats Fig.3 as the MAS-VG candidate selection and gate timeline, not the old Scene C strategy timeline.

## Verification

Current code verification baseline:

```bash
python -m pytest -q
```

Status on 2026-06-18:

- `74 passed`
- Mainline tests now reflect the new six-scenario closure plan.

## Remaining Execution Work

Code and tests are aligned to the new mainline contract. The remaining delivery work is operational:

1. Archive old mixed outputs before the final rerun.
2. Run the expanded adaptive pilot with seeds `777 888`.
3. Freeze `results/adaptive_pilot/pilot_recommendation.csv`.
4. Rerun the formal six-scenario five-seed main experiment.
5. Rerun the four-line five-seed ablation.
6. Regenerate statistics, stratified outputs, tables, and Fig.1 to Fig.9.

## Reproduction Commands

```bash
python -m pytest -q
python scripts/run_adaptive_pilot.py --seeds 777 888 --server_lrs 0.2 0.3 0.4 0.5 --max_coordinate_step_ratios 0.75 1.0 --clip_norms none 2.0
python scripts/run_multi_seed.py --scenarios A A_prime B FEDYOGI VG_FEDYOGI_TR MAS_VG_FEDYOGI_TR --seeds 42 123 456 789 2024
python scripts/run_ablation.py --seeds 42 123 456 789 2024
python scripts/statistical_analysis.py
python scripts/stratified_evaluation.py
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
```
