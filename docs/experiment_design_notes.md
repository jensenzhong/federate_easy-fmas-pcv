# Experiment Design Notes

Last updated: 2026-06-26

## Baseline Positioning

- `A` / centralized GBR is a strong traditional machine-learning baseline for small tabular cost data. It is useful as a sanity/reference baseline, but it should not be described as a same-model centralized upper bound for the federated neural-network methods.
- `A_prime` / centralized MLP is the closest centralized same-model-family baseline for the federated MLP methods.
- For the innovation path, the primary internal baselines should be:
  - `B` / FedAvg
  - `FEDYOGI` / FedYogi-TR
  - `VG_FEDYOGI_TR` / validation-guided FedYogi-TR
  - `MAS_VG_FEDYOGI_TR` / full MAS validation-guided method
- The most direct ablation for the MAS/LLM contribution is `VG_FEDYOGI_TR` vs `MAS_VG_FEDYOGI_TR`, because both use validation-guided FedYogi-TR and differ mainly in candidate selection policy.

## Fixed Split First, Multi-Seed Later

- The first development-stage check should use one fixed data split and one seed, so functional behavior and result direction are easy to inspect.
- Multi-seed experiments should be run only after the fixed-split result direction is acceptable.
- Current multi-seed runs use a fixed data split and vary model/training randomness. This measures training stability under the same split, not full split-level generalization.
- A stronger later robustness design can vary both split seed and training seed.

## Centralized vs Federated Comparability

- The current fair centralized loaders train on the union of client local-train samples, not the full global-train split.
- This means centralized and federated methods use the same effective training examples, the same global validation/test sets, and the same scaler statistics.
- The comparison is usable as a controlled experimental comparison, but it is not a claim that centralized and federated optimization are exactly equivalent.
- Federated training has a different optimization path: local client updates, round-wise aggregation, best-round selection, and optional server-side optimizers.
- To strengthen the comparison, report or inspect:
  - same-data comparison: current fair setup
  - same-model comparison: centralized MLP vs federated MLP
  - traditional baseline: centralized GBR
  - optional full-centralized reference: train on the full 80% global train split as a non-privacy reference
  - optional budget-normalized view: compare against centralized MLP with matched or swept optimizer-step budgets

## Checkpoint Fix

- The centralized ANN checkpoint used to save `model.state_dict().copy()`, which is a shallow copy and can mutate during later training.
- The fixed behavior uses detached tensor clones through `snapshot_model_state()`.
- Any A-prime results generated before this fix should be treated as stale and rerun.
