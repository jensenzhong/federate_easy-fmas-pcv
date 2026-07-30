# Fixed Seed 42 Rerun

Date: 2026-06-26

Purpose: inspect functional result direction after the centralized ANN checkpoint fix, before running multi-seed stability tests.

Command:

```bash
python scripts/run_multi_seed.py --scenarios A A_prime B FEDYOGI VG_FEDYOGI_TR MAS_VG_FEDYOGI_TR --seeds 42
```

Important note: `results/multi_seed/all_results.csv` currently still contains older non-42 seed rows. The seed 42 rows below were rerun after the checkpoint fix.

| Scenario | Role | Test MAPE | RMSE | MAE | R2 | Corrected MAPE |
|---|---|---:|---:|---:|---:|---:|
| A | Centralized GBR reference baseline | 55.05% | 1,390,877 | 1,049,122 | - | - |
| A_prime | Centralized MLP same-model-family baseline | 43.02% | 1,484,185 | 1,039,525 | 0.4817 | 42.99% |
| B | FedAvg baseline | 42.07% | 1,700,182 | 1,198,033 | 0.3198 | 42.10% |
| FEDYOGI | FedYogi-TR server-adaptive baseline | 42.14% | 1,709,283 | 1,220,794 | 0.3125 | 41.87% |
| VG_FEDYOGI_TR | Validation-guided FedYogi-TR | 43.28% | 1,729,997 | 1,232,105 | 0.2957 | 44.93% |
| MAS_VG_FEDYOGI_TR | MAS validation-guided FedYogi-TR | 42.55% | 1,666,848 | 1,188,232 | 0.3462 | 45.16% |

## Immediate Reading

- The checkpoint fix materially improved `A_prime` seed 42 MAPE from the previous stale value around 50.02% to 43.02%.
- Under this fixed seed, centralized MLP and FedAvg/FedYogi/MAS are close in MAPE.
- `B` has the best raw MAPE among federated variants for seed 42, while `MAS_VG_FEDYOGI_TR` has the best RMSE/MAE/R2 among federated variants.
- `VG_FEDYOGI_TR` and `MAS_VG_FEDYOGI_TR` do not clearly beat FedAvg on raw MAPE for seed 42, so innovation tuning should continue before spending on multi-seed stability.
- This fixed-split comparison is usable as a controlled code-development comparison, but not as a claim of optimizer-equivalent centralized-vs-federated training.
