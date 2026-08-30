# FedYogi Calibration V3 Preregistration

## Purpose

Resolve the observed stability boundary between `server_lr=0.01` and `0.02`
without changing the FedYogi formula or any shared experimental condition. V2
showed that `0.01` completed 20 rounds but improved only `0.1286%`, while
`0.02` improved initially and then became unstable. V3 tests only three fixed
learning rates inside that interval.

## Fixed Search

The complete grid is registered before execution:

```text
server_lr = [0.0125, 0.015, 0.0175]
beta1 = 0.9
beta2 = 0.99
tau = 0.001
clip = null
max_coordinate_step_ratio = null
```

No point may be added after results are observed. Model structure, data split,
partition manifest, train-only preprocessing, initialization, local optimizer,
local epochs, client participation, round count, metrics, and checkpoint
selection remain unchanged.

## Eligibility And Selection

V3 reuses the V2 rules without modification. A point is eligible only if it
completes all 20 rounds, has maximum round MAPE at most `2.0`, and has
final-to-best MAPE ratio at most `1.5`. Eligible points are ranked by
best-checkpoint MAPE, RMSE, MAE, and then smaller server learning rate.

The selected point is freeze-ready only if its best MAPE improves by at least
`5%` from its first-round MAPE. A stable but stalled point is not freeze-ready.

## Evidence And Stop Rules

- Use seed 42 Controller Validation only.
- Never read or unlock Locked Test.
- Never call DeepSeek.
- Preserve every round trajectory and failed unit.
- Exact non-finite prediction failures are disqualified without retry.
- Every other failure aborts the complete calibration.
- If V3 has no stable and improving point, stop. Do not add learning-rate
  values; discuss whether a separately preregistered `beta1` or `tau` audit is
  justified before any further calibration.
