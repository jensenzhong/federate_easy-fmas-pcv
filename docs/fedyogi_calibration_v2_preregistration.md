# FedYogi Calibration V2 Preregistration

## Purpose

Refine only the FedYogi server learning rate so the strict baseline is
numerically stable and meaningfully trainable before the FMAS-PCV development
matrix is rerun. This calibration does not modify the model, data partition,
local training, optimizer formula, privacy boundary, checkpoint rule, metrics,
or any PCV/LLM method.

## Prior Evidence

Calibration v1 completed on controller validation only. `server_lr=0.01` was
stable but nearly stalled, `server_lr=0.1` reached its best MAPE at round 4 and
then diverged severely, and `server_lr=0.5` produced non-finite predictions.
An immutable tracked copy of the v1 summary is bound by SHA-256 in the v2
configuration so a fresh clone can validate the prerequisite evidence.

## Fixed Search

The complete grid is registered before execution:

```text
server_lr = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
beta1 = 0.9
beta2 = 0.99
tau = 0.001
clip = null
max_coordinate_step_ratio = null
```

No point may be added after results are observed.

## Eligibility And Selection

A completed point is eligible only when all 20 round MAPE values are finite,
the maximum round MAPE is at most 2.0, and final-round MAPE divided by the best
round MAPE is at most 1.5. Eligible points are ranked by best-checkpoint MAPE,
then RMSE, MAE, and smaller server learning rate.

The selected point is freeze-ready only if its best MAPE improves by at least
5% relative to its first-round MAPE. A stable but stalled selection is recorded
truthfully and does not authorize freezing or matrix reruns.

## Evidence And Stop Rules

- Use seed 42 controller validation only.
- Never read or unlock Locked Test.
- Never call DeepSeek.
- Preserve all 20 round MAPE values and stability checks in the summary.
- Exact non-finite prediction failures are disqualified without retry.
- Every other failure aborts the complete calibration.
- If no stable eligible point exists, stop without extending the grid.
