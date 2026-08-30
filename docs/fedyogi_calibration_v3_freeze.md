# FedYogi Calibration V3 Freeze Decision

## Decision

Freeze the common FedYogi base server learning rate at `0.0175` for the next
seed 42 development-matrix rerun.

The machine-readable source is
`audits/fedyogi_calibration_seed42_v3_summary.json`, SHA-256
`1eb133145293b65881342897fa468d277c56fc157665193ece3ea2e2d47af3f3`.
It was produced from clean commit
`eb680c45c23aaa02a80118efde0be25c88568a78`.

## Selection Evidence

- All three preregistered points completed 20 rounds.
- `0.0125` was stable but improved only `1.5921%` from round 1.
- `0.015` was stable and improved `6.6823%` from round 1.
- `0.0175` was stable and improved `22.2234%` from round 1.
- The selected `0.0175` best checkpoint was round 19 with validation MAPE
  `0.7777662209201127`.
- Its final-to-best MAPE ratio was `1.088799`, below the preregistered `1.5`
  stability ceiling.

The selected point therefore passed the unchanged V2/V3 eligibility and
readiness rules. No post-result point was added.

## Protocol Boundary

Calibration used seed 42 Controller Validation only. DeepSeek was disabled and
Locked Test remained locked. Model structure, client data, preprocessing,
local optimizer, local epochs, client participation, round count, metrics, and
checkpoint selection were unchanged.

The common base value is updated consistently for `FEDAVG_STRICT`,
`FEDYOGI_STRICT`, `DPCV_FEDYOGI`, `SA_PCV_FEDYOGI`, and
`FMAS_PCV_FEDYOGI`. FedAvg does not execute the FedYogi parameter, but retaining
the same registered common field preserves the comparability gate.

Existing seed 42 matrix runs produced with `fedyogi_server_lr=0.1` remain
preserved but are invalid for the final comparison. All affected methods must
be rerun from one clean commit before the development gate is recomputed.
