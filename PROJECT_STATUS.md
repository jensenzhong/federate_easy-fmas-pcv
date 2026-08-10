# FMAS-PCV Project Status

> Last verified: 2026-08-10

## Current State

The strict-federated FMAS-PCV implementation and canonical runner are complete.
Development remains deliberately single-seed, and the formal protocol is not frozen yet.
No development or formal performance claim has been made.

The only paper-line methods are:

- `FEDAVG_STRICT`: federated baseline
- `FEDYOGI_STRICT`: federated baseline
- `DPCV_FEDYOGI`: deterministic PCV ablation
- `SA_PCV_FEDYOGI`: single-agent PCV ablation
- `FMAS_PCV_FEDYOGI`: proposed six-role multi-agent method

Historical VG/MAS/LLM-GCA methods remain available only as legacy code and are not
accepted by the canonical formal runner.

## Strict Data Protocol

- Federation is client-local: raw rows, labels, predictions, and private partitions are
  never exposed to the server or DeepSeek.
- The fixed split contains 480 training rows, 103 controller-validation rows, and 105
  locked-test rows.
- The three partitions are physically separated under `Data/strict_partition_v1/`.
- Development and formal training cannot open the locked-test file.
- Locked-test evaluation requires the formal phase, a frozen study, and explicit unlock.
- Preprocessing is fitted only on each client's training partition.

## Verified Runtime Guarantees

- Five methods share the same client training, data split, metric, checkpoint, and
  provenance path.
- DeepSeek methods use real calls with no retry, fallback model, fake response, or hidden
  heuristic.
- Authentication, connection, HTTP, timeout, schema, parsing, or agent-runtime failures
  stop the run and preserve an immutable numbered `PAUSED*.json` incident.
- Checkpoints restore model, optimizer, RNG, weighting, best-validation, and round state
  only after exact provenance checks.
- Formal evaluation cannot retrain and binds locked-test evidence to the completed
  checkpoint and evaluation provenance hashes.

## Verification Evidence

Offline verification completed before the first network request:

- Full repository: `641 passed`, plus 69 existing SciPy precision-loss warnings.
- Focused privacy/API/partition/runner checks: `247 passed`.
- Plaintext API-key scans: no matches.
- Independent specification review: compliant.
- Independent quality review: no remaining Critical or Important findings.

The user-approved real DeepSeek preflight then completed successfully:

- date: 2026-08-10
- model: `deepseek-chat`
- endpoint: `https://api.deepseek.com/chat/completions`
- role: `preflight`
- real request count: exactly one
- validated response: `{"status":"ready","model":"deepseek-chat"}`
- training/test execution: none
- credential material in telemetry: none detected

The immutable preflight provenance and sanitized call telemetry are stored under
`results/development/seed42/deepseek-preflight/`.

## Active Gate

`study_manifest.yaml` still has `formal_frozen: false`. The seed-42 launcher and
validation-only development gate are implemented and independently reviewed, but the
nine-run matrix has not started yet. Its fixed order is three non-LLM runs, three
`SA_PCV_FEDYOGI` repetitions, and three `FMAS_PCV_FEDYOGI` repetitions.

The predeclared trajectory gate requires validation MAPE not to degrade relative to the
strongest strict baseline, RMSE increase at most 5%, and R2 difference at least -0.02.
At least two of the three FMAS repetitions must pass. These are development screening
criteria, not statistical-significance or equivalence claims.

Real training requires one final approval after displaying the exact matrix, clean Git
commit, development-config hash, partition hash, prompt hashes, and output root. Locked-test
evaluation remains prohibited until a later formal-freeze gate.

## Offline Verification Command

```powershell
python -m pytest -q --basetemp=.pytest_release
```

The current offline baseline is `669 passed` plus the same 69 existing SciPy warnings.
The real preflight is an auditable one-time operation and should not be repeated merely
to increase API-call counts.
