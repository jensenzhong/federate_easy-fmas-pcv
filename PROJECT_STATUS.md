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

Current offline verification after the provider-model switch:

- Full repository: `758 passed`, plus 69 existing SciPy precision-loss warnings.
- Focused strict runner/development/agent checks: `200 passed`.
- Plaintext API-key scans: no matches.

The earlier successful DeepSeek preflight is retained as historical evidence only:

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

The current approved provider contract is now:

- model: `deepseek-v4-flash`
- endpoint: `https://api.deepseek.com/chat/completions`
- scope: every real preflight, SA, and FMAS request
- credential source: process-only `DEEPSEEK_API_KEY`
- retry/fallback: disabled in the strict runner

The first preflight at commit `859a404` used the superseded model and failed its
strict response schema. It remains preserved as failure evidence and is not valid
for the new provider contract. A fresh real preflight is required after the model
switch is committed.

## Active Gate

`study_manifest.yaml` still has `formal_frozen: false`. The partial seed-42 matrix
from commit `c0902ff` was preserved under `results/development/seed42/invalidated/`
after the single-proposer prompt contract changed. No replacement matrix is active.
After a successful `deepseek-v4-flash` preflight, the fixed nine-run order remains
three non-LLM runs, three `SA_PCV_FEDYOGI` repetitions, and three
`FMAS_PCV_FEDYOGI` repetitions on one clean commit.

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
