# FMAS-PCV Project Status

> Last verified: 2026-08-31

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

Current offline verification after the coordinator duplicate-key prompt fix:

- Full repository: `768 passed`, 69 subtests passed, plus 69 existing SciPy
  precision-loss warnings.
- Focused agent, development-runner, protocol, and engine checks: `315 passed`.
- Independent coordinator prompt-contract review: PASS; the reviewer reran
  `tests/test_pcv_agents.py` with `140 passed`.
- The proposer numeric-literal, critic exact-ID, and coordinator unique-key fixes change
  only structured-output instructions. They do not change candidate semantics, client
  telemetry, the data partition, metrics, gates, or the locked-test boundary.

The approved provider contract remains:

- model: `deepseek-v4-flash`
- endpoint: `https://api.deepseek.com/chat/completions`
- scope: every real preflight, SA, and FMAS request
- credential source: process-only `DEEPSEEK_API_KEY`
- retry/fallback: disabled in the strict runner

The successful preflight at commit `79b299c` is retained as historical evidence, but
its coordinator prompt hash predates the unique-key fix. A fresh single real preflight
is therefore required on the replacement clean commit.

## Active Gate

`study_manifest.yaml` still has `formal_frozen: false`. On commit `79b299c`, the real
preflight passed and FedAvg, FedYogi, and DPCV completed. SA repetition 1 completed
round 7, then stopped in round 8 because the real coordinator returned two conflicting
`risk_acknowledgement` keys in one JSON object. No retry, fallback, fake response,
duplicate-key merge, or semantic repair was performed.

The minimal approved fix requires exactly three unique coordinator keys and combines
all risk text into one `risk_acknowledgement` string. The strict JSON parser and
candidate-selection criteria remain unchanged.

Because the coordinator prompt hash and Git commit change, the partial `79b299c`
matrix must be preserved as invalidated evidence. After a successful fresh preflight,
a user-requested full 20-round `SA_PCV_FEDYOGI` repetition-1 schema-soak will run first
under a diagnostic-only run ID. It cannot enter the development gate. Only after it
passes will the fixed formal-development order run three non-LLM methods, three SA
repetitions, and three FMAS repetitions on one clean commit.

The predeclared trajectory gate requires validation MAPE not to degrade relative to the
strongest strict baseline, RMSE increase at most 5%, and R2 difference at least -0.02.
At least two of the three FMAS repetitions must pass. These are development screening
criteria, not statistical-significance or equivalence claims. Locked-test evaluation
remains prohibited until a later formal-freeze gate.

## Offline Verification Command

```powershell
python -m pytest -q --basetemp=.pytest_release
```

The current offline baseline is `768 passed`, 69 subtests passed, plus the same 69
existing SciPy warnings.
The real preflight is an auditable one-time operation and should not be repeated merely
to increase API-call counts.
