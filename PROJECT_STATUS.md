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

Current offline verification after the critic candidate-ID prompt fix:

- Full repository: `767 passed`, 69 subtests passed, plus 69 existing SciPy
  precision-loss warnings.
- Focused agent, development-runner, protocol, and engine checks: `314 passed`.
- Independent critic prompt-contract review: PASS; the reviewer reran
  `tests/test_pcv_agents.py` with `139 passed`.
- The proposer numeric-literal fix and critic exact-ID fix change only structured-output
  instructions. They do not change candidate semantics, client telemetry, the data
  partition, metrics, gates, or the locked-test boundary.

The approved provider contract remains:

- model: `deepseek-v4-flash`
- endpoint: `https://api.deepseek.com/chat/completions`
- scope: every real preflight, SA, and FMAS request
- credential source: process-only `DEEPSEEK_API_KEY`
- retry/fallback: disabled in the strict runner

The successful preflight at commit `3ab68e9` is retained as historical evidence, but
its critic prompt hash predates the exact-ID fix. A fresh single real preflight is
therefore required on the replacement clean commit.

## Active Gate

`study_manifest.yaml` still has `formal_frozen: false`. On commit `3ab68e9`, the real
preflight passed and the replacement matrix completed FedAvg, FedYogi, DPCV, and all
three SA repetitions. FMAS repetition 1 completed round 1, then stopped in round 2
because the real critic renamed provided candidate IDs into sequential labels including
unknown `candidate-3`, `candidate-4`, `candidate-5`, and `candidate-6` values. No retry,
fallback, fake response, ID mapping, or semantic repair was performed.

The minimal approved fix requires the critic to copy every provided candidate ID
character-for-character, prohibits renumbering or aliases, and verifies that accepted
and rejected IDs form an exact partition of the provided set. The strict validator and
critic acceptance criteria remain unchanged.

Because the critic prompt hash and Git commit change, the partial `3ab68e9` matrix must
be preserved as invalidated evidence. After a successful fresh preflight, the fixed
nine-run order remains three non-LLM runs, three `SA_PCV_FEDYOGI` repetitions, and
three `FMAS_PCV_FEDYOGI` repetitions on one clean commit.

The predeclared trajectory gate requires validation MAPE not to degrade relative to the
strongest strict baseline, RMSE increase at most 5%, and R2 difference at least -0.02.
At least two of the three FMAS repetitions must pass. These are development screening
criteria, not statistical-significance or equivalence claims. Locked-test evaluation
remains prohibited until a later formal-freeze gate.

## Offline Verification Command

```powershell
python -m pytest -q --basetemp=.pytest_release
```

The current offline baseline is `767 passed`, 69 subtests passed, plus the same 69
existing SciPy warnings.
The real preflight is an auditable one-time operation and should not be repeated merely
to increase API-call counts.
