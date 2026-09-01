# FMAS-PCV Project Status

> Last verified: 2026-09-01

## Current State

The strict-federated FMAS-PCV implementation and canonical runner are complete.
The complete seed 42 development matrix passed its development gate on commit
`556120ba03668d49f42563febf02ad4a8c4387a8`. Freeze generation, the exact 45-run
formal matrix, immutable training/evaluation batch manifests, and frozen hierarchical
statistics are implemented and verified. The formal protocol is not frozen yet, and no
multi-seed or locked-test claim has been made.

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

- Full repository: `793 passed`, 69 subtests passed, plus 69 existing SciPy
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

The historical preflight at commit `79b299c` is retained as superseded evidence. The
replacement preflight and both diagnostic schema soaks passed on the final prompt
contract before the current matrix was launched. The completed matrix used real
DeepSeek calls throughout; no additional preflight is required before freeze.

## Development Gate

`study_manifest.yaml` still has `formal_frozen: false`. The current matrix contains all
nine required runs: FedAvg, FedYogi, DPCV, three SA repetitions, and three FMAS
repetitions. Every run completed 20 rounds on the same clean commit.

The predeclared trajectory gate requires validation MAPE not to degrade relative to the
strongest strict baseline, RMSE increase at most 5%, and R2 difference at least -0.02.
All three FMAS repetitions passed, exceeding the required two of three. The published
gate is `results/development/seed42/development_gate.json` with SHA-256
`3eac5cb5c13cebb2618067fd8fbad52eb83b36e2ff5d4913196f5c947f7801c7`.

The Baseline Fairness & Protocol Audit is PASS. Its machine-readable record is
`audits/baseline_fairness_audit.json`, and the concise report is
`docs/baseline_fairness_audit.md`. FedYogi implementation, R2 aggregation, checkpoint
selection, comparability, and the Locked Test boundary all passed independent review.
No audit-triggered rerun is required.

These remain development screening results, not statistical-significance or
equivalence claims. Locked-test evaluation remains prohibited until formal freeze and
separate approval.

## Offline Verification Command

```powershell
python -m pytest -q --basetemp=.pytest_release
```

The current offline baseline is `793 passed`, 69 subtests passed, plus the existing
SciPy precision-loss warnings. The formal-infrastructure focused suite adds freeze,
serial fail-stop, evidence-chain, partial-evaluation continuation, and five-seed
hierarchical-statistics coverage.
The real preflight is an auditable one-time operation and should not be repeated merely
to increase API-call counts.
