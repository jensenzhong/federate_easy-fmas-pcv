# FMAS-PCV Strict Federated Project Consolidation Design

Date: 2026-07-30

Status: User-approved design, pending implementation plan
Development seed: 42 (development only; excluded from formal significance tests)

## 1. Objective

Consolidate the project onto one rigorous strict-federated research line and develop a real-time DeepSeek multi-agent aggregation method that improves over strict federated baselines.

The proposed main method is:

> **Federated Multi-Agent Proposal–Critique–Vote FedYogi (FMAS-PCV)**

The paper prioritizes LLM and multi-agent innovation while preserving a conservative evidence standard:

- The server never receives client raw records, raw features, labels, or row-level predictions.
- The LLM and all server-side agents operate only on model updates and approved aggregate telemetry.
- Centralized models are reference points, not mandatory performance targets.
- The main success criterion is stable improvement over the strongest strict federated baseline.
- Seed 42 is used only for development. Formal results use pre-registered new training seeds after the algorithm, prompts, and hyperparameters are frozen.

## 2. Current Project Diagnosis

The repository currently contains multiple partially overlapping method families:

- FedAvg and FedYogi baselines;
- historical strategy-menu MAS methods;
- validation-guided and MAS validation-guided methods;
- coherence and LLM-GCA methods;
- strict no-server-validation variants;
- validation-preview variants;
- numerous pilot, smoke, diagnostic, and multi-seed outputs.

The current documentation, default runner, and aggregate result CSV do not describe the same formal scenario set. Existing five-seed evidence also shows that:

- `MAS_VG_FEDYOGI_TR` has a slightly lower mean MAPE than `VG_FEDYOGI_TR`, but their difference is not significant;
- `LLM_GCA_FEDYOGI_TR` is effectively tied with ordinary FedYogi;
- the current strict seed-42 LLM method does not outperform strict FedAvg;
- the LLM contribution is therefore not yet identifiable.

The consolidation must preserve old code and outputs for auditability while excluding them from new formal statistics.

## 3. Research Question and Success Gates

### 3.1 Primary research question

Can a real-time LLM multi-agent proposal, critique, and client-local voting process improve strict federated aggregation without exposing client data?

### 3.2 Development gate

Seed 42 is the only development seed. The development comparison uses validation evidence only and never loads the locked test partition.

FMAS-PCV advances to the formal stage only when:

- it improves validation MAPE by at least 3% relative to the strongest strict federated baseline;
- validation RMSE increases by no more than 1% and validation R2 decreases by no more than `0.02` relative to that baseline;
- at least two of three independent real-time DeepSeek trajectories pass the threshold;
- all privacy, logging, replay, and correctness tests pass.

The previous numeric target of test MAPE at or below 40.5% is a final evaluation target, not a development-time tuning signal.

### 3.3 Formal evidence gate

For the frozen method:

- seed 42 is excluded;
- five new training seeds are registered before execution;
- the full method must outperform the strongest strict federated baseline on at least four of five training seeds to be described as a stable improvement;
- a statistically significant claim requires Holm-corrected `p < 0.05`;
- without corrected significance, the paper may report an average improvement trend but not a significant improvement.

## 4. Strict Federated Information Boundary

### 4.1 Prohibited information

The server, DeepSeek, and server-side agents must never receive:

- client raw samples;
- raw feature values;
- raw labels;
- row-level predictions or residuals;
- test examples or test metrics during training;
- hand-authored descriptions derived from prior inspection of client data distributions.

### 4.2 Permitted information

Clients may transmit protocol-approved aggregate information produced during federated training:

- model parameters or model updates;
- client sample counts;
- local training loss;
- local validation metric aggregates;
- update norms and direction/coherence statistics;
- candidate rankings;
- relative candidate improvements;
- confidence and failure flags.

All prompt inputs use an explicit allowlist. A prohibited field causes an immediate error; it is never silently dropped and execution never continues.

## 5. Data Partition and Evaluation Protocol

### 5.1 Fixed client-local partition

The canonical dataset contains 688 records across three clients. Every client is partitioned independently:

| Partition | Ratio | Purpose |
|---|---:|---|
| Local Train | 70% | Local model training |
| Local Controller Validation | 15% | Per-round candidate evaluation and voting |
| Locked Local Test | 15% | Final evaluation after method freeze |

For the regression target, client-local target quantile bins are used only to balance the partition. The resulting row-level partition manifest is immutable and contains identifiers or hashes and partition labels, not raw values.

### 5.2 Decoupled randomness

- `split_seed` is fixed for the entire study.
- Training seeds affect initialization, dataloader order, optimizer randomness, and other approved training randomness.
- LLM trajectory identifiers separately capture DeepSeek stochasticity.
- Changing a training seed never changes which records belong to train, validation, or test.

### 5.3 Strict preprocessing

Each client computes feature sufficient statistics using Local Train only. The server aggregates counts, sums, and variance-related sufficient statistics to produce common normalization parameters.

Local Controller Validation and Locked Local Test use the frozen transformation but never contribute to its fitting. Target transformations that require fitted statistics follow the same train-only rule.

### 5.4 Client-local evaluation

Candidate and final models are evaluated locally. Clients return only the sufficient statistics required to aggregate:

- MAPE;
- RMSE;
- MAE;
- R2;
- pseudonymous client-level diagnostic metrics defined in the frozen protocol.

The server does not receive labels or row-level predictions. Locked Local Test cannot be loaded by training, prompt construction, candidate generation, candidate voting, checkpoint selection, or hyperparameter tuning code.

## 6. FMAS-PCV Round Architecture

Each round follows this sequence:

1. Clients train locally and return model updates plus approved telemetry.
2. A diagnostic agent summarizes drift, disagreement, instability, and recent training trends.
3. Three proposal agents independently generate candidates:
   - performance-oriented proposer;
   - stability-oriented proposer;
   - client-balance-oriented proposer.
4. A critic agent identifies unsupported, duplicate, unstable, or unsafe candidates.
5. The server constructs candidate model states without mutating the real FedYogi state.
6. Clients evaluate candidates on Local Controller Validation and return scalar scores and rankings.
7. A coordinator agent selects among admissible candidates using the client feedback and agent debate record.
8. A deterministic safety gate validates the selected action.
9. The selected action is executed, or a planned baseline anchor is executed when every valid LLM proposal is rejected by the algorithmic gate.

All agents use real-time DeepSeek API calls during every training round.

## 7. Candidate Space and Decision Rules

### 7.1 Candidate budget

Each round contains at most eight candidates:

- strict FedAvg anchor;
- strict FedYogi anchor;
- at most two proposals from each of the three proposal agents.

The critic removes duplicates and illegal candidates. It does not create hidden replacement candidates.

### 7.2 Action fields

Each candidate contains:

- three client aggregation weights;
- `server_lr_scale` selected from `0.50`, `0.75`, `1.00`, or `1.25`;
- `update_clip_norm` selected from `0.5`, `1.0`, `2.0`, or disabled.

Version 1 does not change model architecture, local epochs, client optimizer, data partition, or participation schedule.

### 7.3 Hard constraints

- client weights sum to one;
- each client weight is in `[0.05, 0.80]`;
- weight L1 change from the prior accepted round is at most `0.35`;
- candidate preview cannot mutate the real model, optimizer, scheduler, or random-number states;
- NaN, infinity, invalid tensor shape, update explosion, or schema violations reject the candidate.

### 7.4 Client voting

Clients report:

- relative validation MAPE improvement;
- relative validation RMSE improvement;
- candidate rank;
- confidence;
- catastrophic degradation flag.

The deterministic reference score prioritizes weighted validation MAPE improvement, then RMSE, with penalties for worst-client degradation and unstable updates.

The coordinator may select only a candidate that:

- has aggregate validation MAPE within `0.002` absolute MAPE of the round's best legal candidate;
- is no more than `0.001` absolute MAPE worse than the stronger anchor;
- causes no client's validation MAPE to degrade by more than 5% relative to that client's stronger-anchor result;
- passes all trust-region and legality checks.

LLM value is evaluated through candidate generation, proposal diversity, critique, and multi-objective coordination. The gate prevents unsafe execution but must not conceal infrastructure or API failures.

## 8. Method Registry and Ablations

### 8.1 Formal methods

| Internal key | Role |
|---|---|
| `FEDAVG_STRICT` | Basic strict federated baseline |
| `FEDYOGI_STRICT` | Adaptive server-optimizer baseline |
| `DPCV_FEDYOGI` | Deterministic proposal and client-vote baseline |
| `SA_PCV_FEDYOGI` | Single-agent proposal and client-vote method |
| `FMAS_PCV_FEDYOGI` | Full multi-agent method |

Centralized GBR and MLP remain reference-only methods.

### 8.2 Required ablations

The evidence package includes:

- strict FedAvg;
- strict FedYogi;
- deterministic PCV;
- single-agent PCV;
- FMAS-PCV without critic;
- full FMAS-PCV;
- FMAS-PCV without safety gate as a diagnostic-only run.

The no-gate run cannot be used as the deployed or recommended method.

Token and API cost do not constrain the main method. Costs, latency, and communication overhead are still reported. An auxiliary cost-matched single-agent comparison is included to separate organizational benefit from additional inference budget. Candidate count and client-local evaluation budget remain controlled in the principal ablation.

## 9. Experimental Protocol

### 9.1 Shared fairness controls

All formal methods use:

- the same partition manifest;
- the same model architecture;
- paired initialization seeds;
- `20` communication rounds;
- `20` local epochs per communication round;
- the same client optimizer and its frozen hyperparameters;
- the same client participation schedule;
- checkpoint selection by the lowest aggregated client-reported Local Controller Validation MAPE, using the same rule for every method;
- the same primary and secondary metrics.

### 9.2 DeepSeek repetitions

For each formal training seed:

- deterministic baselines run once;
- single-agent and multi-agent methods run three independent DeepSeek trajectories;
- the three LLM trajectories are averaged within the training seed before the seed is treated as one paired statistical observation;
- within-seed LLM variability is reported separately.

This prevents LLM repetitions from being incorrectly treated as independent training seeds.

### 9.3 Metrics and statistics

Primary metric:

- test MAPE.

Secondary metrics:

- RMSE;
- MAE;
- R2;
- worst-client degradation;
- convergence behavior;
- gate rejection and anchor fallback rates;
- DeepSeek calls, tokens, latency, and candidate communication overhead.

Statistical reporting includes:

- paired t-test;
- Wilcoxon signed-rank test;
- 95% confidence intervals;
- paired effect size;
- Holm correction for multiple method comparisons.

## 10. DeepSeek Failure and Experiment Integrity Policy

DeepSeek and experiment infrastructure failures are fail-stop conditions.

The following conditions immediately stop the experiment:

- authentication failure;
- network or connection failure;
- rate-limit failure;
- requested model unavailable;
- malformed or schema-invalid response;
- unhandled API or parsing exception;
- unexpected runtime error in an agent stage.

On failure:

1. No model update for the incomplete round is committed.
2. The system preserves the last fully completed round, including model, FedYogi state, random-number state, configuration, prompts, responses received, and failure logs.
3. No automatic retry, model substitution, simulated response, hidden rule response, or infrastructure fallback is allowed.
4. Other work related to the failed experiment stops.
5. The user receives the exact error category, round, completed progress, and required recovery condition without exposing secrets.
6. The experiment resumes only after the user restores access and explicitly approves resumption.
7. The failed round is re-executed from the last complete round so the trajectory is not partially advanced.

An LLM candidate rejected by the pre-approved algorithmic safety gate is not an infrastructure failure. Planned anchor selection may continue in that case and is logged as an algorithmic outcome.

No unplanned method, hyperparameter, heuristic, retry strategy, or special-case optimization may be introduced during an approved experiment. A new idea requires a separate written change proposal, user approval, and a new experiment version or `freeze_id`.

## 11. Software Architecture

The new implementation is modularized under:

```text
src/federated_learning/pcv/
  protocol.py
  candidates.py
  agents.py
  client_evaluation.py
  voting.py
  gate.py
  telemetry.py
  schemas.py
```

Responsibilities:

- `protocol.py`: privacy boundary and approved data flow;
- `candidates.py`: candidate actions and anchors;
- `agents.py`: DeepSeek role orchestration;
- `client_evaluation.py`: client-local candidate evaluation;
- `voting.py`: ranking and sufficient-statistic aggregation;
- `gate.py`: trust-region, legality, and planned anchor logic;
- `telemetry.py`: auditable round records;
- `schemas.py`: strict serialized input/output contracts.

The project uses one canonical formal experiment entry point:

```text
experiments/run_strict_federated.py
```

Method selection uses `--method`. Historical entry points may remain as compatibility wrappers but cannot produce formal paper outputs.

## 12. Configuration and Result Layout

Canonical configuration:

```text
configs/development_seed42.yaml
configs/formal_frozen.yaml
configs/methods/*.yaml
study_manifest.yaml
```

Canonical result layout:

```text
results/
  archive/
  development/seed42/<run_id>/
  formal/<freeze_id>/<method>/<seed>/<llm_rep>/
  paper/tables/
  paper/figures/
  manifests/
```

Each run records:

- configuration snapshot;
- Git commit and dirty status;
- data partition hash;
- DeepSeek model and API parameters;
- prompts and responses;
- candidates, votes, critiques, gate decisions, and execution action;
- metrics and failure state.

No result file is overwritten. Formal directories are treated as immutable. Any algorithm or prompt change creates a new `freeze_id`.

`study_manifest.yaml` is the single source of truth for:

- formal methods;
- baselines and ablations;
- legacy methods;
- development and formal seeds;
- data protocol version;
- frozen configuration;
- paper-eligible result batches;
- current project stage.

## 13. Historical Archive Policy

Historical methods and outputs are preserved rather than deleted:

- `C`;
- `MAS_ADAPTIVE`;
- `VG_FEDYOGI_TR`;
- `MAS_VG_FEDYOGI_TR`;
- coherence variants;
- LLM-GCA variants;
- validation-preview variants;
- smoke, pilot, and diagnostic outputs not belonging to the new protocol.

Before moving material:

- create an archive manifest;
- record original paths, size, timestamp, and checksum;
- ensure restoration is possible;
- exclude archived outputs from default statistics and plotting.

## 14. Test and Acceptance Requirements

Required tests cover:

- disjoint train, controller-validation, and locked-test partitions;
- fixed partition manifest across methods and training seeds;
- train-only preprocessing statistics;
- locked-test access denial before formal evaluation;
- prompt and telemetry allowlist enforcement;
- candidate weight and action constraints;
- side-effect-free candidate preview;
- correct local metric sufficient-statistic aggregation;
- voting and safety-gate behavior;
- DeepSeek schema validation;
- hard stop on authentication, network, model, parsing, or runtime errors;
- exact checkpoint and random-state preservation on failure;
- explicit user approval requirement before resume;
- no automatic retry or model substitution;
- replay of successful DeepSeek decisions;
- complete run provenance;
- separation of development, formal, and archived results.

Implementation acceptance requires:

1. all existing and new automated tests pass;
2. the privacy audit finds no prohibited information path;
3. seed-42 validation development gate passes in at least two of three DeepSeek trajectories;
4. the project exposes only one default formal method registry;
5. historical methods and results remain recoverable;
6. no formal experiment begins until the frozen manifest is reviewed and approved by the user.

## 15. Claim Discipline

The paper must not claim:

- global optimal aggregation weights;
- guaranteed generalization;
- significant improvement without corrected significance;
- strict privacy guarantees beyond the implemented threat model;
- LLM performance contribution when only the deterministic PCV components improve results.

Claims must follow the observed ablation:

- `DPCV > FedYogi` supports the client-local proposal/vote mechanism;
- `SA-PCV > DPCV` supports LLM proposal generation;
- `FMAS-PCV > SA-PCV` supports multi-agent organization;
- critic and gate ablations support their individual roles;
- otherwise, the conclusion is narrowed to the components actually supported by evidence.

## 16. Implementation Boundary

This document approves the design only. It does not authorize unplanned experiments or deviations.

The next artifact is a file-by-file implementation plan. Implementation begins only after the user reviews this written specification and approves transition to the plan.
