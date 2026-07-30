# LLM-GCA-FedYogi-TR Design

## Method Positioning

LLM-GCA-FedYogi-TR upgrades the previous MAS validation-preview method from candidate selection to generative federated control. The LLM no longer chooses a precomputed candidate ID. It receives aggregate client summaries plus update-coherence diagnostics and generates continuous aggregation weights for the next FedYogi-TR server update.

The method is not reinforcement learning. There is no learned policy network, reward training loop, replay buffer, or exploration schedule. The LLM is used as a zero-temperature reasoning controller that converts structured federated evidence into a bounded aggregation strategy.

## Literature Alignment

- FedCLLM motivates using LLM-readable client/domain descriptions for federated control while keeping raw client records outside the prompt.
- FedGCS motivates treating client control as a generative decision problem rather than a fixed heuristic or reinforcement-learning policy.
- FedLAW and FedAWA motivate dynamic aggregation based on client contribution, update direction, and consistency rather than sample size alone.
- FedYogi supplies the server-side adaptive optimizer. LLM-GCA only changes aggregation weights and an optional server learning-rate scale; the server update remains FedYogi-TR.

## Evidence Available to the LLM

The prompt uses only aggregate, round-level information:

- client summary text, such as validation error level, bias direction, coherence level, and recent weight trend;
- coherence diagnostics derived from model parameter updates;
- recent accepted aggregation weights;
- recent validation trend;
- allowed constraints.

The prompt does not include raw features, raw labels, row-level predictions, test-set metrics, or sample-level records.

## Coherence Diagnostics

For each client, the diagnostics layer computes:

- update norm;
- cosine similarity to the mean client update;
- mean pairwise cosine similarity with other clients;
- cosine similarity to the previous accepted global update;
- drift from the mean update;
- sample-size weight;
- validation MAPE and MPE.

Non-floating tensors are skipped. If an update vector has zero norm, cosine values are set to `0.0`.

## Deterministic Baseline

`COHERENCE_FEDYOGI_TR` is the non-LLM baseline:

```text
alignment_i = max(cosine_to_mean_update_i, 0)
raw_i = sample_size_weight_i * (0.5 + alignment_i)
```

The raw weights are normalized and projected to `[min_client_weight, max_client_weight]`. If every alignment is zero, the method falls back to size-only weights.

## LLM-GCA Strategy

`LLM_GCA_FEDYOGI_TR` asks the LLM to return JSON:

```json
{
  "aggregation_weights": {
    "client_1": 0.30,
    "client_2": 0.45,
    "client_3": 0.25
  },
  "server_lr_scale": 1.0,
  "decision_type": "coherence_driven",
  "reasoning": "short evidence-based reason",
  "risk": "main risk and mitigation"
}
```

The constraint layer then enforces legality and stability:

- weights sum to 1;
- each weight is in `[0.05, 0.80]`;
- large L1 changes from the previous accepted weights are projected back;
- clients with negative coherence cannot exceed their size weight;
- clients with extreme update norm cannot exceed their size weight;
- parse or API failure falls back to `COHERENCE_FEDYOGI_TR`.

These constraints do not choose the objective for the LLM. They only prevent invalid or unstable execution.

## Comparison Protocol

Formal comparison keeps communication rounds, seed, model architecture, data split, and local epochs fixed. The primary comparison is:

```text
FedAvg
FedYogi-TR
COHERENCE_FEDYOGI_TR
LLM_GCA_FEDYOGI_TR
```

`VG_FEDYOGI_TR` and `MAS_VG_FEDYOGI_TR` remain available as historical or appendix methods.
