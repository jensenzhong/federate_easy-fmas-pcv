# Fixed Seed 42 30-Round Tuning Record

Date: 2026-06-26

Scope: code/functionality check only. No paper text, figures, or formal tables were changed.

## Purpose

This record answers the current code-side questions before paper writing:

- Whether increasing federated training to 30 rounds improves the fixed split result.
- Whether a more open MAS/LLM controller contributes more meaningfully.
- Whether centralized and federated results are still comparable under the current setup.
- What should be changed next if the innovation method does not improve test MAPE.

All runs below use the same fixed split/seed first (`seed=42`). Multi-seed stability should be tested only after the single-seed direction is acceptable.

## Baseline Positioning

- `A` / GBR is a traditional machine-learning baseline. It is useful as a reference because Gradient Boosting is a strong tabular-data model, but it is not the same-model centralized upper bound for the federated neural model.
- `A_prime` / centralized MLP is the fair same-architecture centralized baseline. It uses the same global test set and trains on the union of federated local training samples, while local validation samples are still withheld.
- `B`, `FEDYOGI`, `VG_FEDYOGI_TR`, and `MAS_VG_FEDYOGI_TR` are federated neural methods. They are comparable to `A_prime` as different training regimes under the same fixed data split, but they are not optimizer-equivalent to centralized MLP training.
- Therefore, the current comparison is acceptable if it is written as "centralized same-model reference vs federated learning regimes", not as a strict theorem that centralized and FL optimization are equivalent.

## Commands Run

Existing fixed-seed rerun after checkpoint fix:

```powershell
python scripts/run_multi_seed.py --scenarios A A_prime B FEDYOGI VG_FEDYOGI_TR MAS_VG_FEDYOGI_TR --seeds 42
```

30-round FedAvg:

```powershell
python experiments/scenario_B_fedavg.py --num_rounds 30 --seed 42 --output_prefix tuning_b30_seed42
```

30-round FedYogi:

```powershell
python experiments/scenario_C_llm.py --num_rounds 30 --seed 42 --strategy size_only --server_optimizer fedyogi --server_lr 0.5 --max_coordinate_step_ratio 1.0 --output_prefix tuning_fedyogi30_seed42 --method_key FEDYOGI
```

30-round VG with wider candidate search:

```powershell
python experiments/scenario_C_llm.py --num_rounds 30 --seed 42 --strategy size_only --server_optimizer fedyogi --server_lr 0.5 --max_coordinate_step_ratio 1.0 --adaptive_mode validation_guided --candidate_budget 60 --weight_grid_step 0.025 --selection_epsilon 0.005 --weight_l1_change_limit 0.2 --output_prefix tuning_vg30_seed42 --method_key VG_FEDYOGI_TR
```

30-round MAS-open with wider LLM decision room:

```powershell
python experiments/scenario_C_llm.py --num_rounds 30 --seed 42 --use_llm --temperature 0 --server_optimizer fedyogi --server_lr 0.5 --max_coordinate_step_ratio 1.0 --adaptive_mode mas_validation_guided --candidate_budget 60 --weight_grid_step 0.025 --selection_epsilon 0.001 --llm_score_tolerance 0.01 --weight_l1_change_limit 0.8 --output_prefix tuning_mas30_open_seed42 --method_key MAS_VG_FEDYOGI_TR
```

## Fixed Seed 42 Results

Raw test metrics are the first priority here. Corrected metrics are diagnostic only because validation-set bias correction sometimes improves RMSE/R2 while worsening MAPE.

| Method | Rounds | Best Round | Best Val MAPE | Test MAPE | Test RMSE | Test MAE | Test R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A / GBR | n/a | n/a | 45.21% | 55.05% | 1,390,877 | 1,049,122 | n/a |
| A_prime / centralized MLP | n/a | epoch 84 | 32.16% | 43.02% | 1,484,185 | 1,039,525 | 0.4817 |
| B / FedAvg, original | 20 | 20 | n/a | 42.07% | 1,700,183 | 1,198,033 | 0.3198 |
| FEDYOGI, original | 20 | 20 | 40.19% | 42.14% | 1,709,283 | 1,220,794 | 0.3125 |
| VG_FEDYOGI_TR, original | 20 | 20 | 39.46% | 43.28% | 1,729,997 | 1,232,105 | 0.2957 |
| MAS_VG_FEDYOGI_TR, original | 20 | 20 | 39.91% | 42.55% | 1,666,849 | 1,188,232 | 0.3462 |
| B / FedAvg, tuning | 30 | 25 | 38.44% | 44.46% | 1,436,502 | 1,026,089 | 0.5144 |
| FEDYOGI, tuning | 30 | 28 | 36.21% | 46.57% | 1,411,562 | 1,041,428 | 0.5311 |
| VG_FEDYOGI_TR, tuning | 30 | 22 | 36.12% | 46.01% | 1,445,111 | 1,044,217 | 0.5086 |
| MAS-open, tuning | 30 | 22 | 37.07% | 46.67% | 1,514,512 | 1,127,428 | 0.4603 |

## Immediate Interpretation

Increasing to 30 rounds did not improve raw test MAPE on this fixed split. It did improve RMSE/MAE/R2 in several runs, especially FedAvg/FedYogi/VG, which means the models are not simply failing to train. The problem is metric alignment: the best validation MAPE checkpoint and the raw test MAPE are not moving in the same direction.

The best current raw MAPE among neural methods is still the 20-round FedAvg/FedYogi range:

- FedAvg 20 rounds: 42.07%.
- FedYogi 20 rounds: 42.14%.
- MAS 20 rounds: 42.55%.
- Centralized MLP: 43.02%.

So the earlier centralized concern has been mostly resolved after the checkpoint-copy bug fix: centralized MLP is no longer obviously worse than all FL methods. Small differences around 42-43% on one split should not be over-interpreted until multi-seed stability is checked.

## MAS-Open Behavior

The more open MAS setting did make the LLM contribute more:

- `accepted`: 17 rounds.
- `accepted_llm_near_best`: 11 rounds.
- `fallback_conservative_epsilon`: 2 rounds.

So the LLM-requested decision was effectively accepted in 28/30 rounds. This is much more autonomous than the conservative setting.

The LLM also changed client weights meaningfully. Around rounds 16-19 it pushed Client 1 down to the minimum boundary and shifted most weight to Clients 2 and 3; later it moved toward Client 1 again. This fits the multi-agent collaboration idea because client-side validation evidence, server-side preview scoring, and central LLM policy selection are interacting rather than using fixed size-based aggregation.

However, this extra autonomy did not improve test MAPE. The likely reason is validation overfitting or validation-objective mismatch. The LLM was acting on validation-only candidate previews, and the wider tolerance allowed it to choose more aggressive candidates that looked acceptable on validation but did not generalize to test MAPE.

## Current Reliability Judgment

- Code correctness is improved after fixing the checkpoint shallow-copy bug in centralized MLP and model training utilities.
- The current fixed-split result is usable for development diagnosis, but not yet reliable enough for final claims.
- The innovation method is conceptually present: it uses federated learning, validation-guided candidate generation, server-side FedYogi-style updates, and MAS/LLM policy selection.
- The innovation method is not yet empirically strong on raw MAPE. It currently improves some error-scale metrics under 30 rounds but does not improve the headline MAPE.
- Multi-seed testing should wait until the single-seed method is more convincing, otherwise it will only measure an under-tuned method.

## Recommended Next Modifications

Priority 1: improve selection/checkpoint objective.

- Add a composite checkpoint score instead of pure global validation MAPE.
- Candidate score should combine validation MAPE, RMSE, absolute MPE/bias, client gap, update norm, and weight-shift penalty.
- Keep raw MAPE as the headline test metric, but use the composite score to avoid selecting checkpoints that are good on validation percentage error yet worse on test distribution.

Priority 2: make MAS autonomy structured rather than merely wider.

- Keep LLM inside a near-best validation gate.
- Let LLM choose among explicit roles/objectives: `mape_focus`, `rmse_focus`, `bias_control`, `client_balance`, `stability_guard`.
- Log the chosen role each round and use it to select candidate scoring weights.
- This is more defensible as multi-agent collaboration than simply increasing `llm_score_tolerance` and `weight_l1_change_limit`.

Priority 3: add stronger candidate types.

- Add a `performance_anchor` candidate based on inverse client validation MAPE.
- Add a `bias_anchor` candidate based on client validation MPE compensation.
- Add a `stability_anchor` candidate that limits update norm and weight drift.
- Current candidates can change weights, but they do not sufficiently encode "which client is currently generalizing well" as a first-class candidate.

Priority 4: keep 30 rounds as a diagnostic, not default.

- For headline MAPE, 20 rounds is currently better.
- For RMSE/R2, 30 rounds is better.
- The next code change should target selection/generalization, not blindly increasing rounds further.

## Files Produced By This Step

- `results/tuning_fedyogi30_seed42_results.csv`
- `results/tuning_vg30_seed42_results.csv`
- `results/tuning_mas30_open_seed42_results.csv`
- `results/logs/tuning_b30_seed42_round_metrics.csv`
- `results/logs/tuning_fedyogi30_seed42_round_metrics.csv`
- `results/logs/tuning_vg30_seed42_round_metrics.csv`
- `results/logs/tuning_mas30_open_seed42_round_metrics.csv`
- `results/logs/tuning_mas30_open_seed42_llm_decisions.jsonl`

