# MAS-VG-FedYogi-TR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build validation-guided continuous aggregation weight search for FedYogi-TR, then expose no-LLM and LLM-guided MAS variants.

**Architecture:** Add a focused candidate module that generates bounded client-weight candidates and applies deterministic validation gates. Extend FedYogi with preview-safe state cloning, then wire CentralAgent training modes that evaluate candidates on validation before applying one server update.

**Tech Stack:** Python, PyTorch, pandas, existing project experiment runner and unittest/pytest tests.

---

## File Structure

- Create `src/federated_learning/adaptive_candidates.py`: candidate data model, candidate generation, candidate scoring, and gate selection.
- Modify `src/federated_learning/server_optimizers.py`: add clone/restore and preview step APIs without changing existing step behavior.
- Modify `src/federated_learning/mas_agents.py`: add validation-guided training loops, candidate preview evaluation, candidate logs.
- Modify `src/federated_learning/llm_planner.py`: add candidate-selection prompt/parse method while preserving legacy strategy method.
- Modify `experiments/scenario_C_llm.py`: add CLI flags and route `validation_guided` / `mas_validation_guided` modes.
- Modify `src/experiment_names.py`: add `VG_FEDYOGI_TR` and `MAS_VG_FEDYOGI_TR`, keep old `MAS_ADAPTIVE` out of new main order if needed.
- Modify `scripts/run_multi_seed.py`: add new scenarios, remove old `MAS_ADAPTIVE` from default formal run list.
- Add `tests/test_adaptive_candidates.py`: unit tests for candidate generation and gate.
- Extend `tests/test_server_optimizers.py`: preview state tests.
- Extend `tests/test_scenario_c_cli.py` and `tests/test_multi_seed_config.py`: CLI and runner tests.

## Task 1: Candidate Generation And Gate

**Files:**
- Create: `src/federated_learning/adaptive_candidates.py`
- Test: `tests/test_adaptive_candidates.py`

- [ ] **Step 1: Write failing tests**

Create tests covering:
- candidates sum to 1 and respect bounds;
- candidate count stays within budget;
- anchors include `size_anchor`, `uniform_anchor`, and `previous_accepted`;
- validation gate falls back when LLM chooses invalid or worse candidate;
- epsilon prefers conservative candidate.

- [ ] **Step 2: Run candidate tests and verify failure**

Run: `python -m pytest tests/test_adaptive_candidates.py -q`
Expected: fail because module does not exist.

- [ ] **Step 3: Implement candidate module**

Implement:
- `AdaptiveCandidate` dataclass;
- `generate_weight_candidates(client_ids, size_weights, previous_weights, client_metrics, budget, step, min_weight, max_weight)`;
- `score_candidate_metrics(metrics, client_gap, update_norm, weights, previous_weights, profile)`;
- `select_candidate_by_gate(candidates, conservative_candidate_id, requested_candidate_id, epsilon, weight_l1_limit, large_improvement_threshold)`.

- [ ] **Step 4: Run candidate tests**

Run: `python -m pytest tests/test_adaptive_candidates.py -q`
Expected: pass.

## Task 2: FedYogi Preview State

**Files:**
- Modify: `src/federated_learning/server_optimizers.py`
- Test: `tests/test_server_optimizers.py`

- [ ] **Step 1: Add failing tests**

Add tests proving:
- `preview_step` returns the same next state as `step` from the same optimizer state;
- `preview_step` does not mutate `m` or `v`;
- `get_optimizer_state` / `load_optimizer_state` round-trip tensors.

- [ ] **Step 2: Run targeted tests and verify failure**

Run: `python -m pytest tests/test_server_optimizers.py -q`
Expected: fail on missing preview APIs.

- [ ] **Step 3: Implement preview APIs**

Add methods to FedAvg and FedYogi optimizers:
- `get_optimizer_state()`;
- `load_optimizer_state(state)`;
- `preview_step(current_state, weighted_average_state, server_lr_scale=1.0)`.

FedYogi preview must save internal `m/v`, call `step`, then restore `m/v`.

- [ ] **Step 4: Run server optimizer tests**

Run: `python -m pytest tests/test_server_optimizers.py -q`
Expected: pass.

## Task 3: CentralAgent Validation-Guided Training

**Files:**
- Modify: `src/federated_learning/mas_agents.py`
- Test: `tests/test_adaptive_candidates.py`

- [ ] **Step 1: Add tests for candidate preview privacy and gate behavior**

Use lightweight fake metrics/states where possible. Verify candidate previews do not include `test_mape`, `True_Value`, or sample-level fields.

- [ ] **Step 2: Implement central methods**

Add methods:
- `build_continuous_candidate_preview(...)`;
- `select_validation_guided_candidate(...)`;
- `run_training_with_validation_guided_adaptation(...)`;
- `run_training_with_mas_validation_guided_adaptation(...)`.

The MAS method should call the new LLM candidate selector and then apply validation gate.

- [ ] **Step 3: Add candidate logs**

Save per-round candidate decisions into round history and `get_training_history_df` fields:
- `selected_candidate_id`;
- `requested_candidate_id`;
- `gate_status`;
- `candidate_score`;
- `candidate_budget`;
- `weight_l1_from_previous`;
- per-client selected weights.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_adaptive_candidates.py tests/test_llm_decision_inputs.py -q`
Expected: pass.

## Task 4: LLM Candidate Selector

**Files:**
- Modify: `src/federated_learning/llm_planner.py`
- Test: `tests/test_llm_decision_inputs.py`

- [ ] **Step 1: Add parser tests**

Test that candidate selector:
- parses `selected_candidate_id`;
- falls back to best validation candidate when invalid JSON;
- logs only prompt excerpts and not test/sample fields.

- [ ] **Step 2: Implement `choose_candidate`**

Add prompt builder and parser for candidate IDs. Keep legacy `choose_strategy` unchanged for old runs.

- [ ] **Step 3: Run LLM tests**

Run: `python -m pytest tests/test_llm_decision_inputs.py -q`
Expected: pass.

## Task 5: Experiment CLI And Method Names

**Files:**
- Modify: `experiments/scenario_C_llm.py`
- Modify: `src/experiment_names.py`
- Test: `tests/test_scenario_c_cli.py`, `tests/test_adaptive_method_outputs.py`

- [ ] **Step 1: Add failing CLI/name tests**

Assert:
- CLI has `--adaptive_mode`;
- new method names resolve;
- old `MAS_ADAPTIVE` is not used for new validation-guided outputs.

- [ ] **Step 2: Implement CLI routing**

Add flags:
- `--adaptive_mode fixed_strategy|validation_guided|mas_validation_guided`;
- `--candidate_budget`;
- `--weight_grid_step`;
- `--min_client_weight`;
- `--max_client_weight`;
- `--selection_epsilon`;
- `--weight_l1_change_limit`.

Route:
- `validation_guided` -> `VG_FEDYOGI_TR`, prefix `vg_fedyogi_tr`;
- `mas_validation_guided` -> `MAS_VG_FEDYOGI_TR`, prefix `mas_vg_fedyogi_tr`.

- [ ] **Step 3: Save metadata**

Add candidate parameters to result CSV and model info.

- [ ] **Step 4: Run CLI/name tests**

Run: `python -m pytest tests/test_scenario_c_cli.py tests/test_adaptive_method_outputs.py -q`
Expected: pass.

## Task 6: Multi-Seed Runner Integration

**Files:**
- Modify: `scripts/run_multi_seed.py`
- Test: `tests/test_multi_seed_config.py`

- [ ] **Step 1: Add failing runner tests**

Assert:
- default experiment order includes `VG_FEDYOGI_TR` and `MAS_VG_FEDYOGI_TR`;
- `MAS_ADAPTIVE` is not part of default formal run list;
- result files are `vg_fedyogi_tr_results.csv` and `mas_vg_fedyogi_tr_results.csv`.

- [ ] **Step 2: Implement runner changes**

Add new configs using current pilot recommendation for server lr. Keep old `MAS_ADAPTIVE` callable only if explicitly requested or remove from default order.

- [ ] **Step 3: Run runner tests**

Run: `python -m pytest tests/test_multi_seed_config.py -q`
Expected: pass.

## Task 7: Full Verification

**Files:**
- All touched files.

- [ ] **Step 1: Run full tests**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Smoke test VG no-LLM**

Run:
`python experiments/scenario_C_llm.py --num_rounds 2 --server_optimizer fedyogi --server_lr 0.4 --adaptive_mode validation_guided --output_prefix smoke_vg_fedyogi_tr --method_key VG_FEDYOGI_TR --seed 42`

Expected: writes `results/smoke_vg_fedyogi_tr_results.csv`.

- [ ] **Step 3: Smoke test MAS validation-guided with low round count**

Run only if API key/environment is available:
`python experiments/scenario_C_llm.py --num_rounds 2 --use_llm --temperature 0 --server_optimizer fedyogi --server_lr 0.4 --adaptive_mode mas_validation_guided --output_prefix smoke_mas_vg_fedyogi_tr --method_key MAS_VG_FEDYOGI_TR --seed 42`

Expected: writes `results/smoke_mas_vg_fedyogi_tr_results.csv` and an LLM decisions log without sample/test fields.

If API is unavailable, run parser/unit tests and report that LLM smoke was not run.
