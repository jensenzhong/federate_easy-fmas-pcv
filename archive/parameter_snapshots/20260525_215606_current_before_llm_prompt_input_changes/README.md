# Parameter Snapshot: current_before_llm_prompt_input_changes

Created: 2026-05-25 21:56:06 Asia/Shanghai
Workspace: C:\Users\23079\Desktop\科研\federate_easy - 重新搞点图片

Purpose: preserve the current reproducible parameter/code state before changing the LLM prompt and LLM input diagnostics.

## Saved Files

- configs/config.yaml
- experiments/scenario_C_llm.py
- src/federated_learning/llm_planner.py
- src/federated_learning/mas_agents.py
- scripts/run_multi_seed.py
- scripts/run_ablation.py
- tests/test_scenario_c_cli.py, if present
- current_uncommitted_code_diff.patch

## Key Current Parameters

- Scenario B/C num_rounds: 20
- Scenario B/C local_epochs: 20
- Scenario B/C learning_rate: 0.0005
- Scenario B fedprox_mu: 0.0 in scene_b
- Global federated_learning.fedprox_mu: 0.01
- Scenario C default_strategy: size_only
- Scenario C hybrid lambda_hybrid: 0.5
- Scenario C fairness_clip lambda_hybrid: 0.3
- Scenario C fairness_clip alpha_min: 0.1
- Scenario C fairness_clip alpha_max: 0.6
- LLM provider: deepseek
- LLM call_every_n_rounds: 1
- LLM recent_rounds: 10
- LLM max_tokens: 1024
- LLM config temperature: 0.8
- LLM timeout: 60
- CLI supports --temperature override in experiments/scenario_C_llm.py

## Restore Notes

To restore these files manually, copy the saved files back to their matching project paths.
This snapshot intentionally does not overwrite or delete current results.

Current temp=0 LLM seed validation artifacts are stored separately under:
results/temp0_llm_seed_runs/

## Important Boundary

Do not use test-set results inside LLM prompts or training-time decisions. Future LLM prompt/input changes should use validation-only diagnostics and federated aggregate statistics.
