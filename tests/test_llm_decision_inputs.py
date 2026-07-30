import json
import tempfile
import unittest
from pathlib import Path


def _planner():
    from src.federated_learning.llm_planner import LLMPlanner

    class DummyClient:
        pass

    config = {
        "scene_c": {
            "llm": {},
            "strategies": [
                {"name": "size_only"},
                {"name": "perf_only"},
                {"name": "hybrid", "lambda_hybrid": 0.5},
                {"name": "fairness_clip", "lambda_hybrid": 0.3, "alpha_min": 0.1, "alpha_max": 0.6},
            ],
        }
    }
    tmp = tempfile.TemporaryDirectory()
    planner = LLMPlanner(config=config, llm_client=DummyClient(), log_dir=tmp.name)
    planner._tmp = tmp
    return planner


class LLMDecisionInputTests(unittest.TestCase):
    def test_decision_log_name_can_be_overridden(self):
        from src.federated_learning.llm_planner import LLMPlanner

        class DummyClient:
            pass

        tmp = tempfile.TemporaryDirectory()
        planner = LLMPlanner(
            config={"scene_c": {"llm": {}, "strategies": [{"name": "size_only"}]}},
            llm_client=DummyClient(),
            log_dir=tmp.name,
            decisions_log_name="mas_adaptive_llm_decisions.jsonl",
        )

        self.assertEqual(planner.decisions_log_path.name, "mas_adaptive_llm_decisions.jsonl")
        self.assertTrue((Path(tmp.name) / "mas_adaptive_llm_decisions.jsonl").exists())
        tmp.cleanup()

    def test_prompt_uses_validation_only_evidence_without_test_set_terms(self):
        planner = _planner()
        history = [
            {
                "round": 0,
                "strategy_name": "size_only",
                "lr": 0.0005,
                "local_epochs": 20,
                "aggregation_weights": {"Client 1": 0.33, "Client 2": 0.34, "Client 3": 0.33},
                "client_metrics": {
                    "Client 1": {"n_samples": 10, "train_loss": 1.2, "val_mape": 0.8, "val_rmse": 100.0, "val_mae": 80.0, "val_mpe": -0.1},
                    "Client 2": {"n_samples": 12, "train_loss": 1.1, "val_mape": 0.7, "val_rmse": 90.0, "val_mae": 70.0, "val_mpe": -0.05},
                    "Client 3": {"n_samples": 8, "train_loss": 1.3, "val_mape": 0.9, "val_rmse": 110.0, "val_mae": 85.0, "val_mpe": -0.12},
                },
                "global_val": {"mape": 0.82, "rmse": 100.0, "mae": 80.0, "mpe": -0.09, "r2": 0.1},
            },
            {
                "round": 1,
                "strategy_name": "perf_only",
                "lr": 0.0005,
                "local_epochs": 20,
                "aggregation_weights": {"Client 1": 0.31, "Client 2": 0.36, "Client 3": 0.33},
                "client_metrics": {
                    "Client 1": {"n_samples": 10, "train_loss": 1.0, "val_mape": 0.7, "val_rmse": 90.0, "val_mae": 70.0, "val_mpe": -0.08},
                    "Client 2": {"n_samples": 12, "train_loss": 0.9, "val_mape": 0.6, "val_rmse": 80.0, "val_mae": 60.0, "val_mpe": -0.04},
                    "Client 3": {"n_samples": 8, "train_loss": 1.1, "val_mape": 0.8, "val_rmse": 100.0, "val_mae": 75.0, "val_mpe": -0.1},
                },
                "global_val": {"mape": 0.72, "rmse": 88.0, "mae": 68.0, "mpe": -0.07, "r2": 0.2},
            },
        ]
        decision_context = {
            "candidate_weight_preview": {
                "size_only": {"Client 1": 0.33, "Client 2": 0.40, "Client 3": 0.27},
                "perf_only": {"Client 1": 0.32, "Client 2": 0.37, "Client 3": 0.31},
            },
            "candidate_validation_preview": {
                "perf_only": {"global_val_mape": 0.70, "client_mape_gap": 0.15}
            },
        }

        prompt = planner.build_prompt(history, current_round=2, num_rounds=20, decision_context=decision_context)
        lowered = prompt.lower()

        self.assertNotIn("test", lowered)
        self.assertNotIn("测试集", prompt)
        self.assertNotIn("适用", prompt)
        self.assertIn("candidate_weight_preview", prompt)
        self.assertIn("marginal_effect_history", prompt)
        self.assertIn("validation", lowered)

    def test_parse_response_clamps_dynamic_parameters_and_preserves_evidence(self):
        planner = _planner()
        response = json.dumps({
            "chosen_strategy_name": "fairness_clip",
            "lr_scale": 4.0,
            "epoch_delta": 99,
            "lambda_hybrid": 1.7,
            "alpha_min": 0.9,
            "alpha_max": 0.2,
            "evidence": ["validation delta improved"],
            "risk": "weight concentration",
            "reasoning": "validation-only evidence",
        })

        decision = planner.parse_response(response)

        self.assertEqual(decision["chosen_strategy_name"], "fairness_clip")
        self.assertEqual(decision["lr_scale"], 2.0)
        self.assertEqual(decision["epoch_delta"], 5)
        self.assertEqual(decision["lambda_hybrid"], 1.0)
        self.assertLessEqual(decision["alpha_min"], decision["alpha_max"])
        self.assertEqual(decision["alpha_max"], 0.9)
        self.assertEqual(decision["evidence"], ["validation delta improved"])
        self.assertEqual(decision["risk"], "weight concentration")

    def test_choose_strategy_passes_decision_context_into_prompt(self):
        from src.federated_learning.llm_planner import LLMPlanner

        class CapturingClient:
            def __init__(self):
                self.prompt = ""

            def generate(self, prompt, **kwargs):
                self.prompt = prompt
                return json.dumps({
                    "chosen_strategy_name": "hybrid",
                    "lr_scale": 1.0,
                    "epoch_delta": 0,
                    "lambda_hybrid": 0.6,
                    "alpha_min": 0.1,
                    "alpha_max": 0.6,
                    "evidence": ["preview marker"],
                    "risk": "none",
                    "reasoning": "uses supplied validation preview",
                })

        client = CapturingClient()
        tmp = tempfile.TemporaryDirectory()
        planner = LLMPlanner(
            config={
                "scene_c": {
                    "llm": {},
                    "strategies": [{"name": "size_only"}, {"name": "hybrid"}],
                }
            },
            llm_client=client,
            log_dir=tmp.name,
        )

        planner.choose_strategy(
            history_round_metrics=[],
            current_round=4,
            num_rounds=20,
            decision_context={"candidate_validation_preview": {"hybrid": {"preview_marker": 123}}},
        )

        self.assertIn("preview_marker", client.prompt)
        tmp.cleanup()

    def test_choose_candidate_parses_selected_candidate_id(self):
        from src.federated_learning.llm_planner import LLMPlanner

        class CapturingClient:
            def __init__(self):
                self.prompt = ""

            def generate(self, prompt, **kwargs):
                self.prompt = prompt
                return json.dumps({
                    "selected_candidate_id": "candidate_001",
                    "objective_profile": {
                        "primary": "mape",
                        "secondary": ["rmse", "client_gap", "mpe_bias"],
                        "risk_tolerance": "balanced",
                    },
                    "reasoning": "candidate_001 has similar MAPE and better bias control",
                    "risk": "small validation advantage",
                })

        candidate_preview = {
            "size_anchor": {
                "score": 0.410,
                "validation_metrics": {"mape": 0.410, "rmse": 120.0, "mae": 90.0, "mpe": -0.08},
                "client_gap": 0.10,
                "weights": {"Client 1": 0.4, "Client 2": 0.3, "Client 3": 0.3},
            },
            "candidate_001": {
                "score": 0.405,
                "validation_metrics": {"mape": 0.405, "rmse": 115.0, "mae": 88.0, "mpe": -0.02},
                "client_gap": 0.08,
                "weights": {"Client 1": 0.35, "Client 2": 0.35, "Client 3": 0.3},
            },
        }

        client = CapturingClient()
        tmp = tempfile.TemporaryDirectory()
        planner = LLMPlanner(
            config={"scene_c": {"llm": {"temperature": 0.0}, "strategies": [{"name": "size_only"}]}},
            llm_client=client,
            log_dir=tmp.name,
            decisions_log_name="mas_vg_llm_decisions.jsonl",
        )

        decision = planner.choose_candidate(
            history_round_metrics=[],
            current_round=3,
            num_rounds=20,
            candidate_preview=candidate_preview,
        )

        self.assertEqual(decision["selected_candidate_id"], "candidate_001")
        self.assertEqual(decision["fallback_used"], False)
        self.assertIn("candidate_001", client.prompt)
        self.assertIn("validation", client.prompt.lower())

        log_line = (Path(tmp.name) / "mas_vg_llm_decisions.jsonl").read_text(encoding="utf-8").strip()
        self.assertIn("candidate_001", log_line)
        tmp.cleanup()

    def test_client_summaries_are_aggregate_only_and_bucketed(self):
        from src.federated_learning.client_summaries import build_client_summaries

        summaries = build_client_summaries(
            diagnostics={
                "client_1": {
                    "sample_size_weight": 0.6,
                    "val_mape": 0.25,
                    "val_mpe": -0.04,
                    "cosine_to_mean_update": 0.8,
                    "update_norm": 2.0,
                },
                "client_2": {
                    "sample_size_weight": 0.2,
                    "val_mape": 0.7,
                    "val_mpe": 0.05,
                    "cosine_to_mean_update": -0.2,
                    "update_norm": 0.3,
                },
                "client_3": {
                    "sample_size_weight": 0.2,
                    "val_mape": 0.45,
                    "val_mpe": 0.0,
                    "cosine_to_mean_update": 0.2,
                    "update_norm": 1.0,
                },
            },
            previous_weights={"client_1": 0.5, "client_2": 0.3, "client_3": 0.2},
        )

        self.assertEqual(summaries["client_1"]["sample_share_level"], "high")
        self.assertEqual(summaries["client_2"]["coherence_level"], "negative")
        self.assertEqual(summaries["client_1"]["recent_weight_trend"], "increasing")

        def walk_keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield str(key).lower()
                    yield from walk_keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from walk_keys(child)

        for forbidden in ["test", "true_value", "predicted_value", "sample", "feature", "target"]:
            self.assertNotIn(forbidden, set(walk_keys(summaries)))

        serialized = json.dumps(summaries, ensure_ascii=False).lower()
        self.assertIn("summary_text", summaries["client_1"])
        self.assertNotIn("0.600000", serialized)

    def test_candidate_prompt_marks_near_best_and_balanced_recommendation(self):
        planner = _planner()
        candidate_preview = {
            "candidate_best": {
                "score": 0.5000,
                "validation_metrics": {"mape": 0.5000, "rmse": 150.0, "mpe": -0.20},
                "client_gap": 0.20,
                "update_norm": 5.0,
            },
            "candidate_balanced": {
                "score": 0.5005,
                "validation_metrics": {"mape": 0.5005, "rmse": 100.0, "mpe": -0.02},
                "client_gap": 0.05,
                "update_norm": 3.0,
            },
            "candidate_far": {
                "score": 0.5200,
                "validation_metrics": {"mape": 0.5200, "rmse": 80.0, "mpe": 0.0},
                "client_gap": 0.01,
                "update_norm": 1.0,
            },
        }

        prompt = planner.build_candidate_prompt(
            history_round_metrics=[],
            current_round=0,
            num_rounds=20,
            candidate_preview=candidate_preview,
            score_tolerance=0.001,
        )

        self.assertIn('"near_best_candidate_ids"', prompt)
        self.assertIn('"candidate_best"', prompt)
        self.assertIn('"candidate_balanced"', prompt)
        self.assertNotIn('"candidate_far",\n      "candidate_far"', prompt)
        self.assertIn('"balanced_recommended_candidate_id": "candidate_balanced"', prompt)

    def test_choose_candidate_invalid_response_falls_back_to_best_score(self):
        from src.federated_learning.llm_planner import LLMPlanner

        class InvalidClient:
            def generate(self, prompt, **kwargs):
                return "not json"

        candidate_preview = {
            "size_anchor": {"score": 0.410, "validation_metrics": {"mape": 0.410}},
            "candidate_001": {"score": 0.405, "validation_metrics": {"mape": 0.405}},
        }

        tmp = tempfile.TemporaryDirectory()
        planner = LLMPlanner(
            config={"scene_c": {"llm": {}, "strategies": [{"name": "size_only"}]}},
            llm_client=InvalidClient(),
            log_dir=tmp.name,
        )

        decision = planner.choose_candidate(
            history_round_metrics=[],
            current_round=0,
            num_rounds=20,
            candidate_preview=candidate_preview,
        )

        self.assertEqual(decision["selected_candidate_id"], "candidate_001")
        self.assertEqual(decision["fallback_used"], True)
        self.assertEqual(decision["fallback_candidate_id"], "candidate_001")
        tmp.cleanup()

    def test_candidate_prompt_and_log_do_not_include_test_or_sample_fields(self):
        from src.federated_learning.llm_planner import LLMPlanner

        class CapturingClient:
            def __init__(self):
                self.prompt = ""

            def generate(self, prompt, **kwargs):
                self.prompt = prompt
                return json.dumps({"selected_candidate_id": "size_anchor"})

        candidate_preview = {
            "size_anchor": {
                "score": 0.410,
                "validation_metrics": {"mape": 0.410},
                "test_mape": 0.111,
                "True_Value": [1, 2],
                "Predicted_Value": [1.1, 2.2],
                "sample_predictions": [{"target": 1, "feature": 2, "label": 3}],
                "metadata": {"raw_features": [1, 2, 3]},
            }
        }

        client = CapturingClient()
        tmp = tempfile.TemporaryDirectory()
        planner = LLMPlanner(
            config={"scene_c": {"llm": {}, "strategies": [{"name": "size_only"}]}},
            llm_client=client,
            log_dir=tmp.name,
        )

        planner.choose_candidate(
            history_round_metrics=[],
            current_round=0,
            num_rounds=20,
            candidate_preview=candidate_preview,
        )

        log_text = (Path(tmp.name) / "scene_C_llm_decisions.jsonl").read_text(encoding="utf-8")
        combined = (client.prompt + "\n" + log_text).lower()
        for forbidden in ["test_mape", "true_value", "predicted_value", "sample_predictions", "target", "feature", "label", "raw_features"]:
            self.assertNotIn(forbidden, combined)
        tmp.cleanup()


class CandidatePreviewTests(unittest.TestCase):
    def test_candidate_weight_preview_generates_normalized_weights(self):
        from src.federated_learning.mas_agents import CentralAgent

        agent = CentralAgent.__new__(CentralAgent)
        agent.strategies_config = {
            "size_only": {},
            "perf_only": {},
            "hybrid": {"lambda_hybrid": 0.5},
            "fairness_clip": {"lambda_hybrid": 0.3, "alpha_min": 0.1, "alpha_max": 0.6},
        }
        metrics = {
            "Client 1": {"n_samples": 10, "val_mape": 0.5},
            "Client 2": {"n_samples": 20, "val_mape": 0.4},
            "Client 3": {"n_samples": 30, "val_mape": 0.8},
        }

        preview = agent.build_candidate_weight_preview(metrics)

        self.assertEqual(set(preview), {"size_only", "perf_only", "hybrid", "fairness_clip"})
        for weights in preview.values():
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
            self.assertEqual(set(weights), set(metrics))

    def test_candidate_validation_preview_contains_only_aggregate_validation_metrics(self):
        from src.federated_learning.mas_agents import CentralAgent

        class DummyModel:
            def load_state_dict(self, state):
                self.state = state

        class DummyClientAgent:
            def __init__(self, n_val_samples, mape, rmse, mae, mpe):
                self.n_val_samples = n_val_samples
                self.metrics = {"mape": mape, "rmse": rmse, "mae": mae, "mpe": mpe}

            def _create_local_model(self):
                return DummyModel()

            def _evaluate_on_val(self, model):
                return self.metrics

        agent = CentralAgent.__new__(CentralAgent)
        agent.strategies_config = {
            "size_only": {},
            "perf_only": {},
            "hybrid": {"lambda_hybrid": 0.5},
            "fairness_clip": {"lambda_hybrid": 0.3, "alpha_min": 0.1, "alpha_max": 0.6},
        }
        agent.client_agents = {
            "Client 1": DummyClientAgent(2, 0.5, 100.0, 80.0, -0.1),
            "Client 2": DummyClientAgent(3, 0.4, 90.0, 70.0, -0.05),
        }
        import torch
        client_states = {
            "Client 1": {"w": torch.tensor([1.0])},
            "Client 2": {"w": torch.tensor([3.0])},
        }
        metrics = {
            "Client 1": {"n_samples": 10, "val_mape": 0.5},
            "Client 2": {"n_samples": 20, "val_mape": 0.4},
        }

        preview = agent.build_candidate_validation_preview(client_states, metrics)
        serialized = json.dumps(preview).lower()

        self.assertEqual(set(preview), {"size_only", "perf_only", "hybrid", "fairness_clip"})
        self.assertIn("aggregate_val_mape", preview["size_only"])
        self.assertIn("client_metrics", preview["size_only"])
        self.assertNotIn("prediction", serialized)
        self.assertNotIn("target", serialized)
        self.assertNotIn("feature", serialized)
        self.assertNotIn("label", serialized)

    def test_mas_training_passes_score_tolerance_to_candidate_planner(self):
        import inspect
        from src.federated_learning.mas_agents import CentralAgent

        source = inspect.getsource(CentralAgent.run_training_with_mas_validation_guided_adaptation)

        self.assertIn("score_tolerance=llm_score_tolerance", source)
        self.assertIn("llm_score_tolerance = selection_epsilon if llm_score_tolerance is None", source)


if __name__ == "__main__":
    unittest.main()
