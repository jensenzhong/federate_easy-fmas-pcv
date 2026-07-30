import sys
import unittest
from unittest.mock import patch


class ScenarioCCliTests(unittest.TestCase):
    def test_temperature_argument_is_parsed(self):
        from experiments.scenario_C_llm import parse_args

        argv = ["scenario_C_llm.py", "--use_llm", "--temperature", "0"]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.temperature, 0.0)

    def test_adaptive_server_optimizer_arguments_are_parsed(self):
        from experiments.scenario_C_llm import parse_args

        argv = [
            "scenario_C_llm.py",
            "--use_llm",
            "--server_optimizer",
            "fedyogi",
            "--server_lr",
            "0.003",
            "--server_beta1",
            "0.8",
            "--server_beta2",
            "0.95",
            "--server_tau",
            "0.002",
            "--max_coordinate_step_ratio",
            "0.75",
            "--update_clip_norm",
            "1.5",
            "--output_prefix",
            "mas_adaptive",
            "--method_key",
            "MAS_ADAPTIVE",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.server_optimizer, "fedyogi")
        self.assertEqual(args.server_lr, 0.003)
        self.assertEqual(args.server_beta1, 0.8)
        self.assertEqual(args.server_beta2, 0.95)
        self.assertEqual(args.server_tau, 0.002)
        self.assertEqual(args.max_coordinate_step_ratio, 0.75)
        self.assertEqual(args.update_clip_norm, 1.5)
        self.assertEqual(args.output_prefix, "mas_adaptive")
        self.assertEqual(args.method_key, "MAS_ADAPTIVE")

    def test_validation_guided_arguments_are_parsed(self):
        from experiments.scenario_C_llm import parse_args

        argv = [
            "scenario_C_llm.py",
            "--adaptive_mode",
            "validation_guided",
            "--candidate_budget",
            "20",
            "--weight_grid_step",
            "0.1",
            "--min_client_weight",
            "0.05",
            "--max_client_weight",
            "0.8",
            "--selection_epsilon",
            "0.002",
            "--llm_score_tolerance",
            "0.003",
            "--weight_l1_change_limit",
            "0.4",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.adaptive_mode, "validation_guided")
        self.assertEqual(args.candidate_budget, 20)
        self.assertEqual(args.weight_grid_step, 0.1)
        self.assertEqual(args.min_client_weight, 0.05)
        self.assertEqual(args.max_client_weight, 0.8)
        self.assertEqual(args.selection_epsilon, 0.002)
        self.assertEqual(args.llm_score_tolerance, 0.003)
        self.assertEqual(args.weight_l1_change_limit, 0.4)

    def test_generative_coherence_adaptive_modes_are_parsed(self):
        from experiments.scenario_C_llm import parse_args, resolve_method_key, resolve_output_prefix

        argv = [
            "scenario_C_llm.py",
            "--adaptive_mode",
            "llm_generative_coherence",
            "--use_llm",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.adaptive_mode, "llm_generative_coherence")
        self.assertEqual(resolve_method_key(args), "LLM_GCA_FEDYOGI_TR")
        self.assertEqual(resolve_output_prefix(args, "LLM_GCA_FEDYOGI_TR"), "llm_gca_fedyogi_tr")

        argv = ["scenario_C_llm.py", "--adaptive_mode", "coherence_guided"]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(resolve_method_key(args), "COHERENCE_FEDYOGI_TR")
        self.assertEqual(resolve_output_prefix(args, "COHERENCE_FEDYOGI_TR"), "coherence_fedyogi_tr")

        argv = [
            "scenario_C_llm.py",
            "--adaptive_mode",
            "llm_validation_preview_generative",
            "--use_llm",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(resolve_method_key(args), "LLM_VP_GCA_FEDYOGI_TR")
        self.assertEqual(resolve_output_prefix(args, "LLM_VP_GCA_FEDYOGI_TR"), "llm_vp_gca_fedyogi_tr")

        argv = ["scenario_C_llm.py", "--adaptive_mode", "validation_preview_gca"]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(resolve_method_key(args), "VP_GCA_FEDYOGI_TR")
        self.assertEqual(resolve_output_prefix(args, "VP_GCA_FEDYOGI_TR"), "vp_gca_fedyogi_tr")

        argv = ["scenario_C_llm.py", "--adaptive_mode", "strict_coherence_guided"]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(resolve_method_key(args), "STRICT_COHERENCE_FEDYOGI_TR")
        self.assertEqual(resolve_output_prefix(args, "STRICT_COHERENCE_FEDYOGI_TR"), "strict_coherence_fedyogi_tr")

        argv = [
            "scenario_C_llm.py",
            "--adaptive_mode",
            "llm_strict_generative_coherence",
            "--use_llm",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(resolve_method_key(args), "LLM_STRICT_GCA_FEDYOGI_TR")
        self.assertEqual(resolve_output_prefix(args, "LLM_STRICT_GCA_FEDYOGI_TR"), "llm_strict_gca_fedyogi_tr")

    def test_llm_decision_log_uses_output_prefix(self):
        import experiments.scenario_C_llm as scenario_c

        class DummyCentralAgent:
            def run_training_with_llm(self, **kwargs):
                return {
                    "best_round": 0,
                    "best_val_mape": 0.5,
                    "test_metrics": {
                        "mape": 0.5,
                        "rmse": 1.0,
                        "mae": 1.0,
                        "mpe": 0.0,
                        "nrmse": 0.0,
                        "r2": 0.0,
                    },
                }

            def evaluate_global(self, *args, **kwargs):
                return {
                    "mape": 0.5,
                    "rmse": 1.0,
                    "mae": 1.0,
                    "mpe": 0.0,
                    "nrmse": 0.0,
                    "r2": 0.0,
                }

            def calibrate_bias(self):
                self.bias_correction_value = 0.0

        class DummyLogger:
            def info(self, message):
                pass

        argv = [
            "scenario_C_llm.py",
            "--use_llm",
            "--server_optimizer",
            "fedyogi",
            "--output_prefix",
            "mas_adaptive",
            "--method_key",
            "MAS_ADAPTIVE",
        ]
        with patch.object(sys, "argv", argv):
            args = scenario_c.parse_args()

        config = {
            "scene_c": {
                "llm": {"default_provider": "deepseek", "temperature": 0.0},
                "learning_rate": 0.0005,
                "local_epochs": 1,
            },
            "federated_learning": {"client": {"learning_rate": 0.0005, "local_epochs": 1}},
            "output": {"logs_dir": "results/logs", "base_dir": "results", "models_dir": "results/models"},
        }

        with patch.object(scenario_c, "create_llm_planner", return_value=object()) as create_planner, \
             patch.object(scenario_c, "create_central_agent", return_value=(object(), DummyCentralAgent(), None, object())), \
             patch.object(scenario_c, "attach_bias_corrected_metrics", side_effect=lambda _agent, results, _loader: results), \
             patch.object(scenario_c, "save_federated_outputs"):
            scenario_c.run_with_llm(
                args=args,
                config=config,
                client_train_sets={},
                client_val_sets={},
                global_val_set=object(),
                global_test_set=object(),
                preprocessor=object(),
                logger=DummyLogger(),
                device="cpu",
                input_dim=10,
            )

        self.assertEqual(create_planner.call_args.kwargs["decisions_log_name"], "mas_adaptive_llm_decisions.jsonl")

    def test_coherence_guided_dispatch_uses_new_training_method(self):
        import experiments.scenario_C_llm as scenario_c

        class DummyCentralAgent:
            def __init__(self):
                self.called = None

            def run_training_with_coherence_guided_adaptation(self, **kwargs):
                self.called = kwargs
                return {
                    "best_round": 0,
                    "best_val_mape": 0.5,
                    "test_metrics": {
                        "mape": 0.5,
                        "rmse": 1.0,
                        "mae": 1.0,
                        "mpe": 0.0,
                        "nrmse": 0.0,
                        "r2": 0.0,
                    },
                }

            def evaluate_global(self, *args, **kwargs):
                return self.run_training_with_coherence_guided_adaptation()["test_metrics"]

            def calibrate_bias(self):
                self.bias_correction_value = 0.0

        class DummyLogger:
            def info(self, message):
                pass

        argv = ["scenario_C_llm.py", "--adaptive_mode", "coherence_guided"]
        with patch.object(sys, "argv", argv):
            args = scenario_c.parse_args()

        central_agent = DummyCentralAgent()
        config = {
            "scene_c": {"learning_rate": 0.0005, "local_epochs": 1},
            "federated_learning": {"client": {"learning_rate": 0.0005, "local_epochs": 1}},
            "preprocessing": {"random_seed": 42},
            "output": {"logs_dir": "results/logs", "base_dir": "results", "models_dir": "results/models"},
        }

        with patch.object(scenario_c, "create_central_agent", return_value=(object(), central_agent, None, object())), \
             patch.object(scenario_c, "attach_bias_corrected_metrics", side_effect=lambda _agent, results, _loader: results), \
             patch.object(scenario_c, "save_federated_outputs"):
            scenario_c.run_single_strategy(
                "size_only",
                config,
                {},
                {},
                object(),
                object(),
                object(),
                args,
                DummyLogger(),
                "cpu",
                10,
            )

        self.assertIsNotNone(central_agent.called)

    def test_llm_validation_preview_generative_dispatch_uses_new_training_method(self):
        import experiments.scenario_C_llm as scenario_c

        class DummyCentralAgent:
            def __init__(self):
                self.called = None

            def run_training_with_llm_validation_preview_generative(self, **kwargs):
                self.called = kwargs
                return {
                    "best_round": 0,
                    "best_val_mape": 0.5,
                    "test_metrics": {
                        "mape": 0.5,
                        "rmse": 1.0,
                        "mae": 1.0,
                        "mpe": 0.0,
                        "nrmse": 0.0,
                        "r2": 0.0,
                    },
                    "llm_decisions": [],
                }

            def evaluate_global(self, *args, **kwargs):
                return self.run_training_with_llm_validation_preview_generative()["test_metrics"]

            def calibrate_bias(self):
                self.bias_correction_value = 0.0

        class DummyLogger:
            def info(self, message):
                pass

        argv = [
            "scenario_C_llm.py",
            "--adaptive_mode",
            "llm_validation_preview_generative",
            "--use_llm",
            "--candidate_budget",
            "12",
        ]
        with patch.object(sys, "argv", argv):
            args = scenario_c.parse_args()

        central_agent = DummyCentralAgent()
        config = {
            "scene_c": {
                "llm": {"default_provider": "deepseek", "temperature": 0.0},
                "learning_rate": 0.0005,
                "local_epochs": 1,
            },
            "federated_learning": {"client": {"learning_rate": 0.0005, "local_epochs": 1}},
            "preprocessing": {"random_seed": 42},
            "output": {"logs_dir": "results/logs", "base_dir": "results", "models_dir": "results/models"},
        }

        with patch.object(scenario_c, "create_llm_planner", return_value=object()), \
             patch.object(scenario_c, "create_central_agent", return_value=(object(), central_agent, None, object())), \
             patch.object(scenario_c, "attach_bias_corrected_metrics", side_effect=lambda _agent, results, _loader: results), \
             patch.object(scenario_c, "save_federated_outputs"):
            scenario_c.run_single_strategy(
                "size_only",
                config,
                {},
                {},
                object(),
                object(),
                object(),
                args,
                DummyLogger(),
                "cpu",
                10,
            )

        self.assertIsNotNone(central_agent.called)
        self.assertEqual(central_agent.called["candidate_budget"], 12)

    def test_validation_preview_gca_dispatch_uses_non_llm_training_method(self):
        import experiments.scenario_C_llm as scenario_c

        class DummyCentralAgent:
            def __init__(self):
                self.called = None

            def run_training_with_validation_preview_gca(self, **kwargs):
                self.called = kwargs
                return {
                    "best_round": 0,
                    "best_val_mape": 0.5,
                    "test_metrics": {
                        "mape": 0.5,
                        "rmse": 1.0,
                        "mae": 1.0,
                        "mpe": 0.0,
                        "nrmse": 0.0,
                        "r2": 0.0,
                    },
                }

            def evaluate_global(self, *args, **kwargs):
                return self.run_training_with_validation_preview_gca()["test_metrics"]

            def calibrate_bias(self):
                self.bias_correction_value = 0.0

        class DummyLogger:
            def info(self, message):
                pass

        argv = [
            "scenario_C_llm.py",
            "--adaptive_mode",
            "validation_preview_gca",
            "--candidate_budget",
            "12",
        ]
        with patch.object(sys, "argv", argv):
            args = scenario_c.parse_args()

        central_agent = DummyCentralAgent()
        config = {
            "scene_c": {"learning_rate": 0.0005, "local_epochs": 1},
            "federated_learning": {"client": {"learning_rate": 0.0005, "local_epochs": 1}},
            "preprocessing": {"random_seed": 42},
            "output": {"logs_dir": "results/logs", "base_dir": "results", "models_dir": "results/models"},
        }

        with patch.object(scenario_c, "create_central_agent", return_value=(object(), central_agent, None, object())), \
             patch.object(scenario_c, "attach_bias_corrected_metrics", side_effect=lambda _agent, results, _loader: results), \
             patch.object(scenario_c, "save_federated_outputs"):
            scenario_c.run_single_strategy(
                "size_only",
                config,
                {},
                {},
                object(),
                object(),
                object(),
                args,
                DummyLogger(),
                "cpu",
                10,
            )

        self.assertIsNotNone(central_agent.called)
        self.assertEqual(central_agent.called["candidate_budget"], 12)

    def test_strict_gca_dispatch_uses_server_data_free_training_methods(self):
        import experiments.scenario_C_llm as scenario_c

        class DummyCentralAgent:
            def __init__(self):
                self.called = None

            def run_training_with_strict_coherence_guided_adaptation(self, **kwargs):
                self.called = ("strict", kwargs)
                return {
                    "best_round": 0,
                    "best_val_mape": 0.5,
                    "test_metrics": {
                        "mape": 0.5,
                        "rmse": 1.0,
                        "mae": 1.0,
                        "mpe": 0.0,
                        "nrmse": 0.0,
                        "r2": 0.0,
                    },
                }

            def run_training_with_llm_strict_generative_coherence(self, **kwargs):
                self.called = ("llm_strict", kwargs)
                return {
                    "best_round": 0,
                    "best_val_mape": 0.5,
                    "test_metrics": {
                        "mape": 0.5,
                        "rmse": 1.0,
                        "mae": 1.0,
                        "mpe": 0.0,
                        "nrmse": 0.0,
                        "r2": 0.0,
                    },
                    "llm_decisions": [],
                }

            def evaluate_global(self, *args, **kwargs):
                return self.run_training_with_strict_coherence_guided_adaptation()["test_metrics"]

            def calibrate_bias(self):
                self.bias_correction_value = 0.0

        class DummyLogger:
            def info(self, message):
                pass

        config = {
            "scene_c": {
                "llm": {"default_provider": "deepseek", "temperature": 0.0},
                "learning_rate": 0.0005,
                "local_epochs": 1,
            },
            "federated_learning": {"client": {"learning_rate": 0.0005, "local_epochs": 1}},
            "preprocessing": {"random_seed": 42},
            "output": {"logs_dir": "results/logs", "base_dir": "results", "models_dir": "results/models"},
        }

        argv = ["scenario_C_llm.py", "--adaptive_mode", "strict_coherence_guided"]
        with patch.object(sys, "argv", argv):
            args = scenario_c.parse_args()
        central_agent = DummyCentralAgent()
        with patch.object(scenario_c, "create_central_agent", return_value=(object(), central_agent, None, object())), \
             patch.object(scenario_c, "attach_bias_corrected_metrics", side_effect=lambda _agent, results, _loader: results), \
             patch.object(scenario_c, "save_federated_outputs"):
            scenario_c.run_single_strategy(
                "size_only",
                config,
                {},
                {},
                object(),
                object(),
                object(),
                args,
                DummyLogger(),
                "cpu",
                10,
            )
        self.assertEqual(central_agent.called[0], "strict")

        argv = ["scenario_C_llm.py", "--adaptive_mode", "llm_strict_generative_coherence", "--use_llm"]
        with patch.object(sys, "argv", argv):
            args = scenario_c.parse_args()
        central_agent = DummyCentralAgent()
        with patch.object(scenario_c, "create_llm_planner", return_value=object()), \
             patch.object(scenario_c, "create_central_agent", return_value=(object(), central_agent, None, object())), \
             patch.object(scenario_c, "attach_bias_corrected_metrics", side_effect=lambda _agent, results, _loader: results), \
             patch.object(scenario_c, "save_federated_outputs"):
            scenario_c.run_single_strategy(
                "size_only",
                config,
                {},
                {},
                object(),
                object(),
                object(),
                args,
                DummyLogger(),
                "cpu",
                10,
            )
        self.assertEqual(central_agent.called[0], "llm_strict")


if __name__ == "__main__":
    unittest.main()
