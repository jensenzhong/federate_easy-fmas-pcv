import unittest
import tempfile
import sys
from unittest.mock import patch


class AdaptiveMethodOutputTests(unittest.TestCase):
    def test_experiment_names_include_adaptive_methods(self):
        from src.experiment_names import EXPERIMENT_ORDER, experiment_display_name

        self.assertIn("FEDYOGI", EXPERIMENT_ORDER)
        self.assertIn("VG_FEDYOGI_TR", EXPERIMENT_ORDER)
        self.assertIn("MAS_VG_FEDYOGI_TR", EXPERIMENT_ORDER)
        self.assertNotEqual(experiment_display_name("FEDYOGI"), "FEDYOGI")
        self.assertNotEqual(experiment_display_name("VG_FEDYOGI_TR"), "VG_FEDYOGI_TR")
        self.assertNotEqual(experiment_display_name("MAS_VG_FEDYOGI_TR"), "MAS_VG_FEDYOGI_TR")
        self.assertIn("FedYogi", experiment_display_name("FEDYOGI"))

    def test_stratified_loader_includes_adaptive_prediction_files(self):
        import inspect
        import scripts.stratified_evaluation as stratified

        source = inspect.getsource(stratified.load_predictions)

        self.assertIn("fedyogi_predictions.csv", source)
        self.assertIn("vg_fedyogi_tr_predictions.csv", source)
        self.assertIn("mas_vg_fedyogi_tr_predictions.csv", source)
        self.assertIn("fedyogi_predictions_bias_corrected.csv", source)
        self.assertIn("vg_fedyogi_tr_predictions_bias_corrected.csv", source)
        self.assertIn("mas_vg_fedyogi_tr_predictions_bias_corrected.csv", source)

    def test_adaptive_pilot_passes_clip_norm_to_scenario(self):
        import scripts.run_adaptive_pilot as pilot

        class DummyResult:
            returncode = 0
            stdout = ""
            stderr = ""

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return DummyResult()

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pilot.subprocess, "run", side_effect=fake_run):
            row = pilot.run_candidate(
                server_lr=0.02,
                clip_norm=2.0,
                max_coordinate_step_ratio=1.0,
                seed=777,
                output_dir=pilot.Path(tmp),
            )

        self.assertTrue(row["success"])
        self.assertIn("--update_clip_norm", captured["cmd"])
        clip_arg_idx = captured["cmd"].index("--update_clip_norm")
        self.assertEqual(captured["cmd"][clip_arg_idx + 1], "2.0")

    def test_adaptive_pilot_passes_trust_region_to_scenario(self):
        import scripts.run_adaptive_pilot as pilot

        class DummyResult:
            returncode = 0
            stdout = ""
            stderr = ""

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return DummyResult()

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pilot.subprocess, "run", side_effect=fake_run):
            row = pilot.run_candidate(
                server_lr=0.3,
                clip_norm=None,
                max_coordinate_step_ratio=0.75,
                seed=777,
                output_dir=pilot.Path(tmp),
            )

        self.assertTrue(row["success"])
        self.assertIn("--max_coordinate_step_ratio", captured["cmd"])
        ratio_arg_idx = captured["cmd"].index("--max_coordinate_step_ratio")
        self.assertEqual(captured["cmd"][ratio_arg_idx + 1], "0.75")

    def test_round_metrics_include_trust_region_diagnostics(self):
        import inspect
        from src.federated_learning.mas_agents import CentralAgent

        source = inspect.getsource(CentralAgent.get_training_history_df)

        self.assertIn("max_coordinate_step_ratio", source)
        self.assertIn("coordinate_step_clipped", source)
        self.assertIn("coordinate_direction_rejected", source)

    def test_adaptive_pilot_recommendation_uses_cross_seed_mean_validation(self):
        import pandas as pd
        import scripts.run_adaptive_pilot as pilot

        df = pd.DataFrame([
            {"server_lr": 0.3, "update_clip_norm": None, "max_coordinate_step_ratio": 1.0, "seed": 777, "success": True, "best_val_mape": 0.36},
            {"server_lr": 0.3, "update_clip_norm": None, "max_coordinate_step_ratio": 1.0, "seed": 778, "success": True, "best_val_mape": 0.40},
            {"server_lr": 0.5, "update_clip_norm": None, "max_coordinate_step_ratio": 1.0, "seed": 777, "success": True, "best_val_mape": 0.33},
            {"server_lr": 0.5, "update_clip_norm": None, "max_coordinate_step_ratio": 1.0, "seed": 778, "success": True, "best_val_mape": 0.48},
        ])

        recommendation, summary = pilot.select_recommendation(df, expected_seed_count=2, fallback_ratio=1.0)

        self.assertEqual(recommendation["selected_server_lr"], 0.3)
        self.assertEqual(recommendation["selection_metric"], "mean_best_val_mape")
        self.assertEqual(recommendation["selected_n_success"], 2)
        self.assertIn("best_val_mape_mean", summary.columns)

    def test_adaptive_pilot_recommendation_requires_complete_seed_coverage(self):
        import pandas as pd
        import scripts.run_adaptive_pilot as pilot

        df = pd.DataFrame([
            {"server_lr": 0.3, "update_clip_norm": None, "max_coordinate_step_ratio": 1.0, "seed": 777, "success": True, "best_val_mape": 0.36},
            {"server_lr": 0.5, "update_clip_norm": None, "max_coordinate_step_ratio": 1.0, "seed": 888, "success": True, "best_val_mape": 0.35},
        ])

        with self.assertRaisesRegex(RuntimeError, "No pilot configuration completed"):
            pilot.select_recommendation(df, expected_seed_count=2, fallback_ratio=1.0)

    def test_adaptive_pilot_main_accepts_multiple_trust_region_ratios(self):
        import pandas as pd
        import scripts.run_adaptive_pilot as pilot

        calls = []

        def fake_run_candidate(server_lr, clip_norm, max_coordinate_step_ratio, seed, output_dir):
            calls.append((server_lr, clip_norm, max_coordinate_step_ratio, seed, str(output_dir)))
            return {
                "server_lr": server_lr,
                "update_clip_norm": clip_norm,
                "max_coordinate_step_ratio": max_coordinate_step_ratio,
                "seed": seed,
                "success": True,
                "best_val_mape": 0.35,
            }

        recommendation = {
            "selected_server_lr": 0.3,
            "selected_update_clip_norm": None,
            "selected_max_coordinate_step_ratio": 1.0,
            "selection_metric": "mean_best_val_mape",
            "selected_n_success": 2,
        }
        group_summary = pd.DataFrame([{
            "server_lr": 0.3,
            "update_clip_norm": None,
            "max_coordinate_step_ratio": 1.0,
            "best_val_mape_mean": 0.35,
            "best_val_mape_std": 0.01,
            "n_success": 2,
        }])

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pilot, "run_candidate", side_effect=fake_run_candidate), \
             patch.object(pilot, "select_recommendation", return_value=(recommendation, group_summary)), \
             patch.object(sys, "argv", [
                 "run_adaptive_pilot.py",
                 "--seeds", "777", "888",
                 "--server_lrs", "0.3",
                 "--max_coordinate_step_ratios", "0.75", "1.0",
                 "--clip_norms", "none",
                 "--output_dir", tmp,
             ]):
            pilot.main()

        self.assertEqual(len(calls), 4)
        self.assertEqual(sorted({call[2] for call in calls}), [0.75, 1.0])
        self.assertEqual(sorted({call[3] for call in calls}), [777, 888])


if __name__ == "__main__":
    unittest.main()
