import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class MultiSeedConfigTests(unittest.TestCase):
    def test_all_scenarios_receive_seed_argument(self):
        from scripts.run_multi_seed import SCENARIO_CONFIGS

        for scenario in ["A", "A_prime", "B", "B_STRICT", "C", "FEDYOGI", "FEDYOGI_STRICT", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR", "COHERENCE_FEDYOGI_TR", "LLM_GCA_FEDYOGI_TR", "STRICT_COHERENCE_FEDYOGI_TR", "LLM_STRICT_GCA_FEDYOGI_TR", "VP_GCA_FEDYOGI_TR", "LLM_VP_GCA_FEDYOGI_TR"]:
            with self.subTest(scenario=scenario):
                self.assertEqual(SCENARIO_CONFIGS[scenario]["seed_arg"], "--seed")

    def test_user_facing_names_are_semantic_not_abc(self):
        from scripts.run_multi_seed import SCENARIO_CONFIGS
        from src.experiment_names import experiment_display_name

        for scenario in ["A", "A_prime", "B", "B_STRICT", "C", "FEDYOGI", "FEDYOGI_STRICT", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR", "COHERENCE_FEDYOGI_TR", "LLM_GCA_FEDYOGI_TR", "STRICT_COHERENCE_FEDYOGI_TR", "LLM_STRICT_GCA_FEDYOGI_TR", "VP_GCA_FEDYOGI_TR", "LLM_VP_GCA_FEDYOGI_TR"]:
            with self.subTest(scenario=scenario):
                self.assertEqual(SCENARIO_CONFIGS[scenario]["name"], experiment_display_name(scenario))
                self.assertNotEqual(SCENARIO_CONFIGS[scenario]["name"], scenario)

    def test_adaptive_multi_seed_configs_use_trust_region_fedyogi_lr(self):
        from scripts.run_multi_seed import SCENARIO_CONFIGS

        for scenario in ["FEDYOGI", "FEDYOGI_STRICT", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR", "COHERENCE_FEDYOGI_TR", "LLM_GCA_FEDYOGI_TR", "STRICT_COHERENCE_FEDYOGI_TR", "LLM_STRICT_GCA_FEDYOGI_TR", "VP_GCA_FEDYOGI_TR", "LLM_VP_GCA_FEDYOGI_TR"]:
            with self.subTest(scenario=scenario):
                args = SCENARIO_CONFIGS[scenario]["args"]

                self.assertEqual(args[args.index("--server_optimizer") + 1], "fedyogi")
                self.assertNotIn("--server_lr", args)
                self.assertNotIn("--max_coordinate_step_ratio", args)

    def test_default_formal_scenarios_exclude_legacy_mas_adaptive(self):
        from scripts.run_multi_seed import DEFAULT_SCENARIOS, SCENARIO_CONFIGS

        self.assertEqual(
            DEFAULT_SCENARIOS,
            ["A", "A_prime", "B_STRICT", "FEDYOGI_STRICT", "STRICT_COHERENCE_FEDYOGI_TR", "LLM_STRICT_GCA_FEDYOGI_TR"],
        )
        self.assertNotIn("B", DEFAULT_SCENARIOS)
        self.assertNotIn("FEDYOGI", DEFAULT_SCENARIOS)
        self.assertNotIn("C", DEFAULT_SCENARIOS)
        self.assertNotIn("LLM_VP_GCA_FEDYOGI_TR", DEFAULT_SCENARIOS)
        self.assertNotIn("VP_GCA_FEDYOGI_TR", DEFAULT_SCENARIOS)
        self.assertNotIn("MAS_VG_FEDYOGI_TR", DEFAULT_SCENARIOS)
        self.assertNotIn("MAS_ADAPTIVE", DEFAULT_SCENARIOS)
        self.assertIn("MAS_ADAPTIVE", SCENARIO_CONFIGS)
        self.assertIn("MAS_VG_FEDYOGI_TR", SCENARIO_CONFIGS)

    def test_mas_vg_uses_balanced_llm_score_tolerance(self):
        from scripts.run_multi_seed import SCENARIO_CONFIGS

        args = SCENARIO_CONFIGS["MAS_VG_FEDYOGI_TR"]["args"]

        self.assertEqual(args[args.index("--llm_score_tolerance") + 1], "0.003")

    def test_new_gca_scenarios_use_expected_adaptive_modes(self):
        from scripts.run_multi_seed import SCENARIO_CONFIGS

        coherence_args = SCENARIO_CONFIGS["COHERENCE_FEDYOGI_TR"]["args"]
        llm_args = SCENARIO_CONFIGS["LLM_GCA_FEDYOGI_TR"]["args"]

        self.assertEqual(coherence_args[coherence_args.index("--adaptive_mode") + 1], "coherence_guided")
        self.assertNotIn("--use_llm", coherence_args)
        self.assertEqual(llm_args[llm_args.index("--adaptive_mode") + 1], "llm_generative_coherence")
        self.assertIn("--use_llm", llm_args)
        self.assertEqual(llm_args[llm_args.index("--temperature") + 1], "0")

        strict_coherence_args = SCENARIO_CONFIGS["STRICT_COHERENCE_FEDYOGI_TR"]["args"]
        strict_llm_args = SCENARIO_CONFIGS["LLM_STRICT_GCA_FEDYOGI_TR"]["args"]
        self.assertEqual(
            strict_coherence_args[strict_coherence_args.index("--adaptive_mode") + 1],
            "strict_coherence_guided",
        )
        self.assertEqual(
            strict_llm_args[strict_llm_args.index("--adaptive_mode") + 1],
            "llm_strict_generative_coherence",
        )

        vp_baseline_args = SCENARIO_CONFIGS["VP_GCA_FEDYOGI_TR"]["args"]
        self.assertEqual(vp_baseline_args[vp_baseline_args.index("--adaptive_mode") + 1], "validation_preview_gca")
        self.assertNotIn("--use_llm", vp_baseline_args)

        vp_args = SCENARIO_CONFIGS["LLM_VP_GCA_FEDYOGI_TR"]["args"]
        self.assertEqual(vp_args[vp_args.index("--adaptive_mode") + 1], "llm_validation_preview_generative")
        self.assertIn("--use_llm", vp_args)
        self.assertEqual(vp_args[vp_args.index("--temperature") + 1], "0")

    def test_adaptive_multi_seed_configs_can_be_updated_from_pilot_recommendation(self):
        from scripts.run_multi_seed import adaptive_args_from_recommendation

        args = adaptive_args_from_recommendation(
            base_args=["--num_rounds", "20", "--server_lr", "0.3", "--max_coordinate_step_ratio", "1.0"],
            recommendation={
                "selected_server_lr": 0.4,
                "selected_max_coordinate_step_ratio": 1.0,
                "selected_update_clip_norm": None,
            },
        )

        self.assertEqual(args[args.index("--server_lr") + 1], "0.4")
        self.assertEqual(float(args[args.index("--max_coordinate_step_ratio") + 1]), 1.0)
        self.assertNotIn("--update_clip_norm", args)

    def test_adaptive_multi_seed_configs_include_clip_norm_when_recommended(self):
        from scripts.run_multi_seed import adaptive_args_from_recommendation

        args = adaptive_args_from_recommendation(
            base_args=["--num_rounds", "20", "--server_lr", "0.3", "--max_coordinate_step_ratio", "1.0"],
            recommendation={
                "selected_server_lr": 0.2,
                "selected_max_coordinate_step_ratio": 1.0,
                "selected_update_clip_norm": 2.5,
            },
        )

        self.assertEqual(args[args.index("--server_lr") + 1], "0.2")
        self.assertEqual(args[args.index("--update_clip_norm") + 1], "2.5")

    def test_result_csv_reader_adds_run_metadata(self):
        from scripts.run_multi_seed import read_scenario_result
        from src.experiment_names import experiment_display_name

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            pd.DataFrame([{
                "scenario": "B_FedAvg",
                "num_rounds": 20,
                "test_mape": 0.42,
                "test_rmse": 123.0,
            }]).to_csv(output_dir / "fedavg_results.csv", index=False)

            row = read_scenario_result(
                scenario_key="B",
                seed=42,
                output_dir=str(output_dir),
                elapsed_seconds=1.5,
                success=True,
                command=["python", "experiments/scenario_B_fedavg.py"],
                code_commit="abc123",
                code_dirty=True,
                split_seed=42,
                llm_provider="deepseek",
                llm_temperature=0.0,
            )

        self.assertEqual(row["scenario"], experiment_display_name("B"))
        self.assertEqual(row["scenario_key"], "B")
        self.assertEqual(row["seed"], 42)
        self.assertEqual(row["split_seed"], 42)
        self.assertEqual(row["code_commit"], "abc123")
        self.assertTrue(row["code_dirty"])
        self.assertEqual(row["llm_provider"], "deepseek")
        self.assertEqual(row["llm_temperature"], 0.0)
        self.assertEqual(row["test_mape"], 0.42)

    def test_result_csv_reader_supports_adaptive_methods(self):
        from scripts.run_multi_seed import read_scenario_result
        from src.experiment_names import experiment_display_name

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            pd.DataFrame([{
                "scenario": "placeholder",
                "num_rounds": 20,
                "test_mape": 0.39,
                "llm_temperature": 0.0,
                "max_coordinate_step_ratio": 1.0,
            }]).to_csv(output_dir / "mas_vg_fedyogi_tr_results.csv", index=False)

            row = read_scenario_result(
                scenario_key="MAS_VG_FEDYOGI_TR",
                seed=42,
                output_dir=str(output_dir),
                elapsed_seconds=2.0,
                success=True,
                command=["python", "experiments/scenario_C_llm.py"],
                code_commit="abc123",
                code_dirty=False,
                split_seed=42,
                llm_provider="deepseek",
                llm_temperature=0.0,
            )

        self.assertEqual(row["scenario"], experiment_display_name("MAS_VG_FEDYOGI_TR"))
        self.assertEqual(row["scenario_key"], "MAS_VG_FEDYOGI_TR")
        self.assertEqual(row["result_file"], str(output_dir / "mas_vg_fedyogi_tr_results.csv"))
        self.assertEqual(row["llm_temperature"], 0.0)
        self.assertEqual(row["max_coordinate_step_ratio"], 1.0)

    def test_seed_result_snapshot_preserves_single_seed_result(self):
        from scripts.run_multi_seed import _snapshot_seed_result

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result_path = output_dir / "fedyogi_results.csv"
            pd.DataFrame([
                {"scenario_key": "FEDYOGI", "test_mape": 0.42, "seed_marker": 42},
            ]).to_csv(result_path, index=False)

            snapshot_path = _snapshot_seed_result(
                result_file=str(result_path),
                scenario_key="FEDYOGI",
                seed=42,
                output_dir=str(output_dir),
            )

            snapshot = pd.read_csv(snapshot_path)

        self.assertEqual(snapshot.iloc[0]["test_mape"], 0.42)
        self.assertEqual(snapshot.iloc[0]["seed_marker"], 42)
        self.assertTrue(snapshot_path.endswith("FEDYOGI_seed42_result.csv"))

    def test_reparse_logs_prefers_seed_snapshot(self):
        import scripts.run_multi_seed as runner

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "current_code_metadata", return_value={"code_commit": "abc123", "code_dirty": False}), \
             patch.object(runner, "_read_run_metadata", return_value={"split_seed": 42, "llm_provider": "deepseek", "llm_temperature": 0.0}):
            output_dir = Path(tmp)
            multi_seed_dir = output_dir / "multi_seed"
            multi_seed_dir.mkdir(parents=True)
            (multi_seed_dir / "FEDYOGI_seed42.log").write_text(
                "Test MAPE: 99.00%\n",
                encoding="utf-8",
            )
            pd.DataFrame([{
                "scenario_key": "FEDYOGI",
                "test_mape": 0.40,
                "server_lr": 0.4,
            }]).to_csv(multi_seed_dir / "FEDYOGI_seed42_result.csv", index=False)

            rows = runner.reparse_logs(["FEDYOGI"], [42], str(output_dir))

        self.assertEqual(rows[0]["source"], "seed_result_snapshot")
        self.assertEqual(rows[0]["test_mape"], 0.40)
        self.assertEqual(rows[0]["server_lr"], 0.4)

    def test_reparse_logs_without_snapshot_parses_log_not_current_result_csv(self):
        import scripts.run_multi_seed as runner

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "current_code_metadata", return_value={"code_commit": "abc123", "code_dirty": False}), \
             patch.object(runner, "_read_run_metadata", return_value={"split_seed": 42, "llm_provider": "deepseek", "llm_temperature": 0.0}):
            output_dir = Path(tmp)
            multi_seed_dir = output_dir / "multi_seed"
            multi_seed_dir.mkdir(parents=True)
            (multi_seed_dir / "FEDYOGI_seed42.log").write_text(
                "Test MAPE: 40.00%\n",
                encoding="utf-8",
            )
            pd.DataFrame([{
                "scenario_key": "FEDYOGI",
                "test_mape": 0.99,
                "server_lr": 0.4,
            }]).to_csv(output_dir / "fedyogi_results.csv", index=False)

            rows = runner.reparse_logs(["FEDYOGI"], [42], str(output_dir))

        self.assertEqual(rows[0]["source"], "reparsed_log")
        self.assertEqual(rows[0]["test_mape"], 0.40)

    def test_adaptive_formal_run_requires_pilot_recommendation(self):
        import scripts.run_multi_seed as runner

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "load_adaptive_pilot_recommendation", return_value=None), \
             patch.object(runner.subprocess, "run") as fake_run:
            with self.assertRaisesRegex(RuntimeError, "pilot_recommendation"):
                runner.run_scenario("FEDYOGI", seed=42, output_dir=str(tmp))

        self.assertFalse(fake_run.called)

    def test_formal_adaptive_args_require_selected_n_success(self):
        from scripts.run_multi_seed import formal_adaptive_args

        with self.assertRaisesRegex(RuntimeError, "selected_n_success"):
            formal_adaptive_args(
                ["--num_rounds", "20"],
                {
                    "selected_server_lr": 0.3,
                    "selected_max_coordinate_step_ratio": 1.0,
                },
            )

    def test_merge_results_preserves_existing_unrequested_scenarios(self):
        from scripts.run_multi_seed import merge_all_results

        existing = pd.DataFrame([
            {"scenario_key": "A_prime", "seed": 42, "test_mape": 0.50},
            {"scenario_key": "B", "seed": 42, "test_mape": 0.42},
        ])
        new = pd.DataFrame([
            {"scenario_key": "FEDYOGI", "seed": 42, "test_mape": 0.40},
        ])

        merged = merge_all_results(existing, new, scenarios=["FEDYOGI"], seeds=[42])

        self.assertEqual(set(merged["scenario_key"]), {"A_prime", "B", "FEDYOGI"})
        self.assertEqual(
            merged.loc[merged["scenario_key"] == "FEDYOGI", "test_mape"].iloc[0],
            0.40,
        )

    def test_merge_results_replaces_same_scenario_seed(self):
        from scripts.run_multi_seed import merge_all_results

        existing = pd.DataFrame([
            {"scenario_key": "FEDYOGI", "seed": 42, "test_mape": 0.99},
            {"scenario_key": "B", "seed": 42, "test_mape": 0.42},
        ])
        new = pd.DataFrame([
            {"scenario_key": "FEDYOGI", "seed": 42, "test_mape": 0.40},
        ])

        merged = merge_all_results(existing, new, scenarios=["FEDYOGI"], seeds=[42])

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            merged.loc[merged["scenario_key"] == "FEDYOGI", "test_mape"].iloc[0],
            0.40,
        )


if __name__ == "__main__":
    unittest.main()
