import unittest

import pandas as pd


class StatisticalAnalysisTests(unittest.TestCase):
    def test_summary_includes_bias_corrected_metrics(self):
        from scripts.statistical_analysis import compute_summary_stats

        df = pd.DataFrame([
            {"scenario_key": "C", "seed": 42, "success": True, "test_mape": 0.5, "test_mape_corrected": 0.45},
            {"scenario_key": "C", "seed": 123, "success": True, "test_mape": 0.6, "test_mape_corrected": 0.50},
        ])

        summary = compute_summary_stats(df)

        self.assertIn("test_mape_corrected_mean", summary.columns)
        self.assertAlmostEqual(summary.iloc[0]["test_mape_corrected_mean"], 0.475)

    def test_statistical_tests_pair_by_matching_seed(self):
        from scripts.statistical_analysis import perform_statistical_tests
        from src.experiment_names import experiment_display_name

        df = pd.DataFrame([
            {"scenario_key": "B", "seed": 1, "success": True, "test_mape": 0.3, "test_rmse": 10, "test_mae": 8},
            {"scenario_key": "B", "seed": 2, "success": True, "test_mape": 0.4, "test_rmse": 20, "test_mae": 18},
            {"scenario_key": "B", "seed": 3, "success": True, "test_mape": 0.5, "test_rmse": 30, "test_mae": 28},
            {"scenario_key": "FEDYOGI", "seed": 3, "success": True, "test_mape": 0.6, "test_rmse": 40, "test_mae": 38},
            {"scenario_key": "FEDYOGI", "seed": 1, "success": True, "test_mape": 0.4, "test_rmse": 20, "test_mae": 18},
            {"scenario_key": "FEDYOGI", "seed": 2, "success": True, "test_mape": 0.5, "test_rmse": 30, "test_mae": 28},
        ])

        results = perform_statistical_tests(df)

        key = f"{experiment_display_name('B')}_vs_{experiment_display_name('FEDYOGI')}"
        self.assertEqual(results[key]["test_mape"]["paired_seeds"], [1, 2, 3])
        self.assertEqual(results[key]["test_mape"]["t_test_type"], "paired t-test")

    def test_statistical_tests_include_ann_and_bias_corrected_comparisons(self):
        from scripts.statistical_analysis import perform_statistical_tests
        from src.experiment_names import experiment_display_name

        rows = []
        for seed in [1, 2, 3]:
            rows.extend([
                {"scenario_key": "A_prime", "seed": seed, "success": True, "test_mape": 0.52, "test_rmse": 50, "test_mae": 40},
                {
                    "scenario_key": "FEDYOGI",
                    "seed": seed,
                    "success": True,
                    "test_mape": 0.50,
                    "test_rmse": 45,
                    "test_mae": 35,
                    "test_mape_corrected": 0.49,
                    "test_rmse_corrected": 43,
                    "test_mae_corrected": 33,
                },
                {
                    "scenario_key": "VG_FEDYOGI_TR",
                    "seed": seed,
                    "success": True,
                    "test_mape": 0.49,
                    "test_rmse": 44,
                    "test_mae": 34,
                    "test_mape_corrected": 0.48,
                    "test_rmse_corrected": 40,
                    "test_mae_corrected": 30,
                },
                {
                    "scenario_key": "MAS_VG_FEDYOGI_TR",
                    "seed": seed,
                    "success": True,
                    "test_mape": 0.48,
                    "test_rmse": 43,
                    "test_mae": 33,
                    "test_mape_corrected": 0.47,
                    "test_rmse_corrected": 39,
                    "test_mae_corrected": 29,
                },
            ])
        df = pd.DataFrame(rows)

        results = perform_statistical_tests(df)

        self.assertIn(f"{experiment_display_name('A_prime')}_vs_{experiment_display_name('FEDYOGI')}", results)
        self.assertIn(f"{experiment_display_name('A_prime')}_vs_{experiment_display_name('VG_FEDYOGI_TR')}", results)
        self.assertIn(f"{experiment_display_name('FEDYOGI')}_vs_{experiment_display_name('VG_FEDYOGI_TR')}", results)
        corrected_pairs = [
            (
                f"{experiment_display_name('FEDYOGI')}_bias_corrected_vs_"
                f"{experiment_display_name('VG_FEDYOGI_TR')}_bias_corrected"
            ),
            (
                f"{experiment_display_name('FEDYOGI')}_bias_corrected_vs_"
                f"{experiment_display_name('MAS_VG_FEDYOGI_TR')}_bias_corrected"
            ),
            (
                f"{experiment_display_name('VG_FEDYOGI_TR')}_bias_corrected_vs_"
                f"{experiment_display_name('MAS_VG_FEDYOGI_TR')}_bias_corrected"
            ),
        ]
        for corrected_key in corrected_pairs:
            with self.subTest(corrected_key=corrected_key):
                self.assertIn(corrected_key, results)
                self.assertIn("test_mape_corrected", results[corrected_key])

    def test_statistical_tests_include_adaptive_federated_comparisons(self):
        from scripts.statistical_analysis import perform_statistical_tests
        from src.experiment_names import experiment_display_name

        rows = []
        for seed in [1, 2, 3]:
            rows.extend([
                {"scenario_key": "A_prime", "seed": seed, "success": True, "test_mape": 0.52, "test_rmse": 50, "test_mae": 40},
                {"scenario_key": "B", "seed": seed, "success": True, "test_mape": 0.51, "test_rmse": 48, "test_mae": 38},
                {"scenario_key": "FEDYOGI", "seed": seed, "success": True, "test_mape": 0.49, "test_rmse": 46, "test_mae": 36},
                {"scenario_key": "VG_FEDYOGI_TR", "seed": seed, "success": True, "test_mape": 0.48, "test_rmse": 45, "test_mae": 35},
                {"scenario_key": "MAS_VG_FEDYOGI_TR", "seed": seed, "success": True, "test_mape": 0.47, "test_rmse": 44, "test_mae": 34},
                {"scenario_key": "COHERENCE_FEDYOGI_TR", "seed": seed, "success": True, "test_mape": 0.465, "test_rmse": 43, "test_mae": 33},
                {"scenario_key": "LLM_GCA_FEDYOGI_TR", "seed": seed, "success": True, "test_mape": 0.46, "test_rmse": 42, "test_mae": 32},
            ])
        df = pd.DataFrame(rows)

        results = perform_statistical_tests(df)

        expected_pairs = [
            ("A_prime", "B"),
            ("B", "FEDYOGI"),
            ("A_prime", "FEDYOGI"),
            ("A_prime", "VG_FEDYOGI_TR"),
            ("A_prime", "MAS_VG_FEDYOGI_TR"),
            ("B", "VG_FEDYOGI_TR"),
            ("B", "MAS_VG_FEDYOGI_TR"),
            ("FEDYOGI", "VG_FEDYOGI_TR"),
            ("VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR"),
            ("B", "COHERENCE_FEDYOGI_TR"),
            ("B", "LLM_GCA_FEDYOGI_TR"),
            ("FEDYOGI", "COHERENCE_FEDYOGI_TR"),
            ("FEDYOGI", "LLM_GCA_FEDYOGI_TR"),
            ("COHERENCE_FEDYOGI_TR", "LLM_GCA_FEDYOGI_TR"),
        ]
        for left, right in expected_pairs:
            name = f"{experiment_display_name(left)}_vs_{experiment_display_name(right)}"
            with self.subTest(name=name):
                self.assertIn(name, results)
                self.assertIn("test_mape", results[name])

        legacy_names = [
            f"{experiment_display_name('A_prime')}_vs_{experiment_display_name('C')}",
            f"{experiment_display_name('C')}_vs_{experiment_display_name('MAS_ADAPTIVE')}",
            f"{experiment_display_name('FEDYOGI')}_vs_{experiment_display_name('MAS_ADAPTIVE')}",
        ]
        for name in legacy_names:
            with self.subTest(name=name):
                self.assertNotIn(name, results)

    def test_significance_tests_can_be_flattened_for_csv_export(self):
        from scripts.statistical_analysis import flatten_significance_tests

        test_results = {
            "B_vs_C": {
                "test_mape": {
                    "paired_seeds": [1, 2, 3],
                    "t_test_type": "paired t-test",
                    "t_statistic": 1.23,
                    "t_p_value": 0.456,
                    "wilcoxon_type": "Wilcoxon signed-rank",
                    "w_statistic": 2.0,
                    "w_p_value": 0.5,
                    "cohens_d": 0.12,
                    "significant_005": False,
                    "significant_001": False,
                }
            }
        }

        df = flatten_significance_tests(test_results)

        self.assertEqual(list(df["comparison"]), ["B_vs_C"])
        self.assertEqual(list(df["metric"]), ["test_mape"])
        self.assertEqual(list(df["paired_seeds"]), ["1,2,3"])

    def test_significance_tests_export_left_and_right_keys(self):
        from scripts.statistical_analysis import flatten_significance_tests

        test_results = {
            "B_vs_FEDYOGI": {
                "test_mape": {
                    "left_key": "B",
                    "right_key": "FEDYOGI",
                    "paired_seeds": [42, 123],
                    "t_test_type": "paired t-test",
                    "t_statistic": 1.23,
                    "t_p_value": 0.456,
                }
            }
        }

        df = flatten_significance_tests(test_results)

        self.assertEqual(df.loc[0, "left_key"], "B")
        self.assertEqual(df.loc[0, "right_key"], "FEDYOGI")


if __name__ == "__main__":
    unittest.main()
