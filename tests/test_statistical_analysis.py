import unittest

import pandas as pd


class StatisticalAnalysisTests(unittest.TestCase):
    def test_summary_includes_bias_corrected_metrics(self):
        from scripts.statistical_analysis import compute_summary_stats

        df = pd.DataFrame([
            {
                "scenario": "C",
                "seed": 42,
                "success": True,
                "test_mape": 0.5,
                "test_mape_corrected": 0.45,
            },
            {
                "scenario": "C",
                "seed": 123,
                "success": True,
                "test_mape": 0.6,
                "test_mape_corrected": 0.50,
            },
        ])

        summary = compute_summary_stats(df)

        self.assertIn("test_mape_corrected_mean", summary.columns)
        self.assertAlmostEqual(summary.iloc[0]["test_mape_corrected_mean"], 0.475)

    def test_statistical_tests_pair_by_matching_seed(self):
        from scripts.statistical_analysis import perform_statistical_tests

        df = pd.DataFrame([
            {"scenario": "B", "seed": 1, "success": True, "test_mape": 0.3, "test_rmse": 10, "test_mae": 8},
            {"scenario": "B", "seed": 2, "success": True, "test_mape": 0.4, "test_rmse": 20, "test_mae": 18},
            {"scenario": "B", "seed": 3, "success": True, "test_mape": 0.5, "test_rmse": 30, "test_mae": 28},
            {"scenario": "C", "seed": 3, "success": True, "test_mape": 0.6, "test_rmse": 40, "test_mae": 38},
            {"scenario": "C", "seed": 1, "success": True, "test_mape": 0.4, "test_rmse": 20, "test_mae": 18},
            {"scenario": "C", "seed": 2, "success": True, "test_mape": 0.5, "test_rmse": 30, "test_mae": 28},
        ])

        results = perform_statistical_tests(df)

        self.assertEqual(results["B_vs_C"]["test_mape"]["paired_seeds"], [1, 2, 3])
        self.assertEqual(results["B_vs_C"]["test_mape"]["t_test_type"], "paired t-test")

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


if __name__ == "__main__":
    unittest.main()
