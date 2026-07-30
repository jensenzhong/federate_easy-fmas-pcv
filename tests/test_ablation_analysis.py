import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class AblationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        Path("results").mkdir()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def test_collect_prefers_multi_seed_summary_over_logs(self):
        from scripts import ablation_analysis

        pd.DataFrame([
            {
                "id": "ab-1",
                "name": "legacy-single-seed-name",
                "n_seeds": 5,
                "seeds": "42,123,456,789,2024",
                "test_mape_mean": 0.5163,
                "test_mape_std": 0.0573,
                "test_rmse_mean": 1716607,
                "test_rmse_std": 122275,
                "test_r2_mean": 0.3038,
                "test_r2_std": 0.1027,
            },
            {
                "id": "ab-6",
                "name": "legacy-bias-name",
                "n_seeds": 5,
                "seeds": "42,123,456,789,2024",
                "test_mape_corrected_mean": 0.5145,
                "test_mape_corrected_std": 0.0509,
                "test_rmse_corrected_mean": 1530000,
                "test_rmse_corrected_std": 56000,
                "test_r2_corrected_mean": 0.4448,
                "test_r2_corrected_std": 0.0354,
            },
        ]).to_csv("results/ablation_summary.csv", index=False)

        df = ablation_analysis.collect_ablation_results()
        latex = ablation_analysis.generate_latex_table(df)

        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[df["id"] == "ab-1", "n_seeds"].iloc[0], 5)
        self.assertIn("51.63 $\\pm$ 5.73", latex)
        self.assertIn("51.45 $\\pm$ 5.09", latex)
        self.assertNotIn("legacy-single-seed-name", latex)


if __name__ == "__main__":
    unittest.main()
