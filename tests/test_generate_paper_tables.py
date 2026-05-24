import importlib
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class GeneratePaperTablesTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        Path("results/multi_seed").mkdir(parents=True)

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp.cleanup()

    def _module(self):
        return importlib.import_module("scripts.generate_paper_tables")

    def test_main_table_prefers_multi_seed_summary_over_single_run_csv(self):
        pd.DataFrame([{
            "scenario": "B_FedAvg",
            "test_mape": 0.9,
            "test_rmse": 9000000,
            "test_mae": 8000000,
            "test_mpe": 0.4,
            "test_r2": 0.1,
        }]).to_csv("results/fedavg_results.csv", index=False)
        pd.DataFrame([{
            "scenario": "B",
            "n_runs": 5,
            "test_mape_mean": 0.5164,
            "test_mape_std": 0.0467,
            "test_rmse_mean": 1552865,
            "test_rmse_std": 97113,
            "test_mae_mean": 1104931,
            "test_mae_std": 64869,
            "test_mpe_mean": 0.0402,
            "test_mpe_std": 0.0544,
            "test_r2_mean": 0.4308,
            "test_r2_std": 0.0711,
        }]).to_csv("results/multi_seed/statistical_summary.csv", index=False)

        latex = self._module().generate_main_results_table()

        self.assertIn("B (FedAvg) & 5 & 51.64 $\\pm$ 4.67", latex)
        self.assertNotIn("90.00", latex)

    def test_ablation_table_uses_summary_values_instead_of_placeholders(self):
        pd.DataFrame([{
            "id": "ab-1",
            "name": "B-baseline",
            "n_seeds": 5,
            "test_mape_mean": 0.4446,
            "test_mape_std": 0.0123,
        }, {
            "id": "ab-6",
            "name": "C-with-LLM+bias",
            "n_seeds": 5,
            "test_mape_corrected_mean": 0.4218,
            "test_mape_corrected_std": 0.0345,
        }]).to_csv("results/ablation_summary.csv", index=False)

        latex = self._module().generate_ablation_table()

        self.assertIn("B-baseline & No & size\\_only & No & 44.46 $\\pm$ 1.23", latex)
        self.assertIn("C-with-LLM+bias & Yes & Dynamic & Yes & 42.18 $\\pm$ 3.45", latex)
        self.assertNotIn("& - &", latex)


if __name__ == "__main__":
    unittest.main()
