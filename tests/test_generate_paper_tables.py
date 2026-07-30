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

        self.assertIn("传统联邦学习 & 5 & 51.64 $\\pm$ 4.67", latex)
        self.assertNotIn("90.00", latex)

    def test_main_table_uses_semantic_names_and_marks_single_seed_rows(self):
        pd.DataFrame([{
            "scenario": "A_Prime_Neural_Network",
            "test_mape": 0.5058,
            "test_rmse": 1406208,
            "test_mae": 1018121,
            "test_mpe": 0.1984,
        }]).to_csv("results/centralized_nn_results.csv", index=False)
        pd.DataFrame([{
            "scenario": "传统联邦学习",
            "n_runs": 5,
            "test_mape_mean": 0.5164,
            "test_mape_std": 0.0467,
        }]).to_csv("results/multi_seed/statistical_summary.csv", index=False)

        latex = self._module().generate_main_results_table()

        self.assertIn("ANN传统神经网络", latex)
        self.assertIn("单种子", latex)
        self.assertNotIn("A' (NN)", latex)
        self.assertNotIn("B (FedAvg)", latex)

    def test_ablation_table_uses_summary_values_instead_of_placeholders(self):
        pd.DataFrame([{
            "id": "ab-1",
            "name": "传统联邦学习（FedAvg）",
            "n_seeds": 5,
            "test_mape_mean": 0.4446,
            "test_mape_std": 0.0123,
        }, {
            "id": "ab-4",
            "name": "多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）",
            "n_seeds": 5,
            "test_mape_mean": 0.4218,
            "test_mape_std": 0.0345,
        }]).to_csv("results/ablation_summary.csv", index=False)

        latex = self._module().generate_ablation_table()

        self.assertIn("传统联邦学习（FedAvg） & No & size\\_only & No & 44.46 $\\pm$ 1.23", latex)
        self.assertIn("多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）", latex)
        self.assertNotIn("LLM动态决策+偏差校正", latex)
        self.assertNotIn("& - &", latex)

    def test_stratified_table_normalizes_legacy_experiment_names(self):
        pd.DataFrame([{
            "scenario": "A (GBR)",
            "stratum": "Small (<$1M)",
            "n": 2,
            "mape": 0.4,
            "rmse": 100,
            "mae": 90,
            "mpe": 0.1,
        }, {
            "scenario": "A' (NN)",
            "stratum": "Small (<$1M)",
            "n": 2,
            "mape": 0.3,
            "rmse": 80,
            "mae": 70,
            "mpe": 0.05,
        }]).to_csv("results/stratified_evaluation.csv", index=False)

        latex = self._module().generate_stratified_table()

        self.assertIn("GBR传统机器学习", latex)
        self.assertIn("ANN传统神经网络", latex)
        self.assertNotIn("A (GBR)", latex)
        self.assertNotIn("A' (NN)", latex)

    def test_main_table_includes_adaptive_methods_from_multi_seed_summary(self):
        from src.experiment_names import experiment_display_name

        pd.DataFrame([
            {
                "scenario_key": "FEDYOGI",
                "scenario": experiment_display_name("FEDYOGI"),
                "n_runs": 5,
                "test_mape_mean": 0.49,
                "test_mape_std": 0.02,
                "test_rmse_mean": 1200000,
                "test_rmse_std": 100000,
                "test_mae_mean": 900000,
                "test_mae_std": 50000,
                "test_mpe_mean": 0.01,
                "test_mpe_std": 0.02,
                "test_r2_mean": 0.5,
                "test_r2_std": 0.04,
            },
            {
                "scenario_key": "MAS_VG_FEDYOGI_TR",
                "scenario": experiment_display_name("MAS_VG_FEDYOGI_TR"),
                "n_runs": 5,
                "test_mape_mean": 0.47,
                "test_mape_std": 0.03,
                "test_rmse_mean": 1100000,
                "test_rmse_std": 90000,
                "test_mae_mean": 850000,
                "test_mae_std": 40000,
                "test_mpe_mean": -0.01,
                "test_mpe_std": 0.02,
                "test_r2_mean": 0.55,
                "test_r2_std": 0.03,
            },
        ]).to_csv("results/multi_seed/statistical_summary.csv", index=False)

        latex = self._module().generate_main_results_table()

        self.assertIn(experiment_display_name("FEDYOGI"), latex)
        self.assertIn(experiment_display_name("MAS_VG_FEDYOGI_TR"), latex)
        self.assertIn("49.00 $\\pm$ 2.00", latex)
        self.assertIn("47.00 $\\pm$ 3.00", latex)


if __name__ == "__main__":
    unittest.main()
