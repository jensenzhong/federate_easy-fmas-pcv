import tempfile
import unittest
from pathlib import Path

import pandas as pd


class PredictionGapAnalysisTests(unittest.TestCase):
    def test_pairwise_gap_summary_reports_stratified_cancellation(self):
        from scripts.prediction_gap_analysis import build_pairwise_gap_summary

        baseline = pd.DataFrame([
            {"True_Value": 100.0, "Predicted_Value": 90.0, "Client": "Client 1", "Project_Size_Stratum": "Small"},
            {"True_Value": 1000.0, "Predicted_Value": 900.0, "Client": "Client 1", "Project_Size_Stratum": "Medium"},
            {"True_Value": 2000.0, "Predicted_Value": 1500.0, "Client": "Client 2", "Project_Size_Stratum": "Large"},
        ])
        contender = pd.DataFrame([
            {"True_Value": 100.0, "Predicted_Value": 100.0, "Client": "Client 1", "Project_Size_Stratum": "Small"},
            {"True_Value": 1000.0, "Predicted_Value": 850.0, "Client": "Client 1", "Project_Size_Stratum": "Medium"},
            {"True_Value": 2000.0, "Predicted_Value": 1400.0, "Client": "Client 2", "Project_Size_Stratum": "Large"},
        ])

        summary, rowwise = build_pairwise_gap_summary(
            baseline_name="FedAvg",
            contender_name="LLM-GCA",
            baseline_df=baseline,
            contender_df=contender,
            n_bootstrap=200,
            random_seed=7,
        )

        overall = summary[summary["group_type"] == "overall"].iloc[0]
        small = summary[
            (summary["group_type"] == "Project_Size_Stratum")
            & (summary["group_value"] == "Small")
        ].iloc[0]

        self.assertAlmostEqual(overall["mean_ape_diff"], (0.0 - 0.1 + 0.15 - 0.1 + 0.3 - 0.25) / 3)
        self.assertLess(small["mean_ape_diff"], 0)
        self.assertIn("bootstrap_ci_lower", summary.columns)
        self.assertIn("bootstrap_ci_upper", summary.columns)
        self.assertIn("contender_better", rowwise.columns)

    def test_analyze_prediction_files_writes_summary_outputs(self):
        from scripts.prediction_gap_analysis import analyze_prediction_files

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pd.DataFrame([
                {"True_Value": 100.0, "Predicted_Value": 90.0, "Client": "Client 1", "Project_Size_Stratum": "Small"},
            ]).to_csv(base / "fedavg_predictions.csv", index=False)
            pd.DataFrame([
                {"True_Value": 100.0, "Predicted_Value": 100.0, "Client": "Client 1", "Project_Size_Stratum": "Small"},
            ]).to_csv(base / "llm_gca_fedyogi_tr_predictions.csv", index=False)

            summary, rowwise = analyze_prediction_files(
                results_dir=base,
                baseline_key="B",
                contender_key="LLM_GCA_FEDYOGI_TR",
                n_bootstrap=50,
                random_seed=7,
            )
            self.assertTrue((base / "prediction_gap_analysis.csv").exists())
            self.assertTrue((base / "prediction_gap_rowwise.csv").exists())

        self.assertFalse(summary.empty)
        self.assertFalse(rowwise.empty)


if __name__ == "__main__":
    unittest.main()
