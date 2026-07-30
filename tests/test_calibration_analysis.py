import tempfile
import unittest
from pathlib import Path

import pandas as pd


class CalibrationAnalysisTests(unittest.TestCase):
    def test_calibration_summary_reports_raw_to_corrected_gain(self):
        from scripts.calibration_analysis import compute_calibration_summary

        df = pd.DataFrame([
            {"scenario_key": "A", "seed": 1, "success": True, "test_mape": 0.50, "test_mape_corrected": None},
            {"scenario_key": "B", "seed": 1, "success": True, "test_mape": 0.42, "test_mape_corrected": 0.41},
            {"scenario_key": "B", "seed": 2, "success": True, "test_mape": 0.44, "test_mape_corrected": 0.43},
            {"scenario_key": "LLM_GCA_FEDYOGI_TR", "seed": 1, "success": True, "test_mape": 0.43, "test_mape_corrected": 0.40},
            {"scenario_key": "LLM_GCA_FEDYOGI_TR", "seed": 2, "success": True, "test_mape": 0.45, "test_mape_corrected": 0.42},
        ])

        summary = compute_calibration_summary(df)

        self.assertNotIn("A", set(summary["scenario_key"]))
        llm = summary[summary["scenario_key"] == "LLM_GCA_FEDYOGI_TR"].iloc[0]
        self.assertAlmostEqual(llm["mean_mape_delta"], -0.03)
        self.assertAlmostEqual(llm["mean_mape_relative_delta"], -0.03 / 0.44)
        self.assertEqual(llm["n_runs"], 2)

    def test_calibration_analysis_writes_csv(self):
        from scripts.calibration_analysis import run_calibration_analysis

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            results_file = base / "all_results.csv"
            pd.DataFrame([
                {"scenario_key": "B", "seed": 1, "success": True, "test_mape": 0.42, "test_mape_corrected": 0.41},
                {"scenario_key": "B", "seed": 2, "success": True, "test_mape": 0.44, "test_mape_corrected": 0.43},
            ]).to_csv(results_file, index=False)

            output_path = base / "calibration_summary.csv"
            summary = run_calibration_analysis(results_file=results_file, output_path=output_path)
            self.assertTrue(output_path.exists())

        self.assertFalse(summary.empty)


if __name__ == "__main__":
    unittest.main()
