import unittest
from unittest import mock

import pandas as pd


class RunAblationTests(unittest.TestCase):
    def test_bias_correction_config_is_derived_from_llm_run(self):
        from scripts.run_ablation import ABLATION_CONFIGS

        ab6 = next(cfg for cfg in ABLATION_CONFIGS if cfg["id"] == "ab-6")

        self.assertTrue(ab6["derive_from"], "ab-5")

    def test_derived_bias_rows_reuse_ab5_corrected_metrics(self):
        from scripts.run_ablation import derive_bias_correction_rows

        df = pd.DataFrame([
            {
                "id": "ab-5",
                "name": "C-with-LLM",
                "seed": 42,
                "success": True,
                "test_mape": 0.5,
                "test_mape_corrected": 0.45,
            }
        ])

        derived = derive_bias_correction_rows(df)

        self.assertEqual(len(derived), 1)
        self.assertEqual(derived.iloc[0]["id"], "ab-6")
        self.assertEqual(derived.iloc[0]["name"], "C-with-LLM+bias")
        self.assertEqual(derived.iloc[0]["test_mape_corrected"], 0.45)

    def test_main_calls_generate_summary(self):
        import scripts.run_ablation as run_ablation

        fake_metrics = {
            "id": "ab-1",
            "name": "B-baseline",
            "seed": 42,
            "success": True,
        }
        with mock.patch.object(run_ablation, "run_experiment", return_value=fake_metrics), \
             mock.patch.object(run_ablation, "generate_summary") as generate_summary, \
             mock.patch("sys.argv", ["run_ablation.py", "--only", "ab-1", "--seeds", "42"]):
            run_ablation.main()

        generate_summary.assert_called_once()


if __name__ == "__main__":
    unittest.main()
