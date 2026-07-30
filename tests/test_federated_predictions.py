import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


class FederatedPredictionOutputTests(unittest.TestCase):
    def test_mpe_bias_correction_uses_validation_mpe_ratio(self):
        from src.federated_learning.mas_agents import apply_mpe_bias_correction

        corrected, mpe = apply_mpe_bias_correction(
            val_true=np.array([100.0, 200.0]),
            val_pred=np.array([110.0, 220.0]),
            test_pred=np.array([330.0]),
        )

        self.assertAlmostEqual(mpe, 0.1)
        self.assertAlmostEqual(corrected[0], 300.0)

    def test_prediction_frame_includes_client_and_project_size_strata(self):
        from src.federated_learning.mas_agents import build_prediction_frame

        metadata = pd.DataFrame({"Client": ["client_a", "client_b", "client_c"]})

        df = build_prediction_frame(
            y_true=np.array([500_000, 2_000_000, 6_000_000]),
            y_pred=np.array([550_000, 1_800_000, 6_300_000]),
            metadata=metadata,
            thresholds={
                "small_project_max": 1_000_000,
                "medium_project_max": 5_000_000,
            },
            scenario="传统联邦学习",
        )

        self.assertEqual(
            list(df.columns),
            ["scenario", "True_Value", "Predicted_Value", "Client", "Project_Size_Stratum"],
        )
        self.assertEqual(list(df["Client"]), ["client_a", "client_b", "client_c"])
        self.assertEqual(
            list(df["Project_Size_Stratum"]),
            ["Small (<$1M)", "Medium ($1M-$5M)", "Large (>=$5M)"],
        )

    def test_central_agent_save_predictions_writes_bias_corrected_csv(self):
        from src.federated_learning.mas_agents import CentralAgent

        agent = CentralAgent.__new__(CentralAgent)
        agent.config = {
            "thresholds": {
                "small_project_max": 1_000_000,
                "medium_project_max": 5_000_000,
            }
        }
        calls = []

        def fake_evaluate_global(data_loader=None, return_predictions=False, apply_bias_correction=False):
            calls.append({
                "return_predictions": return_predictions,
                "apply_bias_correction": apply_bias_correction,
            })
            return {
                "targets": np.array([2_000_000]),
                "predictions": np.array([1_900_000]),
            }

        agent.evaluate_global = fake_evaluate_global
        loader = SimpleNamespace(dataset=SimpleNamespace(
            prediction_metadata=pd.DataFrame({"Client": ["client_b"]})
        ))

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "predictions.csv"
            agent.save_predictions(
                data_loader=loader,
                output_path=output_path,
                scenario="多智能体协同联邦学习（偏差校正）",
                apply_bias_correction=True,
            )
            df = pd.read_csv(output_path)

        self.assertEqual(calls, [{
            "return_predictions": True,
            "apply_bias_correction": True,
        }])
        self.assertEqual(df.iloc[0]["scenario"], "多智能体协同联邦学习（偏差校正）")
        self.assertEqual(df.iloc[0]["Client"], "client_b")


if __name__ == "__main__":
    unittest.main()
