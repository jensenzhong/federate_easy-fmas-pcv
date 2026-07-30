import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class StratifiedEvaluationTests(unittest.TestCase):
    def test_load_predictions_uses_semantic_experiment_names_for_federated_outputs(self):
        from scripts.stratified_evaluation import load_predictions

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pd.DataFrame([{
                "True_Value": 1_000_000,
                "Predicted_Value": 1_100_000,
            }]).to_csv(base / "fedavg_predictions.csv", index=False)
            pd.DataFrame([{
                "True_Value": 1_000_000,
                "Predicted_Value": 950_000,
            }]).to_csv(base / "fedyogi_predictions.csv", index=False)
            pd.DataFrame([{
                "True_Value": 1_000_000,
                "Predicted_Value": 920_000,
            }]).to_csv(base / "vg_fedyogi_tr_predictions.csv", index=False)
            pd.DataFrame([{
                "True_Value": 1_000_000,
                "Predicted_Value": 910_000,
            }]).to_csv(base / "mas_vg_fedyogi_tr_predictions.csv", index=False)

            predictions = load_predictions(str(base))

        self.assertIn("传统联邦学习", predictions)
        self.assertIn("自适应联邦学习（FedYogi-TR）", predictions)
        self.assertIn("验证引导自适应联邦学习（VG-FedYogi-TR）", predictions)
        self.assertIn("多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）", predictions)
        self.assertNotIn("多智能体协同联邦学习", predictions)
        self.assertNotIn("B (FedAvg)", predictions)
        self.assertNotIn("C (MAS-FL-LLM)", predictions)


if __name__ == "__main__":
    unittest.main()
