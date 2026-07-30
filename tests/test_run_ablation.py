import unittest
from unittest import mock

import pandas as pd


class RunAblationTests(unittest.TestCase):
    def test_default_ablation_configs_match_new_four_line_mainline(self):
        from scripts.run_ablation import ABLATION_CONFIGS

        self.assertEqual([cfg["id"] for cfg in ABLATION_CONFIGS], ["ab-1", "ab-2", "ab-3", "ab-4"])
        self.assertEqual(
            [cfg["name"] for cfg in ABLATION_CONFIGS],
            [
                "传统联邦学习（FedAvg）",
                "自适应联邦学习（FedYogi-TR）",
                "验证引导自适应联邦学习（VG-FedYogi-TR）",
                "多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）",
            ],
        )

    def test_main_calls_generate_summary(self):
        import scripts.run_ablation as run_ablation

        fake_metrics = {
            "id": "ab-1",
            "name": "传统联邦学习（FedAvg）",
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
