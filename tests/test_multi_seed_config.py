import unittest


class MultiSeedConfigTests(unittest.TestCase):
    def test_all_scenarios_receive_seed_argument(self):
        from scripts.run_multi_seed import SCENARIO_CONFIGS

        for scenario in ["A", "A_prime", "B", "C"]:
            with self.subTest(scenario=scenario):
                self.assertEqual(SCENARIO_CONFIGS[scenario]["seed_arg"], "--seed")


if __name__ == "__main__":
    unittest.main()
