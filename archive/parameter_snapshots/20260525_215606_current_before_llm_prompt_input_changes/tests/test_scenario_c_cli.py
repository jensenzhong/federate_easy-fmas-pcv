import sys
import unittest
from unittest.mock import patch


class ScenarioCCliTests(unittest.TestCase):
    def test_temperature_argument_is_parsed(self):
        from experiments.scenario_C_llm import parse_args

        argv = ["scenario_C_llm.py", "--use_llm", "--temperature", "0"]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
