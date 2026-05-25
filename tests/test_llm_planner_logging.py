import tempfile
import unittest
from pathlib import Path


class LLMPlannerLoggingTests(unittest.TestCase):
    def test_new_planner_starts_with_fresh_decision_log(self):
        from src.federated_learning.llm_planner import LLMPlanner

        class DummyClient:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            log_path = log_dir / "scene_C_llm_decisions.jsonl"
            log_path.write_text("old\n", encoding="utf-8")

            LLMPlanner(config={"scene_c": {"llm": {}, "strategies": []}}, llm_client=DummyClient(), log_dir=str(log_dir))

            self.assertEqual(log_path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
