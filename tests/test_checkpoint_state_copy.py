import unittest

import torch


class CheckpointStateCopyTests(unittest.TestCase):
    def test_snapshot_state_dict_is_immutable_after_model_updates(self):
        from src.models import snapshot_model_state

        model = torch.nn.Linear(1, 1)
        snapshot = snapshot_model_state(model)
        saved_weight = snapshot["weight"].clone()

        with torch.no_grad():
            model.weight.add_(10.0)

        self.assertTrue(torch.equal(snapshot["weight"], saved_weight))
        self.assertFalse(torch.equal(snapshot["weight"], model.state_dict()["weight"]))


if __name__ == "__main__":
    unittest.main()
