import unittest

import torch


class ServerOptimizerTests(unittest.TestCase):
    def test_fedavg_optimizer_returns_weighted_average_state(self):
        from src.federated_learning.server_optimizers import FedAvgServerOptimizer

        optimizer = FedAvgServerOptimizer()
        current = {"w": torch.tensor([1.0, 2.0])}
        target = {"w": torch.tensor([3.0, 4.0])}

        updated, info = optimizer.step(current, target)

        torch.testing.assert_close(updated["w"], target["w"])
        self.assertEqual(info["server_optimizer"], "fedavg")
        self.assertAlmostEqual(info["update_norm"], torch.linalg.vector_norm(target["w"] - current["w"]).item())

    def test_central_aggregation_preserves_non_floating_buffers(self):
        from src.federated_learning.mas_agents import CentralAgent

        agent = CentralAgent.__new__(CentralAgent)
        client_states = {
            "client_a": {
                "w": torch.tensor([1.0, 2.0]),
                "num_batches_tracked": torch.tensor(100, dtype=torch.long),
            },
            "client_b": {
                "w": torch.tensor([3.0, 4.0]),
                "num_batches_tracked": torch.tensor(120, dtype=torch.long),
            },
        }

        aggregated = agent.aggregate_with_weights(client_states, {"client_a": 0.25, "client_b": 0.75})

        torch.testing.assert_close(aggregated["w"], torch.tensor([2.5, 3.5]))
        self.assertEqual(aggregated["num_batches_tracked"].dtype, torch.long)
        self.assertEqual(int(aggregated["num_batches_tracked"]), 100)

    def test_fedyogi_first_and_second_step_follow_formula(self):
        from src.federated_learning.server_optimizers import FedYogiServerOptimizer

        optimizer = FedYogiServerOptimizer(
            server_lr=0.01,
            beta1=0.9,
            beta2=0.99,
            tau=1e-3,
        )

        current_1 = {"w": torch.tensor([1.0, 2.0])}
        target_1 = {"w": torch.tensor([0.8, 2.4])}
        delta_1 = target_1["w"] - current_1["w"]
        expected_m_1 = 0.1 * delta_1
        expected_v_1 = torch.zeros_like(delta_1) - 0.01 * delta_1.pow(2) * torch.sign(torch.zeros_like(delta_1) - delta_1.pow(2))
        expected_1 = current_1["w"] + 0.01 * expected_m_1 / (torch.sqrt(expected_v_1) + 1e-3)

        updated_1, info_1 = optimizer.step(current_1, target_1)

        torch.testing.assert_close(updated_1["w"], expected_1)
        torch.testing.assert_close(optimizer.m["w"], expected_m_1)
        torch.testing.assert_close(optimizer.v["w"], expected_v_1)
        self.assertEqual(info_1["server_optimizer"], "fedyogi")

        current_2 = updated_1
        target_2 = {"w": torch.tensor([0.7, 2.3])}
        delta_2 = target_2["w"] - current_2["w"]
        expected_m_2 = 0.9 * expected_m_1 + 0.1 * delta_2
        expected_v_2 = expected_v_1 - 0.01 * delta_2.pow(2) * torch.sign(expected_v_1 - delta_2.pow(2))
        expected_2 = current_2["w"] + 0.01 * expected_m_2 / (torch.sqrt(expected_v_2) + 1e-3)

        updated_2, _ = optimizer.step(current_2, target_2)

        torch.testing.assert_close(updated_2["w"], expected_2)
        torch.testing.assert_close(optimizer.m["w"], expected_m_2)
        torch.testing.assert_close(optimizer.v["w"], expected_v_2)

    def test_fedyogi_applies_lr_scale_and_clip_norm(self):
        from src.federated_learning.server_optimizers import FedYogiServerOptimizer

        optimizer = FedYogiServerOptimizer(
            server_lr=1.0,
            beta1=0.0,
            beta2=0.0,
            tau=1e-9,
            update_clip_norm=0.5,
        )
        current = {"w": torch.tensor([0.0, 0.0])}
        target = {"w": torch.tensor([3.0, 4.0])}

        updated, info = optimizer.step(current, target, server_lr_scale=0.5)

        self.assertLessEqual(torch.linalg.vector_norm(updated["w"] - current["w"]).item(), 0.5 + 1e-6)
        self.assertAlmostEqual(info["server_lr_scale"], 0.5)
        self.assertTrue(info["update_clipped"])

    def test_fedyogi_default_step_does_not_overshoot_weighted_average_target(self):
        from src.federated_learning.server_optimizers import FedYogiServerOptimizer

        optimizer = FedYogiServerOptimizer(
            server_lr=0.1,
            beta1=0.0,
            beta2=0.99,
            tau=1e-3,
        )
        current = {"w": torch.tensor([0.0, 0.0])}
        target = {"w": torch.tensor([0.001, -0.001])}

        updated, info = optimizer.step(current, target)

        torch.testing.assert_close(updated["w"], target["w"])
        self.assertTrue(info["coordinate_step_clipped"])

    def test_fedyogi_default_step_does_not_move_against_weighted_average_direction(self):
        from src.federated_learning.server_optimizers import FedYogiServerOptimizer

        optimizer = FedYogiServerOptimizer(
            server_lr=0.1,
            beta1=0.9,
            beta2=0.99,
            tau=1e-3,
        )
        current = {"w": torch.tensor([0.0])}
        updated, _ = optimizer.step(current, {"w": torch.tensor([1.0])})
        updated, _ = optimizer.step(updated, {"w": torch.tensor([0.8])})

        current_before_reversal = {"w": updated["w"].detach().clone()}
        reversed_target = {"w": torch.tensor([-0.001])}
        updated, info = optimizer.step(current_before_reversal, reversed_target)

        self.assertLessEqual(float(updated["w"][0]), float(current_before_reversal["w"][0]))
        self.assertTrue(info["coordinate_direction_rejected"])

    def test_fedyogi_preview_step_does_not_mutate_optimizer_state(self):
        from src.federated_learning.server_optimizers import FedYogiServerOptimizer

        optimizer = FedYogiServerOptimizer(server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3)
        current = {"w": torch.tensor([0.0, 1.0])}
        target = {"w": torch.tensor([0.5, 1.5])}
        optimizer.step(current, target)
        saved_state = optimizer.get_optimizer_state()

        preview_current = {"w": torch.tensor([0.2, 1.2])}
        preview_target = {"w": torch.tensor([0.8, 1.1])}
        preview_next, preview_info = optimizer.preview_step(preview_current, preview_target, server_lr_scale=0.5)

        torch.testing.assert_close(optimizer.m["w"], saved_state["m"]["w"])
        torch.testing.assert_close(optimizer.v["w"], saved_state["v"]["w"])
        self.assertAlmostEqual(preview_info["server_lr_scale"], 0.5)

        clone = FedYogiServerOptimizer(server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3)
        clone.load_optimizer_state(saved_state)
        expected_next, _ = clone.step(preview_current, preview_target, server_lr_scale=0.5)
        torch.testing.assert_close(preview_next["w"], expected_next["w"])

    def test_fedyogi_preview_clip_overrides_are_side_effect_free(self):
        from src.federated_learning.server_optimizers import FedYogiServerOptimizer

        optimizer = FedYogiServerOptimizer(
            server_lr=0.1,
            beta1=0.9,
            beta2=0.99,
            tau=1e-3,
            update_clip_norm=2.0,
        )
        current = {"w": torch.tensor([0.0, 0.0])}
        target = {"w": torch.tensor([10.0, 0.0])}
        optimizer.step(current, target)
        saved_state = optimizer.get_optimizer_state()

        preview_current = {"w": torch.tensor([0.0, 0.0])}
        preview_target = {"w": torch.tensor([10.0, 0.0])}
        clipped, clipped_info = optimizer.preview_step(
            preview_current,
            preview_target,
            update_clip_norm_override=0.05,
        )
        unclipped, unclipped_info = optimizer.preview_step(
            preview_current,
            preview_target,
            update_clip_norm_override=None,
        )

        self.assertLess(clipped_info["update_norm"], unclipped_info["update_norm"])
        self.assertLess(torch.linalg.vector_norm(clipped["w"]), torch.linalg.vector_norm(unclipped["w"]))
        self.assertEqual(optimizer.update_clip_norm, 2.0)
        restored_state = optimizer.get_optimizer_state()
        self.assertEqual(restored_state["update_clip_norm"], saved_state["update_clip_norm"])
        self.assertEqual(restored_state["server_lr"], saved_state["server_lr"])
        self.assertEqual(restored_state["beta1"], saved_state["beta1"])
        self.assertEqual(restored_state["beta2"], saved_state["beta2"])
        self.assertEqual(restored_state["tau"], saved_state["tau"])
        self.assertEqual(
            restored_state["max_coordinate_step_ratio"],
            saved_state["max_coordinate_step_ratio"],
        )
        torch.testing.assert_close(restored_state["m"]["w"], saved_state["m"]["w"])
        torch.testing.assert_close(restored_state["v"]["w"], saved_state["v"]["w"])

    def test_fedyogi_step_clip_override_does_not_change_default(self):
        from src.federated_learning.server_optimizers import FedYogiServerOptimizer

        optimizer = FedYogiServerOptimizer(
            server_lr=1.0,
            beta1=0.0,
            beta2=0.0,
            tau=1e-3,
            update_clip_norm=2.0,
            max_coordinate_step_ratio=None,
        )
        current = {"w": torch.tensor([0.0])}
        target = {"w": torch.tensor([10.0])}

        updated, info = optimizer.step(
            current,
            target,
            update_clip_norm_override=0.5,
        )

        self.assertAlmostEqual(updated["w"].item(), 0.5)
        self.assertTrue(info["update_clipped"])
        self.assertEqual(optimizer.update_clip_norm, 2.0)

    def test_fedyogi_optimizer_state_round_trips(self):
        from src.federated_learning.server_optimizers import FedYogiServerOptimizer

        optimizer = FedYogiServerOptimizer(server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3)
        optimizer.step({"w": torch.tensor([0.0])}, {"w": torch.tensor([1.0])})
        state = optimizer.get_optimizer_state()

        restored = FedYogiServerOptimizer(server_lr=0.2, beta1=0.1, beta2=0.2, tau=1e-2)
        restored.load_optimizer_state(state)

        torch.testing.assert_close(restored.m["w"], optimizer.m["w"])
        torch.testing.assert_close(restored.v["w"], optimizer.v["w"])
        self.assertAlmostEqual(restored.server_lr, optimizer.server_lr)
        self.assertAlmostEqual(restored.beta1, optimizer.beta1)
        self.assertAlmostEqual(restored.beta2, optimizer.beta2)


if __name__ == "__main__":
    unittest.main()
