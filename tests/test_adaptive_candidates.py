import unittest


class AdaptiveCandidateTests(unittest.TestCase):
    def test_generate_candidates_respects_bounds_budget_and_required_anchors(self):
        from src.federated_learning.adaptive_candidates import generate_weight_candidates

        candidates = generate_weight_candidates(
            client_ids=["Client 1", "Client 2", "Client 3"],
            size_weights={"Client 1": 0.5, "Client 2": 0.2, "Client 3": 0.3},
            previous_weights={"Client 1": 0.4, "Client 2": 0.3, "Client 3": 0.3},
            client_metrics={
                "Client 1": {"val_mape": 0.5},
                "Client 2": {"val_mape": 0.7},
                "Client 3": {"val_mape": 0.4},
            },
            budget=12,
            step=0.1,
            min_weight=0.05,
            max_weight=0.8,
        )

        self.assertLessEqual(len(candidates), 12)
        sources = {candidate.source for candidate in candidates}
        self.assertIn("size_anchor", sources)
        self.assertIn("uniform_anchor", sources)
        self.assertIn("previous_accepted", sources)
        self.assertIn("error_compensation", sources)
        for candidate in candidates:
            self.assertAlmostEqual(sum(candidate.weights.values()), 1.0, places=6)
            for weight in candidate.weights.values():
                self.assertGreaterEqual(weight, 0.05 - 1e-9)
                self.assertLessEqual(weight, 0.8 + 1e-9)

    def test_generate_candidates_adds_literature_backed_evidence_candidates(self):
        from src.federated_learning.adaptive_candidates import generate_weight_candidates

        candidates = generate_weight_candidates(
            client_ids=["Client 1", "Client 2", "Client 3"],
            size_weights={"Client 1": 0.34, "Client 2": 0.34, "Client 3": 0.32},
            previous_weights={"Client 1": 0.34, "Client 2": 0.34, "Client 3": 0.32},
            client_metrics={
                "Client 1": {"val_mape": 0.50, "val_mpe": -0.25},
                "Client 2": {"val_mape": 0.45, "val_mpe": -0.05},
                "Client 3": {"val_mape": 0.55, "val_mpe": 0.02},
            },
            coherence_diagnostics={
                "Client 1": {
                    "sample_size_weight": 0.34,
                    "cosine_to_mean_update": 0.95,
                    "update_norm": 1.0,
                    "val_mape": 0.50,
                    "val_mpe": -0.25,
                },
                "Client 2": {
                    "sample_size_weight": 0.34,
                    "cosine_to_mean_update": 0.40,
                    "update_norm": 3.0,
                    "val_mape": 0.45,
                    "val_mpe": -0.05,
                },
                "Client 3": {
                    "sample_size_weight": 0.32,
                    "cosine_to_mean_update": -0.20,
                    "update_norm": 1.0,
                    "val_mape": 0.55,
                    "val_mpe": 0.02,
                },
            },
            budget=20,
            step=0.05,
            min_weight=0.05,
            max_weight=0.8,
        )

        sources = {candidate.source for candidate in candidates}
        self.assertIn("coherence_prior", sources)
        self.assertIn("fedlaw_shrinkage", sources)
        self.assertIn("drift_limited", sources)
        self.assertIn("bias_sensitive", sources)

        fedlaw = next(candidate for candidate in candidates if candidate.source == "fedlaw_shrinkage")
        self.assertLess(fedlaw.server_lr_scale, 1.0)

        drift_limited = next(candidate for candidate in candidates if candidate.source == "drift_limited")
        self.assertLessEqual(drift_limited.weights["Client 3"], 0.32 + 1e-9)

    def test_gate_falls_back_for_invalid_requested_candidate(self):
        from src.federated_learning.adaptive_candidates import AdaptiveCandidate, select_candidate_by_gate

        candidates = [
            AdaptiveCandidate(
                candidate_id="size_anchor",
                weights={"a": 0.5, "b": 0.5},
                server_lr_scale=1.0,
                source="size_anchor",
                score=0.50,
                validation_metrics={"mape": 0.50},
            ),
            AdaptiveCandidate(
                candidate_id="candidate_001",
                weights={"a": 0.6, "b": 0.4},
                server_lr_scale=1.0,
                source="local_grid",
                score=0.45,
                validation_metrics={"mape": 0.45},
            ),
        ]

        selected, info = select_candidate_by_gate(
            candidates,
            conservative_candidate_id="size_anchor",
            requested_candidate_id="missing",
            epsilon=0.002,
            previous_weights={"a": 0.5, "b": 0.5},
            weight_l1_limit=0.4,
            large_improvement_threshold=0.01,
        )

        self.assertEqual(selected.candidate_id, "candidate_001")
        self.assertEqual(info["gate_status"], "fallback_invalid_request")

    def test_gate_prefers_conservative_candidate_when_improvement_is_below_epsilon(self):
        from src.federated_learning.adaptive_candidates import AdaptiveCandidate, select_candidate_by_gate

        candidates = [
            AdaptiveCandidate(
                candidate_id="size_anchor_lr0p5_ep+0",
                weights={"a": 0.5, "b": 0.5},
                server_lr_scale=1.0,
                source="size_anchor",
                score=0.5000,
                validation_metrics={"mape": 0.5000},
            ),
            AdaptiveCandidate(
                candidate_id="candidate_001",
                weights={"a": 0.55, "b": 0.45},
                server_lr_scale=1.0,
                source="local_grid",
                score=0.4990,
                validation_metrics={"mape": 0.4990},
            ),
        ]

        selected, info = select_candidate_by_gate(
            candidates,
            conservative_candidate_id="size_anchor",
            requested_candidate_id="candidate_001",
            epsilon=0.002,
            previous_weights={"a": 0.5, "b": 0.5},
            weight_l1_limit=0.4,
            large_improvement_threshold=0.01,
        )

        self.assertEqual(selected.candidate_id, "size_anchor_lr0p5_ep+0")
        self.assertEqual(info["gate_status"], "fallback_conservative_epsilon")

    def test_gate_rejects_large_weight_shift_without_large_improvement(self):
        from src.federated_learning.adaptive_candidates import AdaptiveCandidate, select_candidate_by_gate

        candidates = [
            AdaptiveCandidate(
                candidate_id="size_anchor",
                weights={"a": 0.5, "b": 0.5},
                server_lr_scale=1.0,
                source="size_anchor",
                score=0.50,
                validation_metrics={"mape": 0.50},
            ),
            AdaptiveCandidate(
                candidate_id="candidate_001",
                weights={"a": 0.8, "b": 0.2},
                server_lr_scale=1.0,
                source="local_grid",
                score=0.495,
                validation_metrics={"mape": 0.495},
            ),
        ]

        selected, info = select_candidate_by_gate(
            candidates,
            conservative_candidate_id="size_anchor",
            requested_candidate_id="candidate_001",
            epsilon=0.001,
            previous_weights={"a": 0.5, "b": 0.5},
            weight_l1_limit=0.4,
            large_improvement_threshold=0.01,
        )

        self.assertEqual(selected.candidate_id, "size_anchor")
        self.assertEqual(info["gate_status"], "fallback_weight_shift")

    def test_gate_allows_llm_near_best_candidate_within_tolerance(self):
        from src.federated_learning.adaptive_candidates import AdaptiveCandidate, select_candidate_by_gate

        candidates = [
            AdaptiveCandidate(
                candidate_id="size_anchor",
                weights={"a": 0.5, "b": 0.5},
                server_lr_scale=1.0,
                source="size_anchor",
                score=0.500,
                validation_metrics={"mape": 0.500},
            ),
            AdaptiveCandidate(
                candidate_id="candidate_best",
                weights={"a": 0.55, "b": 0.45},
                server_lr_scale=1.0,
                source="local_grid",
                score=0.4800,
                validation_metrics={"mape": 0.4800, "rmse": 120.0},
            ),
            AdaptiveCandidate(
                candidate_id="candidate_llm",
                weights={"a": 0.54, "b": 0.46},
                server_lr_scale=1.0,
                source="local_grid",
                score=0.4808,
                validation_metrics={"mape": 0.4808, "rmse": 100.0},
            ),
        ]

        selected, info = select_candidate_by_gate(
            candidates,
            conservative_candidate_id="size_anchor",
            requested_candidate_id="candidate_llm",
            epsilon=0.001,
            score_tolerance=0.001,
            previous_weights={"a": 0.5, "b": 0.5},
            weight_l1_limit=0.4,
            large_improvement_threshold=0.01,
        )

        self.assertEqual(selected.candidate_id, "candidate_llm")
        self.assertEqual(info["gate_status"], "accepted_llm_near_best")

    def test_gate_rejects_llm_candidate_outside_score_tolerance(self):
        from src.federated_learning.adaptive_candidates import AdaptiveCandidate, select_candidate_by_gate

        candidates = [
            AdaptiveCandidate(
                candidate_id="size_anchor",
                weights={"a": 0.5, "b": 0.5},
                server_lr_scale=1.0,
                source="size_anchor",
                score=0.500,
                validation_metrics={"mape": 0.500},
            ),
            AdaptiveCandidate(
                candidate_id="candidate_best",
                weights={"a": 0.55, "b": 0.45},
                server_lr_scale=1.0,
                source="local_grid",
                score=0.480,
                validation_metrics={"mape": 0.480},
            ),
            AdaptiveCandidate(
                candidate_id="candidate_bad",
                weights={"a": 0.56, "b": 0.44},
                server_lr_scale=1.0,
                source="local_grid",
                score=0.490,
                validation_metrics={"mape": 0.490},
            ),
        ]

        selected, info = select_candidate_by_gate(
            candidates,
            conservative_candidate_id="size_anchor",
            requested_candidate_id="candidate_bad",
            epsilon=0.001,
            score_tolerance=0.001,
            previous_weights={"a": 0.5, "b": 0.5},
            weight_l1_limit=0.4,
            large_improvement_threshold=0.01,
        )

        self.assertEqual(selected.candidate_id, "candidate_best")
        self.assertEqual(info["gate_status"], "fallback_best_score")

    def test_score_candidate_metrics_uses_mape_as_primary_metric(self):
        from src.federated_learning.adaptive_candidates import score_candidate_metrics

        score = score_candidate_metrics(
            metrics={"mape": 0.4, "mpe": -0.2},
            client_gap=0.3,
            update_norm=2.0,
            weights={"a": 0.5, "b": 0.5},
            previous_weights={"a": 0.6, "b": 0.4},
            profile="mape_primary",
        )

        self.assertGreater(score, 0.4)
        self.assertLess(score, 0.5)


if __name__ == "__main__":
    unittest.main()
