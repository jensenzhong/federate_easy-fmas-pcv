import pytest


def test_coherence_weights_increase_for_aligned_client():
    from src.federated_learning.generated_strategy import compute_coherence_weights

    diagnostics = {
        "client_1": {"sample_size_weight": 0.5, "cosine_to_mean_update": 1.0},
        "client_2": {"sample_size_weight": 0.5, "cosine_to_mean_update": 0.0},
    }

    weights = compute_coherence_weights(diagnostics, min_client_weight=0.05, max_client_weight=0.8)

    assert weights["client_1"] > weights["client_2"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(0.05 <= weight <= 0.8 for weight in weights.values())


def test_coherence_weights_fall_back_to_size_when_all_alignment_zero():
    from src.federated_learning.generated_strategy import compute_coherence_weights

    diagnostics = {
        "client_1": {"sample_size_weight": 0.25, "cosine_to_mean_update": -0.3},
        "client_2": {"sample_size_weight": 0.75, "cosine_to_mean_update": 0.0},
    }

    weights = compute_coherence_weights(diagnostics, min_client_weight=0.05, max_client_weight=0.8)

    assert weights["client_1"] == pytest.approx(0.25)
    assert weights["client_2"] == pytest.approx(0.75)


def test_project_generated_strategy_applies_stability_and_drift_constraints():
    from src.federated_learning.generated_strategy import project_generated_strategy

    diagnostics = {
        "client_1": {
            "sample_size_weight": 0.5,
            "cosine_to_mean_update": -0.2,
            "update_norm": 1.0,
        },
        "client_2": {
            "sample_size_weight": 0.5,
            "cosine_to_mean_update": 0.8,
            "update_norm": 1.0,
        },
    }

    result = project_generated_strategy(
        generated_weights={"client_1": 0.8, "client_2": 0.2},
        diagnostics=diagnostics,
        previous_weights={"client_1": 0.5, "client_2": 0.5},
        min_client_weight=0.05,
        max_client_weight=0.8,
        l1_change_limit=0.4,
    )

    assert result.weights["client_1"] <= 0.5
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert result.constraint_status["negative_coherence_limited"] == ["client_1"]
    assert result.constraint_status["l1_projected"] is True


def test_project_generated_strategy_limits_high_norm_client_to_size_weight():
    from src.federated_learning.generated_strategy import project_generated_strategy

    diagnostics = {
        "client_1": {
            "sample_size_weight": 0.3,
            "cosine_to_mean_update": 0.9,
            "update_norm": 10.0,
        },
        "client_2": {
            "sample_size_weight": 0.4,
            "cosine_to_mean_update": 0.4,
            "update_norm": 1.0,
        },
        "client_3": {
            "sample_size_weight": 0.3,
            "cosine_to_mean_update": 0.3,
            "update_norm": 1.0,
        },
    }

    result = project_generated_strategy(
        generated_weights={"client_1": 0.75, "client_2": 0.15, "client_3": 0.10},
        diagnostics=diagnostics,
        previous_weights=None,
    )

    assert result.weights["client_1"] <= 0.3
    assert result.constraint_status["high_norm_limited"] == ["client_1"]


def test_project_generated_strategy_limits_distance_from_generalization_anchor():
    from src.federated_learning.adaptive_candidates import l1_weight_distance
    from src.federated_learning.generated_strategy import (
        compute_robust_prior_weights,
        project_generated_strategy,
    )

    diagnostics = {
        "client_1": {"sample_size_weight": 0.34, "cosine_to_mean_update": 0.10, "update_norm": 1.0},
        "client_2": {"sample_size_weight": 0.34, "cosine_to_mean_update": 0.95, "update_norm": 1.0},
        "client_3": {"sample_size_weight": 0.32, "cosine_to_mean_update": 0.95, "update_norm": 1.0},
    }
    robust_prior = compute_robust_prior_weights(diagnostics, coherence_blend=0.30)

    result = project_generated_strategy(
        generated_weights={"client_1": 0.10, "client_2": 0.50, "client_3": 0.40},
        diagnostics=diagnostics,
        anchor_weights=robust_prior,
        anchor_l1_limit=0.02,
    )

    assert l1_weight_distance(result.weights, robust_prior) <= 0.0200001
    assert result.constraint_status["anchor_projected"] is True


def test_project_generated_strategy_snaps_near_size_balanced_weights_to_size_prior():
    from src.federated_learning.generated_strategy import project_generated_strategy

    diagnostics = {
        "Client 1": {"sample_size_weight": 0.34, "cosine_to_mean_update": 0.8, "update_norm": 1.0},
        "Client 2": {"sample_size_weight": 0.34, "cosine_to_mean_update": 0.7, "update_norm": 1.0},
        "Client 3": {"sample_size_weight": 0.32, "cosine_to_mean_update": 0.6, "update_norm": 1.0},
    }

    result = project_generated_strategy(
        generated_weights={"Client 1": 0.338, "Client 2": 0.343, "Client 3": 0.319},
        diagnostics=diagnostics,
        decision_type="balanced",
        snap_to_size_l1_threshold=0.02,
    )

    assert result.constraint_status["snapped_to_size_prior"] is True
    assert result.weights["Client 1"] == pytest.approx(0.34)
    assert result.weights["Client 2"] == pytest.approx(0.34)
    assert result.weights["Client 3"] == pytest.approx(0.32)
