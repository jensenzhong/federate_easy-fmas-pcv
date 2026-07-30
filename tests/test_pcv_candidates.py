import math

import pytest
import torch

from src.federated_learning.pcv.candidates import (
    build_anchor_candidates,
    build_deterministic_candidates,
    deduplicate_candidates,
    weighted_average_state,
)
from src.federated_learning.pcv.schemas import CandidateAction


def _candidate(
    candidate_id: str,
    weights: dict[str, float],
    *,
    optimizer: str = "fedyogi",
    source: str = "deterministic",
) -> CandidateAction:
    return CandidateAction(
        candidate_id=candidate_id,
        weights=weights,
        server_optimizer=optimizer,
        server_lr_scale=1.0,
        update_clip_norm=1.0 if optimizer == "fedyogi" else None,
        source=source,
        rationale="test candidate",
    )


def _telemetry(
    *,
    val_mapes: tuple[float, float, float] = (0.1, 0.2, 0.4),
    coherences: tuple[float, float, float] = (0.8, 0.4, -0.2),
) -> dict[str, dict[str, float]]:
    return {
        client_id: {
            "val_mape": val_mape,
            "cosine_to_mean": coherence,
        }
        for client_id, val_mape, coherence in zip(
            ("client_01", "client_02", "client_03"),
            val_mapes,
            coherences,
        )
    }


def test_anchor_candidates_are_always_present():
    anchors = build_anchor_candidates(
        sample_counts={"client_01": 2, "client_02": 3, "client_03": 5},
        fedyogi_lr_scale=1.0,
        fedyogi_clip_norm=1.0,
    )

    assert [item.candidate_id for item in anchors] == [
        "anchor_fedavg",
        "anchor_fedyogi",
    ]
    assert anchors[0].server_optimizer == "fedavg"
    assert anchors[1].server_optimizer == "fedyogi"
    assert dict(anchors[0].weights) == {
        "client_01": 0.2,
        "client_02": 0.3,
        "client_03": 0.5,
    }


@pytest.mark.parametrize(
    "sample_counts",
    [
        {"client_01": 0, "client_02": 3, "client_03": 5},
        {"client_01": -1, "client_02": 3, "client_03": 5},
    ],
)
def test_anchor_candidates_reject_nonpositive_sample_counts(sample_counts):
    with pytest.raises(ValueError, match="sample counts"):
        build_anchor_candidates(sample_counts, 1.0, 1.0)


def test_weighted_average_state_matches_candidate_weights():
    states = {
        "client_01": {"w": torch.tensor([1.0])},
        "client_02": {"w": torch.tensor([3.0])},
        "client_03": {"w": torch.tensor([5.0])},
    }

    result = weighted_average_state(
        states,
        {"client_01": 0.2, "client_02": 0.3, "client_03": 0.5},
    )

    assert torch.allclose(result["w"], torch.tensor([3.6]))


def test_weighted_average_state_clones_nonfloating_tensors():
    first_buffer = torch.tensor([7], dtype=torch.int64)
    states = {
        "client_01": {"w": torch.tensor([1.0]), "steps": first_buffer},
        "client_02": {"w": torch.tensor([3.0]), "steps": torch.tensor([9])},
    }

    result = weighted_average_state(
        states,
        {"client_01": 0.25, "client_02": 0.75},
    )

    assert torch.equal(result["steps"], first_buffer)
    assert result["steps"].data_ptr() != first_buffer.data_ptr()
    result["steps"][0] = 99
    assert first_buffer.item() == 7


@pytest.mark.parametrize(
    "states, match",
    [
        (
            {
                "client_01": {"w": torch.ones(2)},
                "client_02": {"other": torch.ones(2)},
            },
            "state keys",
        ),
        (
            {
                "client_01": {"w": torch.ones(2)},
                "client_02": {"w": torch.ones(3)},
            },
            "shape",
        ),
        (
            {
                "client_01": {"w": torch.ones(2, dtype=torch.float32)},
                "client_02": {"w": torch.ones(2, dtype=torch.float64)},
            },
            "dtype",
        ),
    ],
)
def test_weighted_average_state_rejects_incompatible_states(states, match):
    with pytest.raises(ValueError, match=match):
        weighted_average_state(states, {"client_01": 0.5, "client_02": 0.5})


def test_candidate_deduplication_keeps_eight_or_fewer():
    repeated = [
        _candidate(
            f"c{i}",
            {"client_01": 0.3, "client_02": 0.4, "client_03": 0.3},
        )
        for i in range(10)
    ]

    assert len(deduplicate_candidates(repeated, budget=8)) == 1


def test_candidate_deduplication_prioritizes_anchors_and_preserves_budget():
    weights = {"client_01": 0.2, "client_02": 0.3, "client_03": 0.5}
    candidates = [
        _candidate("derived", weights),
        _candidate("anchor_fedyogi", weights, source="anchor"),
        _candidate(
            "anchor_fedavg",
            weights,
            optimizer="fedavg",
            source="anchor",
        ),
        _candidate(
            "derived_2",
            {"client_01": 0.3, "client_02": 0.3, "client_03": 0.4},
        ),
    ]

    selected = deduplicate_candidates(candidates, budget=2)

    assert [item.candidate_id for item in selected] == [
        "anchor_fedavg",
        "anchor_fedyogi",
    ]


def test_candidate_deduplication_validates_actions_beyond_budget():
    valid_weights = {
        "client_01": 0.2,
        "client_02": 0.3,
        "client_03": 0.5,
    }
    candidates = [
        _candidate("anchor_fedavg", valid_weights, optimizer="fedavg", source="anchor"),
        _candidate("anchor_fedyogi", valid_weights, source="anchor"),
        _candidate(
            "invalid_after_budget",
            {"client_01": 0.2, "client_02": 0.3, "client_03": 0.4},
        ),
    ]

    with pytest.raises(ValueError, match="sum to one"):
        deduplicate_candidates(candidates, budget=2)


def test_deterministic_candidates_have_fixed_order_and_projected_weights():
    candidates = build_deterministic_candidates(
        sample_counts={"client_01": 2, "client_02": 3, "client_03": 5},
        previous_weights={
            "client_01": 0.6,
            "client_02": 0.2,
            "client_03": 0.2,
        },
        telemetry=_telemetry(),
        fedyogi_lr_scale=1.0,
        fedyogi_clip_norm=1.0,
    )

    assert [item.candidate_id for item in candidates] == [
        "anchor_fedavg",
        "anchor_fedyogi",
        "deterministic_uniform",
        "deterministic_previous",
        "deterministic_positive_coherence",
        "deterministic_inverse_val_mape",
        "deterministic_error_compensation",
        "deterministic_size_previous_blend",
    ]
    for candidate in candidates[2:]:
        assert sum(candidate.weights.values()) == pytest.approx(1.0)
        assert all(0.05 <= weight <= 0.80 for weight in candidate.weights.values())


def test_extreme_derived_weights_are_projected_to_bounds_and_sum_to_one():
    candidates = build_deterministic_candidates(
        sample_counts={"client_01": 2, "client_02": 3, "client_03": 5},
        previous_weights={
            "client_01": 0.6,
            "client_02": 0.2,
            "client_03": 0.2,
        },
        telemetry=_telemetry(
            val_mapes=(1000.0, 0.001, 0.001),
            coherences=(1.0, 1.0, 1.0),
        ),
        fedyogi_lr_scale=1.0,
        fedyogi_clip_norm=1.0,
    )

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for candidate_id in (
        "deterministic_inverse_val_mape",
        "deterministic_error_compensation",
    ):
        weights = by_id[candidate_id].weights.values()
        assert sum(weights) == pytest.approx(1.0)
        assert all(0.05 <= weight <= 0.80 for weight in weights)


def test_missing_previous_weights_reuses_size_weights_without_losing_anchors():
    candidates = build_deterministic_candidates(
        sample_counts={"client_01": 2, "client_02": 3, "client_03": 5},
        previous_weights=None,
        telemetry=_telemetry(),
        fedyogi_lr_scale=1.0,
        fedyogi_clip_norm=1.0,
    )

    assert [item.candidate_id for item in candidates[:2]] == [
        "anchor_fedavg",
        "anchor_fedyogi",
    ]
    assert "deterministic_previous" not in {
        item.candidate_id for item in candidates
    }
    assert "deterministic_size_previous_blend" not in {
        item.candidate_id for item in candidates
    }


def test_nonfinite_telemetry_rejects_only_affected_candidates():
    telemetry = _telemetry(val_mapes=(0.1, math.nan, 0.4))

    candidates = build_deterministic_candidates(
        sample_counts={"client_01": 2, "client_02": 3, "client_03": 5},
        previous_weights={
            "client_01": 0.6,
            "client_02": 0.2,
            "client_03": 0.2,
        },
        telemetry=telemetry,
        fedyogi_lr_scale=1.0,
        fedyogi_clip_norm=1.0,
    )

    candidate_ids = [item.candidate_id for item in candidates]
    assert candidate_ids[:2] == ["anchor_fedavg", "anchor_fedyogi"]
    assert "deterministic_positive_coherence" in candidate_ids
    assert "deterministic_inverse_val_mape" not in candidate_ids
    assert "deterministic_error_compensation" not in candidate_ids


def test_missing_telemetry_rejects_affected_candidate_without_imputation():
    telemetry = _telemetry()
    del telemetry["client_02"]["cosine_to_mean"]

    candidates = build_deterministic_candidates(
        sample_counts={"client_01": 2, "client_02": 3, "client_03": 5},
        previous_weights={
            "client_01": 0.6,
            "client_02": 0.2,
            "client_03": 0.2,
        },
        telemetry=telemetry,
        fedyogi_lr_scale=1.0,
        fedyogi_clip_norm=1.0,
    )

    candidate_ids = {item.candidate_id for item in candidates}
    assert "deterministic_positive_coherence" not in candidate_ids
    assert "deterministic_inverse_val_mape" in candidate_ids
    assert "deterministic_error_compensation" in candidate_ids
