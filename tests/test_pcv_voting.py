import math
from dataclasses import FrozenInstanceError

import pytest

from src.federated_learning.pcv.schemas import LocalCandidateVote
from src.federated_learning.pcv.voting import (
    CandidateVoteSummary,
    aggregate_candidate_votes,
)


def _vote(client_id="c1", candidate_id="x", **overrides):
    values = {
        "sample_count": 10,
        "val_mape": 0.30,
        "val_rmse": 8.0,
        "relative_mape": -0.10,
        "relative_rmse": -0.05,
        "rank": 1,
        "confidence": 0.9,
        "catastrophic_degradation": False,
    }
    values.update(overrides)
    return LocalCandidateVote(
        client_id=client_id,
        candidate_id=candidate_id,
        **values,
    )


def test_vote_aggregation_is_weighted_by_validation_count():
    votes = [
        _vote("c1", sample_count=90, val_mape=0.30, val_rmse=10.0),
        _vote("c2", sample_count=10, val_mape=0.70, val_rmse=20.0),
    ]

    result = aggregate_candidate_votes(votes)

    assert result["x"].weighted_mape == pytest.approx(0.34)
    assert result["x"].weighted_rmse == pytest.approx(11.0)


def test_vote_summary_reports_rank_confidence_and_catastrophic_clients():
    votes = [
        _vote("c1", rank=1, confidence=0.8),
        _vote(
            "c2",
            rank=3,
            confidence=0.4,
            catastrophic_degradation=True,
        ),
    ]

    summary = aggregate_candidate_votes(votes)["x"]

    assert summary == CandidateVoteSummary(
        candidate_id="x",
        weighted_mape=0.30,
        weighted_rmse=8.0,
        mean_rank=2.0,
        minimum_confidence=0.4,
        catastrophic_client_count=1,
    )
    with pytest.raises(FrozenInstanceError):
        summary.mean_rank = 1.0


def test_vote_aggregation_returns_candidates_in_deterministic_id_order():
    result = aggregate_candidate_votes(
        [_vote("c1", "z"), _vote("c1", "a")]
    )

    assert list(result) == ["a", "z"]


def test_vote_aggregation_handles_huge_counts_without_weighted_overflow():
    huge = 10**400
    result = aggregate_candidate_votes(
        [
            _vote("c1", sample_count=huge, val_mape=1.0e308),
            _vote("c2", sample_count=huge, val_mape=1.0e308),
        ]
    )

    assert result["x"].weighted_mape == pytest.approx(1.0e308)
    assert math.isfinite(result["x"].weighted_mape)


def test_vote_aggregation_rejects_empty_votes():
    with pytest.raises(ValueError, match="at least one vote"):
        aggregate_candidate_votes([])


def test_vote_aggregation_rejects_duplicate_client_candidate_vote():
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_candidate_votes([_vote("c1"), _vote("c1")])


def test_vote_aggregation_rejects_incomplete_candidate_ballots():
    votes = [
        _vote("c1", "x"),
        _vote("c2", "x"),
        _vote("c1", "y"),
    ]

    with pytest.raises(ValueError, match="same clients"):
        aggregate_candidate_votes(votes)


def test_vote_aggregation_rejects_inconsistent_client_sample_counts():
    votes = [
        _vote("c1", "x", sample_count=10),
        _vote("c1", "y", sample_count=11),
    ]

    with pytest.raises(ValueError, match="sample_count"):
        aggregate_candidate_votes(votes)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_count": 0},
        {"sample_count": -1},
        {"sample_count": True},
        {"val_mape": float("nan")},
        {"val_mape": float("inf")},
        {"val_mape": -0.1},
        {"val_rmse": float("nan")},
        {"val_rmse": float("inf")},
        {"val_rmse": -0.1},
        {"relative_mape": float("nan")},
        {"relative_mape": float("inf")},
        {"relative_rmse": float("nan")},
        {"relative_rmse": float("inf")},
        {"rank": 0},
        {"rank": 1.5},
        {"rank": float("inf")},
        {"rank": True},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"catastrophic_degradation": 1},
    ],
)
def test_vote_aggregation_rejects_invalid_vote_numbers(overrides):
    with pytest.raises((TypeError, ValueError)):
        aggregate_candidate_votes([_vote(**overrides)])


@pytest.mark.parametrize(
    ("client_id", "candidate_id"),
    [
        ("", "x"),
        ("  ", "x"),
        (1, "x"),
        ("c1", ""),
        ("c1", "  "),
        ("c1", 1),
    ],
)
def test_vote_aggregation_rejects_invalid_vote_identity(
    client_id,
    candidate_id,
):
    with pytest.raises((TypeError, ValueError)):
        aggregate_candidate_votes([_vote(client_id, candidate_id)])
