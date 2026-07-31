"""Deterministic aggregation of client-local PCV candidate votes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable

from .schemas import LocalCandidateVote


@dataclass(frozen=True)
class CandidateVoteSummary:
    candidate_id: str
    weighted_mape: float
    weighted_rmse: float
    mean_rank: float
    minimum_confidence: float
    catastrophic_client_count: int


def _non_empty_identity(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_real(
    value: object,
    *,
    name: str,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if non_negative and value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validated_vote(vote: object) -> LocalCandidateVote:
    if not isinstance(vote, LocalCandidateVote):
        raise TypeError("votes must contain LocalCandidateVote records")
    _non_empty_identity(vote.client_id, name="client_id")
    _non_empty_identity(vote.candidate_id, name="candidate_id")
    _positive_integer(vote.sample_count, name="sample_count")
    _finite_real(vote.val_mape, name="val_mape", non_negative=True)
    _finite_real(vote.val_rmse, name="val_rmse", non_negative=True)
    _finite_real(vote.relative_mape, name="relative_mape")
    _finite_real(vote.relative_rmse, name="relative_rmse")
    _positive_integer(vote.rank, name="rank")
    confidence = _finite_real(vote.confidence, name="confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between zero and one")
    if type(vote.catastrophic_degradation) is not bool:
        raise TypeError("catastrophic_degradation must be a boolean")
    return vote


def _weighted_average(
    votes: list[LocalCandidateVote],
    field_name: str,
) -> float:
    total_count = sum(int(vote.sample_count) for vote in votes)
    weighted = math.fsum(
        float(getattr(vote, field_name))
        * (int(vote.sample_count) / total_count)
        for vote in votes
    )
    if not math.isfinite(weighted):
        raise ValueError(f"weighted {field_name} must be finite")
    return weighted


def aggregate_candidate_votes(
    votes: Iterable[LocalCandidateVote],
) -> dict[str, CandidateVoteSummary]:
    """Aggregate complete client ballots using validation sample counts."""

    vote_list = [_validated_vote(vote) for vote in votes]
    if not vote_list:
        raise ValueError("candidate aggregation requires at least one vote")

    grouped: dict[str, list[LocalCandidateVote]] = {}
    seen_ballots: set[tuple[str, str]] = set()
    sample_count_by_client: dict[str, int] = {}
    for vote in vote_list:
        ballot = (vote.client_id, vote.candidate_id)
        if ballot in seen_ballots:
            raise ValueError("duplicate client candidate vote")
        seen_ballots.add(ballot)
        grouped.setdefault(vote.candidate_id, []).append(vote)

        previous_count = sample_count_by_client.setdefault(
            vote.client_id,
            int(vote.sample_count),
        )
        if previous_count != vote.sample_count:
            raise ValueError(
                "client sample_count must be consistent across candidates"
            )

    expected_clients = {vote.client_id for vote in vote_list}
    if any(
        {vote.client_id for vote in candidate_votes} != expected_clients
        for candidate_votes in grouped.values()
    ):
        raise ValueError("all candidates must be evaluated by the same clients")

    summaries: dict[str, CandidateVoteSummary] = {}
    for candidate_id in sorted(grouped):
        candidate_votes = grouped[candidate_id]
        mean_rank = math.fsum(float(vote.rank) for vote in candidate_votes) / len(
            candidate_votes
        )
        if not math.isfinite(mean_rank):
            raise ValueError("mean rank must be finite")
        summaries[candidate_id] = CandidateVoteSummary(
            candidate_id=candidate_id,
            weighted_mape=_weighted_average(candidate_votes, "val_mape"),
            weighted_rmse=_weighted_average(candidate_votes, "val_rmse"),
            mean_rank=mean_rank,
            minimum_confidence=min(
                float(vote.confidence) for vote in candidate_votes
            ),
            catastrophic_client_count=sum(
                vote.catastrophic_degradation for vote in candidate_votes
            ),
        )
    return summaries
