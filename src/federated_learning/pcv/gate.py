"""Deterministic PCV safety gate and planned anchor fallback."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from numbers import Real

from .schemas import CandidateAction, CandidateDecision, LocalCandidateVote
from .voting import aggregate_candidate_votes


_TRUST_REGION_L1 = 0.35
_BEST_LEGAL_TOLERANCE = 0.002
_ANCHOR_DEGRADATION_TOLERANCE = 0.001


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_non_negative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _validated_previous_weights(
    previous_weights: Mapping[str, float],
) -> tuple[dict[str, float], tuple[str, ...]]:
    if not isinstance(previous_weights, Mapping):
        raise TypeError("previous_weights must be a mapping")
    copied = dict(previous_weights)
    if not copied:
        raise ValueError("previous_weights must not be empty")
    client_ids = tuple(sorted(copied))
    for client_id in client_ids:
        _identity(client_id, name="previous weight client id")
    values = {
        client_id: _finite_non_negative(
            copied[client_id],
            name=f"previous weight for {client_id}",
        )
        for client_id in client_ids
    }
    if not math.isclose(
        math.fsum(values.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("previous_weights must sum to one")
    return values, client_ids


def _validated_candidates(
    candidates: Mapping[str, CandidateAction],
) -> dict[str, CandidateAction]:
    if not isinstance(candidates, Mapping):
        raise TypeError("candidates must be a mapping")
    copied = dict(candidates)
    if not copied:
        raise ValueError("candidates must not be empty")
    for candidate_id, candidate in copied.items():
        _identity(candidate_id, name="candidate mapping key")
        if type(candidate) is not CandidateAction:
            raise TypeError(
                "candidate values must be exact CandidateAction records"
            )
        if candidate.candidate_id != candidate_id:
            raise ValueError(
                "candidate mapping key must match CandidateAction.candidate_id"
            )
    return copied


def _validated_votes(
    votes: Iterable[LocalCandidateVote],
    *,
    candidate_ids: set[str],
    client_ids: tuple[str, ...],
) -> tuple[LocalCandidateVote, ...]:
    if isinstance(votes, (str, bytes)):
        raise TypeError("votes must be an iterable of LocalCandidateVote records")
    records = tuple(votes)
    clients_by_candidate = {
        candidate_id: set() for candidate_id in candidate_ids
    }
    for vote in records:
        if type(vote) is not LocalCandidateVote:
            raise TypeError("votes must contain exact LocalCandidateVote records")
        client_id = _identity(vote.client_id, name="vote client id")
        candidate_id = _identity(vote.candidate_id, name="vote candidate id")
        if candidate_id not in candidate_ids:
            raise ValueError("vote candidate IDs must exactly match candidate IDs")
        clients_by_candidate[candidate_id].add(client_id)

    expected_clients = set(client_ids)
    if any(
        observed_clients != expected_clients
        for observed_clients in clients_by_candidate.values()
    ):
        raise ValueError(
            "vote client IDs must exactly match previous_weights client IDs"
        )
    return records


def _validated_aggregate_mape(
    aggregate_mape: Mapping[str, float],
    candidate_ids: set[str],
) -> dict[str, float]:
    if not isinstance(aggregate_mape, Mapping):
        raise TypeError("aggregate_mape must be a mapping")
    copied = dict(aggregate_mape)
    if set(copied) != candidate_ids:
        raise ValueError(
            "aggregate_mape keys must exactly match candidate IDs"
        )
    return {
        candidate_id: _finite_non_negative(
            value,
            name=f"aggregate MAPE for {candidate_id}",
        )
        for candidate_id, value in copied.items()
    }


def _rejection(
    *,
    requested_candidate_id: str,
    stronger_anchor_id: str,
    gate_status: str,
    rationale: str,
    diagnostics: Mapping[str, object],
) -> CandidateDecision:
    return CandidateDecision(
        requested_candidate_id=requested_candidate_id,
        selected_candidate_id=stronger_anchor_id,
        gate_status=gate_status,
        rationale=rationale,
        diagnostics=diagnostics,
    )


def _exceeds_boundary(value: float, limit: float) -> bool:
    return value > limit


def select_with_gate(
    *,
    requested_candidate_id: str,
    candidates: Mapping[str, CandidateAction],
    votes: Iterable[LocalCandidateVote],
    aggregate_mape: Mapping[str, float],
    previous_weights: Mapping[str, float],
    stronger_anchor_id: str,
) -> CandidateDecision:
    """Apply the six ordered PCV gate checks to one requested candidate."""

    requested_candidate_id = _identity(
        requested_candidate_id,
        name="requested_candidate_id",
    )
    stronger_anchor_id = _identity(
        stronger_anchor_id,
        name="stronger_anchor_id",
    )
    candidate_map = _validated_candidates(candidates)
    previous, client_ids = _validated_previous_weights(previous_weights)
    if stronger_anchor_id not in candidate_map:
        raise ValueError("stronger anchor must exist among candidates")
    try:
        candidate_map[stronger_anchor_id].validate(client_ids)
    except ValueError as exc:
        raise ValueError("stronger anchor must be legal") from exc

    vote_records = _validated_votes(
        votes,
        candidate_ids=set(candidate_map),
        client_ids=client_ids,
    )
    vote_summaries = aggregate_candidate_votes(vote_records)
    if set(vote_summaries) != set(candidate_map):
        raise ValueError("vote candidate IDs must exactly match candidate IDs")
    scores = _validated_aggregate_mape(
        aggregate_mape,
        set(candidate_map),
    )

    base_diagnostics = {"stronger_anchor_id": stronger_anchor_id}

    if requested_candidate_id not in candidate_map:
        return _rejection(
            requested_candidate_id=requested_candidate_id,
            stronger_anchor_id=stronger_anchor_id,
            gate_status="rejected_missing_candidate",
            rationale="requested candidate does not exist",
            diagnostics=base_diagnostics,
        )

    requested = candidate_map[requested_candidate_id]
    try:
        requested.validate(client_ids)
    except ValueError as exc:
        return _rejection(
            requested_candidate_id=requested_candidate_id,
            stronger_anchor_id=stronger_anchor_id,
            gate_status="rejected_illegal_action",
            rationale="requested candidate action is illegal",
            diagnostics={
                **base_diagnostics,
                "schema_error": str(exc),
            },
        )

    requested_summary = vote_summaries[requested_candidate_id]
    if requested_summary.catastrophic_client_count:
        return _rejection(
            requested_candidate_id=requested_candidate_id,
            stronger_anchor_id=stronger_anchor_id,
            gate_status="rejected_client_degradation",
            rationale="requested candidate causes catastrophic client degradation",
            diagnostics={
                **base_diagnostics,
                "catastrophic_client_count": (
                    requested_summary.catastrophic_client_count
                ),
            },
        )

    l1_distance = math.fsum(
        abs(float(requested.weights[client_id]) - previous[client_id])
        for client_id in client_ids
    )
    if _exceeds_boundary(l1_distance, _TRUST_REGION_L1):
        return _rejection(
            requested_candidate_id=requested_candidate_id,
            stronger_anchor_id=stronger_anchor_id,
            gate_status="rejected_trust_region",
            rationale="requested candidate exceeds the L1 trust region",
            diagnostics={
                **base_diagnostics,
                "l1_distance": l1_distance,
                "l1_limit": _TRUST_REGION_L1,
            },
        )

    legal_candidate_ids: list[str] = []
    for candidate_id in sorted(candidate_map):
        try:
            candidate_map[candidate_id].validate(client_ids)
        except ValueError:
            continue
        legal_candidate_ids.append(candidate_id)
    best_legal_candidate_id = min(
        legal_candidate_ids,
        key=lambda candidate_id: (scores[candidate_id], candidate_id),
    )
    requested_mape = scores[requested_candidate_id]
    best_legal_mape = scores[best_legal_candidate_id]
    score_diagnostics = {
        **base_diagnostics,
        "l1_distance": l1_distance,
        "legal_candidate_ids": legal_candidate_ids,
        "best_legal_candidate_id": best_legal_candidate_id,
        "best_legal_mape": best_legal_mape,
        "requested_mape": requested_mape,
    }
    if _exceeds_boundary(
        requested_mape,
        best_legal_mape + _BEST_LEGAL_TOLERANCE,
    ):
        return _rejection(
            requested_candidate_id=requested_candidate_id,
            stronger_anchor_id=stronger_anchor_id,
            gate_status="rejected_not_near_best",
            rationale="requested candidate is not within the best legal MAPE",
            diagnostics=score_diagnostics,
        )

    anchor_mape = scores[stronger_anchor_id]
    final_diagnostics = {
        **score_diagnostics,
        "anchor_mape": anchor_mape,
    }
    if _exceeds_boundary(
        requested_mape,
        anchor_mape + _ANCHOR_DEGRADATION_TOLERANCE,
    ):
        return _rejection(
            requested_candidate_id=requested_candidate_id,
            stronger_anchor_id=stronger_anchor_id,
            gate_status="rejected_anchor_degradation",
            rationale="requested candidate is too far behind the stronger anchor",
            diagnostics=final_diagnostics,
        )

    return CandidateDecision(
        requested_candidate_id=requested_candidate_id,
        selected_candidate_id=requested_candidate_id,
        gate_status="accepted",
        rationale="requested candidate passed every deterministic gate",
        diagnostics=final_diagnostics,
    )
