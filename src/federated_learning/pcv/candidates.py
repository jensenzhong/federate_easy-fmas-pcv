from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .schemas import CandidateAction


StateDict = Mapping[str, torch.Tensor]

_MIN_WEIGHT = 0.05
_MAX_WEIGHT = 0.80
_MAX_CANDIDATES = 8
_ANCHOR_ORDER = {
    "anchor_fedavg": 0,
    "anchor_fedyogi": 1,
}


def _validated_size_weights(
    sample_counts: Mapping[str, int],
) -> dict[str, float]:
    if not sample_counts:
        raise ValueError("sample counts must not be empty")
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        for count in sample_counts.values()
    ):
        raise ValueError("sample counts must be positive integers")
    total = sum(sample_counts.values())
    return {
        client_id: count / total
        for client_id, count in sample_counts.items()
    }


def build_anchor_candidates(
    sample_counts: Mapping[str, int],
    fedyogi_lr_scale: float,
    fedyogi_clip_norm: float | None,
) -> list[CandidateAction]:
    size_weights = _validated_size_weights(sample_counts)
    client_ids = tuple(sample_counts)
    anchors = [
        CandidateAction(
            candidate_id="anchor_fedavg",
            weights=size_weights,
            server_optimizer="fedavg",
            server_lr_scale=1.0,
            update_clip_norm=None,
            source="anchor",
            rationale="strict FedAvg size-weight anchor",
        ),
        CandidateAction(
            candidate_id="anchor_fedyogi",
            weights=size_weights,
            server_optimizer="fedyogi",
            server_lr_scale=fedyogi_lr_scale,
            update_clip_norm=fedyogi_clip_norm,
            source="anchor",
            rationale="strict FedYogi size-weight anchor",
        ),
    ]
    for anchor in anchors:
        anchor.validate(client_ids)
    return anchors


def weighted_average_state(
    states: Mapping[str, StateDict],
    weights: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must not be empty")
    client_ids = tuple(states)
    if set(weights) != set(client_ids):
        raise ValueError("weights must match state client ids")
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0.0
        for weight in weights.values()
    ):
        raise ValueError("weights must be finite non-negative numbers")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("weights must sum to one")

    first_state = states[client_ids[0]]
    expected_keys = tuple(first_state)
    expected_key_set = set(expected_keys)
    for client_id in client_ids:
        state = states[client_id]
        if set(state) != expected_key_set:
            raise ValueError("state keys must match")
        for key in expected_keys:
            tensor = state[key]
            reference = first_state[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("state values must be tensors")
            if tensor.shape != reference.shape:
                raise ValueError(f"state tensor shape mismatch for {key}")
            if tensor.dtype != reference.dtype:
                raise ValueError(f"state tensor dtype mismatch for {key}")
            if tensor.device != reference.device:
                raise ValueError(f"state tensor device mismatch for {key}")

    averaged: dict[str, torch.Tensor] = {}
    for key in expected_keys:
        reference = first_state[key]
        if not torch.is_floating_point(reference):
            averaged[key] = reference.clone()
            continue
        value = reference.clone().mul(float(weights[client_ids[0]]))
        for client_id in client_ids[1:]:
            value.add_(
                states[client_id][key],
                alpha=float(weights[client_id]),
            )
        averaged[key] = value
    return averaged


def _candidate_key(candidate: CandidateAction) -> tuple[Any, ...]:
    return (
        tuple(
            sorted(
                (client_id, round(float(weight), 12))
                for client_id, weight in candidate.weights.items()
            )
        ),
        candidate.server_optimizer,
        round(float(candidate.server_lr_scale), 12),
        (
            None
            if candidate.update_clip_norm is None
            else round(float(candidate.update_clip_norm), 12)
        ),
    )


def _validate_anchor_identity(candidate: CandidateAction) -> bool:
    is_reserved_id = candidate.candidate_id in _ANCHOR_ORDER
    if candidate.source == "anchor" and not is_reserved_id:
        raise ValueError("anchor source requires a canonical anchor id")
    if not is_reserved_id:
        return False
    if candidate.source != "anchor":
        raise ValueError("reserved anchor id requires anchor source")
    if candidate.candidate_id == "anchor_fedavg" and (
        candidate.server_optimizer != "fedavg"
        or candidate.server_lr_scale != 1.0
        or candidate.update_clip_norm is not None
    ):
        raise ValueError("anchor_fedavg has invalid anchor semantics")
    if (
        candidate.candidate_id == "anchor_fedyogi"
        and candidate.server_optimizer != "fedyogi"
    ):
        raise ValueError("anchor_fedyogi has invalid anchor semantics")
    return True


def deduplicate_candidates(
    candidates: Sequence[CandidateAction],
    budget: int = _MAX_CANDIDATES,
) -> list[CandidateAction]:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("candidate budget must be a positive integer")
    effective_budget = min(budget, _MAX_CANDIDATES)
    client_ids = tuple(candidates[0].weights) if candidates else ()

    canonical_by_id: dict[str, CandidateAction] = {}
    proposals: list[CandidateAction] = []
    for candidate in candidates:
        candidate.validate(client_ids)
        if not _validate_anchor_identity(candidate):
            proposals.append(candidate)
            continue
        existing = canonical_by_id.get(candidate.candidate_id)
        if existing is None:
            canonical_by_id[candidate.candidate_id] = candidate
        elif _candidate_key(existing) != _candidate_key(candidate):
            raise ValueError(
                f"conflicting canonical anchor: {candidate.candidate_id}"
            )

    canonical_anchors = [
        canonical_by_id[candidate_id]
        for candidate_id in _ANCHOR_ORDER
        if candidate_id in canonical_by_id
    ]
    if len(canonical_anchors) > effective_budget:
        raise ValueError("candidate budget cannot discard anchors")

    selected: list[CandidateAction] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in [*canonical_anchors, *proposals]:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) == effective_budget:
            break
    return selected


def _project_weights(
    raw_weights: Mapping[str, float],
    client_ids: tuple[str, ...],
) -> dict[str, float]:
    if set(raw_weights) != set(client_ids):
        raise ValueError("derived weights must match client ids")
    values = {
        client_id: float(raw_weights[client_id])
        for client_id in client_ids
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in values.values()
    ):
        raise ValueError("derived weights must be finite and non-negative")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("derived weights must have positive mass")
    if not (
        len(client_ids) * _MIN_WEIGHT <= 1.0
        <= len(client_ids) * _MAX_WEIGHT
    ):
        raise ValueError("weight bounds are infeasible for client count")

    normalized = {
        client_id: value / total
        for client_id, value in values.items()
    }
    lower = min(
        value - _MAX_WEIGHT
        for value in normalized.values()
    )
    upper = max(
        value - _MIN_WEIGHT
        for value in normalized.values()
    )
    projected: dict[str, float] = {}
    for _ in range(100):
        shift = (lower + upper) / 2.0
        projected = {
            client_id: max(
                _MIN_WEIGHT,
                min(_MAX_WEIGHT, normalized[client_id] - shift),
            )
            for client_id in client_ids
        }
        if sum(projected.values()) > 1.0:
            lower = shift
        else:
            upper = shift

    residual = 1.0 - sum(projected.values())
    if abs(residual) > 1e-12:
        for client_id in client_ids:
            room = (
                _MAX_WEIGHT - projected[client_id]
                if residual > 0.0
                else projected[client_id] - _MIN_WEIGHT
            )
            adjustment = math.copysign(min(abs(residual), room), residual)
            projected[client_id] += adjustment
            residual -= adjustment
            if abs(residual) <= 1e-12:
                break
    return projected


def _validate_telemetry(
    telemetry: Mapping[str, Any] | None,
    sample_counts: Mapping[str, int],
) -> None:
    if telemetry is None:
        return
    if not isinstance(telemetry, Mapping):
        raise ValueError("telemetry must be a mapping")
    unknown_client_ids = set(telemetry) - set(sample_counts)
    if unknown_client_ids:
        raise ValueError(
            f"unknown telemetry client: {sorted(unknown_client_ids)[0]}"
        )

    for client_id, record in telemetry.items():
        if isinstance(record, Mapping):
            record_client_id = record.get("client_id")
            record_sample_count = record.get("sample_count")
        else:
            record_client_id = getattr(record, "client_id", None)
            record_sample_count = getattr(record, "sample_count", None)
        if record_client_id != client_id:
            raise ValueError(
                f"telemetry client_id mismatch for {client_id}"
            )
        if (
            isinstance(record_sample_count, bool)
            or not isinstance(record_sample_count, int)
            or record_sample_count <= 0
            or record_sample_count != sample_counts[client_id]
        ):
            raise ValueError(
                f"telemetry sample_count mismatch for {client_id}"
            )


def _telemetry_value(
    telemetry: Mapping[str, Any],
    client_id: str,
    field: str,
) -> float | None:
    client_telemetry = telemetry.get(client_id)
    if client_telemetry is None:
        return None
    if isinstance(client_telemetry, Mapping):
        value = client_telemetry.get(field)
    else:
        value = getattr(client_telemetry, field, None)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return None
    return float(value)


def _derived_candidate(
    candidate_id: str,
    weights: Mapping[str, float],
    *,
    client_ids: tuple[str, ...],
    fedyogi_lr_scale: float,
    fedyogi_clip_norm: float | None,
    rationale: str,
) -> CandidateAction:
    candidate = CandidateAction(
        candidate_id=candidate_id,
        weights=_project_weights(weights, client_ids),
        server_optimizer="fedyogi",
        server_lr_scale=fedyogi_lr_scale,
        update_clip_norm=fedyogi_clip_norm,
        source="deterministic",
        rationale=rationale,
    )
    candidate.validate(client_ids)
    return candidate


def build_deterministic_candidates(
    sample_counts: Mapping[str, int],
    previous_weights: Mapping[str, float] | None,
    telemetry: Mapping[str, Any] | None,
    fedyogi_lr_scale: float,
    fedyogi_clip_norm: float | None,
    budget: int = _MAX_CANDIDATES,
) -> list[CandidateAction]:
    if budget < len(_ANCHOR_ORDER):
        raise ValueError("candidate budget cannot discard anchors")
    client_ids = tuple(sample_counts)
    size_weights = _validated_size_weights(sample_counts)
    _validate_telemetry(telemetry, sample_counts)
    previous = size_weights if previous_weights is None else previous_weights
    candidates = build_anchor_candidates(
        sample_counts,
        fedyogi_lr_scale,
        fedyogi_clip_norm,
    )

    candidates.append(
        _derived_candidate(
            "deterministic_uniform",
            {client_id: 1.0 for client_id in client_ids},
            client_ids=client_ids,
            fedyogi_lr_scale=fedyogi_lr_scale,
            fedyogi_clip_norm=fedyogi_clip_norm,
            rationale="uniform client weights under FedYogi",
        )
    )
    candidates.append(
        _derived_candidate(
            "deterministic_previous",
            previous,
            client_ids=client_ids,
            fedyogi_lr_scale=fedyogi_lr_scale,
            fedyogi_clip_norm=fedyogi_clip_norm,
            rationale="previous accepted client weights under FedYogi",
        )
    )

    if telemetry is not None:
        coherences = {
            client_id: _telemetry_value(
                telemetry,
                client_id,
                "cosine_to_mean",
            )
            for client_id in client_ids
        }
        if all(value is not None for value in coherences.values()):
            positive_coherence = {
                client_id: size_weights[client_id]
                * max(float(coherences[client_id]), 0.0)
                for client_id in client_ids
            }
            if sum(positive_coherence.values()) > 0.0:
                candidates.append(
                    _derived_candidate(
                        "deterministic_positive_coherence",
                        positive_coherence,
                        client_ids=client_ids,
                        fedyogi_lr_scale=fedyogi_lr_scale,
                        fedyogi_clip_norm=fedyogi_clip_norm,
                        rationale="positive-coherence client weights under FedYogi",
                    )
                )

        val_mapes = {
            client_id: _telemetry_value(
                telemetry,
                client_id,
                "val_mape",
            )
            for client_id in client_ids
        }
        if all(
            value is not None and value > 0.0
            for value in val_mapes.values()
        ):
            candidates.append(
                _derived_candidate(
                    "deterministic_inverse_val_mape",
                    {
                        client_id: 1.0 / float(val_mapes[client_id])
                        for client_id in client_ids
                    },
                    client_ids=client_ids,
                    fedyogi_lr_scale=fedyogi_lr_scale,
                    fedyogi_clip_norm=fedyogi_clip_norm,
                    rationale="inverse validation MAPE weights under FedYogi",
                )
            )
        if all(
            value is not None and value >= 0.0
            for value in val_mapes.values()
        ) and any(float(value) > 0.0 for value in val_mapes.values()):
            candidates.append(
                _derived_candidate(
                    "deterministic_error_compensation",
                    {
                        client_id: float(val_mapes[client_id])
                        for client_id in client_ids
                    },
                    client_ids=client_ids,
                    fedyogi_lr_scale=fedyogi_lr_scale,
                    fedyogi_clip_norm=fedyogi_clip_norm,
                    rationale="validation-error compensation weights under FedYogi",
                )
            )

    candidates.append(
        _derived_candidate(
            "deterministic_size_previous_blend",
            {
                client_id: (
                    0.5 * size_weights[client_id]
                    + 0.5 * float(previous[client_id])
                )
                for client_id in client_ids
            },
            client_ids=client_ids,
            fedyogi_lr_scale=fedyogi_lr_scale,
            fedyogi_clip_norm=fedyogi_clip_norm,
            rationale="equal blend of size and previous accepted weights",
        )
    )
    return deduplicate_candidates(candidates, budget=budget)
