"""Generated aggregation strategy constraints for LLM-GCA-FedYogi-TR."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Optional

from src.federated_learning.adaptive_candidates import l1_weight_distance, normalize_weights


Weights = Dict[str, float]


@dataclass
class StrategyProjectionResult:
    weights: Weights
    constraint_status: Dict[str, Any]


def _size_weights(diagnostics: Dict[str, Dict[str, float]]) -> Weights:
    client_ids = list(diagnostics.keys())
    weights = {
        client_id: max(float(row.get("sample_size_weight", 0.0)), 0.0)
        for client_id, row in diagnostics.items()
    }
    total = sum(weights.values())
    if total <= 0 and client_ids:
        return {client_id: 1.0 / len(client_ids) for client_id in client_ids}
    return {client_id: value / total for client_id, value in weights.items()}


def compute_coherence_weights(
    diagnostics: Dict[str, Dict[str, float]],
    min_client_weight: float = 0.05,
    max_client_weight: float = 0.80,
) -> Weights:
    client_ids = list(diagnostics.keys())
    if not client_ids:
        return {}
    size_weights = _size_weights(diagnostics)
    alignments = {
        client_id: max(float(row.get("cosine_to_mean_update", 0.0)), 0.0)
        for client_id, row in diagnostics.items()
    }
    if all(value <= 0.0 for value in alignments.values()):
        return normalize_weights(size_weights, client_ids, min_client_weight, max_client_weight)

    raw = {
        client_id: size_weights[client_id] * (0.5 + alignments[client_id])
        for client_id in client_ids
    }
    return normalize_weights(raw, client_ids, min_client_weight, max_client_weight)


def compute_robust_prior_weights(
    diagnostics: Dict[str, Dict[str, float]],
    coherence_blend: float = 0.30,
    min_client_weight: float = 0.05,
    max_client_weight: float = 0.80,
) -> Weights:
    """Blend size and coherence priors to keep a generalization anchor."""
    client_ids = list(diagnostics.keys())
    if not client_ids:
        return {}
    blend = max(0.0, min(1.0, float(coherence_blend)))
    size_weights = _size_weights(diagnostics)
    coherence_weights = compute_coherence_weights(
        diagnostics,
        min_client_weight=min_client_weight,
        max_client_weight=max_client_weight,
    )
    raw = {
        client_id: (1.0 - blend) * size_weights[client_id] + blend * coherence_weights[client_id]
        for client_id in client_ids
    }
    return normalize_weights(raw, client_ids, min_client_weight, max_client_weight)


def _normalize_with_caps(
    weights: Weights,
    client_ids: list[str],
    min_weight: float,
    max_weight: float,
    caps: Optional[Dict[str, float]] = None,
) -> Weights:
    caps = caps or {}
    effective_max = {
        client_id: min(float(max_weight), float(caps.get(client_id, max_weight)))
        for client_id in client_ids
    }
    projected = {
        client_id: min(effective_max[client_id], max(float(min_weight), float(weights.get(client_id, 0.0))))
        for client_id in client_ids
    }
    for _ in range(50):
        total = sum(projected.values())
        residual = 1.0 - total
        if abs(residual) <= 1e-10:
            break
        if residual > 0:
            free = [cid for cid in client_ids if projected[cid] < effective_max[cid] - 1e-12]
            if not free:
                break
            room = sum(effective_max[cid] - projected[cid] for cid in free)
            for cid in free:
                projected[cid] += residual * ((effective_max[cid] - projected[cid]) / room) if room > 0 else residual / len(free)
        else:
            free = [cid for cid in client_ids if projected[cid] > min_weight + 1e-12]
            if not free:
                break
            room = sum(projected[cid] - min_weight for cid in free)
            for cid in free:
                projected[cid] += residual * ((projected[cid] - min_weight) / room) if room > 0 else residual / len(free)
    total = sum(projected.values())
    if total <= 0:
        return {client_id: 1.0 / len(client_ids) for client_id in client_ids}
    return {client_id: float(projected[client_id] / total) for client_id in client_ids}


def project_generated_strategy(
    generated_weights: Weights,
    diagnostics: Dict[str, Dict[str, float]],
    previous_weights: Optional[Weights] = None,
    min_client_weight: float = 0.05,
    max_client_weight: float = 0.80,
    l1_change_limit: float = 0.40,
    anchor_weights: Optional[Weights] = None,
    anchor_l1_limit: Optional[float] = None,
    decision_type: Optional[str] = None,
    snap_to_size_l1_threshold: Optional[float] = None,
) -> StrategyProjectionResult:
    client_ids = list(diagnostics.keys())
    if not client_ids:
        return StrategyProjectionResult(weights={}, constraint_status={})

    weights = normalize_weights(generated_weights, client_ids, min_client_weight, max_client_weight)
    status: Dict[str, Any] = {
        "normalized": True,
        "l1_projected": False,
        "negative_coherence_limited": [],
        "high_norm_limited": [],
        "fallback_used": False,
        "anchor_projected": False,
        "snapped_to_size_prior": False,
    }
    size_weights = _size_weights(diagnostics)

    if (
        (decision_type or "").lower() == "balanced"
        and snap_to_size_l1_threshold is not None
        and l1_weight_distance(weights, size_weights) <= float(snap_to_size_l1_threshold)
    ):
        weights = normalize_weights(size_weights, client_ids, min_client_weight, max_client_weight)
        status["snapped_to_size_prior"] = True
        status["snap_to_size_l1_threshold"] = float(snap_to_size_l1_threshold)

    if previous_weights:
        previous = normalize_weights(previous_weights, client_ids, min_client_weight, max_client_weight)
        distance = l1_weight_distance(weights, previous)
        if distance > float(l1_change_limit) > 0:
            ratio = float(l1_change_limit) / distance
            weights = {
                client_id: previous[client_id] + ratio * (weights[client_id] - previous[client_id])
                for client_id in client_ids
            }
            weights = normalize_weights(weights, client_ids, min_client_weight, max_client_weight)
            status["l1_projected"] = True
            status["l1_before_projection"] = distance
            status["l1_after_projection"] = l1_weight_distance(weights, previous)

    caps: Dict[str, float] = {}
    for client_id, row in diagnostics.items():
        if float(row.get("cosine_to_mean_update", 0.0)) < -0.05:
            caps[client_id] = min(caps.get(client_id, max_client_weight), size_weights[client_id])
            status["negative_coherence_limited"].append(client_id)

    norms = [float(row.get("update_norm", 0.0)) for row in diagnostics.values()]
    median_norm = median(norms) if norms else 0.0
    if median_norm > 0:
        for client_id, row in diagnostics.items():
            if float(row.get("update_norm", 0.0)) > median_norm * 2.5:
                caps[client_id] = min(caps.get(client_id, max_client_weight), size_weights[client_id])
                status["high_norm_limited"].append(client_id)

    if caps:
        weights = _normalize_with_caps(weights, client_ids, min_client_weight, max_client_weight, caps)

    if anchor_weights and anchor_l1_limit is not None and float(anchor_l1_limit) >= 0:
        anchor = normalize_weights(anchor_weights, client_ids, min_client_weight, max_client_weight)
        distance = l1_weight_distance(weights, anchor)
        if distance > float(anchor_l1_limit) + 1e-12:
            ratio = float(anchor_l1_limit) / distance if distance > 0 else 0.0
            weights = {
                client_id: anchor[client_id] + ratio * (weights[client_id] - anchor[client_id])
                for client_id in client_ids
            }
            weights = normalize_weights(weights, client_ids, min_client_weight, max_client_weight)
            status["anchor_projected"] = True
            status["anchor_l1_before_projection"] = distance
            status["anchor_l1_after_projection"] = l1_weight_distance(weights, anchor)

    status["l1_from_previous"] = l1_weight_distance(weights, previous_weights or weights)
    return StrategyProjectionResult(weights=weights, constraint_status=status)
