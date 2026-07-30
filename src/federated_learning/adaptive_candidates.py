"""Validation-guided adaptive aggregation candidates.

This module is intentionally independent from model training so candidate
generation and gate behavior can be tested without running federated rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


Weights = Dict[str, float]


@dataclass
class AdaptiveCandidate:
    candidate_id: str
    weights: Weights
    server_lr_scale: float = 1.0
    epoch_delta: int = 0
    source: str = "local_grid"
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    score: Optional[float] = None
    client_gap: float = 0.0
    update_norm: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row = {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "server_lr_scale": self.server_lr_scale,
            "epoch_delta": self.epoch_delta,
            "score": self.score,
            "client_gap": self.client_gap,
            "update_norm": self.update_norm,
        }
        for client_id, weight in self.weights.items():
            row[f"weight_{client_id}"] = weight
        for metric_name, value in self.validation_metrics.items():
            row[f"val_{metric_name}"] = value
        row.update(self.metadata)
        return row


def normalize_weights(
    weights: Weights,
    client_ids: Iterable[str],
    min_weight: float,
    max_weight: float,
) -> Weights:
    client_ids = list(client_ids)
    clipped = {
        client_id: max(float(min_weight), min(float(max_weight), float(weights.get(client_id, 0.0))))
        for client_id in client_ids
    }
    total = sum(clipped.values())
    if total <= 0:
        return {client_id: 1.0 / len(client_ids) for client_id in client_ids}
    normalized = {client_id: value / total for client_id, value in clipped.items()}
    return _project_bounds(normalized, client_ids, min_weight, max_weight)


def _project_bounds(weights: Weights, client_ids: List[str], min_weight: float, max_weight: float) -> Weights:
    projected = dict(weights)
    for _ in range(20):
        changed = False
        for client_id in client_ids:
            value = projected[client_id]
            bounded = max(float(min_weight), min(float(max_weight), value))
            if abs(bounded - value) > 1e-12:
                projected[client_id] = bounded
                changed = True
        total = sum(projected.values())
        free = [
            client_id for client_id in client_ids
            if min_weight + 1e-12 < projected[client_id] < max_weight - 1e-12
        ]
        residual = 1.0 - total
        if abs(residual) <= 1e-10:
            break
        if not free:
            scale = 1.0 / total if total else 1.0
            projected = {client_id: projected[client_id] * scale for client_id in client_ids}
            changed = True
        else:
            share = residual / len(free)
            for client_id in free:
                projected[client_id] += share
            changed = True
        if not changed:
            break
    total = sum(projected.values())
    return {client_id: projected[client_id] / total for client_id in client_ids}


def _weight_key(weights: Weights, client_ids: List[str]) -> Tuple[float, ...]:
    return tuple(round(float(weights[client_id]), 6) for client_id in client_ids)


def _make_candidate(
    candidate_id: str,
    weights: Weights,
    source: str,
    server_lr_scale: float = 1.0,
    epoch_delta: int = 0,
) -> AdaptiveCandidate:
    return AdaptiveCandidate(
        candidate_id=candidate_id,
        weights={client_id: float(weight) for client_id, weight in weights.items()},
        server_lr_scale=float(server_lr_scale),
        epoch_delta=int(epoch_delta),
        source=source,
    )


def generate_weight_candidates(
    client_ids: List[str],
    size_weights: Weights,
    previous_weights: Optional[Weights],
    client_metrics: Dict[str, Dict[str, Any]],
    budget: int = 30,
    step: float = 0.05,
    min_weight: float = 0.05,
    max_weight: float = 0.8,
    server_lr_scales: Optional[List[float]] = None,
    epoch_deltas: Optional[List[int]] = None,
    coherence_diagnostics: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[AdaptiveCandidate]:
    """Generate bounded continuous aggregation candidates.

    The first three anchors are stable and deterministic: sample-size weights,
    uniform weights, and previous accepted weights. Remaining candidates are
    validation-error compensation and local simplex grid candidates.
    """
    if not client_ids:
        raise ValueError("client_ids must not be empty")
    budget = max(1, int(budget))
    server_lr_scales = server_lr_scales or [1.0]
    epoch_deltas = epoch_deltas or [0]
    step = max(float(step), 1e-6)

    candidates: List[AdaptiveCandidate] = []
    seen: set[Tuple[float, ...]] = set()

    def add(weights: Weights, source: str):
        if len(candidates) >= budget:
            return
        normalized = normalize_weights(weights, client_ids, min_weight, max_weight)
        key = _weight_key(normalized, client_ids)
        if key in seen:
            return
        seen.add(key)
        candidate_id = source if source in {"size_anchor", "uniform_anchor", "previous_accepted"} else f"candidate_{len(candidates):03d}"
        candidates.append(_make_candidate(candidate_id, normalized, source))

    def add_special(
        weights: Weights,
        source: str,
        server_lr_scale: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if len(candidates) >= budget:
            return
        normalized = normalize_weights(weights, client_ids, min_weight, max_weight)
        key = (_weight_key(normalized, client_ids), source, round(float(server_lr_scale), 6))
        if key in seen:
            return
        seen.add(key)
        candidate = _make_candidate(
            candidate_id=source,
            weights=normalized,
            source=source,
            server_lr_scale=server_lr_scale,
        )
        candidate.metadata.update(metadata or {})
        candidates.append(candidate)

    add(size_weights, "size_anchor")
    add({client_id: 1.0 / len(client_ids) for client_id in client_ids}, "uniform_anchor")
    if previous_weights:
        add(previous_weights, "previous_accepted")
    else:
        add(size_weights, "previous_accepted")

    error_weights = _error_compensation_weights(client_ids, client_metrics)
    add(error_weights, "error_compensation")

    if coherence_diagnostics:
        coherence_weights = _coherence_prior_weights(
            client_ids,
            size_weights=size_weights,
            diagnostics=coherence_diagnostics,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        add_special(coherence_weights, "coherence_prior")
        add_special(
            size_weights,
            "fedlaw_shrinkage",
            server_lr_scale=0.75,
            metadata={"shrinkage_interpretation": "FedLAW-style smaller aggregate update"},
        )
        add_special(
            _drift_limited_weights(client_ids, size_weights, coherence_diagnostics),
            "drift_limited",
        )
        add_special(
            _bias_sensitive_weights(client_ids, size_weights, client_metrics),
            "bias_sensitive",
        )

    anchors = [candidate.weights for candidate in candidates[:]]
    offsets = [-step, 0.0, step]
    for anchor in anchors:
        if len(candidates) >= budget:
            break
        for deltas in product(offsets, repeat=len(client_ids)):
            if len(candidates) >= budget:
                break
            if abs(sum(deltas)) > 1e-9:
                continue
            weights = {
                client_id: anchor[client_id] + delta
                for client_id, delta in zip(client_ids, deltas)
            }
            add(weights, "local_grid")

    if len(candidates) < budget and len(client_ids) == 3:
        grid_values = [round(i * step, 10) for i in range(int(1.0 / step) + 1)]
        for w1 in grid_values:
            if len(candidates) >= budget:
                break
            for w2 in grid_values:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                weights = dict(zip(client_ids, [w1, w2, w3]))
                if all(min_weight - 1e-9 <= value <= max_weight + 1e-9 for value in weights.values()):
                    add(weights, "simplex_grid")

    # Expand only after required anchors are present, preserving candidate budget.
    if len(server_lr_scales) > 1 or len(epoch_deltas) > 1:
        base_candidates = list(candidates)
        candidates = []
        seen_full = set()
        for base in base_candidates:
            for server_lr_scale in server_lr_scales:
                for epoch_delta in epoch_deltas:
                    if len(candidates) >= budget:
                        break
                    key = (_weight_key(base.weights, client_ids), float(server_lr_scale), int(epoch_delta))
                    if key in seen_full:
                        continue
                    seen_full.add(key)
                    candidate_id = base.candidate_id
                    if server_lr_scale != 1.0 or epoch_delta != 0:
                        candidate_id = f"{base.candidate_id}_lr{str(server_lr_scale).replace('.', 'p')}_ep{epoch_delta:+d}"
                    candidates.append(_make_candidate(
                        candidate_id=candidate_id,
                        weights=base.weights,
                        source=base.source,
                        server_lr_scale=server_lr_scale,
                        epoch_delta=epoch_delta,
                    ))

    return candidates[:budget]


def _coherence_prior_weights(
    client_ids: List[str],
    size_weights: Weights,
    diagnostics: Dict[str, Dict[str, Any]],
    min_weight: float,
    max_weight: float,
) -> Weights:
    alignments = {
        client_id: max(float(diagnostics.get(client_id, {}).get("cosine_to_mean_update", 0.0)), 0.0)
        for client_id in client_ids
    }
    if all(value <= 0.0 for value in alignments.values()):
        return normalize_weights(size_weights, client_ids, min_weight, max_weight)
    raw = {
        client_id: float(size_weights[client_id]) * (0.5 + alignments[client_id])
        for client_id in client_ids
    }
    return normalize_weights(raw, client_ids, min_weight, max_weight)


def _drift_limited_weights(
    client_ids: List[str],
    size_weights: Weights,
    diagnostics: Dict[str, Dict[str, Any]],
) -> Weights:
    norms = [float(diagnostics.get(client_id, {}).get("update_norm", 0.0)) for client_id in client_ids]
    median_norm = median(norms) if norms else 0.0
    raw = dict(size_weights)
    for client_id in client_ids:
        row = diagnostics.get(client_id, {})
        cosine = float(row.get("cosine_to_mean_update", 0.0))
        norm = float(row.get("update_norm", 0.0))
        if cosine < -0.05:
            raw[client_id] = min(raw[client_id], float(size_weights[client_id]) * 0.75)
        if median_norm > 0 and norm > median_norm * 2.5:
            raw[client_id] = min(raw[client_id], float(size_weights[client_id]) * 0.85)
    return raw


def _bias_sensitive_weights(
    client_ids: List[str],
    size_weights: Weights,
    client_metrics: Dict[str, Dict[str, Any]],
) -> Weights:
    bias_magnitudes = {
        client_id: abs(float(client_metrics.get(client_id, {}).get("val_mpe", 0.0)))
        for client_id in client_ids
    }
    if not bias_magnitudes or max(bias_magnitudes.values()) <= 0:
        return dict(size_weights)
    inverse_bias = {
        client_id: 1.0 / (bias_magnitudes[client_id] + 1e-6)
        for client_id in client_ids
    }
    total_inverse = sum(inverse_bias.values())
    bias_prior = {
        client_id: inverse_bias[client_id] / total_inverse
        for client_id in client_ids
    }
    return {
        client_id: 0.75 * float(size_weights[client_id]) + 0.25 * bias_prior[client_id]
        for client_id in client_ids
    }


def _error_compensation_weights(client_ids: List[str], client_metrics: Dict[str, Dict[str, Any]]) -> Weights:
    values = {}
    for client_id in client_ids:
        metric = client_metrics.get(client_id, {})
        values[client_id] = max(float(metric.get("val_mape", 0.0)), 1e-6)
    total = sum(values.values())
    if total <= 0:
        return {client_id: 1.0 / len(client_ids) for client_id in client_ids}
    return {client_id: values[client_id] / total for client_id in client_ids}


def score_candidate_metrics(
    metrics: Dict[str, float],
    client_gap: float,
    update_norm: float,
    weights: Weights,
    previous_weights: Optional[Weights] = None,
    profile: str = "mape_primary",
) -> float:
    """Score lower-is-better validation metrics."""
    mape = float(metrics.get("mape", metrics.get("val_mape", 0.0)))
    mpe = abs(float(metrics.get("mpe", metrics.get("val_mpe", 0.0))))
    gap = float(client_gap or 0.0)
    norm = float(update_norm or 0.0)
    weight_shift = l1_weight_distance(weights, previous_weights or weights)

    if profile != "mape_primary":
        profile = "mape_primary"
    return mape + 0.05 * gap + 0.02 * mpe + 0.001 * norm + 0.01 * weight_shift


def l1_weight_distance(left: Weights, right: Weights) -> float:
    client_ids = set(left) | set(right)
    return sum(abs(float(left.get(client_id, 0.0)) - float(right.get(client_id, 0.0))) for client_id in client_ids)


def select_candidate_by_gate(
    candidates: List[AdaptiveCandidate],
    conservative_candidate_id: str,
    requested_candidate_id: Optional[str] = None,
    epsilon: float = 0.002,
    score_tolerance: float = 0.0,
    previous_weights: Optional[Weights] = None,
    weight_l1_limit: float = 0.4,
    large_improvement_threshold: float = 0.01,
) -> Tuple[AdaptiveCandidate, Dict[str, Any]]:
    """Select a candidate with deterministic validation safeguards."""
    if not candidates:
        raise ValueError("candidates must not be empty")

    scored = [candidate for candidate in candidates if candidate.score is not None]
    pool = scored if scored else candidates
    best = min(pool, key=lambda candidate: float("inf") if candidate.score is None else candidate.score)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    conservative = by_id.get(conservative_candidate_id)
    if conservative is None:
        conservative = next(
            (candidate for candidate in candidates if candidate.source == conservative_candidate_id),
            candidates[0],
        )
    requested = by_id.get(requested_candidate_id) if requested_candidate_id else best
    status = "accepted"

    if requested is None:
        requested = best
        status = "fallback_invalid_request"
    elif requested.score is None:
        requested = best
        status = "fallback_unscored_request"
    elif best.score is not None and requested.score > best.score + float(score_tolerance) + 1e-12:
        requested = best
        status = "fallback_best_score"
    elif requested_candidate_id and requested.candidate_id != best.candidate_id:
        status = "accepted_llm_near_best"

    if (
        requested.candidate_id != conservative.candidate_id
        and requested.score is not None
        and conservative.score is not None
        and conservative.score - requested.score < float(epsilon)
    ):
        requested = conservative
        status = "fallback_conservative_epsilon"

    previous_weights = previous_weights or conservative.weights
    weight_l1 = l1_weight_distance(requested.weights, previous_weights)
    conservative_score = conservative.score if conservative.score is not None else requested.score
    requested_score = requested.score if requested.score is not None else conservative_score
    improvement = (conservative_score - requested_score) if conservative_score is not None and requested_score is not None else 0.0
    if (
        requested.candidate_id != conservative.candidate_id
        and weight_l1 > float(weight_l1_limit)
        and improvement < float(large_improvement_threshold)
    ):
        requested = conservative
        status = "fallback_weight_shift"
        weight_l1 = l1_weight_distance(requested.weights, previous_weights)

    return requested, {
        "gate_status": status,
        "requested_candidate_id": requested_candidate_id,
        "selected_candidate_id": requested.candidate_id,
        "best_candidate_id": best.candidate_id,
        "conservative_candidate_id": conservative.candidate_id,
        "weight_l1_from_previous": weight_l1,
        "epsilon": float(epsilon),
        "score_tolerance": float(score_tolerance),
        "weight_l1_limit": float(weight_l1_limit),
        "large_improvement_threshold": float(large_improvement_threshold),
    }
