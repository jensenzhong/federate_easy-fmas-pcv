"""Aggregate-only client summaries for LLM-guided FL control."""

from __future__ import annotations

from statistics import median
from typing import Dict, Optional


SummaryMap = Dict[str, Dict[str, str]]


def _level_by_tertile(value: float, values: list[float], labels: tuple[str, str, str]) -> str:
    if len(values) <= 1:
        return labels[1]
    ordered = sorted(values)
    low_cut = ordered[max(0, int(len(ordered) / 3) - 1)]
    high_cut = ordered[min(len(ordered) - 1, int(2 * len(ordered) / 3))]
    if value <= low_cut:
        return labels[0]
    if value >= high_cut:
        return labels[2]
    return labels[1]


def _coherence_level(value: float) -> str:
    if value < 0:
        return "negative"
    if value < 0.25:
        return "weak"
    if value < 0.65:
        return "moderate"
    return "strong"


def _bias_direction(value: float) -> str:
    if value < -0.01:
        return "underestimation"
    if value > 0.01:
        return "overestimation"
    return "balanced"


def _weight_trend(client_id: str, diagnostics: Dict[str, Dict[str, float]], previous_weights: Optional[Dict[str, float]]) -> str:
    if not previous_weights or client_id not in previous_weights:
        return "stable"
    current = float(diagnostics[client_id].get("sample_size_weight", 0.0))
    previous = float(previous_weights.get(client_id, current))
    if current > previous + 0.03:
        return "increasing"
    if current < previous - 0.03:
        return "decreasing"
    return "stable"


def build_client_summaries(
    diagnostics: Dict[str, Dict[str, float]],
    previous_weights: Optional[Dict[str, float]] = None,
) -> SummaryMap:
    """Build short FedCLLM-style summaries from aggregate diagnostics."""
    client_ids = list(diagnostics.keys())
    shares = [float(diagnostics[cid].get("sample_size_weight", 0.0)) for cid in client_ids]
    errors = [float(diagnostics[cid].get("val_mape", 0.0)) for cid in client_ids]
    norms = [float(diagnostics[cid].get("update_norm", 0.0)) for cid in client_ids]
    norm_median = median(norms) if norms else 0.0

    summaries: SummaryMap = {}
    for client_id in client_ids:
        row = diagnostics[client_id]
        share_level = _level_by_tertile(float(row.get("sample_size_weight", 0.0)), shares, ("low", "medium", "high"))
        error_level = _level_by_tertile(float(row.get("val_mape", 0.0)), errors, ("low", "medium", "high"))
        norm = float(row.get("update_norm", 0.0))
        if norm_median <= 0:
            norm_level = "normal"
        elif norm < norm_median * 0.5:
            norm_level = "low"
        elif norm > norm_median * 2.0:
            norm_level = "high"
        else:
            norm_level = "normal"
        bias = _bias_direction(float(row.get("val_mpe", 0.0)))
        coherence = _coherence_level(float(row.get("cosine_to_mean_update", 0.0)))
        trend = _weight_trend(client_id, diagnostics, previous_weights)

        summaries[client_id] = {
            "sample_share_level": share_level,
            "validation_error_level": error_level,
            "bias_direction": bias,
            "coherence_level": coherence,
            "update_norm_level": norm_level,
            "recent_weight_trend": trend,
            "summary_text": (
                f"{client_id}: {share_level} client share, {error_level} validation MAPE, "
                f"{coherence} update coherence, {bias} bias, {norm_level} update norm, "
                f"{trend} recent weight trend."
            ),
        }
    return summaries
