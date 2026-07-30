"""Coherence diagnostics for client updates in federated learning."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


Diagnostics = Dict[str, Dict[str, float]]


def _flatten_update(
    global_state: Dict[str, torch.Tensor],
    client_state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    parts = []
    for key, global_value in global_state.items():
        client_value = client_state.get(key)
        if client_value is None:
            continue
        if not torch.is_floating_point(global_value) or not torch.is_floating_point(client_value):
            continue
        parts.append((client_value.detach().float().cpu() - global_value.detach().float().cpu()).reshape(-1))
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts)


def _flatten_delta(delta_state: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
    if not delta_state:
        return torch.zeros(0, dtype=torch.float32)
    parts = []
    for value in delta_state.values():
        if torch.is_floating_point(value):
            parts.append(value.detach().float().cpu().reshape(-1))
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0 or left.numel() != right.numel():
        return 0.0
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        return 0.0
    value = torch.dot(left, right) / (left_norm * right_norm)
    return float(torch.clamp(value, -1.0, 1.0).item())


def compute_coherence_diagnostics(
    global_state: Dict[str, torch.Tensor],
    client_states: Dict[str, Dict[str, torch.Tensor]],
    client_metrics: Dict[str, Dict[str, Any]],
    previous_global_delta: Optional[Dict[str, torch.Tensor]] = None,
) -> Diagnostics:
    """Convert client updates into aggregate-only coherence diagnostics.

    The function uses model parameter deltas and client-level aggregate metrics
    only. Non-floating tensors are ignored, and zero-vector cosines are reported
    as 0.0 to avoid NaN propagation.
    """
    client_ids = list(client_states.keys())
    updates = {
        client_id: _flatten_update(global_state, client_states[client_id])
        for client_id in client_ids
    }
    if updates:
        mean_update = torch.stack(list(updates.values())).mean(dim=0)
    else:
        mean_update = torch.zeros(0, dtype=torch.float32)
    previous_delta = _flatten_delta(previous_global_delta)

    sample_counts = {
        client_id: max(float(client_metrics.get(client_id, {}).get("n_samples", 0.0)), 0.0)
        for client_id in client_ids
    }
    total_samples = sum(sample_counts.values())
    if total_samples <= 0 and client_ids:
        sample_weights = {client_id: 1.0 / len(client_ids) for client_id in client_ids}
    else:
        sample_weights = {
            client_id: (sample_counts[client_id] / total_samples if total_samples > 0 else 0.0)
            for client_id in client_ids
        }

    diagnostics: Diagnostics = {}
    for client_id in client_ids:
        update = updates[client_id]
        pairwise_values = [
            _cosine(update, other_update)
            for other_id, other_update in updates.items()
            if other_id != client_id
        ]
        metrics = client_metrics.get(client_id, {})
        diagnostics[client_id] = {
            "update_norm": float(torch.linalg.vector_norm(update).item()) if update.numel() else 0.0,
            "cosine_to_mean_update": _cosine(update, mean_update),
            "pairwise_mean_cosine": float(sum(pairwise_values) / len(pairwise_values)) if pairwise_values else 0.0,
            "cosine_to_previous_global_update": _cosine(update, previous_delta),
            "drift_from_mean_update": float(torch.linalg.vector_norm(update - mean_update).item()) if update.numel() else 0.0,
            "sample_size_weight": float(sample_weights.get(client_id, 0.0)),
            "val_mape": float(metrics.get("val_mape", 0.0)),
            "val_mpe": float(metrics.get("val_mpe", 0.0)),
        }
    return diagnostics
