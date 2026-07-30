from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np

from .schemas import LocalCandidateVote


@dataclass(frozen=True, slots=True)
class MetricSums:
    """Additive regression sufficient statistics on the original target scale."""

    n: int
    ape_sum: float
    se_sum: float
    ae_sum: float
    y_sum: float
    y_sq_sum: float

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int):
            raise TypeError("n must be an integer")
        if self.n <= 0:
            raise ValueError("n must be positive")
        for name in ("ape_sum", "se_sum", "ae_sum", "y_sum", "y_sq_sum"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in ("ape_sum", "se_sum", "ae_sum", "y_sq_sum"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


def _as_finite_array(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if array.size == 0:
        raise ValueError("metric inputs must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_metric_sums(
    y_true: Any,
    y_pred: Any,
    *,
    inverse_transform: Callable[[np.ndarray], Any],
    epsilon: float = 1e-8,
) -> MetricSums:
    """Compute sufficient statistics after an explicit original-scale transform."""
    truth_input = _as_finite_array(y_true, name="y_true")
    prediction_input = _as_finite_array(y_pred, name="y_pred")
    if truth_input.shape != prediction_input.shape:
        raise ValueError("metric inputs must have equal shape")
    if not callable(inverse_transform):
        raise TypeError("inverse_transform must be callable")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, Real)
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise ValueError("epsilon must be a positive finite number")

    truth = _as_finite_array(
        inverse_transform(truth_input.copy()),
        name="inverse-transformed y_true",
    )
    prediction = _as_finite_array(
        inverse_transform(prediction_input.copy()),
        name="inverse-transformed y_pred",
    )
    if truth.shape != truth_input.shape or prediction.shape != prediction_input.shape:
        raise ValueError("inverse-transform outputs must preserve input shape")

    truth = truth.reshape(-1)
    prediction = prediction.reshape(-1)
    error = prediction - truth
    denominator = np.maximum(np.abs(truth), float(epsilon))
    return MetricSums(
        n=int(truth.size),
        ape_sum=float(np.sum(np.abs(error) / denominator)),
        se_sum=float(np.sum(error * error)),
        ae_sum=float(np.sum(np.abs(error))),
        y_sum=float(np.sum(truth)),
        y_sq_sum=float(np.sum(truth * truth)),
    )


def aggregate_metric_sums(items: Iterable[MetricSums]) -> dict[str, float | int]:
    values = list(items)
    if not values:
        raise ValueError("metric aggregation requires observations")
    if not all(isinstance(item, MetricSums) for item in values):
        raise TypeError("metric aggregation accepts only MetricSums")

    n = sum(item.n for item in values)
    ape_sum = math.fsum(float(item.ape_sum) for item in values)
    se_sum = math.fsum(float(item.se_sum) for item in values)
    ae_sum = math.fsum(float(item.ae_sum) for item in values)
    y_sum = math.fsum(float(item.y_sum) for item in values)
    y_sq_sum = math.fsum(float(item.y_sq_sum) for item in values)
    target_ss = y_sq_sum - y_sum * y_sum / n
    r2 = 1.0 - se_sum / target_ss if target_ss > 0.0 else 0.0
    metrics: dict[str, float | int] = {
        "sample_count": n,
        "mape": ape_sum / n,
        "rmse": math.sqrt(se_sum / n),
        "mae": ae_sum / n,
        "r2": r2,
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("aggregated metrics must be finite")
    return metrics


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _relative_change(candidate: float, anchor: float) -> float:
    denominator = max(abs(anchor), 1e-12)
    relative = (candidate - anchor) / denominator
    if math.isfinite(relative):
        return relative
    return math.copysign(float(np.finfo(float).max), relative)


def build_vote(
    *,
    client_id: str,
    candidate_id: str,
    sample_count: int,
    candidate_mape: float,
    candidate_rmse: float,
    anchor_mape: float,
    anchor_rmse: float,
    rank: int,
    confidence: float,
) -> LocalCandidateVote:
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError("client_id must be a non-empty string")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string")
    sample_count = _positive_int(sample_count, name="sample_count")
    rank = _positive_int(rank, name="rank")
    candidate_mape = _non_negative_finite(candidate_mape, name="candidate_mape")
    candidate_rmse = _non_negative_finite(candidate_rmse, name="candidate_rmse")
    anchor_mape = _non_negative_finite(anchor_mape, name="anchor_mape")
    anchor_rmse = _non_negative_finite(anchor_rmse, name="anchor_rmse")
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise TypeError("confidence must be a real number")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and between zero and one")

    return LocalCandidateVote(
        client_id=client_id,
        candidate_id=candidate_id,
        sample_count=sample_count,
        val_mape=candidate_mape,
        val_rmse=candidate_rmse,
        relative_mape=_relative_change(candidate_mape, anchor_mape),
        relative_rmse=_relative_change(candidate_rmse, anchor_rmse),
        rank=rank,
        confidence=confidence,
        catastrophic_degradation=candidate_mape > anchor_mape * 1.05,
    )
