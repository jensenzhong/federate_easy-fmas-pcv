"""Immutable value objects shared by the FMAS-PCV protocol."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math
from types import MappingProxyType
from typing import Any


ALLOWED_LR_SCALES = (0.50, 0.75, 1.00, 1.25)
ALLOWED_CLIP_NORMS = (None, 0.5, 1.0, 2.0)


def _deep_freeze(value, *, _active_ids: set[int] | None = None):
    if _active_ids is None:
        _active_ids = set()
    if isinstance(value, Mapping) or type(value) in (list, tuple):
        marker = id(value)
        if marker in _active_ids:
            raise TypeError("diagnostics must not contain cycles")
        _active_ids.add(marker)
        try:
            if isinstance(value, Mapping):
                return MappingProxyType(
                    {
                        _deep_freeze(key, _active_ids=_active_ids): _deep_freeze(
                            item,
                            _active_ids=_active_ids,
                        )
                        for key, item in value.items()
                    }
                )
            return tuple(
                _deep_freeze(item, _active_ids=_active_ids) for item in value
            )
        finally:
            _active_ids.remove(marker)
    if type(value) in (str, bytes, int, float, bool, type(None)):
        return value
    raise TypeError(
        f"unsupported diagnostics value type: {type(value).__name__}"
    )


@dataclass(frozen=True)
class ClientTelemetry:
    client_id: str
    sample_count: int
    train_loss: float
    val_mape: float
    val_rmse: float
    update_norm: float
    cosine_to_mean: float
    cosine_to_previous: float

    def to_prompt_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateAction:
    candidate_id: str
    weights: Mapping[str, float]
    server_optimizer: str
    server_lr_scale: float
    update_clip_norm: float | None
    source: str
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "weights",
            MappingProxyType(dict(self.weights)),
        )

    def validate(self, client_ids: tuple[str, ...]) -> None:
        if (
            not client_ids
            or any(
                not isinstance(client_id, str) or not client_id.strip()
                for client_id in client_ids
            )
            or len(set(client_ids)) != len(client_ids)
        ):
            raise ValueError("client ids must be non-empty and unique")
        if not isinstance(self.weights, Mapping):
            raise ValueError("candidate weights must be a mapping")
        if set(self.weights) != set(client_ids):
            raise ValueError("candidate weights must match client ids")

        weight_values = tuple(self.weights.values())
        if any(isinstance(value, bool) for value in weight_values):
            raise ValueError("candidate weights must be numeric")
        try:
            weights_are_finite = all(
                math.isfinite(value) for value in weight_values
            )
        except TypeError as exc:
            raise ValueError("candidate weights must be numeric") from exc
        if not weights_are_finite:
            raise ValueError("candidate weights must be finite")
        if abs(sum(weight_values) - 1.0) > 1e-6:
            raise ValueError("candidate weights must sum to one")
        if any(value < 0.05 or value > 0.80 for value in weight_values):
            raise ValueError("candidate weight outside [0.05, 0.80]")

        for field_name in ("candidate_id", "source", "rationale"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.server_optimizer not in {"fedavg", "fedyogi"}:
            raise ValueError("invalid server_optimizer")
        if (
            isinstance(self.server_lr_scale, bool)
            or not isinstance(self.server_lr_scale, (int, float))
            or not math.isfinite(self.server_lr_scale)
            or self.server_lr_scale not in ALLOWED_LR_SCALES
        ):
            raise ValueError("invalid server_lr_scale")
        if self.update_clip_norm is not None and (
            isinstance(self.update_clip_norm, bool)
            or not isinstance(self.update_clip_norm, (int, float))
            or not math.isfinite(self.update_clip_norm)
            or self.update_clip_norm not in ALLOWED_CLIP_NORMS
        ):
            raise ValueError("invalid update_clip_norm")


@dataclass(frozen=True)
class LocalCandidateVote:
    client_id: str
    candidate_id: str
    sample_count: int
    val_mape: float
    val_rmse: float
    relative_mape: float
    relative_rmse: float
    rank: int
    confidence: float
    catastrophic_degradation: bool


@dataclass(frozen=True)
class CandidateDecision:
    requested_candidate_id: str
    selected_candidate_id: str
    gate_status: str
    rationale: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(
            self,
            "diagnostics",
            _deep_freeze(self.diagnostics),
        )
