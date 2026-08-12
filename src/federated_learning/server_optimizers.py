"""Server-side optimizers for federated model aggregation."""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch


StateDict = Dict[str, torch.Tensor]


def _clone_state(state: StateDict) -> StateDict:
    return {key: value.detach().clone() for key, value in state.items()}


def _state_delta_norm(left: StateDict, right: StateDict) -> float:
    total = 0.0
    for key in left:
        if not torch.is_floating_point(left[key]):
            continue
        delta = left[key].detach().float() - right[key].detach().float()
        total += float(torch.sum(delta * delta).item())
    return total ** 0.5


def _clip_state_update(current_state: StateDict, proposed_state: StateDict, max_norm: float | None) -> tuple[StateDict, bool, float]:
    if max_norm is None or max_norm <= 0:
        return proposed_state, False, _state_delta_norm(proposed_state, current_state)

    update_norm = _state_delta_norm(proposed_state, current_state)
    if update_norm <= max_norm or update_norm == 0:
        return proposed_state, False, update_norm

    scale = max_norm / update_norm
    clipped = {}
    for key, current_value in current_state.items():
        proposed_value = proposed_state[key]
        if torch.is_floating_point(current_value):
            clipped[key] = current_value + (proposed_value - current_value) * scale
        else:
            clipped[key] = proposed_value.detach().clone()
    return clipped, True, max_norm


class FedAvgServerOptimizer:
    """Compatibility optimizer: use the weighted average as the next state."""

    name = "fedavg"

    def get_optimizer_state(self) -> dict:
        return {"name": self.name}

    def load_optimizer_state(self, state: dict) -> None:
        return None

    def preview_step(
        self,
        current_state: StateDict,
        weighted_average_state: StateDict,
        server_lr_scale: float = 1.0,
    ) -> Tuple[StateDict, dict]:
        return self.step(current_state, weighted_average_state, server_lr_scale=server_lr_scale)

    def step(
        self,
        current_state: StateDict,
        weighted_average_state: StateDict,
        server_lr_scale: float = 1.0,
    ) -> Tuple[StateDict, dict]:
        updated = _clone_state(weighted_average_state)
        update_norm = _state_delta_norm(updated, current_state)
        return updated, {
            "server_optimizer": self.name,
            "server_lr": 1.0,
            "server_lr_scale": float(server_lr_scale),
            "effective_server_lr": 1.0,
            "aggregation_delta_norm": update_norm,
            "update_norm": update_norm,
            "update_clipped": False,
        }


_UNSET = object()


class FedYogiServerOptimizer:
    """FedYogi server optimizer.

    Uses the same update shape as Flower FedYogi:
    m_t = beta1 * m_{t-1} + (1 - beta1) * delta_t
    v_t = v_{t-1} - (1 - beta2) * delta_t^2 * sign(v_{t-1} - delta_t^2)
    x_{t+1} = x_t + eta * m_t / (sqrt(v_t) + tau)
    """

    name = "fedyogi"

    def __init__(
        self,
        server_lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
        update_clip_norm: float | None = None,
        max_coordinate_step_ratio: float | None = None,
    ):
        self.server_lr = float(server_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)
        self.update_clip_norm = update_clip_norm
        self.max_coordinate_step_ratio = max_coordinate_step_ratio
        self.m: StateDict = {}
        self.v: StateDict = {}

    def get_optimizer_state(self) -> dict:
        return {
            "name": self.name,
            "server_lr": self.server_lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "tau": self.tau,
            "update_clip_norm": self.update_clip_norm,
            "max_coordinate_step_ratio": self.max_coordinate_step_ratio,
            "m": _clone_state(self.m),
            "v": _clone_state(self.v),
        }

    def load_optimizer_state(self, state: dict) -> None:
        self.server_lr = float(state.get("server_lr", self.server_lr))
        self.beta1 = float(state.get("beta1", self.beta1))
        self.beta2 = float(state.get("beta2", self.beta2))
        self.tau = float(state.get("tau", self.tau))
        self.update_clip_norm = state.get("update_clip_norm", self.update_clip_norm)
        self.max_coordinate_step_ratio = state.get(
            "max_coordinate_step_ratio",
            self.max_coordinate_step_ratio,
        )
        self.m = _clone_state(state.get("m", {}))
        self.v = _clone_state(state.get("v", {}))

    def preview_step(
        self,
        current_state: StateDict,
        weighted_average_state: StateDict,
        server_lr_scale: float = 1.0,
        update_clip_norm_override: float | None | object = _UNSET,
    ) -> Tuple[StateDict, dict]:
        saved = self.get_optimizer_state()
        try:
            return self.step(
                current_state=current_state,
                weighted_average_state=weighted_average_state,
                server_lr_scale=server_lr_scale,
                update_clip_norm_override=update_clip_norm_override,
            )
        finally:
            self.load_optimizer_state(saved)

    def step(
        self,
        current_state: StateDict,
        weighted_average_state: StateDict,
        server_lr_scale: float = 1.0,
        update_clip_norm_override: float | None | object = _UNSET,
    ) -> Tuple[StateDict, dict]:
        effective_lr = self.server_lr * float(server_lr_scale)
        effective_clip_norm = (
            self.update_clip_norm
            if update_clip_norm_override is _UNSET
            else update_clip_norm_override
        )
        updated: StateDict = {}
        coordinate_step_clipped = False
        coordinate_direction_rejected = False

        for key, current_value in current_state.items():
            target_value = weighted_average_state[key]
            if not torch.is_floating_point(current_value):
                updated[key] = copy.deepcopy(target_value)
                continue

            current_float = current_value.detach().float()
            target_float = target_value.detach().float()
            delta = target_float - current_float

            if key not in self.m:
                self.m[key] = torch.zeros_like(delta)
            if key not in self.v:
                self.v[key] = torch.zeros_like(delta)

            self.m[key] = self.beta1 * self.m[key].to(delta.device) + (1.0 - self.beta1) * delta
            delta_sq = delta * delta
            self.v[key] = (
                self.v[key].to(delta.device)
                - (1.0 - self.beta2) * delta_sq * torch.sign(self.v[key].to(delta.device) - delta_sq)
            )
            # Numerical guard only; the FedYogi update should keep v non-negative
            # after a zero initialization.
            v_safe = torch.clamp(self.v[key], min=0.0)
            raw_step = effective_lr * self.m[key] / (torch.sqrt(v_safe) + self.tau)
            if self.max_coordinate_step_ratio is not None and self.max_coordinate_step_ratio > 0:
                opposite_direction = (raw_step != 0) & (raw_step * delta <= 0)
                coordinate_direction_rejected = coordinate_direction_rejected or bool(
                    torch.any(opposite_direction).item()
                )
                raw_step = torch.where(opposite_direction, torch.zeros_like(raw_step), raw_step)

                max_step = torch.abs(delta) * float(self.max_coordinate_step_ratio)
                clipped_step = torch.sign(raw_step) * torch.minimum(torch.abs(raw_step), max_step)
                coordinate_step_clipped = coordinate_step_clipped or bool(
                    torch.any(torch.abs(clipped_step - raw_step) > 0).item()
                )
                raw_step = clipped_step
            next_value = current_float + raw_step
            updated[key] = next_value.to(dtype=current_value.dtype, device=current_value.device)

        updated, clipped, final_norm = _clip_state_update(
            current_state,
            updated,
            effective_clip_norm,
        )
        aggregation_delta_norm = _state_delta_norm(weighted_average_state, current_state)
        return updated, {
            "server_optimizer": self.name,
            "server_lr": self.server_lr,
            "server_lr_scale": float(server_lr_scale),
            "effective_server_lr": effective_lr,
            "server_beta1": self.beta1,
            "server_beta2": self.beta2,
            "server_tau": self.tau,
            "max_coordinate_step_ratio": self.max_coordinate_step_ratio,
            "aggregation_delta_norm": aggregation_delta_norm,
            "update_norm": final_norm,
            "update_clipped": clipped,
            "coordinate_step_clipped": coordinate_step_clipped,
            "coordinate_direction_rejected": coordinate_direction_rejected,
        }


def build_server_optimizer(
    name: str = "fedavg",
    server_lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.99,
    tau: float = 1e-3,
    update_clip_norm: float | None = None,
    max_coordinate_step_ratio: float | None = None,
):
    normalized = (name or "fedavg").lower()
    if normalized == "fedavg":
        return FedAvgServerOptimizer()
    if normalized == "fedyogi":
        return FedYogiServerOptimizer(
            server_lr=server_lr,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
            update_clip_norm=update_clip_norm,
            max_coordinate_step_ratio=max_coordinate_step_ratio,
        )
    raise ValueError(f"Unknown server optimizer: {name}")
