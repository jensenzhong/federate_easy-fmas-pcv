"""
MAS-FL agents for Scenario C (Multi-Agent Federated Learning).

This module provides:
- LocalAgent: client-side training logic
- CentralAgent: aggregation, evaluation, and optional LLM-guided strategy selection
"""

import copy
import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models import CostEstimationMLP
from src.federated_learning.adaptive_candidates import (
    AdaptiveCandidate,
    generate_weight_candidates,
    score_candidate_metrics,
    select_candidate_by_gate,
)
from src.federated_learning.client_summaries import build_client_summaries
from src.federated_learning.coherence_diagnostics import compute_coherence_diagnostics
from src.federated_learning.generated_strategy import compute_coherence_weights
from src.federated_learning.server_optimizers import build_server_optimizer
from src.utils import (
    compute_mape, compute_rmse, compute_mae,
    compute_mpe, compute_nrmse, compute_r2, get_device
)


@contextmanager
def _preserve_rng_state():
    """Prevent validation previews from consuming training RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _project_size_strata(y_true: np.ndarray, thresholds: dict) -> list[str]:
    small_max = thresholds.get("small_project_max", 1_000_000)
    medium_max = thresholds.get("medium_project_max", 5_000_000)
    strata = []
    for value in np.array(y_true).flatten():
        if value < small_max:
            strata.append(f"Small (<${small_max/1e6:.0f}M)")
        elif value < medium_max:
            strata.append(f"Medium (${small_max/1e6:.0f}M-${medium_max/1e6:.0f}M)")
        else:
            strata.append(f"Large (>=${medium_max/1e6:.0f}M)")
    return strata


def apply_mpe_bias_correction(
    val_true: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Apply the shared validation-MPE multiplicative bias correction."""
    mpe_ratio = compute_mpe(val_true, val_pred)
    corrected_pred = np.array(test_pred).flatten() / (1.0 + mpe_ratio)
    return corrected_pred, mpe_ratio


def build_prediction_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: Optional[pd.DataFrame],
    thresholds: dict,
    scenario: str,
) -> pd.DataFrame:
    """Build a stratified prediction table for downstream residual analysis."""
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    df = pd.DataFrame({
        "scenario": scenario,
        "True_Value": y_true_flat,
        "Predicted_Value": y_pred_flat,
    })

    if metadata is not None and "Client" in metadata.columns and len(metadata) == len(df):
        df["Client"] = metadata["Client"].values
    else:
        df["Client"] = "unknown"

    df["Project_Size_Stratum"] = _project_size_strata(y_true_flat, thresholds)
    return df[["scenario", "True_Value", "Predicted_Value", "Client", "Project_Size_Stratum"]]


class LocalAgent:
    """
    Client-side agent for local training and validation.
    """

    def __init__(
        self,
        client_id: str,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        config: dict,
        device: torch.device = None,
        input_dim: Optional[int] = None,
        fedprox_mu: Optional[float] = None,
        preprocessor: Optional[Any] = None
    ):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device if device else get_device("auto")
        self.preprocessor = preprocessor

        self.input_dim = input_dim
        if self.input_dim is None:
            scene_c_cfg = config.get('scene_c', {})
            data_cfg = scene_c_cfg.get('data', {})
            feature_columns = data_cfg.get('feature_columns') or config.get('data', {}).get('feature_columns', [])
            self.input_dim = len(feature_columns) if feature_columns else 10

        fl_config = config.get('federated_learning', {})
        client_config = fl_config.get('client', {})

        self.batch_size = client_config.get('batch_size', 32)
        self.default_lr = client_config.get('learning_rate', 0.0005)
        self.default_local_epochs = client_config.get('local_epochs', 20)
        self.weight_decay = 1e-4

        # 允许外部显式覆盖fedprox_mu（场景B传入0.0确保纯FedAvg）
        if fedprox_mu is not None:
            self.fedprox_mu = fedprox_mu
        else:
            self.fedprox_mu = fl_config.get('fedprox_mu', 0.0)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        self.n_train_samples = len(train_dataset)
        self.n_val_samples = len(val_dataset)

    def _create_local_model(self) -> CostEstimationMLP:
        model = CostEstimationMLP(
            input_dim=self.input_dim,
            hidden_dims=[128, 128, 64, 32],
            output_dim=1,
            activation='gelu',
            dropout=0.1
        )
        return model.to(self.device)

    def train_one_round(
        self,
        global_model_state: Dict[str, torch.Tensor],
        lr: Optional[float] = None,
        local_epochs: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        lr = lr if lr is not None else self.default_lr
        local_epochs = local_epochs if local_epochs is not None else self.default_local_epochs

        model = self._create_local_model()
        model.load_state_dict(global_model_state)

        global_model_ref = {}
        if self.fedprox_mu > 0:
            for name, param in model.named_parameters():
                global_model_ref[name] = param.detach().clone()

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        train_losses = []
        for _ in range(local_epochs):
            epoch_losses = []
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = model(batch_X)
                task_loss = criterion(outputs, batch_y)

                loss = task_loss
                if self.fedprox_mu > 0:
                    prox_loss = 0.0
                    for name, param in model.named_parameters():
                        ref_param = global_model_ref[name]
                        prox_loss += ((param - ref_param) ** 2).sum()
                    loss = loss + (self.fedprox_mu / 2) * prox_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
            train_losses.append(np.mean(epoch_losses))

        val_metrics = self._evaluate_on_val(model)

        metrics = {
            "n_samples": self.n_train_samples,
            "train_loss": train_losses[-1] if train_losses else 0.0,
            "val_mape": val_metrics["mape"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_mpe": val_metrics["mpe"],
        }

        updated_state_dict = copy.deepcopy(model.state_dict())
        return updated_state_dict, metrics

    def _evaluate_on_val(self, model: nn.Module) -> Dict[str, float]:
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = model(batch_X)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        y_pred = np.concatenate(all_preds, axis=0).reshape(-1, 1)
        y_true = np.concatenate(all_targets, axis=0).reshape(-1, 1)

        # Keep client-side validation metrics on the same original target scale
        # used by server-side global evaluation.
        if self.preprocessor is not None:
            y_pred = self.preprocessor.inverse_transform_target(y_pred)
            y_true = self.preprocessor.inverse_transform_target(y_true)

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        return {
            "mape": compute_mape(y_true, y_pred),
            "rmse": compute_rmse(y_true, y_pred),
            "mae": compute_mae(y_true, y_pred),
            "mpe": compute_mpe(y_true, y_pred),
        }


class CentralAgent:
    """
    Server-side agent for aggregation, evaluation, and strategy selection.
    """

    def __init__(
        self,
        global_model: CostEstimationMLP,
        client_agents: Dict[str, LocalAgent],
        global_val_loader: DataLoader,
        global_test_loader: DataLoader,
        preprocessor,
        config: dict,
        device: torch.device = None,
        llm_planner=None,
        server_optimizer: str = "fedavg",
        server_lr: float = 0.01,
        server_beta1: float = 0.9,
        server_beta2: float = 0.99,
        server_tau: float = 1e-3,
        update_clip_norm: Optional[float] = None,
        max_coordinate_step_ratio: Optional[float] = 1.0,
    ):
        self.global_model = global_model
        self.client_agents = client_agents
        self.global_val_loader = global_val_loader
        self.global_test_loader = global_test_loader
        self.preprocessor = preprocessor
        self.config = config
        self.device = device if device else get_device("auto")
        self.llm_planner = llm_planner
        self.server_optimizer = build_server_optimizer(
            name=server_optimizer,
            server_lr=server_lr,
            beta1=server_beta1,
            beta2=server_beta2,
            tau=server_tau,
            update_clip_norm=update_clip_norm,
            max_coordinate_step_ratio=max_coordinate_step_ratio,
        )

        self.history_round_metrics: List[Dict] = []
        self.best_val_mape = float('inf')
        self.best_model_state = None
        self.best_round = 0

        # 偏差校正参数
        self.bias_correction_value = 0.0
        self.bias_correction_enabled = False

        scene_c_config = config.get('scene_c', {})
        fl_config = config.get('federated_learning', {})
        llm_config = scene_c_config.get('llm', {})

        self.default_lr = scene_c_config.get('learning_rate',
                          fl_config.get('client', {}).get('learning_rate', 0.0005))
        self.default_local_epochs = scene_c_config.get('local_epochs',
                                    fl_config.get('client', {}).get('local_epochs', 20))

        self.default_strategy = scene_c_config.get('default_strategy', 'size_only')
        self.strategies_config = self._parse_strategies_config(scene_c_config.get('strategies', []))
        self.llm_call_every_n_rounds = max(1, int(llm_config.get('call_every_n_rounds', 1)))
        input_enhancement = llm_config.get('input_enhancement', {})
        self.enable_candidate_weight_preview = input_enhancement.get('candidate_weight_preview', True)
        self.enable_candidate_validation_preview = input_enhancement.get('candidate_validation_preview', True)
        self.previous_accepted_weights: Optional[Dict[str, float]] = None

    @staticmethod
    def _state_l2_norm(state: Dict[str, torch.Tensor]) -> float:
        total = 0.0
        for value in state.values():
            if torch.is_floating_point(value):
                tensor = value.detach().float()
                total += float(torch.sum(tensor * tensor).item())
        return total ** 0.5

    @staticmethod
    def _state_delta_l2_norm(
        left: Dict[str, torch.Tensor],
        right: Dict[str, torch.Tensor],
    ) -> float:
        total = 0.0
        for key, value in left.items():
            if torch.is_floating_point(value):
                delta = value.detach().float() - right[key].detach().float()
                total += float(torch.sum(delta * delta).item())
        return total ** 0.5

    def _parse_strategies_config(self, strategies_list: List[Dict]) -> Dict[str, Dict]:
        strategies = {}
        for s in strategies_list:
            name = s.get('name', '')
            if name:
                strategies[name] = {
                    'description': s.get('description', ''),
                    'lambda_hybrid': s.get('lambda_hybrid', 0.5),
                    'alpha_min': s.get('alpha_min', 0.0),
                    'alpha_max': s.get('alpha_max', 1.0),
                }
        if 'size_only' not in strategies:
            strategies['size_only'] = {'description': 'Weighted by sample size'}
        return strategies

    def compute_client_weights(
        self,
        strategy_name: str,
        client_metrics: Dict[str, Dict[str, Any]],
        strategy_params_override: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        strategy_params = copy.deepcopy(self.strategies_config.get(strategy_name, {}))
        if strategy_params_override:
            strategy_params.update({
                key: value for key, value in strategy_params_override.items()
                if value is not None
            })
        lambda_hybrid = strategy_params.get('lambda_hybrid', 0.5)
        alpha_min = strategy_params.get('alpha_min', 0.0)
        alpha_max = strategy_params.get('alpha_max', 1.0)

        client_ids = list(client_metrics.keys())

        total_samples = sum(m["n_samples"] for m in client_metrics.values())
        size_weights = {cid: client_metrics[cid]["n_samples"] / total_samples for cid in client_ids}

        epsilon = 1e-6
        inv_mapes = {cid: 1.0 / (client_metrics[cid]["val_mape"] + epsilon) for cid in client_ids}
        total_inv_mape = sum(inv_mapes.values())
        perf_weights = {cid: inv_mapes[cid] / total_inv_mape for cid in client_ids}

        if strategy_name == "size_only":
            weights = size_weights
        elif strategy_name == "perf_only":
            weights = perf_weights
        elif strategy_name == "hybrid":
            weights = {
                cid: (1 - lambda_hybrid) * size_weights[cid] + lambda_hybrid * perf_weights[cid]
                for cid in client_ids
            }
        elif strategy_name == "fairness_clip":
            raw_weights = {
                cid: (1 - lambda_hybrid) * size_weights[cid] + lambda_hybrid * perf_weights[cid]
                for cid in client_ids
            }
            clipped_weights = {
                cid: max(alpha_min, min(alpha_max, w))
                for cid, w in raw_weights.items()
            }
            total_clipped = sum(clipped_weights.values())
            weights = {cid: w / total_clipped for cid, w in clipped_weights.items()}
        else:
            print(f"  [Warning] Unknown strategy '{strategy_name}', falling back to 'size_only'")
            weights = size_weights

        return weights

    def build_candidate_weight_preview(
        self,
        client_metrics: Dict[str, Dict[str, Any]],
        strategy_params_override: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        preview = {}
        for strategy_name in ["size_only", "perf_only", "hybrid", "fairness_clip"]:
            weights = self.compute_client_weights(
                strategy_name,
                client_metrics,
                strategy_params_override=strategy_params_override
            )
            preview[strategy_name] = {
                cid: float(weight)
                for cid, weight in weights.items()
            }
        return preview

    def build_candidate_validation_preview(
        self,
        client_states: Dict[str, Dict[str, torch.Tensor]],
        client_metrics: Dict[str, Dict[str, Any]],
        strategy_params_override: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, Any]]:
        preview = {}
        for strategy_name in ["size_only", "perf_only", "hybrid", "fairness_clip"]:
            weights = self.compute_client_weights(
                strategy_name,
                client_metrics,
                strategy_params_override=strategy_params_override
            )
            candidate_state = self.aggregate_with_weights(client_states, weights)

            per_client = {}
            total_val_samples = 0
            weighted_sums = {"mape": 0.0, "rmse": 0.0, "mae": 0.0, "mpe": 0.0}
            for client_id, agent in self.client_agents.items():
                candidate_model = agent._create_local_model()
                candidate_model.load_state_dict(candidate_state)
                metrics = agent._evaluate_on_val(candidate_model)
                n_val_samples = int(getattr(agent, "n_val_samples", 0))
                total_val_samples += n_val_samples
                for metric_name in weighted_sums:
                    weighted_sums[metric_name] += float(metrics[metric_name]) * n_val_samples
                per_client[client_id] = {
                    "n_val_samples": n_val_samples,
                    "val_mape": float(metrics["mape"]),
                    "val_rmse": float(metrics["rmse"]),
                    "val_mae": float(metrics["mae"]),
                    "val_mpe": float(metrics["mpe"]),
                }

            if total_val_samples > 0:
                aggregate_metrics = {
                    metric_name: value / total_val_samples
                    for metric_name, value in weighted_sums.items()
                }
            else:
                aggregate_metrics = {metric_name: 0.0 for metric_name in weighted_sums}

            client_mapes = [m["val_mape"] for m in per_client.values()]
            preview[strategy_name] = {
                "weights": {cid: float(weight) for cid, weight in weights.items()},
                "aggregate_val_mape": aggregate_metrics["mape"],
                "aggregate_val_rmse": aggregate_metrics["rmse"],
                "aggregate_val_mae": aggregate_metrics["mae"],
                "aggregate_val_mpe": aggregate_metrics["mpe"],
                "client_mape_gap": (max(client_mapes) - min(client_mapes)) if client_mapes else 0.0,
                "client_metrics": per_client,
            }

        return preview

    def _evaluate_state_on_global_val(self, state: Dict[str, torch.Tensor]) -> Dict[str, float]:
        original_state = copy.deepcopy(self.global_model.state_dict())
        was_training = self.global_model.training
        try:
            with _preserve_rng_state():
                self.global_model.load_state_dict(state)
                return self.evaluate_global(data_loader=self.global_val_loader)
        finally:
            self.global_model.load_state_dict(original_state)
            self.global_model.train(was_training)

    def _evaluate_state_on_client_vals(self, state: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        with _preserve_rng_state():
            per_client = {}
            for client_id, agent in self.client_agents.items():
                model = agent._create_local_model()
                model.load_state_dict(state)
                metrics = agent._evaluate_on_val(model)
                per_client[client_id] = {
                    "val_mape": float(metrics["mape"]),
                    "val_rmse": float(metrics["rmse"]),
                    "val_mae": float(metrics["mae"]),
                    "val_mpe": float(metrics["mpe"]),
                    "n_val_samples": int(getattr(agent, "n_val_samples", 0)),
                }
            return per_client

    def build_continuous_candidate_preview(
        self,
        client_states: Dict[str, Dict[str, torch.Tensor]],
        client_metrics: Dict[str, Dict[str, Any]],
        current_state: Dict[str, torch.Tensor],
        candidate_budget: int = 30,
        weight_grid_step: float = 0.05,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        server_lr_scales: Optional[List[float]] = None,
        epoch_deltas: Optional[List[int]] = None,
        score_profile: str = "mape_primary",
        coherence_diagnostics: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[AdaptiveCandidate]:
        size_weights = self.compute_client_weights("size_only", client_metrics)
        candidates = generate_weight_candidates(
            client_ids=list(client_metrics.keys()),
            size_weights=size_weights,
            previous_weights=self.previous_accepted_weights,
            client_metrics=client_metrics,
            budget=candidate_budget,
            step=weight_grid_step,
            min_weight=min_client_weight,
            max_weight=max_client_weight,
            server_lr_scales=server_lr_scales or [1.0],
            epoch_deltas=epoch_deltas or [0],
            coherence_diagnostics=coherence_diagnostics,
        )

        for candidate in candidates:
            aggregated_state = self.aggregate_with_weights(client_states, candidate.weights)
            next_state, server_update = self.server_optimizer.preview_step(
                current_state=current_state,
                weighted_average_state=aggregated_state,
                server_lr_scale=candidate.server_lr_scale,
            )
            global_metrics = self._evaluate_state_on_global_val(next_state)
            per_client = self._evaluate_state_on_client_vals(next_state)
            client_mapes = [metrics["val_mape"] for metrics in per_client.values()]
            client_gap = (max(client_mapes) - min(client_mapes)) if client_mapes else 0.0
            candidate.validation_metrics = {
                "mape": float(global_metrics["mape"]),
                "rmse": float(global_metrics["rmse"]),
                "mae": float(global_metrics["mae"]),
                "mpe": float(global_metrics["mpe"]),
                "nrmse": float(global_metrics["nrmse"]),
                "r2": float(global_metrics["r2"]),
            }
            candidate.client_gap = float(client_gap)
            candidate.update_norm = float(server_update.get("update_norm", 0.0))
            candidate.score = score_candidate_metrics(
                metrics=candidate.validation_metrics,
                client_gap=candidate.client_gap,
                update_norm=candidate.update_norm,
                weights=candidate.weights,
                previous_weights=self.previous_accepted_weights or size_weights,
                profile=score_profile,
            )
            candidate.metadata = {
                "aggregation_delta_norm": float(server_update.get("aggregation_delta_norm", 0.0)),
                "coordinate_step_clipped": bool(server_update.get("coordinate_step_clipped", False)),
                "coordinate_direction_rejected": bool(server_update.get("coordinate_direction_rejected", False)),
                "client_validation_metrics": per_client,
            }
        return candidates

    @staticmethod
    def _candidate_preview_for_history(candidates: List[AdaptiveCandidate]) -> Dict[str, Any]:
        preview = {}
        for candidate in candidates:
            preview[candidate.candidate_id] = {
                "source": candidate.source,
                "weights": candidate.weights,
                "server_lr_scale": candidate.server_lr_scale,
                "epoch_delta": candidate.epoch_delta,
                "score": candidate.score,
                "client_gap": candidate.client_gap,
                "update_norm": candidate.update_norm,
                "validation_metrics": candidate.validation_metrics,
                "metadata": {
                    key: value for key, value in candidate.metadata.items()
                    if key != "client_validation_metrics"
                },
            }
        return preview

    def aggregate_with_weights(
        self,
        client_states: Dict[str, Dict[str, torch.Tensor]],
        weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        aggregated_state = {}
        first_client_id = list(client_states.keys())[0]
        param_keys = client_states[first_client_id].keys()

        for key in param_keys:
            first_value = client_states[first_client_id][key]
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = copy.deepcopy(first_value)
                continue

            aggregated_param = None
            for cid, state in client_states.items():
                weighted_param = weights[cid] * state[key].float()
                aggregated_param = weighted_param if aggregated_param is None else aggregated_param + weighted_param
            aggregated_state[key] = aggregated_param.to(dtype=first_value.dtype, device=first_value.device)

        return aggregated_state

    def aggregate_fedavg(
        self,
        client_states: Dict[str, Dict[str, torch.Tensor]],
        client_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        weights = self.compute_client_weights("size_only", client_metrics)
        return self.aggregate_with_weights(client_states, weights)

    def evaluate_global(self, data_loader: DataLoader = None, return_predictions: bool = False,
                        apply_bias_correction: bool = False) -> Dict[str, float]:
        loader = data_loader if data_loader is not None else self.global_val_loader

        self.global_model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.global_model(batch_X)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        y_pred = np.concatenate(all_preds, axis=0).reshape(-1, 1)
        y_true = np.concatenate(all_targets, axis=0).reshape(-1, 1)

        # Inverse transform to original scale
        if self.preprocessor is not None:
            y_pred = self.preprocessor.inverse_transform_target(y_pred)
            y_true = self.preprocessor.inverse_transform_target(y_true)

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        # 应用偏差校正（百分比方式）
        if apply_bias_correction and self.bias_correction_enabled:
            y_pred = y_pred / (1.0 + self.bias_correction_value)

        metrics = {
            "mape": compute_mape(y_true, y_pred),
            "rmse": compute_rmse(y_true, y_pred),
            "mae": compute_mae(y_true, y_pred),
            "mpe": compute_mpe(y_true, y_pred),
            "nrmse": compute_nrmse(y_true, y_pred),
            "r2": compute_r2(y_true, y_pred),
        }

        if return_predictions:
            metrics["predictions"] = y_pred
            metrics["targets"] = y_true

        return metrics

    def save_predictions(
        self,
        data_loader: DataLoader,
        output_path: Path,
        scenario: str,
        apply_bias_correction: bool = False,
    ) -> pd.DataFrame:
        metrics = self.evaluate_global(
            data_loader=data_loader,
            return_predictions=True,
            apply_bias_correction=apply_bias_correction,
        )
        metadata = getattr(getattr(data_loader, "dataset", None), "prediction_metadata", None)
        predictions_df = build_prediction_frame(
            y_true=metrics["targets"],
            y_pred=metrics["predictions"],
            metadata=metadata,
            thresholds=self.config.get("thresholds", {}),
            scenario=scenario,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        print(f"  Predictions saved: {output_path}")
        return predictions_df

    def _train_clients_for_round(
        self,
        global_state: Dict[str, torch.Tensor],
        lr: float,
        local_epochs: int,
        verbose: bool = True
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, Any]]]:
        client_states = {}
        client_metrics = {}
        for client_id, agent in self.client_agents.items():
            updated_state, metrics = agent.train_one_round(
                global_model_state=global_state,
                lr=lr,
                local_epochs=local_epochs
            )
            update_norm = self._state_delta_l2_norm(updated_state, global_state)
            metrics["update_norm"] = update_norm
            metrics["update_delta_norm"] = update_norm
            metrics["validation_gap"] = float(metrics["val_mape"] - metrics["val_mpe"])
            client_states[client_id] = updated_state
            client_metrics[client_id] = metrics

            if verbose:
                print(
                    f"    {client_id}: {metrics['n_samples']} samples, "
                    f"train_loss={metrics['train_loss']:.6f}, "
                    f"val_mape={metrics['val_mape']*100:.2f}%"
                )

        return client_states, client_metrics

    @staticmethod
    def _safe_client_metric_summary(client_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        return {
            cid: {
                "n_samples": int(metrics["n_samples"]),
                "train_loss": float(metrics["train_loss"]),
                "val_mape": float(metrics["val_mape"]),
                "val_rmse": float(metrics["val_rmse"]),
                "val_mae": float(metrics["val_mae"]),
                "val_mpe": float(metrics["val_mpe"]),
                "update_norm": float(metrics.get("update_norm", 0.0)),
                "update_delta_norm": float(metrics.get("update_delta_norm", 0.0)),
                "validation_gap": float(metrics.get("validation_gap", 0.0)),
            }
            for cid, metrics in client_metrics.items()
        }

    @staticmethod
    def _strategy_params_from_decision(decision: Optional[Dict[str, Any]]) -> Dict[str, float]:
        if not decision:
            return {}
        return {
            "lambda_hybrid": decision.get("lambda_hybrid"),
            "alpha_min": decision.get("alpha_min"),
            "alpha_max": decision.get("alpha_max"),
        }

    @staticmethod
    def _server_lr_scale_from_decision(decision: Optional[Dict[str, Any]]) -> float:
        if not decision:
            return 1.0
        return float(decision.get("server_lr_scale", decision.get("lr_scale", 1.0)))

    @staticmethod
    def _state_delta(
        next_state: Dict[str, torch.Tensor],
        previous_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        delta = {}
        for key, value in next_state.items():
            if key in previous_state and torch.is_floating_point(value):
                delta[key] = value.detach().float().cpu() - previous_state[key].detach().float().cpu()
        return delta

    def _coherence_artifacts_for_round(
        self,
        current_state: Dict[str, torch.Tensor],
        client_states: Dict[str, Dict[str, torch.Tensor]],
        client_metrics: Dict[str, Dict[str, Any]],
        previous_global_delta: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]:
        diagnostics = compute_coherence_diagnostics(
            global_state=current_state,
            client_states=client_states,
            client_metrics=client_metrics,
            previous_global_delta=previous_global_delta,
        )
        summaries = build_client_summaries(
            diagnostics=diagnostics,
            previous_weights=self.previous_accepted_weights,
        )
        return diagnostics, summaries

    @staticmethod
    def _aggregate_client_validation_metrics(
        per_client: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        total = sum(
            int(metrics.get("n_val_samples", metrics.get("n_samples", 0)))
            for metrics in per_client.values()
        )
        metric_keys = ["val_mape", "val_rmse", "val_mae", "val_mpe"]
        if total <= 0:
            return {key.replace("val_", ""): 0.0 for key in metric_keys} | {"nrmse": 0.0, "r2": 0.0}

        aggregated = {}
        for key in metric_keys:
            value = sum(
                float(metrics.get(key, 0.0))
                * int(metrics.get("n_val_samples", metrics.get("n_samples", 0)))
                for metrics in per_client.values()
            ) / total
            aggregated[key.replace("val_", "")] = float(value)
        aggregated["nrmse"] = 0.0
        aggregated["r2"] = 0.0
        return aggregated

    def _apply_aggregated_round(
        self,
        round_idx: int,
        strategy_name: str,
        lr: float,
        local_epochs: int,
        client_states: Dict[str, Dict[str, torch.Tensor]],
        client_metrics: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
        strategy_params_override: Optional[Dict[str, float]] = None,
        llm_decision: Optional[Dict[str, Any]] = None,
        candidate_weight_preview: Optional[Dict[str, Any]] = None,
        candidate_validation_preview: Optional[Dict[str, Any]] = None,
        server_lr_scale: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        if verbose:
            weights_str = ", ".join([f"{cid}={w:.3f}" for cid, w in weights.items()])
            print(f"    Weights: {weights_str}")

        current_state = copy.deepcopy(self.global_model.state_dict())
        aggregated_state = self.aggregate_with_weights(client_states, weights)
        next_state, server_update_info = self.server_optimizer.step(
            current_state=current_state,
            weighted_average_state=aggregated_state,
            server_lr_scale=server_lr_scale,
        )
        self.global_model.load_state_dict(next_state)

        global_metrics = self.evaluate_global()

        if verbose:
            print(
                f"    Global Val: MAPE={global_metrics['mape']*100:.2f}%, "
                f"RMSE=${global_metrics['rmse']:,.2f}"
            )

        round_record = {
            "round": round_idx,
            "strategy_name": strategy_name,
            "lr": lr,
            "local_epochs": local_epochs,
            "aggregation_weights": weights,
            "strategy_params_override": strategy_params_override or {},
            "server_optimizer": server_update_info.get("server_optimizer"),
            "server_update": server_update_info,
            "llm_decision": llm_decision,
            "candidate_weight_preview": candidate_weight_preview or {},
            "candidate_validation_preview": candidate_validation_preview or {},
            "client_metrics": self._safe_client_metric_summary(client_metrics),
            "global_val": {
                "mape": global_metrics["mape"],
                "mae": global_metrics["mae"],
                "rmse": global_metrics["rmse"],
                "mpe": global_metrics["mpe"],
                "nrmse": global_metrics["nrmse"],
                "r2": global_metrics["r2"],
            }
        }
        self.history_round_metrics.append(round_record)

        if global_metrics["mape"] < self.best_val_mape:
            self.best_val_mape = global_metrics["mape"]
            self.best_model_state = copy.deepcopy(self.global_model.state_dict())
            self.best_round = round_idx
            if verbose:
                print(f"    >>> New best model! Val MAPE={global_metrics['mape']*100:.2f}%")

        return round_record

    def _apply_aggregated_round_client_reported(
        self,
        round_idx: int,
        strategy_name: str,
        lr: float,
        local_epochs: int,
        client_states: Dict[str, Dict[str, torch.Tensor]],
        client_metrics: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
        strategy_params_override: Optional[Dict[str, float]] = None,
        llm_decision: Optional[Dict[str, Any]] = None,
        server_lr_scale: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Apply a round without server-side validation data.

        Strict modes use the already reported client local-validation summaries
        for logging only and keep the fixed final-round checkpoint protocol.
        They do not preview or evaluate the aggregated global model during
        training.
        """
        if verbose:
            weights_str = ", ".join([f"{cid}={w:.3f}" for cid, w in weights.items()])
            print(f"    Weights: {weights_str}")

        current_state = copy.deepcopy(self.global_model.state_dict())
        aggregated_state = self.aggregate_with_weights(client_states, weights)
        next_state, server_update_info = self.server_optimizer.step(
            current_state=current_state,
            weighted_average_state=aggregated_state,
            server_lr_scale=server_lr_scale,
        )
        self.global_model.load_state_dict(next_state)

        client_reported_metrics = self._aggregate_client_validation_metrics(client_metrics)

        if verbose:
            print(
                f"    Client-reported Local Val: MAPE={client_reported_metrics['mape']*100:.2f}%, "
                f"RMSE=${client_reported_metrics['rmse']:,.2f}"
            )

        round_record = {
            "round": round_idx,
            "strategy_name": strategy_name,
            "lr": lr,
            "local_epochs": local_epochs,
            "aggregation_weights": weights,
            "strategy_params_override": strategy_params_override or {},
            "server_optimizer": server_update_info.get("server_optimizer"),
            "server_update": server_update_info,
            "llm_decision": llm_decision,
            "validation_source": "client_reported",
            "strict_checkpoint_policy": "final_round",
            "client_metrics": self._safe_client_metric_summary(client_metrics),
            "global_val": {
                "mape": client_reported_metrics["mape"],
                "mae": client_reported_metrics["mae"],
                "rmse": client_reported_metrics["rmse"],
                "mpe": client_reported_metrics["mpe"],
                "nrmse": client_reported_metrics["nrmse"],
                "r2": client_reported_metrics["r2"],
            }
        }
        self.history_round_metrics.append(round_record)

        self.best_val_mape = client_reported_metrics["mape"]
        self.best_model_state = copy.deepcopy(self.global_model.state_dict())
        self.best_round = round_idx
        if verbose:
            print(
                f"    >>> Strict checkpoint set to current fixed round; "
                f"client-reported MAPE={client_reported_metrics['mape']*100:.2f}%"
            )

        return round_record

    def calibrate_bias(self):
        """
        在验证集上计算并保存偏差校正值。
        使用百分比校正（基于MPE），而非绝对值校正。
        这样对不同规模的项目更公平。

        校正方式: y_pred_corrected = y_pred / (1 + mpe_ratio)
        """
        val_metrics = self.evaluate_global(
            data_loader=self.global_val_loader, return_predictions=True
        )
        y_pred = val_metrics["predictions"]
        y_true = val_metrics["targets"]

        _, mpe_ratio = apply_mpe_bias_correction(
            val_true=y_true,
            val_pred=y_pred,
            test_pred=y_pred,
        )
        self.bias_correction_value = mpe_ratio
        self.bias_correction_enabled = True

        print(f"  [Bias Correction] Validation MPE = {mpe_ratio * 100:.2f}%")
        print(f"  [Bias Correction] Correction factor: 1 / (1 + {mpe_ratio:.4f}) = {1/(1+mpe_ratio):.4f}")

        # 验证校正后的效果
        corrected_pred = y_pred / (1.0 + mpe_ratio)
        corrected_mpe = compute_mpe(y_true, corrected_pred)
        corrected_mape = compute_mape(y_true, corrected_pred)
        print(f"  [Bias Correction] Validation MPE after correction: {corrected_mpe * 100:.2f}%")
        print(f"  [Bias Correction] Validation MAPE after correction: {corrected_mape * 100:.2f}%")

        return mpe_ratio

    def run_round(
        self,
        round_idx: int,
        strategy_name: str,
        lr: Optional[float] = None,
        local_epochs: Optional[int] = None,
        server_lr_scale: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        lr = lr if lr is not None else self.default_lr
        local_epochs = local_epochs if local_epochs is not None else self.default_local_epochs

        if verbose:
            print(f"\n[Round {round_idx + 1}] Strategy: {strategy_name}, LR={lr}, Epochs={local_epochs}")

        global_state = copy.deepcopy(self.global_model.state_dict())
        client_states, client_metrics = self._train_clients_for_round(
            global_state=global_state,
            lr=lr,
            local_epochs=local_epochs,
            verbose=verbose
        )
        weights = self.compute_client_weights(strategy_name, client_metrics)
        return self._apply_aggregated_round(
            round_idx=round_idx,
            strategy_name=strategy_name,
            lr=lr,
            local_epochs=local_epochs,
            client_states=client_states,
            client_metrics=client_metrics,
            weights=weights,
            server_lr_scale=server_lr_scale,
            verbose=verbose
        )

    def run_training(
        self,
        num_rounds: int,
        strategy_name: Optional[str] = None,
        lr: Optional[float] = None,
        local_epochs: Optional[int] = None,
        server_lr_scale: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        strategy_name = strategy_name if strategy_name is not None else self.default_strategy

        if verbose:
            print("=" * 60)
            print("Starting MAS-FL Training (Phase 2: Multi-Strategy)")
            print("=" * 60)
            print(f"  Rounds: {num_rounds}")
            print(f"  Strategy: {strategy_name}")
            print(f"  Clients: {list(self.client_agents.keys())}")
            print(f"  Learning Rate: {lr if lr else self.default_lr}")
            print(f"  Local Epochs: {local_epochs if local_epochs else self.default_local_epochs}")

        for round_idx in range(num_rounds):
            self.run_round(
                round_idx=round_idx,
                strategy_name=strategy_name,
                lr=lr,
                local_epochs=local_epochs,
                server_lr_scale=server_lr_scale,
                verbose=verbose
            )

        if verbose:
            print("\n" + "-" * 60)
            print("Restoring Best Checkpoint...")
            print("-" * 60)
            last_round_mape = self.history_round_metrics[-1]["global_val"]["mape"] if self.history_round_metrics else float('inf')
            print(f"  Last round (Round {num_rounds}) Val MAPE: {last_round_mape * 100:.2f}%")
            print(f"  Best round (Round {self.best_round + 1}) Val MAPE: {self.best_val_mape * 100:.2f}%")

        if self.best_model_state is not None:
            self.global_model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"  >>> Model restored to Round {self.best_round + 1} (best checkpoint)")
        else:
            if verbose:
                print("  [Warning] No best checkpoint found, using last round model")

        if verbose:
            print("\n  Evaluating best checkpoint on test set...")
        test_metrics = self.evaluate_global(data_loader=self.global_test_loader)

        if verbose:
            print("\n" + "=" * 60)
            print("Training Completed!")
            print("=" * 60)
            print(f"  Strategy: {strategy_name}")
            print(f"  Best Round: {self.best_round + 1}")
            print(f"  Best Val MAPE: {self.best_val_mape * 100:.2f}%")
            print("\nTest Set Results (using best checkpoint):")
            print(f"  MAPE:  {test_metrics['mape'] * 100:.2f}%")
            print(f"  RMSE:  ${test_metrics['rmse']:,.2f}")
            print(f"  MAE:   ${test_metrics['mae']:,.2f}")
            print(f"  MPE:   {test_metrics['mpe'] * 100:.2f}%")
            print(f"  NRMSE: {test_metrics['nrmse'] * 100:.2f}%")
            print(f"  R2:    {test_metrics['r2']:.4f}")

        return {
            "history": self.history_round_metrics,
            "best_round": self.best_round,
            "best_val_mape": self.best_val_mape,
            "test_metrics": test_metrics,
            "strategy_used": strategy_name
        }

    def run_training_strict_final_round(
        self,
        num_rounds: int,
        strategy_name: Optional[str] = None,
        lr: Optional[float] = None,
        local_epochs: Optional[int] = None,
        server_lr_scale: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run a fixed strategy without server-side validation data.

        This path is for strict federated baselines. During training the server
        only uses uploaded client model states and aggregate client reports. The
        held-out server test loader is touched once after the final round.
        """
        strategy_name = strategy_name if strategy_name is not None else self.default_strategy
        lr = lr if lr is not None else self.default_lr
        local_epochs = local_epochs if local_epochs is not None else self.default_local_epochs

        if verbose:
            print("=" * 60)
            print("Starting strict no-server-validation federated training")
            print("=" * 60)
            print(f"  Rounds: {num_rounds}")
            print(f"  Strategy: {strategy_name}")
            print(f"  Checkpoint policy: final_round")
            print(f"  Clients: {list(self.client_agents.keys())}")
            print(f"  Learning Rate: {lr}")
            print(f"  Local Epochs: {local_epochs}")

        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n[Round {round_idx + 1}] Strategy: {strategy_name}, LR={lr}, Epochs={local_epochs}")

            global_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=global_state,
                lr=lr,
                local_epochs=local_epochs,
                verbose=verbose
            )
            weights = self.compute_client_weights(strategy_name, client_metrics)
            self._apply_aggregated_round_client_reported(
                round_idx=round_idx,
                strategy_name=strategy_name,
                lr=lr,
                local_epochs=local_epochs,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=weights,
                server_lr_scale=server_lr_scale,
                verbose=verbose
            )

        if verbose:
            print("\n  Evaluating final-round model on test set...")
        test_metrics = self.evaluate_global(data_loader=self.global_test_loader)

        if verbose:
            print("\n" + "=" * 60)
            print("Strict federated training completed!")
            print("=" * 60)
            print(f"  Strategy: {strategy_name}")
            print(f"  Final Round: {self.best_round + 1}")
            print(f"  Client-reported Val MAPE: {self.best_val_mape * 100:.2f}%")
            print("\nTest Set Results (final round only):")
            print(f"  MAPE:  {test_metrics['mape'] * 100:.2f}%")
            print(f"  RMSE:  ${test_metrics['rmse']:,.2f}")
            print(f"  MAE:   ${test_metrics['mae']:,.2f}")
            print(f"  MPE:   {test_metrics['mpe'] * 100:.2f}%")
            print(f"  NRMSE: {test_metrics['nrmse'] * 100:.2f}%")
            print(f"  R2:    {test_metrics['r2']:.4f}")

        return {
            "history": self.history_round_metrics,
            "best_round": self.best_round,
            "best_val_mape": self.best_val_mape,
            "test_metrics": test_metrics,
            "strategy_used": strategy_name,
            "validation_source": "client_reported",
            "checkpoint_policy": "final_round",
        }

    def run_training_with_llm(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
        base_server_lr_scale: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        if self.llm_planner is None:
            raise ValueError("LLM Planner not initialized")

        base_lr = base_lr if base_lr is not None else self.default_lr
        base_local_epochs = base_local_epochs if base_local_epochs is not None else self.default_local_epochs

        if verbose:
            print("=" * 60)
            print("Starting MAS-FL Training (Phase 3: LLM-Guided)")
            print("=" * 60)
            print(f"  Rounds: {num_rounds}")
            print(f"  Clients: {list(self.client_agents.keys())}")
            print(f"  Base Learning Rate: {base_lr}")
            print(f"  Base Local Epochs: {base_local_epochs}")
            print(f"  LLM call interval: every {self.llm_call_every_n_rounds} round(s)")

        llm_decisions = []
        next_training_decision = {
            "lr_scale": 1.0,
            "server_lr_scale": base_server_lr_scale,
            "epoch_delta": 0,
            "reasoning": "Initial training parameters before first validation preview"
        }

        for round_idx in range(num_rounds):
            lr = base_lr * next_training_decision.get("lr_scale", 1.0)
            server_lr_scale = self._server_lr_scale_from_decision(next_training_decision)
            local_epochs = max(5, min(30, base_local_epochs + next_training_decision.get("epoch_delta", 0)))

            if verbose:
                print(f"\n[Round {round_idx + 1}] LLM-preview flow, LR={lr:.6f}, Epochs={local_epochs}")

            global_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=global_state,
                lr=lr,
                local_epochs=local_epochs,
                verbose=verbose
            )

            candidate_weight_preview = (
                self.build_candidate_weight_preview(client_metrics)
                if self.enable_candidate_weight_preview else {}
            )
            candidate_validation_preview = (
                self.build_candidate_validation_preview(client_states, client_metrics)
                if self.enable_candidate_validation_preview else {}
            )
            decision_context = {
                "current_client_metrics": self._safe_client_metric_summary(client_metrics),
                "candidate_weight_preview": candidate_weight_preview,
                "candidate_validation_preview": candidate_validation_preview,
            }
            should_call_llm = (
                round_idx % self.llm_call_every_n_rounds == 0
                or self.llm_planner.last_llm_decision is None
            )

            reused_decision = False
            if should_call_llm:
                decision = self.llm_planner.choose_strategy(
                    history_round_metrics=self.history_round_metrics,
                    current_round=round_idx,
                    num_rounds=num_rounds,
                    decision_context=decision_context
                )
            else:
                decision = copy.deepcopy(self.llm_planner.last_llm_decision)
                decision["reasoning"] = (
                    f"Reused previous LLM decision because "
                    f"call_every_n_rounds={self.llm_call_every_n_rounds}"
                )
                reused_decision = True

            decision_record = copy.deepcopy(decision)
            decision_record["is_reused"] = reused_decision
            decision_record["training_lr_used_this_round"] = lr
            decision_record["server_lr_scale_used_this_round"] = server_lr_scale
            decision_record["training_epochs_used_this_round"] = local_epochs
            decision_record["hyperparameter_effect"] = (
                "lr_scale, server_lr_scale, and epoch_delta apply to the next training round"
            )
            llm_decisions.append(decision_record)

            strategy_name = decision["chosen_strategy_name"]
            strategy_params_override = self._strategy_params_from_decision(decision)
            weights = self.compute_client_weights(
                strategy_name,
                client_metrics,
                strategy_params_override=strategy_params_override
            )

            if verbose:
                decision_mode = "reused" if reused_decision else "new"
                print(
                    f"\n  [LLM Decision] Round {round_idx + 1}: "
                    f"strategy={strategy_name}, mode={decision_mode}, "
                    f"next_lr_scale={decision.get('lr_scale', 1.0):.2f}, "
                    f"next_server_lr_scale={self._server_lr_scale_from_decision(decision):.2f}, "
                    f"next_epoch_delta={decision.get('epoch_delta', 0)}"
                )

            self._apply_aggregated_round(
                round_idx=round_idx,
                strategy_name=strategy_name,
                lr=lr,
                local_epochs=local_epochs,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=weights,
                strategy_params_override=strategy_params_override,
                llm_decision=decision_record,
                candidate_weight_preview=candidate_weight_preview,
                candidate_validation_preview=candidate_validation_preview,
                server_lr_scale=server_lr_scale,
                verbose=verbose
            )
            next_training_decision = copy.deepcopy(decision)

        if verbose:
            print("\n" + "-" * 60)
            print("Restoring Best Checkpoint...")
            print("-" * 60)
            last_round_mape = self.history_round_metrics[-1]["global_val"]["mape"] if self.history_round_metrics else float('inf')
            print(f"  Last round (Round {num_rounds}) Val MAPE: {last_round_mape * 100:.2f}%")
            print(f"  Best round (Round {self.best_round + 1}) Val MAPE: {self.best_val_mape * 100:.2f}%")

        if self.best_model_state is not None:
            self.global_model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"  >>> Model restored to Round {self.best_round + 1} (best checkpoint)")
        else:
            if verbose:
                print("  [Warning] No best checkpoint found, using last round model")

        if verbose:
            print("\n  Evaluating best checkpoint on test set...")
        test_metrics = self.evaluate_global(data_loader=self.global_test_loader)

        if verbose:
            print("\n" + "=" * 60)
            print("LLM-Guided Training Completed!")
            print("=" * 60)
            print(f"  Best Round: {self.best_round + 1}")
            print(f"  Best Val MAPE: {self.best_val_mape * 100:.2f}%")
            print("\nTest Set Results (using best checkpoint):")
            print(f"  MAPE:  {test_metrics['mape'] * 100:.2f}%")
            print(f"  RMSE:  ${test_metrics['rmse']:,.2f}")
            print(f"  MAE:   ${test_metrics['mae']:,.2f}")
            print(f"  MPE:   {test_metrics['mpe'] * 100:.2f}%")
            print(f"  NRMSE: {test_metrics['nrmse'] * 100:.2f}%")
            print(f"  R2:    {test_metrics['r2']:.4f}")

            strategy_counts = {}
            for d in llm_decisions:
                s = d["chosen_strategy_name"]
                strategy_counts[s] = strategy_counts.get(s, 0) + 1
            print("\nLLM Strategy Selection Summary:")
            for s, count in strategy_counts.items():
                print(f"  {s}: {count} rounds ({count/num_rounds*100:.1f}%)")

        return {
            "history": self.history_round_metrics,
            "best_round": self.best_round,
            "best_val_mape": self.best_val_mape,
            "test_metrics": test_metrics,
            "llm_decisions": llm_decisions,
            "mode": "llm_guided"
        }

    def _finalize_training(
        self,
        num_rounds: int,
        mode_name: str,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if verbose:
            print("\n" + "-" * 60)
            print("Restoring Best Checkpoint...")
            print("-" * 60)
            last_round_mape = self.history_round_metrics[-1]["global_val"]["mape"] if self.history_round_metrics else float('inf')
            print(f"  Last round (Round {num_rounds}) Val MAPE: {last_round_mape * 100:.2f}%")
            print(f"  Best round (Round {self.best_round + 1}) Val MAPE: {self.best_val_mape * 100:.2f}%")

        if self.best_model_state is not None:
            self.global_model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"  >>> Model restored to Round {self.best_round + 1} (best checkpoint)")
        elif verbose:
            print("  [Warning] No best checkpoint found, using last round model")

        if verbose:
            print("\n  Evaluating best checkpoint on test set...")
        test_metrics = self.evaluate_global(data_loader=self.global_test_loader)

        if verbose:
            print("\n" + "=" * 60)
            print(f"{mode_name} Training Completed!")
            print("=" * 60)
            print(f"  Best Round: {self.best_round + 1}")
            print(f"  Best Val MAPE: {self.best_val_mape * 100:.2f}%")
            print("\nTest Set Results (using best checkpoint):")
            print(f"  MAPE:  {test_metrics['mape'] * 100:.2f}%")
            print(f"  RMSE:  ${test_metrics['rmse']:,.2f}")
            print(f"  MAE:   ${test_metrics['mae']:,.2f}")
            print(f"  MPE:   {test_metrics['mpe'] * 100:.2f}%")
            print(f"  NRMSE: {test_metrics['nrmse'] * 100:.2f}%")
            print(f"  R2:    {test_metrics['r2']:.4f}")

        return {
            "history": self.history_round_metrics,
            "best_round": self.best_round,
            "best_val_mape": self.best_val_mape,
            "test_metrics": test_metrics,
            "mode": mode_name,
        }

    def run_training_with_validation_guided_adaptation(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
        candidate_budget: int = 30,
        weight_grid_step: float = 0.05,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        selection_epsilon: float = 0.002,
        llm_score_tolerance: Optional[float] = None,
        weight_l1_change_limit: float = 0.40,
        large_improvement_threshold: float = 0.01,
        score_profile: str = "mape_primary",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        base_lr = base_lr if base_lr is not None else self.default_lr
        base_local_epochs = base_local_epochs if base_local_epochs is not None else self.default_local_epochs

        if verbose:
            print("=" * 60)
            print("Starting Validation-Guided FedYogi-TR Training")
            print("=" * 60)
            print(f"  Rounds: {num_rounds}")
            print(f"  Candidate budget: {candidate_budget}")
            print(f"  Weight step: {weight_grid_step}")
            print(f"  Client weight bounds: [{min_client_weight}, {max_client_weight}]")

        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n[Round {round_idx + 1}] Validation-guided candidate search")

            current_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=current_state,
                lr=base_lr,
                local_epochs=base_local_epochs,
                verbose=verbose,
            )
            candidates = self.build_continuous_candidate_preview(
                client_states=client_states,
                client_metrics=client_metrics,
                current_state=current_state,
                candidate_budget=candidate_budget,
                weight_grid_step=weight_grid_step,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
                server_lr_scales=[0.5, 1.0, 1.5],
                epoch_deltas=[0],
                score_profile=score_profile,
            )
            selected, gate_info = select_candidate_by_gate(
                candidates=candidates,
                conservative_candidate_id="size_anchor",
                requested_candidate_id=None,
                epsilon=selection_epsilon,
                score_tolerance=0.0,
                previous_weights=self.previous_accepted_weights,
                weight_l1_limit=weight_l1_change_limit,
                large_improvement_threshold=large_improvement_threshold,
            )
            self.previous_accepted_weights = copy.deepcopy(selected.weights)

            if verbose:
                print(
                    f"    Selected candidate: {selected.candidate_id}, "
                    f"score={selected.score:.6f}, gate={gate_info['gate_status']}"
                )

            record = self._apply_aggregated_round(
                round_idx=round_idx,
                strategy_name="continuous_weight_search",
                lr=base_lr,
                local_epochs=base_local_epochs + selected.epoch_delta,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=selected.weights,
                strategy_params_override={
                    "candidate_id": selected.candidate_id,
                    "candidate_source": selected.source,
                    "score_profile": score_profile,
                },
                llm_decision=None,
                candidate_weight_preview={
                    candidate.candidate_id: candidate.weights
                    for candidate in candidates
                },
                candidate_validation_preview=self._candidate_preview_for_history(candidates),
                server_lr_scale=selected.server_lr_scale,
                verbose=verbose,
            )
            record["selected_candidate_id"] = selected.candidate_id
            record["requested_candidate_id"] = gate_info.get("requested_candidate_id")
            record["gate_status"] = gate_info["gate_status"]
            record["candidate_score"] = selected.score
            record["candidate_source"] = selected.source
            record["candidate_budget"] = candidate_budget
            record["weight_l1_from_previous"] = gate_info.get("weight_l1_from_previous", 0.0)
            record["selection_epsilon"] = selection_epsilon
            record["weight_l1_change_limit"] = weight_l1_change_limit

        return self._finalize_training(
            num_rounds=num_rounds,
            mode_name="validation_guided",
            verbose=verbose,
        )

    def run_training_with_coherence_guided_adaptation(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        base_lr = base_lr if base_lr is not None else self.default_lr
        base_local_epochs = base_local_epochs if base_local_epochs is not None else self.default_local_epochs
        previous_global_delta = None

        if verbose:
            print("=" * 60)
            print("Starting Coherence-Guided FedYogi-TR Training")
            print("=" * 60)
            print(f"  Rounds: {num_rounds}")
            print(f"  Client weight bounds: [{min_client_weight}, {max_client_weight}]")

        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n[Round {round_idx + 1}] Coherence-guided aggregation")

            current_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=current_state,
                lr=base_lr,
                local_epochs=base_local_epochs,
                verbose=verbose,
            )
            with _preserve_rng_state():
                diagnostics, summaries = self._coherence_artifacts_for_round(
                    current_state=current_state,
                    client_states=client_states,
                    client_metrics=client_metrics,
                    previous_global_delta=previous_global_delta,
                )
                weights = compute_coherence_weights(
                    diagnostics,
                    min_client_weight=min_client_weight,
                    max_client_weight=max_client_weight,
                )
            self.previous_accepted_weights = copy.deepcopy(weights)

            record = self._apply_aggregated_round(
                round_idx=round_idx,
                strategy_name="coherence_guided",
                lr=base_lr,
                local_epochs=base_local_epochs,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=weights,
                strategy_params_override={"source": "coherence_baseline"},
                llm_decision=None,
                server_lr_scale=1.0,
                verbose=verbose,
            )
            record.update({
                "coherence_diagnostics": diagnostics,
                "client_summaries": summaries,
                "generated_weights_raw": weights,
                "generated_weights_projected": weights,
                "constraint_status": {"source": "coherence_baseline", "fallback_used": False},
                "llm_reasoning": None,
                "llm_risk": None,
            })
            previous_global_delta = self._state_delta(
                copy.deepcopy(self.global_model.state_dict()),
                current_state,
            )

        return self._finalize_training(
            num_rounds=num_rounds,
            mode_name="coherence_guided",
            verbose=verbose,
        )

    def run_training_with_llm_generative_coherence(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        weight_l1_change_limit: float = 0.40,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if self.llm_planner is None:
            raise ValueError("LLM Planner not initialized")

        base_lr = base_lr if base_lr is not None else self.default_lr
        base_local_epochs = base_local_epochs if base_local_epochs is not None else self.default_local_epochs
        previous_global_delta = None
        llm_decisions = []

        if verbose:
            print("=" * 60)
            print("Starting LLM-GCA FedYogi-TR Training")
            print("=" * 60)
            print(f"  Rounds: {num_rounds}")
            print(f"  Client weight bounds: [{min_client_weight}, {max_client_weight}]")
            print(f"  Weight L1 limit: {weight_l1_change_limit}")

        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n[Round {round_idx + 1}] LLM generative coherence aggregation")

            current_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=current_state,
                lr=base_lr,
                local_epochs=base_local_epochs,
                verbose=verbose,
            )
            with _preserve_rng_state():
                diagnostics, summaries = self._coherence_artifacts_for_round(
                    current_state=current_state,
                    client_states=client_states,
                    client_metrics=client_metrics,
                    previous_global_delta=previous_global_delta,
                )
                llm_decision = self.llm_planner.choose_generated_weights(
                    history_round_metrics=self.history_round_metrics,
                    current_round=round_idx,
                    num_rounds=num_rounds,
                    client_summaries=summaries,
                    coherence_diagnostics=diagnostics,
                    previous_weights=self.previous_accepted_weights,
                    min_client_weight=min_client_weight,
                    max_client_weight=max_client_weight,
                    weight_l1_change_limit=weight_l1_change_limit,
                )
            weights = copy.deepcopy(llm_decision["projected_weights"])
            self.previous_accepted_weights = copy.deepcopy(weights)

            decision_record = copy.deepcopy(llm_decision)
            decision_record.update({
                "round": round_idx + 1,
                "selected_weights": weights,
                "note": "LLM generated continuous aggregation weights; constraint layer enforced legality and stability.",
            })
            llm_decisions.append(decision_record)

            record = self._apply_aggregated_round(
                round_idx=round_idx,
                strategy_name="llm_generative_coherence",
                lr=base_lr,
                local_epochs=base_local_epochs,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=weights,
                strategy_params_override={
                    "decision_type": llm_decision.get("decision_type", "balanced"),
                    "fallback_used": llm_decision.get("fallback_used", False),
                },
                llm_decision=decision_record,
                server_lr_scale=float(llm_decision.get("server_lr_scale", 1.0)),
                verbose=verbose,
            )
            record.update({
                "coherence_diagnostics": diagnostics,
                "client_summaries": summaries,
                "generated_weights_raw": llm_decision.get("generated_weights_raw", {}),
                "generated_weights_projected": weights,
                "constraint_status": llm_decision.get("constraint_status", {}),
                "llm_reasoning": llm_decision.get("reasoning", ""),
                "llm_risk": llm_decision.get("risk", ""),
                "llm_generative_decision": decision_record,
            })
            previous_global_delta = self._state_delta(
                copy.deepcopy(self.global_model.state_dict()),
                current_state,
            )

        results = self._finalize_training(
            num_rounds=num_rounds,
            mode_name="llm_generative_coherence",
            verbose=verbose,
        )
        results["llm_decisions"] = llm_decisions
        return results

    def run_training_with_strict_coherence_guided_adaptation(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        base_lr = base_lr if base_lr is not None else self.default_lr
        base_local_epochs = base_local_epochs if base_local_epochs is not None else self.default_local_epochs
        previous_global_delta = None

        if verbose:
            print("=" * 60)
            print("Starting Strict Client-Reported Coherence FedYogi-TR Training")
            print("=" * 60)
            print("  Server validation preview: disabled")
            print(f"  Rounds: {num_rounds}")
            print(f"  Client weight bounds: [{min_client_weight}, {max_client_weight}]")

        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n[Round {round_idx + 1}] Strict coherence-guided aggregation")

            current_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=current_state,
                lr=base_lr,
                local_epochs=base_local_epochs,
                verbose=verbose,
            )
            diagnostics, summaries = self._coherence_artifacts_for_round(
                current_state=current_state,
                client_states=client_states,
                client_metrics=client_metrics,
                previous_global_delta=previous_global_delta,
            )
            weights = compute_coherence_weights(
                diagnostics,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
            )
            self.previous_accepted_weights = copy.deepcopy(weights)

            record = self._apply_aggregated_round_client_reported(
                round_idx=round_idx,
                strategy_name="strict_coherence_guided",
                lr=base_lr,
                local_epochs=base_local_epochs,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=weights,
                strategy_params_override={"source": "strict_coherence_baseline"},
                llm_decision=None,
                server_lr_scale=1.0,
                verbose=verbose,
            )
            record.update({
                "coherence_diagnostics": diagnostics,
                "client_summaries": summaries,
                "generated_weights_raw": weights,
                "generated_weights_projected": weights,
                "constraint_status": {"source": "strict_coherence_baseline", "fallback_used": False},
                "llm_reasoning": None,
                "llm_risk": None,
            })
            previous_global_delta = self._state_delta(
                copy.deepcopy(self.global_model.state_dict()),
                current_state,
            )

        return self._finalize_training(
            num_rounds=num_rounds,
            mode_name="strict_coherence_guided",
            verbose=verbose,
        )

    def run_training_with_llm_strict_generative_coherence(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        weight_l1_change_limit: float = 0.40,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if self.llm_planner is None:
            raise ValueError("LLM Planner not initialized")

        base_lr = base_lr if base_lr is not None else self.default_lr
        base_local_epochs = base_local_epochs if base_local_epochs is not None else self.default_local_epochs
        previous_global_delta = None
        llm_decisions = []

        if verbose:
            print("=" * 60)
            print("Starting Strict LLM-GCA FedYogi-TR Training")
            print("=" * 60)
            print("  Server validation preview: disabled")
            print(f"  Rounds: {num_rounds}")
            print(f"  Client weight bounds: [{min_client_weight}, {max_client_weight}]")
            print(f"  Weight L1 limit: {weight_l1_change_limit}")

        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n[Round {round_idx + 1}] Strict LLM generative aggregation")

            current_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=current_state,
                lr=base_lr,
                local_epochs=base_local_epochs,
                verbose=verbose,
            )
            diagnostics, summaries = self._coherence_artifacts_for_round(
                current_state=current_state,
                client_states=client_states,
                client_metrics=client_metrics,
                previous_global_delta=previous_global_delta,
            )
            llm_decision = self.llm_planner.choose_generated_weights(
                history_round_metrics=self.history_round_metrics,
                current_round=round_idx,
                num_rounds=num_rounds,
                client_summaries=summaries,
                coherence_diagnostics=diagnostics,
                previous_weights=self.previous_accepted_weights,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
                weight_l1_change_limit=weight_l1_change_limit,
            )
            weights = copy.deepcopy(llm_decision["projected_weights"])
            self.previous_accepted_weights = copy.deepcopy(weights)

            decision_record = copy.deepcopy(llm_decision)
            decision_record.update({
                "round": round_idx + 1,
                "selected_weights": weights,
                "validation_source": "client_reported",
                "control_action": llm_decision.get("control_action", "balanced"),
                "server_lr_scale": float(llm_decision.get("server_lr_scale", 1.0)),
                "anchor_l1_limit": llm_decision.get("anchor_l1_limit"),
                "note": "LLM generated federated control parameters from client summaries and update diagnostics; server validation preview is disabled.",
            })
            llm_decisions.append(decision_record)

            record = self._apply_aggregated_round_client_reported(
                round_idx=round_idx,
                strategy_name="llm_strict_generative_coherence",
                lr=base_lr,
                local_epochs=base_local_epochs,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=weights,
                strategy_params_override={
                    "decision_type": llm_decision.get("decision_type", "balanced"),
                    "control_action": llm_decision.get("control_action", "balanced"),
                    "fallback_used": llm_decision.get("fallback_used", False),
                    "validation_source": "client_reported",
                    "anchor_l1_limit": llm_decision.get("anchor_l1_limit"),
                },
                llm_decision=decision_record,
                server_lr_scale=float(llm_decision.get("server_lr_scale", 1.0)),
                verbose=verbose,
            )
            record.update({
                "coherence_diagnostics": diagnostics,
                "client_summaries": summaries,
                "generated_weights_raw": llm_decision.get("generated_weights_raw", {}),
                "generated_weights_projected": weights,
                "constraint_status": llm_decision.get("constraint_status", {}),
                "llm_reasoning": llm_decision.get("reasoning", ""),
                "llm_risk": llm_decision.get("risk", ""),
                "llm_generative_decision": decision_record,
            })
            previous_global_delta = self._state_delta(
                copy.deepcopy(self.global_model.state_dict()),
                current_state,
            )

        results = self._finalize_training(
            num_rounds=num_rounds,
            mode_name="llm_strict_generative_coherence",
            verbose=verbose,
        )
        results["llm_decisions"] = llm_decisions
        return results

    def run_training_with_validation_preview_gca(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
        candidate_budget: int = 30,
        weight_grid_step: float = 0.05,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        selection_epsilon: float = 0.002,
        weight_l1_change_limit: float = 0.40,
        large_improvement_threshold: float = 0.01,
        score_profile: str = "mape_primary",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        base_lr = base_lr if base_lr is not None else self.default_lr
        base_local_epochs = base_local_epochs if base_local_epochs is not None else self.default_local_epochs
        previous_global_delta = None

        if verbose:
            print("=" * 60)
            print("Starting Validation-Preview GCA FedYogi-TR Training")
            print("=" * 60)
            print(f"  Rounds: {num_rounds}")
            print(f"  Candidate budget: {candidate_budget}")
            print(f"  Weight step: {weight_grid_step}")
            print(f"  Client weight bounds: [{min_client_weight}, {max_client_weight}]")

        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n[Round {round_idx + 1}] Validation-preview GCA candidate search")

            current_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=current_state,
                lr=base_lr,
                local_epochs=base_local_epochs,
                verbose=verbose,
            )
            diagnostics, summaries = self._coherence_artifacts_for_round(
                current_state=current_state,
                client_states=client_states,
                client_metrics=client_metrics,
                previous_global_delta=previous_global_delta,
            )
            candidates = self.build_continuous_candidate_preview(
                client_states=client_states,
                client_metrics=client_metrics,
                current_state=current_state,
                candidate_budget=candidate_budget,
                weight_grid_step=weight_grid_step,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
                server_lr_scales=[0.5, 0.75, 1.0, 1.5],
                epoch_deltas=[0],
                score_profile=score_profile,
                coherence_diagnostics=diagnostics,
            )
            candidate_preview = self._candidate_preview_for_history(candidates)
            selected, gate_info = select_candidate_by_gate(
                candidates=candidates,
                conservative_candidate_id="size_anchor",
                requested_candidate_id=None,
                epsilon=selection_epsilon,
                score_tolerance=0.0,
                previous_weights=self.previous_accepted_weights,
                weight_l1_limit=weight_l1_change_limit,
                large_improvement_threshold=large_improvement_threshold,
            )
            self.previous_accepted_weights = copy.deepcopy(selected.weights)

            if verbose:
                print(
                    f"    Selected candidate: {selected.candidate_id}, "
                    f"score={selected.score:.6f}, gate={gate_info['gate_status']}"
                )

            record = self._apply_aggregated_round(
                round_idx=round_idx,
                strategy_name="validation_preview_gca",
                lr=base_lr,
                local_epochs=base_local_epochs + selected.epoch_delta,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=selected.weights,
                strategy_params_override={
                    "candidate_id": selected.candidate_id,
                    "candidate_source": selected.source,
                    "score_profile": score_profile,
                    "llm_used": False,
                },
                llm_decision=None,
                candidate_weight_preview={
                    candidate.candidate_id: candidate.weights
                    for candidate in candidates
                },
                candidate_validation_preview=candidate_preview,
                server_lr_scale=selected.server_lr_scale,
                verbose=verbose,
            )
            record.update({
                "selected_candidate_id": selected.candidate_id,
                "requested_candidate_id": gate_info.get("requested_candidate_id"),
                "gate_status": gate_info["gate_status"],
                "candidate_score": selected.score,
                "candidate_source": selected.source,
                "candidate_budget": candidate_budget,
                "weight_l1_from_previous": gate_info.get("weight_l1_from_previous", 0.0),
                "selection_epsilon": selection_epsilon,
                "weight_l1_change_limit": weight_l1_change_limit,
                "coherence_diagnostics": diagnostics,
                "client_summaries": summaries,
                "llm_reasoning": None,
                "llm_risk": None,
            })
            previous_global_delta = self._state_delta(
                copy.deepcopy(self.global_model.state_dict()),
                current_state,
            )

        return self._finalize_training(
            num_rounds=num_rounds,
            mode_name="validation_preview_gca",
            verbose=verbose,
        )

    def run_training_with_llm_validation_preview_generative(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
        candidate_budget: int = 30,
        weight_grid_step: float = 0.05,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        selection_epsilon: float = 0.002,
        llm_score_tolerance: Optional[float] = None,
        weight_l1_change_limit: float = 0.40,
        large_improvement_threshold: float = 0.01,
        score_profile: str = "mape_primary",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if self.llm_planner is None:
            raise ValueError("LLM Planner not initialized")

        base_lr = base_lr if base_lr is not None else self.default_lr
        base_local_epochs = base_local_epochs if base_local_epochs is not None else self.default_local_epochs
        llm_score_tolerance = selection_epsilon if llm_score_tolerance is None else float(llm_score_tolerance)
        previous_global_delta = None
        llm_decisions = []

        if verbose:
            print("=" * 60)
            print("Starting LLM Validation-Preview Generative FedYogi-TR Training")
            print("=" * 60)
            print(f"  Rounds: {num_rounds}")
            print(f"  Candidate budget: {candidate_budget}")
            print(f"  Weight step: {weight_grid_step}")
            print(f"  Client weight bounds: [{min_client_weight}, {max_client_weight}]")
            print(f"  LLM score tolerance: {llm_score_tolerance}")

        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n[Round {round_idx + 1}] LLM validation-preview generative control")

            current_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=current_state,
                lr=base_lr,
                local_epochs=base_local_epochs,
                verbose=verbose,
            )
            diagnostics, summaries = self._coherence_artifacts_for_round(
                current_state=current_state,
                client_states=client_states,
                client_metrics=client_metrics,
                previous_global_delta=previous_global_delta,
            )
            candidates = self.build_continuous_candidate_preview(
                client_states=client_states,
                client_metrics=client_metrics,
                current_state=current_state,
                candidate_budget=candidate_budget,
                weight_grid_step=weight_grid_step,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
                server_lr_scales=[0.5, 0.75, 1.0, 1.5],
                epoch_deltas=[0],
                score_profile=score_profile,
                coherence_diagnostics=diagnostics,
            )
            candidate_preview = self._candidate_preview_for_history(candidates)
            llm_decision = self.llm_planner.choose_validation_preview_generative_strategy(
                history_round_metrics=self.history_round_metrics,
                current_round=round_idx,
                num_rounds=num_rounds,
                candidate_preview=candidate_preview,
                client_summaries=summaries,
                coherence_diagnostics=diagnostics,
                score_tolerance=llm_score_tolerance,
            )

            requested_candidate_id = llm_decision.get("selected_candidate_ids", [None])[0]
            mixture_candidate = None
            if llm_decision.get("projected_weights"):
                mixture_candidate = AdaptiveCandidate(
                    candidate_id="llm_mixture",
                    weights={
                        client_id: float(weight)
                        for client_id, weight in llm_decision["projected_weights"].items()
                    },
                    server_lr_scale=float(llm_decision.get("server_lr_scale", 1.0)),
                    epoch_delta=0,
                    source="llm_validation_preview_mixture",
                    metadata={
                        "selected_candidate_ids": llm_decision.get("selected_candidate_ids", []),
                        "mixture_weights": llm_decision.get("mixture_weights", {}),
                    },
                )
                aggregated_state = self.aggregate_with_weights(client_states, mixture_candidate.weights)
                next_state, server_update = self.server_optimizer.preview_step(
                    current_state=current_state,
                    weighted_average_state=aggregated_state,
                    server_lr_scale=mixture_candidate.server_lr_scale,
                )
                global_metrics = self._evaluate_state_on_global_val(next_state)
                per_client = self._evaluate_state_on_client_vals(next_state)
                client_mapes = [metrics["val_mape"] for metrics in per_client.values()]
                mixture_candidate.validation_metrics = {
                    "mape": float(global_metrics["mape"]),
                    "rmse": float(global_metrics["rmse"]),
                    "mae": float(global_metrics["mae"]),
                    "mpe": float(global_metrics["mpe"]),
                    "nrmse": float(global_metrics["nrmse"]),
                    "r2": float(global_metrics["r2"]),
                }
                mixture_candidate.client_gap = (max(client_mapes) - min(client_mapes)) if client_mapes else 0.0
                mixture_candidate.update_norm = float(server_update.get("update_norm", 0.0))
                mixture_candidate.score = score_candidate_metrics(
                    metrics=mixture_candidate.validation_metrics,
                    client_gap=mixture_candidate.client_gap,
                    update_norm=mixture_candidate.update_norm,
                    weights=mixture_candidate.weights,
                    previous_weights=self.previous_accepted_weights or self.compute_client_weights("size_only", client_metrics),
                    profile=score_profile,
                )
                mixture_candidate.metadata.update({
                    "aggregation_delta_norm": float(server_update.get("aggregation_delta_norm", 0.0)),
                    "coordinate_step_clipped": bool(server_update.get("coordinate_step_clipped", False)),
                    "coordinate_direction_rejected": bool(server_update.get("coordinate_direction_rejected", False)),
                    "client_validation_metrics": per_client,
                })
                candidates.append(mixture_candidate)
                candidate_preview = self._candidate_preview_for_history(candidates)
                requested_candidate_id = "llm_mixture"

            selected, gate_info = select_candidate_by_gate(
                candidates=candidates,
                conservative_candidate_id="size_anchor",
                requested_candidate_id=requested_candidate_id,
                epsilon=selection_epsilon,
                score_tolerance=llm_score_tolerance,
                previous_weights=self.previous_accepted_weights,
                weight_l1_limit=weight_l1_change_limit,
                large_improvement_threshold=large_improvement_threshold,
            )
            self.previous_accepted_weights = copy.deepcopy(selected.weights)

            decision_record = copy.deepcopy(llm_decision)
            decision_record.update({
                "round": round_idx + 1,
                "requested_candidate_id": requested_candidate_id,
                "selected_candidate_id": selected.candidate_id,
                "gate_status": gate_info["gate_status"],
                "candidate_score": selected.score,
                "candidate_source": selected.source,
                "llm_mixture_score": mixture_candidate.score if mixture_candidate else None,
                "note": "LLM selects a validation-preview candidate mixture; deterministic gate executes only a validated stable action.",
            })
            llm_decisions.append(decision_record)

            if verbose:
                print(
                    f"    LLM mixture requested: {requested_candidate_id}, "
                    f"selected: {selected.candidate_id}, "
                    f"score={selected.score:.6f}, gate={gate_info['gate_status']}"
                )

            record = self._apply_aggregated_round(
                round_idx=round_idx,
                strategy_name="llm_validation_preview_generative",
                lr=base_lr,
                local_epochs=base_local_epochs + selected.epoch_delta,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=selected.weights,
                strategy_params_override={
                    "candidate_id": selected.candidate_id,
                    "candidate_source": selected.source,
                    "score_profile": score_profile,
                    "decision_type": llm_decision.get("decision_type", "balanced"),
                    "fallback_used": llm_decision.get("fallback_used", False),
                },
                llm_decision=decision_record,
                candidate_weight_preview={
                    candidate.candidate_id: candidate.weights
                    for candidate in candidates
                },
                candidate_validation_preview=candidate_preview,
                server_lr_scale=selected.server_lr_scale,
                verbose=verbose,
            )
            record.update({
                "selected_candidate_id": selected.candidate_id,
                "requested_candidate_id": gate_info.get("requested_candidate_id"),
                "gate_status": gate_info["gate_status"],
                "candidate_score": selected.score,
                "candidate_source": selected.source,
                "candidate_budget": candidate_budget,
                "weight_l1_from_previous": gate_info.get("weight_l1_from_previous", 0.0),
                "selection_epsilon": selection_epsilon,
                "llm_score_tolerance": llm_score_tolerance,
                "weight_l1_change_limit": weight_l1_change_limit,
                "coherence_diagnostics": diagnostics,
                "client_summaries": summaries,
                "llm_reasoning": llm_decision.get("reasoning", ""),
                "llm_risk": llm_decision.get("risk", ""),
                "llm_validation_preview_decision": decision_record,
            })
            previous_global_delta = self._state_delta(
                copy.deepcopy(self.global_model.state_dict()),
                current_state,
            )

        results = self._finalize_training(
            num_rounds=num_rounds,
            mode_name="llm_validation_preview_generative",
            verbose=verbose,
        )
        results["llm_decisions"] = llm_decisions
        return results

    def run_training_with_mas_validation_guided_adaptation(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
        candidate_budget: int = 30,
        weight_grid_step: float = 0.05,
        min_client_weight: float = 0.05,
        max_client_weight: float = 0.80,
        selection_epsilon: float = 0.002,
        llm_score_tolerance: Optional[float] = None,
        weight_l1_change_limit: float = 0.40,
        large_improvement_threshold: float = 0.01,
        score_profile: str = "mape_primary",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if self.llm_planner is None:
            raise ValueError("LLM Planner not initialized")

        base_lr = base_lr if base_lr is not None else self.default_lr
        base_local_epochs = base_local_epochs if base_local_epochs is not None else self.default_local_epochs
        llm_score_tolerance = selection_epsilon if llm_score_tolerance is None else float(llm_score_tolerance)
        llm_decisions = []

        if verbose:
            print("=" * 60)
            print("Starting MAS Validation-Guided FedYogi-TR Training")
            print("=" * 60)
            print(f"  Rounds: {num_rounds}")
            print(f"  Candidate budget: {candidate_budget}")
            print(f"  Weight step: {weight_grid_step}")
            print(f"  Client weight bounds: [{min_client_weight}, {max_client_weight}]")
            print(f"  LLM score tolerance: {llm_score_tolerance}")

        for round_idx in range(num_rounds):
            if verbose:
                print(f"\n[Round {round_idx + 1}] MAS validation-guided candidate search")

            current_state = copy.deepcopy(self.global_model.state_dict())
            client_states, client_metrics = self._train_clients_for_round(
                global_state=current_state,
                lr=base_lr,
                local_epochs=base_local_epochs,
                verbose=verbose,
            )
            candidates = self.build_continuous_candidate_preview(
                client_states=client_states,
                client_metrics=client_metrics,
                current_state=current_state,
                candidate_budget=candidate_budget,
                weight_grid_step=weight_grid_step,
                min_client_weight=min_client_weight,
                max_client_weight=max_client_weight,
                server_lr_scales=[0.5, 1.0, 1.5],
                epoch_deltas=[0],
                score_profile=score_profile,
            )
            candidate_preview = self._candidate_preview_for_history(candidates)
            llm_decision = self.llm_planner.choose_candidate(
                history_round_metrics=self.history_round_metrics,
                current_round=round_idx,
                num_rounds=num_rounds,
                candidate_preview=candidate_preview,
                score_tolerance=llm_score_tolerance,
            )
            requested_candidate_id = llm_decision.get("selected_candidate_id")
            selected, gate_info = select_candidate_by_gate(
                candidates=candidates,
                conservative_candidate_id="size_anchor",
                requested_candidate_id=requested_candidate_id,
                epsilon=selection_epsilon,
                score_tolerance=llm_score_tolerance,
                previous_weights=self.previous_accepted_weights,
                weight_l1_limit=weight_l1_change_limit,
                large_improvement_threshold=large_improvement_threshold,
            )
            self.previous_accepted_weights = copy.deepcopy(selected.weights)

            decision_record = copy.deepcopy(llm_decision)
            decision_record.update({
                "round": round_idx + 1,
                "requested_candidate_id": requested_candidate_id,
                "selected_candidate_id": selected.candidate_id,
                "gate_status": gate_info["gate_status"],
                "candidate_score": selected.score,
                "candidate_source": selected.source,
                "note": "LLM requests a validation-preview candidate; deterministic gate selects the executed candidate.",
            })
            llm_decisions.append(decision_record)

            if verbose:
                print(
                    f"    LLM requested: {requested_candidate_id}, "
                    f"selected: {selected.candidate_id}, "
                    f"score={selected.score:.6f}, gate={gate_info['gate_status']}"
                )

            record = self._apply_aggregated_round(
                round_idx=round_idx,
                strategy_name="mas_validation_guided_candidate",
                lr=base_lr,
                local_epochs=base_local_epochs + selected.epoch_delta,
                client_states=client_states,
                client_metrics=client_metrics,
                weights=selected.weights,
                strategy_params_override={
                    "candidate_id": selected.candidate_id,
                    "candidate_source": selected.source,
                    "score_profile": score_profile,
                },
                llm_decision=decision_record,
                candidate_weight_preview={
                    candidate.candidate_id: candidate.weights
                    for candidate in candidates
                },
                candidate_validation_preview=candidate_preview,
                server_lr_scale=selected.server_lr_scale,
                verbose=verbose,
            )
            record["selected_candidate_id"] = selected.candidate_id
            record["requested_candidate_id"] = gate_info.get("requested_candidate_id")
            record["gate_status"] = gate_info["gate_status"]
            record["candidate_score"] = selected.score
            record["candidate_source"] = selected.source
            record["candidate_budget"] = candidate_budget
            record["weight_l1_from_previous"] = gate_info.get("weight_l1_from_previous", 0.0)
            record["selection_epsilon"] = selection_epsilon
            record["llm_score_tolerance"] = llm_score_tolerance
            record["weight_l1_change_limit"] = weight_l1_change_limit
            record["llm_candidate_decision"] = decision_record

        results = self._finalize_training(
            num_rounds=num_rounds,
            mode_name="mas_validation_guided",
            verbose=verbose,
        )
        results["llm_decisions"] = llm_decisions
        return results

    def get_training_history_df(self):
        import pandas as pd

        records = []
        for record in self.history_round_metrics:
            row = {
                "round": record["round"] + 1,
                "strategy": record["strategy_name"],
                "lr": record["lr"],
                "local_epochs": record["local_epochs"],
                "global_val_mape": record["global_val"]["mape"],
                "global_val_rmse": record["global_val"]["rmse"],
                "global_val_mae": record["global_val"]["mae"],
                "global_val_mpe": record["global_val"].get("mpe", 0),
                "global_val_nrmse": record["global_val"].get("nrmse", 0),
                "global_val_r2": record["global_val"].get("r2", 0),
                "server_optimizer": record.get("server_optimizer", "fedavg"),
                "server_lr_scale": record.get("server_update", {}).get("server_lr_scale", 1.0),
                "effective_server_lr": record.get("server_update", {}).get("effective_server_lr", 1.0),
                "max_coordinate_step_ratio": record.get("server_update", {}).get("max_coordinate_step_ratio"),
                "aggregation_delta_norm": record.get("server_update", {}).get("aggregation_delta_norm", 0.0),
                "server_update_norm": record.get("server_update", {}).get("update_norm", 0.0),
                "server_update_clipped": record.get("server_update", {}).get("update_clipped", False),
                "coordinate_step_clipped": record.get("server_update", {}).get("coordinate_step_clipped", False),
                "coordinate_direction_rejected": record.get("server_update", {}).get("coordinate_direction_rejected", False),
                "llm_control_action": (record.get("llm_decision") or {}).get("control_action"),
                "llm_anchor_l1_limit": (record.get("llm_decision") or {}).get("anchor_l1_limit"),
                "selected_candidate_id": record.get("selected_candidate_id"),
                "requested_candidate_id": record.get("requested_candidate_id"),
                "gate_status": record.get("gate_status"),
                "candidate_score": record.get("candidate_score"),
                "candidate_source": record.get("candidate_source"),
                "candidate_budget": record.get("candidate_budget"),
                "weight_l1_from_previous": record.get("weight_l1_from_previous"),
                "selection_epsilon": record.get("selection_epsilon"),
                "llm_score_tolerance": record.get("llm_score_tolerance"),
                "weight_l1_change_limit": record.get("weight_l1_change_limit"),
                "validation_source": record.get("validation_source"),
                "strict_checkpoint_policy": record.get("strict_checkpoint_policy"),
            }
            for cid, metrics in record["client_metrics"].items():
                row[f"{cid}_val_mape"] = metrics["val_mape"]
                row[f"{cid}_n_samples"] = metrics["n_samples"]
                row[f"{cid}_train_loss"] = metrics["train_loss"]
                row[f"{cid}_update_norm"] = metrics.get("update_norm", 0)
                row[f"{cid}_update_delta_norm"] = metrics.get("update_delta_norm", 0)
                row[f"{cid}_validation_gap"] = metrics.get("validation_gap", 0)
            if "aggregation_weights" in record:
                for cid, weight in record["aggregation_weights"].items():
                    row[f"{cid}_weight"] = weight
            records.append(row)

        return pd.DataFrame(records)

    def get_client_metrics_df(self):
        import pandas as pd

        records = []
        for record in self.history_round_metrics:
            for cid, metrics in record["client_metrics"].items():
                row = {
                    "round": record["round"] + 1,
                    "strategy": record["strategy_name"],
                    "client_id": cid,
                    "n_samples": metrics["n_samples"],
                    "train_loss": metrics["train_loss"],
                    "val_mape": metrics["val_mape"],
                    "val_rmse": metrics.get("val_rmse", 0),
                    "val_mae": metrics.get("val_mae", 0),
                    "val_mpe": metrics.get("val_mpe", 0),
                    "update_norm": metrics.get("update_norm", 0),
                    "update_delta_norm": metrics.get("update_delta_norm", 0),
                    "validation_gap": metrics.get("validation_gap", 0),
                }
                if "aggregation_weights" in record:
                    row["aggregation_weight"] = record["aggregation_weights"].get(cid, 0)
                records.append(row)

        return pd.DataFrame(records)

    def save_training_logs(self, output_dir: str = "results/logs", prefix: str = "scene_C"):
        import json
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        round_metrics_df = self.get_training_history_df()
        round_metrics_path = output_path / f"{prefix}_round_metrics.csv"
        round_metrics_df.to_csv(round_metrics_path, index=False)
        print(f"  Saved: {round_metrics_path}")

        client_metrics_df = self.get_client_metrics_df()
        client_metrics_path = output_path / f"{prefix}_client_metrics.csv"
        client_metrics_df.to_csv(client_metrics_path, index=False)
        print(f"  Saved: {client_metrics_path}")

        history_json = {
            "training_config": {
                "default_strategy": self.default_strategy,
                "default_lr": self.default_lr,
                "default_local_epochs": self.default_local_epochs,
                "strategies_config": self.strategies_config,
                "server_optimizer": getattr(self.server_optimizer, "name", "fedavg"),
                "max_coordinate_step_ratio": getattr(self.server_optimizer, "max_coordinate_step_ratio", None),
            },
            "training_summary": {
                "total_rounds": len(self.history_round_metrics),
                "best_round": self.best_round + 1,
                "best_val_mape": self.best_val_mape,
            },
            "round_history": self.history_round_metrics,
        }

        history_json_path = output_path / f"{prefix}_training_history.json"
        with open(history_json_path, 'w', encoding='utf-8') as f:
            json.dump(history_json, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved: {history_json_path}")

        return {
            "round_metrics": str(round_metrics_path),
            "client_metrics": str(client_metrics_path),
            "training_history": str(history_json_path),
        }

    def get_available_strategies(self) -> List[str]:
        return list(self.strategies_config.keys())

    def get_strategy_description(self, strategy_name: str) -> str:
        if strategy_name in self.strategies_config:
            return self.strategies_config[strategy_name].get('description', 'No description')
        return f"Unknown strategy: {strategy_name}"
