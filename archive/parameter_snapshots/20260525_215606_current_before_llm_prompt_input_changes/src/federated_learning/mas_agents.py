"""
MAS-FL agents for Scenario C (Multi-Agent Federated Learning).

This module provides:
- LocalAgent: client-side training logic
- CentralAgent: aggregation, evaluation, and optional LLM-guided strategy selection
"""

import copy
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models import CostEstimationMLP
from src.utils import (
    compute_mape, compute_rmse, compute_mae,
    compute_mpe, compute_nrmse, compute_r2, get_device
)


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
        llm_planner=None
    ):
        self.global_model = global_model
        self.client_agents = client_agents
        self.global_val_loader = global_val_loader
        self.global_test_loader = global_test_loader
        self.preprocessor = preprocessor
        self.config = config
        self.device = device if device else get_device("auto")
        self.llm_planner = llm_planner

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
        client_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        strategy_params = self.strategies_config.get(strategy_name, {})
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

    def aggregate_with_weights(
        self,
        client_states: Dict[str, Dict[str, torch.Tensor]],
        weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        aggregated_state = {}
        first_client_id = list(client_states.keys())[0]
        param_keys = client_states[first_client_id].keys()

        for key in param_keys:
            aggregated_param = None
            for cid, state in client_states.items():
                weighted_param = weights[cid] * state[key].float()
                aggregated_param = weighted_param if aggregated_param is None else aggregated_param + weighted_param
            aggregated_state[key] = aggregated_param

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

        # 计算百分比偏差 (MPE)
        mpe_ratio = val_metrics["mpe"]  # 已经是 mean((y_pred - y_true) / y_true)
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
        verbose: bool = True
    ) -> Dict[str, Any]:
        lr = lr if lr is not None else self.default_lr
        local_epochs = local_epochs if local_epochs is not None else self.default_local_epochs

        if verbose:
            print(f"\n[Round {round_idx + 1}] Strategy: {strategy_name}, LR={lr}, Epochs={local_epochs}")

        global_state = copy.deepcopy(self.global_model.state_dict())

        client_states = {}
        client_metrics = {}
        for client_id, agent in self.client_agents.items():
            updated_state, metrics = agent.train_one_round(
                global_model_state=global_state,
                lr=lr,
                local_epochs=local_epochs
            )
            client_states[client_id] = updated_state
            client_metrics[client_id] = metrics

            if verbose:
                print(
                    f"    {client_id}: {metrics['n_samples']} samples, "
                    f"train_loss={metrics['train_loss']:.6f}, "
                    f"val_mape={metrics['val_mape']*100:.2f}%"
                )

        weights = self.compute_client_weights(strategy_name, client_metrics)

        if verbose:
            weights_str = ", ".join([f"{cid}={w:.3f}" for cid, w in weights.items()])
            print(f"    Weights: {weights_str}")

        aggregated_state = self.aggregate_with_weights(client_states, weights)
        self.global_model.load_state_dict(aggregated_state)

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
            "client_metrics": {
                cid: {
                    "n_samples": m["n_samples"],
                    "train_loss": m["train_loss"],
                    "val_mape": m["val_mape"],
                    "val_rmse": m["val_rmse"],
                    "val_mae": m["val_mae"],
                    "val_mpe": m["val_mpe"],
                }
                for cid, m in client_metrics.items()
            },
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

    def run_training(
        self,
        num_rounds: int,
        strategy_name: Optional[str] = None,
        lr: Optional[float] = None,
        local_epochs: Optional[int] = None,
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

    def run_training_with_llm(
        self,
        num_rounds: int,
        base_lr: Optional[float] = None,
        base_local_epochs: Optional[int] = None,
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

        for round_idx in range(num_rounds):
            should_call_llm = (
                round_idx < 4
                or round_idx % self.llm_call_every_n_rounds == 0
                or self.llm_planner.last_llm_decision is None
            )

            reused_decision = False
            if should_call_llm:
                decision = self.llm_planner.choose_strategy(
                    history_round_metrics=self.history_round_metrics,
                    current_round=round_idx,
                    num_rounds=num_rounds
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
            llm_decisions.append(decision_record)

            strategy_name = decision["chosen_strategy_name"]
            lr = base_lr * decision.get("lr_scale", 1.0)
            local_epochs = max(5, min(30, base_local_epochs + decision.get("epoch_delta", 0)))

            if verbose:
                decision_mode = "reused" if reused_decision else "new"
                print(
                    f"\n  [LLM Decision] Round {round_idx + 1}: "
                    f"strategy={strategy_name}, lr={lr:.6f}, epochs={local_epochs}, mode={decision_mode}"
                )

            self.run_round(
                round_idx=round_idx,
                strategy_name=strategy_name,
                lr=lr,
                local_epochs=local_epochs,
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
            }
            for cid, metrics in record["client_metrics"].items():
                row[f"{cid}_val_mape"] = metrics["val_mape"]
                row[f"{cid}_n_samples"] = metrics["n_samples"]
                row[f"{cid}_train_loss"] = metrics["train_loss"]
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
