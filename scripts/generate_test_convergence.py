"""
Generate test-set convergence curves for Scenario B and Scenario C.

This is an analysis-only runner. It evaluates the test set after each
federated aggregation round, but does not expose test metrics to the
training loop, checkpoint selection, or LLM decision process.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.scenario_C_llm import create_llm_planner
from src.data_preprocessing import (
    load_federated_datasets,
    load_federated_datasets_for_scene_c,
)
from src.federated_learning.mas_agents import CentralAgent, LocalAgent
from src.models import CostEstimationMLP
from src.utils import get_device, load_config, set_seed


class TestTrackingCentralAgent(CentralAgent):
    """CentralAgent variant that records test metrics after each round."""

    def __init__(self, *args, scenario_label: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenario_label = scenario_label
        self.test_history: List[Dict[str, Any]] = []

    def _apply_aggregated_round(self, *args, **kwargs) -> Dict[str, Any]:
        round_record = super()._apply_aggregated_round(*args, **kwargs)
        test_metrics = self.evaluate_global(data_loader=self.global_test_loader)
        round_record["global_test"] = {
            "mape": test_metrics["mape"],
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "mpe": test_metrics["mpe"],
            "nrmse": test_metrics["nrmse"],
            "r2": test_metrics["r2"],
        }
        self.test_history.append(
            {
                "scenario": self.scenario_label,
                "round": round_record["round"] + 1,
                "strategy": round_record["strategy_name"],
                "lr": round_record["lr"],
                "local_epochs": round_record["local_epochs"],
                "test_mape": test_metrics["mape"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_mpe": test_metrics["mpe"],
                "test_nrmse": test_metrics["nrmse"],
                "test_r2": test_metrics["r2"],
                "val_mape": round_record["global_val"]["mape"],
                "val_rmse": round_record["global_val"]["rmse"],
                "val_mae": round_record["global_val"]["mae"],
                "val_r2": round_record["global_val"]["r2"],
            }
        )
        return round_record


def create_model(input_dim: int, device: torch.device) -> CostEstimationMLP:
    return CostEstimationMLP(
        input_dim=input_dim,
        hidden_dims=[128, 128, 64, 32],
        output_dim=1,
        activation="gelu",
        dropout=0.1,
    ).to(device)


def create_agents(
    client_train_sets,
    client_val_sets,
    config: dict,
    device: torch.device,
    input_dim: int,
    preprocessor,
    fedprox_mu=None,
) -> Dict[str, LocalAgent]:
    return {
        client_id: LocalAgent(
            client_id=client_id,
            train_dataset=client_train_sets[client_id],
            val_dataset=client_val_sets[client_id],
            config=config,
            device=device,
            input_dim=input_dim,
            fedprox_mu=fedprox_mu,
            preprocessor=preprocessor,
        )
        for client_id in client_train_sets.keys()
    }


def make_central_agent(
    scenario_label: str,
    config: dict,
    client_train_sets,
    client_val_sets,
    global_val_set,
    global_test_set,
    preprocessor,
    device: torch.device,
    input_dim: int,
    batch_size: int,
    fedprox_mu=None,
    llm_planner=None,
    agent_cls=TestTrackingCentralAgent,
    **agent_kwargs,
) -> TestTrackingCentralAgent:
    global_model = create_model(input_dim=input_dim, device=device)
    client_agents = create_agents(
        client_train_sets=client_train_sets,
        client_val_sets=client_val_sets,
        config=config,
        device=device,
        input_dim=input_dim,
        preprocessor=preprocessor,
        fedprox_mu=fedprox_mu,
    )
    return agent_cls(
        global_model=global_model,
        client_agents=client_agents,
        global_val_loader=DataLoader(global_val_set, batch_size=batch_size, shuffle=False),
        global_test_loader=DataLoader(global_test_set, batch_size=batch_size, shuffle=False),
        preprocessor=preprocessor,
        config=config,
        device=device,
        llm_planner=llm_planner,
        scenario_label=scenario_label,
        **agent_kwargs,
    )


def run_scenario_b(config: dict, seed: int, num_rounds: int, verbose: bool):
    scene_b_cfg = config.get("scene_b", {})
    set_seed(seed)
    device = get_device(config["compute"]["device"])

    datasets = load_federated_datasets(config, config_key="scene_b")
    client_train_sets, client_val_sets, global_val_set, global_test_set, preprocessor = datasets

    input_dim = len(scene_b_cfg.get("data", {}).get("feature_columns", []))
    batch_size = scene_b_cfg.get("batch_size", 32)
    lr = scene_b_cfg.get("learning_rate", 0.0005)
    local_epochs = scene_b_cfg.get("local_epochs", 20)
    strategy = scene_b_cfg.get("strategy", "size_only")

    agent = make_central_agent(
        scenario_label="B_FedAvg",
        config=config,
        client_train_sets=client_train_sets,
        client_val_sets=client_val_sets,
        global_val_set=global_val_set,
        global_test_set=global_test_set,
        preprocessor=preprocessor,
        device=device,
        input_dim=input_dim,
        batch_size=batch_size,
        fedprox_mu=scene_b_cfg.get("fedprox_mu", 0.0),
    )
    result = agent.run_training(
        num_rounds=num_rounds,
        strategy_name=strategy,
        lr=lr,
        local_epochs=local_epochs,
        verbose=verbose,
    )
    return agent.test_history, result


def run_scenario_c(config: dict, seed: int, num_rounds: int, verbose: bool, temperature=None):
    scene_c_cfg = config.get("scene_c", {})
    if temperature is not None:
        config = copy.deepcopy(config)
        config.setdefault("scene_c", {}).setdefault("llm", {})["temperature"] = temperature

    set_seed(seed)
    device = get_device(config["compute"]["device"])

    datasets = load_federated_datasets_for_scene_c(config)
    client_train_sets, client_val_sets, global_val_set, global_test_set, preprocessor = datasets

    input_dim = len(scene_c_cfg.get("data", {}).get("feature_columns", []))
    batch_size = scene_c_cfg.get("batch_size", 32)
    base_lr = scene_c_cfg.get("learning_rate", 0.0005)
    base_local_epochs = scene_c_cfg.get("local_epochs", 20)

    log_dir = Path("results/logs/test_convergence_llm")
    llm_planner = create_llm_planner(config, str(log_dir))

    agent = make_central_agent(
        scenario_label="C_MAS_FL_LLM",
        config=config,
        client_train_sets=client_train_sets,
        client_val_sets=client_val_sets,
        global_val_set=global_val_set,
        global_test_set=global_test_set,
        preprocessor=preprocessor,
        device=device,
        input_dim=input_dim,
        batch_size=batch_size,
        llm_planner=llm_planner,
    )
    result = agent.run_training_with_llm(
        num_rounds=num_rounds,
        base_lr=base_lr,
        base_local_epochs=base_local_epochs,
        verbose=verbose,
    )
    return agent.test_history, result


def save_figure(df: pd.DataFrame, output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    colors = {"B_FedAvg": "#2ca02c", "C_MAS_FL_LLM": "#d62728"}
    labels = {"B_FedAvg": "B (FedAvg)", "C_MAS_FL_LLM": "C (MAS-FL-LLM)"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Test-Set Convergence: B (FedAvg) vs C (MAS-FL-LLM)", fontweight="bold")

    for scenario in ["B_FedAvg", "C_MAS_FL_LLM"]:
        sdf = df[df["scenario"] == scenario]
        axes[0].plot(
            sdf["round"],
            sdf["test_mape"] * 100,
            marker="o" if scenario == "B_FedAvg" else "s",
            linewidth=2,
            markersize=4,
            color=colors[scenario],
            label=labels[scenario],
        )
        axes[1].plot(
            sdf["round"],
            sdf["test_rmse"] / 1e6,
            marker="o" if scenario == "B_FedAvg" else "s",
            linewidth=2,
            markersize=4,
            color=colors[scenario],
            label=labels[scenario],
        )

    axes[0].set_xlabel("Federated Round")
    axes[0].set_ylabel("Test MAPE (%)")
    axes[0].set_title("(a) Test MAPE")
    axes[0].legend()

    axes[1].set_xlabel("Federated Round")
    axes[1].set_ylabel("Test RMSE (M$)")
    axes[1].set_title("(b) Test RMSE")
    axes[1].legend()

    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate B/C test-set convergence comparison.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("configs/config.yaml")

    print(f"[Run] B test convergence: seed={args.seed}, rounds={args.num_rounds}")
    b_history, b_result = run_scenario_b(config, args.seed, args.num_rounds, args.verbose)

    print(f"[Run] C test convergence: seed={args.seed}, rounds={args.num_rounds}")
    c_history, c_result = run_scenario_c(
        config=config,
        seed=args.seed,
        num_rounds=args.num_rounds,
        verbose=args.verbose,
        temperature=args.temperature,
    )

    df = pd.DataFrame(b_history + c_history)
    csv_path = Path(f"results/test_convergence_b_c_seed{args.seed}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    fig_stem = Path("results/figures/fig10_test_convergence_comparison")
    save_figure(df, fig_stem)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {fig_stem}.png")
    print(f"[Saved] {fig_stem}.pdf")
    print(
        "[Final] "
        f"B test MAPE={b_result['test_metrics']['mape'] * 100:.2f}%, "
        f"C test MAPE={c_result['test_metrics']['mape'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
