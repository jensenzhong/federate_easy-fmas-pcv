"""
Run B/C federated learning until validation-based convergence.

This exploratory experiment uses validation MAPE for early stopping and model
selection. Test metrics are recorded for post-hoc convergence analysis only and
are not used to stop training or choose a checkpoint.
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

from experiments.scenario_C_llm import create_llm_planner
from scripts.generate_test_convergence import (
    TestTrackingCentralAgent,
    make_central_agent,
)
from src.data_preprocessing import (
    load_federated_datasets,
    load_federated_datasets_for_scene_c,
)
from src.utils import get_device, load_config, set_seed


class EarlyStopTriggered(Exception):
    def __init__(self, stop_round: int):
        super().__init__(f"Early stopping triggered at round {stop_round}")
        self.stop_round = stop_round


class EarlyStoppingTestAgent(TestTrackingCentralAgent):
    def __init__(self, *args, patience: int, min_delta: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self.early_best_val_mape = float("inf")
        self.early_wait = 0

    def _apply_aggregated_round(self, *args, **kwargs) -> Dict[str, Any]:
        round_record = super()._apply_aggregated_round(*args, **kwargs)
        current_val = round_record["global_val"]["mape"]
        if current_val < self.early_best_val_mape - self.min_delta:
            self.early_best_val_mape = current_val
            self.early_wait = 0
        else:
            self.early_wait += 1

        if self.early_wait >= self.patience:
            raise EarlyStopTriggered(round_record["round"] + 1)
        return round_record


def finalize_agent(agent: EarlyStoppingTestAgent, stopped_round: int = None) -> Dict[str, Any]:
    if agent.best_model_state is not None:
        agent.global_model.load_state_dict(agent.best_model_state)
    test_metrics = agent.evaluate_global(data_loader=agent.global_test_loader)
    return {
        "scenario": agent.scenario_label,
        "best_round": agent.best_round + 1,
        "stopped_round": stopped_round or len(agent.history_round_metrics),
        "best_val_mape": agent.best_val_mape,
        "test_mape": test_metrics["mape"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_mpe": test_metrics["mpe"],
        "test_r2": test_metrics["r2"],
    }


def run_b(config: dict, seed: int, max_rounds: int, patience: int, min_delta: float):
    scene_b_cfg = config.get("scene_b", {})
    set_seed(seed)
    device = get_device(config["compute"]["device"])
    datasets = load_federated_datasets(config, config_key="scene_b")
    client_train_sets, client_val_sets, global_val_set, global_test_set, preprocessor = datasets

    agent = make_central_agent(
        scenario_label="B_FedAvg",
        config=config,
        client_train_sets=client_train_sets,
        client_val_sets=client_val_sets,
        global_val_set=global_val_set,
        global_test_set=global_test_set,
        preprocessor=preprocessor,
        device=device,
        input_dim=len(scene_b_cfg.get("data", {}).get("feature_columns", [])),
        batch_size=scene_b_cfg.get("batch_size", 32),
        fedprox_mu=scene_b_cfg.get("fedprox_mu", 0.0),
        agent_cls=EarlyStoppingTestAgent,
        patience=patience,
        min_delta=min_delta,
    )

    stopped_round = None
    try:
        agent.run_training(
            num_rounds=max_rounds,
            strategy_name=scene_b_cfg.get("strategy", "size_only"),
            lr=scene_b_cfg.get("learning_rate", 0.0005),
            local_epochs=scene_b_cfg.get("local_epochs", 20),
            verbose=False,
        )
    except EarlyStopTriggered as exc:
        stopped_round = exc.stop_round

    return agent.test_history, finalize_agent(agent, stopped_round)


def run_c(
    config: dict,
    seed: int,
    max_rounds: int,
    patience: int,
    min_delta: float,
    temperature: float,
):
    config = copy.deepcopy(config)
    config.setdefault("scene_c", {}).setdefault("llm", {})["temperature"] = temperature

    scene_c_cfg = config.get("scene_c", {})
    set_seed(seed)
    device = get_device(config["compute"]["device"])
    datasets = load_federated_datasets_for_scene_c(config)
    client_train_sets, client_val_sets, global_val_set, global_test_set, preprocessor = datasets

    llm_log_dir = Path("results/logs/convergence_to_best_llm") / f"seed{seed}"
    llm_planner = create_llm_planner(config, str(llm_log_dir))

    agent = make_central_agent(
        scenario_label="C_MAS_FL_LLM",
        config=config,
        client_train_sets=client_train_sets,
        client_val_sets=client_val_sets,
        global_val_set=global_val_set,
        global_test_set=global_test_set,
        preprocessor=preprocessor,
        device=device,
        input_dim=len(scene_c_cfg.get("data", {}).get("feature_columns", [])),
        batch_size=scene_c_cfg.get("batch_size", 32),
        llm_planner=llm_planner,
        agent_cls=EarlyStoppingTestAgent,
        patience=patience,
        min_delta=min_delta,
    )

    stopped_round = None
    try:
        agent.run_training_with_llm(
            num_rounds=max_rounds,
            base_lr=scene_c_cfg.get("learning_rate", 0.0005),
            base_local_epochs=scene_c_cfg.get("local_epochs", 20),
            verbose=False,
        )
    except EarlyStopTriggered as exc:
        stopped_round = exc.stop_round

    return agent.test_history, finalize_agent(agent, stopped_round)


def plot_summary(history_df: pd.DataFrame, output_stem: Path) -> None:
    colors = {"B_FedAvg": "#2ca02c", "C_MAS_FL_LLM": "#d62728"}
    labels = {"B_FedAvg": "B (FedAvg)", "C_MAS_FL_LLM": "C (MAS-FL-LLM)"}
    fig, ax = plt.subplots(figsize=(9, 5))

    for (scenario, seed), group in history_df.groupby(["scenario", "seed"]):
        ax.plot(
            group["round"],
            group["test_mape"] * 100,
            color=colors[scenario],
            alpha=0.45,
            linewidth=1.5,
            label=f"{labels[scenario]} seed={seed}",
        )

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Test MAPE Under Validation-Based Early Stopping")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run convergence-to-best-validation experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20, 42])
    parser.add_argument("--max_rounds", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("configs/config.yaml")

    all_history: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for seed in args.seeds:
        print(f"[Seed {seed}] B convergence run")
        b_history, b_summary = run_b(config, seed, args.max_rounds, args.patience, args.min_delta)
        for row in b_history:
            row["seed"] = seed
        b_summary["seed"] = seed
        all_history.extend(b_history)
        summaries.append(b_summary)
        print(
            f"  B: best_round={b_summary['best_round']}, "
            f"stop={b_summary['stopped_round']}, "
            f"test_mape={b_summary['test_mape'] * 100:.2f}%"
        )

        print(f"[Seed {seed}] C convergence run, temperature={args.temperature}")
        c_history, c_summary = run_c(
            config,
            seed,
            args.max_rounds,
            args.patience,
            args.min_delta,
            args.temperature,
        )
        for row in c_history:
            row["seed"] = seed
        c_summary["seed"] = seed
        all_history.extend(c_history)
        summaries.append(c_summary)
        print(
            f"  C: best_round={c_summary['best_round']}, "
            f"stop={c_summary['stopped_round']}, "
            f"test_mape={c_summary['test_mape'] * 100:.2f}%"
        )

    history_df = pd.DataFrame(all_history)
    summary_df = pd.DataFrame(summaries)

    history_path = Path("results/convergence_to_best_history.csv")
    summary_path = Path("results/convergence_to_best_summary.csv")
    history_df.to_csv(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_summary(history_df, Path("results/figures/fig11_convergence_to_best"))

    print(f"[Saved] {history_path}")
    print(f"[Saved] {summary_path}")
    print("[Summary]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
