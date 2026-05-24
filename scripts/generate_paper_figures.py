"""
论文级可视化脚本 (Paper-Quality Figures)

生成9张论文所需的图表，所有图表使用英文标签、300 DPI、同时输出PNG+PDF。

图表列表:
  Fig.1 - 四场景(A/A'/B/C)性能指标对比条形图
  Fig.2 - B与C收敛曲线对比(MAPE vs Round)
  Fig.3 - LLM策略选择时间序列图
  Fig.4 - 三客户端val_mape随轮次变化
  Fig.5 - 预测vs真实散点图（多场景）
  Fig.6 - 消融实验对比图
  Fig.7 - 多种子结果箱线图(带显著性标记)
  Fig.8 - 分层评估对比图（按项目规模）
  Fig.9 - 偏差校正前后对比图

用法:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --fig 1 3 5   # 仅生成指定图
"""

import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 论文级样式设置
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "A": "#1f77b4",
    "A_prime": "#ff7f0e",
    "B": "#2ca02c",
    "C": "#d62728",
    "C_corrected": "#9467bd",
}

OUTPUT_DIR = Path("results/figures")


def save_fig(fig, name: str):
    """保存图表为PNG和PDF"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png + .pdf")


# ============================================================
# Fig.1: 四场景性能指标对比条形图
# ============================================================
def fig1_scenario_comparison():
    """四场景(A/A'/B/C)性能指标对比"""
    print("\n[Fig.1] Scenario Performance Comparison")

    base = Path("results")
    scenarios = {}

    # 加载各场景结果CSV
    files = {
        "A (GBR)": "centralized_results.csv",
        "A' (NN)": "centralized_nn_results.csv",
        "B (FedAvg)": "fedavg_results.csv",
        "C (MAS-FL)": "scenario_c_results.csv",
    }

    for name, filename in files.items():
        path = base / filename
        if path.exists():
            df = pd.read_csv(path)
            scenarios[name] = {
                "MAPE": df["test_mape"].values[0] * 100,
                "RMSE": df["test_rmse"].values[0] / 1000,  # 转为k$
                "MAE": df["test_mae"].values[0] / 1000,
                "MPE": df["test_mpe"].values[0] * 100,
            }

    if len(scenarios) < 2:
        print("  [SKIP] Not enough scenario results found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Performance Comparison Across Scenarios", fontsize=14, fontweight="bold")

    names = list(scenarios.keys())
    x = np.arange(len(names))
    colors = [COLORS.get(k.split()[0], "#333333") for k in names]

    # (a) MAPE
    ax = axes[0, 0]
    values = [scenarios[n]["MAPE"] for n in names]
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("(a) MAPE Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    # (b) RMSE
    ax = axes[0, 1]
    values = [scenarios[n]["RMSE"] for n in names]
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("RMSE (k$)")
    ax.set_title("(b) RMSE Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{val:.0f}k", ha="center", va="bottom", fontsize=8)

    # (c) MAE
    ax = axes[1, 0]
    values = [scenarios[n]["MAE"] for n in names]
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor="black")
    ax.set_ylabel("MAE (k$)")
    ax.set_title("(c) MAE Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{val:.0f}k", ha="center", va="bottom", fontsize=8)

    # (d) MPE (Bias)
    ax = axes[1, 1]
    values = [scenarios[n]["MPE"] for n in names]
    bar_colors = ["red" if v < 0 else "blue" for v in values]
    bars = ax.bar(x, values, color=bar_colors, alpha=0.7, edgecolor="black")
    ax.axhline(y=0, color="black", linewidth=1.5)
    ax.axhspan(-10, 10, alpha=0.1, color="green")
    ax.set_ylabel("MPE (%)")
    ax.set_title("(d) Prediction Bias (MPE)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    for bar, val in zip(bars, values):
        offset = 1 if val >= 0 else -2
        ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "fig1_scenario_comparison")


# ============================================================
# Fig.2: B与C收敛曲线对比
# ============================================================
def fig2_convergence_comparison():
    """B与C收敛曲线对比(MAPE vs Round)"""
    print("\n[Fig.2] Convergence Comparison (B vs C)")

    base = Path("results/logs")

    b_path = base / "scene_B_round_metrics.csv"
    c_path = base / "scene_C_round_metrics.csv"

    if not b_path.exists() or not c_path.exists():
        print("  [SKIP] Missing round_metrics.csv for B or C")
        return

    df_b = pd.read_csv(b_path)
    df_c = pd.read_csv(c_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Federated Learning Convergence: B (FedAvg) vs C (MAS-FL-LLM)",
                 fontsize=13, fontweight="bold")

    # (a) MAPE convergence
    ax1.plot(df_b["round"], df_b["global_val_mape"] * 100, "o-",
             color=COLORS["B"], linewidth=2, markersize=4, label="B (FedAvg)")
    ax1.plot(df_c["round"], df_c["global_val_mape"] * 100, "s-",
             color=COLORS["C"], linewidth=2, markersize=4, label="C (MAS-FL-LLM)")
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Global Validation MAPE (%)")
    ax1.set_title("(a) MAPE Convergence")
    ax1.legend()

    # (b) RMSE convergence
    ax2.plot(df_b["round"], df_b["global_val_rmse"] / 1e6, "o-",
             color=COLORS["B"], linewidth=2, markersize=4, label="B (FedAvg)")
    ax2.plot(df_c["round"], df_c["global_val_rmse"] / 1e6, "s-",
             color=COLORS["C"], linewidth=2, markersize=4, label="C (MAS-FL-LLM)")
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Global Validation RMSE (M$)")
    ax2.set_title("(b) RMSE Convergence")
    ax2.legend()

    fig.tight_layout()
    save_fig(fig, "fig2_convergence_comparison")


# ============================================================
# Fig.3: LLM策略选择时间序列图
# ============================================================
def fig3_llm_strategy_timeline():
    """LLM策略选择随轮次变化"""
    print("\n[Fig.3] LLM Strategy Selection Timeline")

    decisions_path = Path("results/logs/scene_C_llm_decisions.jsonl")
    if not decisions_path.exists():
        print("  [SKIP] No LLM decisions log found")
        return

    decisions = []
    with open(decisions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                decisions.append(json.loads(line.strip()))

    rounds = [d["round"] for d in decisions]
    strategies = [d["decision"]["chosen_strategy_name"] for d in decisions]
    lr_scales = [d["decision"].get("lr_scale", 1.0) for d in decisions]
    epoch_deltas = [d["decision"].get("epoch_delta", 0) for d in decisions]

    strategy_names = sorted(set(strategies))
    strategy_to_num = {s: i for i, s in enumerate(strategy_names)}
    strategy_nums = [strategy_to_num[s] for s in strategies]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("LLM Decision Timeline (Scenario C)", fontsize=13, fontweight="bold")

    # (a) Strategy selection
    scatter_colors = plt.cm.Set2(np.linspace(0, 1, len(strategy_names)))
    for i, sname in enumerate(strategy_names):
        mask = [s == sname for s in strategies]
        r = [rounds[j] for j in range(len(rounds)) if mask[j]]
        ax1.scatter(r, [i] * len(r), color=scatter_colors[i], s=50, label=sname, zorder=3)
    ax1.set_ylabel("Strategy")
    ax1.set_yticks(range(len(strategy_names)))
    ax1.set_yticklabels(strategy_names)
    ax1.set_title("(a) Strategy Selection per Round")
    ax1.legend(loc="upper right", fontsize=8)

    # (b) Learning rate scale
    ax2.plot(rounds, lr_scales, "o-", color="#1f77b4", markersize=4)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("LR Scale")
    ax2.set_title("(b) Learning Rate Scale Factor")

    # (c) Epoch delta
    ax3.bar(rounds, epoch_deltas, color=["green" if d >= 0 else "red" for d in epoch_deltas],
            alpha=0.7, edgecolor="black", linewidth=0.5)
    ax3.axhline(y=0, color="black", linewidth=1)
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Epoch Delta")
    ax3.set_title("(c) Local Epoch Adjustment")

    fig.tight_layout()
    save_fig(fig, "fig3_llm_strategy_timeline")


# ============================================================
# Fig.4: 三客户端val_mape随轮次变化
# ============================================================
def fig4_client_mape_trends():
    """三客户端验证MAPE随轮次变化"""
    print("\n[Fig.4] Client Validation MAPE Trends")

    client_path = Path("results/logs/scene_C_client_metrics.csv")
    if not client_path.exists():
        print("  [SKIP] No client metrics found")
        return

    df = pd.read_csv(client_path)
    clients = df["client_id"].unique()

    fig, ax = plt.subplots(figsize=(10, 5))
    client_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, client in enumerate(clients):
        client_data = df[df["client_id"] == client]
        color = client_colors[i % len(client_colors)]
        ax.plot(client_data["round"], client_data["val_mape"] * 100,
                "o-", color=color, markersize=3, linewidth=1.5, label=client)

    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Client Validation MAPE (%)")
    ax.set_title("Client-wise Validation MAPE Trends (Scenario C)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig4_client_mape_trends")


# ============================================================
# Fig.5: 预测vs真实散点图
# ============================================================
def fig5_prediction_scatter():
    """预测 vs 真实散点图"""
    print("\n[Fig.5] Prediction vs True Value Scatter")

    base = Path("results")
    pred_files = {
        "A (GBR)": "centralized_predictions.csv",
        "A' (NN)": "centralized_nn_predictions.csv",
    }

    loaded = {}
    for name, filename in pred_files.items():
        path = base / filename
        if path.exists():
            loaded[name] = pd.read_csv(path)

    if not loaded:
        print("  [SKIP] No prediction files found")
        return

    n_plots = len(loaded)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle("Predicted vs True Contract Amount", fontsize=13, fontweight="bold")

    for ax, (name, df) in zip(axes, loaded.items()):
        y_true = df["True_Value"].values / 1e6
        y_pred = df["Predicted_Value"].values / 1e6

        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="black", linewidth=0.3)

        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")

        ax.set_xlabel("True Value (M$)")
        ax.set_ylabel("Predicted Value (M$)")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    save_fig(fig, "fig5_prediction_scatter")


# ============================================================
# Fig.6: 消融实验对比图
# ============================================================
def fig6_ablation():
    """消融实验对比条形图"""
    print("\n[Fig.6] Ablation Study Results")

    ablation_path = Path("results/ablation_summary.csv")
    if not ablation_path.exists():
        print("  [SKIP] No ablation results found. Run: python scripts/run_ablation.py")
        return

    df = pd.read_csv(ablation_path)
    if df.empty or "test_mape" not in df.columns:
        print("  [SKIP] No valid ablation data")
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    label_map = {
        "B-baseline": "Fixed-weight baseline",
        "B+FedProx": "Baseline + proximal constraint",
        "C-perf_only": "Performance-only weighting",
        "C-perf\\_only": "Performance-only weighting",
        "C-hybrid": "Hybrid weighting",
        "C-LLM": "LLM-driven weighting",
        "C-LLM+bias": "LLM weighting + bias correction",
    }
    raw_names = [name.replace("\\_", "_") for name in df["name"].tolist()]
    names = [label_map.get(name, name) for name in raw_names]
    mape = df["test_mape"].to_numpy(dtype=float) * 100
    x = np.arange(len(names))

    line_color = "#D55E00"
    ax.plot(
        x,
        mape,
        color=line_color,
        linewidth=2.2,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor=line_color,
    )

    best_idx = int(np.argmin(mape))
    ax.scatter(
        x[best_idx],
        mape[best_idx],
        s=90,
        color="#0072B2",
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    y_min = float(np.floor((mape.min() - 0.15) * 10) / 10)
    y_max = float(np.ceil((mape.max() + 0.15) * 10) / 10)
    for idx, val in enumerate(mape):
        offset = 0.035 if idx != best_idx else -0.06
        va = "bottom" if idx != best_idx else "top"
        ax.text(
            x[idx],
            val + offset,
            f"{val:.2f}%",
            ha="center",
            va=va,
            fontsize=8,
        )

    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Ablation Study: Fine-Grained Comparison of Component Effects", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 0.001, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, "fig6_ablation_study")


# ============================================================
# Fig.7: 多种子结果箱线图
# ============================================================
def fig7_multi_seed_boxplot():
    """多种子实验箱线图（带显著性标记）"""
    print("\n[Fig.7] Multi-Seed Results Boxplot")

    results_path = Path("results/multi_seed/all_results.csv")
    if not results_path.exists():
        print("  [SKIP] No multi-seed results. Run: python scripts/run_multi_seed.py")
        return

    df = pd.read_csv(results_path)
    df = df[df["success"] == True]

    if "test_mape" not in df.columns:
        print("  [SKIP] No MAPE data in results")
        return

    scenarios = sorted(df["scenario"].unique())
    data = [df[df["scenario"] == s]["test_mape"].values * 100 for s in scenarios]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(data, labels=scenarios, patch_artist=True, widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
                    medianprops=dict(color="black", linewidth=2))

    colors = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Multi-Seed Experiment Results (5 seeds)", fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "fig7_multi_seed_boxplot")


# ============================================================
# Fig.8: 分层评估对比图
# ============================================================
def fig8_stratified_evaluation():
    """分层评估对比图（按项目规模）"""
    print("\n[Fig.8] Stratified Evaluation by Project Size")

    strat_path = Path("results/stratified_evaluation.csv")
    if not strat_path.exists():
        print("  [SKIP] No stratified results. Run: python scripts/stratified_evaluation.py")
        return

    df = pd.read_csv(strat_path)
    if df.empty:
        print("  [SKIP] No data")
        return

    scenarios = df["scenario"].unique()
    strata = df["stratum"].unique()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(strata))
    width = 0.8 / len(scenarios)
    colors_list = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))

    for i, scenario in enumerate(scenarios):
        sdf = df[df["scenario"] == scenario]
        mape_values = []
        for stratum in strata:
            row = sdf[sdf["stratum"] == stratum]
            mape_values.append(row["mape"].values[0] * 100 if len(row) > 0 and row["n"].values[0] > 0 else 0)

        bars = ax.bar(x + i * width, mape_values, width, label=scenario,
                      color=colors_list[i], edgecolor="black", alpha=0.8)

    ax.set_ylabel("MAPE (%)")
    ax.set_title("Stratified Performance by Project Size", fontweight="bold")
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(strata, rotation=15)
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "fig8_stratified_evaluation")


# ============================================================
# Fig.9: 偏差校正前后对比图
# ============================================================
def fig9_bias_correction():
    """偏差校正前后对比"""
    print("\n[Fig.9] Bias Correction Before vs After")

    result_path = Path("results/scenario_c_results.csv")
    if not result_path.exists():
        print("  [SKIP] No scenario C results found")
        return

    df = pd.read_csv(result_path)

    has_corrected = "test_mape_corrected" in df.columns
    if not has_corrected:
        print("  [SKIP] No bias-corrected results in CSV")
        return

    metrics = ["MAPE", "RMSE", "MAE", "MPE"]
    before = [
        df["test_mape"].values[0] * 100,
        df["test_rmse"].values[0] / 1000,
        df["test_mae"].values[0] / 1000,
        df["test_mpe"].values[0] * 100,
    ]
    after = [
        df["test_mape_corrected"].values[0] * 100,
        df["test_rmse_corrected"].values[0] / 1000,
        df["test_mae_corrected"].values[0] / 1000,
        df["test_mpe_corrected"].values[0] * 100,
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("Bias Correction Effect on Scenario C", fontsize=13, fontweight="bold")

    for ax, metric, bval, aval in zip(axes, metrics, before, after):
        x = [0, 1]
        colors_ba = ["#ff7f0e", "#2ca02c"]
        bars = ax.bar(x, [bval, aval], color=colors_ba, edgecolor="black", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Before", "After"])

        unit = "%" if metric in ["MAPE", "MPE"] else "k$"
        ax.set_ylabel(f"{metric} ({unit})")
        ax.set_title(metric)

        for bar, val in zip(bars, [bval, aval]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # 添加改善百分比
        if bval != 0:
            change = (aval - bval) / abs(bval) * 100
            color = "green" if change < 0 else "red"
            ax.text(0.5, 0.95, f"{change:+.1f}%", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, color=color, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, "fig9_bias_correction")


# ============================================================
# Main
# ============================================================
FIGURE_FUNCTIONS = {
    1: ("Scenario Performance Comparison", fig1_scenario_comparison),
    2: ("Convergence Comparison (B vs C)", fig2_convergence_comparison),
    3: ("LLM Strategy Selection Timeline", fig3_llm_strategy_timeline),
    4: ("Client MAPE Trends", fig4_client_mape_trends),
    5: ("Prediction vs True Scatter", fig5_prediction_scatter),
    6: ("Ablation Study", fig6_ablation),
    7: ("Multi-Seed Boxplot", fig7_multi_seed_boxplot),
    8: ("Stratified Evaluation", fig8_stratified_evaluation),
    9: ("Bias Correction", fig9_bias_correction),
}


def main():
    parser = argparse.ArgumentParser(description="Generate paper-quality figures")
    parser.add_argument("--fig", nargs="*", type=int, default=None,
                        help="Figure numbers to generate (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("PAPER FIGURE GENERATOR")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}/")

    figs_to_generate = args.fig or list(FIGURE_FUNCTIONS.keys())

    generated = 0
    for fig_num in figs_to_generate:
        if fig_num in FIGURE_FUNCTIONS:
            name, func = FIGURE_FUNCTIONS[fig_num]
            try:
                func()
                generated += 1
            except Exception as e:
                print(f"  [ERROR] Fig.{fig_num} ({name}): {e}")

    print(f"\n{'='*70}")
    print(f"Generated {generated} figures in {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
