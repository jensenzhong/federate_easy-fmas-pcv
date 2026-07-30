"""
论文级可视化脚本 (Paper-Quality Figures)

生成9张论文所需的图表，所有图表使用英文标签、300 DPI、同时输出PNG+PDF。

图表列表:
  Fig.1 - Method performance comparison
  Fig.2 - Federated method convergence comparison (MAPE vs Round)
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

from src.experiment_names import experiment_display_name

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
    "FEDYOGI": "#9467bd",
    "VG_FEDYOGI_TR": "#d62728",
    "MAS_VG_FEDYOGI_TR": "#17becf",
}

OUTPUT_DIR = Path("results/figures")
MAINLINE_SCENARIOS = ["A", "A_prime", "B", "FEDYOGI", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR"]
FEDERATED_MAINLINE_SCENARIOS = ["B", "FEDYOGI", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR"]
SCATTER_SCENARIOS = ["A_prime", "B", "FEDYOGI", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR"]
BIAS_CORRECTION_SCENARIOS = ["FEDYOGI", "VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR"]

RESULT_FILES = {
    "A": "centralized_results.csv",
    "A_prime": "centralized_nn_results.csv",
    "B": "fedavg_results.csv",
    "FEDYOGI": "fedyogi_results.csv",
    "VG_FEDYOGI_TR": "vg_fedyogi_tr_results.csv",
    "MAS_VG_FEDYOGI_TR": "mas_vg_fedyogi_tr_results.csv",
}

PREDICTION_FILES = {
    "A": "centralized_predictions.csv",
    "A_prime": "centralized_nn_predictions.csv",
    "B": "fedavg_predictions.csv",
    "FEDYOGI": "fedyogi_predictions.csv",
    "VG_FEDYOGI_TR": "vg_fedyogi_tr_predictions.csv",
    "MAS_VG_FEDYOGI_TR": "mas_vg_fedyogi_tr_predictions.csv",
}

ROUND_METRIC_FILES = {
    "B": "scene_B_round_metrics.csv",
    "FEDYOGI": "fedyogi_round_metrics.csv",
    "VG_FEDYOGI_TR": "vg_fedyogi_tr_round_metrics.csv",
    "MAS_VG_FEDYOGI_TR": "mas_vg_fedyogi_tr_round_metrics.csv",
}

CLIENT_METRIC_FILES = {
    "MAS_VG_FEDYOGI_TR": "mas_vg_fedyogi_tr_client_metrics.csv",
}

LLM_DECISION_FILES = {
    "MAS_VG_FEDYOGI_TR": "mas_vg_fedyogi_tr_llm_decisions.jsonl",
}


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
    """Method-level performance comparison."""
    print("\n[Fig.1] Scenario Performance Comparison")

    base = Path("results")
    scenarios = {}

    # 加载各场景结果CSV
    color_keys = []
    for key in MAINLINE_SCENARIOS:
        filename = RESULT_FILES[key]
        path = base / filename
        if path.exists():
            df = pd.read_csv(path)
            name = experiment_display_name(key)
            color_keys.append(key)
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
    colors = [COLORS.get(k, "#333333") for k in color_keys]

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
    print("\n[Fig.2] Federated Method Convergence Comparison")

    base = Path("results/logs")

    available = {
        key: base / filename
        for key, filename in ROUND_METRIC_FILES.items()
        if (base / filename).exists()
    }
    if len(available) < 2:
        print("  [SKIP] Missing round_metrics.csv for federated methods")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Federated Learning Convergence by Method",
                 fontsize=13, fontweight="bold")

    # (a) MAPE convergence
    markers = {"B": "o", "FEDYOGI": "^", "VG_FEDYOGI_TR": "s", "MAS_VG_FEDYOGI_TR": "D"}
    for key, path in available.items():
        df_method = pd.read_csv(path)
        ax1.plot(df_method["round"], df_method["global_val_mape"] * 100,
                 marker=markers.get(key, "o"), linestyle="-",
                 color=COLORS.get(key, "#333333"), linewidth=2, markersize=4,
                 label=experiment_display_name(key))
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Global Validation MAPE (%)")
    ax1.set_title("(a) MAPE Convergence")
    ax1.legend()

    # (b) RMSE convergence
    for key, path in available.items():
        df_method = pd.read_csv(path)
        ax2.plot(df_method["round"], df_method["global_val_rmse"] / 1e6,
                 marker=markers.get(key, "o"), linestyle="-",
                 color=COLORS.get(key, "#333333"), linewidth=2, markersize=4,
                 label=experiment_display_name(key))
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
    """MAS-VG candidate request, selection, and gate timeline."""
    print("\n[Fig.3] LLM Strategy Selection Timeline")

    round_path = Path("results/logs/mas_vg_fedyogi_tr_round_metrics.csv")
    decisions_path = Path("results/logs/mas_vg_fedyogi_tr_llm_decisions.jsonl")
    if not round_path.exists():
        print("  [SKIP] No MAS-VG round metrics found")
        return
    if not decisions_path.exists():
        print("  [SKIP] No MAS-VG LLM decisions log found")
        return

    rounds_df = pd.read_csv(round_path)
    rounds = rounds_df["round"].tolist()
    requested_ids = rounds_df["requested_candidate_id"].fillna("none").astype(str).tolist()
    selected_ids = rounds_df["selected_candidate_id"].fillna("none").astype(str).tolist()
    gate_statuses = rounds_df["gate_status"].fillna("unknown").astype(str).tolist()

    candidate_ids = sorted({*requested_ids, *selected_ids})
    candidate_to_y = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    requested_y = [candidate_to_y[candidate_id] for candidate_id in requested_ids]
    selected_y = [candidate_to_y[candidate_id] for candidate_id in selected_ids]

    decision_rounds = []
    with open(decisions_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line.strip())
            decision_rounds.append(payload.get("round"))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("MAS-VG Candidate Selection and Gate Timeline", fontsize=13, fontweight="bold")

    ax1.plot(rounds, requested_y, "o--", color="#7f7f7f", linewidth=1.5, markersize=4, label="Requested")
    ax1.plot(rounds, selected_y, "o-", color=COLORS["MAS_VG_FEDYOGI_TR"], linewidth=2, markersize=4, label="Selected")
    for round_id in decision_rounds:
        ax1.axvline(round_id, color="#d9d9d9", linestyle=":", linewidth=0.8, zorder=0)
    ax1.set_ylabel("Candidate ID")
    ax1.set_yticks(range(len(candidate_ids)))
    ax1.set_yticklabels(candidate_ids)
    ax1.set_title("(a) Requested vs Selected Candidate")
    ax1.legend(loc="upper right", fontsize=8)

    gate_names = sorted(set(gate_statuses))
    gate_to_y = {name: index for index, name in enumerate(gate_names)}
    gate_colors = plt.cm.Set2(np.linspace(0, 1, len(gate_names)))
    for gate_name, color in zip(gate_names, gate_colors):
        gate_rounds = [rounds[index] for index, value in enumerate(gate_statuses) if value == gate_name]
        gate_y = [gate_to_y[gate_name]] * len(gate_rounds)
        ax2.scatter(gate_rounds, gate_y, color=color, s=45, label=gate_name, zorder=3)
    ax2.set_ylabel("Gate Status")
    ax2.set_yticks(range(len(gate_names)))
    ax2.set_yticklabels(gate_names)
    ax2.set_title("(b) Gate Decision per Round")
    ax2.legend(loc="upper right", fontsize=8)

    candidate_score = pd.to_numeric(rounds_df.get("candidate_score"), errors="coerce")
    ax3.plot(rounds, candidate_score, "o-", color="#1f77b4", markersize=4)
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Candidate Score")
    ax3.set_title("(c) Validation-Based Candidate Score")

    fig.tight_layout()
    save_fig(fig, "fig3_llm_strategy_timeline")


# ============================================================
# Fig.4: 三客户端val_mape随轮次变化
# ============================================================
def fig4_client_mape_trends():
    """三客户端验证MAPE随轮次变化"""
    print("\n[Fig.4] Client Validation MAPE Trends")

    client_path = Path("results/logs/mas_vg_fedyogi_tr_client_metrics.csv")
    if not client_path.exists():
        print("  [SKIP] No MAS-VG client metrics found")
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
    ax.set_title("Client-wise Validation MAPE Trends in MAS-VG-FedYogi-TR", fontweight="bold")
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
    loaded = {}
    for key in SCATTER_SCENARIOS:
        name = experiment_display_name(key)
        filename = PREDICTION_FILES[key]
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
    if df.empty:
        print("  [SKIP] No valid ablation data")
        return

    if "test_mape_mean" in df.columns:
        metric_col = "test_mape_mean"
        std_col = "test_mape_std" if "test_mape_std" in df.columns else None
    elif "test_mape" in df.columns:
        metric_col = "test_mape"
        std_col = None
    else:
        print("  [SKIP] No valid ablation MAPE columns")
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    names = [str(name).replace("\\_", "_") for name in df["name"].tolist()]
    mape = df[metric_col].to_numpy(dtype=float) * 100
    mape_std = (
        df[std_col].fillna(0).to_numpy(dtype=float) * 100
        if std_col is not None
        else None
    )
    x = np.arange(len(names))

    line_color = "#D55E00"
    if mape_std is not None:
        ax.errorbar(
            x,
            mape,
            yerr=mape_std,
            color=line_color,
            linewidth=2.2,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=2,
            markeredgecolor=line_color,
            capsize=4,
        )
    else:
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

    lower = mape - mape_std if mape_std is not None else mape
    upper = mape + mape_std if mape_std is not None else mape
    y_min = float(np.floor((lower.min() - 0.15) * 10) / 10)
    y_max = float(np.ceil((upper.max() + 0.15) * 10) / 10)
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

    scenario_order = [experiment_display_name(key) for key in MAINLINE_SCENARIOS]
    scenarios = [scenario for scenario in scenario_order if scenario in set(df["scenario"].unique())]
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

    rows = []
    for scenario_key in BIAS_CORRECTION_SCENARIOS:
        result_path = Path("results") / RESULT_FILES[scenario_key]
        if not result_path.exists():
            continue
        df = pd.read_csv(result_path)
        if df.empty or "test_mape_corrected" not in df.columns:
            continue
        row = df.iloc[0]
        rows.append({
            "label": experiment_display_name(scenario_key),
            "before": float(row["test_mape"]) * 100,
            "after": float(row["test_mape_corrected"]) * 100,
        })

    if not rows:
        print("  [SKIP] No bias-corrected adaptive results found")
        return

    labels = [row["label"] for row in rows]
    before = [row["before"] for row in rows]
    after = [row["after"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.suptitle("Bias Correction Effect Across Adaptive Mainline Methods", fontsize=13, fontweight="bold")

    before_bars = ax.bar(x - width / 2, before, width, label="Before", color="#ff7f0e", edgecolor="black", alpha=0.8)
    after_bars = ax.bar(x + width / 2, after, width, label="After", color="#2ca02c", edgecolor="black", alpha=0.8)

    ax.set_ylabel("Test MAPE (%)")
    ax.set_title("Bias Correction Before vs After")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.legend()

    for bars in (before_bars, after_bars):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    save_fig(fig, "fig9_bias_correction")


# ============================================================
# Main
# ============================================================
FIGURE_FUNCTIONS = {
    1: ("Scenario Performance Comparison", fig1_scenario_comparison),
    2: ("Federated Method Convergence Comparison", fig2_convergence_comparison),
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
