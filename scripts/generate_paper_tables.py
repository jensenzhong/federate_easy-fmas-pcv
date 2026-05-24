"""Generate LaTeX tables from the current experiment outputs.

The main table prefers multi-seed summaries when available. Single-seed CSV
files are used only for scenarios that do not have multi-seed evidence.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


SCENARIO_ORDER = ["A", "A_prime", "B", "C"]
SCENARIO_LABELS = {
    "A": "A (GBR)",
    "A_prime": "A' (NN)",
    "B": "B (FedAvg)",
    "C": "C (MAS-FL-LLM)",
}
SINGLE_RESULT_FILES = {
    "A": "centralized_results.csv",
    "A_prime": "centralized_nn_results.csv",
    "B": "fedavg_results.csv",
    "C": "scenario_c_results.csv",
}


def _fmt_mean_std(mean, std, pct=False, dollar=False):
    if pd.isna(mean):
        return "-"
    std = 0 if pd.isna(std) else std
    if pct:
        return f"{mean * 100:.2f} $\\pm$ {std * 100:.2f}"
    if dollar:
        return f"{mean:,.0f} $\\pm$ {std:,.0f}"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def _fmt_single(value, pct=False, dollar=False):
    if pd.isna(value):
        return "-"
    if pct:
        return f"{value * 100:.2f}"
    if dollar:
        return f"{value:,.0f}"
    return f"{value:.4f}"


def _single_result_row(base, scenario):
    path = base / SINGLE_RESULT_FILES[scenario]
    if not path.exists():
        return None

    data = pd.read_csv(path).iloc[0]
    return {
        "label": SCENARIO_LABELS[scenario],
        "n": "1",
        "mape": _fmt_single(data.get("test_mape"), pct=True),
        "rmse": _fmt_single(data.get("test_rmse"), dollar=True),
        "mae": _fmt_single(data.get("test_mae"), dollar=True),
        "mpe": _fmt_single(data.get("test_mpe"), pct=True),
        "r2": _fmt_single(data.get("test_r2")) if "test_r2" in data else "-",
    }


def _multi_seed_rows(base):
    path = base / "multi_seed" / "statistical_summary.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    rows = []
    for scenario in SCENARIO_ORDER:
        match = df[df["scenario"] == scenario]
        if match.empty:
            continue

        data = match.iloc[0]
        rows.append({
            "label": SCENARIO_LABELS.get(scenario, scenario),
            "n": str(int(data.get("n_runs", 0))),
            "mape": _fmt_mean_std(data.get("test_mape_mean"), data.get("test_mape_std"), pct=True),
            "rmse": _fmt_mean_std(data.get("test_rmse_mean"), data.get("test_rmse_std"), dollar=True),
            "mae": _fmt_mean_std(data.get("test_mae_mean"), data.get("test_mae_std"), dollar=True),
            "mpe": _fmt_mean_std(data.get("test_mpe_mean"), data.get("test_mpe_std"), pct=True),
            "r2": _fmt_mean_std(data.get("test_r2_mean"), data.get("test_r2_std")),
        })

        if scenario == "C" and "test_mape_corrected_mean" in data and pd.notna(data.get("test_mape_corrected_mean")):
            rows.append({
                "label": "C + Bias Corr.",
                "n": str(int(data.get("n_runs", 0))),
                "mape": _fmt_mean_std(data.get("test_mape_corrected_mean"), data.get("test_mape_corrected_std"), pct=True),
                "rmse": _fmt_mean_std(data.get("test_rmse_corrected_mean"), data.get("test_rmse_corrected_std"), dollar=True),
                "mae": _fmt_mean_std(data.get("test_mae_corrected_mean"), data.get("test_mae_corrected_std"), dollar=True),
                "mpe": _fmt_mean_std(data.get("test_mpe_corrected_mean"), data.get("test_mpe_corrected_std"), pct=True),
                "r2": _fmt_mean_std(data.get("test_r2_corrected_mean"), data.get("test_r2_corrected_std")),
            })

    return rows


def generate_main_results_table():
    """Generate the main comparison table."""
    print("\n" + "=" * 70)
    print("Table 1: Main Experiment Results")
    print("=" * 70)

    base = Path("results")
    rows = []
    multi_rows = _multi_seed_rows(base)
    multi_labels = {row["label"] for row in multi_rows}

    for scenario in SCENARIO_ORDER:
        label = SCENARIO_LABELS[scenario]
        if label in multi_labels:
            rows.extend([row for row in multi_rows if row["label"] == label])
        else:
            single = _single_result_row(base, scenario)
            if single:
                rows.append(single)

    rows.extend([row for row in multi_rows if row["label"] == "C + Bias Corr."])

    if not rows:
        print("  [SKIP] No result files found")
        return ""

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Performance Comparison of Different Scenarios on Highway Cost Prediction}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
Scenario & N & MAPE (\%) $\downarrow$ & RMSE (\$) $\downarrow$ & MAE (\$) $\downarrow$ & MPE (\%) & $R^2$ $\uparrow$ \\
\midrule
"""

    for row in rows:
        latex += (
            f"{row['label']} & {row['n']} & {row['mape']} & {row['rmse']} & "
            f"{row['mae']} & {row['mpe']} & {row['r2']} \\\\\n"
        )

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    print(latex)
    return latex


def generate_ablation_table():
    """Generate the ablation table from ablation_summary.csv."""
    print("\n" + "=" * 70)
    print("Table 2: Ablation Study Results")
    print("=" * 70)

    path = Path("results/ablation_summary.csv")
    if not path.exists():
        print("  [SKIP] Run ablation experiments first")
        return ""

    df = pd.read_csv(path)
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study: Component Contribution Analysis}
\label{tab:ablation}
\begin{tabular}{lccccl}
\toprule
Config & FedProx & Strategy & LLM & MAPE (\%) & Contribution \\
\midrule
"""

    configs = [
        ("ab-1", "B-baseline", "No", "size\\_only", "No", "Baseline", "test_mape"),
        ("ab-2", "B+FedProx", "Yes", "size\\_only", "No", "+FedProx regularization", "test_mape"),
        ("ab-3", "C-fixed-perf", "Yes", "perf\\_only", "No", "+Performance weighting", "test_mape"),
        ("ab-4", "C-fixed-hybrid", "Yes", "hybrid", "No", "+Hybrid weighting", "test_mape"),
        ("ab-5", "C-with-LLM", "Yes", "Dynamic", "Yes", "+LLM decision making", "test_mape"),
        ("ab-6", "C-with-LLM+bias", "Yes", "Dynamic", "Yes", "+Bias correction", "test_mape_corrected"),
    ]

    for exp_id, name, fedprox, strategy, llm, contribution, metric_prefix in configs:
        match = df[df["id"] == exp_id]
        if match.empty:
            continue

        row = match.iloc[0]
        mean_key = f"{metric_prefix}_mean"
        std_key = f"{metric_prefix}_std"
        if mean_key not in row or pd.isna(row.get(mean_key)):
            continue

        mape = _fmt_mean_std(row.get(mean_key), row.get(std_key), pct=True)
        latex += f"{name} & {fedprox} & {strategy} & {llm} & {mape} & {contribution} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    print(latex)
    return latex


def generate_stratified_table():
    """Generate the stratified evaluation table when available."""
    print("\n" + "=" * 70)
    print("Table 3: Stratified Evaluation by Project Size")
    print("=" * 70)

    path = Path("results/stratified_evaluation.csv")
    if not path.exists():
        print("  [SKIP] Run stratified evaluation first")
        return ""

    df = pd.read_csv(path)
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Stratified Performance by Project Size}
\label{tab:stratified}
\begin{tabular}{llccccc}
\toprule
Scenario & Size Category & N & MAPE (\%) & RMSE (\$) & MAE (\$) & MPE (\%) \\
\midrule
"""

    for scenario in df["scenario"].unique():
        sdf = df[df["scenario"] == scenario]
        first = True
        for _, row in sdf.iterrows():
            name = scenario if first else ""
            first = False
            n = int(row["n"])
            if n == 0:
                latex += f"{name} & {row['stratum']} & 0 & - & - & - & - \\\\\n"
            else:
                latex += (
                    f"{name} & {row['stratum']} & {n} & {row['mape'] * 100:.2f} & "
                    f"{row['rmse']:,.0f} & {row['mae']:,.0f} & {row['mpe'] * 100:.2f} \\\\\n"
                )
        latex += r"\midrule" + "\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    print(latex)
    return latex


def main():
    print("=" * 70)
    print("PAPER TABLES GENERATOR")
    print("=" * 70)

    all_latex = ""
    all_latex += generate_main_results_table() or ""
    all_latex += generate_ablation_table() or ""
    all_latex += generate_stratified_table() or ""

    output_path = Path("results/paper_tables.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Paper Tables (LaTeX Source)\n\n")
        f.write("Generated from the current CSV outputs. Main B/C rows use multi-seed summaries when available.\n\n")
        f.write("```latex\n")
        f.write(all_latex)
        f.write("```\n")

    print(f"\n\nAll tables saved to {output_path}")


if __name__ == "__main__":
    main()
