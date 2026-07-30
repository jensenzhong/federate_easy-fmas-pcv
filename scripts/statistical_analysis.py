"""
统计分析脚本 (Statistical Analysis)

基于多种子实验结果：
1. 计算 mean ± std
2. 执行配对t检验或Wilcoxon检验（B vs C）
3. 生成带置信区间的结果汇总表
4. 输出LaTeX格式表格

用法:
    python scripts/statistical_analysis.py
    python scripts/statistical_analysis.py --results_file results/multi_seed/all_results.csv
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from scipy import stats

from src.experiment_names import (
    EXPERIMENT_ORDER,
    canonical_experiment_key,
    experiment_display_name,
)


def load_results(results_file: str) -> pd.DataFrame:
    """加载多种子实验结果"""
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} result entries from {results_file}")
    return df


def _success_mask(df: pd.DataFrame) -> pd.Series:
    if "success" not in df.columns:
        return pd.Series(True, index=df.index)
    success = df["success"]
    if success.dtype == bool:
        return success
    return success.astype(str).str.lower().isin(["true", "1", "yes", "ok"])


def normalize_experiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical keys while keeping user-facing names semantic."""
    normalized = df.copy()
    if "scenario_key" in normalized.columns:
        keys = normalized["scenario_key"].map(canonical_experiment_key)
    else:
        keys = normalized["scenario"].map(canonical_experiment_key)
    normalized["scenario_key"] = keys
    normalized["scenario"] = keys.map(experiment_display_name)
    return normalized


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算各场景的汇总统计"""
    df = normalize_experiment_columns(df)
    scenarios = sorted(
        df["scenario_key"].dropna().unique(),
        key=lambda key: EXPERIMENT_ORDER.index(key) if key in EXPERIMENT_ORDER else len(EXPERIMENT_ORDER)
    )
    metrics = [
        "test_mape", "test_rmse", "test_mae", "test_mpe", "test_r2",
        "test_mape_corrected", "test_rmse_corrected", "test_mae_corrected",
        "test_mpe_corrected", "test_r2_corrected",
    ]

    summary_rows = []
    for scenario in scenarios:
        scenario_df = df[(df["scenario_key"] == scenario) & _success_mask(df)]
        n_runs = len(scenario_df)

        row = {
            "scenario": experiment_display_name(scenario),
            "scenario_key": scenario,
            "n_runs": n_runs,
        }
        for metric in metrics:
            if metric in scenario_df.columns and scenario_df[metric].notna().any():
                values = scenario_df[metric].dropna()
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std()
                row[f"{metric}_min"] = values.min()
                row[f"{metric}_max"] = values.max()

                # 95%置信区间
                if len(values) > 1:
                    ci = stats.t.interval(
                        0.95, len(values) - 1,
                        loc=values.mean(), scale=stats.sem(values)
                    )
                    row[f"{metric}_ci_lower"] = ci[0]
                    row[f"{metric}_ci_upper"] = ci[1]

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _paired_or_independent_test(
    df: pd.DataFrame,
    left_key: str,
    right_key: str,
    left_metric: str,
    right_metric: str,
) -> dict | None:
    left_data = df[(df["scenario_key"] == left_key) & _success_mask(df)]
    right_data = df[(df["scenario_key"] == right_key) & _success_mask(df)]

    if len(left_data) < 3 or len(right_data) < 3:
        print(
            f"\n  [WARN] Insufficient data for "
            f"{experiment_display_name(left_key)} vs {experiment_display_name(right_key)} comparison"
        )
        print(f"    {left_key}: {len(left_data)} runs, {right_key}: {len(right_data)} runs")
        return None

    if left_metric not in left_data.columns or right_metric not in right_data.columns:
        return None

    if "seed" in left_data.columns and "seed" in right_data.columns:
        paired = (
            left_data[["seed", left_metric]]
            .dropna()
            .rename(columns={left_metric: "left_value"})
            .merge(
                right_data[["seed", right_metric]].dropna().rename(columns={right_metric: "right_value"}),
                on="seed",
            )
            .sort_values("seed")
        )
    else:
        paired = pd.DataFrame()

    if len(paired) >= 2:
        paired_seeds = paired["seed"].astype(int).tolist()
        left_values = paired["left_value"].values
        right_values = paired["right_value"].values
        t_stat, p_value = stats.ttest_rel(left_values, right_values)
        test_type = "paired t-test"
    else:
        paired_seeds = []
        left_values = left_data[left_metric].dropna().values
        right_values = right_data[right_metric].dropna().values
        if len(left_values) < 2 or len(right_values) < 2:
            return None
        t_stat, p_value = stats.ttest_ind(left_values, right_values)
        test_type = "independent t-test"

    if len(left_values) < 2 or len(right_values) < 2:
        return None

    try:
        if paired_seeds:
            w_stat, w_pvalue = stats.wilcoxon(left_values, right_values)
            wilcoxon_type = "Wilcoxon signed-rank"
        else:
            w_stat, w_pvalue = stats.mannwhitneyu(
                left_values, right_values, alternative="two-sided"
            )
            wilcoxon_type = "Mann-Whitney U"
    except Exception:
        w_stat, w_pvalue = np.nan, np.nan
        wilcoxon_type = "N/A"

    pooled_std = np.sqrt((np.var(left_values) + np.var(right_values)) / 2)
    cohens_d = (
        (np.mean(left_values) - np.mean(right_values)) / pooled_std
        if pooled_std > 0 else 0
    )

    return {
        "left_key": left_key,
        "right_key": right_key,
        "left_scenario": experiment_display_name(left_key),
        "right_scenario": experiment_display_name(right_key),
        "left_metric": left_metric,
        "right_metric": right_metric,
        "left_mean": np.mean(left_values),
        "left_std": np.std(left_values),
        "right_mean": np.mean(right_values),
        "right_std": np.std(right_values),
        # Legacy column names retained for downstream CSV/table compatibility.
        "b_mean": np.mean(left_values),
        "b_std": np.std(left_values),
        "c_mean": np.mean(right_values),
        "c_std": np.std(right_values),
        "paired_seeds": paired_seeds,
        "t_test_type": test_type,
        "t_statistic": t_stat,
        "t_p_value": p_value,
        "wilcoxon_type": wilcoxon_type,
        "w_statistic": w_stat,
        "w_p_value": w_pvalue,
        "cohens_d": cohens_d,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def perform_statistical_tests(df: pd.DataFrame) -> dict:
    """执行主实验统计检验。"""
    df = normalize_experiment_columns(df)
    results = {}

    comparison_specs = [
        ("A_prime", "B", None),
        ("A_prime", "FEDYOGI", None),
        ("A_prime", "VG_FEDYOGI_TR", None),
        ("A_prime", "MAS_VG_FEDYOGI_TR", None),
        ("B", "FEDYOGI", None),
        ("B", "VG_FEDYOGI_TR", None),
        ("B", "MAS_VG_FEDYOGI_TR", None),
        ("B", "COHERENCE_FEDYOGI_TR", None),
        ("B", "LLM_GCA_FEDYOGI_TR", None),
        ("FEDYOGI", "VG_FEDYOGI_TR", None),
        ("FEDYOGI", "COHERENCE_FEDYOGI_TR", None),
        ("FEDYOGI", "LLM_GCA_FEDYOGI_TR", None),
        ("VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR", None),
        ("COHERENCE_FEDYOGI_TR", "LLM_GCA_FEDYOGI_TR", None),
        ("FEDYOGI", "VG_FEDYOGI_TR", "both_corrected"),
        ("FEDYOGI", "MAS_VG_FEDYOGI_TR", "both_corrected"),
        ("FEDYOGI", "COHERENCE_FEDYOGI_TR", "both_corrected"),
        ("FEDYOGI", "LLM_GCA_FEDYOGI_TR", "both_corrected"),
        ("VG_FEDYOGI_TR", "MAS_VG_FEDYOGI_TR", "both_corrected"),
        ("COHERENCE_FEDYOGI_TR", "LLM_GCA_FEDYOGI_TR", "both_corrected"),
    ]
    comparisons = []
    for left_key, right_key, mode in comparison_specs:
        comparison_name = f"{experiment_display_name(left_key)}_vs_{experiment_display_name(right_key)}"
        if mode == "both_corrected":
            comparison_name = (
                f"{experiment_display_name(left_key)}_bias_corrected_vs_"
                f"{experiment_display_name(right_key)}_bias_corrected"
            )
        elif mode == "corrected":
            comparison_name = (
                f"{experiment_display_name(left_key)}_vs_"
                f"{experiment_display_name(right_key)}_bias_corrected"
            )
        comparisons.append((left_key, right_key, comparison_name, mode))

    for left_key, right_key, comparison_name, mode in comparisons:
        test_results = {}
        for metric in ["test_mape", "test_rmse", "test_mae"]:
            left_metric = f"{metric}_corrected" if mode == "both_corrected" else metric
            right_metric = f"{metric}_corrected" if mode in ("corrected", "both_corrected") else metric
            result = _paired_or_independent_test(
                df=df,
                left_key=left_key,
                right_key=right_key,
                left_metric=left_metric,
                right_metric=right_metric,
            )
            if result is not None:
                test_results[right_metric] = result
        if test_results:
            results[comparison_name] = test_results

    return results


def generate_latex_tables(summary: pd.DataFrame, test_results: dict):
    """生成LaTeX格式表格"""

    # 主实验对比表
    print("\n" + "=" * 70)
    print("LaTeX Table: Main Experiment Results (mean ± std)")
    print("=" * 70)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Main Experiment Results across 5 Random Seeds (mean $\pm$ std)}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
Scenario & N & MAPE (\%) & RMSE (\$) & MAE (\$) & MPE (\%) & $R^2$ \\
\midrule
"""
    for _, row in summary.iterrows():
        name = row["scenario"]
        n = int(row["n_runs"])
        mape = _fmt_mean_std(row, "test_mape", pct=True)
        rmse = _fmt_mean_std(row, "test_rmse", dollar=True)
        mae = _fmt_mean_std(row, "test_mae", dollar=True)
        mpe = _fmt_mean_std(row, "test_mpe", pct=True)
        r2 = _fmt_mean_std(row, "test_r2")
        latex += f"{name} & {n} & {mape} & {rmse} & {mae} & {mpe} & {r2} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    print(latex)

    # 统计检验表
    if test_results:
        print("\n" + "=" * 70)
        print("LaTeX Table: Statistical Test Results")
        print("=" * 70)

        latex2 = r"""
\begin{table}[htbp]
\centering
\caption{Statistical Significance Tests for Main Experiment Comparisons}
\label{tab:significance}
\begin{tabular}{lcccc}
\toprule
Metric & Test & Statistic & p-value & Significant \\
\midrule
"""
        for comparison, metrics in test_results.items():
            for metric, result in metrics.items():
                metric_name = metric.replace("test_", "").upper()
                t_type = result["t_test_type"]
                t_stat = f"{result['t_statistic']:.3f}"
                p_val = f"{result['t_p_value']:.4f}"
                sig = "Yes***" if result["significant_001"] else ("Yes*" if result["significant_005"] else "No")
                latex2 += f"{comparison}: {metric_name} & {t_type} & {t_stat} & {p_val} & {sig} \\\\\n"

        latex2 += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        print(latex2)

    return latex


def flatten_significance_tests(test_results: dict) -> pd.DataFrame:
    """Convert nested B-vs-C test results into a CSV-friendly table."""
    rows = []
    for comparison, metrics in test_results.items():
        for metric, result in metrics.items():
            paired_seeds = result.get("paired_seeds", [])
            rows.append({
                "comparison": comparison,
                "metric": metric,
                "left_key": result.get("left_key"),
                "right_key": result.get("right_key"),
                "left_scenario": result.get("left_scenario"),
                "right_scenario": result.get("right_scenario"),
                "left_metric": result.get("left_metric"),
                "right_metric": result.get("right_metric"),
                "left_mean": result.get("left_mean"),
                "left_std": result.get("left_std"),
                "right_mean": result.get("right_mean"),
                "right_std": result.get("right_std"),
                "paired_seeds": ",".join(str(seed) for seed in paired_seeds),
                "t_test_type": result.get("t_test_type"),
                "t_statistic": result.get("t_statistic"),
                "t_p_value": result.get("t_p_value"),
                "wilcoxon_type": result.get("wilcoxon_type"),
                "w_statistic": result.get("w_statistic"),
                "w_p_value": result.get("w_p_value"),
                "cohens_d": result.get("cohens_d"),
                "significant_005": result.get("significant_005"),
                "significant_001": result.get("significant_001"),
            })

    return pd.DataFrame(rows)


def _fmt_mean_std(row, metric_prefix, pct=False, dollar=False):
    """格式化 mean ± std"""
    mean_key = f"{metric_prefix}_mean"
    std_key = f"{metric_prefix}_std"

    if mean_key not in row or pd.isna(row[mean_key]):
        return "-"

    mean_val = row[mean_key]
    std_val = row.get(std_key, 0)

    if pct:
        return f"{mean_val*100:.2f} $\\pm$ {std_val*100:.2f}"
    elif dollar:
        return f"{mean_val:,.0f} $\\pm$ {std_val:,.0f}"
    else:
        return f"{mean_val:.4f} $\\pm$ {std_val:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of multi-seed results")
    parser.add_argument("--results_file", type=str,
                        default="results/multi_seed/all_results.csv",
                        help="Path to multi-seed results CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"\n[ERROR] Results file not found: {results_path}")
        print("Please run multi-seed experiments first:")
        print("  python scripts/run_multi_seed.py")
        return

    df = load_results(str(results_path))

    # 汇总统计
    print("\n--- Summary Statistics ---")
    summary = compute_summary_stats(df)
    print(summary.to_string(index=False))

    # 统计检验
    print("\n--- Statistical Tests ---")
    test_results = perform_statistical_tests(df)

    for comparison, metrics in test_results.items():
        print(f"\n  {comparison}:")
        for metric, result in metrics.items():
            sig = "***" if result["significant_001"] else ("*" if result["significant_005"] else "ns")
            print(f"    {metric}: p={result['t_p_value']:.4f} ({sig}), "
                  f"Cohen's d={result['cohens_d']:.3f}")

    # 生成LaTeX表格
    generate_latex_tables(summary, test_results)

    # 保存结果
    summary_path = Path("results") / "multi_seed" / "statistical_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved statistical summary to {summary_path}")

    tests_path = Path("results") / "multi_seed" / "significance_tests.csv"
    flatten_significance_tests(test_results).to_csv(tests_path, index=False)
    print(f"Saved significance tests to {tests_path}")


if __name__ == "__main__":
    main()
