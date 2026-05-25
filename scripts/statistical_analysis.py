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


def load_results(results_file: str) -> pd.DataFrame:
    """加载多种子实验结果"""
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} result entries from {results_file}")
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算各场景的汇总统计"""
    scenarios = df["scenario"].unique()
    metrics = [
        "test_mape", "test_rmse", "test_mae", "test_mpe", "test_r2",
        "test_mape_corrected", "test_rmse_corrected", "test_mae_corrected",
        "test_mpe_corrected", "test_r2_corrected",
    ]

    summary_rows = []
    for scenario in sorted(scenarios):
        scenario_df = df[(df["scenario"] == scenario) & (df["success"] == True)]
        n_runs = len(scenario_df)

        row = {"scenario": scenario, "n_runs": n_runs}
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


def perform_statistical_tests(df: pd.DataFrame) -> dict:
    """执行统计检验（B vs C各变体）"""
    results = {}

    # 获取B场景数据
    b_data = df[(df["scenario"] == "B") & (df["success"] == True)]

    # 对比的场景
    comparison_scenarios = ["C"]

    metrics_to_test = ["test_mape", "test_rmse", "test_mae"]

    for comp_scenario in comparison_scenarios:
        comp_data = df[(df["scenario"] == comp_scenario) & (df["success"] == True)]

        if len(b_data) < 3 or len(comp_data) < 3:
            print(f"\n  [WARN] Insufficient data for B vs {comp_scenario} comparison")
            print(f"    B: {len(b_data)} runs, {comp_scenario}: {len(comp_data)} runs")
            continue

        test_results = {}
        for metric in metrics_to_test:
            if metric not in b_data.columns or metric not in comp_data.columns:
                continue

            if "seed" in b_data.columns and "seed" in comp_data.columns:
                paired = (
                    b_data[["seed", metric]]
                    .dropna()
                    .merge(
                        comp_data[["seed", metric]].dropna(),
                        on="seed",
                        suffixes=("_b", "_c"),
                    )
                    .sort_values("seed")
                )
            else:
                paired = pd.DataFrame()

            if len(paired) >= 2:
                paired_seeds = paired["seed"].astype(int).tolist()
                b_values = paired[f"{metric}_b"].values
                c_values = paired[f"{metric}_c"].values
                t_stat, p_value = stats.ttest_rel(b_values, c_values)
                test_type = "paired t-test"
            else:
                paired_seeds = []
                b_values = b_data[metric].dropna().values
                c_values = comp_data[metric].dropna().values
                if len(b_values) < 2 or len(c_values) < 2:
                    continue
                t_stat, p_value = stats.ttest_ind(b_values, c_values)
                test_type = "independent t-test"

            if len(b_values) < 2 or len(c_values) < 2:
                continue

            # Wilcoxon检验（非参数）
            try:
                if paired_seeds:
                    w_stat, w_pvalue = stats.wilcoxon(b_values, c_values)
                else:
                    w_stat, w_pvalue = stats.mannwhitneyu(b_values, c_values, alternative="two-sided")
                wilcoxon_type = "Wilcoxon signed-rank" if paired_seeds else "Mann-Whitney U"
            except Exception:
                w_stat, w_pvalue = np.nan, np.nan
                wilcoxon_type = "N/A"

            # 效应量 (Cohen's d)
            pooled_std = np.sqrt((np.var(b_values) + np.var(c_values)) / 2)
            cohens_d = (np.mean(b_values) - np.mean(c_values)) / pooled_std if pooled_std > 0 else 0

            test_results[metric] = {
                "b_mean": np.mean(b_values),
                "b_std": np.std(b_values),
                "c_mean": np.mean(c_values),
                "c_std": np.std(c_values),
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

        results[f"B_vs_{comp_scenario}"] = test_results

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
\caption{Statistical Significance Tests (B vs C)}
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
                latex2 += f"{metric_name} & {t_type} & {t_stat} & {p_val} & {sig} \\\\\n"

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
