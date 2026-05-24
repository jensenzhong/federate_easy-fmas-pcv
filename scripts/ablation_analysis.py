"""
消融实验结果分析脚本 (Ablation Analysis)

从ablation_logs中提取结果，生成对比表格（含LaTeX格式）
"""

import sys
import re
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


# 消融实验配置元数据
ABLATION_META = {
    "ab-1": {"name": "B-baseline", "fedprox": "No", "strategy": "size\\_only", "llm": "No", "rounds": 20},
    "ab-2": {"name": "B+FedProx", "fedprox": "Yes", "strategy": "size\\_only", "llm": "No", "rounds": 20},
    "ab-3": {"name": "C-perf\\_only", "fedprox": "Yes", "strategy": "perf\\_only", "llm": "No", "rounds": 20},
    "ab-4": {"name": "C-hybrid", "fedprox": "Yes", "strategy": "hybrid", "llm": "No", "rounds": 20},
    "ab-5": {"name": "C-LLM", "fedprox": "Yes", "strategy": "Dynamic", "llm": "Yes", "rounds": 20},
    "ab-6": {"name": "C-LLM+bias", "fedprox": "Yes", "strategy": "Dynamic+BC", "llm": "Yes", "rounds": 20},
}


def extract_metrics_from_log(log_path: Path, use_bias_corrected: bool = False) -> dict:
    """从日志文件中提取测试集指标"""
    content = log_path.read_text(encoding="utf-8", errors="replace")

    if use_bias_corrected:
        # 提取偏差校正后的结果
        section = content.split("[Bias Correction] Test Results")
        if len(section) >= 2:
            block = section[1][:500]
        else:
            # 无偏差校正结果，回退到普通结果
            block = content
    else:
        # 提取 "Test Set Results (using best checkpoint):" 之后的内容
        section = content.split("Test Set Results (using best checkpoint):")
        if len(section) >= 2:
            block = section[1][:500]
        else:
            block = content

    metrics = {}
    patterns = {
        "test_mape": r"MAPE:\s+([\d.]+)%",
        "test_rmse": r"RMSE:\s+\$([\d,]+\.?\d*)",
        "test_mae": r"MAE:\s+\$([\d,]+\.?\d*)",
        "test_mpe": r"MPE:\s+([-\d.]+)%",
        "test_r2": r"R2:\s+([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, block)
        if match:
            val_str = match.group(1).replace(",", "")
            val = float(val_str)
            if key in ("test_mape", "test_mpe"):
                val = val / 100.0  # 转为小数
            metrics[key] = val

    # 提取最佳轮次
    best_round_match = re.search(r"Best [Rr]ound[:\s]+(\d+)", content)
    if best_round_match:
        metrics["best_round"] = int(best_round_match.group(1))

    return metrics


def collect_ablation_results(results_dir: str = "results") -> pd.DataFrame:
    """收集所有消融实验结果"""
    log_dir = Path(results_dir) / "ablation_logs"
    if not log_dir.exists():
        print("[WARN] ablation_logs directory not found.")
        return pd.DataFrame()

    rows = []
    for ab_id in sorted(ABLATION_META.keys()):
        meta = ABLATION_META[ab_id]
        # 查找对应日志文件
        log_files = list(log_dir.glob(f"{ab_id}_*.log"))
        if not log_files:
            print(f"  [SKIP] {ab_id}: no log file found")
            continue

        log_path = log_files[0]
        use_bias = (ab_id == "ab-6")
        metrics = extract_metrics_from_log(log_path, use_bias_corrected=use_bias)

        if not metrics:
            print(f"  [SKIP] {ab_id}: could not extract metrics from {log_path.name}")
            continue

        row = {
            "id": ab_id,
            "name": meta["name"],
            "fedprox": meta["fedprox"],
            "strategy": meta["strategy"],
            "llm": meta["llm"],
            "rounds": meta["rounds"],
            **metrics,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame):
    """打印可读的汇总表"""
    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    print(f"{'ID':<6} {'Name':<18} {'MAPE%':>8} {'RMSE$':>12} {'MAE$':>12} {'MPE%':>8} {'R2':>8} {'Best':>5}")
    print("-" * 90)

    for _, row in df.iterrows():
        mape = f"{row['test_mape']*100:.2f}" if pd.notna(row.get('test_mape')) else "-"
        rmse = f"{row['test_rmse']:,.0f}" if pd.notna(row.get('test_rmse')) else "-"
        mae = f"{row['test_mae']:,.0f}" if pd.notna(row.get('test_mae')) else "-"
        mpe = f"{row['test_mpe']*100:.2f}" if pd.notna(row.get('test_mpe')) else "-"
        r2 = f"{row['test_r2']:.4f}" if pd.notna(row.get('test_r2')) else "-"
        best = str(int(row.get('best_round', 0))) if pd.notna(row.get('best_round')) else "-"
        print(f"{row['id']:<6} {row['name']:<18} {mape:>8} {rmse:>12} {mae:>12} {mpe:>8} {r2:>8} {best:>5}")

    print("=" * 90)

    # 分析各组件贡献
    print("\nComponent Contribution Analysis:")
    if len(df) >= 2:
        mape_vals = df.set_index("id")["test_mape"]
        if "ab-1" in mape_vals.index and "ab-2" in mape_vals.index:
            delta = (mape_vals["ab-1"] - mape_vals["ab-2"]) * 100
            print(f"  FedProx effect (ab-1 → ab-2): {delta:+.2f}% MAPE")
        if "ab-2" in mape_vals.index and "ab-3" in mape_vals.index:
            delta = (mape_vals["ab-2"] - mape_vals["ab-3"]) * 100
            print(f"  perf_only strategy (ab-2 → ab-3): {delta:+.2f}% MAPE")
        if "ab-3" in mape_vals.index and "ab-4" in mape_vals.index:
            delta = (mape_vals["ab-3"] - mape_vals["ab-4"]) * 100
            print(f"  hybrid vs perf_only (ab-3 → ab-4): {delta:+.2f}% MAPE")
        if "ab-4" in mape_vals.index and "ab-5" in mape_vals.index:
            delta = (mape_vals["ab-4"] - mape_vals["ab-5"]) * 100
            print(f"  LLM dynamic selection (ab-4 → ab-5): {delta:+.2f}% MAPE")
        if "ab-5" in mape_vals.index and "ab-6" in mape_vals.index:
            delta = (mape_vals["ab-5"] - mape_vals["ab-6"]) * 100
            print(f"  Bias correction (ab-5 → ab-6): {delta:+.2f}% MAPE")
        if "ab-1" in mape_vals.index and "ab-5" in mape_vals.index:
            delta = (mape_vals["ab-1"] - mape_vals["ab-5"]) * 100
            print(f"  Total improvement (ab-1 → ab-5): {delta:+.2f}% MAPE")


def generate_latex_table(df: pd.DataFrame) -> str:
    """生成LaTeX格式消融表"""
    print("\n" + "=" * 70)
    print("LaTeX Table: Ablation Study Results")
    print("=" * 70)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation Study: Component Contribution Analysis}
\label{tab:ablation}
\begin{tabular}{lcccccc}
\toprule
Configuration & FedProx & Strategy & LLM & MAPE (\%) $\downarrow$ & RMSE (\$) $\downarrow$ & $R^2$ $\uparrow$ \\
\midrule
"""

    for _, row in df.iterrows():
        name = row["name"]
        fedprox = row["fedprox"]
        strategy = row["strategy"]
        llm = row["llm"]
        mape = f"{row['test_mape']*100:.2f}" if pd.notna(row.get("test_mape")) else "-"
        rmse = f"{row['test_rmse']:,.0f}" if pd.notna(row.get("test_rmse")) else "-"
        r2 = f"{row['test_r2']:.4f}" if pd.notna(row.get("test_r2")) else "-"
        latex += f"{name} & {fedprox} & {strategy} & {llm} & {mape} & {rmse} & {r2} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    print(latex)
    return latex


def main():
    print("=" * 70)
    print("ABLATION ANALYSIS")
    print("=" * 70)

    df = collect_ablation_results()

    if df.empty:
        print("\nNo results found. Please run ablation experiments first:")
        print("  python scripts/run_ablation.py")
        return

    print_summary_table(df)
    latex = generate_latex_table(df)

    # 保存
    output_path = Path("results") / "ablation_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved summary to {output_path}")

    # 保存LaTeX
    latex_path = Path("results") / "ablation_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved LaTeX to {latex_path}")


if __name__ == "__main__":
    main()
