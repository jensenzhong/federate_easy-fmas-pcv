"""
分层评估脚本 (Stratified Evaluation)

按项目规模分层评估各场景的模型性能:
- small: < $1M
- medium: $1M - $5M
- large: >= $5M

用法:
    python scripts/stratified_evaluation.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.utils import (
    load_config, compute_mape, compute_rmse, compute_mae,
    compute_mpe, compute_r2
)


def load_predictions(results_dir: str = "results") -> dict:
    """加载各场景的预测结果"""
    base = Path(results_dir)
    predictions = {}

    # 场景A
    pred_a_path = base / "centralized_predictions.csv"
    if pred_a_path.exists():
        predictions["A (GBR)"] = pd.read_csv(pred_a_path)

    # 场景A'
    pred_a_prime_path = base / "centralized_nn_predictions.csv"
    if pred_a_prime_path.exists():
        predictions["A' (NN)"] = pd.read_csv(pred_a_prime_path)

    # 对于B和C场景，需要从训练历史推导或从保存的预测文件中加载
    # 如果没有单独的预测文件，我们需要重新运行模型推理
    for name, filename in [("B (FedAvg)", "fedavg_predictions.csv"),
                           ("C (MAS-FL-LLM)", "scenario_c_predictions.csv")]:
        path = base / filename
        if path.exists():
            predictions[name] = pd.read_csv(path)

    return predictions


def stratify_by_size(y_true: np.ndarray, y_pred: np.ndarray, config: dict) -> dict:
    """按项目规模分层计算指标"""
    thresholds = config.get("thresholds", {})
    small_max = thresholds.get("small_project_max", 1_000_000)
    medium_max = thresholds.get("medium_project_max", 5_000_000)

    strata = {
        f"Small (<${small_max/1e6:.0f}M)": (y_true < small_max),
        f"Medium (${small_max/1e6:.0f}M-${medium_max/1e6:.0f}M)": (y_true >= small_max) & (y_true < medium_max),
        f"Large (>=${medium_max/1e6:.0f}M)": (y_true >= medium_max),
    }

    results = {}
    for stratum_name, mask in strata.items():
        n = mask.sum()
        if n == 0:
            results[stratum_name] = {"n": 0, "mape": np.nan, "rmse": np.nan,
                                      "mae": np.nan, "mpe": np.nan, "r2": np.nan}
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        results[stratum_name] = {
            "n": int(n),
            "mape": compute_mape(y_t, y_p),
            "rmse": compute_rmse(y_t, y_p),
            "mae": compute_mae(y_t, y_p),
            "mpe": compute_mpe(y_t, y_p),
            "r2": compute_r2(y_t, y_p),
        }

    return results


def generate_latex_table(all_results: dict):
    """生成LaTeX分层评估表格"""
    print("\n" + "=" * 70)
    print("LaTeX Table: Stratified Evaluation by Project Size")
    print("=" * 70)

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Stratified Evaluation by Project Size}
\label{tab:stratified}
\begin{tabular}{llccccc}
\toprule
Scenario & Stratum & N & MAPE (\%) & RMSE (\$) & MAE (\$) & MPE (\%) \\
\midrule
"""

    for scenario_name, strata_results in all_results.items():
        first = True
        for stratum_name, metrics in strata_results.items():
            name_col = scenario_name if first else ""
            first = False
            n = metrics["n"]
            if n == 0:
                latex += f"{name_col} & {stratum_name} & 0 & - & - & - & - \\\\\n"
            else:
                mape = f"{metrics['mape']*100:.2f}"
                rmse = f"{metrics['rmse']:,.0f}"
                mae = f"{metrics['mae']:,.0f}"
                mpe = f"{metrics['mpe']*100:.2f}"
                latex += f"{name_col} & {stratum_name} & {n} & {mape} & {rmse} & {mae} & {mpe} \\\\\n"
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
    print("STRATIFIED EVALUATION BY PROJECT SIZE")
    print("=" * 70)

    config = load_config("configs/config.yaml")
    predictions = load_predictions()

    if not predictions:
        print("\n[ERROR] No prediction files found in results/")
        print("Please run experiments first to generate prediction files.")
        return

    print(f"\nLoaded predictions for {len(predictions)} scenarios: {list(predictions.keys())}")

    all_results = {}
    for scenario_name, pred_df in predictions.items():
        # 确定列名
        true_col = "True_Value" if "True_Value" in pred_df.columns else "y_true"
        pred_col = "Predicted_Value" if "Predicted_Value" in pred_df.columns else "y_pred"

        if true_col not in pred_df.columns or pred_col not in pred_df.columns:
            print(f"  [WARN] Cannot find required columns in {scenario_name}, skipping")
            continue

        y_true = pred_df[true_col].values
        y_pred = pred_df[pred_col].values

        print(f"\n  {scenario_name}: {len(y_true)} samples")
        strata_results = stratify_by_size(y_true, y_pred, config)
        all_results[scenario_name] = strata_results

        for stratum_name, metrics in strata_results.items():
            if metrics["n"] > 0:
                print(f"    {stratum_name}: N={metrics['n']}, "
                      f"MAPE={metrics['mape']*100:.2f}%, "
                      f"RMSE=${metrics['rmse']:,.0f}, "
                      f"MPE={metrics['mpe']*100:.2f}%")

    # 生成LaTeX表格
    if all_results:
        generate_latex_table(all_results)

    # 保存CSV
    rows = []
    for scenario, strata in all_results.items():
        for stratum, metrics in strata.items():
            row = {"scenario": scenario, "stratum": stratum}
            row.update(metrics)
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        output_path = Path("results") / "stratified_evaluation.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
