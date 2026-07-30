"""Analyze row-wise and stratified prediction gaps between two methods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiment_names import canonical_experiment_key, experiment_display_name


PREDICTION_FILES = {
    "A": "centralized_predictions.csv",
    "A_prime": "centralized_nn_predictions.csv",
    "B": "fedavg_predictions.csv",
    "FEDYOGI": "fedyogi_predictions.csv",
    "COHERENCE_FEDYOGI_TR": "coherence_fedyogi_tr_predictions.csv",
    "LLM_GCA_FEDYOGI_TR": "llm_gca_fedyogi_tr_predictions.csv",
    "VG_FEDYOGI_TR": "vg_fedyogi_tr_predictions.csv",
    "MAS_VG_FEDYOGI_TR": "mas_vg_fedyogi_tr_predictions.csv",
}


def _ape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return np.abs((y - p) / np.maximum(np.abs(y), 1e-9))


def _required_columns(df: pd.DataFrame) -> list[str]:
    columns = ["True_Value", "Predicted_Value"]
    for optional in ("Client", "Project_Size_Stratum"):
        if optional in df.columns:
            columns.append(optional)
    return columns


def _validate_alignment(baseline_df: pd.DataFrame, contender_df: pd.DataFrame) -> None:
    if len(baseline_df) != len(contender_df):
        raise ValueError("Prediction files must contain the same number of rows.")
    for column in ("True_Value", "Client", "Project_Size_Stratum"):
        if column in baseline_df.columns and column in contender_df.columns:
            if not baseline_df[column].reset_index(drop=True).equals(
                contender_df[column].reset_index(drop=True)
            ):
                raise ValueError(f"Prediction files are not aligned on {column}.")


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int, random_seed: int) -> tuple[float, float]:
    if len(values) == 0 or n_bootstrap <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    samples = []
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, len(values), len(values))
        samples.append(float(np.mean(values[indices])))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def _summary_row(
    rowwise: pd.DataFrame,
    group_type: str,
    group_value: str,
    baseline_name: str,
    contender_name: str,
    n_bootstrap: int,
    random_seed: int,
) -> dict:
    diff = rowwise["ape_diff_contender_minus_baseline"].to_numpy(float)
    ci_lower, ci_upper = _bootstrap_ci(diff, n_bootstrap=n_bootstrap, random_seed=random_seed)
    return {
        "baseline": baseline_name,
        "contender": contender_name,
        "group_type": group_type,
        "group_value": group_value,
        "n": int(len(rowwise)),
        "baseline_mape": float(rowwise["baseline_ape"].mean()),
        "contender_mape": float(rowwise["contender_ape"].mean()),
        "mean_ape_diff": float(np.mean(diff)),
        "sum_ape_diff": float(np.sum(diff)),
        "median_ape_diff": float(np.median(diff)),
        "contender_better_count": int(rowwise["contender_better"].sum()),
        "baseline_better_count": int((rowwise["ape_diff_contender_minus_baseline"] > 0).sum()),
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
    }


def build_pairwise_gap_summary(
    baseline_name: str,
    contender_name: str,
    baseline_df: pd.DataFrame,
    contender_df: pd.DataFrame,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return stratified gap summary and row-wise APE differences."""
    _validate_alignment(baseline_df, contender_df)

    baseline = baseline_df.reset_index(drop=True)
    contender = contender_df.reset_index(drop=True)
    rowwise = baseline[_required_columns(baseline)].copy()
    rowwise["baseline"] = baseline_name
    rowwise["contender"] = contender_name
    rowwise["baseline_prediction"] = baseline["Predicted_Value"].astype(float)
    rowwise["contender_prediction"] = contender["Predicted_Value"].astype(float)
    rowwise["baseline_ape"] = _ape(baseline["True_Value"], baseline["Predicted_Value"])
    rowwise["contender_ape"] = _ape(contender["True_Value"], contender["Predicted_Value"])
    rowwise["ape_diff_contender_minus_baseline"] = (
        rowwise["contender_ape"] - rowwise["baseline_ape"]
    )
    rowwise["prediction_diff_contender_minus_baseline"] = (
        rowwise["contender_prediction"] - rowwise["baseline_prediction"]
    )
    rowwise["contender_better"] = rowwise["ape_diff_contender_minus_baseline"] < 0

    rows = [
        _summary_row(
            rowwise,
            group_type="overall",
            group_value="all",
            baseline_name=baseline_name,
            contender_name=contender_name,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
    ]
    for group_type in ("Project_Size_Stratum", "Client"):
        if group_type not in rowwise.columns:
            continue
        for group_value, group_df in rowwise.groupby(group_type, sort=False):
            rows.append(
                _summary_row(
                    group_df,
                    group_type=group_type,
                    group_value=str(group_value),
                    baseline_name=baseline_name,
                    contender_name=contender_name,
                    n_bootstrap=n_bootstrap,
                    random_seed=random_seed,
                )
            )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        overall_n = int(summary.loc[summary["group_type"] == "overall", "n"].iloc[0])
        summary["overall_mean_contribution"] = summary["sum_ape_diff"] / max(overall_n, 1)
    return summary, rowwise


def _prediction_path(results_dir: Path, key: str) -> Path:
    scenario_key = canonical_experiment_key(key)
    if scenario_key not in PREDICTION_FILES:
        raise ValueError(f"Unsupported prediction key: {key}")
    return results_dir / PREDICTION_FILES[scenario_key]


def analyze_prediction_files(
    results_dir: str | Path = "results",
    baseline_key: str = "B",
    contender_key: str = "LLM_GCA_FEDYOGI_TR",
    n_bootstrap: int = 2000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_path = Path(results_dir)
    baseline_key = canonical_experiment_key(baseline_key)
    contender_key = canonical_experiment_key(contender_key)
    baseline_path = _prediction_path(results_path, baseline_key)
    contender_path = _prediction_path(results_path, contender_key)
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    if not contender_path.exists():
        raise FileNotFoundError(contender_path)

    baseline_df = pd.read_csv(baseline_path)
    contender_df = pd.read_csv(contender_path)
    summary, rowwise = build_pairwise_gap_summary(
        baseline_name=experiment_display_name(baseline_key),
        contender_name=experiment_display_name(contender_key),
        baseline_df=baseline_df,
        contender_df=contender_df,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
    )

    summary_path = results_path / "prediction_gap_analysis.csv"
    rowwise_path = results_path / "prediction_gap_rowwise.csv"
    summary.to_csv(summary_path, index=False)
    rowwise.to_csv(rowwise_path, index=False)
    return summary, rowwise


def _print_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("No prediction gap rows produced.")
        return
    columns = [
        "group_type",
        "group_value",
        "n",
        "baseline_mape",
        "contender_mape",
        "mean_ape_diff",
        "overall_mean_contribution",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
    ]
    print(summary[columns].to_string(index=False))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze prediction gaps between two methods.")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--baseline", default="B")
    parser.add_argument("--contender", default="LLM_GCA_FEDYOGI_TR")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    summary, _ = analyze_prediction_files(
        results_dir=args.results_dir,
        baseline_key=args.baseline,
        contender_key=args.contender,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
    )
    _print_summary(summary)
    print(f"\nSaved: {Path(args.results_dir) / 'prediction_gap_analysis.csv'}")
    print(f"Saved: {Path(args.results_dir) / 'prediction_gap_rowwise.csv'}")


if __name__ == "__main__":
    main()
