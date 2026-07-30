"""Summarize validation-MPE calibration gains across methods and seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiment_names import (
    EXPERIMENT_ORDER,
    canonical_experiment_key,
    experiment_display_name,
)


def _success_mask(df: pd.DataFrame) -> pd.Series:
    if "success" not in df.columns:
        return pd.Series(True, index=df.index)
    success = df["success"]
    if success.dtype == bool:
        return success
    return success.astype(str).str.lower().isin(["true", "1", "yes", "ok"])


def compute_calibration_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw-to-corrected metric deltas for each method."""
    required = {"scenario_key", "test_mape", "test_mape_corrected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    working = df.copy()
    working["scenario_key"] = working["scenario_key"].map(canonical_experiment_key)
    working = working[_success_mask(working)].copy()
    working = working.dropna(subset=["test_mape", "test_mape_corrected"]).copy()
    working["mape_delta_corrected_minus_raw"] = (
        working["test_mape_corrected"].astype(float) - working["test_mape"].astype(float)
    )
    working["mape_relative_delta_corrected_minus_raw"] = (
        working["mape_delta_corrected_minus_raw"] / working["test_mape"].astype(float)
    )

    optional_pairs = [
        ("test_rmse", "test_rmse_corrected", "rmse_delta_corrected_minus_raw"),
        ("test_mae", "test_mae_corrected", "mae_delta_corrected_minus_raw"),
        ("test_mpe", "test_mpe_corrected", "mpe_delta_corrected_minus_raw"),
        ("test_r2", "test_r2_corrected", "r2_delta_corrected_minus_raw"),
    ]
    for raw_col, corrected_col, delta_col in optional_pairs:
        if raw_col in working.columns and corrected_col in working.columns:
            working[delta_col] = working[corrected_col].astype(float) - working[raw_col].astype(float)

    rows = []
    for scenario_key, group in working.groupby("scenario_key", sort=False):
        row = {
            "scenario_key": scenario_key,
            "scenario": experiment_display_name(scenario_key),
            "n_runs": int(len(group)),
            "mean_raw_mape": float(group["test_mape"].mean()),
            "mean_corrected_mape": float(group["test_mape_corrected"].mean()),
            "mean_mape_delta": float(group["mape_delta_corrected_minus_raw"].mean()),
            "std_mape_delta": float(group["mape_delta_corrected_minus_raw"].std()),
            "calibration_improved_mape_runs": int((group["mape_delta_corrected_minus_raw"] < 0).sum()),
        }
        row["mean_mape_relative_delta"] = row["mean_mape_delta"] / max(row["mean_raw_mape"], 1e-12)
        for delta_col in [
            "rmse_delta_corrected_minus_raw",
            "mae_delta_corrected_minus_raw",
            "mpe_delta_corrected_minus_raw",
            "r2_delta_corrected_minus_raw",
        ]:
            if delta_col in group.columns:
                row[f"mean_{delta_col}"] = float(group[delta_col].mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        order = {key: index for index, key in enumerate(EXPERIMENT_ORDER)}
        summary["_order"] = summary["scenario_key"].map(order).fillna(len(order))
        summary = summary.sort_values("_order", kind="stable").drop(columns=["_order"]).reset_index(drop=True)
    return summary


def run_calibration_analysis(
    results_file: str | Path = "results/multi_seed/all_results.csv",
    output_path: str | Path = "results/multi_seed/calibration_summary.csv",
) -> pd.DataFrame:
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    df = pd.read_csv(results_path)
    summary = compute_calibration_summary(df)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze validation-based calibration gains.")
    parser.add_argument("--results_file", default="results/multi_seed/all_results.csv")
    parser.add_argument("--output_path", default="results/multi_seed/calibration_summary.csv")
    args = parser.parse_args(argv)

    summary = run_calibration_analysis(args.results_file, args.output_path)
    print(summary.to_string(index=False))
    print(f"\nSaved calibration summary to {args.output_path}")


if __name__ == "__main__":
    main()
