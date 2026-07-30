"""Pilot FedYogi server learning-rate grid on a non-main seed."""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


project_root = Path(__file__).parent.parent


def _format_float_for_name(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace(".", "p").replace("-", "m")


def run_candidate(
    server_lr: float,
    clip_norm: float | None,
    max_coordinate_step_ratio: float,
    seed: int,
    output_dir: Path,
) -> dict:
    clip_suffix = "" if clip_norm is None else f"_clip_{_format_float_for_name(clip_norm)}"
    trust_suffix = (
        "" if max_coordinate_step_ratio == 1.0
        else f"_trust_{_format_float_for_name(max_coordinate_step_ratio)}"
    )
    output_prefix = f"adaptive_pilot_fedyogi_lr_{_format_float_for_name(server_lr)}{clip_suffix}{trust_suffix}"
    cmd = [
        sys.executable,
        "experiments/scenario_C_llm.py",
        "--num_rounds", "20",
        "--strategy", "size_only",
        "--server_optimizer", "fedyogi",
        "--server_lr", str(server_lr),
        "--max_coordinate_step_ratio", str(max_coordinate_step_ratio),
        "--output_prefix", output_prefix,
        "--method_key", "FEDYOGI",
        "--seed", str(seed),
    ]
    if clip_norm is not None:
        cmd.extend(["--update_clip_norm", str(clip_norm)])

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{output_prefix}.log"
    log_path.write_text(
        f"Command: {' '.join(cmd)}\nReturn code: {result.returncode}\nElapsed: {elapsed:.1f}s\n\n"
        f"{result.stdout or ''}\n--- STDERR ---\n{result.stderr or ''}",
        encoding="utf-8",
    )

    source_result = project_root / "results" / f"{output_prefix}_results.csv"
    pilot_result = output_dir / f"{output_prefix}_results.csv"
    row = {
        "server_lr": server_lr,
        "update_clip_norm": clip_norm,
        "max_coordinate_step_ratio": max_coordinate_step_ratio,
        "seed": seed,
        "success": result.returncode == 0,
        "elapsed_seconds": elapsed,
        "log_file": str(log_path),
        "result_file": str(pilot_result),
    }
    if source_result.exists():
        shutil.copy2(source_result, pilot_result)
        metrics = pd.read_csv(source_result).iloc[-1].to_dict()
        row.update(metrics)
    else:
        row["result_read_error"] = f"Missing {source_result}"
    return row


def select_recommendation(
    df: pd.DataFrame,
    expected_seed_count: int,
    fallback_ratio: float,
) -> tuple[dict, pd.DataFrame]:
    if df.empty or "best_val_mape" not in df.columns:
        raise RuntimeError(
            "Pilot summary is empty or missing best_val_mape. "
            "No pilot configuration completed successfully."
        )

    work = df.copy()
    work["best_val_mape"] = pd.to_numeric(work["best_val_mape"], errors="coerce")
    work["success_bool"] = work.get("success", True).astype(str).str.lower().isin(["true", "1", "yes", "ok"])
    finite_mask = np.isfinite(work["best_val_mape"].to_numpy(dtype=float))
    work = work[work["success_bool"] & finite_mask].copy()
    if work.empty:
        raise RuntimeError(
            "Pilot summary contains no successful rows with finite best_val_mape. "
            "No pilot configuration completed successfully."
        )

    work["update_clip_norm_key"] = work["update_clip_norm"].where(work["update_clip_norm"].notna(), "none")
    group_cols = ["server_lr", "update_clip_norm_key", "max_coordinate_step_ratio"]
    summary = (
        work.groupby(group_cols, dropna=False)
        .agg(
            best_val_mape_mean=("best_val_mape", "mean"),
            best_val_mape_std=("best_val_mape", "std"),
            best_val_mape_min=("best_val_mape", "min"),
            best_val_mape_max=("best_val_mape", "max"),
            n_success=("seed", "count"),
        )
        .reset_index()
    )
    summary["update_clip_norm"] = summary["update_clip_norm_key"].apply(
        lambda value: None if value == "none" else float(value)
    )
    summary = summary.drop(columns=["update_clip_norm_key"])

    complete = summary[summary["n_success"] >= expected_seed_count]
    if complete.empty:
        raise RuntimeError(
            "No pilot configuration completed all expected seeds. "
            "No pilot configuration completed successfully for formal freezing."
        )

    best = complete.sort_values(
        ["best_val_mape_mean", "best_val_mape_std", "server_lr"],
        na_position="last",
    ).iloc[0]

    recommendation = {
        "selected_server_lr": float(best["server_lr"]),
        "selected_update_clip_norm": best.get("update_clip_norm"),
        "selected_max_coordinate_step_ratio": float(best["max_coordinate_step_ratio"]),
        "selection_metric": "mean_best_val_mape",
        "selected_best_val_mape_mean": float(best["best_val_mape_mean"]),
        "selected_best_val_mape_std": (
            None if pd.isna(best["best_val_mape_std"]) else float(best["best_val_mape_std"])
        ),
        "selected_n_success": int(best["n_success"]),
        "expected_seed_count": int(expected_seed_count),
        "fallback_to_default": False,
    }
    return recommendation, summary


def main():
    parser = argparse.ArgumentParser(description="Run FedYogi server_lr pilot grid.")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--server_lrs", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--max_coordinate_step_ratio", type=float, default=None)
    parser.add_argument(
        "--max_coordinate_step_ratios",
        nargs="*",
        type=float,
        default=None,
        help="One or more trust-region max coordinate step ratios to evaluate.",
    )
    parser.add_argument(
        "--clip_norms",
        nargs="*",
        type=str,
        default=["none"],
        help="Optional update clip norms. Use 'none' for no clipping.",
    )
    parser.add_argument("--output_dir", type=str, default="results/adaptive_pilot")
    args = parser.parse_args()

    clip_norms = []
    for value in args.clip_norms:
        if str(value).lower() in {"none", "null", "nan"}:
            clip_norms.append(None)
        else:
            clip_norms.append(float(value))

    output_dir = project_root / args.output_dir
    seeds = args.seeds if args.seeds else [args.seed]
    step_ratios = args.max_coordinate_step_ratios
    if step_ratios is None:
        step_ratios = [args.max_coordinate_step_ratio if args.max_coordinate_step_ratio is not None else 1.0]

    rows = []
    for seed in seeds:
        for server_lr in args.server_lrs:
            for max_coordinate_step_ratio in step_ratios:
                for clip_norm in clip_norms:
                    print(
                        f"\n[Pilot] FedYogi server_lr={server_lr}, "
                        f"max_coordinate_step_ratio={max_coordinate_step_ratio}, "
                        f"clip_norm={clip_norm}, seed={seed}"
                    )
                    row = run_candidate(
                        server_lr=server_lr,
                        clip_norm=clip_norm,
                        max_coordinate_step_ratio=max_coordinate_step_ratio,
                        seed=seed,
                        output_dir=output_dir,
                    )
                    rows.append(row)
                    status = "OK" if row.get("success") else "FAIL"
                    print(
                        f"  [{status}] test_mape={row.get('test_mape', 'N/A')}, "
                        f"best_val_mape={row.get('best_val_mape', 'N/A')}"
                    )

    df = pd.DataFrame(rows)
    summary_path = output_dir / "pilot_summary.csv"
    df.to_csv(summary_path, index=False)

    recommendation, group_summary = select_recommendation(
        df,
        expected_seed_count=len(seeds),
        fallback_ratio=step_ratios[0],
    )
    recommendation["seeds"] = ",".join(str(seed) for seed in seeds)
    group_summary_path = output_dir / "pilot_group_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)
    rec_path = output_dir / "pilot_recommendation.csv"
    pd.DataFrame([recommendation]).to_csv(rec_path, index=False)
    print(f"\nSaved pilot summary: {summary_path}")
    print(f"Saved pilot group summary: {group_summary_path}")
    print(f"Saved pilot recommendation: {rec_path}")


if __name__ == "__main__":
    main()
