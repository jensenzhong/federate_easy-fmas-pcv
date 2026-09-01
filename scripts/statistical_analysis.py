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
import hashlib
import json
import math
from pathlib import Path
import re

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from scipy import stats
import yaml

from src.formal_protocol import (
    ANALYSIS_PROTOCOL,
    METHOD_PROMPT_ROLES,
    METHOD_REPETITIONS,
    canonical_json_bytes,
    file_sha256,
    validate_freeze_record,
)
from src.federated_learning.pcv.provider_config import deepseek_provenance
from src.study_manifest import StudyManifest, load_study_manifest

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


def _read_formal_json(path: Path, *, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must contain one JSON object")
    return value


def load_frozen_formal_results(
    results_root: Path,
    manifest: StudyManifest,
    freeze_id: str,
) -> pd.DataFrame:
    """Load only a complete, evaluated, provenance-bound frozen batch."""

    if freeze_id not in manifest.paper_eligible_freeze_ids:
        raise ValueError(f"freeze id is not paper eligible: {freeze_id}")
    if manifest.development_seed in manifest.formal_seeds:
        raise ValueError("development seed 42 is forbidden in formal statistics")
    batch_root = Path(results_root) / "formal" / freeze_id
    project_root = Path(results_root).resolve().parent
    frozen_document = yaml.safe_load(
        (project_root / "configs/formal_frozen.yaml").read_text(encoding="utf-8")
    )
    if type(frozen_document) is not dict:
        raise ValueError("formal frozen configuration is unreadable")
    freeze_payload = validate_freeze_record(
        project_root,
        freeze_id=freeze_id,
        frozen_document=frozen_document,
    )
    evaluation_batch = _read_formal_json(
        batch_root / "EVALUATION_BATCH_COMPLETE.json",
        label="formal evaluation batch",
    )
    expected_count = sum(
        len(METHOD_REPETITIONS[method]) * len(manifest.formal_seeds)
        for method in manifest.formal_methods
    )
    if (
        evaluation_batch.get("schema_version") != 1
        or evaluation_batch.get("status") != "complete"
        or evaluation_batch.get("phase") != "formal_evaluate"
        or evaluation_batch.get("freeze_id") != freeze_id
        or evaluation_batch.get("run_count") != expected_count
        or type(evaluation_batch.get("runs")) is not list
        or len(evaluation_batch["runs"]) != expected_count
    ):
        raise ValueError("formal evaluation batch is incomplete")

    batch_records = {}
    for record in evaluation_batch["runs"]:
        if (
            type(record) is not dict
            or set(record)
            != {
                "method", "training_seed", "llm_rep", "completion_sha256",
                "validation_sha256", "checkpoint_sha256",
                "evaluation_completion_sha256", "locked_test_sha256",
            }
        ):
            raise ValueError("formal evaluation batch run schema mismatch")
        identity = (record["method"], record["training_seed"], record["llm_rep"])
        if identity in batch_records:
            raise ValueError("formal evaluation batch contains duplicate runs")
        batch_records[identity] = record

    rows = []
    for method in manifest.formal_methods:
        if method not in METHOD_REPETITIONS:
            raise ValueError(f"formal result batch contains a legacy method: {method}")
        for seed in manifest.formal_seeds:
            for rep in METHOD_REPETITIONS[method]:
                run_root = batch_root / method / str(seed) / str(rep)
                locked_path = run_root / "locked_test_metrics.json"
                locked = _read_formal_json(locked_path, label="locked-test metrics")
                provenance = _read_formal_json(
                    run_root / "provenance.json", label="formal provenance"
                )
                completion = _read_formal_json(
                    run_root / "EVALUATION_COMPLETE.json",
                    label="formal evaluation completion",
                )
                evaluation_name = completion.get("evaluation_provenance")
                if (
                    type(evaluation_name) is not str
                    or not re.fullmatch(
                        r"evaluation_provenance(?:\.\d{3})?\.json", evaluation_name
                    )
                ):
                    raise ValueError("formal evaluation provenance is missing")
                evaluation_path = run_root / evaluation_name
                evaluation = _read_formal_json(
                    evaluation_path, label="formal evaluation provenance"
                )
                training_completion_path = run_root / "TRAINING_COMPLETE.json"
                training_completion = _read_formal_json(
                    training_completion_path, label="formal training completion"
                )
                validation_path = run_root / "validation_metrics.json"
                validation = _read_formal_json(
                    validation_path, label="formal validation metrics"
                )
                metrics = locked.get("locked_test")
                if (
                    type(metrics) is not dict
                    or set(metrics) != {"sample_count", "mape", "rmse", "mae", "r2"}
                    or type(metrics.get("sample_count")) is not int
                    or metrics["sample_count"] <= 0
                    or any(
                        type(metrics.get(name)) not in {int, float}
                        or not math.isfinite(metrics[name])
                        for name in ("mape", "rmse", "mae", "r2")
                    )
                    or any(metrics[name] < 0 for name in ("mape", "rmse", "mae"))
                    or metrics["r2"] > 1
                ):
                    raise ValueError("formal locked-test metrics are invalid")
                identity = {
                    "method": method,
                    "training_seed": seed,
                    "llm_rep": rep,
                }
                stable_fields = {
                    "schema_version", "method", "training_seed", "llm_rep", "run_id",
                    "freeze_id", "git_commit", "git_dirty", "partition_sha256",
                    "sealed_partition_metadata_sha256", "method_config_sha256",
                    "base_config_sha256", "effective_config_sha256", "prompt_hashes", "deepseek",
                }
                prompt_hashes = provenance.get("prompt_hashes")
                expected_prompts = {
                    role: freeze_payload["prompt_sha256s"][role]
                    for role in METHOD_PROMPT_ROLES[method]
                }
                pause_names = sorted(path.name for path in run_root.glob("PAUSED*.json"))
                if (
                    locked.get("schema_version") != 1
                    or locked.get("phase") != "formal_evaluate"
                    or any(locked.get(key) != value for key, value in identity.items())
                    or provenance.get("phase") != "formal_train"
                    or provenance.get("freeze_id") != freeze_id
                    or provenance.get("schema_version") != 1
                    or provenance.get("run_id") is not None
                    or provenance.get("git_dirty") is not False
                    or provenance.get("locked_test_unlocked") is not False
                    or any(provenance.get(key) != value for key, value in identity.items())
                    or evaluation.get("phase") != "formal_evaluate"
                    or evaluation.get("freeze_id") != freeze_id
                    or evaluation.get("locked_test_unlocked") is not True
                    or any(evaluation.get(key) != value for key, value in identity.items())
                    or set(evaluation) != {*set(provenance), "training_checkpoint_sha256"}
                    or any(
                        evaluation.get(field) != provenance.get(field)
                        for field in stable_fields
                    )
                    or evaluation.get("resume_requested") is not True
                    or locked.get("training_checkpoint_sha256")
                    != evaluation.get("training_checkpoint_sha256")
                    or locked.get("evaluation_provenance_sha256")
                    != file_sha256(evaluation_path)
                    or completion.get("status") != "complete"
                    or completion.get("phase") != "formal_evaluate"
                    or any(completion.get(key) != value for key, value in identity.items())
                    or set(completion)
                    != {"status", "phase", "method", "training_seed", "llm_rep",
                        "last_complete_round", "resolved_pause_reports", "resume_approved",
                        "provenance", "evaluation_provenance", "result_status", "result_file",
                        "result_sha256"}
                    or completion.get("last_complete_round") != 20
                    or completion.get("resolved_pause_reports") != pause_names
                    or completion.get("resume_approved") is not True
                    or completion.get("provenance") != "provenance.json"
                    or completion.get("evaluation_provenance") != evaluation_name
                    or completion.get("result_status") != "complete"
                    or completion.get("result_file") != "locked_test_metrics.json"
                    or completion.get("result_sha256") != file_sha256(locked_path)
                    or training_completion.get("status") != "complete"
                    or training_completion.get("phase") != "formal_train"
                    or any(
                        training_completion.get(key) != value
                        for key, value in identity.items()
                    )
                    or training_completion.get("last_complete_round") != 20
                    or training_completion.get("result_file") != "validation_metrics.json"
                    or training_completion.get("result_sha256") != file_sha256(validation_path)
                    or validation.get("status") != "complete"
                    or validation.get("phase") != "formal_train"
                    or any(validation.get(key) != value for key, value in identity.items())
                    or validation.get("completed_rounds") != 20
                ):
                    raise ValueError("formal result provenance identity mismatch")
                batch_record = batch_records.get((method, seed, rep))
                checkpoint_path = run_root / "last_complete.pt"
                if (
                    batch_record is None
                    or batch_record["completion_sha256"]
                    != file_sha256(training_completion_path)
                    or batch_record["validation_sha256"] != file_sha256(validation_path)
                    or batch_record["checkpoint_sha256"] != file_sha256(checkpoint_path)
                    or batch_record["evaluation_completion_sha256"]
                    != file_sha256(run_root / "EVALUATION_COMPLETE.json")
                    or batch_record["locked_test_sha256"] != file_sha256(locked_path)
                    or locked.get("training_checkpoint_sha256")
                    != batch_record["checkpoint_sha256"]
                    or provenance.get("partition_sha256")
                    != freeze_payload["partition_sha256"]
                    or provenance.get("sealed_partition_metadata_sha256")
                    != freeze_payload["sealed_partition_metadata_sha256"]
                    or provenance.get("method_config_sha256")
                    != freeze_payload["method_config_sha256s"][method]
                    or type(prompt_hashes) is not dict
                    or prompt_hashes != expected_prompts
                    or provenance.get("deepseek")
                    != deepseek_provenance(
                        enabled=bool(manifest.methods[method]["uses_llm"])
                    )
                ):
                    raise ValueError("formal evaluation batch evidence mismatch")
                rows.append(
                    {
                        "freeze_id": freeze_id,
                        "method": method,
                        "training_seed": seed,
                        "llm_rep": rep,
                        "success": True,
                        "test_unlocked": True,
                        "test_sample_count": metrics["sample_count"],
                        "test_mape": float(metrics["mape"]),
                        "test_rmse": float(metrics["rmse"]),
                        "test_mae": float(metrics["mae"]),
                        "test_r2": float(metrics["r2"]),
                        "partition_sha256": provenance.get("partition_sha256"),
                        "config_sha256": provenance.get("effective_config_sha256"),
                        "prompt_bundle_sha256": hashlib.sha256(
                            canonical_json_bytes(prompt_hashes)
                        ).hexdigest(),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["method", "training_seed", "llm_rep"]
    ).reset_index(drop=True)
    if len(frame) != expected_count or set(frame["training_seed"]) != set(manifest.formal_seeds):
        raise ValueError("formal result coverage mismatch")
    if frame.duplicated(["method", "training_seed", "llm_rep"]).any():
        raise ValueError("formal result batch contains duplicate runs")
    if 42 in set(frame["training_seed"]):
        raise ValueError("development seed 42 is forbidden in formal statistics")
    if len(batch_records) != len(frame):
        raise ValueError("formal evaluation batch contains unexpected runs")
    return frame


def aggregate_formal_repetitions(raw: pd.DataFrame) -> pd.DataFrame:
    """Average LLM repetitions within seed before any paired inference."""

    required = {
        "method", "training_seed", "llm_rep", "test_mape", "test_rmse",
        "test_mae", "test_r2",
    }
    if missing := required - set(raw.columns):
        raise ValueError(f"formal results missing columns: {sorted(missing)}")
    rows = []
    for (method, seed), group in raw.groupby(["method", "training_seed"], sort=False):
        expected_reps = set(METHOD_REPETITIONS.get(method, ()))
        if group["llm_rep"].duplicated().any():
            raise ValueError(f"duplicate repetition for {method} seed {seed}")
        if (
            not expected_reps
            or len(group) != len(expected_reps)
            or set(group["llm_rep"].astype(int)) != expected_reps
        ):
            raise ValueError(f"repetition coverage mismatch for {method} seed {seed}")
        row = {"method": method, "training_seed": int(seed), "n_repetitions": len(group)}
        for metric in ("test_mape", "test_rmse", "test_mae", "test_r2"):
            values = group[metric].astype(float)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite formal metric: {metric}")
            if metric in {"test_mape", "test_rmse", "test_mae"} and (values < 0).any():
                raise ValueError(f"formal metric must be non-negative: {metric}")
            if metric == "test_r2" and (values > 1).any():
                raise ValueError("formal R2 cannot exceed one")
            row[metric] = float(values.mean())
            row[f"{metric}_within_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "training_seed"]).reset_index(drop=True)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _paired_metric_statistics(
    fmas_values: np.ndarray,
    comparator_values: np.ndarray,
    *,
    higher_is_better: bool,
) -> dict:
    improvement = (
        fmas_values - comparator_values
        if higher_is_better
        else comparator_values - fmas_values
    )
    t_result = stats.ttest_rel(fmas_values, comparator_values)
    try:
        wilcoxon = stats.wilcoxon(fmas_values, comparator_values)
        w_stat = float(wilcoxon.statistic)
        w_p = float(wilcoxon.pvalue)
    except ValueError:
        w_stat = math.nan
        w_p = math.nan
    improvement_std = float(np.std(improvement, ddof=1))
    improvement_mean = float(np.mean(improvement))
    if len(improvement) > 1:
        margin = float(
            stats.t.ppf(0.975, len(improvement) - 1)
            * stats.sem(improvement)
        )
    else:
        margin = math.nan
    wins = (
        fmas_values > comparator_values
        if higher_is_better
        else fmas_values < comparator_values
    )
    return {
        "paired_n": int(len(improvement)),
        "seed_wins": int(np.sum(wins)),
        "fmas_mean": float(np.mean(fmas_values)),
        "comparator_mean": float(np.mean(comparator_values)),
        "mean_improvement": improvement_mean,
        "mean_improvement_ci95": [
            improvement_mean - margin,
            improvement_mean + margin,
        ],
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p_value": float(t_result.pvalue),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_value": w_p,
        "paired_effect_size": improvement_mean / improvement_std
        if improvement_std > 0
        else 0.0,
    }


def analyze_fmas_formal_results(per_seed: pd.DataFrame) -> dict:
    """Apply the preregistered five-seed paired comparison family."""

    fmas = per_seed[per_seed["method"] == "FMAS_PCV_FEDYOGI"]
    if len(fmas) != ANALYSIS_PROTOCOL["formal_seed_count"]:
        raise ValueError("FMAS formal analysis requires exactly five seed observations")
    comparisons = []
    raw_p_values = []
    for comparator in (
        "FEDAVG_STRICT",
        "FEDYOGI_STRICT",
        "DPCV_FEDYOGI",
        "SA_PCV_FEDYOGI",
    ):
        metric_columns = ["test_mape", "test_rmse", "test_mae", "test_r2"]
        paired = fmas[["training_seed", *metric_columns]].merge(
            per_seed[per_seed["method"] == comparator][["training_seed", *metric_columns]],
            on="training_seed",
            suffixes=("_fmas", "_baseline"),
            validate="one_to_one",
        ).sort_values("training_seed")
        if len(paired) != ANALYSIS_PROTOCOL["formal_seed_count"]:
            raise ValueError(f"paired seed coverage mismatch for {comparator}")
        metric_results = {}
        for metric in metric_columns:
            metric_results[metric] = _paired_metric_statistics(
                paired[f"{metric}_fmas"].to_numpy(dtype=float),
                paired[f"{metric}_baseline"].to_numpy(dtype=float),
                higher_is_better=metric == "test_r2",
            )
        mape = metric_results["test_mape"]
        raw_p = mape["paired_t_p_value"]
        raw_p_values.append(raw_p)
        baseline_values = paired["test_mape_baseline"].to_numpy(dtype=float)
        fmas_values = paired["test_mape_fmas"].to_numpy(dtype=float)
        comparisons.append(
            {
                "comparison": f"FMAS_PCV_FEDYOGI_vs_{comparator}",
                "comparator": comparator,
                "paired_n": mape["paired_n"],
                "seed_wins": mape["seed_wins"],
                "fmas_mean_mape": float(np.mean(fmas_values)),
                "comparator_mean_mape": float(np.mean(baseline_values)),
                "mean_relative_improvement": float(
                    np.mean((baseline_values - fmas_values) / baseline_values)
                ),
                "mean_mape_improvement_ci95": mape["mean_improvement_ci95"],
                "paired_t_statistic": mape["paired_t_statistic"],
                "paired_t_p_value": raw_p,
                "wilcoxon_statistic": mape["wilcoxon_statistic"],
                "wilcoxon_p_value": mape["wilcoxon_p_value"],
                "paired_effect_size": mape["paired_effect_size"],
                "metrics": metric_results,
            }
        )
    adjusted = _holm_adjust(raw_p_values)
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_p_value"] = value
    primary = next(
        row
        for row in comparisons
        if row["comparator"] == ANALYSIS_PROTOCOL["primary_strict_baseline"]
    )
    stable = primary["seed_wins"] >= ANALYSIS_PROTOCOL["stable_improvement_min_seed_wins"]
    significant = stable and primary["holm_adjusted_p_value"] < ANALYSIS_PROTOCOL["alpha"]
    if significant:
        claim_status = "significant_improvement"
    elif stable:
        claim_status = "stable_improvement"
    elif primary["fmas_mean_mape"] < primary["comparator_mean_mape"]:
        claim_status = "mean_improvement_trend"
    else:
        claim_status = "no_supported_improvement"
    return {
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "comparisons": comparisons,
        "stable_improvement": stable,
        "significant_improvement": significant,
        "claim_status": claim_status,
    }


def run_frozen_formal_statistics(
    *,
    results_root: Path,
    freeze_id: str,
    manifest_path: Path,
) -> dict:
    manifest = load_study_manifest(Path(manifest_path))
    raw = load_frozen_formal_results(Path(results_root), manifest, freeze_id)
    per_seed = aggregate_formal_repetitions(raw)
    report = analyze_fmas_formal_results(per_seed)
    output_root = Path(results_root) / "paper" / freeze_id / "statistics"
    paths = {
        "raw": output_root / "formal_raw_runs.csv",
        "per_seed": output_root / "formal_per_seed.csv",
        "report": output_root / "formal_analysis.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite frozen statistical artifacts: {existing}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    raw.to_csv(paths["raw"], index=False)
    per_seed.to_csv(paths["per_seed"], index=False)
    paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Statistical analysis of multi-seed results")
    parser.add_argument("--results_file", type=str,
                        default="results/multi_seed/all_results.csv",
                        help="Path to multi-seed results CSV")
    parser.add_argument(
        "--freeze-id",
        help="Analyze one complete paper-eligible FMAS formal batch",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root containing formal/<freeze-id>",
    )
    parser.add_argument(
        "--study-manifest",
        type=Path,
        default=Path("study_manifest.yaml"),
    )
    args = parser.parse_args(argv)

    if args.freeze_id:
        report = run_frozen_formal_statistics(
            results_root=args.results_root,
            freeze_id=args.freeze_id,
            manifest_path=args.study_manifest,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"\n[ERROR] Results file not found: {results_path}")
        print("Please run multi-seed experiments first:")
        print("  python scripts/run_multi_seed.py")
        return 1

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
    return 0


if __name__ == "__main__":
    main()
