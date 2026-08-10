"""
多种子批量运行脚本 (Multi-Seed Runner)

对每个方法实验 × 5个种子运行实验
种子列表: [42, 123, 456, 789, 2024]

用法:
    python scripts/run_multi_seed.py                    # 运行全部
    python scripts/run_multi_seed.py --scenarios B C    # 仅运行指定内部实验键（显示为方法名）
    python scripts/run_multi_seed.py --seeds 42 123     # 仅运行指定种子
"""

import sys
import os
import subprocess
import argparse
import time
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.experiment_names import (
    EXPERIMENT_ORDER,
    canonical_experiment_key,
    experiment_display_name,
)
from src.utils import load_config

# Historical compatibility only. New strict-paper methods have one canonical
# entry point: experiments/run_strict_federated.py.
NEW_FORMAL_METHODS = frozenset(
    {
        "FEDAVG_STRICT",
        "FEDYOGI_STRICT",
        "DPCV_FEDYOGI",
        "SA_PCV_FEDYOGI",
        "FMAS_PCV_FEDYOGI",
    }
)


def reject_new_formal_methods(scenarios) -> None:
    requested = set(scenarios)
    forbidden = sorted(requested & NEW_FORMAL_METHODS)
    if forbidden:
        raise RuntimeError(
            "FMAS formal methods must use experiments/run_strict_federated.py "
            "and study_manifest.yaml"
        )


DEFAULT_SEEDS = [42, 123, 456, 789, 2024]
DEFAULT_SCENARIOS = [
    "A",
    "A_prime",
    "B_STRICT",
    "FEDYOGI",
    "STRICT_COHERENCE_FEDYOGI_TR",
    "LLM_STRICT_GCA_FEDYOGI_TR",
]
PILOT_RECOMMENDATION_PATH = project_root / "results" / "adaptive_pilot" / "pilot_recommendation.csv"
ADAPTIVE_SCENARIOS = {
    "FEDYOGI",
    "FEDYOGI_STRICT",
    "VG_FEDYOGI_TR",
    "MAS_VG_FEDYOGI_TR",
    "COHERENCE_FEDYOGI_TR",
    "LLM_GCA_FEDYOGI_TR",
    "STRICT_COHERENCE_FEDYOGI_TR",
    "LLM_STRICT_GCA_FEDYOGI_TR",
    "VP_GCA_FEDYOGI_TR",
    "LLM_VP_GCA_FEDYOGI_TR",
}

RESULT_FILES = {
    "A": "centralized_results.csv",
    "A_prime": "centralized_nn_results.csv",
    "B": "fedavg_results.csv",
    "B_STRICT": "fedavg_strict_results.csv",
    "C": "scenario_c_results.csv",
    "FEDYOGI": "fedyogi_results.csv",
    "FEDYOGI_STRICT": "fedyogi_strict_results.csv",
    "VG_FEDYOGI_TR": "vg_fedyogi_tr_results.csv",
    "MAS_VG_FEDYOGI_TR": "mas_vg_fedyogi_tr_results.csv",
    "COHERENCE_FEDYOGI_TR": "coherence_fedyogi_tr_results.csv",
    "LLM_GCA_FEDYOGI_TR": "llm_gca_fedyogi_tr_results.csv",
    "STRICT_COHERENCE_FEDYOGI_TR": "strict_coherence_fedyogi_tr_results.csv",
    "LLM_STRICT_GCA_FEDYOGI_TR": "llm_strict_gca_fedyogi_tr_results.csv",
    "VP_GCA_FEDYOGI_TR": "vp_gca_fedyogi_tr_results.csv",
    "LLM_VP_GCA_FEDYOGI_TR": "llm_vp_gca_fedyogi_tr_results.csv",
}

SCENARIO_CONFIGS = {
    "A": {
        "name": experiment_display_name("A"),
        "script": "experiments/scenario_A_centralized.py",
        "args": [],
        "seed_arg": "--seed",
    },
    "A_prime": {
        "name": experiment_display_name("A_prime"),
        "script": "experiments/scenario_A_prime.py",
        "args": [],
        "seed_arg": "--seed",
    },
    "B": {
        "name": experiment_display_name("B"),
        "script": "experiments/scenario_B_fedavg.py",
        "args": ["--num_rounds", "20"],
        "seed_arg": "--seed",
    },
    "B_STRICT": {
        "name": experiment_display_name("B_STRICT"),
        "script": "experiments/scenario_B_fedavg.py",
        "args": [
            "--num_rounds", "20",
            "--strict_no_server_validation",
            "--output_prefix", "fedavg_strict",
            "--method_key", "B_STRICT",
        ],
        "seed_arg": "--seed",
    },
    "C": {
        "name": experiment_display_name("C"),
        "script": "experiments/scenario_C_llm.py",
        "args": ["--num_rounds", "20", "--use_llm"],
        "seed_arg": "--seed",
    },
    "FEDYOGI": {
        "name": experiment_display_name("FEDYOGI"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--strategy", "size_only",
            "--server_optimizer", "fedyogi",
            "--output_prefix", "fedyogi",
            "--method_key", "FEDYOGI",
        ],
        "seed_arg": "--seed",
    },
    "FEDYOGI_STRICT": {
        "name": experiment_display_name("FEDYOGI_STRICT"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--strategy", "size_only",
            "--server_optimizer", "fedyogi",
            "--strict_no_server_validation",
            "--output_prefix", "fedyogi_strict",
            "--method_key", "FEDYOGI_STRICT",
        ],
        "seed_arg": "--seed",
    },
    "VG_FEDYOGI_TR": {
        "name": experiment_display_name("VG_FEDYOGI_TR"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--strategy", "size_only",
            "--server_optimizer", "fedyogi",
            "--adaptive_mode", "validation_guided",
            "--output_prefix", "vg_fedyogi_tr",
            "--method_key", "VG_FEDYOGI_TR",
        ],
        "seed_arg": "--seed",
    },
    "MAS_VG_FEDYOGI_TR": {
        "name": experiment_display_name("MAS_VG_FEDYOGI_TR"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--use_llm",
            "--temperature", "0",
            "--server_optimizer", "fedyogi",
            "--adaptive_mode", "mas_validation_guided",
            "--llm_score_tolerance", "0.003",
            "--output_prefix", "mas_vg_fedyogi_tr",
            "--method_key", "MAS_VG_FEDYOGI_TR",
        ],
        "seed_arg": "--seed",
    },
    "COHERENCE_FEDYOGI_TR": {
        "name": experiment_display_name("COHERENCE_FEDYOGI_TR"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--strategy", "size_only",
            "--server_optimizer", "fedyogi",
            "--adaptive_mode", "coherence_guided",
            "--output_prefix", "coherence_fedyogi_tr",
            "--method_key", "COHERENCE_FEDYOGI_TR",
        ],
        "seed_arg": "--seed",
    },
    "LLM_GCA_FEDYOGI_TR": {
        "name": experiment_display_name("LLM_GCA_FEDYOGI_TR"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--use_llm",
            "--temperature", "0",
            "--server_optimizer", "fedyogi",
            "--adaptive_mode", "llm_generative_coherence",
            "--output_prefix", "llm_gca_fedyogi_tr",
            "--method_key", "LLM_GCA_FEDYOGI_TR",
        ],
        "seed_arg": "--seed",
    },
    "STRICT_COHERENCE_FEDYOGI_TR": {
        "name": experiment_display_name("STRICT_COHERENCE_FEDYOGI_TR"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--strategy", "size_only",
            "--server_optimizer", "fedyogi",
            "--adaptive_mode", "strict_coherence_guided",
            "--output_prefix", "strict_coherence_fedyogi_tr",
            "--method_key", "STRICT_COHERENCE_FEDYOGI_TR",
        ],
        "seed_arg": "--seed",
    },
    "LLM_STRICT_GCA_FEDYOGI_TR": {
        "name": experiment_display_name("LLM_STRICT_GCA_FEDYOGI_TR"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--use_llm",
            "--temperature", "0",
            "--strategy", "size_only",
            "--server_optimizer", "fedyogi",
            "--adaptive_mode", "llm_strict_generative_coherence",
            "--output_prefix", "llm_strict_gca_fedyogi_tr",
            "--method_key", "LLM_STRICT_GCA_FEDYOGI_TR",
        ],
        "seed_arg": "--seed",
    },
    "VP_GCA_FEDYOGI_TR": {
        "name": experiment_display_name("VP_GCA_FEDYOGI_TR"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--strategy", "size_only",
            "--server_optimizer", "fedyogi",
            "--adaptive_mode", "validation_preview_gca",
            "--output_prefix", "vp_gca_fedyogi_tr",
            "--method_key", "VP_GCA_FEDYOGI_TR",
        ],
        "seed_arg": "--seed",
    },
    "LLM_VP_GCA_FEDYOGI_TR": {
        "name": experiment_display_name("LLM_VP_GCA_FEDYOGI_TR"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--use_llm",
            "--temperature", "0",
            "--server_optimizer", "fedyogi",
            "--adaptive_mode", "llm_validation_preview_generative",
            "--llm_score_tolerance", "0.003",
            "--output_prefix", "llm_vp_gca_fedyogi_tr",
            "--method_key", "LLM_VP_GCA_FEDYOGI_TR",
        ],
        "seed_arg": "--seed",
    },
    "MAS_ADAPTIVE": {
        "name": experiment_display_name("MAS_ADAPTIVE"),
        "script": "experiments/scenario_C_llm.py",
        "args": [
            "--num_rounds", "20",
            "--use_llm",
            "--temperature", "0",
            "--server_optimizer", "fedyogi",
            "--output_prefix", "mas_adaptive",
            "--method_key", "MAS_ADAPTIVE",
        ],
        "seed_arg": "--seed",
    },
}


def _format_arg_value(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.10g}"
    return str(value)


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return str(value).lower() in {"", "nan", "none", "null"}


def _set_cli_arg(args: list[str], flag: str, value) -> list[str]:
    updated = list(args)
    value_str = _format_arg_value(value)
    if flag in updated:
        index = updated.index(flag)
        if index + 1 < len(updated):
            updated[index + 1] = value_str
        else:
            updated.append(value_str)
    else:
        updated.extend([flag, value_str])
    return updated


def _remove_cli_arg(args: list[str], flag: str) -> list[str]:
    updated = []
    skip_next = False
    for item in args:
        if skip_next:
            skip_next = False
            continue
        if item == flag:
            skip_next = True
            continue
        updated.append(item)
    return updated


def adaptive_args_from_recommendation(base_args: list[str], recommendation: dict | None) -> list[str]:
    args = list(base_args)
    if not recommendation:
        return args

    if not _is_missing(recommendation.get("selected_server_lr")):
        args = _set_cli_arg(args, "--server_lr", recommendation["selected_server_lr"])
    if not _is_missing(recommendation.get("selected_max_coordinate_step_ratio")):
        args = _set_cli_arg(
            args,
            "--max_coordinate_step_ratio",
            recommendation["selected_max_coordinate_step_ratio"],
        )

    args = _remove_cli_arg(args, "--update_clip_norm")
    clip_norm = recommendation.get("selected_update_clip_norm")
    if not _is_missing(clip_norm):
        args = _set_cli_arg(args, "--update_clip_norm", clip_norm)
    return args


def load_adaptive_pilot_recommendation(path: Path = PILOT_RECOMMENDATION_PATH) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def formal_adaptive_args(base_args: list[str], recommendation: dict | None) -> list[str]:
    """Apply frozen validation-only pilot parameters for formal FedYogi-TR runs."""
    if not recommendation:
        raise RuntimeError(
            f"Adaptive formal runs require {PILOT_RECOMMENDATION_PATH}. "
            "Run scripts/run_adaptive_pilot.py first."
        )
    if _is_missing(recommendation.get("selected_server_lr")):
        raise RuntimeError(f"{PILOT_RECOMMENDATION_PATH} is missing selected_server_lr.")
    if _is_missing(recommendation.get("selected_max_coordinate_step_ratio")):
        raise RuntimeError(
            f"{PILOT_RECOMMENDATION_PATH} is missing selected_max_coordinate_step_ratio."
        )
    if _is_missing(recommendation.get("selected_n_success")):
        raise RuntimeError(f"{PILOT_RECOMMENDATION_PATH} is missing selected_n_success.")
    if _is_missing(recommendation.get("expected_seed_count")):
        raise RuntimeError(f"{PILOT_RECOMMENDATION_PATH} is missing expected_seed_count.")

    n_success = int(float(recommendation["selected_n_success"]))
    expected_seed_count = int(float(recommendation["expected_seed_count"]))
    if n_success < expected_seed_count:
        raise RuntimeError(
            f"{PILOT_RECOMMENDATION_PATH} selected_n_success={n_success} is below "
            f"expected_seed_count={expected_seed_count}."
        )
    return adaptive_args_from_recommendation(base_args, recommendation)


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return ""
    return (result.stdout or "").strip() if result.returncode == 0 else ""


def current_code_metadata() -> dict:
    """Return lightweight code provenance for experiment CSV rows."""
    commit = _git_output(["rev-parse", "--short", "HEAD"]) or "unknown"
    dirty = bool(_git_output(["status", "--porcelain"]))
    return {"code_commit": commit, "code_dirty": dirty}


def _read_run_metadata() -> dict:
    config = load_config("configs/config.yaml")
    llm_config = config.get("scene_c", {}).get("llm", {})
    return {
        "split_seed": config.get("preprocessing", {}).get("random_seed"),
        "llm_provider": llm_config.get("default_provider"),
        "llm_temperature": llm_config.get("temperature"),
    }


def read_scenario_result(
    scenario_key: str,
    seed: int,
    output_dir: str,
    elapsed_seconds: float,
    success: bool,
    command: list[str],
    code_commit: str,
    code_dirty: bool,
    split_seed: int | None,
    llm_provider: str | None,
    llm_temperature: float | None,
) -> dict:
    """Read the experiment CSV as source of truth and attach run metadata."""
    result_path = Path(output_dir) / RESULT_FILES[scenario_key]
    if not result_path.exists():
        raise FileNotFoundError(f"Result CSV not found for {scenario_key}: {result_path}")

    df = pd.read_csv(result_path)
    if df.empty:
        raise ValueError(f"Result CSV is empty for {scenario_key}: {result_path}")

    row = df.iloc[-1].to_dict()
    csv_llm_provider = row.get("llm_provider", llm_provider)
    csv_llm_temperature = row.get("llm_temperature", llm_temperature)
    row.update({
        "scenario": experiment_display_name(scenario_key),
        "scenario_key": scenario_key,
        "seed": seed,
        "split_seed": split_seed,
        "elapsed_seconds": elapsed_seconds,
        "success": success,
        "command": " ".join(command),
        "result_file": str(result_path),
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "llm_provider": csv_llm_provider,
        "llm_temperature": csv_llm_temperature,
    })
    return row


def _snapshot_seed_result(result_file: str, scenario_key: str, seed: int, output_dir: str) -> str | None:
    source = Path(result_file)
    if not source.exists():
        return None
    snapshot_path = Path(output_dir) / "multi_seed" / f"{scenario_key}_seed{seed}_result.csv"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    if df.empty:
        return None
    df.tail(1).to_csv(snapshot_path, index=False)
    return str(snapshot_path)


def read_seed_snapshot_result(
    scenario_key: str,
    seed: int,
    output_dir: str,
    code_commit: str,
    code_dirty: bool,
    split_seed: int | None,
    llm_provider: str | None,
    llm_temperature: float | None,
) -> dict | None:
    snapshot_path = Path(output_dir) / "multi_seed" / f"{scenario_key}_seed{seed}_result.csv"
    if not snapshot_path.exists():
        return None

    df = pd.read_csv(snapshot_path)
    if df.empty:
        return None

    row = df.iloc[-1].to_dict()
    row.update({
        "scenario": experiment_display_name(scenario_key),
        "scenario_key": scenario_key,
        "seed": seed,
        "split_seed": split_seed,
        "success": True,
        "code_commit": code_commit,
        "code_dirty": code_dirty,
        "llm_provider": row.get("llm_provider", llm_provider),
        "llm_temperature": row.get("llm_temperature", llm_temperature),
        "seed_result_file": str(snapshot_path),
        "source": "seed_result_snapshot",
    })
    return row


def run_scenario(scenario_key: str, seed: int, output_dir: str) -> dict:
    """运行单个场景单个种子的实验"""
    config = SCENARIO_CONFIGS[scenario_key]
    script = config["script"]
    args = list(config["args"])
    if scenario_key in ADAPTIVE_SCENARIOS:
        args = formal_adaptive_args(args, load_adaptive_pilot_recommendation())

    if config["seed_arg"]:
        args.extend([config["seed_arg"], str(seed)])

    cmd = [sys.executable, script] + args

    print(f"\n  Running {experiment_display_name(scenario_key)} (seed={seed})...")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    elapsed = time.time() - start_time

    # 保存日志
    log_dir = Path(output_dir) / "multi_seed"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{scenario_key}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n\n")
        f.write(result.stdout or "")
        if result.stderr:
            f.write(f"\n--- STDERR ---\n{result.stderr}")

    code_meta = current_code_metadata()
    run_meta = _read_run_metadata()
    try:
        metrics = read_scenario_result(
            scenario_key=scenario_key,
            seed=seed,
            output_dir=output_dir,
            elapsed_seconds=elapsed,
            success=result.returncode == 0,
            command=cmd,
            code_commit=code_meta["code_commit"],
            code_dirty=code_meta["code_dirty"],
            split_seed=run_meta["split_seed"],
            llm_provider=run_meta["llm_provider"],
            llm_temperature=run_meta["llm_temperature"],
        )
        snapshot_path = _snapshot_seed_result(
            result_file=metrics["result_file"],
            scenario_key=scenario_key,
            seed=seed,
            output_dir=output_dir,
        )
        if snapshot_path:
            metrics["seed_result_file"] = snapshot_path
    except Exception as exc:
        metrics = _parse_output_metrics(result.stdout or "")
        metrics.update({
            "seed": seed,
            "scenario": experiment_display_name(scenario_key),
            "scenario_key": scenario_key,
            "split_seed": run_meta["split_seed"],
            "elapsed_seconds": elapsed,
            "success": result.returncode == 0,
            "command": " ".join(cmd),
            "code_commit": code_meta["code_commit"],
            "code_dirty": code_meta["code_dirty"],
            "llm_provider": run_meta["llm_provider"],
            "llm_temperature": run_meta["llm_temperature"],
            "result_read_error": str(exc),
        })

    status = "OK" if result.returncode == 0 else "FAIL"
    mape_str = f"MAPE={metrics.get('test_mape', 'N/A')}" if metrics.get("test_mape") else ""
    print(f"    [{status}] {elapsed:.1f}s {mape_str}")

    return metrics


def _parse_output_metrics(stdout: str) -> dict:
    """从stdout中解析测试集指标

    查找 "Test Set Results" 或 "COMPLETED" 部分提取最终测试指标。
    使用 "Test MAPE:" / "Test RMSE:" 等明确前缀，或从 "Test Set Results" 块中提取。
    """
    import re
    metrics = {}

    # 方法1: 从明确标记的 "Test MAPE:", "Test RMSE:" 等提取
    explicit_patterns = {
        "test_mape": r"Test\s+MAPE:\s+([\d.]+)%",
        "test_rmse": r"Test\s+RMSE:\s+\$([\d,]+\.?\d*)",
        "test_mae": r"Test\s+MAE:\s+\$([\d,]+\.?\d*)",
        "test_r2": r"Test\s+R2:\s+([\d.]+)",
    }

    for key, pattern in explicit_patterns.items():
        match = re.search(pattern, stdout)
        if match:
            val_str = match.group(1).replace(",", "")
            val = float(val_str)
            if key == "test_mape":
                val = val / 100.0
            metrics[key] = val

    # 方法2: 从 "Test Set Results (using best checkpoint):" 块提取
    # 这个块的格式是确定的
    test_block_match = re.search(
        r"Test Set Results.*?:\s*\n((?:\s+\w+:.*\n)+)",
        stdout
    )
    if test_block_match:
        block = test_block_match.group(1)
        block_patterns = {
            "test_mape": r"MAPE:\s+([\d.]+)%",
            "test_rmse": r"RMSE:\s+\$([\d,]+\.?\d*)",
            "test_mae": r"MAE:\s+\$([\d,]+\.?\d*)",
            "test_mpe": r"MPE:\s+([-\d.]+)%",
            "test_r2": r"R2:\s+([\d.]+)",
        }
        for key, pattern in block_patterns.items():
            if key not in metrics:  # 不覆盖已提取的值
                match = re.search(pattern, block)
                if match:
                    val_str = match.group(1).replace(",", "")
                    val = float(val_str)
                    if key in ("test_mape", "test_mpe"):
                        val = val / 100.0
                    metrics[key] = val

    # 方法3: 从 Bias Correction Test Results 提取（场景C）
    bias_block_match = re.search(
        r"\[Bias Correction\] Test Results.*?:\s*\n((?:\s+\w+:.*\n)+)",
        stdout
    )
    if bias_block_match:
        block = bias_block_match.group(1)
        bias_patterns = {
            "test_mape_corrected": r"MAPE:\s+([\d.]+)%",
            "test_rmse_corrected": r"RMSE:\s+\$([\d,]+\.?\d*)",
            "test_mae_corrected": r"MAE:\s+\$([\d,]+\.?\d*)",
            "test_mpe_corrected": r"MPE:\s+([-\d.]+)%",
            "test_r2_corrected": r"R2:\s+([\d.]+)",
        }
        for key, pattern in bias_patterns.items():
            match = re.search(pattern, block)
            if match:
                val_str = match.group(1).replace(",", "")
                val = float(val_str)
                if key in ("test_mape_corrected", "test_mpe_corrected"):
                    val = val / 100.0
                metrics[key] = val

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        choices=list(SCENARIO_CONFIGS.keys()),
                        help="Scenarios to run (default: formal experiment order)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help=f"Seeds to use (default: {DEFAULT_SEEDS})")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--reparse", action="store_true",
                        help="Reparse existing logs without rerunning experiments")
    args = parser.parse_args()

    seeds = args.seeds or DEFAULT_SEEDS
    scenarios = args.scenarios or DEFAULT_SCENARIOS

    print(
        "WARNING: scripts/run_multi_seed.py is a historical non-formal runner; "
        "it cannot create FMAS paper evidence."
    )
    reject_new_formal_methods(scenarios)

    print("=" * 70)
    print("MULTI-SEED EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Scenarios: {scenarios}")
    print(f"Seeds: {seeds}")

    if args.reparse:
        print("Mode: REPARSE (reading existing logs)")
        all_results = reparse_logs(scenarios, seeds, args.output_dir)
    else:
        print(f"Total runs: {len(scenarios) * len(seeds)}")
        all_results = []
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"Experiment: {SCENARIO_CONFIGS[scenario]['name']} [{scenario}]")
            print(f"{'='*60}")

            for seed in seeds:
                metrics = run_scenario(scenario, seed, args.output_dir)
                all_results.append(metrics)

    # 保存所有结果
    import pandas as pd

    df = pd.DataFrame(all_results)
    output_path = Path(args.output_dir) / "multi_seed" / "all_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        df = merge_all_results(existing_df, df, scenarios=scenarios, seeds=seeds)
    df.to_csv(output_path, index=False)
    print(f"\n\nAll results saved to {output_path}")

    # 打印汇总
    print_summary(df, scenarios)
    print(f"\n\nNext step: python scripts/statistical_analysis.py")


def reparse_logs(scenarios: list, seeds: list, output_dir: str) -> list:
    """从已有日志重新提取指标"""
    log_dir = Path(output_dir) / "multi_seed"
    all_results = []
    code_meta = current_code_metadata()
    run_meta = _read_run_metadata()

    for scenario in scenarios:
        for seed in seeds:
            log_path = log_dir / f"{scenario}_seed{seed}.log"
            if not log_path.exists():
                print(f"  [SKIP] {log_path.name} not found")
                continue

            content = log_path.read_text(encoding="utf-8", errors="replace")
            metrics = read_seed_snapshot_result(
                scenario_key=scenario,
                seed=seed,
                output_dir=output_dir,
                code_commit=code_meta["code_commit"],
                code_dirty=code_meta["code_dirty"],
                split_seed=run_meta["split_seed"],
                llm_provider=run_meta["llm_provider"],
                llm_temperature=run_meta["llm_temperature"],
            )
            if metrics is None:
                metrics = _parse_output_metrics(content)
                metrics.update({
                    "seed": seed,
                    "scenario": experiment_display_name(scenario),
                    "scenario_key": scenario,
                    "split_seed": run_meta["split_seed"],
                    "success": True,
                    "code_commit": code_meta["code_commit"],
                    "code_dirty": code_meta["code_dirty"],
                    "llm_provider": run_meta["llm_provider"],
                    "llm_temperature": run_meta["llm_temperature"],
                    "source": "reparsed_log",
                })
            mape_str = f"MAPE={metrics.get('test_mape', 'N/A')}" if metrics.get("test_mape") else "MAPE=N/A"
            print(f"  {experiment_display_name(scenario)} seed={seed}: {mape_str}")
            all_results.append(metrics)

    return all_results


def _ensure_scenario_key(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "scenario_key" not in normalized.columns and "scenario" in normalized.columns:
        normalized["scenario_key"] = normalized["scenario"].map(canonical_experiment_key)
    elif "scenario_key" in normalized.columns:
        normalized["scenario_key"] = normalized["scenario_key"].map(canonical_experiment_key)
    return normalized


def merge_all_results(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    scenarios: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    """Merge newly run scenario/seed rows without dropping unrequested rows."""
    if existing_df is None or existing_df.empty:
        merged = new_df.copy()
    elif new_df is None or new_df.empty:
        merged = existing_df.copy()
    else:
        existing = _ensure_scenario_key(existing_df)
        new = _ensure_scenario_key(new_df)
        replace_scenarios = {canonical_experiment_key(scenario) for scenario in scenarios}
        replace_seeds = {int(seed) for seed in seeds}
        existing_seed = pd.to_numeric(existing.get("seed"), errors="coerce")
        replace_mask = (
            existing["scenario_key"].isin(replace_scenarios)
            & existing_seed.isin(replace_seeds)
        )
        merged = pd.concat([existing.loc[~replace_mask], new], ignore_index=True, sort=False)

    if "scenario_key" in merged.columns and "seed" in merged.columns:
        order = {key: index for index, key in enumerate(EXPERIMENT_ORDER)}
        merged = merged.copy()
        merged["_scenario_order"] = merged["scenario_key"].map(order).fillna(len(order))
        merged["_seed_order"] = pd.to_numeric(merged["seed"], errors="coerce")
        merged = (
            merged.sort_values(["_scenario_order", "_seed_order"], kind="stable")
            .drop(columns=["_scenario_order", "_seed_order"])
            .drop_duplicates(subset=["scenario_key", "seed"], keep="last")
            .reset_index(drop=True)
        )
    return merged


def print_summary(df, scenarios):
    """打印汇总统计"""
    import pandas as pd

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY (mean +/- std)")
    print("=" * 70)

    for scenario in scenarios:
        if scenario not in SCENARIO_CONFIGS:
            continue
        if "scenario_key" in df.columns:
            scenario_df = df[df["scenario_key"] == scenario]
        else:
            scenario_df = df[df["scenario"].isin([scenario, experiment_display_name(scenario)])]
        if scenario_df.empty:
            continue

        success_df = scenario_df[scenario_df.get("success", True) == True]
        if success_df.empty:
            print(f"\n  {experiment_display_name(scenario)}: all runs failed")
            continue

        name = SCENARIO_CONFIGS[scenario]["name"]
        print(f"\n  {name}: n={len(success_df)}")

        for col, label, multiplier, fmt in [
            ("test_mape", "MAPE", 100, ".2f"),
            ("test_rmse", "RMSE", 1, ",.0f"),
            ("test_mae", "MAE", 1, ",.0f"),
            ("test_mpe", "MPE", 100, ".2f"),
            ("test_r2", "R2", 1, ".4f"),
        ]:
            if col in success_df.columns and success_df[col].notna().any():
                mean_val = success_df[col].mean() * multiplier
                std_val = success_df[col].std() * multiplier
                unit = "%" if multiplier == 100 else ("$" if col in ("test_rmse", "test_mae") else "")
                if unit == "$":
                    print(f"    {label}: ${mean_val:{fmt}} +/- ${std_val:{fmt}}")
                elif unit == "%":
                    print(f"    {label}: {mean_val:{fmt}}% +/- {std_val:{fmt}}%")
                else:
                    print(f"    {label}: {mean_val:{fmt}} +/- {std_val:{fmt}}")

        # 偏差校正后的指标
        if "test_mape_corrected" in success_df.columns and success_df["test_mape_corrected"].notna().any():
            mape_c_mean = success_df["test_mape_corrected"].mean() * 100
            mape_c_std = success_df["test_mape_corrected"].std() * 100
            print(f"    MAPE (bias corrected): {mape_c_mean:.2f}% +/- {mape_c_std:.2f}%")


if __name__ == "__main__":
    main()
