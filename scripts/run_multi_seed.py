"""
多种子批量运行脚本 (Multi-Seed Runner)

对每个场景(A, A', B, C, C+bias) × 5个种子运行实验
种子列表: [42, 123, 456, 789, 2024]

用法:
    python scripts/run_multi_seed.py                    # 运行全部
    python scripts/run_multi_seed.py --scenarios B C    # 仅运行指定场景
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

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

SCENARIO_CONFIGS = {
    "A": {
        "name": "Centralized (GBR)",
        "script": "experiments/scenario_A_centralized.py",
        "args": [],
        "seed_arg": "--seed",
    },
    "A_prime": {
        "name": "Centralized (NN)",
        "script": "experiments/scenario_A_prime.py",
        "args": [],
        "seed_arg": "--seed",
    },
    "B": {
        "name": "FedAvg Baseline",
        "script": "experiments/scenario_B_fedavg.py",
        "args": ["--num_rounds", "20"],
        "seed_arg": "--seed",
    },
    "C": {
        "name": "MAS-FL-LLM",
        "script": "experiments/scenario_C_llm.py",
        "args": ["--num_rounds", "20", "--use_llm"],
        "seed_arg": "--seed",
    },
}


def run_scenario(scenario_key: str, seed: int, output_dir: str) -> dict:
    """运行单个场景单个种子的实验"""
    config = SCENARIO_CONFIGS[scenario_key]
    script = config["script"]
    args = list(config["args"])

    if config["seed_arg"]:
        args.extend([config["seed_arg"], str(seed)])

    cmd = [sys.executable, script] + args

    print(f"\n  Running {scenario_key} (seed={seed})...")

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

    # 解析结果（从stdout中提取关键指标）
    metrics = _parse_output_metrics(result.stdout or "")
    metrics["seed"] = seed
    metrics["scenario"] = scenario_key
    metrics["elapsed_seconds"] = elapsed
    metrics["success"] = result.returncode == 0

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
                        help="Scenarios to run (default: all)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help=f"Seeds to use (default: {DEFAULT_SEEDS})")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--reparse", action="store_true",
                        help="Reparse existing logs without rerunning experiments")
    args = parser.parse_args()

    seeds = args.seeds or DEFAULT_SEEDS
    scenarios = args.scenarios or list(SCENARIO_CONFIGS.keys())

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
            print(f"Scenario: {scenario} ({SCENARIO_CONFIGS[scenario]['name']})")
            print(f"{'='*60}")

            for seed in seeds:
                metrics = run_scenario(scenario, seed, args.output_dir)
                all_results.append(metrics)

    # 保存所有结果
    import pandas as pd

    df = pd.DataFrame(all_results)
    output_path = Path(args.output_dir) / "multi_seed" / "all_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n\nAll results saved to {output_path}")

    # 打印汇总
    print_summary(df, scenarios)
    print(f"\n\nNext step: python scripts/statistical_analysis.py")


def reparse_logs(scenarios: list, seeds: list, output_dir: str) -> list:
    """从已有日志重新提取指标"""
    log_dir = Path(output_dir) / "multi_seed"
    all_results = []

    for scenario in scenarios:
        for seed in seeds:
            log_path = log_dir / f"{scenario}_seed{seed}.log"
            if not log_path.exists():
                print(f"  [SKIP] {log_path.name} not found")
                continue

            content = log_path.read_text(encoding="utf-8", errors="replace")
            metrics = _parse_output_metrics(content)
            metrics["seed"] = seed
            metrics["scenario"] = scenario
            metrics["success"] = True

            mape_str = f"MAPE={metrics.get('test_mape', 'N/A')}" if metrics.get("test_mape") else "MAPE=N/A"
            print(f"  {scenario} seed={seed}: {mape_str}")
            all_results.append(metrics)

    return all_results


def print_summary(df, scenarios):
    """打印汇总统计"""
    import pandas as pd

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY (mean +/- std)")
    print("=" * 70)

    for scenario in scenarios:
        if scenario not in SCENARIO_CONFIGS:
            continue
        scenario_df = df[df["scenario"] == scenario]
        if scenario_df.empty:
            continue

        success_df = scenario_df[scenario_df.get("success", True) == True]
        if success_df.empty:
            print(f"\n  {scenario}: all runs failed")
            continue

        name = SCENARIO_CONFIGS[scenario]["name"]
        print(f"\n  {scenario} ({name}): n={len(success_df)}")

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
