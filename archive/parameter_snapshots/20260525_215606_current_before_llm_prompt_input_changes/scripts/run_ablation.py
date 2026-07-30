"""
消融实验批量运行脚本 (Ablation Study Runner)

消融矩阵:
| 编号  | 配置名           | 轮数 | FedProx mu | 策略               | LLM |
|-------|------------------|------|------------|-------------------|-----|
| ab-1  | B-baseline       | 20   | 0.0        | size_only         | 无  |
| ab-2  | B+FedProx        | 20   | 0.01       | size_only         | 无  |
| ab-3  | C-fixed-perf     | 20   | 0.01       | perf_only(固定)   | 无  |
| ab-4  | C-fixed-hybrid   | 20   | 0.01       | hybrid(固定)      | 无  |
| ab-5  | C-with-LLM       | 20   | 0.01       | LLM动态           | 有  |
| ab-6  | C-with-LLM+bias  | 20   | 0.01       | LLM动态+偏差校正  | 有  |

用法:
    python scripts/run_ablation.py                          # 运行全部消融实验 (5个种子)
    python scripts/run_ablation.py --only ab-1 ab-2         # 仅运行指定实验
    python scripts/run_ablation.py --seeds 42               # 仅运行单个种子
    python scripts/run_ablation.py --seeds 42 123 456       # 指定多个种子
"""

import sys
import os
import re
import subprocess
import argparse
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

ABLATION_CONFIGS = [
    {
        "id": "ab-1",
        "name": "B-baseline",
        "description": "Pure FedAvg baseline (no FedProx)",
        "script": "experiments/scenario_B_fedavg.py",
        "args": ["--num_rounds", "20", "--fedprox_mu", "0.0", "--strategy", "size_only",
                 "--output_prefix", "ablation_ab1"],
    },
    {
        "id": "ab-2",
        "name": "B+FedProx",
        "description": "FedAvg + FedProx regularization",
        "script": "experiments/scenario_B_fedavg.py",
        "args": ["--num_rounds", "20", "--fedprox_mu", "0.01", "--strategy", "size_only",
                 "--output_prefix", "ablation_ab2"],
    },
    {
        "id": "ab-3",
        "name": "C-fixed-perf",
        "description": "Fixed perf_only strategy + FedProx",
        "script": "experiments/scenario_C_llm.py",
        "args": ["--num_rounds", "20", "--strategy", "perf_only"],
    },
    {
        "id": "ab-4",
        "name": "C-fixed-hybrid",
        "description": "Fixed hybrid strategy + FedProx",
        "script": "experiments/scenario_C_llm.py",
        "args": ["--num_rounds", "20", "--strategy", "hybrid"],
    },
    {
        "id": "ab-5",
        "name": "C-with-LLM",
        "description": "Full MAS-FL-LLM (dynamic strategy selection, 20 rounds)",
        "script": "experiments/scenario_C_llm.py",
        "args": ["--num_rounds", "20", "--use_llm"],
    },
    {
        "id": "ab-6",
        "name": "C-with-LLM+bias",
        "description": "Bias-corrected metrics derived from the same ab-5 LLM run",
        "derive_from": "ab-5",
        "note": "Derived from ab-5 corrected metrics to avoid a second stochastic LLM run."
    },
]


def run_experiment(config: dict, seed: int, output_dir: str) -> dict:
    """运行单个消融实验，返回解析后的指标字典"""
    exp_id = config["id"]
    exp_name = config["name"]
    script = config["script"]
    args = list(config["args"])

    # 添加seed参数
    if "--seed" not in args:
        args.extend(["--seed", str(seed)])

    cmd = [sys.executable, script] + args

    print(f"\n{'='*60}")
    print(f"Running {exp_id}: {exp_name} (seed={seed})")
    print(f"  Description: {config['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

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
    log_dir = Path(output_dir) / "ablation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{exp_id}_{exp_name}_seed{seed}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"Elapsed time: {elapsed:.1f}s\n")
        f.write(f"\n{'='*40} STDOUT {'='*40}\n")
        f.write(result.stdout or "")
        f.write(f"\n{'='*40} STDERR {'='*40}\n")
        f.write(result.stderr or "")

    success = result.returncode == 0
    if success:
        print(f"  [OK] {exp_id} seed={seed} completed in {elapsed:.1f}s")
    else:
        print(f"  [FAIL] {exp_id} seed={seed} failed (return code: {result.returncode})")
        print(f"  See log: {log_path}")
        if result.stderr:
            lines = result.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"    {line}")

    # 解析指标
    metrics = _parse_output_metrics(result.stdout or "")
    metrics["id"] = exp_id
    metrics["name"] = exp_name
    metrics["seed"] = seed
    metrics["success"] = success
    metrics["elapsed_seconds"] = elapsed

    return metrics


def _parse_output_metrics(stdout: str) -> dict:
    """从stdout中解析测试集指标"""
    metrics = {}

    # 从 "Test Set Results" 块提取
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
            match = re.search(pattern, block)
            if match:
                val_str = match.group(1).replace(",", "")
                val = float(val_str)
                if key in ("test_mape", "test_mpe"):
                    val = val / 100.0
                metrics[key] = val

    # 从 Bias Correction Test Results 提取
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


def derive_bias_correction_rows(df):
    """Create ab-6 rows from ab-5 corrected metrics without rerunning LLM."""
    import pandas as pd

    if df.empty or "id" not in df.columns:
        return pd.DataFrame()

    source = df[(df["id"] == "ab-5") & (df["success"] == True)].copy()
    if source.empty:
        return pd.DataFrame()

    corrected_cols = [
        "test_mape_corrected", "test_rmse_corrected", "test_mae_corrected",
        "test_mpe_corrected", "test_r2_corrected",
    ]
    if not any(col in source.columns and source[col].notna().any() for col in corrected_cols):
        return pd.DataFrame()

    source["id"] = "ab-6"
    source["name"] = "C-with-LLM+bias"
    source["description"] = "Derived bias-corrected metrics from ab-5"
    return source


def generate_summary(all_results: list, output_dir: str):
    """生成消融实验统计汇总CSV"""
    import pandas as pd

    df = pd.DataFrame(all_results)
    if df.empty:
        print("  No results to summarize.")
        return

    derived_df = derive_bias_correction_rows(df)
    if not derived_df.empty:
        df = pd.concat([df, derived_df], ignore_index=True)

    # 保存原始结果
    raw_path = Path(output_dir) / "ablation_logs" / "ablation_all_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n  Raw results saved to {raw_path}")

    # 按实验ID分组统计
    metric_cols = ["test_mape", "test_rmse", "test_mae", "test_mpe", "test_r2",
                   "test_mape_corrected", "test_rmse_corrected", "test_mae_corrected",
                   "test_mpe_corrected", "test_r2_corrected"]

    # 仅统计成功的实验
    success_df = df[df["success"] == True]

    summary_rows = []
    for exp_id in success_df["id"].unique():
        group = success_df[success_df["id"] == exp_id]
        row = {
            "id": exp_id,
            "name": group["name"].iloc[0],
            "n_seeds": len(group),
            "seeds": ",".join(str(s) for s in sorted(group["seed"].tolist())),
        }
        for col in metric_cols:
            if col in group.columns and group[col].notna().any():
                vals = group[col].dropna()
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = vals.std()
                row[f"{col}_min"] = vals.min()
                row[f"{col}_max"] = vals.max()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(output_dir) / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Statistical summary saved to {summary_path}")

    # 打印汇总表
    print("\n" + "=" * 90)
    print("ABLATION STATISTICAL SUMMARY (mean +/- std)")
    print("=" * 90)
    for _, row in summary_df.iterrows():
        print(f"\n  {row['id']} ({row['name']}): n={row['n_seeds']} seeds=[{row['seeds']}]")
        if "test_mape_mean" in row and pd.notna(row.get("test_mape_mean")):
            print(f"    MAPE: {row['test_mape_mean']*100:.2f}% +/- {row.get('test_mape_std', 0)*100:.2f}%")
        if "test_rmse_mean" in row and pd.notna(row.get("test_rmse_mean")):
            print(f"    RMSE: ${row['test_rmse_mean']:,.0f} +/- ${row.get('test_rmse_std', 0):,.0f}")
        if "test_r2_mean" in row and pd.notna(row.get("test_r2_mean")):
            print(f"    R2:   {row['test_r2_mean']:.4f} +/- {row.get('test_r2_std', 0):.4f}")
        if "test_mape_corrected_mean" in row and pd.notna(row.get("test_mape_corrected_mean")):
            print(f"    MAPE (corrected): {row['test_mape_corrected_mean']*100:.2f}% +/- {row.get('test_mape_corrected_std', 0)*100:.2f}%")
            print(f"    R2   (corrected): {row.get('test_r2_corrected_mean', 0):.4f} +/- {row.get('test_r2_corrected_std', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments (multi-seed)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run specific experiments (e.g., ab-1 ab-2)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help=f"Random seeds (default: {DEFAULT_SEEDS})")
    # 兼容旧的 --seed 参数
    parser.add_argument("--seed", type=int, default=None,
                        help="Single random seed (deprecated, use --seeds)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory")
    args = parser.parse_args()

    # 确定种子列表
    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = DEFAULT_SEEDS

    print("=" * 70)
    print("ABLATION STUDY RUNNER (Multi-Seed)")
    print("=" * 70)
    print(f"Seeds: {seeds}")

    configs = ABLATION_CONFIGS
    if args.only:
        requested = set(args.only)
        for config in ABLATION_CONFIGS:
            if config.get("derive_from") and config["id"] in requested:
                requested.add(config["derive_from"])
        configs = [c for c in configs if c["id"] in requested]
        print(f"Running {len(configs)} selected experiments: {[c['id'] for c in configs]}")
    else:
        print(f"Running all {len(configs)} experiments")

    runnable_configs = [c for c in configs if "derive_from" not in c]
    derived_configs = [c for c in configs if "derive_from" in c]
    if derived_configs:
        print(f"Derived experiments: {[c['id'] for c in derived_configs]} will be computed from source runs.")

    total_runs = len(runnable_configs) * len(seeds)
    print(f"Total runs: {total_runs} ({len(runnable_configs)} runnable configs x {len(seeds)} seeds)")

    all_results = []
    run_count = 0
    for config in runnable_configs:
        for seed in seeds:
            run_count += 1
            print(f"\n>>> Run {run_count}/{total_runs}")
            metrics = run_experiment(config, seed, args.output_dir)
            all_results.append(metrics)

    # 总结
    print("\n" + "=" * 70)
    print("ABLATION STUDY RUN STATUS")
    print("=" * 70)
    for config in configs:
        if "derive_from" in config:
            print(f"  [DERIVED] {config['id']}: {config['name']} from {config['derive_from']}")
            continue
        exp_results = [r for r in all_results if r["id"] == config["id"]]
        ok_count = sum(1 for r in exp_results if r["success"])
        total = len(exp_results)
        status = "OK" if ok_count == total else f"PARTIAL ({ok_count}/{total})"
        print(f"  [{status}] {config['id']}: {config['name']}")

    generate_summary(all_results, args.output_dir)

    failed_runs = [r for r in all_results if not r["success"]]
    if failed_runs:
        print(f"\n  {len(failed_runs)} runs failed. Check logs in results/ablation_logs/")
    else:
        print(f"\n  All {total_runs} runs completed successfully!")

    print("\n  Next step: python scripts/ablation_analysis.py")


if __name__ == "__main__":
    main()
