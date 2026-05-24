"""
偏差诊断脚本 (Bias Diagnosis)

分析场景C的系统性低估偏差根因：
1. 解析训练日志，分析每个客户端的MPE趋势
2. 分析perf_only策略是否系统性偏好低估的客户端
3. 分析best checkpoint的MPE分布
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_round_metrics(log_dir: str = "results/logs"):
    """分析每轮的全局MPE变化趋势"""
    log_path = Path(log_dir)

    # 尝试加载场景C的轮次指标
    round_metrics_path = log_path / "scene_C_round_metrics.csv"
    if not round_metrics_path.exists():
        print(f"[WARN] {round_metrics_path} not found, skipping round metrics analysis.")
        return None

    df = pd.read_csv(round_metrics_path)
    print(f"\n[Round Metrics] Loaded {len(df)} rounds from {round_metrics_path}")

    if "global_val_mpe" in df.columns:
        print(f"  Global Val MPE range: {df['global_val_mpe'].min()*100:.2f}% ~ {df['global_val_mpe'].max()*100:.2f}%")
        print(f"  Global Val MPE mean: {df['global_val_mpe'].mean()*100:.2f}%")
        print(f"  Global Val MPE final: {df['global_val_mpe'].iloc[-1]*100:.2f}%")

    if "global_val_mape" in df.columns:
        best_idx = df["global_val_mape"].idxmin()
        best_round = df.loc[best_idx, "round"]
        best_mape = df.loc[best_idx, "global_val_mape"]
        best_mpe = df.loc[best_idx, "global_val_mpe"] if "global_val_mpe" in df.columns else 0
        print(f"\n  Best MAPE round: {best_round} (MAPE={best_mape*100:.2f}%, MPE={best_mpe*100:.2f}%)")

    return df


def analyze_client_metrics(log_dir: str = "results/logs"):
    """分析各客户端的MPE趋势"""
    log_path = Path(log_dir)

    client_metrics_path = log_path / "scene_C_client_metrics.csv"
    if not client_metrics_path.exists():
        print(f"[WARN] {client_metrics_path} not found, skipping client metrics analysis.")
        return None

    df = pd.read_csv(client_metrics_path)
    print(f"\n[Client Metrics] Loaded {len(df)} records from {client_metrics_path}")

    clients = df["client_id"].unique()
    for client in clients:
        client_data = df[df["client_id"] == client]
        if "val_mpe" in client_data.columns:
            print(f"\n  {client}:")
            print(f"    Val MPE mean: {client_data['val_mpe'].mean()*100:.2f}%")
            print(f"    Val MPE final: {client_data['val_mpe'].iloc[-1]*100:.2f}%")
            print(f"    Val MAPE mean: {client_data['val_mape'].mean()*100:.2f}%")
            print(f"    Samples: {client_data['n_samples'].iloc[0]}")

    return df


def analyze_llm_decisions(log_dir: str = "results/logs"):
    """分析LLM决策模式"""
    log_path = Path(log_dir)

    decisions_path = log_path / "scene_C_llm_decisions.jsonl"
    if not decisions_path.exists():
        print(f"[WARN] {decisions_path} not found, skipping LLM decision analysis.")
        return None

    decisions = []
    with open(decisions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))

    print(f"\n[LLM Decisions] Loaded {len(decisions)} decisions")

    # 统计策略选择频次
    strategy_counts = {}
    for d in decisions:
        strategy = d.get("decision", {}).get("chosen_strategy_name", "unknown")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    print("  Strategy selection frequency:")
    for s, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {count} ({count/len(decisions)*100:.1f}%)")

    return decisions


def analyze_aggregation_weights(log_dir: str = "results/logs"):
    """分析聚合权重分布"""
    log_path = Path(log_dir)

    history_path = log_path / "scene_C_training_history.json"
    if not history_path.exists():
        print(f"[WARN] {history_path} not found, skipping weight analysis.")
        return None

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    rounds = history.get("round_history", [])
    print(f"\n[Aggregation Weights] Analyzing {len(rounds)} rounds")

    # 对比perf_only和size_only轮次的权重差异
    for strategy_name in ["size_only", "perf_only", "hybrid", "fairness_clip"]:
        strategy_rounds = [r for r in rounds if r["strategy_name"] == strategy_name]
        if not strategy_rounds:
            continue

        print(f"\n  Strategy: {strategy_name} ({len(strategy_rounds)} rounds)")
        # 汇总各客户端的平均权重
        all_weights = {}
        for r in strategy_rounds:
            for cid, w in r.get("aggregation_weights", {}).items():
                if cid not in all_weights:
                    all_weights[cid] = []
                all_weights[cid].append(w)

        for cid, weights in all_weights.items():
            print(f"    {cid}: mean_weight={np.mean(weights):.3f}, std={np.std(weights):.3f}")

    return rounds


def main():
    print("=" * 70)
    print("BIAS DIAGNOSIS: Analyzing Scenario C Systematic Bias")
    print("=" * 70)

    log_dir = "results/logs"

    round_df = analyze_round_metrics(log_dir)
    client_df = analyze_client_metrics(log_dir)
    decisions = analyze_llm_decisions(log_dir)
    rounds = analyze_aggregation_weights(log_dir)

    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    if round_df is not None and "global_val_mpe" in round_df.columns:
        final_mpe = round_df["global_val_mpe"].iloc[-1] * 100
        if abs(final_mpe) > 15:
            print(f"\n  [CRITICAL] Final MPE = {final_mpe:.2f}% (> 15% threshold)")
            print("  Recommendation: Enable bias correction in evaluation")
        elif abs(final_mpe) > 10:
            print(f"\n  [WARNING] Final MPE = {final_mpe:.2f}% (> 10%)")
            print("  Recommendation: Consider bias correction")
        else:
            print(f"\n  [OK] Final MPE = {final_mpe:.2f}% (within acceptable range)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
