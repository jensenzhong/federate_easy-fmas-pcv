"""
LLM决策模式分析脚本 (Analyze LLM Decisions)

解析 scene_C_llm_decisions.jsonl，分析:
1. 策略选择频次分布
2. lr_scale / epoch_delta 调整分布
3. 策略选择随时间的变化趋势
4. LLM"探索→收敛"的模式稳定性

用法:
    python scripts/analyze_llm_decisions.py
    python scripts/analyze_llm_decisions.py --log_dir results/logs
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def load_decisions(log_dir: str = "results/logs") -> list:
    """加载LLM决策日志"""
    decisions_path = Path(log_dir) / "scene_C_llm_decisions.jsonl"

    if not decisions_path.exists():
        print(f"[ERROR] {decisions_path} not found")
        return []

    decisions = []
    with open(decisions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    decisions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(decisions)} LLM decisions")
    return decisions


def analyze_strategy_frequency(decisions: list):
    """分析策略选择频次"""
    print("\n--- Strategy Selection Frequency ---")

    strategies = [d.get("decision", {}).get("chosen_strategy_name", "unknown")
                  for d in decisions]

    counter = Counter(strategies)
    total = len(strategies)

    for strategy, count in counter.most_common():
        print(f"  {strategy}: {count}/{total} ({count/total*100:.1f}%)")

    return counter


def analyze_hyperparameter_adjustments(decisions: list):
    """分析超参数调整分布"""
    print("\n--- Hyperparameter Adjustments ---")

    lr_scales = []
    epoch_deltas = []
    lambda_hybrids = []

    for d in decisions:
        decision = d.get("decision", {})
        lr_scales.append(decision.get("lr_scale", 1.0))
        epoch_deltas.append(decision.get("epoch_delta", 0))
        lh = decision.get("lambda_hybrid")
        if lh is not None:
            lambda_hybrids.append(lh)

    lr_scales = np.array(lr_scales)
    epoch_deltas = np.array(epoch_deltas)

    print(f"\n  lr_scale:")
    print(f"    mean={lr_scales.mean():.3f}, std={lr_scales.std():.3f}")
    print(f"    min={lr_scales.min():.3f}, max={lr_scales.max():.3f}")
    print(f"    >1.0: {(lr_scales > 1.0).sum()}, =1.0: {(lr_scales == 1.0).sum()}, <1.0: {(lr_scales < 1.0).sum()}")

    print(f"\n  epoch_delta:")
    print(f"    mean={epoch_deltas.mean():.2f}, std={epoch_deltas.std():.2f}")
    print(f"    >0: {(epoch_deltas > 0).sum()}, =0: {(epoch_deltas == 0).sum()}, <0: {(epoch_deltas < 0).sum()}")

    if lambda_hybrids:
        lambda_hybrids = np.array(lambda_hybrids)
        print(f"\n  lambda_hybrid (hybrid/fairness_clip rounds only):")
        print(f"    mean={lambda_hybrids.mean():.3f}, std={lambda_hybrids.std():.3f}")

    return lr_scales, epoch_deltas


def analyze_temporal_patterns(decisions: list):
    """分析决策随时间的变化模式"""
    print("\n--- Temporal Decision Patterns ---")

    rounds = []
    strategies = []
    is_reused = []

    for d in decisions:
        rounds.append(d.get("round", 0))
        strategies.append(d.get("decision", {}).get("chosen_strategy_name", "unknown"))
        is_reused.append(d.get("decision", {}).get("is_reused", False))

    # 分阶段统计
    if len(rounds) >= 10:
        early = strategies[:10]
        mid = strategies[10:20] if len(strategies) > 20 else strategies[10:]
        late = strategies[20:] if len(strategies) > 20 else []

        print(f"\n  Early phase (rounds 1-10):")
        for s, c in Counter(early).most_common():
            print(f"    {s}: {c}")

        if mid:
            print(f"\n  Mid phase (rounds 11-20):")
            for s, c in Counter(mid).most_common():
                print(f"    {s}: {c}")

        if late:
            print(f"\n  Late phase (rounds 21+):")
            for s, c in Counter(late).most_common():
                print(f"    {s}: {c}")

    # 检测收敛：后半段是否趋向于单一策略
    if len(strategies) >= 10:
        last_half = strategies[len(strategies)//2:]
        dominant = Counter(last_half).most_common(1)[0]
        dominance_ratio = dominant[1] / len(last_half)
        print(f"\n  Convergence analysis (last {len(last_half)} rounds):")
        print(f"    Dominant strategy: {dominant[0]} ({dominance_ratio*100:.1f}%)")
        if dominance_ratio > 0.7:
            print(f"    [CONVERGED] LLM has converged to '{dominant[0]}'")
        else:
            print(f"    [EXPLORING] LLM is still exploring multiple strategies")


def analyze_reasoning_themes(decisions: list):
    """分析LLM推理文本的主题"""
    print("\n--- Reasoning Themes ---")

    keywords = {
        "decline": ["下降", "降低", "改善", "improved", "decline"],
        "stagnate": ["停滞", "持平", "plateau", "stagnant"],
        "worsen": ["恶化", "升高", "worse", "increased"],
        "explore": ["探索", "尝试", "try", "explore"],
        "maintain": ["保持", "继续", "maintain", "continue"],
    }

    theme_counts = Counter()
    for d in decisions:
        reasoning = d.get("decision", {}).get("reasoning", "")
        for theme, words in keywords.items():
            if any(w in reasoning.lower() for w in words):
                theme_counts[theme] += 1

    for theme, count in theme_counts.most_common():
        print(f"  {theme}: {count}/{len(decisions)} ({count/len(decisions)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM decisions")
    parser.add_argument("--log_dir", type=str, default="results/logs",
                        help="Log directory")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM DECISION PATTERN ANALYSIS")
    print("=" * 70)

    decisions = load_decisions(args.log_dir)

    if not decisions:
        print("\nNo decisions found. Run scenario C with LLM first:")
        print("  python experiments/scenario_C_llm.py --use_llm --num_rounds 30")
        return

    analyze_strategy_frequency(decisions)
    analyze_hyperparameter_adjustments(decisions)
    analyze_temporal_patterns(decisions)
    analyze_reasoning_themes(decisions)

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
