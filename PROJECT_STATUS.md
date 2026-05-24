# Project Status

> **Last Updated**: 2026-02-26 18:05
> **Data Version**: All scenarios rerun with seed=42 on 2026-02-26
> **Configuration**: Verified identical to reference version (226)

---

## 1. Overall Status

All four experimental scenarios (A / A' / B / C) have been successfully rerun under a unified protocol on 2026-02-26. The DeepSeek LLM API functioned correctly throughout all 20 rounds of Scenario C. Results are consistent and reproducible.

**Project Phase**: Core experiments complete. Reports updated with latest data.

---

## 2. Latest Test Results (seed=42, 2026-02-26)

| Scenario | Method | Rounds | Best Round | Test MAPE | Test RMSE | Test MAE | Test MPE | Test R² |
|----------|--------|--------|-----------|----------:|----------:|---------:|---------:|--------:|
| A | Centralized GBR | — | — | 64.33% | $1,639,791 | $1,189,341 | +29.67% | — |
| A' | Centralized MLP | 800(ES@86) | 86 | 50.58% | $1,406,209 | $1,018,121 | +19.84% | — |
| B | FedAvg (size_only) | 30 | 25 | 44.46% | **$1,436,502** | **$1,026,089** | +6.25% | **0.5144** |
| **C** | **MAS-FL-LLM** | **20** | **19** | **40.56%** | $1,582,833 | $1,089,729 | -7.58% | 0.4105 |
| **C** (bias corr.) | MAS-FL-LLM + correction | 20 | 19 | **40.93%** | $1,529,465 | $1,052,134 | **-1.83%** | 0.4495 |

### Key Findings

1. **MAPE Champion**: Scenario C achieves the best MAPE (40.56%), outperforming FedAvg (B) by 3.90 percentage points (8.77% relative improvement)
2. **RMSE Trade-off**: C's RMSE ($1,582,833) is higher than B's ($1,436,502) by ~$146K, narrowing to ~$93K after bias correction
3. **Near-Zero Bias**: After bias correction, C's MPE drops to -1.83% (near-unbiased), vs B's +6.25%
4. **Communication Efficiency**: C achieves better MAPE in 20 rounds vs B's 30 rounds (33% fewer communication rounds)
5. **LLM Strategy Convergence**: LLM converges to perf_only strategy after 4-round exploration (75% of rounds use perf_only)

---

## 3. LLM Decision Summary (Scenario C)

### Strategy Distribution
| Strategy | Rounds Used | Percentage |
|----------|:-----------:|:----------:|
| perf_only | 15 | 75.0% |
| hybrid | 2 | 10.0% |
| fairness_clip | 2 | 10.0% |
| size_only | 1 | 5.0% |

### Learning Rate Adjustment Pattern
- Rounds 1-4: lr_scale=1.0 (exploration phase, base values)
- Round 5: lr_scale=1.2 (accelerate exploration)
- Rounds 8-18: lr_scale=1.1 (stable moderate acceleration)
- Round 19: lr_scale=1.05 (approaching convergence, reduced perturbation)
- Round 20: lr_scale=1.0 (final round, return to base)

---

## 4. Experiment Configuration (B vs C)

| Parameter | Scenario B | Scenario C |
|-----------|-----------|-----------|
| Rounds | 30 | 20 |
| FedProx mu | 0.0 (pure FedAvg) | 0.01 (light proximal) |
| Strategy | size_only (fixed) | LLM dynamic selection |
| LR Schedule | Fixed 0.0005 | LLM-adjusted (0.0005 * lr_scale) |
| Local Epochs | 20 (fixed) | 20 + epoch_delta (LLM-adjusted) |

### Shared Configuration
- Model: CostEstimationMLP [128, 128, 64, 32], GELU, BatchNorm1d, Dropout=0.1
- Parameters: 28,993
- Optimizer: Adam (weight_decay=1e-4)
- Data: Same split (seed=42), 3 clients (~148/150/140 train samples)
- Target transform: power_0.25

---

## 5. File Structure and Canonical Sources

### Result Files (authoritative, freshly generated 2026-02-26)
| File | Content |
|------|---------|
| `results/centralized_results.csv` | Scenario A test metrics |
| `results/centralized_nn_results.csv` | Scenario A' test metrics |
| `results/fedavg_results.csv` | Scenario B test metrics |
| `results/scenario_c_results.csv` | Scenario C test metrics (raw + bias corrected) |

### Log Files
| File | Content |
|------|---------|
| `results/logs/scene_B_round_metrics.csv` | B: per-round global metrics |
| `results/logs/scene_B_client_metrics.csv` | B: per-round per-client metrics |
| `results/logs/scene_C_round_metrics.csv` | C: per-round global metrics |
| `results/logs/scene_C_client_metrics.csv` | C: per-round per-client metrics |
| `results/logs/scene_C_llm_decisions.jsonl` | C: LLM decision log (strategy, lr_scale, reasoning) |
| `results/logs/scene_C_training_history.json` | C: complete training history |

### Model Checkpoints
| File | Content |
|------|---------|
| `results/models/centralized_gbr_model.pkl` | Scenario A: GBR model |
| `results/models/centralized_nn_model.pth` | Scenario A': NN model |
| `results/models/fedavg_global_model.pth` | Scenario B: best checkpoint (Round 25) |
| `results/models/scenario_C_llm_model.pt` | Scenario C: best checkpoint (Round 19) |

### Reports
| File | Content |
|------|---------|
| `实验C配置与算法分析报告.md` | Detailed algorithm analysis with engineering significance |
| `PROJECT_STATUS.md` | This file - project overview and latest results |
| `results/experiment_ABC_comparison.md` | Cross-scenario comparison summary |

---

## 6. Completed Work

- [x] Unified data pipeline for all 4 scenarios
- [x] Scenario A: Centralized GBR baseline
- [x] Scenario A': Centralized MLP (NN upper bound)
- [x] Scenario B: FedAvg baseline (30 rounds, size_only, mu=0.0)
- [x] Scenario C: MAS-FL-LLM (20 rounds, LLM dynamic strategy, mu=0.01)
- [x] Bias correction mechanism (validation-based MPE correction)
- [x] R² metric added to federated scenarios
- [x] Configuration verified against reference version (226)
- [x] Dead code cleanup (removed 5 unused files + __pycache__)
- [x] Detailed algorithm analysis report
- [x] All results freshly regenerated (2026-02-26)

---

## 7. Open Gaps (for Paper-Level Completeness)

| Gap | Priority | Description |
|-----|----------|-------------|
| Multi-seed runs | High | Only seed=42 tested; need 5+ seeds for statistical significance |
| Statistical tests | High | No paired t-test or Wilcoxon test between B and C |
| Ablation study | High | Cannot isolate FedProx vs strategy selection vs LLM contributions |
| Stratified evaluation | Medium | No per-scale analysis (small/medium/large projects) |
| Communication cost | Medium | No formal communication round efficiency analysis |
| Paper figures | Medium | `results/figures/` needs publication-quality plots |
| LLM reproducibility | Low | temperature=0.8 introduces stochasticity across runs |

---

## 8. How to Reproduce

```bash
# Ensure Python 3.12+ with PyTorch, sklearn, numpy, pandas

# Run all 4 scenarios in order
python experiments/scenario_A_centralized.py
python experiments/scenario_A_prime.py
python experiments/scenario_B_fedavg.py --seed 42
python experiments/scenario_C_llm.py --use_llm --num_rounds 20

# Results will be saved to results/ directory
# Logs will be saved to results/logs/ directory
```

**Requirements for Scenario C**: DeepSeek API key with sufficient balance. Set in `configs/config.yaml` under `scene_c.llm.api_key` or via environment variable.
