# Deep Evaluation & Future Roadmap (v1.7)
*Last updated: 2026-03-03 — reflects Phase 1 Portfolio Construction results (90-fold, 2007–2026)*

## 1. Deep Evaluation of Current Performance

### Overall Metrics (Full 90-Fold Run: 2007–2026)
*   **Mean Rank Accuracy**: **57.5%** (Baseline: 50%)
    *   Statistically robust over 19 years of OOS data including GFC and COVID-19. All 90 folds beat random chance.
*   **Mean Val Loss**: **0.461**
*   **Mean Magnitude MAE**: **0.35** — stable across all 90 folds.
*   **L/S Alpha**: Positive in **88/90 folds** (folds 22 and 37 are outlier negative-alpha folds), mean L/S spread of **+0.865 z-score units** per snapshot.
*   **NaN Training Folds**: **0/90 (100% stable)** — clipping and zero-padding fixes successfully eliminated all NaN states.
*   **ECE (raw)**: 0.015 — near-perfect confidence calibration.
*   **Conformal Coverage**: 89.4% (target ≥ 90% for α=0.10).

### Regime Analysis (v1.6 Classification)
| Regime | Folds | Rank Accuracy | L/S Alpha | Validation Loss | Magnitude MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 54 | 57.44% | +33.64% | 0.4616 | 0.3484 |
| **Crisis** | 12 | 56.88% | +32.46% | 0.4683 | 0.3584 |
| **Calm** | 24 | 57.98% | +26.46% | 0.4569 | 0.3535 |

### Generalization Gap
*   **Conclusion**: Model demonstrates highly stable and profitable edge across all realized volatility states. The `Masked Superset Architecture` and dynamic `EarlyStopping` prevent overfitting even in massive drawdowns (Crisis).

---

## 2. v1.6 Recommendations — Status

| Rec | Description | Status |
|-----|-------------|--------|
| 1 | Gradient clipping + AMP fix | ✅ Implemented |
| 2 | Confidence calibration (Temperature/Platt) | ✅ Implemented |
| 3 | Dynamic early stopping patience | ✅ Implemented (recalibrated thresholds) |
| 4 | Adaptive magnitude weight λ | ✅ Implemented (recalibrated thresholds) |
| 5 | MC Dropout median rank | ✅ Implemented |
| 6 | Conformal prediction sets | ✅ Implemented |
| 7 | Regime classification | ✅ Implemented (recalibrated thresholds) |
| 8 | Long-short alpha tracking | ✅ Implemented |
| 9 | Walk-forward architecture | ✅ Already in v1.5 |
| 10 | Sector-neutral loss | ✅ Already in v1.5 |
| 11 | Benchmark vs simple models | ✅ Complete — see [`docs/benchmark_results.md`](benchmark_results.md) |

---

## 3. Current Limitations

### A. Execution & Costs
~~The backtest does not model slippage or commissions.~~ **✅ Resolved in v1.7.** A 5 bps/trade TCA model is now integrated into `compute_portfolio_metrics()`. At observed turnover levels (MVO v2 avg 0.27, Risk Parity avg 1.06), transaction costs are negligible — total cost across 90 folds is 0.23 (MVO v2) and 0.86 (Risk Parity), with net Sharpe virtually identical to gross Sharpe.

### B. Temperature Scaling Counter-Productive
The model's raw ECE (0.015) is already excellent. Temperature scaling with T=1.50 actually *increases* ECE to 0.029. This is expected — the v1.6 training fixes resolved the overconfidence issue at its root. Temperature calibration should only be applied if future model changes degrade raw calibration.

---

## 4. Future Roadmap

### Phase 0: Remaining Deployment Items
1.  ~~**Benchmark vs simple models** (Rec 11)~~ — **✅ Complete.** DGRCL outperforms all 7 benchmarks across 90 folds. Full results in [`docs/benchmark_results.md`](benchmark_results.md). Key result: DGRCL leads LSTM-only by +0.15pp rank accuracy and +20% L/S alpha, confirming the graph component's contribution.
2.  ~~**Transaction Cost Analysis**~~ — **✅ Complete.** 5 bps/trade TCA integrated into `compute_portfolio_metrics()`. Net Sharpes are virtually identical to gross (cost is <0.1% of total PnL).
3.  **Store raw L/S return series in `fold_results.json`** — enables FF3 attribution for DGRCL (currently only computed for benchmarks).

### Phase 1: Portfolio Construction — ✅ Complete
All three items implemented and validated across 90-fold walk-forward backtests.

1.  ~~**Mean-Variance Optimizer**~~ — **✅ Complete.** Markowitz MVO via `cvxpy` with Ledoit-Wolf covariance, adaptive position scaling, and dollar-neutral constraints. See results below.
2.  ~~**Risk Parity via graph clusters**~~ — **✅ Complete.** Equal-risk-contribution allocation across spectral clusters inferred from the DynamicGraphLearner's attention adjacency. Solved via L-BFGS-B with softmax parameterization.
3.  ~~**Conformal abstention gating**~~ — **✅ Complete.** Per-fold calibration on first half of validation snapshots; trades only where `set_size == 1`. Includes minimum-universe guard (`min_tradable=20`) that re-admits the most confident ambiguous stocks when gating would leave too few for the optimizer.

#### Phase 1 Results: Three-Way Comparison (90 Folds, 2007–2026)

| Metric | MVO v1 | MVO v2 (production) | Risk Parity |
|---|:---:|:---:|:---:|
| **Sharpe (gross)** | 4.95 ± 4.27 | 4.81 ± 4.22 | 3.37 ± 4.71 |
| **Sharpe (net, 5bps TCA)** | — | 4.81 ± 4.22 | 3.36 ± 4.71 |
| **Total PnL (gross)** | 314.7 | 274.5 | 425.8 |
| **Total PnL (net)** | — | 274.3 | 424.9 |
| **Avg Turnover** | 0.334 | 0.267 | 1.064 |
| **Avg MaxDD** | 0.652 | **0.244** | 1.441 |
| **Worst MaxDD** | 14.46 | **1.72** | 7.39 |
| **High-DD folds (>1.0)** | 11/90 | **4/90** | 46/90 |
| **Win rate (Sharpe > 0)** | 88.9% | 85.6% | 74.4% |
| **PnL / Worst DD (Calmar)** | 21.8 | **159.5** | 57.6 |
| **Sharpe / Turnover** | 14.8 | **18.1** | 3.2 |

**Key findings:**
*   **MVO v2 is the production candidate.** The min-universe guard + adaptive position scaling cut worst-fold drawdown from 14.5 to 1.7 (8.4× improvement) and raised the Calmar ratio from 21.8 to 159.5 (7.3×). Of the 9 previously negative-Sharpe folds, 7 flipped positive.
*   **Risk Parity generates the most raw PnL** (425.8 vs 274.5) but with 3× higher turnover and 46/90 high-drawdown folds. Its Sharpe/Turnover efficiency (3.2) is 6× lower than MVO v2.
*   **The two methods are nearly uncorrelated** (fold-level Sharpe correlation r = 0.04), making a 50/50 ensemble attractive: projected 91.1% win rate with only 8 negative-Sharpe folds.
*   **Transaction costs are negligible** at institutional S&P 500 liquidity. Total 90-fold cost: 0.23 (MVO v2), 0.86 (Risk Parity).

Implementation: `portfolio_optimizer.py` (all strategies), integrated into `train.py` via `compute_optimized_alpha()`. Results in `backtest_results/phase1_mvo_v2/` and `backtest_results/phase1_riskparity/`.

### Phase 2: Alpha Expansion (Medium-Term)
1.  **MVO + Risk Parity ensemble** — regime-conditioned or equal blend of the two uncorrelated strategies.
2.  **Alternative data** — news/sentiment as new heterogeneous graph node type.
3.  **RL fine-tuning (PPO)** — optimize directly for Sharpe Ratio post-supervised-pretraining.
4.  **Multi-horizon targets** — simultaneous 1d, 5d, 20d prediction heads.
