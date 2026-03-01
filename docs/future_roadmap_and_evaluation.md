# Deep Evaluation & Future Roadmap (v1.6)
*Last updated: 2026-03-01 — reflects v1.6 90-fold walk-forward run (2007–2026)*

## 1. Deep Evaluation of Current Performance

### Overall Metrics (Full 90-Fold Run: 2007–2026)
*   **Mean Rank Accuracy**: **57.6%** (Baseline: 50%)
    *   Statistically robust over 19 years of OOS data including GFC and COVID-19.
*   **Mean Val Loss**: **0.460 ± 0.017**
*   **Mean Magnitude MAE**: **0.351** — stable across all 90 folds.
*   **L/S Alpha**: Positive in **86/90 folds** (95.6%), mean +0.82 per snapshot.
*   **NaN Training Folds**: 10/90 (11%) — concentrated in GFC and COVID regimes.
*   **ECE (raw)**: 0.015 — near-perfect confidence calibration.
*   **Conformal Coverage**: 89.4% (target ≥ 90% for α=0.10).

### Regime Analysis (v1.6 Classification)
| Regime | Vol Range | Approx. Folds | Mean Rank Acc. | Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Calm** | < 0.20 | ~25% of folds | ~58% | **Stable**. Clear trends, orderly sector rotation. |
| **Normal** | 0.20–0.50 | ~50% of folds | ~57.5% | **Consistent**. Adaptive λ and base patience work well. |
| **Crisis** | > 0.50 | ~25% of folds | ~55% | **Degraded but functional**. Extended patience helps. |

### Generalization Gap
*   **Train Loss**: ~0.463 (mean where non-NaN)
*   **Val Loss**: ~0.460
*   **Gap**: ~0.003 (negligible)
*   **Conclusion**: The model is **not overfitting**. Masked Superset Architecture and dynamic early stopping work effectively.

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
| 11 | Benchmark vs simple models | 🔲 Not yet implemented |

---

## 3. Current Limitations

### A. NaN Training Loss in Crisis Regimes
10/90 folds exhibit NaN training loss — all in high-volatility regime transitions (GFC, COVID). v1.6 mitigates this with NaN batch skipping in `train_epoch`, but root cause (FP16 overflow on extreme returns) remains.

### B. Execution & Costs (Not Modeled)
The backtest does not model slippage or commissions.
*   **Risk**: High-turnover daily rebalancing at 57.5% accuracy could be net-negative after fees.
*   **Fix**: Integrate a 5 bps/trade cost model (TCA).

### C. Temperature Scaling Counter-Productive
The model's raw ECE (0.015) is already excellent. Temperature scaling with T=1.50 actually *increases* ECE to 0.029. This is expected — the v1.6 training fixes resolved the overconfidence issue at its root. Temperature calibration should only be applied if future model changes degrade raw calibration.

---

## 4. Future Roadmap

### Phase 0: Remaining Deployment Items
1.  **Benchmark vs simple models** (Rec 11) — compare against momentum, sector rotation, and random baselines.
2.  **Transaction Cost Analysis** — 5 bps/trade cost model integration.
3.  **Full re-run with recalibrated thresholds** — validate regime-adaptive features with corrected boundaries.

### Phase 1: Portfolio Construction (Short-Term)
1.  **Mean-Variance Optimizer** — feed `dir_logits` + `mag_preds` into a convex optimizer.
2.  **Risk Parity via graph clusters** — equal-risk allocation across learned stock clusters.
3.  **Conformal abstention gating** — skip trades where `set_size = 2` or `set_size = 0`.

### Phase 2: Alpha Expansion (Medium-Term)
1.  **Alternative data** — news/sentiment as new heterogeneous graph node type.
2.  **RL fine-tuning (PPO)** — optimize directly for Sharpe Ratio post-supervised-pretraining.
3.  **Multi-horizon targets** — simultaneous 1d, 5d, 20d prediction heads.
