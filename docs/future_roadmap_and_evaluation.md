# Deep Evaluation & Future Roadmap (v1.5)
*Last updated: 2026-02-22 after full 90-fold walk-forward run (2007–2026)*

## 1. Deep Evaluation of Current Performance

### Overall Metrics (Full 90-Fold Run: 2007–2026)
*   **Mean Rank Accuracy**: **57.5%** (Baseline: 50%)
    *   Statistically robust over 19 years of OOS data including GFC and COVID-19. Correctly ranking ~57–60% of same-sector pairs is sufficient for a profitable sector-neutral long-short strategy.
*   **Mean Val Loss**: **0.4615 ± 0.0157**
*   **Mean Magnitude MAE**: **0.3517** (Log-Scale); spikes to >0.41 in high-volatility regimes.
*   **NaN Training Folds**: 7/90 (7.8%) — concentrated in GFC and COVID regimes (see `backtest_findings.md`).

### Regime Analysis
The model exhibits performance cyclicity correlated with market regimes:

| Regime | Approx. Folds | Mean Rank Acc. | Assessment |
| :--- | :--- | :--- | :--- |
| **Recovery / Bull** | 21–35 | **~57.6%** | **Stable**. Clear trends and orderly sector rotation — most learnable. |
| **High-Vol / Crisis** | 36–39, 59–61 | **~53–55%** | **Most degraded**. NaN train losses; model under-trains entering these folds. |
| **Post-Crisis Trending** | 62–90 | **~57.8%** | **Strongest individual folds** (F69: 61.0%, F76: 60.5%). |

### Generalization Gap
*   **Train Loss**: ~0.453 (mean where non-NaN)
*   **Val Loss**: ~0.4615
*   **Gap**: ~0.008 (tighter than previous runs)
*   **Conclusion**: The model is **not overfitting**. The Masked Superset Architecture and early stopping work effectively across all 90 folds.

---

## 2. Current Fail Points & Limitations

### A. NaN Training Loss in Crisis Regimes (Critical)
Seven folds (1, 20, 21, 36, 37, 59, 60, 61) exhibit NaN training loss — all in high-volatility regimes (GFC and COVID). The model enters the corresponding validation windows under-trained, capping rank accuracy at ~52–55% in those folds.
*   **Fix**: Reduce `gradient_clip_norm` from 1.0 → 0.5; add per-layer gradient norm monitoring. See `backtest_findings.md` § Phase 0.

### B. Confidence Head Overconfidence
MC Dropout (10 passes, Fold 90) gives a mean confidence of 87.1% despite direction score std of 0.15 — the model is not calibrated.
*   **Fix**: Apply Platt Scaling or Isotonic Regression to the confidence head outputs before using them for position sizing.

### C. Rank Stability Under Uncertainty
Only 73.5% of stock rank orderings are stable across MC Dropout passes — ~26.5% flip between passes. Using raw single-pass ranks for portfolio construction amplifies this instability.
*   **Fix**: Use median rank across 10 MC Dropout passes at inference time.

### D. Execution & Costs (Not Modeled)
The current backtest does not model slippage or commissions.
*   **Risk**: A high-turnover daily rebalancing strategy at 57.5% accuracy could be net-negative after fees.
*   **Fix**: Integrate a 5 bps/trade cost model (TCA) before evaluating live viability.

---

## 3. Future Roadmap

> See `backtest_findings.md` for the full prioritized next-steps breakdown.

### Phase 0: Deployment Blockers (Do First)
1.  **Fix NaN gradient cascades** — clip `max_norm` to 0.5, add per-layer monitoring.
2.  **Calibrate confidence head** — Platt Scaling on held-out fold predictions.
3.  **Validate long-short returns** — quintile return spread as live P&L proxy.

### Phase 1: Robustness (Short-Term)
1.  **Dynamic early stopping patience** — increase patience in high-vol regimes.
2.  **Adaptive magnitude weight λ** — `λ = 0.05 * (1 + α * σ_regime)`.
3.  **Transaction Cost Analysis (TCA)** — 5 bps/trade cost model integration.
4.  **MC Dropout ensemble ranking** — median rank across 10 passes for stability.

### Phase 2: Portfolio Construction (Medium-Term)
1.  **Mean-Variance Optimizer** — feed `dir_logits` + `mag_preds` into a convex optimizer.
2.  **Risk Parity via graph clusters** — equal-risk allocation across learned stock clusters.
3.  **Regime-conditional training** — HMM/VIX tagger with regime-specific hyperparameters.

### Phase 3: Alpha Expansion (Long-Term)
1.  **Alternative data** — news/sentiment as new heterogeneous graph node type.
2.  **Conformalized prediction intervals** — replace confidence head with distribution-free coverage.
3.  **RL fine-tuning (PPO)** — optimize directly for Sharpe Ratio post-supervised-pretraining.
