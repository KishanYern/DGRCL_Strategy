# Backtest Findings & Next Steps (v1.5 — 90-Fold Run)
*Run date: 2026-02-22 | Period: 2007-01-25 → 2026-02-20 | 4,799 trading days*

---

## Summary Statistics

| Metric | Value |
|---|---|
| Total Folds | 90 (walk-forward, 1-day step) |
| Universe | 150 stocks, 4 macro factors |
| Mean Val Loss | **0.4615 ± 0.0157** |
| Best Fold Val Loss | 0.4315 (Fold 76) |
| Worst Fold Val Loss | 0.4991 (Folds 20, 21, 36, 37, 61) |
| **Mean Rank Accuracy** | **57.5%** (+7.5pp vs 50% random baseline) |
| Peak Rank Accuracy | 61.0% (Fold 69) |
| Min Rank Accuracy | 52.4% (Fold 61) |
| Mean Magnitude MAE | 0.3517 |
| Folds w/ NaN Train Loss | **7/90 (7.8%)** — Folds 1, 20, 21, 36, 37, 59, 60, 61 |

---

## Key Findings

### 1. The Model Has Real Alpha
A 57.5% out-of-sample rank accuracy maintained across 19 years — including the GFC, COVID-19 crash, and Quantitative Tightening — is a statistically robust signal. The IQR of rank accuracy is **55.8%–59.4%**, meaning even weak folds are well above random. This is the primary result: the Masked Superset Architecture + MTL approach successfully learns portable cross-sectional structure.

### 2. NaN Training Loss Is Unresolved (Critical Blocker)
Seven folds exhibit `NaN` training loss, clustering in two high-volatility regimes:
- **GFC cluster:** Folds 1, 20, 21 (approx. 2008–2011)
- **COVID/Vol cluster:** Folds 36, 37, 59, 60, 61 (approx. 2015–2021)

The `GradScaler` prevents val loss from going NaN, but the model enters these validation windows **under-trained**, as early stopping fires with partially corrupted weights. This is the #1 blocker for live deployment.

### 3. Regime Sensitivity
| Regime | Approx. Folds | Mean Rank Acc. | Note |
|---|---|---|---|
| Recovery / Bull | 21–35 | ~57.6% | Most stable learning |
| High-Vol / Crisis | 36–39, 59–61 | ~53–55% | Worst performance |
| Post-Crisis Trending | 62–90 | ~57.8% | Best individual folds |

The model degrades in crisis regimes but **does not break** — it remains above random even in worst-case. Regime-adaptive training could push these crisis folds from 53% toward 57%.

### 4. Magnitude Head Spikes in Stress
In 4+ folds (15, 25, 35, 39), Mag MAE exceeds 0.41 during market stress:
- Average Mag MAE in calm regimes: ~0.340
- Average Mag MAE in stress regimes: ~0.415
- The λ=0.05 magnitude weight is too rigid — it needs to be dynamic.

### 5. MC Dropout Confidence Is Not Calibrated
Final-fold MC Dropout (10 passes) results:
| Stat | Value | Concern |
|---|---|---|
| Direction score mean | -0.075 | Slight systematic short bias |
| Direction score std | **0.150** | Very high epistemic uncertainty |
| Confidence mean | **0.871** | Overconfident — not calibrated |
| Rank stability | **0.735** | 26.5% of rank pairs flip between passes |

Raw confidence outputs **cannot** be used for position sizing. They require calibration (Platt Scaling or Isotonic Regression) first.

### 6. Early Stopping Fires Too Aggressively
Median stopping epoch across all folds: ~epoch 22. Worst-performing folds stopped at epoch 13–17. When early stopping is triggered inside high-volatility training windows, the model captures noise rather than signal. Folds that ran to epoch 60–85 systematically outperformed.

---

## Next Steps

### 🔴 Phase 0 — Deployment Blockers (Fix Before Live Trading)

| # | Task | Rationale |
|---|---|---|
| 0.1 | **Reduce gradient clip norm** from 1.0 → 0.5; add per-layer grad norm monitoring | Eliminates NaN training cascades in 7 crisis folds |
| 0.2 | **Calibrate confidence head** via Platt Scaling on held-out fold predictions | Required before using confidence for position sizing |
| 0.3 | **Validate long-short returns** — sort by predicted rank → compute 5-day forward return spread across quintiles | Converts rank accuracy into tradeable P&L estimate |

### 🟠 Phase 1 — Robustness (Short-Term)

| # | Task | Rationale |
|---|---|---|
| 1.1 | **Dynamic early stopping patience** — increase patience when trailing 20-day realized vol > 2σ | Lets model train longer during exactly the periods where it currently under-trains |
| 1.2 | **Adaptive magnitude weight λ** — `λ = 0.05 * (1 + α * σ_regime)` | Prevents magnitude head from polluting direction signal in normal markets |
| 1.3 | **Transaction Cost Analysis (TCA)** — integrate 5 bps/trade cost model | Verifies edge survives execution |
| 1.4 | **MC Dropout ensemble ranking** — use median rank across 10 passes instead of single-pass rank | Mechanically improves rank stability from 73.5% → est. 90%+ |

### 🟡 Phase 2 — Portfolio Construction (Medium-Term)

| # | Task | Rationale |
|---|---|---|
| 2.1 | **Mean-Variance Optimizer** — feed `dir_logits` (alpha) + `mag_preds` (variance proxy) into a convex optimizer | Replaces naive quintile selection with risk-efficient weights |
| 2.2 | **Risk Parity via Graph Clusters** — use learned adjacency matrix to cluster correlated stocks | Allocates risk equally across clusters rather than stocks, reducing correlation risk |
| 2.3 | **Regime-conditional training** — HMM or VIX-threshold regime tagger; separate early stopping thresholds per regime | Structural fix for regime sensitivity |

### 🟢 Phase 3 — Alpha Expansion (Long-Term)

| # | Task | Rationale |
|---|---|---|
| 3.1 | **Alternative data ingestion** — news/sentiment as new graph node type | Adds orthogonal signal to price-based features |
| 3.2 | **Conformalized prediction intervals** — replace confidence head with distribution-free coverage guarantees | Rigorous uncertainty quantification for position sizing |
| 3.3 | **RL fine-tuning (PPO)** — optimize directly for Sharpe Ratio post-supervised-pretraining | Closes gap between training objective and live P&L |

---

## Artifacts

| File | Description |
|---|---|
| `backtest_results/fold_results.json` | Raw per-fold metrics (90 entries) |
| `backtest_results/backtest_summary.png` | Val loss + rank accuracy timeline |
| `backtest_results/training_curve_fold_*.png` | Per-fold loss curves (all 90 folds) |
| `backtest_results/attention_heatmap_fold_90.png` | Final fold macro-to-stock attention |
| `backtest_gpu.log` | Full training log |
