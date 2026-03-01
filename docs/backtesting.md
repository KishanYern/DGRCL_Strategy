# Backtesting & Metrics
*Last updated: 2026-03-01 — reflects v1.6 90-fold walk-forward run (2007–2026)*

## 1. Walk-Forward Validation

DGRCL uses **Walk-Forward Validation** (Rolling Window Backtesting) to simulate real-time performance. This prevents look-ahead bias and tests adaptability to changing market regimes.

### Fold Strategy (v1.6 — 90-Fold Configuration)
- **Lookback Window**: 60 days (feature construction)
- **Forecast Horizon**: 5 days
- **Step Size**: 1 day (dense, maximally overlapping folds)
- **Snapshots per fold**: 136 training, 36 validation
- **Total Folds**: 90 (covering 4,799 trading days: 2007-01-25 → 2026-02-20)
- **Universe**: Top 150 stocks by avg absolute returns + 4 macro factors
- **Active stocks per timestep**: min=137, mean=144, max=150 (dynamic via Masked Superset)

## 2. Metrics

### A. Rank Accuracy (Direction)
Measures the percentage of correctly ranked pairs within the same sector.
- **Random Baseline**: 50%
- **Edge**: >52% consistently is considered alpha.
- **v1.6 Result**: Mean **57.5%**, peak **61.0%** (Fold 50), all 90 folds beat random chance (min 51.7%).

$$ \text{Rank Acc} = \frac{\text{Correct Pairs}}{\text{Total Valid Pairs}} $$

### B. Magnitude Mean Absolute Error (MAE)
- **v1.6 Result**: Mean **0.352** — stable across all folds regardless of regime.
- Error of ~0.35 log-standard-deviations ≈ ±0.5% in raw return terms.

### C. Long-Short Alpha (v1.6 — Rec 8)
Sector-balanced long-short return spread: buy top-20% and sell bottom-20% within each GICS sector.
- **v1.6 Result**: Mean **+31.57%** annualized active return, positive in 90/90 folds. Peak regime: Normal (+33.64%), Calm (+26.46%).

### D. Confidence Calibration (v1.6 — Recs 2, 6)
- **ECE (raw)**: 0.015 — near-perfect calibration.
- **Conformal coverage**: 89.4% (target ≥ 90% for α=0.10).

## 3. Visualization

The script automatically generates plots in `backtest_results/`:

- **`training_curve_fold_X.png`**: Loss curve per fold. Smooth decay without immediate overfitting.
- **`attention_heatmap_fold_X.png`**: Adjacency matrix—shows which stocks attend to which.
- **`backtest_summary.png`**:
  - **Top**: Timeline of market regimes covered.
  - **Middle Left**: Train vs Val Loss per fold.
  - **Middle Center**: Loss trend across folds.
  - **Middle Right**: Rank Accuracy per fold vs random baseline.
  - **Bottom**: Magnitude MAE per fold.
- **`reliability_diagram_before.png`**: Confidence calibration diagram (pre-calibration).
- **`reliability_diagram_after_temp.png`**: After Temperature Scaling.

---

## 4. Market Regimes

### Macro Regime Labels (Backtest Summary Plot)
The backtest automatically labels periods:
- **Bull Run**: S&P 500 > 200d MA + Low Volatility
- **Recovery**: S&P 500 crossing above 50d MA after a drawdown
- **Bear Market**: S&P 500 < 200d MA
- **Consolidation**: Choppy price action within a range

### Per-Fold Regime Classification (v1.6 — Rec 7)
Each fold is tagged by its training-window volatility:
- **Calm** (vol < 0.20): Quiet trending — λ ≈ 0.05, base patience.
- **Normal** (0.20–0.50): Base case — λ ≈ 0.08–0.12, base patience.
- **Crisis** (vol > 0.50): GFC/COVID-type — λ = 0.15, patience doubled.

Regime, realized_vol, and adaptive_lambda are persisted per fold in `fold_results.json`.

---

## 5. NaN Stability (v1.6)

| Backtest Version | NaN Folds | Mitigation |
|-----------------|-----------|------------|
| v1.5 | ~12% | None |
| v1.6 | **0% (0/90)** | `max_grad_norm=0.5` + explicit zero-padding isolation + active-only $\sigma$ scaling |

The v1.6 interventions completely eliminated the NaN cascade problem. Even during extreme volatility spikes (GFC, COVID), the model remained 100% stable in all 90 out-of-sample walk-forward folds.

## 6. Running the Full Pipeline

```bash
# AMD GPU (recommended):
bash run_gpu_training.sh --real-data --save-calibration-data

# Manual (CPU / NVIDIA):
python3 train.py --real-data --save-calibration-data
python3 confidence_calibration.py --results-dir ./backtest_results
```
