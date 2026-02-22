# Backtesting & Metrics
*Last updated: 2026-02-22 — reflects 90-fold walk-forward run (2007–2026)*

## 1. Walk-Forward Validation

DGRCL uses **Walk-Forward Validation** (also known as Rolling Window Backtesting) to simulate how the model would have performed in real-time. This prevents look-ahead bias and tests the model's adaptability to changing market regimes.

### Fold Strategy (v1.5 — 90-Fold Configuration)
- **Lookback Window**: 60 days (feature construction)
- **Forecast Horizon**: 5 days
- **Step Size**: 1 day (dense, maximally overlapping folds)
- **Snapshots per fold**: 136 training snapshots, 36 validation snapshots
- **Total Folds**: 90 (covering 4,799 trading days: 2007-01-25 → 2026-02-20)
- **Universe**: Top 150 stocks by avg absolute returns + 4 macro factors
- **Active stocks per timestep**: min=137, mean=144, max=150 (dynamic via Masked Superset)

## 2. Metrics

### A. Rank Accuracy (Direction)
Measures the percentage of correctly ranked pairs within the same sector.
- **Random Baseline**: 50%
- **Edge**: >52% consistently is considered alpha.
- **Goal**: >55% (DGRCL achieves a **mean of 57.5%**, peak 61.0%, over the 90-fold run).

$$ \text{Rank Acc} = \frac{\text{Correct Pairs}}{\text{Total Valid Pairs}} $$

A pair $(i, j)$ is valid if $|r_i - r_j| > 1\%$.

### B. Magnitude Mean Absolute Error (MAE)
Measures the error in predicting the log-scaled magnitude of returns.
- **Target**: log(1 + |return| / σ)
- **Interpretation**: Lower is better. A value of ~0.34 means the model is, on average, off by roughly 0.34 log-standard-deviations. This translates to an error of approx 0.5% in raw return terms.

### C. Sharpe Ratio (Implicit)
While not directly optimized, high Rank Accuracy combined with low Magnitude MAE implies a high Sharpe Ratio strategy: betting larger on high-confidence, high-magnitude setups.

## 3. Visualization

The script automatically generates plots in `backtest_results/`:

- **`training_curve_fold_X.png`**: Loss curve for each fold. Look for smooth decay without immediate overfitting.
- **`attention_heatmap_fold_X.png`**: Adjacency matrix of the graph. Shows which stocks are attending to which. Sparse patterns indicate learned structure.
- **`backtest_summary.png`**: 
  - **Top**: Timeline of market regimes covered.
  - **Bottom Left**: Train vs Val Loss per fold.
  - **Bottom Right**: Rank Accuracy per fold vs Random Baseline.

---

## 4. Market Regimes

The backtest automatically labels periods:
- **Bull Run**: S&P 500 > 200d MA + Low Volatility
- **Recovery**: S&P 500 crossing above 50d MA after a drawdown
- **Bear Market**: S&P 500 < 200d MA
- **Consolidation**: Choppy price action within a range

DGRCL's dynamic graph structure is designed to adapt its attention weights ($A_t$) based on the current regime.
