# Backtesting & Metrics
*Last updated: 2026-03-03 — reflects v1.7 Phase 1 Portfolio Construction (2007–2026)*

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

## 6. Phase 1: Portfolio Construction (v1.7)

Phase 1 replaces the naive sector L/S baseline with optimized portfolio weights derived from model outputs. Three strategies are available, all sharing a common pre-processing pipeline:

```
MacroDGRCL outputs → ConformalGate (abstention) → ExpectedReturn (μ) → Covariance (Σ) → Optimizer → weights w
```

### Strategies

| Strategy | `--portfolio-method` | Description |
|---|---|---|
| **MVO** | `mvo` | Markowitz Mean-Variance via cvxpy. Dollar-neutral, gross leverage ≤ 2.0, adaptive per-stock caps. |
| **Risk Parity** | `riskparity` | Equal-risk-contribution across spectral clusters from the attention graph. |
| **Naive L/S** | `naive` | Baseline: top-20%/bottom-20% equal-weight long-short (no optimizer). |

### Portfolio Metrics

| Metric | Description |
|---|---|
| **Sharpe (gross)** | Annualized Sharpe on raw P&L (5-day horizon, 252/5 periods/year) |
| **Sharpe (net)** | Sharpe after 5 bps/trade TCA deduction |
| **Turnover** | Mean absolute weight change between consecutive snapshots |
| **MaxDD** | Maximum drawdown on cumulative P&L within a fold |
| **Calmar** | Total PnL / worst single-fold max drawdown |
| **Gated %** | Fraction of active stocks removed by conformal abstention |
| **TCA Cost** | Cumulative transaction cost at 5 bps per unit of turnover |

### Conformal Abstention Gate

Per-fold calibration on the first half of validation snapshots, applied to the second half:
- `set_size == 1` → trade (model assigns exactly one direction)
- `set_size == 0` → abstain (model rejects both directions)
- `set_size == 2` → abstain (model is ambiguous)

**Minimum-universe guard**: If gating leaves fewer than `min_tradable=20` stocks, the most confident ambiguous stocks are progressively re-admitted to prevent dangerous optimizer concentration.

### Running Portfolio Backtests

```bash
# MVO (recommended — production candidate):
python3 train.py --real-data --epochs 100 --portfolio-method mvo \
    --output-dir ./backtest_results/phase1_mvo_v2

# Risk Parity:
python3 train.py --real-data --epochs 100 --portfolio-method riskparity \
    --output-dir ./backtest_results/phase1_riskparity

# Naive L/S baseline:
python3 train.py --real-data --epochs 100 --portfolio-method naive \
    --output-dir ./backtest_results/phase1_naive
```

---

## 7. Running the Base Pipeline

```bash
# AMD GPU (recommended):
bash run_gpu_training.sh --real-data --save-calibration-data

# Manual (CPU / NVIDIA):
python3 train.py --real-data --save-calibration-data
python3 confidence_calibration.py --results-dir ./backtest_results
```

---

## 8. Benchmark Comparison

After training, run the benchmark suite to compare DGRCL against a tiered set of baseline models on the same 90 folds:

```bash
# Full run — all 7 models (includes LSTM-Only and LightGBM training per fold):
python run_benchmarks.py --real-data

# Classical factors only — fast, no GPU required:
python run_benchmarks.py --real-data --skip-trainable

# Skip FF3 download (offline environments):
python run_benchmarks.py --real-data --skip-ff3
```

### Benchmark Tiers

| Tier | Models | Purpose |
|------|--------|---------|
| Tier 1 — Null | Random, Prior-Day Persistence | Floor; confirms framework is unbiased |
| Tier 2 — Classical | Momentum 12-1M, Short-Term Reversal, Low Volatility | Rules-based factor benchmarks |
| Tier 3 — ML Ablations | LSTM-Only (No Graph), LightGBM LambdaRank | Isolate contribution of graph/macro architecture |

### v1.6 Benchmark Results Summary

| Model | Rank Accuracy | L/S Spread (z) | FF3 α | FF3 t-stat |
|-------|:---:|:---:|:---:|:---:|
| **DGRCL v1.6** | **57.56%** | **+0.865** | — | — |
| LSTM-Only | 57.41% | +0.719 | +0.720 | +15.5 |
| LightGBM | 56.52% | +0.792 | +0.792 | +11.9 |
| Prior-Day Persistence | 50.37% | +0.004 | -0.004 | -0.07 |
| Random | 50.00% | -0.063 | — | — |
| Short-Term Reversal | 49.25% | +0.042 | -0.082 | -1.41 |
| Low Volatility | 48.60% | -0.225 | -0.225 | -3.39 |
| Momentum 12-1M | **46.41%** | **-0.384** | **-0.395** | **-6.41** |

Full benchmark analysis, per-regime breakdowns, and artifact paths: see [`docs/benchmark_results.md`](benchmark_results.md).
