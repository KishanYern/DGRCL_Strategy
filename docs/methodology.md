# DGRCL v1.6 Methodology

## Core Philosophy

DGRCL (Dynamic Graph Relation Contrastive Learning) treats the market as a heterogeneous graph where individual stocks and macroeconomic factors interact dynamically. Instead of treating each stock as an isolated time series, we model the **relationships** between assets to capture systemic risks and sector-wide movements.

### Why "Dynamic"?

Market correlations are not static. A tech stock might correlate with rates during inflation scares but decouple during earnings season. DGRCL learns a new adjacency matrix $A_t$ at every timestep using multi-head self-attention, allowing it to adapt to changing regimes.

---

## 1. Heterogeneous Graph Architecture

The model uses two types of nodes:
- **Stock Nodes ($N_s$)**: Individual equities with features like Return, RSI, MACD, Volatility.
- **Macro Nodes ($N_m$)**: Global factors like Oil, 10Y Yields, VIX, DXY.

### Message Passing
The graph update step involves two distinct aggregations:
1. **Stock $\to$ Stock**: Aggregating information from similar peers (learned via attention).
2. **Macro $\to$ Stock**: Injecting global context into every stock node.

$$ H^{(l+1)}_s = \text{GRU}( \text{Agg}_{\text{stock}}(H^{(l)}_s, A_t) + \text{Agg}_{\text{macro}}(H^{(l)}_m), H^{(l)}_s ) $$

---

## 2. Training Objectives

### A. Sector-Aware Pairwise Ranking Loss (Direction)

Instead of asking "Will stock X go up?", we ask "Will stock X outperform stock Y in the same sector?". This removes systemic market beta.

**Loss Function:**
$$ L_{dir} = \frac{1}{|P|} \sum_{(i,j) \in P} \max(0, m - (s_i - s_j) \cdot \text{sign}(r_i - r_j)) $$

- **Pairs ($P$)**: Only stocks $(i, j)$ within the **same GICS sector**.
- **Margin ($m$)**: Set to `0.5`.
- **Significance Threshold**: Only pairs where $|r_i - r_j| > 1\%$.

### B. Log-Scaled Magnitude Prediction

$$ y_{mag} = \log(1 + \frac{|r_i|}{\sigma}) $$

- **Log-scaling** compresses outliers.
- **Loss**: SmoothL1Loss (Huber Loss).

### C. Adaptive Magnitude Weight λ (v1.6 — Rec 4)

Instead of a fixed `λ = 0.05`, the magnitude weight scales with the realized cross-sectional volatility of each fold:

$$ \lambda = \text{clip}(0.05 + 2.0 \cdot \sigma_{\text{regime}},\; 0.05,\; 0.15) $$

In crisis regimes, magnitude predictions are harder — increasing λ forces the model to invest more capacity in getting magnitudes right when they matter most.

**Total Loss:**
$$ L_{total} = L_{dir} + \lambda(\sigma) \cdot L_{mag} $$

---

## 3. Regime-Adaptive Training (v1.6 — Recs 3, 4, 7)

Each fold is classified into a market regime based on the trailing cross-sectional return volatility:

| Regime | Vol Range | λ | Patience |
|--------|-----------|---|----------|
| **Calm** | < 0.20 | 0.05–0.08 | Base (10) |
| **Normal** | 0.20–0.50 | 0.08–0.12 | Base (10) |
| **Crisis** | > 0.50 | 0.12–0.15 | Extended (20) |

This volatility is computed via `compute_regime_vol()` — the standard deviation of mean absolute cross-sectional returns over the last 20 training snapshots.

---

## 4. Confidence Calibration (v1.6 — Recs 2, 6)

### Temperature Scaling (Rec 2)
Learns a scalar $T$ such that $P(up) = \sigma(logit / T)$ minimizes BCE on held-out data.

### Conformal Prediction (Rec 6)
Provides distribution-free coverage guarantee:
$$ P(\text{true label} \in \text{prediction set}) \geq 1 - \alpha $$

At test time, stocks with `set_size = 2` (both UP and DOWN included) should trigger **abstention** — the model has genuine uncertainty.

---

## 5. Regularization & Stability

### Gradient Clipping (Rec 1)
- `max_grad_norm = 0.5` (tightened from 1.0) to prevent NaN cascades.
- NaN batches are **skipped** in metric accumulation rather than corrupting the epoch.

### MC Dropout Inference (Rec 5)
10 stochastic forward passes. v1.6 uses **median rank** across passes (instead of mean) for more robust stock selection.

### Per-Fold GradScaler (Rec 1)
`GradScaler` is reinitialized per fold to prevent AMP state leakage between folds.

---

## 6. Hyperparameters (Default)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lookback` | 60 | Feature window (days) |
| `hidden_dim` | 64 | Node embedding size |
| `top_k` | 10 | Max neighbors per stock |
| `learning_rate` | 1e-4 | With Cosine Annealing |
| `patience` | 10 (base) / 20 (crisis) | Dynamic early stopping |
| `loss_margin` | 0.5 | Ranking margin |
| `mag_weight` | 0.05–0.15 | Adaptive λ |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `mc_passes` | 10 | MC Dropout inference passes |
