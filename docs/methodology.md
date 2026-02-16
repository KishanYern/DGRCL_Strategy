# DGRCL v1.5 Methodology

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

## 2. Training Objectives (v1.5)

In v1.5, we decoupled the learning problem into two distinct heads to solve the "training collapse" issue where the model would output safe, near-zero predictions.

### A. Sector-Aware Pairwise Ranking Loss (Direction)

Instead of asking "Will stock X go up?", we ask "Will stock X outperform stock Y in the same sector?". This is a much easier task because it removes systemic market beta.

**Loss Function:**
$$ L_{dir} = \frac{1}{|P|} \sum_{(i,j) \in P} \max(0, m - (s_i - s_j) \cdot \text{sign}(r_i - r_j)) $$

- **Pairs ($P$)**: We only compare stocks $(i, j)$ within the **same GICS sector**.
- **Margin ($m$)**: Set to `0.5`. The model must separate winners from losers by at least this margin.
- **Significance Threshold**: We only train on pairs where $|r_i - r_j| > 1\%$. If two stocks have nearly identical returns, we don't force the model to rank them.

**Why this works:** It forces the model to learn **relative alpha** rather than just predicting the market index.

### B. Log-Scaled Magnitude Prediction (Magnitude)

Predicting absolute returns is hard because returns are heavy-tailed (mostly noise, occasional 10% jumps). In v1.4, the model collapsed to predicting the mean (zero).

In v1.5, we transform the target:
$$ y_{mag} = \log(1 + \frac{|r_i|}{\sigma}) $$

- **Log-scaling** compresses outliers, making the distribution closer to normal.
- **Normalization ($\sigma$)** scales returns to a range roughly $[0, 3]$.
- **Loss**: SmoothL1Loss (Huber Loss) is used to be robust to outliers.

$$ L_{mag} = \text{SmoothL1}(\hat{y}_{mag}, y_{mag}) $$

---

## 3. Regularization & Stability

### Weight Decay
Reduced to `1e-3` (from `1e-1`) to allow the model to learn sharper decision boundaries without weights collapsing to zero.

### MC Dropout Inference
We use Monte Carlo Dropout during inference to estimate **uncertainty**. We run 10 stochastic forward passes and measure the variance of the raw logits.
- **Confidence**: Inverse of the logit standard deviation.
- **Rank Stability**: Spearman correlation of rankings across MC passes (1.0 = stable, 0.0 = random).

---

## 4. Hyperparameters (Default)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lookback` | 60 | Feature window (days) |
| `hidden_dim` | 64 | Node embedding size |
| `top_k` | 10 | Max neighbors per stock |
| `learning_rate` | 1e-4 | With Cosine Annealing |
| `patience` | 40 | Early stopping epochs |
| `loss_margin` | 0.5 | Ranking margin |

