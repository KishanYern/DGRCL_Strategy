# Deep Evaluation & Future Roadmap (v1.5)

## 1. Deep Evaluation of Current Performance

### Overall Metrics (2021-2026 Backtest)
*   **Mean Rank Accuracy**: **58.2%** (Baseline: 50%)
    *   This indicates a strong edge. Correctly ranking ~6 out of 10 pairs consistently is sufficient for a profitable long-short strategy.
*   **Mean Magnitude MAE**: **0.339** (Log-Scale)
    *   The model's magnitude predictions are stable and low-variance, unlike v1.4.

### Regime Analysis
The model exhibits performance cyclicity correlated with market regimes:

| Regime | Period (Approx) | Rank Accuracy | Assessment |
| :--- | :--- | :--- | :--- |
| **Recovery / Bull** | 2021 - Mid 2022 | **~60%** | **Strongest**. The model excels when trends are clear and sector rotation is orderly. |
| **Bear / Transition** | Mid 2022 - 2023 | **~59%** | **Resilient**. The pairwise ranking approach neutralizes market beta, allowing it to find relative winners even in down markets. |
| **Consolidation** | 2023 - 2024 | **~56%** | **Weakest**. In choppy, range-bound markets, sector dispersion decreases, making ranking harder and noisier. |
| **Late Cycle** | 2025 - Present | **~57.5%** | **Moderate**. As trends mature and volatility decreases, the edge compresses slightly but remains positive. |

### Generalization Gap
*   **Train Loss**: ~0.42
*   **Val Loss**: ~0.45
*   **Gap**: ~0.03
*   **Conclusion**: The model is **not overfitting**. The dedicated validation set and Early Stopping (patience=40) are working effectively. The gap is consistent across folds, indicating robust hyperparameters.

---

## 2. Current Fail Points & Limitations

### A. Consolidation Weakness
The drop to ~56% accuracy in consolidation periods (Folds 8-9) is the primary model weakness.
*   **Cause**: When the market moves sideways, the "signal-to-noise" ratio of returns drops. A 1% difference between stocks might be noise rather than alpha.
*   **Risk**: If transaction costs are high, a 56% win rate might purely cover costs without profit (churn).

### B. Magnitude Calibration
While stable, `SmoothL1Loss` on log-targets is a heuristic.
*   **Issue**: It treats under-prediction and over-prediction symmetrically. In trading, missing a big move (under-prediction) is an opportunity cost, but over-betting on a false move (over-prediction) is a realized loss.

### C. Execution & Costs (Not Modeled)
The current backtest assumes varying position sizes based on magnitude but **does not model slippage or commissions**.
*   **Risk**: A high-turnover strategy (rebalancing every day/week) with 58% accuracy could be net negative after fees.

---

## 3. Future Roadmap

### Phase 1: Robustness (Immediate)
1.  **Regime-Weighted Training**: Implement a sample weighting scheme where losses during "high confidence" regimes count more.
2.  **Transaction Cost Analysis (TCA)**: Integrate a simple cost model (e.g., 5bps per trade) into the backtest to see if the edge survives.
3.  **Dynamic Margin**: Instead of fixed `margin=0.5`, make the margin a function of current market volatility (e.g., `margin = 0.5 * VIX_factor`).

### Phase 2: Portfolio Construction (Short Term)
1.  **Mean-Variance Optimizer**: Instead of raw predictions, feed `dir_logits` (Alpha) and `mag_preds` (Variance proxy) into a convex optimizer to generate weights.
2.  **Risk Parity**: Use the graph's learned adjacency matrix to cluster correlated stocks and allocate risk equally across clusters, rather than stocks.

### Phase 3: Alpha Expansion (Long Term)
1.  **Alternative Data**: Ingest Sentiment Data (News/Twitter) as a new node type in the heterogeneous graph.
2.  **Higher Frequency**: Move from Daily to Hourly bars. The GNN architecture is time-agnostic and should scale.
3.  **Reinforcement Learning**: Fine-tune the trained model using RL (PPO) to optimize directly for Sharpe Ratio rather than classification/regression loss.
