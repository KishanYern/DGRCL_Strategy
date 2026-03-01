# DGRCL API Reference

## Class: `MacroDGRCL`
(`macro_dgrcl.py`)
Inherits from `nn.Module`.

### Parameters
- **num_stocks** (`int`): Number of stock nodes.
- **num_macros** (`int`): Number of macro nodes.
- **stock_feature_dim** (`int`): Input feature dimension for stocks (8).
- **macro_feature_dim** (`int`): Input feature dimension for macro (4).
- **hidden_dim** (`int`): Hidden state size (64).
- **top_k** (`int`): Max neighbors per stock in dynamic graph (10).
- **head_dropout** (`float`): Dropout rate in output heads (0.3).

### Methods
- **forward(stock_features, macro_features)**:
  - **Returns**: `(dir_logits, mag_preds)`
  - **Shapes**: `[batch, 1]`, `[batch, 1]`

## Class: `DynamicGraphLearner`
(`macro_dgrcl.py`)

Learns the adjacency matrix $A_t$ for stock-to-stock connections.

### Parameters
- **input_dim** (`int`): Feature dimension of stock nodes.
- **top_k** (`int`): Sparsity constraint.

### Methods
- **forward(x, batch_indices)**:
  - **Returns**: `(edge_index, edge_weight)`
  - **Shape**: `[2, edges]`, `[edges]`

## Class: `VisualizedMultiTaskHead`
(`macro_dgrcl.py`)

Shared backbone -> separate heads for direction and magnitude.

### Parameters
- **input_dim** (`int`): Hidden state from GNN (64).
- **dropout** (`float`): Output dropout (0.3).

### Methods
- **forward(x)**:
  - **Returns**: `(dir_logits, mag_output)`
  - **Notes**: `mag_output` passes through `Softplus()` to ensure non-negative predictions.

---

## Training Functions (`train.py`)

### `EarlyStopping`
Early stopping with dynamic patience (Rec 3).
- **Parameters**: `base_patience=10`, `max_patience=20`
- **update_patience(realized_vol, high_vol_threshold=0.50)**: Doubles patience in crisis regimes.

### `compute_regime_vol(train_snapshots, lookback=20)`
Computes trailing cross-sectional return volatility from the most recent training snapshots. Returns the standard deviation of mean absolute returns — higher values indicate GFC/COVID-like regimes.

### `classify_regime(realized_vol, low_threshold=0.20, high_threshold=0.50)`
Tags a fold as `'calm'`, `'normal'`, or `'crisis'` based on realized vol (Rec 7).

### `compute_adaptive_lambda(realized_vol, base_lambda=0.05, max_lambda=0.15, sensitivity=2.0)`
Adjusts magnitude loss weight dynamically per fold (Rec 4). Returns λ clipped to `[base_lambda, max_lambda]`.

### `train_step(...)`
Single gradient descent step with NaN guard.
- **Returns**: dict containing `loss`, `rank_accuracy`, `mag_mae`, `grad_norm`.
- **NaN guard**: If `loss_total` is non-finite, skips backward pass and returns NaN metrics.

### `train_epoch(...)`
Full epoch of training with NaN batch skipping.
- **Returns**: dict with epoch-level metrics (averaged across valid batches only).
- **Extra fields**: `num_batches`, `nan_batches`.

### `evaluate(...)`
Evaluates model on validation data (no gradient updates).
- **Returns**: dict containing `val_loss`, `rank_accuracy`, `mag_mae`.

### `mc_dropout_inference(model, stock_feat, macro_feat, ...)`
MC Dropout with 10 passes (Rec 5). Returns `median_dir_rank` for stable stock selection.

### `compute_long_short_alpha(dir_logits, returns, sector_ids)`
Sector-balanced long-short alpha: buy top-20%, sell bottom-20% within each GICS sector (Rec 8).

### `main(...)`
Orchestrates the entire training pipeline:
- Data loading and walk-forward fold splitting
- Per-fold regime classification and adaptive hyperparameter tuning
- Training with dynamic patience and NaN-guarded epochs
- Model checkpointing and backtest summary generation
- Optional: save calibration data for post-processing

---

## Calibration Module (`confidence_calibration.py`)

### `TemperatureScaler` (Rec 2)
Learns scalar T to scale logits before sigmoid: `P(up) = σ(logit / T)`.
- **fit(logits, labels)**: Minimizes BCE via L-BFGS.
- **calibrate(logits)**: Returns calibrated probabilities.

### `PlattScaler` (Rec 2 — alternative)
Learns `P(up) = σ(a · logit + b)` via logistic regression (requires scikit-learn).
- **fit(logits, labels)**: Fits via sklearn `LogisticRegression`.
- Gracefully skipped if scikit-learn is not installed.

### `ConformalPredictor` (Rec 6)
Distribution-free prediction sets with coverage guarantee ≥ 1-α.
- **calibrate(logits, labels)**: Computes conformal threshold `q_hat`.
- **predict_set(logits)**: Returns prediction sets with `include_up`, `include_down`, `set_size`.
- **empirical_coverage(logits, labels)**: Measures actual coverage on test data.

### Utility Functions
- **`compute_ece(probs, labels)`**: Expected Calibration Error.
- **`reliability_diagram(probs, labels)`**: Plots confidence reliability diagram.
- **`run_post_processing(results_dir)`**: Standalone entry point for calibration post-processing.
