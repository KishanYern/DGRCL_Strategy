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

### `train_step(...)`
Executes one gradient descent step.
- **Args**: Model, features, returns, optimizer, loss fn, etc.
- **Returns**: dict containing `loss`, `rank_accuracy`, `mag_mae`, `grad_norm`.

### `evaluate(...)`
Evaluates the model on validation data (no gradient updates).
- **Args**: Model, data loader, loss fn.
- **Returns**: dict containing `val_loss`, `rank_accuracy`, `mag_mae`.

### `train_epoch(...)`
Runs a full epoch of training.
- **Args**: Model, loader, optimizer, loss fn.
- **Returns**: Aggregated metrics for the epoch.

### `main(...)`
Orchestrates the entire training pipeline, including:
- Data loading/splitting (Walk-Forward)
- Model initialization
- Training loop with Early Stopping
- Model checkpointing
- Backtest summary generation
