# DGRCL_Strategy

Macro-Aware Dynamic Graph Relation Contrastive Learning (DGRCL) v1.1 for market-neutral trading.

## Overview

A Heterogeneous Graph Neural Network that models the stock market as a graph where:

-   **Stock nodes** represent individual equities with technical features
-   **Macro nodes** represent macroeconomic factors (Oil, Yields, VIX, Currency) as first-class graph citizens

## Key Features

-   **Heterogeneous Graph Topology**: Two node types (Stock, Macro) with hybrid edge connections
-   **Dynamic Edge Learning**: Stock→Stock adjacency computed via self-attention at each timestep
-   **Custom Message Passing**: Dual aggregation from stock neighbors and macro factors
-   **Monte Carlo Dropout**: Epistemic uncertainty estimation via stochastic inference
-   **Pairwise Ranking Loss**: Predicts relative rank rather than absolute price

## Architecture

```
Raw Time-Series → LSTM Encoder → Dynamic Graph Learner → MacroPropagation → MC Dropout Head → Ranking Scores
                                       ↑                        ↑
                                  Stock Embeddings         Macro Embeddings
```

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install torch torch_geometric pytest
```

## Usage

```python
from macro_dgrcl import MacroDGRCL

# Initialize model
model = MacroDGRCL(
    num_stocks=50,
    num_macros=4,
    stock_feature_dim=8,
    macro_feature_dim=4,
    hidden_dim=64,
    top_k=10,
    mc_dropout=0.3
)

# Forward pass
scores, embeddings = model(stock_features, macro_features)

# Inference with uncertainty
mu, sigma2, embeddings = model.predict_with_uncertainty(
    stock_features, macro_features, n_samples=30
)
```

## Training

```bash
# Train with synthetic data (quick demo)
python train.py

# Train with real market data (requires data_ingest.py first)
python train.py --real-data
```

### Backtesting with Walk-Forward Validation

For proper backtesting on historical data:

```bash
# 1. Download and process S&P 500 + macro data
python data_ingest.py

# 2. Run walk-forward backtest
python train.py --real-data
```

This performs walk-forward cross-validation:
- **Train window**: 252 days (~1 year)
- **Validation window**: 63 days (~1 quarter)
- **Step**: Advance 63 days between folds

## Testing

```bash
python -m pytest test_macro_dgrcl.py -v
```

## Files

| File                  | Description                                    |
| --------------------- | ---------------------------------------------- |
| `macro_dgrcl.py`      | Core model architecture                        |
| `losses.py`           | Pairwise Ranking + InfoNCE Contrastive losses  |
| `train.py`            | Training loop with walk-forward backtesting    |
| `data_ingest.py`      | S&P 500 + Macro data ingestion                 |
| `data_loader.py`      | CSV→Tensor loader + WalkForwardSplitter        |
| `test_macro_dgrcl.py` | Pytest suite (22 tests)                        |

## License

MIT
