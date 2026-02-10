"""
Data Loader for Macro-Aware DGRCL

Provides:
- load_processed_data(): Load CSVs from data_ingest.py and convert to tensors
- WalkForwardSplitter: Time-series aware train/val splitting
- create_backtest_folds(): Generate walk-forward snapshots for backtesting
"""

import os
import glob
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# DATA LOADING
# =============================================================================

def load_processed_data(
    data_dir: str = "./data/processed",
    device: torch.device = None,
    max_stocks: Optional[int] = 150
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DatetimeIndex, List[str]]:
    """
    Load processed CSV files from data_ingest.py and convert to tensors.
    
    Args:
        data_dir: Path to processed data directory
        device: Torch device (default: CPU)
        
    Returns:
        stock_tensor: [N_stocks, T, 8] stock features (N_stocks ≤ max_stocks)
        macro_tensor: [N_macros, T, 4] macro features  
        returns_tensor: [N_stocks, T] stock returns (for labels)
        dates: DatetimeIndex of time dimension
        stock_tickers: List of stock ticker symbols
    """
    if device is None:
        device = torch.device('cpu')
    
    # Load common index
    common_index_path = os.path.join(data_dir, "common_index.txt")
    if not os.path.exists(common_index_path):
        raise FileNotFoundError(
            f"Common index not found at {common_index_path}. "
            "Run data_ingest.py first to generate processed data."
        )
    
    dates = pd.to_datetime(pd.read_csv(common_index_path, header=None)[0])
    T = len(dates)
    
    # Load stock data
    stock_files = sorted(glob.glob(os.path.join(data_dir, "stock_*.csv")))
    if not stock_files:
        raise FileNotFoundError(f"No stock files found in {data_dir}")
    
    stock_dfs = []
    stock_tickers = []
    for f in stock_files:
        ticker = os.path.basename(f).replace("stock_", "").replace(".csv", "")
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        stock_dfs.append(df)
        stock_tickers.append(ticker)
    
    # Limit stock universe for VRAM management
    if max_stocks is not None and len(stock_dfs) > max_stocks:
        print(f"  Selecting top {max_stocks} stocks from {len(stock_dfs)} by avg absolute returns (liquidity proxy)...")
        avg_abs_returns = [
            df['Returns'].abs().mean() if 'Returns' in df.columns else df.iloc[:, -1].abs().mean()
            for df in stock_dfs
        ]
        top_indices = np.argsort(avg_abs_returns)[::-1][:max_stocks]
        top_indices = sorted(top_indices)  # Preserve alphabetical order
        stock_dfs = [stock_dfs[i] for i in top_indices]
        stock_tickers = [stock_tickers[i] for i in top_indices]
    
    # Load macro data
    macro_files = sorted(glob.glob(os.path.join(data_dir, "macro_*.csv")))
    if not macro_files:
        raise FileNotFoundError(f"No macro files found in {data_dir}")
    
    macro_dfs = []
    macro_names = []
    for f in macro_files:
        name = os.path.basename(f).replace("macro_", "").replace(".csv", "")
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        macro_dfs.append(df)
        macro_names.append(name)
    
    # Convert to tensors
    # Stock features: [N_stocks, T, 8]
    # Columns: ['Close', 'High', 'Low', 'Log_Vol', 'RSI_14', 'MACD', 'Volatility_5', 'Returns']
    stock_arrays = [df.values for df in stock_dfs]  # Each is [T, 8]
    stock_tensor = torch.tensor(
        np.stack(stock_arrays, axis=0),  # [N_stocks, T, 8]
        dtype=torch.float32,
        device=device
    )
    
    # Extract returns for labels (last column is 'Returns')
    returns_tensor = stock_tensor[:, :, -1].clone()  # [N_stocks, T]
    
    # Macro features: [N_macros, T, 4]
    # Columns: ['Close', 'Returns', 'MA_50', 'MA_200']
    macro_arrays = [df.values for df in macro_dfs]
    macro_tensor = torch.tensor(
        np.stack(macro_arrays, axis=0),  # [N_macros, T, 4]
        dtype=torch.float32,
        device=device
    )
    
    print(f"Loaded {len(stock_tickers)} stocks, {len(macro_names)} macro factors")
    print(f"Time range: {dates.iloc[0]} to {dates.iloc[-1]} ({T} trading days)")
    print(f"Stock tensor: {stock_tensor.shape}")
    print(f"Macro tensor: {macro_tensor.shape}")
    
    return stock_tensor, macro_tensor, returns_tensor, dates, stock_tickers


# =============================================================================
# WALK-FORWARD SPLITTER
# =============================================================================

@dataclass
class WalkForwardFold:
    """Single fold in walk-forward validation."""
    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    
    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)
    
    @property
    def val_slice(self) -> slice:
        return slice(self.val_start, self.val_end)


class WalkForwardSplitter:
    """
    Walk-forward cross-validation splitter for time-series.
    
    Creates expanding or sliding window train/val splits that respect
    temporal ordering (no future data leakage).
    
    Example with train_size=252, val_size=63, step_size=63:
    
        |----train (252 days)----|--val (63)--|           Fold 1
             |----train (252 days)----|--val (63)--|      Fold 2
                  |----train (252 days)----|--val (63)--| Fold 3
    """
    
    def __init__(
        self,
        train_size: int = 252,
        val_size: int = 63,
        step_size: int = 63,
        expanding: bool = False
    ):
        """
        Args:
            train_size: Number of days for training window
            val_size: Number of days for validation window
            step_size: Days to advance between folds
            expanding: If True, training window expands (includes all prior data)
                       If False, uses sliding window of fixed train_size
        """
        self.train_size = train_size
        self.val_size = val_size
        self.step_size = step_size
        self.expanding = expanding
    
    def split(self, total_days: int) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds.
        
        Args:
            total_days: Total number of days in dataset
            
        Returns:
            List of WalkForwardFold objects
        """
        folds = []
        fold_idx = 0
        
        # First fold: train from 0 to train_size, val from train_size to train_size+val_size
        train_start = 0
        train_end = self.train_size
        val_start = train_end
        val_end = val_start + self.val_size
        
        while val_end <= total_days:
            folds.append(WalkForwardFold(
                fold_idx=fold_idx,
                train_start=train_start if self.expanding else train_end - self.train_size,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end
            ))
            
            # Advance by step_size
            train_end += self.step_size
            val_start = train_end
            val_end = val_start + self.val_size
            fold_idx += 1
        
        return folds
    
    def get_n_folds(self, total_days: int) -> int:
        """Calculate number of folds for given data length."""
        return len(self.split(total_days))


# =============================================================================
# BACKTEST FOLD CREATION
# =============================================================================

def create_backtest_folds(
    stock_data: torch.Tensor,  # [N_s, Total_T, d_s]
    macro_data: torch.Tensor,  # [N_m, Total_T, d_m]
    returns_data: torch.Tensor,  # [N_s, Total_T]
    splitter: WalkForwardSplitter,
    window_size: int = 60,
    forecast_horizon: int = 5,
    step_size: int = 1
) -> List[Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    """
    Create backtest folds with walk-forward validation.
    
    Each fold contains:
    - train_snapshots: List of (stock_window, macro_window, future_returns) tuples
    - val_snapshots: List of (stock_window, macro_window, future_returns) tuples
    
    Args:
        stock_data: [N_s, Total_T, d_s] full stock time-series
        macro_data: [N_m, Total_T, d_m] full macro time-series
        returns_data: [N_s, Total_T] stock returns
        splitter: WalkForwardSplitter instance
        window_size: Lookback window T for each snapshot
        forecast_horizon: Days ahead for return labels
        step_size: Stride between snapshots within each fold
        
    Returns:
        List of (train_snapshots, val_snapshots) tuples, one per fold
    """
    total_t = stock_data.size(1)
    folds_data = []
    
    # Get fold boundaries
    folds = splitter.split(total_t)
    
    print(f"\nCreating {len(folds)} backtest folds:")
    print(f"  Window size: {window_size} days")
    print(f"  Forecast horizon: {forecast_horizon} days")
    print(f"  Snapshot step size: {step_size} days")
    
    for fold in folds:
        # Create snapshots for training period
        train_snapshots = _create_snapshots(
            stock_data, macro_data, returns_data,
            start_idx=fold.train_start,
            end_idx=fold.train_end,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            step_size=step_size
        )
        
        # Create snapshots for validation period
        val_snapshots = _create_snapshots(
            stock_data, macro_data, returns_data,
            start_idx=fold.val_start,
            end_idx=fold.val_end,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            step_size=step_size
        )
        
        folds_data.append((train_snapshots, val_snapshots))
        
        print(f"  Fold {fold.fold_idx + 1}: "
              f"train days [{fold.train_start}:{fold.train_end}] → {len(train_snapshots)} snapshots, "
              f"val days [{fold.val_start}:{fold.val_end}] → {len(val_snapshots)} snapshots")
    
    return folds_data


def _create_snapshots(
    stock_data: torch.Tensor,
    macro_data: torch.Tensor,
    returns_data: torch.Tensor,
    start_idx: int,
    end_idx: int,
    window_size: int,
    forecast_horizon: int,
    step_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create snapshots within a specific time range.
    
    Args:
        stock_data: Full stock data
        macro_data: Full macro data
        returns_data: Full returns data
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        window_size: Lookback window
        forecast_horizon: Days ahead for labels
        step_size: Stride between snapshots
        
    Returns:
        List of (stock_window, macro_window, future_returns) tuples
    """
    snapshots = []
    
    # We need at least window_size history and forecast_horizon future
    # First valid snapshot starts at: start_idx + window_size - 1
    # Last valid snapshot ends at: end_idx - forecast_horizon
    
    for t in range(start_idx, end_idx - window_size - forecast_horizon + 1, step_size):
        # Extract window ending at t + window_size
        stock_window = stock_data[:, t:t+window_size, :]  # [N_s, window_size, d_s]
        macro_window = macro_data[:, t:t+window_size, :]  # [N_m, window_size, d_m]
        
        # Future returns as labels (cumulative over forecast horizon)
        future_returns = returns_data[:, t+window_size:t+window_size+forecast_horizon].sum(dim=1)
        
        snapshots.append((stock_window, macro_window, future_returns))
    
    return snapshots


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_and_prepare_backtest(
    data_dir: str = "./data/processed",
    train_size: int = 252,
    val_size: int = 63,
    step_size: int = 63,
    window_size: int = 60,
    forecast_horizon: int = 5,
    snapshot_step: int = 1,
    device: torch.device = None,
    max_stocks: Optional[int] = 150
) -> Tuple[List, pd.DatetimeIndex, List[str], int, int]:
    """
    Convenience function to load data and create backtest folds in one call.
    
    Returns:
        folds: List of (train_snapshots, val_snapshots) tuples
        dates: Date index
        tickers: Stock ticker list
        num_stocks: Number of stocks
        num_macros: Number of macro factors
    """
    # Load data
    stock_data, macro_data, returns_data, dates, tickers = load_processed_data(
        data_dir=data_dir,
        device=device,
        max_stocks=max_stocks
    )
    
    # Create splitter
    splitter = WalkForwardSplitter(
        train_size=train_size,
        val_size=val_size,
        step_size=step_size
    )
    
    # Create folds
    folds = create_backtest_folds(
        stock_data=stock_data,
        macro_data=macro_data,
        returns_data=returns_data,
        splitter=splitter,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        step_size=snapshot_step
    )
    
    return folds, dates, tickers, stock_data.size(0), macro_data.size(0)


if __name__ == "__main__":
    # Demo/test loading
    try:
        folds, dates, tickers, n_stocks, n_macros = load_and_prepare_backtest()
        print(f"\nBacktest ready: {len(folds)} folds, {n_stocks} stocks, {n_macros} macro factors")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python data_ingest.py' first to generate processed data.")
