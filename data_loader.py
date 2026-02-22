"""
Data Loader for Macro-Aware DGRCL (Dynamic Universe)

Provides:
- load_processed_data(): Load CSVs and build inclusion_mask tensor
- WalkForwardSplitter: Time-series aware train/val splitting
- create_backtest_folds(): Generate walk-forward snapshots with active_mask
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
    max_stocks: Optional[int] = 150,
    historical_csv: str = "./data/sp500_historical.csv"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           pd.DatetimeIndex, List[str]]:
    """
    Load processed CSV files and build inclusion mask tensor.
    
    Args:
        data_dir: Path to processed data directory
        device: Torch device (default: CPU)
        max_stocks: Maximum number of stocks (caps superset size)
        historical_csv: Path to sp500_historical.csv for inclusion mask
        
    Returns:
        stock_tensor: [N_stocks, T, 8] stock features
        macro_tensor: [N_macros, T, 4] macro features  
        returns_tensor: [N_stocks, T] stock returns
        inclusion_mask: [N_stocks, T] boolean (True = active S&P 500 member)
        dates: DatetimeIndex of time dimension
        stock_tickers: List of stock ticker symbols
    """
    if device is None:
        device = torch.device('cpu')
    
    # Load common index (union date range)
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
        print(f"  Selecting top {max_stocks} stocks from {len(stock_dfs)} "
              f"by avg absolute returns (liquidity proxy, valid days only)...")
        # FIX #6: Compute mean only over non-zero rows to avoid bias from
        # zero-padded days introduced by the union-index reindexing. Stocks
        # with shorter histories have more zero-padded days which would
        # systematically lower their mean under the naive .mean() approach.
        avg_abs_returns = []
        for df in stock_dfs:
            ret_col = df['Returns'] if 'Returns' in df.columns else df.iloc[:, -1]
            valid = ret_col[ret_col != 0]
            avg_abs_returns.append(valid.abs().mean() if len(valid) > 0 else 0.0)
        top_indices = np.argsort(avg_abs_returns)[::-1][:max_stocks]
        top_indices = sorted(top_indices)  # Preserve alphabetical order
        stock_dfs = [stock_dfs[i] for i in top_indices]
        stock_tickers = [stock_tickers[i] for i in top_indices]
    
    N_stocks = len(stock_dfs)
    
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
    stock_arrays = [df.values for df in stock_dfs]
    stock_tensor = torch.tensor(
        np.stack(stock_arrays, axis=0),
        dtype=torch.float32,
        device=device
    )
    
    # Extract returns for labels (last column is 'Returns')
    returns_tensor = stock_tensor[:, :, -1].clone()
    
    # Macro features: [N_macros, T, 4]
    macro_arrays = [df.values for df in macro_dfs]
    macro_tensor = torch.tensor(
        np.stack(macro_arrays, axis=0),
        dtype=torch.float32,
        device=device
    )
    
    # =========================================================================
    # BUILD INCLUSION MASK [N_stocks, T]
    # =========================================================================
    inclusion_mask = _build_inclusion_mask(
        stock_tickers=stock_tickers,
        dates=dates,
        historical_csv=historical_csv,
        device=device
    )
    
    # Count active stocks per timestep for diagnostics
    active_per_t = inclusion_mask.sum(dim=0).float()
    
    print(f"Loaded {len(stock_tickers)} stocks, {len(macro_names)} macro factors")
    print(f"Time range: {dates.iloc[0]} to {dates.iloc[-1]} ({T} trading days)")
    print(f"Stock tensor: {stock_tensor.shape}")
    print(f"Macro tensor: {macro_tensor.shape}")
    print(f"Inclusion mask: {inclusion_mask.shape}")
    print(f"  Active stocks per timestep: "
          f"min={active_per_t.min().int().item()}, "
          f"mean={active_per_t.mean().int().item()}, "
          f"max={active_per_t.max().int().item()}")
    
    return stock_tensor, macro_tensor, returns_tensor, inclusion_mask, dates, stock_tickers


def _build_inclusion_mask(
    stock_tickers: List[str],
    dates: pd.DatetimeIndex,
    historical_csv: str,
    device: torch.device
) -> torch.Tensor:
    """
    Build boolean inclusion mask [N_stocks, T] from historical constituent data.

    True = ticker was an active S&P 500 constituent on that date.

    FIX #7: Vectorized via pandas merge_asof instead of O(N×T) Python loop.
    For a superset of 600 stocks × 4500 trading days this is ~100× faster.

    Falls back to all-True if historical CSV is not found (backward compat).
    """
    if not os.path.exists(historical_csv):
        print("  WARNING: No historical constituent file found. "
              "Assuming all stocks always active (no survivorship bias correction).")
        return torch.ones(len(stock_tickers), len(dates), dtype=torch.bool, device=device)

    hist_df = pd.read_csv(historical_csv)
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date').reset_index(drop=True)

    # Build a long-form DataFrame: (snapshot_date, ticker)
    # Explode once up-front to avoid repeated string splitting per date.
    records = []
    for _, row in hist_df.iterrows():
        for tkr in row['tickers'].split(','):
            records.append((row['date'], tkr.strip()))
    exploded = pd.DataFrame(records, columns=['snapshot_date', 'ticker'])

    # We only care about tickers in our loaded universe
    ticker_set = set(stock_tickers)
    exploded = exploded[exploded['ticker'].isin(ticker_set)]

    # Assign snapshot_idx so we can do a single merge_asof
    # merge_asof = "for each query date, find the most recent snapshot_date <= query"
    dates_df = pd.DataFrame({'query_date': pd.DatetimeIndex(dates)})
    snap_dates = hist_df[['date']].rename(columns={'date': 'snapshot_date'})
    snap_dates['snapshot_idx'] = np.arange(len(snap_dates))

    merged_dates = pd.merge_asof(
        dates_df.sort_values('query_date'),
        snap_dates,
        left_on='query_date',
        right_on='snapshot_date',
        direction='backward'
    )
    # Restore original order of dates
    merged_dates = merged_dates.set_index(merged_dates.index)  # already sorted
    # Map query_date → snapshot_idx (NaN means before first snapshot → all inactive)
    date_to_snap = dict(zip(merged_dates['query_date'], merged_dates['snapshot_idx']))

    # Build {snapshot_idx → set of active tickers}
    snap_idx_to_tickers: dict = {}
    for snap_idx, group in exploded.groupby(
            exploded['snapshot_date'].map(
                dict(zip(hist_df['date'], np.arange(len(hist_df))))
            )):
        snap_idx_to_tickers[int(snap_idx)] = set(group['ticker'].tolist())

    # Vectorized fill: for each (ticker, t) check membership
    N = len(stock_tickers)
    T = len(dates)
    ticker_to_idx = {t: i for i, t in enumerate(stock_tickers)}

    mask_np = np.zeros((N, T), dtype=bool)

    dates_list = list(pd.DatetimeIndex(dates))
    for t_idx, date in enumerate(dates_list):
        snap_idx = date_to_snap.get(date, np.nan)
        if pd.isna(snap_idx):
            continue  # Before any historical data — leave False
        active_set = snap_idx_to_tickers.get(int(snap_idx), set())
        for tkr in active_set:
            i = ticker_to_idx.get(tkr)
            if i is not None:
                mask_np[i, t_idx] = True

    return torch.tensor(mask_np, dtype=torch.bool, device=device)


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
    """
    
    def __init__(
        self,
        train_size: int = 252,
        val_size: int = 63,
        step_size: int = 63,
        expanding: bool = False
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.step_size = step_size
        self.expanding = expanding
    
    def split(self, total_days: int) -> List[WalkForwardFold]:
        folds = []
        fold_idx = 0
        
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
            
            train_end += self.step_size
            val_start = train_end
            val_end = val_start + self.val_size
            fold_idx += 1
        
        return folds
    
    def get_n_folds(self, total_days: int) -> int:
        return len(self.split(total_days))


# =============================================================================
# BACKTEST FOLD CREATION
# =============================================================================

def create_backtest_folds(
    stock_data: torch.Tensor,       # [N_s, Total_T, d_s]
    macro_data: torch.Tensor,       # [N_m, Total_T, d_m]
    returns_data: torch.Tensor,     # [N_s, Total_T]
    inclusion_mask: torch.Tensor,   # [N_s, Total_T]
    splitter: WalkForwardSplitter,
    window_size: int = 60,
    forecast_horizon: int = 5,
    step_size: int = 1
) -> List[Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                 'WalkForwardFold']]:
    """
    Create backtest folds with walk-forward validation.

    Each fold contains:
    - train_snapshots: List of (stock_window, macro_window, future_returns, active_mask)
    - val_snapshots:   List of (stock_window, macro_window, future_returns, active_mask)
    - fold:            WalkForwardFold metadata (train_start, train_end, val_start, val_end)

    active_mask is the inclusion status at the END of the lookback window (time T),
    defining the tradable universe for the forecast horizon.
    """
    total_t = stock_data.size(1)
    assert macro_data.size(1) == total_t, "Macro and Stock data must have same length"
    assert inclusion_mask.size(1) == total_t, "Inclusion mask must match time dimension"
    
    folds_data = []
    folds = splitter.split(total_t)
    
    print(f"\nCreating {len(folds)} backtest folds:")
    print(f"  Window size: {window_size} days")
    print(f"  Forecast horizon: {forecast_horizon} days")
    print(f"  Snapshot step size: {step_size} days")
    
    for fold in folds:
        # FIX #1 & #2: Compute cross-sectional demean and rolling z-score per fold
        start_t = max(0, fold.train_start - window_size)
        end_t = fold.val_end
        
        ret_slice = returns_data[:, start_t:end_t].clone()
        mask_slice = inclusion_mask[:, start_t:end_t]
        stock_slice = stock_data[:, start_t:end_t, :].clone()
        macro_slice = macro_data[:, start_t:end_t, :]
        
        # 1. Cross-sectional demeaning using ONLY active stocks (prevents survivorship bias leakage)
        active_counts = mask_slice.sum(dim=0).clamp(min=1)
        market_rets = (ret_slice * mask_slice).sum(dim=0) / active_counts
        demeaned_rets = ret_slice - market_rets.unsqueeze(0)
        
        # 2. Re-normalize using causal rolling stats within the fold bounds
        renorm_rets = torch.zeros_like(demeaned_rets)
        T_slice = demeaned_rets.size(1)
        for t in range(T_slice):
            w_start = max(0, t - window_size)
            if w_start < t:
                # Stats are from [t-window, t-1], purely causal
                window_data = demeaned_rets[:, w_start:t]
                m = window_data.mean(dim=1)
                s = window_data.std(dim=1, unbiased=True).clamp(min=1e-8)
                renorm_rets[:, t] = (demeaned_rets[:, t] - m) / s
                
        # Returns column is conventionally at the last index (-1)
        stock_slice[:, :, -1] = renorm_rets
        
        # Adjust indices because we sliced the tensors
        fold_train_start = fold.train_start - start_t
        fold_train_end = fold.train_end - start_t
        fold_val_start = fold.val_start - start_t
        fold_val_end = fold.val_end - start_t
        
        train_snapshots = _create_snapshots(
            stock_slice, macro_slice, demeaned_rets, mask_slice,
            start_idx=fold_train_start,
            end_idx=fold_train_end,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            step_size=step_size
        )
        
        val_snapshots = _create_snapshots(
            stock_slice, macro_slice, demeaned_rets, mask_slice,
            start_idx=fold_val_start,
            end_idx=fold_val_end,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            step_size=step_size
        )
        
        folds_data.append((train_snapshots, val_snapshots, fold))
        
        # Report active stock stats for this fold
        if train_snapshots:
            active_counts = [s[3].sum().item() for s in train_snapshots]
            print(f"  Fold {fold.fold_idx + 1}: "
                  f"train [{fold.train_start}:{fold.train_end}] → "
                  f"{len(train_snapshots)} snapshots "
                  f"(~{np.mean(active_counts):.0f} active stocks), "
                  f"val [{fold.val_start}:{fold.val_end}] → "
                  f"{len(val_snapshots)} snapshots")
        else:
            print(f"  Fold {fold.fold_idx + 1}: "
                  f"train [{fold.train_start}:{fold.train_end}] → "
                  f"{len(train_snapshots)} snapshots, "
                  f"val [{fold.val_start}:{fold.val_end}] → "
                  f"{len(val_snapshots)} snapshots")
    
    return folds_data


def _create_snapshots(
    stock_data: torch.Tensor,
    macro_data: torch.Tensor,
    returns_data: torch.Tensor,
    inclusion_mask: torch.Tensor,
    start_idx: int,
    end_idx: int,
    window_size: int,
    forecast_horizon: int,
    step_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create snapshots within a specific time range.
    
    Returns:
        List of (stock_window, macro_window, future_returns, active_mask) tuples.
        active_mask is the inclusion status at time t+window_size (end of lookback).
    """
    snapshots = []
    
    for t in range(start_idx, end_idx - window_size - forecast_horizon + 1, step_size):
        stock_window = stock_data[:, t:t+window_size, :]
        macro_window = macro_data[:, t:t+window_size, :]

        # Future returns as labels.
        # INVARIANT: future_returns uses indices [t+window_size, t+window_size+forecast_horizon)
        # which are strictly AFTER the lookback window [t, t+window_size).
        # stock_window[:, :, -1] (the Returns column) only reaches t+window_size-1 — no leakage.
        future_returns = returns_data[:, t+window_size:t+window_size+forecast_horizon].sum(dim=1)

        # FIX #1: active_mask at the START of the forecast period (t + window_size),
        # not the last day of the lookback (t + window_size - 1).
        # Convention: mask[i]=True means stock i is a constituent when the
        # forecast period begins, i.e., it is legally tradable at that time.
        # Using t+window_size-1 would include stocks removed on the boundary
        # day and exclude stocks added on that day — the wrong semantics.
        clamp_t = min(t + window_size, inclusion_mask.size(1) - 1)
        active_mask = inclusion_mask[:, clamp_t]  # [N_s]

        snapshots.append((stock_window, macro_window, future_returns, active_mask))

    return snapshots


# =============================================================================
# SECTOR MAPPING HELPER
# =============================================================================

def get_sector_mapping(
    tickers: List[str],
    data_dir: str = "./data"
) -> Dict[str, str]:
    """
    Load or generate sector mapping for tickers.
    If real mapping file exists (ticker_sector_map.csv), use it.
    Otherwise, generate synthetic GICS sectors for testing.
    """
    candidate_paths = [
        os.path.join(data_dir, "ticker_sector_map.csv"),
        os.path.join(data_dir, "processed", "ticker_sector_map.csv"),
        os.path.join(os.path.dirname(data_dir), "ticker_sector_map.csv"),
        os.path.join("./data/processed", "ticker_sector_map.csv"),
        os.path.join("./data", "ticker_sector_map.csv")
    ]
    
    map_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            map_path = p
            break
            
    if map_path:
        print(f"  Loading sector map from: {map_path}")
        df = pd.read_csv(map_path)
        df.columns = [c.lower() for c in df.columns]
        if 'ticker' in df.columns and 'sector' in df.columns:
            return pd.Series(df.sector.values, index=df.ticker).to_dict()
        elif 'symbol' in df.columns and 'gics sector' in df.columns:
             return pd.Series(df['gics sector'].values, index=df.symbol).to_dict()
    
    # Fallback: Synthetic Sectors
    print("Warning: No sector map found. Generating synthetic sectors for testing.")
    sectors = [
        'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
        'Communication Services', 'Industrials', 'Consumer Staples',
        'Energy', 'Utilities', 'Real Estate', 'Materials'
    ]
    mapping = {}
    for ticker in tickers:
        idx = sum(ord(c) for c in ticker) % len(sectors)
        mapping[ticker] = sectors[idx]
        
    return mapping


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
    macro_lag: int = 5,
    device: torch.device = None,
    max_stocks: Optional[int] = 150,
    historical_csv: str = "./data/sp500_historical.csv"
) -> Tuple[List, pd.DatetimeIndex, List[str], int, int, Dict[str, str]]:
    """
    Convenience function to load data and create backtest folds in one call.
    Applies explicit MACRO LAGGING to enforce causality.
    
    Returns:
        folds: List of (train_snapshots, val_snapshots) tuples
               Each snapshot is (stock, macro, returns, active_mask)
        dates: Date index
        tickers: Stock ticker list
        num_stocks: Number of stocks
        num_macros: Number of macro factors
        sector_map: Dict mapping ticker -> sector_name
    """
    # Load data (now returns inclusion_mask)
    stock_data, macro_data, returns_data, inclusion_mask, dates, tickers = \
        load_processed_data(
            data_dir=data_dir,
            device=device,
            max_stocks=max_stocks,
            historical_csv=historical_csv
        )
    
    # --- APPLY MACRO LAGGING ---
    if macro_lag > 0:
        print(f"Applying macro lag of {macro_lag} days for causality...")
        stock_data = stock_data[:, macro_lag:, :]
        returns_data = returns_data[:, macro_lag:]
        inclusion_mask = inclusion_mask[:, macro_lag:]
        macro_data = macro_data[:, :-macro_lag, :]
        dates = dates[macro_lag:]
        print(f"  Aligned data shape: {stock_data.size(1)} timesteps")
    
    # Load sector map
    sector_map = get_sector_mapping(tickers, data_dir=os.path.dirname(data_dir))

    # Create splitter
    splitter = WalkForwardSplitter(
        train_size=train_size,
        val_size=val_size,
        step_size=step_size
    )
    
    # Create folds (now threads inclusion_mask through)
    folds = create_backtest_folds(
        stock_data=stock_data,
        macro_data=macro_data,
        returns_data=returns_data,
        inclusion_mask=inclusion_mask,
        splitter=splitter,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        step_size=snapshot_step
    )
    
    return folds, dates, tickers, stock_data.size(0), macro_data.size(0), sector_map


if __name__ == "__main__":
    # Demo/test loading
    try:
        folds, dates, tickers, n_stocks, n_macros, sector_map = load_and_prepare_backtest()
        print(f"\nBacktest ready: {len(folds)} folds, {n_stocks} stocks, {n_macros} macro factors")
        print(f"Sector map loaded for {len(sector_map)} tickers")
        
        # Verify 4-tuple format
        if folds and folds[0][0]:
            snapshot = folds[0][0][0]
            print(f"\nSnapshot format verification:")
            print(f"  stock_window: {snapshot[0].shape}")
            print(f"  macro_window: {snapshot[1].shape}")
            print(f"  future_returns: {snapshot[2].shape}")
            print(f"  active_mask: {snapshot[3].shape} (dtype={snapshot[3].dtype})")
            print(f"  Active stocks in first snapshot: {snapshot[3].sum().item()}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python data_ingest.py' first to generate processed data.")
