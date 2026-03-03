"""
Live Data Pipeline for DGRCL Paper Trading

Fetches and featurizes the most recent market data from yfinance and assembles
the snapshot tensors that the model expects.  Reuses the exact same feature
engineering functions from data_ingest.py to guarantee identical preprocessing.

Key design:
- Downloads the last (lookback_days + warmup_days) trading days per ticker.
  The warmup window (default 120 days) is needed to compute RSI and MACD and
  to warm up the rolling z-score normalization (60-day window).
- Aligns the stock universe to the order stored in the checkpoint's `tickers`
  list.  Stocks that no longer trade or fail to download are given a zero
  feature vector and marked inactive.
- Builds tensors shaped exactly like the backtest feed:
    stock_tensor:   [N_stocks, T, S] — S = number of selected features
    macro_tensor:   [N_macros, T, 4]
    active_mask:    [N_stocks, T] bool

Usage:
    from live_data import LiveDataFeed
    feed = LiveDataFeed(tickers=bundle.tickers, feature_indices=bundle.feature_indices)
    stock_t, macro_t, active_t = feed.get_snapshot(lookback=60)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Reuse feature engineering from data_ingest
from data_ingest import (
    calculate_rsi,
    calculate_macd,
    calculate_volatility,
    rolling_zscore_normalize,
    MACRO_TICKERS,
)

logger = logging.getLogger(__name__)

# Number of extra calendar days to request beyond the lookback window to
# account for weekends, holidays, and yfinance padding.
_CALENDAR_BUFFER = 2.0  # multiplier: request 2× the target trading days

ALL_FEATURE_NAMES = [
    "Close", "High", "Low", "Log_Vol", "RSI_14", "MACD", "Volatility_5", "Returns"
]

ALL_MACRO_FEATURE_NAMES = ["Close", "Returns", "MA_50", "MA_200"]


# =============================================================================
# SINGLE-TICKER FEATURE ENGINEERING
# =============================================================================

def _compute_stock_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute the 8 stock features on a raw yfinance OHLCV DataFrame.
    Returns None if there is insufficient data.
    """
    if df is None or len(df) < 30:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.copy()
        df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Log_Vol"] = np.log(df["Volume"].clip(lower=1))
        df["RSI_14"] = calculate_rsi(df["Close"])
        macd, _ = calculate_macd(df["Close"])
        df["MACD"] = macd
        df["Volatility_5"] = calculate_volatility(df["Close"])
        cols = ["Close", "High", "Low", "Log_Vol", "RSI_14", "MACD", "Volatility_5", "Returns"]
        df = df[cols].dropna()
        if len(df) < 10:
            return None
        df = rolling_zscore_normalize(df, window=60)
        df = df.dropna()
        return df
    except Exception as e:
        logger.debug("Feature error: %s", e)
        return None


def _compute_macro_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute 4 macro features from a raw yfinance DataFrame."""
    if df is None or len(df) < 30:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.copy()
        df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["MA_50"] = df["Close"].rolling(window=50).mean()
        df["MA_200"] = df["Close"].rolling(window=200).mean()
        df = df[["Close", "Returns", "MA_50", "MA_200"]].dropna()
        if len(df) < 10:
            return None
        df = rolling_zscore_normalize(df, window=60)
        df = df.dropna()
        return df
    except Exception as e:
        logger.debug("Macro feature error: %s", e)
        return None


# =============================================================================
# LIVE DATA FEED
# =============================================================================

class LiveDataFeed:
    """
    Builds live snapshot tensors aligned to a fixed stock universe.

    Args:
        tickers:         Ordered list matching the checkpoint's training universe.
        feature_indices: Which of the 8 stock features to use (from checkpoint).
        warmup_days:     Extra trading days to download for indicator warmup.
                         MACD needs 26 days, RSI needs 14; 120 is conservative.
        device:          Torch device for returned tensors.
    """

    MACRO_NAMES = list(MACRO_TICKERS.keys())   # ["Energy", "Rates", "Currency", "Fear"]
    MACRO_TICKERS_MAP = MACRO_TICKERS

    def __init__(
        self,
        tickers: List[str],
        feature_indices: Optional[List[int]] = None,
        warmup_days: int = 120,
        device: Optional[torch.device] = None,
    ):
        self.tickers = tickers
        self.feature_indices = feature_indices if feature_indices is not None else list(range(8))
        self.warmup_days = warmup_days
        self.device = device or torch.device("cpu")
        self._selected_features = [ALL_FEATURE_NAMES[i] for i in self.feature_indices]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_snapshot(
        self,
        lookback: int = 60,
        as_of: Optional[datetime] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Download data and build tensors for the most recent `lookback` trading days.

        Args:
            lookback:  Number of trading days to include in the returned tensors
                       (= WINDOW_SIZE in the backtest, default 60).
            as_of:     Reference date (default: today).  The download covers
                       [as_of - warmup_days - lookback, as_of].

        Returns:
            stock_tensor:  [N_stocks, lookback, S]  (S = len(feature_indices))
            macro_tensor:  [N_macros, lookback, 4]
            active_mask:   [N_stocks, lookback] bool
        """
        as_of = as_of or datetime.utcnow()
        total_days = lookback + self.warmup_days
        start_dt = as_of - timedelta(days=int(total_days * _CALENDAR_BUFFER))
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = (as_of + timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info("Fetching live data: %d tickers + 4 macros (%s → %s)",
                    len(self.tickers), start_str, end_str)

        stock_dfs = self._fetch_stocks(start_str, end_str)
        macro_dfs = self._fetch_macros(start_str, end_str)

        return self._build_tensors(stock_dfs, macro_dfs, lookback)

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _fetch_stocks(self, start: str, end: str) -> Dict[str, Optional[pd.DataFrame]]:
        """Download and featurize all stock tickers in batch."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install via: pip install yfinance")

        result: Dict[str, Optional[pd.DataFrame]] = {}
        failed = 0
        for ticker in self.tickers:
            try:
                raw = yf.download(ticker, start=start, end=end,
                                  progress=False, auto_adjust=True)
                result[ticker] = _compute_stock_features(raw)
                if result[ticker] is None:
                    failed += 1
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", ticker, e)
                result[ticker] = None
                failed += 1

        if failed:
            logger.warning("%d/%d tickers failed to download or had insufficient data",
                           failed, len(self.tickers))
        return result

    def _fetch_macros(self, start: str, end: str) -> Dict[str, Optional[pd.DataFrame]]:
        """Download and featurize macro indicators."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install via: pip install yfinance")

        result: Dict[str, Optional[pd.DataFrame]] = {}
        for name, yticker in self.MACRO_TICKERS_MAP.items():
            try:
                raw = yf.download(yticker, start=start, end=end,
                                  progress=False, auto_adjust=True)
                result[name] = _compute_macro_features(raw)
            except Exception as e:
                logger.debug("Failed to fetch macro %s (%s): %s", name, yticker, e)
                result[name] = None
        return result

    # ------------------------------------------------------------------
    # Tensor assembly
    # ------------------------------------------------------------------

    def _build_tensors(
        self,
        stock_dfs: Dict[str, Optional[pd.DataFrame]],
        macro_dfs: Dict[str, Optional[pd.DataFrame]],
        lookback: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Align all DataFrames to a common date index and build tensors.

        Stocks that failed to download receive a zero feature vector and are
        marked inactive (active_mask = False) for all timesteps.
        """
        N_s = len(self.tickers)
        S = len(self.feature_indices)
        N_m = len(self.MACRO_NAMES)

        # Build common date index from valid data
        valid_dates = self._common_index(stock_dfs, macro_dfs)
        if len(valid_dates) < lookback:
            raise ValueError(
                f"Only {len(valid_dates)} valid trading days available "
                f"(need {lookback}). Try increasing warmup_days or check your "
                f"yfinance connection."
            )
        # Keep the most recent `lookback` dates
        date_window = valid_dates[-lookback:]
        T = len(date_window)

        stock_tensor = torch.zeros(N_s, T, S, dtype=torch.float32)
        active_mask = torch.zeros(N_s, T, dtype=torch.bool)

        for si, ticker in enumerate(self.tickers):
            df = stock_dfs.get(ticker)
            if df is None or df.empty:
                continue
            # Reindex to the common date window; fill missing with 0
            df_aligned = df[self._selected_features].reindex(date_window, fill_value=0.0)
            valid_rows = df_aligned.abs().sum(axis=1) > 0
            arr = df_aligned.values.astype(np.float32)
            stock_tensor[si] = torch.from_numpy(arr)
            active_mask[si] = torch.from_numpy(valid_rows.values)

        macro_tensor = torch.zeros(N_m, T, 4, dtype=torch.float32)
        for mi, name in enumerate(self.MACRO_NAMES):
            df = macro_dfs.get(name)
            if df is None or df.empty:
                continue
            df_aligned = df[ALL_MACRO_FEATURE_NAMES].reindex(date_window, fill_value=0.0)
            macro_tensor[mi] = torch.from_numpy(df_aligned.values.astype(np.float32))

        return (
            stock_tensor.to(self.device),
            macro_tensor.to(self.device),
            active_mask.to(self.device),
        )

    @staticmethod
    def _common_index(
        stock_dfs: Dict[str, Optional[pd.DataFrame]],
        macro_dfs: Dict[str, Optional[pd.DataFrame]],
    ) -> pd.DatetimeIndex:
        """
        Build the intersection of all available date indices.
        Falls back to the union of stock dates if macros are sparse.
        """
        all_indices = []
        for df in stock_dfs.values():
            if df is not None and not df.empty:
                all_indices.append(df.index)
        for df in macro_dfs.values():
            if df is not None and not df.empty:
                all_indices.append(df.index)

        if not all_indices:
            raise ValueError("No valid data downloaded for any ticker.")

        # Use the intersection so every timestep has at least some data
        common = all_indices[0]
        for idx in all_indices[1:]:
            common = common.intersection(idx)
        return common.sort_values()
