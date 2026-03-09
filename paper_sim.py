"""
DGRCL Historical Paper Trading Simulator
=========================================
Replays the DGRCL strategy from a given start date to present using simulated
paper money, producing a dollar-denominated equity curve, trade log, and
comprehensive performance report.

Risk management layers (all configurable via CLI):
  Existing (from portfolio_optimizer.py / alpaca_broker.py):
    - Dollar-neutral constraint (sum(w) = 0)
    - Gross leverage cap (||w||_1 <= max_leverage)
    - Per-stock position limit (|w_i| <= 5%)
    - Conformal abstention gate (skip ambiguous predictions)

  New (added in this simulator):
    - Max-drawdown circuit breaker: go flat for cooldown_days rebalance
      periods when equity drops >= max_dd_pct from peak
    - Per-rebalance loss guard: skip rebalance when equity fell more than
      rebalance_loss_pct since the last executed rebalance
    - Regime-conditioned leverage: reduce MVO gross leverage cap in crisis
      regimes (high cross-sectional vol periods)

Usage:
    # Requires a pre-trained checkpoint (run train.py --real-data --save-checkpoint first):
    python paper_sim.py --start-date 2022-01-01 --capital 10000

    # Train checkpoint on pre-simulation data first, then simulate:
    python paper_sim.py --start-date 2022-01-01 --capital 10000 --train-first

    # Custom risk parameters:
    python paper_sim.py --start-date 2022-01-01 --capital 10000 \\
        --max-drawdown-pct 0.15 --cooldown-days 10 --crisis-leverage 1.0
"""

import argparse
import json
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*")

logger = logging.getLogger(__name__)


# =============================================================================
# RISK MANAGER
# =============================================================================

class RiskManager:
    """
    Portfolio-level risk controls that sit on top of the existing per-stock
    constraints (MVO leverage/position limits, conformal gate).

    Three new mechanisms not present in the existing backtest code:

    1. Max-drawdown circuit breaker
       If the running equity falls >= max_dd_pct below its all-time peak,
       the strategy goes flat (zero weights) for cooldown_days rebalance
       periods before resuming.  This limits catastrophic tail losses.

    2. Per-rebalance loss guard
       If equity dropped >= rebalance_loss_pct since the LAST EXECUTED
       rebalance, skip this rebalance and hold current positions.  This
       prevents "doubling down" into an ongoing adverse move.

    3. Regime-conditioned leverage
       The MVO gross leverage cap is reduced in crisis regimes (high
       cross-sectional volatility periods).  During calm or normal regimes
       the full leverage cap is used; in crisis regimes a lower cap applies.
    """

    def __init__(
        self,
        max_dd_pct: float = 0.20,
        cooldown_days: int = 2,
        rebalance_loss_pct: float = 0.08,
        normal_leverage: float = 2.0,
        crisis_leverage: float = 1.0,
        calm_leverage: float = 2.0,
    ):
        """
        Args:
            max_dd_pct:          Peak-to-trough drawdown threshold for circuit
                                 breaker (default 20%).
            cooldown_days:       Rebalance periods to stay flat after the
                                 breaker trips (default 15 ≈ 75 trading days).
            rebalance_loss_pct:  Skip rebalance when equity fell more than this
                                 fraction since the last executed rebalance
                                 (default 8%).
            normal_leverage:     MVO gross leverage cap in normal regime (2.0).
            crisis_leverage:     MVO gross leverage cap in crisis regime (1.0).
            calm_leverage:       MVO gross leverage cap in calm regime (2.0).
        """
        self.max_dd_pct = max_dd_pct
        self.cooldown_days = cooldown_days
        self.rebalance_loss_pct = rebalance_loss_pct
        self.normal_leverage = normal_leverage
        self.crisis_leverage = crisis_leverage
        self.calm_leverage = calm_leverage

        self._peak_equity: float = 0.0
        self._cooldown_remaining: int = 0
        self._last_rebalance_equity: float = 0.0
        self._breaker_trips: int = 0

    def reset(self, initial_equity: float):
        """Initialize state at simulation start."""
        self._peak_equity = initial_equity
        self._cooldown_remaining = 0
        self._last_rebalance_equity = initial_equity
        self._breaker_trips = 0

    def update_peak(self, equity: float):
        """Call every day to maintain the running equity peak."""
        if equity > self._peak_equity:
            self._peak_equity = equity

    def current_drawdown(self, equity: float) -> float:
        """Fraction of peak equity that has been lost (0 = at peak)."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - equity) / self._peak_equity)

    def check_rebalance(
        self,
        equity: float,
        regime: str = "normal",
    ) -> Tuple[bool, float, Dict]:
        """
        Determine whether to execute a rebalance and with what leverage.

        Args:
            equity:  Current portfolio equity (USD).
            regime:  Market regime: "calm" | "normal" | "crisis".

        Returns:
            (ok_to_trade, effective_leverage, diagnostics_dict)
            ok_to_trade       — True if the strategy should rebalance.
            effective_leverage — MVO gross leverage cap to apply.
            diagnostics_dict  — Debug information logged per rebalance.
        """
        diag: Dict = {
            "peak_equity": round(self._peak_equity, 2),
            "current_dd": round(self.current_drawdown(equity), 4),
            "cooldown_remaining": self._cooldown_remaining,
            "breaker_trips": self._breaker_trips,
            "regime": regime,
            "skip_reason": None,
        }

        # --- Check 1: Active cooldown from a previous breaker trip ---
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            diag["skip_reason"] = (
                f"circuit_breaker_cooldown "
                f"({self._cooldown_remaining + 1} periods left)"
            )
            diag["cooldown_remaining"] = self._cooldown_remaining
            return False, 0.0, diag

        # --- Check 2: Circuit breaker (max drawdown exceeded) ---
        dd = self.current_drawdown(equity)
        if dd >= self.max_dd_pct:
            self._cooldown_remaining = self.cooldown_days - 1
            self._breaker_trips += 1
            diag["skip_reason"] = (
                f"circuit_breaker_tripped "
                f"(dd={dd:.1%} >= {self.max_dd_pct:.1%})"
            )
            diag["cooldown_remaining"] = self._cooldown_remaining
            return False, 0.0, diag

        # --- Check 3: Per-rebalance loss guard ---
        if self._last_rebalance_equity > 0:
            loss_since_last = (
                (self._last_rebalance_equity - equity)
                / self._last_rebalance_equity
            )
            if loss_since_last >= self.rebalance_loss_pct:
                diag["skip_reason"] = (
                    f"rebalance_loss_guard "
                    f"({loss_since_last:.1%} >= {self.rebalance_loss_pct:.1%})"
                )
                return False, 0.0, diag

        # --- Regime-conditioned leverage ---
        lev_by_regime = {
            "crisis": self.crisis_leverage,
            "calm":   self.calm_leverage,
            "normal": self.normal_leverage,
        }
        effective_lev = lev_by_regime.get(regime, self.normal_leverage)

        # Update last-rebalance equity ONLY when we actually trade
        self._last_rebalance_equity = equity
        return True, effective_lev, diag


# =============================================================================
# HISTORICAL DATA CACHE
# =============================================================================

class HistoricalDataCache:
    """
    Downloads and caches the complete stock + macro history for the simulation
    window in one batch, then provides efficient per-date slicing.

    Design goals:
      - Feature engineering identical to live_data.py (same rolling z-score,
        same RSI/MACD/volatility functions) so the model sees consistent inputs.
      - Raw close prices stored separately for dollar P&L computation.
      - Batch yfinance download (group_by="ticker") for speed; individual
        fallback per-ticker on errors.
    """

    MACRO_TICKERS_MAP: Dict[str, str] = {
        "Energy":   "CL=F",
        "Rates":    "^TNX",
        "Currency": "DX-Y.NYB",
        "Fear":     "^VIX",
    }
    ALL_FEATURE_NAMES = [
        "Close", "High", "Low", "Log_Vol", "RSI_14", "MACD", "Volatility_5", "Returns"
    ]
    ALL_MACRO_FEATURE_NAMES = ["Close", "Returns", "MA_50", "MA_200"]

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        feature_indices: Optional[List[int]] = None,
        warmup_days: int = 300,
    ):
        self.tickers = tickers
        self.feature_indices = feature_indices if feature_indices is not None else list(range(8))
        self._selected_features = [self.ALL_FEATURE_NAMES[i] for i in self.feature_indices]

        sim_start = pd.Timestamp(start_date)
        sim_end   = pd.Timestamp(end_date)
        # Request 2× warmup in calendar days to cover weekends/holidays
        dl_start  = sim_start - pd.Timedelta(days=warmup_days * 2)

        self._sim_start = sim_start
        self._sim_end   = sim_end

        print(
            f"  Fetching historical data: {dl_start.date()} → {sim_end.date()} "
            f"({len(tickers)} stocks + 4 macro factors)"
        )

        self._stock_feat: Dict[str, Optional[pd.DataFrame]] = {}
        self._stock_close: Dict[str, Optional[pd.Series]]   = {}
        self._macro_feat: Dict[str, Optional[pd.DataFrame]] = {}

        self._download_all(
            dl_start.strftime("%Y-%m-%d"),
            (sim_end + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
        )

        self._all_days, self._trading_days = self._build_trading_days()
        print(
            f"  Cache ready: {len(self._trading_days)} sim days "
            f"({self._trading_days[0].date()} → {self._trading_days[-1].date()}), "
            f"{len(self._all_days)} total (incl. warmup)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model_input(
        self,
        as_of_date: pd.Timestamp,
        lookback: int = 60,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return model-ready tensors for the window ending on as_of_date.

        Returns:
            stock_tensor:  [N_stocks, lookback, S]
            macro_tensor:  [N_macros, lookback, 4]
            active_mask:   [N_stocks] bool (tradable on the last day)
        """
        avail = [d for d in self._all_days if d <= as_of_date]
        if len(avail) < lookback:
            raise ValueError(
                f"Insufficient history before {as_of_date.date()} "
                f"(have {len(avail)}, need {lookback}). "
                "Increase warmup_days or start the simulation later."
            )

        window = pd.DatetimeIndex(avail[-lookback:])
        T  = len(window)
        S  = len(self.feature_indices)
        Ns = len(self.tickers)
        Nm = len(self.MACRO_TICKERS_MAP)

        stock_t = torch.zeros(Ns, T, S, dtype=torch.float32)
        active  = torch.zeros(Ns, dtype=torch.bool)

        for si, ticker in enumerate(self.tickers):
            df = self._stock_feat.get(ticker)
            if df is None or df.empty:
                continue
            aligned    = df[self._selected_features].reindex(window, fill_value=0.0)
            valid_rows = aligned.abs().sum(axis=1) > 0
            stock_t[si] = torch.from_numpy(aligned.values.astype(np.float32))
            active[si]  = bool(valid_rows.iloc[-1])

        macro_t = torch.zeros(Nm, T, 4, dtype=torch.float32)
        for mi, name in enumerate(self.MACRO_TICKERS_MAP):
            df = self._macro_feat.get(name)
            if df is None or df.empty:
                continue
            aligned    = df[self.ALL_MACRO_FEATURE_NAMES].reindex(window, fill_value=0.0)
            macro_t[mi] = torch.from_numpy(aligned.values.astype(np.float32))

        return stock_t, macro_t, active

    def get_period_returns(
        self,
        from_date: pd.Timestamp,
        to_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        Simple (arithmetic) returns for each ticker over the period.
        Used for daily mark-to-market P&L.
        """
        result: Dict[str, float] = {}
        for ticker in self.tickers:
            close = self._stock_close.get(ticker)
            if close is None or close.empty:
                result[ticker] = 0.0
                continue
            avail_from = close.index[close.index >= from_date]
            avail_to   = close.index[close.index <= to_date]
            if len(avail_from) == 0 or len(avail_to) == 0:
                result[ticker] = 0.0
                continue
            p0 = float(close.loc[avail_from[0]])
            p1 = float(close.loc[avail_to[-1]])
            result[ticker] = (p1 / p0 - 1.0) if p0 > 0 else 0.0
        return result

    def get_returns_matrix(
        self,
        as_of_date: pd.Timestamp,
        n_days: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Daily log-returns matrix for covariance estimation.

        Returns:
            ret_matrix: [N_stocks, T_cov] float32
            active_np:  [N_stocks] bool
        """
        avail  = [d for d in self._all_days if d <= as_of_date]
        window = avail[-n_days:] if len(avail) >= n_days else avail
        dt_win = pd.DatetimeIndex(window)

        Ns    = len(self.tickers)
        T_cov = max(len(window) - 1, 1)
        ret_matrix = np.zeros((Ns, T_cov), dtype=np.float32)
        active_np  = np.zeros(Ns, dtype=bool)

        for si, ticker in enumerate(self.tickers):
            close = self._stock_close.get(ticker)
            if close is None or close.empty:
                continue
            aligned = close.reindex(dt_win).ffill()
            log_ret = np.log(aligned / aligned.shift(1)).dropna()
            if len(log_ret) > 0:
                n = min(len(log_ret), T_cov)
                ret_matrix[si, -n:] = log_ret.values[-n:].astype(np.float32)
                active_np[si] = True

        return ret_matrix, active_np

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        return self._trading_days

    # ------------------------------------------------------------------
    # Download & feature engineering
    # ------------------------------------------------------------------

    def _download_all(self, start: str, end: str):
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required: pip install yfinance")

        from data_ingest import (
            calculate_rsi, calculate_macd, calculate_volatility,
            rolling_zscore_normalize,
        )

        def _featurize_stock(df_raw) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
            if df_raw is None or len(df_raw) < 30:
                return None, None
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw = df_raw.copy()
                df_raw.columns = df_raw.columns.get_level_values(0)
            df = df_raw.copy()
            close_series = df["Close"].dropna().copy()
            df["Returns"]     = np.log(df["Close"] / df["Close"].shift(1))
            df["Log_Vol"]     = np.log(df["Volume"].clip(lower=1))
            df["RSI_14"]      = calculate_rsi(df["Close"])
            macd, _           = calculate_macd(df["Close"])
            df["MACD"]        = macd
            df["Volatility_5"] = calculate_volatility(df["Close"])
            cols = ["Close", "High", "Low", "Log_Vol", "RSI_14", "MACD",
                    "Volatility_5", "Returns"]
            df = df[cols].dropna()
            if len(df) < 10:
                return None, None
            df = rolling_zscore_normalize(df, window=60)
            return df.dropna(), close_series

        # --- Batch download (faster for large universes) ---
        print("  Batch downloading stocks ...", end="", flush=True)
        batch = None
        try:
            batch = yf.download(
                self.tickers, start=start, end=end,
                progress=False, auto_adjust=True,
                group_by="ticker", threads=True,
            )
        except Exception as e:
            logger.debug("Batch download failed (%s), will use individual downloads.", e)
        print(" done")

        failed = 0
        for ticker in self.tickers:
            try:
                raw = None
                if batch is not None:
                    if isinstance(batch.columns, pd.MultiIndex):
                        lvl0 = batch.columns.get_level_values(0)
                        if ticker in lvl0:
                            raw = batch[ticker].copy()
                    elif ticker in batch.columns:
                        raw = batch[[ticker]].rename(columns={ticker: "Close"})

                if raw is None or (hasattr(raw, "empty") and raw.empty):
                    raw = yf.download(
                        ticker, start=start, end=end,
                        progress=False, auto_adjust=True,
                    )

                feat_df, close_s = _featurize_stock(raw)
                self._stock_feat[ticker]  = feat_df
                self._stock_close[ticker] = close_s
                if feat_df is None:
                    failed += 1
            except Exception as e:
                logger.debug("Failed %s: %s", ticker, e)
                self._stock_feat[ticker]  = None
                self._stock_close[ticker] = None
                failed += 1

        ok = len(self.tickers) - failed
        print(f"  Stocks: {ok}/{len(self.tickers)} OK ({failed} failed/skipped)")

        # --- Macro indicators (individual downloads) ---
        for name, yticker in self.MACRO_TICKERS_MAP.items():
            try:
                raw = yf.download(yticker, start=start, end=end,
                                  progress=False, auto_adjust=True)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                if raw is None or len(raw) < 30:
                    self._macro_feat[name] = None
                    continue
                df = raw.copy()
                df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
                df["MA_50"]   = df["Close"].rolling(50).mean()
                df["MA_200"]  = df["Close"].rolling(200).mean()
                df = df[["Close", "Returns", "MA_50", "MA_200"]].dropna()
                if len(df) >= 10:
                    from data_ingest import rolling_zscore_normalize
                    df = rolling_zscore_normalize(df, window=60)
                    self._macro_feat[name] = df.dropna()
                else:
                    self._macro_feat[name] = None
            except Exception as e:
                logger.debug("Macro %s (%s) failed: %s", name, yticker, e)
                self._macro_feat[name] = None

    def _build_trading_days(self) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Union of all available stock date indices.

        Returns two calendars:
          - all_days:  full range (including warmup before sim_start) — for lookback
          - sim_days:  clipped to [sim_start, sim_end] — for the simulation loop

        We use union (not intersection) because each ticker may have gaps due to
        listing dates, halts, or data availability.  get_model_input() handles
        missing tickers gracefully via reindex(fill_value=0.0) and active_mask.
        """
        stock_indices = []
        for df in self._stock_feat.values():
            if df is not None and not df.empty:
                stock_indices.append(df.index)

        if not stock_indices:
            raise ValueError("No stock data was downloaded. Check your tickers and date range.")

        all_days = stock_indices[0]
        for idx in stock_indices[1:]:
            all_days = all_days.union(idx)

        all_days = all_days[all_days <= self._sim_end].sort_values()
        sim_days = all_days[(all_days >= self._sim_start) & (all_days <= self._sim_end)]
        return all_days, sim_days


# =============================================================================
# SIMULATION DATA STRUCTURES
# =============================================================================

@dataclass
class RebalanceRecord:
    """One rebalance decision and its outcome."""
    date: pd.Timestamp
    equity_before: float
    equity_after: float           # after deducting transaction cost
    weights: Dict[str, float]     # {ticker: signed weight}
    n_active: int
    n_trade: int
    gated_pct: float
    gross_leverage: float
    turnover: float
    transaction_cost: float
    regime: str
    regime_vol: float
    rank_stability: float
    risk_action: str              # "trade" | skip reason string
    drawdown_at_rebalance: float
    effective_leverage: float
    net_exposure: float = 0.0
    top_longs: List[str]  = field(default_factory=list)
    top_shorts: List[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Complete output of one simulation run."""
    equity_curve: pd.Series                # date → equity ($)
    daily_pnl: pd.Series                   # date → daily dollar P&L
    rebalance_log: List[RebalanceRecord]
    spy_curve: Optional[pd.Series]         # SPY buy-and-hold benchmark
    initial_capital: float
    start_date: str
    end_date: str
    config: Dict

    # ---- Performance metrics ----

    def _daily_returns(self) -> pd.Series:
        return self.equity_curve.pct_change().fillna(0.0)

    def total_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0

    def cagr(self) -> float:
        years = len(self.equity_curve) / 252.0
        if years <= 0:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (1.0 / years) - 1.0

    def sharpe(self, rf: float = 0.05) -> float:
        """Annualized Sharpe ratio (risk-free rate 5% p.a., daily compounded)."""
        dr = self._daily_returns()
        rf_d = (1 + rf) ** (1 / 252) - 1
        exc  = dr - rf_d
        if exc.std() < 1e-10:
            return 0.0
        return (exc.mean() / exc.std()) * math.sqrt(252)

    def sortino(self, rf: float = 0.05) -> float:
        dr  = self._daily_returns()
        rf_d = (1 + rf) ** (1 / 252) - 1
        exc  = dr - rf_d
        neg  = exc[exc < 0]
        if len(neg) < 2 or neg.std() < 1e-10:
            return 0.0
        return (exc.mean() / neg.std()) * math.sqrt(252)

    def max_drawdown(self) -> float:
        eq  = self.equity_curve
        peak = eq.cummax()
        dd   = (peak - eq) / peak
        return float(dd.max())

    def calmar(self) -> float:
        mdd = self.max_drawdown()
        return self.cagr() / mdd if mdd > 1e-10 else float("inf")

    def drawdown_series(self) -> pd.Series:
        eq   = self.equity_curve
        peak = eq.cummax()
        return (peak - eq) / peak

    def rolling_sharpe(self, window: int = 90, rf: float = 0.05) -> pd.Series:
        dr   = self._daily_returns()
        rf_d = (1 + rf) ** (1 / 252) - 1
        exc  = dr - rf_d
        mu   = exc.rolling(window).mean()
        sig  = exc.rolling(window).std().clip(lower=1e-10)
        return (mu / sig) * math.sqrt(252)

    def monthly_returns_grid(self) -> pd.DataFrame:
        """Year × Month (1-12) grid of cumulative monthly returns."""
        dr      = self._daily_returns()
        monthly = dr.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly.index = monthly.index.to_period("M")
        df = pd.DataFrame({
            "year":  monthly.index.year,
            "month": monthly.index.month,
            "ret":   monthly.values,
        })
        return df.pivot(index="year", columns="month", values="ret")

    def metrics_dict(self) -> Dict:
        trades = [r for r in self.rebalance_log if r.risk_action == "trade"]
        return {
            "initial_capital":        f"${self.initial_capital:,.2f}",
            "final_equity":           f"${self.equity_curve.iloc[-1]:,.2f}",
            "total_return":           f"{self.total_return():.2%}",
            "cagr":                   f"{self.cagr():.2%}",
            "sharpe_rf5pct":          f"{self.sharpe():.3f}",
            "sortino_rf5pct":         f"{self.sortino():.3f}",
            "max_drawdown":           f"{self.max_drawdown():.2%}",
            "calmar":                 f"{self.calmar():.3f}",
            "n_rebalances_total":     str(len(self.rebalance_log)),
            "n_rebalances_traded":    str(len(trades)),
            "circuit_breaker_trips":  str(sum(
                1 for r in self.rebalance_log if "circuit_breaker_tripped" in r.risk_action
            )),
            "cooldown_skips":         str(sum(
                1 for r in self.rebalance_log if "cooldown" in r.risk_action
            )),
            "avg_gross_leverage":     (
                f"{np.mean([r.gross_leverage for r in trades]):.3f}"
                if trades else "N/A"
            ),
            "avg_gated_pct":          (
                f"{np.mean([r.gated_pct for r in trades]):.1%}"
                if trades else "N/A"
            ),
            "avg_net_exposure":       (
                f"{np.mean([r.net_exposure for r in trades]):.3f}"
                if trades else "N/A"
            ),
            "low_confidence_skips":   str(sum(
                1 for r in self.rebalance_log if "low_confidence" in r.risk_action
            )),
            "low_turnover_skips":     str(sum(
                1 for r in self.rebalance_log if "low_turnover" in r.risk_action
            )),
            "total_transaction_costs": f"${sum(r.transaction_cost for r in self.rebalance_log):,.2f}",
        }


# =============================================================================
# PAPER SIMULATOR
# =============================================================================

class PaperSimulator:
    """
    Orchestrates the historical paper-trading simulation.

    Simulation loop (every FORECAST_HORIZON trading days):
        1. Build 60-day lookback window from HistoricalDataCache
        2. Classify market regime (calm / normal / crisis)
        3. Ask RiskManager whether to trade and at what leverage
        4. Run MC Dropout inference → direction logits + magnitude
        5. Apply ConformalGate → trade_mask
        6. Estimate covariance from recent 200 trading days
        7. Run MVO optimizer (leverage capped by RiskManager)
        8. Compute turnover → deduct transaction cost from equity
        9. Update daily equity between rebalances

    Between rebalances: equity updated daily using actual stock returns
    weighted by the fixed target weights from the last rebalance.
    """

    WINDOW_SIZE      = 60   # lookback (must match training)
    FORECAST_HORIZON = 7    # rebalance every N trading days (~1 week)
    TCA_BPS          = 5.0  # transaction cost per unit of turnover (5 bps)
    MC_SAMPLES       = 10   # Monte Carlo Dropout passes per inference
    MIN_TURNOVER     = 0.15 # skip rebalance if total turnover < this
    MIN_RANK_STAB    = 0.40 # go flat if MC rank stability below this

    def __init__(
        self,
        checkpoint_path: str,
        risk_manager: RiskManager,
        portfolio_method: str = "mvo",
        portfolio_gamma: float = 1.0,
        min_tradable: int = 20,
        device: Optional[torch.device] = None,
        retrain_interval: int = 42,
        retrain_epochs: int = 60,
    ):
        self.checkpoint_path  = checkpoint_path
        self.risk_manager     = risk_manager
        self.portfolio_method = portfolio_method
        self.portfolio_gamma  = portfolio_gamma
        self.min_tradable     = min_tradable
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.retrain_interval = retrain_interval
        self.retrain_epochs   = retrain_epochs

        self._bundle  = None
        self._model   = None
        self._tickers: Optional[List[str]] = None
        self._feature_indices: Optional[List[int]] = None
        self._conformal_q_hat: Optional[float] = None
        self._conformal_calibrated: bool = False
        self._conformal_alpha: float = 0.10
        self._sector_mask: Optional[torch.Tensor] = None
        self._retrain_count: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        initial_capital: float = 10_000.0,
    ) -> SimulationResult:
        """
        Execute the full historical simulation.

        Args:
            start_date:      First trading date (YYYY-MM-DD).
            end_date:        Last trading date (default: today).
            initial_capital: Starting portfolio value in USD.

        Returns:
            SimulationResult with equity curve, trade log, and all metrics.
        """
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")

        print(f"\n{'='*65}")
        print("DGRCL Historical Paper Trading Simulator")
        print(f"  Period    : {start_date} → {end_date}")
        print(f"  Capital   : ${initial_capital:,.0f}")
        print(f"  Device    : {self.device}")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"{'='*65}\n")

        bundle  = self._get_bundle()
        model   = self._get_model()
        tickers = self._tickers
        feat_idx = self._feature_indices

        self._init_conformal_state()

        warmup = 700 if self.retrain_interval > 0 else 300
        cache = HistoricalDataCache(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            feature_indices=feat_idx,
            warmup_days=warmup,
        )

        trading_days = cache.trading_days
        if len(trading_days) == 0:
            raise ValueError(
                f"No trading days found between {start_date} and {end_date}. "
                "Check your date range and internet connection."
            )

        rebalance_dates = trading_days[:: self.FORECAST_HORIZON]
        rebalance_set   = set(rebalance_dates)

        retrain_label = (
            f"every {self.retrain_interval} days"
            if self.retrain_interval > 0 else "DISABLED"
        )
        print(f"\nSimulation plan:")
        print(f"  Trading days : {len(trading_days)}")
        print(f"  Rebalances   : {len(rebalance_dates)}")
        print(f"  Universe     : {len(tickers)} tickers")
        print(f"  Method       : {self.portfolio_method.upper()}")
        print(f"  Retraining   : {retrain_label}")

        # ---- Initialize state ----
        self.risk_manager.reset(initial_capital)
        equity          = initial_capital
        current_weights = np.zeros(len(tickers))  # start flat (all cash)
        prev_weights    = np.zeros(len(tickers))

        equity_curve: Dict[pd.Timestamp, float] = {}
        daily_pnl:    Dict[pd.Timestamp, float] = {}
        rebalance_log: List[RebalanceRecord] = []
        days_since_retrain = 0

        print(f"\n{'Date':<14}{'Equity':>11}{'Return':>8}  {'Regime':<9}{'Action'}")
        print("-" * 70)

        for i, day in enumerate(trading_days):
            # ---- Daily mark-to-market ----
            if i > 0:
                prev_day = trading_days[i - 1]
                period_rets = cache.get_period_returns(prev_day, day)
                port_ret = sum(
                    current_weights[j] * period_rets.get(tickers[j], 0.0)
                    for j in range(len(tickers))
                )
                day_pnl = equity * port_ret
                equity  = max(equity + day_pnl, 0.01)  # floor at $0.01
                daily_pnl[day] = day_pnl
            else:
                daily_pnl[day] = 0.0

            self.risk_manager.update_peak(equity)
            equity_curve[day] = equity
            days_since_retrain += 1

            if equity <= 0.01:
                print(f"\n⚠  Equity reached zero on {day.date()}. Stopping simulation.")
                break

            # ---- Rebalance (if scheduled) ----
            if day in rebalance_set:
                # Periodic retrain: refresh model on recent data
                if (self.retrain_interval > 0
                        and days_since_retrain >= self.retrain_interval):
                    try:
                        self._retrain_cycle(cache, day)
                        model = self._model
                    except Exception as e:
                        logger.warning("Retrain failed (%s), continuing with current model.", e)
                    days_since_retrain = 0

                rec = self._do_rebalance(
                    day=day,
                    equity=equity,
                    prev_weights=prev_weights,
                    current_weights=current_weights,
                    cache=cache,
                    bundle=bundle,
                    model=model,
                    tickers=tickers,
                )
                rebalance_log.append(rec)

                prev_weights    = current_weights.copy()
                current_weights = np.array([rec.weights.get(t, 0.0) for t in tickers])
                equity         -= rec.transaction_cost
                equity          = max(equity, 0.01)
                equity_curve[day] = equity

                if len(rebalance_log) % 4 == 1:
                    cum_ret = equity / initial_capital - 1.0
                    print(
                        f"{str(day.date()):<14}"
                        f"${equity:>9,.0f}  "
                        f"{cum_ret:>+6.1%}  "
                        f"{rec.regime:<9}"
                        f"{rec.risk_action}"
                    )

        print(f"\n{'='*65}")

        # ---- SPY benchmark ----
        spy_curve = self._fetch_spy_curve(
            start_date, end_date, initial_capital, trading_days
        )

        eq_series  = pd.Series(equity_curve)
        pnl_series = pd.Series(daily_pnl)

        result = SimulationResult(
            equity_curve=eq_series,
            daily_pnl=pnl_series,
            rebalance_log=rebalance_log,
            spy_curve=spy_curve,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            config=self._build_config_dict(),
        )

        print("\nPERFORMANCE SUMMARY")
        print("-" * 42)
        for k, v in result.metrics_dict().items():
            print(f"  {k:<30}: {v}")

        return result

    # ------------------------------------------------------------------
    # Rebalance logic
    # ------------------------------------------------------------------

    def _do_rebalance(
        self,
        day: pd.Timestamp,
        equity: float,
        prev_weights: np.ndarray,
        current_weights: np.ndarray,
        cache: HistoricalDataCache,
        bundle,
        model,
        tickers: List[str],
    ) -> RebalanceRecord:
        """Execute one rebalance cycle: data → inference → risk → weights."""

        # 1. Build model input
        try:
            stock_t, macro_t, active_mask = cache.get_model_input(day, self.WINDOW_SIZE)
        except ValueError as e:
            logger.warning("Skipping %s (insufficient history): %s", day.date(), e)
            return self._skip_record(
                day, equity, current_weights, tickers,
                "insufficient_history", prev_weights
            )

        stock_t      = stock_t.to(self.device)
        macro_t      = macro_t.to(self.device)
        active_np    = active_mask.numpy().astype(bool)

        # 2. Estimate regime from trailing cross-sectional vol
        regime, regime_vol = self._estimate_regime(cache, day, active_np, tickers)

        # 3. Risk manager decision
        ok_to_trade, effective_lev, risk_diag = self.risk_manager.check_rebalance(
            equity=equity, regime=regime
        )

        if not ok_to_trade:
            return self._skip_record(
                day, equity, current_weights, tickers,
                risk_diag.get("skip_reason", "risk_manager_skip"),
                prev_weights, regime=regime, regime_vol=regime_vol,
                risk_diag=risk_diag,
            )

        # 4. MC Dropout inference
        mc_out        = self._mc_inference(model, stock_t, macro_t)
        dir_logits_np = mc_out["dir_score_mean"].cpu().numpy()
        mag_preds_np  = mc_out["mag_mean"].cpu().numpy()
        rank_stability = float(mc_out["rank_stability"].item())

        # 4b. Confidence gate: if MC rank stability is too low, go flat.
        # Uses logger.warning (visible at default log level) so every skip
        # is traceable in the output without requiring --log-level DEBUG.
        if rank_stability < self.MIN_RANK_STAB:
            logger.warning(
                "%s: LOW RANK STABILITY (%.3f < %.2f) — skipping rebalance.",
                day.date(), rank_stability, self.MIN_RANK_STAB,
            )
            return self._skip_record(
                day, equity, current_weights, tickers,
                f"low_confidence_rank_stab={rank_stability:.3f}", prev_weights,
                regime=regime, regime_vol=regime_vol,
            )

        # 5. Conformal gate (uses local state, refreshed during retrain)
        from portfolio_optimizer import ConformalGate
        gate = ConformalGate(
            alpha=self._conformal_alpha,
            min_tradable=self.min_tradable,
        )
        if self._conformal_calibrated and self._conformal_q_hat is not None:
            gate._predictor._q_hat = self._conformal_q_hat
            gate._calibrated = True

        if gate._calibrated:
            trade_mask_np, gate_stats = gate.gate(dir_logits_np, active_np)
        else:
            trade_mask_np = active_np.copy()
            gate_stats = {
                "n_active":  int(active_np.sum()),
                "n_trade":   int(active_np.sum()),
                "gated_pct": 0.0,
            }

        n_active  = gate_stats.get("n_active", int(active_np.sum()))
        n_trade   = gate_stats.get("n_trade",  int(trade_mask_np.sum()))
        gated_pct = gate_stats.get("gated_pct", 0.0)

        # 6. Expected return estimate
        from portfolio_optimizer import ExpectedReturnEstimator, CovarianceEstimator, MeanVarianceOptimizer
        mu_est = ExpectedReturnEstimator()
        mu     = mu_est.compute(dir_logits_np, mag_preds_np, trade_mask_np)

        # 7. Fresh covariance from recent data (rolling 200-day window)
        ret_matrix, active_cov = cache.get_returns_matrix(day, n_days=200)
        cov_est = CovarianceEstimator()
        try:
            combined_active = active_np & active_cov
            cov_est.fit(ret_matrix, combined_active)
            sigma = cov_est.sigma
        except Exception as e:
            logger.warning("Covariance estimation failed (%s), using identity.", e)
            sigma = np.eye(len(tickers)) * 1e-2

        # Dollar-neutral: keep sum(w) == 0 unconditionally.
        # The model was trained and backtested as market-neutral; injecting a
        # directional beta tilt via net_exposure violates that assumption and
        # introduces uncompensated market risk when the ranking signal is weak.
        net_exp = 0.0

        # 8. MVO with regime-conditioned leverage
        if n_trade < 4:
            new_weights     = np.zeros(len(tickers))
            optimizer_status = "no_signal"
        else:
            mvo = MeanVarianceOptimizer(
                gamma=self.portfolio_gamma,
                max_leverage=effective_lev,
                max_position=0.05,
                net_exposure=net_exp,
                turnover_penalty=0.001,
            )
            try:
                new_weights      = mvo.optimize(mu, sigma, trade_mask_np,
                                                prev_weights=current_weights)
                optimizer_status = "optimal"
            except Exception as e:
                logger.warning("MVO failed (%s), using equal-weight fallback.", e)
                new_weights      = self._equal_ls_fallback(mu, trade_mask_np)
                optimizer_status = "fallback"

        # 8b. Suppress trivially small weight changes to reduce churn
        WEIGHT_DUST = 0.005
        delta = new_weights - current_weights
        dust_mask = np.abs(delta) < WEIGHT_DUST
        new_weights[dust_mask] = current_weights[dust_mask]

        # 9. Transaction cost + turnover threshold
        turnover         = float(np.abs(new_weights - current_weights).sum())
        transaction_cost = turnover * (self.TCA_BPS / 10_000.0) * equity

        if turnover < self.MIN_TURNOVER:
            return self._skip_record(
                day, equity, current_weights, tickers,
                "low_turnover", prev_weights,
                regime=regime, regime_vol=regime_vol,
            )

        # Top picks for logging
        long_idx  = np.where(new_weights > 1e-4)[0]
        short_idx = np.where(new_weights < -1e-4)[0]
        top_longs  = [tickers[i] for i in sorted(long_idx,  key=lambda i: -new_weights[i])[:5]]
        top_shorts = [tickers[i] for i in sorted(short_idx, key=lambda i:  new_weights[i])[:5]]

        weights_dict = {
            tickers[i]: float(new_weights[i])
            for i in range(len(tickers))
            if abs(new_weights[i]) > 1e-6
        }

        return RebalanceRecord(
            date=day,
            equity_before=equity,
            equity_after=equity - transaction_cost,
            weights=weights_dict,
            n_active=n_active,
            n_trade=n_trade,
            gated_pct=gated_pct,
            gross_leverage=float(np.abs(new_weights).sum()),
            turnover=turnover,
            transaction_cost=transaction_cost,
            regime=regime,
            regime_vol=regime_vol,
            rank_stability=rank_stability,
            risk_action="trade",
            drawdown_at_rebalance=self.risk_manager.current_drawdown(equity),
            effective_leverage=effective_lev,
            net_exposure=net_exp,
            top_longs=top_longs,
            top_shorts=top_shorts,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_bundle(self):
        if self._bundle is None:
            from checkpoint import CheckpointManager
            mgr = CheckpointManager(self.checkpoint_path)
            self._bundle = mgr.load()
            self._tickers        = self._bundle.tickers
            self._feature_indices = self._bundle.feature_indices
        return self._bundle

    def _get_model(self):
        if self._model is None:
            bundle = self._get_bundle()
            self._model = bundle.build_model(self.device)
            self._model.eval()
        return self._model

    def _init_conformal_state(self):
        """Seed local conformal state from the initial checkpoint bundle."""
        bundle = self._get_bundle()
        self._conformal_q_hat = bundle.conformal_q_hat
        self._conformal_calibrated = bundle.conformal_calibrated
        self._conformal_alpha = bundle.conformal_alpha

        sector_ids = bundle.sector_ids
        if sector_ids is not None:
            ids = torch.tensor(sector_ids, dtype=torch.long)
            self._sector_mask = (ids.unsqueeze(0) == ids.unsqueeze(1))
        else:
            self._sector_mask = None

    def _build_tensors_from_cache(
        self,
        cache: HistoricalDataCache,
        as_of_date: pd.Timestamp,
        n_days: int = 400,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Assemble [N, T, F] feature/return/mask tensors from the cache for
        the last n_days of trading data up to as_of_date.
        """
        avail = [d for d in cache._all_days if d <= as_of_date]
        if len(avail) < n_days:
            logger.warning(
                "Not enough history for retrain (%d < %d days). Skipping.",
                len(avail), n_days,
            )
            return None

        window = pd.DatetimeIndex(avail[-n_days:])
        T = len(window)
        S = len(self._feature_indices)
        Ns = len(self._tickers)
        Nm = len(cache.MACRO_TICKERS_MAP)

        selected_feat = [cache.ALL_FEATURE_NAMES[i] for i in self._feature_indices]

        stock_tensor = torch.zeros(Ns, T, S, dtype=torch.float32)
        inclusion_mask = torch.zeros(Ns, T, dtype=torch.bool)
        returns_tensor = torch.zeros(Ns, T, dtype=torch.float32)

        for si, ticker in enumerate(self._tickers):
            df = cache._stock_feat.get(ticker)
            if df is None or df.empty:
                continue
            aligned = df[selected_feat].reindex(window, fill_value=0.0)
            valid = aligned.abs().sum(axis=1) > 0
            stock_tensor[si] = torch.from_numpy(aligned.values.astype(np.float32))
            inclusion_mask[si] = torch.from_numpy(valid.values)

            close = cache._stock_close.get(ticker)
            if close is not None and not close.empty:
                close_aligned = close.reindex(window).ffill()
                log_ret = np.log(close_aligned / close_aligned.shift(1)).fillna(0.0)
                returns_tensor[si] = torch.from_numpy(log_ret.values.astype(np.float32))

        macro_tensor = torch.zeros(Nm, T, 4, dtype=torch.float32)
        for mi, name in enumerate(cache.MACRO_TICKERS_MAP):
            df = cache._macro_feat.get(name)
            if df is None or df.empty:
                continue
            aligned = df[cache.ALL_MACRO_FEATURE_NAMES].reindex(window, fill_value=0.0)
            macro_tensor[mi] = torch.from_numpy(aligned.values.astype(np.float32))

        return stock_tensor, macro_tensor, returns_tensor, inclusion_mask

    def _retrain_cycle(
        self,
        cache: HistoricalDataCache,
        as_of_date: pd.Timestamp,
    ) -> bool:
        """
        Train a fresh model on the recent data window ending at as_of_date
        and recalibrate the conformal gate.  Returns True on success.
        """
        from train import (
            train_epoch, evaluate, EarlyStopping,
            create_sequential_snapshots, compute_regime_vol,
            classify_regime, compute_adaptive_lambda,
        )
        from macro_dgrcl import MacroDGRCL

        self._retrain_count += 1
        print(f"\n  ** Retraining #{self._retrain_count} "
              f"(as of {as_of_date.date()}) **")

        tensors = self._build_tensors_from_cache(cache, as_of_date, n_days=600)
        if tensors is None:
            return False
        stock_t, macro_t, returns_t, inclusion = tensors

        snapshots = create_sequential_snapshots(
            stock_t, macro_t, returns_t,
            window_size=self.WINDOW_SIZE,
            step_size=1,
            forecast_horizon=self.FORECAST_HORIZON,
            inclusion_mask=inclusion,
        )
        if len(snapshots) < 80:
            print(f"     Only {len(snapshots)} snapshots — too few. Skipping.")
            return False

        n_val = min(100, len(snapshots) // 4)
        n_train = len(snapshots) - n_val
        train_data = snapshots[:n_train]
        val_data = snapshots[n_train:]

        bundle = self._get_bundle()
        cfg = bundle.model_config
        model = MacroDGRCL(
            num_stocks=cfg["num_stocks"],
            num_macros=cfg["num_macros"],
            stock_feature_dim=cfg["stock_feature_dim"],
            macro_feature_dim=cfg["macro_feature_dim"],
            hidden_dim=cfg["hidden_dim"],
            temporal_layers=cfg["temporal_layers"],
            mp_layers=cfg["mp_layers"],
            heads=cfg["heads"],
            top_k=cfg["top_k"],
            dropout=cfg["dropout"],
            head_dropout=cfg["head_dropout"],
        ).to(self.device)

        # Warm-start from the current model's weights so fine-tuning preserves
        # learned cross-sectional structure rather than training from scratch.
        if self._model is not None:
            model.load_state_dict(self._model.state_dict())
            print(f"     Warm-starting from current model weights.")

        # Lower learning rate for fine-tuning (vs cold-start training)
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.retrain_epochs, eta_min=1e-7,
        )
        mse_loss_fn = nn.SmoothL1Loss()

        realized_vol = compute_regime_vol(train_data, lookback=20)
        regime = classify_regime(realized_vol)
        adaptive_lambda = compute_adaptive_lambda(realized_vol, base_lambda=0.05)
        print(f"     Train: {n_train} snaps, Val: {n_val} snaps | "
              f"Regime: {regime} | vol={realized_vol:.4f} | λ={adaptive_lambda:.3f}")

        early_stopping = EarlyStopping(base_patience=7, max_patience=14)
        early_stopping.update_patience(realized_vol, high_vol_threshold=0.50)

        sector_mask = self._sector_mask
        best_val = float("inf")
        for epoch in range(self.retrain_epochs):
            train_epoch(
                model=model,
                data_loader=train_data,
                optimizer=optimizer,
                mse_loss_fn=mse_loss_fn,
                device=self.device,
                mag_weight=adaptive_lambda,
                max_grad_norm=0.5,
                sector_mask=sector_mask,
            )
            scheduler.step()

            if val_data:
                val_metrics = evaluate(
                    model=model,
                    data_loader=val_data,
                    mse_loss_fn=mse_loss_fn,
                    device=self.device,
                    mag_weight=0.05,
                    sector_mask=sector_mask,
                )
                val_loss = val_metrics["loss"]
                if val_loss < best_val:
                    best_val = val_loss
                if early_stopping(val_loss, model):
                    early_stopping.restore_best(model)
                    break

        final_epoch = epoch + 1
        print(f"     Trained {final_epoch} epochs (best val loss={best_val:.4f})")

        # --- Recalibrate conformal gate on validation data ---
        model.eval()
        from portfolio_optimizer import ConformalGate
        gate = ConformalGate(
            alpha=self._conformal_alpha, min_tradable=self.min_tradable,
        )
        cal_logits, cal_labels = [], []
        cal_end = max(1, len(val_data) // 2)
        with torch.no_grad():
            for snap in val_data[:cal_end]:
                sf = snap[0].to(self.device)
                mf = snap[1].to(self.device)
                am = snap[3].to(self.device) if len(snap) == 4 else None
                logits_snap, _ = model(
                    stock_features=sf, macro_features=mf, active_mask=am,
                )
                rets_np = snap[2].numpy()
                logits_np = logits_snap.squeeze(-1).cpu().numpy()
                mask_np = (
                    am.cpu().numpy().astype(bool) if am is not None
                    else np.ones(len(logits_np), dtype=bool)
                )
                cal_logits.extend(logits_np[mask_np].tolist())
                cal_labels.extend((rets_np[mask_np] > 0).astype(float).tolist())

        if len(cal_logits) > 10:
            gate.calibrate(
                np.array(cal_logits, dtype=np.float32),
                np.array(cal_labels, dtype=np.float32),
            )
            self._conformal_q_hat = float(gate._predictor._q_hat)
            self._conformal_calibrated = True
            print(f"     Conformal gate recalibrated "
                  f"(q_hat={self._conformal_q_hat:.4f}, "
                  f"{len(cal_logits)} samples)")
        else:
            print("     Insufficient data for conformal recalibration")

        self._model = model
        self._model.eval()
        print(f"     Retrain complete. Model updated.")
        return True

    def _mc_inference(
        self,
        model,
        stock_t: torch.Tensor,
        macro_t: torch.Tensor,
    ) -> Dict:
        """Monte Carlo Dropout inference (model.train() keeps dropout active)."""
        model.train()
        dir_scores_all, mag_preds_all = [], []

        with torch.no_grad():
            for _ in range(self.MC_SAMPLES):
                dir_l, mag_o = model(stock_features=stock_t, macro_features=macro_t)
                dir_scores_all.append(dir_l.squeeze(-1))
                mag_preds_all.append(mag_o.squeeze(-1))

        model.eval()

        dir_stack = torch.stack(dir_scores_all, dim=0)
        mag_stack = torch.stack(mag_preds_all,  dim=0)
        dir_mean  = dir_stack.mean(0)
        dir_std   = dir_stack.std(0)

        # Rank stability: mean Spearman ρ across MC passes
        ranks = torch.stack(
            [s.argsort().argsort().float() for s in dir_scores_all], dim=0
        )
        rank_stab = torch.tensor(1.0)
        if self.MC_SAMPLES > 1:
            base_r = ranks[0]
            rhos   = []
            for i in range(1, self.MC_SAMPLES):
                r_i = ranks[i]
                d_sq = ((base_r - r_i) ** 2).sum()
                n    = float(r_i.numel())
                rhos.append(1.0 - 6.0 * float(d_sq) / max(n * (n * n - 1), 1.0))
            rank_stab = torch.tensor(float(np.mean(rhos)))

        return {
            "dir_score_mean": dir_mean,
            "mag_mean":       mag_stack.mean(0),
            "confidence":     1.0 / (1.0 + dir_std),
            "rank_stability": rank_stab,
        }

    def _estimate_regime(
        self,
        cache: HistoricalDataCache,
        day: pd.Timestamp,
        active_np: np.ndarray,
        tickers: List[str],
    ) -> Tuple[str, float]:
        """
        Classify market regime from trailing 20-day cross-sectional return vol.

        Thresholds are calibrated for raw arithmetic returns from
        get_period_returns() (typical range 0.001–0.015 per day), which are
        ~20-30x smaller than the z-score-normalized returns used in train.py.
        The empirical distribution of this raw-return vol across the 2022-2026
        simulation period is:
            calm   < 0.002  (quiet, trending market)
            normal   0.002–0.005
            crisis > 0.005  (high cross-sectional dispersion day)
        """
        try:
            avail  = [d for d in cache._all_days if d <= day]
            window = avail[-21:] if len(avail) >= 21 else avail
            if len(window) < 5:
                return "normal", 0.0

            active_tickers = [tickers[i] for i in range(len(tickers)) if active_np[i]]
            if not active_tickers:
                return "normal", 0.0

            mean_abs_rets = []
            for d_idx in range(1, len(window)):
                rets = cache.get_period_returns(window[d_idx - 1], window[d_idx])
                vals = [abs(rets.get(t, 0.0)) for t in active_tickers]
                if vals:
                    mean_abs_rets.append(float(np.mean(vals)))

            if len(mean_abs_rets) < 2:
                return "normal", 0.0

            vol = float(np.std(mean_abs_rets, ddof=1))
            if vol < 0.002:
                return "calm",   vol
            elif vol >= 0.005:
                return "crisis", vol
            else:
                return "normal", vol

        except Exception:
            return "normal", 0.0

    def _compute_net_exposure(
        self,
        cache: HistoricalDataCache,
        day: pd.Timestamp,
        active_np: np.ndarray,
        tickers: List[str],
        regime: str,
    ) -> float:
        """
        Regime-aware directional tilt derived from the equal-weighted universe
        momentum over 60 days.  Returns a net exposure in [-0.3, +0.3] that
        replaces the old sum(w)==0 dollar-neutral constraint.

        Calm  bull  → +0.30  (full long tilt)
        Normal bull → +0.20
        Crisis      → -0.10  (slight defensive short tilt)
        Bear market → 0 (stay neutral if trend is down)
        """
        TILT_CAP = {"calm": 0.30, "normal": 0.20, "crisis": -0.10}
        try:
            avail = [d for d in cache._all_days if d <= day]
            lookback = min(60, len(avail) - 1)
            if lookback < 10:
                return 0.0

            start_d = avail[-(lookback + 1)]
            end_d = avail[-1]

            active_tickers = [tickers[i] for i in range(len(tickers)) if active_np[i]]
            if len(active_tickers) < 10:
                return 0.0

            rets = cache.get_period_returns(start_d, end_d)
            cum_rets = [rets.get(t, 0.0) for t in active_tickers]
            mean_ret = float(np.mean(cum_rets))

            cap = TILT_CAP.get(regime, 0.20)
            if regime == "crisis":
                return cap

            if mean_ret > 0:
                strength = min(mean_ret / 0.10, 1.0)
                return cap * strength
            else:
                return 0.0

        except Exception:
            return 0.0

    @staticmethod
    def _equal_ls_fallback(mu: np.ndarray, trade_mask: np.ndarray) -> np.ndarray:
        """Top-20% long / bottom-20% short equal-weight fallback."""
        N          = len(mu)
        active_idx = np.where(trade_mask)[0]
        if len(active_idx) < 2:
            return np.zeros(N)
        k          = max(1, int(0.20 * len(active_idx)))
        sorted_i   = np.argsort(mu[active_idx])
        w          = np.zeros(N)
        w[active_idx[sorted_i[-k:]]] = 1.0 / k
        w[active_idx[sorted_i[:k]]] = -1.0 / k
        return w

    def _skip_record(
        self,
        day: pd.Timestamp,
        equity: float,
        current_weights: np.ndarray,
        tickers: List[str],
        risk_action: str,
        prev_weights: np.ndarray,
        regime: str = "normal",
        regime_vol: float = 0.0,
        risk_diag: Optional[Dict] = None,
    ) -> RebalanceRecord:
        """No-trade record — hold current positions unchanged."""
        weights_dict = {
            tickers[i]: float(current_weights[i])
            for i in range(len(tickers))
            if abs(current_weights[i]) > 1e-6
        }
        return RebalanceRecord(
            date=day,
            equity_before=equity,
            equity_after=equity,
            weights=weights_dict,
            n_active=0,
            n_trade=0,
            gated_pct=0.0,
            gross_leverage=float(np.abs(current_weights).sum()),
            turnover=0.0,
            transaction_cost=0.0,
            regime=regime,
            regime_vol=regime_vol,
            rank_stability=0.0,
            risk_action=risk_action,
            drawdown_at_rebalance=self.risk_manager.current_drawdown(equity),
            effective_leverage=0.0,
        )

    def _fetch_spy_curve(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float,
        trading_days: pd.DatetimeIndex,
    ) -> Optional[pd.Series]:
        """Download SPY close prices and build a buy-and-hold equity curve."""
        try:
            import yfinance as yf
            raw = yf.download("SPY", start=start_date, end=end_date,
                              progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"].reindex(trading_days).ffill().dropna()
            if len(close) == 0:
                return None
            spy_curve = initial_capital * (close / close.iloc[0])
            spy_curve.name = "SPY"
            return spy_curve
        except Exception as e:
            logger.warning("Could not fetch SPY benchmark: %s", e)
            return None

    def _build_config_dict(self) -> Dict:
        return {
            "checkpoint_path":  self.checkpoint_path,
            "portfolio_method": self.portfolio_method,
            "portfolio_gamma":  self.portfolio_gamma,
            "min_tradable":     self.min_tradable,
            "mc_samples":       self.MC_SAMPLES,
            "tca_bps":          self.TCA_BPS,
            "window_size":      self.WINDOW_SIZE,
            "forecast_horizon": self.FORECAST_HORIZON,
            "min_turnover":     self.MIN_TURNOVER,
            "min_rank_stability": self.MIN_RANK_STAB,
            "retrain_interval": self.retrain_interval,
            "retrain_epochs":   self.retrain_epochs,
            "retrain_count":    self._retrain_count,
            "risk": {
                "max_dd_pct":          self.risk_manager.max_dd_pct,
                "cooldown_days":       self.risk_manager.cooldown_days,
                "rebalance_loss_pct":  self.risk_manager.rebalance_loss_pct,
                "normal_leverage":     self.risk_manager.normal_leverage,
                "crisis_leverage":     self.risk_manager.crisis_leverage,
                "calm_leverage":       self.risk_manager.calm_leverage,
            },
        }


# =============================================================================
# REPORTER
# =============================================================================

class Reporter:
    """
    Generates all output artefacts for a completed simulation:
      - paper_sim_report.png  : 4-panel figure (equity, drawdown, monthly
                                returns, rolling Sharpe)
      - paper_sim_metrics.json: performance summary
      - paper_sim_trades.csv  : full rebalance trade log
      - paper_sim_equity.csv  : daily equity values
    """

    MONTH_ABBR = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",  6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    @classmethod
    def save_all(cls, result: SimulationResult, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        cls._save_report_figure(result, output_dir)
        cls._save_metrics_json(result, output_dir)
        cls._save_trade_log(result, output_dir)
        cls._save_equity_csv(result, output_dir)
        print(f"\nAll outputs saved to: {output_dir}/")

    # ------------------------------------------------------------------
    # Multi-panel figure
    # ------------------------------------------------------------------

    @classmethod
    def _save_report_figure(cls, result: SimulationResult, output_dir: str):
        fig = plt.figure(figsize=(18, 14))
        gs  = gridspec.GridSpec(
            2, 2, figure=fig,
            hspace=0.38, wspace=0.32,
            left=0.07, right=0.96, top=0.93, bottom=0.07,
        )

        ax_eq  = fig.add_subplot(gs[0, 0])
        ax_dd  = fig.add_subplot(gs[0, 1])
        ax_mo  = fig.add_subplot(gs[1, 0])
        ax_sh  = fig.add_subplot(gs[1, 1])

        cls._plot_equity(ax_eq, result)
        cls._plot_drawdown(ax_dd, result)
        cls._plot_monthly_heatmap(ax_mo, result)
        cls._plot_rolling_sharpe(ax_sh, result)

        fig.suptitle(
            f"DGRCL Paper Trading  |  {result.start_date} → {result.end_date}  "
            f"|  ${result.initial_capital:,.0f} initial capital",
            fontsize=14, fontweight="bold", y=0.975,
        )

        path = os.path.join(output_dir, "paper_sim_report.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    @classmethod
    def _plot_equity(cls, ax: plt.Axes, result: SimulationResult):
        eq = result.equity_curve
        ax.plot(eq.index, eq.values, color="#1f77b4", linewidth=1.8,
                label=f"DGRCL  ({result.total_return():+.1%})")

        if result.spy_curve is not None:
            spy = result.spy_curve.reindex(eq.index).ffill()
            spy_ret = spy.iloc[-1] / spy.iloc[0] - 1.0
            ax.plot(spy.index, spy.values, color="#d62728", linewidth=1.4,
                    alpha=0.7, linestyle="--", label=f"SPY B&H  ({spy_ret:+.1%})")

        ax.axhline(result.initial_capital, color="grey", linewidth=0.8,
                   linestyle=":", alpha=0.7, label="Initial capital")

        m = result.metrics_dict()
        info = (
            f"CAGR: {m['cagr']}  |  Sharpe: {m['sharpe_rf5pct']}  |  "
            f"MaxDD: {m['max_drawdown']}"
        )
        ax.set_title(f"Equity Curve\n{info}", fontsize=10)
        ax.set_ylabel("Portfolio Value ($)")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    @classmethod
    def _plot_drawdown(cls, ax: plt.Axes, result: SimulationResult):
        dd = result.drawdown_series() * 100

        ax.fill_between(dd.index, 0, -dd.values, color="#d62728", alpha=0.5,
                        label=f"DGRCL (max {result.max_drawdown():.1%})")

        if result.spy_curve is not None:
            eq   = result.equity_curve
            spy  = result.spy_curve.reindex(eq.index).ffill()
            spy_dd = ((spy.cummax() - spy) / spy.cummax()) * 100
            # spy_dd is already in percentage points (×100); divide back to a
            # fraction before passing to :.1% which multiplies by 100 again.
            spy_max_dd = float(spy_dd.max()) / 100.0
            ax.fill_between(spy_dd.index, 0, -spy_dd.values, color="#aec7e8",
                            alpha=0.4, label=f"SPY (max {spy_max_dd:.1%})")

        # Mark circuit-breaker trips
        for rec in result.rebalance_log:
            if "circuit_breaker_tripped" in rec.risk_action:
                ax.axvline(rec.date, color="orange", linewidth=1.0,
                           alpha=0.7, linestyle="--")

        ax.set_title("Drawdown from Peak", fontsize=10)
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    @classmethod
    def _plot_monthly_heatmap(cls, ax: plt.Axes, result: SimulationResult):
        grid = result.monthly_returns_grid()
        if grid.empty:
            ax.text(0.5, 0.5, "No monthly data", ha="center", va="center")
            ax.set_title("Monthly Returns", fontsize=10)
            return

        vmax = max(abs(grid.values[~np.isnan(grid.values)]).max(), 0.01)
        im   = ax.imshow(grid.values, cmap="RdYlGn", aspect="auto",
                         vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([cls.MONTH_ABBR.get(c, str(c)) for c in grid.columns],
                           fontsize=8)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([str(y) for y in grid.index], fontsize=8)

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                val = grid.values[r, c]
                if not np.isnan(val):
                    txt_color = "white" if abs(val) > vmax * 0.5 else "black"
                    ax.text(c, r, f"{val:.1%}", ha="center", va="center",
                            fontsize=6.5, color=txt_color)

        plt.colorbar(im, ax=ax, format=matplotlib.ticker.PercentFormatter(1.0),
                     fraction=0.046, pad=0.04)
        ax.set_title("Monthly Returns Heatmap", fontsize=10)

    @classmethod
    def _plot_rolling_sharpe(cls, ax: plt.Axes, result: SimulationResult):
        roll_sh = result.rolling_sharpe(window=90)
        ax.plot(roll_sh.index, roll_sh.values, color="#2ca02c", linewidth=1.5,
                label="DGRCL 90-day rolling Sharpe")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.axhline(result.sharpe(), color="#1f77b4", linewidth=1.0,
                   linestyle=":", alpha=0.8,
                   label=f"Full-period Sharpe = {result.sharpe():.2f}")

        if result.spy_curve is not None:
            spy_dr = result.spy_curve.reindex(result.equity_curve.index).ffill().pct_change()
            spy_roll = (spy_dr.rolling(90).mean() / spy_dr.rolling(90).std().clip(1e-10)) * math.sqrt(252)
            ax.plot(spy_roll.index, spy_roll.values, color="#d62728",
                    linewidth=1.2, alpha=0.6, linestyle="--", label="SPY 90-day Sharpe")

        ax.set_title("Rolling 90-Day Sharpe Ratio (rf=5%)", fontsize=10)
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    # ------------------------------------------------------------------
    # JSON + CSV outputs
    # ------------------------------------------------------------------

    @classmethod
    def _save_metrics_json(cls, result: SimulationResult, output_dir: str):
        m    = result.metrics_dict()
        data = {
            "summary":       m,
            "config":        result.config,
            "start_date":    result.start_date,
            "end_date":      result.end_date,
            "generated_at":  datetime.utcnow().isoformat() + "Z",
        }
        if result.spy_curve is not None:
            spy_ret = result.spy_curve.iloc[-1] / result.spy_curve.iloc[0] - 1.0
            data["spy_total_return"] = f"{spy_ret:.2%}"

        path = os.path.join(output_dir, "paper_sim_metrics.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {path}")

    @classmethod
    def _save_trade_log(cls, result: SimulationResult, output_dir: str):
        path = os.path.join(output_dir, "paper_sim_trades.csv")
        rows = []
        for r in result.rebalance_log:
            rows.append({
                "date":                   str(r.date.date()),
                "equity_before":          round(r.equity_before, 2),
                "equity_after":           round(r.equity_after, 2),
                "risk_action":            r.risk_action,
                "regime":                 r.regime,
                "regime_vol":             round(r.regime_vol, 6),
                "effective_leverage":     round(r.effective_leverage, 3),
                "gross_leverage":         round(r.gross_leverage, 4),
                "n_active":               r.n_active,
                "n_trade":                r.n_trade,
                "gated_pct":              round(r.gated_pct, 4),
                "turnover":               round(r.turnover, 6),
                "transaction_cost":       round(r.transaction_cost, 4),
                "rank_stability":         round(r.rank_stability, 4),
                "drawdown_at_rebalance":  round(r.drawdown_at_rebalance, 4),
                "top_longs":              "|".join(r.top_longs),
                "top_shorts":             "|".join(r.top_shorts),
            })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(rows)} records)")

    @classmethod
    def _save_equity_csv(cls, result: SimulationResult, output_dir: str):
        path = os.path.join(output_dir, "paper_sim_equity.csv")
        df   = result.equity_curve.rename("equity").to_frame()
        df["daily_pnl"]   = result.daily_pnl.reindex(df.index)
        df["daily_return"] = result.equity_curve.pct_change()
        df["drawdown"]     = result.drawdown_series()
        if result.spy_curve is not None:
            df["spy_equity"] = result.spy_curve.reindex(df.index)
        df.index.name = "date"
        df.to_csv(path, float_format="%.6f")
        print(f"  Saved: {path}  ({len(df)} days)")


# =============================================================================
# TRAIN-FIRST HELPER
# =============================================================================

def train_checkpoint(
    checkpoint_dir: str,
    portfolio_method: str,
    portfolio_gamma: float,
    force_cpu: bool,
):
    """
    Train the DGRCL model on all available historical data and save a
    checkpoint for use by the paper simulator.

    Delegates to train.main() which runs the full walk-forward pipeline.
    NOTE: For a strictly look-ahead-free simulation, users should manually
    limit training to folds that end before their simulation start date using
    `--end-fold` in train.py.  This helper trains on all data as a convenience.
    """
    print("\nTraining DGRCL model (this may take a while)...")
    print("Tip: run  train.py --real-data --save-checkpoint  manually to")
    print("control exactly which folds are trained.\n")

    try:
        import train as train_module
        train_module.main(
            use_real_data=True,
            epochs=100,
            save_checkpoint=True,
            checkpoint_dir=checkpoint_dir,
            portfolio_method=portfolio_method,
            portfolio_gamma=portfolio_gamma,
            force_cpu=force_cpu,
        )
    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("Please run:  python train.py --real-data --save-checkpoint")
        sys.exit(1)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "DGRCL Historical Paper Trading Simulator\n"
            "Replays the strategy from a start date to present with simulated capital."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Date / capital ----
    parser.add_argument(
        "--start-date", type=str, default="2022-01-01",
        help="Simulation start date YYYY-MM-DD (default: 2022-01-01)",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Simulation end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0,
        help="Initial capital in USD (default: 10000)",
    )

    # ---- Checkpoint ----
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/latest.pt",
        help="Path to trained model checkpoint (default: ./checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--train-first", action="store_true",
        help="Train the model on all available data before simulating",
    )

    # ---- Portfolio ----
    parser.add_argument(
        "--portfolio-method", type=str, default="mvo",
        choices=["mvo", "riskparity", "naive"],
        help="Portfolio construction method (default: mvo)",
    )
    parser.add_argument(
        "--portfolio-gamma", type=float, default=1.0,
        help="MVO risk-aversion coefficient (default: 1.0)",
    )
    parser.add_argument(
        "--min-tradable", type=int, default=20,
        help="Min stocks after conformal gating (default: 20)",
    )

    # ---- Risk management ----
    parser.add_argument(
        "--max-drawdown-pct", type=float, default=0.18,
        help="Circuit breaker: max peak-to-trough drawdown before going flat (default: 0.18 = 18%%)",
    )
    parser.add_argument(
        "--cooldown-days", type=int, default=2,
        help="Rebalance periods to stay flat after circuit breaker trips (default: 2 ≈ 14 trading days)",
    )
    parser.add_argument(
        "--rebalance-loss-pct", type=float, default=0.08,
        help="Skip rebalance if equity fell > X%% since last rebalance (default: 0.08 = 8%%)",
    )
    parser.add_argument(
        "--normal-leverage", type=float, default=1.5,
        help="MVO gross leverage cap in normal regime (default: 1.5)",
    )
    parser.add_argument(
        "--crisis-leverage", type=float, default=0.8,
        help="MVO gross leverage cap in crisis regime (default: 0.8)",
    )
    parser.add_argument(
        "--calm-leverage", type=float, default=1.5,
        help="MVO gross leverage cap in calm regime (default: 1.5)",
    )

    # ---- Retraining ----
    parser.add_argument(
        "--retrain-interval", type=int, default=42,
        help="Retrain the model every N trading days (default: 42 ≈ every 2 months). "
             "Set to 0 to disable retraining (original frozen-model behavior).",
    )
    parser.add_argument(
        "--retrain-epochs", type=int, default=60,
        help="Max training epochs per retrain cycle (default: 60)",
    )

    # ---- Output ----
    parser.add_argument(
        "--output-dir", type=str, default="./paper_sim_results",
        help="Directory for all output files (default: ./paper_sim_results)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU mode (slower but avoids GPU memory issues)",
    )
    parser.add_argument(
        "--log-level", type=str, default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )

    args = parser.parse_args()

    # ---- Logging ----
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Device ----
    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ---- Train first (optional) ----
    if args.train_first:
        train_checkpoint(
            checkpoint_dir=os.path.dirname(args.checkpoint) or "./checkpoints",
            portfolio_method=args.portfolio_method,
            portfolio_gamma=args.portfolio_gamma,
            force_cpu=args.cpu,
        )

    # ---- Validate checkpoint ----
    if not os.path.exists(args.checkpoint):
        print(
            f"\nERROR: Checkpoint not found: {args.checkpoint}\n"
            "Please run one of:\n"
            "  python train.py --real-data --save-checkpoint\n"
            "  python paper_sim.py --train-first ...\n"
        )
        sys.exit(1)

    # ---- Build risk manager ----
    risk_mgr = RiskManager(
        max_dd_pct=args.max_drawdown_pct,
        cooldown_days=args.cooldown_days,
        rebalance_loss_pct=args.rebalance_loss_pct,
        normal_leverage=args.normal_leverage,
        crisis_leverage=args.crisis_leverage,
        calm_leverage=args.calm_leverage,
    )

    print("\nRisk Management Settings:")
    print(f"  Max drawdown circuit breaker : {args.max_drawdown_pct:.0%}")
    print(f"  Cooldown after breaker trips  : {args.cooldown_days} rebalance periods")
    print(f"  Per-rebalance loss guard      : {args.rebalance_loss_pct:.0%}")
    print(f"  Leverage (calm/normal/crisis) : "
          f"{args.calm_leverage:.1f} / {args.normal_leverage:.1f} / {args.crisis_leverage:.1f}")
    print(f"  Retrain interval              : "
          f"{args.retrain_interval} trading days" if args.retrain_interval > 0 else "  Retrain interval              : DISABLED")

    # ---- Run simulation ----
    sim = PaperSimulator(
        checkpoint_path=args.checkpoint,
        risk_manager=risk_mgr,
        portfolio_method=args.portfolio_method,
        portfolio_gamma=args.portfolio_gamma,
        min_tradable=args.min_tradable,
        device=device,
        retrain_interval=args.retrain_interval,
        retrain_epochs=args.retrain_epochs,
    )

    result = sim.run(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
    )

    # ---- Save all outputs ----
    Reporter.save_all(result, args.output_dir)


if __name__ == "__main__":
    main()
