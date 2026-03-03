"""
Alpaca Broker Integration for DGRCL Paper Trading

Handles the full order lifecycle:
  1. Auth  — connects to Alpaca paper-trading API
  2. Status — reads current positions and account equity
  3. Diff   — computes target vs current weight differences
  4. Orders — submits fractional market orders, respects safety limits
  5. Fills  — polls for fill confirmation and logs execution

Paper trading base URL: https://paper-api.alpaca.markets
Live trading base URL:  https://api.alpaca.markets  (⚠️ real money)

Requirements:
    pip install alpaca-trade-api

Environment variables (set in .env or shell):
    ALPACA_API_KEY    — your Alpaca API key ID
    ALPACA_SECRET_KEY — your Alpaca API secret key
    ALPACA_BASE_URL   — defaults to paper trading endpoint
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PAPER_URL = "https://paper-api.alpaca.markets"
_LIVE_URL = "https://api.alpaca.markets"

# Alpaca's minimum notional order size ($1 for fractional shares)
_MIN_ORDER_NOTIONAL = 1.0

# Maximum retries when checking for fills
_FILL_POLL_RETRIES = 12
_FILL_POLL_INTERVAL_S = 5


# =============================================================================
# POSITION / ORDER DATACLASSES
# =============================================================================

@dataclass
class Position:
    ticker: str
    qty: float          # fractional shares held (negative = short)
    market_value: float # signed dollar value
    weight: float       # as fraction of account equity


@dataclass
class Order:
    ticker: str
    side: str           # "buy" or "sell"
    notional: float     # dollar amount to trade
    qty: Optional[float] = None  # shares (alternative to notional)
    order_id: Optional[str] = None
    status: str = "pending"
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0


@dataclass
class ExecutionReport:
    timestamp: str
    account_equity: float
    orders_submitted: List[Order]
    orders_failed: List[Dict]
    total_notional_traded: float
    estimated_cost_bps: float
    error: Optional[str] = None


# =============================================================================
# ALPACA BROKER
# =============================================================================

class AlpacaBroker:
    """
    Thin wrapper around the Alpaca REST API for paper trading.

    All order sizes are computed from target fractional weights and current
    account equity.  A min-notional filter ($1) prevents tiny stub orders.
    Safety checks:
      - max_order_pct:    No single order > X% of equity (default 10%)
      - daily_loss_limit: Halt trading if daily P&L < -Y% of equity (default 5%)
      - kill_switch:      Set cfg.kill_switch = True to disable all trading
    """

    def __init__(self, config):
        """
        Args:
            config: LiveConfig object with Alpaca credentials and risk limits.
        """
        self.cfg = config
        self._api = None  # lazy-initialized

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_account_equity(self) -> float:
        """Return current portfolio equity in USD."""
        api = self._get_api()
        account = api.get_account()
        return float(account.equity)

    def get_positions(self, equity: Optional[float] = None) -> Dict[str, Position]:
        """
        Return current positions as a dict {ticker: Position}.
        Weights are computed from provided equity (or fetched live if None).
        """
        api = self._get_api()
        if equity is None:
            equity = self.get_account_equity()
        raw_positions = api.list_positions()
        result: Dict[str, Position] = {}
        for pos in raw_positions:
            ticker = pos.symbol
            qty = float(pos.qty)
            market_value = float(pos.market_value)
            weight = market_value / equity if equity > 0 else 0.0
            result[ticker] = Position(
                ticker=ticker,
                qty=qty,
                market_value=market_value,
                weight=weight,
            )
        return result

    def compute_orders(
        self,
        target_weights: np.ndarray,  # [N_stocks]
        tickers: List[str],          # length N_stocks
        equity: float,
    ) -> Tuple[List[Order], float]:
        """
        Compute the set of orders needed to move from current positions to
        target weights.

        Returns:
            orders:          List of Order objects (buys and sells)
            total_notional:  Gross notional to trade (USD)
        """
        current_positions = self.get_positions(equity)
        current_weights: Dict[str, float] = {
            t: p.weight for t, p in current_positions.items()
        }

        orders: List[Order] = []
        for i, ticker in enumerate(tickers):
            target_w = float(target_weights[i])
            current_w = current_weights.get(ticker, 0.0)
            delta_w = target_w - current_w

            # Skip if the weight change is below the minimum threshold
            if abs(delta_w) < self.cfg.min_weight_change:
                continue

            notional = abs(delta_w) * equity

            # Clamp to max single order size
            max_notional = self.cfg.max_order_pct * equity
            if notional > max_notional:
                logger.warning(
                    "Order for %s clamped from $%.0f to $%.0f (max_order_pct=%.0f%%)",
                    ticker, notional, max_notional, self.cfg.max_order_pct * 100,
                )
                notional = max_notional

            if notional < _MIN_ORDER_NOTIONAL:
                continue

            side = "buy" if delta_w > 0 else "sell"
            orders.append(Order(ticker=ticker, side=side, notional=notional))

        total_notional = sum(o.notional for o in orders)
        return orders, total_notional

    def execute(
        self,
        target_weights: np.ndarray,
        tickers: List[str],
    ) -> ExecutionReport:
        """
        Full execution cycle: compute orders → safety check → submit → log.

        Args:
            target_weights: [N_stocks] MVO portfolio weights
            tickers:        Stock universe matching target_weights order

        Returns:
            ExecutionReport with submission results
        """
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        submitted: List[Order] = []
        failed: List[Dict] = []
        error_msg: Optional[str] = None

        # Safety: kill switch
        if self.cfg.kill_switch:
            logger.warning("KILL SWITCH ACTIVE — no orders submitted")
            return ExecutionReport(
                timestamp=ts,
                account_equity=0.0,
                orders_submitted=[],
                orders_failed=[],
                total_notional_traded=0.0,
                estimated_cost_bps=0.0,
                error="kill_switch_active",
            )

        equity = self.get_account_equity()

        # Safety: daily loss limit
        if self._daily_loss_exceeded(equity):
            logger.error(
                "Daily loss limit exceeded (equity=$%.0f). No orders submitted.", equity
            )
            return ExecutionReport(
                timestamp=ts,
                account_equity=equity,
                orders_submitted=[],
                orders_failed=[],
                total_notional_traded=0.0,
                estimated_cost_bps=0.0,
                error="daily_loss_limit",
            )

        orders, total_notional = self.compute_orders(target_weights, tickers, equity)
        logger.info(
            "Submitting %d orders, total notional $%.0f (equity $%.0f)",
            len(orders), total_notional, equity,
        )

        api = self._get_api()
        for order in orders:
            try:
                alpaca_order = api.submit_order(
                    symbol=order.ticker,
                    side=order.side,
                    type="market",
                    time_in_force="day",
                    notional=round(order.notional, 2),
                )
                order.order_id = alpaca_order.id
                order.status = "submitted"
                submitted.append(order)
                logger.debug(
                    "Submitted %s %s $%.2f (id=%s)",
                    order.side.upper(), order.ticker, order.notional, order.order_id,
                )
            except Exception as e:
                logger.error("Order failed %s %s: %s", order.side, order.ticker, e)
                failed.append({
                    "ticker": order.ticker,
                    "side": order.side,
                    "notional": order.notional,
                    "error": str(e),
                })

        # Estimated TCA at 5 bps/trade
        estimated_cost = (total_notional * 5 / 10_000) if equity > 0 else 0.0

        report = ExecutionReport(
            timestamp=ts,
            account_equity=equity,
            orders_submitted=submitted,
            orders_failed=failed,
            total_notional_traded=total_notional,
            estimated_cost_bps=5.0 * len(submitted),
            error=error_msg,
        )

        if submitted:
            self._poll_fills(submitted)

        return report

    def cancel_all_orders(self):
        """Cancel all open orders (safety utility)."""
        api = self._get_api()
        api.cancel_all_orders()
        logger.info("All open orders cancelled")

    def flatten_all_positions(self):
        """Close all open positions immediately (emergency exit)."""
        api = self._get_api()
        api.close_all_positions()
        logger.warning("All positions closed via flatten_all_positions()")

    def is_market_open(self) -> bool:
        """Return True if the US equity market is currently open."""
        api = self._get_api()
        clock = api.get_clock()
        return bool(clock.is_open)

    def next_market_open(self) -> datetime:
        """Return the UTC datetime of the next market open."""
        api = self._get_api()
        clock = api.get_clock()
        return clock.next_open.replace(tzinfo=None)

    def next_market_close(self) -> datetime:
        """Return the UTC datetime of the next market close."""
        api = self._get_api()
        clock = api.get_clock()
        return clock.next_close.replace(tzinfo=None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_api(self):
        """Lazy-initialize the Alpaca REST client."""
        if self._api is not None:
            return self._api
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError(
                "alpaca-trade-api is required for live trading.\n"
                "Install via: pip install alpaca-trade-api"
            )
        self._api = tradeapi.REST(
            key_id=self.cfg.alpaca_api_key,
            secret_key=self.cfg.alpaca_secret_key,
            base_url=self.cfg.alpaca_base_url,
            api_version="v2",
        )
        # Validate credentials on first connect
        account = self._api.get_account()
        logger.info(
            "Connected to Alpaca: account_id=%s, status=%s, equity=$%.2f",
            account.id, account.status, float(account.equity),
        )
        return self._api

    def _daily_loss_exceeded(self, current_equity: float) -> bool:
        """Check if today's P&L has breached the daily loss limit."""
        try:
            api = self._get_api()
            # Use Alpaca's portfolio history for today
            history = api.get_portfolio_history(
                period="1D",
                timeframe="1H",
                extended_hours=False,
            )
            if history.profit_loss and len(history.profit_loss) > 0:
                daily_pnl = float(history.profit_loss[-1])
                limit = -abs(self.cfg.daily_loss_limit * current_equity)
                if daily_pnl < limit:
                    logger.error(
                        "Daily P&L $%.2f < limit $%.2f", daily_pnl, limit
                    )
                    return True
        except Exception as e:
            logger.debug("Could not check daily loss: %s", e)
        return False

    def _poll_fills(self, orders: List[Order]):
        """Poll until all submitted orders are filled or timeout."""
        pending_ids = {o.order_id: o for o in orders if o.order_id}
        if not pending_ids:
            return

        api = self._get_api()
        for attempt in range(_FILL_POLL_RETRIES):
            time.sleep(_FILL_POLL_INTERVAL_S)
            still_pending = {}
            for oid, order in pending_ids.items():
                try:
                    ao = api.get_order(oid)
                    if ao.status in ("filled", "partially_filled"):
                        order.status = ao.status
                        order.filled_qty = float(ao.filled_qty or 0)
                        order.filled_avg_price = float(ao.filled_avg_price or 0)
                        logger.debug(
                            "Filled %s %s: qty=%.4f @ $%.2f",
                            order.ticker, order.side,
                            order.filled_qty, order.filled_avg_price,
                        )
                    elif ao.status in ("canceled", "expired", "rejected"):
                        order.status = ao.status
                        logger.warning("Order %s %s: %s", order.ticker, oid, ao.status)
                    else:
                        still_pending[oid] = order
                except Exception as e:
                    logger.debug("Error polling order %s: %s", oid, e)
                    still_pending[oid] = order
            pending_ids = still_pending
            if not pending_ids:
                logger.info("All orders filled after %d poll(s)", attempt + 1)
                return

        # Timeout — mark remaining as pending
        for order in pending_ids.values():
            logger.warning("Order %s still pending after timeout", order.ticker)
