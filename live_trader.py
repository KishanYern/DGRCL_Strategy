"""
DGRCL Live Paper Trading Orchestrator

Two modes:

  --once      Run a single inference + trade cycle and exit.
              Useful for manual triggers or cron jobs.

  --daemon    Run on a schedule (default: daily at 18:00 ET after market close).
              Requires APScheduler: pip install APScheduler

  --dry-run   Run inference and log signals, but do NOT submit orders to Alpaca.
              Useful for monitoring signals before going live.

  --flatten   Close all open positions immediately and exit (emergency use).

Usage:
    # Set credentials in .env or as env vars first, then:

    # One-shot (dry run — no real orders):
    python live_trader.py --once --dry-run

    # One-shot (real paper orders):
    python live_trader.py --once

    # Daemon mode (runs every day at 18:00 ET):
    python live_trader.py --daemon

    # Emergency: close everything
    python live_trader.py --flatten
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_US_EASTERN = "US/Eastern"  # used for scheduling


# =============================================================================
# CORE CYCLE
# =============================================================================

def run_once(dry_run: bool = False, as_of: Optional[datetime] = None) -> dict:
    """
    Execute a single inference + trade cycle.

    Args:
        dry_run:  If True, compute signals but skip order submission.
        as_of:    Reference datetime for data download (default: now).

    Returns:
        Dict suitable for JSON logging.
    """
    from live_config import load_config
    from live_inference import InferenceEngine
    from alpaca_broker import AlpacaBroker

    cfg = load_config()
    cfg.validate()

    engine = InferenceEngine(cfg)
    broker = AlpacaBroker(cfg)

    # Check market status
    try:
        market_open = broker.is_market_open()
        if market_open:
            logger.warning(
                "Market is currently OPEN. Running inference during market hours "
                "may use stale closing prices. Consider waiting for market close."
            )
    except Exception as e:
        logger.debug("Could not check market status: %s", e)
        market_open = None

    # --- Run inference ---
    logger.info("Running inference...")
    result = engine.run(as_of=as_of)
    logger.info("\n%s", result.summary())

    report = {
        "cycle_timestamp": result.timestamp,
        "dry_run": dry_run,
        "market_open_at_run": market_open,
        "signal": result.to_dict(),
    }

    # --- Execute orders ---
    if not result.has_trades:
        logger.info("No confident signals — holding current positions")
        report["execution"] = {"status": "no_signal", "orders_submitted": 0}
        _save_log(report, cfg.results_dir)
        return report

    if dry_run:
        logger.info("DRY RUN — signals computed but NO orders submitted")
        report["execution"] = {
            "status": "dry_run",
            "orders_submitted": 0,
            "top_longs": result.top_longs,
            "top_shorts": result.top_shorts,
        }
        _save_log(report, cfg.results_dir)
        return report

    logger.info("Submitting orders via Alpaca paper trading...")
    exec_report = broker.execute(
        target_weights=result.target_weights,
        tickers=result.tickers,
    )

    report["execution"] = {
        "status": "submitted" if not exec_report.error else exec_report.error,
        "account_equity": exec_report.account_equity,
        "orders_submitted": len(exec_report.orders_submitted),
        "orders_failed": len(exec_report.orders_failed),
        "total_notional_traded": round(exec_report.total_notional_traded, 2),
        "estimated_cost_bps": exec_report.estimated_cost_bps,
        "failed_details": exec_report.orders_failed,
        "fills": [
            {
                "ticker": o.ticker,
                "side": o.side,
                "notional": round(o.notional, 2),
                "status": o.status,
                "filled_qty": round(o.filled_qty, 6),
                "filled_avg_price": round(o.filled_avg_price, 4),
            }
            for o in exec_report.orders_submitted
        ],
    }

    logger.info(
        "Execution: %d orders submitted, $%.0f notional traded",
        len(exec_report.orders_submitted),
        exec_report.total_notional_traded,
    )
    if exec_report.orders_failed:
        logger.error("%d orders failed", len(exec_report.orders_failed))

    _save_log(report, cfg.results_dir)
    return report


# =============================================================================
# DAEMON MODE
# =============================================================================

def run_daemon(dry_run: bool = False):
    """
    Run inference daily at the configured time (default 18:00 ET).
    Uses APScheduler for reliable scheduling.
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print(
            "APScheduler is required for daemon mode.\n"
            "Install via: pip install APScheduler"
        )
        sys.exit(1)

    from live_config import load_config
    cfg = load_config()
    cfg.validate()

    hour, minute = cfg.run_time_et.split(":")
    logger.info(
        "Daemon starting — will run daily at %s ET (dry_run=%s)",
        cfg.run_time_et, dry_run,
    )

    scheduler = BlockingScheduler()
    scheduler.add_job(
        func=lambda: run_once(dry_run=dry_run),
        trigger=CronTrigger(
            hour=int(hour),
            minute=int(minute),
            timezone=_US_EASTERN,
        ),
        id="dgrcl_daily_inference",
        name="DGRCL Daily Inference",
        replace_existing=True,
        misfire_grace_time=3600,  # allow up to 1h late execution
    )

    # Also run immediately on startup so first-run feedback is instant
    logger.info("Running initial cycle on daemon startup...")
    try:
        run_once(dry_run=dry_run)
    except Exception as e:
        logger.error("Initial cycle failed: %s", e)

    logger.info("Scheduler running. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Daemon stopped.")


# =============================================================================
# LOG UTILITY
# =============================================================================

def _save_log(report: dict, results_dir: str):
    """Save cycle report as a timestamped JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    ts = report.get("cycle_timestamp", datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    safe_ts = ts.replace(":", "").replace("-", "")
    path = os.path.join(results_dir, f"{safe_ts}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Cycle log saved: %s", path)

    # Also maintain a rolling summary (last 90 cycles)
    _update_summary(report, results_dir)


def _update_summary(report: dict, results_dir: str):
    """Append this cycle to a rolling summary JSON array."""
    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                history = json.load(f)
        except Exception:
            history = []
    else:
        history = []

    history.append({
        "timestamp": report.get("cycle_timestamp"),
        "has_trades": report.get("signal", {}).get("has_trades", False),
        "n_trade": report.get("signal", {}).get("n_trade", 0),
        "gated_pct": report.get("signal", {}).get("gated_pct", 0),
        "gross_leverage": report.get("signal", {}).get("gross_leverage", 0),
        "rank_stability": report.get("signal", {}).get("rank_stability", 0),
        "top_longs": report.get("signal", {}).get("top_longs", []),
        "top_shorts": report.get("signal", {}).get("top_shorts", []),
        "orders_submitted": report.get("execution", {}).get("orders_submitted", 0),
        "total_notional": report.get("execution", {}).get("total_notional_traded", 0),
        "dry_run": report.get("dry_run", True),
    })

    # Keep most recent 90 entries
    history = history[-90:]
    with open(summary_path, "w") as f:
        json.dump(history, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DGRCL Live Paper Trading — run inference and execute via Alpaca",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Run a single inference + trade cycle and exit",
    )
    mode_group.add_argument(
        "--daemon",
        action="store_true",
        help="Run daily at the configured time (requires APScheduler)",
    )
    mode_group.add_argument(
        "--flatten",
        action="store_true",
        help="Close all open positions immediately and exit (emergency)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals but do NOT submit orders (safe for testing)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (default: from DGRCL_CHECKPOINT_PATH or ./checkpoints/latest.pt)",
    )

    args = parser.parse_args()

    # Override checkpoint path if provided
    if args.checkpoint:
        os.environ["DGRCL_CHECKPOINT_PATH"] = args.checkpoint

    if args.flatten:
        from live_config import load_config
        from alpaca_broker import AlpacaBroker
        cfg = load_config()
        cfg.validate()
        broker = AlpacaBroker(cfg)
        broker.flatten_all_positions()
        print("All positions closed.")
        sys.exit(0)

    if args.once:
        report = run_once(dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
        sys.exit(0)

    if args.daemon:
        run_daemon(dry_run=args.dry_run)
