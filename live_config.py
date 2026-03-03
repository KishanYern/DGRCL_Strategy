"""
Configuration for DGRCL Live Paper Trading

All settings are read from environment variables (with sensible defaults).
Create a .env file in the project root for local development:

    ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
    ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ALPACA_BASE_URL=https://paper-api.alpaca.markets

Never commit the .env file to version control.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

# Load .env if python-dotenv is installed (soft dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LiveConfig:
    """
    All configuration for a live trading run.

    Instantiate via load_config() rather than directly — that function
    resolves environment variables and validates required fields.
    """

    # --- Alpaca credentials ---
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # --- Model / checkpoint ---
    checkpoint_path: str = "./checkpoints/latest.pt"
    window_size: int = 60           # WINDOW_SIZE from training (must match)
    mc_dropout_samples: int = 10    # MC Dropout passes per inference run

    # --- Portfolio optimizer ---
    portfolio_method: str = "mvo"   # "mvo" | "naive"
    portfolio_gamma: float = 1.0    # MVO risk aversion
    max_leverage: float = 2.0       # gross leverage cap
    max_position: float = 0.05      # per-stock position limit (5%)
    min_tradable: int = 20          # min universe after conformal gating

    # --- Execution ---
    min_weight_change: float = 0.005    # ignore weight deltas < 0.5%
    max_order_pct: float = 0.10         # max single order as % of equity (10%)
    daily_loss_limit: float = 0.05      # halt if daily P&L < -5% of equity
    kill_switch: bool = False           # set True to prevent all order submission

    # --- Scheduling ---
    # live_trader.py daemon mode: run inference daily after market close.
    # Format: "HH:MM" in US/Eastern time.  18:00 ET = ~2h after 4pm close.
    run_time_et: str = "18:00"

    # --- Output ---
    results_dir: str = "./live_results"
    log_level: str = "INFO"

    # --- Computed ---
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def validate(self):
        """Raise ValueError if any required field is missing."""
        if not self.alpaca_api_key:
            raise ValueError(
                "ALPACA_API_KEY is not set. "
                "Export it as an environment variable or add it to .env"
            )
        if not self.alpaca_secret_key:
            raise ValueError(
                "ALPACA_SECRET_KEY is not set. "
                "Export it as an environment variable or add it to .env"
            )
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Run train.py with --save-checkpoint first."
            )


def load_config(**overrides) -> LiveConfig:
    """
    Build a LiveConfig from environment variables + optional keyword overrides.

    Environment variables:
        ALPACA_API_KEY          Alpaca key ID
        ALPACA_SECRET_KEY       Alpaca secret key
        ALPACA_BASE_URL         API base URL (defaults to paper trading)
        DGRCL_CHECKPOINT_PATH   Path to .pt checkpoint file
        DGRCL_RESULTS_DIR       Directory for daily JSON logs
        DGRCL_PORTFOLIO_METHOD  mvo | naive (default: mvo)
        DGRCL_PORTFOLIO_GAMMA   Float (default: 1.0)
        DGRCL_MAX_LEVERAGE      Float (default: 2.0)
        DGRCL_MAX_POSITION      Float (default: 0.05)
        DGRCL_MIN_TRADABLE      Int (default: 20)
        DGRCL_KILL_SWITCH       "true" to disable trading
        DGRCL_LOG_LEVEL         DEBUG | INFO | WARNING (default: INFO)
        DGRCL_MC_SAMPLES        Int MC Dropout passes (default: 10)

    Example:
        export ALPACA_API_KEY=PKxxx
        export ALPACA_SECRET_KEY=xxx
        python live_trader.py --once
    """
    # Resolve device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cfg = LiveConfig(
        alpaca_api_key=os.environ.get("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
        alpaca_base_url=os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        checkpoint_path=os.environ.get("DGRCL_CHECKPOINT_PATH", "./checkpoints/latest.pt"),
        results_dir=os.environ.get("DGRCL_RESULTS_DIR", "./live_results"),
        portfolio_method=os.environ.get("DGRCL_PORTFOLIO_METHOD", "mvo"),
        portfolio_gamma=float(os.environ.get("DGRCL_PORTFOLIO_GAMMA", "1.0")),
        max_leverage=float(os.environ.get("DGRCL_MAX_LEVERAGE", "2.0")),
        max_position=float(os.environ.get("DGRCL_MAX_POSITION", "0.05")),
        min_tradable=int(os.environ.get("DGRCL_MIN_TRADABLE", "20")),
        kill_switch=os.environ.get("DGRCL_KILL_SWITCH", "").lower() == "true",
        log_level=os.environ.get("DGRCL_LOG_LEVEL", "INFO"),
        mc_dropout_samples=int(os.environ.get("DGRCL_MC_SAMPLES", "10")),
        device=device,
    )

    # Apply any programmatic overrides
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise ValueError(f"Unknown LiveConfig field: {k}")

    return cfg
