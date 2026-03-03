"""
Live Inference Engine for DGRCL Paper Trading

Loads a saved checkpoint bundle, fetches the latest market data, runs the
forward pass + conformal gate + MVO optimizer, and returns a SignalResult
with per-stock target weights and full diagnostics.

Usage:
    from live_inference import InferenceEngine
    from live_config import load_config

    cfg = load_config()
    engine = InferenceEngine(cfg)
    result = engine.run()

    if result.has_trades:
        print(result.summary())
        # Pass result.target_weights to alpaca_broker.py
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

from checkpoint import CheckpointManager, CheckpointBundle
from live_data import LiveDataFeed
from portfolio_optimizer import (
    ConformalGate,
    CovarianceEstimator,
    MeanVarianceOptimizer,
    ExpectedReturnEstimator,
    ConformalPredictor,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SIGNAL RESULT
# =============================================================================

@dataclass
class SignalResult:
    """
    Output of a single inference run.  Contains target weights + diagnostics.
    """
    timestamp: str
    tickers: List[str]

    # Target portfolio weights — positive = long, negative = short, 0 = no position.
    # Dollar-neutral (sum ≈ 0), gross leverage ≤ max_leverage.
    target_weights: np.ndarray          # [N_stocks]

    # Per-stock signals
    dir_logits: np.ndarray              # [N_stocks] raw direction logits
    mag_preds: np.ndarray               # [N_stocks] magnitude predictions
    expected_returns: np.ndarray        # [N_stocks] mu = (2P_up - 1) * |mag|
    confidence: np.ndarray             # [N_stocks] MC Dropout confidence [0,1]

    # Gate diagnostics
    active_mask: np.ndarray             # [N_stocks] tradable before gating
    trade_mask: np.ndarray              # [N_stocks] after conformal gate
    n_active: int
    n_trade: int
    gated_pct: float
    conformal_q_hat: Optional[float]
    gate_relaxed: int                   # stocks re-admitted by min_tradable guard

    # Optimizer info
    method: str
    optimizer_status: str               # "optimal" | "fallback" | "no_signal"

    # MC Dropout stability
    rank_stability: float               # Spearman ρ across passes

    # Human-readable top picks
    top_longs: List[str]                # top 5 long candidates by weight
    top_shorts: List[str]               # top 5 short candidates by weight

    @property
    def has_trades(self) -> bool:
        """True if the gate passed enough stocks to build a non-trivial portfolio."""
        return self.n_trade >= 4 and np.abs(self.target_weights).sum() > 1e-6

    def summary(self) -> str:
        lines = [
            f"=== DGRCL Signal @ {self.timestamp} ===",
            f"  Universe: {self.n_active} active  |  "
            f"Gate: {self.n_trade} tradable ({self.gated_pct*100:.1f}% filtered)",
            f"  Method: {self.method}  |  Status: {self.optimizer_status}",
            f"  Rank stability (MC): {self.rank_stability:.3f}",
            f"  Gross leverage: {np.abs(self.target_weights).sum():.3f}",
            f"  Top longs:  {', '.join(self.top_longs) if self.top_longs else '—'}",
            f"  Top shorts: {', '.join(self.top_shorts) if self.top_shorts else '—'}",
        ]
        if not self.has_trades:
            lines.append("  [NO TRADES — gate filtered all stocks or optimizer failed]")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "has_trades": self.has_trades,
            "n_active": self.n_active,
            "n_trade": self.n_trade,
            "gated_pct": round(self.gated_pct, 4),
            "gate_relaxed": self.gate_relaxed,
            "conformal_q_hat": self.conformal_q_hat,
            "method": self.method,
            "optimizer_status": self.optimizer_status,
            "rank_stability": round(float(self.rank_stability), 4),
            "gross_leverage": round(float(np.abs(self.target_weights).sum()), 4),
            "top_longs": self.top_longs,
            "top_shorts": self.top_shorts,
            "weights": {
                t: round(float(w), 6)
                for t, w in zip(self.tickers, self.target_weights)
                if abs(w) > 1e-6
            },
        }


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """
    Orchestrates a single live inference run.

    Steps:
        1. Load checkpoint (model weights, conformal gate, covariance)
        2. Fetch live snapshot from yfinance via LiveDataFeed
        3. Run MC Dropout forward passes → dir_logits, mag_preds, confidence
        4. Apply ConformalGate → trade_mask
        5. Run MVO optimizer → target_weights
        6. Return SignalResult
    """

    def __init__(self, config):
        """
        Args:
            config: LiveConfig object (from live_config.py)
        """
        self.cfg = config
        self._bundle: Optional[CheckpointBundle] = None
        self._model = None
        self._gate: Optional[ConformalGate] = None
        self._cov: Optional[np.ndarray] = None
        self._mvo: Optional[MeanVarianceOptimizer] = None
        self._mu_estimator = ExpectedReturnEstimator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, as_of: Optional[datetime] = None) -> SignalResult:
        """
        Run a full inference cycle and return a SignalResult.

        Args:
            as_of:  Reference datetime for data download (default: now).
        """
        as_of = as_of or datetime.utcnow()
        ts = as_of.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("Starting inference run at %s", ts)

        # 1. Load checkpoint (cached after first call)
        bundle = self._get_bundle()

        # 2. Fetch live snapshot
        logger.info("Fetching live data for %d tickers...", len(bundle.tickers))
        feed = LiveDataFeed(
            tickers=bundle.tickers,
            feature_indices=bundle.feature_indices,
            warmup_days=120,
            device=self.cfg.device,
        )
        stock_t, macro_t, active_t = feed.get_snapshot(
            lookback=self.cfg.window_size,
            as_of=as_of,
        )
        logger.info(
            "Snapshot: %d stocks × %d timesteps, %d active",
            stock_t.shape[0], stock_t.shape[1], active_t[:, -1].sum().item()
        )

        # 3. Run MC Dropout inference on last timestep slice [N, 1, F] → [N, F]
        model = self._get_model()
        last_stock = stock_t[:, -self.cfg.window_size:, :]
        last_macro = macro_t[:, -self.cfg.window_size:, :]
        active_last = active_t[:, -1]  # [N_stocks] bool

        mc_out = self._mc_inference(model, last_stock, last_macro)
        dir_logits_np = mc_out["dir_score_mean"].cpu().numpy()
        mag_preds_np = mc_out["mag_mean"].cpu().numpy()
        confidence_np = mc_out["confidence"].cpu().numpy()
        rank_stability = float(mc_out["rank_stability"].item())
        active_np = active_last.cpu().numpy().astype(bool)

        # 4. Apply conformal gate
        gate = self._get_gate(bundle)
        if gate._calibrated:
            trade_mask_np, gate_stats = gate.gate(dir_logits_np, active_np)
        else:
            logger.warning("Conformal gate not calibrated — using active mask directly")
            trade_mask_np = active_np.copy()
            gate_stats = {
                "n_active": int(active_np.sum()),
                "n_trade": int(active_np.sum()),
                "n_gated": 0,
                "gated_pct": 0.0,
                "relaxed": 0,
                "q_hat": None,
            }

        n_active = gate_stats.get("n_active", int(active_np.sum()))
        n_trade = gate_stats.get("n_trade", int(trade_mask_np.sum()))
        gated_pct = gate_stats.get("gated_pct", 0.0)
        n_relaxed = gate_stats.get("relaxed", 0)
        q_hat = gate_stats.get("q_hat")

        logger.info(
            "Gate: %d active → %d tradable (%.1f%% filtered, %d relaxed)",
            n_active, n_trade, gated_pct * 100, n_relaxed,
        )

        # 5. Run optimizer
        mu = self._mu_estimator.compute(dir_logits_np, mag_preds_np, trade_mask_np)
        target_weights, optimizer_status = self._optimize(
            mu, trade_mask_np, bundle
        )

        # 6. Build result
        result = self._build_result(
            ts=ts,
            tickers=bundle.tickers,
            target_weights=target_weights,
            dir_logits_np=dir_logits_np,
            mag_preds_np=mag_preds_np,
            mu=mu,
            confidence_np=confidence_np,
            active_np=active_np,
            trade_mask_np=trade_mask_np,
            n_active=n_active,
            n_trade=n_trade,
            gated_pct=gated_pct,
            q_hat=q_hat,
            n_relaxed=n_relaxed,
            rank_stability=rank_stability,
            optimizer_status=optimizer_status,
        )

        logger.info("Inference complete. has_trades=%s", result.has_trades)
        return result

    def reload_checkpoint(self):
        """Force reload of checkpoint from disk (call after retraining)."""
        self._bundle = None
        self._model = None
        self._gate = None
        self._cov = None
        self._mvo = None
        logger.info("Checkpoint cache cleared — will reload on next run()")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_bundle(self) -> CheckpointBundle:
        if self._bundle is None:
            mgr = CheckpointManager(self.cfg.checkpoint_path)
            self._bundle = mgr.load()
        return self._bundle

    def _get_model(self):
        if self._model is None:
            bundle = self._get_bundle()
            self._model = bundle.build_model(self.cfg.device)
            self._model.eval()
        return self._model

    def _get_gate(self, bundle: CheckpointBundle) -> ConformalGate:
        if self._gate is None:
            gate = ConformalGate(
                alpha=bundle.conformal_alpha,
                min_tradable=self.cfg.min_tradable,
            )
            if bundle.conformal_calibrated and bundle.conformal_q_hat is not None:
                # Inject saved q_hat directly into the predictor
                gate._predictor._q_hat = bundle.conformal_q_hat
                gate._calibrated = True
            self._gate = gate
        return self._gate

    def _mc_inference(
        self, model, stock_t: torch.Tensor, macro_t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run MC Dropout inference — imported from train.py logic."""
        n_samples = self.cfg.mc_dropout_samples

        model.train()  # keep dropout active
        dir_scores_all = []
        mag_preds_all = []

        with torch.no_grad():
            for _ in range(n_samples):
                dir_logits, mag_out = model(
                    stock_features=stock_t,
                    macro_features=macro_t,
                )
                dir_scores_all.append(dir_logits.squeeze(-1))
                mag_preds_all.append(mag_out.squeeze(-1))

        model.eval()

        dir_stack = torch.stack(dir_scores_all, dim=0)   # [n, N]
        mag_stack = torch.stack(mag_preds_all, dim=0)    # [n, N]

        dir_mean = dir_stack.mean(0)
        dir_std = dir_stack.std(0)
        confidence = 1.0 / (1.0 + dir_std)

        # Median rank across passes
        ranks_per_pass = torch.stack(
            [s.argsort().argsort().float() for s in dir_scores_all], dim=0
        )
        median_rank = ranks_per_pass.median(dim=0).values

        # Rank stability: mean Spearman ρ between pass 0 and passes 1..n
        rank_stab = torch.tensor(1.0)
        if n_samples > 1:
            base_rank = ranks_per_pass[0]
            rhos = []
            for i in range(1, n_samples):
                r = ranks_per_pass[i]
                d_sq = ((base_rank - r) ** 2).sum()
                n = float(r.numel())
                rho = 1.0 - 6.0 * float(d_sq) / max(n * (n * n - 1), 1.0)
                rhos.append(rho)
            rank_stab = torch.tensor(float(np.mean(rhos)))

        return {
            "dir_score_mean": dir_mean,
            "dir_score_std": dir_std,
            "median_dir_rank": median_rank,
            "mag_mean": mag_stack.mean(0),
            "mag_std": mag_stack.std(0),
            "confidence": confidence,
            "rank_stability": rank_stab,
        }

    def _optimize(
        self,
        mu: np.ndarray,
        trade_mask: np.ndarray,
        bundle: CheckpointBundle,
    ) -> tuple:
        """Run MVO on the trade universe. Returns (weights, status_string)."""
        if trade_mask.sum() < 4:
            return np.zeros(len(mu)), "no_signal"

        # Covariance
        sigma = bundle.covariance_sigma
        if sigma is None:
            # Fallback: identity covariance (equal-weight in Sigma)
            N = len(mu)
            sigma = np.eye(N) * 1e-2

        if self._mvo is None:
            self._mvo = MeanVarianceOptimizer(
                gamma=self.cfg.portfolio_gamma,
                max_leverage=self.cfg.max_leverage,
                max_position=self.cfg.max_position,
            )

        try:
            w = self._mvo.optimize(mu, sigma, trade_mask)
            status = "optimal"
        except Exception as e:
            logger.warning("MVO failed (%s), falling back to equal-weight L/S", e)
            w = self._equal_ls_fallback(mu, trade_mask)
            status = "fallback"

        return w, status

    @staticmethod
    def _equal_ls_fallback(mu: np.ndarray, trade_mask: np.ndarray) -> np.ndarray:
        """Top-20% long / bottom-20% short equal-weight fallback."""
        N = len(mu)
        active_idx = np.where(trade_mask)[0]
        if len(active_idx) < 2:
            return np.zeros(N)
        k = max(1, int(0.20 * len(active_idx)))
        sorted_i = np.argsort(mu[active_idx])
        w = np.zeros(N)
        w[active_idx[sorted_i[-k:]]] = 1.0 / k
        w[active_idx[sorted_i[:k]]] = -1.0 / k
        return w

    def _build_result(self, **kwargs) -> SignalResult:
        tickers = kwargs["tickers"]
        target_weights = kwargs["target_weights"]

        # Top 5 longs and shorts by absolute weight
        long_idx = np.where(target_weights > 1e-6)[0]
        short_idx = np.where(target_weights < -1e-6)[0]
        top_longs = [
            tickers[i] for i in sorted(long_idx, key=lambda i: -target_weights[i])[:5]
        ]
        top_shorts = [
            tickers[i] for i in sorted(short_idx, key=lambda i: target_weights[i])[:5]
        ]

        return SignalResult(
            timestamp=kwargs["ts"],
            tickers=tickers,
            target_weights=target_weights,
            dir_logits=kwargs["dir_logits_np"],
            mag_preds=kwargs["mag_preds_np"],
            expected_returns=kwargs["mu"],
            confidence=kwargs["confidence_np"],
            active_mask=kwargs["active_np"],
            trade_mask=kwargs["trade_mask_np"],
            n_active=kwargs["n_active"],
            n_trade=kwargs["n_trade"],
            gated_pct=kwargs["gated_pct"],
            conformal_q_hat=kwargs["q_hat"],
            gate_relaxed=kwargs["n_relaxed"],
            method=self.cfg.portfolio_method,
            optimizer_status=kwargs["optimizer_status"],
            rank_stability=kwargs["rank_stability"],
            top_longs=top_longs,
            top_shorts=top_shorts,
        )
