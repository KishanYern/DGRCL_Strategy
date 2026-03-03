"""
Unit tests for portfolio_optimizer.py (Phase 1 Portfolio Construction)

Tests cover:
  - ExpectedReturnEstimator: sign and magnitude of mu
  - CovarianceEstimator: PSD guarantee, active-mask exclusion
  - MeanVarianceOptimizer: dollar-neutral, leverage, position constraints
  - GraphClusterer: eigengap heuristic, sector fallback
  - RiskParityOptimizer: ERC convergence, dollar-neutral output
  - ConformalGate: set_size gating logic, calibration requirement
  - PortfolioConstructor: end-to-end pipeline on synthetic data
  - compute_portfolio_metrics: Sharpe / drawdown / turnover correctness
"""

import math
import warnings
import numpy as np
import pytest

from portfolio_optimizer import (
    ExpectedReturnEstimator,
    CovarianceEstimator,
    MeanVarianceOptimizer,
    GraphClusterer,
    RiskParityOptimizer,
    ConformalGate,
    PortfolioConstructor,
    compute_portfolio_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def make_active_mask(N: int, pct_active: float = 0.80) -> np.ndarray:
    mask = RNG.random(N) < pct_active
    if not mask.any():
        mask[0] = True
    return mask


def make_returns(N: int, T: int) -> np.ndarray:
    return RNG.normal(0, 0.01, size=(N, T)).astype(np.float32)


def make_logits(N: int) -> np.ndarray:
    return RNG.normal(0, 1, size=N).astype(np.float32)


def make_mag(N: int) -> np.ndarray:
    return np.abs(RNG.normal(0, 0.01, size=N)).astype(np.float32)


# =============================================================================
# ExpectedReturnEstimator
# =============================================================================

class TestExpectedReturnEstimator:
    def setup_method(self):
        self.est = ExpectedReturnEstimator()

    def test_positive_logit_gives_positive_mu(self):
        N = 10
        logits = np.ones(N) * 5.0   # very confident UP
        mag = np.ones(N) * 0.01
        mask = np.ones(N, dtype=bool)
        mu = self.est.compute(logits, mag, mask)
        assert (mu > 0).all(), "Strongly positive logits should yield positive mu"

    def test_negative_logit_gives_negative_mu(self):
        N = 10
        logits = np.ones(N) * -5.0  # very confident DOWN
        mag = np.ones(N) * 0.01
        mask = np.ones(N, dtype=bool)
        mu = self.est.compute(logits, mag, mask)
        assert (mu < 0).all(), "Strongly negative logits should yield negative mu"

    def test_zero_logit_zero_mu(self):
        N = 5
        logits = np.zeros(N)
        mag = np.ones(N) * 0.01
        mask = np.ones(N, dtype=bool)
        mu = self.est.compute(logits, mag, mask)
        np.testing.assert_allclose(mu, 0.0, atol=1e-6)

    def test_inactive_stocks_zeroed(self):
        N = 10
        logits = make_logits(N)
        mag = make_mag(N)
        mask = np.zeros(N, dtype=bool)
        mask[:5] = True
        mu = self.est.compute(logits, mag, mask)
        np.testing.assert_allclose(mu[5:], 0.0, atol=1e-8)

    def test_magnitude_scales_mu(self):
        logits = np.array([2.0, 2.0])
        mag = np.array([0.01, 0.02])
        mask = np.ones(2, dtype=bool)
        mu = self.est.compute(logits, mag, mask)
        assert mu[1] == pytest.approx(2 * mu[0], rel=1e-5), \
            "mu should scale linearly with magnitude"


# =============================================================================
# CovarianceEstimator
# =============================================================================

class TestCovarianceEstimator:
    def test_output_is_psd(self):
        N, T = 30, 60
        returns = make_returns(N, T)
        active = make_active_mask(N)
        est = CovarianceEstimator()
        est.fit(returns, active)
        sigma = est.sigma
        eigvals = np.linalg.eigvalsh(sigma)
        assert eigvals.min() >= 0, "Covariance matrix must be PSD"

    def test_inactive_stocks_get_eps_diagonal(self):
        N, T = 20, 50
        returns = make_returns(N, T)
        active = np.zeros(N, dtype=bool)
        active[:10] = True
        eps = 1e-4
        est = CovarianceEstimator(eps=eps)
        est.fit(returns, active)
        sigma = est.sigma
        # Inactive stocks (10:) should have diagonal ≥ eps and off-diagonal ~ 0.
        # PSD flooring may shift the diagonal slightly above eps, so we only
        # assert a lower bound and a reasonable upper bound (e.g., 10×eps).
        inactive_diag = np.diag(sigma)[10:]
        assert (inactive_diag >= eps).all(), "Inactive diagonals must be >= eps"
        assert (inactive_diag <= 10 * eps).all(), \
            f"Inactive diagonals {inactive_diag} are much larger than eps={eps}"

    def test_shape(self):
        N, T = 15, 40
        returns = make_returns(N, T)
        est = CovarianceEstimator()
        est.fit(returns)
        assert est.sigma.shape == (N, N)

    def test_symmetric(self):
        N, T = 20, 60
        returns = make_returns(N, T)
        est = CovarianceEstimator()
        est.fit(returns)
        np.testing.assert_allclose(est.sigma, est.sigma.T, atol=1e-10)

    def test_raises_before_fit(self):
        est = CovarianceEstimator()
        with pytest.raises(RuntimeError, match="fit"):
            _ = est.sigma

    def test_degenerate_few_active(self):
        N, T = 10, 50
        returns = make_returns(N, T)
        active = np.zeros(N, dtype=bool)
        active[0] = True  # Only 1 active stock → degenerate
        est = CovarianceEstimator(eps=1e-4)
        est.fit(returns, active)
        assert est.sigma.shape == (N, N), "Should still return [N, N] matrix"


# =============================================================================
# MeanVarianceOptimizer
# =============================================================================

class TestMeanVarianceOptimizer:
    def setup_method(self):
        self.N = 40
        self.active = make_active_mask(self.N, 0.70)
        self.mu = make_logits(self.N) * 0.01
        self.mu[~self.active] = 0.0

        returns = make_returns(self.N, 100)
        est = CovarianceEstimator()
        est.fit(returns, self.active)
        self.sigma = est.sigma

        self.mvo = MeanVarianceOptimizer(gamma=1.0, max_leverage=2.0, max_position=0.05)

    def test_dollar_neutral(self):
        pytest.importorskip("cvxpy")
        w = self.mvo.optimize(self.mu, self.sigma, self.active)
        assert abs(w.sum()) < 1e-4, f"Portfolio must be dollar-neutral, got sum={w.sum():.6f}"

    def test_inactive_zero(self):
        pytest.importorskip("cvxpy")
        w = self.mvo.optimize(self.mu, self.sigma, self.active)
        np.testing.assert_allclose(w[~self.active], 0.0, atol=1e-6)

    def test_gross_leverage_cap(self):
        pytest.importorskip("cvxpy")
        w = self.mvo.optimize(self.mu, self.sigma, self.active)
        assert np.abs(w).sum() <= 2.0 + 1e-4, \
            f"Gross leverage {np.abs(w).sum():.4f} exceeds 2.0 cap"

    def test_position_limits(self):
        pytest.importorskip("cvxpy")
        w = self.mvo.optimize(self.mu, self.sigma, self.active)
        assert np.abs(w).max() <= 0.05 + 1e-4, \
            f"Max position {np.abs(w).max():.4f} exceeds 5% cap"

    def test_fallback_when_too_few_active(self):
        """With < 4 active stocks, optimizer uses equal-weight L/S fallback."""
        pytest.importorskip("cvxpy")
        active = np.zeros(self.N, dtype=bool)
        active[:3] = True
        w = self.mvo.optimize(self.mu, self.sigma, active)
        assert w.shape == (self.N,)


# =============================================================================
# GraphClusterer
# =============================================================================

class TestGraphClusterer:
    def _make_attention_graph(self, N: int, n_edges: int):
        src = RNG.integers(0, N, size=n_edges)
        dst = RNG.integers(0, N, size=n_edges)
        weights = np.abs(RNG.normal(0, 0.1, size=n_edges)).astype(np.float32)
        weights /= max(weights.max(), 1e-8)
        edge_index = np.stack([src, dst], axis=0)
        return edge_index, weights

    def test_cluster_ids_shape(self):
        pytest.importorskip("sklearn")
        N = 50
        active = make_active_mask(N)
        ei, ew = self._make_attention_graph(N, 200)
        gc = GraphClusterer(min_clusters=3, max_clusters=6)
        cluster_ids = gc.infer_clusters(ei, ew, N, active)
        assert cluster_ids.shape == (N,), "cluster_ids must have length N"

    def test_inactive_get_minus_one(self):
        pytest.importorskip("sklearn")
        N = 30
        active = np.ones(N, dtype=bool)
        active[15:] = False
        ei, ew = self._make_attention_graph(N, 100)
        gc = GraphClusterer()
        cluster_ids = gc.infer_clusters(ei, ew, N, active)
        assert (cluster_ids[15:] == -1).all(), "Inactive stocks must have cluster_id -1"

    def test_sector_fallback_used_when_too_few_active(self):
        N = 10
        active = np.zeros(N, dtype=bool)
        active[:2] = True   # Only 2 active — below min_clusters * 2 = 6
        sector_ids = np.arange(N) % 3
        ei, ew = self._make_attention_graph(N, 20)
        gc = GraphClusterer(min_clusters=3)
        cluster_ids = gc.infer_clusters(ei, ew, N, active, fallback_sector_ids=sector_ids)
        # Should use sector_ids for the active stocks
        for i in np.where(active)[0]:
            assert cluster_ids[i] == sector_ids[i]

    def test_eigengap_returns_int_in_range(self):
        pytest.importorskip("sklearn")
        N = 40
        affinity = np.abs(RNG.normal(0, 0.5, (N, N)).astype(np.float32))
        affinity = 0.5 * (affinity + affinity.T)
        gc = GraphClusterer(min_clusters=3, max_clusters=8)
        k = gc._eigengap_n_clusters(affinity)
        assert 3 <= k <= 8, f"Eigengap returned k={k} outside [3, 8]"


# =============================================================================
# RiskParityOptimizer
# =============================================================================

class TestRiskParityOptimizer:
    def test_dollar_neutral(self):
        N = 30
        active = make_active_mask(N, 0.8)
        mu = make_logits(N) * 0.01
        mu[~active] = 0.0
        returns = make_returns(N, 80)
        est = CovarianceEstimator()
        est.fit(returns, active)

        # 3 clusters of ~10 each
        cluster_ids = np.full(N, -1, dtype=int)
        active_idx = np.where(active)[0]
        for i, idx in enumerate(active_idx):
            cluster_ids[idx] = i % 3

        rpo = RiskParityOptimizer()
        w = rpo.optimize(mu, est.sigma, cluster_ids, active)
        assert abs(w.sum()) < 1e-5, f"Risk-parity portfolio must be dollar-neutral, got {w.sum():.6f}"

    def test_inactive_stocks_zero(self):
        N = 20
        active = make_active_mask(N, 0.6)
        mu = make_logits(N) * 0.01
        mu[~active] = 0.0
        returns = make_returns(N, 60)
        est = CovarianceEstimator()
        est.fit(returns, active)

        cluster_ids = np.full(N, -1, dtype=int)
        active_idx = np.where(active)[0]
        for i, idx in enumerate(active_idx):
            cluster_ids[idx] = i % 4

        rpo = RiskParityOptimizer()
        w = rpo.optimize(mu, est.sigma, cluster_ids, active)
        np.testing.assert_allclose(w[~active], 0.0, atol=1e-8)

    def test_gross_leverage_one(self):
        """Risk parity normalizes to gross leverage = 1."""
        N = 20
        active = np.ones(N, dtype=bool)
        mu = make_logits(N) * 0.01
        returns = make_returns(N, 60)
        est = CovarianceEstimator()
        est.fit(returns)
        cluster_ids = np.array([i % 3 for i in range(N)])

        rpo = RiskParityOptimizer()
        w = rpo.optimize(mu, est.sigma, cluster_ids, active)
        assert np.abs(w).sum() <= 1.0 + 1e-5, \
            f"Risk parity gross leverage {np.abs(w).sum():.4f} > 1"

    def test_single_cluster_fallback(self):
        """Degenerate: only one cluster → naive L/S fallback, still returns [N] array."""
        N = 20
        active = np.ones(N, dtype=bool)
        mu = make_logits(N) * 0.01
        cluster_ids = np.zeros(N, dtype=int)  # All in cluster 0
        returns = make_returns(N, 50)
        est = CovarianceEstimator()
        est.fit(returns)
        rpo = RiskParityOptimizer()
        w = rpo.optimize(mu, est.sigma, cluster_ids, active)
        assert w.shape == (N,)


# =============================================================================
# ConformalGate
# =============================================================================

class TestConformalGate:
    def _make_calibration_data(self, N: int = 200):
        logits = make_logits(N)
        labels = (make_logits(N) > 0).astype(float)
        return logits, labels

    def test_gate_returns_subset_of_active(self):
        gate = ConformalGate(alpha=0.10)
        logits_cal, labels_cal = self._make_calibration_data()
        gate.calibrate(logits_cal, labels_cal)

        N = 50
        logits_test = make_logits(N)
        active = make_active_mask(N)
        trade_mask, stats = gate.gate(logits_test, active)

        # trade_mask must be a subset of active
        assert not (trade_mask & ~active).any(), \
            "Gate cannot add stocks that were not in active_mask"

    def test_gate_reduces_or_equal_active(self):
        gate = ConformalGate(alpha=0.10)
        logits_cal, labels_cal = self._make_calibration_data()
        gate.calibrate(logits_cal, labels_cal)

        N = 100
        logits_test = make_logits(N)
        active = np.ones(N, dtype=bool)
        trade_mask, stats = gate.gate(logits_test, active)

        assert trade_mask.sum() <= N, "Gate cannot trade more than active stocks"

    def test_uncalibrated_warns_and_passthrough(self):
        gate = ConformalGate(alpha=0.10)
        N = 20
        logits = make_logits(N)
        active = make_active_mask(N)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trade_mask, stats = gate.gate(logits, active)
            assert len(w) == 1
            assert "calibrate" in str(w[0].message).lower()
        np.testing.assert_array_equal(trade_mask, active)

    def test_stats_gated_pct_in_range(self):
        gate = ConformalGate(alpha=0.10)
        logits_cal, labels_cal = self._make_calibration_data()
        gate.calibrate(logits_cal, labels_cal)
        N = 80
        logits_test = make_logits(N)
        active = np.ones(N, dtype=bool)
        _, stats = gate.gate(logits_test, active)
        assert 0.0 <= stats["gated_pct"] <= 1.0

    def test_calibrate_and_save_load(self, tmp_path):
        gate1 = ConformalGate(alpha=0.10)
        logits_cal, labels_cal = self._make_calibration_data()
        gate1.calibrate(logits_cal, labels_cal)

        save_path = str(tmp_path / "conformal.json")
        gate1.save(save_path)

        gate2 = ConformalGate(alpha=0.10)
        gate2.load(save_path)

        N = 30
        logits_test = make_logits(N)
        active = np.ones(N, dtype=bool)
        m1, _ = gate1.gate(logits_test, active)
        m2, _ = gate2.gate(logits_test, active)
        np.testing.assert_array_equal(m1, m2)

    def test_min_tradable_guard_readmits_ambiguous(self):
        """When gating leaves fewer than min_tradable stocks, the gate
        should progressively re-admit ambiguous (set_size==2) stocks."""
        gate = ConformalGate(alpha=0.10, min_tradable=15)
        logits_cal, labels_cal = self._make_calibration_data(500)
        gate.calibrate(logits_cal, labels_cal)

        N = 50
        # Near-zero logits → most will have set_size == 2 (ambiguous)
        logits_test = np.random.RandomState(42).randn(N) * 0.1
        active = np.ones(N, dtype=bool)
        trade_mask, stats = gate.gate(logits_test, active)

        # With high ambiguity, the guard should ensure at least min_tradable
        assert trade_mask.sum() >= min(15, active.sum()), (
            f"Expected >= 15 tradable, got {trade_mask.sum()}"
        )
        assert stats["relaxed"] >= 0

    def test_min_tradable_not_triggered_when_enough(self):
        """When gating naturally leaves enough stocks, no relaxation occurs."""
        gate = ConformalGate(alpha=0.10, min_tradable=5)
        logits_cal, labels_cal = self._make_calibration_data(500)
        gate.calibrate(logits_cal, labels_cal)

        N = 100
        # Strong logits → most will have set_size == 1
        logits_test = np.random.RandomState(7).randn(N) * 3.0
        active = np.ones(N, dtype=bool)
        trade_mask, stats = gate.gate(logits_test, active)

        assert stats["relaxed"] == 0, "Should not relax when enough stocks pass"


# =============================================================================
# compute_portfolio_metrics
# =============================================================================

class TestComputePortfolioMetrics:
    def test_empty_series(self):
        result = compute_portfolio_metrics([], [])
        assert math.isnan(result["sharpe"])
        assert result["n_snapshots"] == 0

    def test_single_snapshot(self):
        w = np.array([0.5, -0.5])
        r = np.array([0.01, -0.02])
        result = compute_portfolio_metrics([w], [r])
        assert result["n_snapshots"] == 1
        assert math.isnan(result["sharpe"]) or isinstance(result["sharpe"], float)

    def test_positive_pnl_positive_sharpe(self):
        N = 10
        # Constant positive P&L → positive Sharpe
        w = np.zeros(N)
        w[0] = 0.5
        w[1] = -0.5
        r_pos = np.zeros(N)
        r_pos[0] = 0.01
        r_pos[1] = -0.01  # Long wins, short loses → spread positive
        weights = [w.copy() for _ in range(30)]
        returns = [r_pos.copy() for _ in range(30)]
        result = compute_portfolio_metrics(weights, returns)
        assert result["sharpe"] > 0, "Constant positive PnL must give positive Sharpe"

    def test_turnover_zero_for_constant_weights(self):
        N = 10
        w = np.zeros(N)
        w[:5] = 0.1
        w[5:] = -0.1
        weights = [w.copy() for _ in range(10)]
        returns = [make_returns(1, N).squeeze() for _ in range(10)]
        result = compute_portfolio_metrics(weights, returns)
        assert result["turnover_mean"] == pytest.approx(0.0, abs=1e-8)

    def test_max_drawdown_non_negative(self):
        N = 5
        weights = [make_returns(1, N).squeeze() for _ in range(20)]
        returns = [make_returns(1, N).squeeze() for _ in range(20)]
        result = compute_portfolio_metrics(weights, returns)
        assert result["max_drawdown"] >= 0.0

    def test_tca_fields_present(self):
        w = np.array([0.5, -0.5])
        r = np.array([0.01, -0.02])
        result = compute_portfolio_metrics([w, w], [r, r], cost_bps=5.0)
        for key in ("sharpe_net", "total_pnl_net", "total_cost", "max_drawdown_net"):
            assert key in result, f"Missing TCA field: {key}"

    def test_tca_cost_reduces_pnl(self):
        N = 10
        w = np.zeros(N); w[0] = 0.5; w[1] = -0.5
        r = np.zeros(N); r[0] = 0.01; r[1] = -0.01
        result = compute_portfolio_metrics([w] * 5, [r] * 5, cost_bps=10.0)
        assert result["total_pnl_net"] < result["total_pnl"]
        assert result["total_cost"] > 0

    def test_zero_cost_gives_equal_gross_net(self):
        N = 4
        w = np.array([0.25, 0.25, -0.25, -0.25])
        r = np.array([0.01, 0.005, -0.005, -0.01])
        result = compute_portfolio_metrics([w] * 3, [r] * 3, cost_bps=0.0)
        assert result["sharpe"] == pytest.approx(result["sharpe_net"], abs=1e-6)
        assert result["total_pnl"] == pytest.approx(result["total_pnl_net"], abs=1e-8)


# =============================================================================
# PortfolioConstructor end-to-end
# =============================================================================

class TestPortfolioConstructorEndToEnd:
    def _setup(self, N: int = 50, T_train: int = 100):
        returns_train = make_returns(N, T_train)
        active = make_active_mask(N, 0.75)
        logits = make_logits(N)
        mag = make_mag(N)
        logits[~active] = 0.0
        mag[~active] = 0.0
        sector_ids = np.arange(N) % 5
        return returns_train, active, logits, mag, sector_ids

    def test_naive_method(self):
        N = 40
        returns_train, active, logits, mag, sec = self._setup(N)
        pc = PortfolioConstructor(method="naive", use_conformal_gate=False)
        pc.fit_covariance(returns_train, active)
        w, stats = pc.construct(logits, mag, active, sector_ids=sec)
        assert w.shape == (N,)
        assert abs(w.sum()) < 1e-8

    def test_mvo_method(self):
        pytest.importorskip("cvxpy")
        N = 40
        returns_train, active, logits, mag, sec = self._setup(N)
        pc = PortfolioConstructor(method="mvo", gamma=1.0, use_conformal_gate=False)
        pc.fit_covariance(returns_train, active)
        w, stats = pc.construct(logits, mag, active, sector_ids=sec)
        assert w.shape == (N,)
        assert abs(w.sum()) < 1e-4, f"MVO not dollar-neutral: sum={w.sum()}"

    def test_riskparity_method(self):
        pytest.importorskip("scipy")
        N = 40
        returns_train, active, logits, mag, sec = self._setup(N)
        pc = PortfolioConstructor(method="riskparity", use_conformal_gate=False)
        pc.fit_covariance(returns_train, active)
        w, stats = pc.construct(logits, mag, active, sector_ids=sec)
        assert w.shape == (N,)
        # Allow small numerical error for risk parity dollar neutrality
        assert abs(w.sum()) < 1e-4, f"RiskParity not dollar-neutral: sum={w.sum()}"

    def test_with_conformal_gate(self):
        N = 60
        returns_train, active, logits, mag, sec = self._setup(N)
        pc = PortfolioConstructor(method="naive", use_conformal_gate=True, alpha=0.10)
        pc.fit_covariance(returns_train, active)

        # Calibrate gate
        cal_logits = make_logits(200)
        cal_labels = (make_logits(200) > 0).astype(float)
        pc.fit_conformal(cal_logits, cal_labels)

        w, stats = pc.construct(logits, mag, active, sector_ids=sec)
        assert w.shape == (N,)
        assert "gate" in stats
        assert 0.0 <= stats["gate"]["gated_pct"] <= 1.0

    def test_all_gated_returns_zero_weights(self):
        N = 20
        returns_train, active, logits, mag, sec = self._setup(N)
        pc = PortfolioConstructor(method="naive", use_conformal_gate=True, alpha=0.10)
        pc.fit_covariance(returns_train, active)

        # Calibrate with extreme threshold that gates everything
        # by making all calibration scores nonconforming
        cal_logits = np.zeros(200)
        cal_labels = np.ones(200)  # All true = UP, but logits = 0 → 50% prob → nonconforming
        pc.fit_conformal(cal_logits, cal_labels)

        # Force all-zero active mask → should return zeros
        zero_active = np.zeros(N, dtype=bool)
        w, stats = pc.construct(logits, mag, zero_active, sector_ids=sec)
        np.testing.assert_allclose(w, 0.0, atol=1e-8)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            PortfolioConstructor(method="invalid_method")

    def test_construct_without_cov_fit_uses_fallback(self):
        N = 30
        active = make_active_mask(N)
        logits = make_logits(N)
        mag = make_mag(N)
        pc = PortfolioConstructor(method="mvo", use_conformal_gate=False)
        # Do not call fit_covariance
        w, stats = pc.construct(logits, mag, active)
        assert w.shape == (N,)
        assert stats.get("fallback") == "cov_not_fitted"
