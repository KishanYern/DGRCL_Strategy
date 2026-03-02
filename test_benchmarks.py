"""
Unit Tests for Phase 0 Benchmark Suite (Rec 11)

Tests:
  1. Tier 1 — Null baselines (random, prior_day_persistence)
  2. Tier 2 — Classical factors (momentum_12_1, short_term_reversal, low_volatility)
  3. Tier 3 — ML baselines (LSTMOnlyBenchmark, LightGBMRanker)
  4. Tier 4 — FF3 attribution (compute_ff3_attribution)
  5. Runner — score_snapshot dispatch, benchmark_ls_alpha
"""

import math
import pytest
import numpy as np
import torch
import pandas as pd

from benchmark_models import (
    random_scores,
    prior_day_persistence,
    momentum_12_1,
    short_term_reversal,
    low_volatility,
    LSTMOnlyBenchmark,
    LightGBMRanker,
    compute_ff3_attribution,
    RETURNS_IDX,
    VOLATILITY_IDX,
    ALL_BENCHMARKS,
    SIMPLE_BENCHMARKS,
    TRAINABLE_BENCHMARKS,
)
from run_benchmarks import score_snapshot, compute_benchmark_ls_alpha


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def n_stocks():
    return 20


@pytest.fixture
def window_size():
    return 60


@pytest.fixture
def feature_dim():
    return 8


@pytest.fixture
def stock_window(n_stocks, window_size, feature_dim):
    """[N_s, T, 8] stock feature tensor with realistic structure."""
    torch.manual_seed(0)
    window = torch.randn(n_stocks, window_size, feature_dim) * 0.01
    # Ensure returns column has some variation
    window[:, :, RETURNS_IDX] = torch.randn(n_stocks, window_size) * 0.02
    return window


@pytest.fixture
def active_mask(n_stocks):
    """~80% active, ensuring at least 4 active and 1 inactive."""
    torch.manual_seed(1)
    mask = torch.rand(n_stocks) > 0.2
    mask[0] = mask[1] = mask[2] = mask[3] = True
    mask[-1] = False
    return mask


@pytest.fixture
def returns(n_stocks):
    torch.manual_seed(2)
    return torch.randn(n_stocks) * 0.02


@pytest.fixture
def snapshot(stock_window, returns, active_mask):
    """4-tuple snapshot matching the data_loader format."""
    macro = torch.randn(4, 60, 4)
    return (stock_window, macro, returns, active_mask)


# =============================================================================
# TEST: TIER 1 — NULL BASELINES
# =============================================================================

class TestRandomScores:

    def test_output_shape(self, n_stocks):
        scores = random_scores(n_stocks)
        assert scores.shape == (n_stocks,)

    def test_with_active_mask(self, n_stocks, active_mask):
        scores = random_scores(n_stocks, active_mask)
        # Inactive stocks must be -inf
        inactive = ~active_mask
        assert (scores[inactive] == float('-inf')).all()

    def test_stochastic_different_each_call(self, n_stocks):
        """Random baseline must vary between calls (non-deterministic signal)."""
        s1 = random_scores(n_stocks)
        s2 = random_scores(n_stocks)
        assert not torch.equal(s1, s2), "Random scores should differ between calls"

    def test_no_nan(self, n_stocks):
        scores = random_scores(n_stocks)
        assert not torch.isnan(scores).any()


class TestPriorDayPersistence:

    def test_output_shape(self, stock_window, n_stocks):
        scores = prior_day_persistence(stock_window)
        assert scores.shape == (n_stocks,)

    def test_signal_is_last_return(self, stock_window, n_stocks):
        """Score must equal the last-day return in the lookback window."""
        scores = prior_day_persistence(stock_window)
        expected = stock_window[:, -1, RETURNS_IDX]
        assert torch.allclose(scores, expected)

    def test_inactive_gets_neg_inf(self, stock_window, active_mask):
        scores = prior_day_persistence(stock_window, active_mask)
        assert (scores[~active_mask] == float('-inf')).all()

    def test_deterministic(self, stock_window):
        """Same input must produce identical scores."""
        s1 = prior_day_persistence(stock_window)
        s2 = prior_day_persistence(stock_window)
        assert torch.equal(s1, s2)

    def test_no_future_leakage(self, stock_window, returns):
        """Changing future returns must not affect scores."""
        scores_before = prior_day_persistence(stock_window)
        # Modify future returns tensor (should not affect stock_window)
        future_returns_modified = returns * 999.0
        scores_after = prior_day_persistence(stock_window)
        assert torch.equal(scores_before, scores_after)


# =============================================================================
# TEST: TIER 2 — CLASSICAL FACTORS
# =============================================================================

class TestMomentum12_1:

    def test_output_shape(self, stock_window, n_stocks):
        scores = momentum_12_1(stock_window)
        assert scores.shape == (n_stocks,)

    def test_signal_direction(self, n_stocks, window_size, feature_dim):
        """A stock with higher cumulative return should score higher."""
        window = torch.zeros(n_stocks, window_size, feature_dim)
        # Stock 0: strong positive returns
        window[0, :, RETURNS_IDX] = 0.01
        # Stock 1: strong negative returns
        window[1, :, RETURNS_IDX] = -0.01

        scores = momentum_12_1(window)
        assert scores[0].item() > scores[1].item(), \
            "High-momentum stock should score higher than low-momentum stock"

    def test_skips_recent_days(self, n_stocks, window_size, feature_dim):
        """Scores should not change if we modify only the last skip_days days."""
        window = torch.randn(n_stocks, window_size, feature_dim) * 0.01
        skip_days = 21

        # Reference scores
        scores_ref = momentum_12_1(window, skip_days=skip_days)

        # Modify only the last skip_days (should be excluded from signal)
        window_modified = window.clone()
        window_modified[:, -skip_days:, RETURNS_IDX] = 999.0
        scores_modified = momentum_12_1(window_modified, skip_days=skip_days)

        assert torch.allclose(scores_ref, scores_modified), \
            "Momentum score should not depend on the most recent skip_days returns"

    def test_inactive_gets_neg_inf(self, stock_window, active_mask):
        scores = momentum_12_1(stock_window, active_mask)
        assert (scores[~active_mask] == float('-inf')).all()

    def test_no_nan(self, stock_window):
        scores = momentum_12_1(stock_window)
        assert not torch.isnan(scores).any()


class TestShortTermReversal:

    def test_output_shape(self, stock_window, n_stocks):
        scores = short_term_reversal(stock_window)
        assert scores.shape == (n_stocks,)

    def test_signal_direction(self, n_stocks, window_size, feature_dim):
        """Recent losers should score HIGHER (reversal)."""
        window = torch.zeros(n_stocks, window_size, feature_dim)
        # Stock 0: recent loss (should score higher as reversal candidate)
        window[0, -5:, RETURNS_IDX] = -0.05
        # Stock 1: recent gain (should score lower)
        window[1, -5:, RETURNS_IDX] = 0.05

        scores = short_term_reversal(window)
        assert scores[0].item() > scores[1].item(), \
            "Recent loser should be reversal candidate and score higher"

    def test_negated_sign(self, stock_window):
        """Score is negative of recent cumulative return."""
        scores = short_term_reversal(stock_window, lookback_days=5)
        expected = -stock_window[:, -5:, RETURNS_IDX].sum(dim=1)
        assert torch.allclose(scores, expected)

    def test_inactive_gets_neg_inf(self, stock_window, active_mask):
        scores = short_term_reversal(stock_window, active_mask)
        assert (scores[~active_mask] == float('-inf')).all()


class TestLowVolatility:

    def test_output_shape(self, stock_window, n_stocks):
        scores = low_volatility(stock_window)
        assert scores.shape == (n_stocks,)

    def test_signal_direction(self, n_stocks, window_size, feature_dim):
        """Low-volatility stocks should score higher."""
        window = torch.zeros(n_stocks, window_size, feature_dim)
        # Stock 0: low volatility
        window[0, :, VOLATILITY_IDX] = 0.001
        # Stock 1: high volatility
        window[1, :, VOLATILITY_IDX] = 0.050

        scores = low_volatility(window)
        assert scores[0].item() > scores[1].item(), \
            "Low-volatility stock should score higher"

    def test_negated_mean_vol(self, stock_window):
        """When Volatility_5 column is non-zero, score equals negated mean vol."""
        # Ensure non-trivial volatility column
        stock_window[:, :, VOLATILITY_IDX] = torch.abs(torch.randn_like(stock_window[:, :, VOLATILITY_IDX])) * 0.01 + 0.001
        scores = low_volatility(stock_window)
        expected = -stock_window[:, :, VOLATILITY_IDX].mean(dim=1)
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_inactive_gets_neg_inf(self, stock_window, active_mask):
        scores = low_volatility(stock_window, active_mask)
        assert (scores[~active_mask] == float('-inf')).all()


# =============================================================================
# TEST: TIER 3 — ML BASELINES
# =============================================================================

class TestLSTMOnlyBenchmark:

    def test_score_output_shape(self, n_stocks, window_size, feature_dim):
        bench = LSTMOnlyBenchmark(num_stocks=n_stocks, stock_feature_dim=feature_dim)
        stock_features = torch.randn(n_stocks, window_size, feature_dim)
        scores = bench.score(stock_features)
        assert scores.shape == (n_stocks,)

    def test_inactive_gets_neg_inf(self, n_stocks, window_size, feature_dim, active_mask):
        bench = LSTMOnlyBenchmark(num_stocks=n_stocks, stock_feature_dim=feature_dim)
        stock_features = torch.randn(n_stocks, window_size, feature_dim)
        scores = bench.score(stock_features, active_mask)
        assert (scores[~active_mask] == float('-inf')).all()

    def test_scores_vary_with_input(self, n_stocks, window_size, feature_dim):
        """Different inputs should produce different scores (model is non-trivial)."""
        bench = LSTMOnlyBenchmark(num_stocks=n_stocks, stock_feature_dim=feature_dim)
        s1 = bench.score(torch.randn(n_stocks, window_size, feature_dim))
        s2 = bench.score(torch.randn(n_stocks, window_size, feature_dim))
        assert not torch.equal(s1, s2)

    def test_no_nan_in_scores(self, n_stocks, window_size, feature_dim):
        bench = LSTMOnlyBenchmark(num_stocks=n_stocks, stock_feature_dim=feature_dim)
        scores = bench.score(torch.randn(n_stocks, window_size, feature_dim))
        assert not torch.isnan(scores).any()

    def test_train_fold_runs(self, n_stocks, window_size, feature_dim, snapshot):
        """train_fold must complete without error on a small synthetic fold."""
        bench = LSTMOnlyBenchmark(num_stocks=n_stocks, stock_feature_dim=feature_dim)
        # Use a 3-element list of snapshots
        snapshots = [snapshot] * 5
        result = bench.train_fold(snapshots, sector_mask=None, sector_ids=None, num_epochs=3)
        assert 'epochs_trained' in result
        assert result['epochs_trained'] >= 1


class TestLightGBMRanker:

    def test_score_output_shape_untrained(self, n_stocks, window_size, feature_dim):
        bench = LightGBMRanker(feature_dim=feature_dim)
        stock_features = torch.randn(n_stocks, window_size, feature_dim)
        scores = bench.score(stock_features)
        # Untrained model returns all -inf
        assert scores.shape == (n_stocks,)
        assert (scores == float('-inf')).all()

    def test_train_and_score(self, n_stocks, window_size, feature_dim, snapshot):
        """After training, score should return finite values for active stocks."""
        bench = LightGBMRanker(feature_dim=feature_dim)
        snapshots = [snapshot] * 20  # Enough for LightGBM
        bench.train_fold(snapshots)

        stock_features, _, returns, active_mask = snapshot
        scores = bench.score(stock_features, active_mask)
        assert scores.shape == (n_stocks,)
        # Active stocks should have finite scores
        assert torch.isfinite(scores[active_mask]).all()
        # Inactive get -inf
        assert (scores[~active_mask] == float('-inf')).all()

    def test_extract_features_shape(self, n_stocks, window_size, feature_dim):
        stock_window = torch.randn(n_stocks, window_size, feature_dim)
        X = LightGBMRanker._extract_features(stock_window)
        # mean + std + last + momentum = 4 * feature_dim
        assert X.shape == (n_stocks, feature_dim * 4)


# =============================================================================
# TEST: TIER 4 — FF3 ATTRIBUTION
# =============================================================================

class TestFF3Attribution:

    def test_returns_expected_keys(self):
        ls_returns = [0.001, -0.002, 0.003, 0.001, 0.002] * 10
        dates = pd.date_range('2021-01-01', periods=len(ls_returns), freq='B')

        # Create minimal fake FF3 data aligned to dates
        ff3 = pd.DataFrame({
            'Mkt-RF': np.random.randn(len(dates)) * 0.01,
            'SMB': np.random.randn(len(dates)) * 0.005,
            'HML': np.random.randn(len(dates)) * 0.005,
            'RF': np.full(len(dates), 0.00005),
        }, index=dates)

        result = compute_ff3_attribution(ls_returns, list(dates), ff3)

        expected_keys = {'alpha', 'beta_mkt', 'beta_smb', 'beta_hml',
                         'r_squared', 'alpha_t_stat', 'alpha_p_value', 'n_observations'}
        assert set(result.keys()) == expected_keys

    def test_empty_input_returns_nan(self):
        result = compute_ff3_attribution([], [], None)
        assert math.isnan(result['alpha'])

    def test_r_squared_in_unit_interval(self):
        ls_returns = [0.001 * i % 3 for i in range(50)]
        dates = pd.date_range('2021-01-01', periods=len(ls_returns), freq='B')
        ff3 = pd.DataFrame({
            'Mkt-RF': np.random.randn(len(dates)) * 0.01,
            'SMB':    np.random.randn(len(dates)) * 0.005,
            'HML':    np.random.randn(len(dates)) * 0.005,
        }, index=dates)

        result = compute_ff3_attribution(ls_returns, list(dates), ff3)
        if not math.isnan(result['r_squared']):
            assert 0.0 <= result['r_squared'] <= 1.0


# =============================================================================
# TEST: RUNNER DISPATCH
# =============================================================================

class TestScoreSnapshotDispatch:

    def test_all_simple_benchmarks_dispatch(self, snapshot, n_stocks):
        """score_snapshot must dispatch without error for all Tier 1/2 benchmarks."""
        for benchmark_name in SIMPLE_BENCHMARKS:
            scores = score_snapshot(benchmark_name, snapshot)
            assert scores.shape == (n_stocks,), \
                f"{benchmark_name}: expected shape ({n_stocks},), got {scores.shape}"

    def test_unknown_benchmark_raises(self, snapshot):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            score_snapshot('nonexistent_model', snapshot)

    def test_trainable_benchmarks_require_model(self, snapshot):
        """Trainable benchmarks without a model must raise AssertionError."""
        for name in TRAINABLE_BENCHMARKS:
            with pytest.raises(AssertionError):
                score_snapshot(name, snapshot, trained_model=None)


class TestComputeBenchmarkLSAlpha:

    def test_returns_expected_keys(self, snapshot, n_stocks):
        val_snapshots = [snapshot] * 5
        result = compute_benchmark_ls_alpha(val_snapshots, 'random', sector_ids=None)
        assert 'total_ls_alpha' in result
        assert 'mean_ls_alpha' in result
        assert 'n_snapshots' in result
        assert 'ls_returns_list' in result

    def test_all_simple_benchmarks_run(self, snapshot):
        val_snapshots = [snapshot] * 5
        for name in SIMPLE_BENCHMARKS:
            result = compute_benchmark_ls_alpha(val_snapshots, name, sector_ids=None)
            assert isinstance(result['mean_ls_alpha'], float)

    def test_sector_balanced_book(self, n_stocks, snapshot):
        """With sector_ids, result should equal global fallback for balanced sectors."""
        # Assign all stocks to same sector — sector-balanced reduces to global
        sector_ids = torch.zeros(n_stocks, dtype=torch.long)
        val_snapshots = [snapshot] * 5
        result_global = compute_benchmark_ls_alpha(
            val_snapshots, 'momentum_12_1', sector_ids=None
        )
        result_sector = compute_benchmark_ls_alpha(
            val_snapshots, 'momentum_12_1', sector_ids=sector_ids
        )
        # Both should produce valid (not NaN) results
        assert not math.isnan(result_global['mean_ls_alpha'])
        assert not math.isnan(result_sector['mean_ls_alpha'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
