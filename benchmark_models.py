"""
Benchmark Models for Macro-Aware DGRCL (Phase 0 — Rec 11)

Provides scoring functions that return per-snapshot stock scores [N_s] in the
same format as DGRCL's dir_logits.squeeze(-1), so all downstream evaluation
(compute_pairwise_ranking_loss, compute_long_short_alpha) runs identically.

Tiers:
  1. Null Baselines       — random, prior-day persistence
  2. Classical Factors    — momentum (12-1), short-term reversal, low-volatility
  3. ML Baselines         — LSTM-only (no graph), LightGBM LambdaRank
  4. Risk Decomposition   — Fama-French 3-Factor attribution (post-hoc on L/S returns)

Score contract:
    Each scoring function returns a torch.Tensor of shape [N_s].
    Higher score = predicted outperformer.
    Inactive padded stocks are assigned -inf so they sort to the bottom.
"""

import math
import os
import io
import json
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Absolute returns feature index in stock_window ([N_s, T, 8])
# Feature order: Close, High, Low, Log_Vol, RSI_14, MACD, Volatility_5, Returns
RETURNS_IDX = 7
VOLATILITY_IDX = 6  # Volatility_5 column


# =============================================================================
# TIER 1: NULL BASELINES
# =============================================================================

def random_scores(
    n_stocks: int,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Random ranking baseline.

    Expected rank accuracy is 50% (chance), providing the floor for all
    other benchmarks. Any model worse than this is pathological.

    Args:
        n_stocks: Number of stocks in universe (N_s)
        active_mask: [N_s] bool — inactive stocks receive -inf score

    Returns:
        [N_s] random scores drawn from N(0,1)
    """
    scores = torch.randn(n_stocks)
    if active_mask is not None:
        scores[~active_mask] = float('-inf')
    return scores


def prior_day_persistence(
    stock_window: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Naive momentum: yesterday's return predicts today's return.

    Uses the last observed return in the lookback window as the signal.
    Captures pure autocorrelation with a 1-day lag.

    Args:
        stock_window: [N_s, T, 8] — lookback window of stock features
        active_mask: [N_s] bool — inactive stocks receive -inf score

    Returns:
        [N_s] scores (last-day return)
    """
    # stock_window[:, -1, RETURNS_IDX] = return on the most recent lookback day
    scores = stock_window[:, -1, RETURNS_IDX].clone()
    if active_mask is not None:
        scores[~active_mask] = float('-inf')
    return scores


# =============================================================================
# TIER 2: CLASSICAL FACTOR MODELS
# =============================================================================

def momentum_12_1(
    stock_window: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
    skip_days: int = 21,
) -> torch.Tensor:
    """
    Cross-Sectional Momentum (12-1 month).

    Jegadeesh & Titman (1993): sum returns over the past 252 trading days
    excluding the most recent month (skip_days=21) to remove short-term
    reversal contamination.

    When the lookback window is shorter than 252 days (which it always is here
    at T=60), we use the available window minus the skip period. This is a
    compressed version of the factor but preserves its direction signal.

    Args:
        stock_window: [N_s, T, 8] lookback window
        active_mask: [N_s] bool
        skip_days: Days to skip at the end to avoid reversal (default 21 ≈ 1 month)

    Returns:
        [N_s] cumulative momentum scores
    """
    T = stock_window.size(1)
    returns = stock_window[:, :, RETURNS_IDX]   # [N_s, T]

    # Sum returns from start of window up to (T - skip_days)
    # If skip_days >= T, fall back to full window (edge case for very short windows)
    end_idx = max(T - skip_days, 1)
    scores = returns[:, :end_idx].sum(dim=1)    # [N_s]

    if active_mask is not None:
        scores[~active_mask] = float('-inf')
    return scores


def short_term_reversal(
    stock_window: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
    lookback_days: int = 5,
) -> torch.Tensor:
    """
    Short-Term Reversal (5-day).

    De Bondt & Thaler (1985): recent losers tend to outperform in the near
    term. Negating the recent cumulative return makes ranking interpretable
    (higher score = stronger reversal candidate = predicted outperformer).

    Args:
        stock_window: [N_s, T, 8]
        active_mask: [N_s] bool
        lookback_days: Number of recent days to sum for short-term return (default 5)

    Returns:
        [N_s] reversal scores (negated recent return)
    """
    returns = stock_window[:, :, RETURNS_IDX]   # [N_s, T]
    recent_return = returns[:, -lookback_days:].sum(dim=1)  # [N_s]
    # Negate: stocks with large recent losses rank highest (reversal candidates)
    scores = -recent_return

    if active_mask is not None:
        scores[~active_mask] = float('-inf')
    return scores


def low_volatility(
    stock_window: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Low-Volatility Anomaly.

    Baker, Bradley & Wurgler (2011): low-volatility stocks outperform
    high-volatility stocks on a risk-adjusted basis. We use the Volatility_5
    feature (5-day rolling log-return std) from the stock_window; if that
    column is absent or zero, we fall back to computing return std directly.

    Negating volatility ensures low-vol stocks rank highest.

    Args:
        stock_window: [N_s, T, 8]
        active_mask: [N_s] bool

    Returns:
        [N_s] scores (negated mean 5-day volatility)
    """
    vol_series = stock_window[:, :, VOLATILITY_IDX]   # [N_s, T]
    # Use mean of the pre-computed volatility column over the entire lookback
    mean_vol = vol_series.mean(dim=1)   # [N_s]

    # Fallback: if the column is uniformly zero (synthetic data), compute directly
    if mean_vol.abs().sum() < 1e-6:
        returns = stock_window[:, :, RETURNS_IDX]
        std_per_stock = returns.std(dim=1)
        mean_vol = std_per_stock

    scores = -mean_vol  # Higher score = lower volatility = predicted outperformer

    if active_mask is not None:
        scores[~active_mask] = float('-inf')
    return scores


# =============================================================================
# TIER 3: ML BASELINES
# =============================================================================

class LSTMOnlyBenchmark:
    """
    LSTM-Only model: temporal encoder + multi-task head with no graph.

    Ablates the graph component (DynamicGraphLearner + MacroPropagation) by
    using only the TemporalEncoder and MultiTaskHead from macro_dgrcl.py.
    Trained per fold on the same snapshots as DGRCL.

    The model is initialized with identical hyperparameters to DGRCL
    (hidden_dim=64, 2 layers, dropout=0.5) so any performance difference
    is attributable solely to the absence of the graph structure.
    """

    def __init__(
        self,
        num_stocks: int,
        stock_feature_dim: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        device: torch.device = None,
    ):
        from macro_dgrcl import TemporalEncoder, MultiTaskHead

        requested_device = device or torch.device('cpu')
        self.hidden_dim = hidden_dim
        self.num_stocks = num_stocks

        # Stock encoder only (no macro encoder, no graph)
        self.encoder = TemporalEncoder(
            input_dim=stock_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.head = MultiTaskHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Attempt to move to requested device; fall back to CPU on HIP/CUDA errors
        # (common in AMD ROCm environments without pre-compiled GPU kernels).
        try:
            dummy = torch.zeros(1, 1, stock_feature_dim, device=requested_device)
            self.encoder = self.encoder.to(requested_device)
            self.head = self.head.to(requested_device)
            self.device = requested_device
        except (RuntimeError, AssertionError) as e:
            if 'HIP' in str(e) or 'CUDA' in str(e) or 'hip' in str(e).lower():
                print(f"  [LSTMOnly] GPU unavailable ({type(e).__name__}), falling back to CPU")
                self.device = torch.device('cpu')
                self.encoder = self.encoder.to(self.device)
                self.head = self.head.to(self.device)
            else:
                raise

        self._all_params = list(self.encoder.parameters()) + list(self.head.parameters())
        self.optimizer = torch.optim.AdamW(self._all_params, lr=5e-5, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

    def score(
        self,
        stock_features: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute direction scores for all stocks in one snapshot.

        Args:
            stock_features: [N_s, T, d_s]
            active_mask: [N_s] bool

        Returns:
            [N_s] direction scores (logits from direction head)
        """
        self.encoder.eval()
        self.head.eval()

        with torch.no_grad():
            stock_h = self.encoder(stock_features.to(self.device))   # [N_s, H]
            dir_logits, _ = self.head(stock_h)                        # [N_s, 1]
            scores = dir_logits.squeeze(-1)                            # [N_s]

        if active_mask is not None:
            scores = scores.clone()
            scores[~active_mask.to(self.device)] = float('-inf')

        return scores.cpu()

    def train_fold(
        self,
        train_snapshots: List[Tuple],
        sector_mask: Optional[torch.Tensor],
        sector_ids: Optional[torch.Tensor],
        num_epochs: int = 100,
        max_grad_norm: float = 0.5,
        patience: int = 10,
    ) -> Dict[str, float]:
        """
        Train the LSTM-only model on one fold's training snapshots.

        Uses the same pairwise ranking loss as DGRCL for a like-for-like
        comparison. Magnitude head is trained but not used for scoring.

        Args:
            train_snapshots: List of (stock_window, macro_window, returns, active_mask) tuples
            sector_mask: [N_s, N_s] bool — sector constraint for pairwise loss
            sector_ids: [N_s] long — sector IDs for L/S alpha computation
            num_epochs: Max epochs (default 100, same as DGRCL)
            max_grad_norm: Gradient clipping threshold (same as DGRCL 0.5)
            patience: Early stopping patience (default 10)

        Returns:
            Dict with training metrics averaged over last epoch
        """
        from train import compute_pairwise_ranking_loss, compute_log_scaled_mag_target

        self.encoder.train()
        self.head.train()

        mse_loss = nn.SmoothL1Loss()
        best_loss = float('inf')
        patience_counter = 0
        best_encoder_state = None
        best_head_state = None

        if sector_mask is not None:
            sector_mask = sector_mask.to(self.device)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_valid = 0

            for snap in train_snapshots:
                stock_feat = snap[0].to(self.device)
                returns_t = snap[2].to(self.device)
                active_mask = snap[3].to(self.device) if len(snap) == 4 else None

                self.optimizer.zero_grad()

                stock_h = self.encoder(stock_feat)       # [N_s, H]
                dir_logits, mag_preds = self.head(stock_h)

                scores = dir_logits.squeeze(-1)
                loss_dir, _ = compute_pairwise_ranking_loss(
                    scores=scores,
                    returns=returns_t,
                    sector_mask=sector_mask,
                    active_mask=active_mask,
                )

                mag_target = compute_log_scaled_mag_target(returns_t, active_mask=active_mask)
                if active_mask is not None and active_mask.any():
                    loss_mag = mse_loss(mag_preds.squeeze(-1)[active_mask], mag_target[active_mask])
                else:
                    loss_mag = mse_loss(mag_preds.squeeze(-1), mag_target)

                loss = loss_dir + 0.05 * loss_mag

                if not torch.isfinite(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(self._all_params, max_grad_norm)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_valid += 1

            self.scheduler.step()
            mean_loss = epoch_loss / max(n_valid, 1)

            # Simple early stopping on training loss (no val set used here —
            # val snapshots are reserved for unbiased evaluation)
            if mean_loss < best_loss - 1e-4:
                best_loss = mean_loss
                patience_counter = 0
                # Save best state for restoration
                import copy
                best_encoder_state = copy.deepcopy(self.encoder.state_dict())
                best_head_state = copy.deepcopy(self.head.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Restore best weights
        if best_encoder_state is not None:
            self.encoder.load_state_dict(best_encoder_state)
            self.head.load_state_dict(best_head_state)

        return {'best_train_loss': best_loss, 'epochs_trained': epoch + 1}


class LightGBMRanker:
    """
    LightGBM LambdaRank benchmark.

    Treats stock ranking as a learning-to-rank problem using the industry-standard
    LambdaRank objective. Features are the mean of each stock's last-window OHLCV
    and technical indicator columns (collapsed T-dimension to reduce dimensionality).

    Trained per fold on train_snapshots, evaluated on val_snapshots.
    Not suitable for intraday rebalancing but provides a strong tree-based baseline.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        n_estimators: int = 200,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
    ):
        try:
            import lightgbm as lgb
            self._lgb = lgb
        except ImportError:
            raise ImportError(
                "lightgbm is required for LightGBMRanker. "
                "Install with: pip install lightgbm"
            )

        self.feature_dim = feature_dim
        self.model = self._lgb.LGBMRanker(
            objective='lambdarank',
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            verbose=-1,
        )
        self._trained = False

    @staticmethod
    def _extract_features(stock_window: torch.Tensor) -> np.ndarray:
        """
        Collapse [N_s, T, d_s] to [N_s, d_s * 4] summary statistics.

        For each feature dimension, computes mean, std, last value, and
        momentum (last - first) to preserve temporal information without
        passing full sequences to the tree model.

        Args:
            stock_window: [N_s, T, d_s]

        Returns:
            [N_s, d_s * 4] numpy array
        """
        arr = stock_window.numpy()      # [N_s, T, d_s]
        mean = arr.mean(axis=1)         # [N_s, d_s]
        std = arr.std(axis=1)           # [N_s, d_s]
        last = arr[:, -1, :]            # [N_s, d_s]
        momentum = arr[:, -1, :] - arr[:, 0, :]  # [N_s, d_s]
        return np.concatenate([mean, std, last, momentum], axis=1)

    def train_fold(
        self,
        train_snapshots: List[Tuple],
    ) -> None:
        """
        Train LightGBM ranker on one fold's training snapshots.

        Uses binary-direction labels (+1 / 0) derived from cross-sectional
        returns, grouped by snapshot for LambdaRank.

        Args:
            train_snapshots: List of (stock_window, macro_window, returns, active_mask) tuples
        """
        all_X = []
        all_y = []
        group_sizes = []

        for snap in train_snapshots:
            stock_feat = snap[0]    # [N_s, T, d_s] on CPU
            returns_t = snap[2]     # [N_s]
            active_mask = snap[3] if len(snap) == 4 else None

            if active_mask is not None:
                active = active_mask.bool()
                if active.sum() < 2:
                    continue
                stock_feat_active = stock_feat[active]
                returns_active = returns_t[active]
            else:
                stock_feat_active = stock_feat
                returns_active = returns_t

            X = self._extract_features(stock_feat_active)   # [n_active, d]
            # Binary relevance labels for LambdaRank: top half of stocks (by return)
            # get label=1, bottom half get label=0. LightGBM LambdaRank requires
            # labels in [0, num_label_classes-1]; binary {0, 1} is simplest valid form.
            n_active = len(returns_active)
            sorted_idx = returns_active.argsort()  # ascending
            y = np.zeros(n_active, dtype=int)
            top_k = n_active // 2
            y[sorted_idx[-top_k:].numpy()] = 1  # top half = relevant

            all_X.append(X)
            all_y.append(y)
            group_sizes.append(len(y))

        if not all_X:
            return

        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)

        self.model.fit(
            X_train, y_train,
            group=group_sizes,
        )
        self._trained = True

    def score(
        self,
        stock_window: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score all stocks using trained LightGBM ranker.

        Args:
            stock_window: [N_s, T, d_s]
            active_mask: [N_s] bool

        Returns:
            [N_s] ranking scores (higher = predicted outperformer)
        """
        N_s = stock_window.size(0)
        scores = torch.full((N_s,), float('-inf'))

        if not self._trained:
            return scores

        if active_mask is not None:
            active = active_mask.bool()
        else:
            active = torch.ones(N_s, dtype=torch.bool)

        if active.sum() == 0:
            return scores

        X = self._extract_features(stock_window[active])
        preds = self.model.predict(X)   # [n_active]
        scores[active] = torch.tensor(preds, dtype=torch.float32)

        return scores


# =============================================================================
# TIER 4: FAMA-FRENCH 3-FACTOR ATTRIBUTION
# =============================================================================

def download_ff3_factors(
    start_date: str = "2007-01-01",
    end_date: str = "2026-12-31",
    cache_path: str = "./data/ff3_factors.csv",
) -> "pd.DataFrame":
    """
    Download Fama-French 3-Factor returns from Ken French's data library.

    Downloads the "Fama/French 3 Factors (Daily)" CSV from the WRDS/French
    data library. Caches locally to avoid repeated downloads.

    Factors returned (decimal, not percent):
        Mkt-RF — daily excess market return
        SMB    — small-minus-big size factor
        HML    — high-minus-low value factor
        RF     — daily risk-free rate

    Args:
        start_date: Start date for factor data (YYYY-MM-DD)
        end_date: End date for factor data (YYYY-MM-DD)
        cache_path: Local path to cache the downloaded CSV

    Returns:
        pd.DataFrame with DatetimeIndex and columns [Mkt-RF, SMB, HML, RF]
    """
    import pandas as pd
    import urllib.request

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download FF3 factors: {e}\n"
            "Please download manually from http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html "
            "and save to ./data/ff3_factors.csv with columns: Date, Mkt-RF, SMB, HML, RF"
        )

    # Unzip and parse
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
        with zf.open(csv_name) as f:
            raw = f.read().decode('latin-1')

    # The FF3 daily CSV has a header section before the data rows
    lines = raw.split('\n')
    # Find the first line that starts with a date (8 digits)
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and len(stripped.split(',')[0].strip()) == 8:
            data_start = i
            break

    data_lines = '\n'.join(lines[data_start:])
    df = pd.read_csv(
        io.StringIO(data_lines),
        header=None,
        names=['Date', 'Mkt-RF', 'SMB', 'HML', 'RF'],
        skipinitialspace=True,
    )

    # Drop footer rows (FF3 CSVs sometimes include annual data at the bottom)
    df = df[df['Date'].astype(str).str.len() == 8]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df = df.set_index('Date').sort_index()

    # Convert from percent to decimal
    for col in ['Mkt-RF', 'SMB', 'HML', 'RF']:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0

    df = df.dropna()
    df = df.loc[start_date:end_date]

    df.to_csv(cache_path)
    print(f"  Downloaded and cached FF3 factors: {cache_path}")
    return df


def compute_ff3_attribution(
    fold_ls_returns: List[float],
    fold_dates: List,
    ff3_factors: "pd.DataFrame",
) -> Dict[str, float]:
    """
    Fama-French 3-Factor attribution of DGRCL's L/S returns.

    Regresses the per-fold L/S return series on Mkt-RF, SMB, HML to decompose
    DGRCL's alpha into factor-explained and unexplained components.

    Equation:
        R_LS(t) = α + β_Mkt * Mkt-RF(t) + β_SMB * SMB(t) + β_HML * HML(t) + ε(t)

    The intercept α is the factor-unexplained daily alpha. A statistically
    significant positive α indicates DGRCL's edge is not a disguised bet on
    known risk premia.

    Args:
        fold_ls_returns: List of per-snapshot L/S spreads (one per val snapshot)
        fold_dates: Corresponding dates for each snapshot (pd.Timestamp or datetime)
        ff3_factors: DataFrame from download_ff3_factors()

    Returns:
        Dict with: alpha, beta_mkt, beta_smb, beta_hml, r_squared,
                   alpha_t_stat, alpha_p_value, n_observations
    """
    import pandas as pd
    from scipy import stats

    if not fold_ls_returns or ff3_factors is None or ff3_factors.empty:
        return {k: float('nan') for k in [
            'alpha', 'beta_mkt', 'beta_smb', 'beta_hml',
            'r_squared', 'alpha_t_stat', 'alpha_p_value', 'n_observations'
        ]}

    # Align L/S returns with FF3 factor data by date
    ls_series = pd.Series(fold_ls_returns, index=pd.DatetimeIndex(fold_dates))
    aligned = ls_series.to_frame('LS').join(ff3_factors[['Mkt-RF', 'SMB', 'HML']], how='inner')
    aligned = aligned.dropna()

    if len(aligned) < 5:
        return {k: float('nan') for k in [
            'alpha', 'beta_mkt', 'beta_smb', 'beta_hml',
            'r_squared', 'alpha_t_stat', 'alpha_p_value', 'n_observations'
        ]}

    y = aligned['LS'].values
    X = np.column_stack([
        np.ones(len(aligned)),
        aligned['Mkt-RF'].values,
        aligned['SMB'].values,
        aligned['HML'].values,
    ])

    # OLS regression via least squares
    result = np.linalg.lstsq(X, y, rcond=None)
    coeffs = result[0]

    alpha, beta_mkt, beta_smb, beta_hml = coeffs

    # Compute R² and t-stat for alpha
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = len(y)
    k = X.shape[1]  # 4 parameters (intercept + 3 factors)
    residual_std = np.sqrt(ss_res / max(n - k, 1))

    # Standard error of alpha via (X'X)^{-1} * residual_var
    try:
        XtXinv = np.linalg.inv(X.T @ X)
        alpha_se = residual_std * np.sqrt(XtXinv[0, 0])
        alpha_t_stat = alpha / alpha_se if alpha_se > 0 else float('nan')
        # Two-tailed p-value from t-distribution
        alpha_p_value = 2.0 * (1.0 - stats.t.cdf(abs(alpha_t_stat), df=n - k))
    except np.linalg.LinAlgError:
        alpha_t_stat = float('nan')
        alpha_p_value = float('nan')

    return {
        'alpha': float(alpha),
        'beta_mkt': float(beta_mkt),
        'beta_smb': float(beta_smb),
        'beta_hml': float(beta_hml),
        'r_squared': float(r_squared),
        'alpha_t_stat': float(alpha_t_stat),
        'alpha_p_value': float(alpha_p_value),
        'n_observations': int(n),
    }


# =============================================================================
# REGISTRY: ALL BENCHMARKS
# =============================================================================

#: Names of Tier 1/2 benchmarks that require no training
SIMPLE_BENCHMARKS = [
    'random',
    'prior_day_persistence',
    'momentum_12_1',
    'short_term_reversal',
    'low_volatility',
]

#: Names of Tier 3 benchmarks that train per fold
TRAINABLE_BENCHMARKS = [
    'lstm_only',
    'lgbm_ranker',
]

ALL_BENCHMARKS = SIMPLE_BENCHMARKS + TRAINABLE_BENCHMARKS
