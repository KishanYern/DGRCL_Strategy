"""
Phase 1 Portfolio Construction for Macro-DGRCL

Three complementary strategies built on top of model outputs (dir_logits, mag_preds):

1. MeanVarianceOptimizer  -- Markowitz MVO with Ledoit-Wolf covariance, solved via cvxpy
2. RiskParityOptimizer    -- Equal-risk-contribution across graph-learned clusters
3. ConformalGate          -- Pre-trade filter: skip stocks where set_size != 1

All strategies share a common pre-processing pipeline:
    MacroDGRCL outputs
        -> ConformalGate (abstention mask)
        -> ExpectedReturnEstimator (mu vector)
        -> CovarianceEstimator (Sigma matrix, per fold)
        -> MVO or RiskParity
        -> portfolio weights w [N_s]

Usage:
    from portfolio_optimizer import PortfolioConstructor

    constructor = PortfolioConstructor(method="mvo", gamma=1.0)
    constructor.fit_covariance(train_returns_tensor)          # once per fold
    constructor.fit_conformal(val_logits, val_labels)         # once per fold

    for snapshot in val_snapshots:
        weights = constructor.construct(dir_logits, mag_preds, active_mask)
        realized_pnl = (weights * future_returns).sum()
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional heavy dependencies — imported lazily so the module can be imported
# even if cvxpy / sklearn / scipy are not installed. Methods that require them
# will raise a clear ImportError at call time.
# ---------------------------------------------------------------------------
_CVXPY_AVAILABLE = False
_SKLEARN_AVAILABLE = False
_SCIPY_AVAILABLE = False

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.cluster import SpectralClustering
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    from scipy.optimize import minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    pass

from confidence_calibration import ConformalPredictor, sigmoid_np


# =============================================================================
# EXPECTED RETURN ESTIMATOR
# =============================================================================

class ExpectedReturnEstimator:
    """
    Combines direction and magnitude model heads into a signed expected return.

    Formula:
        P_up = sigmoid(dir_logits)
        mu   = (2 * P_up - 1) * mag_preds

    Interpretation:
        - P_up > 0.5  →  positive expected return, scaled by predicted magnitude
        - P_up < 0.5  →  negative expected return (short candidate)
        - P_up = 0.5  →  mu = 0, no edge (will receive near-zero weight)
    """

    def compute(
        self,
        dir_logits: np.ndarray,   # [N] raw direction logits
        mag_preds: np.ndarray,    # [N] positive magnitude predictions
        active_mask: np.ndarray,  # [N] bool — inactive stocks forced to 0
    ) -> np.ndarray:
        """
        Returns:
            mu: [N] signed expected return vector (0 for inactive stocks)
        """
        p_up = 1.0 / (1.0 + np.exp(-np.clip(dir_logits, -50, 50)))  # stable sigmoid
        mu = (2.0 * p_up - 1.0) * np.abs(mag_preds)
        mu = mu * active_mask.astype(float)
        return mu


# =============================================================================
# COVARIANCE ESTIMATOR
# =============================================================================

class CovarianceEstimator:
    """
    Per-fold covariance estimation using Ledoit-Wolf shrinkage.

    Ledoit-Wolf is critical here because N_stocks (~150) is close to T_train
    (~200), making the sample covariance ill-conditioned. The shrinkage
    estimator regularizes toward a scaled identity, giving an invertible Sigma
    without manual tuning.

    Falls back to diagonal-loaded sample covariance if sklearn is unavailable.
    """

    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self._sigma: Optional[np.ndarray] = None

    def fit(self, returns: np.ndarray, active_mask: Optional[np.ndarray] = None):
        """
        Estimate covariance from the training-window return matrix.

        Args:
            returns:     [N_stocks, T] return series (training window, per fold)
            active_mask: [N_stocks] bool — inactive stocks excluded from estimation;
                         their rows/columns are set to eps * I in the final matrix.
        """
        N = returns.shape[0]

        if active_mask is not None:
            active_idx = np.where(active_mask)[0]
        else:
            active_idx = np.arange(N)

        # Require at least 2 active stocks and 2 time steps
        if len(active_idx) < 2 or returns.shape[1] < 2:
            self._sigma = self.eps * np.eye(N)
            return self

        ret_active = returns[active_idx, :].T  # [T, n_active] — sklearn expects (n_samples, n_features)

        if _SKLEARN_AVAILABLE:
            lw = LedoitWolf(assume_centered=False)
            try:
                lw.fit(ret_active)
                sigma_active = lw.covariance_  # [n_active, n_active]
            except Exception:
                sigma_active = np.cov(ret_active, rowvar=False)
        else:
            sigma_active = np.cov(ret_active, rowvar=False)

        # Ensure 2-D even if only 1 active stock
        sigma_active = np.atleast_2d(sigma_active)

        # Build full N×N matrix: inactive = eps * I, active block = estimated
        sigma_full = self.eps * np.eye(N)
        sigma_full[np.ix_(active_idx, active_idx)] = sigma_active

        # Symmetrize and floor eigenvalues to ensure PSD
        sigma_full = 0.5 * (sigma_full + sigma_full.T)
        eigvals = np.linalg.eigvalsh(sigma_full)
        if eigvals.min() < self.eps:
            sigma_full += (self.eps - eigvals.min()) * np.eye(N)

        self._sigma = sigma_full
        return self

    @property
    def sigma(self) -> np.ndarray:
        if self._sigma is None:
            raise RuntimeError("CovarianceEstimator.fit() must be called before accessing sigma.")
        return self._sigma


# =============================================================================
# MEAN-VARIANCE OPTIMIZER
# =============================================================================

class MeanVarianceOptimizer:
    """
    Markowitz Mean-Variance Optimization via cvxpy.

    Problem:
        maximize   w^T mu - (gamma/2) w^T Sigma w
        subject to sum(w) = 0          (dollar neutral)
                   ||w||_1 <= 2.0      (gross leverage cap)
                   |w_i| <= max_pos    (per-stock position limit)
                   w[~active] = 0      (no-trade mask)

    Returns the optimal weight vector w [N].
    """

    def __init__(
        self,
        gamma: float = 1.0,
        max_leverage: float = 2.0,
        max_position: float = 0.05,
    ):
        self.gamma = gamma
        self.max_leverage = max_leverage
        self.max_position = max_position

    def optimize(
        self,
        mu: np.ndarray,          # [N] expected returns
        sigma: np.ndarray,       # [N, N] covariance matrix
        active_mask: np.ndarray, # [N] bool
    ) -> np.ndarray:
        """
        Returns:
            w: [N] portfolio weights (0 for inactive stocks)

        Falls back to equal-weight L/S if cvxpy solve fails.
        """
        if not _CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy is required for MeanVarianceOptimizer. "
                "Install via: pip install 'cvxpy>=1.4.0'"
            )

        N = len(mu)
        active_idx = np.where(active_mask)[0]
        n_active = len(active_idx)

        if n_active < 4:
            return self._fallback_equal_ls(mu, active_mask)

        # Adaptive position limit: when the tradable universe is small,
        # tighten per-stock cap to prevent dangerous concentration.
        # With 50+ stocks we use the configured max; below that we scale
        # down to 1/n_active (equal-weight ceiling) as a smooth ramp.
        if n_active >= 50:
            effective_max_pos = self.max_position
        else:
            floor = 1.0 / n_active
            effective_max_pos = min(self.max_position, max(floor, 0.5 / n_active + 0.02))

        # Sub-problem on active stocks only (faster, avoids ill-conditioning)
        mu_a = mu[active_idx]
        sig_a = sigma[np.ix_(active_idx, active_idx)]

        # Ensure PSD for cvxpy
        sig_a = 0.5 * (sig_a + sig_a.T)
        min_eig = np.linalg.eigvalsh(sig_a).min()
        if min_eig < 1e-8:
            sig_a += (1e-8 - min_eig) * np.eye(n_active)

        w_a = cp.Variable(n_active)

        objective = cp.Maximize(
            mu_a @ w_a - (self.gamma / 2.0) * cp.quad_form(w_a, sig_a)
        )
        constraints = [
            cp.sum(w_a) == 0,                        # dollar neutral
            cp.norm1(w_a) <= self.max_leverage,       # gross leverage
            w_a >= -effective_max_pos,                # short limit (adaptive)
            w_a <= effective_max_pos,                 # long limit (adaptive)
        ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            try:
                prob.solve(solver=cp.SCS)
            except Exception:
                return self._fallback_equal_ls(mu, active_mask)

        if w_a.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            return self._fallback_equal_ls(mu, active_mask)

        # Re-embed active weights into full N-dim vector
        w_full = np.zeros(N)
        w_full[active_idx] = w_a.value
        return w_full

    def _fallback_equal_ls(
        self,
        mu: np.ndarray,
        active_mask: np.ndarray,
        top_pct: float = 0.20,
    ) -> np.ndarray:
        """Naive equal-weight long-short fallback when cvxpy fails."""
        N = len(mu)
        active_idx = np.where(active_mask)[0]
        if len(active_idx) < 2:
            return np.zeros(N)

        scores = mu[active_idx]
        k = max(1, int(top_pct * len(active_idx)))
        sorted_idx = np.argsort(scores)

        w = np.zeros(N)
        long_idx = active_idx[sorted_idx[-k:]]
        short_idx = active_idx[sorted_idx[:k]]
        w[long_idx] = 1.0 / k
        w[short_idx] = -1.0 / k
        return w


# =============================================================================
# GRAPH CLUSTERER
# =============================================================================

class GraphClusterer:
    """
    Infers stock clusters from the DynamicGraphLearner's attention adjacency.

    Converts the sparse (edge_index, edge_weights) output into a dense
    affinity matrix, then runs spectral clustering. The number of clusters
    is chosen via the eigengap heuristic on the normalized graph Laplacian.

    Falls back to sector-based clusters if spectral clustering fails or
    sklearn is unavailable.
    """

    def __init__(
        self,
        min_clusters: int = 3,
        max_clusters: int = 10,
        random_state: int = 42,
    ):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

    def infer_clusters(
        self,
        edge_index: np.ndarray,   # [2, E] attention edge indices
        edge_weights: np.ndarray, # [E] softmax-normalized attention weights
        n_nodes: int,
        active_mask: np.ndarray,  # [N] bool
        fallback_sector_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Determine cluster assignment for each stock.

        Returns:
            cluster_ids: [N] int array — cluster label per stock.
                         Inactive stocks get label -1.
        """
        cluster_ids = np.full(n_nodes, -1, dtype=int)
        active_idx = np.where(active_mask)[0]
        n_active = len(active_idx)

        if n_active < self.min_clusters * 2:
            return self._sector_fallback(active_mask, fallback_sector_ids)

        # Build dense affinity matrix from sparse attention edges
        affinity = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        src, dst = edge_index[0], edge_index[1]
        affinity[dst, src] = edge_weights

        # Symmetrize: A = (A + A^T) / 2
        affinity = 0.5 * (affinity + affinity.T)

        # Sub-matrix for active stocks only
        affinity_a = affinity[np.ix_(active_idx, active_idx)]

        if not _SKLEARN_AVAILABLE:
            return self._sector_fallback(active_mask, fallback_sector_ids)

        # Eigengap heuristic: find n_clusters that maximises the gap between
        # consecutive eigenvalues of the normalized Laplacian
        n_clusters = self._eigengap_n_clusters(affinity_a)

        try:
            sc = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=self.random_state,
                n_init=10,
            )
            labels_active = sc.fit_predict(affinity_a.clip(0))  # affinity must be non-negative
        except Exception:
            return self._sector_fallback(active_mask, fallback_sector_ids)

        cluster_ids[active_idx] = labels_active
        return cluster_ids

    def _eigengap_n_clusters(self, affinity: np.ndarray) -> int:
        """Eigengap heuristic on normalized graph Laplacian eigenvalues."""
        n = affinity.shape[0]
        k_max = min(self.max_clusters, n - 1)
        k_min = self.min_clusters

        # Degree matrix
        deg = affinity.sum(axis=1)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(deg_inv_sqrt)

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        L_norm = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt
        L_norm = 0.5 * (L_norm + L_norm.T)  # symmetrize numerical noise

        try:
            eigvals = np.linalg.eigvalsh(L_norm)
            eigvals = np.sort(eigvals)
            # Eigengap = difference between consecutive eigenvalues
            gaps = np.diff(eigvals[k_min - 1 : k_max + 1])
            best_k = k_min + int(np.argmax(gaps))
        except Exception:
            best_k = k_min

        return int(np.clip(best_k, k_min, k_max))

    def _sector_fallback(
        self,
        active_mask: np.ndarray,
        sector_ids: Optional[np.ndarray],
    ) -> np.ndarray:
        """Use sector_ids as cluster labels when spectral clustering is unavailable."""
        n = len(active_mask)
        if sector_ids is not None and len(sector_ids) == n:
            result = np.where(active_mask, sector_ids, -1)
            return result.astype(int)
        # Last resort: assign all active stocks to cluster 0
        result = np.where(active_mask, 0, -1)
        return result.astype(int)


# =============================================================================
# RISK PARITY OPTIMIZER
# =============================================================================

class RiskParityOptimizer:
    """
    Equal-Risk-Contribution (ERC) portfolio across graph-learned clusters.

    Inter-cluster allocation: solve for cluster weights such that each cluster
    contributes equal risk to the total portfolio.

    Intra-cluster allocation: distribute each cluster's weight across stocks
    proportional to |mu_i| (signed direction, magnitude-weighted).

    The result is dollar-neutral by construction (long/short within each cluster
    according to mu sign, then clusters balanced for equal risk).
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def optimize(
        self,
        mu: np.ndarray,           # [N] expected returns
        sigma: np.ndarray,        # [N, N] covariance matrix
        cluster_ids: np.ndarray,  # [N] cluster labels (-1 = inactive)
        active_mask: np.ndarray,  # [N] bool
    ) -> np.ndarray:
        """
        Returns:
            w: [N] portfolio weights summing to 0 (dollar neutral)
        """
        N = len(mu)
        unique_clusters = [c for c in np.unique(cluster_ids) if c >= 0]
        K = len(unique_clusters)

        if K < 2:
            # Degenerate: single cluster → fall back to naive L/S
            return self._naive_ls(mu, active_mask)

        # Build cluster-level covariance by aggregating stocks within each cluster.
        # Proxy: cluster return = equal-weighted average of member returns.
        cluster_to_stocks: Dict[int, List[int]] = {c: [] for c in unique_clusters}
        for i, c in enumerate(cluster_ids):
            if c >= 0 and active_mask[i]:
                cluster_to_stocks[c].append(i)

        # Remove empty clusters (can happen if all members are inactive)
        unique_clusters = [c for c in unique_clusters if cluster_to_stocks[c]]
        K = len(unique_clusters)
        if K < 2:
            return self._naive_ls(mu, active_mask)

        # Cluster covariance [K, K]: cov(c_i, c_j) = avg covariance between members
        sigma_c = np.zeros((K, K))
        for ki, ci in enumerate(unique_clusters):
            for kj, cj in enumerate(unique_clusters):
                idx_i = cluster_to_stocks[ci]
                idx_j = cluster_to_stocks[cj]
                block = sigma[np.ix_(idx_i, idx_j)]
                sigma_c[ki, kj] = block.mean()

        # Symmetrize and ensure PSD
        sigma_c = 0.5 * (sigma_c + sigma_c.T)
        eigmin = np.linalg.eigvalsh(sigma_c).min()
        if eigmin < 1e-8:
            sigma_c += (1e-8 - eigmin) * np.eye(K)

        # Solve ERC: minimize sum_k (RC_k - port_vol/K)^2
        cluster_weights = self._solve_erc(sigma_c, K)  # [K] positive weights

        # Build full weight vector: sign from mu, magnitude from cluster weight
        w_full = np.zeros(N)
        for ki, ci in enumerate(unique_clusters):
            members = cluster_to_stocks[ci]
            if not members:
                continue

            mu_members = mu[members]   # [n_members]
            w_cluster = cluster_weights[ki]

            # Within cluster: dollar-neutral split based on mu sign + magnitude
            pos_mask = mu_members > 0
            neg_mask = mu_members < 0

            pos_sum = np.abs(mu_members[pos_mask]).sum()
            neg_sum = np.abs(mu_members[neg_mask]).sum()

            for j, idx in enumerate(members):
                if mu_members[j] > 0 and pos_sum > 0:
                    w_full[idx] = +w_cluster * (mu_members[j] / pos_sum)
                elif mu_members[j] < 0 and neg_sum > 0:
                    w_full[idx] = -w_cluster * (abs(mu_members[j]) / neg_sum)

        # Dollar-neutral renormalization: subtract mean over active stocks
        active_sum = w_full[active_mask].sum()
        n_active = active_mask.sum()
        if n_active > 0:
            w_full[active_mask] -= active_sum / n_active

        # Leverage normalization: scale to gross leverage = 1
        gross = np.abs(w_full).sum()
        if gross > 1e-8:
            w_full /= gross

        return w_full

    def _solve_erc(self, sigma_c: np.ndarray, K: int) -> np.ndarray:
        """
        Solve for equal-risk-contribution weights across K clusters.

        Uses scipy.optimize.minimize with SLSQP. Falls back to 1/K if
        scipy is unavailable or optimization fails.
        """
        if not _SCIPY_AVAILABLE:
            return np.ones(K) / K

        def erc_objective(w_raw):
            # Softmax parameterization: w = softmax(w_raw) to enforce positivity + sum=1
            w = np.exp(w_raw - w_raw.max())
            w = w / w.sum()
            port_var = w @ sigma_c @ w
            port_vol = max(math.sqrt(port_var), 1e-10)
            rc = w * (sigma_c @ w) / port_vol
            target = port_vol / K
            return float(np.sum((rc - target) ** 2))

        w_raw_init = np.zeros(K)
        try:
            result = minimize(
                erc_objective,
                w_raw_init,
                method="L-BFGS-B",
                options={"maxiter": self.max_iter, "ftol": self.tol},
            )
            w_opt_raw = result.x
            w_opt = np.exp(w_opt_raw - w_opt_raw.max())
            w_opt = w_opt / w_opt.sum()
            return w_opt
        except Exception:
            return np.ones(K) / K

    def _naive_ls(self, mu: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        """Naive equal-weight L/S fallback."""
        N = len(mu)
        active_idx = np.where(active_mask)[0]
        if len(active_idx) < 2:
            return np.zeros(N)
        k = max(1, int(0.20 * len(active_idx)))
        sorted_idx = np.argsort(mu[active_idx])
        w = np.zeros(N)
        w[active_idx[sorted_idx[-k:]]] = 1.0 / k
        w[active_idx[sorted_idx[:k]]] = -1.0 / k
        return w


# =============================================================================
# CONFORMAL GATE
# =============================================================================

class ConformalGate:
    """
    Abstention filter based on conformal prediction set size.

    Trading rule:
        set_size == 1  →  trade (model assigns exactly one direction)
        set_size == 0  →  abstain (model rejects both directions)
        set_size == 2  →  abstain (model is ambiguous between directions)

    When gating leaves fewer than ``min_tradable`` stocks, the gate
    progressively relaxes by re-admitting the most confident set_size==2
    stocks (sorted by |softmax_prob - 0.5|, descending) until the minimum
    threshold is reached.  This prevents MVO from concentrating into
    a dangerously small universe during high-ambiguity periods.

    The gate is calibrated per fold on the first half of validation snapshots,
    then applied to the second half to prevent look-ahead bias.
    """

    def __init__(self, alpha: float = 0.10, min_tradable: int = 20):
        self.alpha = alpha
        self.min_tradable = min_tradable
        self._predictor = ConformalPredictor(alpha=alpha)
        self._calibrated = False

    def calibrate(
        self,
        logits: np.ndarray,  # [N_samples] raw direction logits
        labels: np.ndarray,  # [N_samples] binary ground-truth labels
    ):
        """Fit conformal threshold q_hat from held-out calibration data."""
        self._predictor.calibrate(logits, labels.astype(int))
        self._calibrated = True
        return self

    def gate(
        self,
        logits: np.ndarray,   # [N] per-stock direction logits for this snapshot
        active_mask: np.ndarray,  # [N] bool — existing tradability mask
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply conformal abstention to the active mask.

        Returns:
            trade_mask:  [N] bool — active AND conformal-gated
            stats:       Dict with abstention diagnostics
        """
        if not self._calibrated:
            warnings.warn(
                "ConformalGate has not been calibrated. "
                "Call calibrate() on held-out validation data first. "
                "Returning active_mask unchanged.",
                stacklevel=2,
            )
            return active_mask.copy(), {"gated_pct": 0.0, "q_hat": None, "relaxed": 0}

        result = self._predictor.predict_set(logits)
        set_sizes = result["set_size"]             # [N]
        confident_mask = set_sizes == 1            # [N] bool
        trade_mask = active_mask & confident_mask

        n_active = int(active_mask.sum())
        n_trade = int(trade_mask.sum())
        n_relaxed = 0

        # --- Minimum-universe guard ---
        # When gating is too aggressive, re-admit the most confident
        # ambiguous stocks (set_size == 2) to keep the optimizer from
        # concentrating into a dangerously small universe.
        if n_trade < self.min_tradable and n_active >= self.min_tradable:
            ambiguous_idx = np.where(active_mask & (set_sizes == 2))[0]
            if len(ambiguous_idx) > 0:
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
                confidence = np.abs(probs - 0.5)  # higher = more decisive
                # Sort ambiguous stocks by confidence (most confident first)
                order = np.argsort(-confidence[ambiguous_idx])
                deficit = self.min_tradable - n_trade
                readmit = ambiguous_idx[order[:deficit]]
                trade_mask[readmit] = True
                n_relaxed = len(readmit)
                n_trade = int(trade_mask.sum())

        gated = n_active - n_trade
        gated_pct = gated / max(n_active, 1)

        stats = {
            "n_active": n_active,
            "n_trade": n_trade,
            "n_gated": gated,
            "gated_pct": gated_pct,
            "relaxed": n_relaxed,
            "q_hat": float(result["q_hat"][0]) if len(result["q_hat"]) > 0 else None,
        }
        return trade_mask, stats

    def save(self, path: str):
        self._predictor.save(path)

    def load(self, path: str):
        self._predictor.load(path)
        self._calibrated = True


# =============================================================================
# PORTFOLIO PERFORMANCE METRICS
# =============================================================================

def compute_portfolio_metrics(
    weight_series: List[np.ndarray],   # List of [N] weight vectors
    return_series: List[np.ndarray],   # List of [N] realized returns
    annualization: int = 252,
    cost_bps: float = 5.0,
) -> Dict:
    """
    Compute Sharpe ratio, turnover, and max drawdown from a series of weights
    and realized returns.  Includes a transaction cost model.

    Args:
        weight_series: One weight vector per snapshot
        return_series: Corresponding realized 5-day forward returns per stock
        annualization: Trading days per year (default 252)
        cost_bps:      One-way transaction cost in basis points (default 5 bps).
                       Applied as: cost = cost_bps/10000 * turnover_per_snapshot.
                       Covers commissions, half-spread, and market impact for
                       liquid S&P 500 names at institutional ticket sizes.

    Returns:
        Dict with sharpe, sharpe_net, turnover_mean, max_drawdown, total_pnl,
        total_pnl_net, total_cost, n_snapshots
    """
    empty_result = {
        "sharpe": float("nan"),
        "sharpe_net": float("nan"),
        "turnover_mean": float("nan"),
        "max_drawdown": float("nan"),
        "max_drawdown_net": float("nan"),
        "total_pnl": 0.0,
        "total_pnl_net": 0.0,
        "total_cost": 0.0,
        "n_snapshots": 0,
    }
    if not weight_series or not return_series:
        return empty_result

    cost_per_unit = cost_bps / 10_000.0  # 5 bps = 0.0005

    pnl_gross = []
    pnl_net = []
    costs = []
    for i, (w, r) in enumerate(zip(weight_series, return_series)):
        gross = float(np.dot(w, r))
        pnl_gross.append(gross)

        # Turnover cost: proportional to absolute weight change
        if i == 0:
            # First snapshot: full build cost = sum(|w|) * cost
            turnover_i = float(np.abs(w).sum())
        else:
            turnover_i = float(np.abs(w - weight_series[i - 1]).sum())
        cost_i = turnover_i * cost_per_unit
        costs.append(cost_i)
        pnl_net.append(gross - cost_i)

    pnl_g = np.array(pnl_gross)
    pnl_n = np.array(pnl_net)
    total_pnl = float(pnl_g.sum())
    total_cost = float(sum(costs))
    total_pnl_net = float(pnl_n.sum())

    n = len(pnl_g)
    periods_per_year = annualization / 5  # 5-day forecast horizon

    def _sharpe(arr):
        if len(arr) < 2:
            return float("nan")
        return (arr.mean() / max(arr.std(ddof=1), 1e-10)) * math.sqrt(periods_per_year)

    sharpe_gross = _sharpe(pnl_g)
    sharpe_net = _sharpe(pnl_n)

    # Turnover: mean absolute weight change between consecutive snapshots
    turnovers = []
    for i in range(1, len(weight_series)):
        turnovers.append(float(np.abs(weight_series[i] - weight_series[i - 1]).sum()))
    turnover_mean = float(np.mean(turnovers)) if turnovers else float("nan")

    # Max drawdown on cumulative P&L (gross and net)
    def _max_dd(arr):
        cum = np.cumsum(arr)
        running_max = np.maximum.accumulate(cum)
        dd = running_max - cum
        return float(dd.max()) if len(dd) > 0 else 0.0

    max_dd_gross = _max_dd(pnl_g)
    max_dd_net = _max_dd(pnl_n)

    return {
        "sharpe": round(float(sharpe_gross), 4),
        "sharpe_net": round(float(sharpe_net), 4),
        "turnover_mean": round(float(turnover_mean), 6),
        "max_drawdown": round(float(max_dd_gross), 6),
        "max_drawdown_net": round(float(max_dd_net), 6),
        "total_pnl": round(total_pnl, 6),
        "total_pnl_net": round(total_pnl_net, 6),
        "total_cost": round(total_cost, 6),
        "n_snapshots": n,
    }


# =============================================================================
# PORTFOLIO CONSTRUCTOR (ORCHESTRATOR)
# =============================================================================

class PortfolioConstructor:
    """
    Orchestrates the full portfolio construction pipeline for a single fold.

    Pipeline:
        1. ConformalGate     -- abstention pre-filter (optional)
        2. ExpectedReturn    -- mu from (dir_logits, mag_preds)
        3. CovarianceEstimator -- Ledoit-Wolf Sigma from training returns
        4. MVO or RiskParity -- compute weights w

    Usage (per fold):
        constructor = PortfolioConstructor(method="mvo", gamma=1.0)
        constructor.fit_covariance(train_returns, train_active_mask)
        constructor.fit_conformal(cal_logits, cal_labels)  # optional

        weights, stats = constructor.construct(
            dir_logits, mag_preds, active_mask,
            edge_index=ei, edge_weights=ew, sector_ids=sids
        )
    """

    def __init__(
        self,
        method: str = "mvo",
        gamma: float = 1.0,
        max_leverage: float = 2.0,
        max_position: float = 0.05,
        alpha: float = 0.10,
        use_conformal_gate: bool = True,
        min_tradable: int = 20,
    ):
        """
        Args:
            method:            "mvo" | "riskparity" | "naive"
            gamma:             Risk aversion for MVO (default 1.0)
            max_leverage:      Gross leverage cap for MVO (default 2.0)
            max_position:      Per-stock position limit for MVO (default 5%)
            alpha:             Conformal miscoverage level (default 0.10)
            use_conformal_gate: Whether to apply conformal abstention
            min_tradable:      Minimum stocks the conformal gate must preserve;
                               if gating would leave fewer, the most confident
                               ambiguous stocks are re-admitted (default 20)
        """
        if method not in ("mvo", "riskparity", "naive"):
            raise ValueError(f"method must be 'mvo', 'riskparity', or 'naive', got '{method}'")

        self.method = method
        self.use_conformal_gate = use_conformal_gate

        self._mu_estimator = ExpectedReturnEstimator()
        self._cov_estimator = CovarianceEstimator()
        self._mvo = MeanVarianceOptimizer(
            gamma=gamma,
            max_leverage=max_leverage,
            max_position=max_position,
        )
        self._risk_parity = RiskParityOptimizer()
        self._clusterer = GraphClusterer()
        self._gate = ConformalGate(alpha=alpha, min_tradable=min_tradable) if use_conformal_gate else None

        self._cov_fitted = False

    def fit_covariance(
        self,
        train_returns: np.ndarray,       # [N_stocks, T] return series (training window)
        active_mask: Optional[np.ndarray] = None,
    ):
        """Estimate covariance matrix from the training-window returns."""
        self._cov_estimator.fit(train_returns, active_mask)
        self._cov_fitted = True

    def fit_conformal(
        self,
        cal_logits: np.ndarray,   # [N_cal_samples] flattened direction logits
        cal_labels: np.ndarray,   # [N_cal_samples] binary labels
    ):
        """Calibrate the conformal gate on held-out calibration data."""
        if self._gate is not None:
            self._gate.calibrate(cal_logits, cal_labels)

    def construct(
        self,
        dir_logits: np.ndarray,          # [N] direction logits
        mag_preds: np.ndarray,           # [N] magnitude predictions
        active_mask: np.ndarray,         # [N] bool tradability mask
        edge_index: Optional[np.ndarray] = None,   # [2, E] attention edges (risk parity)
        edge_weights: Optional[np.ndarray] = None, # [E] attention weights (risk parity)
        sector_ids: Optional[np.ndarray] = None,   # [N] int sector labels (fallback)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run the full pipeline for one snapshot and return portfolio weights.

        Returns:
            weights: [N] portfolio weight vector
            stats:   Dict with gate stats and optimizer info
        """
        stats: Dict = {"method": self.method}

        # --- Step 1: Conformal Abstention Gate ---
        if self._gate is not None and self._gate._calibrated:
            trade_mask, gate_stats = self._gate.gate(dir_logits, active_mask)
            stats["gate"] = gate_stats
        else:
            trade_mask = active_mask.copy()
            stats["gate"] = {"gated_pct": 0.0}

        if not trade_mask.any():
            # All stocks gated — return zero weights
            return np.zeros(len(dir_logits)), {**stats, "fallback": "all_gated"}

        # --- Step 2: Expected Return Estimation ---
        mu = self._mu_estimator.compute(dir_logits, mag_preds, trade_mask)

        # --- Step 3: Portfolio Weights ---
        if self.method == "naive":
            w = self._naive_ls(mu, trade_mask)

        elif self.method == "mvo":
            if not self._cov_fitted:
                w = self._naive_ls(mu, trade_mask)
                stats["fallback"] = "cov_not_fitted"
            else:
                w = self._mvo.optimize(mu, self._cov_estimator.sigma, trade_mask)

        elif self.method == "riskparity":
            if not self._cov_fitted:
                w = self._naive_ls(mu, trade_mask)
                stats["fallback"] = "cov_not_fitted"
            else:
                # Cluster via attention graph (or sector fallback)
                N = len(dir_logits)
                if edge_index is not None and edge_weights is not None:
                    cluster_ids = self._clusterer.infer_clusters(
                        edge_index, edge_weights, N, trade_mask, sector_ids
                    )
                else:
                    cluster_ids = self._clusterer._sector_fallback(trade_mask, sector_ids)

                stats["n_clusters"] = int((cluster_ids >= 0).sum() > 0) and len(
                    np.unique(cluster_ids[cluster_ids >= 0])
                )
                w = self._risk_parity.optimize(
                    mu, self._cov_estimator.sigma, cluster_ids, trade_mask
                )

        else:
            w = np.zeros(len(mu))

        return w, stats

    @staticmethod
    def _naive_ls(
        mu: np.ndarray,
        active_mask: np.ndarray,
        top_pct: float = 0.20,
    ) -> np.ndarray:
        N = len(mu)
        active_idx = np.where(active_mask)[0]
        if len(active_idx) < 2:
            return np.zeros(N)
        k = max(1, int(top_pct * len(active_idx)))
        sorted_idx = np.argsort(mu[active_idx])
        w = np.zeros(N)
        w[active_idx[sorted_idx[-k:]]] = 1.0 / k
        w[active_idx[sorted_idx[:k]]] = -1.0 / k
        return w
