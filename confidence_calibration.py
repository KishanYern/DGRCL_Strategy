"""
Confidence Calibration Module for DGRCL Strategy

Platt Scaling / Temperature Calibration:
    The MC Dropout confidence head outputs raw logits that map to probabilities
    via sigmoid. If the model is overconfident (e.g., 87% avg confidence when
    actual accuracy is lower), this module recalibrates.

    Two complementary approaches:
      • TemperatureScaler: pure-PyTorch, learns scalar T that scales logits
        before sigmoid so that P(correct | confidence=c) ≈ c.
      • PlattScaler: sklearn-based logistic regression (a, b learned separately).

    Both accept raw direction logits [N] and return calibrated probabilities [N].

Conformal Prediction Intervals (ConformalPredictor):
    Provides distribution-free, coverage-guaranteed prediction sets.
    Unlike confidence logits (which can be arbitrarily overconfident),
    conformal prediction guarantees: P(true label ∈ prediction_set) ≥ 1 − α
    on any exchangeable (IID or walk-forward) data, with no model assumptions.

    Usage pattern:
        1. On a held-out calibration fold: predictor.calibrate(scores, labels)
        2. At test time: predictor.predict_set(test_scores)

Standalone post-processing usage:
    $ python confidence_calibration.py --results-dir ./backtest_results \\
          --logits-json fold_logits.json --labels-json fold_labels.json
"""

import json
import os
import math
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for server environments
import matplotlib.pyplot as plt


# =============================================================================
# TEMPERATURE SCALER
# =============================================================================

class TemperatureScaler(nn.Module):
    """
    Scalar temperature calibration for direction logits.

    Learns a single temperature T ∈ (0, ∞) such that:
        P(up | stock_i) = sigmoid(logit_i / T)

    A perfect calibration has T=1. T>1 means the model is overconfident
    (logits are too large → probabilities are too extreme). T<1 means
    underconfident. The DGRCL model is expected to have T>1 given the
    87% mean confidence observed in the quant analysis.

    Training via minimizing binary cross-entropy on a held-out calibration set.
    """

    def __init__(self):
        super().__init__()
        # Initialize T=1.5 to give the optimizer a warm start toward T>1,
        # since we know a priori the model is overconfident.
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to raw direction logits.

        Args:
            logits: [N] raw direction logits from model (unbounded)

        Returns:
            [N] scaled logits — pass through sigmoid to get calibrated probs
        """
        return logits / self.temperature.clamp(min=1e-3)  # Prevent division by zero

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Return calibrated probabilities P(return > 0) for each stock.

        Args:
            logits: [N] raw direction logits

        Returns:
            [N] calibrated probabilities in (0, 1)
        """
        scaled = self.forward(logits)
        return torch.sigmoid(scaled)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 1000,
        verbose: bool = False
    ) -> float:
        """
        Fit temperature T to minimize BCE on held-out calibration data.

        Args:
            logits: [N] raw direction logits (from all val snapshots)
            labels: [N] binary labels — 1 if return > 0, else 0 (float tensor)
            lr: Learning rate for LBFGS optimizer
            max_iter: LBFGS max iterations
            verbose: Whether to print final ECE

        Returns:
            Final ECE (Expected Calibration Error) after fitting
        """
        self.train()

        # BCE loss for calibration optimization
        criterion = nn.BCEWithLogitsLoss()

        # LBFGS converges in very few steps for 1-parameter optimization
        optimizer = optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)

        labels = labels.float()

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        self.eval()
        with torch.no_grad():
            cal_probs = self.calibrate(logits)
            ece = compute_ece(cal_probs.numpy(), labels.numpy())

        if verbose:
            print(f"  Temperature after fitting: {self.temperature.item():.4f}")
            print(f"  Post-calibration ECE: {ece:.4f}")

        return ece


# =============================================================================
# PLATT SCALER (sklearn-based alternative)
# =============================================================================

class PlattScaler:
    """
    Platt Scaling (a, b) variant: learns separate scale and bias.
        P(up | stock_i) = sigmoid(a * logit_i + b)

    Slightly more expressive than TemperatureScaler (2 parameters vs 1).
    Uses scikit-learn logistic regression to fit — no gradient loop required.

    Prefer TemperatureScaler for simplicity; use PlattScaler when val set is
    large enough to reliably estimate an offset (b ≠ 0) as well.
    """

    def __init__(self):
        self._a = 1.0   # Scale parameter (logit coefficient)
        self._b = 0.0   # Offset parameter (intercept)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        verbose: bool = False
    ) -> float:
        """
        Fit Platt parameters via logistic regression with existing logits as features.

        Args:
            logits: [N] raw direction logits (numpy array)
            labels: [N] binary labels (0 or 1)
            verbose: Whether to print fitted parameters and ECE

        Returns:
            ECE after fitting
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError(
                "scikit-learn is required for PlattScaler. "
                "Install via: pip install scikit-learn"
            )

        # Standardize logits for numerical stability
        scaler = StandardScaler()
        X = scaler.fit_transform(logits.reshape(-1, 1))  # [N, 1]

        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        clf.fit(X, labels)

        # Un-standardize parameters back to original logit space
        # logistic_pred = sigmoid(clf.coef_[0] * (logit - mean) / std + clf.intercept_[0])
        # = sigmoid((clf.coef_[0]/std) * logit + (clf.intercept_[0] - clf.coef_[0]*mean/std))
        std = scaler.scale_[0]
        mean = scaler.mean_[0]
        self._a = float(clf.coef_[0][0]) / std
        self._b = float(clf.intercept_[0]) - float(clf.coef_[0][0]) * mean / std

        # Compute ECE on calibration data
        cal_probs = self.calibrate(logits)
        ece = compute_ece(cal_probs, labels)

        if verbose:
            print(f"  Platt a={self._a:.4f}, b={self._b:.4f}")
            print(f"  Post-calibration ECE: {ece:.4f}")

        return ece

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Return calibrated probabilities.

        Args:
            logits: [N] raw direction logits

        Returns:
            [N] calibrated probabilities in (0, 1)
        """
        return sigmoid_np(self._a * logits + self._b)

    def save(self, path: str):
        """Persist Platt parameters to JSON."""
        with open(path, 'w') as f:
            json.dump({'a': self._a, 'b': self._b, 'type': 'platt'}, f, indent=2)

    def load(self, path: str):
        """Load previously saved Platt parameters."""
        with open(path, 'r') as f:
            data = json.load(f)
        self._a = data['a']
        self._b = data['b']


# =============================================================================
# CONFORMAL PREDICTOR
# =============================================================================

class ConformalPredictor:
    """
    Distribution-Free Conformal Prediction Sets for Direction Signals.

    Provides the guarantee: P(true_label ∈ prediction_set) ≥ 1 − α
    on any exchangeable test data — without any model distributional assumptions.

    CONCEPT:
        1. For each held-out (calibration) sample:
           nonconformity_score = 1 - sigmoid(dir_logit) if label=1
                               = sigmoid(dir_logit)     if label=0
           (i.e., how inconsistent is the model's score with the true label?)

        2. The (1-α) quantile of these scores, q_hat, becomes the threshold.

        3. At test time: include label y=1 in prediction set if
           sigmoid(dir_logit) >= 1 - q_hat  (model is confident enough in UP)
           include label y=0 in prediction set if
           1 - sigmoid(dir_logit) >= 1 - q_hat  (model is confident enough in DOWN)

    The width of the prediction set (both, one, or neither label included) is
    a honest measure of uncertainty — wider = model is less certain.

    Args:
        alpha: Miscoverage level (e.g., 0.1 = 90% coverage guarantee)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._q_hat: Optional[float] = None  # Quantile threshold, set after calibrate()

    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute conformal threshold q_hat from held-out calibration data.

        Args:
            logits: [N] raw direction logits from model
            labels: [N] binary ground-truth labels (0 or 1)

        Returns:
            q_hat: The conformal quantile threshold
        """
        probs = sigmoid_np(logits)  # P(label=1), [N]

        # Nonconformity score: prob of the TRUE label (higher = more conforming)
        # Nonconformity = 1 - P(true_label)
        scores = np.where(labels == 1, 1.0 - probs, probs)  # [N]

        # Conformal quantile: (1-alpha) * (1 + 1/N) upper bound
        # The +1/N Bonferroni correction provides strict finite-sample coverage.
        n = len(scores)
        level = math.ceil((1 - self.alpha) * (n + 1)) / n
        level = min(level, 1.0)  # Clamp: cannot exceed 1
        self._q_hat = float(np.quantile(scores, level))

        return self._q_hat

    def predict_set(self, logits: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate conformal prediction sets for new test samples.

        Args:
            logits: [N] raw direction logits at test time

        Returns:
            Dict with:
                include_up:    [N] bool — include UP (label=1) in prediction set
                include_down:  [N] bool — include DOWN (label=0) in prediction set
                set_size:      [N] int  — 0, 1, or 2 (how many labels in set)
                coverage_ok:   [N] bool — set is non-empty (False = abstain signal)
        """
        if self._q_hat is None:
            raise RuntimeError(
                "ConformalPredictor.calibrate() must be called before predict_set(). "
                "Fit on held-out validation data first."
            )

        probs = sigmoid_np(logits)  # P(label=1)

        # Include UP if P(up) score is conforming: 1 - P(up) < q_hat
        # Equivalently: P(up) > 1 - q_hat
        include_up = (1.0 - probs) < self._q_hat    # [N] bool
        # Include DOWN if P(down) = 1-P(up) is conforming: P(up) < q_hat
        include_down = probs < self._q_hat           # [N] bool

        set_size = include_up.astype(int) + include_down.astype(int)  # 0, 1, or 2

        return {
            'include_up': include_up,
            'include_down': include_down,
            'set_size': set_size,
            'coverage_ok': set_size > 0,
            'q_hat': np.full(len(logits), self._q_hat)
        }

    def empirical_coverage(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute empirical coverage on test data (should be ≥ 1−alpha).

        Args:
            logits: [N] raw direction logits
            labels: [N] true binary labels

        Returns:
            Fraction of test samples where true label is in prediction set
        """
        prediction = self.predict_set(logits)
        # Check if true label is covered
        covered = np.where(
            labels == 1,
            prediction['include_up'],
            prediction['include_down']
        )
        return float(covered.mean())

    def save(self, path: str):
        """Persist conformal threshold to JSON."""
        with open(path, 'w') as f:
            json.dump({'q_hat': self._q_hat, 'alpha': self.alpha, 'type': 'conformal'}, f, indent=2)

    def load(self, path: str):
        """Load previously saved conformal threshold."""
        with open(path, 'r') as f:
            data = json.load(f)
        self._q_hat = data['q_hat']
        self.alpha = data['alpha']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE) — lower is better, 0 is perfect.

    Splits predictions into n_bins confidence buckets and measures the
    mean absolute difference between bucket confidence and bucket accuracy.

    Args:
        probs: [N] predicted probabilities (after sigmoid/calibration)
        labels: [N] true binary labels (0 or 1)
        n_bins: Number of confidence buckets

    Returns:
        ECE in [0, 1]
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi)
        if in_bin.sum() == 0:
            continue  # Skip empty bins
        bin_conf = probs[in_bin].mean()    # Mean confidence in bucket
        bin_acc = labels[in_bin].mean()    # Mean accuracy in bucket
        bin_weight = in_bin.sum() / n      # Fraction of samples in bucket
        ece += bin_weight * abs(bin_conf - bin_acc)

    return float(ece)


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None
) -> float:
    """
    Plot confidence reliability diagram and compute ECE.

    A perfectly calibrated model has points on the diagonal (y=x).
    Points above the diagonal = model is underconfident.
    Points below the diagonal = model is overconfident (our case — 87% mean confidence).

    Args:
        probs: [N] predicted probabilities from the model or calibrator
        labels: [N] binary true labels
        n_bins: Number of confidence buckets for histogram
        title: Plot title
        save_path: If provided, save figure to this path

    Returns:
        ECE value
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi)
        if in_bin.sum() == 0:
            continue
        bin_centers.append(probs[in_bin].mean())
        bin_accuracies.append(labels[in_bin].mean())
        bin_counts.append(in_bin.sum())

    ece = compute_ece(probs, labels, n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)
    ax1.bar(
        bin_centers,
        bin_accuracies,
        width=1.0 / n_bins,
        alpha=0.7,
        color='steelblue',
        edgecolor='white',
        label='Model accuracy'
    )
    ax1.set_xlabel('Confidence (predicted probability)', fontsize=12)
    ax1.set_ylabel('Accuracy (fraction correct)', fontsize=12)
    ax1.set_title(f'{title}\nECE = {ece:.4f}', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Right: histogram of confidence distribution
    ax2.bar(
        bin_centers,
        bin_counts,
        width=1.0 / n_bins,
        alpha=0.7,
        color='coral',
        edgecolor='white'
    )
    ax2.set_xlabel('Predicted probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Reliability diagram saved: {save_path}")

    plt.close()
    return ece


# =============================================================================
# STANDALONE POST-PROCESSING ENTRY POINT
# =============================================================================

def run_post_processing(
    results_dir: str,
    logits_file: Optional[str] = None,
    labels_file: Optional[str] = None,
    alpha: float = 0.1,
    n_bins: int = 10
):
    """
    Standalone post-processing: load saved logits/labels, fit calibrators,
    generate reliability diagram and conformal threshold.

    Logits/labels are saved by train.py per fold when --save-calibration-data
    is passed. The calibration is fit on the second half of folds (held-out).

    Args:
        results_dir: Path to backtest_results directory
        logits_file: JSON file with list of raw logits (one per val sample)
        labels_file: JSON file with list of binary labels (0/1)
        alpha: Conformal coverage level (1-alpha = coverage guarantee)
        n_bins: Number of bins for ECE computation
    """
    # Determine file paths
    if logits_file is None:
        logits_file = os.path.join(results_dir, 'calibration_logits.json')
    if labels_file is None:
        labels_file = os.path.join(results_dir, 'calibration_labels.json')

    if not os.path.exists(logits_file) or not os.path.exists(labels_file):
        print(
            f"Calibration data files not found:\n"
            f"  {logits_file}\n"
            f"  {labels_file}\n"
            "Run train.py with --save-calibration-data to generate them."
        )
        return

    with open(logits_file, 'r') as f:
        logits_raw = json.load(f)
    with open(labels_file, 'r') as f:
        labels_raw = json.load(f)

    logits_np = np.array(logits_raw, dtype=np.float32)
    labels_np = np.array(labels_raw, dtype=np.float32)

    print(f"\n=== Confidence Calibration Post-Processing ===")
    print(f"  Samples: {len(logits_np)}")
    print(f"  Raw mean confidence (sigmoid): {sigmoid_np(logits_np).mean():.4f}")
    print(f"  Raw ECE: {compute_ece(sigmoid_np(logits_np), labels_np, n_bins):.4f}")

    # --- Temperature Scaling ---
    print("\n[1] Temperature Scaling")
    logits_t = torch.tensor(logits_np)
    labels_t = torch.tensor(labels_np)
    temp_scaler = TemperatureScaler()
    ece_after = temp_scaler.fit(logits_t, labels_t, verbose=True)

    with torch.no_grad():
        cal_probs_ts = temp_scaler.calibrate(logits_t).numpy()

    # Save temperature parameter
    temp_path = os.path.join(results_dir, 'temperature_calibration.json')
    with open(temp_path, 'w') as f:
        json.dump({
            'temperature': temp_scaler.temperature.item(),
            'ece_before': compute_ece(sigmoid_np(logits_np), labels_np, n_bins),
            'ece_after': ece_after,
            'type': 'temperature'
        }, f, indent=2)
    print(f"  Saved temperature calibration: {temp_path}")

    # Reliability diagram — before and after
    diagram_path = os.path.join(results_dir, 'reliability_diagram_before.png')
    reliability_diagram(
        sigmoid_np(logits_np), labels_np,
        title="Before Calibration (Raw Logits)", save_path=diagram_path
    )

    diagram_path_after = os.path.join(results_dir, 'reliability_diagram_after_temp.png')
    reliability_diagram(
        cal_probs_ts, labels_np,
        title="After Temperature Scaling", save_path=diagram_path_after
    )

    # --- Platt Scaling (alternative, requires scikit-learn) ---
    print("\n[2] Platt Scaling (alternative)")
    ece_platt = None
    try:
        platt = PlattScaler()
        ece_platt = platt.fit(logits_np, labels_np.astype(int), verbose=True)
        platt_path = os.path.join(results_dir, 'platt_calibration.json')
        platt.save(platt_path)
        print(f"  Saved Platt calibration: {platt_path}")

        diagram_path_platt = os.path.join(results_dir, 'reliability_diagram_after_platt.png')
        reliability_diagram(
            platt.calibrate(logits_np), labels_np,
            title="After Platt Scaling", save_path=diagram_path_platt
        )
    except ImportError:
        print("  ⚠ Skipped — scikit-learn not installed.")
        print("  To enable Platt Scaling: pip install scikit-learn")
        print("  (Temperature Scaling above is equally effective and has no extra dependencies.)")

    # --- Conformal Prediction ---
    print(f"\n[3] Conformal Prediction (α={alpha})")
    # Use first half as calibration, second half to measure empirical coverage
    split = len(logits_np) // 2
    conformal = ConformalPredictor(alpha=alpha)
    q = conformal.calibrate(logits_np[:split], labels_np[:split].astype(int))
    print(f"  q_hat (conformal threshold): {q:.4f}")

    cov = conformal.empirical_coverage(logits_np[split:], labels_np[split:].astype(int))
    print(f"  Empirical coverage on test half: {cov:.4f} (target ≥ {1-alpha:.2f})")

    conformal_path = os.path.join(results_dir, 'conformal_predictor.json')
    conformal.save(conformal_path)
    print(f"  Saved conformal predictor: {conformal_path}")

    print("\n=== Post-Processing Complete ===")
    print(f"  ECE Before:              {compute_ece(sigmoid_np(logits_np), labels_np, n_bins):.4f}")
    print(f"  ECE After (Temperature): {ece_after:.4f}")
    if ece_platt is not None:
        print(f"  ECE After (Platt):       {ece_platt:.4f}")
    else:
        print(f"  ECE After (Platt):       N/A (sklearn not installed)")
    print(f"  Conformal Coverage:      {cov:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DGRCL Confidence Calibration Post-Processing (Recs 2 & 6)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./backtest_results",
        help="Directory containing fold results and where calibration artifacts are saved"
    )
    parser.add_argument(
        "--logits-json",
        type=str,
        default=None,
        help="Path to JSON file with raw direction logits (default: results_dir/calibration_logits.json)"
    )
    parser.add_argument(
        "--labels-json",
        type=str,
        default=None,
        help="Path to JSON file with binary labels (default: results_dir/calibration_labels.json)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Conformal miscoverage level (default: 0.1 → 90%% coverage guarantee)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins for reliability diagram / ECE (default: 10)"
    )

    args = parser.parse_args()
    run_post_processing(
        results_dir=args.results_dir,
        logits_file=args.logits_json,
        labels_file=args.labels_json,
        alpha=args.alpha,
        n_bins=args.bins
    )
