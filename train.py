"""
Training Loop for Macro-Aware DGRCL v1.5 (Pairwise Ranking)

Implements:
- Pairwise Margin Ranking Loss: Sector-aware margin ranking (direction)
- SmoothL1Loss on log-scaled |returns| (magnitude)
- EarlyStopping on combined validation loss
- Walk-forward cross-validation
- Mixed Precision Training (FP16) for GPU memory efficiency
"""

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import gc

from macro_dgrcl import MacroDGRCL


# Visualization output directory
# NOTE: This global constant is now overridden inside main() based on arguments
# Leaving it here as a default fallback if functions are called directly without main()
VIS_DIR = "./backtest_results"


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    
    Monitors the combined validation loss (L_BCE + λ·L_MSE).
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Combined validation loss
            model: Model to save best state of
            
        Returns:
            True if training should stop
        """
        # FIX #NaN-2: NaN val_loss must NEVER be accepted as a 'best' checkpoint.
        # Prior behaviour: on first call when best_loss is None, NaN was stored as best_loss.
        # Then nan < nan-delta is always False, so counter hit patience and restore_best()
        # loaded the NaN-weight checkpoint, guaranteeing corrupt weights for the whole fold.
        if math.isnan(val_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_best(self, model: nn.Module):
        """Restore model to best checkpoint (mapped to model's current device)."""
        if self.best_state is not None:
            # FIX #NaN-2: Don't restore a checkpoint whose recorded best_loss is NaN.
            if self.best_loss is not None and math.isnan(self.best_loss):
                return
            device = next(model.parameters()).device
            state = {k: v.to(device) for k, v in self.best_state.items()}
            model.load_state_dict(state)


# =============================================================================
# TRAINING STEP
# =============================================================================

# Ranking loss hyperparameters
RANKING_MARGIN = 0.5        # Logit separation required between winner/loser
SIGNIFICANCE_THRESHOLD = 0.003  # 0.3% return divergence to qualify as valid pair
# FIX #NoLearn-3: Lowered from 1% → 0.3% to produce ~3× more valid pairs per snapshot.
# 1% was too restrictive in calm market regimes, causing most snapshots to emit
# zero valid pairs → zero loss → zero gradients → model stuck at random init.


def compute_pairwise_ranking_loss(
    scores: torch.Tensor,        # [N_s] model output logits
    returns: torch.Tensor,       # [N_s] actual returns
    sector_mask: Optional[torch.Tensor] = None,  # [N_s, N_s] bool
    active_mask: Optional[torch.Tensor] = None,   # [N_s] bool
    margin: float = RANKING_MARGIN,
    threshold: float = SIGNIFICANCE_THRESHOLD
) -> Tuple[torch.Tensor, float]:
    """
    Sector-Aware Pairwise Margin Ranking Loss.
    
    For each valid pair (i, j) within the same sector where
    returns[i] - returns[j] > threshold, enforce:
        score[i] - score[j] > margin
    
    Loss = mean(relu(margin - (score_i - score_j))) over valid pairs.
    
    Args:
        scores: [N_s] unbounded direction logits from model
        returns: [N_s] actual future returns
        sector_mask: [N_s, N_s] True where stocks share a sector
        margin: Required logit separation between winner and loser
        threshold: Minimum return difference to form a valid pair
        
    Returns:
        Tuple of (ranking_loss, pairwise_rank_accuracy)
    """
    # 1. Pairwise return differences: ret_diff[i, j] = returns[i] - returns[j]
    ret_diff = returns.unsqueeze(1) - returns.unsqueeze(0)  # [N_s, N_s]
    
    # 2. Pairwise score differences: score_diff[i, j] = scores[i] - scores[j]
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # [N_s, N_s]
    
    # 3. Valid pairs mask: same sector AND significant return divergence
    # Only consider directed pairs where i outperformed j by > threshold
    valid_pairs = (ret_diff > threshold)
    if sector_mask is not None:
        valid_pairs = valid_pairs & sector_mask
    
    # 3b. Filter to active-active pairs only (both stocks must be tradable)
    if active_mask is not None:
        active_pairs = active_mask.unsqueeze(1) & active_mask.unsqueeze(0)  # [N, N]
        valid_pairs = valid_pairs & active_pairs
    
    # 4. Compute margin ranking loss: max(0, margin - score_diff)
    ranking_loss_matrix = torch.relu(margin - score_diff)
    
    if valid_pairs.sum() > 0:
        loss = ranking_loss_matrix[valid_pairs].mean()
        # Pairwise ranking accuracy: % of valid pairs where score_i > score_j
        rank_acc = (score_diff[valid_pairs] > 0).float().mean().item()
    else:
        # FIX #NoLearn-2: Use scores.sum() * 0.0 instead of torch.tensor(0.0, requires_grad=True).
        # The leaf-tensor fallback has requires_grad=True but NO connection to the model's
        # computation graph, so loss.backward() produces zero gradients for ALL parameters.
        # scores.sum() * 0.0 IS connected to the graph (scores flows from the model forward
        # pass), so gradients still propagate correctly even when loss value is 0.
        loss = scores.sum() * 0.0
        rank_acc = 0.0
    
    return loss, rank_acc


def compute_log_scaled_mag_target(
    returns: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Log-scale magnitude targets: log(1 + |returns| / σ).

    Normalizes raw absolute returns (~0.001–0.02 range) into a
    well-behaved ~[0, 3] range so SmoothL1Loss can produce meaningful
    gradients instead of behaving as pure MSE.

    FIX #5: σ is computed only over ACTIVE stocks. Zero-padded inactive
    stocks have returns==0 after union-index reindexing; including them
    deflates σ and inflates the normalized targets for active stocks,
    causing the magnitude head to see artificially large targets.

    Args:
        returns: [N_s] raw future returns
        active_mask: [N_s] bool — if provided, sigma is computed over
                     active stocks only; all N_s targets are still returned.

    Returns:
        [N_s] log-scaled magnitude targets
    """
    abs_ret = torch.abs(returns)
    if active_mask is not None and active_mask.any():
        sigma = abs_ret[active_mask].std().clamp(min=1e-6)
    else:
        sigma = abs_ret.std().clamp(min=1e-6)
    return torch.log1p(abs_ret / sigma)


def train_step(
    model: MacroDGRCL,
    stock_features: torch.Tensor,  # [N_s, T, d_s]
    macro_features: torch.Tensor,  # [N_m, T, d_m]
    returns: torch.Tensor,  # [N_s] future returns (labels)
    optimizer: torch.optim.Optimizer,
    mse_loss_fn: nn.Module,
    mag_weight: float = 0.05,
    macro_stock_edges: Optional[torch.Tensor] = None,
    sector_mask: Optional[torch.Tensor] = None,
    sector_ids: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
    max_grad_norm: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accumulation_steps: int = 4,
    batch_idx: int = 0
) -> Dict[str, float]:
    """
    Single training step with Pairwise Ranking + Magnitude regression.
    
    Direction Loss:  Sector-Aware Pairwise Margin Ranking
        For pairs (i, j) in same sector where ret_i - ret_j > 1%,
        enforce: score_i - score_j > margin (0.5)
    
    Magnitude Loss:  SmoothL1Loss on log(1 + |returns| / σ)
    
    Args:
        model: MacroDGRCL model
        stock_features: [N_s, T, d_s] time-series
        macro_features: [N_m, T, d_m] time-series
        returns: [N_s] future returns (labels)
        optimizer: Optimizer
        mse_loss_fn: SmoothL1Loss for magnitude
        mag_weight: λ weight for magnitude loss
        macro_stock_edges: Optional fixed edges
        sector_mask: [N_s, N_s] boolean mask
        sector_ids: [N_s] sector IDs (unused, kept for API compat)
        max_grad_norm: Gradient clipping threshold
        scaler: Optional GradScaler for mixed precision training
        
    Returns:
        Dict with loss metrics and task-specific metrics
    """
    model.train()
    
    # Gradient Accumulation: Only zero gradients at the start of an accumulation cycle
    if batch_idx % accumulation_steps == 0:
        optimizer.zero_grad()
    
    # --------------- Forward Pass ---------------
    use_amp = scaler is not None
    with torch.cuda.amp.autocast(enabled=use_amp):
        dir_logits, mag_preds = model(
            stock_features=stock_features,
            macro_features=macro_features,
            macro_stock_edges=macro_stock_edges,
            sector_mask=sector_mask,
            active_mask=active_mask
        )
        
        scores = dir_logits.squeeze(-1)  # [N_s]
        
        # --------------- Direction: Pairwise Margin Ranking Loss ---------------
        loss_dir, rank_accuracy = compute_pairwise_ranking_loss(
            scores=scores,
            returns=returns,
            sector_mask=sector_mask,
            active_mask=active_mask
        )
        
        # --------------- Magnitude: SmoothL1 on log-scaled targets ---------------
        # FIX #5: pass active_mask so sigma is computed over active stocks only
        mag_target = compute_log_scaled_mag_target(returns, active_mask=active_mask)
        # Mask magnitude loss: only compute over active stocks
        if active_mask is not None and active_mask.any():
            loss_mag = mse_loss_fn(
                mag_preds.squeeze(-1)[active_mask],
                mag_target[active_mask]
            )
        else:
            loss_mag = mse_loss_fn(mag_preds.squeeze(-1), mag_target)

        loss_total = loss_dir + mag_weight * loss_mag

    # FIX #NaN-3: Skip the entire backward pass if loss is non-finite.
    # A single degenerate snapshot (all-zero future returns, pure-padding window,
    # or FP16 overflow on a high-volatility window) can produce NaN/Inf that would
    # corrupt all model weights. Silently skip and return NaN metrics so callers
    # can detect the bad batch without crashing the rest of the epoch.
    if not torch.isfinite(loss_total):
        # Still zero_grad to keep accumulation state clean
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.zero_grad()
        return {
            'loss': float('nan'),
            'loss_dir': loss_dir.item() if torch.isfinite(loss_dir) else float('nan'),
            'loss_mag': loss_mag.item() if torch.isfinite(loss_mag) else float('nan'),
            'rank_accuracy': 0.0,
            'mag_mae': float('nan'),
            'grad_norm': 0.0
        }

    # Scale the loss since gradients are accumulated
    loss_item = loss_total.item()
    loss_total = loss_total / accumulation_steps

    # --------------- Backward Pass ---------------
    # FIX #2: Removed duplicate optimizer.zero_grad() that was here.
    # The first zero_grad() at the top of the function is sufficient.
    # A second call here wiped all gradients computed in the autocast block,
    # silently preventing the model from learning.
    
    if use_amp:
        scaler.scale(loss_total).backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.tensor(0.0)
    else:
        loss_total.backward()
        if (batch_idx + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        else:
            grad_norm = torch.tensor(0.0)
    
    # --- Metrics ---
    with torch.no_grad():
        if active_mask is not None and active_mask.any():
            mag_mae = torch.abs(
                mag_preds.squeeze(-1)[active_mask] - mag_target[active_mask]
            ).mean().item()
        else:
            mag_mae = torch.abs(mag_preds.squeeze(-1) - mag_target).mean().item()
    
    return {
        'loss': loss_item,
        'loss_dir': loss_dir.item(),
        'loss_mag': loss_mag.item(),
        'rank_accuracy': rank_accuracy,
        'mag_mae': mag_mae,
        'grad_norm': grad_norm.item()
    }


def train_epoch(
    model: MacroDGRCL,
    data_loader: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    mse_loss_fn: nn.Module,
    device: torch.device,
    mag_weight: float = 0.05,
    max_grad_norm: float = 1.0,
    sector_mask: Optional[torch.Tensor] = None,
    sector_ids: Optional[torch.Tensor] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    accumulation_steps: int = 4
) -> Dict[str, float]:
    """
    Train for one epoch over sequential snapshots.
    
    Args:
        model: MacroDGRCL model
        data_loader: List of (stock_features, macro_features, returns) tuples
        optimizer: Optimizer
        mse_loss_fn: SmoothL1Loss for magnitude
        device: Torch device
        mag_weight: λ for magnitude loss
        max_grad_norm: Gradient clipping threshold
        sector_mask: [N_s, N_s] boolean mask
        sector_ids: [N_s] sector IDs
        scaler: Optional GradScaler for mixed precision training
        
    Returns:
        Dict with epoch-level metrics (averaged across batches)
    """
    model.train()
    epoch_metrics = {
        'loss': 0.0, 'loss_dir': 0.0, 'loss_mag': 0.0,
        'rank_accuracy': 0.0, 'mag_mae': 0.0
    }
    num_batches = 0
    
    # Ensure masks are on device
    if sector_mask is not None:
        sector_mask = sector_mask.to(device)
    if sector_ids is not None:
        sector_ids = sector_ids.to(device)
    
    for snapshot in data_loader:
        # Support both 3-tuple (legacy) and 4-tuple (with active_mask) formats
        if len(snapshot) == 4:
            stock_feat, macro_feat, returns, active_mask = snapshot
            active_mask = active_mask.to(device)
        else:
            stock_feat, macro_feat, returns = snapshot
            active_mask = None
        
        stock_feat = stock_feat.to(device)
        macro_feat = macro_feat.to(device)
        returns = returns.to(device)
        
        metrics = train_step(
            model=model,
            stock_features=stock_feat,
            macro_features=macro_feat,
            returns=returns,
            optimizer=optimizer,
            mse_loss_fn=mse_loss_fn,
            mag_weight=mag_weight,
            max_grad_norm=max_grad_norm,
            sector_mask=sector_mask,
            sector_ids=sector_ids,
            active_mask=active_mask,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
            batch_idx=num_batches
        )
        
        for key in epoch_metrics:
            epoch_metrics[key] += metrics[key]
        num_batches += 1
    
    # Average across batches
    for key in epoch_metrics:
        epoch_metrics[key] /= max(num_batches, 1)
    epoch_metrics['num_batches'] = num_batches
    
    return epoch_metrics


@torch.no_grad()
def evaluate(
    model: MacroDGRCL,
    data_loader: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    mse_loss_fn: nn.Module,
    device: torch.device,
    mag_weight: float = 0.05,
    sector_mask: Optional[torch.Tensor] = None,
    sector_ids: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Evaluate model on validation data with pairwise ranking + magnitude metrics.
    """
    model.eval()
    eval_metrics = {
        'loss': 0.0, 'loss_dir': 0.0, 'loss_mag': 0.0,
        'rank_accuracy': 0.0, 'mag_mae': 0.0
    }
    num_batches = 0
    
    if sector_mask is not None:
        sector_mask = sector_mask.to(device)
    
    for snapshot in data_loader:
        # Support both 3-tuple (legacy) and 4-tuple (with active_mask) formats
        if len(snapshot) == 4:
            stock_feat, macro_feat, returns, active_mask = snapshot
            active_mask = active_mask.to(device)
        else:
            stock_feat, macro_feat, returns = snapshot
            active_mask = None
        
        stock_feat = stock_feat.to(device)
        macro_feat = macro_feat.to(device)
        returns = returns.to(device)
        
        # Forward pass with mixed precision (consistent with training)
        with torch.cuda.amp.autocast(enabled=stock_feat.is_cuda):
            dir_logits, mag_preds = model(
                stock_features=stock_feat,
                macro_features=macro_feat,
                sector_mask=sector_mask,
                active_mask=active_mask
            )
        
        scores = dir_logits.squeeze(-1)
        
        # Direction: Pairwise Margin Ranking Loss
        loss_dir, rank_accuracy = compute_pairwise_ranking_loss(
            scores=scores,
            returns=returns,
            sector_mask=sector_mask,
            active_mask=active_mask
        )
        
        # Magnitude: SmoothL1 on log-scaled targets (masked)
        # FIX #5: pass active_mask so sigma is computed over active stocks only
        mag_target = compute_log_scaled_mag_target(returns, active_mask=active_mask)
        if active_mask is not None and active_mask.any():
            loss_mag = mse_loss_fn(
                mag_preds.squeeze(-1)[active_mask],
                mag_target[active_mask]
            )
            mag_mae = torch.abs(
                mag_preds.squeeze(-1)[active_mask] - mag_target[active_mask]
            ).mean().item()
        else:
            loss_mag = mse_loss_fn(mag_preds.squeeze(-1), mag_target)
            mag_mae = torch.abs(mag_preds.squeeze(-1) - mag_target).mean().item()
        
        loss_total = loss_dir + mag_weight * loss_mag
        
        eval_metrics['loss'] += loss_total.item()
        eval_metrics['loss_dir'] += loss_dir.item()
        eval_metrics['loss_mag'] += loss_mag.item()
        eval_metrics['rank_accuracy'] += rank_accuracy
        eval_metrics['mag_mae'] += mag_mae
        num_batches += 1
    
    for key in eval_metrics:
        eval_metrics[key] /= max(num_batches, 1)
    
    return eval_metrics


def create_sequential_snapshots(
    stock_data: torch.Tensor,  # [N_s, Total_T, d_s]
    macro_data: torch.Tensor,  # [N_m, Total_T, d_m]
    returns_data: torch.Tensor,  # [N_s, Total_T]
    window_size: int = 60,
    step_size: int = 1,
    forecast_horizon: int = 5,
    inclusion_mask: Optional[torch.Tensor] = None  # [N_s, Total_T]
) -> List[Tuple]:
    """
    Create sequential snapshot batches from time-series data.
    
    Returns:
        If inclusion_mask is provided:
            List of (stock_window, macro_window, future_returns, active_mask) 4-tuples
        Otherwise:
            List of (stock_window, macro_window, future_returns) 3-tuples (legacy)
    """
    total_t = stock_data.size(1)
    snapshots = []
    
    for t in range(0, total_t - window_size - forecast_horizon + 1, step_size):
        stock_window = stock_data[:, t:t+window_size, :]
        macro_window = macro_data[:, t:t+window_size, :]
        
        # Future returns as labels.
        # INVARIANT: future_returns uses indices [t+window_size, t+window_size+forecast_horizon)
        # which are strictly AFTER the lookback window [t, t+window_size).
        # stock_window[:, :, -1] (the Returns column) only reaches t+window_size-1 — no leakage.
        future_returns = returns_data[:, t+window_size:t+window_size+forecast_horizon].sum(dim=1)
        
        if inclusion_mask is not None:
            # FIX #1 (same as _create_snapshots in data_loader.py):
            # Use t+window_size (forecast START) not t+window_size-1 (last lookback day).
            # This correctly identifies which stocks are tradable when the
            # forecast period begins, matching the semantics of future_returns.
            clamp_t = min(t + window_size, inclusion_mask.size(1) - 1)
            active_mask = inclusion_mask[:, clamp_t]  # [N_s]
            snapshots.append((stock_window, macro_window, future_returns, active_mask))
        else:
            snapshots.append((stock_window, macro_window, future_returns))
    
    return snapshots


def mc_dropout_inference(
    model: MacroDGRCL,
    stock_features: torch.Tensor,
    macro_features: torch.Tensor,
    n_samples: int = 10,
    macro_stock_edges: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Monte Carlo Dropout inference for uncertainty estimation.
    
    Runs n_samples stochastic forward passes with dropout enabled,
    then computes mean prediction and confidence from raw score variance.
    
    For pairwise ranking models, confidence is derived from the stability
    of the raw direction scores (logits) across MC passes — NOT from
    sigmoid probabilities, which compress near 0.5 and hide uncertainty.
    
    Args:
        model: MacroDGRCL model
        stock_features: [N_s, T, d_s]
        macro_features: [N_m, T, d_m]
        n_samples: Number of stochastic forward passes
        macro_stock_edges: Optional fixed edges
        
    Returns:
        Dict with keys:
            dir_score_mean: [N_s] mean raw direction score (logit)
            dir_score_std:  [N_s] std of raw scores — uncertainty
            mag_mean:       [N_s] mean magnitude prediction
            mag_std:        [N_s] std of magnitude prediction
            confidence:     [N_s] 1/(1+score_std) — higher = more confident
            rank_stability: scalar — fraction of pairwise rankings that are
                            consistent across all MC passes (0=chaotic, 1=stable)
    """
    model.train()  # Keep dropout active
    
    dir_scores_all = []
    mag_preds_all = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            dir_logits, mag_preds = model(
                stock_features=stock_features,
                macro_features=macro_features,
                macro_stock_edges=macro_stock_edges
            )
            dir_scores_all.append(dir_logits.squeeze(-1))  # [N_s] raw scores
            mag_preds_all.append(mag_preds.squeeze(-1))    # [N_s]
    
    # Stack: [n_samples, N_s]
    dir_scores_stack = torch.stack(dir_scores_all, dim=0)
    mag_preds_stack = torch.stack(mag_preds_all, dim=0)
    
    dir_score_mean = dir_scores_stack.mean(dim=0)
    dir_score_std = dir_scores_stack.std(dim=0)
    mag_mean = mag_preds_stack.mean(dim=0)
    mag_std = mag_preds_stack.std(dim=0)
    
    # Confidence: inverse of raw score uncertainty
    # score_std captures how much the model's ranking changes under dropout
    confidence = 1.0 / (1.0 + dir_score_std)
    
    # Rank stability: how consistent are the stock rankings across MC passes?
    # Uses Spearman rank correlation (vectorized) instead of O(N²) Kendall tau.
    if n_samples >= 2 and dir_scores_stack.shape[1] >= 2:
        ref_ranks = dir_scores_stack[0].argsort().argsort().float()  # [N_s]
        correlations = []
        for i in range(1, n_samples):
            sample_ranks = dir_scores_stack[i].argsort().argsort().float()
            # Spearman correlation via Pearson on ranks
            corr_matrix = torch.corrcoef(torch.stack([ref_ranks, sample_ranks]))
            correlations.append(corr_matrix[0, 1].item())
        rank_stability = sum(correlations) / len(correlations)
        # Clamp to [0, 1] — negative correlation means chaotic
        rank_stability = max(0.0, rank_stability)
    else:
        rank_stability = 0.0
    
    model.eval()  # Restore eval mode
    
    return {
        'dir_score_mean': dir_score_mean,
        'dir_score_std': dir_score_std,
        'mag_mean': mag_mean,
        'mag_std': mag_std,
        'confidence': confidence,
        'rank_stability': torch.tensor(rank_stability),
    }


def save_attention_heatmap(
    model: MacroDGRCL,
    stock_features: torch.Tensor,
    macro_features: torch.Tensor,
    fold_idx: int,
    output_dir: str,
    tickers: Optional[List[str]] = None,
    max_display: int = 30,
    sector_mask: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,   # FIX #4: added parameter
):
    """
    Visualize the DynamicGraphLearner attention weights as a heatmap.

    Shows which stocks attend to which other stocks, revealing whether
    the learned edges capture meaningful relationships or memorize noise.

    Args:
        model: Trained MacroDGRCL model
        stock_features: [N_s, T, d_s] input features
        macro_features: [N_m, T, d_m] macro features
        fold_idx: Current fold number (for filename)
        output_dir: Directory to save the heatmap
        tickers: Optional list of ticker names for axis labels
        max_display: Maximum number of stocks to show (for readability)
        sector_mask: [N_s, N_s] boolean sector mask (same as used in training)
        active_mask: [N_s] bool — inactive nodes are zeroed before LayerNorm
                     to match the Safeguard 1 applied in MacroDGRCL.forward()
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        # Replicate MacroDGRCL.forward() exactly:
        # encode → Safeguard 1 (zero inactive) → normalize → graph_learner
        stock_h = model.stock_encoder(stock_features)  # [N_s, H]

        # FIX #4: Apply Safeguard 1 before LayerNorm (same as forward pass).
        # Without this, LSTM bias on inactive stocks skews LayerNorm stats
        # AND the graph learner receives non-zero embeddings for dummy nodes.
        if active_mask is not None:
            device = stock_features.device
            active_mask = active_mask.to(device)
            stock_h = stock_h * active_mask.unsqueeze(-1).float()

        stock_h = model.stock_embedding_norm(stock_h)  # LayerNorm
        
        # LayerNorm bias re-introduces non-zero values for inactive nodes
        if active_mask is not None:
            stock_h = stock_h * active_mask.unsqueeze(-1).float()

        # Ensure sector_mask is on the correct device
        if sector_mask is not None:
            sector_mask = sector_mask.to(stock_features.device)

        # FIX #4: Pass active_mask so inactive→* and *→inactive edges
        # receive -inf attention, keeping the heatmap accurate.
        edge_index, edge_weights = model.graph_learner(
            stock_h,
            sector_mask=sector_mask,
            active_mask=active_mask,
            return_weights=True,
        )
    
    N_s = stock_features.size(0)
    n_display = min(N_s, max_display)
    
    # Build dense attention matrix from sparse edges
    attn_matrix = torch.zeros(N_s, N_s, device=stock_features.device)
    src, dst = edge_index
    attn_matrix[dst, src] = edge_weights  # dst attends to src
    
    # Apply row-wise softmax over non-zero entries so heatmap shows
    # meaningful probability distributions instead of raw scores.
    # Mask zero entries (no edge) with -inf before softmax.
    mask = (attn_matrix == 0)
    attn_matrix_softmax = attn_matrix.masked_fill(mask, float('-inf'))
    attn_matrix_softmax = torch.softmax(attn_matrix_softmax, dim=1)
    # Replace any NaN rows (fully masked) with 0
    attn_matrix_softmax = torch.nan_to_num(attn_matrix_softmax, nan=0.0)
    
    # Crop to displayable size
    attn_np = attn_matrix_softmax[:n_display, :n_display].cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn_np, cmap='YlOrRd', aspect='auto')
    ax.set_title(f'Stock→Stock Attention Weights — Fold {fold_idx + 1}', fontsize=14)
    ax.set_xlabel('Source (attends FROM)', fontsize=12)
    ax.set_ylabel('Destination (attends TO)', fontsize=12)
    
    if tickers and len(tickers) >= n_display:
        tick_labels = tickers[:n_display]
        ax.set_xticks(range(n_display))
        ax.set_yticks(range(n_display))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(tick_labels, fontsize=6)
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'attention_heatmap_fold_{fold_idx + 1}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved attention heatmap: {filepath}")


def save_training_curve(epochs: List[int], losses: List[float], fold_idx: int, output_dir: str):
    """Save training loss curve to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss - Fold {fold_idx + 1}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'training_curve_fold_{fold_idx + 1}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  Saved: {filepath}")


def save_backtest_summary(all_fold_results: List[Dict], output_dir: str, dates=None):
    """Generate and save backtest summary visualization with market regimes."""
    os.makedirs(output_dir, exist_ok=True)
    
    folds = [r['fold'] for r in all_fold_results]
    train_losses = [r['train_loss'] for r in all_fold_results]
    val_losses = [r['val_loss'] if r['val_loss'] is not None else 0 for r in all_fold_results]
    dir_accs = [r.get('rank_accuracy', 0) for r in all_fold_results]
    mag_maes = [r.get('mag_mae', 0) for r in all_fold_results]
    
    # Create figure with market regime context
    fig = plt.figure(figsize=(16, 12))
    
    # --- Top Row: Market Regime Timeline ---
    ax_market = fig.add_subplot(3, 1, 1)
    if dates is not None and len(dates) > 0:
        import pandas as pd
        n_days = len(dates)
        
        regimes = [
            ("2022-01-01", "2022-10-15", "Bear Market", "#FF6B6B"),
            ("2022-10-16", "2023-07-31", "Recovery", "#87CEEB"),
            ("2023-08-01", "2024-01-01", "Consolidation", "#FFD93D"),
            ("2024-01-02", "2025-03-01", "Bull Run", "#90EE90"),
            ("2025-03-02", "2026-12-31", "Late Cycle", "#DDA0DD"),
        ]
        
        for start, end, label, color in regimes:
            try:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                if start_date >= dates.iloc[0] and start_date <= dates.iloc[-1]:
                    ax_market.axvspan(start_date, min(end_date, dates.iloc[-1]), 
                                     alpha=0.3, color=color, label=label)
            except:
                pass
        
        fold_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, result in enumerate(all_fold_results):
            if 'train_start_idx' in result and 'val_end_idx' in result:
                fold_start = dates.iloc[result['train_start_idx']]
                fold_end = dates.iloc[min(result['val_end_idx'], len(dates)-1)]
                ax_market.axvline(fold_start, color=fold_colors[i % len(fold_colors)], 
                                 linestyle='--', alpha=0.7)
        
        ax_market.set_title('Market Regimes & Fold Coverage (2021-2026)', fontsize=14, fontweight='bold')
        ax_market.set_xlim(dates.iloc[0], dates.iloc[-1])
        ax_market.legend(loc='upper left', fontsize=9)
        ax_market.set_ylabel('Market Regime')
    else:
        ax_market.text(0.5, 0.5, 'Market regime data not available', 
                      ha='center', va='center', fontsize=12)
    
    # --- Middle Row: Loss Charts ---
    ax1 = fig.add_subplot(3, 3, 4)
    ax2 = fig.add_subplot(3, 3, 5)
    ax3 = fig.add_subplot(3, 3, 6)
    
    x = np.arange(len(folds))
    width = 0.35
    ax1.bar(x - width/2, train_losses, width, label='Train', color='steelblue')
    if any(val_losses):
        ax1.bar(x + width/2, val_losses, width, label='Val', color='coral')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Val Loss (Combined)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(folds, train_losses, 'o-', color='steelblue', linewidth=2, markersize=10, label='Train Loss')
    if any(val_losses):
        ax2.plot(folds, val_losses, 's-', color='coral', linewidth=2, markersize=10, label='Val Loss')
    ax2.axhline(y=np.mean(train_losses), color='steelblue', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Trend Across Folds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Direction accuracy per fold
    if any(dir_accs):
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(folds)))
        ax3.bar(folds, [a * 100 for a in dir_accs], color=colors)
        ax3.axhline(y=50, color='red', linestyle='--', label='Random Baseline')
        mean_acc = np.mean(dir_accs) * 100
        ax3.axhline(y=mean_acc, color='green', linestyle='--', 
                    label=f'Mean: {mean_acc:.1f}%')
        ax3.legend()
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Rank Accuracy (%)')
    ax3.set_title('Pairwise Rank Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # --- Bottom Row: Magnitude MAE ---
    ax4 = fig.add_subplot(3, 3, 7)
    if any(mag_maes):
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(folds)))
        ax4.bar(folds, mag_maes, color=colors)
        ax4.axhline(y=np.mean(mag_maes), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(mag_maes):.4f}')
        ax4.legend()
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('MAE')
    ax4.set_title('Magnitude Head MAE')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'backtest_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved backtest summary: {filepath}")


# =============================================================================
# FEATURE ABLATION CONFIGURATION
# =============================================================================

ABLATION_CONFIGS = {
    "baseline":        [0, 1, 2, 3, 4, 5, 6, 7],  # All 8 features
    "stationary_only": [3, 4, 5, 6, 7],            # Log_Vol, RSI, MACD, Vol5, Returns
    "pure_momentum":   [4, 5, 7],                  # RSI, MACD, Returns
    "pure_volatility": [3, 6, 7],                  # Log_Vol, Vol5, Returns
}


def slice_stock_features(
    snapshots: List[Tuple],
    feature_indices: List[int]
) -> List[Tuple]:
    """
    Slice stock features to keep only selected dimensions.
    Supports both 3-tuple (stock, macro, returns) and 4-tuple 
    (stock, macro, returns, active_mask) formats.
    """
    if not feature_indices:
        return snapshots
        
    sliced_snapshots = []
    for snapshot in snapshots:
        stock = snapshot[0]
        stock_sliced = stock[:, :, feature_indices]
        # Preserve all other elements (macro, ret, and optional active_mask)
        sliced_snapshots.append((stock_sliced,) + snapshot[1:])
        
    return sliced_snapshots

def main(
    use_real_data: bool = False,
    start_fold: int = 1,
    end_fold: Optional[int] = None,
    mag_weight: float = 0.05,
    force_cpu: bool = False,
    use_amp: bool = True,
    ablation: str = "baseline",
    output_dir: str = "./backtest_results",
    epochs: int = 100
):
    """
    Training loop with Multi-Task Learning.
    
    Args:
        use_real_data: If True, load data from ./data/processed/
                       If False, use synthetic random data
        start_fold: First fold to run (1-based index)
        end_fold: Last fold to run (1-based index, inclusive). If None, run all.
        mag_weight: λ weight for magnitude loss (default 0.1)
        force_cpu: If True, force CPU mode (avoids GPU memory issues)
        use_amp: If True, enable mixed precision (FP16) training for memory efficiency
        ablation: Feature ablation experiment name (one of ABLATION_CONFIGS keys)
        output_dir: Directory to save results
    """
    # Hyperparameters
    STOCK_DIM_FULL = 8  # ['Close', 'High', 'Low', 'Log_Vol', 'RSI_14', 'MACD', 'Volatility_5', 'Returns']
    FEATURE_NAMES = ['Close', 'High', 'Low', 'Log_Vol', 'RSI_14', 'MACD', 'Volatility_5', 'Returns']
    
    # Feature ablation: select experiment
    feature_indices = ABLATION_CONFIGS[ablation]
    STOCK_DIM = len(feature_indices)
    selected_names = [FEATURE_NAMES[i] for i in feature_indices]
    print(f"Feature ablation: '{ablation}' → {STOCK_DIM} features: {selected_names}")
    MACRO_DIM = 4
    HIDDEN_DIM = 64
    WINDOW_SIZE = 60
    NUM_EPOCHS = epochs
    LR = 5e-5
    WEIGHT_DECAY = 1e-3
    PATIENCE = 10  # FIX #NoLearn-1: Reduced from 40 → 10.
    # With patience=40 and LR=5e-5 the model was effectively never improving:
    # every fold early-stopped at epoch 41, meaning the best checkpoint was always
    # epoch 1 (random init). A shorter patience reacts faster to plateaus and
    # frees compute for folds that actually show improvement.
    MAX_STOCKS = 150  # Stock universe size
    
    # Walk-forward parameters (only used with real data)
    TRAIN_SIZE = 200
    VAL_SIZE = 100
    FOLD_STEP = 50
    
    # Visualization output directory (use passed argument)
    VIS_DIR = output_dir
    os.makedirs(VIS_DIR, exist_ok=True)
    
    # Checkpoint path for fold-level resume
    CHECKPOINT_PATH = os.path.join(VIS_DIR, "fold_results.json")
    
    # Device
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Magnitude loss weight (λ): {mag_weight}")
    
    # Initialize GradScaler for mixed precision training
    scaler = None
    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed Precision Training (FP16): ENABLED")
    else:
        print("Mixed Precision Training (FP16): DISABLED")
    
    if use_real_data:
        from data_loader import load_and_prepare_backtest
        
        print("\n=== Loading Real Market Data ===")
        # v1.4: Now returns sector_map
        folds, dates, tickers, NUM_STOCKS, NUM_MACROS, sector_map = load_and_prepare_backtest(
            data_dir="./data/processed",
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            step_size=FOLD_STEP,
            window_size=WINDOW_SIZE,
            forecast_horizon=5,
            snapshot_step=1,
            macro_lag=5,  # v1.4: Explicit macro lagging
            device=torch.device('cpu'),
            max_stocks=MAX_STOCKS
        )
        
        print(f"\nReady for walk-forward backtest:")
        print(f"  {len(folds)} folds, {NUM_STOCKS} stocks, {NUM_MACROS} macro factors")
        
        # v1.4: Initialize Graph Builder with Sector Info
        from macro_dgrcl import HeteroGraphBuilder
        graph_builder = HeteroGraphBuilder(
            sector_mapping=sector_map,
            tickers=tickers
        )
        # Precompute sector mask [N, N] and IDs [N]
        # We need these on the correct device later
        sector_mask = graph_builder.sector_mask
        
        # Convert sector mapping to integer IDs for target generation
        # tickers is list of strings
        # sector_map is dict {ticker: sector_name}
        if sector_map and tickers:
            unique_sectors = sorted(list(set(sector_map.values())))
            sector_to_id = {s: i for i, s in enumerate(unique_sectors)}
            sector_ids_list = [sector_to_id.get(sector_map.get(t, "Unknown"), -1) for t in tickers]
            sector_ids = torch.tensor(sector_ids_list, dtype=torch.long)
            print(f"  Sector constraints enabled. Found {len(unique_sectors)} sectors.")
        else:
            sector_ids = None
            print("  No sector mapping found. Running in sector-agnostic mode.")

        # Defensive alignment check: ensure sector tensors match the loaded universe.
        # A mismatch here (e.g., if load_processed_data reorders tickers) would
        # silently corrupt pairwise comparisons for every snapshot in every fold.
        if sector_mask is not None:
            assert sector_mask.shape[0] == len(tickers), (
                f"sector_mask rows ({sector_mask.shape[0]}) ≠ len(tickers) ({len(tickers)}). "
                "Ticker order in graph_builder must match stock_data."
            )
        if sector_ids is not None:
            assert sector_ids.shape[0] == len(tickers), (
                f"sector_ids length ({sector_ids.shape[0]}) ≠ len(tickers) ({len(tickers)}). "
                "Ticker order in sector_to_id mapping must match stock_data."
            )
            
    else:
        NUM_STOCKS = 50
        NUM_MACROS = 4
        print("\n=== Generating Synthetic Data ===")
        total_timesteps = 500
        
        stock_data = torch.randn(NUM_STOCKS, total_timesteps, STOCK_DIM_FULL)
        macro_data = torch.randn(NUM_MACROS, total_timesteps, MACRO_DIM)
        returns_data = torch.randn(NUM_STOCKS, total_timesteps) * 0.02
        
        # Create synthetic inclusion mask (~80% active at any time)
        inclusion_mask = torch.rand(NUM_STOCKS, total_timesteps) > 0.2
        
        train_data = create_sequential_snapshots(
            stock_data, macro_data, returns_data,
            window_size=WINDOW_SIZE,
            step_size=5,
            forecast_horizon=5,
            inclusion_mask=inclusion_mask
        )
        folds = [(train_data, [])]
        print(f"Created {len(train_data)} training snapshots (synthetic)")
        
        # No sector constraints for synthetic data
        sector_mask = None
        sector_ids = None
    
    # Loss function (shared across folds)
    mse_loss_fn = nn.SmoothL1Loss()
    
    # Load any previously completed fold results for resume
    all_fold_results = []
    completed_folds = set()
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                saved_results = json.load(f)
            all_fold_results = saved_results
            completed_folds = {r['fold'] for r in saved_results}
            print(f"\nResuming: found {len(completed_folds)} completed folds from checkpoint")
        except (json.JSONDecodeError, KeyError):
            print("Warning: corrupt checkpoint file, starting fresh")
    
    for fold_idx, fold_tuple in enumerate(folds):
        # Unpack 3-tuple: (train_snapshots, val_snapshots, WalkForwardFold metadata).
        # The fold metadata carries accurate train_start / val_end indices from the
        # WalkForwardSplitter, preventing the hardcoded synthetic-formula bug (Issue #7).
        train_data_fold, val_data_fold = fold_tuple[0], fold_tuple[1]
        fold_meta = fold_tuple[2] if len(fold_tuple) == 3 else None

        # Apply feature ablation (slices stock_features dim for each snapshot)
        train_data = slice_stock_features(train_data_fold, feature_indices)
        val_data = slice_stock_features(val_data_fold, feature_indices)
        current_fold_num = fold_idx + 1
        
        # Skip folds outside requested range
        if current_fold_num < start_fold:
            continue
        if end_fold is not None and current_fold_num > end_fold:
            break
        if current_fold_num in completed_folds:
            print(f"\nFold {current_fold_num} already completed (checkpoint), skipping...")
            continue
            
        # Enhanced GPU memory clearing from previous fold
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all kernels to finish
            torch.cuda.empty_cache()  # Release cached memory
        gc.collect()  # Python garbage collection

        # FIX #NaN-1 (PRIMARY): Recreate the GradScaler at the start of every fold.
        #
        # The GradScaler tracks inf/NaN gradient events via internal state
        # (_scale, _growth_tracker, _found_inf_per_device). When a bad snapshot
        # in Fold N triggers a NaN, the scaler lowers its scale factor. This
        # degraded state carries over into Fold N+1, N+2, … even though the model
        # is freshly re-initialised each fold.
        #
        # In this backtest: Fold 1 already showed NaN training loss (line 143 of
        # the log). The scaler kept degrading until Fold 19, where the scale hit
        # a floor that caused FP16 overflow on the *very first* forward pass,
        # producing NaN in *every* subsequent fold.
        #
        # Recreating the scaler gives each fold a fresh scale = 2^16 and a clean
        # inf-tracking state, decoupling folds from each other.
        if use_amp and device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'='*60}")
        print(f"Training snapshots: {len(train_data)}")
        print(f"Validation snapshots: {len(val_data)}")
        
        # Guard: warn if validation set is empty
        if not val_data:
            print(f"  ⚠ WARNING: Fold {fold_idx + 1} has ZERO validation snapshots!")
            print(f"  → Training will run full {NUM_EPOCHS} epochs without early stopping.")
        
        # Fresh model per fold to prevent weight leakage across walk-forward splits
        model = MacroDGRCL(
            num_stocks=NUM_STOCKS,
            num_macros=NUM_MACROS,
            stock_feature_dim=STOCK_DIM,
            macro_feature_dim=MACRO_DIM,
            hidden_dim=HIDDEN_DIM,
            temporal_layers=2,
            mp_layers=2,
            heads=4,
            top_k=min(5, NUM_STOCKS - 1),
            dropout=0.5,
            head_dropout=0.5
        ).to(device)
        
        optimizer = AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        early_stopping = EarlyStopping(patience=PATIENCE)
        
        # Training loop for this fold
        print("\nTraining...")
        epoch_losses = []
        for epoch in range(NUM_EPOCHS):
            epoch_metrics = train_epoch(
                model=model,
                data_loader=train_data,
                optimizer=optimizer,
                mse_loss_fn=mse_loss_fn,
                device=device,
                mag_weight=mag_weight,
                max_grad_norm=1.0,
                sector_mask=sector_mask,
                sector_ids=sector_ids,
                scaler=scaler
            )
            epoch_losses.append(epoch_metrics['loss'])
            
            scheduler.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Loss: {epoch_metrics['loss']:.4f} "
                      f"(Dir: {epoch_metrics['loss_dir']:.4f}, Mag: {epoch_metrics['loss_mag']:.4f}) | "
                      f"Rank Acc: {epoch_metrics['rank_accuracy']*100:.1f}% | "
                      f"Mag MAE: {epoch_metrics['mag_mae']:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Early stopping check on validation data
            if val_data:
                val_metrics = evaluate(
                    model=model,
                    data_loader=val_data,
                    mse_loss_fn=mse_loss_fn,
                    device=device,
                    mag_weight=mag_weight,
                    sector_mask=sector_mask,
                    sector_ids=sector_ids
                )
                if early_stopping(val_metrics['loss'], model):
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(best val loss: {early_stopping.best_loss:.4f})")
                    early_stopping.restore_best(model)
                    break
        
        # Save training curve for this fold
        save_training_curve(
            epochs=list(range(1, len(epoch_losses) + 1)),
            losses=epoch_losses,
            fold_idx=fold_idx,
            output_dir=VIS_DIR
        )
        
        # Final validation (if we have validation data)
        if val_data:
            print("\nValidating...")
            val_metrics = evaluate(
                model=model,
                data_loader=val_data,
                mse_loss_fn=mse_loss_fn,
                device=device,
                mag_weight=mag_weight,
                sector_mask=sector_mask,
                sector_ids=sector_ids
            )
            print(f"  Val Loss: {val_metrics['loss']:.4f} "
                  f"(Dir: {val_metrics['loss_dir']:.4f}, Mag: {val_metrics['loss_mag']:.4f})")
            print(f"  Rank Accuracy: {val_metrics['rank_accuracy']*100:.1f}%")
            print(f"  Mag MAE: {val_metrics['mag_mae']:.4f}")
            
            all_fold_results.append({
                'fold': fold_idx + 1,
                'train_loss': epoch_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'rank_accuracy': val_metrics['rank_accuracy'],
                'mag_mae': val_metrics['mag_mae'],
                # Use actual splitter indices when available (real data 3-tuple folds);
                # fall back to synthetic formula only if metadata is absent.
                'train_start_idx': fold_meta.train_start if fold_meta is not None else fold_idx * FOLD_STEP,
                'val_end_idx': fold_meta.val_end if fold_meta is not None else fold_idx * FOLD_STEP + TRAIN_SIZE + VAL_SIZE
            })
        else:
            all_fold_results.append({
                'fold': fold_idx + 1,
                'train_loss': epoch_metrics['loss'],
                'val_loss': None,
                'rank_accuracy': epoch_metrics['rank_accuracy'],
                'mag_mae': epoch_metrics['mag_mae'],
                'train_start_idx': fold_meta.train_start if fold_meta is not None else fold_idx * FOLD_STEP,
                'val_end_idx': fold_meta.val_end if fold_meta is not None else fold_idx * FOLD_STEP + TRAIN_SIZE + VAL_SIZE
            })
        
        # Save checkpoint after each fold
        os.makedirs(VIS_DIR, exist_ok=True)
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(all_fold_results, f, indent=2)
        print(f"  Checkpoint saved ({len(all_fold_results)} folds complete)")
    
    # Summary
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY (v1.5 Pairwise Ranking)")
    print(f"{'='*60}")
    
    val_losses = [r['val_loss'] for r in all_fold_results if r['val_loss'] is not None]
    rank_accs = [r['rank_accuracy'] for r in all_fold_results]
    mag_maes = [r['mag_mae'] for r in all_fold_results]
    
    if val_losses:
        print(f"Average Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"Best Fold: {np.argmin(val_losses) + 1} (Loss: {min(val_losses):.4f})")
    
    print(f"Average Rank Accuracy: {np.mean(rank_accs)*100:.1f}%")
    print(f"Average Mag MAE: {np.mean(mag_maes):.4f}")
    
    # Save backtest summary visualization
    save_backtest_summary(all_fold_results, VIS_DIR, dates=dates if use_real_data else None)
    
    # MC Dropout inference on last fold (only if model exists)
    if 'model' in locals():
        print("\nMC Dropout Inference (10 stochastic passes)...")
        test_stock = train_data[0][0].to(device)
        test_macro = train_data[0][1].to(device)
        
        mc_results = mc_dropout_inference(
            model=model,
            stock_features=test_stock,
            macro_features=test_macro,
            n_samples=10,
            macro_stock_edges=None  # Can be passed if we had fixed edges
        )
        
        print(f"  Direction score mean:  {mc_results['dir_score_mean'].mean().item():.4f}")
        print(f"  Direction score std:   {mc_results['dir_score_std'].mean().item():.4f}")
        print(f"  Magnitude mean:        {mc_results['mag_mean'].mean().item():.4f}")
        print(f"  Magnitude uncertainty: {mc_results['mag_std'].mean().item():.4f}")
        print(f"  Confidence mean:       {mc_results['confidence'].mean().item():.4f}")
        print(f"  Confidence min/max:    {mc_results['confidence'].min().item():.4f} / "
              f"{mc_results['confidence'].max().item():.4f}")
        print(f"  Rank stability:        {mc_results['rank_stability'].item():.4f}")
        
        # Attention weight visualization on last fold
        print("\nGenerating attention heatmap...")
        save_attention_heatmap(
            model=model,
            stock_features=test_stock,
            macro_features=test_macro,
            fold_idx=len(folds) - 1,
            output_dir=VIS_DIR,
            tickers=tickers if use_real_data else None,
            sector_mask=sector_mask
        )
    else:
        print("\nNo model trained (all folds skipped). Skipping inference.")
    
    print(f"\nBacktest complete! Results saved to: {VIS_DIR}")
    return all_fold_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Macro-DGRCL v1.4 (Sector-Aware)")
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real market data from ./data/processed/ instead of synthetic data"
    )
    parser.add_argument(
        "--start-fold",
        type=int,
        default=1,
        help="Start fold index (1-based, default: 1)"
    )
    parser.add_argument(
        "--end-fold",
        type=int,
        default=None,
        help="End fold index (1-based, inclusive, default: All)"
    )
    parser.add_argument(
        "--mag-weight",
        type=float,
        default=0.05,
        help="Weight λ for magnitude loss (default: 0.05)"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (slower but avoids GPU memory issues)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    
    parser.add_argument(
        "--ablation",
        type=str,
        default="baseline",
        choices=list(ABLATION_CONFIGS.keys()),
        help="Feature ablation experiment (default: baseline, uses all 8 features)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./backtest_results",
        help="Directory to save visualization and checkpoint results (default: ./backtest_results)"
    )
    
    args = parser.parse_args()
    main(
        use_real_data=args.real_data,
        start_fold=args.start_fold,
        end_fold=args.end_fold,
        mag_weight=args.mag_weight,
        force_cpu=args.cpu,
        ablation=args.ablation,
        output_dir=args.output_dir,
        epochs=args.epochs
    )
