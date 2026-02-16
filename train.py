"""
Training Loop for Macro-Aware DGRCL v1.3 (Multi-Task Learning)

Implements:
- Multi-Task Loss: BCEWithLogitsLoss (direction) + λ·MSELoss (magnitude)
- Dynamic target engineering from raw returns
- EarlyStopping on combined validation loss
- Walk-forward cross-validation
- Mixed Precision Training (FP16) for GPU memory efficiency
"""

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
            device = next(model.parameters()).device
            state = {k: v.to(device) for k, v in self.best_state.items()}
            model.load_state_dict(state)


# =============================================================================
# TRAINING STEP
# =============================================================================

def train_step(
    model: MacroDGRCL,
    stock_features: torch.Tensor,  # [N_s, T, d_s]
    macro_features: torch.Tensor,  # [N_m, T, d_m]
    returns: torch.Tensor,  # [N_s] future returns (labels)
    optimizer: torch.optim.Optimizer,
    bce_loss_fn: nn.BCEWithLogitsLoss,
    mse_loss_fn: nn.MSELoss,
    mag_weight: float = 0.1,
    macro_stock_edges: Optional[torch.Tensor] = None,
    sector_mask: Optional[torch.Tensor] = None,
    sector_ids: Optional[torch.Tensor] = None,
    max_grad_norm: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Single training step with Multi-Task Learning.
    
    Target Engineering:
        dir_target = (returns > Sector_Median(returns)).float()   → binary {0, 1}
        mag_target = |returns|                                     → ℝ≥0
    
    Args:
        model: MacroDGRCL model
        stock_features: [N_s, T, d_s] time-series
        macro_features: [N_m, T, d_m] time-series
        returns: [N_s] future returns (labels)
        optimizer: Optimizer
        bce_loss_fn: BCEWithLogitsLoss for direction
        mse_loss_fn: MSELoss for magnitude
        mag_weight: λ weight for magnitude loss
        macro_stock_edges: Optional fixed edges
        sector_mask: [N_s, N_s] boolean mask
        sector_ids: [N_s] sector IDs for sector-neutral targets
        max_grad_norm: Gradient clipping threshold
        scaler: Optional GradScaler for mixed precision training
        
    Returns:
        Dict with loss metrics and task-specific metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # --- Target Engineering ---
    # Direction: Return > Sector Median ? 1 : 0
    # This neutralizes sector beta and forces relative value learning.
    
    if sector_ids is not None:
        # Compute median return per sector
        unique_sectors = torch.unique(sector_ids)
        dir_target = torch.zeros_like(returns)
        
        for s_id in unique_sectors:
            mask = (sector_ids == s_id)
            sector_returns = returns[mask]
            
            if len(sector_returns) > 1:
                # Calculate median for this sector
                median_val = sector_returns.median()
                # 1 if > median, 0 otherwise
                dir_target[mask] = (sector_returns > median_val).float()
            else:
                # If only 1 stock in sector, can't compute relative value
                # Fallback to global 0 (hold) or ignore? 
                # Let's set to 0.5 (uncertain) -> binary 0 or 1 arbitrarily?
                # Actually, standard practice is to exclude or set to 0.
                dir_target[mask] = 0.0 # Default to SELL/HOLD
                
    else:
        # Fallback to Global Median if no sectors provided (Backward Compatibility)
        N_s = returns.size(0)
        ranks = returns.argsort().argsort().float()
        dir_target = (ranks >= N_s / 2).float()  # [N_s]
    
    # Magnitude: absolute return
    mag_target = torch.abs(returns)
    
    # --------------- Forward Pass ---------------
    # Use automatic mixed precision (AMP) if scaler is provided
    use_amp = scaler is not None
    with torch.cuda.amp.autocast(enabled=use_amp):
        dir_logits, mag_preds = model(
            stock_features=stock_features,
            macro_features=macro_features,
            macro_stock_edges=macro_stock_edges,
            sector_mask=sector_mask
        )
        
        # --------------- Loss Computation ---------------
        loss_dir = bce_loss_fn(dir_logits.squeeze(-1), dir_target)
        loss_mag = mse_loss_fn(mag_preds.squeeze(-1), mag_target)
        loss_total = loss_dir + mag_weight * loss_mag
    
    # --------------- Backward Pass ---------------
    optimizer.zero_grad()
    
    if use_amp:
        # Mixed precision: use scaler for backward pass
        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)  # Unscale before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Full precision: standard backward pass
        loss_total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    
    # --- Metrics ---
    with torch.no_grad():
        # Direction accuracy
        dir_preds = (dir_logits.squeeze(-1) > 0.0).float()
        dir_accuracy = (dir_preds == dir_target).float().mean().item()
        
        # Magnitude MAE
        mag_mae = torch.abs(mag_preds.squeeze(-1) - mag_target).mean().item()
    
    return {
        'loss': loss_total.item(),
        'loss_dir': loss_dir.item(),
        'loss_mag': loss_mag.item(),
        'dir_accuracy': dir_accuracy,
        'mag_mae': mag_mae,
        'grad_norm': grad_norm.item()
    }


def train_epoch(
    model: MacroDGRCL,
    data_loader: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    bce_loss_fn: nn.BCEWithLogitsLoss,
    mse_loss_fn: nn.MSELoss,
    device: torch.device,
    mag_weight: float = 0.1,
    max_grad_norm: float = 1.0,
    sector_mask: Optional[torch.Tensor] = None,
    sector_ids: Optional[torch.Tensor] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Train for one epoch over sequential snapshots.
    
    Args:
        model: MacroDGRCL model
        data_loader: List of (stock_features, macro_features, returns) tuples
        optimizer: Optimizer
        bce_loss_fn: BCEWithLogitsLoss
        mse_loss_fn: MSELoss
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
        'dir_accuracy': 0.0, 'mag_mae': 0.0
    }
    num_batches = 0
    
    # Ensure masks are on device
    if sector_mask is not None:
        sector_mask = sector_mask.to(device)
    if sector_ids is not None:
        sector_ids = sector_ids.to(device)
    
    for stock_feat, macro_feat, returns in data_loader:
        stock_feat = stock_feat.to(device)
        macro_feat = macro_feat.to(device)
        returns = returns.to(device)
        
        metrics = train_step(
            model=model,
            stock_features=stock_feat,
            macro_features=macro_feat,
            returns=returns,
            optimizer=optimizer,
            bce_loss_fn=bce_loss_fn,
            mse_loss_fn=mse_loss_fn,
            mag_weight=mag_weight,
            max_grad_norm=max_grad_norm,
            sector_mask=sector_mask,
            sector_ids=sector_ids,
            scaler=scaler
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
    data_loader: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    bce_loss_fn: nn.BCEWithLogitsLoss,
    mse_loss_fn: nn.MSELoss,
    device: torch.device,
    mag_weight: float = 0.1,
    sector_mask: Optional[torch.Tensor] = None,
    sector_ids: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Evaluate model on validation data with multi-task metrics.
    
    Args:
        model: MacroDGRCL model
        data_loader: Validation data
        bce_loss_fn: BCEWithLogitsLoss
        mse_loss_fn: MSELoss
        device: Torch device
        mag_weight: λ for magnitude loss
        sector_mask: [N_s, N_s] boolean mask
        sector_ids: [N_s] sector IDs
        
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    eval_metrics = {
        'loss': 0.0, 'loss_dir': 0.0, 'loss_mag': 0.0,
        'dir_accuracy': 0.0, 'mag_mae': 0.0
    }
    num_batches = 0
    
    if sector_mask is not None:
        sector_mask = sector_mask.to(device)
    if sector_ids is not None:
        sector_ids = sector_ids.to(device)
    
    for stock_feat, macro_feat, returns in data_loader:
        stock_feat = stock_feat.to(device)
        macro_feat = macro_feat.to(device)
        returns = returns.to(device)
        
        # Target engineering (Sector-Neutral)
        if sector_ids is not None:
            unique_sectors = torch.unique(sector_ids)
            dir_target = torch.zeros_like(returns)
            for s_id in unique_sectors:
                mask = (sector_ids == s_id)
                sector_returns = returns[mask]
                if len(sector_returns) > 1:
                    median_val = sector_returns.median()
                    dir_target[mask] = (sector_returns > median_val).float()
                else:
                    dir_target[mask] = 0.0
        else:
            # Fallback
            N_s = returns.size(0)
            ranks = returns.argsort().argsort().float()
            dir_target = (ranks >= N_s / 2).float()
            
        mag_target = torch.abs(returns)
        
        # Forward pass with mixed precision (consistent with training)
        # Note: no scaler needed in eval since we don't do backward pass
        with torch.cuda.amp.autocast(enabled=stock_feat.is_cuda):
            dir_logits, mag_preds = model(
                stock_features=stock_feat,
                macro_features=macro_feat,
                sector_mask=sector_mask
            )
        
        # Losses
        loss_dir = bce_loss_fn(dir_logits.squeeze(-1), dir_target)
        loss_mag = mse_loss_fn(mag_preds.squeeze(-1), mag_target)
        loss_total = loss_dir + mag_weight * loss_mag
        
        # Metrics
        dir_preds = (dir_logits.squeeze(-1) > 0.0).float()
        dir_accuracy = (dir_preds == dir_target).float().mean().item()
        mag_mae = torch.abs(mag_preds.squeeze(-1) - mag_target).mean().item()
        
        eval_metrics['loss'] += loss_total.item()
        eval_metrics['loss_dir'] += loss_dir.item()
        eval_metrics['loss_mag'] += loss_mag.item()
        eval_metrics['dir_accuracy'] += dir_accuracy
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
    forecast_horizon: int = 5
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create sequential snapshot batches from time-series data.
    
    Args:
        stock_data: [N_s, Total_T, d_s] full stock time-series
        macro_data: [N_m, Total_T, d_m] full macro time-series
        returns_data: [N_s, Total_T] stock returns
        window_size: Lookback window T
        step_size: Stride between windows
        forecast_horizon: Days ahead for return labels
        
    Returns:
        List of (stock_window, macro_window, future_returns) tuples
    """
    total_t = stock_data.size(1)
    snapshots = []
    
    for t in range(0, total_t - window_size - forecast_horizon, step_size):
        # Extract windows
        stock_window = stock_data[:, t:t+window_size, :]  # [N_s, T, d_s]
        macro_window = macro_data[:, t:t+window_size, :]  # [N_m, T, d_m]
        
        # Future returns as labels (cumulative over forecast horizon)
        future_returns = returns_data[:, t+window_size:t+window_size+forecast_horizon].sum(dim=1)
        
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
    then computes mean prediction and confidence (inverse of std).
    
    Args:
        model: MacroDGRCL model
        stock_features: [N_s, T, d_s]
        macro_features: [N_m, T, d_m]
        n_samples: Number of stochastic forward passes
        macro_stock_edges: Optional fixed edges
        
    Returns:
        Dict with keys:
            dir_prob_mean: [N_s] mean P(up) across samples
            dir_prob_std:  [N_s] std of P(up) — uncertainty
            mag_mean:      [N_s] mean magnitude prediction
            mag_std:       [N_s] std of magnitude prediction
            confidence:    [N_s] 1/(1+std) — higher = more confident
    """
    model.train()  # Keep dropout active
    
    dir_probs_all = []
    mag_preds_all = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            dir_logits, mag_preds = model(
                stock_features=stock_features,
                macro_features=macro_features,
                macro_stock_edges=macro_stock_edges
            )
            dir_probs_all.append(torch.sigmoid(dir_logits.squeeze(-1)))  # [N_s]
            mag_preds_all.append(mag_preds.squeeze(-1))  # [N_s]
    
    # Stack: [n_samples, N_s]
    dir_probs_stack = torch.stack(dir_probs_all, dim=0)
    mag_preds_stack = torch.stack(mag_preds_all, dim=0)
    
    dir_prob_mean = dir_probs_stack.mean(dim=0)
    dir_prob_std = dir_probs_stack.std(dim=0)
    mag_mean = mag_preds_stack.mean(dim=0)
    mag_std = mag_preds_stack.std(dim=0)
    
    # Confidence: inverse of combined uncertainty
    confidence = 1.0 / (1.0 + dir_prob_std)
    
    model.eval()  # Restore eval mode
    
    return {
        'dir_prob_mean': dir_prob_mean,
        'dir_prob_std': dir_prob_std,
        'mag_mean': mag_mean,
        'mag_std': mag_std,
        'confidence': confidence,
    }


def save_attention_heatmap(
    model: MacroDGRCL,
    stock_features: torch.Tensor,
    macro_features: torch.Tensor,
    fold_idx: int,
    output_dir: str,
    tickers: Optional[List[str]] = None,
    max_display: int = 30,
    sector_mask: Optional[torch.Tensor] = None
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
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        # Match model's actual forward path: encode → normalize → graph_learner
        stock_h = model.stock_encoder(stock_features)  # [N_s, H]
        stock_h = model.embedding_norm(stock_h)         # LayerNorm (was missing!)
        
        # Ensure sector_mask is on the same device as the model
        if sector_mask is not None:
            sector_mask = sector_mask.to(stock_features.device)
        
        # Pass sector_mask so cross-sector pairs are masked (was missing!)
        edge_index, edge_weights = model.graph_learner(
            stock_h, sector_mask=sector_mask, return_weights=True
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
    dir_accs = [r.get('dir_accuracy', 0) for r in all_fold_results]
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
    ax3.set_ylabel('Direction Accuracy (%)')
    ax3.set_title('Direction Head Accuracy')
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


def main(
    use_real_data: bool = False,
    start_fold: int = 1,
    end_fold: Optional[int] = None,
    mag_weight: float = 0.1,
    force_cpu: bool = False,
    use_amp: bool = True
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
    """
    # Hyperparameters
    STOCK_DIM = 8  # ['Close', 'High', 'Low', 'Log_Vol', 'RSI_14', 'MACD', 'Volatility_5', 'Returns']
    MACRO_DIM = 4
    HIDDEN_DIM = 64
    WINDOW_SIZE = 60
    NUM_EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 1e-1
    PATIENCE = 25  # Early stopping patience
    MAX_STOCKS = 150  # Stock universe size
    
    # Walk-forward parameters (only used with real data)
    TRAIN_SIZE = 200
    VAL_SIZE = 100
    FOLD_STEP = 50
    
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
            
    else:
        NUM_STOCKS = 50
        NUM_MACROS = 4
        print("\n=== Generating Synthetic Data ===")
        total_timesteps = 500
        
        stock_data = torch.randn(NUM_STOCKS, total_timesteps, STOCK_DIM)
        macro_data = torch.randn(NUM_MACROS, total_timesteps, MACRO_DIM)
        returns_data = torch.randn(NUM_STOCKS, total_timesteps) * 0.02
        
        train_data = create_sequential_snapshots(
            stock_data, macro_data, returns_data,
            window_size=WINDOW_SIZE,
            step_size=5,
            forecast_horizon=5
        )
        folds = [(train_data, [])]
        print(f"Created {len(train_data)} training snapshots (synthetic)")
        
        # No sector constraints for synthetic data
        sector_mask = None
        sector_ids = None
    
    # Loss functions (shared across folds)
    bce_loss_fn = nn.BCEWithLogitsLoss()
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
    
    for fold_idx, (train_data, val_data) in enumerate(folds):
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
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        early_stopping = EarlyStopping(patience=PATIENCE)
        
        # Training loop for this fold
        print("\nTraining...")
        epoch_losses = []
        for epoch in range(NUM_EPOCHS):
            epoch_metrics = train_epoch(
                model=model,
                data_loader=train_data,
                optimizer=optimizer,
                bce_loss_fn=bce_loss_fn,
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
                      f"Dir Acc: {epoch_metrics['dir_accuracy']*100:.1f}% | "
                      f"Mag MAE: {epoch_metrics['mag_mae']:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Early stopping check on validation data
            if val_data:
                val_metrics = evaluate(
                    model=model,
                    data_loader=val_data,
                    bce_loss_fn=bce_loss_fn,
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
                bce_loss_fn=bce_loss_fn,
                mse_loss_fn=mse_loss_fn,
                device=device,
                mag_weight=mag_weight,
                sector_mask=sector_mask,
                sector_ids=sector_ids
            )
            print(f"  Val Loss: {val_metrics['loss']:.4f} "
                  f"(Dir: {val_metrics['loss_dir']:.4f}, Mag: {val_metrics['loss_mag']:.4f})")
            print(f"  Dir Accuracy: {val_metrics['dir_accuracy']*100:.1f}%")
            print(f"  Mag MAE: {val_metrics['mag_mae']:.4f}")
            
            all_fold_results.append({
                'fold': fold_idx + 1,
                'train_loss': epoch_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'dir_accuracy': val_metrics['dir_accuracy'],
                'mag_mae': val_metrics['mag_mae'],
                'train_start_idx': fold_idx * FOLD_STEP,
                'val_end_idx': fold_idx * FOLD_STEP + TRAIN_SIZE + VAL_SIZE
            })
        else:
            all_fold_results.append({
                'fold': fold_idx + 1,
                'train_loss': epoch_metrics['loss'],
                'val_loss': None,
                'dir_accuracy': epoch_metrics['dir_accuracy'],
                'mag_mae': epoch_metrics['mag_mae'],
                'train_start_idx': fold_idx * FOLD_STEP,
                'val_end_idx': fold_idx * FOLD_STEP + TRAIN_SIZE + VAL_SIZE
            })
        
        # Save checkpoint after each fold
        os.makedirs(VIS_DIR, exist_ok=True)
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(all_fold_results, f, indent=2)
        print(f"  Checkpoint saved ({len(all_fold_results)} folds complete)")
    
    # Summary
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY (v1.4 Multi-Task Sector-Aware)")
    print(f"{'='*60}")
    
    val_losses = [r['val_loss'] for r in all_fold_results if r['val_loss'] is not None]
    dir_accs = [r['dir_accuracy'] for r in all_fold_results]
    mag_maes = [r['mag_mae'] for r in all_fold_results]
    
    if val_losses:
        print(f"Average Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"Best Fold: {np.argmin(val_losses) + 1} (Loss: {min(val_losses):.4f})")
    
    print(f"Average Dir Accuracy: {np.mean(dir_accs)*100:.1f}%")
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
        
        print(f"  Direction P(up) mean:  {mc_results['dir_prob_mean'].mean().item():.4f}")
        print(f"  Direction uncertainty:  {mc_results['dir_prob_std'].mean().item():.4f}")
        print(f"  Magnitude mean:        {mc_results['mag_mean'].mean().item():.4f}")
        print(f"  Magnitude uncertainty: {mc_results['mag_std'].mean().item():.4f}")
        print(f"  Confidence mean:       {mc_results['confidence'].mean().item():.4f}")
        print(f"  Confidence min/max:    {mc_results['confidence'].min().item():.4f} / "
              f"{mc_results['confidence'].max().item():.4f}")
        
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
        default=0.1,
        help="Weight λ for magnitude loss (default: 1.0)"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (slower but avoids GPU memory issues)"
    )
    
    args = parser.parse_args()
    main(
        use_real_data=args.real_data,
        start_fold=args.start_fold,
        end_fold=args.end_fold,
        mag_weight=args.mag_weight,
        force_cpu=args.cpu
    )
