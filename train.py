"""
Training Loop for Macro-Aware DGRCL v1.1

Implements:
- train_step: Single batch forward/backward pass
- train_epoch: Full epoch training
- DataLoader utilities for sequential snapshots
- Live visualization during backtesting
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import os

from macro_dgrcl import MacroDGRCL
from losses import DGRCLLoss


# Visualization output directory
VIS_DIR = "./backtest_results"


def train_step(
    model: MacroDGRCL,
    stock_features: torch.Tensor,  # [N_s, T, d_s]
    macro_features: torch.Tensor,  # [N_m, T, d_m]
    returns: torch.Tensor,  # [N_s] future returns (labels)
    criterion: DGRCLLoss,
    optimizer: torch.optim.Optimizer,
    macro_stock_edges: Optional[torch.Tensor] = None,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Single training step.
    
    Args:
        model: MacroDGRCL model
        stock_features: [N_s, T, d_s] stock time-series
        macro_features: [N_m, T, d_m] macro time-series
        returns: [N_s] future returns for ranking labels
        criterion: DGRCL loss function
        optimizer: Optimizer
        macro_stock_edges: Optional fixed macro→stock edges
        max_grad_norm: Gradient clipping threshold
        
    Returns:
        Dict with loss metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    scores, embeddings = model(
        stock_features=stock_features,
        macro_features=macro_features,
        macro_stock_edges=macro_stock_edges,
        force_dropout=False
    )
    
    # Compute loss
    loss = criterion(scores, embeddings, returns)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # Optimizer step
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'grad_norm': grad_norm.item()
    }


def train_epoch(
    model: MacroDGRCL,
    data_loader: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: DGRCLLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch over sequential snapshots.
    
    Args:
        model: MacroDGRCL model
        data_loader: List of (stock_features, macro_features, returns) tuples
        criterion: DGRCL loss function
        optimizer: Optimizer
        device: Torch device
        max_grad_norm: Gradient clipping threshold
        
    Returns:
        Dict with epoch-level metrics
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for stock_feat, macro_feat, returns in data_loader:
        stock_feat = stock_feat.to(device)
        macro_feat = macro_feat.to(device)
        returns = returns.to(device)
        
        metrics = train_step(
            model=model,
            stock_features=stock_feat,
            macro_features=macro_feat,
            returns=returns,
            criterion=criterion,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm
        )
        
        epoch_loss += metrics['loss']
        num_batches += 1
    
    return {
        'loss': epoch_loss / max(num_batches, 1),
        'num_batches': num_batches
    }


@torch.no_grad()
def evaluate(
    model: MacroDGRCL,
    data_loader: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: DGRCLLoss,
    device: torch.device,
    n_mc_samples: int = 30
) -> Dict[str, float]:
    """
    Evaluate model with uncertainty estimation.
    
    Args:
        model: MacroDGRCL model
        data_loader: Validation data
        criterion: Loss function
        device: Torch device
        n_mc_samples: Number of MC Dropout samples
        
    Returns:
        Dict with evaluation metrics including uncertainty
    """
    model.eval()
    total_loss = 0.0
    all_uncertainties = []
    num_batches = 0
    
    for stock_feat, macro_feat, returns in data_loader:
        stock_feat = stock_feat.to(device)
        macro_feat = macro_feat.to(device)
        returns = returns.to(device)
        
        # Get predictions with uncertainty
        mu, sigma2, embeddings = model.predict_with_uncertainty(
            stock_feat, macro_feat, n_samples=n_mc_samples
        )
        
        # Compute loss with mean predictions
        loss = criterion(mu, embeddings, returns)
        total_loss += loss.item()
        
        # Track uncertainty
        all_uncertainties.append(sigma2.mean().item())
        num_batches += 1
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'mean_uncertainty': np.mean(all_uncertainties) if all_uncertainties else 0.0
    }


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
    uncertainties = [r['uncertainty'] if r['uncertainty'] is not None else 0 for r in all_fold_results]
    
    # Create figure with market regime context
    fig = plt.figure(figsize=(16, 10))
    
    # Main layout: 2 rows
    # Top row: Market context (SPY-like returns visualization)
    # Bottom row: 3 metric charts
    
    # --- Top Row: Market Regime Timeline ---
    ax_market = fig.add_subplot(2, 1, 1)
    if dates is not None and len(dates) > 0:
        import pandas as pd
        # Create cumulative returns proxy from date count
        n_days = len(dates)
        date_range = pd.date_range(dates.iloc[0], dates.iloc[-1], periods=n_days)
        
        # Define market regimes with colors
        regimes = [
            ("2021-05-05", "2021-12-31", "Recovery Rally", "#90EE90"),
            ("2022-01-01", "2022-10-15", "Bear Market", "#FF6B6B"),
            ("2022-10-16", "2023-07-31", "Recovery", "#87CEEB"),
            ("2023-08-01", "2024-01-01", "Consolidation", "#FFD93D"),
            ("2024-01-02", "2026-02-06", "Bull Run", "#90EE90"),
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
        
        # Mark fold boundaries
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
    
    # --- Bottom Row: 3 Metric Charts ---
    ax1 = fig.add_subplot(2, 3, 4)
    ax2 = fig.add_subplot(2, 3, 5)
    ax3 = fig.add_subplot(2, 3, 6)
    
    # Plot 1: Train vs Val Loss per fold
    x = np.arange(len(folds))
    width = 0.35
    ax1.bar(x - width/2, train_losses, width, label='Train', color='steelblue')
    if any(val_losses):
        ax1.bar(x + width/2, val_losses, width, label='Val', color='coral')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Validation Loss')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training loss trend across folds
    ax2.plot(folds, train_losses, 'o-', color='steelblue', linewidth=2, markersize=10, label='Train Loss')
    if any(val_losses):
        ax2.plot(folds, val_losses, 's-', color='coral', linewidth=2, markersize=10, label='Val Loss')
    ax2.axhline(y=np.mean(train_losses), color='steelblue', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Trend Across Folds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty per fold
    if any(uncertainties):
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(folds)))
        ax3.bar(folds, uncertainties, color=colors)
        ax3.axhline(y=np.mean(uncertainties), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(uncertainties):.4f}')
        ax3.legend()
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Mean Uncertainty (σ²)')
    ax3.set_title('Model Uncertainty by Fold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'backtest_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved backtest summary: {filepath}")


def main(use_real_data: bool = False):
    """
    Training loop with optional real data support.
    
    Args:
        use_real_data: If True, load data from ./data/processed/
                       If False, use synthetic random data
    """
    # Hyperparameters
    STOCK_DIM = 8  # ['Close', 'High', 'Low', 'Log_Vol', 'RSI_14', 'MACD', 'Volatility_5', 'Returns']
    MACRO_DIM = 4
    HIDDEN_DIM = 64
    WINDOW_SIZE = 60
    NUM_EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Walk-forward parameters (only used with real data)
    TRAIN_SIZE = 200  # ~10 months
    VAL_SIZE = 100    # ~20 weeks (must exceed WINDOW_SIZE + forecast_horizon)
    FOLD_STEP = 50    # Advance by ~10 weeks between folds
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if use_real_data:
        # Load real market data (always load on CPU, training loop moves to GPU)
        from data_loader import load_and_prepare_backtest
        
        print("\n=== Loading Real Market Data ===")
        folds, dates, tickers, NUM_STOCKS, NUM_MACROS = load_and_prepare_backtest(
            data_dir="./data/processed",
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            step_size=FOLD_STEP,
            window_size=WINDOW_SIZE,
            forecast_horizon=5,
            snapshot_step=1,
            device=torch.device('cpu')  # Load on CPU, move to GPU during training
        )
        
        print(f"\nReady for walk-forward backtest:")
        print(f"  {len(folds)} folds, {NUM_STOCKS} stocks, {NUM_MACROS} macro factors")
        
    else:
        # Generate synthetic data for demonstration
        NUM_STOCKS = 50
        NUM_MACROS = 4
        print("\n=== Generating Synthetic Data ===")
        total_timesteps = 500
        
        stock_data = torch.randn(NUM_STOCKS, total_timesteps, STOCK_DIM)
        macro_data = torch.randn(NUM_MACROS, total_timesteps, MACRO_DIM)
        returns_data = torch.randn(NUM_STOCKS, total_timesteps) * 0.02
        
        # Create single fold for synthetic data
        train_data = create_sequential_snapshots(
            stock_data, macro_data, returns_data,
            window_size=WINDOW_SIZE,
            step_size=5,
            forecast_horizon=5
        )
        # Wrap as single fold with empty validation for consistency
        folds = [(train_data, [])]
        print(f"Created {len(train_data)} training snapshots (synthetic)")
    
    # Create model
    model = MacroDGRCL(
        num_stocks=NUM_STOCKS,
        num_macros=NUM_MACROS,
        stock_feature_dim=STOCK_DIM,
        macro_feature_dim=MACRO_DIM,
        hidden_dim=HIDDEN_DIM,
        temporal_layers=2,
        mp_layers=2,
        heads=4,
        top_k=min(10, NUM_STOCKS - 1),  # Ensure top_k < num_stocks
        dropout=0.1,
        mc_dropout=0.3
    ).to(device)
    
    # Loss
    criterion = DGRCLLoss(
        ranking_margin=1.0,
        nce_temperature=0.07,
        nce_weight=0.1
    )
    
    # Walk-forward backtest loop
    all_fold_results = []
    
    for fold_idx, (train_data, val_data) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'='*60}")
        print(f"Training snapshots: {len(train_data)}")
        print(f"Validation snapshots: {len(val_data)}")
        
        # Reset optimizer and scheduler for each fold
        optimizer = AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        
        # Training loop for this fold with loss tracking
        print("\nTraining...")
        epoch_losses = []
        for epoch in range(NUM_EPOCHS):
            metrics = train_epoch(
                model=model,
                data_loader=train_data,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )
            epoch_losses.append(metrics['loss'])
            
            scheduler.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {metrics['loss']:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save training curve for this fold
        save_training_curve(
            epochs=list(range(1, NUM_EPOCHS + 1)),
            losses=epoch_losses,
            fold_idx=fold_idx,
            output_dir=VIS_DIR
        )
        
        # Validation (if we have validation data)
        if val_data:
            print("\nValidating...")
            val_metrics = evaluate(
                model=model,
                data_loader=val_data,
                criterion=criterion,
                device=device,
                n_mc_samples=30
            )
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Mean Uncertainty: {val_metrics['mean_uncertainty']:.6f}")
            all_fold_results.append({
                'fold': fold_idx + 1,
                'train_loss': metrics['loss'],
                'val_loss': val_metrics['loss'],
                'uncertainty': val_metrics['mean_uncertainty']
            })
        else:
            all_fold_results.append({
                'fold': fold_idx + 1,
                'train_loss': metrics['loss'],
                'val_loss': None,
                'uncertainty': None
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    
    val_losses = [r['val_loss'] for r in all_fold_results if r['val_loss'] is not None]
    if val_losses:
        print(f"Average Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"Best Fold: {np.argmin(val_losses) + 1} (Loss: {min(val_losses):.4f})")
    
    # Save backtest summary visualization with market regimes
    save_backtest_summary(all_fold_results, VIS_DIR, dates=dates if use_real_data else None)
    
    # Test MC Dropout uncertainty on last fold
    print("\nTesting MC Dropout uncertainty estimation...")
    model.eval()
    test_stock = train_data[0][0].to(device)
    test_macro = train_data[0][1].to(device)
    
    mu, sigma2, _ = model.predict_with_uncertainty(test_stock, test_macro, n_samples=50)
    
    print(f"Prediction mean: {mu.mean().item():.4f}")
    print(f"Prediction std (uncertainty): {sigma2.sqrt().mean().item():.4f}")
    
    print(f"\nBacktest complete! Results saved to: {VIS_DIR}")
    return all_fold_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Macro-DGRCL model")
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real market data from ./data/processed/ instead of synthetic data"
    )
    
    args = parser.parse_args()
    main(use_real_data=args.real_data)

