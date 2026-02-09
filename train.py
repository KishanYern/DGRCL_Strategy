"""
Training Loop for Macro-Aware DGRCL v1.1

Implements:
- train_step: Single batch forward/backward pass
- train_epoch: Full epoch training
- DataLoader utilities for sequential snapshots
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, Optional, List
import numpy as np

from macro_dgrcl import MacroDGRCL
from losses import DGRCLLoss


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


def main():
    """Example training loop."""
    # Hyperparameters
    NUM_STOCKS = 50
    NUM_MACROS = 4
    STOCK_DIM = 8  # ['Close', 'High', 'Low', 'Log_Vol', 'RSI_14', 'MACD', 'Volatility_5', 'Returns']
    MACRO_DIM = 4
    HIDDEN_DIM = 64
    WINDOW_SIZE = 60
    NUM_EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
        top_k=10,
        dropout=0.1,
        mc_dropout=0.3
    ).to(device)
    
    # Loss and optimizer
    criterion = DGRCLLoss(
        ranking_margin=1.0,
        nce_temperature=0.07,
        nce_weight=0.1
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Generate synthetic data for demonstration
    print("Generating synthetic data...")
    total_timesteps = 500
    
    stock_data = torch.randn(NUM_STOCKS, total_timesteps, STOCK_DIM)
    macro_data = torch.randn(NUM_MACROS, total_timesteps, MACRO_DIM)
    returns_data = torch.randn(NUM_STOCKS, total_timesteps) * 0.02  # ~2% daily vol
    
    # Create snapshots
    train_data = create_sequential_snapshots(
        stock_data, macro_data, returns_data,
        window_size=WINDOW_SIZE,
        step_size=5,
        forecast_horizon=5
    )
    
    print(f"Created {len(train_data)} training snapshots")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        metrics = train_epoch(
            model=model,
            data_loader=train_data,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {metrics['loss']:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Test MC Dropout uncertainty
    print("\nTesting MC Dropout uncertainty estimation...")
    model.eval()
    test_stock = train_data[0][0].unsqueeze(0).squeeze(0).to(device)  # Use first snapshot
    test_macro = train_data[0][1].unsqueeze(0).squeeze(0).to(device)
    
    mu, sigma2, _ = model.predict_with_uncertainty(test_stock, test_macro, n_samples=50)
    
    print(f"Prediction mean: {mu.mean().item():.4f}")
    print(f"Prediction std (uncertainty): {sigma2.sqrt().mean().item():.4f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
