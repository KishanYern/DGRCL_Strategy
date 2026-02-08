"""
Loss Functions for Macro-Aware DGRCL v1.1

Implements:
- Pairwise Margin Ranking Loss (for relative rank prediction)
- InfoNCE Contrastive Loss (for representation regularization)
- Combined DGRCL Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def pairwise_ranking_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """
    Pairwise Margin Ranking Loss.
    
    For pairs (i, j) where y_i > y_j (stock i has higher return):
        L = max(0, margin - (score_i - score_j))
    
    We want the model to assign higher scores to higher-return stocks.
    
    Args:
        scores: [N, 1] predicted ranking scores
        returns: [N] actual returns (ground truth for ranking)
        margin: Margin for the ranking loss
        
    Returns:
        Scalar loss value
    """
    scores = scores.squeeze(-1)  # [N]
    N = scores.size(0)
    
    # Create all pairs
    scores_i = scores.unsqueeze(1)  # [N, 1]
    scores_j = scores.unsqueeze(0)  # [1, N]
    
    returns_i = returns.unsqueeze(1)  # [N, 1]
    returns_j = returns.unsqueeze(0)  # [1, N]
    
    # Mask: y_i > y_j (stock i should be ranked higher)
    mask = (returns_i > returns_j).float()  # [N, N]
    
    # Pairwise loss: max(0, margin - (s_i - s_j)) where y_i > y_j
    diff = scores_i - scores_j  # [N, N]
    loss_matrix = F.relu(margin - diff)  # [N, N]
    
    # Apply mask and compute mean
    loss = (loss_matrix * mask).sum()
    num_pairs = mask.sum()
    
    if num_pairs > 0:
        loss = loss / num_pairs
    
    return loss


def info_nce_loss(
    embeddings: torch.Tensor,
    temperature: float = 0.07,
    returns: Optional[torch.Tensor] = None,
    similarity_threshold: float = 0.5
) -> torch.Tensor:
    """
    InfoNCE Contrastive Loss.
    
    Positive pairs: stocks with similar returns
    Negative pairs: all other stocks
    
    L = -log(exp(sim(z_i, z_i+) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
    
    Args:
        embeddings: [N, H] node embeddings
        temperature: Softmax temperature τ
        returns: [N] returns to define positive pairs (required for meaningful contrastive learning)
        similarity_threshold: Threshold for defining positive pairs by returns (in std units)
        
    Returns:
        Scalar loss value
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)  # [N, H]
    N = embeddings.size(0)
    device = embeddings.device
    
    # Compute similarity matrix (normalized embeddings give sim in [-1, 1])
    sim_matrix = torch.mm(embeddings, embeddings.t())  # [N, N]
    
    # Clamp temperature-scaled similarities for numerical stability
    logits = sim_matrix / temperature
    logits = logits.clamp(-50, 50)  # Prevent overflow in exp
    
    # Create positive pair mask based on returns
    if returns is not None:
        # Positive pairs: stocks with similar returns
        returns_i = returns.unsqueeze(1)  # [N, 1]
        returns_j = returns.unsqueeze(0)  # [1, N]
        return_diff = torch.abs(returns_i - returns_j)  # [N, N]
        
        # Define positives as those with similar returns
        return_std = returns.std() + 1e-8
        positive_mask = (return_diff < similarity_threshold * return_std).float()
    else:
        # Without returns, use top-k most similar embeddings as pseudo-positives
        # (This is a fallback - ideally you always have returns for financial data)
        k = max(1, N // 10)  # Top 10% as positives
        _, topk_indices = torch.topk(logits, k=k+1, dim=1)  # +1 to include self
        positive_mask = torch.zeros_like(logits)
        positive_mask.scatter_(1, topk_indices, 1.0)
    
    # Remove self from both positives and denominators
    self_mask = torch.eye(N, device=device).bool()
    positive_mask = positive_mask.masked_fill(self_mask, 0)
    
    # Check if there are any positives per row
    has_positives = positive_mask.sum(dim=1) > 0  # [N]
    
    if not has_positives.any():
        # No valid positive pairs - return zero loss to avoid NaN
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Use large negative value instead of -inf for numerical stability
    LARGE_NEG = -1e9
    logits_masked = logits.masked_fill(self_mask, LARGE_NEG)
    
    # Compute log-softmax with numerical stability
    log_prob = F.log_softmax(logits_masked, dim=1)  # [N, N]
    
    # Clamp log_prob to avoid -inf propagating
    log_prob = log_prob.clamp(min=-100)
    
    # Normalize positive mask per row
    pos_count = positive_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [N, 1]
    positive_mask_normalized = positive_mask / pos_count
    
    # Compute loss: -mean of log probs for positive pairs
    loss_per_sample = -(log_prob * positive_mask_normalized).sum(dim=1)  # [N]
    
    # Only compute loss for samples with valid positives
    loss = loss_per_sample[has_positives].mean()
    
    return loss


def dgrcl_loss(
    scores: torch.Tensor,
    embeddings: torch.Tensor,
    returns: torch.Tensor,
    ranking_margin: float = 1.0,
    nce_temperature: float = 0.07,
    nce_weight: float = 0.1,
    similarity_threshold: float = 0.5
) -> torch.Tensor:
    """
    Combined DGRCL Loss.
    
    L = L_rank + λ * L_NCE
    
    Args:
        scores: [N, 1] predicted ranking scores
        embeddings: [N, H] node embeddings for contrastive loss
        returns: [N] actual returns
        ranking_margin: Margin for ranking loss
        nce_temperature: Temperature for InfoNCE
        nce_weight: Weight λ for contrastive term
        similarity_threshold: Threshold for positive pairs in InfoNCE
        
    Returns:
        Combined loss value
    """
    l_rank = pairwise_ranking_loss(scores, returns, margin=ranking_margin)
    l_nce = info_nce_loss(
        embeddings,
        temperature=nce_temperature,
        returns=returns,
        similarity_threshold=similarity_threshold
    )
    
    return l_rank + nce_weight * l_nce


class DGRCLLoss(nn.Module):
    """
    Module wrapper for DGRCL loss with configurable hyperparameters.
    """
    
    def __init__(
        self,
        ranking_margin: float = 1.0,
        nce_temperature: float = 0.07,
        nce_weight: float = 0.1,
        similarity_threshold: float = 0.5
    ):
        super().__init__()
        self.ranking_margin = ranking_margin
        self.nce_temperature = nce_temperature
        self.nce_weight = nce_weight
        self.similarity_threshold = similarity_threshold
    
    def forward(
        self,
        scores: torch.Tensor,
        embeddings: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        return dgrcl_loss(
            scores=scores,
            embeddings=embeddings,
            returns=returns,
            ranking_margin=self.ranking_margin,
            nce_temperature=self.nce_temperature,
            nce_weight=self.nce_weight,
            similarity_threshold=self.similarity_threshold
        )
