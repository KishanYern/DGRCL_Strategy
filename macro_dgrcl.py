"""
Macro-Aware DGRCL v1.5 - Sector-Aware Macro-Lag Architecture

A Heterogeneous Graph Neural Network for market-neutral trading with:
- Explicit Macro factor nodes (not feature vectors)
- Dynamic Stock→Stock edge learning via attention (SECTOR-CONSTRAINED)
- Hybrid Macro→Stock edges (fixed + learned)
- Multi-Task Head: Direction (binary) + Magnitude (regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Dict, Tuple, Optional, List


# =============================================================================
# HETERO GRAPH BUILDER
# =============================================================================

class HeteroGraphBuilder:
    """
    Constructs PyG HeteroData from raw stock/macro tensors.
    
    Node Types:
        - 'stock': N_s nodes with d_s=8 features
        - 'macro': N_m nodes with d_m=4 features
    
    Edge Types:
        - ('macro', 'influences', 'stock'): Hybrid (fixed + learned)
        - ('stock', 'relates', 'stock'): Dynamic via attention (Sector-Constrained)
    """
    
    def __init__(
        self,
        macro_stock_fixed_edges: Optional[torch.Tensor] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        tickers: Optional[List[str]] = None
    ):
        """
        Args:
            macro_stock_fixed_edges: [2, E] tensor of fixed macro→stock edges
            sector_mapping: Dict mapping Ticker → Sector Name
            tickers: List of tickers corresponding to stock_features indices
        """
        self.macro_stock_fixed_edges = macro_stock_fixed_edges
        self.sector_mapping = sector_mapping
        self.tickers = tickers
        
        # Precompute sector mask if mapping is provided
        self.sector_mask = None
        if sector_mapping and tickers:
            self._precompute_sector_mask()
    
    def _precompute_sector_mask(self):
        """
        Create a boolean mask [N, N] where Mask[i, j] = True iff Sector[i] == Sector[j].
        """
        N = len(self.tickers)
        sectors = [self.sector_mapping.get(t, "Unknown") for t in self.tickers]
        
        # Convert sectors to integer IDs
        unique_sectors = list(set(sectors))
        sector_to_id = {s: i for i, s in enumerate(unique_sectors)}
        sector_ids = torch.tensor([sector_to_id[s] for s in sectors])
        
        # Broadcast equality -> [N, N]
        # mask[i, j] is True if sector_ids[i] == sector_ids[j]
        self.sector_mask = sector_ids.unsqueeze(0) == sector_ids.unsqueeze(1)
        print(f"HeteroGraphBuilder: Precomputed sector mask for {N} stocks.")

    def build_correlation_edges(
        self,
        return_series: torch.Tensor,   # [N_s, T] rolling returns from training window
        active_mask: torch.Tensor,      # [N_s] bool
        top_k: int = 15,
        min_corr: float = 0.20,
    ):
        """
        Compute top-K pairwise Pearson correlation edges from training-window returns.

        Called once per fold on training-split returns (point-in-time safe). The
        resulting edges are held fixed for all snapshots within the fold.
        High-correlation pairs form the structural prior for the graph learner;
        the model learns whether to upweight or downweight them via the attention
        branch of DynamicGraphLearner.

        Args:
            return_series: [N_s, T] — daily log-returns over the training window
            active_mask:   [N_s] bool — inactive/padded stocks are excluded
            top_k:         Maximum neighbors to keep per node
            min_corr:      Minimum |correlation| threshold to include an edge

        Returns:
            edge_index:  [2, E] long tensor (src, dst)
            edge_weight: [E] float tensor of Pearson correlations in [-1, 1]
        """
        N_s = return_series.size(0)
        device = return_series.device

        # Restrict to active stocks only to avoid zero-padded rows
        active_idx = active_mask.nonzero(as_tuple=True)[0]  # [n_active]
        n_active = active_idx.numel()

        if n_active < 2:
            # Degenerate: return empty edge set
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.float32, device=device),
            )

        ret_active = return_series[active_idx].float()  # [n_active, T]

        # Mean-center each row for Pearson correlation
        ret_c = ret_active - ret_active.mean(dim=1, keepdim=True)  # [n_active, T]
        std = ret_c.norm(dim=1, keepdim=True).clamp(min=1e-8)      # [n_active, 1]
        ret_norm = ret_c / std                                       # [n_active, T]

        # Full pairwise correlation matrix [n_active, n_active]
        corr_matrix = (ret_norm @ ret_norm.T) / ret_active.size(1)  # [n, n]

        # Zero diagonal (self-loops not useful)
        corr_matrix.fill_diagonal_(0.0)

        # Top-K per row (absolute correlation)
        k = min(top_k, n_active - 1)
        abs_corr = corr_matrix.abs()
        topk_vals, topk_local_idx = torch.topk(abs_corr, k=k, dim=1)  # [n, k]

        # Apply minimum correlation threshold
        valid = topk_vals >= min_corr  # [n, k]

        # Build edges in the full-universe index space
        row_local = torch.arange(n_active, device=device).unsqueeze(1).expand_as(topk_local_idx)
        src_local = topk_local_idx[valid]  # neighbor (local index)
        dst_local = row_local[valid]        # receiving node (local index)

        # Map local active indices back to full universe indices
        src_full = active_idx[src_local]
        dst_full = active_idx[dst_local]

        edge_index = torch.stack([src_full, dst_full], dim=0)  # [2, E]
        edge_weight = corr_matrix[src_local, dst_local]        # raw Pearson in [-1,1]

        return edge_index, edge_weight

    def build(
        self,
        stock_features: torch.Tensor,  # [N_s, T, d_s]
        macro_features: torch.Tensor,  # [N_m, T, d_m]
        stock_stock_edges: Optional[torch.Tensor] = None,  # [2, E_ss]
        macro_stock_learned_edges: Optional[torch.Tensor] = None,  # [2, E_ms]
    ) -> HeteroData:
        """
        Build a HeteroData object for a single timestep.
        
        Returns:
            HeteroData with proper node features and edge indices
        """
        data = HeteroData()
        
        # Node features
        data['stock'].x = stock_features
        data['stock'].num_nodes = stock_features.size(0)
        
        data['macro'].x = macro_features
        data['macro'].num_nodes = macro_features.size(0)
        
        # Stock→Stock edges (dynamic, computed by DynamicGraphLearner)
        if stock_stock_edges is not None:
            data['stock', 'relates', 'stock'].edge_index = stock_stock_edges
        
        # Macro→Stock edges (hybrid: fixed + learned)
        macro_stock_edges = self._build_macro_stock_edges(
            macro_features.size(0),
            stock_features.size(0),
            macro_stock_learned_edges
        )
        if macro_stock_edges is not None:
            data['macro', 'influences', 'stock'].edge_index = macro_stock_edges
        
        return data
    
    def _build_macro_stock_edges(
        self,
        num_macro: int,
        num_stock: int,
        learned_edges: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Combine fixed and learned macro→stock edges."""
        edges_list = []
        
        if self.macro_stock_fixed_edges is not None:
            edges_list.append(self.macro_stock_fixed_edges)
        
        if learned_edges is not None:
            edges_list.append(learned_edges)
        
        # If no fixed edges defined, create default: all macros connect to all stocks
        if len(edges_list) == 0:
            src = torch.arange(num_macro).repeat_interleave(num_stock)
            dst = torch.arange(num_stock).repeat(num_macro)
            return torch.stack([src, dst], dim=0)
        else:
            return torch.cat(edges_list, dim=1)


# =============================================================================
# TEMPORAL ENCODER
# =============================================================================

class TemporalEncoder(nn.Module):
    """
    Shared LSTM that projects raw time-series into latent embeddings.
    
    Input: [batch, T, F] where F is feature dimension
    Output: [batch, H] where H is hidden dimension (last hidden state)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, T, F] time-series features
        Returns:
            h: [N, H] latent embeddings (last hidden state)
        """
        # LSTM output: (output, (h_n, c_n))
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape: [num_layers * num_directions, batch, hidden_size]
        # Take the last layer's hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        
        return h


# =============================================================================
# DYNAMIC GRAPH LEARNER
# =============================================================================

class DynamicGraphLearner(nn.Module):
    """
    Hybrid dynamic graph learner: correlation prior + multi-head GAT attention.

    Combines two complementary signals:
      1. Structural prior: pre-computed per-fold Pearson correlation edges
         capture actual return co-movement independent of sector labels.
      2. Learned attention: true multi-head GAT attention (concat [W·h_i || W·h_j]
         per head, separate attention vector per head) learns which connections
         are predictively useful beyond raw correlation.

    The final edge weight is: softmax_per_node(α_attn + λ·corr_prior)
    where λ is a learnable scalar initialized to 1.0. If no correlation prior
    is provided the model falls back to pure attention.

    Compared to the previous single-vector `a` design:
      - Per-head projection avoids the bottleneck of a single attention direction
      - Pre-normalized softmax output (rather than raw LeakyReLU scores) removes
        the double-softmax issue in MacroPropagation
      - Cross-sector edges allowed when correlation is high (sector mask moved
        to be optional, not mandatory)
    """

    def __init__(
        self,
        hidden_dim: int,
        top_k: int = 15,
        num_heads: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_heads = num_heads
        self.temperature = temperature
        self.head_dim = max(hidden_dim // num_heads, 1)

        # Per-head projections: W_k ∈ R^{H × head_dim}
        self.W = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)

        # Per-head attention vectors: a_k ∈ R^{2*head_dim}
        self.att = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))

        # Learnable weight balancing attention vs. correlation prior
        self.corr_lambda = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(p=dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        # Use Glorot-uniform for per-head attention vectors
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(
        self,
        embeddings: torch.Tensor,           # [N, H]
        sector_mask: Optional[torch.Tensor] = None,   # [N, N] bool — now optional
        active_mask: Optional[torch.Tensor] = None,   # [N] bool
        corr_edge_index: Optional[torch.Tensor] = None,  # [2, E_c] correlation prior edges
        corr_edge_weight: Optional[torch.Tensor] = None,  # [E_c] Pearson weights
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute dynamic adjacency combining correlation prior and GAT attention.

        Args:
            embeddings:       [N, H] LSTM hidden states
            sector_mask:      Optional [N, N] bool — if provided, restricts edges
                              to same-sector pairs (used as a hard gate)
            active_mask:      [N] bool — inactive stocks get no edges
            corr_edge_index:  [2, E_c] pre-computed correlation edges (per fold)
            corr_edge_weight: [E_c] Pearson correlations for prior edges
            return_weights:   Whether to return pre-normalized edge weights

        Returns:
            edge_index:  [2, E] sparse edge indices
            edge_weight: [E] softmax-normalized weights in [0, 1] (or None)
        """
        N = embeddings.size(0)
        device = embeddings.device

        # Project and reshape: [N, num_heads, head_dim]
        h = self.W(embeddings).view(N, self.num_heads, self.head_dim)

        # --- Build candidate edge set ---
        # Start with correlation prior edges (cross-sector allowed)
        if corr_edge_index is not None and corr_edge_index.size(1) > 0:
            src_c, dst_c = corr_edge_index
            # Augment with full top_k attention-only edges for remaining pairs
            # Strategy: compute full attention matrix, take top_k per node,
            # then union with correlation edges
            use_corr = True
        else:
            use_corr = False

        # Compute full pairwise GAT attention scores [N, N]
        # a_k^T [W_k·h_i || W_k·h_j]  averaged across heads
        h_i = h.unsqueeze(1).expand(N, N, self.num_heads, self.head_dim)  # [N, N, H, D]
        h_j = h.unsqueeze(0).expand(N, N, self.num_heads, self.head_dim)  # [N, N, H, D]
        concat = torch.cat([h_i, h_j], dim=-1)  # [N, N, H, 2D]

        # self.att: [H, 2D] -> broadcast over [N, N, H, 2D] -> [N, N, H]
        attn_scores = (concat * self.att.unsqueeze(0).unsqueeze(0)).sum(-1)  # [N, N, H]
        attn_scores = F.leaky_relu(attn_scores, negative_slope=0.2)
        attn_scores = attn_scores.mean(dim=-1) / self.temperature  # [N, N] mean over heads

        # --- Apply masks ---
        # Combined mask: both nodes must be active
        combined_mask = torch.ones(N, N, dtype=torch.bool, device=device)
        if active_mask is not None:
            active_pairs = active_mask.unsqueeze(0) & active_mask.unsqueeze(1)
            combined_mask = combined_mask & active_pairs
        if sector_mask is not None:
            # Sector mask now optional: only apply if explicitly passed
            combined_mask = combined_mask & sector_mask.to(device)

        attn_scores = attn_scores.masked_fill(~combined_mask, float('-inf'))

        # --- Add correlation prior to attention scores ---
        if use_corr:
            # Scatter corr weights into a dense [N, N] prior matrix
            corr_matrix = torch.zeros(N, N, device=device)
            src_c_dev = corr_edge_index[0].to(device)
            dst_c_dev = corr_edge_index[1].to(device)
            w_dev = corr_edge_weight.to(device).float()
            corr_matrix[src_c_dev, dst_c_dev] = w_dev
            # Only add prior where edges are valid (mask-consistent)
            corr_matrix = corr_matrix * combined_mask.float()
            attn_scores = attn_scores + self.corr_lambda * corr_matrix
        else:
            # No correlation prior: add zero contribution so corr_lambda always
            # participates in the computation graph and receives gradients.
            attn_scores = attn_scores + self.corr_lambda * 0.0

        # --- Top-k selection per node ---
        k = min(self.top_k, N - 1)
        topk_scores, topk_idx = torch.topk(attn_scores, k=k, dim=1)  # [N, k]

        # Filter truly invalid edges (-inf from masking)
        valid = torch.isfinite(topk_scores)  # [N, k]

        row_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, k)  # [N, k]
        src_idx = topk_idx[valid]   # neighbor j
        dst_idx = row_idx[valid]    # receiving node i

        edge_index = torch.stack([src_idx, dst_idx], dim=0)  # [2, E]

        if return_weights:
            # Pre-normalize with softmax PER destination node so downstream
            # MacroPropagation can use weights directly without re-softmax.
            # We compute a per-node softmax over selected neighbors.
            raw = topk_scores.clone()
            raw[~valid] = float('-inf')
            # Softmax over k dimension (per row = per receiving node)
            normalized = torch.softmax(raw, dim=1)  # [N, k]
            edge_weight = normalized[valid]  # [E] in [0, 1], sums to 1 per node
            edge_weight = self.dropout(edge_weight)
            return edge_index, edge_weight

        return edge_index, None


# =============================================================================
# MACRO PROPAGATION LAYER
# =============================================================================

class MacroPropagation(MessagePassing):
    """
    Custom message passing layer where Stock nodes aggregate messages from:
    1. Top-k similar Stock neighbors (dynamic edges)
    2. Connected Macro nodes (hybrid edges)
    
    Equation:
        h'_i = σ(Σ_{j ∈ N_s(i)} α_ij W_s h_j + Σ_{m ∈ N_m(i)} β_im W_m h_m)
    
    This is NOT a standard GATConv - it implements dual aggregation with
    separate transformations for stock and macro messages.
    """
    
    def __init__(
        self,
        stock_dim: int,
        macro_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.stock_dim = stock_dim
        self.macro_dim = macro_dim
        self.out_dim = out_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Separate transformations for stock and macro
        self.W_stock = nn.Linear(stock_dim, heads * out_dim, bias=False)
        self.W_macro = nn.Linear(macro_dim, heads * out_dim, bias=False)

        # att_stock removed: DynamicGraphLearner now always returns pre-normalized
        # per-edge weights (softmax over selected neighbors). _aggregate_stock uses
        # those weights directly and no longer needs an in-layer attention vector.

        # Attention parameters for macro-stock
        self.att_macro = nn.Parameter(torch.randn(1, heads, 2 * out_dim))
        
        # Output projection
        self.out_proj = nn.Linear(heads * out_dim, out_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_stock.weight)
        nn.init.xavier_uniform_(self.W_macro.weight)
        nn.init.xavier_uniform_(self.att_macro)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        stock_h: torch.Tensor,  # [N_s, H_s]
        macro_h: torch.Tensor,  # [N_m, H_m]
        stock_stock_edge_index: torch.Tensor,  # [2, E_ss]
        macro_stock_edge_index: torch.Tensor,  # [2, E_ms]
        stock_stock_edge_weight: Optional[torch.Tensor] = None,  # [E_ss]
    ) -> torch.Tensor:
        """
        Dual aggregation from stock and macro neighbors.
        
        Returns:
            Updated stock embeddings [N_s, out_dim]
        """
        N_s = stock_h.size(0)
        H = self.heads
        D = self.out_dim
        
        # Transform stock embeddings
        stock_h_transformed = self.W_stock(stock_h).view(-1, H, D)  # [N_s, H, D]
        
        # Transform macro embeddings
        macro_h_transformed = self.W_macro(macro_h).view(-1, H, D)  # [N_m, H, D]
        
        # 1. Stock-Stock aggregation
        stock_agg = self._aggregate_stock(
            stock_h_transformed,
            stock_stock_edge_index,
            stock_stock_edge_weight
        )  # [N_s, H, D]
        
        # 2. Macro-Stock aggregation
        macro_agg = self._aggregate_macro(
            stock_h_transformed,
            macro_h_transformed,
            macro_stock_edge_index
        )  # [N_s, H, D]
        
        # Combine aggregations
        out = stock_agg + macro_agg  # [N_s, H, D]
        out = out.view(-1, H * D)  # [N_s, H*D]
        out = self.out_proj(out)  # [N_s, out_dim]
        out = F.elu(out)
        
        return out
    
    def _aggregate_stock(
        self,
        h: torch.Tensor,           # [N_s, H, D]
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate messages from stock neighbors using pre-normalized edge weights.

        DynamicGraphLearner always returns softmax-normalized per-edge weights
        (one probability distribution per destination node). We apply them directly
        as a weighted sum over neighbor embeddings — no redundant in-layer softmax.
        """
        src, dst = edge_index
        h_j = h[src]  # [E, H, D]

        if edge_weight is not None:
            # Pre-normalized weights from DynamicGraphLearner
            w = edge_weight.view(-1, 1, 1).to(h.dtype)  # [E, 1, 1]
            out_msgs = h_j * w
        else:
            # Fallback: uniform aggregation (no edge_weight provided)
            out_msgs = h_j

        out = torch.zeros(h.shape, dtype=out_msgs.dtype, device=h.device)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand_as(out_msgs), out_msgs)
        return out
    
    def _aggregate_macro(
        self,
        stock_h: torch.Tensor,  # [N_s, H, D]
        macro_h: torch.Tensor,  # [N_m, H, D]
        edge_index: torch.Tensor  # [2, E_ms] where src=macro, dst=stock
    ) -> torch.Tensor:
        """Aggregate messages from macro neighbors."""
        src, dst = edge_index  # src: macro indices, dst: stock indices
        
        # Get embeddings
        h_stock = stock_h[dst]  # [E, H, D]
        h_macro = macro_h[src]  # [E, H, D]
        
        # Compute attention β_im
        concat = torch.cat([h_stock, h_macro], dim=-1)  # [E, H, 2D]
        beta = (concat * self.att_macro).sum(dim=-1)  # [E, H]
        beta = F.leaky_relu(beta, negative_slope=self.negative_slope)
        
        # Normalize via softmax over macro neighbors per stock
        beta = softmax(beta, dst, num_nodes=stock_h.size(0))  # [E, H]
        beta = F.dropout(beta, p=self.dropout, training=self.training)
        
        # Aggregate macro messages to stock nodes
        out = h_macro * beta.unsqueeze(-1)  # [E, H, D]
        # Use matching dtype for scatter_add_ (autocast may mix FP16/FP32)
        out = torch.zeros(stock_h.shape, dtype=out.dtype, device=stock_h.device).scatter_add_(
            0, dst.view(-1, 1, 1).expand_as(out), out
        )
        
        return out


# =============================================================================
# MULTI-TASK HEAD
# =============================================================================

class MultiTaskHead(nn.Module):
    """
    Multi-Task output head for decoupled direction and magnitude prediction.
    
    Architecture:
        Shared:  Linear(H, H) -> ReLU -> Dropout
        dir_head: Linear(H, H//2) -> ReLU -> Dropout -> Linear(1)  [logits]
        mag_head: Linear(H, H//2) -> ReLU -> Dropout -> Linear(1) -> Softplus  [positive]
    
    Direction predicts P(return > cross-sectional median) via BCEWithLogitsLoss.
    Magnitude predicts |R_t| via MSELoss, guaranteed non-negative by Softplus.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Shared feature extraction block
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Direction head: outputs unbounded logits for BCEWithLogitsLoss
        self.dir_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Magnitude head: outputs positive scalar via Softplus
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through shared block then both task heads.
        
        Args:
            x: [N, hidden_dim] GNN embeddings
            
        Returns:
            dir_logits: [N, 1] unbounded logits for direction classification
            mag_preds:  [N, 1] positive magnitude predictions
        """
        shared_features = self.shared(x)  # [N, H]
        dir_logits = self.dir_head(shared_features)  # [N, 1]
        mag_preds = self.mag_head(shared_features)   # [N, 1]
        return dir_logits, mag_preds


# =============================================================================
# MACRO DGRCL - MAIN MODEL
# =============================================================================

class MacroDGRCL(nn.Module):
    """
    Macro-Aware Dynamic Graph Relation Contrastive Learning Model v1.5.
    
    Multi-Task Learning Architecture:
        1. TemporalEncoder: LSTM projects time-series → embeddings
        2. DynamicGraphLearner: Computes dynamic Stock→Stock adjacency (Sector-Constrained)
        3. MacroPropagation: Custom message passing with dual aggregation
        4. MultiTaskHead: Direction (binary) + Magnitude (regression)
    
    Forward returns (direction_logits, magnitude_preds) for MTL training.
    """
    
    def __init__(
        self,
        num_stocks: int,
        num_macros: int,
        stock_feature_dim: int = 8,  # ['Close', 'High', 'Low', 'Log_Vol', 'RSI_14', 'MACD', 'Volatility_5', 'Returns']
        macro_feature_dim: int = 4,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        mp_layers: int = 2,
        heads: int = 4,
        top_k: int = 10,
        dropout: float = 0.1,
        head_dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_stocks = num_stocks
        self.num_macros = num_macros
        self.hidden_dim = hidden_dim
        
        # Temporal Encoders (separate for stock and macro due to different dims)
        self.stock_encoder = TemporalEncoder(
            input_dim=stock_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout
        )
        self.macro_encoder = TemporalEncoder(
            input_dim=macro_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout
        )

        # Embedding Normalization — separate norms for stocks and macros.
        # Stocks and macros come from different domains (equity prices vs. macro
        # indicators) with different latent-space statistics. A single shared
        # LayerNorm would force one set of learned scale/bias to serve both,
        # which conflates gradients from fundamentally different distributions.
        self.stock_embedding_norm = nn.LayerNorm(hidden_dim)
        self.macro_embedding_norm = nn.LayerNorm(hidden_dim)
        
        # Dynamic Graph Learner (upgraded: multi-head GAT + correlation prior)
        self.graph_learner = DynamicGraphLearner(
            hidden_dim=hidden_dim,
            top_k=top_k,
            num_heads=heads,
            dropout=dropout,
        )
        
        # Message Passing Layers
        self.mp_layers = nn.ModuleList([
            MacroPropagation(
                stock_dim=hidden_dim,
                macro_dim=hidden_dim,
                out_dim=hidden_dim,
                heads=heads,
                dropout=dropout
            )
            for _ in range(mp_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(mp_layers)
        ])
        
        # Multi-Task Head (Direction + Magnitude)
        self.output_head = MultiTaskHead(
            hidden_dim=hidden_dim,
            dropout=head_dropout
        )
        
        # Graph builder
        self.graph_builder = HeteroGraphBuilder()
    
    def forward(
        self,
        stock_features: torch.Tensor,  # [N_s, T, d_s]
        macro_features: torch.Tensor,  # [N_m, T, d_m]
        macro_stock_edges: Optional[torch.Tensor] = None,  # [2, E_ms]
        sector_mask: Optional[torch.Tensor] = None,        # [N_s, N_s] boolean mask
        active_mask: Optional[torch.Tensor] = None,        # [N_s] boolean mask
        corr_edge_index: Optional[torch.Tensor] = None,   # [2, E_c] per-fold correlation edges
        corr_edge_weight: Optional[torch.Tensor] = None,  # [E_c] Pearson weights
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN backbone and Multi-Task head.

        Args:
            stock_features:   [N_s, T, d_s] stock time-series
            macro_features:   [N_m, T, d_m] macro time-series
            macro_stock_edges: [2, E_ms] macro→stock edge index
            sector_mask:      Optional [N_s, N_s] boolean mask — when provided,
                              restricts learned edges to within-sector pairs.
                              Leave None to allow cross-sector edges (recommended
                              when correlation prior is active).
            active_mask:      [N_s] bool — inactive padded stocks
            corr_edge_index:  [2, E_c] pre-computed correlation edges (per fold)
            corr_edge_weight: [E_c] Pearson correlations for structural prior edges

        Returns:
            direction_logits: [N_s, 1] unbounded logits for P(return > median)
            magnitude_preds:  [N_s, 1] predicted |R_t| (non-negative)
        """
        # 1. Temporal Encoding
        stock_h = self.stock_encoder(stock_features)  # [N_s, H]
        macro_h = self.macro_encoder(macro_features)  # [N_m, H]

        # SAFEGUARD 1: Zero out inactive embeddings BEFORE LayerNorm
        # LSTM produces non-zero outputs even for zero inputs (due to bias).
        # These meaningless vectors would skew LayerNorm batch statistics.
        if active_mask is not None:
            stock_h = stock_h * active_mask.unsqueeze(-1).float()  # [N_s, H]

        # Normalize embeddings (separate norms per domain — see __init__)
        stock_h = self.stock_embedding_norm(stock_h)
        macro_h = self.macro_embedding_norm(macro_h)
        
        # LayerNorm's learned bias re-introduces non-zero values for inactive nodes
        # Re-zero after norm to prevent these from entering the graph learner
        if active_mask is not None:
            stock_h = stock_h * active_mask.unsqueeze(-1).float()
        
        # 2. Dynamic Graph Learning (Stock→Stock)
        # Pass correlation prior edges; sector_mask is now optional (pass None to
        # allow cross-sector edges when correlation prior is active).
        stock_stock_edges, edge_weights = self.graph_learner(
            stock_h,
            sector_mask=sector_mask,    # None when using correlation-based adjacency
            active_mask=active_mask,
            corr_edge_index=corr_edge_index,
            corr_edge_weight=corr_edge_weight,
            return_weights=True
        )
        
        # 3. Build Macro→Stock edges if not provided
        if macro_stock_edges is None:
            # Default: all macros connect to all stocks
            N_m, N_s = macro_features.size(0), stock_features.size(0)
            src = torch.arange(N_m, device=stock_features.device).repeat_interleave(N_s)
            dst = torch.arange(N_s, device=stock_features.device).repeat(N_m)
            macro_stock_edges = torch.stack([src, dst], dim=0)
        
        # SAFEGUARD 3: Filter macro→stock edges to active destinations only
        # Prevents macro embeddings from broadcasting into zero-padded inactive nodes
        if active_mask is not None:
            dst_active = active_mask[macro_stock_edges[1]]  # [E] bool
            macro_stock_edges = macro_stock_edges[:, dst_active]
        
        # 4. Message Passing
        h = stock_h
        for i, (mp_layer, ln) in enumerate(zip(self.mp_layers, self.layer_norms)):
            h_new = mp_layer(
                stock_h=h,
                macro_h=macro_h,
                stock_stock_edge_index=stock_stock_edges,
                macro_stock_edge_index=macro_stock_edges,
                stock_stock_edge_weight=edge_weights
            )
            # Residual connection + LayerNorm
            h = ln(h + h_new)
            # SAFEGUARD 4: Re-zero inactive nodes after each residual+LayerNorm.
            # LayerNorm's learned bias can re-introduce non-zero values for
            # nodes that were zeroed by Safeguard 1. Without re-zeroing here,
            # those spurious embeddings propagate into subsequent MP layers
            # via the residual path, contaminating active-stock aggregation.
            if active_mask is not None:
                h = h * active_mask.unsqueeze(-1).float()
        
        # 5. Multi-Task Head
        direction_logits, magnitude_preds = self.output_head(h)
        
        # SAFEGUARD 5: Mask inactive logits
        # Even with zeroed inputs from Safeguard 4, the linear layers
        # inside output_head have biases that produce non-zero outputs.
        # This keeps inactive node logits strictly at 0.0.
        if active_mask is not None:
            direction_logits = direction_logits * active_mask.unsqueeze(-1).float()
            magnitude_preds = magnitude_preds * active_mask.unsqueeze(-1).float()
        
        return direction_logits, magnitude_preds
