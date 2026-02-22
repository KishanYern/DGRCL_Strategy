"""
Macro-Aware DGRCL v1.4 - Sector-Aware Macro-Lag Architecture

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
    Computes dynamic adjacency matrix for Stock→Stock edges via attention.
    
    SECTOR-CONSTRAINED:
    Applies a sector mask to attention scores so that stocks ONLY attend
    to other stocks in the same sector.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        top_k: int = 10,
        temperature: float = 1.0
    ):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        
        # Learnable projection for attention
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.a = nn.Parameter(torch.randn(2 * hidden_dim))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        # Use Xavier-scale std so attention scores have meaningful variance.
        # std=0.01 on a 2H-dim vector produces near-zero dot products,
        # collapsing all attention scores to ~0 (uniform attention).
        nn.init.normal_(self.a, std=1.0 / (2 * self.W.in_features) ** 0.5)
    
    def forward(
        self,
        embeddings: torch.Tensor,  # [N, H]
        sector_mask: Optional[torch.Tensor] = None, # [N, N] bool
        active_mask: Optional[torch.Tensor] = None,  # [N] bool
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute dynamic adjacency via attention.
        
        Args:
            embeddings: [N, H] node embeddings
            sector_mask: [N, N] boolean mask (True = allowed edge)
            return_weights: Whether to return attention weights
            
        Returns:
            edge_index: [2, E] sparse edge indices (top-k per node)
            edge_weight: [E] attention weights (if return_weights=True)
        """
        N = embeddings.size(0)
        
        # Project embeddings
        h = self.W(embeddings)  # [N, H]
        
        # Compute pairwise attention scores
        # a^T [W h_i || W h_j] for all pairs
        h_repeat = h.unsqueeze(1).repeat(1, N, 1)  # [N, N, H]
        h_repeat_t = h.unsqueeze(0).repeat(N, 1, 1)  # [N, N, H]
        
        concat = torch.cat([h_repeat, h_repeat_t], dim=-1)  # [N, N, 2H]
        attention = F.leaky_relu(torch.matmul(concat, self.a), negative_slope=0.2)  # [N, N]
        attention = attention / self.temperature
        
        # --- COMBINE SECTOR MASK + ACTIVE MASK ---
        # Build combined mask: both nodes must be active AND in same sector
        combined_mask = None
        if active_mask is not None:
            # Pairwise active mask: both src and dst must be active
            active_pairs = active_mask.unsqueeze(0) & active_mask.unsqueeze(1)  # [N, N]
            combined_mask = active_pairs
        
        if sector_mask is not None:
            if sector_mask.shape != attention.shape:
               raise ValueError(f"Sector mask shape {sector_mask.shape} mismatch with attention {attention.shape}")
            if combined_mask is not None:
                combined_mask = combined_mask & sector_mask
            else:
                combined_mask = sector_mask
        
        if combined_mask is not None:
            # Mask out invalid pairs with -inf.
            # float('-inf') is safe for all dtypes (FP16/FP32/BF16) and
            # guarantees isfinite() cleanly separates valid from masked edges.
            attention = attention.masked_fill(~combined_mask, float('-inf'))

        # Top-k selection per row (each node keeps top-k neighbors)
        k = min(self.top_k, N - 1)
        topk_values, topk_indices = torch.topk(attention, k=k, dim=1)  # [N, k]
        
        # Filter out masked edges (those filled with -inf above).
        # isfinite() is dtype-agnostic and correctly identifies -inf in FP16/FP32/BF16.
        valid_mask = torch.isfinite(topk_values)  # [N, k]
        
        # Build sparse edge_index
        # CRITICAL: In PyG MessagePassing, info flows src → dst
        # Node i computed which neighbors j are relevant, so i should RECEIVE from j
        # Therefore edges are: src=j (neighbors), dst=i (computing node)
        
        # Create row indices [N, k]
        row_indices = torch.arange(N, device=embeddings.device).unsqueeze(1).repeat(1, k)  # [N, k]
        
        # Apply filter
        src_indices = topk_indices[valid_mask]  # neighbors j (flattened)
        dst_indices = row_indices[valid_mask]   # node i (flattened)
        
        # Edges: j → i
        edge_index = torch.stack([src_indices, dst_indices], dim=0)  # [2, E_valid]
        
        if return_weights:
            # Normalize weights via softmax over selected neighbors (subset)
            # We need to re-softmax only over valid edges per node?
            # Actually, the original softmax was over all N. 
            # But usually we softmax over the selected neighbors for GAT-style.
            # Here we just return the raw attention scores (passed through leaky_relu/temp)
            # The downstream layer applies softmax.
            
            # Extract valid values
            edge_weight = topk_values[valid_mask]
            
            # NOTE: The downstream MacroPropagation layer calls softmax(alpha, dst, ...)
            # So passing raw scores is correct.
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
        
        # Attention parameters for stock-stock
        self.att_stock = nn.Parameter(torch.randn(1, heads, 2 * out_dim))
        
        # Attention parameters for macro-stock
        self.att_macro = nn.Parameter(torch.randn(1, heads, 2 * out_dim))
        
        # Output projection
        self.out_proj = nn.Linear(heads * out_dim, out_dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_stock.weight)
        nn.init.xavier_uniform_(self.W_macro.weight)
        nn.init.xavier_uniform_(self.att_stock)
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
        h: torch.Tensor,  # [N_s, H, D]
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate messages from stock neighbors."""
        src, dst = edge_index
        
        # Compute attention
        h_i = h[dst]  # [E, H, D]
        h_j = h[src]  # [E, H, D]
        
        concat = torch.cat([h_i, h_j], dim=-1)  # [E, H, 2D]
        alpha = (concat * self.att_stock).sum(dim=-1)  # [E, H]
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        
        # Normalize via softmax over incoming edges
        alpha = softmax(alpha, dst, num_nodes=h.size(0))  # [E, H]
        # NaN-safe: fully inactive nodes have all-(-inf) inputs → NaN after softmax
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Incorporate precomputed edge weights if available
        if edge_weight is not None:
            alpha = alpha * edge_weight.unsqueeze(-1)  # [E, H]
        
        # Aggregate
        out = h_j * alpha.unsqueeze(-1)  # [E, H, D]
        # Use matching dtype for scatter_add_ (autocast may mix FP16/FP32)
        out = torch.zeros(h.shape, dtype=out.dtype, device=h.device).scatter_add_(0, dst.view(-1, 1, 1).expand_as(out), out)
        
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
    Macro-Aware Dynamic Graph Relation Contrastive Learning Model v1.4.
    
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
        
        # Dynamic Graph Learner
        self.graph_learner = DynamicGraphLearner(
            hidden_dim=hidden_dim,
            top_k=top_k
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
        sector_mask: Optional[torch.Tensor] = None, # [N_s, N_s] boolean mask
        active_mask: Optional[torch.Tensor] = None  # [N_s] boolean mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN backbone and Multi-Task head.
        
        Args:
            stock_features: [N_s, T, d_s] stock time-series
            macro_features: [N_m, T, d_m] macro time-series
            macro_stock_edges: [2, E_ms] macro→stock edge index
            sector_mask: [N_s, N_s] boolean mask for sector constraints
            
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
        
        # 2. Dynamic Graph Learning (Stock→Stock)
        # SAFEGUARD 2: Pass active_mask to sever inactive nodes via -inf masking
        stock_stock_edges, edge_weights = self.graph_learner(
            stock_h, 
            sector_mask=sector_mask,
            active_mask=active_mask,
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
        
        return direction_logits, magnitude_preds
