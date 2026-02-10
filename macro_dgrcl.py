"""
Macro-Aware DGRCL v1.3 - Multi-Task Learning Architecture

A Heterogeneous Graph Neural Network for market-neutral trading with:
- Explicit Macro factor nodes (not feature vectors)
- Dynamic Stock→Stock edge learning via attention
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
        - ('stock', 'relates', 'stock'): Dynamic via attention
    """
    
    def __init__(
        self,
        macro_stock_fixed_edges: Optional[torch.Tensor] = None,
        sector_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Args:
            macro_stock_fixed_edges: [2, E] tensor of fixed macro→stock edges
            sector_mapping: Dict mapping stock_idx → sector_idx for fixed edges
        """
        self.macro_stock_fixed_edges = macro_stock_fixed_edges
        self.sector_mapping = sector_mapping or {}
    
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
        
        return torch.cat(edges_list, dim=1) if edges_list else None


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
    
    At each timestep, computes pairwise attention scores and keeps top-k
    neighbors per node to maintain sparsity.
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
        nn.init.normal_(self.a, std=0.01)
    
    def forward(
        self,
        embeddings: torch.Tensor,  # [N, H]
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute dynamic adjacency via attention.
        
        Args:
            embeddings: [N, H] node embeddings
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
        
        # Top-k selection per row (each node keeps top-k neighbors)
        k = min(self.top_k, N - 1)
        topk_values, topk_indices = torch.topk(attention, k=k, dim=1)  # [N, k]
        
        # Build sparse edge_index
        # CRITICAL: In PyG MessagePassing, info flows src → dst
        # Node i computed which neighbors j are relevant, so i should RECEIVE from j
        # Therefore edges are: src=j (neighbors), dst=i (computing node)
        row_indices = torch.arange(N, device=embeddings.device).unsqueeze(1).repeat(1, k)  # [N, k]  (node i)
        col_indices = topk_indices  # [N, k]  (neighbors j that node i selected)
        
        # Edges: j → i (so node i receives messages from its selected neighbors j)
        src = col_indices.flatten()  # neighbors j
        dst = row_indices.flatten()  # node i that selected them
        
        edge_index = torch.stack([src, dst], dim=0)  # [2, N*k]
        
        if return_weights:
            # Normalize weights via softmax over selected neighbors
            edge_weight = F.softmax(topk_values, dim=1).flatten()  # [N*k]
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
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Incorporate precomputed edge weights if available
        if edge_weight is not None:
            alpha = alpha * edge_weight.unsqueeze(-1)  # [E, H]
        
        # Aggregate
        out = h_j * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros_like(h).scatter_add_(0, dst.view(-1, 1, 1).expand_as(out), out)
        
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
        out = torch.zeros_like(stock_h).scatter_add_(
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
    Macro-Aware Dynamic Graph Relation Contrastive Learning Model v1.3.
    
    Multi-Task Learning Architecture:
        1. TemporalEncoder: LSTM projects time-series → embeddings
        2. DynamicGraphLearner: Computes dynamic Stock→Stock adjacency
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN backbone and Multi-Task head.
        
        Args:
            stock_features: [N_s, T, d_s] stock time-series
            macro_features: [N_m, T, d_m] macro time-series
            macro_stock_edges: [2, E_ms] macro→stock edge index
            
        Returns:
            direction_logits: [N_s, 1] unbounded logits for P(return > median)
            magnitude_preds:  [N_s, 1] predicted |R_t| (non-negative)
        """
        # 1. Temporal Encoding
        stock_h = self.stock_encoder(stock_features)  # [N_s, H]
        macro_h = self.macro_encoder(macro_features)  # [N_m, H]
        
        # 2. Dynamic Graph Learning (Stock→Stock)
        stock_stock_edges, edge_weights = self.graph_learner(
            stock_h, return_weights=True
        )
        
        # 3. Build Macro→Stock edges if not provided
        if macro_stock_edges is None:
            # Default: all macros connect to all stocks
            N_m, N_s = macro_features.size(0), stock_features.size(0)
            src = torch.arange(N_m, device=stock_features.device).repeat_interleave(N_s)
            dst = torch.arange(N_s, device=stock_features.device).repeat(N_m)
            macro_stock_edges = torch.stack([src, dst], dim=0)
        
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
        
        # 5. Multi-Task Head
        direction_logits, magnitude_preds = self.output_head(h)
        
        return direction_logits, magnitude_preds
