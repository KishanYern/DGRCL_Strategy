"""
Unit Tests for Macro-Aware DGRCL v1.1

Tests:
1. HeteroData graph construction
2. Temporal encoder output shapes
3. Dynamic graph learner sparsity
4. MacroPropagation aggregation
5. MC Dropout variance
6. Full forward pass
7. Loss computation
"""

import pytest
import torch
import torch.nn.functional as F

from macro_dgrcl import (
    HeteroGraphBuilder,
    TemporalEncoder,
    DynamicGraphLearner,
    MacroPropagation,
    MCDropoutHead,
    MacroDGRCL
)
from losses import pairwise_ranking_loss, info_nce_loss, dgrcl_loss, DGRCLLoss


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    N_s = 50  # Number of stocks
    N_m = 4   # Number of macro factors
    T = 60    # Sequence length
    d_s = 7   # Stock feature dim
    d_m = 4   # Macro feature dim
    H = 64    # Hidden dim
    
    return {
        'stock_features': torch.randn(N_s, T, d_s),
        'macro_features': torch.randn(N_m, T, d_m),
        'stock_embeddings': torch.randn(N_s, H),
        'macro_embeddings': torch.randn(N_m, H),
        'returns': torch.randn(N_s) * 0.02,
        'N_s': N_s,
        'N_m': N_m,
        'T': T,
        'd_s': d_s,
        'd_m': d_m,
        'H': H
    }


# =============================================================================
# TEST: HETERO GRAPH CONSTRUCTION
# =============================================================================

class TestHeteroGraphBuilder:
    
    def test_basic_construction(self, sample_data):
        """Test basic HeteroData construction."""
        builder = HeteroGraphBuilder()
        
        data = builder.build(
            stock_features=sample_data['stock_features'],
            macro_features=sample_data['macro_features']
        )
        
        # Check node types exist
        assert 'stock' in data.node_types
        assert 'macro' in data.node_types
        
        # Check node counts
        assert data['stock'].num_nodes == sample_data['N_s']
        assert data['macro'].num_nodes == sample_data['N_m']
        
        # Check feature shapes
        assert data['stock'].x.shape == sample_data['stock_features'].shape
        assert data['macro'].x.shape == sample_data['macro_features'].shape
    
    def test_edge_types_present(self, sample_data):
        """Test that default edges are created."""
        builder = HeteroGraphBuilder()
        
        stock_stock_edges = torch.randint(0, sample_data['N_s'], (2, 100))
        
        data = builder.build(
            stock_features=sample_data['stock_features'],
            macro_features=sample_data['macro_features'],
            stock_stock_edges=stock_stock_edges
        )
        
        # Check edge types
        assert ('stock', 'relates', 'stock') in data.edge_types
        assert ('macro', 'influences', 'stock') in data.edge_types


# =============================================================================
# TEST: TEMPORAL ENCODER
# =============================================================================

class TestTemporalEncoder:
    
    def test_output_shape(self, sample_data):
        """Test LSTM output dimensions."""
        encoder = TemporalEncoder(
            input_dim=sample_data['d_s'],
            hidden_dim=sample_data['H'],
            num_layers=2
        )
        
        output = encoder(sample_data['stock_features'])
        
        assert output.shape == (sample_data['N_s'], sample_data['H'])
    
    def test_bidirectional_doubles_dim(self, sample_data):
        """Test bidirectional LSTM doubles output dimension."""
        encoder = TemporalEncoder(
            input_dim=sample_data['d_s'],
            hidden_dim=sample_data['H'],
            num_layers=2,
            bidirectional=True
        )
        
        output = encoder(sample_data['stock_features'])
        
        assert output.shape == (sample_data['N_s'], 2 * sample_data['H'])
    
    def test_gradient_flow(self, sample_data):
        """Test gradients flow through encoder."""
        encoder = TemporalEncoder(
            input_dim=sample_data['d_s'],
            hidden_dim=sample_data['H']
        )
        
        x = sample_data['stock_features'].clone().requires_grad_(True)
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# TEST: DYNAMIC GRAPH LEARNER
# =============================================================================

class TestDynamicGraphLearner:
    
    def test_output_shapes(self, sample_data):
        """Test edge_index shape."""
        learner = DynamicGraphLearner(
            hidden_dim=sample_data['H'],
            top_k=10
        )
        
        edge_index, edge_weight = learner(
            sample_data['stock_embeddings'],
            return_weights=True
        )
        
        # edge_index should be [2, N * top_k]
        expected_edges = sample_data['N_s'] * 10
        assert edge_index.shape == (2, expected_edges)
        assert edge_weight.shape == (expected_edges,)
    
    def test_top_k_sparsity(self, sample_data):
        """Test that each node has exactly top_k neighbors."""
        top_k = 5
        learner = DynamicGraphLearner(
            hidden_dim=sample_data['H'],
            top_k=top_k
        )
        
        edge_index, _ = learner(sample_data['stock_embeddings'])
        
        # Count edges per source node
        src = edge_index[0]
        unique, counts = torch.unique(src, return_counts=True)
        
        assert (counts == top_k).all()
    
    def test_dynamic_changes_with_embeddings(self, sample_data):
        """Test that different embeddings produce different graphs."""
        learner = DynamicGraphLearner(
            hidden_dim=sample_data['H'],
            top_k=10
        )
        
        edge_index1, _ = learner(sample_data['stock_embeddings'])
        edge_index2, _ = learner(torch.randn_like(sample_data['stock_embeddings']))
        
        # Edges should differ
        assert not torch.equal(edge_index1, edge_index2)


# =============================================================================
# TEST: MACRO PROPAGATION
# =============================================================================

class TestMacroPropagation:
    
    def test_output_shape(self, sample_data):
        """Test MacroPropagation output shape."""
        mp = MacroPropagation(
            stock_dim=sample_data['H'],
            macro_dim=sample_data['H'],
            out_dim=sample_data['H'],
            heads=4
        )
        
        # Create edges
        stock_stock_edges = torch.randint(0, sample_data['N_s'], (2, 200))
        macro_stock_edges = torch.stack([
            torch.arange(sample_data['N_m']).repeat_interleave(sample_data['N_s']),
            torch.arange(sample_data['N_s']).repeat(sample_data['N_m'])
        ])
        
        output = mp(
            stock_h=sample_data['stock_embeddings'],
            macro_h=sample_data['macro_embeddings'],
            stock_stock_edge_index=stock_stock_edges,
            macro_stock_edge_index=macro_stock_edges
        )
        
        assert output.shape == (sample_data['N_s'], sample_data['H'])
    
    def test_aggregation_uses_both_sources(self, sample_data):
        """Test that output changes with macro input."""
        mp = MacroPropagation(
            stock_dim=sample_data['H'],
            macro_dim=sample_data['H'],
            out_dim=sample_data['H'],
            heads=4
        )
        
        stock_stock_edges = torch.randint(0, sample_data['N_s'], (2, 200))
        macro_stock_edges = torch.stack([
            torch.arange(sample_data['N_m']).repeat_interleave(sample_data['N_s']),
            torch.arange(sample_data['N_s']).repeat(sample_data['N_m'])
        ])
        
        # Run with different macro embeddings
        out1 = mp(
            stock_h=sample_data['stock_embeddings'],
            macro_h=sample_data['macro_embeddings'],
            stock_stock_edge_index=stock_stock_edges,
            macro_stock_edge_index=macro_stock_edges
        )
        
        out2 = mp(
            stock_h=sample_data['stock_embeddings'],
            macro_h=torch.randn_like(sample_data['macro_embeddings']),
            stock_stock_edge_index=stock_stock_edges,
            macro_stock_edge_index=macro_stock_edges
        )
        
        # Outputs should differ
        assert not torch.allclose(out1, out2)


# =============================================================================
# TEST: MC DROPOUT HEAD
# =============================================================================

class TestMCDropoutHead:
    
    def test_basic_output_shape(self, sample_data):
        """Test output shape."""
        head = MCDropoutHead(
            input_dim=sample_data['H'],
            hidden_dim=64,
            output_dim=1
        )
        
        output = head(sample_data['stock_embeddings'])
        
        assert output.shape == (sample_data['N_s'], 1)
    
    def test_force_dropout_produces_variance(self, sample_data):
        """Test that force_dropout=True produces different outputs."""
        head = MCDropoutHead(
            input_dim=sample_data['H'],
            hidden_dim=64,
            output_dim=1,
            dropout=0.5  # High dropout for visible variance
        )
        head.eval()  # Switch to eval mode
        
        # Multiple forward passes with force_dropout
        outputs = [
            head(sample_data['stock_embeddings'], force_dropout=True)
            for _ in range(10)
        ]
        
        # Check outputs differ
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "MC Dropout should produce different outputs"
    
    def test_mc_forward_returns_mu_sigma(self, sample_data):
        """Test mc_forward returns mean and variance."""
        head = MCDropoutHead(
            input_dim=sample_data['H'],
            hidden_dim=64,
            output_dim=1,
            dropout=0.3
        )
        
        mu, sigma2 = head.mc_forward(sample_data['stock_embeddings'], n_samples=50)
        
        assert mu.shape == (sample_data['N_s'], 1)
        assert sigma2.shape == (sample_data['N_s'], 1)
        
        # Variance should be positive
        assert (sigma2 >= 0).all()
        
        # With dropout, variance should be non-zero
        assert sigma2.sum() > 0


# =============================================================================
# TEST: FULL MODEL
# =============================================================================

class TestMacroDGRCL:
    
    def test_forward_pass(self, sample_data, device):
        """Test full forward pass."""
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        
        scores, embeddings = model(stock_feat, macro_feat)
        
        assert scores.shape == (sample_data['N_s'], 1)
        assert embeddings.shape == (sample_data['N_s'], sample_data['H'])
    
    def test_forward_with_force_dropout(self, sample_data, device):
        """Test forward with MC Dropout enabled."""
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H'],
            mc_dropout=0.5
        ).to(device)
        model.eval()
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        
        # Multiple passes with force_dropout
        outputs = [
            model(stock_feat, macro_feat, force_dropout=True)[0]
            for _ in range(5)
        ]
        
        # Outputs should differ
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same
    
    def test_predict_with_uncertainty(self, sample_data, device):
        """Test uncertainty estimation."""
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        
        mu, sigma2, embeddings = model.predict_with_uncertainty(
            stock_feat, macro_feat, n_samples=30
        )
        
        assert mu.shape == (sample_data['N_s'], 1)
        assert sigma2.shape == (sample_data['N_s'], 1)
        assert (sigma2 >= 0).all()


# =============================================================================
# TEST: LOSS FUNCTIONS
# =============================================================================

class TestLossFunctions:
    
    def test_pairwise_ranking_loss_positive(self, sample_data):
        """Test ranking loss is positive."""
        scores = torch.randn(sample_data['N_s'], 1)
        returns = sample_data['returns']
        
        loss = pairwise_ranking_loss(scores, returns)
        
        assert loss >= 0
        assert not torch.isnan(loss)
    
    def test_ranking_loss_zero_for_perfect_ranking(self, sample_data):
        """Test loss approaches zero for perfect predictions."""
        # Create scores that perfectly match ranking
        returns = sample_data['returns']
        scores = returns.clone().unsqueeze(-1) * 100  # Large margin
        
        loss = pairwise_ranking_loss(scores, returns, margin=1.0)
        
        assert loss < 0.5  # Should be relatively small with perfect ranking
    
    def test_info_nce_loss_positive(self, sample_data):
        """Test InfoNCE loss is positive."""
        embeddings = sample_data['stock_embeddings']
        
        loss = info_nce_loss(embeddings)
        
        assert loss >= 0
        assert not torch.isnan(loss)
    
    def test_dgrcl_loss_gradient_flow(self, sample_data):
        """Test gradients flow through combined loss."""
        scores = torch.randn(sample_data['N_s'], 1, requires_grad=True)
        embeddings = torch.randn(sample_data['N_s'], sample_data['H'], requires_grad=True)
        returns = sample_data['returns']
        
        loss = dgrcl_loss(scores, embeddings, returns)
        loss.backward()
        
        assert scores.grad is not None
        assert embeddings.grad is not None
        assert not torch.isnan(scores.grad).any()
        assert not torch.isnan(embeddings.grad).any()
    
    def test_dgrcl_loss_module(self, sample_data):
        """Test DGRCLLoss module."""
        criterion = DGRCLLoss(
            ranking_margin=1.0,
            nce_temperature=0.07,
            nce_weight=0.1
        )
        
        scores = torch.randn(sample_data['N_s'], 1)
        embeddings = sample_data['stock_embeddings']
        returns = sample_data['returns']
        
        loss = criterion(scores, embeddings, returns)
        
        assert loss >= 0
        assert not torch.isnan(loss)


# =============================================================================
# TEST: END-TO-END
# =============================================================================

class TestEndToEnd:
    
    def test_training_step_no_nan(self, sample_data, device):
        """Test one training step produces no NaN."""
        from train import train_step
        
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        criterion = DGRCLLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        returns = sample_data['returns'].to(device)
        
        metrics = train_step(
            model=model,
            stock_features=stock_feat,
            macro_features=macro_feat,
            returns=returns,
            criterion=criterion,
            optimizer=optimizer
        )
        
        assert not torch.isnan(torch.tensor(metrics['loss']))
        assert metrics['loss'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
