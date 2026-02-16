"""
Unit Tests for Macro-Aware DGRCL v1.3 (Multi-Task Learning)

Tests:
1. HeteroData graph construction
2. Temporal encoder output shapes
3. Dynamic graph learner sparsity
4. MacroPropagation aggregation
5. MultiTaskHead outputs and constraints
6. Full forward pass (MTL)
7. End-to-end training step
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from macro_dgrcl import (
    HeteroGraphBuilder,
    TemporalEncoder,
    DynamicGraphLearner,
    MacroPropagation,
    MultiTaskHead,
    MacroDGRCL
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def device():
    # Force CPU for unit tests to avoid GPU OOM on memory-constrained systems.
    # Architecture is device-agnostic; correctness is identical on CPU vs GPU.
    return torch.device('cpu')


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Reduced sizes to prevent OOM on memory-constrained GPUs.
    # Architecture is identical — same layer types, heads, forward logic.
    # Only input counts are smaller (N_s 50→10 eliminates O(N²) attention bottleneck).
    N_s = 10  # Number of stocks (reduced from 50)
    N_m = 4   # Number of macro factors
    T = 20    # Sequence length (reduced from 60)
    d_s = 8   # Stock feature dim (v1.3: 8 features)
    d_m = 4   # Macro feature dim
    H = 32    # Hidden dim (reduced from 64)
    
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
        
        assert 'stock' in data.node_types
        assert 'macro' in data.node_types
        assert data['stock'].num_nodes == sample_data['N_s']
        assert data['macro'].num_nodes == sample_data['N_m']
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
        top_k = 10
        learner = DynamicGraphLearner(
            hidden_dim=sample_data['H'],
            top_k=top_k
        )
        
        edge_index, edge_weight = learner(
            sample_data['stock_embeddings'],
            return_weights=True
        )
        
        # top_k is clamped to N-1 inside forward()
        effective_k = min(top_k, sample_data['N_s'] - 1)
        expected_edges = sample_data['N_s'] * effective_k
        assert edge_index.shape == (2, expected_edges)
        assert edge_weight.shape == (expected_edges,)
    
    def test_top_k_sparsity(self, sample_data):
        """Test that each node RECEIVES from exactly top_k neighbors."""
        top_k = 5
        learner = DynamicGraphLearner(
            hidden_dim=sample_data['H'],
            top_k=top_k
        )
        
        edge_index, _ = learner(sample_data['stock_embeddings'])
        
        # Each destination node selects exactly k neighbors to receive from
        effective_k = min(top_k, sample_data['N_s'] - 1)
        dst = edge_index[1]
        unique, counts = torch.unique(dst, return_counts=True)
        assert (counts == effective_k).all()
    
    def test_dynamic_changes_with_embeddings(self, sample_data):
        """Test that different embeddings produce different graphs."""
        learner = DynamicGraphLearner(
            hidden_dim=sample_data['H'],
            top_k=10
        )
        
        edge_index1, _ = learner(sample_data['stock_embeddings'])
        edge_index2, _ = learner(torch.randn_like(sample_data['stock_embeddings']))
        
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
        
        assert not torch.allclose(out1, out2)


# =============================================================================
# TEST: MULTI-TASK HEAD
# =============================================================================

class TestMultiTaskHead:
    
    def test_output_shapes(self, sample_data):
        """Test both heads return [N, 1] tensors."""
        head = MultiTaskHead(
            hidden_dim=sample_data['H'],
            dropout=0.3
        )
        
        dir_logits, mag_preds = head(sample_data['stock_embeddings'])
        
        assert dir_logits.shape == (sample_data['N_s'], 1)
        assert mag_preds.shape == (sample_data['N_s'], 1)
    
    def test_magnitude_non_negative(self, sample_data):
        """Test that mag_head output is always non-negative (Softplus guarantee)."""
        head = MultiTaskHead(
            hidden_dim=sample_data['H'],
            dropout=0.0  # No dropout for deterministic test
        )
        
        # Test with various inputs including extreme values
        for _ in range(10):
            x = torch.randn(sample_data['N_s'], sample_data['H']) * 5  # Large magnitude inputs
            _, mag_preds = head(x)
            assert (mag_preds >= 0).all(), \
                f"Magnitude predictions should be non-negative, got min={mag_preds.min().item()}"
    
    def test_direction_logits_unbounded(self, sample_data):
        """Test that dir_head logits can take any real value."""
        head = MultiTaskHead(
            hidden_dim=sample_data['H'],
            dropout=0.0
        )
        
        # Collect logits from multiple random inputs with large magnitude
        all_logits = []
        for _ in range(50):
            x = torch.randn(sample_data['N_s'], sample_data['H']) * 10
            dir_logits, _ = head(x)
            all_logits.append(dir_logits)
        
        all_logits = torch.cat(all_logits)
        # Should have both positive and negative values
        assert all_logits.min() < 0, "Direction logits should include negative values"
        assert all_logits.max() > 0, "Direction logits should include positive values"
    
    def test_gradient_flow(self, sample_data):
        """Test gradients flow through both heads."""
        head = MultiTaskHead(
            hidden_dim=sample_data['H'],
            dropout=0.0
        )
        
        x = sample_data['stock_embeddings'].clone().requires_grad_(True)
        dir_logits, mag_preds = head(x)
        
        # Backprop through direction head
        loss_dir = dir_logits.sum()
        loss_dir.backward(retain_graph=True)
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        x.grad.zero_()
        
        # Backprop through magnitude head
        loss_mag = mag_preds.sum()
        loss_mag.backward()
        assert not torch.isnan(x.grad).any()
    
    def test_shared_block_connects_heads(self, sample_data):
        """Test that both heads share the initial linear block."""
        head = MultiTaskHead(
            hidden_dim=sample_data['H'],
            dropout=0.0
        )
        
        # Check that shared block parameters affect both outputs
        x = sample_data['stock_embeddings']
        dir_logits_1, mag_preds_1 = head(x)
        
        # Perturb shared block weights
        with torch.no_grad():
            head.shared[0].weight.add_(torch.randn_like(head.shared[0].weight) * 0.5)
        
        dir_logits_2, mag_preds_2 = head(x)
        
        # Both outputs should change
        assert not torch.allclose(dir_logits_1, dir_logits_2)
        assert not torch.allclose(mag_preds_1, mag_preds_2)


# =============================================================================
# TEST: FULL MODEL
# =============================================================================

class TestMacroDGRCL:
    
    def test_forward_pass(self, sample_data, device):
        """Test full forward pass returns (dir_logits, mag_preds)."""
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        
        dir_logits, mag_preds = model(stock_feat, macro_feat)
        
        assert dir_logits.shape == (sample_data['N_s'], 1)
        assert mag_preds.shape == (sample_data['N_s'], 1)
    
    def test_magnitude_always_positive(self, sample_data, device):
        """Test model magnitude output is always non-negative."""
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        
        _, mag_preds = model(stock_feat, macro_feat)
        
        assert (mag_preds >= 0).all(), "Magnitude predictions must be non-negative"
    
    def test_no_nan_in_outputs(self, sample_data, device):
        """Test no NaN values in model outputs."""
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        
        dir_logits, mag_preds = model(stock_feat, macro_feat)
        
        assert not torch.isnan(dir_logits).any(), "NaN in direction logits"
        assert not torch.isnan(mag_preds).any(), "NaN in magnitude predictions"
    
    def test_gradient_flow_full_model(self, sample_data, device):
        """Test gradients flow through the full model for both tasks."""
        from train import compute_pairwise_ranking_loss, compute_log_scaled_mag_target
        
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        returns = sample_data['returns'].to(device)
        
        dir_logits, mag_preds = model(stock_feat, macro_feat)
        
        # Compute pairwise ranking loss (direction) + SmoothL1 (magnitude)
        scores = dir_logits.squeeze(-1)
        loss_dir, _ = compute_pairwise_ranking_loss(scores, returns)
        
        mag_target = compute_log_scaled_mag_target(returns)
        loss_mag = nn.SmoothL1Loss()(mag_preds.squeeze(-1), mag_target)
        total_loss = loss_dir + loss_mag
        
        total_loss.backward()
        
        # Check gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


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
        
        mse_loss_fn = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        returns = sample_data['returns'].to(device)
        
        metrics = train_step(
            model=model,
            stock_features=stock_feat,
            macro_features=macro_feat,
            returns=returns,
            optimizer=optimizer,
            mse_loss_fn=mse_loss_fn,
            mag_weight=1.0
        )
        
        assert not torch.isnan(torch.tensor(metrics['loss']))
        assert metrics['loss'] >= 0
        assert 0 <= metrics['rank_accuracy'] <= 1
        assert metrics['mag_mae'] >= 0
    
    def test_training_step_returns_all_metrics(self, sample_data, device):
        """Test that train_step returns all expected metric keys."""
        from train import train_step
        
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        mse_loss_fn = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        returns = sample_data['returns'].to(device)
        
        metrics = train_step(
            model=model,
            stock_features=stock_feat,
            macro_features=macro_feat,
            returns=returns,
            optimizer=optimizer,
            mse_loss_fn=mse_loss_fn,
        )
        
        expected_keys = {'loss', 'loss_dir', 'loss_mag', 'rank_accuracy', 'mag_mae', 'grad_norm'}
        assert set(metrics.keys()) == expected_keys
    
    def test_evaluate_returns_metrics(self, sample_data, device):
        """Test that evaluate returns pairwise ranking + magnitude metrics."""
        from train import evaluate
        
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        mse_loss_fn = nn.SmoothL1Loss()
        
        # Create a small validation set
        val_data = [
            (sample_data['stock_features'], sample_data['macro_features'], sample_data['returns'])
        ]
        
        val_metrics = evaluate(
            model=model,
            data_loader=val_data,
            mse_loss_fn=mse_loss_fn,
            device=device,
            mag_weight=1.0
        )
        
        expected_keys = {'loss', 'loss_dir', 'loss_mag', 'rank_accuracy', 'mag_mae'}
        assert set(val_metrics.keys()) == expected_keys
        assert val_metrics['loss'] >= 0
    
    def test_early_stopping(self):
        """Test EarlyStopping behavior."""
        from train import EarlyStopping
        
        model = nn.Linear(10, 1)
        es = EarlyStopping(patience=3, min_delta=0.01)
        
        # Simulate improving losses
        assert not es(1.0, model)
        assert not es(0.9, model)
        assert not es(0.8, model)
        
        # Simulate stagnation
        assert not es(0.8, model)  # counter=1
        assert not es(0.8, model)  # counter=2
        assert es(0.8, model)       # counter=3 >= patience
        
        assert es.early_stop
    
    def test_mag_weight_affects_loss(self, sample_data, device):
        """Test that changing mag_weight changes the total loss."""
        from train import train_step
        
        # Create two identical models
        torch.manual_seed(42)
        model1 = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        torch.manual_seed(42)
        model2 = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        mse_loss_fn = nn.SmoothL1Loss()
        opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        returns = sample_data['returns'].to(device)
        
        m1 = train_step(model1, stock_feat, macro_feat, returns, opt1,
                        mse_loss_fn, mag_weight=0.1)
        m2 = train_step(model2, stock_feat, macro_feat, returns, opt2,
                        mse_loss_fn, mag_weight=10.0)
        
        # Direction loss should be very similar (not identical due to GPU non-determinism),
        # but total loss should clearly differ due to mag_weight
        assert abs(m1['loss_dir'] - m2['loss_dir']) < 0.05
        assert m1['loss'] != m2['loss']


# =============================================================================
# TEST: MC DROPOUT INFERENCE
# =============================================================================

class TestMCDropout:
    
    def test_mc_dropout_variance(self, sample_data, device):
        """MC Dropout must produce non-zero variance (dropout is active)."""
        from train import mc_dropout_inference
        
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H'],
            dropout=0.5,
            head_dropout=0.5
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        
        results = mc_dropout_inference(
            model, stock_feat, macro_feat, n_samples=20
        )
        
        # With dropout=0.5 and 20 samples, std should be non-zero
        assert results['dir_score_std'].mean().item() > 0, \
            "MC Dropout should produce non-zero variance in direction scores"
        assert results['mag_std'].mean().item() > 0, \
            "MC Dropout should produce non-zero variance in magnitude predictions"
    
    def test_mc_dropout_output_shapes(self, sample_data, device):
        """MC Dropout returns correct shapes for all outputs."""
        from train import mc_dropout_inference
        
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        
        results = mc_dropout_inference(
            model, stock_feat, macro_feat, n_samples=5
        )
        
        N_s = sample_data['N_s']
        assert results['dir_score_mean'].shape == (N_s,)
        assert results['dir_score_std'].shape == (N_s,)
        assert results['mag_mean'].shape == (N_s,)
        assert results['mag_std'].shape == (N_s,)
        assert results['confidence'].shape == (N_s,)
        assert results['rank_stability'].shape == ()
        
        # Confidence should be in (0, 1]
        assert (results['confidence'] > 0).all()
        assert (results['confidence'] <= 1).all()
        
        # Rank stability should be in [0, 1]
        assert 0.0 <= results['rank_stability'].item() <= 1.0


# =============================================================================
# TEST: ROLLING Z-SCORE NO LOOK-AHEAD
# =============================================================================

class TestRollingZScore:
    
    def test_no_lookahead(self):
        """Modifying future values must not change past normalized values."""
        import pandas as pd
        from data_ingest import rolling_zscore_normalize
        
        # Create a deterministic series
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df1 = pd.DataFrame({'A': range(100)}, index=dates, dtype=float)
        df2 = df1.copy()
        
        # Modify future values (from index 50 onward)
        df2.loc[df2.index[50]:, 'A'] = 9999.0
        
        norm1 = rolling_zscore_normalize(df1, window=10)
        norm2 = rolling_zscore_normalize(df2, window=10)
        
        # All normalized values BEFORE the modification point should be identical
        # (checking up to index 49, since shift(1) means index 50 uses data up to 49)
        pd.testing.assert_frame_equal(
            norm1.iloc[1:50],  # skip first row (NaN from shift)
            norm2.iloc[1:50],
            check_names=True,
            obj="No look-ahead: past values should be identical"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
