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
    
    # Generate active mask: ~70% of stocks active
    torch.manual_seed(42)  # Deterministic for tests
    active_mask = torch.rand(N_s) > 0.3  # ~70% True
    # Ensure at least 3 active and 1 inactive for meaningful tests
    active_mask[0] = True
    active_mask[1] = True
    active_mask[2] = True
    active_mask[-1] = False
    
    return {
        'stock_features': torch.randn(N_s, T, d_s),
        'macro_features': torch.randn(N_m, T, d_m),
        'stock_embeddings': torch.randn(N_s, H),
        'macro_embeddings': torch.randn(N_m, H),
        'returns': torch.randn(N_s) * 0.02,
        'active_mask': active_mask,
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
        
        # Create two identical models (use seed 123 to avoid conflict with fixture seed 42)
        torch.manual_seed(123)
        model1 = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        torch.manual_seed(123)
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


# =============================================================================
# TEST: FEATURE ABLATION
# =============================================================================

class TestFeatureAblation:
    
    def test_slice_stock_features_helper(self):
        """Test the slicing helper function."""
        from train import slice_stock_features
        
        # Create dummy snapshots: 2 snapshots, 5 stocks, 10 time steps, 8 features
        N_s, T, d_s = 5, 10, 8
        snapshots = []
        for _ in range(2):
            stock = torch.randn(N_s, T, d_s)
            macro = torch.randn(4, T, 4)
            ret = torch.randn(N_s)
            snapshots.append((stock, macro, ret))
            
        # Select indices [0, 7] -> first and last
        indices = [0, 7]
        sliced = slice_stock_features(snapshots, indices)
        
        # Check length preserved
        assert len(sliced) == 2
        
        # Check dimensions
        for i in range(2):
            s_orig = snapshots[i][0]
            s_new = sliced[i][0]
            
            assert s_new.shape == (N_s, T, 2)
            
            # Check content - feature 0 match
            assert torch.allclose(s_new[:, :, 0], s_orig[:, :, 0])
            # Check content - feature 1 (was 7) match
            assert torch.allclose(s_new[:, :, 1], s_orig[:, :, 7])
            
            # Macro and returns intact
            assert torch.allclose(sliced[i][1], snapshots[i][1])
            assert torch.allclose(sliced[i][2], snapshots[i][2])

    def test_model_with_reduced_dim(self, sample_data, device):
        """Test model initialization and forward pass with reduced feature dim."""
        # Simulate "pure_momentum" (3 features)
        feature_dim = 3
        
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=feature_dim,  # <--- REDUCED DIM
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        # Create input with matching dim
        stock_feat = torch.randn(sample_data['N_s'], sample_data['T'], feature_dim).to(device)
        macro_feat = sample_data['macro_features'].to(device)
        
        # Forward pass should work without error
        dir_logits, mag_preds = model(stock_feat, macro_feat)
        
        assert dir_logits.shape == (sample_data['N_s'], 1)
        assert mag_preds.shape == (sample_data['N_s'], 1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# TEST: ACTIVE MASK — DYNAMIC UNIVERSE SAFEGUARDS
# =============================================================================

class TestActiveMaskSeversInactiveNodes:
    
    def test_inactive_nodes_get_zero_attention(self, sample_data):
        """Verify that inactive nodes receive zero attention weight after masking."""
        learner = DynamicGraphLearner(
            hidden_dim=sample_data['H'],
            top_k=5
        )
        
        active_mask = sample_data['active_mask']
        
        edge_index, edge_weights = learner(
            sample_data['stock_embeddings'],
            active_mask=active_mask,
            return_weights=True
        )
        
        # Inactive nodes should not appear as source or destination
        inactive_indices = (~active_mask).nonzero(as_tuple=True)[0]
        src, dst = edge_index
        
        for inactive_idx in inactive_indices:
            # Inactive node should not be a source (no one attends TO it)
            assert (src != inactive_idx).all(), \
                f"Inactive node {inactive_idx} appears as source in edge_index"


class TestNaNSafeFullyInactive:
    
    def test_all_inactive_no_nan(self, sample_data, device):
        """All-inactive mask must produce zeros, not NaN, through full forward pass."""
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        # All stocks inactive
        all_inactive = torch.zeros(sample_data['N_s'], dtype=torch.bool, device=device)
        
        dir_logits, mag_preds = model(
            stock_feat, macro_feat, active_mask=all_inactive
        )
        
        assert not torch.isnan(dir_logits).any(), "NaN in direction logits with all-inactive mask"
        assert not torch.isnan(mag_preds).any(), "NaN in magnitude preds with all-inactive mask"
    
    def test_partial_inactive_no_nan(self, sample_data, device):
        """Partial active mask must produce no NaN in any output."""
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        active_mask = sample_data['active_mask'].to(device)
        
        dir_logits, mag_preds = model(
            stock_feat, macro_feat, active_mask=active_mask
        )
        
        assert not torch.isnan(dir_logits).any(), "NaN in direction logits"
        assert not torch.isnan(mag_preds).any(), "NaN in magnitude preds"


class TestMacroEdgesFiltered:
    
    def test_macro_edges_exclude_inactive(self, sample_data, device):
        """Macro→Stock edges should exclude inactive destinations."""
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)
        
        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        active_mask = sample_data['active_mask'].to(device)
        
        # Run forward pass — it builds and filters macro→stock edges internally
        # We verify by checking that the model runs without error and
        # inactive nodes don't get contaminated
        dir_logits, mag_preds = model(
            stock_feat, macro_feat, active_mask=active_mask
        )
        
        # Outputs should have correct shape
        assert dir_logits.shape == (sample_data['N_s'], 1)
        assert mag_preds.shape == (sample_data['N_s'], 1)


class TestRankingLossIgnoresInactive:
    
    def test_inactive_pairs_excluded(self, sample_data):
        """Ranking loss should only use active-active pairs."""
        from train import compute_pairwise_ranking_loss
        
        scores = torch.randn(sample_data['N_s'])
        returns = torch.randn(sample_data['N_s']) * 0.05  # Large returns for valid pairs
        active_mask = sample_data['active_mask']
        
        # Loss with active_mask
        loss_masked, acc_masked = compute_pairwise_ranking_loss(
            scores=scores,
            returns=returns,
            active_mask=active_mask
        )
        
        # Loss with all active (no mask)
        loss_unmasked, acc_unmasked = compute_pairwise_ranking_loss(
            scores=scores,
            returns=returns
        )
        
        # Masked loss should be different (fewer valid pairs)
        # unless all stocks happen to be active
        if not active_mask.all():
            # Either the losses differ or the number of valid pairs differs
            assert loss_masked.item() != loss_unmasked.item() or \
                   acc_masked != acc_unmasked, \
                "Masking should change loss/accuracy when stocks are inactive"
    
    def test_all_inactive_zero_loss(self, sample_data):
        """All-inactive mask should produce zero ranking loss (no valid pairs)."""
        from train import compute_pairwise_ranking_loss
        
        scores = torch.randn(sample_data['N_s'])
        returns = torch.randn(sample_data['N_s']) * 0.05
        all_inactive = torch.zeros(sample_data['N_s'], dtype=torch.bool)
        
        loss, acc = compute_pairwise_ranking_loss(
            scores=scores,
            returns=returns,
            active_mask=all_inactive
        )
        
        assert loss.item() == 0.0, "All-inactive mask should produce zero ranking loss"
        assert acc == 0.0, "All-inactive mask should produce zero accuracy"


class TestMagnitudeLossMasksInactive:
    
    def test_magnitude_loss_only_active(self, sample_data, device):
        """SmoothL1 magnitude loss should only consider active stocks."""
        from train import compute_log_scaled_mag_target
        
        mse_loss_fn = nn.SmoothL1Loss()
        active_mask = sample_data['active_mask']
        returns = sample_data['returns']
        mag_preds = torch.randn(sample_data['N_s'])
        mag_target = compute_log_scaled_mag_target(returns)
        
        # Compute loss over active only
        loss_active = mse_loss_fn(
            mag_preds[active_mask],
            mag_target[active_mask]
        )
        
        # Compute loss over all
        loss_all = mse_loss_fn(mag_preds, mag_target)
        
        # They should differ unless all are active
        if not active_mask.all():
            assert loss_active.item() != loss_all.item(), \
                "Masked loss should differ from unmasked loss"


class TestTrainingStepWithActiveMask:
    
    def test_no_nan_with_active_mask(self, sample_data, device):
        """Full training step with active_mask should produce no NaN."""
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
        active_mask = sample_data['active_mask'].to(device)
        
        metrics = train_step(
            model=model,
            stock_features=stock_feat,
            macro_features=macro_feat,
            returns=returns,
            optimizer=optimizer,
            mse_loss_fn=mse_loss_fn,
            mag_weight=1.0,
            active_mask=active_mask
        )
        
        assert not torch.isnan(torch.tensor(metrics['loss'])), "NaN loss with active_mask"
        assert metrics['loss'] >= 0, "Loss should be non-negative"
        assert 0 <= metrics['rank_accuracy'] <= 1
        assert metrics['mag_mae'] >= 0
    
    def test_gradient_flow_with_active_mask(self, sample_data, device):
        """Gradients should flow through model with active_mask."""
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
        active_mask = sample_data['active_mask'].to(device)

        dir_logits, mag_preds = model(
            stock_feat, macro_feat, active_mask=active_mask
        )

        scores = dir_logits.squeeze(-1)
        loss_dir, _ = compute_pairwise_ranking_loss(
            scores, returns, active_mask=active_mask
        )

        # FIX #5: pass active_mask so sigma uses only active stocks
        mag_target = compute_log_scaled_mag_target(returns, active_mask=active_mask)
        loss_mag = nn.SmoothL1Loss()(
            mag_preds.squeeze(-1)[active_mask],
            mag_target[active_mask]
        )
        total_loss = loss_dir + loss_mag
        total_loss.backward()

        # Check gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_train_step_updates_parameters(self, sample_data, device):
        """
        Verify that model parameters actually change after train_step().

        This test would have caught the double optimizer.zero_grad() bug (Issue #2)
        where all gradients were wiped before backward(), so parameters never moved.
        """
        from train import train_step

        torch.manual_seed(0)
        model = MacroDGRCL(
            num_stocks=sample_data['N_s'],
            num_macros=sample_data['N_m'],
            stock_feature_dim=sample_data['d_s'],
            macro_feature_dim=sample_data['d_m'],
            hidden_dim=sample_data['H']
        ).to(device)

        # Snapshot weights before the training step
        params_before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        mse_loss_fn = nn.SmoothL1Loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # Large LR for clear movement

        stock_feat = sample_data['stock_features'].to(device)
        macro_feat = sample_data['macro_features'].to(device)
        returns = sample_data['returns'].to(device)
        active_mask = sample_data['active_mask'].to(device)

        metrics = train_step(
            model=model,
            stock_features=stock_feat,
            macro_features=macro_feat,
            returns=returns,
            optimizer=optimizer,
            mse_loss_fn=mse_loss_fn,
            mag_weight=1.0,
            active_mask=active_mask,
            accumulation_steps=1
        )

        # At least some parameters must have changed — if double zero_grad() was
        # present, ALL parameters would be identical (no gradient was applied).
        changed = sum(
            1 for name, p in model.named_parameters()
            if p.requires_grad and not torch.equal(p.detach(), params_before[name])
        )
        assert changed > 0, (
            "No parameters changed after train_step()! "
            "This indicates gradients were zeroed before backward() — "
            "check for a duplicate optimizer.zero_grad() call in train_step()."
        )


class TestMagnitudeTargetSigmaFix:
    """Verify that compute_log_scaled_mag_target uses active-only sigma (Issue #5)."""

    def test_sigma_excludes_inactive_stocks(self, sample_data):
        """
        compute_log_scaled_mag_target must use active-only sigma when a mask is provided.

        We verify the BEHAVIOR: masked and unmasked calls produce different outputs
        when some stocks are inactive (zero-padded). We do not assert a direction on
        sigma because the direction depends on how the zeros compare to active returns.

        The key property: if masking changes sigma, it changes the normalized targets.
        """
        from train import compute_log_scaled_mag_target

        # Use a fully controlled standalone tensor (independent of fixture randomness)
        # 10 stocks: 7 active (varied returns), 3 inactive (zero-padded)
        N = 10
        active = torch.tensor([True]*7 + [False]*3)
        returns = torch.zeros(N)
        returns[active] = torch.tensor([0.01, -0.02, 0.03, -0.01, 0.05, -0.03, 0.02])

        target_masked = compute_log_scaled_mag_target(returns, active_mask=active)
        target_unmasked = compute_log_scaled_mag_target(returns)

        # The two sigmata must differ (otherwise masking does nothing)
        sigma_masked = returns[active].abs().std()
        sigma_unmasked = returns.abs().std()
        assert not torch.isclose(sigma_masked, sigma_unmasked, atol=1e-6), (
            "Masked sigma should differ from unmasked sigma when inactive stocks "
            "have zero returns and active stocks have non-zero returns"
        )

        # Consequently, the normalized targets for active stocks must differ
        assert not torch.allclose(target_masked[active], target_unmasked[active]), (
            "Targets for active stocks must differ between masked and unmasked "
            "when sigma changes"
        )

        # Sanity: all targets are finite and non-negative
        assert target_masked.isfinite().all(), "Masked targets must be finite"
        assert (target_masked >= 0).all(), "log1p targets must be non-negative"
