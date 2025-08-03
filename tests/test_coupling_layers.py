"""
Tests for coupling layer implementations.

Tests reversible coupling functions to ensure mathematical correctness
and gradient computation accuracy.
"""

import torch
import torch.nn as nn
import pytest
from revnet_zero.layers.coupling_layers import (
    AdditiveCoupling, 
    AffineCoupling, 
    LearnedCoupling
)


class TestAdditiveCoupling:
    """Test additive coupling implementation."""
    
    def test_forward_inverse_consistency(self):
        """Test that forward and inverse operations are consistent."""
        d_model = 512
        batch_size = 4
        seq_len = 128
        
        coupling = AdditiveCoupling(d_model)
        
        # Create test input
        x = torch.randn(batch_size, seq_len, d_model)
        x1, x2 = coupling.split_input(x)
        
        # Forward and inverse
        y1, y2 = coupling.forward(x1, x2)
        x1_reconstructed, x2_reconstructed = coupling.inverse(y1, y2)
        
        # Check reconstruction accuracy
        torch.testing.assert_close(x1, x1_reconstructed, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(x2, x2_reconstructed, atol=1e-6, rtol=1e-6)
    
    def test_gradient_computation(self):
        """Test gradient computation through coupling."""
        d_model = 256
        batch_size = 2
        seq_len = 64
        
        coupling = AdditiveCoupling(d_model)
        
        # Create test input with gradients
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        x1, x2 = coupling.split_input(x)
        
        # Forward pass
        y1, y2 = coupling.forward(x1, x2)
        
        # Create dummy loss
        loss = (y1.sum() + y2.sum())
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
    
    def test_different_dimensions(self):
        """Test coupling with different tensor dimensions."""
        test_configs = [
            (128, 2, 32),   # Small
            (512, 4, 128),  # Medium  
            (1024, 8, 256), # Large
        ]
        
        for d_model, batch_size, seq_len in test_configs:
            coupling = AdditiveCoupling(d_model)
            
            x = torch.randn(batch_size, seq_len, d_model)
            x1, x2 = coupling.split_input(x)
            
            y1, y2 = coupling.forward(x1, x2)
            x1_rec, x2_rec = coupling.inverse(y1, y2)
            
            torch.testing.assert_close(x1, x1_rec, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(x2, x2_rec, atol=1e-5, rtol=1e-5)


class TestAffineCoupling:
    """Test affine coupling implementation."""
    
    def test_forward_inverse_consistency(self):
        """Test forward and inverse consistency for affine coupling."""
        d_model = 512
        coupling = AffineCoupling(d_model, scale_init=0.1)
        
        x = torch.randn(2, 64, d_model)
        x1, x2 = coupling.split_input(x)
        
        y1, y2 = coupling.forward(x1, x2)
        x1_rec, x2_rec = coupling.inverse(y1, y2)
        
        torch.testing.assert_close(x1, x1_rec, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(x2, x2_rec, atol=1e-5, rtol=1e-5)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        d_model = 256
        coupling = AffineCoupling(d_model, scale_init=0.01)  # Small scale for stability
        
        # Test with large values
        x = torch.randn(2, 32, d_model) * 10
        x1, x2 = coupling.split_input(x)
        
        y1, y2 = coupling.forward(x1, x2)
        
        # Check for NaN or Inf
        assert not torch.isnan(y1).any()
        assert not torch.isnan(y2).any()
        assert not torch.isinf(y1).any()
        assert not torch.isinf(y2).any()
        
        # Test reconstruction
        x1_rec, x2_rec = coupling.inverse(y1, y2)
        torch.testing.assert_close(x1, x1_rec, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x2, x2_rec, atol=1e-4, rtol=1e-4)


class TestLearnedCoupling:
    """Test learned coupling with gating."""
    
    def test_forward_inverse_consistency(self):
        """Test consistency for learned coupling."""
        d_model = 512
        coupling = LearnedCoupling(d_model)
        
        x = torch.randn(2, 64, d_model)
        x1, x2 = coupling.split_input(x)
        
        y1, y2 = coupling.forward(x1, x2)
        x1_rec, x2_rec = coupling.inverse(y1, y2)
        
        torch.testing.assert_close(x1, x1_rec, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(x2, x2_rec, atol=1e-6, rtol=1e-6)
    
    def test_gating_behavior(self):
        """Test that gating affects the transformation."""
        d_model = 256
        coupling = LearnedCoupling(d_model)
        
        # Two different inputs
        x1_a = torch.randn(1, 32, d_model // 2)
        x1_b = torch.randn(1, 32, d_model // 2)
        x2 = torch.randn(1, 32, d_model // 2)
        
        # Get outputs for different x1 values
        _, y2_a = coupling.forward(x1_a, x2)
        _, y2_b = coupling.forward(x1_b, x2)
        
        # Outputs should be different due to gating
        assert not torch.allclose(y2_a, y2_b, atol=1e-6)


class TestCouplingMemoryEfficiency:
    """Test memory efficiency properties of coupling layers."""
    
    def test_memory_usage_comparison(self):
        """Compare memory usage between coupling types."""
        d_model = 1024
        batch_size = 4
        seq_len = 256
        
        couplings = [
            AdditiveCoupling(d_model),
            AffineCoupling(d_model),
            LearnedCoupling(d_model),
        ]
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        for coupling in couplings:
            x1, x2 = coupling.split_input(x)
            
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem_before = torch.cuda.memory_allocated()
            
            y1, y2 = coupling.forward(x1, x2)
            
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                memory_increase = mem_after - mem_before
                
                # Memory increase should be reasonable
                expected_memory = x.numel() * 4  # float32
                assert memory_increase < expected_memory * 2  # Allow some overhead
    
    def test_gradient_checkpointing_compatibility(self):
        """Test compatibility with gradient checkpointing."""
        from torch.utils.checkpoint import checkpoint
        
        d_model = 512
        coupling = AdditiveCoupling(d_model)
        
        def coupling_fn(x1, x2):
            return coupling.forward(x1, x2)
        
        x = torch.randn(2, 64, d_model, requires_grad=True)
        x1, x2 = coupling.split_input(x)
        
        # Use gradient checkpointing
        y1, y2 = checkpoint(coupling_fn, x1, x2)
        
        loss = (y1.sum() + y2.sum())
        loss.backward()
        
        # Check gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


if __name__ == "__main__":
    pytest.main([__file__])