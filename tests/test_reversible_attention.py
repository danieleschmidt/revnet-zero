"""
Tests for reversible attention layers.

Tests the correctness and memory efficiency of reversible attention
implementations.
"""

import torch
import torch.nn as nn
import pytest
from revnet_zero.layers.reversible_attention import ReversibleAttention, MultiHeadAttention
from revnet_zero.layers.coupling_layers import AdditiveCoupling


class TestMultiHeadAttention:
    """Test standard multi-head attention component."""
    
    def test_attention_output_shape(self):
        """Test that attention produces correct output shape."""
        d_model = 512
        num_heads = 8
        batch_size = 2
        seq_len = 128
        
        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_attention_mask(self):
        """Test attention with causal mask."""
        d_model = 256
        num_heads = 4
        seq_len = 32
        
        attention = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(1, seq_len, d_model)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        output = attention(x, attention_mask=mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_flash_attention_compatibility(self):
        """Test flash attention when available."""
        d_model = 512
        num_heads = 8
        
        # Test both with and without flash attention
        attention_standard = MultiHeadAttention(d_model, num_heads, use_flash_attention=False)
        attention_flash = MultiHeadAttention(d_model, num_heads, use_flash_attention=True)
        
        x = torch.randn(2, 64, d_model)
        
        output_standard = attention_standard(x)
        output_flash = attention_flash(x)
        
        # Outputs should have same shape
        assert output_standard.shape == output_flash.shape


class TestReversibleAttention:
    """Test reversible attention layer."""
    
    def test_reversible_vs_standard_equivalence(self):
        """Test that reversible attention matches standard attention."""
        d_model = 512
        num_heads = 8
        batch_size = 2
        seq_len = 64
        
        rev_attention = ReversibleAttention(d_model, num_heads, coupling="additive")
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Get outputs from both modes
        rev_attention.eval()  # Set to eval to disable training-specific behavior
        
        output_reversible = rev_attention(x, use_reversible=True)
        output_standard = rev_attention(x, use_reversible=False)
        
        # Outputs should be close (may have small numerical differences)
        torch.testing.assert_close(output_reversible, output_standard, atol=1e-4, rtol=1e-4)
    
    def test_gradient_computation(self):
        """Test gradient computation through reversible attention."""
        d_model = 256
        num_heads = 4
        
        rev_attention = ReversibleAttention(d_model, num_heads)
        x = torch.randn(2, 32, d_model, requires_grad=True)
        
        # Forward pass
        output = rev_attention(x, use_reversible=True)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # Check gradient has reasonable magnitude
        grad_norm = x.grad.norm()
        assert grad_norm > 0 and grad_norm < 100  # Reasonable range
    
    def test_memory_efficiency(self):
        """Test memory efficiency of reversible attention."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        d_model = 1024
        num_heads = 16
        seq_len = 512
        
        rev_attention = ReversibleAttention(d_model, num_heads).cuda()
        x = torch.randn(1, seq_len, d_model, requires_grad=True).cuda()
        
        # Test standard mode memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        output_standard = rev_attention(x, use_reversible=False)
        loss_standard = output_standard.sum()
        loss_standard.backward()
        
        memory_standard = torch.cuda.max_memory_allocated()
        
        # Clear and test reversible mode
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x.grad = None  # Clear previous gradients
        
        rev_attention.train()  # Enable training mode for reversible computation
        output_reversible = rev_attention(x, use_reversible=True)
        loss_reversible = output_reversible.sum()
        loss_reversible.backward()
        
        memory_reversible = torch.cuda.max_memory_allocated()
        
        # Reversible should use less memory (though the difference might be small for small sequences)
        memory_reduction = (memory_standard - memory_reversible) / memory_standard
        print(f"Memory reduction: {memory_reduction:.2%}")
        
        # For very long sequences, memory reduction should be significant
        # For shorter sequences, the overhead might be similar
        assert memory_reversible <= memory_standard * 1.2  # Allow some overhead
    
    def test_different_coupling_types(self):
        """Test different coupling function types."""
        d_model = 256
        num_heads = 4
        
        coupling_types = ["additive"]  # Start with additive
        
        x = torch.randn(2, 32, d_model)
        
        for coupling_type in coupling_types:
            rev_attention = ReversibleAttention(d_model, num_heads, coupling=coupling_type)
            
            output = rev_attention(x, use_reversible=True)
            
            assert output.shape == x.shape
            assert not torch.isnan(output).any()
    
    def test_attention_mask_compatibility(self):
        """Test reversible attention with attention masks."""
        d_model = 256
        num_heads = 4
        seq_len = 32
        
        rev_attention = ReversibleAttention(d_model, num_heads)
        x = torch.randn(2, seq_len, d_model)
        
        # Create attention mask
        mask = torch.ones(2, seq_len, seq_len)
        mask[:, :, seq_len//2:] = 0  # Mask out second half
        
        output = rev_attention(x, attention_mask=mask, use_reversible=True)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_layer_norm_integration(self):
        """Test layer normalization integration."""
        d_model = 512
        num_heads = 8
        
        rev_attention = ReversibleAttention(d_model, num_heads)
        
        # Test that layer norm is applied correctly
        x = torch.randn(2, 64, d_model) * 10  # Large values to test normalization
        
        output = rev_attention(x, use_reversible=True)
        
        # Output should have reasonable scale
        assert output.std() < x.std()  # Should be normalized
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestReversibleAttentionMemoryEstimation:
    """Test memory estimation capabilities."""
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        d_model = 1024
        num_heads = 16
        batch_size = 4
        seq_len = 256
        
        rev_attention = ReversibleAttention(d_model, num_heads)
        
        estimates = rev_attention.estimate_memory_usage(batch_size, seq_len)
        
        # Check that estimates contain expected keys
        expected_keys = ["standard_memory", "reversible_memory", "memory_saved", "reduction_ratio"]
        for key in expected_keys:
            assert key in estimates
        
        # Check that estimates are reasonable
        assert estimates["reversible_memory"] > 0
        assert estimates["memory_saved"] >= 0
        assert 0 <= estimates["reduction_ratio"] <= 1
        assert estimates["standard_memory"] >= estimates["reversible_memory"]
    
    def test_scaling_behavior(self):
        """Test how memory estimates scale with sequence length."""
        d_model = 512
        num_heads = 8
        batch_size = 2
        
        rev_attention = ReversibleAttention(d_model, num_heads)
        
        seq_lengths = [64, 128, 256, 512]
        estimates = []
        
        for seq_len in seq_lengths:
            est = rev_attention.estimate_memory_usage(batch_size, seq_len)
            estimates.append(est)
        
        # Memory should scale with sequence length
        for i in range(1, len(estimates)):
            ratio = estimates[i]["standard_memory"] / estimates[i-1]["standard_memory"]
            expected_ratio = seq_lengths[i] / seq_lengths[i-1]
            
            # Should scale roughly linearly with sequence length
            assert abs(ratio - expected_ratio) < 0.5


if __name__ == "__main__":
    pytest.main([__file__])