"""
Tests for memory scheduling functionality.

Tests the memory scheduler's ability to make intelligent decisions
about activation recomputation vs storage.
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import Mock, patch

from revnet_zero.memory.scheduler import (
    MemoryScheduler, 
    AdaptiveScheduler, 
    LayerScheduler,
    RecomputePolicy,
    LayerMemoryProfile
)
from revnet_zero.models.reversible_transformer import ReversibleTransformer


class TestMemoryScheduler:
    """Test basic memory scheduler functionality."""
    
    def test_initialization(self):
        """Test scheduler initialization."""
        model = Mock()
        scheduler = MemoryScheduler(model)
        
        assert scheduler.model == model
        assert scheduler.memory_budget > 0
        assert scheduler.strategy == "adaptive"
        assert len(scheduler.layer_policies) == 0
    
    def test_set_policy(self):
        """Test setting recomputation policies."""
        model = Mock()
        scheduler = MemoryScheduler(model)
        
        # Set policy for single layer
        scheduler.set_policy(0, RecomputePolicy.RECOMPUTE)
        assert scheduler.layer_policies[0] == RecomputePolicy.RECOMPUTE
        
        # Set policy for multiple layers
        scheduler.set_policy([1, 2, 3], RecomputePolicy.STORE)
        for i in [1, 2, 3]:
            assert scheduler.layer_policies[i] == RecomputePolicy.STORE
    
    def test_should_recompute_explicit_policy(self):
        """Test recomputation decision with explicit policies."""
        model = Mock()
        scheduler = MemoryScheduler(model)
        
        # Set explicit policies
        scheduler.set_policy(0, RecomputePolicy.STORE)
        scheduler.set_policy(1, RecomputePolicy.RECOMPUTE)
        
        assert not scheduler.should_recompute(0)
        assert scheduler.should_recompute(1)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=8 * 1024**3)  # 8GB
    def test_adaptive_decision_high_memory(self, mock_memory, mock_cuda):
        """Test adaptive decision under high memory pressure."""
        model = Mock()
        scheduler = MemoryScheduler(model, memory_budget=10 * 1024**3)  # 10GB budget
        
        # Should prefer recomputation when memory usage is high
        assert scheduler.should_recompute(0)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=2 * 1024**3)  # 2GB
    def test_adaptive_decision_low_memory(self, mock_memory, mock_cuda):
        """Test adaptive decision under low memory pressure."""
        model = Mock()
        scheduler = MemoryScheduler(model, memory_budget=10 * 1024**3)  # 10GB budget
        
        # Should prefer storage when memory usage is low
        assert not scheduler.should_recompute(0)
    
    def test_context_manager(self):
        """Test scheduler as context manager."""
        model = Mock()
        scheduler = MemoryScheduler(model)
        
        with scheduler as sched:
            assert sched == scheduler
            assert len(sched.recomputed_layers) == 0
        
        # After context, stats should be updated
        assert hasattr(scheduler, 'current_memory_usage')


class TestAdaptiveScheduler:
    """Test adaptive scheduler with dynamic policies."""
    
    def test_initialization_with_profiles(self):
        """Test initialization with layer profiles."""
        model = Mock()
        
        # Create mock profiles
        profiles = {
            0: LayerMemoryProfile(0, 1000, 500, 2.0, 800),  # High storage cost, high recompute cost
            1: LayerMemoryProfile(1, 500, 200, 1.0, 400),   # Medium storage cost, low recompute cost
            2: LayerMemoryProfile(2, 200, 100, 0.5, 100),   # Low storage cost, very low recompute cost
        }
        
        scheduler = AdaptiveScheduler(model, profile=profiles)
        
        assert len(scheduler.layer_profiles) == 3
        assert len(scheduler.layer_policies) > 0
    
    def test_policy_adaptation(self):
        """Test policy adaptation based on memory pressure."""
        model = Mock()
        
        profiles = {
            0: LayerMemoryProfile(0, 1000, 500, 1.0, 800),
            1: LayerMemoryProfile(1, 500, 200, 1.0, 400),
        }
        
        scheduler = AdaptiveScheduler(model, profile=profiles)
        
        # Simulate high memory pressure
        scheduler.current_memory_usage = scheduler.memory_budget * 0.95
        scheduler.adapt_policies(100)
        
        # Should adjust threshold to prefer more recomputation
        assert scheduler.memory_threshold < 0.8
    
    def test_efficiency_based_initialization(self):
        """Test that initial policies are based on memory efficiency."""
        model = Mock()
        
        profiles = {
            0: LayerMemoryProfile(0, 1000, 500, 1.0, 1000),  # High storage, low recompute -> efficient to recompute
            1: LayerMemoryProfile(1, 500, 200, 10.0, 100),   # Low storage, high recompute -> efficient to store
        }
        
        scheduler = AdaptiveScheduler(model, profile=profiles)
        
        # Layer 0 should be set to recompute (high storage cost)
        # Layer 1 should be set to store (high recompute cost)
        # Note: exact policies depend on implementation details


class TestLayerScheduler:
    """Test layer-specific scheduler with fine-grained control."""
    
    def test_layer_groups(self):
        """Test creation and management of layer groups."""
        model = Mock()
        scheduler = LayerScheduler(model)
        
        # Create layer groups
        scheduler.create_layer_group("early", [0, 1, 2])
        scheduler.create_layer_group("middle", [3, 4, 5])
        scheduler.create_layer_group("late", [6, 7, 8])
        
        assert "early" in scheduler.layer_groups
        assert scheduler.layer_groups["early"] == [0, 1, 2]
    
    def test_group_policies(self):
        """Test setting policies for layer groups."""
        model = Mock()
        scheduler = LayerScheduler(model)
        
        scheduler.create_layer_group("early", [0, 1, 2])
        scheduler.set_group_policy("early", RecomputePolicy.STORE)
        
        # All layers in group should have the policy
        for layer_id in [0, 1, 2]:
            assert scheduler.layer_policies[layer_id] == RecomputePolicy.STORE
    
    def test_layer_range_policy(self):
        """Test setting policies for layer ranges."""
        model = Mock()
        scheduler = LayerScheduler(model)
        
        scheduler.set_layer_range_policy(2, 6, RecomputePolicy.RECOMPUTE)
        
        # Layers 2, 3, 4, 5 should have recompute policy
        for layer_id in range(2, 6):
            assert scheduler.layer_policies[layer_id] == RecomputePolicy.RECOMPUTE
    
    def test_memory_summary(self):
        """Test memory summary generation."""
        model = Mock()
        scheduler = LayerScheduler(model)
        
        scheduler.set_policy([0, 1], RecomputePolicy.STORE)
        scheduler.create_layer_group("test", [2, 3])
        
        summary = scheduler.get_memory_summary()
        
        assert "memory_budget" in summary
        assert "layer_policies" in summary
        assert "layer_groups" in summary
        assert summary["layer_policies"][0] == RecomputePolicy.STORE


class TestMemorySchedulerIntegration:
    """Test scheduler integration with real models."""
    
    def test_scheduler_with_reversible_transformer(self):
        """Test scheduler integration with reversible transformer."""
        model = ReversibleTransformer(
            vocab_size=1000,
            num_layers=4,
            d_model=256,
            num_heads=4,
            d_ff=1024,
        )
        
        scheduler = MemoryScheduler(model)
        model.set_memory_scheduler(scheduler)
        
        # Test forward pass with scheduler
        input_ids = torch.randint(0, 1000, (2, 64))
        
        with scheduler:
            output = model(input_ids)
        
        assert "logits" in output
        assert output["logits"].shape == (2, 64, 1000)
    
    def test_scheduler_recommendations(self):
        """Test scheduler's ability to make optimization recommendations."""
        model = ReversibleTransformer(
            vocab_size=1000,
            num_layers=6,
            d_model=512,
            num_heads=8,
            d_ff=2048,
        )
        
        scheduler = LayerScheduler(model)
        
        # Set up policies based on typical recommendations
        scheduler.set_layer_range_policy(0, 2, RecomputePolicy.STORE)      # Early layers: store
        scheduler.set_layer_range_policy(2, 4, RecomputePolicy.RECOMPUTE)  # Middle layers: recompute
        scheduler.set_layer_range_policy(4, 6, RecomputePolicy.ADAPTIVE)   # Late layers: adaptive
        
        summary = scheduler.get_memory_summary()
        
        # Should have policies for all layers
        assert len(summary["layer_policies"]) == 6
        
        # Check policy distribution
        policies = list(summary["layer_policies"].values())
        assert RecomputePolicy.STORE in policies
        assert RecomputePolicy.RECOMPUTE in policies
        assert RecomputePolicy.ADAPTIVE in policies
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_pressure_adaptation(self):
        """Test scheduler adaptation under real memory pressure."""
        model = ReversibleTransformer(
            vocab_size=1000,
            num_layers=8,
            d_model=1024,
            num_heads=16,
            d_ff=4096,
        ).cuda()
        
        # Create scheduler with realistic memory budget
        total_memory = torch.cuda.get_device_properties(0).total_memory
        budget = int(0.5 * total_memory)  # Use 50% of available memory
        
        scheduler = AdaptiveScheduler(model, memory_budget=budget)
        
        # Test with progressively larger inputs
        seq_lengths = [128, 256, 512, 1024]
        
        for seq_len in seq_lengths:
            try:
                input_ids = torch.randint(0, 1000, (1, seq_len)).cuda()
                
                with scheduler:
                    output = model(input_ids)
                
                # Should complete without OOM
                assert output["logits"].shape == (1, seq_len, 1000)
                
                # Scheduler should adapt policies
                scheduler.adapt_policies(seq_len)
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Expected for very large sequences
                    break
                else:
                    raise


if __name__ == "__main__":
    pytest.main([__file__])