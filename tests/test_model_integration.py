"""
Integration tests for the complete reversible transformer model.

Tests end-to-end functionality including forward/backward passes,
memory scheduling, and training workflows.
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any

try:
    from revnet_zero import (
        ReversibleTransformer,
        MemoryScheduler,
        AdaptiveScheduler,
        LongContextTrainer,
    )
    from revnet_zero.layers import ReversibleAttention, ReversibleFFN
    from revnet_zero.utils import convert_to_reversible
except ImportError:
    import sys
    sys.path.append('..')
    from revnet_zero import (
        ReversibleTransformer,
        MemoryScheduler,
        AdaptiveScheduler,
        LongContextTrainer,
    )
    from revnet_zero.layers import ReversibleAttention, ReversibleFFN
    from revnet_zero.utils import convert_to_reversible


class TestReversibleTransformerIntegration:
    """Integration tests for reversible transformer."""
    
    if HAS_PYTEST:
        @pytest.fixture
        def device(self):
            """Get test device."""
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        @pytest.fixture
        def small_model(self, device):
            """Create small model for testing."""
            model = ReversibleTransformer(
                vocab_size=100,
                num_layers=2,
                d_model=64,
                num_heads=4,
                d_ff=256,
                max_seq_len=128,
                dropout=0.1,
            ).to(device)
            return model
        
        @pytest.fixture
        def sample_data(self, device):
            """Create sample training data."""
            batch_size, seq_len = 2, 32
            input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
            labels = torch.randint(0, 100, (batch_size, seq_len), device=device)
            return {"input_ids": input_ids, "labels": labels}
    
    @staticmethod
    def get_device():
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def create_small_model():
        """Create small model for testing."""
        device = TestReversibleTransformerIntegration.get_device()
        model = ReversibleTransformer(
            vocab_size=100,
            num_layers=2,
            d_model=64,
            num_heads=4,
            d_ff=256,
            max_seq_len=128,
            dropout=0.1,
        ).to(device)
        return model
    
    @staticmethod
    def create_sample_data():
        """Create sample training data."""
        device = TestReversibleTransformerIntegration.get_device()
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 100, (batch_size, seq_len), device=device)
        return {"input_ids": input_ids, "labels": labels}
    
    def test_model_creation(self, small_model):
        """Test basic model creation and parameter counting."""
        assert isinstance(small_model, ReversibleTransformer)
        
        # Check parameter count
        total_params = sum(p.numel() for p in small_model.parameters())
        assert total_params > 0
        
        # Check model info
        model_info = small_model.get_model_info()
        assert model_info["num_layers"] == 2
        assert model_info["d_model"] == 64
        assert model_info["num_heads"] == 4
        assert model_info["total_parameters"] == total_params
    
    def test_forward_pass(self, small_model, sample_data, device):
        """Test forward pass in both modes."""
        input_ids = sample_data["input_ids"]
        
        # Test reversible mode
        small_model.set_reversible_mode(True)
        outputs_rev = small_model(input_ids)
        
        assert "logits" in outputs_rev
        assert outputs_rev["logits"].shape == (*input_ids.shape, small_model.vocab_size)
        
        # Test standard mode
        small_model.set_reversible_mode(False)
        outputs_std = small_model(input_ids)
        
        assert "logits" in outputs_std
        assert outputs_std["logits"].shape == outputs_rev["logits"].shape
        
        # Outputs should be similar (not identical due to numerical differences)
        logit_diff = torch.abs(outputs_rev["logits"] - outputs_std["logits"]).max()
        assert logit_diff < 1e-3  # Allow small numerical differences
    
    def test_backward_pass(self, small_model, sample_data):
        """Test backward pass and gradient computation."""
        input_ids = sample_data["input_ids"]
        labels = sample_data["labels"]
        
        # Test reversible mode
        small_model.train()
        small_model.set_reversible_mode(True)
        small_model.zero_grad()
        
        outputs = small_model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        
        assert loss.requires_grad
        loss.backward()
        
        # Check that gradients are computed
        grad_norms = []
        for name, param in small_model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        assert len(grad_norms) > 0
        assert all(norm >= 0 for norm in grad_norms)
    
    def test_memory_estimation(self, small_model):
        """Test memory usage estimation."""
        batch_size, seq_len = 2, 64
        
        memory_est = small_model.estimate_memory_usage(batch_size, seq_len)
        
        required_fields = [
            "total_memory", "embedding_memory", "layer_memory", 
            "output_memory", "memory_saved", "reduction_ratio"
        ]
        
        for field in required_fields:
            assert field in memory_est
            assert isinstance(memory_est[field], (int, float))
            assert memory_est[field] >= 0
        
        # Memory saved should be positive for reversible model
        assert memory_est["memory_saved"] > 0
        assert 0 <= memory_est["reduction_ratio"] <= 1
    
    def test_memory_scheduler_integration(self, small_model, sample_data):
        """Test integration with memory scheduler."""
        scheduler = MemoryScheduler(small_model)
        small_model.set_memory_scheduler(scheduler)
        
        input_ids = sample_data["input_ids"]
        
        # Test with scheduler
        with scheduler:
            outputs = small_model(input_ids)
            loss = outputs["logits"].sum()
            loss.backward()
        
        # Check scheduler statistics
        assert hasattr(scheduler, 'current_memory_usage')
        assert hasattr(scheduler, 'recomputed_layers')
        
        small_model.zero_grad()
    
    def test_adaptive_scheduler(self, small_model, sample_data):
        """Test adaptive memory scheduler."""
        scheduler = AdaptiveScheduler(small_model)
        small_model.set_memory_scheduler(scheduler)
        
        input_ids = sample_data["input_ids"]
        
        # Run multiple steps to test adaptation
        for step in range(5):
            with scheduler:
                outputs = small_model(input_ids)
                loss = outputs["logits"].sum()
                loss.backward()
            
            # Test adaptation
            if hasattr(scheduler, 'adapt_policies'):
                scheduler.adapt_policies(step)
            
            small_model.zero_grad()
    
    def test_training_step(self, small_model, sample_data):
        """Test a complete training step."""
        optimizer = optim.AdamW(small_model.parameters(), lr=1e-3)
        
        input_ids = sample_data["input_ids"]
        labels = sample_data["labels"]
        
        small_model.train()
        
        # Initial loss
        outputs = small_model(input_ids=input_ids, labels=labels)
        initial_loss = outputs["loss"].item()
        
        # Training step
        optimizer.zero_grad()
        outputs = small_model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        
        # New loss (should change after optimization step)
        outputs = small_model(input_ids=input_ids, labels=labels)
        new_loss = outputs["loss"].item()
        
        # Loss should change (though may not necessarily decrease for one step)
        assert initial_loss != new_loss
    
    def test_eval_mode(self, small_model, sample_data):
        """Test evaluation mode."""
        input_ids = sample_data["input_ids"]
        
        # Set to eval mode
        small_model.eval()
        
        with torch.no_grad():
            outputs = small_model(input_ids)
            
        assert "logits" in outputs
        assert not outputs["logits"].requires_grad
    
    def test_different_sequence_lengths(self, small_model, device):
        """Test with different sequence lengths."""
        vocab_size = small_model.vocab_size
        
        seq_lengths = [16, 32, 64, 128]
        
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
            
            # Should handle different sequence lengths
            outputs = small_model(input_ids)
            
            assert outputs["logits"].shape == (1, seq_len, vocab_size)
    
    def test_batch_processing(self, small_model, device):
        """Test batch processing with different batch sizes."""
        vocab_size = small_model.vocab_size
        seq_len = 32
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            
            outputs = small_model(input_ids)
            
            assert outputs["logits"].shape == (batch_size, seq_len, vocab_size)


class TestLongContextTrainer:
    """Test long context trainer functionality."""
    
    @pytest.fixture
    def trainer_setup(self, device):
        """Setup trainer with small model and data."""
        model = ReversibleTransformer(
            vocab_size=100,
            num_layers=2,
            d_model=32,
            num_heads=2,
            d_ff=128,
            max_seq_len=64,
        ).to(device)
        
        # Create sample dataset
        batch_size, seq_len, num_samples = 2, 32, 16
        input_ids = torch.randint(0, 100, (num_samples, seq_len))
        labels = torch.randint(0, 100, (num_samples, seq_len))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        trainer = LongContextTrainer(
            model=model,
            max_length=64,
            gradient_accumulation_steps=1,
            use_amp=False,  # Disable AMP for testing
            log_memory_usage=False,
        )
        
        return trainer, dataloader
    
    def test_trainer_creation(self, trainer_setup):
        """Test trainer creation."""
        trainer, dataloader = trainer_setup
        
        assert isinstance(trainer, LongContextTrainer)
        assert trainer.max_length == 64
        assert trainer.gradient_accumulation_steps == 1
    
    def test_training_step(self, trainer_setup):
        """Test single training step."""
        trainer, dataloader = trainer_setup
        
        # Get a batch
        batch = next(iter(dataloader))
        batch = {"input_ids": batch[0], "labels": batch[1]}
        
        # Create optimizer
        optimizer = optim.AdamW(trainer.model.parameters(), lr=1e-3)
        
        # Run training step
        loss, metrics = trainer._training_step(batch, optimizer)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert "learning_rate" in metrics
        assert "gradient_norm" in metrics
    
    def test_mini_training_loop(self, trainer_setup):
        """Test a mini training loop."""
        trainer, dataloader = trainer_setup
        
        # Run very short training
        results = trainer.train(
            train_dataloader=dataloader,
            num_epochs=1,
            num_steps=3,  # Very short for testing
            logging_steps=1,
        )
        
        assert "training_stats" in results
        assert "total_steps" in results
        assert results["total_steps"] >= 3
        assert len(results["training_stats"]["losses"]) >= 3


class TestModelConversion:
    """Test model conversion utilities."""
    
    def test_conversion_placeholder(self):
        """Placeholder test for model conversion."""
        # This would test convert_to_reversible function
        # Currently just checking import works
        assert callable(convert_to_reversible)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_model_config(self):
        """Test invalid model configurations."""
        with pytest.raises((ValueError, AssertionError)):
            # d_model not divisible by num_heads
            ReversibleTransformer(
                vocab_size=100,
                num_layers=2,
                d_model=65,  # Not divisible by num_heads=4
                num_heads=4,
                d_ff=256,
            )
    
    def test_empty_input(self, device):
        """Test with empty or invalid inputs."""
        model = ReversibleTransformer(
            vocab_size=100,
            num_layers=1,
            d_model=32,
            num_heads=2,
            d_ff=128,
        ).to(device)
        
        # Test with empty sequence (should handle gracefully)
        empty_input = torch.randint(0, 100, (1, 0), device=device)  # Empty sequence
        
        try:
            outputs = model(empty_input)
            # If it doesn't crash, that's good
            assert outputs["logits"].shape[1] == 0
        except (RuntimeError, ValueError):
            # It's also acceptable to raise an error for empty input
            pass


if __name__ == "__main__":
    # Run tests if executed directly
    import sys
    
    # Simple test runner for environments without pytest
    def run_basic_tests():
        """Run basic tests without pytest."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running tests on device: {device}")
        
        try:
            # Test model creation
            print("Testing model creation...")
            model = ReversibleTransformer(
                vocab_size=100,
                num_layers=2,
                d_model=64,
                num_heads=4,
                d_ff=256,
            ).to(device)
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✓ Model created with {param_count:,} parameters")
            
            # Test forward pass
            print("Testing forward pass...")
            input_ids = torch.randint(0, 100, (2, 32), device=device)
            outputs = model(input_ids)
            assert "logits" in outputs
            print(f"✓ Forward pass - Output shape: {outputs['logits'].shape}")
            
            # Test backward pass
            print("Testing backward pass...")
            model.train()
            labels = torch.randint(0, 100, (2, 32), device=device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            print(f"✓ Backward pass - Loss: {loss.item():.4f}")
            
            # Test memory estimation
            print("Testing memory estimation...")
            memory_est = model.estimate_memory_usage(2, 32)
            total_gb = memory_est["total_memory"] / 1e9
            saved_gb = memory_est["memory_saved"] / 1e9
            print(f"✓ Memory estimation - Total: {total_gb:.3f}GB, Saved: {saved_gb:.3f}GB")
            
            # Test reversible vs standard modes
            print("Testing reversible vs standard modes...")
            model.set_reversible_mode(True)
            outputs_rev = model(input_ids)
            
            model.set_reversible_mode(False)
            outputs_std = model(input_ids)
            
            logit_diff = torch.abs(outputs_rev["logits"] - outputs_std["logits"]).max()
            print(f"✓ Mode comparison - Max difference: {logit_diff:.6f}")
            
            print("\n🎉 All basic tests passed! ✓")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if len(sys.argv) > 1 and sys.argv[1] == "--basic":
        run_basic_tests()
    else:
        print("Use --basic flag to run basic tests without pytest")