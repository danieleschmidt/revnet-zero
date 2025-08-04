"""
Basic usage example for RevNet-Zero reversible transformers.

This example demonstrates the core functionality and basic API usage
for training memory-efficient transformers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math
from typing import Dict, Any

try:
    from ..revnet_zero import (
        ReversibleTransformer,
        MemoryScheduler,
        LongContextTrainer,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('..')
    from revnet_zero import (
        ReversibleTransformer,
        MemoryScheduler, 
        LongContextTrainer,
    )


class BasicExample:
    """
    Basic example demonstrating RevNet-Zero functionality.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize basic example.
        
        Args:
            device: Device to use ("cuda", "cpu", or "auto")
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
    
    def create_simple_model(self) -> ReversibleTransformer:
        """
        Create a simple reversible transformer model.
        
        Returns:
            Initialized reversible transformer
        """
        model = ReversibleTransformer(
            vocab_size=1000,
            num_layers=4,
            d_model=128,
            num_heads=4,
            d_ff=512,
            max_seq_len=512,
            coupling="additive",
            dropout=0.1,
            use_flash_attention=False,
        ).to(self.device)
        
        print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def create_sample_data(self, batch_size: int = 4, seq_len: int = 128, 
                          num_batches: int = 10) -> DataLoader:
        """
        Create sample training data.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_batches: Number of batches
            
        Returns:
            DataLoader with synthetic data
        """
        # Generate random token sequences
        input_ids = torch.randint(0, 1000, (num_batches * batch_size, seq_len))
        labels = torch.randint(0, 1000, (num_batches * batch_size, seq_len))
        
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Created dataset with {len(dataset)} samples")
        return dataloader
    
    def compare_memory_usage(self, model: ReversibleTransformer, 
                           batch_size: int = 2, seq_len: int = 128) -> Dict[str, Any]:
        """
        Compare memory usage between reversible and standard modes.
        
        Args:
            model: Reversible transformer model
            batch_size: Batch size for comparison
            seq_len: Sequence length
            
        Returns:
            Memory usage comparison results
        """
        # Create test input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        
        results = {}
        
        # Test reversible mode
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        model.train()
        model.set_reversible_mode(True)
        
        # Forward pass with reversible computation
        outputs = model(input_ids)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss.backward()
        
        if self.device.type == "cuda":
            reversible_memory = torch.cuda.max_memory_allocated()
            results["reversible_memory"] = reversible_memory
        
        # Clear gradients
        model.zero_grad()
        
        # Test standard mode
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        model.set_reversible_mode(False)
        
        # Forward pass with standard computation
        outputs = model(input_ids)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss.backward()
        
        if self.device.type == "cuda":
            standard_memory = torch.cuda.max_memory_allocated()
            results["standard_memory"] = standard_memory
            results["memory_saved"] = standard_memory - reversible_memory
            results["reduction_percentage"] = (standard_memory - reversible_memory) / standard_memory * 100
        
        model.zero_grad()
        
        return results
    
    def basic_training_loop(self, model: ReversibleTransformer, 
                          dataloader: DataLoader, num_epochs: int = 2) -> Dict[str, Any]:
        """
        Run a basic training loop with memory scheduling.
        
        Args:
            model: Model to train
            dataloader: Training data
            num_epochs: Number of epochs
            
        Returns:
            Training statistics
        """
        # Setup optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Setup memory scheduler
        memory_scheduler = MemoryScheduler(model)
        model.set_memory_scheduler(memory_scheduler)
        
        # Training loop
        model.train()
        training_stats = {"losses": [], "steps": [], "memory_usage": []}
        
        step = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (input_ids, labels) in enumerate(dataloader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass with memory scheduling
                with memory_scheduler:
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # Track statistics
                step += 1
                epoch_loss += loss.item()
                training_stats["losses"].append(loss.item())
                training_stats["steps"].append(step)
                
                # Track memory usage
                if self.device.type == "cuda":
                    memory_usage = torch.cuda.memory_allocated()
                    training_stats["memory_usage"].append(memory_usage)
                
                if batch_idx % 5 == 0:
                    memory_str = ""
                    if self.device.type == "cuda":
                        memory_gb = torch.cuda.memory_allocated() / 1e9
                        memory_str = f", Memory: {memory_gb:.2f}GB"
                    
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}{memory_str}")
        
        training_time = time.time() - start_time
        avg_loss = sum(training_stats["losses"]) / len(training_stats["losses"])
        
        print(f"Training completed in {training_time:.2f}s, Average loss: {avg_loss:.4f}")
        
        return {
            "training_time": training_time,
            "average_loss": avg_loss,
            "final_loss": training_stats["losses"][-1],
            "total_steps": step,
            "training_stats": training_stats,
        }
    
    def demonstrate_model_info(self, model: ReversibleTransformer):
        """
        Demonstrate model information and memory estimation features.
        
        Args:
            model: Model to analyze
        """
        # Get model info
        model_info = model.get_model_info()
        
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        
        for key, value in model_info.items():
            if isinstance(value, (int, float)):
                if "memory" in key.lower() or "size" in key.lower():
                    print(f"{key}: {value:,}")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: {value}")
        
        # Memory estimation
        print("\nMEMORY ESTIMATION")
        print("-" * 30)
        
        batch_sizes = [1, 2, 4, 8]
        seq_lengths = [512, 1024, 2048, 4096]
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                memory_est = model.estimate_memory_usage(batch_size, seq_len)
                total_gb = memory_est["total_memory"] / 1e9
                saved_gb = memory_est["memory_saved"] / 1e9
                reduction = memory_est["reduction_ratio"] * 100
                
                print(f"Batch {batch_size}, Seq {seq_len}: "
                      f"{total_gb:.2f}GB total, "
                      f"{saved_gb:.2f}GB saved ({reduction:.1f}% reduction)")
    
    def run_complete_example(self) -> Dict[str, Any]:
        """
        Run the complete basic example.
        
        Returns:
            Complete example results
        """
        print("🚀 RevNet-Zero Basic Example")
        print("="*50)
        
        # 1. Create model
        print("\n1. Creating reversible transformer model...")
        model = self.create_simple_model()
        
        # 2. Show model information
        print("\n2. Model information and memory estimation...")
        self.demonstrate_model_info(model)
        
        # 3. Create sample data
        print("\n3. Creating sample training data...")
        dataloader = self.create_sample_data(batch_size=2, seq_len=128, num_batches=8)
        
        # 4. Memory comparison
        print("\n4. Comparing memory usage (reversible vs standard)...")
        try:
            memory_comparison = self.compare_memory_usage(model, batch_size=2, seq_len=128)
            
            if self.device.type == "cuda" and memory_comparison:
                rev_gb = memory_comparison["reversible_memory"] / 1e9
                std_gb = memory_comparison["standard_memory"] / 1e9
                saved_gb = memory_comparison["memory_saved"] / 1e9
                reduction = memory_comparison["reduction_percentage"]
                
                print(f"Reversible mode: {rev_gb:.2f}GB")
                print(f"Standard mode: {std_gb:.2f}GB") 
                print(f"Memory saved: {saved_gb:.2f}GB ({reduction:.1f}% reduction)")
            else:
                print("Memory comparison only available on CUDA devices")
        except Exception as e:
            print(f"Memory comparison failed: {e}")
        
        # 5. Basic training
        print("\n5. Running basic training loop...")
        training_results = self.basic_training_loop(model, dataloader, num_epochs=2)
        
        # 6. Summary
        print("\n" + "="*50)
        print("EXAMPLE COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"✓ Training completed in {training_results['training_time']:.2f}s")
        print(f"✓ Final loss: {training_results['final_loss']:.4f}")
        print(f"✓ Total training steps: {training_results['total_steps']}")
        
        return {
            "model_info": model.get_model_info(),
            "memory_comparison": memory_comparison if 'memory_comparison' in locals() else None,
            "training_results": training_results,
        }


def main():
    """Run the basic example."""
    example = BasicExample()
    results = example.run_complete_example()
    return results


if __name__ == "__main__":
    main()