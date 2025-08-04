"""
Example: Long context training with RevNet-Zero.

This example demonstrates how to train a reversible transformer
on very long sequences with memory efficiency.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from revnet_zero import ReversibleTransformer, MemoryScheduler, LongContextTrainer
from revnet_zero.training.mixed_precision import ReversibleAMPTrainer
from revnet_zero.memory.profiler import MemoryProfiler
from revnet_zero.utils.benchmarking import PerformanceBenchmark
import numpy as np


class LongSequenceDataset(Dataset):
    """
    Simple dataset for long sequence training.
    
    Generates random sequences of specified length for demonstration.
    In practice, you would load your actual long-context data here.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        sequence_length: int = 32768,
        vocab_size: int = 50257,
    ):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequence
        input_ids = torch.randint(0, self.vocab_size, (self.sequence_length,))
        
        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def create_long_context_model(
    sequence_length: int = 32768,
    vocab_size: int = 50257,
    num_layers: int = 12,
    d_model: int = 768,
    num_heads: int = 12,
    coupling: str = "additive",
    use_flash_attention: bool = True,
) -> ReversibleTransformer:
    """
    Create a reversible transformer optimized for long sequences.
    
    Args:
        sequence_length: Maximum sequence length
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers
        d_model: Hidden dimension
        num_heads: Number of attention heads
        coupling: Coupling function type
        use_flash_attention: Whether to use flash attention
        
    Returns:
        Configured ReversibleTransformer model
    """
    model = ReversibleTransformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_model * 4,
        max_seq_len=sequence_length,
        coupling=coupling,
        dropout=0.1,
        use_flash_attention=use_flash_attention,
        use_rational_attention=True,  # Use rational attention for stability
        tie_weights=True,
    )
    
    return model


def setup_memory_scheduler(
    model: ReversibleTransformer,
    memory_budget_gb: float = 24.0,
    strategy: str = 'adaptive',
) -> MemoryScheduler:
    """
    Setup memory scheduler for efficient training.
    
    Args:
        model: Model to schedule
        memory_budget_gb: Available memory budget in GB
        strategy: Scheduling strategy
        
    Returns:
        Configured memory scheduler
    """
    # Create memory scheduler
    scheduler = MemoryScheduler(
        model=model,
        strategy=strategy,
        memory_budget=int(memory_budget_gb * 1024**3),  # Convert to bytes
        recompute_granularity='layer',
    )
    
    # Set scheduler on model
    model.set_memory_scheduler(scheduler)
    
    return scheduler


def benchmark_memory_usage(
    model: ReversibleTransformer,
    sequence_lengths: list = [4096, 8192, 16384, 32768],
    batch_size: int = 1,
):
    """
    Benchmark memory usage across different sequence lengths.
    
    Args:
        model: Model to benchmark
        sequence_lengths: List of sequence lengths to test
        batch_size: Batch size for testing
    """
    print("\n📊 Benchmarking Memory Usage")
    print("=" * 50)
    
    device = next(model.parameters()).device
    benchmark = PerformanceBenchmark(device=device)
    
    for seq_len in sequence_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Create test input
        input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=device)
        
        # Measure memory with reversible computation
        model.set_reversible_mode(True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        memory_before = torch.cuda.memory_allocated() / (1024**3)
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        memory_after = torch.cuda.memory_allocated() / (1024**3)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        
        memory_used = memory_after - memory_before
        
        print(f"  Memory used: {memory_used:.2f} GB")
        print(f"  Peak memory: {peak_memory:.2f} GB")
        
        # Estimate memory for standard transformer
        estimated_standard_memory = seq_len * batch_size * model.d_model * model.num_layers * 4 / (1024**3)
        memory_reduction = (1 - memory_used / estimated_standard_memory) * 100 if estimated_standard_memory > 0 else 0
        
        print(f"  Estimated standard memory: {estimated_standard_memory:.2f} GB")
        print(f"  Memory reduction: {memory_reduction:.1f}%")


def train_long_context_model():
    """
    Main training function for long context model.
    """
    print("🚀 Starting Long Context Training with RevNet-Zero")
    print("=" * 60)
    
    # Configuration
    config = {
        'sequence_length': 16384,  # 16K tokens
        'vocab_size': 50257,
        'num_layers': 12,
        'd_model': 768,
        'num_heads': 12,
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
        'learning_rate': 5e-5,
        'num_training_steps': 1000,
        'memory_budget_gb': 20.0,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_long_context_model(
        sequence_length=config['sequence_length'],
        vocab_size=config['vocab_size'],
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
    )
    
    model = model.to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"Model type: {model_info['model_type']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Model size: {model_info['parameter_size_mb']:.1f} MB")
    
    # Setup memory scheduler
    print("\n🧠 Setting up memory scheduler...")
    scheduler = setup_memory_scheduler(
        model=model,
        memory_budget_gb=config['memory_budget_gb'],
        strategy='adaptive',
    )
    
    # Benchmark memory usage
    if torch.cuda.is_available():
        benchmark_memory_usage(
            model=model,
            sequence_lengths=[4096, 8192, 16384],
            batch_size=config['batch_size'],
        )
    
    # Create dataset and dataloader
    print("\n📚 Creating dataset...")
    dataset = LongSequenceDataset(
        num_samples=100,  # Small dataset for demonstration
        sequence_length=config['sequence_length'],
        vocab_size=config['vocab_size'],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available(),
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # Setup mixed precision trainer
    print("\n⚡ Setting up mixed precision training...")
    amp_trainer = ReversibleAMPTrainer(
        model=model,
        fp16=torch.cuda.is_available(),
        loss_scale='dynamic',
        keep_batchnorm_fp32=True,
    )
    
    # Training loop
    print(f"\n🎯 Starting training for {config['num_training_steps']} steps...")
    
    stats = amp_trainer.train(
        train_loader=dataloader,
        optimizer=optimizer,
        num_steps=config['num_training_steps'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        max_grad_norm=1.0,
        log_interval=50,
    )
    
    # Print training results
    print("\n📈 Training Results:")
    print(f"  Total steps: {len(stats['losses'])}")
    print(f"  Final loss: {stats['losses'][-1]:.4f}")
    print(f"  Average loss: {np.mean(stats['losses']):.4f}")
    print(f"  Total overflows: {stats['total_overflows']}")
    
    if torch.cuda.is_available():
        print(f"  Peak memory: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    
    # Memory profiling
    print("\n🔍 Running detailed memory profiling...")
    memory_profiler = MemoryProfiler(model)
    
    # Create test input
    test_input = torch.randint(
        0, config['vocab_size'],
        (1, config['sequence_length'] // 4),  # Smaller sequence for profiling
        device=device
    )
    
    memory_results = memory_profiler.profile_memory_usage(
        input_data=test_input,
        num_steps=3,
        include_backward=True,
    )
    
    print(f"Peak memory usage: {memory_results['peak_memory_mb']:.2f} MB")
    print(f"Average memory usage: {memory_results['average_memory_mb']:.2f} MB")
    print(f"Memory growth: {memory_results['memory_growth_mb']:.2f} MB")
    
    print("\n✅ Long context training completed successfully!")
    
    return model, stats


def main():
    """Main function."""
    try:
        model, stats = train_long_context_model()
        
        print("\n🎉 Training completed successfully!")
        print("This example demonstrated:")
        print("  ✅ Training with very long sequences (16K+ tokens)")
        print("  ✅ Memory-efficient reversible computation")
        print("  ✅ Automatic mixed precision training")
        print("  ✅ Adaptive memory scheduling")
        print("  ✅ Performance benchmarking and profiling")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()