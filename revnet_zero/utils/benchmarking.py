"""
Benchmarking utilities for reversible transformers.

This module provides comprehensive benchmarking tools for measuring
performance, memory usage, and throughput of reversible models.
"""

import torch
import torch.nn as nn
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for reversible transformers.
    
    Measures throughput, memory usage, latency, and energy consumption
    across different model configurations and sequence lengths.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = defaultdict(list)
    
    def benchmark_model(
        self,
        model: nn.Module,
        sequence_lengths: List[int] = [512, 1024, 2048, 4096],
        batch_sizes: List[int] = [1, 2, 4, 8],
        num_iterations: int = 10,
        warmup_iterations: int = 3,
        measure_memory: bool = True,
        measure_backward: bool = True,
    ) -> Dict[str, Any]:
        """
        Benchmark model performance across different configurations.
        
        Args:
            model: Model to benchmark
            sequence_lengths: List of sequence lengths to test
            batch_sizes: List of batch sizes to test
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            measure_memory: Whether to measure memory usage
            measure_backward: Whether to measure backward pass
            
        Returns:
            Comprehensive benchmark results
        """
        model = model.to(self.device)
        model.eval()
        
        results = {
            'forward_times': defaultdict(list),
            'backward_times': defaultdict(list),
            'memory_usage': defaultdict(list),
            'throughput': defaultdict(list),
            'configurations': [],
        }
        
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                config_name = f"bs{batch_size}_seq{seq_len}"
                results['configurations'].append({
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'config_name': config_name,
                })
                
                print(f"Benchmarking {config_name}...")
                
                # Create test input
                input_ids = torch.randint(
                    0, model.vocab_size if hasattr(model, 'vocab_size') else 50257,
                    (batch_size, seq_len),
                    device=self.device
                )
                
                # Warmup
                for _ in range(warmup_iterations):
                    with torch.no_grad():
                        _ = model(input_ids)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Forward pass benchmarking
                forward_times = []
                memory_usage = []
                
                for i in range(num_iterations):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Measure memory before
                    if measure_memory and torch.cuda.is_available():
                        memory_before = torch.cuda.memory_allocated()
                    
                    # Forward pass
                    start_time = time.perf_counter()
                    
                    if measure_backward:
                        outputs = model(input_ids)
                        if isinstance(outputs, dict):
                            loss = outputs.get('logits', outputs['last_hidden_state']).sum()
                        else:
                            loss = outputs.sum()
                    else:
                        with torch.no_grad():
                            outputs = model(input_ids)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    forward_time = time.perf_counter() - start_time
                    forward_times.append(forward_time)
                    
                    # Measure memory after forward
                    if measure_memory and torch.cuda.is_available():
                        memory_after_forward = torch.cuda.memory_allocated()
                        memory_usage.append(memory_after_forward - memory_before)
                    
                    # Backward pass
                    if measure_backward:
                        backward_start = time.perf_counter()
                        loss.backward()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        backward_time = time.perf_counter() - backward_start
                        
                        # Clear gradients
                        model.zero_grad()
                    else:
                        backward_time = 0
                    
                    results['backward_times'][config_name].append(backward_time)
                
                # Calculate statistics
                avg_forward_time = np.mean(forward_times)
                avg_memory = np.mean(memory_usage) if memory_usage else 0
                throughput = (batch_size * seq_len) / avg_forward_time  # tokens/second
                
                results['forward_times'][config_name] = forward_times
                results['memory_usage'][config_name] = memory_usage
                results['throughput'][config_name].append(throughput)
        
        # Add model information
        results['model_info'] = {
            'device': str(self.device),
            'model_type': type(model).__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
        }
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        sequence_length: int = 2048,
        batch_size: int = 2,
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Compare performance between different models.
        
        Args:
            models: Dictionary of model name -> model instance
            sequence_length: Sequence length for comparison
            batch_size: Batch size for comparison
            num_iterations: Number of iterations for each model
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"Benchmarking {model_name}...")
            
            results = self.benchmark_model(
                model=model,
                sequence_lengths=[sequence_length],
                batch_sizes=[batch_size],
                num_iterations=num_iterations,
            )
            
            config_name = f"bs{batch_size}_seq{sequence_length}"
            comparison_results[model_name] = {
                'avg_forward_time': np.mean(results['forward_times'][config_name]),
                'avg_backward_time': np.mean(results['backward_times'][config_name]),
                'avg_memory_usage': np.mean(results['memory_usage'][config_name]) if results['memory_usage'][config_name] else 0,
                'throughput': results['throughput'][config_name][0],
                'total_parameters': results['model_info']['total_parameters'],
            }
        
        return comparison_results
    
    def plot_results(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_plots: bool = True,
    ):
        """
        Plot benchmark results.
        
        Args:
            results: Results from benchmark_model
            save_path: Path to save plots
            show_plots: Whether to display plots
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RevNet-Zero Performance Benchmark Results', fontsize=16)
        
        # Extract data for plotting
        configs = results['configurations']
        seq_lengths = sorted(list(set(c['sequence_length'] for c in configs)))
        batch_sizes = sorted(list(set(c['batch_size'] for c in configs)))
        
        # Forward time vs sequence length
        ax1 = axes[0, 0]
        for bs in batch_sizes:
            times = []
            seq_lens = []
            for seq_len in seq_lengths:
                config_name = f"bs{bs}_seq{seq_len}"
                if config_name in results['forward_times']:
                    times.append(np.mean(results['forward_times'][config_name]))
                    seq_lens.append(seq_len)
            if times:
                ax1.plot(seq_lens, times, marker='o', label=f'Batch Size {bs}')
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Forward Time (seconds)')
        ax1.set_title('Forward Pass Time vs Sequence Length')
        ax1.legend()
        ax1.set_xscale('log', base=2)
        
        # Memory usage vs sequence length
        ax2 = axes[0, 1]
        for bs in batch_sizes:
            memory_usage = []
            seq_lens = []
            for seq_len in seq_lengths:
                config_name = f"bs{bs}_seq{seq_len}"
                if config_name in results['memory_usage'] and results['memory_usage'][config_name]:
                    memory_mb = np.mean(results['memory_usage'][config_name]) / (1024 * 1024)
                    memory_usage.append(memory_mb)
                    seq_lens.append(seq_len)
            if memory_usage:
                ax2.plot(seq_lens, memory_usage, marker='o', label=f'Batch Size {bs}')
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Sequence Length')
        ax2.legend()
        ax2.set_xscale('log', base=2)
        
        # Throughput vs sequence length
        ax3 = axes[1, 0]
        for bs in batch_sizes:
            throughputs = []
            seq_lens = []
            for seq_len in seq_lengths:
                config_name = f"bs{bs}_seq{seq_len}"
                if config_name in results['throughput'] and results['throughput'][config_name]:
                    throughputs.append(results['throughput'][config_name][0])
                    seq_lens.append(seq_len)
            if throughputs:
                ax3.plot(seq_lens, throughputs, marker='o', label=f'Batch Size {bs}')
        
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Throughput (tokens/second)')
        ax3.set_title('Throughput vs Sequence Length')
        ax3.legend()
        ax3.set_xscale('log', base=2)
        
        # Memory efficiency (throughput per MB)
        ax4 = axes[1, 1]
        for bs in batch_sizes:
            efficiency = []
            seq_lens = []
            for seq_len in seq_lengths:
                config_name = f"bs{bs}_seq{seq_len}"
                if (config_name in results['throughput'] and 
                    config_name in results['memory_usage'] and
                    results['throughput'][config_name] and 
                    results['memory_usage'][config_name]):
                    
                    throughput = results['throughput'][config_name][0]
                    memory_mb = np.mean(results['memory_usage'][config_name]) / (1024 * 1024)
                    if memory_mb > 0:
                        efficiency.append(throughput / memory_mb)
                        seq_lens.append(seq_len)
            
            if efficiency:
                ax4.plot(seq_lens, efficiency, marker='o', label=f'Batch Size {bs}')
        
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Efficiency (tokens/second/MB)')
        ax4.set_title('Memory Efficiency vs Sequence Length')
        ax4.legend()
        ax4.set_xscale('log', base=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Benchmark plots saved to {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()


class MemoryProfiler:
    """
    Detailed memory profiling for reversible transformers.
    
    Tracks memory usage throughout training and inference,
    identifying memory hotspots and optimization opportunities.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.memory_snapshots = []
        self.hooks = []
    
    def profile_memory_usage(
        self,
        input_data: torch.Tensor,
        num_steps: int = 10,
        include_backward: bool = True,
    ) -> Dict[str, Any]:
        """
        Profile memory usage during model execution.
        
        Args:
            input_data: Input tensor for profiling
            num_steps: Number of steps to profile
            include_backward: Whether to include backward pass
            
        Returns:
            Memory profiling results
        """
        self.memory_snapshots = []
        
        # Register memory tracking hooks
        self._register_memory_hooks()
        
        device = next(self.model.parameters()).device
        input_data = input_data.to(device)
        
        for step in range(num_steps):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Take initial snapshot
            self._take_memory_snapshot(f"step_{step}_start")
            
            # Forward pass
            outputs = self.model(input_data)
            self._take_memory_snapshot(f"step_{step}_forward")
            
            if include_backward:
                # Backward pass
                if isinstance(outputs, dict):
                    loss = outputs.get('logits', outputs['last_hidden_state']).sum()
                else:
                    loss = outputs.sum()
                
                loss.backward()
                self._take_memory_snapshot(f"step_{step}_backward")
                
                # Clear gradients
                self.model.zero_grad()
                self._take_memory_snapshot(f"step_{step}_cleanup")
        
        # Remove hooks
        self._remove_memory_hooks()
        
        return self._analyze_memory_snapshots()
    
    def _register_memory_hooks(self):
        """Register hooks to track memory usage per layer."""
        def memory_hook(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated()
                    self.memory_snapshots.append({
                        'layer': name,
                        'memory': current_memory,
                        'timestamp': time.time(),
                    })
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                handle = module.register_forward_hook(memory_hook(name))
                self.hooks.append(handle)
    
    def _remove_memory_hooks(self):
        """Remove memory tracking hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _take_memory_snapshot(self, label: str):
        """Take a memory snapshot with label."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            
            self.memory_snapshots.append({
                'label': label,
                'memory': current_memory,
                'max_memory': max_memory,
                'reserved_memory': reserved_memory,
                'timestamp': time.time(),
            })
    
    def _analyze_memory_snapshots(self) -> Dict[str, Any]:
        """Analyze collected memory snapshots."""
        if not self.memory_snapshots:
            return {'error': 'No memory snapshots collected'}
        
        # Extract memory usage patterns
        memory_timeline = []
        peak_memory = 0
        memory_efficiency = []
        
        for snapshot in self.memory_snapshots:
            if 'memory' in snapshot:
                memory_mb = snapshot['memory'] / (1024 * 1024)
                memory_timeline.append({
                    'label': snapshot.get('label', snapshot.get('layer', 'unknown')),
                    'memory_mb': memory_mb,
                    'timestamp': snapshot['timestamp'],
                })
                peak_memory = max(peak_memory, memory_mb)
        
        # Calculate memory efficiency metrics
        if len(memory_timeline) > 1:
            memory_growth = memory_timeline[-1]['memory_mb'] - memory_timeline[0]['memory_mb']
            avg_memory = np.mean([m['memory_mb'] for m in memory_timeline])
        else:
            memory_growth = 0
            avg_memory = memory_timeline[0]['memory_mb'] if memory_timeline else 0
        
        return {
            'peak_memory_mb': peak_memory,
            'average_memory_mb': avg_memory,
            'memory_growth_mb': memory_growth,
            'memory_timeline': memory_timeline,
            'total_snapshots': len(self.memory_snapshots),
        }


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for reversible transformers.
    
    Combines performance benchmarking, memory profiling, and
    comparison tools in a unified interface.
    """
    
    def __init__(self):
        self.performance_benchmark = PerformanceBenchmark()
        self.results_cache = {}
    
    def run_comprehensive_benchmark(
        self,
        models: Dict[str, nn.Module],
        sequence_lengths: List[int] = [1024, 4096, 16384],
        batch_sizes: List[int] = [1, 2, 4],
        metrics: List[str] = ['memory', 'throughput', 'latency'],
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across multiple models and configurations.
        
        Args:
            models: Dictionary of model name -> model instance
            sequence_lengths: List of sequence lengths to test
            batch_sizes: List of batch sizes to test
            metrics: List of metrics to measure
            
        Returns:
            Comprehensive benchmark results
        """
        results = {
            'models': {},
            'comparisons': {},
            'summary': {},
        }
        
        # Benchmark each model
        for model_name, model in models.items():
            print(f"\n🔍 Benchmarking {model_name}...")
            
            model_results = self.performance_benchmark.benchmark_model(
                model=model,
                sequence_lengths=sequence_lengths,
                batch_sizes=batch_sizes,
                measure_memory='memory' in metrics,
                measure_backward='latency' in metrics,
            )
            
            results['models'][model_name] = model_results
            
            # Memory profiling if requested
            if 'memory' in metrics:
                print(f"  📊 Memory profiling {model_name}...")
                memory_profiler = MemoryProfiler(model)
                
                # Use a representative configuration for memory profiling
                test_input = torch.randint(
                    0, getattr(model, 'vocab_size', 50257),
                    (2, 2048),  # batch_size=2, seq_len=2048
                    device=next(model.parameters()).device
                )
                
                memory_results = memory_profiler.profile_memory_usage(
                    input_data=test_input,
                    num_steps=3,
                )
                
                results['models'][model_name]['detailed_memory'] = memory_results
        
        # Generate comparisons
        results['comparisons'] = self._generate_comparisons(results['models'])
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_comparisons(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison metrics between models."""
        comparisons = {}
        
        if len(model_results) < 2:
            return comparisons
        
        model_names = list(model_results.keys())
        base_model = model_names[0]  # Use first model as baseline
        
        for model_name in model_names[1:]:
            comparison_key = f"{model_name}_vs_{base_model}"
            comparisons[comparison_key] = {}
            
            # Compare performance metrics
            base_results = model_results[base_model]
            current_results = model_results[model_name]
            
            # Find common configurations
            base_configs = set(base_results['forward_times'].keys())
            current_configs = set(current_results['forward_times'].keys())
            common_configs = base_configs & current_configs
            
            for config in common_configs:
                base_time = np.mean(base_results['forward_times'][config])
                current_time = np.mean(current_results['forward_times'][config])
                
                speedup = base_time / current_time if current_time > 0 else 0
                
                base_memory = (np.mean(base_results['memory_usage'][config]) 
                              if base_results['memory_usage'][config] else 0)
                current_memory = (np.mean(current_results['memory_usage'][config]) 
                                 if current_results['memory_usage'][config] else 0)
                
                memory_ratio = current_memory / base_memory if base_memory > 0 else 1
                
                comparisons[comparison_key][config] = {
                    'speedup': speedup,
                    'memory_ratio': memory_ratio,
                    'base_time': base_time,
                    'current_time': current_time,
                    'base_memory_mb': base_memory / (1024 * 1024),
                    'current_memory_mb': current_memory / (1024 * 1024),
                }
        
        return comparisons
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            'total_models': len(results['models']),
            'best_performance': {},
            'recommendations': [],
        }
        
        # Find best performing model for each metric
        model_names = list(results['models'].keys())
        
        if len(model_names) > 1:
            # Compare average performance across all configurations
            avg_performances = {}
            
            for model_name in model_names:
                model_results = results['models'][model_name]
                
                # Calculate average forward time
                all_forward_times = []
                all_memory_usage = []
                
                for config_times in model_results['forward_times'].values():
                    all_forward_times.extend(config_times)
                
                for config_memory in model_results['memory_usage'].values():
                    if config_memory:
                        all_memory_usage.extend(config_memory)
                
                avg_performances[model_name] = {
                    'avg_forward_time': np.mean(all_forward_times) if all_forward_times else float('inf'),
                    'avg_memory_usage': np.mean(all_memory_usage) if all_memory_usage else float('inf'),
                }
            
            # Find best models
            best_speed = min(avg_performances.items(), key=lambda x: x[1]['avg_forward_time'])
            best_memory = min(avg_performances.items(), key=lambda x: x[1]['avg_memory_usage'])
            
            summary['best_performance'] = {
                'fastest_model': best_speed[0],
                'most_memory_efficient': best_memory[0],
            }
            
            # Generate recommendations
            if best_speed[0] == best_memory[0]:
                summary['recommendations'].append(
                    f"{best_speed[0]} provides the best balance of speed and memory efficiency"
                )
            else:
                summary['recommendations'].extend([
                    f"Use {best_speed[0]} for maximum speed",
                    f"Use {best_memory[0]} for maximum memory efficiency"
                ])
        
        return summary
    
    def plot_comparison(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_plots: bool = True,
    ):
        """Plot comparison results across models."""
        if 'models' not in results or len(results['models']) < 2:
            print("Need at least 2 models for comparison plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RevNet-Zero Model Comparison', fontsize=16)
        
        model_names = list(results['models'].keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        # Extract common sequence lengths for comparison
        all_configs = set()
        for model_results in results['models'].values():
            all_configs.update(model_results['forward_times'].keys())
        
        # Parse sequence lengths from config names
        seq_lengths = sorted(list(set(
            int(config.split('_seq')[1]) for config in all_configs 
            if '_seq' in config
        )))
        
        # Forward time comparison
        ax1 = axes[0, 0]
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            model_results = results['models'][model_name]
            times = []
            seq_lens = []
            
            for seq_len in seq_lengths:
                # Find config with this sequence length (assuming batch_size=1)
                config_name = f"bs1_seq{seq_len}"
                if config_name in model_results['forward_times']:
                    times.append(np.mean(model_results['forward_times'][config_name]))
                    seq_lens.append(seq_len)
            
            if times:
                ax1.plot(seq_lens, times, marker='o', color=color, label=model_name)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Forward Time (seconds)')
        ax1.set_title('Forward Pass Time Comparison')
        ax1.legend()
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        
        # Memory usage comparison
        ax2 = axes[0, 1]
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            model_results = results['models'][model_name]
            memory_usage = []
            seq_lens = []
            
            for seq_len in seq_lengths:
                config_name = f"bs1_seq{seq_len}"
                if (config_name in model_results['memory_usage'] and 
                    model_results['memory_usage'][config_name]):
                    
                    memory_mb = np.mean(model_results['memory_usage'][config_name]) / (1024 * 1024)
                    memory_usage.append(memory_mb)
                    seq_lens.append(seq_len)
            
            if memory_usage:
                ax2.plot(seq_lens, memory_usage, marker='o', color=color, label=model_name)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Comparison')
        ax2.legend()
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        
        # Throughput comparison
        ax3 = axes[1, 0]
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            model_results = results['models'][model_name]
            throughputs = []
            seq_lens = []
            
            for seq_len in seq_lengths:
                config_name = f"bs1_seq{seq_len}"
                if (config_name in model_results['throughput'] and 
                    model_results['throughput'][config_name]):
                    
                    throughputs.append(model_results['throughput'][config_name][0])
                    seq_lens.append(seq_len)
            
            if throughputs:
                ax3.plot(seq_lens, throughputs, marker='o', color=color, label=model_name)
        
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Throughput (tokens/second)')
        ax3.set_title('Throughput Comparison')
        ax3.legend()
        ax3.set_xscale('log', base=2)
        
        # Parameter count comparison (bar chart)
        ax4 = axes[1, 1]
        param_counts = []
        names = []
        
        for model_name in model_names:
            model_info = results['models'][model_name]['model_info']
            param_counts.append(model_info['total_parameters'] / 1e6)  # Convert to millions
            names.append(model_name)
        
        bars = ax4.bar(names, param_counts, color=colors[:len(names)])
        ax4.set_ylabel('Parameters (Millions)')
        ax4.set_title('Model Parameter Count')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, param_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count:.1f}M', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved to {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()