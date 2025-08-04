"""
Comprehensive benchmarking utilities for RevNet-Zero.

This module provides tools for benchmarking memory usage, throughput,
and scaling behavior of reversible transformers.
"""

import torch
import torch.nn as nn
import time
import gc
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

try:
    from ..revnet_zero import ReversibleTransformer, MemoryScheduler
except ImportError:
    import sys
    sys.path.append('..')
    from revnet_zero import ReversibleTransformer, MemoryScheduler


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    model_name: str
    batch_size: int
    seq_length: int
    memory_usage: int
    peak_memory: int
    forward_time: float
    backward_time: float
    total_time: float
    throughput: float  # tokens/second
    memory_per_token: float
    reversible_mode: bool
    

class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for reversible transformers.
    
    Measures memory usage, throughput, and scaling behavior across
    different model configurations and sequence lengths.
    """
    
    def __init__(self, device: str = "auto", output_dir: str = "./benchmark_results"):
        """
        Initialize benchmark suite.
        
        Args:
            device: Device to use for benchmarking
            output_dir: Directory to save results
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Benchmarking on device: {self.device}")
    
    def benchmark_single_config(
        self,
        model: nn.Module,
        model_name: str,
        batch_size: int,
        seq_length: int,
        reversible_mode: bool = True,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark a single model configuration.
        
        Args:
            model: Model to benchmark
            model_name: Name of the model
            batch_size: Batch size
            seq_length: Sequence length
            reversible_mode: Whether to use reversible computation
            num_warmup: Number of warmup iterations
            num_iterations: Number of measurement iterations
            
        Returns:
            Benchmark results
        """
        model.eval()
        
        # Set reversible mode if supported
        if hasattr(model, 'set_reversible_mode'):
            model.set_reversible_mode(reversible_mode)
        
        # Create input data
        input_ids = torch.randint(0, model.vocab_size if hasattr(model, 'vocab_size') else 1000, 
                                 (batch_size, seq_length), device=self.device)
        
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(input_ids)
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Measure forward pass
        forward_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            
            outputs = model(input_ids)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            forward_times.append(time.time() - start_time)
        
        # Measure memory after forward pass
        if self.device.type == "cuda":
            memory_usage = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            memory_usage = peak_memory = 0
        
        # Measure backward pass (for training scenario)
        backward_times = []
        for _ in range(num_iterations):
            model.zero_grad()
            outputs = model(input_ids)
            loss = outputs.sum() if isinstance(outputs, torch.Tensor) else outputs["logits"].sum()
            
            start_time = time.time()
            loss.backward()
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                
            backward_times.append(time.time() - start_time)
        
        # Calculate statistics
        avg_forward_time = np.mean(forward_times)
        avg_backward_time = np.mean(backward_times)
        total_time = avg_forward_time + avg_backward_time
        
        total_tokens = batch_size * seq_length
        throughput = total_tokens / total_time
        memory_per_token = memory_usage / total_tokens if total_tokens > 0 else 0
        
        return BenchmarkResult(
            model_name=model_name,
            batch_size=batch_size,
            seq_length=seq_length,
            memory_usage=memory_usage,
            peak_memory=peak_memory,
            forward_time=avg_forward_time,
            backward_time=avg_backward_time,
            total_time=total_time,
            throughput=throughput,
            memory_per_token=memory_per_token,
            reversible_mode=reversible_mode,
        )
    
    def scaling_benchmark(
        self,
        model_configs: List[Dict[str, Any]],
        sequence_lengths: List[int] = [512, 1024, 2048, 4096, 8192],
        batch_sizes: List[int] = [1, 2, 4, 8],
        test_reversible: bool = True,
    ) -> List[BenchmarkResult]:
        """
        Run scaling benchmark across different configurations.
        
        Args:
            model_configs: List of model configurations to test
            sequence_lengths: Sequence lengths to test
            batch_sizes: Batch sizes to test
            test_reversible: Whether to test reversible mode
            
        Returns:
            List of benchmark results
        """
        results = []
        
        print(f"Running scaling benchmark with {len(model_configs)} models, "
              f"{len(sequence_lengths)} sequence lengths, {len(batch_sizes)} batch sizes")
        
        for config_idx, config in enumerate(model_configs):
            print(f"\nTesting model {config_idx + 1}/{len(model_configs)}: {config['name']}")
            
            # Create model
            model_config = config.copy()
            model_name = model_config.pop('name')
            
            try:
                model = ReversibleTransformer(**model_config).to(self.device)
            except Exception as e:
                print(f"Failed to create model {model_name}: {e}")
                continue
            
            for seq_len in sequence_lengths:
                for batch_size in batch_sizes:
                    # Skip configurations that are too large
                    total_tokens = batch_size * seq_len
                    if total_tokens > 32768:  # Skip very large configs to avoid OOM
                        continue
                    
                    print(f"  Testing batch_size={batch_size}, seq_len={seq_len}")
                    
                    try:
                        # Test standard mode
                        result_standard = self.benchmark_single_config(
                            model, f"{model_name}_standard", batch_size, seq_len, 
                            reversible_mode=False
                        )
                        results.append(result_standard)
                        
                        # Test reversible mode if requested
                        if test_reversible:
                            result_reversible = self.benchmark_single_config(
                                model, f"{model_name}_reversible", batch_size, seq_len,
                                reversible_mode=True
                            )
                            results.append(result_reversible)
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"    OOM for batch_size={batch_size}, seq_len={seq_len}")
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                        else:
                            print(f"    Error: {e}")
                    except Exception as e:
                        print(f"    Unexpected error: {e}")
        
        return results
    
    def memory_efficiency_analysis(
        self,
        results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """
        Analyze memory efficiency from benchmark results.
        
        Args:
            results: Benchmark results to analyze
            
        Returns:
            Memory efficiency analysis
        """
        # Group results by model and configuration
        grouped_results = {}
        for result in results:
            key = (result.batch_size, result.seq_length)
            if key not in grouped_results:
                grouped_results[key] = {"standard": None, "reversible": None}
            
            if result.reversible_mode:
                grouped_results[key]["reversible"] = result
            else:
                grouped_results[key]["standard"] = result
        
        # Calculate memory savings
        memory_savings = []
        throughput_comparison = []
        
        for (batch_size, seq_len), group in grouped_results.items():
            if group["standard"] and group["reversible"]:
                standard = group["standard"]
                reversible = group["reversible"]
                
                memory_saved = standard.memory_usage - reversible.memory_usage
                memory_reduction = memory_saved / standard.memory_usage * 100
                
                throughput_ratio = reversible.throughput / standard.throughput
                
                memory_savings.append({
                    "batch_size": batch_size,
                    "seq_length": seq_len,
                    "memory_saved": memory_saved,
                    "memory_reduction_percent": memory_reduction,
                    "throughput_ratio": throughput_ratio,
                })
        
        # Summary statistics
        if memory_savings:
            avg_memory_reduction = np.mean([s["memory_reduction_percent"] for s in memory_savings])
            avg_throughput_ratio = np.mean([s["throughput_ratio"] for s in memory_savings])
            
            analysis = {
                "average_memory_reduction": avg_memory_reduction,
                "average_throughput_ratio": avg_throughput_ratio,
                "memory_savings_breakdown": memory_savings,
                "total_configurations_tested": len(memory_savings),
            }
        else:
            analysis = {
                "error": "No comparable results found (need both standard and reversible modes)",
                "total_results": len(results),
            }
        
        return analysis
    
    def plot_scaling_results(
        self,
        results: List[BenchmarkResult],
        save_plots: bool = True
    ):
        """
        Plot scaling benchmark results.
        
        Args:
            results: Benchmark results to plot
            save_plots: Whether to save plots to disk
        """
        # Convert results to arrays for plotting
        data = {}
        for result in results:
            model_type = "reversible" if result.reversible_mode else "standard"
            key = f"{result.model_name}_{model_type}"
            
            if key not in data:
                data[key] = {
                    "seq_lengths": [],
                    "memory_usage": [],
                    "throughput": [],
                    "batch_sizes": [],
                }
            
            data[key]["seq_lengths"].append(result.seq_length)
            data[key]["memory_usage"].append(result.memory_usage / 1e9)  # GB
            data[key]["throughput"].append(result.throughput)
            data[key]["batch_sizes"].append(result.batch_size)
        
        # Plot 1: Memory usage vs sequence length
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for key, values in data.items():
            if values["seq_lengths"]:
                plt.plot(values["seq_lengths"], values["memory_usage"], 
                        marker='o', label=key)
        plt.xlabel("Sequence Length")
        plt.ylabel("Memory Usage (GB)")
        plt.title("Memory Usage vs Sequence Length")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Throughput vs sequence length
        plt.subplot(2, 2, 2)
        for key, values in data.items():
            if values["seq_lengths"]:
                plt.plot(values["seq_lengths"], values["throughput"], 
                        marker='s', label=key)
        plt.xlabel("Sequence Length")
        plt.ylabel("Throughput (tokens/sec)")
        plt.title("Throughput vs Sequence Length")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Memory efficiency (reversible vs standard)
        plt.subplot(2, 2, 3)
        reversible_data = [v for k, v in data.items() if "reversible" in k]
        standard_data = [v for k, v in data.items() if "standard" in k]
        
        if reversible_data and standard_data:
            for rev_data, std_data in zip(reversible_data, standard_data):
                if rev_data["seq_lengths"] and std_data["seq_lengths"]:
                    memory_reduction = [(s - r) / s * 100 
                                      for s, r in zip(std_data["memory_usage"], 
                                                     rev_data["memory_usage"])]
                    plt.plot(rev_data["seq_lengths"], memory_reduction, 
                            marker='^', label="Memory Reduction %")
        
        plt.xlabel("Sequence Length")
        plt.ylabel("Memory Reduction (%)")
        plt.title("Memory Reduction: Reversible vs Standard")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Memory per token
        plt.subplot(2, 2, 4)
        for key, values in data.items():
            if values["seq_lengths"] and values["batch_sizes"]:
                memory_per_token = [m / (s * b) for m, s, b in 
                                  zip(values["memory_usage"], 
                                      values["seq_lengths"], 
                                      values["batch_sizes"])]
                plt.plot(values["seq_lengths"], memory_per_token, 
                        marker='d', label=key)
        
        plt.xlabel("Sequence Length")
        plt.ylabel("Memory per Token (GB)")
        plt.title("Memory Efficiency per Token")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "scaling_benchmark.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {plot_path}")
        
        plt.show()
    
    def save_results(self, results: List[BenchmarkResult], filename: str = "benchmark_results.json"):
        """
        Save benchmark results to JSON file.
        
        Args:
            results: Results to save
            filename: Output filename
        """
        results_data = [asdict(result) for result in results]
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def load_results(self, filename: str = "benchmark_results.json") -> List[BenchmarkResult]:
        """
        Load benchmark results from JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded benchmark results
        """
        input_path = self.output_dir / filename
        
        with open(input_path, 'r') as f:
            results_data = json.load(f)
        
        results = [BenchmarkResult(**data) for data in results_data]
        print(f"Loaded {len(results)} results from {input_path}")
        
        return results
    
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            results: Benchmark results
            
        Returns:
            Formatted report string
        """
        if not results:
            return "No benchmark results to report."
        
        # Analyze results
        memory_analysis = self.memory_efficiency_analysis(results)
        
        # Group by model
        model_results = {}
        for result in results:
            model_name = result.model_name.split('_')[0]  # Remove _standard/_reversible suffix
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)
        
        report = []
        report.append("=" * 80)
        report.append("REVNET-ZERO BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total benchmark runs: {len(results)}")
        report.append(f"Models tested: {len(model_results)}")
        report.append(f"Device: {self.device}")
        report.append("")
        
        # Memory efficiency analysis
        if "average_memory_reduction" in memory_analysis:
            report.append("MEMORY EFFICIENCY")
            report.append("-" * 40)
            report.append(f"Average memory reduction: {memory_analysis['average_memory_reduction']:.1f}%")
            report.append(f"Average throughput ratio: {memory_analysis['average_throughput_ratio']:.3f}")
            report.append(f"Configurations tested: {memory_analysis['total_configurations_tested']}")
            report.append("")
        
        # Detailed results by model
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        
        for model_name, model_results_list in model_results.items():
            report.append(f"\nModel: {model_name}")
            report.append("  " + "-" * 35)
            
            # Best and worst performance
            best_throughput = max(model_results_list, key=lambda x: x.throughput)
            worst_memory = max(model_results_list, key=lambda x: x.memory_usage)
            
            report.append(f"  Best throughput: {best_throughput.throughput:.0f} tokens/sec "
                         f"(batch={best_throughput.batch_size}, seq={best_throughput.seq_length})")
            report.append(f"  Highest memory: {worst_memory.memory_usage/1e9:.2f}GB "
                         f"(batch={worst_memory.batch_size}, seq={worst_memory.seq_length})")
            
            # Memory savings summary for this model
            reversible_results = [r for r in model_results_list if r.reversible_mode]
            standard_results = [r for r in model_results_list if not r.reversible_mode]
            
            if reversible_results and standard_results:
                avg_rev_memory = np.mean([r.memory_usage for r in reversible_results])
                avg_std_memory = np.mean([r.memory_usage for r in standard_results])
                memory_reduction = (avg_std_memory - avg_rev_memory) / avg_std_memory * 100
                
                report.append(f"  Memory reduction (avg): {memory_reduction:.1f}%")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


class BenchmarkExample:
    """Example class demonstrating benchmarking functionality."""
    
    def __init__(self, device: str = "auto"):
        self.benchmark_suite = BenchmarkSuite(device=device)
    
    def run_quick_benchmark(self) -> List[BenchmarkResult]:
        """
        Run a quick benchmark with a few configurations.
        
        Returns:
            Benchmark results
        """
        print("🚀 Running Quick Benchmark")
        print("=" * 50)
        
        # Define test configurations
        model_configs = [
            {
                "name": "small_model",
                "vocab_size": 1000,
                "num_layers": 4,
                "d_model": 128,
                "num_heads": 4,
                "d_ff": 512,
            },
            # {
            #     "name": "medium_model",
            #     "vocab_size": 1000,
            #     "num_layers": 8,
            #     "d_model": 256,
            #     "num_heads": 8,
            #     "d_ff": 1024,
            # },
        ]
        
        # Run benchmark
        results = self.benchmark_suite.scaling_benchmark(
            model_configs=model_configs,
            sequence_lengths=[128, 256, 512],
            batch_sizes=[1, 2, 4],
            test_reversible=True,
        )
        
        # Generate and print report
        report = self.benchmark_suite.generate_report(results)
        print(report)
        
        # Save results
        self.benchmark_suite.save_results(results, "quick_benchmark.json")
        
        return results
    
    def run_memory_analysis(self) -> Dict[str, Any]:
        """
        Run memory analysis benchmark.
        
        Returns:
            Memory analysis results
        """
        print("🔍 Running Memory Analysis")
        print("=" * 50)
        
        # Create a single model for detailed analysis
        model = ReversibleTransformer(
            vocab_size=1000,
            num_layers=6,
            d_model=256,
            num_heads=8,
            d_ff=1024,
        ).to(self.benchmark_suite.device)
        
        results = []
        
        # Test different sequence lengths
        for seq_len in [256, 512, 1024, 2048]:
            try:
                print(f"Testing sequence length: {seq_len}")
                
                # Standard mode
                result_std = self.benchmark_suite.benchmark_single_config(
                    model, "memory_test", batch_size=2, seq_length=seq_len,
                    reversible_mode=False
                )
                results.append(result_std)
                
                # Reversible mode
                result_rev = self.benchmark_suite.benchmark_single_config(
                    model, "memory_test", batch_size=2, seq_length=seq_len,
                    reversible_mode=True
                )
                results.append(result_rev)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at sequence length {seq_len}")
                    break
        
        # Analyze results
        analysis = self.benchmark_suite.memory_efficiency_analysis(results)
        
        print("\nMEMORY ANALYSIS RESULTS")
        print("-" * 30)
        if "average_memory_reduction" in analysis:
            print(f"Average memory reduction: {analysis['average_memory_reduction']:.1f}%")
            print(f"Average throughput ratio: {analysis['average_throughput_ratio']:.3f}")
        
        return analysis


def main():
    """Run benchmark examples."""
    example = BenchmarkExample()
    
    # Run quick benchmark
    results = example.run_quick_benchmark()
    
    # Run memory analysis
    analysis = example.run_memory_analysis()
    
    return {"benchmark_results": results, "memory_analysis": analysis}


if __name__ == "__main__":
    main()