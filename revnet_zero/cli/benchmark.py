"""
Benchmark CLI for RevNet-Zero.

Command-line interface for benchmarking reversible transformer models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from ..examples.benchmarking import BenchmarkSuite
    from .. import ReversibleTransformer
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from revnet_zero.examples.benchmarking import BenchmarkSuite
    from revnet_zero import ReversibleTransformer


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark RevNet-Zero reversible transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  revnet-zero-benchmark --quick
  revnet-zero-benchmark --model-size small --seq-lengths 512 1024 2048
  revnet-zero-benchmark --custom-config model_config.json --output results/
        """,
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-size", 
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Predefined model size (default: small)"
    )
    model_group.add_argument(
        "--custom-config",
        type=str,
        help="Path to custom model configuration JSON file"
    )
    
    # Benchmark parameters
    bench_group = parser.add_argument_group("Benchmark Parameters")
    bench_group.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Sequence lengths to test (default: 512 1024 2048)"
    )
    bench_group.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+", 
        default=[1, 2, 4],
        help="Batch sizes to test (default: 1 2 4)"
    )
    bench_group.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)"
    )
    bench_group.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)"
    )
    
    # Test modes
    mode_group = parser.add_argument_group("Test Modes")
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with minimal configurations"
    )
    mode_group.add_argument(
        "--memory-only",
        action="store_true",
        help="Only test memory usage (faster)"
    )
    mode_group.add_argument(
        "--no-reversible",
        action="store_true",
        help="Skip reversible mode testing"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        type=str,
        default="./benchmark_results",
        help="Output directory for results (default: ./benchmark_results)"
    )
    output_group.add_argument(
        "--save-plots",
        action="store_true",
        help="Save benchmark plots"
    )
    output_group.add_argument(
        "--json-output",
        type=str,
        help="Save results to JSON file"
    )
    output_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    # Device selection
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for benchmarking (default: auto)"
    )
    
    return parser


def get_model_config(size: str) -> Dict[str, Any]:
    """Get predefined model configuration."""
    configs = {
        "tiny": {
            "name": "tiny",
            "vocab_size": 1000,
            "num_layers": 2,
            "d_model": 64,
            "num_heads": 2,
            "d_ff": 128,
        },
        "small": {
            "name": "small", 
            "vocab_size": 1000,
            "num_layers": 4,
            "d_model": 128,
            "num_heads": 4,
            "d_ff": 512,
        },
        "medium": {
            "name": "medium",
            "vocab_size": 1000,
            "num_layers": 8,
            "d_model": 256,
            "num_heads": 8,
            "d_ff": 1024,
        },
        "large": {
            "name": "large",
            "vocab_size": 1000,
            "num_layers": 12,
            "d_model": 512,
            "num_heads": 16,
            "d_ff": 2048,
        },
    }
    return configs[size]


def load_custom_config(config_path: str) -> Dict[str, Any]:
    """Load custom model configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ["name", "vocab_size", "num_layers", "d_model", "num_heads", "d_ff"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file")
    
    return config


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run benchmark based on CLI arguments."""
    print("🚀 RevNet-Zero Benchmark")
    print("=" * 50)
    
    # Initialize benchmark suite
    benchmark_suite = BenchmarkSuite(device=args.device, output_dir=args.output)
    
    # Get model configurations
    if args.custom_config:
        model_configs = [load_custom_config(args.custom_config)]
    else:
        model_configs = [get_model_config(args.model_size)]
    
    # Adjust parameters for quick mode
    if args.quick:
        args.seq_lengths = [256, 512]
        args.batch_sizes = [1, 2]
        args.iterations = 5
        args.warmup = 2
        print("Quick mode: Using reduced test parameters")
    
    print(f"Testing configurations: {[c['name'] for c in model_configs]}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Device: {benchmark_suite.device}")
    
    # Run benchmark
    try:
        results = benchmark_suite.scaling_benchmark(
            model_configs=model_configs,
            sequence_lengths=args.seq_lengths,
            batch_sizes=args.batch_sizes,
            test_reversible=not args.no_reversible,
        )
        
        if not results:
            print("❌ No benchmark results obtained")
            return {"error": "No results"}
        
        print(f"✓ Completed {len(results)} benchmark runs")
        
        # Generate report
        report = benchmark_suite.generate_report(results)
        print("\n" + report)
        
        # Save results
        if args.json_output:
            benchmark_suite.save_results(results, args.json_output)
        else:
            benchmark_suite.save_results(results)
        
        # Generate plots
        if args.save_plots:
            try:
                benchmark_suite.plot_scaling_results(results, save_plots=True)
                print("✓ Plots saved")
            except Exception as e:
                print(f"⚠️  Failed to generate plots: {e}")
        
        # Memory analysis
        memory_analysis = benchmark_suite.memory_efficiency_analysis(results)
        
        return {
            "results": results,
            "memory_analysis": memory_analysis,
            "report": report,
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def benchmark_cli():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Arguments: {vars(args)}")
    
    try:
        results = run_benchmark(args)
        
        if "error" in results:
            sys.exit(1)
        else:
            print("\n✅ Benchmark completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def main():
    """Entry point for console script."""
    benchmark_cli()


if __name__ == "__main__":
    main()