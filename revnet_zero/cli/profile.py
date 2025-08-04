"""
Memory profiling CLI for RevNet-Zero.

Command-line interface for profiling memory usage of reversible transformer models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from ..memory.profiler import DetailedMemoryProfiler
    from .. import ReversibleTransformer
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from revnet_zero.memory.profiler import DetailedMemoryProfiler
    from revnet_zero import ReversibleTransformer


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Profile memory usage of RevNet-Zero models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  revnet-zero-profile --model-config config.json --batch-size 4 --seq-length 1024
  revnet-zero-profile --preset medium --profile-layers --save-trace
        """,
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--preset",
        choices=["tiny", "small", "medium", "large"],
        help="Use predefined model configuration"
    )
    model_group.add_argument(
        "--model-config",
        type=str,
        help="Path to model configuration JSON file"
    )
    
    # Manual model parameters
    manual_group = parser.add_argument_group("Manual Model Parameters")
    manual_group.add_argument("--vocab-size", type=int, default=1000)
    manual_group.add_argument("--num-layers", type=int, default=6)
    manual_group.add_argument("--d-model", type=int, default=256)
    manual_group.add_argument("--num-heads", type=int, default=8)
    manual_group.add_argument("--d-ff", type=int, default=1024)
    
    # Input configuration
    input_group = parser.add_argument_group("Input Configuration")
    input_group.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for profiling (default: 2)"
    )
    input_group.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for profiling (default: 512)"
    )
    
    # Profiling options
    profile_group = parser.add_argument_group("Profiling Options")
    profile_group.add_argument(
        "--profile-layers",
        action="store_true",
        help="Profile individual layers"
    )
    profile_group.add_argument(
        "--compare-modes",
        action="store_true",
        help="Compare reversible vs standard modes"
    )
    profile_group.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of profiling iterations (default: 5)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./profiling_results",
        help="Output directory for results (default: ./profiling_results)"
    )
    output_group.add_argument(
        "--save-trace",
        action="store_true",
        help="Save Chrome trace file for visualization"
    )
    output_group.add_argument(
        "--save-plots",
        action="store_true",
        help="Save memory timeline plots"
    )
    output_group.add_argument(
        "--json-report",
        type=str,
        help="Save detailed report to JSON file"
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
        help="Device to use for profiling (default: auto)"
    )
    
    return parser


def get_model_config_from_preset(preset: str) -> Dict[str, Any]:
    """Get model configuration from preset."""
    presets = {
        "tiny": {
            "vocab_size": 1000,
            "num_layers": 2,
            "d_model": 64,
            "num_heads": 2,
            "d_ff": 128,
        },
        "small": {
            "vocab_size": 1000,
            "num_layers": 4,
            "d_model": 128,
            "num_heads": 4,
            "d_ff": 512,
        },
        "medium": {
            "vocab_size": 1000,
            "num_layers": 6,
            "d_model": 256,
            "num_heads": 8,
            "d_ff": 1024,
        },
        "large": {
            "vocab_size": 1000,
            "num_layers": 12,
            "d_model": 512,
            "num_heads": 16,
            "d_ff": 2048,
        },
    }
    return presets[preset]


def get_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Get model configuration from arguments."""
    if args.preset:
        return get_model_config_from_preset(args.preset)
    elif args.model_config:
        with open(args.model_config, 'r') as f:
            return json.load(f)
    else:
        return {
            "vocab_size": args.vocab_size,
            "num_layers": args.num_layers,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
        }


def create_model(config: Dict[str, Any], device: str) -> ReversibleTransformer:
    """Create model from configuration."""
    import torch
    
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    model = ReversibleTransformer(**config).to(device)
    return model


def run_profiling(args: argparse.Namespace) -> Dict[str, Any]:
    """Run memory profiling based on CLI arguments."""
    print("🔍 RevNet-Zero Memory Profiler")
    print("=" * 50)
    
    # Get model configuration
    config = get_model_config(args)
    print(f"Model config: {config}")
    
    # Create model
    model = create_model(config, args.device)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {model.device if hasattr(model, 'device') else 'unknown'}")
    
    # Initialize profiler
    output_dir = Path(args.output_dir)
    profiler = DetailedMemoryProfiler(output_dir=output_dir)
    
    # Create sample input
    import torch
    device = next(model.parameters()).device
    sample_input = torch.randint(
        0, config["vocab_size"], 
        (args.batch_size, args.seq_length), 
        device=device
    )
    
    print(f"Sample input shape: {sample_input.shape}")
    
    results = {}
    
    try:
        # Basic profiling
        print("\n1. Running basic memory profiling...")
        profiler.start_profiling()
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(sample_input)
        
        # Training simulation
        model.train()
        for i in range(args.iterations):
            model.zero_grad()
            outputs = model(sample_input)
            loss = outputs["logits"].sum() if isinstance(outputs, dict) else outputs.sum()
            loss.backward()
            
            if args.verbose:
                print(f"  Iteration {i+1}/{args.iterations}")
        
        profiler.stop_profiling()
        
        # Generate report
        report = profiler.generate_report()
        results["basic_profiling"] = report
        
        print(f"✓ Basic profiling completed")
        print(f"  Peak memory: {report['summary']['peak_memory'] / 1e9:.2f}GB")
        print(f"  Memory overhead: {report['summary']['memory_overhead'] / 1e9:.2f}GB")
        
        # Layer-by-layer profiling
        if args.profile_layers:
            print("\n2. Running layer-by-layer profiling...")
            layer_results = []
            
            layer_id = 0
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf modules
                    try:
                        profile = profiler.profile_layer(layer_id, name, module, sample_input)
                        layer_results.append({
                            "layer_id": layer_id,
                            "name": name,
                            "forward_memory": profile.forward_memory,
                            "backward_memory": profile.backward_memory,
                            "peak_memory": profile.peak_memory,
                        })
                        layer_id += 1
                        
                        if args.verbose:
                            print(f"  {name}: {profile.peak_memory / 1e6:.1f}MB")
                            
                    except Exception as e:
                        if args.verbose:
                            print(f"  Skipped {name}: {e}")
            
            results["layer_profiling"] = layer_results
            print(f"✓ Profiled {len(layer_results)} layers")
        
        # Mode comparison
        if args.compare_modes:
            print("\n3. Comparing reversible vs standard modes...")
            
            comparison_results = {}
            
            # Test reversible mode
            model.set_reversible_mode(True)
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            model.train()
            for _ in range(3):
                model.zero_grad()
                outputs = model(sample_input)
                loss = outputs["logits"].sum() if isinstance(outputs, dict) else outputs.sum()
                loss.backward()
            
            if hasattr(torch.cuda, 'max_memory_allocated'):
                reversible_memory = torch.cuda.max_memory_allocated()
            else:
                reversible_memory = 0
            
            # Test standard mode
            model.set_reversible_mode(False)
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
                
            for _ in range(3):
                model.zero_grad()
                outputs = model(sample_input)
                loss = outputs["logits"].sum() if isinstance(outputs, dict) else outputs.sum()
                loss.backward()
            
            if hasattr(torch.cuda, 'max_memory_allocated'):
                standard_memory = torch.cuda.max_memory_allocated()
            else:
                standard_memory = 0
            
            if standard_memory > 0 and reversible_memory > 0:
                memory_reduction = (standard_memory - reversible_memory) / standard_memory * 100
                
                comparison_results = {
                    "reversible_memory": reversible_memory,
                    "standard_memory": standard_memory,
                    "memory_saved": standard_memory - reversible_memory,
                    "reduction_percentage": memory_reduction,
                }
                
                results["mode_comparison"] = comparison_results
                
                print(f"✓ Mode comparison completed")
                print(f"  Reversible: {reversible_memory / 1e9:.2f}GB")
                print(f"  Standard: {standard_memory / 1e9:.2f}GB")
                print(f"  Reduction: {memory_reduction:.1f}%")
            else:
                print("⚠️  Mode comparison only available on CUDA")
        
        # Save outputs
        if args.save_trace:
            trace_path = output_dir / "memory_trace.json"
            profiler.export_chrome_trace(trace_path)
            print(f"✓ Chrome trace saved to {trace_path}")
        
        if args.save_plots:
            plot_path = output_dir / "memory_timeline.png"
            profiler.plot_memory_timeline(plot_path)
            print(f"✓ Memory timeline plot saved to {plot_path}")
        
        if args.json_report:
            report_path = Path(args.json_report)
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✓ Detailed report saved to {report_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def profile_cli():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Arguments: {vars(args)}")
    
    try:
        results = run_profiling(args)
        
        if "error" in results:
            sys.exit(1)
        else:
            print("\n✅ Profiling completed successfully!")
            
            # Print summary
            if "basic_profiling" in results:
                report = results["basic_profiling"]
                print(f"Peak memory usage: {report['summary']['peak_memory'] / 1e9:.2f}GB")
                
                if "recommendations" in report and report["recommendations"]:
                    print("\nRecommendations:")
                    for rec in report["recommendations"][:3]:  # Show top 3
                        print(f"  • {rec['description']}")
            
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n❌ Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def main():
    """Entry point for console script."""
    profile_cli()


if __name__ == "__main__":
    main()