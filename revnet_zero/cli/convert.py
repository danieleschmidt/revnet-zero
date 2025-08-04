"""
Model conversion CLI for RevNet-Zero.

Command-line interface for converting existing transformer models to reversible.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

try:
    from ..utils.conversion import convert_to_reversible
    from .. import ReversibleTransformer
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from revnet_zero.utils.conversion import convert_to_reversible
    from revnet_zero import ReversibleTransformer


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert existing transformer models to reversible",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  revnet-zero-convert --model-path model.pt --output-path reversible_model.pt
  revnet-zero-convert --huggingface-model gpt2 --verify --save-comparison
        """,
    )
    
    # Input model
    input_group = parser.add_argument_group("Input Model")
    input_group.add_argument(
        "--model-path",
        type=str,
        help="Path to PyTorch model file (.pt or .pth)"
    )
    input_group.add_argument(
        "--huggingface-model",
        type=str,
        help="HuggingFace model name or path"
    )
    input_group.add_argument(
        "--model-type",
        choices=["gpt2", "bert", "generic"],
        help="Model type (auto-detected if not specified)"
    )
    
    # Conversion options
    conv_group = parser.add_argument_group("Conversion Options")
    conv_group.add_argument(
        "--coupling",
        choices=["additive", "affine", "learned"],
        default="additive",
        help="Coupling function type (default: additive)"
    )
    conv_group.add_argument(
        "--preserve-weights",
        action="store_true",
        default=True,
        help="Preserve original model weights (default: True)"
    )
    conv_group.add_argument(
        "--no-preserve-weights",
        action="store_true",
        help="Don't preserve original weights"
    )
    conv_group.add_argument(
        "--checkpoint-segments",
        type=int,
        default=4,
        help="Number of gradient checkpointing segments (default: 4)"
    )
    
    # Verification options
    verify_group = parser.add_argument_group("Verification Options")
    verify_group.add_argument(
        "--verify",
        action="store_true",
        help="Verify output equivalence after conversion"
    )
    verify_group.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for verification (default: 1e-5)"
    )
    verify_group.add_argument(
        "--test-inputs",
        type=str,
        help="Path to test inputs for verification"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-path",
        "-o",
        type=str,
        required=True,
        help="Output path for converted model"
    )
    output_group.add_argument(
        "--save-config",
        type=str,
        help="Save model configuration to JSON file"
    )
    output_group.add_argument(
        "--save-comparison",
        type=str,
        help="Save conversion comparison report"
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
        help="Device to use for conversion (default: auto)"
    )
    
    return parser


def load_model(args: argparse.Namespace):
    """Load the original model."""
    import torch
    
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        
        try:
            # Try loading as a state dict first
            state_dict = torch.load(args.model_path, map_location="cpu")
            
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                # Model checkpoint format
                state_dict = state_dict["state_dict"]
            
            # Would need to infer model architecture here
            # For now, return a placeholder
            print("⚠️  Loading from state dict not fully implemented")
            return None
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    elif args.huggingface_model:
        print(f"Loading HuggingFace model: {args.huggingface_model}")
        
        try:
            # Try importing transformers
            from transformers import AutoModel, AutoConfig
            
            config = AutoConfig.from_pretrained(args.huggingface_model)
            model = AutoModel.from_pretrained(args.huggingface_model)
            
            print(f"Loaded {config.model_type} model with {model.num_parameters():,} parameters")
            return model
            
        except ImportError:
            print("❌ transformers library not installed. Install with: pip install transformers")
            return None
        except Exception as e:
            print(f"❌ Failed to load HuggingFace model: {e}")
            return None
    
    else:
        print("❌ Must specify either --model-path or --huggingface-model")
        return None


def perform_conversion(model, args: argparse.Namespace):
    """Perform the model conversion."""
    print(f"\nConverting model to reversible...")
    print(f"Coupling type: {args.coupling}")
    print(f"Preserve weights: {not args.no_preserve_weights}")
    
    try:
        reversible_model = convert_to_reversible(
            model=model,
            coupling=args.coupling,
            checkpoint_segments=args.checkpoint_segments,
            preserve_weights=not args.no_preserve_weights,
            verify_equivalence=args.verify,
        )
        
        print("✓ Model conversion completed")
        
        # Print model info
        if hasattr(reversible_model, 'get_model_info'):
            info = reversible_model.get_model_info()
            print(f"  Parameters: {info['total_parameters']:,}")
            print(f"  Model type: {info['model_type']}")
        
        return reversible_model
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_conversion(original_model, reversible_model, args: argparse.Namespace):
    """Verify the conversion results."""
    import torch
    
    print(f"\nVerifying conversion...")
    
    try:
        # Create test input
        vocab_size = getattr(reversible_model, 'vocab_size', 1000)
        test_input = torch.randint(0, vocab_size, (2, 32))
        
        # Get outputs from both models
        original_model.eval()
        reversible_model.eval()
        
        with torch.no_grad():
            original_output = original_model(test_input)
            reversible_output = reversible_model(test_input, use_reversible=False)
        
        # Extract logits for comparison
        if hasattr(original_output, 'logits'):
            orig_logits = original_output.logits
        elif isinstance(original_output, dict) and 'logits' in original_output:
            orig_logits = original_output['logits']
        else:
            orig_logits = original_output[0] if isinstance(original_output, tuple) else original_output
        
        if isinstance(reversible_output, dict) and 'logits' in reversible_output:
            rev_logits = reversible_output['logits']
        else:
            rev_logits = reversible_output[0] if isinstance(reversible_output, tuple) else reversible_output
        
        # Calculate difference
        max_diff = torch.max(torch.abs(orig_logits - rev_logits)).item()
        
        print(f"Maximum output difference: {max_diff:.2e}")
        
        if max_diff <= args.tolerance:
            print("✓ Verification passed")
            return True
        else:
            print(f"⚠️  Verification failed (tolerance: {args.tolerance:.2e})")
            return False
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def save_model(model, output_path: str, args: argparse.Namespace):
    """Save the converted model."""
    import torch
    import json
    
    print(f"\nSaving converted model to {output_path}")
    
    try:
        # Save model
        torch.save(model.state_dict(), output_path)
        print("✓ Model saved")
        
        # Save configuration if requested
        if args.save_config:
            if hasattr(model, 'get_model_info'):
                config = model.get_model_info()
                with open(args.save_config, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✓ Configuration saved to {args.save_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False


def run_conversion(args: argparse.Namespace) -> Dict[str, Any]:
    """Run model conversion based on CLI arguments."""
    print("🔄 RevNet-Zero Model Converter")
    print("=" * 50)
    
    results = {"success": False}
    
    # Load original model
    original_model = load_model(args)
    if original_model is None:
        return results
    
    # Perform conversion
    reversible_model = perform_conversion(original_model, args)
    if reversible_model is None:
        return results
    
    # Verify conversion if requested
    verification_passed = True
    if args.verify:
        verification_passed = verify_conversion(original_model, reversible_model, args)
        results["verification_passed"] = verification_passed
    
    # Save converted model
    save_success = save_model(reversible_model, args.output_path, args)
    if not save_success:
        return results
    
    results["success"] = True
    results["output_path"] = args.output_path
    
    if hasattr(reversible_model, 'get_model_info'):
        results["model_info"] = reversible_model.get_model_info()
    
    return results


def convert_cli():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Arguments: {vars(args)}")
    
    # Handle conflicting arguments
    if args.no_preserve_weights:
        args.preserve_weights = False
    
    try:
        results = run_conversion(args)
        
        if not results["success"]:
            sys.exit(1)
        else:
            print("\n✅ Model conversion completed successfully!")
            print(f"Converted model saved to: {results['output_path']}")
            
            if "verification_passed" in results:
                if results["verification_passed"]:
                    print("✓ Output verification passed")
                else:
                    print("⚠️  Output verification failed")
            
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n❌ Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def main():
    """Entry point for console script."""
    convert_cli()


if __name__ == "__main__":
    main()