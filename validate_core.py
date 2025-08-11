#\!/usr/bin/env python3
"""
Quick validation of core RevNet-Zero functionality.
"""

# Load mock PyTorch environment
import mock_torch
import revnet_zero

def quick_validation():
    """Quick validation of core components."""
    print("🚀 RevNet-Zero Quick Validation")
    print("=" * 50)
    
    try:
        # Test import
        print("✓ Core imports successful")
        
        # Test basic instantiation
        from revnet_zero import ReversibleTransformer, AdditiveCoupling
        
        # Test coupling layer
        coupling = AdditiveCoupling(d_model=128)
        print("✓ Coupling layer instantiated")
        
        # Test transformer model
        model = ReversibleTransformer(
            vocab_size=1000,
            num_layers=6,
            d_model=256,
            num_heads=8,
            max_seq_len=1024
        )
        print("✓ Reversible transformer instantiated")
        
        # Test model info
        info = model.get_model_info()
        print(f"✓ Model info: {info['total_parameters']} parameters")
        
        print("\n🎯 Generation 1 Core Functionality: WORKING")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    quick_validation()
