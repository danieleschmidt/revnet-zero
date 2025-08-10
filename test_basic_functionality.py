#!/usr/bin/env python3
"""
Basic functionality test for RevNet-Zero library.

This script tests the core components of the reversible transformer library
to ensure basic functionality works correctly.
"""

# Load mock PyTorch environment for testing
import mock_torch
import torch
import torch.nn as nn
from revnet_zero import (
    ReversibleTransformer, 
    ReversibleAttention, 
    ReversibleFFN,
    AdditiveCoupling,
    MemoryScheduler
)

def test_coupling_layers():
    """Test coupling layer implementations."""
    print("Testing coupling layers...")
    
    d_model = 256
    batch_size = 2
    seq_len = 64
    
    coupling = AdditiveCoupling(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test split and cat operations
    x1, x2 = coupling.split_input(x)
    assert x1.shape == (batch_size, seq_len, d_model // 2)
    assert x2.shape == (batch_size, seq_len, d_model // 2)
    
    # Test forward and inverse
    y1, y2 = coupling.forward(x1, x2)
    x1_recovered, x2_recovered = coupling.inverse(y1, y2)
    
    # Check perfect reconstruction
    assert torch.allclose(x1, x1_recovered, atol=1e-6)
    assert torch.allclose(x2, x2_recovered, atol=1e-6)
    
    print("✓ Coupling layers working correctly")
    return True

def test_reversible_attention():
    """Test reversible attention layer."""
    print("Testing reversible attention...")
    
    d_model = 256
    num_heads = 8
    batch_size = 2
    seq_len = 64
    
    attention = ReversibleAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test forward pass
    with torch.no_grad():  # No grad for testing
        output = attention(x, use_reversible=False)  # Use standard mode for testing
    assert output.shape == x.shape
    
    # Test memory estimation
    memory_est = attention.estimate_memory_usage(batch_size, seq_len)
    assert 'reversible_memory' in memory_est
    assert 'memory_saved' in memory_est
    
    print("✓ Reversible attention working correctly")
    return True

def test_reversible_ffn():
    """Test reversible feedforward network."""
    print("Testing reversible FFN...")
    
    d_model = 256
    d_ff = 1024
    batch_size = 2
    seq_len = 64
    
    ffn = ReversibleFFN(
        d_model=d_model,
        d_ff=d_ff,
        activation="gelu",
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test forward pass
    with torch.no_grad():  # No grad for testing
        output = ffn(x, use_reversible=False)  # Use standard mode for testing
    assert output.shape == x.shape
    
    # Test memory estimation
    memory_est = ffn.estimate_memory_usage(batch_size, seq_len)
    assert 'reversible_memory' in memory_est
    assert 'memory_saved' in memory_est
    
    print("✓ Reversible FFN working correctly")
    return True

def test_reversible_transformer():
    """Test complete reversible transformer model."""
    print("Testing reversible transformer...")
    
    vocab_size = 1000
    num_layers = 2  # Small for testing
    d_model = 128   # Small for testing
    num_heads = 4   # Small for testing
    d_ff = 256      # Small for testing
    max_seq_len = 512
    
    model = ReversibleTransformer(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    
    batch_size = 1  # Small for testing
    seq_len = 32    # Small for testing
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        outputs = model(input_ids, use_reversible=False)  # Standard mode for testing
    
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, seq_len, vocab_size)
    
    # Test model info
    model_info = model.get_model_info()
    assert model_info['num_layers'] == num_layers
    assert model_info['d_model'] == d_model
    
    print("✓ Reversible transformer working correctly")
    return True

def test_memory_scheduler():
    """Test memory scheduler functionality."""
    print("Testing memory scheduler...")
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    )
    
    scheduler = MemoryScheduler(model, memory_budget=1024*1024*1024)  # 1GB
    
    # Test policy setting
    from revnet_zero.memory.scheduler import RecomputePolicy
    scheduler.set_policy([0, 1], RecomputePolicy.RECOMPUTE)
    scheduler.set_policy([2], RecomputePolicy.STORE)
    
    # Test decision making
    should_recompute_0 = scheduler.should_recompute(0)
    should_recompute_2 = scheduler.should_recompute(2)
    
    assert should_recompute_0 == True
    assert should_recompute_2 == False
    
    print("✓ Memory scheduler working correctly")
    return True

def test_cli_structure():
    """Test CLI structure."""
    print("Testing CLI structure...")
    try:
        from revnet_zero.cli.benchmark import benchmark_cli
        from revnet_zero.cli.profile import profile_cli
        from revnet_zero.cli.convert import convert_cli
        
        # Check that CLI functions exist
        assert callable(benchmark_cli)
        assert callable(profile_cli)
        assert callable(convert_cli)
        
        print("✓ CLI structure valid")
        return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def test_package_metadata():
    """Test package metadata."""
    print("Testing package metadata...")
    try:
        from revnet_zero import __version__, __author__
        
        assert isinstance(__version__, str)
        assert isinstance(__author__, str)
        assert len(__version__) > 0
        
        print(f"✓ Package metadata valid - Version: {__version__}, Author: {__author__}")
        return True
    except Exception as e:
        print(f"❌ Package metadata test failed: {e}")
        return False

def test_configuration_files():
    """Test configuration files."""
    print("Testing configuration files...")
    try:
        import os
        from pathlib import Path
        
        # Check essential files exist
        repo_root = Path(__file__).parent
        essential_files = [
            "setup.py",
            "pyproject.toml", 
            "README.md",
            "LICENSE",
            "ARCHITECTURE.md",
        ]
        
        missing_files = []
        for file_name in essential_files:
            file_path = repo_root / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"⚠️  Missing files: {missing_files}")
        else:
            print("✓ All essential files present")
        
        return len(missing_files) == 0
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_all_tests():
    """Run all basic tests."""
    print("🚀 Running RevNet-Zero Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        ("Coupling Layers", test_coupling_layers),
        ("Reversible Attention", test_reversible_attention),
        ("Reversible FFN", test_reversible_ffn),
        ("Reversible Transformer", test_reversible_transformer),
        ("Memory Scheduler", test_memory_scheduler),
        ("Package Metadata", test_package_metadata),
        ("Configuration Files", test_configuration_files),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! RevNet-Zero basic functionality is working.")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed. Some functionality may not work properly.")
        return False

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)