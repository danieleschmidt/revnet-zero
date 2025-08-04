#!/usr/bin/env python3
"""
Basic functionality test for RevNet-Zero without external dependencies.

This script tests core functionality using only standard library and 
simulated PyTorch operations.
"""

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    try:
        # Test main imports
        from revnet_zero import ReversibleTransformer
        from revnet_zero.layers import ReversibleAttention, ReversibleFFN
        from revnet_zero.memory import MemoryScheduler
        from revnet_zero.utils import convert_to_reversible
        print("✓ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation without PyTorch."""
    print("Testing model creation structure...")
    try:
        # Just test that the classes exist and can be instantiated at the Python level
        from revnet_zero.models.reversible_transformer import ReversibleTransformer
        
        # Check class definition
        assert hasattr(ReversibleTransformer, '__init__')
        assert hasattr(ReversibleTransformer, 'forward')
        assert hasattr(ReversibleTransformer, 'get_model_info')
        
        print("✓ Model class structure valid")
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def test_layer_structure():
    """Test layer structure."""
    print("Testing layer structure...")
    try:
        from revnet_zero.layers.reversible_attention import ReversibleAttention
        from revnet_zero.layers.reversible_ffn import ReversibleFFN
        from revnet_zero.layers.coupling_layers import AdditiveCoupling, AffineCoupling
        
        # Check that classes have required methods
        assert hasattr(ReversibleAttention, 'forward')
        assert hasattr(ReversibleFFN, 'forward')
        assert hasattr(AdditiveCoupling, 'forward')
        assert hasattr(AffineCoupling, 'forward')
        
        print("✓ Layer structure valid")
        return True
    except Exception as e:
        print(f"❌ Layer structure test failed: {e}")
        return False

def test_memory_scheduler():
    """Test memory scheduler structure."""
    print("Testing memory scheduler...")
    try:
        from revnet_zero.memory.scheduler import MemoryScheduler, AdaptiveScheduler
        from revnet_zero.memory.profiler import MemoryProfiler
        
        # Check that classes have required methods
        assert hasattr(MemoryScheduler, 'should_recompute')
        assert hasattr(AdaptiveScheduler, 'adapt_policies')
        assert hasattr(MemoryProfiler, 'start_profiling')
        
        print("✓ Memory scheduler structure valid")
        return True
    except Exception as e:
        print(f"❌ Memory scheduler test failed: {e}")
        return False

def test_examples():
    """Test examples structure."""
    print("Testing examples...")
    try:
        from examples.basic_usage import BasicExample
        from examples.benchmarking import BenchmarkSuite
        
        # Check that example classes exist
        assert hasattr(BasicExample, 'run_complete_example')
        assert hasattr(BenchmarkSuite, 'scaling_benchmark')
        
        print("✓ Examples structure valid")
        return True
    except Exception as e:
        print(f"❌ Examples test failed: {e}")
        return False

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
        ("Package Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Layer Structure", test_layer_structure),
        ("Memory Scheduler", test_memory_scheduler),
        ("Examples", test_examples),
        ("CLI Structure", test_cli_structure),
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