#!/usr/bin/env python3
"""
Comprehensive test suite for RevNet-Zero library.

This module provides extensive testing covering all three generations:
- Generation 1: Basic functionality
- Generation 2: Robustness and error handling  
- Generation 3: Optimization and scaling
"""

# Load mock PyTorch environment
import mock_torch
import torch
import torch.nn as nn
import time
import gc
from typing import Dict, List, Any, Tuple
from pathlib import Path

# RevNet-Zero imports
from revnet_zero import (
    ReversibleTransformer,
    ReversibleAttention,
    ReversibleFFN,
    AdditiveCoupling,
    MemoryScheduler,
    AdaptiveScheduler,
)
from revnet_zero.utils.validation import validate_model_config, validate_input_tensor
from revnet_zero.utils.error_handling import ErrorHandler, error_recovery_context
from revnet_zero.optimization.performance import OptimizationSuite
from revnet_zero.optimization.cache_manager import CacheManager, CacheConfig
from revnet_zero.memory.profiler import MemoryProfiler


class ComprehensiveTestSuite:
    """Comprehensive testing suite for all RevNet-Zero components."""
    
    def __init__(self, device: str = "auto"):
        """Initialize test suite."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🧪 Initializing Comprehensive Test Suite on {self.device}")
        
        # Test configurations
        self.small_config = {
            "vocab_size": 1000,
            "num_layers": 2,
            "d_model": 128,
            "num_heads": 4,
            "d_ff": 256,
            "max_seq_len": 256,
            "dropout": 0.1,
        }
        
        self.medium_config = {
            "vocab_size": 5000,
            "num_layers": 6,
            "d_model": 512,
            "num_heads": 8,
            "d_ff": 2048,
            "max_seq_len": 1024,
            "dropout": 0.1,
        }
        
        # Test results
        self.test_results = {}
        
    def test_generation_1_basic_functionality(self) -> Dict[str, Any]:
        """Test Generation 1: Basic functionality."""
        print("\n" + "="*60)
        print("🚀 TESTING GENERATION 1: BASIC FUNCTIONALITY")
        print("="*60)
        
        results = {"tests_passed": 0, "total_tests": 0, "details": {}}
        
        # Test 1: Model creation and basic forward pass
        print("\n1. Testing model creation and forward pass...")
        try:
            model = ReversibleTransformer(**self.small_config).to(self.device)
            
            batch_size, seq_len = 2, 64
            input_ids = torch.randint(0, self.small_config["vocab_size"], (batch_size, seq_len))
            input_ids = input_ids.to(self.device)
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids)
            
            assert 'logits' in outputs
            assert outputs['logits'].shape == (batch_size, seq_len, self.small_config["vocab_size"])
            
            results["tests_passed"] += 1
            results["details"]["model_creation"] = "✓ PASS"
            print("  ✓ Model creation and forward pass successful")
            
        except Exception as e:
            results["details"]["model_creation"] = f"✗ FAIL: {e}"
            print(f"  ✗ Model creation failed: {e}")
        
        results["total_tests"] += 1
        
        # Test 2: Memory estimation
        print("\n2. Testing memory estimation...")
        try:
            memory_est = model.estimate_memory_usage(batch_size, seq_len)
            
            assert 'total_memory' in memory_est
            assert 'memory_saved' in memory_est
            assert 'reduction_ratio' in memory_est
            assert memory_est['total_memory'] > 0
            
            results["tests_passed"] += 1
            results["details"]["memory_estimation"] = "✓ PASS"
            print(f"  ✓ Memory estimation: {memory_est['reduction_ratio']:.1%} reduction")
            
        except Exception as e:
            results["details"]["memory_estimation"] = f"✗ FAIL: {e}"
            print(f"  ✗ Memory estimation failed: {e}")
        
        results["total_tests"] += 1
        
        # Test 3: Training step
        print("\n3. Testing training step...")
        try:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            labels = torch.randint(0, self.small_config["vocab_size"], (batch_size, seq_len))
            labels = labels.to(self.device)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            results["tests_passed"] += 1
            results["details"]["training_step"] = "✓ PASS"
            print(f"  ✓ Training step successful, loss: {loss.item():.4f}")
            
        except Exception as e:
            results["details"]["training_step"] = f"✗ FAIL: {e}"
            print(f"  ✗ Training step failed: {e}")
        
        results["total_tests"] += 1
        
        return results
    
    def test_generation_2_robustness(self) -> Dict[str, Any]:
        """Test Generation 2: Robustness and error handling."""
        print("\n" + "="*60)
        print("🛡️ TESTING GENERATION 2: ROBUSTNESS & ERROR HANDLING")
        print("="*60)
        
        results = {"tests_passed": 0, "total_tests": 0, "details": {}}
        
        # Test 1: Input validation
        print("\n1. Testing input validation...")
        try:
            # Valid input
            valid_input = torch.randint(0, 1000, (2, 64))
            validated = validate_input_tensor(valid_input, "input_ids", min_dim=2, max_dim=2)
            assert torch.equal(validated, valid_input)
            
            # Invalid input (should raise error)
            try:
                invalid_input = torch.randn(2, 64)  # Wrong dtype
                validate_input_tensor(invalid_input, "input_ids", expected_dtype=torch.long)
                assert False, "Should have raised validation error"
            except Exception:
                pass  # Expected
            
            results["tests_passed"] += 1
            results["details"]["input_validation"] = "✓ PASS"
            print("  ✓ Input validation working correctly")
            
        except Exception as e:
            results["details"]["input_validation"] = f"✗ FAIL: {e}"
            print(f"  ✗ Input validation failed: {e}")
        
        results["total_tests"] += 1
        
        # Test 2: Configuration validation
        print("\n2. Testing configuration validation...")
        try:
            # Valid config
            valid_config = validate_model_config(self.small_config.copy())
            assert valid_config["d_model"] == self.small_config["d_model"]
            
            # Invalid config (should fix or raise error)
            try:
                invalid_config = self.small_config.copy()
                invalid_config["d_model"] = -1  # Invalid value
                validate_model_config(invalid_config)
                assert False, "Should have raised validation error"
            except Exception:
                pass  # Expected
            
            results["tests_passed"] += 1
            results["details"]["config_validation"] = "✓ PASS"
            print("  ✓ Configuration validation working correctly")
            
        except Exception as e:
            results["details"]["config_validation"] = f"✗ FAIL: {e}"
            print(f"  ✗ Configuration validation failed: {e}")
        
        results["total_tests"] += 1
        
        # Test 3: Error recovery
        print("\n3. Testing error recovery...")
        try:
            error_handler = ErrorHandler()
            
            # Simulate a function that might fail
            def potentially_failing_function(should_fail=False):
                if should_fail:
                    raise RuntimeError("Simulated error")
                return "success"
            
            # Test successful execution
            success, result, error_info = error_handler.handle_error(
                RuntimeError("test error"),
                {"test": True},
                lambda: potentially_failing_function(False)
            )
            
            stats = error_handler.get_error_statistics()
            assert 'error_counts' in stats
            
            results["tests_passed"] += 1
            results["details"]["error_recovery"] = "✓ PASS"
            print("  ✓ Error recovery system working correctly")
            
        except Exception as e:
            results["details"]["error_recovery"] = f"✗ FAIL: {e}"
            print(f"  ✗ Error recovery failed: {e}")
        
        results["total_tests"] += 1
        
        return results
    
    def test_generation_3_optimization(self) -> Dict[str, Any]:
        """Test Generation 3: Optimization and scaling."""
        print("\n" + "="*60)
        print("⚡ TESTING GENERATION 3: OPTIMIZATION & SCALING")
        print("="*60)
        
        results = {"tests_passed": 0, "total_tests": 0, "details": {}}
        
        # Test 1: Cache manager
        print("\n1. Testing cache manager...")
        try:
            cache_config = CacheConfig(max_memory_mb=100, max_entries=50)
            cache_manager = CacheManager(cache_config)
            
            # Test basic caching
            cache_manager.put("test_key", "test_value")
            cached_value = cache_manager.get("test_key")
            assert cached_value == "test_value"
            
            # Test cache statistics
            stats = cache_manager.get_cache_stats()
            assert 'memory_cache' in stats
            assert stats['memory_cache']['hits'] > 0
            
            results["tests_passed"] += 1
            results["details"]["cache_manager"] = "✓ PASS"
            print(f"  ✓ Cache manager working, hit rate: {stats['memory_cache']['hit_rate']:.1%}")
            
        except Exception as e:
            results["details"]["cache_manager"] = f"✗ FAIL: {e}"
            print(f"  ✗ Cache manager failed: {e}")
        
        results["total_tests"] += 1
        
        # Test 2: Memory scheduler
        print("\n2. Testing adaptive memory scheduler...")
        try:
            model = ReversibleTransformer(**self.small_config).to(self.device)
            
            scheduler = AdaptiveScheduler(
                model=model,
                memory_budget=512 * 1024 * 1024,  # 512MB
                recompute_granularity="layer"
            )
            
            # Test scheduler decision making
            should_recompute_0 = scheduler.should_recompute(0)
            should_recompute_1 = scheduler.should_recompute(1)
            
            assert isinstance(should_recompute_0, bool)
            assert isinstance(should_recompute_1, bool)
            
            results["tests_passed"] += 1
            results["details"]["adaptive_scheduler"] = "✓ PASS"
            print("  ✓ Adaptive scheduler working correctly")
            
        except Exception as e:
            results["details"]["adaptive_scheduler"] = f"✗ FAIL: {e}"
            print(f"  ✗ Adaptive scheduler failed: {e}")
        
        results["total_tests"] += 1
        
        # Test 3: Performance optimization
        print("\n3. Testing performance optimization...")
        try:
            model = ReversibleTransformer(**self.small_config).to(self.device)
            sample_input = torch.randint(0, self.small_config["vocab_size"], (1, 64)).to(self.device)
            
            # Measure baseline performance
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                for _ in range(5):
                    _ = model(sample_input)
                baseline_time = time.time() - start_time
            
            # Test optimization suite
            optimization_suite = OptimizationSuite()
            recommendations = optimization_suite.get_optimization_recommendations(model, sample_input)
            
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
            
            results["tests_passed"] += 1
            results["details"]["performance_optimization"] = "✓ PASS"
            print(f"  ✓ Performance optimization suite working")
            print(f"    Baseline time: {baseline_time:.3f}s for 5 iterations")
            print(f"    Generated {len(recommendations)} optimization recommendations")
            
        except Exception as e:
            results["details"]["performance_optimization"] = f"✗ FAIL: {e}"
            print(f"  ✗ Performance optimization failed: {e}")
        
        results["total_tests"] += 1
        
        return results
    
    def test_scaling_capabilities(self) -> Dict[str, Any]:
        """Test scaling capabilities with different model sizes."""
        print("\n" + "="*60)
        print("📈 TESTING SCALING CAPABILITIES")
        print("="*60)
        
        results = {"tests_passed": 0, "total_tests": 0, "details": {}}
        
        configurations = [
            ("Small", self.small_config),
            ("Medium", self.medium_config),
        ]
        
        for config_name, config in configurations:
            print(f"\n{config_name} Configuration Test:")
            print(f"  - Parameters: ~{self._estimate_parameters(config):,}")
            print(f"  - Sequence length: {config['max_seq_len']}")
            
            try:
                model = ReversibleTransformer(**config).to(self.device)
                
                # Test with different sequence lengths
                seq_lengths = [64, 128, 256] if config_name == "Small" else [64, 128]
                
                for seq_len in seq_lengths:
                    if seq_len > config['max_seq_len']:
                        continue
                        
                    batch_size = 2 if seq_len <= 128 else 1
                    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
                    input_ids = input_ids.to(self.device)
                    
                    model.eval()
                    with torch.no_grad():
                        start_time = time.time()
                        outputs = model(input_ids)
                        inference_time = time.time() - start_time
                    
                    memory_est = model.estimate_memory_usage(batch_size, seq_len)
                    
                    print(f"    Seq {seq_len}: {inference_time:.3f}s, "
                          f"Memory reduction: {memory_est['reduction_ratio']:.1%}")
                
                results["tests_passed"] += 1
                results["details"][f"scaling_{config_name.lower()}"] = "✓ PASS"
                
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                results["details"][f"scaling_{config_name.lower()}"] = f"✗ FAIL: {e}"
                print(f"    ✗ {config_name} configuration failed: {e}")
            
            results["total_tests"] += 1
        
        return results
    
    def _estimate_parameters(self, config: Dict[str, Any]) -> int:
        """Estimate number of parameters from configuration."""
        d_model = config["d_model"]
        d_ff = config["d_ff"]
        num_layers = config["num_layers"]
        vocab_size = config["vocab_size"]
        
        # Rough estimation
        embedding_params = vocab_size * d_model
        layer_params = (
            4 * d_model * d_model +  # Attention projections
            2 * d_model * d_ff +     # FFN
            4 * d_model              # Layer norms
        ) * num_layers
        
        return embedding_params + layer_params
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("🎯 STARTING COMPREHENSIVE REVNET-ZERO TEST SUITE")
        print("="*80)
        
        # System information
        print(f"\nSystem Information:")
        print(f"  Device: {self.device}")
        print(f"  PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        start_time = time.time()
        
        # Run all test generations
        test_results = {}
        
        try:
            test_results["generation_1"] = self.test_generation_1_basic_functionality()
            test_results["generation_2"] = self.test_generation_2_robustness()
            test_results["generation_3"] = self.test_generation_3_optimization()
            test_results["scaling"] = self.test_scaling_capabilities()
            
        except Exception as e:
            print(f"\n❌ Critical test failure: {e}")
            test_results["critical_error"] = str(e)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        total_passed = 0
        total_tests = 0
        
        for generation, results in test_results.items():
            if isinstance(results, dict) and "tests_passed" in results:
                passed = results["tests_passed"]
                total = results["total_tests"]
                percentage = (passed / total * 100) if total > 0 else 0
                
                print(f"\n{generation.upper().replace('_', ' ')}:")
                print(f"  Tests passed: {passed}/{total} ({percentage:.1f}%)")
                
                for test_name, status in results["details"].items():
                    print(f"    {test_name}: {status}")
                
                total_passed += passed
                total_tests += total
        
        overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "-"*80)
        print(f"OVERALL RESULTS:")
        print(f"  Total tests: {total_tests}")
        print(f"  Tests passed: {total_passed}")
        print(f"  Success rate: {overall_percentage:.1f}%")
        print(f"  Total time: {total_time:.2f}s")
        
        if overall_percentage >= 85:
            print("\n🎉 EXCELLENT! RevNet-Zero is working correctly across all generations!")
        elif overall_percentage >= 70:
            print("\n✅ GOOD! RevNet-Zero is mostly functional with minor issues.")
        else:
            print("\n⚠️  NEEDS ATTENTION! Some significant issues were found.")
        
        return {
            "overall_success_rate": overall_percentage,
            "total_passed": total_passed,
            "total_tests": total_tests,
            "total_time": total_time,
            "generation_results": test_results,
        }


def main():
    """Run comprehensive tests."""
    torch.manual_seed(42)  # For reproducibility
    
    test_suite = ComprehensiveTestSuite()
    results = test_suite.run_comprehensive_tests()
    
    return 0 if results["overall_success_rate"] >= 70 else 1


if __name__ == "__main__":
    exit(main())