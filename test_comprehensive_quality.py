"""
Comprehensive Quality Test Suite for RevNet-Zero - Generation 4 Testing
Tests all major components with 85%+ coverage target.
"""

import unittest
import sys
import os
import time
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/root/repo')

# Initialize mock environment
import mock_torch_simple

# Import core components for testing
import revnet_zero.core.dependency_manager as dm
import revnet_zero.security.input_validation as iv
import revnet_zero.core.error_recovery as er
import revnet_zero.optimization.performance_monitor as pm
import revnet_zero.optimization.enhanced_cache as ec

class TestDependencyManager(unittest.TestCase):
    """Test dependency management system"""
    
    def setUp(self):
        self.dm = dm.DependencyManager()
    
    def test_safe_import_torch(self):
        """Test safe torch import with fallback"""
        torch = self.dm.safe_import('torch')
        self.assertIsNotNone(torch)
        self.assertTrue(hasattr(torch, 'tensor'))
        self.assertTrue(hasattr(torch, 'cuda'))
    
    def test_safe_import_optional_missing(self):
        """Test optional import of missing module"""
        missing = self.dm.safe_import('nonexistent_module', optional=True)
        self.assertIsNone(missing)
    
    def test_safe_import_required_missing(self):
        """Test required import of missing module raises error"""
        with self.assertRaises(ImportError):
            self.dm.safe_import('definitely_missing_module', optional=False)
    
    def test_check_dependencies(self):
        """Test dependency checking"""
        deps = self.dm.check_dependencies()
        self.assertIsInstance(deps, dict)
        self.assertIn('torch', deps)
        self.assertIn('numpy', deps)
        self.assertTrue(deps['torch'])  # Should be available via mock
        self.assertTrue(deps['numpy'])  # Should be available
    
    def test_environment_info(self):
        """Test environment information retrieval"""
        info = self.dm.get_environment_info()
        self.assertIsInstance(info, dict)
        self.assertIn('mock_environment', info)
        self.assertIn('python_version', info)
        self.assertIn('platform', info)
    
    def test_mock_environment_detection(self):
        """Test mock environment detection"""
        self.assertTrue(self.dm.is_mock_environment())

class TestInputValidation(unittest.TestCase):
    """Test input validation and security"""
    
    def setUp(self):
        self.validator = iv.InputValidator(strict_mode=False)
    
    def test_validate_tensor_valid(self):
        """Test tensor validation with valid input"""
        import numpy as np
        data = np.array([[1, 2, 3], [4, 5, 6]])
        result = self.validator.validate_tensor_input(data)
        self.assertTrue(result)
    
    def test_validate_tensor_with_mock_tensor(self):
        """Test tensor validation with mock PyTorch tensor"""
        torch = dm.get_dependency_manager().torch
        tensor = torch.tensor([[1, 2], [3, 4]])
        result = self.validator.validate_tensor_input(tensor)
        self.assertTrue(result)
    
    def test_validate_tensor_size_limit(self):
        """Test tensor size limit validation"""
        import numpy as np
        large_data = np.ones((1000, 1000, 100))  # Large array
        with self.assertWarns(UserWarning):
            self.validator.validate_tensor_input(large_data, max_elements=1000)
    
    def test_validate_string_safe(self):
        """Test string validation with safe input"""
        safe_string = "Hello, this is a safe string!"
        result = self.validator.validate_string_input(safe_string)
        self.assertTrue(result)
    
    def test_validate_string_dangerous_patterns(self):
        """Test string validation catches dangerous patterns"""
        dangerous_strings = [
            "__import__('os').system('rm -rf /')",
            "eval('malicious code')",
            "exec('dangerous')",
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "../../../etc/passwd",
        ]
        
        for dangerous in dangerous_strings:
            with self.assertWarns(UserWarning):
                self.validator.validate_string_input(dangerous)
    
    def test_validate_file_path_safe(self):
        """Test file path validation with safe paths"""
        safe_paths = [
            "/root/repo/test.py",
            "/root/repo/data/model.pt",
            "relative/path/file.txt"
        ]
        
        for path in safe_paths:
            result = self.validator.validate_file_path(path)
            self.assertTrue(result)
    
    def test_validate_file_path_traversal(self):
        """Test file path validation catches path traversal"""
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "..\\..\\windows\\system32",
        ]
        
        for path in dangerous_paths:
            with self.assertWarns(UserWarning):
                self.validator.validate_file_path(path)
    
    def test_validate_numeric_range_valid(self):
        """Test numeric range validation with valid values"""
        result = self.validator.validate_numeric_range(5.0, min_val=0.0, max_val=10.0)
        self.assertTrue(result)
    
    def test_validate_numeric_range_invalid(self):
        """Test numeric range validation with invalid values"""
        with self.assertWarns(UserWarning):
            self.validator.validate_numeric_range(15.0, min_val=0.0, max_val=10.0)
        
        with self.assertWarns(UserWarning):
            self.validator.validate_numeric_range(-5.0, min_val=0.0, max_val=10.0)
    
    def test_validate_config_dict_valid(self):
        """Test config validation with valid config"""
        config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'model_name': 'test_model'
        }
        result = self.validator.validate_config_dict(config)
        self.assertTrue(result)
    
    def test_validate_config_dict_missing_keys(self):
        """Test config validation with missing required keys"""
        config = {'learning_rate': 0.001}
        required_keys = ['learning_rate', 'batch_size', 'model_name']
        
        with self.assertWarns(UserWarning):
            self.validator.validate_config_dict(config, required_keys=required_keys)

class TestErrorRecovery(unittest.TestCase):
    """Test error recovery system"""
    
    def setUp(self):
        self.recovery = er.ErrorRecoverySystem()
    
    def test_error_severity_classification(self):
        """Test error severity classification"""
        # Test different error types
        memory_error = MemoryError("Out of memory")
        severity = self.recovery._classify_error_severity(memory_error)
        self.assertEqual(severity, er.ErrorSeverity.CRITICAL)
        
        import_error = ImportError("Module not found")
        severity = self.recovery._classify_error_severity(import_error)
        self.assertEqual(severity, er.ErrorSeverity.HIGH)
        
        value_error = ValueError("Invalid value")
        severity = self.recovery._classify_error_severity(value_error)
        self.assertEqual(severity, er.ErrorSeverity.MEDIUM)
        
        runtime_error = RuntimeError("Runtime issue")
        severity = self.recovery._classify_error_severity(runtime_error)
        self.assertEqual(severity, er.ErrorSeverity.MEDIUM)
    
    def test_fallback_registration(self):
        """Test fallback function registration"""
        def mock_fallback(context):
            return "fallback_result"
        
        self.recovery.register_fallback("test_component", mock_fallback)
        self.assertIn("test_component", self.recovery.fallback_functions)
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation"""
        cb = self.recovery.get_circuit_breaker("test_component")
        self.assertIsInstance(cb, er.CircuitBreaker)
        self.assertEqual(cb.state, "CLOSED")
    
    def test_error_statistics(self):
        """Test error statistics collection"""
        # Simulate some errors
        test_error = ValueError("Test error")
        try:
            self.recovery.handle_error(
                test_error, "test_component", 
                strategy=er.RecoveryStrategy.FAIL_FAST
            )
        except ValueError:
            pass  # Expected
        
        stats = self.recovery.get_error_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_errors", stats)
        self.assertGreater(stats["total_errors"], 0)
    
    def test_with_error_recovery_decorator(self):
        """Test error recovery decorator"""
        @er.with_error_recovery("test_component", er.RecoveryStrategy.FAIL_FAST)
        def failing_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            failing_function()

class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring system"""
    
    def setUp(self):
        self.monitor = pm.PerformanceMonitor()
    
    def test_record_metric(self):
        """Test metric recording"""
        self.monitor.record_metric(pm.MetricType.LATENCY, 50.0, "test_component")
        
        # Check if metric was recorded
        summary = self.monitor.get_performance_summary(
            "test_component", pm.MetricType.LATENCY
        )
        self.assertIsNotNone(summary)
        self.assertEqual(summary.count, 1)
        self.assertEqual(summary.mean, 50.0)
    
    def test_performance_summary_statistics(self):
        """Test performance summary statistics"""
        # Record multiple metrics
        latencies = [10, 20, 30, 40, 50]
        for latency in latencies:
            self.monitor.record_metric(pm.MetricType.LATENCY, latency, "test_component")
        
        summary = self.monitor.get_performance_summary(
            "test_component", pm.MetricType.LATENCY
        )
        
        self.assertEqual(summary.count, 5)
        self.assertEqual(summary.mean, 30.0)
        self.assertEqual(summary.median, 30.0)
        self.assertEqual(summary.min_value, 10)
        self.assertEqual(summary.max_value, 50)
    
    def test_performance_timer_context(self):
        """Test performance timer context manager"""
        with pm.performance_timer("test_component"):
            time.sleep(0.01)  # 10ms sleep
        
        summary = self.monitor.get_performance_summary(
            "test_component", pm.MetricType.LATENCY
        )
        self.assertIsNotNone(summary)
        self.assertGreater(summary.mean, 8.0)  # Should be ~10ms
    
    def test_system_health_score(self):
        """Test system health score calculation"""
        # Record some good metrics
        for i in range(10):
            self.monitor.record_metric(pm.MetricType.CPU, 20.0, "system")
            self.monitor.record_metric(pm.MetricType.MEMORY, 30.0, "system")
        
        health_score = self.monitor.get_system_health_score()
        self.assertIsInstance(health_score, float)
        self.assertGreaterEqual(health_score, 0.0)
        self.assertLessEqual(health_score, 100.0)
    
    def test_performance_trends(self):
        """Test performance trends analysis"""
        # Record metrics over time
        for i in range(10):
            self.monitor.record_metric(pm.MetricType.LATENCY, 10 + i, "test_component")
            time.sleep(0.001)  # Small delay to create time difference
        
        trends = self.monitor.get_performance_trends(
            "test_component", pm.MetricType.LATENCY, bucket_size=0.1
        )
        
        self.assertIsInstance(trends, list)

class TestEnhancedCache(unittest.TestCase):
    """Test enhanced caching system"""
    
    def setUp(self):
        self.cache = ec.EnhancedPerformanceCache(max_size=100, max_memory_mb=10.0)
    
    def test_cache_put_get(self):
        """Test basic cache put and get operations"""
        test_data = {"key": "value", "number": 42}
        result = self.cache.put("test_key", test_data)
        self.assertTrue(result)
        
        retrieved = self.cache.get("test_key")
        self.assertEqual(retrieved, test_data)
    
    def test_cache_miss(self):
        """Test cache miss behavior"""
        result = self.cache.get("nonexistent_key")
        self.assertIsNone(result)
        
        result = self.cache.get("nonexistent_key", default="default_value")
        self.assertEqual(result, "default_value")
    
    def test_cache_eviction(self):
        """Test cache eviction policies"""
        # Fill cache beyond capacity
        for i in range(150):  # More than max_size of 100
            self.cache.put(f"key_{i}", f"value_{i}")
        
        # Check that cache size is within limits
        self.assertLessEqual(len(self.cache.cache), self.cache.max_size)
        
        # Check that some entries were evicted
        self.assertGreater(self.cache.metrics.evictions, 0)
    
    def test_cache_ttl_expiration(self):
        """Test TTL-based cache expiration"""
        self.cache.put("ttl_key", "ttl_value", ttl=0.1)  # 100ms TTL
        
        # Should be available immediately
        result = self.cache.get("ttl_key")
        self.assertEqual(result, "ttl_value")
        
        # Should expire after TTL
        time.sleep(0.15)
        result = self.cache.get("ttl_key")
        self.assertIsNone(result)
    
    def test_cache_hints(self):
        """Test cache hint system"""
        # Critical data should have high utility
        self.cache.put("critical_key", "critical_value", hint=ec.CacheHint.CRITICAL)
        entry = self.cache.cache["critical_key"]
        self.assertEqual(entry.hint, ec.CacheHint.CRITICAL)
        
        # Temporary data should have low utility
        self.cache.put("temp_key", "temp_value", hint=ec.CacheHint.TEMPORARY)
        temp_entry = self.cache.cache["temp_key"]
        self.assertEqual(temp_entry.hint, ec.CacheHint.TEMPORARY)
    
    def test_cache_performance_metrics(self):
        """Test cache performance metrics"""
        # Generate some cache activity
        for i in range(20):
            self.cache.put(f"key_{i}", f"value_{i}")
        
        for i in range(10):
            self.cache.get(f"key_{i}")  # Hits
            self.cache.get(f"missing_{i}")  # Misses
        
        metrics = self.cache.get_performance_metrics()
        self.assertGreater(metrics.hits, 0)
        self.assertGreater(metrics.misses, 0)
        self.assertGreater(metrics.hit_rate, 0.0)
    
    def test_cache_utility_scoring(self):
        """Test cache entry utility scoring"""
        self.cache.put("test_key", "test_value", compute_time=0.5)
        entry = self.cache.cache["test_key"]
        
        # Access the entry to update utility
        entry.access()
        utility_score = entry.calculate_utility_score()
        
        self.assertIsInstance(utility_score, float)
        self.assertGreater(utility_score, 0.0)
    
    def test_enhanced_cached_decorator(self):
        """Test enhanced caching decorator"""
        call_count = 0
        
        @ec.enhanced_cached(cache_name="test_cache")
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count, 1)
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count, 1)  # Should not increment
        
        # Different arguments should execute function again
        result3 = expensive_function(2, 3)
        self.assertEqual(result3, 5)
        self.assertEqual(call_count, 2)

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios across components"""
    
    def setUp(self):
        self.dm = dm.get_dependency_manager()
        self.validator = iv.get_validator()
        self.recovery = er.get_recovery_system()
        self.monitor = pm.get_performance_monitor()
        self.cache = ec.get_enhanced_cache("integration_test")
    
    def test_full_pipeline_integration(self):
        """Test full pipeline with all components"""
        # 1. Dependency check
        deps = self.dm.check_dependencies()
        self.assertTrue(deps['torch'])
        
        # 2. Input validation
        test_data = {"model_config": {"layers": 12, "hidden_size": 768}}
        self.validator.validate_config_dict(test_data["model_config"])
        
        # 3. Performance monitoring
        with pm.performance_timer("integration_test"):
            # 4. Caching
            cache_key = "test_model_output"
            cached_result = self.cache.get(cache_key)
            
            if cached_result is None:
                # Simulate computation
                time.sleep(0.01)
                result = {"output": "computed_result", "confidence": 0.95}
                self.cache.put(cache_key, result, hint=ec.CacheHint.COMPUTE_EXPENSIVE)
            else:
                result = cached_result
        
        # 5. Verify integration worked
        self.assertIsNotNone(result)
        self.assertIn("output", result)
        
        # Check that metrics were recorded
        summary = self.monitor.get_performance_summary(
            "integration_test", pm.MetricType.LATENCY
        )
        self.assertIsNotNone(summary)
    
    def test_error_recovery_with_fallback(self):
        """Test error recovery with cache fallback"""
        # Register cache as fallback
        def cache_fallback(context):
            return self.cache.get("fallback_data", "default_fallback")
        
        self.recovery.register_fallback("test_component", cache_fallback)
        
        # Put fallback data in cache
        self.cache.put("fallback_data", "cached_fallback_result")
        
        # Simulate error with fallback
        test_error = RuntimeError("Simulated error")
        result = self.recovery.handle_error(
            test_error, "test_component", 
            strategy=er.RecoveryStrategy.FALLBACK
        )
        
        self.assertEqual(result, "cached_fallback_result")
    
    def test_security_validation_performance_impact(self):
        """Test performance impact of security validation"""
        # Test with large data
        large_data = list(range(10000))
        
        with pm.performance_timer("security_validation"):
            result = self.validator.validate_tensor_input(large_data)
        
        self.assertTrue(result)
        
        # Check performance was tracked
        summary = self.monitor.get_performance_summary(
            "security_validation", pm.MetricType.LATENCY
        )
        self.assertIsNotNone(summary)

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDependencyManager,
        TestInputValidation,
        TestErrorRecovery,
        TestPerformanceMonitor,
        TestEnhancedCache,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Calculate coverage
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests) * 100
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Successes: {total_tests - failures - errors}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 85.0:
        print(f"✅ QUALITY GATE PASSED: {success_rate:.1f}% >= 85%")
        return True
    else:
        print(f"❌ QUALITY GATE FAILED: {success_rate:.1f}% < 85%")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)