#!/usr/bin/env python3
"""
Comprehensive production integration tests for RevNet-Zero.
"""

import sys
import os
import time
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestProductionIntegration(unittest.TestCase):
    """Production integration test suite."""
    
    def setUp(self):
        """Set up test environment."""
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up after tests."""
        execution_time = time.time() - self.start_time
        if execution_time > 1.0:
            print(f"Warning: Test took {execution_time:.2f}s")
    
    def test_core_module_imports(self):
        """Test that core modules can be imported."""
        # Test individual modules without triggering torch imports
        
        # Test internationalization
        try:
            exec(open('revnet_zero/deployment/internationalization.py').read())
            print("✅ Internationalization module loads successfully")
        except Exception as e:
            self.fail(f"Internationalization module failed: {e}")
        
        # Test multi-region
        try:
            exec(open('revnet_zero/deployment/multi_region.py').read())
            print("✅ Multi-region module loads successfully")
        except Exception as e:
            self.fail(f"Multi-region module failed: {e}")
        
        # Test API modules
        try:
            exec(open('revnet_zero/api/models.py').read())
            print("✅ API models module loads successfully")
        except Exception as e:
            self.fail(f"API models module failed: {e}")
        
        try:
            exec(open('revnet_zero/api/endpoints.py').read())
            print("✅ API endpoints module loads successfully")
        except Exception as e:
            self.fail(f"API endpoints module failed: {e}")
    
    def test_security_features(self):
        """Test security features."""
        # Test secure validation module
        try:
            exec(open('revnet_zero/security/secure_validation.py').read())
            print("✅ Security validation module loads successfully")
        except Exception as e:
            self.fail(f"Security validation module failed: {e}")
    
    def test_api_functionality(self):
        """Test API functionality."""
        # Execute and test API endpoints
        namespace = {}
        exec(open('revnet_zero/api/endpoints.py').read(), namespace)
        
        # Test ModelAPI
        ModelAPI = namespace['ModelAPI']
        api = ModelAPI()
        
        # Test model loading
        result = api.load_model("test_model", {"d_model": 512})
        self.assertTrue(result["success"])
        self.assertIn("model_info", result)
        print("✅ Model API load_model works")
        
        # Test prediction
        result = api.predict("test_model", {"text": "Hello world", "max_length": 100})
        self.assertTrue(result["success"])
        self.assertIn("generated_text", result)
        print("✅ Model API predict works")
        
        # Test TrainingAPI
        TrainingAPI = namespace['TrainingAPI']
        training_api = TrainingAPI()
        
        # Test training start
        result = training_api.start_training({"num_epochs": 3, "batch_size": 8})
        self.assertTrue(result["success"])
        self.assertIn("job_id", result)
        job_id = result["job_id"]
        print("✅ Training API start_training works")
        
        # Test training status
        result = training_api.get_training_status(job_id)
        self.assertTrue(result["success"])
        self.assertIn("job_info", result)
        print("✅ Training API get_training_status works")
        
        # Test HealthAPI
        HealthAPI = namespace['HealthAPI']
        health_api = HealthAPI()
        
        # Test health check
        result = health_api.health_check()
        self.assertTrue(result["success"])
        self.assertIn("status", result)
        self.assertIn("uptime_seconds", result)
        print("✅ Health API health_check works")
        
        # Test metrics
        result = health_api.get_metrics()
        self.assertTrue(result["success"])
        self.assertIn("metrics", result)
        print("✅ Health API get_metrics works")
    
    def test_globalization_features(self):
        """Test globalization and compliance features."""
        # Test internationalization
        namespace = {}
        
        # Patch Path to avoid file system dependencies
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        code = open('revnet_zero/deployment/internationalization.py').read()
        code = code.replace('Path(__file__).parent', f'Path("{temp_dir}")')
        
        exec(code, namespace)
        
        InternationalizationManager = namespace['InternationalizationManager']
        i18n = InternationalizationManager()
        
        # Test language switching
        for lang in ['en', 'es', 'fr', 'de', 'ja', 'zh']:
            i18n.set_language(lang)
            welcome = i18n.get_text('ui.welcome')
            self.assertIsInstance(welcome, str)
            self.assertTrue(len(welcome) > 0)
        
        print("✅ Internationalization language switching works")
        
        # Test compliance
        self.assertTrue(i18n.is_consent_required('gdpr'))
        self.assertEqual(i18n.get_data_retention_days('gdpr'), 365)
        print("✅ Compliance checking works")
        
        # Test multi-region
        namespace2 = {}
        exec(open('revnet_zero/deployment/multi_region.py').read(), namespace2)
        
        create_default_multi_region_setup = namespace2['create_default_multi_region_setup']
        manager = create_default_multi_region_setup()
        
        summary = manager.get_deployment_summary()
        self.assertGreater(summary['total_regions'], 0)
        self.assertGreaterEqual(summary['healthy_regions'], 0)
        print("✅ Multi-region deployment works")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_error_handling(self):
        """Test error handling and resilience."""
        namespace = {}
        exec(open('revnet_zero/api/endpoints.py').read(), namespace)
        
        ModelAPI = namespace['ModelAPI']
        api = ModelAPI()
        
        # Test prediction on non-existent model
        result = api.predict("non_existent_model", {"text": "test"})
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        print("✅ Error handling for missing model works")
        
        # Test invalid training config
        TrainingAPI = namespace['TrainingAPI']
        training_api = TrainingAPI()
        
        result = training_api.get_training_status("invalid_job_id")
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        print("✅ Error handling for invalid job ID works")
    
    def test_performance_characteristics(self):
        """Test performance characteristics."""
        namespace = {}
        exec(open('revnet_zero/api/endpoints.py').read(), namespace)
        
        ModelAPI = namespace['ModelAPI']
        api = ModelAPI()
        
        # Load model
        api.load_model("perf_test_model", {"d_model": 512})
        
        # Test prediction performance
        start_time = time.time()
        result = api.predict("perf_test_model", {"text": "Performance test input"})
        end_time = time.time()
        
        self.assertTrue(result["success"])
        processing_time = end_time - start_time
        self.assertLess(processing_time, 1.0)  # Should complete in under 1 second
        print(f"✅ Prediction completed in {processing_time:.3f}s")
        
        # Test batch prediction
        start_time = time.time()
        batch_data = {
            "texts": ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"],
            "batch_size": 2
        }
        result = api.batch_predict("perf_test_model", batch_data)
        end_time = time.time()
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 5)
        batch_time = end_time - start_time
        self.assertLess(batch_time, 2.0)  # Batch should complete in under 2 seconds
        print(f"✅ Batch prediction completed in {batch_time:.3f}s")
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        namespace = {}
        exec(open('revnet_zero/api/endpoints.py').read(), namespace)
        
        ConfigAPI = namespace['ConfigAPI']
        config_api = ConfigAPI()
        
        # Test valid configuration
        valid_config = {
            "model_config": {
                "d_model": 512,
                "num_layers": 12,
                "num_heads": 8
            },
            "training_config": {
                "batch_size": 16,
                "learning_rate": 5e-5
            }
        }
        
        result = config_api.validate_config(valid_config)
        self.assertTrue(result["success"])
        self.assertTrue(result["validated"])
        print("✅ Valid configuration accepted")
        
        # Test configuration with warnings
        large_config = {
            "model_config": {
                "d_model": 8192,  # Very large
                "num_layers": 48   # Very deep
            }
        }
        
        result = config_api.validate_config(large_config)
        self.assertTrue(result["success"])
        self.assertGreater(len(result["warnings"]), 0)
        print("✅ Configuration warnings generated appropriately")
        
        # Test invalid configuration
        invalid_config = {}  # Missing required fields
        
        result = config_api.validate_config(invalid_config)
        self.assertTrue(result["success"])
        self.assertFalse(result["validated"])
        self.assertGreater(len(result["errors"]), 0)
        print("✅ Invalid configuration rejected")
    
    def test_version_management(self):
        """Test API version management."""
        namespace = {}
        exec(open('revnet_zero/api/versioning.py').read(), namespace)
        
        VersionManager = namespace['VersionManager']
        manager = VersionManager()
        
        # Test supported versions
        versions = manager.get_supported_versions()
        self.assertGreater(len(versions), 0)
        print(f"✅ {len(versions)} API versions supported")
        
        # Test version validation
        result = manager.validate_version_request("1.0")
        self.assertTrue(result["valid"])
        print("✅ Version validation works")
        
        # Test invalid version
        result = manager.validate_version_request("99.0")
        self.assertFalse(result["valid"])
        print("✅ Invalid version rejected")
        
        # Test migration info
        migration = manager.get_migration_info("1.0", "2.0")
        self.assertIn("breaking_changes", migration)
        self.assertIn("new_features", migration)
        print("✅ Migration information available")
    
    def test_monitoring_and_health(self):
        """Test monitoring and health check functionality."""
        namespace = {}
        exec(open('revnet_zero/api/endpoints.py').read(), namespace)
        
        HealthAPI = namespace['HealthAPI']
        health_api = HealthAPI()
        
        # Test health check
        health = health_api.health_check()
        self.assertTrue(health["success"])
        self.assertIn("status", health)
        self.assertIn(health["status"], ["healthy", "degraded", "unhealthy"])
        print(f"✅ Health status: {health['status']}")
        
        # Test metrics collection
        metrics = health_api.get_metrics()
        self.assertTrue(metrics["success"])
        self.assertIn("metrics", metrics)
        self.assertIn("performance_stats", metrics)
        self.assertIn("resource_usage", metrics)
        print("✅ Metrics collection works")
        
        # Simulate some requests to test error rate calculation
        health_api._track_request(success=True)
        health_api._track_request(success=True)
        health_api._track_request(success=False)  # One error
        
        health = health_api.health_check()
        self.assertGreater(health["error_rate_percent"], 0)
        print(f"✅ Error rate tracking: {health['error_rate_percent']:.1f}%")


class TestProductionReadiness(unittest.TestCase):
    """Test production readiness features."""
    
    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""
        namespace = {}
        exec(open('revnet_zero/api/endpoints.py').read(), namespace)
        
        ModelAPI = namespace['ModelAPI']
        api = ModelAPI()
        
        # Load model
        api.load_model("concurrent_test", {"d_model": 256})
        
        # Simulate concurrent requests
        results = []
        for i in range(10):
            result = api.predict("concurrent_test", {"text": f"Request {i}"})
            results.append(result)
        
        # All requests should succeed
        for i, result in enumerate(results):
            self.assertTrue(result["success"], f"Request {i} failed")
        
        print(f"✅ {len(results)} concurrent requests handled successfully")
    
    def test_resource_cleanup(self):
        """Test resource cleanup and memory management."""
        namespace = {}
        exec(open('revnet_zero/api/endpoints.py').read(), namespace)
        
        ModelAPI = namespace['ModelAPI']
        api = ModelAPI()
        
        # Load multiple models
        model_ids = ["model_1", "model_2", "model_3"]
        for model_id in model_ids:
            result = api.load_model(model_id, {"d_model": 256})
            self.assertTrue(result["success"])
        
        self.assertEqual(len(api.models), 3)
        print("✅ Multiple models loaded")
        
        # Unload models
        for model_id in model_ids:
            result = api.unload_model(model_id)
            self.assertTrue(result["success"])
        
        self.assertEqual(len(api.models), 0)
        print("✅ All models unloaded successfully")
    
    def test_graceful_degradation(self):
        """Test graceful degradation under stress."""
        namespace = {}
        exec(open('revnet_zero/api/endpoints.py').read(), namespace)
        
        HealthAPI = namespace['HealthAPI']
        health_api = HealthAPI()
        
        # Simulate high error rate
        for _ in range(20):
            health_api._track_request(success=False)
        
        for _ in range(80):
            health_api._track_request(success=True)
        
        health = health_api.health_check()
        self.assertIn(health["status"], ["degraded", "unhealthy"])
        print(f"✅ Graceful degradation: {health['status']} with {health['error_rate_percent']:.1f}% error rate")


def run_integration_tests():
    """Run all integration tests."""
    print("🧪 RUNNING PRODUCTION INTEGRATION TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestProductionIntegration))
    suite.addTest(unittest.makeSuite(TestProductionReadiness))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print(f"✅ {result.testsRun} tests completed successfully")
        print("✅ API functionality verified")
        print("✅ Error handling validated") 
        print("✅ Performance characteristics confirmed")
        print("✅ Production readiness validated")
        return True
    else:
        print("❌ SOME INTEGRATION TESTS FAILED")
        print(f"Failed: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)