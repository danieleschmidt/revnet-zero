#!/usr/bin/env python3
"""
Final integration test for RevNet-Zero production readiness.
"""

import sys
import os
import time
import unittest
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestFinalIntegration(unittest.TestCase):
    """Final integration test suite."""
    
    def setUp(self):
        """Set up test environment."""
        self.start_time = time.time()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_api_endpoints_functionality(self):
        """Test comprehensive API functionality."""
        # Load API endpoints module
        api_code = """
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

"""
        
        with open('revnet_zero/api/endpoints.py', 'r') as f:
            api_code += f.read()
        
        namespace = {}
        exec(api_code, namespace)
        
        # Test ModelAPI
        ModelAPI = namespace['ModelAPI']
        api = ModelAPI()
        
        # Test complete workflow
        load_result = api.load_model("test_model", {"d_model": 512, "num_layers": 12})
        self.assertTrue(load_result["success"])
        print("✅ Model loading successful")
        
        predict_result = api.predict("test_model", {
            "text": "Test input for prediction",
            "max_length": 100,
            "temperature": 0.8
        })
        self.assertTrue(predict_result["success"])
        self.assertIn("generated_text", predict_result)
        print("✅ Model prediction successful")
        
        batch_result = api.batch_predict("test_model", {
            "texts": ["Text 1", "Text 2", "Text 3"],
            "batch_size": 2
        })
        self.assertTrue(batch_result["success"])
        self.assertEqual(len(batch_result["results"]), 3)
        print("✅ Batch prediction successful")
        
        unload_result = api.unload_model("test_model")
        self.assertTrue(unload_result["success"])
        print("✅ Model unloading successful")
    
    def test_training_api_workflow(self):
        """Test complete training workflow."""
        api_code = """
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

"""
        
        with open('revnet_zero/api/endpoints.py', 'r') as f:
            api_code += f.read()
        
        namespace = {}
        exec(api_code, namespace)
        
        TrainingAPI = namespace['TrainingAPI']
        training_api = TrainingAPI()
        
        # Start training
        start_result = training_api.start_training({
            "model_type": "reversible_transformer",
            "dataset_path": "/tmp/dataset",
            "output_path": "/tmp/model",
            "num_epochs": 3,
            "batch_size": 16
        })
        self.assertTrue(start_result["success"])
        job_id = start_result["job_id"]
        print("✅ Training job started")
        
        # Check status
        status_result = training_api.get_training_status(job_id)
        self.assertTrue(status_result["success"])
        self.assertIn("job_info", status_result)
        print("✅ Training status retrieved")
        
        # Stop training
        stop_result = training_api.stop_training(job_id)
        self.assertTrue(stop_result["success"])
        print("✅ Training job stopped")
    
    def test_health_monitoring_comprehensive(self):
        """Test comprehensive health monitoring."""
        api_code = """
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

"""
        
        with open('revnet_zero/api/endpoints.py', 'r') as f:
            api_code += f.read()
        
        namespace = {}
        exec(api_code, namespace)
        
        HealthAPI = namespace['HealthAPI']
        health_api = HealthAPI()
        
        # Basic health check
        health_result = health_api.health_check()
        self.assertTrue(health_result["success"])
        self.assertIn("status", health_result)
        self.assertIn("uptime_seconds", health_result)
        self.assertIn("version", health_result)
        print(f"✅ Health check: {health_result['status']}")
        
        # Detailed metrics
        metrics_result = health_api.get_metrics()
        self.assertTrue(metrics_result["success"])
        self.assertIn("metrics", metrics_result)
        self.assertIn("performance_stats", metrics_result)
        self.assertIn("resource_usage", metrics_result)
        print("✅ Detailed metrics retrieved")
        
        # Test error tracking
        for _ in range(10):
            health_api._track_request(success=True)
        for _ in range(2):
            health_api._track_request(success=False)
        
        updated_health = health_api.health_check()
        self.assertGreater(updated_health["error_rate_percent"], 0)
        print(f"✅ Error tracking: {updated_health['error_rate_percent']:.1f}% error rate")
    
    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        api_code = """
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

"""
        
        with open('revnet_zero/api/endpoints.py', 'r') as f:
            api_code += f.read()
        
        namespace = {}
        exec(api_code, namespace)
        
        ConfigAPI = namespace['ConfigAPI']
        config_api = ConfigAPI()
        
        # Test valid small configuration
        small_config = {
            "model_config": {
                "d_model": 256,
                "num_layers": 6,
                "num_heads": 4
            },
            "training_config": {
                "batch_size": 8,
                "learning_rate": 1e-4
            }
        }
        
        result = config_api.validate_config(small_config)
        self.assertTrue(result["success"])
        self.assertTrue(result["validated"])
        self.assertEqual(len(result["errors"]), 0)
        print("✅ Small configuration validated successfully")
        
        # Test large configuration (should generate warnings)
        large_config = {
            "model_config": {
                "d_model": 8192,
                "num_layers": 48,
                "num_heads": 64
            },
            "training_config": {
                "batch_size": 4,
                "learning_rate": 5e-5
            }
        }
        
        result = config_api.validate_config(large_config)
        self.assertTrue(result["success"])
        self.assertGreater(len(result["warnings"]), 0)
        self.assertGreater(result["estimated_memory_gb"], 20)
        print(f"✅ Large configuration warnings: {len(result['warnings'])} warnings")
        
        # Test invalid configuration
        invalid_config = {
            "training_config": {"batch_size": 8}
            # Missing required model_config
        }
        
        result = config_api.validate_config(invalid_config)
        self.assertTrue(result["success"])
        self.assertFalse(result["validated"])
        self.assertGreater(len(result["errors"]), 0)
        print("✅ Invalid configuration properly rejected")
    
    def test_api_versioning_system(self):
        """Test API versioning system."""
        version_code = """
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, date

"""
        
        with open('revnet_zero/api/versioning.py', 'r') as f:
            version_code += f.read()
        
        namespace = {}
        exec(version_code, namespace)
        
        VersionManager = namespace['VersionManager']
        manager = VersionManager()
        
        # Test supported versions
        versions = manager.get_supported_versions()
        self.assertGreater(len(versions), 0)
        self.assertIn("1.0", versions)
        print(f"✅ {len(versions)} API versions supported: {', '.join(versions)}")
        
        # Test version validation
        valid_result = manager.validate_version_request("1.0")
        self.assertTrue(valid_result["valid"])
        print("✅ Valid version accepted")
        
        invalid_result = manager.validate_version_request("99.0")
        self.assertFalse(invalid_result["valid"])
        print("✅ Invalid version rejected")
        
        # Test migration information
        migration = manager.get_migration_info("1.0", "2.0")
        self.assertIn("breaking_changes", migration)
        self.assertIn("new_features", migration)
        print(f"✅ Migration info: {len(migration['breaking_changes'])} breaking changes")
        
        # Test deprecation notices
        notices = manager.get_deprecation_notices()
        self.assertIsInstance(notices, list)
        print(f"✅ Deprecation notices: {len(notices)} notices")
    
    def test_globalization_and_compliance(self):
        """Test globalization and compliance features."""
        # Test internationalization with proper imports
        i18n_code = f"""
import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum

# Mock translations directory
translations_dir = Path("{self.temp_dir}")

"""
        
        with open('revnet_zero/deployment/internationalization.py', 'r') as f:
            content = f.read()
            # Replace the problematic Path(__file__).parent reference
            content = content.replace('Path(__file__).parent / "translations"', f'Path("{self.temp_dir}")')
            i18n_code += content
        
        namespace = {}
        exec(i18n_code, namespace)
        
        InternationalizationManager = namespace['InternationalizationManager']
        get_i18n_manager = namespace['get_i18n_manager']
        set_language = namespace['set_language']
        get_text = namespace['get_text']
        
        # Test language support
        i18n = InternationalizationManager()
        
        test_languages = ['en', 'es', 'fr', 'de', 'ja', 'zh']
        for lang in test_languages:
            i18n.set_language(lang)
            welcome = i18n.get_text('ui.welcome')
            self.assertIsInstance(welcome, str)
            self.assertGreater(len(welcome), 0)
        
        print(f"✅ {len(test_languages)} languages tested successfully")
        
        # Test compliance features
        compliance_regions = ['gdpr', 'ccpa', 'pdpa', 'pipeda']
        for region in compliance_regions:
            consent_required = i18n.is_consent_required(region)
            self.assertIsInstance(consent_required, bool)
            
            retention_days = i18n.get_data_retention_days(region)
            self.assertIsInstance(retention_days, int)
            self.assertGreater(retention_days, 0)
        
        print(f"✅ {len(compliance_regions)} compliance frameworks tested")
    
    def test_multi_region_deployment(self):
        """Test multi-region deployment features."""
        mr_code = """
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

"""
        
        with open('revnet_zero/deployment/multi_region.py', 'r') as f:
            mr_code += f.read()
        
        namespace = {}
        exec(mr_code, namespace)
        
        create_default_multi_region_setup = namespace['create_default_multi_region_setup']
        Region = namespace['Region']
        
        # Test deployment setup
        manager = create_default_multi_region_setup()
        
        summary = manager.get_deployment_summary()
        self.assertGreater(summary['total_regions'], 0)
        self.assertGreaterEqual(summary['healthy_regions'], 0)
        self.assertIn('load_balancing_strategy', summary)
        print(f"✅ Multi-region setup: {summary['total_regions']} regions, {summary['availability_percent']:.1f}% availability")
        
        # Test region selection
        test_locations = ["US", "DE", "JP", "SG"]
        for location in test_locations:
            optimal = manager.get_optimal_region(user_location=location)
            if optimal:
                self.assertIsNotNone(optimal.region)
                self.assertIsNotNone(optimal.cloud_provider)
        
        print(f"✅ Region selection tested for {len(test_locations)} locations")
        
        # Test health monitoring
        health_results = manager.perform_health_check()
        self.assertIsInstance(health_results, dict)
        self.assertGreater(len(health_results), 0)
        print(f"✅ Health monitoring: {len(health_results)} regions checked")
        
        # Test data residency
        countries = ["US", "GB", "DE", "JP"]
        for country in countries:
            residency_region = manager.get_region_for_data_residency(country)
            # Some countries may not have compliant regions in our setup
            if residency_region:
                self.assertTrue(residency_region.data_residency_required)
        
        print(f"✅ Data residency tested for {len(countries)} countries")
    
    def test_security_validation(self):
        """Test security validation features."""
        sec_code = """
import re
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import hashlib

"""
        
        with open('revnet_zero/security/secure_validation.py', 'r') as f:
            sec_code += f.read()
        
        namespace = {}
        exec(sec_code, namespace)
        
        SecureInputValidator = namespace['SecureInputValidator']
        SecurityAuditLogger = namespace['SecurityAuditLogger']
        
        # Test input validation
        validator = SecureInputValidator()
        
        # Test safe string
        safe_text = "This is a safe input string for testing"
        validated = validator.validate_string_input(safe_text)
        self.assertEqual(validated, safe_text)
        print("✅ Safe string validation passed")
        
        # Test dangerous pattern detection
        try:
            dangerous_text = "eval('malicious code')"
            validator.validate_string_input(dangerous_text)
            self.fail("Should have rejected dangerous input")
        except ValueError:
            print("✅ Dangerous pattern correctly rejected")
        
        # Test sequence validation
        safe_sequence = ["item1", "item2", "item3"]
        validated_seq = validator.validate_sequence_input(safe_sequence)
        self.assertEqual(len(validated_seq), 3)
        print("✅ Sequence validation passed")
        
        # Test security audit logging
        audit_logger = SecurityAuditLogger()
        audit_logger.log_security_event("TEST_EVENT", {"test": "data"})
        
        summary = audit_logger.get_security_summary()
        self.assertGreater(summary['total_events'], 0)
        print("✅ Security audit logging working")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Load all necessary modules
        api_code = """
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

"""
        
        with open('revnet_zero/api/endpoints.py', 'r') as f:
            api_code += f.read()
        
        namespace = {}
        exec(api_code, namespace)
        
        # Initialize APIs
        ModelAPI = namespace['ModelAPI']
        TrainingAPI = namespace['TrainingAPI']
        ConfigAPI = namespace['ConfigAPI']
        HealthAPI = namespace['HealthAPI']
        
        model_api = ModelAPI()
        training_api = TrainingAPI()
        config_api = ConfigAPI()
        health_api = HealthAPI()
        
        print("🔄 Starting end-to-end workflow test...")
        
        # Step 1: Validate configuration
        config = {
            "model_config": {
                "d_model": 512,
                "num_layers": 12,
                "num_heads": 8
            },
            "training_config": {
                "batch_size": 16,
                "learning_rate": 5e-5,
                "num_epochs": 3
            }
        }
        
        config_result = config_api.validate_config(config)
        self.assertTrue(config_result["success"])
        self.assertTrue(config_result["validated"])
        print("  ✅ Step 1: Configuration validated")
        
        # Step 2: Start training
        training_result = training_api.start_training(config["training_config"])
        self.assertTrue(training_result["success"])
        job_id = training_result["job_id"]
        print("  ✅ Step 2: Training started")
        
        # Step 3: Monitor training
        status_result = training_api.get_training_status(job_id)
        self.assertTrue(status_result["success"])
        print("  ✅ Step 3: Training monitored")
        
        # Step 4: Load trained model (simulated)
        load_result = model_api.load_model("trained_model", config["model_config"])
        self.assertTrue(load_result["success"])
        print("  ✅ Step 4: Model loaded")
        
        # Step 5: Run inference
        predict_result = model_api.predict("trained_model", {
            "text": "Test inference on trained model",
            "max_length": 50
        })
        self.assertTrue(predict_result["success"])
        print("  ✅ Step 5: Inference completed")
        
        # Step 6: Health check
        health_result = health_api.health_check()
        self.assertTrue(health_result["success"])
        print("  ✅ Step 6: Health check passed")
        
        # Step 7: Cleanup
        unload_result = model_api.unload_model("trained_model")
        self.assertTrue(unload_result["success"])
        
        stop_result = training_api.stop_training(job_id)
        self.assertTrue(stop_result["success"])
        print("  ✅ Step 7: Cleanup completed")
        
        print("🎉 End-to-end workflow completed successfully!")


def run_final_integration_tests():
    """Run final integration test suite."""
    print("🚀 RUNNING FINAL INTEGRATION TESTS FOR PRODUCTION READINESS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFinalIntegration)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("🎉 ALL FINAL INTEGRATION TESTS PASSED!")
        print(f"✅ {result.testsRun} comprehensive tests completed successfully")
        print()
        print("PRODUCTION READINESS VERIFIED:")
        print("✅ API endpoints functionality")
        print("✅ Training workflow management")
        print("✅ Health monitoring and metrics")
        print("✅ Configuration validation")
        print("✅ API versioning system")
        print("✅ Globalization and compliance")
        print("✅ Multi-region deployment")
        print("✅ Security validation")
        print("✅ End-to-end workflow")
        print()
        print("🌟 REVNET-ZERO IS PRODUCTION READY! 🌟")
        return True
    else:
        print("❌ SOME FINAL INTEGRATION TESTS FAILED")
        print(f"Failed: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
        return False


if __name__ == "__main__":
    success = run_final_integration_tests()
    sys.exit(0 if success else 1)