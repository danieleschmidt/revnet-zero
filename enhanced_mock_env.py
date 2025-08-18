#!/usr/bin/env python3
"""
Enhanced Mock Environment for RevNet-Zero Development
Provides better mocks for numpy and yaml
"""

import sys
import os
from unittest.mock import MagicMock
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_enhanced_numpy_mock():
    """Create enhanced numpy mock with basic functionality."""
    mock_numpy = MagicMock()
    mock_numpy.__name__ = 'numpy'
    mock_numpy.__version__ = '1.24.0'
    
    # Mock basic numpy functions
    mock_numpy.array = lambda x, dtype=None: x
    mock_numpy.zeros = lambda *args, **kwargs: [0] * (args[0] if args else 10)
    mock_numpy.ones = lambda *args, **kwargs: [1] * (args[0] if args else 10)
    mock_numpy.empty = lambda *args, **kwargs: [0] * (args[0] if args else 10)
    mock_numpy.arange = lambda *args, **kwargs: list(range(args[0] if args else 10))
    
    # Mock data types
    mock_numpy.float32 = 'float32'
    mock_numpy.float64 = 'float64'
    mock_numpy.int32 = 'int32'
    mock_numpy.int64 = 'int64'
    mock_numpy.bool_ = bool
    
    # Mock mathematical functions
    mock_numpy.sqrt = lambda x: x
    mock_numpy.mean = lambda x: sum(x) / len(x) if hasattr(x, '__len__') else x
    mock_numpy.sum = lambda x: sum(x) if hasattr(x, '__iter__') else x
    mock_numpy.exp = lambda x: x
    mock_numpy.log = lambda x: x
    
    # Mock shape and reshaping
    mock_numpy.shape = lambda x: getattr(x, 'shape', (10,))
    mock_numpy.reshape = lambda x, shape: x
    
    return mock_numpy

def create_enhanced_yaml_mock():
    """Create enhanced YAML mock with JSON fallback."""
    mock_yaml = MagicMock()
    mock_yaml.__name__ = 'yaml'
    
    def safe_load(content):
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Simple YAML-like parsing for basic cases
                result = {}
                for line in content.strip().split('\n'):
                    if ':' in line and not line.strip().startswith('#'):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        # Try to convert value
                        try:
                            if value.lower() in ('true', 'false'):
                                value = value.lower() == 'true'
                            elif value.isdigit():
                                value = int(value)
                            elif '.' in value and value.replace('.', '').isdigit():
                                value = float(value)
                        except:
                            pass
                        result[key] = value
                return result
        return content

    def dump(data, stream=None, **kwargs):
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                lines.append(f"{key}: {value}")
            result = '\n'.join(lines)
        else:
            result = json.dumps(data, indent=2)
        
        if stream:
            stream.write(result)
        return result

    mock_yaml.safe_load = safe_load
    mock_yaml.load = safe_load
    mock_yaml.dump = dump
    mock_yaml.safe_dump = dump
    
    return mock_yaml

def setup_enhanced_mocks():
    """Set up enhanced mocks for missing dependencies."""
    
    # Enhanced NumPy mock
    if 'numpy' not in sys.modules:
        numpy_mock = create_enhanced_numpy_mock()
        sys.modules['numpy'] = numpy_mock
        sys.modules['np'] = numpy_mock
        print("✅ Enhanced NumPy mock activated")
    
    # Enhanced YAML mock
    if 'yaml' not in sys.modules:
        yaml_mock = create_enhanced_yaml_mock()
        sys.modules['yaml'] = yaml_mock
        print("✅ Enhanced YAML mock activated")
    
    # Initialize PyTorch mock from existing setup
    if os.path.exists('setup_mock_env.py'):
        with open('setup_mock_env.py', 'r') as f:
            setup_code = f.read()
        exec(setup_code)
    
    print("🚀 Enhanced mock environment ready")

if __name__ == "__main__":
    setup_enhanced_mocks()