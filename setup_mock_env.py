"""
Setup mock environment for RevNet-Zero development and testing.
This ensures imports work correctly without requiring heavy dependencies.
"""

import sys
from unittest.mock import MagicMock, Mock


def setup_mock_torch():
    """Setup mock PyTorch environment."""
    
    # Create mock torch module
    mock_torch = MagicMock()
    
    # Mock tensor class
    class MockTensor:
        def __init__(self, *args, **kwargs):
            self.shape = kwargs.get('shape', (1, 1))
            self.dtype = kwargs.get('dtype', 'float32')
            self.device = kwargs.get('device', 'cpu')
            self.grad = None
        
        def __call__(self, *args, **kwargs):
            return MockTensor()
        
        def size(self, dim=None):
            return self.shape if dim is None else self.shape[dim] if dim < len(self.shape) else 1
        
        def view(self, *shape):
            result = MockTensor()
            result.shape = shape
            return result
        
        def to(self, *args, **kwargs):
            return self
        
        def cuda(self):
            result = MockTensor()
            result.device = 'cuda'
            return result
        
        def cpu(self):
            result = MockTensor()
            result.device = 'cpu'
            return result
        
        def backward(self):
            pass
        
        def zero_grad(self):
            pass
    
    # Setup torch module attributes
    mock_torch.Tensor = MockTensor
    mock_torch.tensor = lambda *args, **kwargs: MockTensor()
    mock_torch.zeros = lambda *args, **kwargs: MockTensor()
    mock_torch.ones = lambda *args, **kwargs: MockTensor()
    mock_torch.randn = lambda *args, **kwargs: MockTensor()
    mock_torch.rand = lambda *args, **kwargs: MockTensor()
    mock_torch.empty = lambda *args, **kwargs: MockTensor()
    
    # Mock nn module
    mock_nn = MagicMock()
    mock_nn.Module = Mock
    mock_nn.Linear = lambda *args, **kwargs: Mock()
    mock_nn.LayerNorm = lambda *args, **kwargs: Mock()
    mock_nn.Dropout = lambda *args, **kwargs: Mock()
    mock_nn.GELU = lambda *args, **kwargs: Mock()
    
    # Mock functional with specific methods
    mock_functional = MagicMock()
    mock_functional.gelu = lambda x: x
    mock_functional.relu = lambda x: x
    mock_functional.dropout = lambda x, *args, **kwargs: x
    mock_functional.softmax = lambda x, *args, **kwargs: x
    mock_functional.layer_norm = lambda x, *args, **kwargs: x
    mock_functional.scaled_dot_product_attention = lambda q, k, v, *args, **kwargs: v
    mock_nn.functional = mock_functional
    
    # Mock nn.parallel
    mock_parallel = MagicMock()
    mock_parallel.DistributedDataParallel = lambda model, *args, **kwargs: model
    mock_nn.parallel = mock_parallel
    
    mock_torch.nn = mock_nn
    
    # Mock optim
    mock_optim = MagicMock()
    mock_optim.Adam = lambda *args, **kwargs: Mock()
    mock_optim.AdamW = lambda *args, **kwargs: Mock()
    mock_torch.optim = mock_optim
    
    # Mock cuda
    mock_cuda = MagicMock()
    mock_cuda.is_available = lambda: True
    mock_cuda.device_count = lambda: 1
    mock_cuda.max_memory_allocated = lambda: 1024**3
    mock_torch.cuda = mock_cuda
    
    # Mock utils
    mock_utils = MagicMock()
    mock_utils.data = MagicMock()
    mock_utils.data.DataLoader = lambda *args, **kwargs: Mock()
    mock_torch.utils = mock_utils
    
    # Mock distributed
    mock_distributed = MagicMock()
    mock_distributed.is_available = lambda: True
    mock_distributed.is_initialized = lambda: False
    mock_distributed.init_process_group = lambda *args, **kwargs: None
    mock_distributed.get_world_size = lambda: 1
    mock_distributed.get_rank = lambda: 0
    mock_torch.distributed = mock_distributed
    
    # Register in sys.modules
    sys.modules['torch'] = mock_torch
    sys.modules['torch.nn'] = mock_nn
    sys.modules['torch.nn.functional'] = mock_functional
    sys.modules['torch.nn.parallel'] = mock_parallel
    sys.modules['torch.optim'] = mock_optim
    sys.modules['torch.cuda'] = mock_cuda
    sys.modules['torch.distributed'] = mock_distributed
    sys.modules['torch.utils'] = mock_utils
    sys.modules['torch.utils.data'] = mock_utils.data
    
    print("Mock PyTorch environment ready for development")
    
    return mock_torch


def setup_mock_dependencies():
    """Setup other mock dependencies."""
    
    # Mock einops
    mock_einops = MagicMock()
    mock_einops.rearrange = lambda x, *args, **kwargs: x
    mock_einops.reduce = lambda x, *args, **kwargs: x
    sys.modules['einops'] = mock_einops
    
    # Mock psutil
    mock_psutil = MagicMock()
    mock_psutil.virtual_memory = lambda: Mock(available=8*1024**3, total=16*1024**3)
    mock_psutil.cpu_count = lambda: 8
    sys.modules['psutil'] = mock_psutil
    
    # Mock packaging
    mock_packaging = MagicMock()
    sys.modules['packaging'] = mock_packaging
    
    # Mock triton (optional)
    mock_triton = MagicMock()
    sys.modules['triton'] = mock_triton
    
    # Mock jax (optional)
    mock_jax = MagicMock()
    sys.modules['jax'] = mock_jax
    sys.modules['jax.numpy'] = MagicMock()
    
    # Mock web frameworks
    mock_fastapi = MagicMock()
    mock_fastapi.FastAPI = lambda *args, **kwargs: Mock()
    mock_fastapi.HTTPException = Exception
    mock_fastapi.BackgroundTasks = Mock
    sys.modules['fastapi'] = mock_fastapi
    
    mock_uvicorn = MagicMock()
    sys.modules['uvicorn'] = mock_uvicorn
    
    mock_pydantic = MagicMock()
    sys.modules['pydantic'] = mock_pydantic
    
    # Mock numpy (basic)
    try:
        import numpy as np
    except ImportError:
        mock_numpy = MagicMock()
        mock_numpy.array = lambda *args: Mock()
        mock_numpy.zeros = lambda *args: Mock()
        mock_numpy.ones = lambda *args: Mock()
        mock_numpy.exp = lambda x: x
        sys.modules['numpy'] = mock_numpy
        sys.modules['np'] = mock_numpy


def setup_full_mock_environment():
    """Setup complete mock environment for testing."""
    setup_mock_torch()
    setup_mock_dependencies()
    print("Complete mock environment initialized")


if __name__ == "__main__":
    setup_full_mock_environment()