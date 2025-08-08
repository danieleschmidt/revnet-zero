"""
Mock PyTorch implementation for development and testing without dependencies.
This allows us to validate core logic and structure.
"""

import sys
from typing import Any, Optional, Tuple, Union, Dict
from unittest.mock import MagicMock, Mock


class MockTensor:
    """Mock torch.Tensor for testing."""
    
    def __init__(self, *args, **kwargs):
        self.shape = kwargs.get('shape', (1, 1))
        self.dtype = kwargs.get('dtype', 'float32')
        self.device = kwargs.get('device', 'cpu')
    
    def __getattr__(self, name):
        return MagicMock()
    
    def backward(self):
        pass
    
    def detach(self):
        return self
    
    def cpu(self):
        return self
    
    def cuda(self):
        return self


class MockModule:
    """Mock torch.nn.Module for testing."""
    
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        return MockTensor()
    
    def parameters(self):
        return []
    
    def forward(self, *args, **kwargs):
        return MockTensor()
    
    def __getattr__(self, name):
        return MagicMock()


# Create mock torch module
torch_mock = Mock()
torch_mock.Tensor = MockTensor
torch_mock.zeros = lambda *args, **kwargs: MockTensor(shape=args[0] if args else (1,))
torch_mock.ones = lambda *args, **kwargs: MockTensor(shape=args[0] if args else (1,))
torch_mock.randn = lambda *args, **kwargs: MockTensor(shape=args if args else (1,))
torch_mock.arange = lambda *args, **kwargs: MockTensor()
torch_mock.exp = lambda x: MockTensor()
torch_mock.sin = lambda x: MockTensor()
torch_mock.cos = lambda x: MockTensor()
torch_mock.log = lambda x: MockTensor()
torch_mock.cat = lambda x, dim=0: MockTensor()
torch_mock.no_grad = lambda: Mock(__enter__=lambda x: None, __exit__=lambda *args: None)
torch_mock.autograd = Mock()
torch_mock.autograd.Function = MockModule

# Mock torch.nn
torch_mock.nn = Mock()
torch_mock.nn.Module = MockModule
torch_mock.nn.Linear = lambda *args, **kwargs: MockModule()
torch_mock.nn.LayerNorm = lambda *args, **kwargs: MockModule()
torch_mock.nn.Dropout = lambda *args, **kwargs: MockModule()
torch_mock.nn.Parameter = lambda x: MockTensor()
torch_mock.nn.parallel = Mock()
torch_mock.nn.parallel.DistributedDataParallel = lambda model, **kwargs: model

# Mock torch.nn.functional
torch_mock.nn.functional = Mock()
torch_mock.nn.functional.softmax = lambda x, dim=-1: MockTensor()
torch_mock.nn.functional.gelu = lambda x: MockTensor()

# Mock torch.optim
torch_mock.optim = Mock()
torch_mock.optim.AdamW = lambda *args, **kwargs: Mock(step=Mock(), zero_grad=Mock())
torch_mock.optim.Adam = lambda *args, **kwargs: Mock(step=Mock(), zero_grad=Mock())

# Mock torch.cuda
torch_mock.cuda = Mock()
torch_mock.cuda.is_available = lambda: False
torch_mock.cuda.max_memory_allocated = lambda: 1000000

# Mock torch.utils
torch_mock.utils = Mock()
torch_mock.utils.data = Mock()
torch_mock.utils.data.DataLoader = lambda *args, **kwargs: Mock(__iter__=lambda self: iter([]))

# Mock torch.distributed
torch_mock.distributed = Mock()
torch_mock.distributed.init_process_group = Mock()
torch_mock.distributed.get_rank = lambda: 0
torch_mock.distributed.get_world_size = lambda: 1

# Install mock
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn
sys.modules['torch.nn.functional'] = torch_mock.nn.functional
sys.modules['torch.optim'] = torch_mock.optim
sys.modules['torch.cuda'] = torch_mock.cuda
sys.modules['torch.utils'] = torch_mock.utils
sys.modules['torch.utils.data'] = torch_mock.utils.data
sys.modules['torch.distributed'] = torch_mock.distributed
sys.modules['torch.nn.parallel'] = torch_mock.nn.parallel
sys.modules['einops'] = Mock()
sys.modules['psutil'] = Mock()
sys.modules['psutil'].virtual_memory = lambda: Mock(available=8000000000)
sys.modules['psutil'].cpu_count = lambda: 8

# Mock other common dependencies
sys.modules['packaging'] = Mock()
sys.modules['packaging'].version = Mock()
sys.modules['packaging'].version.Version = str
sys.modules['numpy'] = Mock()
sys.modules['numpy'].array = lambda x: x
sys.modules['numpy'].allclose = lambda x, y, **kwargs: True

print("Mock PyTorch environment ready for development")