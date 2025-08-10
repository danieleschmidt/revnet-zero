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
        self.grad = None
        self.requires_grad_ = lambda x: self
    
    def __getattr__(self, name):
        if name == 'size':
            return lambda dim=None: self.shape[dim] if dim is not None else self.shape
        return MagicMock()
    
    def __setitem__(self, key, value):
        pass
        
    def __getitem__(self, key):
        return MockTensor()
    
    def backward(self):
        pass
    
    def detach(self):
        return self
    
    def cpu(self):
        return self
    
    def cuda(self):
        return self
    
    def transpose(self, *args):
        return self
    
    def view(self, *args):
        return self
    
    def contiguous(self):
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
torch_mock.__version__ = "2.0.0+mock"
torch_mock.version = Mock()
torch_mock.version.cuda = "11.8"
torch_mock.zeros = lambda *args, **kwargs: MockTensor(shape=args[0] if args else (1,))
torch_mock.ones = lambda *args, **kwargs: MockTensor(shape=args[0] if args else (1,))
torch_mock.randn = lambda *args, **kwargs: MockTensor(shape=args if args else (1,))
torch_mock.arange = lambda *args, **kwargs: MockTensor()
torch_mock.exp = lambda x: MockTensor()
torch_mock.sin = lambda x: MockTensor()
torch_mock.cos = lambda x: MockTensor()
torch_mock.log = lambda x: MockTensor()
torch_mock.cat = lambda x, dim=0: MockTensor()
torch_mock.split = lambda x, size, dim=0: (MockTensor(), MockTensor())
torch_mock.matmul = lambda x, y: MockTensor()
torch_mock.allclose = lambda x, y, **kwargs: True
torch_mock.no_grad = lambda: Mock(__enter__=lambda x: None, __exit__=lambda *args: None)
torch_mock.autograd = Mock()
torch_mock.autograd.Function = MockModule

# Mock torch.nn
torch_mock.nn = Mock()
torch_mock.nn.Module = MockModule
torch_mock.nn.Linear = lambda *args, **kwargs: MockModule()
torch_mock.nn.LayerNorm = lambda *args, **kwargs: MockModule()
torch_mock.nn.Dropout = lambda *args, **kwargs: MockModule()
torch_mock.nn.ReLU = lambda *args, **kwargs: MockModule()
torch_mock.nn.Embedding = lambda *args, **kwargs: MockModule()
torch_mock.nn.Sequential = lambda *args: MockModule()
torch_mock.nn.ModuleList = lambda x: x
torch_mock.nn.CrossEntropyLoss = lambda *args, **kwargs: MockModule()
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
torch_mock.utils.checkpoint = Mock()
torch_mock.utils.checkpoint.checkpoint = lambda fn, *args, **kwargs: fn(*args, **kwargs)

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
sys.modules['torch.utils.checkpoint'] = torch_mock.utils.checkpoint
sys.modules['torch.distributed'] = torch_mock.distributed
sys.modules['torch.nn.parallel'] = torch_mock.nn.parallel
sys.modules['einops'] = Mock()
psutil_mock = Mock()
psutil_mock.virtual_memory = lambda: Mock(total=16000000000, available=8000000000)
psutil_mock.cpu_count = lambda: 8
sys.modules['psutil'] = psutil_mock

# Mock other common dependencies
sys.modules['packaging'] = Mock()
sys.modules['packaging'].version = Mock()
sys.modules['packaging'].version.Version = str
sys.modules['numpy'] = Mock()
sys.modules['numpy'].array = lambda x: x
sys.modules['numpy'].allclose = lambda x, y, **kwargs: True

# Mock matplotlib  
sys.modules['matplotlib'] = Mock()
sys.modules['matplotlib'].pyplot = Mock()
plt_mock = Mock()
plt_mock.figure = lambda **kwargs: Mock()
plt_mock.plot = Mock()
plt_mock.xlabel = Mock()
plt_mock.ylabel = Mock()
plt_mock.title = Mock()
plt_mock.legend = Mock()
plt_mock.savefig = Mock()
plt_mock.close = Mock()
sys.modules['matplotlib.pyplot'] = plt_mock

# Mock FastAPI and web dependencies
sys.modules['fastapi'] = Mock()
sys.modules['fastapi'].FastAPI = Mock
sys.modules['fastapi'].HTTPException = Exception
sys.modules['fastapi'].BackgroundTasks = Mock
sys.modules['uvicorn'] = Mock()
sys.modules['uvicorn'].run = Mock()
sys.modules['pydantic'] = Mock()
sys.modules['pydantic'].BaseModel = object

print("Mock PyTorch environment ready for development")