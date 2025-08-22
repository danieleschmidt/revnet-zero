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
            self.requires_grad = kwargs.get('requires_grad', False)
            self.data = self  # Self-reference for .data attribute
        
        def __call__(self, *args, **kwargs):
            return MockTensor()
        
        def __getitem__(self, key):
            return MockTensor()
        
        def __setitem__(self, key, value):
            pass  # Mock assignment
        
        def size(self, dim=None):
            return self.shape if dim is None else self.shape[dim] if dim < len(self.shape) else 1
        
        def view(self, *shape):
            result = MockTensor()
            result.shape = shape
            return result
        
        def unsqueeze(self, dim):
            new_shape = list(self.shape)
            new_shape.insert(dim, 1)
            result = MockTensor()
            result.shape = tuple(new_shape)
            return result
        
        def transpose(self, dim0, dim1):
            return MockTensor()
        
        def expand(self, *size):
            result = MockTensor()
            result.shape = size
            return result
        
        def contiguous(self):
            return self
        
        def masked_fill(self, mask, value):
            return MockTensor()
        
        def chunk(self, chunks, dim=0):
            return [MockTensor() for _ in range(chunks)]
        
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
        
        def requires_grad_(self, requires_grad=True):
            self.requires_grad = requires_grad
            return self
        
        def item(self):
            return 0.5  # Mock scalar value
        
        def mean(self):
            return MockTensor()
        
        def numel(self):
            return 1000  # Mock parameter count
        
        def __add__(self, other):
            return MockTensor()
        
        def __mul__(self, other):
            return MockTensor()
        
        def __matmul__(self, other):
            return MockTensor()
        
        def float(self):
            return MockTensor()
        
        def long(self):
            return MockTensor()
        
        def int(self):
            return MockTensor()
    
    # Setup torch module attributes
    mock_torch.Tensor = MockTensor
    mock_torch.tensor = lambda *args, **kwargs: MockTensor()
    mock_torch.zeros = lambda *args, **kwargs: MockTensor()
    mock_torch.ones = lambda *args, **kwargs: MockTensor()
    mock_torch.randn = lambda *args, **kwargs: MockTensor()
    mock_torch.rand = lambda *args, **kwargs: MockTensor()
    mock_torch.empty = lambda *args, **kwargs: MockTensor()
    mock_torch.arange = lambda *args, **kwargs: MockTensor()
    mock_torch.exp = lambda x: MockTensor()
    mock_torch.sin = lambda x: MockTensor()
    mock_torch.cos = lambda x: MockTensor()
    mock_torch.cat = lambda tensors, dim=0: MockTensor()
    mock_torch.stack = lambda tensors, dim=0: MockTensor()
    mock_torch.matmul = lambda a, b: MockTensor()
    mock_torch.sum = lambda x, *args, **kwargs: MockTensor()
    mock_torch.sigmoid = lambda x: MockTensor()
    mock_torch.no_grad = lambda: MockContextManager()
    
    # Mock context manager for torch.no_grad()
    class MockContextManager:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    # Mock nn module
    mock_nn = MagicMock()
    
    # Mock nn.Module with proper parameter handling
    class MockModule:
        def __init__(self):
            self.training = True
            self._parameters = {}
            self._modules = {}
            self.weight = MockTensor()  # Mock weight parameter
            self.bias = MockTensor()    # Mock bias parameter
        
        def parameters(self):
            return [MockTensor() for _ in range(5)]  # Mock some parameters
        
        def __call__(self, *args, **kwargs):
            return MockTensor()
        
        def apply(self, fn):
            return self
        
        def register_buffer(self, name, tensor):
            setattr(self, name, tensor)
        
        def cuda(self):
            return self
        
        def eval(self):
            self.training = False
            return self
        
        def train(self, mode=True):
            self.training = mode
            return self
    
    mock_nn.Module = MockModule
    mock_nn.Linear = lambda *args, **kwargs: MockModule()
    mock_nn.LayerNorm = lambda *args, **kwargs: MockModule()
    mock_nn.Embedding = lambda *args, **kwargs: MockModule()
    mock_nn.Dropout = lambda *args, **kwargs: MockModule()
    mock_nn.GELU = lambda *args, **kwargs: MockModule()
    mock_nn.Parameter = lambda x: MockTensor()
    mock_nn.ModuleList = lambda modules: list(modules)
    mock_nn.CrossEntropyLoss = lambda *args, **kwargs: MockModule()
    
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
    
    # Mock checkpoint
    mock_checkpoint = MagicMock()
    mock_checkpoint.checkpoint = lambda fn, *args, **kwargs: fn(*args, **kwargs)
    mock_utils.checkpoint = mock_checkpoint
    
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
    sys.modules['torch.utils.checkpoint'] = mock_checkpoint
    
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
    
    # Mock scipy
    mock_scipy = MagicMock()
    mock_stats = MagicMock()
    mock_stats.ttest_rel = lambda a, b: Mock(pvalue=0.001, statistic=5.0)
    mock_stats.wilcoxon = lambda a, b: Mock(pvalue=0.002, statistic=10.0) 
    mock_stats.kstest = lambda a, b: Mock(pvalue=0.05, statistic=0.1)
    mock_scipy.stats = mock_stats
    sys.modules['scipy'] = mock_scipy
    sys.modules['scipy.stats'] = mock_stats
    
    # Mock networkx
    mock_networkx = MagicMock()
    mock_networkx.DiGraph = lambda: Mock()
    mock_networkx.Graph = lambda: Mock()
    sys.modules['networkx'] = mock_networkx
    
    # Mock matplotlib
    mock_matplotlib = MagicMock()
    mock_pyplot = MagicMock()
    mock_pyplot.figure = lambda *args, **kwargs: Mock()
    mock_pyplot.plot = lambda *args, **kwargs: Mock()
    mock_pyplot.show = lambda: None
    mock_pyplot.savefig = lambda *args, **kwargs: None
    mock_matplotlib.pyplot = mock_pyplot
    sys.modules['matplotlib'] = mock_matplotlib
    sys.modules['matplotlib.pyplot'] = mock_pyplot


def setup_full_mock_environment():
    """Setup complete mock environment for testing."""
    setup_mock_torch()
    setup_mock_dependencies()
    print("Complete mock environment initialized")


if __name__ == "__main__":
    setup_full_mock_environment()