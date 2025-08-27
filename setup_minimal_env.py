#!/usr/bin/env python3
"""
Minimal environment setup for RevNet-Zero development.
Creates mock PyTorch and numpy implementations for testing.
"""

import sys
import os
import types
import warnings
from pathlib import Path

def setup_numpy_mock():
    """Create comprehensive numpy mock"""
    class MockArray:
        def __init__(self, data, dtype=None):
            if isinstance(data, (list, tuple)):
                self.data = data
                if isinstance(data[0], (list, tuple)):
                    self.shape = (len(data), len(data[0]))
                else:
                    self.shape = (len(data),)
            else:
                self.data = [data]
                self.shape = (1,)
            
            self.size = 1
            for dim in self.shape:
                self.size *= dim
            self.dtype = dtype or 'float32'
        
        def __getitem__(self, key):
            return MockArray(self.data[key])
        
        def __setitem__(self, key, value):
            self.data[key] = value
            
        def mean(self):
            return MockArray([sum(self.data) / len(self.data)])
        
        def sum(self):
            return MockArray([sum(self.data)])
        
        def __repr__(self):
            return f"MockArray({self.data})"
    
    # Create numpy mock
    np = types.ModuleType('numpy')
    
    # Basic array functions
    np.array = lambda x, dtype=None: MockArray(x, dtype)
    np.zeros = lambda shape: MockArray([0] * (shape if isinstance(shape, int) else shape[0]))
    np.ones = lambda shape: MockArray([1] * (shape if isinstance(shape, int) else shape[0]))
    np.asarray = lambda x: MockArray(x) if not isinstance(x, MockArray) else x
    
    # Data types
    np.float32 = 'float32'
    np.float64 = 'float64'
    np.int32 = 'int32'
    np.int64 = 'int64'
    np.number = (int, float)
    np.floating = (float,)
    
    # Math functions
    np.isnan = lambda x: False
    np.isinf = lambda x: False
    np.any = lambda x: True
    np.abs = lambda x: x
    np.mean = lambda x: sum(x.data if hasattr(x, 'data') else x) / len(x.data if hasattr(x, 'data') else x)
    np.issubdtype = lambda dtype, base: True
    
    # Constants
    np.pi = 3.14159265359
    np.e = 2.71828182846
    
    # Install in sys.modules
    sys.modules['numpy'] = np
    return np

def setup_torch_mock():
    """Create comprehensive PyTorch mock"""
    
    class MockTensor:
        def __init__(self, data, requires_grad=False, device='cpu'):
            if hasattr(data, 'data'):
                self.data = data.data
                self.shape = getattr(data, 'shape', (1,))
            elif isinstance(data, (list, tuple)):
                self.data = data
                # Calculate shape for nested lists
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)):
                    self.shape = (len(data), len(data[0]))
                else:
                    self.shape = (len(data),)
            else:
                self.data = [data]
                self.shape = (1,)
            
            self.requires_grad = requires_grad
            self.device = device
            self.grad = None
            self.dtype = 'torch.float32'
        
        def size(self, dim=None):
            if dim is None:
                return self.shape
            return self.shape[dim]
        
        def numel(self):
            size = 1
            for dim in self.shape:
                size *= dim
            return size
        
        def view(self, *args):
            return MockTensor(self.data, self.requires_grad, self.device)
        
        def transpose(self, dim0, dim1):
            return MockTensor(self.data, self.requires_grad, self.device)
        
        def contiguous(self):
            return self
        
        def detach(self):
            return MockTensor(self.data, False, self.device)
        
        def clone(self):
            return MockTensor(self.data, self.requires_grad, self.device)
        
        def requires_grad_(self, requires_grad=True):
            self.requires_grad = requires_grad
            return self
        
        def backward(self):
            pass
        
        def mean(self):
            return MockTensor([0.0])
        
        def unsqueeze(self, dim):
            return MockTensor(self.data, self.requires_grad, self.device)
        
        def expand(self, *args):
            return MockTensor(self.data, self.requires_grad, self.device)
        
        def masked_fill(self, mask, value):
            return MockTensor(self.data, self.requires_grad, self.device)
        
        def __add__(self, other):
            return MockTensor(self.data, self.requires_grad, self.device)
        
        def __mul__(self, other):
            return MockTensor(self.data, self.requires_grad, self.device)
        
        def __getitem__(self, key):
            return MockTensor(self.data, self.requires_grad, self.device)
        
        def __repr__(self):
            return f"MockTensor({self.data}, device='{self.device}')"
    
    # Create torch mock
    torch = types.ModuleType('torch')
    
    # Tensor creation functions
    torch.Tensor = MockTensor
    torch.tensor = lambda data, dtype=None, device='cpu', requires_grad=False: MockTensor(data, requires_grad, device)
    
    def _create_zeros(*shape, dtype=None, device='cpu'):
        if len(shape) == 1:
            return MockTensor([0.0] * shape[0], device=device)
        elif len(shape) == 2:
            data = [[0.0] * shape[1] for _ in range(shape[0])]
            return MockTensor(data, device=device)
        else:
            # For higher dimensions, flatten
            total = 1
            for dim in shape:
                total *= dim
            return MockTensor([0.0] * total, device=device)
    
    def _create_ones(*shape, dtype=None, device='cpu'):
        if len(shape) == 1:
            return MockTensor([1.0] * shape[0], device=device)
        elif len(shape) == 2:
            data = [[1.0] * shape[1] for _ in range(shape[0])]
            return MockTensor(data, device=device)
        else:
            # For higher dimensions, flatten
            total = 1
            for dim in shape:
                total *= dim
            return MockTensor([1.0] * total, device=device)
    
    torch.zeros = _create_zeros
    torch.ones = _create_ones
    torch.randn = lambda *shape, dtype=None, device='cpu': _create_zeros(*shape, device=device)  # Use zeros for simplicity
    torch.randint = lambda low, high, size, dtype=None, device='cpu': MockTensor([1] * size[0] if isinstance(size, (list, tuple)) else [1] * size, device=device)
    torch.arange = lambda start, end=None, step=1, dtype=None, device='cpu': MockTensor(list(range(start, end or start, step)), device=device)
    
    # Data types
    torch.float = 'torch.float32'
    torch.float32 = 'torch.float32'
    torch.long = 'torch.int64'
    torch.int64 = 'torch.int64'
    torch.bool = 'torch.bool'
    
    # Math functions
    torch.matmul = lambda x, y: MockTensor([1.0])
    torch.cat = lambda tensors, dim=0: tensors[0] if tensors else MockTensor([0])
    torch.chunk = lambda tensor, chunks, dim=0: [MockTensor(tensor.data), MockTensor(tensor.data)]
    torch.split = lambda tensor, split_size, dim=0: [MockTensor(tensor.data), MockTensor(tensor.data)]
    torch.stack = lambda tensors, dim=0: tensors[0] if tensors else MockTensor([0])
    torch.exp = lambda x: MockTensor([1.0])
    torch.log = lambda x: MockTensor([0.0])
    torch.sin = lambda x: MockTensor([0.0])
    torch.cos = lambda x: MockTensor([1.0])
    torch.tanh = lambda x: MockTensor([0.0])
    torch.sigmoid = lambda x: MockTensor([0.5])
    torch.softmax = lambda x, dim=-1: MockTensor([0.5] * (x.size(dim) if hasattr(x, 'size') else 1))
    torch.isnan = lambda x: MockTensor([False])
    torch.isinf = lambda x: MockTensor([False])
    torch.clamp = lambda x, min=None, max=None: x
    
    # CUDA mock
    class MockCuda:
        @staticmethod
        def is_available():
            return False
        
        @staticmethod
        def device_count():
            return 0
        
        @staticmethod
        def get_device_properties(device):
            class Props:
                total_memory = 8 * 1024**3  # 8GB
            return Props()
        
        @staticmethod
        def memory_allocated(device=None):
            return 1024**2  # 1MB
        
        @staticmethod
        def max_memory_allocated(device=None):
            return 2 * 1024**2  # 2MB
        
        @staticmethod
        def empty_cache():
            pass
    
    torch.cuda = MockCuda()
    torch.device = lambda x: x
    
    # Context managers
    class MockNoGrad:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    torch.no_grad = lambda: MockNoGrad()
    
    # Autograd mock
    class MockAutograd:
        @staticmethod
        def backward(tensors, grad_tensors=None, retain_graph=None):
            pass
        
        class Function:
            @staticmethod
            def apply(*args):
                return args[0] if args else MockTensor([0])
        
        grad = lambda fn: lambda *args: [MockTensor([0])] * len(args)
    
    torch.autograd = MockAutograd()
    
    # Neural network mock
    nn = types.ModuleType('torch.nn')
    
    class MockModule:
        def __init__(self):
            self.training = True
            self._parameters = {}
            self._modules = {}
        
        def parameters(self, recurse=True):
            return []
        
        def named_parameters(self, prefix='', recurse=True):
            return []
        
        def train(self, mode=True):
            self.training = mode
            return self
        
        def eval(self):
            self.training = False
            return self
        
        def apply(self, fn):
            return self
        
        def register_buffer(self, name, tensor):
            setattr(self, name, tensor)
        
        def to(self, device):
            return self
        
        def cuda(self):
            return self
        
        def cpu(self):
            return self
        
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
        
        def forward(self, x):
            return x
    
    nn.Module = MockModule
    
    class MockModuleList:
        def __init__(self, modules=None):
            self._modules = list(modules or [])
        
        def __iter__(self):
            return iter(self._modules)
        
        def __getitem__(self, idx):
            return self._modules[idx]
        
        def __len__(self):
            return len(self._modules)
        
        def append(self, module):
            self._modules.append(module)
        
        def enumerate(self):
            return enumerate(self._modules)
    
    nn.ModuleList = MockModuleList
    
    class MockSequential(MockModule):
        def __init__(self, *args):
            super().__init__()
            self._modules = list(args)
        
        def forward(self, x):
            for module in self._modules:
                if hasattr(module, 'forward'):
                    x = module.forward(x)
                elif callable(module):
                    x = module(x)
            return x
        
        def __call__(self, x):
            return self.forward(x)
    
    nn.Sequential = MockSequential
    
    class MockLinear(MockModule):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = MockTensor([[0.1] * in_features] * out_features, requires_grad=True)
            if bias:
                self.bias = MockTensor([0.1] * out_features, requires_grad=True)
            else:
                self.bias = None
        
        def forward(self, x):
            return MockTensor([0.1] * self.out_features)
    
    class MockLayerNorm(MockModule):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
            super().__init__()
            self.normalized_shape = normalized_shape
            self.eps = eps
            if elementwise_affine:
                self.weight = MockTensor([1.0] * normalized_shape, requires_grad=True)
                self.bias = MockTensor([0.0] * normalized_shape, requires_grad=True)
            else:
                self.weight = None
                self.bias = None
        
        def forward(self, x):
            return x
    
    class MockEmbedding(MockModule):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = MockTensor([[0.1] * embedding_dim] * num_embeddings, requires_grad=True)
        
        def forward(self, input):
            batch_size = input.shape[0] if hasattr(input, 'shape') else 1
            seq_len = input.shape[1] if hasattr(input, 'shape') and len(input.shape) > 1 else 1
            return MockTensor([[0.1] * self.embedding_dim] * seq_len)
    
    class MockDropout(MockModule):
        def __init__(self, p=0.5, inplace=False):
            super().__init__()
            self.p = p
            self.inplace = inplace
        
        def forward(self, x):
            return x
    
    class MockCrossEntropyLoss(MockModule):
        def __init__(self, weight=None, size_average=None, ignore_index=-100, 
                     reduce=None, reduction='mean', label_smoothing=0.0):
            super().__init__()
            self.reduction = reduction
        
        def forward(self, input, target):
            return MockTensor([1.0])
    
    class MockReLU(MockModule):
        def __init__(self, inplace=False):
            super().__init__()
            self.inplace = inplace
        
        def forward(self, x):
            return x  # Just pass through for mock
    
    class MockGELU(MockModule):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return x  # Just pass through for mock
    
    nn.Linear = MockLinear
    nn.LayerNorm = MockLayerNorm  
    nn.Embedding = MockEmbedding
    nn.Dropout = MockDropout
    nn.CrossEntropyLoss = MockCrossEntropyLoss
    nn.ReLU = MockReLU
    nn.GELU = MockGELU
    
    # Functional
    F = types.ModuleType('torch.nn.functional')
    F.relu = lambda x: MockTensor([0.1])
    F.gelu = lambda x: MockTensor([0.1])
    F.silu = lambda x: MockTensor([0.1])
    F.softmax = lambda x, dim=-1: MockTensor([0.5])
    F.scaled_dot_product_attention = lambda q, k, v, attn_mask=None, dropout_p=0.0: MockTensor([0.1])
    
    nn.functional = F
    torch.nn = nn
    
    # Parameter class
    class MockParameter(MockTensor):
        def __init__(self, data, requires_grad=True):
            super().__init__(data, requires_grad=requires_grad)
    
    nn.Parameter = MockParameter
    torch.nn.Parameter = MockParameter
    
    # Initialization functions
    class MockInit:
        @staticmethod
        def normal_(tensor, mean=0.0, std=1.0):
            pass
        
        @staticmethod
        def zeros_(tensor):
            pass
        
        @staticmethod
        def ones_(tensor):
            pass
        
        @staticmethod
        def xavier_uniform_(tensor, gain=1.0):
            pass
        
        @staticmethod
        def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
            pass
    
    torch.nn.init = MockInit()
    
    # Optimizer mock
    optim = types.ModuleType('torch.optim')
    
    class MockOptimizer:
        def __init__(self, params, **kwargs):
            self.param_groups = []
        
        def step(self):
            pass
        
        def zero_grad(self):
            pass
    
    class MockAdamW(MockOptimizer):
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
            super().__init__(params)
    
    optim.AdamW = MockAdamW
    torch.optim = optim
    
    # Install in sys.modules  
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = nn
    sys.modules['torch.nn.functional'] = F
    sys.modules['torch.optim'] = optim
    sys.modules['torch.autograd'] = torch.autograd
    
    return torch

def setup_additional_mocks():
    """Setup additional mocks for dependencies"""
    
    # Einops mock
    einops = types.ModuleType('einops')
    einops.rearrange = lambda tensor, pattern: tensor
    einops.reduce = lambda tensor, pattern, reduction: tensor
    einops.repeat = lambda tensor, pattern: tensor
    sys.modules['einops'] = einops
    
    # PSUtil mock
    psutil = types.ModuleType('psutil')
    class MockProcess:
        @staticmethod
        def memory_info():
            class MemInfo:
                rss = 1024 * 1024 * 1024  # 1GB
            return MemInfo()
    
    class MockVirtualMemory:
        total = 8 * 1024 * 1024 * 1024  # 8GB
    
    psutil.Process = MockProcess
    psutil.virtual_memory = lambda: MockVirtualMemory()
    sys.modules['psutil'] = psutil

def main():
    """Setup complete mock environment"""
    print("Setting up minimal mock environment...")
    
    # Setup mocks
    np = setup_numpy_mock()
    torch = setup_torch_mock()
    setup_additional_mocks()
    
    print("✅ Mock environment setup complete!")
    print(f"  - numpy: {type(sys.modules['numpy'])}")
    print(f"  - torch: {type(sys.modules['torch'])}")
    print(f"  - einops: {type(sys.modules['einops'])}")
    print(f"  - psutil: {type(sys.modules['psutil'])}")
    
    return True

if __name__ == "__main__":
    main()