"""
Simplified mock PyTorch implementation for RevNet-Zero testing in environments without full PyTorch.
"""

import sys
import types
import numpy as np
from typing import Any, Optional, Union, Tuple, List

class MockTensor:
    """Mock PyTorch tensor for basic operations"""
    def __init__(self, data, requires_grad=False, dtype=None, device='cpu'):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype if dtype else np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype if dtype else np.float32)
        else:
            self.data = np.array([data], dtype=dtype if dtype else np.float32)
        
        self.requires_grad = requires_grad
        self.grad = None
        self.device = device
        self.dtype = dtype if dtype else 'float32'
        self._shape = self.data.shape
    
    @property
    def shape(self):
        return self._shape
    
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]
    
    def dim(self):
        return len(self.shape)
    
    def backward(self, gradient=None):
        """Mock backward pass"""
        if self.requires_grad:
            if gradient is None:
                gradient = np.ones_like(self.data)
            self.grad = MockTensor(gradient) if not isinstance(gradient, MockTensor) else gradient
    
    def detach(self):
        return MockTensor(self.data.copy(), requires_grad=False)
    
    def clone(self):
        return MockTensor(self.data.copy(), requires_grad=self.requires_grad)
    
    def mean(self, dim=None, keepdim=False):
        result = np.mean(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, requires_grad=self.requires_grad)
    
    def sum(self, dim=None, keepdim=False):
        result = np.sum(self.data, axis=dim, keepdims=keepdim)
        return MockTensor(result, requires_grad=self.requires_grad)
    
    def view(self, *shape):
        return MockTensor(self.data.reshape(shape), requires_grad=self.requires_grad)
    
    def transpose(self, dim0, dim1):
        result = np.swapaxes(self.data, dim0, dim1)
        return MockTensor(result, requires_grad=self.requires_grad)
    
    def unsqueeze(self, dim):
        result = np.expand_dims(self.data, axis=dim)
        return MockTensor(result, requires_grad=self.requires_grad)
    
    def squeeze(self, dim=None):
        result = np.squeeze(self.data, axis=dim)
        return MockTensor(result, requires_grad=self.requires_grad)
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            result = self.data + other.data
        else:
            result = self.data + other
        return MockTensor(result, requires_grad=self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad))
    
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            result = self.data * other.data
        else:
            result = self.data * other
        return MockTensor(result, requires_grad=self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad))
    
    def __truediv__(self, other):
        if isinstance(other, MockTensor):
            result = self.data / other.data
        else:
            result = self.data / other
        return MockTensor(result, requires_grad=self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad))
    
    def __matmul__(self, other):
        if isinstance(other, MockTensor):
            result = np.matmul(self.data, other.data)
        else:
            result = np.matmul(self.data, other)
        return MockTensor(result, requires_grad=self.requires_grad or (hasattr(other, 'requires_grad') and other.requires_grad))
    
    def __getitem__(self, key):
        result = self.data[key]
        return MockTensor(result, requires_grad=self.requires_grad)
    
    def __repr__(self):
        return f"MockTensor({self.data}, requires_grad={self.requires_grad})"

class MockParameter(MockTensor):
    """Mock PyTorch Parameter"""
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)

def tensor(data, requires_grad=False, dtype=None, device='cpu'):
    """Mock torch.tensor function"""
    return MockTensor(data, requires_grad=requires_grad, dtype=dtype, device=device)

def zeros(*shape, requires_grad=False, dtype=None, device='cpu'):
    """Mock torch.zeros function"""
    data = np.zeros(shape, dtype=dtype if dtype else np.float32)
    return MockTensor(data, requires_grad=requires_grad, dtype=dtype, device=device)

def ones(*shape, requires_grad=False, dtype=None, device='cpu'):
    """Mock torch.ones function"""
    data = np.ones(shape, dtype=dtype if dtype else np.float32)
    return MockTensor(data, requires_grad=requires_grad, dtype=dtype, device=device)

def randn(*shape, requires_grad=False, dtype=None, device='cpu'):
    """Mock torch.randn function"""
    data = np.random.randn(*shape).astype(dtype if dtype else np.float32)
    return MockTensor(data, requires_grad=requires_grad, dtype=dtype, device=device)

def randint(low, high, size, dtype=None, device='cpu'):
    """Mock torch.randint function"""
    data = np.random.randint(low, high, size=size)
    return MockTensor(data, dtype=dtype, device=device)

# Mock activation functions
def relu(input_tensor):
    result = np.maximum(0, input_tensor.data)
    return MockTensor(result, requires_grad=input_tensor.requires_grad)

def gelu(input_tensor):
    """GELU activation function"""
    x = input_tensor.data
    result = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    return MockTensor(result, requires_grad=input_tensor.requires_grad)

def softmax(input_tensor, dim=-1):
    """Softmax activation"""
    x = input_tensor.data
    exp_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    result = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    return MockTensor(result, requires_grad=input_tensor.requires_grad)

# Mock nn module
class MockModule:
    """Base mock PyTorch Module"""
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x):
        return x

class MockLinear(MockModule):
    """Mock PyTorch Linear layer"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias
        weight_data = np.random.randn(out_features, in_features) * 0.1
        self.weight = MockParameter(weight_data)
        self._parameters['weight'] = self.weight
        
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = MockParameter(bias_data)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None
    
    def forward(self, x):
        # Simple matrix multiplication
        result = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            result = result + self.bias
        return result

class MockLayerNorm(MockModule):
    """Mock PyTorch LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Initialize weight and bias
        self.weight = MockParameter(np.ones(normalized_shape))
        self.bias = MockParameter(np.zeros(normalized_shape))
        self._parameters['weight'] = self.weight
        self._parameters['bias'] = self.bias
    
    def forward(self, x):
        # Simple layer normalization
        mean = np.mean(x.data, axis=-1, keepdims=True)
        std = np.std(x.data, axis=-1, keepdims=True)
        normalized = (x.data - mean) / (std + self.eps)
        result = normalized * self.weight.data + self.bias.data
        return MockTensor(result, requires_grad=x.requires_grad)

class MockDropout(MockModule):
    """Mock PyTorch Dropout"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training:
            return x
        # Simple dropout simulation
        mask = np.random.random(x.shape) > self.p
        result = x.data * mask / (1 - self.p)
        return MockTensor(result, requires_grad=x.requires_grad)

# Create mock torch module
torch_module = types.ModuleType('torch')
torch_module.Tensor = MockTensor
torch_module.tensor = tensor
torch_module.zeros = zeros
torch_module.ones = ones
torch_module.randn = randn
torch_module.randint = randint
torch_module.relu = relu
torch_module.Parameter = MockParameter

# Mock autograd
class MockFunction:
    """Mock PyTorch autograd Function"""
    @staticmethod
    def forward(ctx, *args):
        # Simple pass-through
        return args[0] if args else None
    
    @staticmethod
    def backward(ctx, grad_output):
        # Simple gradient pass-through
        return grad_output, None

autograd_module = types.ModuleType('torch.autograd')
autograd_module.Function = MockFunction
autograd_module.grad = lambda outputs, inputs: [MockTensor(np.random.randn(*inp.shape)) for inp in inputs]
torch_module.autograd = autograd_module

# Mock nn module
nn_module = types.ModuleType('torch.nn')
nn_module.Module = MockModule
nn_module.Linear = MockLinear
nn_module.LayerNorm = MockLayerNorm
nn_module.Dropout = MockDropout

# Mock functional
functional_module = types.ModuleType('torch.nn.functional')
functional_module.relu = relu
functional_module.gelu = gelu
functional_module.softmax = softmax
nn_module.functional = functional_module

torch_module.nn = nn_module

# Mock CUDA functionality
class MockCuda:
    @staticmethod
    def is_available():
        return False
    
    @staticmethod
    def device_count():
        return 0
    
    @staticmethod
    def max_memory_allocated():
        return 1000000000  # 1GB mock

torch_module.cuda = MockCuda()

# Mock optim
class MockAdamW:
    def __init__(self, params, lr=1e-3, **kwargs):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        pass
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None

optim_module = types.ModuleType('torch.optim')
optim_module.AdamW = MockAdamW
torch_module.optim = optim_module

# Mock utils
utils_module = types.ModuleType('torch.utils')

# Mock checkpoint
def mock_checkpoint(function, *args, **kwargs):
    """Mock gradient checkpointing"""
    return function(*args, **kwargs)

checkpoint_module = types.ModuleType('torch.utils.checkpoint')
checkpoint_module.checkpoint = mock_checkpoint
checkpoint_module.CheckpointFunction = MockFunction

utils_module.checkpoint = checkpoint_module
torch_module.utils = utils_module

# Install mock torch in sys.modules
sys.modules['torch'] = torch_module
sys.modules['torch.nn'] = nn_module
sys.modules['torch.nn.functional'] = functional_module
sys.modules['torch.autograd'] = autograd_module
sys.modules['torch.optim'] = optim_module
sys.modules['torch.utils'] = utils_module
sys.modules['torch.utils.checkpoint'] = checkpoint_module

print("✅ Mock PyTorch installed successfully for testing environment")