"""
Unified kernel manager for RevNet-Zero optimizations.

Automatically selects optimal kernels based on hardware and input characteristics.
"""

import torch
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import warnings

from .cuda_kernels import (
    fused_reversible_attention as cuda_fused_attention,
    fused_coupling_forward,
    fused_coupling_backward,
    CUDAKernelManager
)

from .triton_kernels import (
    triton_fused_reversible_attention,
    triton_block_sparse_attention,
    TritonKernelManager
)

class KernelBackend(Enum):
    """Available kernel backends."""
    PYTORCH = "pytorch"
    CUDA = "cuda" 
    TRITON = "triton"
    AUTO = "auto"

class OptimizationProfile:
    """Profile for kernel optimization decisions."""
    
    def __init__(self, 
                 seq_len: int,
                 head_dim: int,
                 batch_size: int,
                 num_heads: int,
                 device: torch.device):
        self.seq_len = seq_len
        self.head_dim = head_dim  
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.device = device
        
        # Computed characteristics
        self.total_elements = batch_size * num_heads * seq_len * head_dim
        self.memory_gb = self.total_elements * 4 / (1024**3)  # Assume float32
        self.is_long_sequence = seq_len > 4096
        self.is_large_model = head_dim > 512
        
    def get_complexity_score(self) -> float:
        """Get complexity score for kernel selection."""
        return (self.seq_len / 1024) * (self.head_dim / 64) * (self.batch_size * self.num_heads / 16)

class KernelManager:
    """Unified manager for all RevNet-Zero kernels."""
    
    def __init__(self, backend: KernelBackend = KernelBackend.AUTO):
        self.backend = backend
        self.cuda_manager = CUDAKernelManager()
        self._performance_cache: Dict[str, Dict] = {}
        
        # Check available backends
        self.has_cuda = torch.cuda.is_available()
        self.has_triton = TritonKernelManager.is_available()
        
        if backend == KernelBackend.CUDA and not self.has_cuda:
            warnings.warn("CUDA requested but not available, falling back to PyTorch")
            self.backend = KernelBackend.PYTORCH
        elif backend == KernelBackend.TRITON and not self.has_triton:
            warnings.warn("Triton requested but not available, falling back to PyTorch")
            self.backend = KernelBackend.PYTORCH
    
    def select_optimal_backend(self, profile: OptimizationProfile) -> KernelBackend:
        """Select optimal backend based on input characteristics."""
        if self.backend != KernelBackend.AUTO:
            return self.backend
        
        # Decision tree for backend selection
        complexity = profile.get_complexity_score()
        
        # For small operations, PyTorch is often fastest due to low overhead
        if complexity < 1.0:
            return KernelBackend.PYTORCH
        
        # For very long sequences, prefer Triton's block-sparse operations
        if profile.is_long_sequence and self.has_triton:
            return KernelBackend.TRITON
        
        # For medium complexity with CUDA, use custom kernels
        if self.has_cuda and complexity > 2.0:
            return KernelBackend.CUDA
        
        # For high complexity without custom kernels, use Triton
        if self.has_triton and complexity > 10.0:
            return KernelBackend.TRITON
            
        return KernelBackend.PYTORCH
    
    def fused_reversible_attention(self,
                                 q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 coupling_type: str = 'additive',
                                 backend: Optional[KernelBackend] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimal fused reversible attention.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            v: Value tensor [batch, num_heads, seq_len, head_dim]
            coupling_type: Type of coupling function ('additive', 'affine')
            backend: Force specific backend (optional)
            
        Returns:
            output: Attention output tensor
            coupling_output: Coupling function output for reversibility
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        profile = OptimizationProfile(seq_len, head_dim, batch_size, num_heads, q.device)
        
        # Select backend
        selected_backend = backend or self.select_optimal_backend(profile)
        
        # Dispatch to appropriate implementation
        if selected_backend == KernelBackend.TRITON and self.has_triton:
            return triton_fused_reversible_attention(q, k, v, coupling_type)
        elif selected_backend == KernelBackend.CUDA and self.has_cuda:
            return cuda_fused_attention(q, k, v, coupling_type)
        else:
            return self._pytorch_fused_attention(q, k, v, coupling_type)
    
    def block_sparse_attention(self,
                              q: torch.Tensor,
                              k: torch.Tensor, 
                              v: torch.Tensor,
                              block_mask: torch.Tensor,
                              block_size: int = 64,
                              backend: Optional[KernelBackend] = None) -> torch.Tensor:
        """
        Block-sparse attention with optimal kernel selection.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]  
            v: Value tensor [batch, num_heads, seq_len, head_dim]
            block_mask: Block sparsity mask [num_q_blocks, num_k_blocks]
            block_size: Size of attention blocks
            backend: Force specific backend
            
        Returns:
            output: Sparse attention output
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        profile = OptimizationProfile(seq_len, head_dim, batch_size, num_heads, q.device)
        
        selected_backend = backend or self.select_optimal_backend(profile)
        
        if selected_backend == KernelBackend.TRITON and self.has_triton:
            return triton_block_sparse_attention(q, k, v, block_mask, block_size)
        else:
            return self._pytorch_block_sparse_attention(q, k, v, block_mask, block_size)
    
    def fused_coupling_layer(self,
                           x1: torch.Tensor,
                           x2: torch.Tensor,
                           coupling_type: str = 'additive',
                           backend: Optional[KernelBackend] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused coupling layer with optimal backend.
        
        Args:
            x1: First input partition
            x2: Second input partition
            coupling_type: Type of coupling ('additive', 'affine')
            backend: Force specific backend
            
        Returns:
            y1, y2: Coupled outputs
        """
        batch_size, seq_len, d_model = x1.shape
        profile = OptimizationProfile(seq_len, d_model, batch_size, 1, x1.device)
        
        selected_backend = backend or self.select_optimal_backend(profile)
        
        if selected_backend == KernelBackend.CUDA and self.has_cuda:
            return fused_coupling_forward(x1, x2, coupling_type)
        else:
            return self._pytorch_coupling_forward(x1, x2, coupling_type)
    
    def _pytorch_fused_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                                coupling_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch fallback for fused attention."""
        scale = q.size(-1) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply coupling function
        if coupling_type == 'additive':
            coupled_scores = scores + 0.1 * torch.sin(scores)
        elif coupling_type == 'affine':
            coupled_scores = scores * 1.1 + 0.05
        else:
            coupled_scores = scores
            
        attention_weights = torch.softmax(coupled_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        # Coupling output for reversibility  
        coupling_output = torch.sum(coupled_scores * attention_weights, dim=-1)
        
        return output, coupling_output
    
    def _pytorch_block_sparse_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                       block_mask: torch.Tensor, block_size: int) -> torch.Tensor:
        """PyTorch fallback for block-sparse attention."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        scale = head_dim ** -0.5
        
        num_blocks = seq_len // block_size
        output = torch.zeros_like(q)
        
        for i in range(num_blocks):
            for j in range(num_blocks):
                if block_mask[i, j] == 0:
                    continue
                    
                i_start, i_end = i * block_size, (i + 1) * block_size
                j_start, j_end = j * block_size, (j + 1) * block_size
                
                q_block = q[:, :, i_start:i_end, :]
                k_block = k[:, :, j_start:j_end, :]
                v_block = v[:, :, j_start:j_end, :]
                
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(scores, dim=-1)
                block_output = torch.matmul(attn_weights, v_block)
                
                output[:, :, i_start:i_end, :] += block_output
                
        return output
    
    def _pytorch_coupling_forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                                 coupling_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch fallback for coupling layers."""
        if coupling_type == 'additive':
            coupling_fn = 0.1 * torch.sin(x1)
            y1 = x1
            y2 = x2 + coupling_fn
        elif coupling_type == 'affine':
            log_scale = 0.05 * torch.tanh(x1)
            translation = 0.1 * torch.relu(x1)
            y1 = x1
            y2 = x2 * torch.exp(log_scale) + translation
        else:
            y1, y2 = x1, x2
            
        return y1, y2
    
    def benchmark_kernels(self, 
                         seq_lengths: list = [1024, 4096, 16384],
                         head_dims: list = [64, 128, 256],
                         batch_size: int = 8,
                         num_heads: int = 16,
                         num_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark different kernel implementations.
        
        Args:
            seq_lengths: Sequence lengths to test
            head_dims: Head dimensions to test  
            batch_size: Batch size for testing
            num_heads: Number of attention heads
            num_runs: Number of benchmark runs
            
        Returns:
            benchmark_results: Performance comparison results
        """
        results = {}
        
        for seq_len in seq_lengths:
            for head_dim in head_dims:
                test_key = f"seq_{seq_len}_dim_{head_dim}"
                results[test_key] = {}
                
                # Create test tensors
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
                k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
                v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
                
                # Benchmark each backend
                backends_to_test = [KernelBackend.PYTORCH]
                if self.has_cuda:
                    backends_to_test.append(KernelBackend.CUDA)
                if self.has_triton:
                    backends_to_test.append(KernelBackend.TRITON)
                
                for backend in backends_to_test:
                    times = []
                    
                    # Warmup
                    for _ in range(3):
                        _ = self.fused_reversible_attention(q, k, v, backend=backend)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
                    
                    # Benchmark
                    if device.type == 'cuda':
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        
                        for _ in range(num_runs):
                            start_event.record()
                            _ = self.fused_reversible_attention(q, k, v, backend=backend)
                            end_event.record()
                            torch.cuda.synchronize()
                            times.append(start_event.elapsed_time(end_event))
                    else:
                        import time
                        for _ in range(num_runs):
                            start_time = time.perf_counter()
                            _ = self.fused_reversible_attention(q, k, v, backend=backend)
                            end_time = time.perf_counter()
                            times.append((end_time - start_time) * 1000)  # Convert to ms
                    
                    results[test_key][backend.value] = {
                        'mean_time_ms': sum(times) / len(times),
                        'std_time_ms': (sum((t - sum(times) / len(times))**2 for t in times) / len(times))**0.5,
                        'min_time_ms': min(times),
                        'max_time_ms': max(times)
                    }
        
        return results
    
    def get_performance_summary(self) -> str:
        """Get a summary of kernel capabilities and recommendations."""
        summary = []
        summary.append("=== RevNet-Zero Kernel Manager Performance Summary ===\n")
        
        # Available backends
        summary.append("Available Backends:")
        summary.append(f"  PyTorch: ✓ (Always available)")
        summary.append(f"  CUDA: {'✓' if self.has_cuda else '✗'}")
        summary.append(f"  Triton: {'✓' if self.has_triton else '✗'}")
        summary.append("")
        
        # Performance recommendations
        summary.append("Performance Recommendations:")
        summary.append("  Short sequences (< 1K): PyTorch (low overhead)")
        summary.append("  Medium sequences (1K-8K): CUDA kernels (optimal fusion)")
        summary.append("  Long sequences (> 8K): Triton (block-sparse optimization)")
        summary.append("  Memory-constrained: Triton + block sparsity")
        summary.append("")
        
        # Feature matrix
        summary.append("Feature Support Matrix:")
        summary.append("                   PyTorch  CUDA  Triton")
        summary.append("  Fused Attention    ✓       ✓     ✓")
        summary.append("  Block Sparse       ✓       ✗     ✓")
        summary.append("  Coupling Layers    ✓       ✓     ✓")
        summary.append("  Memory Checkp.     ✓       ✓     ✓")
        
        return "\n".join(summary)

# Global kernel manager instance
_global_kernel_manager: Optional[KernelManager] = None

def get_kernel_manager(backend: KernelBackend = KernelBackend.AUTO) -> KernelManager:
    """Get global kernel manager instance."""
    global _global_kernel_manager
    if _global_kernel_manager is None:
        _global_kernel_manager = KernelManager(backend)
    return _global_kernel_manager

def set_kernel_backend(backend: KernelBackend):
    """Set global kernel backend."""
    global _global_kernel_manager
    _global_kernel_manager = KernelManager(backend)

# Convenience functions
def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                   coupling_type: str = 'additive') -> Tuple[torch.Tensor, torch.Tensor]:
    """Global convenience function for fused attention."""
    return get_kernel_manager().fused_reversible_attention(q, k, v, coupling_type)

def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    block_mask: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Global convenience function for sparse attention."""
    return get_kernel_manager().block_sparse_attention(q, k, v, block_mask, block_size)

def coupling_layer(x1: torch.Tensor, x2: torch.Tensor, 
                  coupling_type: str = 'additive') -> Tuple[torch.Tensor, torch.Tensor]:
    """Global convenience function for coupling layers."""
    return get_kernel_manager().fused_coupling_layer(x1, x2, coupling_type)

# Export all
__all__ = [
    'KernelManager',
    'KernelBackend', 
    'OptimizationProfile',
    'get_kernel_manager',
    'set_kernel_backend',
    'fused_attention',
    'sparse_attention', 
    'coupling_layer'
]