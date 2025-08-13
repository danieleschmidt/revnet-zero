"""
Comprehensive Baseline Implementations

This module implements state-of-the-art long-context attention methods for
comparative evaluation against RevNet-Zero's reversible architectures.

Baselines implemented:
- Longformer: Sliding window + global attention
- BigBird: Sparse attention with random, window, and global components
- Performer: Kernel-based linear attention approximation  
- Flash Attention: Hardware-optimized attention computation
- Mamba: Selective state space models for long sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod


class BaselineModel(nn.Module, ABC):
    """Base class for all baseline implementations."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_length: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
    
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the baseline model."""
        pass
    
    @abstractmethod
    def get_memory_complexity(self) -> str:
        """Return the memory complexity of this attention mechanism."""
        pass


class LongformerAttention(BaselineModel):
    """
    Longformer: Sliding window + global attention pattern.
    
    Reference: "Longformer: The Long-Document Transformer" (2020)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_length: int,
        window_size: int = 512,
        num_global_tokens: int = 1,
        dropout: float = 0.1
    ):
        super().__init__(d_model, num_heads, max_seq_length, dropout)
        
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_longformer_mask(
        self,
        seq_length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create Longformer attention mask with sliding window + global attention."""
        mask = torch.full(
            (seq_length, seq_length),
            float('-inf'),
            device=device,
            dtype=torch.float
        )
        
        # Sliding window attention
        for i in range(seq_length):
            start = max(0, i - self.window_size // 2)
            end = min(seq_length, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0.0
        
        # Global attention for first few tokens
        mask[:self.num_global_tokens, :] = 0.0
        mask[:, :self.num_global_tokens] = 0.0
        
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear projections
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        
        # Create Longformer attention mask
        longformer_mask = self.create_longformer_mask(seq_length, hidden_states.device)
        
        # Combine with input mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            longformer_mask = longformer_mask + attention_mask
        
        # Scaled dot-product attention with sparse mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + longformer_mask.unsqueeze(0).unsqueeze(0)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, self.d_model)
        
        return self.output(attention_output)
    
    def get_memory_complexity(self) -> str:
        return f"O(n * w) where n={self.max_seq_length}, w={self.window_size}"


class BigBirdAttention(BaselineModel):
    """
    BigBird: Sparse attention with random, window, and global components.
    
    Reference: "Big Bird: Transformers for Longer Sequences" (2020)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_length: int,
        window_size: int = 128,
        num_random_blocks: int = 3,
        num_global_tokens: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(d_model, num_heads, max_seq_length, dropout)
        
        self.window_size = window_size
        self.num_random_blocks = num_random_blocks
        self.num_global_tokens = num_global_tokens
        
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_bigbird_mask(
        self,
        seq_length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create BigBird attention mask with window + random + global attention."""
        mask = torch.full(
            (seq_length, seq_length),
            float('-inf'),
            device=device,
            dtype=torch.float
        )
        
        # Local window attention
        for i in range(seq_length):
            start = max(0, i - self.window_size // 2)
            end = min(seq_length, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0.0
        
        # Global attention
        mask[:self.num_global_tokens, :] = 0.0
        mask[:, :self.num_global_tokens] = 0.0
        
        # Random attention connections
        for i in range(self.num_global_tokens, seq_length):
            # Add random connections
            random_indices = torch.randperm(seq_length, device=device)[:self.num_random_blocks]
            mask[i, random_indices] = 0.0
        
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear projections
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        
        # Create BigBird attention mask
        bigbird_mask = self.create_bigbird_mask(seq_length, hidden_states.device)
        
        # Combine with input mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            bigbird_mask = bigbird_mask + attention_mask
        
        # Scaled dot-product attention with sparse mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + bigbird_mask.unsqueeze(0).unsqueeze(0)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, self.d_model)
        
        return self.output(attention_output)
    
    def get_memory_complexity(self) -> str:
        return f"O(n * (w + r)) where n={self.max_seq_length}, w={self.window_size}, r={self.num_random_blocks}"


class PerformerAttention(BaselineModel):
    """
    Performer: Kernel-based linear attention approximation.
    
    Reference: "Rethinking Attention with Performers" (2020)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_length: int,
        num_features: int = 256,
        kernel_type: str = 'relu',
        dropout: float = 0.1
    ):
        super().__init__(d_model, num_heads, max_seq_length, dropout)
        
        self.num_features = num_features
        self.kernel_type = kernel_type
        
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Random feature projection matrix
        self.register_buffer(
            'projection_matrix',
            torch.randn(self.d_head, num_features) / math.sqrt(self.d_head)
        )
        
    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply kernel feature map to input tensor."""
        # Project to random features
        x_projected = torch.matmul(x, self.projection_matrix)
        
        if self.kernel_type == 'relu':
            return F.relu(x_projected)
        elif self.kernel_type == 'elu':
            return F.elu(x_projected) + 1.0
        elif self.kernel_type == 'softmax':
            return torch.exp(x_projected - torch.max(x_projected, dim=-1, keepdim=True)[0])
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear projections
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        
        # Apply kernel feature maps
        q_prime = self.kernel_feature_map(q)  # [batch, heads, seq, features]
        k_prime = self.kernel_feature_map(k)  # [batch, heads, seq, features]
        
        # Linear attention computation: O(n) complexity
        # Compute K^T V first (features x d_head)
        kv = torch.matmul(k_prime.transpose(-2, -1), v)  # [batch, heads, features, d_head]
        
        # Then compute Q (K^T V)
        attention_output = torch.matmul(q_prime, kv)  # [batch, heads, seq, d_head]
        
        # Normalize by row sums
        normalizer = torch.matmul(q_prime, k_prime.sum(dim=-2, keepdim=True).transpose(-2, -1))
        attention_output = attention_output / (normalizer + 1e-8)
        
        # Apply mask if provided
        if attention_mask is not None:
            # For linear attention, we need to handle masking differently
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1)
            attention_output = attention_output * mask_expanded
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, self.d_model)
        
        return self.output(attention_output)
    
    def get_memory_complexity(self) -> str:
        return f"O(n * f) where n={self.max_seq_length}, f={self.num_features}"


class FlashAttention(BaselineModel):
    """
    Flash Attention: Hardware-optimized attention computation.
    
    Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
    Note: This is a simplified implementation for comparison purposes.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_length: int,
        block_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__(d_model, num_heads, max_seq_length, dropout)
        
        self.block_size = block_size
        
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def flash_attention_block(
        self,
        q_block: torch.Tensor,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
        mask_block: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute attention for a block with online softmax."""
        # Compute attention scores
        scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Apply mask if provided
        if mask_block is not None:
            scores = scores + mask_block
        
        # Online softmax computation
        max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(scores - max_scores)
        sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
        
        # Attention weights
        attention_weights = exp_scores / sum_exp_scores
        attention_weights = self.dropout(attention_weights)
        
        # Weighted values
        weighted_values = torch.matmul(attention_weights, v_block)
        
        return weighted_values, max_scores, sum_exp_scores
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Linear projections
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.d_head).transpose(1, 2)
        
        # Process in blocks to simulate Flash Attention
        num_blocks = (seq_length + self.block_size - 1) // self.block_size
        
        output_blocks = []
        
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, seq_length)
            
            q_block = q[:, :, start_idx:end_idx, :]
            
            # For simplicity, use full attention within each block
            # Real Flash Attention uses more sophisticated blocking
            block_output, _, _ = self.flash_attention_block(
                q_block, k, v,
                attention_mask.unsqueeze(1).unsqueeze(1) if attention_mask is not None else None
            )
            
            output_blocks.append(block_output)
        
        # Concatenate blocks
        attention_output = torch.cat(output_blocks, dim=2)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, self.d_model)
        
        return self.output(attention_output)
    
    def get_memory_complexity(self) -> str:
        return f"O(n * b) where n={self.max_seq_length}, b={self.block_size}"


class MambaBlock(BaselineModel):
    """
    Mamba: Selective State Space Model.
    
    Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
    Note: Simplified implementation focusing on the core selective mechanism.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,  # Not used in Mamba but kept for interface compatibility
        max_seq_length: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(d_model, num_heads, max_seq_length, dropout)
        
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand_factor
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # State space parameters (selective mechanism)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)  # For B and C
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # State transition matrix (learnable)
        A = torch.randn(self.d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def selective_ssm(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective State Space Model computation.
        
        Args:
            x: Input tensor [batch, seq_len, d_inner]
            delta: Selection mechanism [batch, seq_len, d_inner]
            B: Input matrix [batch, seq_len, d_state]
            C: Output matrix [batch, seq_len, d_state]
        """
        batch_size, seq_len, d_inner = x.shape
        
        # State transition matrix
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Discretize state space model
        # Simplified version - real Mamba uses more sophisticated discretization
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [batch, seq, d_inner, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(-2)  # [batch, seq, d_inner, d_state]
        
        # Selective scan (simplified - real implementation is more efficient)
        states = []
        current_state = torch.zeros(batch_size, d_inner, self.d_state, device=x.device)
        
        for t in range(seq_len):
            # Update state
            current_state = deltaA[:, t] * current_state + deltaB[:, t] * x[:, t:t+1].unsqueeze(-1)
            
            # Compute output
            y_t = torch.sum(current_state * C[:, t:t+1].unsqueeze(-2), dim=-1)  # [batch, d_inner]
            states.append(y_t)
        
        # Stack outputs
        y = torch.stack(states, dim=1)  # [batch, seq_len, d_inner]
        
        return y
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Input projection and split
        x_and_res = self.in_proj(hidden_states)  # [batch, seq, d_inner * 2]
        x, res = x_and_res.chunk(2, dim=-1)  # Each [batch, seq, d_inner]
        
        # Apply convolution for local context
        x = x.transpose(1, 2)  # [batch, d_inner, seq]
        x = self.conv1d(x)[:, :, :seq_length]  # Trim padding
        x = x.transpose(1, 2)  # [batch, seq, d_inner]
        
        # Apply SiLU activation
        x = F.silu(x)
        
        # Selective mechanism - compute B, C, and delta
        x_proj = self.x_proj(x)  # [batch, seq, d_state * 2]
        B, C = x_proj.chunk(2, dim=-1)  # Each [batch, seq, d_state]
        
        delta = F.softplus(self.dt_proj(x))  # [batch, seq, d_inner]
        
        # Apply selective SSM
        y = self.selective_ssm(x, delta, B, C)
        
        # Apply gate (residual connection)
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        # Apply mask if provided
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        
        return self.dropout(output)
    
    def get_memory_complexity(self) -> str:
        return f"O(n * d) where n={self.max_seq_length}, d={self.d_state}"


class ComprehensiveBaselines:
    """
    Comprehensive baseline collection for systematic comparison.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        max_seq_length: int = 4096,
        dropout: float = 0.1
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        
        self.baselines = {}
        self._initialize_baselines()
    
    def _initialize_baselines(self):
        """Initialize all baseline models."""
        self.baselines['longformer'] = LongformerAttention(
            self.d_model, self.num_heads, self.max_seq_length,
            window_size=512, num_global_tokens=1, dropout=self.dropout
        )
        
        self.baselines['bigbird'] = BigBirdAttention(
            self.d_model, self.num_heads, self.max_seq_length,
            window_size=128, num_random_blocks=3, num_global_tokens=2,
            dropout=self.dropout
        )
        
        self.baselines['performer'] = PerformerAttention(
            self.d_model, self.num_heads, self.max_seq_length,
            num_features=256, kernel_type='relu', dropout=self.dropout
        )
        
        self.baselines['flash_attention'] = FlashAttention(
            self.d_model, self.num_heads, self.max_seq_length,
            block_size=128, dropout=self.dropout
        )
        
        self.baselines['mamba'] = MambaBlock(
            self.d_model, self.num_heads, self.max_seq_length,
            d_state=64, d_conv=4, expand_factor=2, dropout=self.dropout
        )
    
    def get_baseline(self, name: str) -> BaselineModel:
        """Get a specific baseline model."""
        if name not in self.baselines:
            raise ValueError(f"Unknown baseline: {name}. Available: {list(self.baselines.keys())}")
        return self.baselines[name]
    
    def get_all_baselines(self) -> Dict[str, BaselineModel]:
        """Get all baseline models."""
        return self.baselines.copy()
    
    def get_memory_complexities(self) -> Dict[str, str]:
        """Get memory complexity for all baselines."""
        return {name: model.get_memory_complexity() for name, model in self.baselines.items()}
    
    def comparative_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        baselines_to_run: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass on multiple baselines for comparison.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            baselines_to_run: List of baseline names to run (default: all)
            
        Returns:
            Dictionary mapping baseline names to outputs
        """
        if baselines_to_run is None:
            baselines_to_run = list(self.baselines.keys())
        
        results = {}
        for name in baselines_to_run:
            if name in self.baselines:
                try:
                    with torch.no_grad():
                        output = self.baselines[name](hidden_states, attention_mask)
                    results[name] = output
                except Exception as e:
                    print(f"Error running baseline {name}: {e}")
                    results[name] = None
            else:
                print(f"Unknown baseline: {name}")
        
        return results


def create_baseline_suite(
    model_config: Dict[str, Any],
    baseline_names: Optional[List[str]] = None
) -> ComprehensiveBaselines:
    """
    Factory function to create a comprehensive baseline suite.
    
    Args:
        model_config: Configuration dictionary with model parameters
        baseline_names: List of baseline names to include (default: all)
        
    Returns:
        Configured ComprehensiveBaselines instance
    """
    suite = ComprehensiveBaselines(**model_config)
    
    if baseline_names:
        # Filter baselines if specific ones requested
        filtered_baselines = {
            name: suite.baselines[name] 
            for name in baseline_names 
            if name in suite.baselines
        }
        suite.baselines = filtered_baselines
    
    return suite


__all__ = [
    'BaselineModel',
    'LongformerAttention',
    'BigBirdAttention', 
    'PerformerAttention',
    'FlashAttention',
    'MambaBlock',
    'ComprehensiveBaselines',
    'create_baseline_suite'
]