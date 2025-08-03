"""
Coupling layer implementations for reversible networks.

This module provides base classes and implementations for coupling functions
used in reversible neural networks, enabling memory-efficient training.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseCoupling(nn.Module, ABC):
    """
    Base class for reversible coupling functions.
    
    Coupling functions split input into two parts and apply reversible transformations
    that can be inverted exactly during backpropagation for memory efficiency.
    """
    
    def __init__(self, split_dim: int = -1):
        super().__init__()
        self.split_dim = split_dim
    
    @abstractmethod
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward coupling transformation."""
        pass
    
    @abstractmethod
    def inverse(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply inverse coupling transformation for activation reconstruction."""
        pass
    
    def split_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split input tensor along specified dimension."""
        split_size = x.size(self.split_dim) // 2
        return torch.split(x, split_size, dim=self.split_dim)
    
    def cat_output(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Concatenate output tensors along split dimension."""
        return torch.cat([y1, y2], dim=self.split_dim)


class AdditiveCoupling(BaseCoupling):
    """
    Additive coupling function: y1 = x1, y2 = x2 + F(x1)
    
    This is the most memory-efficient coupling as it only requires storing
    the transformation F(x1) during forward pass.
    """
    
    def __init__(self, d_model: int, d_hidden: Optional[int] = None, split_dim: int = -1):
        super().__init__(split_dim)
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model
        
        # Transformation network F
        self.transform_net = nn.Sequential(
            nn.Linear(d_model // 2, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, d_model // 2),
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward additive coupling: y1 = x1, y2 = x2 + F(x1)
        
        Args:
            x1: First half of input tensor
            x2: Second half of input tensor
            
        Returns:
            y1, y2: Transformed tensors
        """
        y1 = x1
        transform = self.transform_net(x1)
        y2 = x2 + transform
        return y1, y2
    
    def inverse(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse additive coupling: x1 = y1, x2 = y2 - F(y1)
        
        Args:
            y1: First half of output tensor
            y2: Second half of output tensor
            
        Returns:
            x1, x2: Original input tensors
        """
        x1 = y1
        transform = self.transform_net(y1)
        x2 = y2 - transform
        return x1, x2


class AffineCoupling(BaseCoupling):
    """
    Affine coupling function: y1 = x1, y2 = x2 * exp(s(x1)) + t(x1)
    
    More expressive than additive coupling but requires storing both
    scale and translation parameters.
    """
    
    def __init__(self, d_model: int, d_hidden: Optional[int] = None, 
                 split_dim: int = -1, scale_init: float = 0.1):
        super().__init__(split_dim)
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model
        self.scale_init = scale_init
        
        # Scale network s(x1)
        self.scale_net = nn.Sequential(
            nn.Linear(d_model // 2, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, d_model // 2),
        )
        
        # Translation network t(x1)
        self.translate_net = nn.Sequential(
            nn.Linear(d_model // 2, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, d_model // 2),
        )
        
        # Initialize scale weights for stability
        with torch.no_grad():
            self.scale_net[-1].weight.data *= self.scale_init
            self.scale_net[-1].bias.data.zero_()
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward affine coupling: y1 = x1, y2 = x2 * exp(s(x1)) + t(x1)
        
        Args:
            x1: First half of input tensor
            x2: Second half of input tensor
            
        Returns:
            y1, y2: Transformed tensors
        """
        y1 = x1
        scale = self.scale_net(x1)
        translate = self.translate_net(x1)
        y2 = x2 * torch.exp(scale) + translate
        return y1, y2
    
    def inverse(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse affine coupling: x1 = y1, x2 = (y2 - t(y1)) / exp(s(y1))
        
        Args:
            y1: First half of output tensor
            y2: Second half of output tensor
            
        Returns:
            x1, x2: Original input tensors
        """
        x1 = y1
        scale = self.scale_net(y1)
        translate = self.translate_net(y1)
        x2 = (y2 - translate) * torch.exp(-scale)
        return x1, x2


class LearnedCoupling(BaseCoupling):
    """
    Learned coupling with gating mechanism for improved expressiveness.
    
    Combines additive and multiplicative transformations with learnable gates.
    """
    
    def __init__(self, d_model: int, d_hidden: Optional[int] = None, split_dim: int = -1):
        super().__init__(split_dim)
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model
        
        # Main transformation network
        self.transform_net = nn.Sequential(
            nn.Linear(d_model // 2, self.d_hidden),
            nn.GELU(),
            nn.Linear(self.d_hidden, d_model // 2),
        )
        
        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(d_model // 2, self.d_hidden // 2),
            nn.GELU(),
            nn.Linear(self.d_hidden // 2, d_model // 2),
            nn.Sigmoid(),
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward learned coupling with gating.
        
        Args:
            x1: First half of input tensor
            x2: Second half of input tensor
            
        Returns:
            y1, y2: Transformed tensors
        """
        y1 = x1
        transform = self.transform_net(x1)
        gate = self.gate_net(x1)
        y2 = x2 + gate * transform
        return y1, y2
    
    def inverse(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse learned coupling.
        
        Args:
            y1: First half of output tensor
            y2: Second half of output tensor
            
        Returns:
            x1, x2: Original input tensors
        """
        x1 = y1
        transform = self.transform_net(y1)
        gate = self.gate_net(y1)
        x2 = y2 - gate * transform
        return x1, x2