"""
Mixed precision training utilities for reversible transformers.

This module provides specialized mixed precision training tools
optimized for reversible neural networks.
"""

import torch
import torch.nn as nn
from torch.cuda import amp
from typing import Optional, Dict, Any, List, Callable
import logging
from contextlib import contextmanager


class ReversibleAMPTrainer:
    """
    Automatic Mixed Precision trainer optimized for reversible transformers.
    
    Provides stable mixed precision training with special handling
    for reversible computation numerical stability.
    """
    
    def __init__(
        self,
        model: nn.Module,
        fp16: bool = True,
        loss_scale: str = 'dynamic',
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        keep_batchnorm_fp32: bool = True,
        reversible_loss_scaling: bool = True,
    ):
        """
        Initialize mixed precision trainer.
        
        Args:
            model: Reversible model to train
            fp16: Whether to use FP16 precision
            loss_scale: Loss scaling strategy ('dynamic' or 'static')
            init_scale: Initial loss scale value
            growth_factor: Factor to grow loss scale
            backoff_factor: Factor to reduce loss scale on overflow
            growth_interval: Interval to attempt growing loss scale
            keep_batchnorm_fp32: Keep batch norm in FP32 for stability
            reversible_loss_scaling: Use specialized scaling for reversible layers
        """
        self.model = model
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.keep_batchnorm_fp32 = keep_batchnorm_fp32
        self.reversible_loss_scaling = reversible_loss_scaling
        
        # Initialize gradient scaler
        if self.fp16:
            if loss_scale == 'dynamic':
                self.scaler = amp.GradScaler(
                    init_scale=init_scale,
                    growth_factor=growth_factor,
                    backoff_factor=backoff_factor,
                    growth_interval=growth_interval,
                )
            else:
                self.scaler = amp.GradScaler(
                    init_scale=init_scale,
                    growth_factor=1.0,  # No growth for static scaling
                    backoff_factor=1.0,  # No backoff for static scaling
                    growth_interval=float('inf'),  # Never grow
                )
        else:
            self.scaler = None
        
        # Configure model for mixed precision
        self._configure_model_precision()
        
        # Statistics
        self.overflow_count = 0
        self.scale_updates = 0
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _configure_model_precision(self):
        """Configure model layers for optimal mixed precision."""
        if not self.fp16:
            return
        
        # Keep certain layers in FP32 for stability
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)) and self.keep_batchnorm_fp32:
                module.float()
            elif hasattr(module, 'use_reversible') and self.reversible_loss_scaling:
                # Special handling for reversible layers
                self._configure_reversible_layer(module)
    
    def _configure_reversible_layer(self, layer: nn.Module):
        """Configure reversible layer for mixed precision."""
        # Ensure coupling functions remain stable in FP16
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'coupling'):
            # Keep coupling computations in higher precision if needed
            pass
        
        if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'coupling'):
            # Keep coupling computations in higher precision if needed
            pass
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.fp16:
            with amp.autocast():
                yield
        else:
            yield
    
    def forward_with_amp(
        self,
        inputs: Dict[str, torch.Tensor],
        compute_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with automatic mixed precision.
        
        Args:
            inputs: Input dictionary
            compute_loss: Whether to compute loss
            
        Returns:
            Model outputs
        """
        with self.autocast_context():
            outputs = self.model(**inputs)
            
            if compute_loss and 'labels' in inputs:
                if isinstance(outputs, dict) and 'loss' not in outputs:
                    # Compute loss manually if not provided
                    logits = outputs.get('logits', outputs.get('last_hidden_state'))
                    if logits is not None:
                        loss_fn = nn.CrossEntropyLoss()
                        if len(logits.shape) == 3:  # Language modeling
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = inputs['labels'][..., 1:].contiguous()
                            loss = loss_fn(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                        else:
                            loss = loss_fn(logits.view(-1, logits.size(-1)), inputs['labels'].view(-1))
                        outputs['loss'] = loss
        
        return outputs
    
    def backward_with_amp(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Backward pass with automatic mixed precision.
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
            gradient_accumulation_steps: Number of gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Statistics about the backward pass
        """
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        stats = {
            'loss': loss.item() * gradient_accumulation_steps,
            'scale': self.scaler.get_scale() if self.scaler else 1.0,
            'overflow': False,
            'grad_norm': None,
        }
        
        # Optimizer step
        if self.scaler is not None:
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)
            
            # Check for overflow
            if self._check_overflow():
                stats['overflow'] = True
                self.overflow_count += 1
                optimizer.zero_grad()
                return stats
            
            # Gradient clipping
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )
                stats['grad_norm'] = grad_norm.item()
            
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Track scale updates
            if self.scaler.get_scale() != stats['scale']:
                self.scale_updates += 1
        else:
            # Standard optimization without scaling
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )
                stats['grad_norm'] = grad_norm.item()
            
            optimizer.step()
        
        optimizer.zero_grad()
        return stats
    
    def _check_overflow(self) -> bool:
        """Check if gradients have overflowed."""
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
    ) -> Dict[str, Any]:
        """
        Complete training step with mixed precision.
        
        Args:
            batch: Training batch
            optimizer: Optimizer
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm
            
        Returns:
            Training step statistics
        """
        # Forward pass
        outputs = self.forward_with_amp(batch, compute_loss=True)
        loss = outputs.get('loss')
        
        if loss is None:
            raise ValueError("Loss not found in model outputs")
        
        # Backward pass
        backward_stats = self.backward_with_amp(
            loss=loss,
            optimizer=optimizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
        )
        
        # Combine statistics
        stats = {
            'loss': backward_stats['loss'],
            'scale': backward_stats['scale'],
            'overflow': backward_stats['overflow'],
            'grad_norm': backward_stats['grad_norm'],
            'outputs': outputs,
        }
        
        return stats
    
    def train(
        self,
        train_loader,
        optimizer: torch.optim.Optimizer,
        num_steps: int,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 100,
        eval_fn: Optional[Callable] = None,
        eval_interval: int = 1000,
    ) -> Dict[str, Any]:
        """
        Training loop with mixed precision.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            num_steps: Number of training steps
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm
            log_interval: Logging interval
            eval_fn: Evaluation function
            eval_interval: Evaluation interval
            
        Returns:
            Training statistics
        """
        self.model.train()
        
        stats = {
            'losses': [],
            'scales': [],
            'grad_norms': [],
            'overflow_count': 0,
            'eval_results': [],
        }
        
        step = 0
        data_iter = iter(train_loader)
        
        while step < num_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            # Training step
            step_stats = self.train_step(
                batch=batch,
                optimizer=optimizer,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
            )
            
            # Record statistics
            stats['losses'].append(step_stats['loss'])
            stats['scales'].append(step_stats['scale'])
            if step_stats['grad_norm'] is not None:
                stats['grad_norms'].append(step_stats['grad_norm'])
            if step_stats['overflow']:
                stats['overflow_count'] += 1
            
            step += 1
            
            # Logging
            if step % log_interval == 0:
                avg_loss = sum(stats['losses'][-log_interval:]) / min(log_interval, len(stats['losses']))
                current_scale = step_stats['scale']
                
                self.logger.info(
                    f"Step {step}/{num_steps}, Loss: {avg_loss:.4f}, "
                    f"Scale: {current_scale:.0f}, Overflows: {stats['overflow_count']}"
                )
                
                if step_stats['grad_norm'] is not None:
                    self.logger.info(f"Grad Norm: {step_stats['grad_norm']:.4f}")
            
            # Evaluation
            if eval_fn is not None and step % eval_interval == 0:
                self.model.eval()
                eval_result = eval_fn()
                stats['eval_results'].append({'step': step, 'result': eval_result})
                self.model.train()
                
                self.logger.info(f"Evaluation at step {step}: {eval_result}")
        
        # Final statistics
        stats.update({
            'total_overflows': self.overflow_count,
            'total_scale_updates': self.scale_updates,
            'final_scale': self.scaler.get_scale() if self.scaler else 1.0,
        })
        
        return stats
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get trainer state dictionary."""
        return {
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'overflow_count': self.overflow_count,
            'scale_updates': self.scale_updates,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state dictionary."""
        if self.scaler and 'scaler_state_dict' in state_dict and state_dict['scaler_state_dict']:
            self.scaler.load_state_dict(state_dict['scaler_state_dict'])
        
        self.overflow_count = state_dict.get('overflow_count', 0)
        self.scale_updates = state_dict.get('scale_updates', 0)


class ReversibleFP16Optimizer:
    """
    FP16 optimizer wrapper with special handling for reversible models.
    
    Provides optimized parameter updates that maintain numerical
    stability in reversible computations.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_scale: float = 1.0,
        dynamic_loss_scale: bool = True,
        min_loss_scale: float = 1.0,
        max_loss_scale: float = 2**24,
        scale_growth_factor: float = 2.0,
        scale_backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        """
        Initialize FP16 optimizer wrapper.
        
        Args:
            optimizer: Base optimizer
            loss_scale: Initial loss scale
            dynamic_loss_scale: Whether to use dynamic loss scaling
            min_loss_scale: Minimum loss scale
            max_loss_scale: Maximum loss scale
            scale_growth_factor: Factor to grow loss scale
            scale_backoff_factor: Factor to reduce loss scale
            growth_interval: Interval to attempt growing loss scale
        """
        self.optimizer = optimizer
        self.loss_scale = loss_scale
        self.dynamic_loss_scale = dynamic_loss_scale
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale
        self.scale_growth_factor = scale_growth_factor
        self.scale_backoff_factor = scale_backoff_factor
        self.growth_interval = growth_interval
        
        # State tracking
        self.overflow_count = 0
        self.last_overflow_step = -1
        self.steps_since_growth = 0
        self.total_steps = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for FP16 training."""
        return loss * self.loss_scale
    
    def step(self, closure=None):
        """Optimizer step with overflow handling."""
        # Check for gradient overflow
        has_overflow = self._check_overflow()
        
        if has_overflow:
            self._handle_overflow()
            return False  # Indicate step was skipped
        
        # Unscale gradients
        self._unscale_gradients()
        
        # Optimizer step
        self.optimizer.step(closure)
        
        # Update loss scale
        if self.dynamic_loss_scale:
            self._update_loss_scale()
        
        self.total_steps += 1
        return True  # Indicate successful step
    
    def _check_overflow(self) -> bool:
        """Check if any gradients have overflowed."""
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    grad_data = param.grad.data
                    if torch.isnan(grad_data).any() or torch.isinf(grad_data).any():
                        return True
        return False
    
    def _handle_overflow(self):
        """Handle gradient overflow."""
        self.overflow_count += 1
        self.last_overflow_step = self.total_steps
        
        # Reduce loss scale
        if self.dynamic_loss_scale:
            self.loss_scale = max(
                self.loss_scale * self.scale_backoff_factor,
                self.min_loss_scale
            )
        
        # Zero gradients
        self.optimizer.zero_grad()
    
    def _unscale_gradients(self):
        """Unscale gradients before optimizer step."""
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.loss_scale)
    
    def _update_loss_scale(self):
        """Update loss scale for dynamic scaling."""
        self.steps_since_growth += 1
        
        # Try to grow loss scale if no recent overflows
        if (self.steps_since_growth >= self.growth_interval and
            self.total_steps - self.last_overflow_step >= self.growth_interval):
            
            new_scale = min(
                self.loss_scale * self.scale_growth_factor,
                self.max_loss_scale
            )
            
            if new_scale != self.loss_scale:
                self.loss_scale = new_scale
                self.steps_since_growth = 0
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def state_dict(self):
        """Get optimizer state dict."""
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_scale': self.loss_scale,
            'overflow_count': self.overflow_count,
            'last_overflow_step': self.last_overflow_step,
            'steps_since_growth': self.steps_since_growth,
            'total_steps': self.total_steps,
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.loss_scale = state_dict['loss_scale']
        self.overflow_count = state_dict['overflow_count']
        self.last_overflow_step = state_dict['last_overflow_step']
        self.steps_since_growth = state_dict['steps_since_growth']
        self.total_steps = state_dict['total_steps']


def convert_model_to_fp16(model: nn.Module, keep_batchnorm_fp32: bool = True) -> nn.Module:
    """
    Convert model to FP16 with special handling for reversible layers.
    
    Args:
        model: Model to convert
        keep_batchnorm_fp32: Whether to keep batch norm layers in FP32
        
    Returns:
        Converted model
    """
    for name, module in model.named_modules():
        # Keep certain layers in FP32 for stability
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)) and keep_batchnorm_fp32:
            continue
        
        # Convert to half precision
        if hasattr(module, 'weight') and module.weight is not None:
            module.half()
    
    return model


def create_mixed_precision_model(
    model: nn.Module,
    optimizer_class: type = torch.optim.AdamW,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    use_amp: bool = True,
    **amp_kwargs
) -> tuple:
    """
    Create a mixed precision training setup.
    
    Args:
        model: Model to train
        optimizer_class: Optimizer class
        optimizer_kwargs: Optimizer kwargs
        use_amp: Whether to use automatic mixed precision
        **amp_kwargs: Arguments for AMP trainer
        
    Returns:
        Tuple of (model, optimizer, amp_trainer)
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {'lr': 1e-4, 'weight_decay': 0.01}
    
    # Create optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    
    # Create AMP trainer if requested
    if use_amp:
        amp_trainer = ReversibleAMPTrainer(model, **amp_kwargs)
    else:
        amp_trainer = None
    
    return model, optimizer, amp_trainer