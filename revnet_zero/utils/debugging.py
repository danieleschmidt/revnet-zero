"""
Debugging utilities for reversible transformers.

This module provides tools for validating reversible computations,
checking gradients, and debugging memory issues.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from collections import defaultdict
import warnings


class ReversibleGradientChecker:
    """
    Utility for checking gradient correctness in reversible layers.
    
    Compares gradients computed through reversible computation
    against standard backpropagation to ensure correctness.
    """
    
    def __init__(self, tolerance: float = 1e-6, verbose: bool = True):
        self.tolerance = tolerance
        self.verbose = verbose
    
    def check_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ) -> Dict[str, bool]:
        """
        Check gradient correctness for all reversible layers in a model.
        
        Args:
            model: Model to check
            input_shape: Shape of input tensor for testing
            device: Device to run tests on
            
        Returns:
            Dictionary mapping layer names to correctness status
        """
        if device is None:
            device = next(model.parameters()).device
        
        results = {}
        
        # Find all reversible layers
        reversible_layers = self._find_reversible_layers(model)
        
        for name, layer in reversible_layers.items():
            if self.verbose:
                print(f"Checking layer: {name}")
            
            try:
                is_correct = self.check_layer(layer, input_shape, device)
                results[name] = is_correct
                
                if self.verbose:
                    status = "✓" if is_correct else "✗"
                    print(f"  {status} Gradient check {'passed' if is_correct else 'failed'}")
                    
            except Exception as e:
                results[name] = False
                if self.verbose:
                    print(f"  ✗ Error during check: {e}")
        
        return results
    
    def check_layer(
        self,
        layer: nn.Module,
        input_shape: Tuple[int, ...],
        device: torch.device,
    ) -> bool:
        """
        Check gradient correctness for a single reversible layer.
        
        Args:
            layer: Reversible layer to check
            input_shape: Shape of input tensor
            device: Device to run test on
            
        Returns:
            Whether gradients are correct within tolerance
        """
        # Create test input
        x = torch.randn(input_shape, device=device, requires_grad=True)
        
        # Clone for non-reversible computation
        x_standard = x.clone().detach().requires_grad_(True)
        layer_standard = self._clone_layer_for_standard_computation(layer)
        
        # Forward pass with reversible computation
        layer.set_reversible_mode(True)
        y_rev = layer(x)
        
        # Forward pass with standard computation
        layer_standard.set_reversible_mode(False)
        y_std = layer_standard(x_standard)
        
        # Check forward pass similarity
        if not torch.allclose(y_rev, y_std, atol=self.tolerance):
            if self.verbose:
                max_diff = torch.max(torch.abs(y_rev - y_std)).item()
                print(f"    Forward pass differs by {max_diff}")
            return False
        
        # Create dummy loss
        loss_rev = y_rev.sum()
        loss_std = y_std.sum()
        
        # Backward pass
        loss_rev.backward(retain_graph=True)
        loss_std.backward(retain_graph=True)
        
        # Check input gradients
        if not torch.allclose(x.grad, x_standard.grad, atol=self.tolerance):
            if self.verbose:
                max_diff = torch.max(torch.abs(x.grad - x_standard.grad)).item()
                print(f"    Input gradients differ by {max_diff}")
            return False
        
        # Check parameter gradients
        for (name_rev, param_rev), (name_std, param_std) in zip(
            layer.named_parameters(), layer_standard.named_parameters()
        ):
            if param_rev.grad is None or param_std.grad is None:
                continue
                
            if not torch.allclose(param_rev.grad, param_std.grad, atol=self.tolerance):
                if self.verbose:
                    max_diff = torch.max(torch.abs(param_rev.grad - param_std.grad)).item()
                    print(f"    Parameter {name_rev} gradients differ by {max_diff}")
                return False
        
        return True
    
    def _find_reversible_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Find all reversible layers in a model."""
        reversible_layers = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'set_reversible_mode'):
                reversible_layers[name] = module
        
        return reversible_layers
    
    def _clone_layer_for_standard_computation(self, layer: nn.Module) -> nn.Module:
        """Clone a layer for standard (non-reversible) computation."""
        # This is a simplified version - in practice, you'd need more sophisticated cloning
        import copy
        cloned = copy.deepcopy(layer)
        return cloned


class StabilityAnalyzer:
    """
    Analyze numerical stability of reversible computations.
    
    Tracks gradient norms, activation statistics, and potential
    numerical issues during training.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_norms = []
        self.activation_stats = defaultdict(list)
        self.hooks = []
    
    def analyze(
        self,
        num_forward_passes: int = 10,
        input_shape: Tuple[int, ...] = (2, 512, 768),
        check_gradient_norm: bool = True,
        check_activation_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze numerical stability over multiple forward passes.
        
        Args:
            num_forward_passes: Number of forward passes to analyze
            input_shape: Shape of test inputs
            check_gradient_norm: Whether to check gradient norms
            check_activation_stats: Whether to collect activation statistics
            
        Returns:
            Dictionary with stability analysis results
        """
        device = next(self.model.parameters()).device
        
        if check_activation_stats:
            self._register_activation_hooks()
        
        gradient_norms = []
        activation_means = defaultdict(list)
        activation_stds = defaultdict(list)
        
        for i in range(num_forward_passes):
            # Create random input
            x = torch.randn(input_shape, device=device, requires_grad=True)
            
            # Forward pass
            output = self.model(x)
            if isinstance(output, dict):
                loss = output.get('loss', output['logits'].sum())
            else:
                loss = output.sum() if not hasattr(output, 'sum') else output.sum()
            
            # Backward pass
            if check_gradient_norm:
                loss.backward(retain_graph=True)
                
                # Calculate gradient norm
                total_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                gradient_norms.append(total_norm)
                
                # Clear gradients
                self.model.zero_grad()
        
        if check_activation_stats:
            # Process activation statistics
            for name, stats in self.activation_stats.items():
                means = [s['mean'] for s in stats]
                stds = [s['std'] for s in stats]
                activation_means[name] = means
                activation_stds[name] = stds
            
            self._remove_activation_hooks()
        
        return {
            'max_grad_norm': max(gradient_norms) if gradient_norms else None,
            'mean_grad_norm': np.mean(gradient_norms) if gradient_norms else None,
            'grad_norm_std': np.std(gradient_norms) if gradient_norms else None,
            'gradient_norms': gradient_norms,
            'activation_means': dict(activation_means),
            'activation_stds': dict(activation_stds),
            'potential_issues': self._identify_potential_issues(
                gradient_norms, activation_means, activation_stds
            ),
        }
    
    def _register_activation_hooks(self):
        """Register hooks to collect activation statistics."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    stats = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                    }
                    self.activation_stats[name].append(stats)
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(handle)
    
    def _remove_activation_hooks(self):
        """Remove activation hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _identify_potential_issues(
        self,
        gradient_norms: List[float],
        activation_means: Dict[str, List[float]],
        activation_stds: Dict[str, List[float]],
    ) -> List[str]:
        """Identify potential numerical stability issues."""
        issues = []
        
        if gradient_norms:
            max_grad_norm = max(gradient_norms)
            if max_grad_norm > 100:
                issues.append(f"Very large gradient norm detected: {max_grad_norm:.2f}")
            elif max_grad_norm < 1e-6:
                issues.append(f"Very small gradient norm detected: {max_grad_norm:.2e}")
        
        for name, means in activation_means.items():
            if means:
                max_mean = max(abs(m) for m in means)
                if max_mean > 100:
                    issues.append(f"Large activation magnitudes in {name}: {max_mean:.2f}")
        
        for name, stds in activation_stds.items():
            if stds:
                min_std = min(stds)
                max_std = max(stds)
                if min_std < 1e-6:
                    issues.append(f"Very small activation std in {name}: {min_std:.2e}")
                if max_std > 100:
                    issues.append(f"Very large activation std in {name}: {max_std:.2f}")
        
        return issues


class MemoryLeakDetector:
    """
    Detect memory leaks in reversible computations.
    
    Monitors GPU memory usage and identifies potential leaks
    during training or inference.
    """
    
    def __init__(self):
        self.memory_snapshots = []
        self.baseline_memory = None
    
    def start_monitoring(self):
        """Start monitoring memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.baseline_memory = torch.cuda.memory_allocated()
        else:
            self.baseline_memory = 0
        self.memory_snapshots = []
    
    def take_snapshot(self, label: str = ""):
        """Take a memory usage snapshot."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            memory_diff = current_memory - self.baseline_memory
            
            snapshot = {
                'label': label,
                'total_memory': current_memory,
                'memory_diff': memory_diff,
                'cached_memory': torch.cuda.memory_reserved(),
            }
            self.memory_snapshots.append(snapshot)
        
    def analyze_leaks(self, threshold_mb: float = 100.0) -> Dict[str, Any]:
        """
        Analyze memory snapshots for potential leaks.
        
        Args:
            threshold_mb: Memory increase threshold in MB to flag as potential leak
            
        Returns:
            Analysis results
        """
        if not self.memory_snapshots:
            return {"status": "No snapshots available"}
        
        threshold_bytes = threshold_mb * 1024 * 1024
        potential_leaks = []
        max_memory_increase = 0
        
        for i, snapshot in enumerate(self.memory_snapshots):
            memory_increase = snapshot['memory_diff']
            max_memory_increase = max(max_memory_increase, memory_increase)
            
            if memory_increase > threshold_bytes:
                potential_leaks.append({
                    'snapshot_index': i,
                    'label': snapshot['label'],
                    'memory_increase_mb': memory_increase / (1024 * 1024),
                })
        
        return {
            'baseline_memory_mb': self.baseline_memory / (1024 * 1024) if self.baseline_memory else 0,
            'max_memory_increase_mb': max_memory_increase / (1024 * 1024),
            'potential_leaks': potential_leaks,
            'total_snapshots': len(self.memory_snapshots),
            'snapshots': self.memory_snapshots,
        }


class CommonPitfalls:
    """
    Checker for common pitfalls in reversible transformer implementations.
    
    Identifies configuration issues, implementation problems, and
    provides suggestions for fixes.
    """
    
    def __init__(self):
        self.issues = []
    
    def check(
        self,
        model: nn.Module,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        Check for common pitfalls and issues.
        
        Args:
            model: Model to check
            training_config: Training configuration dictionary
            
        Returns:
            List of issues found
        """
        self.issues = []
        
        # Check model architecture
        self._check_model_architecture(model)
        
        # Check training configuration
        if training_config:
            self._check_training_config(training_config)
        
        # Check memory scheduler configuration
        self._check_memory_scheduler(model)
        
        return self.issues
    
    def _check_model_architecture(self, model: nn.Module):
        """Check model architecture for common issues."""
        # Check layer count
        if hasattr(model, 'num_layers'):
            if model.num_layers < 6:
                self.issues.append({
                    'description': 'Very few layers may not benefit significantly from reversible computation',
                    'solution': 'Consider using standard transformers for small models (<6 layers)',
                    'severity': 'warning'
                })
            elif model.num_layers > 48:
                self.issues.append({
                    'description': 'Very deep models may have gradient flow issues',
                    'solution': 'Monitor gradient norms and consider gradient clipping',
                    'severity': 'warning'
                })
        
        # Check hidden dimension
        if hasattr(model, 'd_model'):
            if model.d_model % 64 != 0:
                self.issues.append({
                    'description': 'Hidden dimension not divisible by 64 may reduce GPU efficiency',
                    'solution': 'Use hidden dimensions divisible by 64 (e.g., 768, 1024)',
                    'severity': 'info'
                })
        
        # Check attention heads
        if hasattr(model, 'num_heads') and hasattr(model, 'd_model'):
            if model.d_model % model.num_heads != 0:
                self.issues.append({
                    'description': 'Hidden dimension not divisible by number of attention heads',
                    'solution': 'Ensure d_model is divisible by num_heads',
                    'severity': 'error'
                })
    
    def _check_training_config(self, config: Dict[str, Any]):
        """Check training configuration for issues."""
        # Check batch size
        batch_size = config.get('batch_size', 1)
        if batch_size > 8:
            self.issues.append({
                'description': 'Large batch sizes may negate memory benefits of reversible computation',
                'solution': 'Consider using smaller batch sizes with gradient accumulation',
                'severity': 'warning'
            })
        
        # Check sequence length
        seq_len = config.get('max_seq_len', 512)
        if seq_len < 2048:
            self.issues.append({
                'description': 'Short sequences may not benefit from reversible computation overhead',
                'solution': 'Reversible computation is most beneficial for sequences >2048 tokens',
                'severity': 'info'
            })
        
        # Check learning rate
        lr = config.get('learning_rate', 1e-4)
        if lr > 1e-3:
            self.issues.append({
                'description': 'High learning rate may cause instability with reversible computation',
                'solution': 'Consider using lower learning rates (1e-4 to 5e-4)',
                'severity': 'warning'
            })
    
    def _check_memory_scheduler(self, model: nn.Module):
        """Check memory scheduler configuration."""
        if hasattr(model, 'memory_scheduler'):
            if model.memory_scheduler is None:
                self.issues.append({
                    'description': 'No memory scheduler configured',
                    'solution': 'Consider using AdaptiveScheduler for optimal memory usage',
                    'severity': 'info'
                })
        
        # Check if reversible mode is enabled
        reversible_layers = []
        for module in model.modules():
            if hasattr(module, 'use_reversible'):
                reversible_layers.append(module)
        
        if reversible_layers:
            disabled_layers = [layer for layer in reversible_layers if not layer.use_reversible]
            if disabled_layers:
                self.issues.append({
                    'description': f'{len(disabled_layers)} reversible layers have reversible mode disabled',
                    'solution': 'Enable reversible mode with model.set_reversible_mode(True)',
                    'severity': 'warning'
                })


def validate_reversible_computation(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (2, 512, 768),
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive validation of reversible computation.
    
    Args:
        model: Reversible model to validate
        input_shape: Shape of test inputs
        tolerance: Numerical tolerance for comparisons
        verbose: Whether to print detailed results
        
    Returns:
        Validation results dictionary
    """
    results = {}
    
    # Gradient checking
    if verbose:
        print("🔍 Checking gradient correctness...")
    grad_checker = ReversibleGradientChecker(tolerance=tolerance, verbose=verbose)
    gradient_results = grad_checker.check_model(model, input_shape)
    results['gradient_check'] = gradient_results
    
    # Stability analysis
    if verbose:
        print("📊 Analyzing numerical stability...")
    stability_analyzer = StabilityAnalyzer(model)
    stability_results = stability_analyzer.analyze(
        num_forward_passes=5,
        input_shape=input_shape,
    )
    results['stability_analysis'] = stability_results
    
    # Common pitfalls
    if verbose:
        print("⚠️  Checking for common pitfalls...")
    pitfall_checker = CommonPitfalls()
    pitfall_results = pitfall_checker.check(model)
    results['common_pitfalls'] = pitfall_results
    
    # Overall assessment
    gradient_passed = all(gradient_results.values())
    has_stability_issues = len(stability_results.get('potential_issues', [])) > 0
    has_critical_pitfalls = any(
        issue['severity'] == 'error' for issue in pitfall_results
    )
    
    overall_status = "PASS"
    if not gradient_passed or has_critical_pitfalls:
        overall_status = "FAIL"
    elif has_stability_issues:
        overall_status = "WARNING"
    
    results['overall_status'] = overall_status
    
    if verbose:
        print(f"\n🏁 Overall validation status: {overall_status}")
        if overall_status == "FAIL":
            print("❌ Critical issues found - model may not work correctly")
        elif overall_status == "WARNING":
            print("⚠️  Some issues found - monitor during training")
        else:
            print("✅ All checks passed - model ready for use")
    
    return results