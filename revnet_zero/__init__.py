"""
RevNet-Zero: Memory-efficient reversible transformers for long-context training.

This library implements reversible transformer layers with on-the-fly activation
recomputation, reducing GPU memory usage by >70% during training.
"""

__version__ = "1.0.0"
__author__ = "RevNet-Zero Team"
__email__ = "team@revnet-zero.org"

# Initialize robust dependency management first
from .core.dependency_manager import get_dependency_manager, check_environment

# Get dependency manager and verify environment
_dm = get_dependency_manager()
_env_status = check_environment()

# Safe imports with fallback handling
try:
    from .models.reversible_transformer import ReversibleTransformer
    from .layers.reversible_attention import ReversibleAttention
    from .layers.reversible_ffn import ReversibleFFN
    from .layers.coupling_layers import AdditiveCoupling, AffineCoupling
    from .memory.scheduler import MemoryScheduler, AdaptiveScheduler
    from .utils.conversion import convert_to_reversible
    from .training.trainer import LongContextTrainer
    
    # Full functionality available
    __all__ = [
        "ReversibleTransformer",
        "ReversibleAttention", 
        "ReversibleFFN",
        "AdditiveCoupling",
        "AffineCoupling",
        "MemoryScheduler",
        "AdaptiveScheduler",
        "convert_to_reversible",
        "LongContextTrainer",
        "get_dependency_manager",
        "check_environment",
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Some RevNet-Zero components unavailable due to missing dependencies: {e}. "
        f"Environment status: {_env_status}. Limited functionality available.",
        ImportWarning
    )
    
    # Minimal exports for degraded environment
    __all__ = [
        "get_dependency_manager",
        "check_environment",
    ]

# Export environment info for debugging
def get_revnet_environment():
    """Get comprehensive RevNet-Zero environment information"""
    return {
        'version': __version__,
        'environment_status': _env_status,
        'mock_environment': _dm.is_mock_environment(),
        'available_components': __all__,
    }