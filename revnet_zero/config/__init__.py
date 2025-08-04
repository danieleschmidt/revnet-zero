"""
Configuration management system for RevNet-Zero.

This module provides flexible configuration management with
validation, presets, and environment variable support.
"""

from .config_manager import ConfigManager, load_config, save_config
from .presets import get_preset_config, list_presets
from .validation import validate_config, ConfigValidationError

__all__ = [
    "ConfigManager",
    "load_config",
    "save_config",
    "get_preset_config",
    "list_presets",
    "validate_config",
    "ConfigValidationError",
]