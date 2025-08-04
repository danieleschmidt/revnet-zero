"""
Configuration manager for RevNet-Zero.

Handles loading, saving, and managing model and training configurations
with support for environment variables and validation.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
import logging

from ..utils.validation import validate_model_config, RevNetConfigurationError


@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    vocab_size: int = 50000
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.1
    coupling: str = "additive"
    use_flash_attention: bool = False
    use_rational_attention: bool = False
    layer_norm_eps: float = 1e-5
    tie_weights: bool = True


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 100000
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_loss_scale: str = "dynamic"


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    memory_budget: Optional[int] = None
    use_reversible: bool = True
    recompute_granularity: str = "layer"
    checkpoint_segments: int = 4
    gradient_checkpointing: bool = False
    cpu_offload: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: Optional[str] = None
    structured_logging: bool = True
    console_output: bool = True
    log_interval: int = 100
    save_logs: bool = True


@dataclass
class RevNetConfig:
    """Complete RevNet-Zero configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Meta information
    config_version: str = "1.0"
    description: str = ""
    created_by: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RevNetConfig':
        """Create from dictionary."""
        return cls(
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {})),
            memory=MemoryConfig(**data.get('memory', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            config_version=data.get('config_version', '1.0'),
            description=data.get('description', ''),
            created_by=data.get('created_by', ''),
        )


class ConfigManager:
    """
    Configuration manager for RevNet-Zero.
    
    Handles loading, saving, validation, and environment variable
    integration for model and training configurations.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        if config_dir is None:
            config_dir = Path.home() / ".revnet_zero" / "configs"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def load_config(
        self,
        config_path: Union[str, Path],
        validate: bool = True,
        apply_env_vars: bool = True,
    ) -> RevNetConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            validate: Whether to validate configuration
            apply_env_vars: Whether to apply environment variable overrides
            
        Returns:
            Loaded configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            RevNetConfigurationError: If configuration is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
        else:
            raise RevNetConfigurationError(f"Unsupported config format: {config_path.suffix}")
        
        # Create config object
        config = RevNetConfig.from_dict(data)
        
        # Apply environment variable overrides
        if apply_env_vars:
            config = self._apply_env_overrides(config)
        
        # Validate if requested
        if validate:
            self._validate_config(config)
        
        self.logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def save_config(
        self,
        config: RevNetConfig,
        config_path: Union[str, Path],
        format: str = "yaml",
    ):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
            format: File format ("yaml" or "json")
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = config.to_dict()
        
        if format.lower() == "yaml":
            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved configuration to {config_path}")
    
    def create_default_config(self, name: str = "default") -> RevNetConfig:
        """
        Create a default configuration.
        
        Args:
            name: Configuration name
            
        Returns:
            Default configuration
        """
        config = RevNetConfig()
        config.description = f"Default RevNet-Zero configuration"
        config.created_by = "RevNet-Zero ConfigManager"
        
        # Save to config directory
        config_path = self.config_dir / f"{name}.yaml"
        self.save_config(config, config_path)
        
        return config
    
    def list_configs(self) -> List[str]:
        """
        List available configuration files.
        
        Returns:
            List of configuration file names
        """
        config_files = []
        for pattern in ["*.yaml", "*.yml", "*.json"]:
            config_files.extend(self.config_dir.glob(pattern))
        
        return [f.stem for f in config_files]
    
    def get_config_path(self, name: str, format: str = "yaml") -> Path:
        """
        Get path for a named configuration.
        
        Args:
            name: Configuration name
            format: File format
            
        Returns:
            Configuration file path
        """
        extension = "yaml" if format == "yaml" else "json"
        return self.config_dir / f"{name}.{extension}"
    
    def _apply_env_overrides(self, config: RevNetConfig) -> RevNetConfig:
        """Apply environment variable overrides."""
        env_mappings = {
            # Model config
            "REVNET_VOCAB_SIZE": ("model", "vocab_size", int),
            "REVNET_NUM_LAYERS": ("model", "num_layers", int),
            "REVNET_D_MODEL": ("model", "d_model", int),
            "REVNET_NUM_HEADS": ("model", "num_heads", int),
            "REVNET_D_FF": ("model", "d_ff", int),
            "REVNET_MAX_SEQ_LEN": ("model", "max_seq_len", int),
            "REVNET_DROPOUT": ("model", "dropout", float),
            "REVNET_COUPLING": ("model", "coupling", str),
            
            # Training config
            "REVNET_LEARNING_RATE": ("training", "learning_rate", float),
            "REVNET_BATCH_SIZE": ("training", "batch_size", int),
            "REVNET_MAX_STEPS": ("training", "max_steps", int),
            "REVNET_WARMUP_STEPS": ("training", "warmup_steps", int),
            "REVNET_WEIGHT_DECAY": ("training", "weight_decay", float),
            
            # Memory config
            "REVNET_USE_REVERSIBLE": ("memory", "use_reversible", lambda x: x.lower() == "true"),
            "REVNET_MEMORY_BUDGET": ("memory", "memory_budget", int),
            "REVNET_RECOMPUTE_GRANULARITY": ("memory", "recompute_granularity", str),
            
            # Logging config
            "REVNET_LOG_LEVEL": ("logging", "level", str),
            "REVNET_LOG_DIR": ("logging", "log_dir", str),
        }
        
        for env_var, (section, key, type_converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = type_converter(os.environ[env_var])
                    section_obj = getattr(config, section)
                    setattr(section_obj, key, value)
                    self.logger.debug(f"Applied env override: {env_var} = {value}")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid env var {env_var}: {e}")
        
        return config
    
    def _validate_config(self, config: RevNetConfig):
        """Validate configuration."""
        try:
            # Validate model config
            model_dict = asdict(config.model)
            validate_model_config(model_dict)
            
            # Additional validations
            if config.training.learning_rate <= 0:
                raise RevNetConfigurationError("learning_rate must be positive")
            
            if config.training.batch_size <= 0:
                raise RevNetConfigurationError("batch_size must be positive")
            
            if config.training.max_steps <= 0:
                raise RevNetConfigurationError("max_steps must be positive")
            
            if config.memory.recompute_granularity not in ["layer", "block", "attention"]:
                raise RevNetConfigurationError(
                    "recompute_granularity must be one of ['layer', 'block', 'attention']"
                )
            
            if config.logging.level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise RevNetConfigurationError(
                    "logging level must be one of ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']"
                )
            
        except Exception as e:
            raise RevNetConfigurationError(f"Configuration validation failed: {e}")
    
    def merge_configs(self, base_config: RevNetConfig, override_config: RevNetConfig) -> RevNetConfig:
        """
        Merge two configurations, with override_config taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        # Convert to dictionaries
        base_dict = base_config.to_dict()
        override_dict = override_config.to_dict()
        
        # Deep merge
        merged_dict = self._deep_merge(base_dict, override_dict)
        
        return RevNetConfig.from_dict(merged_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


# Convenience functions
def load_config(config_path: Union[str, Path], **kwargs) -> RevNetConfig:
    """Load configuration from file."""
    manager = ConfigManager()
    return manager.load_config(config_path, **kwargs)


def save_config(config: RevNetConfig, config_path: Union[str, Path], **kwargs):
    """Save configuration to file."""
    manager = ConfigManager()
    manager.save_config(config, config_path, **kwargs)