"""
Distributed training utilities for reversible transformers.

This module provides tools for distributed training of reversible
models across multiple GPUs and nodes with memory optimization.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, Any, List, Tuple, Union
import os
import logging
from contextlib import contextmanager


class DistributedReversibleTrainer:
    """
    Distributed trainer for reversible transformer models.
    
    Supports various distributed training strategies optimized
    for memory-efficient reversible computation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        sharding_strategy: str = 'ddp',
        gradient_checkpointing_policy: str = 'selective',
        cpu_offload: bool = False,
        mixed_precision: bool = True,
        find_unused_parameters: bool = False,
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Reversible model to train
            sharding_strategy: Strategy for parameter sharding ('ddp', 'fsdp', 'deepspeed')
            gradient_checkpointing_policy: Gradient checkpointing policy
            cpu_offload: Whether to offload parameters to CPU
            mixed_precision: Whether to use mixed precision training
            find_unused_parameters: Whether to find unused parameters in DDP
        """
        self.model = model
        self.sharding_strategy = sharding_strategy
        self.gradient_checkpointing_policy = gradient_checkpointing_policy
        self.cpu_offload = cpu_offload
        self.mixed_precision = mixed_precision
        self.find_unused_parameters = find_unused_parameters
        
        # Initialize distributed training
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Set device
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        
        # Initialize process group
        if not dist.is_initialized() and self.world_size > 1:
            dist.init_process_group(backend='nccl')
        
        # Setup model for distributed training
        self._setup_distributed_model()
        
        # Setup mixed precision
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Setup logging
        self.logger = self._setup_logging()
    
    def _setup_distributed_model(self):
        """Setup model for distributed training."""
        self.model = self.model.to(self.device)
        
        if self.world_size > 1:
            if self.sharding_strategy == 'ddp':
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    find_unused_parameters=self.find_unused_parameters,
                )
            elif self.sharding_strategy == 'fsdp':
                self._setup_fsdp()
            elif self.sharding_strategy == 'deepspeed':
                self._setup_deepspeed()
            else:
                raise ValueError(f"Unknown sharding strategy: {self.sharding_strategy}")
    
    def _setup_fsdp(self):
        """Setup Fully Sharded Data Parallel."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch, ShardingStrategy
            
            # Define auto wrap policy for transformer blocks
            auto_wrap_policy = transformer_auto_wrap_policy
            
            # Mixed precision policy
            if self.mixed_precision:
                mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
            else:
                mixed_precision_policy = None
            
            # Sharding strategy
            if self.cpu_offload:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            
            self.model = FSDP(
                self.model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_policy,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                sharding_strategy=sharding_strategy,
                device_id=self.local_rank,
            )
            
        except ImportError:
            self.logger.warning("FSDP not available, falling back to DDP")
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.find_unused_parameters,
            )
    
    def _setup_deepspeed(self):
        """Setup DeepSpeed training."""
        try:
            import deepspeed
            
            # DeepSpeed configuration
            ds_config = {
                "train_batch_size": 16,  # Will be overridden
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 16,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": 1e-4,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01
                    }
                },
                "fp16": {
                    "enabled": self.mixed_precision,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                },
                "zero_optimization": {
                    "stage": 3 if self.cpu_offload else 2,
                    "offload_optimizer": {
                        "device": "cpu" if self.cpu_offload else "none"
                    },
                    "offload_param": {
                        "device": "cpu" if self.cpu_offload else "none"
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "sub_group_size": 1e9,
                    "reduce_bucket_size": 1e6,
                    "stage3_prefetch_bucket_size": 1e6,
                    "stage3_param_persistence_threshold": 1e4,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "gather_16bit_weights_on_model_save": True
                },
                "gradient_clipping": 1.0,
                "wall_clock_breakdown": False
            }
            
            # Initialize DeepSpeed
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                config=ds_config,
            )
            
            self.model = model_engine
            self.optimizer = optimizer
            
        except ImportError:
            self.logger.warning("DeepSpeed not available, falling back to DDP")
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.find_unused_parameters,
            )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for distributed training."""
        logger = logging.getLogger(f'DistributedTrainer_rank_{self.rank}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train(
        self,
        train_dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: int = 1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 100,
        save_interval: int = 1000,
        checkpoint_dir: Optional[str] = None,
        log_memory_usage: bool = True,
    ) -> Dict[str, Any]:
        """
        Run distributed training.
        
        Args:
            train_dataloader: Training dataloader
            optimizer: Optimizer (if not using DeepSpeed)
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: Logging interval in steps
            save_interval: Checkpoint saving interval in steps
            checkpoint_dir: Directory to save checkpoints
            log_memory_usage: Whether to log memory usage
            
        Returns:
            Training statistics
        """
        # Initialize optimizer if not provided and not using DeepSpeed
        if optimizer is None and self.sharding_strategy != 'deepspeed':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
        
        # Training statistics
        stats = {
            'total_steps': 0,
            'total_loss': 0.0,
            'epoch_losses': [],
            'memory_stats': [],
        }
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            step_count = 0
            
            # Set epoch for distributed sampler
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass with mixed precision
                with self._autocast_context():
                    outputs = self.model(batch['input_ids'], labels=batch.get('labels'))
                    
                    if isinstance(outputs, dict):
                        loss = outputs['loss']
                    else:
                        loss = outputs[0] if isinstance(outputs, tuple) else outputs
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass
                if self.sharding_strategy == 'deepspeed':
                    self.model.backward(loss)
                else:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                # Accumulate loss
                epoch_loss += loss.item() * gradient_accumulation_steps
                stats['total_loss'] += loss.item() * gradient_accumulation_steps
                
                # Gradient update
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if self.sharding_strategy == 'deepspeed':
                        self.model.step()
                    else:
                        # Gradient clipping
                        if self.scaler is not None:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                            optimizer.step()
                        
                        optimizer.zero_grad()
                    
                    stats['total_steps'] += 1
                    step_count += 1
                    
                    # Logging
                    if stats['total_steps'] % log_interval == 0:
                        avg_loss = epoch_loss / step_count if step_count > 0 else 0
                        
                        if self.rank == 0:
                            self.logger.info(
                                f"Epoch {epoch+1}/{num_epochs}, Step {stats['total_steps']}, "
                                f"Loss: {avg_loss:.4f}"
                            )
                            
                            if log_memory_usage and torch.cuda.is_available():
                                memory_stats = self._get_memory_stats()
                                stats['memory_stats'].append(memory_stats)
                                self.logger.info(
                                    f"Memory - Allocated: {memory_stats['allocated_gb']:.2f}GB, "
                                    f"Cached: {memory_stats['cached_gb']:.2f}GB"
                                )
                    
                    # Checkpointing
                    if checkpoint_dir and stats['total_steps'] % save_interval == 0:
                        self._save_checkpoint(
                            checkpoint_dir,
                            epoch,
                            stats['total_steps'],
                            optimizer if self.sharding_strategy != 'deepspeed' else None,
                        )
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            stats['epoch_losses'].append(avg_epoch_loss)
            
            if self.rank == 0:
                self.logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        return stats
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to appropriate device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device, non_blocking=True)
            else:
                moved_batch[key] = value
        return moved_batch
    
    @contextmanager
    def _autocast_context(self):
        """Context manager for mixed precision."""
        if self.mixed_precision and self.sharding_strategy != 'deepspeed':
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'cached_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
            }
        else:
            return {'allocated_gb': 0, 'cached_gb': 0, 'max_allocated_gb': 0}
    
    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Save training checkpoint."""
        if self.rank != 0:
            return  # Only save on rank 0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self._get_model_state_dict(),
        }
        
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if self.scaler is not None:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _get_model_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict, handling different distributed strategies."""
        if self.sharding_strategy == 'deepspeed':
            return self.model.module.state_dict()
        elif hasattr(self.model, 'module'):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint['epoch'], checkpoint['step']
    
    def cleanup(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Setup distributed dataloader with appropriate sampler.
    
    Args:
        dataset: Training dataset
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        Configured DataLoader
    """
    # Use distributed sampler if in distributed mode
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=drop_last,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class ReversibleModelSharding:
    """
    Specialized sharding utilities for reversible models.
    
    Provides optimized parameter sharding that takes into account
    the unique structure of reversible transformers.
    """
    
    @staticmethod
    def analyze_memory_distribution(model: nn.Module) -> Dict[str, Any]:
        """
        Analyze memory distribution across model components.
        
        Args:
            model: Reversible model to analyze
            
        Returns:
            Memory distribution analysis
        """
        memory_breakdown = {}
        total_params = 0
        
        # Analyze embeddings
        embedding_params = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                params = sum(p.numel() for p in module.parameters())
                embedding_params += params
                memory_breakdown[f'embedding_{name}'] = params
        
        # Analyze transformer layers
        layer_params = []
        for name, module in model.named_modules():
            if 'layer' in name.lower() or 'block' in name.lower():
                if hasattr(module, 'attention') or hasattr(module, 'feed_forward'):
                    params = sum(p.numel() for p in module.parameters())
                    layer_params.append(params)
                    memory_breakdown[f'layer_{name}'] = params
        
        # Calculate totals
        total_params = sum(p.numel() for p in model.parameters())
        layer_total = sum(layer_params)
        
        return {
            'total_parameters': total_params,
            'embedding_parameters': embedding_params,
            'layer_parameters': layer_total,
            'other_parameters': total_params - embedding_params - layer_total,
            'parameters_per_layer': layer_params,
            'memory_breakdown': memory_breakdown,
            'parameter_distribution': {
                'embeddings': embedding_params / total_params if total_params > 0 else 0,
                'layers': layer_total / total_params if total_params > 0 else 0,
                'other': (total_params - embedding_params - layer_total) / total_params if total_params > 0 else 0,
            }
        }
    
    @staticmethod
    def recommend_sharding_strategy(
        model: nn.Module,
        available_memory_gb: float,
        num_gpus: int,
    ) -> Dict[str, Any]:
        """
        Recommend optimal sharding strategy based on model size and available resources.
        
        Args:
            model: Model to analyze
            available_memory_gb: Available memory per GPU in GB
            num_gpus: Number of available GPUs
            
        Returns:
            Sharding strategy recommendations
        """
        analysis = ReversibleModelSharding.analyze_memory_distribution(model)
        
        # Estimate memory requirements
        param_memory_gb = analysis['total_parameters'] * 4 / (1024**3)  # Assuming float32
        gradient_memory_gb = param_memory_gb  # Gradients same size as parameters
        optimizer_memory_gb = param_memory_gb * 2  # Adam optimizer states
        
        total_memory_per_gpu = param_memory_gb + gradient_memory_gb + optimizer_memory_gb
        
        recommendations = {
            'estimated_memory_per_gpu_gb': total_memory_per_gpu,
            'available_memory_per_gpu_gb': available_memory_gb,
            'memory_ratio': total_memory_per_gpu / available_memory_gb,
            'recommended_strategies': [],
        }
        
        # Make recommendations based on memory requirements
        if total_memory_per_gpu <= available_memory_gb * 0.8:  # 80% threshold
            recommendations['recommended_strategies'].append({
                'strategy': 'ddp',
                'reason': 'Model fits comfortably in GPU memory',
                'cpu_offload': False,
            })
        elif total_memory_per_gpu <= available_memory_gb * 1.2:  # Can fit with optimization
            recommendations['recommended_strategies'].append({
                'strategy': 'ddp',
                'reason': 'Model fits with memory optimization',
                'cpu_offload': False,
                'mixed_precision': True,
            })
        else:
            # Need advanced sharding
            recommendations['recommended_strategies'].extend([
                {
                    'strategy': 'fsdp',
                    'reason': 'Model too large for DDP, need parameter sharding',
                    'cpu_offload': total_memory_per_gpu > available_memory_gb * 2,
                    'mixed_precision': True,
                },
                {
                    'strategy': 'deepspeed',
                    'reason': 'Maximum memory efficiency needed',
                    'zero_stage': 3 if total_memory_per_gpu > available_memory_gb * 3 else 2,
                    'cpu_offload': True,
                }
            ])
        
        return recommendations