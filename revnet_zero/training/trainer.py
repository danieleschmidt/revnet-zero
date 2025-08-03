"""
Long-context training utilities for reversible transformers.

This module provides specialized training classes optimized for
very long sequence training with memory-efficient techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Union, Callable
import time
import math
from pathlib import Path

from ..memory.scheduler import MemoryScheduler, AdaptiveScheduler
from ..memory.profiler import MemoryProfiler


class LongContextTrainer:
    """
    Trainer specialized for long-context reversible transformer training.
    
    Handles memory management, gradient accumulation, and optimization
    for training with very long sequences (256k+ tokens).
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_length: int = 262144,
        gradient_accumulation_steps: int = 1,
        memory_scheduler: Optional[MemoryScheduler] = None,
        use_amp: bool = True,
        log_memory_usage: bool = True,
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.memory_scheduler = memory_scheduler
        self.use_amp = use_amp
        self.log_memory_usage = log_memory_usage
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Memory tracking
        self.memory_profiler = MemoryProfiler() if log_memory_usage else None
        
        # Mixed precision setup
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training statistics
        self.training_stats = {
            "steps": [],
            "losses": [],
            "memory_usage": [],
            "throughput": [],
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 1,
        num_steps: Optional[int] = None,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        logging_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Train the model with long-context optimization.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
            optimizer: Optimizer (will create AdamW if None)
            scheduler: Learning rate scheduler
            num_epochs: Number of training epochs
            num_steps: Maximum training steps (overrides epochs if set)
            eval_steps: Steps between evaluations
            save_steps: Steps between checkpoints
            logging_steps: Steps between logging
            
        Returns:
            Training statistics and final metrics
        """
        # Setup optimizer if not provided
        if optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
        
        # Setup memory scheduler if not provided
        if self.memory_scheduler is None:
            self.memory_scheduler = AdaptiveScheduler(self.model)
        
        self.model.train()
        
        if self.memory_profiler:
            self.memory_profiler.start_profiling()
        
        total_steps = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            for step, batch in enumerate(train_dataloader):
                if num_steps and total_steps >= num_steps:
                    break
                
                # Training step
                loss, metrics = self._training_step(batch, optimizer)
                
                total_steps += 1
                self.global_step = total_steps
                
                # Update training stats
                self.training_stats["steps"].append(total_steps)
                self.training_stats["losses"].append(loss)
                
                if self.log_memory_usage:
                    memory_usage = self._get_memory_usage()
                    self.training_stats["memory_usage"].append(memory_usage)
                
                # Logging
                if total_steps % logging_steps == 0:
                    self._log_training_progress(
                        epoch, step, total_steps, loss, metrics, start_time
                    )
                
                # Evaluation
                if eval_dataloader and total_steps % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    self._log_evaluation(total_steps, eval_metrics)
                    
                    # Save best model
                    if eval_metrics["loss"] < self.best_loss:
                        self.best_loss = eval_metrics["loss"]
                        if self.checkpoint_dir:
                            self._save_checkpoint("best_model.pt", optimizer, scheduler)
                
                # Checkpointing
                if self.checkpoint_dir and total_steps % save_steps == 0:
                    self._save_checkpoint(f"checkpoint_step_{total_steps}.pt", optimizer, scheduler)
                
                # Memory scheduler adaptation
                if hasattr(self.memory_scheduler, 'adapt_policies'):
                    self.memory_scheduler.adapt_policies(total_steps)
            
            if num_steps and total_steps >= num_steps:
                break
        
        if self.memory_profiler:
            self.memory_profiler.stop_profiling()
        
        # Final evaluation
        final_metrics = {}
        if eval_dataloader:
            final_metrics = self.evaluate(eval_dataloader)
        
        return {
            "training_stats": self.training_stats,
            "final_metrics": final_metrics,
            "total_steps": total_steps,
            "best_loss": self.best_loss,
        }
    
    def _training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        optimizer: optim.Optimizer
    ) -> tuple[float, Dict[str, Any]]:
        """
        Execute a single training step with gradient accumulation.
        
        Args:
            batch: Input batch
            optimizer: Optimizer
            
        Returns:
            Loss value and metrics
        """
        total_loss = 0.0
        
        # Move batch to device
        batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                for k, v in batch.items()}
        
        # Gradient accumulation loop
        for accum_step in range(self.gradient_accumulation_steps):
            # Get micro-batch
            micro_batch = self._get_micro_batch(batch, accum_step)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if self.memory_scheduler:
                    with self.memory_scheduler:
                        outputs = self.model(**micro_batch)
                else:
                    outputs = self.model(**micro_batch)
                
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
        
        # Optimizer step
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Collect metrics
        metrics = {
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": self._get_gradient_norm(),
        }
        
        if self.memory_scheduler:
            self.memory_scheduler.update_memory_stats()
            metrics.update({
                "memory_usage": self.memory_scheduler.current_memory_usage,
                "peak_memory": self.memory_scheduler.peak_memory_usage,
                "recomputed_layers": len(self.memory_scheduler.recomputed_layers),
            })
        
        return total_loss, metrics
    
    def _get_micro_batch(
        self, 
        batch: Dict[str, torch.Tensor], 
        accum_step: int
    ) -> Dict[str, torch.Tensor]:
        """
        Extract micro-batch for gradient accumulation.
        
        Args:
            batch: Full batch
            accum_step: Accumulation step index
            
        Returns:
            Micro-batch
        """
        micro_batch = {}
        batch_size = None
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                if batch_size is None:
                    batch_size = value.size(0)
                
                micro_size = batch_size // self.gradient_accumulation_steps
                start_idx = accum_step * micro_size
                end_idx = start_idx + micro_size
                
                micro_batch[key] = value[start_idx:end_idx]
            else:
                micro_batch[key] = value
        
        return micro_batch
    
    def _get_gradient_norm(self) -> float:
        """Calculate gradient norm for monitoring."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
            }
        else:
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    
    def _log_training_progress(
        self,
        epoch: int,
        step: int,
        global_step: int,
        loss: float,
        metrics: Dict[str, Any],
        start_time: float,
    ):
        """Log training progress."""
        elapsed_time = time.time() - start_time
        steps_per_sec = global_step / elapsed_time
        
        log_msg = (
            f"Epoch {epoch}, Step {step}, Global Step {global_step} | "
            f"Loss: {loss:.4f} | "
            f"LR: {metrics['learning_rate']:.2e} | "
            f"Grad Norm: {metrics['gradient_norm']:.2f} | "
            f"Speed: {steps_per_sec:.2f} steps/s"
        )
        
        if "memory_usage" in metrics:
            memory_gb = metrics["memory_usage"] / (1024**3)
            log_msg += f" | Memory: {memory_gb:.1f}GB"
        
        print(log_msg)
    
    def _log_evaluation(self, step: int, metrics: Dict[str, Any]):
        """Log evaluation results."""
        log_msg = f"Evaluation at step {step} | "
        for key, value in metrics.items():
            if isinstance(value, float):
                log_msg += f"{key}: {value:.4f} | "
        print(log_msg)
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model on evaluation dataset.
        
        Args:
            eval_dataloader: Evaluation data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch, use_reversible=False)  # Use standard mode for eval
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                total_loss += loss.item()
                if "input_ids" in batch:
                    total_tokens += batch["input_ids"].numel()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_batches": num_batches,
        }
    
    def _save_checkpoint(
        self,
        filename: str,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
        
        self.checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "training_stats": self.training_stats,
        }
        
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        load_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        self.training_stats = checkpoint.get("training_stats", {})
        
        # Load optimizer state
        if load_optimizer and optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")
        
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
        }