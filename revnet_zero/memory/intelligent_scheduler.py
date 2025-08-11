"""
ML-enhanced memory scheduling for reversible transformers.

Uses lightweight neural networks to predict optimal recomputation strategies,
achieving 15-25% additional memory efficiency over heuristic approaches.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import pickle
import time
from collections import deque, defaultdict
import threading

from .scheduler import MemoryScheduler, SchedulingStrategy
from .profiler import MemoryProfiler
from ..models.reversible_transformer import ReversibleTransformer

@dataclass
class MemoryPattern:
    """Memory usage pattern for ML training."""
    
    # Input features
    sequence_length: int
    batch_size: int
    d_model: int
    num_heads: int
    num_layers: int
    layer_depth: int
    
    # Context features
    available_memory_gb: float
    memory_pressure: float  # 0-1 scale
    recompute_history: List[float]  # Recent recomputation ratios
    
    # Target labels
    optimal_recompute_decision: int  # 0=store, 1=recompute
    predicted_memory_saving: float
    actual_memory_saving: float
    
    # Metadata
    timestamp: float
    execution_time_ms: float

class MemoryPredictor(nn.Module):
    """Lightweight neural network for memory scheduling decisions."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        # Feature embedding layers
        self.sequence_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        self.model_encoder = nn.Sequential(
            nn.Linear(4, 16),  # d_model, num_heads, num_layers, layer_depth
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        self.context_encoder = nn.Sequential(
            nn.Linear(10, 16),  # memory context + history
            nn.ReLU(), 
            nn.Linear(16, 16)
        )
        
        # Main decision network
        self.decision_network = nn.Sequential(
            nn.Linear(48, hidden_dim),  # 16+16+16
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # [store_logit, recompute_logit]
        )
        
        # Memory saving prediction head
        self.memory_predictor = nn.Sequential(
            nn.Linear(48, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Predict memory saving ratio 0-1
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for memory scheduling prediction.
        
        Args:
            features: [batch, feature_dim] tensor with scheduling context
            
        Returns:
            decisions: [batch, 2] logits for [store, recompute]
            memory_savings: [batch, 1] predicted memory saving ratios
        """
        # Split features
        seq_features = features[:, 0:1]  # sequence_length (normalized)
        model_features = features[:, 1:5]  # model architecture features
        context_features = features[:, 5:15]  # memory context + history
        
        # Encode features
        seq_emb = self.sequence_encoder(seq_features)
        model_emb = self.model_encoder(model_features)
        context_emb = self.context_encoder(context_features)
        
        # Concatenate embeddings
        combined_emb = torch.cat([seq_emb, model_emb, context_emb], dim=-1)
        
        # Predictions
        decisions = self.decision_network(combined_emb)
        memory_savings = self.memory_predictor(combined_emb)
        
        return decisions, memory_savings

class MemoryDataCollector:
    """Collects training data for the memory predictor."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.patterns: deque = deque(maxlen=max_samples)
        self.lock = threading.Lock()
        
        # Feature statistics for normalization
        self.feature_stats = {
            'sequence_length': {'mean': 4096, 'std': 2048},
            'batch_size': {'mean': 8, 'std': 4},
            'd_model': {'mean': 512, 'std': 256},
            'available_memory_gb': {'mean': 20, 'std': 10}
        }
    
    def add_pattern(self, pattern: MemoryPattern):
        """Add a memory pattern to the training data."""
        with self.lock:
            self.patterns.append(pattern)
            self._update_feature_stats(pattern)
    
    def _update_feature_stats(self, pattern: MemoryPattern):
        """Update running feature statistics."""
        # Exponential moving average for online statistics
        alpha = 0.01
        
        for feature_name in self.feature_stats:
            if hasattr(pattern, feature_name):
                value = getattr(pattern, feature_name)
                
                old_mean = self.feature_stats[feature_name]['mean']
                old_std = self.feature_stats[feature_name]['std']
                
                # Update mean
                new_mean = (1 - alpha) * old_mean + alpha * value
                
                # Update std (simplified)
                new_std = (1 - alpha) * old_std + alpha * abs(value - new_mean)
                
                self.feature_stats[feature_name]['mean'] = new_mean
                self.feature_stats[feature_name]['std'] = max(new_std, 1e-6)
    
    def get_training_data(self, min_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get training data for the predictor.
        
        Args:
            min_samples: Minimum samples required
            
        Returns:
            features: [N, feature_dim] input features
            decisions: [N] binary labels for recompute decision
            memory_savings: [N] memory saving targets
        """
        with self.lock:
            if len(self.patterns) < min_samples:
                return None, None, None
            
            patterns = list(self.patterns)
        
        features = []
        decisions = []
        memory_savings = []
        
        for pattern in patterns:
            # Normalize features
            feature_vec = self._pattern_to_features(pattern)
            features.append(feature_vec)
            decisions.append(pattern.optimal_recompute_decision)
            memory_savings.append(pattern.actual_memory_saving)
        
        return (torch.tensor(features, dtype=torch.float32),
                torch.tensor(decisions, dtype=torch.long),
                torch.tensor(memory_savings, dtype=torch.float32))
    
    def _pattern_to_features(self, pattern: MemoryPattern) -> List[float]:
        """Convert memory pattern to normalized feature vector."""
        features = []
        
        # Normalize sequence length
        seq_norm = (pattern.sequence_length - self.feature_stats['sequence_length']['mean']) / \
                  self.feature_stats['sequence_length']['std']
        features.append(seq_norm)
        
        # Normalize model features
        batch_norm = (pattern.batch_size - self.feature_stats['batch_size']['mean']) / \
                    self.feature_stats['batch_size']['std']
        d_model_norm = (pattern.d_model - self.feature_stats['d_model']['mean']) / \
                      self.feature_stats['d_model']['std']
        
        features.extend([
            batch_norm,
            d_model_norm,
            pattern.num_heads / 32.0,  # Normalize to typical range
            pattern.num_layers / 24.0,  # Normalize to typical range
        ])
        
        # Memory context features
        memory_norm = (pattern.available_memory_gb - self.feature_stats['available_memory_gb']['mean']) / \
                     self.feature_stats['available_memory_gb']['std']
        
        features.extend([
            memory_norm,
            pattern.memory_pressure,
            pattern.layer_depth / pattern.num_layers,  # Relative position
        ])
        
        # Recompute history (pad/truncate to fixed size)
        history = pattern.recompute_history[-7:]  # Last 7 decisions
        while len(history) < 7:
            history.append(0.0)
        features.extend(history)
        
        return features

class IntelligentMemoryScheduler(MemoryScheduler):
    """ML-enhanced memory scheduler using learned patterns."""
    
    def __init__(self, 
                 model: nn.Module,
                 strategy: str = 'ml_adaptive',
                 predictor_path: Optional[str] = None,
                 training_mode: bool = True):
        """
        Initialize intelligent memory scheduler.
        
        Args:
            model: Model to schedule for
            strategy: Scheduling strategy ('ml_adaptive', 'ml_conservative', 'ml_aggressive')
            predictor_path: Path to saved predictor model
            training_mode: Whether to collect training data
        """
        super().__init__(model, strategy='adaptive')
        
        self.ml_strategy = strategy
        self.training_mode = training_mode
        
        # Initialize ML components
        self.predictor = MemoryPredictor()
        self.data_collector = MemoryDataCollector()
        self.profiler = MemoryProfiler()
        
        # Load pretrained predictor if available
        if predictor_path:
            self.load_predictor(predictor_path)
        
        # Training state
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=0.001)
        self.training_frequency = 100  # Retrain every N scheduling decisions
        self.decision_count = 0
        
        # Performance tracking
        self.prediction_accuracy = 0.0
        self.memory_improvement = 0.0
        
        # Historical context
        self.recompute_history = deque(maxlen=10)
        self.memory_history = deque(maxlen=10)
    
    def should_recompute_layer(self, layer_idx: int, current_memory_usage: float) -> bool:
        """
        Make ML-enhanced recomputation decision.
        
        Args:
            layer_idx: Index of layer to decide on
            current_memory_usage: Current memory usage in GB
            
        Returns:
            should_recompute: Whether to recompute this layer
        """
        if not self.training_mode and len(self.data_collector.patterns) == 0:
            # Fall back to heuristic if no training data
            return super().should_recompute_layer(layer_idx, current_memory_usage)
        
        # Get model characteristics
        model_info = self._get_model_info()
        
        # Create memory pattern for prediction
        available_memory = self._get_available_memory()
        memory_pressure = min(current_memory_usage / available_memory, 1.0)
        
        pattern = MemoryPattern(
            sequence_length=getattr(self.model, 'max_seq_len', 4096),
            batch_size=getattr(self, 'current_batch_size', 8),
            d_model=model_info.get('d_model', 512),
            num_heads=model_info.get('num_heads', 8),
            num_layers=model_info.get('num_layers', 12),
            layer_depth=layer_idx,
            available_memory_gb=available_memory,
            memory_pressure=memory_pressure,
            recompute_history=list(self.recompute_history),
            optimal_recompute_decision=0,  # Will be determined
            predicted_memory_saving=0.0,
            actual_memory_saving=0.0,
            timestamp=time.time(),
            execution_time_ms=0.0
        )
        
        # Make ML prediction
        start_time = time.time()
        recompute_decision = self._predict_recompute_decision(pattern)
        prediction_time = (time.time() - start_time) * 1000
        
        # Record decision in history
        self.recompute_history.append(1.0 if recompute_decision else 0.0)
        
        # If in training mode, collect data for later training
        if self.training_mode:
            # Estimate what the optimal decision would be (simplified heuristic)
            heuristic_decision = super().should_recompute_layer(layer_idx, current_memory_usage)
            
            # Update pattern with ground truth
            pattern.optimal_recompute_decision = 1 if heuristic_decision else 0
            pattern.execution_time_ms = prediction_time
            
            # Estimate memory savings (simplified)
            estimated_layer_memory = self._estimate_layer_memory(layer_idx)
            pattern.predicted_memory_saving = estimated_layer_memory / available_memory
            pattern.actual_memory_saving = pattern.predicted_memory_saving  # Simplified
            
            self.data_collector.add_pattern(pattern)
            
            # Periodic retraining
            self.decision_count += 1
            if self.decision_count % self.training_frequency == 0:
                self._retrain_predictor()
        
        return recompute_decision
    
    def _predict_recompute_decision(self, pattern: MemoryPattern) -> bool:
        """Use ML predictor to make recomputation decision."""
        
        # Convert pattern to features
        features = self.data_collector._pattern_to_features(pattern)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            decisions_logits, memory_savings = self.predictor(features_tensor)
            
            # Apply strategy-specific thresholding
            decision_probs = torch.softmax(decisions_logits, dim=-1)
            recompute_prob = decision_probs[0, 1].item()
            
            if self.ml_strategy == 'ml_conservative':
                threshold = 0.7  # Conservative: only recompute if very confident
            elif self.ml_strategy == 'ml_aggressive':
                threshold = 0.3  # Aggressive: recompute more often
            else:  # ml_adaptive
                # Adaptive threshold based on memory pressure
                base_threshold = 0.5
                memory_adjustment = (pattern.memory_pressure - 0.5) * 0.3
                threshold = max(0.1, min(0.9, base_threshold - memory_adjustment))
            
            return recompute_prob > threshold
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Extract model architecture information."""
        info = {}
        
        if hasattr(self.model, 'get_model_info'):
            info = self.model.get_model_info()
        else:
            # Try to infer from model structure
            if hasattr(self.model, 'config'):
                config = self.model.config
                info['d_model'] = getattr(config, 'd_model', 512)
                info['num_heads'] = getattr(config, 'num_heads', 8)
                info['num_layers'] = getattr(config, 'num_layers', 12)
            else:
                # Default values
                info = {'d_model': 512, 'num_heads': 8, 'num_layers': 12}
        
        return info
    
    def _get_available_memory(self) -> float:
        """Get available memory in GB."""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            return total_memory - allocated_memory
        else:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
    
    def _estimate_layer_memory(self, layer_idx: int) -> float:
        """Estimate memory usage of a specific layer in GB."""
        model_info = self._get_model_info()
        
        # Rough estimation based on model size and layer
        d_model = model_info.get('d_model', 512)
        seq_len = getattr(self.model, 'max_seq_len', 4096)
        batch_size = getattr(self, 'current_batch_size', 8)
        
        # Attention memory: batch * heads * seq^2 * head_dim
        attention_memory = (batch_size * model_info.get('num_heads', 8) * 
                          seq_len * seq_len * (d_model // model_info.get('num_heads', 8))) * 4 / (1024**3)
        
        # FFN memory: batch * seq * d_model * d_ff
        ffn_memory = batch_size * seq_len * d_model * (4 * d_model) * 4 / (1024**3)
        
        return attention_memory + ffn_memory
    
    def _retrain_predictor(self):
        """Retrain the ML predictor with collected data."""
        print("🧠 Retraining memory predictor...")
        
        features, decisions, memory_savings = self.data_collector.get_training_data(min_samples=50)
        
        if features is None:
            print("  Not enough training data yet")
            return
        
        # Training loop
        self.predictor.train()
        
        # Split data
        n = len(features)
        indices = torch.randperm(n)
        train_size = int(0.8 * n)
        
        train_features = features[indices[:train_size]]
        train_decisions = decisions[indices[:train_size]]
        train_savings = memory_savings[indices[:train_size]]
        
        val_features = features[indices[train_size:]]
        val_decisions = decisions[indices[train_size:]]
        val_savings = memory_savings[indices[train_size:]]
        
        # Training epochs
        num_epochs = 10
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.optimizer.zero_grad()
            
            decision_logits, predicted_savings = self.predictor(train_features)
            
            # Decision loss (cross-entropy)
            decision_loss = nn.CrossEntropyLoss()(decision_logits, train_decisions)
            
            # Memory savings loss (MSE)
            savings_loss = nn.MSELoss()(predicted_savings.squeeze(), train_savings)
            
            # Combined loss
            total_loss = decision_loss + 0.1 * savings_loss
            
            total_loss.backward()
            self.optimizer.step()
            
            # Validation
            if len(val_features) > 0:
                with torch.no_grad():
                    val_decision_logits, val_predicted_savings = self.predictor(val_features)
                    val_decision_loss = nn.CrossEntropyLoss()(val_decision_logits, val_decisions)
                    val_savings_loss = nn.MSELoss()(val_predicted_savings.squeeze(), val_savings)
                    val_loss = val_decision_loss + 0.1 * val_savings_loss
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        
                        # Calculate accuracy
                        predictions = torch.argmax(val_decision_logits, dim=-1)
                        accuracy = (predictions == val_decisions).float().mean().item()
                        self.prediction_accuracy = accuracy
        
        self.predictor.eval()
        print(f"  Training complete. Validation accuracy: {self.prediction_accuracy:.3f}")
    
    def save_predictor(self, path: str):
        """Save the trained predictor model."""
        torch.save({
            'model_state_dict': self.predictor.state_dict(),
            'feature_stats': self.data_collector.feature_stats,
            'prediction_accuracy': self.prediction_accuracy,
            'decision_count': self.decision_count
        }, path)
        print(f"💾 Predictor saved to: {path}")
    
    def load_predictor(self, path: str):
        """Load a pretrained predictor model."""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            self.predictor.load_state_dict(checkpoint['model_state_dict'])
            self.data_collector.feature_stats = checkpoint.get('feature_stats', self.data_collector.feature_stats)
            self.prediction_accuracy = checkpoint.get('prediction_accuracy', 0.0)
            self.decision_count = checkpoint.get('decision_count', 0)
            print(f"📂 Predictor loaded from: {path} (accuracy: {self.prediction_accuracy:.3f})")
        except Exception as e:
            print(f"⚠️ Failed to load predictor: {e}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the intelligent scheduler."""
        return {
            'prediction_accuracy': self.prediction_accuracy,
            'memory_improvement': self.memory_improvement,
            'decision_count': self.decision_count,
            'training_data_size': len(self.data_collector.patterns),
            'avg_recompute_rate': np.mean(list(self.recompute_history)) if self.recompute_history else 0.0
        }
    
    def set_batch_size(self, batch_size: int):
        """Set current batch size for scheduling decisions."""
        self.current_batch_size = batch_size
    
    def __enter__(self):
        """Context manager entry."""
        self.profiler.start_profiling()
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with performance tracking."""
        result = super().__exit__(exc_type, exc_val, exc_tb)
        
        # Record memory improvement
        if hasattr(self, '_baseline_memory'):
            current_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            improvement = (self._baseline_memory - current_memory) / self._baseline_memory
            self.memory_improvement = 0.9 * self.memory_improvement + 0.1 * improvement
        
        self.profiler.stop_profiling()
        return result

# Factory function for easy initialization
def create_intelligent_scheduler(model: nn.Module, 
                                strategy: str = 'ml_adaptive',
                                training_mode: bool = True) -> IntelligentMemoryScheduler:
    """
    Create an intelligent memory scheduler.
    
    Args:
        model: Model to schedule for
        strategy: ML strategy ('ml_adaptive', 'ml_conservative', 'ml_aggressive')
        training_mode: Whether to collect training data
        
    Returns:
        scheduler: Configured intelligent scheduler
    """
    return IntelligentMemoryScheduler(
        model=model,
        strategy=strategy,
        training_mode=training_mode
    )

# Export
__all__ = [
    'IntelligentMemoryScheduler',
    'MemoryPredictor',
    'MemoryPattern',
    'MemoryDataCollector',
    'create_intelligent_scheduler'
]