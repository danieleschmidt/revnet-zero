"""
Experimental Suite for Comprehensive Model Evaluation

This module provides automated experimental infrastructure for comparing
RevNet-Zero against baseline methods with rigorous scientific methodology.

Features:
- Automated experiment orchestration
- Memory usage profiling during training
- Performance benchmarking across multiple metrics
- Statistical validation with publication standards
- Reproducible experiment protocols
- Real-time monitoring and adaptive stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from collections import defaultdict

from .comprehensive_baselines import ComprehensiveBaselines
from .statistical_framework import StatisticalFramework, ExperimentResult
from ..layers.quantum_coupling import create_quantum_coupling
from ..memory.wavelet_scheduler import WaveletMemoryScheduler
from ..models.reversible_transformer import ReversibleTransformer


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    model_name: str
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_metrics: List[str]
    num_runs: int = 5
    max_training_steps: int = 1000
    eval_frequency: int = 100
    early_stopping_patience: int = 5
    memory_budget_gb: float = 8.0
    
    def __post_init__(self):
        # Ensure minimum number of runs for statistical validity
        if self.num_runs < 3:
            logging.warning(f"num_runs={self.num_runs} may be too low for statistical significance")


@dataclass 
class PerformanceMetrics:
    """Container for performance measurements."""
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    training_time_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    energy_consumption_joules: Optional[float] = None
    convergence_steps: Optional[int] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MemoryProfiler:
    """Real-time memory usage profiler."""
    
    def __init__(self):
        self.measurements = []
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record_measurement()
    
    def record_measurement(self):
        """Record current memory usage."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            gpu_memory = 0
        
        cpu_memory = psutil.virtual_memory().used / (1024**2)  # MB
        
        self.measurements.append({
            'timestamp': time.time() - (self.start_time or time.time()),
            'gpu_memory_mb': gpu_memory,
            'cpu_memory_mb': cpu_memory
        })
    
    def get_peak_memory(self) -> float:
        """Get peak GPU memory usage in MB."""
        if not self.measurements:
            return 0.0
        return max(m['gpu_memory_mb'] for m in self.measurements)
    
    def get_average_memory(self) -> float:
        """Get average GPU memory usage in MB."""
        if not self.measurements:
            return 0.0
        return np.mean([m['gpu_memory_mb'] for m in self.measurements])


class TrainingEnvironment:
    """Controlled training environment for fair comparison."""
    
    def __init__(
        self,
        device: torch.device = None,
        memory_budget_gb: float = 8.0,
        enable_amp: bool = True,
        enable_profiling: bool = True
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_budget_gb = memory_budget_gb
        self.enable_amp = enable_amp
        self.enable_profiling = enable_profiling
        
        # Training state
        self.current_model = None
        self.current_optimizer = None
        self.scaler = torch.cuda.amp.GradScaler() if enable_amp else None
        
        # Profiling
        self.memory_profiler = MemoryProfiler() if enable_profiling else None
        
    def setup_model(self, model: nn.Module, learning_rate: float = 1e-4):
        """Setup model and optimizer."""
        self.current_model = model.to(self.device)
        self.current_optimizer = optim.AdamW(
            self.current_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Perform one training step."""
        self.current_model.train()
        self.current_optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        start_time = time.time()
        
        if self.enable_amp:
            with torch.cuda.amp.autocast():
                outputs = self.current_model(batch['input_ids'])
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), 
                               batch['labels'].view(-1))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.current_optimizer)
            self.scaler.update()
        else:
            outputs = self.current_model(batch['input_ids'])
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), 
                           batch['labels'].view(-1))
            loss.backward()
            self.current_optimizer.step()
        
        training_time = time.time() - start_time
        
        # Record memory if profiling enabled
        if self.memory_profiler:
            self.memory_profiler.record_measurement()
        
        return {
            'loss': loss.item(),
            'training_time': training_time
        }
    
    def evaluate_model(
        self,
        eval_dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        max_eval_batches: int = 50
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        self.current_model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= max_eval_batches:
                    break
                
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.enable_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.current_model(batch['input_ids'])
                        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                       batch['labels'].view(-1))
                else:
                    outputs = self.current_model(batch['input_ids'])
                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                   batch['labels'].view(-1))
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                mask = batch['labels'] != -100  # Ignore padding tokens
                correct_predictions += (predictions == batch['labels'])[mask].sum().item()
                total_tokens += mask.sum().item()
        
        eval_time = time.time() - start_time
        
        metrics = {
            'eval_loss': total_loss / min(max_eval_batches, len(eval_dataloader)),
            'accuracy': correct_predictions / max(total_tokens, 1),
            'perplexity': np.exp(total_loss / min(max_eval_batches, len(eval_dataloader))),
            'eval_time': eval_time
        }
        
        return metrics
    
    def get_memory_metrics(self) -> Dict[str, float]:
        """Get memory usage metrics."""
        metrics = {}
        
        if torch.cuda.is_available():
            metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            metrics['gpu_memory_peak_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
        
        if self.memory_profiler and self.memory_profiler.measurements:
            metrics['peak_memory_mb'] = self.memory_profiler.get_peak_memory()
            metrics['avg_memory_mb'] = self.memory_profiler.get_average_memory()
        
        return metrics


class ExperimentalSuite:
    """
    Main experimental suite for comprehensive model comparison.
    """
    
    def __init__(
        self,
        output_dir: str = "./experiment_results",
        statistical_framework: Optional[StatisticalFramework] = None,
        device: torch.device = None,
        random_seed: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.statistical_framework = statistical_framework or StatisticalFramework(
            output_dir=str(self.output_dir / "statistics")
        )
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = random_seed
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize experiment tracking
        self.experiment_configs: List[ExperimentConfig] = []
        self.completed_experiments: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_experiment_config(self, config: ExperimentConfig):
        """Add experiment configuration to the suite."""
        self.experiment_configs.append(config)
        self.logger.info(f"Added experiment config: {config.model_name}")
    
    def create_revnet_experiment(
        self,
        model_config: Dict[str, Any],
        coupling_type: str = 'quantum_rotation',
        use_wavelet_scheduler: bool = True,
        **kwargs
    ) -> ExperimentConfig:
        """Create experiment config for RevNet-Zero with novel features."""
        config = ExperimentConfig(
            model_name=f"revnet_zero_{coupling_type}{'_wavelet' if use_wavelet_scheduler else ''}",
            model_config={
                'model_type': 'revnet_zero',
                'coupling_type': coupling_type,
                'use_wavelet_scheduler': use_wavelet_scheduler,
                **model_config
            },
            **kwargs
        )
        return config
    
    def create_baseline_experiment(
        self,
        baseline_name: str,
        model_config: Dict[str, Any],
        **kwargs
    ) -> ExperimentConfig:
        """Create experiment config for a baseline method."""
        config = ExperimentConfig(
            model_name=f"baseline_{baseline_name}",
            model_config={
                'model_type': 'baseline',
                'baseline_name': baseline_name,
                **model_config
            },
            **kwargs
        )
        return config
    
    def build_model(self, config: ExperimentConfig) -> nn.Module:
        """Build model from experiment configuration."""
        model_config = config.model_config
        
        if model_config['model_type'] == 'revnet_zero':
            # Create RevNet-Zero model with novel features
            model = ReversibleTransformer(
                num_layers=model_config.get('num_layers', 12),
                d_model=model_config.get('d_model', 512),
                num_heads=model_config.get('num_heads', 8),
                max_seq_len=model_config.get('max_seq_len', 4096)
            )
            
            # Replace coupling layers with quantum coupling if specified
            if 'coupling_type' in model_config and model_config['coupling_type'].startswith('quantum'):
                coupling_type = model_config['coupling_type'].replace('quantum_', '')
                for layer in model.layers:
                    if hasattr(layer, 'coupling_fn'):
                        layer.coupling_fn = create_quantum_coupling(
                            coupling_type=coupling_type,
                            dim=model_config.get('d_model', 512) // 2
                        )
            
            return model
            
        elif model_config['model_type'] == 'baseline':
            # Create baseline model
            baseline_suite = ComprehensiveBaselines(
                d_model=model_config.get('d_model', 512),
                num_heads=model_config.get('num_heads', 8),
                max_seq_length=model_config.get('max_seq_len', 4096)
            )
            return baseline_suite.get_baseline(model_config['baseline_name'])
        
        else:
            raise ValueError(f"Unknown model type: {model_config['model_type']}")
    
    def create_dummy_dataloader(
        self,
        batch_size: int = 8,
        seq_length: int = 1024,
        vocab_size: int = 50257,
        num_batches: int = 100
    ) -> torch.utils.data.DataLoader:
        """Create dummy dataloader for benchmarking."""
        
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples, seq_length, vocab_size):
                self.num_samples = num_samples
                self.seq_length = seq_length
                self.vocab_size = vocab_size
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
                labels = input_ids.clone()
                return {'input_ids': input_ids, 'labels': labels}
        
        dataset = DummyDataset(num_batches * batch_size, seq_length, vocab_size)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def run_single_experiment(
        self,
        config: ExperimentConfig,
        run_id: int
    ) -> PerformanceMetrics:
        """Run a single experiment instance."""
        self.logger.info(f"Starting experiment: {config.model_name}, run {run_id}")
        
        # Set random seed for this run
        torch.manual_seed(self.random_seed + run_id)
        np.random.seed(self.random_seed + run_id)
        
        # Create model
        model = self.build_model(config)
        
        # Create training environment
        training_env = TrainingEnvironment(
            device=self.device,
            memory_budget_gb=config.memory_budget_gb,
            enable_amp=True,
            enable_profiling=True
        )
        
        # Setup model and optimizer
        training_env.setup_model(
            model, 
            learning_rate=config.training_config.get('learning_rate', 1e-4)
        )
        
        # Create dataloaders
        train_dataloader = self.create_dummy_dataloader(
            batch_size=config.training_config.get('batch_size', 8),
            seq_length=config.dataset_config.get('seq_length', 1024),
            num_batches=config.max_training_steps
        )
        
        eval_dataloader = self.create_dummy_dataloader(
            batch_size=config.training_config.get('batch_size', 8),
            seq_length=config.dataset_config.get('seq_length', 1024),
            num_batches=50  # Smaller eval set
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training loop with measurements
        metrics = PerformanceMetrics()
        
        start_time = time.time()
        best_eval_loss = float('inf')
        patience_counter = 0
        convergence_step = None
        
        with training_env.memory_profiler:
            for step, batch in enumerate(train_dataloader):
                if step >= config.max_training_steps:
                    break
                
                # Training step
                step_metrics = training_env.training_step(batch, criterion)
                
                # Evaluation
                if step % config.eval_frequency == 0:
                    eval_metrics = training_env.evaluate_model(eval_dataloader, criterion)
                    
                    self.logger.info(
                        f"Step {step}: Loss={step_metrics['loss']:.4f}, "
                        f"Eval Loss={eval_metrics['eval_loss']:.4f}, "
                        f"Accuracy={eval_metrics['accuracy']:.4f}"
                    )
                    
                    # Early stopping check
                    if eval_metrics['eval_loss'] < best_eval_loss:
                        best_eval_loss = eval_metrics['eval_loss']
                        patience_counter = 0
                        if convergence_step is None:
                            convergence_step = step
                    else:
                        patience_counter += 1
                        if patience_counter >= config.early_stopping_patience:
                            self.logger.info(f"Early stopping at step {step}")
                            break
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_eval_metrics = training_env.evaluate_model(eval_dataloader, criterion)
        memory_metrics = training_env.get_memory_metrics()
        
        # Compile performance metrics
        metrics.accuracy = final_eval_metrics['accuracy']
        metrics.perplexity = final_eval_metrics['perplexity']
        metrics.memory_usage_mb = memory_metrics.get('peak_memory_mb', 0)
        metrics.training_time_seconds = training_time
        metrics.inference_time_ms = final_eval_metrics['eval_time'] * 1000  # Convert to ms
        metrics.convergence_steps = convergence_step
        
        # Clean up
        del model, training_env, train_dataloader, eval_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        self.logger.info(f"Completed experiment: {config.model_name}, run {run_id}")
        return metrics
    
    def run_experiment_config(self, config: ExperimentConfig) -> List[PerformanceMetrics]:
        """Run all repetitions of an experiment configuration."""
        results = []
        
        for run_id in range(config.num_runs):
            try:
                metrics = self.run_single_experiment(config, run_id)
                results.append(metrics)
                
                # Add to statistical framework
                for metric_name, value in metrics.to_dict().items():
                    if metric_name in config.evaluation_metrics:
                        self.statistical_framework.add_experiment_result(
                            method_name=config.model_name,
                            metric_name=metric_name,
                            values=[value],
                            run_id=run_id,
                            config=config.model_config
                        )
                        
            except Exception as e:
                self.logger.error(f"Experiment failed: {config.model_name}, run {run_id}: {e}")
                continue
        
        self.completed_experiments[config.model_name] = results
        return results
    
    def run_all_experiments(self) -> Dict[str, List[PerformanceMetrics]]:
        """Run all configured experiments."""
        self.logger.info(f"Starting experimental suite with {len(self.experiment_configs)} configurations")
        
        for config in self.experiment_configs:
            self.logger.info(f"Running experiment configuration: {config.model_name}")
            self.run_experiment_config(config)
        
        self.logger.info("All experiments completed")
        return self.completed_experiments
    
    def perform_statistical_analysis(
        self,
        metrics_to_analyze: List[str] = None
    ) -> pd.DataFrame:
        """Perform comprehensive statistical analysis of results."""
        if metrics_to_analyze is None:
            # Use all available metrics
            all_metrics = set()
            for results in self.completed_experiments.values():
                for result in results:
                    all_metrics.update(result.to_dict().keys())
            metrics_to_analyze = list(all_metrics)
        
        methods = list(self.completed_experiments.keys())
        
        # Perform comprehensive comparison
        comparison_df = self.statistical_framework.comprehensive_comparison(
            methods=methods,
            metrics=metrics_to_analyze,
            correction_method='benjamini_hochberg'  # More appropriate for multiple comparisons
        )
        
        return comparison_df
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive experimental report."""
        # Perform statistical analysis
        statistical_results = self.perform_statistical_analysis()
        
        # Create summary statistics
        summary_stats = {}
        for method_name, results in self.completed_experiments.items():
            if not results:
                continue
                
            method_stats = {}
            all_metrics = {}
            
            for result in results:
                for metric_name, value in result.to_dict().items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
            
            for metric_name, values in all_metrics.items():
                method_stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n': len(values)
                }
            
            summary_stats[method_name] = method_stats
        
        # Compile final report
        report = {
            'experiment_summary': {
                'total_experiments': len(self.experiment_configs),
                'total_completed_runs': sum(len(results) for results in self.completed_experiments.values()),
                'methods_compared': list(self.completed_experiments.keys()),
                'random_seed': self.random_seed
            },
            'summary_statistics': summary_stats,
            'statistical_tests': statistical_results.to_dict('records') if not statistical_results.empty else [],
            'significant_findings': self._extract_significant_findings(statistical_results),
            'performance_rankings': self._create_performance_rankings(summary_stats)
        }
        
        # Save report
        report_path = self.output_dir / 'final_experiment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Final report saved to {report_path}")
        return report
    
    def _extract_significant_findings(self, statistical_results: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract statistically significant findings."""
        if statistical_results.empty:
            return []
        
        significant_results = statistical_results[statistical_results.get('is_significant', False)]
        
        findings = []
        for _, row in significant_results.iterrows():
            finding = {
                'comparison': f"{row['method1']} vs {row['method2']}",
                'metric': row['metric'],
                'p_value': row.get('corrected_p_value', row.get('p_value')),
                'effect_size': row.get('effect_size'),
                'significance_level': row.get('significance_level'),
                'interpretation': row.get('interpretation', 'Significant difference found')
            }
            findings.append(finding)
        
        return findings
    
    def _create_performance_rankings(self, summary_stats: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Create performance rankings for each metric."""
        rankings = {}
        
        # Get all metrics
        all_metrics = set()
        for method_stats in summary_stats.values():
            all_metrics.update(method_stats.keys())
        
        for metric in all_metrics:
            metric_values = {}
            for method, stats in summary_stats.items():
                if metric in stats:
                    metric_values[method] = stats[metric]['mean']
            
            # Sort methods by performance (lower is better for loss/perplexity, higher for accuracy)
            reverse_sort = metric in ['accuracy']  # Metrics where higher is better
            sorted_methods = sorted(metric_values.items(), key=lambda x: x[1], reverse=reverse_sort)
            
            rankings[metric] = {method: rank + 1 for rank, (method, _) in enumerate(sorted_methods)}
        
        return rankings


# Convenience functions for common experiment setups

def create_memory_efficiency_experiment(
    seq_lengths: List[int] = [1024, 2048, 4096, 8192],
    model_sizes: List[str] = ['small', 'medium'],
    num_runs: int = 5
) -> List[ExperimentConfig]:
    """Create experiment configs focused on memory efficiency comparison."""
    configs = []
    
    model_configs = {
        'small': {'d_model': 256, 'num_heads': 8, 'num_layers': 6},
        'medium': {'d_model': 512, 'num_heads': 8, 'num_layers': 12}
    }
    
    for size in model_sizes:
        for seq_len in seq_lengths:
            base_config = {
                **model_configs[size],
                'max_seq_len': seq_len
            }
            
            # RevNet-Zero with quantum coupling
            configs.append(ExperimentConfig(
                model_name=f"revnet_quantum_{size}_seq{seq_len}",
                model_config={
                    'model_type': 'revnet_zero',
                    'coupling_type': 'quantum_rotation',
                    'use_wavelet_scheduler': True,
                    **base_config
                },
                dataset_config={'seq_length': seq_len},
                training_config={'batch_size': max(1, 64 // (seq_len // 512)), 'learning_rate': 1e-4},
                evaluation_metrics=['memory_usage_mb', 'training_time_seconds', 'accuracy'],
                num_runs=num_runs,
                max_training_steps=500
            ))
            
            # Baseline comparisons
            for baseline in ['longformer', 'performer', 'flash_attention']:
                configs.append(ExperimentConfig(
                    model_name=f"{baseline}_{size}_seq{seq_len}",
                    model_config={
                        'model_type': 'baseline',
                        'baseline_name': baseline,
                        **base_config
                    },
                    dataset_config={'seq_length': seq_len},
                    training_config={'batch_size': max(1, 64 // (seq_len // 512)), 'learning_rate': 1e-4},
                    evaluation_metrics=['memory_usage_mb', 'training_time_seconds', 'accuracy'],
                    num_runs=num_runs,
                    max_training_steps=500
                ))
    
    return configs


__all__ = [
    'ExperimentConfig',
    'PerformanceMetrics', 
    'MemoryProfiler',
    'TrainingEnvironment',
    'ExperimentalSuite',
    'create_memory_efficiency_experiment'
]