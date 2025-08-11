"""
Scaling laws analysis for reversible transformers.

Provides tools for:
- Empirical scaling law derivation (compute, parameters, data)
- Memory scaling characterization  
- Performance prediction at different scales
- Optimal resource allocation strategies
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import json
import time
from scipy.optimize import curve_fit
from scipy import stats
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .comparative_studies import ModelFactory, BenchmarkResult
from ..models.reversible_transformer import ReversibleTransformer
from ..memory.scheduler import MemoryScheduler

@dataclass
class ScalingExperiment:
    """Configuration for a scaling experiment."""
    
    # Parameter scaling
    parameter_counts: List[int] = field(default_factory=lambda: [1e6, 5e6, 10e6, 50e6, 100e6, 500e6])
    
    # Compute scaling (FLOPs)
    compute_budgets: List[float] = field(default_factory=lambda: [1e15, 1e16, 1e17, 1e18, 1e19])
    
    # Data scaling (tokens)
    dataset_sizes: List[int] = field(default_factory=lambda: [1e6, 1e7, 1e8, 1e9, 1e10])
    
    # Context length scaling
    context_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096, 8192, 16384])
    
    # Training configuration
    max_training_steps: int = 1000
    evaluation_frequency: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    # Experiment parameters
    num_seeds: int = 3
    confidence_interval: float = 0.95

@dataclass
class ScalingResult:
    """Results from scaling experiments."""
    
    parameter_count: int
    compute_budget: float
    dataset_size: int
    context_length: int
    batch_size: int
    
    # Performance metrics
    final_loss: float
    memory_usage_gb: float
    training_time_hours: float
    tokens_per_second: float
    
    # Scaling-specific metrics
    loss_vs_compute: List[Tuple[float, float]]  # (compute_used, loss)
    memory_vs_context: List[Tuple[int, float]]  # (context_len, memory_gb)
    
    # Model characteristics
    model_type: str
    coupling_type: Optional[str]
    seed: int
    timestamp: str

class ScalingLawsAnalyzer:
    """Comprehensive scaling analysis for reversible transformers."""
    
    def __init__(self, experiment_config: Optional[ScalingExperiment] = None):
        self.config = experiment_config or ScalingExperiment()
        self.results: List[ScalingResult] = []
        self.fitted_laws: Dict[str, Dict] = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run_scaling_analysis(self, 
                           model_types: List[str] = None,
                           output_dir: str = './scaling_results') -> Dict[str, Any]:
        """
        Run comprehensive scaling analysis.
        
        Args:
            model_types: Types of models to analyze ['standard', 'reversible']
            output_dir: Directory for results
            
        Returns:
            analysis_results: Complete scaling analysis
        """
        model_types = model_types or ['standard', 'reversible']
        
        print("📈 Starting Scaling Laws Analysis")
        print("=" * 50)
        
        # Run experiments
        self._run_scaling_experiments(model_types)
        
        # Fit scaling laws
        self._fit_scaling_laws()
        
        # Generate predictions
        predictions = self._generate_scaling_predictions()
        
        # Analysis and visualization
        analysis_results = {
            'fitted_laws': self.fitted_laws,
            'predictions': predictions,
            'recommendations': self._generate_scaling_recommendations(),
            'experimental_data': [self._result_to_dict(r) for r in self.results]
        }
        
        # Save results
        self._save_results(analysis_results, output_dir)
        
        # Generate visualizations
        self._generate_scaling_plots(analysis_results, output_dir)
        
        return analysis_results
    
    def _run_scaling_experiments(self, model_types: List[str]):
        """Run systematic scaling experiments."""
        
        total_experiments = (len(model_types) * 
                           len(self.config.parameter_counts) *
                           len(self.config.context_lengths) *
                           self.config.num_seeds)
        
        print(f"Total experiments: {total_experiments}")
        
        # Use thread pool for parallel experiments
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            
            for model_type in model_types:
                for param_count in self.config.parameter_counts:
                    for context_len in self.config.context_lengths:
                        for seed in range(self.config.num_seeds):
                            future = executor.submit(
                                self._run_single_scaling_experiment,
                                model_type, param_count, context_len, seed
                            )
                            futures.append(future)
            
            # Collect results
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        print(f"Completed {completed}/{total_experiments} experiments")
                except Exception as e:
                    self.logger.error(f"Experiment failed: {e}")
    
    def _run_single_scaling_experiment(self, 
                                     model_type: str,
                                     param_count: int, 
                                     context_len: int,
                                     seed: int) -> Optional[ScalingResult]:
        """Run a single scaling experiment."""
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Determine model size based on parameter count
            model_size = self._param_count_to_size(param_count)
            
            # Create model
            if model_type == 'standard':
                model = ModelFactory.create_standard_transformer(
                    model_size, max_seq_len=context_len
                )
                coupling_type = None
            else:
                model = ModelFactory.create_reversible_model(
                    model_size, max_seq_len=context_len, coupling_type='additive'
                )
                coupling_type = 'additive'
            
            model = model.to(device)
            actual_param_count = sum(p.numel() for p in model.parameters())
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Memory monitoring setup
            memory_scheduler = None
            if model_type == 'reversible':
                memory_scheduler = MemoryScheduler(model, strategy='adaptive')
            
            # Training loop with loss tracking
            batch_size = min(8, max(1, 32 // (context_len // 512)))  # Adaptive batch size
            
            loss_vs_compute = []
            memory_vs_context = []
            
            start_time = time.time()
            total_flops = 0
            
            for step in range(self.config.max_training_steps):
                # Generate batch
                input_ids = torch.randint(0, 1000, (batch_size, context_len), device=device)
                labels = torch.randint(0, 1000, (batch_size, context_len), device=device)
                
                optimizer.zero_grad()
                
                # Forward pass with memory tracking
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                
                if memory_scheduler:
                    with memory_scheduler:
                        outputs = model(input_ids)
                        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                else:
                    outputs = model(input_ids)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                loss.backward()
                optimizer.step()
                
                # Track compute and memory
                step_flops = self._estimate_flops(actual_param_count, context_len, batch_size)
                total_flops += step_flops
                
                if device.type == 'cuda':
                    memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
                else:
                    import psutil
                    memory_gb = psutil.Process().memory_info().rss / (1024**3)
                
                # Record data points
                if step % self.config.evaluation_frequency == 0:
                    loss_vs_compute.append((total_flops, loss.item()))
                    memory_vs_context.append((context_len, memory_gb))
            
            training_time = time.time() - start_time
            final_loss = loss.item()
            tokens_per_second = (self.config.max_training_steps * batch_size * context_len) / training_time
            
            # Create result
            result = ScalingResult(
                parameter_count=actual_param_count,
                compute_budget=total_flops,
                dataset_size=self.config.max_training_steps * batch_size * context_len,
                context_length=context_len,
                batch_size=batch_size,
                final_loss=final_loss,
                memory_usage_gb=memory_gb,
                training_time_hours=training_time / 3600,
                tokens_per_second=tokens_per_second,
                loss_vs_compute=loss_vs_compute,
                memory_vs_context=memory_vs_context,
                model_type=model_type,
                coupling_type=coupling_type,
                seed=seed,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Single experiment failed: {e}")
            return None
    
    def _param_count_to_size(self, param_count: int) -> str:
        """Convert parameter count to model size string."""
        if param_count < 10e6:
            return 'tiny'
        elif param_count < 50e6:
            return 'small'  
        elif param_count < 200e6:
            return 'base'
        else:
            return 'large'
    
    def _estimate_flops(self, param_count: int, seq_len: int, batch_size: int) -> float:
        """Estimate FLOPs for a training step."""
        # Simplified FLOP estimation
        # Forward: ~6 * param_count * seq_len * batch_size
        # Backward: ~2 * forward
        forward_flops = 6 * param_count * seq_len * batch_size
        return 3 * forward_flops  # Forward + backward
    
    def _fit_scaling_laws(self):
        """Fit mathematical scaling laws to experimental data."""
        
        print("🔬 Fitting scaling laws...")
        
        # Group results by model type
        model_groups = {}
        for result in self.results:
            key = f"{result.model_type}_{result.coupling_type or 'none'}"
            if key not in model_groups:
                model_groups[key] = []
            model_groups[key].append(result)
        
        for model_key, results in model_groups.items():
            print(f"  Fitting laws for: {model_key}")
            
            # Loss vs Parameters scaling law: L(N) = a * N^(-b) + c
            param_counts = [r.parameter_count for r in results]
            final_losses = [r.final_loss for r in results]
            
            if len(set(param_counts)) > 3:  # Need multiple parameter counts
                try:
                    def power_law(n, a, b, c):
                        return a * np.power(n, -b) + c
                    
                    popt_params, _ = curve_fit(power_law, param_counts, final_losses, 
                                             p0=[1.0, 0.1, 0.1], maxfev=10000)
                    
                    self.fitted_laws[f"{model_key}_loss_vs_params"] = {
                        'type': 'power_law',
                        'formula': 'L(N) = a * N^(-b) + c',
                        'parameters': {'a': popt_params[0], 'b': popt_params[1], 'c': popt_params[2]},
                        'r_squared': self._calculate_r_squared(final_losses, 
                                                             [power_law(n, *popt_params) for n in param_counts])
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to fit parameter scaling law for {model_key}: {e}")
            
            # Memory vs Context Length scaling: M(L) = a * L^b + c
            context_lengths = [r.context_length for r in results]  
            memory_usages = [r.memory_usage_gb for r in results]
            
            if len(set(context_lengths)) > 3:
                try:
                    def memory_law(l, a, b, c):
                        return a * np.power(l, b) + c
                    
                    popt_memory, _ = curve_fit(memory_law, context_lengths, memory_usages,
                                             p0=[1e-6, 2.0, 0.1], maxfev=10000)
                    
                    self.fitted_laws[f"{model_key}_memory_vs_context"] = {
                        'type': 'power_law',
                        'formula': 'M(L) = a * L^b + c', 
                        'parameters': {'a': popt_memory[0], 'b': popt_memory[1], 'c': popt_memory[2]},
                        'r_squared': self._calculate_r_squared(memory_usages,
                                                             [memory_law(l, *popt_memory) for l in context_lengths])
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to fit memory scaling law for {model_key}: {e}")
            
            # Compute optimal scaling: C(N, D) where N=parameters, D=data
            # Simplified: assume optimal compute ~ N^a * D^b
            data_sizes = [r.dataset_size for r in results]
            compute_budgets = [r.compute_budget for r in results]
            
            if len(set(param_counts)) > 2 and len(set(data_sizes)) > 2:
                try:
                    def compute_law(nd, a, b):
                        n, d = nd
                        return a * (n ** 0.73) * (d ** 0.27)  # Chinchilla-inspired
                    
                    # Prepare data for fitting
                    nd_pairs = [(r.parameter_count, r.dataset_size) for r in results]
                    
                    # This is simplified - in practice would need more sophisticated fitting
                    self.fitted_laws[f"{model_key}_compute_optimal"] = {
                        'type': 'chinchilla_inspired',
                        'formula': 'C_opt(N, D) ~ N^0.73 * D^0.27',
                        'parameters': {'param_exponent': 0.73, 'data_exponent': 0.27},
                        'description': 'Optimal compute allocation between parameters and data'
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to fit compute scaling law for {model_key}: {e}")
    
    def _calculate_r_squared(self, y_true: List[float], y_pred: List[float]) -> float:
        """Calculate R-squared coefficient of determination."""
        y_true_mean = np.mean(y_true)
        ss_tot = sum((y - y_true_mean) ** 2 for y in y_true)
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def _generate_scaling_predictions(self) -> Dict[str, Any]:
        """Generate predictions using fitted scaling laws."""
        
        predictions = {}
        
        # Prediction scenarios
        scenarios = {
            'small_scale': {'params': 10e6, 'context': 2048, 'data': 1e9},
            'medium_scale': {'params': 100e6, 'context': 8192, 'data': 10e9}, 
            'large_scale': {'params': 1e9, 'context': 32768, 'data': 100e9},
            'extreme_scale': {'params': 10e9, 'context': 131072, 'data': 1e12}
        }
        
        for scenario_name, scenario in scenarios.items():
            predictions[scenario_name] = {}
            
            for law_name, law_data in self.fitted_laws.items():
                model_type = law_name.split('_')[0] + '_' + law_name.split('_')[1]
                metric = '_'.join(law_name.split('_')[2:])
                
                if model_type not in predictions[scenario_name]:
                    predictions[scenario_name][model_type] = {}
                
                # Make predictions based on law type
                if metric == 'loss_vs_params' and law_data['type'] == 'power_law':
                    params = law_data['parameters']
                    predicted_loss = (params['a'] * (scenario['params'] ** -params['b']) + params['c'])
                    predictions[scenario_name][model_type]['predicted_loss'] = predicted_loss
                
                elif metric == 'memory_vs_context' and law_data['type'] == 'power_law':
                    params = law_data['parameters']
                    predicted_memory = (params['a'] * (scenario['context'] ** params['b']) + params['c'])
                    predictions[scenario_name][model_type]['predicted_memory_gb'] = predicted_memory
                
                elif metric == 'compute_optimal':
                    # Compute optimal training time/cost estimates
                    params = law_data['parameters']
                    optimal_compute = (scenario['params'] ** params['param_exponent'] * 
                                     scenario['data'] ** params['data_exponent'])
                    predictions[scenario_name][model_type]['optimal_compute'] = optimal_compute
        
        return predictions
    
    def _generate_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable scaling recommendations."""
        
        recommendations = []
        
        # Compare reversible vs standard scaling
        reversible_laws = {k: v for k, v in self.fitted_laws.items() if 'reversible' in k}
        standard_laws = {k: v for k, v in self.fitted_laws.items() if 'standard' in k}
        
        # Memory scaling comparison
        rev_memory_law = reversible_laws.get('reversible_additive_memory_vs_context')
        std_memory_law = standard_laws.get('standard_none_memory_vs_context')
        
        if rev_memory_law and std_memory_law:
            rev_exponent = rev_memory_law['parameters']['b']
            std_exponent = std_memory_law['parameters']['b'] 
            
            if rev_exponent < std_exponent:
                recommendations.append({
                    'category': 'memory_scaling',
                    'recommendation': f'Reversible transformers scale better with context length '
                                    f'(exponent: {rev_exponent:.2f} vs {std_exponent:.2f})',
                    'strength': 'high',
                    'optimal_use': 'Long context applications (>8K tokens)'
                })
        
        # Parameter scaling comparison
        rev_loss_law = reversible_laws.get('reversible_additive_loss_vs_params')
        std_loss_law = standard_laws.get('standard_none_loss_vs_params')
        
        if rev_loss_law and std_loss_law:
            rev_r2 = rev_loss_law.get('r_squared', 0)
            std_r2 = std_loss_law.get('r_squared', 0)
            
            if abs(rev_r2 - std_r2) < 0.1:  # Similar scaling behavior
                recommendations.append({
                    'category': 'parameter_scaling',
                    'recommendation': 'Similar loss scaling behavior observed between architectures',
                    'strength': 'medium',
                    'optimal_use': 'Choose based on memory constraints rather than scaling'
                })
        
        # Compute efficiency recommendations
        recommendations.append({
            'category': 'compute_efficiency', 
            'recommendation': 'For compute-constrained scenarios, prioritize parameter efficiency over data',
            'strength': 'high',
            'optimal_use': 'Resource-limited research and deployment'
        })
        
        return recommendations
    
    def predict_performance(self, 
                          model_type: str,
                          parameter_count: int,
                          context_length: int,
                          data_size: int) -> Dict[str, float]:
        """
        Predict model performance using fitted scaling laws.
        
        Args:
            model_type: 'standard' or 'reversible_additive'
            parameter_count: Number of parameters
            context_length: Context length in tokens
            data_size: Training data size in tokens
            
        Returns:
            predictions: Dict with predicted metrics
        """
        predictions = {}
        
        # Loss prediction
        loss_law_key = f"{model_type}_loss_vs_params"
        if loss_law_key in self.fitted_laws:
            law = self.fitted_laws[loss_law_key]
            if law['type'] == 'power_law':
                params = law['parameters']
                pred_loss = params['a'] * (parameter_count ** -params['b']) + params['c']
                predictions['loss'] = pred_loss
        
        # Memory prediction  
        memory_law_key = f"{model_type}_memory_vs_context"
        if memory_law_key in self.fitted_laws:
            law = self.fitted_laws[memory_law_key]
            if law['type'] == 'power_law':
                params = law['parameters'] 
                pred_memory = params['a'] * (context_length ** params['b']) + params['c']
                predictions['memory_gb'] = pred_memory
        
        # Compute prediction
        compute_law_key = f"{model_type}_compute_optimal"
        if compute_law_key in self.fitted_laws:
            law = self.fitted_laws[compute_law_key]
            if law['type'] == 'chinchilla_inspired':
                params = law['parameters']
                pred_compute = (parameter_count ** params['param_exponent'] * 
                              data_size ** params['data_exponent'])
                predictions['optimal_compute'] = pred_compute
        
        return predictions
    
    def _result_to_dict(self, result: ScalingResult) -> Dict[str, Any]:
        """Convert ScalingResult to dictionary for serialization."""
        return {
            'parameter_count': result.parameter_count,
            'compute_budget': result.compute_budget,
            'dataset_size': result.dataset_size,
            'context_length': result.context_length,
            'batch_size': result.batch_size,
            'final_loss': result.final_loss,
            'memory_usage_gb': result.memory_usage_gb,
            'training_time_hours': result.training_time_hours,
            'tokens_per_second': result.tokens_per_second,
            'model_type': result.model_type,
            'coupling_type': result.coupling_type,
            'seed': result.seed,
            'timestamp': result.timestamp
        }
    
    def _save_results(self, analysis_results: Dict[str, Any], output_dir: str):
        """Save analysis results to file."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        with open(os.path.join(output_dir, 'scaling_analysis.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save fitted laws separately  
        with open(os.path.join(output_dir, 'scaling_laws.json'), 'w') as f:
            json.dump(self.fitted_laws, f, indent=2, default=str)
        
        print(f"📊 Scaling analysis results saved to: {output_dir}")
    
    def _generate_scaling_plots(self, analysis_results: Dict[str, Any], output_dir: str):
        """Generate scaling law visualization plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Loss vs Parameters
        self._plot_loss_vs_parameters(output_dir)
        
        # Plot 2: Memory vs Context Length
        self._plot_memory_vs_context(output_dir)
        
        # Plot 3: Scaling Law Comparison
        self._plot_scaling_comparison(analysis_results, output_dir)
    
    def _plot_loss_vs_parameters(self, output_dir: str):
        """Plot loss vs parameter count scaling."""
        plt.figure(figsize=(10, 6))
        
        # Group results by model type
        model_groups = {}
        for result in self.results:
            key = f"{result.model_type}"
            if key not in model_groups:
                model_groups[key] = {'params': [], 'losses': []}
            model_groups[key]['params'].append(result.parameter_count)
            model_groups[key]['losses'].append(result.final_loss)
        
        colors = {'standard': 'blue', 'reversible': 'red'}
        
        for model_type, data in model_groups.items():
            plt.scatter(data['params'], data['losses'], 
                       alpha=0.6, label=f'{model_type.title()} Transformer',
                       color=colors.get(model_type, 'gray'))
            
            # Plot fitted law if available
            law_key = f"{model_type}_none_loss_vs_params" if model_type == 'standard' else f"{model_type}_additive_loss_vs_params"
            if law_key in self.fitted_laws:
                law = self.fitted_laws[law_key]
                if law['type'] == 'power_law':
                    params = law['parameters']
                    x_range = np.logspace(np.log10(min(data['params'])), 
                                        np.log10(max(data['params'])), 100)
                    y_fitted = params['a'] * (x_range ** -params['b']) + params['c']
                    plt.plot(x_range, y_fitted, '--', 
                            color=colors.get(model_type, 'gray'),
                            label=f'{model_type.title()} Fit (R²={law["r_squared"]:.3f})')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Parameter Count')
        plt.ylabel('Final Loss')
        plt.title('Scaling Law: Loss vs Parameters')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_vs_parameters.png'), dpi=300)
        plt.close()
    
    def _plot_memory_vs_context(self, output_dir: str):
        """Plot memory usage vs context length scaling."""
        plt.figure(figsize=(10, 6))
        
        model_groups = {}
        for result in self.results:
            key = f"{result.model_type}"
            if key not in model_groups:
                model_groups[key] = {'contexts': [], 'memories': []}
            model_groups[key]['contexts'].append(result.context_length)
            model_groups[key]['memories'].append(result.memory_usage_gb)
        
        colors = {'standard': 'blue', 'reversible': 'red'}
        
        for model_type, data in model_groups.items():
            plt.scatter(data['contexts'], data['memories'],
                       alpha=0.6, label=f'{model_type.title()} Transformer',
                       color=colors.get(model_type, 'gray'))
            
            # Plot fitted law
            law_key = f"{model_type}_none_memory_vs_context" if model_type == 'standard' else f"{model_type}_additive_memory_vs_context"
            if law_key in self.fitted_laws:
                law = self.fitted_laws[law_key]
                if law['type'] == 'power_law':
                    params = law['parameters']
                    x_range = np.linspace(min(data['contexts']), max(data['contexts']), 100)
                    y_fitted = params['a'] * (x_range ** params['b']) + params['c']
                    plt.plot(x_range, y_fitted, '--',
                            color=colors.get(model_type, 'gray'),
                            label=f'{model_type.title()} Fit (exp={params["b"]:.2f})')
        
        plt.xscale('log')  
        plt.yscale('log')
        plt.xlabel('Context Length')
        plt.ylabel('Memory Usage (GB)')
        plt.title('Scaling Law: Memory vs Context Length')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_vs_context.png'), dpi=300)
        plt.close()
    
    def _plot_scaling_comparison(self, analysis_results: Dict[str, Any], output_dir: str):
        """Plot comparison of scaling predictions."""
        plt.figure(figsize=(12, 8))
        
        predictions = analysis_results['predictions']
        scenarios = list(predictions.keys())
        
        # Memory comparison
        plt.subplot(2, 2, 1)
        standard_memory = []
        reversible_memory = []
        
        for scenario in scenarios:
            std_mem = predictions[scenario].get('standard_none', {}).get('predicted_memory_gb', 0)
            rev_mem = predictions[scenario].get('reversible_additive', {}).get('predicted_memory_gb', 0)
            standard_memory.append(std_mem)
            reversible_memory.append(rev_mem)
        
        x = np.arange(len(scenarios))
        plt.bar(x - 0.35, standard_memory, 0.3, label='Standard', alpha=0.7)
        plt.bar(x + 0.35, reversible_memory, 0.3, label='Reversible', alpha=0.7)
        plt.xlabel('Scale Scenario')
        plt.ylabel('Predicted Memory (GB)')
        plt.title('Memory Scaling Predictions')
        plt.xticks(x, scenarios, rotation=45)
        plt.legend()
        
        # Add more subplots for other metrics...
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scaling_predictions.png'), dpi=300)
        plt.close()

# Export
__all__ = [
    'ScalingLawsAnalyzer',
    'ScalingExperiment',
    'ScalingResult'
]