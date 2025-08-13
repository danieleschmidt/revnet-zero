"""
Hypothesis-Driven Development Engine for Research and Optimization
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path
import time
from scipy import stats

@dataclass
class Hypothesis:
    """Represents a research hypothesis with measurable criteria"""
    id: str
    name: str
    description: str
    success_criteria: Dict[str, float]
    baseline_metrics: Optional[Dict[str, float]] = None
    experimental_metrics: Optional[Dict[str, float]] = None
    statistical_significance: Optional[float] = None
    p_value: Optional[float] = None
    status: str = "pending"  # pending, testing, completed, failed
    created_at: float = field(default_factory=time.time)
    
class ExperimentResult:
    """Container for experiment results with statistical analysis"""
    
    def __init__(self, hypothesis: Hypothesis):
        self.hypothesis = hypothesis
        self.baseline_results: List[Dict[str, float]] = []
        self.experimental_results: List[Dict[str, float]] = []
        self.statistical_tests: Dict[str, Dict] = {}
        
    def add_baseline_result(self, metrics: Dict[str, float]) -> None:
        """Add a baseline measurement"""
        self.baseline_results.append(metrics)
        
    def add_experimental_result(self, metrics: Dict[str, float]) -> None:
        """Add an experimental measurement"""
        self.experimental_results.append(metrics)
        
    def analyze_statistical_significance(self) -> Dict[str, Any]:
        """Perform statistical analysis of results"""
        if not self.baseline_results or not self.experimental_results:
            return {}
            
        analysis = {}
        
        for metric in self.hypothesis.success_criteria.keys():
            if metric not in self.baseline_results[0] or metric not in self.experimental_results[0]:
                continue
                
            baseline_values = [r[metric] for r in self.baseline_results]
            experimental_values = [r[metric] for r in self.experimental_results]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(baseline_values, experimental_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_values) - 1) * np.var(baseline_values, ddof=1) + 
                                (len(experimental_values) - 1) * np.var(experimental_values, ddof=1)) / 
                               (len(baseline_values) + len(experimental_values) - 2))
            effect_size = (np.mean(experimental_values) - np.mean(baseline_values)) / pooled_std
            
            analysis[metric] = {
                'baseline_mean': np.mean(baseline_values),
                'experimental_mean': np.mean(experimental_values),
                'improvement': (np.mean(experimental_values) - np.mean(baseline_values)) / np.mean(baseline_values),
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05,
                'baseline_std': np.std(baseline_values),
                'experimental_std': np.std(experimental_values)
            }
            
        self.statistical_tests = analysis
        return analysis

class HypothesisEngine:
    """Engine for managing and testing research hypotheses"""
    
    def __init__(self, output_dir: str = "experiments"):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.experiment_results: Dict[str, ExperimentResult] = {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def create_hypothesis(self, 
                         name: str,
                         description: str, 
                         success_criteria: Dict[str, float],
                         hypothesis_id: Optional[str] = None) -> str:
        """Create a new research hypothesis"""
        
        if hypothesis_id is None:
            hypothesis_id = f"hyp_{len(self.hypotheses):04d}"
            
        hypothesis = Hypothesis(
            id=hypothesis_id,
            name=name,
            description=description,
            success_criteria=success_criteria
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        self.experiment_results[hypothesis_id] = ExperimentResult(hypothesis)
        
        self.logger.info(f"Created hypothesis {hypothesis_id}: {name}")
        return hypothesis_id
    
    def test_hypothesis(self, 
                       hypothesis_id: str,
                       baseline_function: Callable,
                       experimental_function: Callable,
                       num_trials: int = 10,
                       test_data: Any = None) -> Dict[str, Any]:
        """Test a hypothesis with statistical rigor"""
        
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
            
        hypothesis = self.hypotheses[hypothesis_id]
        hypothesis.status = "testing"
        result = self.experiment_results[hypothesis_id]
        
        self.logger.info(f"Testing hypothesis {hypothesis_id}: {hypothesis.name}")
        
        # Run baseline experiments
        self.logger.info(f"Running {num_trials} baseline trials...")
        for trial in range(num_trials):
            try:
                baseline_metrics = baseline_function(test_data)
                result.add_baseline_result(baseline_metrics)
                self.logger.debug(f"Baseline trial {trial + 1}: {baseline_metrics}")
            except Exception as e:
                self.logger.error(f"Baseline trial {trial + 1} failed: {e}")
                
        # Run experimental trials
        self.logger.info(f"Running {num_trials} experimental trials...")
        for trial in range(num_trials):
            try:
                experimental_metrics = experimental_function(test_data)
                result.add_experimental_result(experimental_metrics)
                self.logger.debug(f"Experimental trial {trial + 1}: {experimental_metrics}")
            except Exception as e:
                self.logger.error(f"Experimental trial {trial + 1} failed: {e}")
                
        # Analyze results
        analysis = result.analyze_statistical_significance()
        
        # Determine if hypothesis succeeded
        hypothesis_success = True
        for criterion, threshold in hypothesis.success_criteria.items():
            if criterion in analysis:
                improvement = analysis[criterion]['improvement']
                significant = analysis[criterion]['significant']
                
                # Check if improvement meets threshold and is statistically significant
                if improvement < threshold or not significant:
                    hypothesis_success = False
                    break
                    
        hypothesis.status = "completed" if hypothesis_success else "failed"
        
        # Save results
        self._save_experiment_results(hypothesis_id, result, analysis)
        
        return {
            'hypothesis_id': hypothesis_id,
            'success': hypothesis_success,
            'analysis': analysis,
            'baseline_trials': len(result.baseline_results),
            'experimental_trials': len(result.experimental_results)
        }
    
    def _save_experiment_results(self, 
                                hypothesis_id: str,
                                result: ExperimentResult, 
                                analysis: Dict) -> None:
        """Save experiment results to disk"""
        
        output_file = self.output_dir / f"{hypothesis_id}_results.json"
        
        results_data = {
            'hypothesis': {
                'id': result.hypothesis.id,
                'name': result.hypothesis.name,
                'description': result.hypothesis.description,
                'success_criteria': result.hypothesis.success_criteria,
                'status': result.hypothesis.status,
                'created_at': result.hypothesis.created_at
            },
            'baseline_results': result.baseline_results,
            'experimental_results': result.experimental_results,
            'statistical_analysis': analysis,
            'summary': self._generate_summary(result.hypothesis, analysis)
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        self.logger.info(f"Saved results to {output_file}")
    
    def _generate_summary(self, hypothesis: Hypothesis, analysis: Dict) -> str:
        """Generate human-readable summary of results"""
        
        if not analysis:
            return "Insufficient data for analysis"
            
        summary_lines = [
            f"Hypothesis: {hypothesis.name}",
            f"Status: {hypothesis.status}",
            "",
            "Results:"
        ]
        
        for metric, stats in analysis.items():
            improvement_pct = stats['improvement'] * 100
            significance = "✓" if stats['significant'] else "✗"
            
            summary_lines.append(
                f"  {metric}: {improvement_pct:+.2f}% improvement "
                f"(p={stats['p_value']:.4f}) {significance}"
            )
            
        return "\n".join(summary_lines)
    
    def get_successful_hypotheses(self) -> List[Hypothesis]:
        """Get all successfully tested hypotheses"""
        return [h for h in self.hypotheses.values() if h.status == "completed"]
    
    def generate_research_report(self) -> str:
        """Generate comprehensive research report"""
        
        report_lines = [
            "# Research Hypothesis Testing Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Total Hypotheses: {len(self.hypotheses)}",
            f"Completed: {len([h for h in self.hypotheses.values() if h.status == 'completed'])}",
            f"Failed: {len([h for h in self.hypotheses.values() if h.status == 'failed'])}",
            f"In Progress: {len([h for h in self.hypotheses.values() if h.status == 'testing'])}",
            "",
            "## Successful Hypotheses"
        ]
        
        for hypothesis in self.get_successful_hypotheses():
            if hypothesis.id in self.experiment_results:
                result = self.experiment_results[hypothesis.id]
                if result.statistical_tests:
                    report_lines.extend([
                        f"### {hypothesis.name}",
                        f"**Description:** {hypothesis.description}",
                        "",
                        "**Results:**"
                    ])
                    
                    for metric, stats in result.statistical_tests.items():
                        improvement = stats['improvement'] * 100
                        report_lines.append(
                            f"- {metric}: {improvement:+.2f}% improvement "
                            f"(p={stats['p_value']:.4f}, d={stats['effect_size']:.3f})"
                        )
                    
                    report_lines.append("")
        
        return "\n".join(report_lines)

class ExperimentTracker:
    """Tracks ongoing experiments and their progress"""
    
    def __init__(self):
        self.active_experiments: Dict[str, Dict] = {}
        self.experiment_history: List[Dict] = []
        
    def start_experiment(self, 
                        experiment_id: str,
                        hypothesis_id: str, 
                        parameters: Dict) -> None:
        """Start tracking a new experiment"""
        
        self.active_experiments[experiment_id] = {
            'hypothesis_id': hypothesis_id,
            'parameters': parameters,
            'start_time': time.time(),
            'status': 'running',
            'metrics': []
        }
        
    def record_metric(self, 
                     experiment_id: str,
                     metric_name: str, 
                     value: float,
                     step: Optional[int] = None) -> None:
        """Record a metric for an active experiment"""
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        metric_record = {
            'name': metric_name,
            'value': value,
            'timestamp': time.time(),
            'step': step
        }
        
        self.active_experiments[experiment_id]['metrics'].append(metric_record)
        
    def complete_experiment(self, experiment_id: str) -> Dict:
        """Mark an experiment as completed and archive it"""
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = self.active_experiments[experiment_id]
        experiment['status'] = 'completed'
        experiment['end_time'] = time.time()
        experiment['duration'] = experiment['end_time'] - experiment['start_time']
        
        # Archive experiment
        self.experiment_history.append(experiment.copy())
        del self.active_experiments[experiment_id]
        
        return experiment