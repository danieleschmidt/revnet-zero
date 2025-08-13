"""
Statistical Validation Framework

This module provides comprehensive statistical testing and validation capabilities
for comparing RevNet-Zero against baseline methods with rigorous scientific standards.

Features:
- Automated experiment design and execution
- Multiple comparison corrections (Bonferroni, Benjamini-Hochberg)
- Effect size calculations (Cohen's d, eta-squared)
- Power analysis and sample size determination
- Reproducibility protocols with detailed logging
- Publication-ready results formatting
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import hashlib
from collections import defaultdict
import warnings


@dataclass
class ExperimentResult:
    """Container for experimental results with metadata."""
    method_name: str
    metric_name: str
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S'))
    
    def __post_init__(self):
        self.values = [float(v) for v in self.values]  # Ensure float conversion
    
    @property
    def mean(self) -> float:
        return np.mean(self.values)
    
    @property
    def std(self) -> float:
        return np.std(self.values, ddof=1)
    
    @property
    def median(self) -> float:
        return np.median(self.values)
    
    @property
    def n(self) -> int:
        return len(self.values)


@dataclass
class StatisticalTest:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    method1: str
    method2: str
    metric: str
    interpretation: str
    corrected_p_value: Optional[float] = None
    
    @property
    def is_significant(self) -> bool:
        p_val = self.corrected_p_value if self.corrected_p_value is not None else self.p_value
        return p_val < 0.05
    
    @property
    def significance_level(self) -> str:
        p_val = self.corrected_p_value if self.corrected_p_value is not None else self.p_value
        if p_val < 0.001:
            return "***"
        elif p_val < 0.01:
            return "**"
        elif p_val < 0.05:
            return "*"
        else:
            return "ns"


class StatisticalFramework:
    """
    Comprehensive statistical framework for model comparison.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        min_effect_size: float = 0.5,
        power: float = 0.8,
        seed: int = 42,
        output_dir: str = "./statistical_results"
    ):
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.power = power
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Storage for results
        self.experiment_results: List[ExperimentResult] = []
        self.statistical_tests: List[StatisticalTest] = []
        
        # Experiment metadata
        self.experiment_id = self._generate_experiment_id()
        self.start_time = time.time()
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{random_hash}"
    
    def add_experiment_result(
        self,
        method_name: str,
        metric_name: str,
        values: List[float],
        **metadata
    ) -> None:
        """Add experimental result to the framework."""
        result = ExperimentResult(
            method_name=method_name,
            metric_name=metric_name,
            values=values,
            metadata=metadata
        )
        self.experiment_results.append(result)
    
    def calculate_sample_size(
        self,
        effect_size: float = None,
        alpha: float = None,
        power: float = None
    ) -> int:
        """
        Calculate required sample size for given statistical parameters.
        
        Uses Cohen's method for t-test sample size calculation.
        """
        effect_size = effect_size or self.min_effect_size
        alpha = alpha or self.alpha
        power = power or self.power
        
        # Approximate formula for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        return (mean1 - mean2) / pooled_std
    
    def interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def two_sample_t_test(
        self,
        method1: str,
        method2: str,
        metric: str,
        alternative: str = 'two-sided'
    ) -> StatisticalTest:
        """
        Perform two-sample t-test between methods.
        
        Args:
            method1, method2: Names of methods to compare
            metric: Name of metric to compare
            alternative: 'two-sided', 'greater', or 'less'
        """
        # Find results for both methods
        results1 = [r for r in self.experiment_results 
                   if r.method_name == method1 and r.metric_name == metric]
        results2 = [r for r in self.experiment_results 
                   if r.method_name == method2 and r.metric_name == metric]
        
        if not results1 or not results2:
            raise ValueError(f"No results found for comparison: {method1} vs {method2} on {metric}")
        
        # Combine all values from multiple experiments
        values1 = []
        values2 = []
        for r in results1:
            values1.extend(r.values)
        for r in results2:
            values2.extend(r.values)
        
        # Perform t-test
        statistic, p_value = ttest_ind(values1, values2, alternative=alternative)
        
        # Calculate effect size
        effect_size = self.cohens_d(values1, values2)
        
        # Confidence interval for difference of means
        mean_diff = np.mean(values1) - np.mean(values2)
        se_diff = np.sqrt(np.var(values1, ddof=1)/len(values1) + np.var(values2, ddof=1)/len(values2))
        df = len(values1) + len(values2) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        interpretation = f"{method1} vs {method2}: " + self.interpret_effect_size(effect_size)
        
        test_result = StatisticalTest(
            test_name="Two-sample t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            method1=method1,
            method2=method2,
            metric=metric,
            interpretation=interpretation
        )
        
        self.statistical_tests.append(test_result)
        return test_result
    
    def mann_whitney_u_test(
        self,
        method1: str,
        method2: str,
        metric: str,
        alternative: str = 'two-sided'
    ) -> StatisticalTest:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        """
        # Find results for both methods
        results1 = [r for r in self.experiment_results 
                   if r.method_name == method1 and r.metric_name == metric]
        results2 = [r for r in self.experiment_results 
                   if r.method_name == method2 and r.metric_name == metric]
        
        if not results1 or not results2:
            raise ValueError(f"No results found for comparison: {method1} vs {method2} on {metric}")
        
        # Combine all values
        values1 = []
        values2 = []
        for r in results1:
            values1.extend(r.values)
        for r in results2:
            values2.extend(r.values)
        
        # Perform Mann-Whitney U test
        statistic, p_value = mannwhitneyu(values1, values2, alternative=alternative)
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(values1), len(values2)
        effect_size = 2 * statistic / (n1 * n2) - 1
        
        # Confidence interval (approximation)
        se = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = effect_size - z_critical * se
        ci_upper = effect_size + z_critical * se
        
        interpretation = f"{method1} vs {method2}: non-parametric comparison"
        
        test_result = StatisticalTest(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            method1=method1,
            method2=method2,
            metric=metric,
            interpretation=interpretation
        )
        
        self.statistical_tests.append(test_result)
        return test_result
    
    def multiple_comparison_correction(
        self,
        method: str = 'bonferroni'
    ) -> List[StatisticalTest]:
        """
        Apply multiple comparison correction to all stored tests.
        
        Args:
            method: 'bonferroni', 'benjamini_hochberg', or 'none'
        """
        if not self.statistical_tests:
            return []
        
        p_values = [test.p_value for test in self.statistical_tests]
        
        if method == 'bonferroni':
            corrected_p = [min(1.0, p * len(p_values)) for p in p_values]
        elif method == 'benjamini_hochberg':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p = np.array(p_values)[sorted_indices]
            
            corrected_sorted_p = []
            for i, p in enumerate(sorted_p):
                correction_factor = len(p_values) / (i + 1)
                corrected_sorted_p.append(min(1.0, p * correction_factor))
            
            # Ensure monotonicity
            for i in range(len(corrected_sorted_p) - 2, -1, -1):
                corrected_sorted_p[i] = min(corrected_sorted_p[i], corrected_sorted_p[i + 1])
            
            # Unsort
            corrected_p = [0] * len(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = corrected_sorted_p[i]
        else:  # No correction
            corrected_p = p_values
        
        # Update test results with corrected p-values
        for i, test in enumerate(self.statistical_tests):
            test.corrected_p_value = corrected_p[i]
        
        return self.statistical_tests
    
    def comprehensive_comparison(
        self,
        methods: List[str],
        metrics: List[str],
        correction_method: str = 'bonferroni'
    ) -> pd.DataFrame:
        """
        Perform comprehensive pairwise comparisons between all methods and metrics.
        
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        # Perform all pairwise comparisons
        for metric in metrics:
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    try:
                        # Try parametric test first
                        test_result = self.two_sample_t_test(method1, method2, metric)
                        comparison_results.append({
                            'metric': metric,
                            'method1': method1,
                            'method2': method2,
                            'test_type': 'parametric',
                            'statistic': test_result.statistic,
                            'p_value': test_result.p_value,
                            'effect_size': test_result.effect_size,
                            'ci_lower': test_result.confidence_interval[0],
                            'ci_upper': test_result.confidence_interval[1],
                            'interpretation': test_result.interpretation
                        })
                    except Exception as e:
                        warnings.warn(f"Parametric test failed for {method1} vs {method2} on {metric}: {e}")
                        
                        try:
                            # Fallback to non-parametric test
                            test_result = self.mann_whitney_u_test(method1, method2, metric)
                            comparison_results.append({
                                'metric': metric,
                                'method1': method1,
                                'method2': method2,
                                'test_type': 'non_parametric',
                                'statistic': test_result.statistic,
                                'p_value': test_result.p_value,
                                'effect_size': test_result.effect_size,
                                'ci_lower': test_result.confidence_interval[0],
                                'ci_upper': test_result.confidence_interval[1],
                                'interpretation': test_result.interpretation
                            })
                        except Exception as e2:
                            warnings.warn(f"Both tests failed for {method1} vs {method2} on {metric}: {e2}")
        
        # Apply multiple comparison correction
        self.multiple_comparison_correction(correction_method)
        
        # Add corrected p-values to results
        for i, result in enumerate(comparison_results):
            if i < len(self.statistical_tests):
                result['corrected_p_value'] = self.statistical_tests[i].corrected_p_value
                result['is_significant'] = self.statistical_tests[i].is_significant
                result['significance_level'] = self.statistical_tests[i].significance_level
        
        return pd.DataFrame(comparison_results)
    
    def power_analysis(
        self,
        method1: str,
        method2: str,
        metric: str,
        effect_sizes: List[float] = None
    ) -> Dict[str, float]:
        """
        Perform post-hoc power analysis for a specific comparison.
        """
        if effect_sizes is None:
            effect_sizes = [0.2, 0.5, 0.8, 1.0, 1.2]
        
        # Get sample sizes from actual data
        results1 = [r for r in self.experiment_results 
                   if r.method_name == method1 and r.metric_name == metric]
        results2 = [r for r in self.experiment_results 
                   if r.method_name == method2 and r.metric_name == metric]
        
        if not results1 or not results2:
            raise ValueError(f"No results found for {method1} vs {method2} on {metric}")
        
        n1 = sum(len(r.values) for r in results1)
        n2 = sum(len(r.values) for r in results2)
        
        power_results = {}
        for effect_size in effect_sizes:
            # Calculate power for given effect size
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = effect_size * np.sqrt((n1 * n2) / (n1 + n2)) / np.sqrt(2) - z_alpha
            power = stats.norm.cdf(z_beta)
            power_results[f"effect_size_{effect_size}"] = power
        
        return power_results
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        report = {
            'experiment_id': self.experiment_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'alpha': self.alpha,
                'min_effect_size': self.min_effect_size,
                'power': self.power,
                'seed': self.seed
            },
            'data_summary': {},
            'statistical_tests': len(self.statistical_tests),
            'significant_comparisons': sum(1 for t in self.statistical_tests if t.is_significant),
            'methods_compared': list(set(r.method_name for r in self.experiment_results)),
            'metrics_analyzed': list(set(r.metric_name for r in self.experiment_results))
        }
        
        # Data summary by method and metric
        for result in self.experiment_results:
            key = f"{result.method_name}_{result.metric_name}"
            if key not in report['data_summary']:
                report['data_summary'][key] = {
                    'n_experiments': 0,
                    'total_samples': 0,
                    'mean_performance': 0,
                    'std_performance': 0
                }
            
            summary = report['data_summary'][key]
            summary['n_experiments'] += 1
            summary['total_samples'] += len(result.values)
            summary['mean_performance'] = (summary['mean_performance'] * (summary['n_experiments'] - 1) + result.mean) / summary['n_experiments']
            summary['std_performance'] = np.sqrt(((summary['std_performance'] ** 2) * (summary['n_experiments'] - 1) + (result.std ** 2)) / summary['n_experiments'])
        
        return report
    
    def save_results(self, filename_prefix: str = "statistical_analysis") -> Path:
        """Save all results to files."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_path = self.output_dir / f"{filename_prefix}_{self.experiment_id}"
        
        # Save experiment results
        results_df = pd.DataFrame([
            {
                'experiment_id': self.experiment_id,
                'method_name': r.method_name,
                'metric_name': r.metric_name,
                'mean': r.mean,
                'std': r.std,
                'median': r.median,
                'n_samples': r.n,
                'timestamp': r.timestamp,
                **r.metadata
            }
            for r in self.experiment_results
        ])
        results_df.to_csv(base_path.with_suffix('_results.csv'), index=False)
        
        # Save statistical tests
        tests_df = pd.DataFrame([
            {
                'test_name': t.test_name,
                'method1': t.method1,
                'method2': t.method2,
                'metric': t.metric,
                'statistic': t.statistic,
                'p_value': t.p_value,
                'corrected_p_value': t.corrected_p_value,
                'effect_size': t.effect_size,
                'ci_lower': t.confidence_interval[0],
                'ci_upper': t.confidence_interval[1],
                'is_significant': t.is_significant,
                'significance_level': t.significance_level,
                'interpretation': t.interpretation
            }
            for t in self.statistical_tests
        ])
        tests_df.to_csv(base_path.with_suffix('_tests.csv'), index=False)
        
        # Save summary report
        summary_report = self.generate_summary_report()
        with open(base_path.with_suffix('_summary.json'), 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        return base_path
    
    def create_visualization(
        self,
        methods: List[str],
        metric: str,
        plot_type: str = 'boxplot',
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create visualization of method comparison.
        
        Args:
            methods: List of method names to compare
            metric: Metric name to visualize
            plot_type: 'boxplot', 'violin', or 'barplot'
            save_path: Optional path to save figure
        """
        # Collect data for plotting
        plot_data = []
        for method in methods:
            results = [r for r in self.experiment_results 
                      if r.method_name == method and r.metric_name == metric]
            for r in results:
                for value in r.values:
                    plot_data.append({'Method': method, 'Value': value, 'Metric': metric})
        
        if not plot_data:
            raise ValueError(f"No data found for methods {methods} and metric {metric}")
        
        df = pd.DataFrame(plot_data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'boxplot':
            sns.boxplot(data=df, x='Method', y='Value')
        elif plot_type == 'violin':
            sns.violinplot(data=df, x='Method', y='Value')
        elif plot_type == 'barplot':
            sns.barplot(data=df, x='Method', y='Value', ci=95)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        plt.title(f'Comparison of {metric} across methods')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


def create_statistical_framework(
    alpha: float = 0.01,  # More stringent for research
    min_effect_size: float = 0.8,  # Large effect size
    power: float = 0.95,  # High power
    **kwargs
) -> StatisticalFramework:
    """
    Factory function to create a research-grade statistical framework.
    
    Uses more stringent parameters suitable for research publication.
    """
    return StatisticalFramework(
        alpha=alpha,
        min_effect_size=min_effect_size,
        power=power,
        **kwargs
    )


__all__ = [
    'ExperimentResult',
    'StatisticalTest',
    'StatisticalFramework',
    'create_statistical_framework'
]