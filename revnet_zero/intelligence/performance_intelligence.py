"""
Performance Intelligence System for autonomous optimization and monitoring
"""

import torch
import torch.nn as nn
import psutil
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
import json
from collections import deque, defaultdict
import numpy as np
from pathlib import Path

@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time"""
    timestamp: float
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_percent: float
    memory_percent: float
    throughput: float  # tokens/second
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    
@dataclass
class PerformanceAlert:
    """Performance alert when thresholds are exceeded"""
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    timestamp: float
    metrics: Dict[str, float]
    suggested_actions: List[str] = field(default_factory=list)

class RealTimeProfiler:
    """Real-time performance profiling and monitoring"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.is_profiling = False
        self.snapshots = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.profiling_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.thresholds = {
            'gpu_memory_percent': 90.0,
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'gradient_norm': 10.0,
            'throughput_drop': 0.3  # 30% drop from baseline
        }
        
        # Baseline metrics
        self.baseline_throughput: Optional[float] = None
        self.performance_alerts: List[PerformanceAlert] = []
        
        self.logger = logging.getLogger(__name__)
        
    def start_profiling(self) -> None:
        """Start continuous performance monitoring"""
        if self.is_profiling:
            return
            
        self.is_profiling = True
        self.profiling_thread = threading.Thread(target=self._profiling_loop)
        self.profiling_thread.daemon = True
        self.profiling_thread.start()
        
        self.logger.info("Started real-time performance profiling")
        
    def stop_profiling(self) -> None:
        """Stop continuous performance monitoring"""
        self.is_profiling = False
        if self.profiling_thread:
            self.profiling_thread.join()
            
        self.logger.info("Stopped real-time performance profiling")
        
    def _profiling_loop(self) -> None:
        """Main profiling loop running in background thread"""
        while self.is_profiling:
            try:
                snapshot = self._capture_snapshot()
                self.snapshots.append(snapshot)
                self._check_performance_alerts(snapshot)
                time.sleep(self.sampling_interval)
            except Exception as e:
                self.logger.error(f"Error in profiling loop: {e}")
                
    def _capture_snapshot(self) -> PerformanceSnapshot:
        """Capture current performance snapshot"""
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.max_memory_allocated()
        else:
            gpu_memory_used = 0
            gpu_memory_total = 1  # Avoid division by zero
            
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            throughput=0.0  # Will be updated by training loop
        )
        
        return snapshot
        
    def update_training_metrics(self, 
                               throughput: float,
                               loss: float,
                               learning_rate: float,
                               gradient_norm: float) -> None:
        """Update training-specific metrics"""
        if self.snapshots:
            latest = self.snapshots[-1]
            latest.throughput = throughput
            latest.loss = loss
            latest.learning_rate = learning_rate
            latest.gradient_norm = gradient_norm
            
            # Update baseline throughput
            if self.baseline_throughput is None:
                self.baseline_throughput = throughput
                
    def _check_performance_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Check for performance issues and generate alerts"""
        
        alerts = []
        
        # GPU memory alert
        gpu_memory_percent = (snapshot.gpu_memory_used / snapshot.gpu_memory_total) * 100
        if gpu_memory_percent > self.thresholds['gpu_memory_percent']:
            alerts.append(PerformanceAlert(
                alert_type='memory',
                severity='high' if gpu_memory_percent > 95 else 'medium',
                message=f"GPU memory usage at {gpu_memory_percent:.1f}%",
                timestamp=snapshot.timestamp,
                metrics={'gpu_memory_percent': gpu_memory_percent},
                suggested_actions=[
                    'Increase gradient accumulation steps',
                    'Reduce batch size',
                    'Enable gradient checkpointing'
                ]
            ))
            
        # CPU usage alert
        if snapshot.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(PerformanceAlert(
                alert_type='cpu',
                severity='medium',
                message=f"CPU usage at {snapshot.cpu_percent:.1f}%",
                timestamp=snapshot.timestamp,
                metrics={'cpu_percent': snapshot.cpu_percent},
                suggested_actions=[
                    'Reduce data loading workers',
                    'Optimize preprocessing pipeline'
                ]
            ))
            
        # Throughput drop alert
        if (self.baseline_throughput and snapshot.throughput > 0 and
            (self.baseline_throughput - snapshot.throughput) / self.baseline_throughput > 
            self.thresholds['throughput_drop']):
            
            drop_percent = ((self.baseline_throughput - snapshot.throughput) / 
                          self.baseline_throughput) * 100
            alerts.append(PerformanceAlert(
                alert_type='throughput',
                severity='medium',
                message=f"Throughput dropped by {drop_percent:.1f}%",
                timestamp=snapshot.timestamp,
                metrics={
                    'current_throughput': snapshot.throughput,
                    'baseline_throughput': self.baseline_throughput,
                    'drop_percent': drop_percent
                },
                suggested_actions=[
                    'Check for memory fragmentation',
                    'Restart training process',
                    'Reduce sequence length'
                ]
            ))
            
        # Gradient explosion alert
        if snapshot.gradient_norm and snapshot.gradient_norm > self.thresholds['gradient_norm']:
            alerts.append(PerformanceAlert(
                alert_type='gradient',
                severity='high',
                message=f"Gradient norm explosion: {snapshot.gradient_norm:.3f}",
                timestamp=snapshot.timestamp,
                metrics={'gradient_norm': snapshot.gradient_norm},
                suggested_actions=[
                    'Enable gradient clipping',
                    'Reduce learning rate',
                    'Check for numerical instability'
                ]
            ))
            
        # Add new alerts
        self.performance_alerts.extend(alerts)
        
        # Log critical alerts immediately
        for alert in alerts:
            if alert.severity == 'high':
                self.logger.warning(f"Performance Alert: {alert.message}")
                
    def get_performance_summary(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get performance summary for the last N minutes"""
        
        cutoff_time = time.time() - (window_minutes * 60)
        recent_snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]
        
        if not recent_snapshots:
            return {}
            
        # Calculate statistics
        gpu_memory_percents = [(s.gpu_memory_used / s.gpu_memory_total) * 100 
                              for s in recent_snapshots]
        cpu_percents = [s.cpu_percent for s in recent_snapshots]
        memory_percents = [s.memory_percent for s in recent_snapshots]
        throughputs = [s.throughput for s in recent_snapshots if s.throughput > 0]
        
        summary = {
            'window_minutes': window_minutes,
            'num_samples': len(recent_snapshots),
            'gpu_memory': {
                'mean': np.mean(gpu_memory_percents),
                'max': np.max(gpu_memory_percents),
                'std': np.std(gpu_memory_percents)
            } if gpu_memory_percents else {},
            'cpu_usage': {
                'mean': np.mean(cpu_percents),
                'max': np.max(cpu_percents),
                'std': np.std(cpu_percents)
            } if cpu_percents else {},
            'memory_usage': {
                'mean': np.mean(memory_percents),
                'max': np.max(memory_percents),
                'std': np.std(memory_percents)
            } if memory_percents else {},
            'throughput': {
                'mean': np.mean(throughputs),
                'min': np.min(throughputs),
                'std': np.std(throughputs)
            } if throughputs else {}
        }
        
        return summary

class PerformanceIntelligence:
    """Intelligent performance monitoring and optimization system"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        self.model = model
        self.config = config or {}
        
        # Components
        self.profiler = RealTimeProfiler(
            sampling_interval=self.config.get('sampling_interval', 1.0)
        )
        
        # Performance history
        self.optimization_history: List[Dict] = []
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Optimization strategies
        self.optimization_strategies = {
            'memory_pressure': self._handle_memory_pressure,
            'throughput_degradation': self._handle_throughput_degradation,
            'gradient_instability': self._handle_gradient_instability,
            'resource_contention': self._handle_resource_contention
        }
        
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self) -> None:
        """Start intelligent performance monitoring"""
        self.profiler.start_profiling()
        self.logger.info("Started intelligent performance monitoring")
        
    def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self.profiler.stop_profiling()
        self.logger.info("Stopped performance monitoring")
        
    def update_training_metrics(self, **metrics) -> None:
        """Update training metrics and trigger optimization if needed"""
        self.profiler.update_training_metrics(**metrics)
        
        # Check for optimization opportunities
        self._analyze_and_optimize()
        
    def _analyze_and_optimize(self) -> None:
        """Analyze current performance and apply optimizations"""
        
        # Get recent alerts
        recent_alerts = [a for a in self.profiler.performance_alerts 
                        if time.time() - a.timestamp < 60]  # Last minute
        
        if not recent_alerts:
            return
            
        # Group alerts by type
        alert_types = defaultdict(list)
        for alert in recent_alerts:
            alert_types[alert.alert_type].append(alert)
            
        # Apply optimizations
        for alert_type, alerts in alert_types.items():
            if alert_type == 'memory' and len(alerts) >= 2:  # Multiple memory alerts
                self._handle_memory_pressure(alerts)
            elif alert_type == 'throughput' and len(alerts) >= 1:
                self._handle_throughput_degradation(alerts)
            elif alert_type == 'gradient' and len(alerts) >= 1:
                self._handle_gradient_instability(alerts)
                
    def _handle_memory_pressure(self, alerts: List[PerformanceAlert]) -> None:
        """Handle memory pressure situations"""
        
        self.logger.info("Handling memory pressure...")
        
        optimizations = []
        
        # Increase gradient accumulation
        if hasattr(self.model, 'gradient_accumulation_steps'):
            old_steps = self.model.gradient_accumulation_steps
            self.model.gradient_accumulation_steps = min(old_steps * 2, 32)
            optimizations.append(f'gradient_accumulation: {old_steps} -> {self.model.gradient_accumulation_steps}')
            
        # Enable more aggressive memory scheduling
        if hasattr(self.model, 'memory_scheduler'):
            self.model.memory_scheduler.set_strategy('aggressive')
            optimizations.append('memory_scheduler: aggressive')
            
        # Record optimization
        self.optimization_history.append({
            'timestamp': time.time(),
            'trigger': 'memory_pressure',
            'optimizations': optimizations,
            'alerts': len(alerts)
        })
        
        self.logger.info(f"Applied memory optimizations: {optimizations}")
        
    def _handle_throughput_degradation(self, alerts: List[PerformanceAlert]) -> None:
        """Handle throughput degradation"""
        
        self.logger.info("Handling throughput degradation...")
        
        optimizations = []
        
        # Enable flash attention if available
        for layer in getattr(self.model, 'layers', []):
            if hasattr(layer, 'use_flash_attention'):
                layer.use_flash_attention = True
                optimizations.append('flash_attention: enabled')
                
        # Optimize cache settings
        if hasattr(self.model, 'cache_manager'):
            self.model.cache_manager.optimize_for_throughput()
            optimizations.append('cache_optimization: throughput')
            
        self.optimization_history.append({
            'timestamp': time.time(),
            'trigger': 'throughput_degradation', 
            'optimizations': optimizations,
            'alerts': len(alerts)
        })
        
        self.logger.info(f"Applied throughput optimizations: {optimizations}")
        
    def _handle_gradient_instability(self, alerts: List[PerformanceAlert]) -> None:
        """Handle gradient instability"""
        
        self.logger.info("Handling gradient instability...")
        
        optimizations = []
        
        # Enable gradient clipping
        if hasattr(self.model, 'gradient_clip_value'):
            self.model.gradient_clip_value = 1.0
            optimizations.append('gradient_clipping: 1.0')
            
        self.optimization_history.append({
            'timestamp': time.time(),
            'trigger': 'gradient_instability',
            'optimizations': optimizations,
            'alerts': len(alerts)
        })
        
        self.logger.info(f"Applied stability optimizations: {optimizations}")
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        summary = self.profiler.get_performance_summary()
        recent_alerts = [a for a in self.profiler.performance_alerts 
                        if time.time() - a.timestamp < 3600]  # Last hour
        
        report = {
            'timestamp': time.time(),
            'performance_summary': summary,
            'recent_alerts': len(recent_alerts),
            'alert_breakdown': defaultdict(int),
            'optimizations_applied': len(self.optimization_history),
            'recent_optimizations': self.optimization_history[-5:] if self.optimization_history else []
        }
        
        # Alert breakdown
        for alert in recent_alerts:
            report['alert_breakdown'][alert.alert_type] += 1
            
        return report
        
    def export_performance_data(self, filepath: str) -> None:
        """Export performance data for analysis"""
        
        data = {
            'snapshots': [
                {
                    'timestamp': s.timestamp,
                    'gpu_memory_used': s.gpu_memory_used,
                    'gpu_memory_total': s.gpu_memory_total,
                    'cpu_percent': s.cpu_percent,
                    'memory_percent': s.memory_percent,
                    'throughput': s.throughput,
                    'loss': s.loss,
                    'gradient_norm': s.gradient_norm
                }
                for s in self.profiler.snapshots
            ],
            'alerts': [
                {
                    'type': a.alert_type,
                    'severity': a.severity,
                    'message': a.message,
                    'timestamp': a.timestamp,
                    'metrics': a.metrics
                }
                for a in self.profiler.performance_alerts
            ],
            'optimizations': self.optimization_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        self.logger.info(f"Exported performance data to {filepath}")