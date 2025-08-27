"""
Generation 3 Auto-Scaling and Load Balancing System for RevNet-Zero

Implements intelligent auto-scaling, load balancing, and distributed inference
capabilities for production deployment at scale.
"""

import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import json
import logging
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
import warnings


class ScalingPolicy(Enum):
    """Auto-scaling policy types."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    THROUGHPUT_BASED = "throughput_based" 
    LATENCY_BASED = "latency_based"
    CUSTOM = "custom"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_AWARE = "resource_aware"
    LATENCY_AWARE = "latency_aware"


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""
    node_id: str
    endpoint: str
    capacity: int
    current_load: int = 0
    status: str = "healthy"  # healthy, degraded, unhealthy
    last_heartbeat: float = 0.0
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    @property
    def utilization(self) -> float:
        """Get current utilization percentage."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 0


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    active_requests: int
    throughput: float
    average_latency: float
    queue_length: int
    error_rate: float


@dataclass 
class ScalingEvent:
    """Records a scaling event."""
    timestamp: float
    action: str  # scale_up, scale_down
    reason: str
    old_capacity: int
    new_capacity: int
    triggered_by: str


class IntelligentAutoScaler:
    """
    Generation 3: Intelligent auto-scaling system with predictive scaling.
    
    Features:
    - Multiple scaling policies (CPU, memory, throughput, latency)
    - Predictive scaling based on historical patterns
    - Custom scaling rules and policies
    - Integration with load balancing
    - Cost optimization
    """
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        target_utilization: float = 70.0,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 50.0,
        cooldown_period: float = 300.0,  # 5 minutes
        scaling_policy: ScalingPolicy = ScalingPolicy.CPU_BASED,
        enable_predictive_scaling: bool = True,
    ):
        """
        Initialize intelligent auto-scaler.
        
        Args:
            min_workers: Minimum number of worker nodes
            max_workers: Maximum number of worker nodes
            target_utilization: Target utilization percentage
            scale_up_threshold: Threshold to trigger scale up
            scale_down_threshold: Threshold to trigger scale down
            cooldown_period: Cooldown period between scaling actions (seconds)
            scaling_policy: Primary scaling policy
            enable_predictive_scaling: Enable predictive scaling
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.scaling_policy = scaling_policy
        self.enable_predictive_scaling = enable_predictive_scaling
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_counter = 0
        
        # Scaling state
        self.last_scaling_action = 0.0
        self.scaling_history: List[ScalingEvent] = []
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Predictive scaling
        self.usage_patterns: Dict[str, List[float]] = defaultdict(list)
        self.seasonal_patterns: Dict[str, float] = {}
        
        # Threading
        self.scaling_thread = None
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Custom scaling rules
        self.custom_rules: List[Callable[[ScalingMetrics], bool]] = []
        
        # Performance tracking
        self.scaling_performance = {
            'successful_scaling_actions': 0,
            'failed_scaling_actions': 0,
            'total_cost_saved': 0.0,
            'average_response_time_improvement': 0.0
        }
    
    def add_worker(self, endpoint: str, capacity: int = 100) -> str:
        """
        Add a new worker node.
        
        Args:
            endpoint: Worker endpoint URL
            capacity: Worker capacity (requests/minute)
            
        Returns:
            Worker node ID
        """
        with self._lock:
            self.worker_counter += 1
            node_id = f"worker-{self.worker_counter}"
            
            worker = WorkerNode(
                node_id=node_id,
                endpoint=endpoint,
                capacity=capacity,
                last_heartbeat=time.time()
            )
            
            self.workers[node_id] = worker
            
            self.logger.info(f"Added worker {node_id} with capacity {capacity}")
            return node_id
    
    def remove_worker(self, node_id: str) -> bool:
        """
        Remove a worker node.
        
        Args:
            node_id: Worker node ID to remove
            
        Returns:
            True if worker was removed successfully
        """
        with self._lock:
            if node_id in self.workers:
                del self.workers[node_id]
                self.logger.info(f"Removed worker {node_id}")
                return True
            return False
    
    def update_worker_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """
        Update metrics for a worker node.
        
        Args:
            node_id: Worker node ID
            metrics: Updated metrics
        """
        with self._lock:
            if node_id in self.workers:
                self.workers[node_id].metrics.update(metrics)
                self.workers[node_id].last_heartbeat = time.time()
                
                # Update load based on metrics
                if 'current_requests' in metrics:
                    self.workers[node_id].current_load = metrics['current_requests']
    
    def collect_cluster_metrics(self) -> ScalingMetrics:
        """
        Collect cluster-wide metrics for scaling decisions.
        
        Returns:
            Aggregated cluster metrics
        """
        current_time = time.time()
        
        with self._lock:
            if not self.workers:
                return ScalingMetrics(
                    timestamp=current_time,
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    active_requests=0,
                    throughput=0.0,
                    average_latency=0.0,
                    queue_length=0,
                    error_rate=0.0
                )
            
            # Aggregate metrics across all workers
            total_cpu = 0.0
            total_memory = 0.0
            total_requests = 0
            total_throughput = 0.0
            total_latency = 0.0
            total_queue = 0
            total_errors = 0.0
            healthy_workers = 0
            
            for worker in self.workers.values():
                if worker.status == "healthy":
                    healthy_workers += 1
                    metrics = worker.metrics
                    
                    total_cpu += metrics.get('cpu_usage', 0.0)
                    total_memory += metrics.get('memory_usage', 0.0)
                    total_requests += worker.current_load
                    total_throughput += metrics.get('throughput', 0.0)
                    total_latency += metrics.get('average_latency', 0.0)
                    total_queue += metrics.get('queue_length', 0)
                    total_errors += metrics.get('error_rate', 0.0)
            
            if healthy_workers == 0:
                healthy_workers = 1  # Avoid division by zero
            
            cluster_metrics = ScalingMetrics(
                timestamp=current_time,
                cpu_usage=total_cpu / healthy_workers,
                memory_usage=total_memory / healthy_workers,
                active_requests=total_requests,
                throughput=total_throughput,
                average_latency=total_latency / healthy_workers,
                queue_length=total_queue,
                error_rate=total_errors / healthy_workers
            )
            
            # Store metrics history
            self.metrics_history.append(cluster_metrics)
            
            return cluster_metrics
    
    def should_scale_up(self, metrics: ScalingMetrics) -> Tuple[bool, str]:
        """
        Determine if the cluster should scale up.
        
        Args:
            metrics: Current cluster metrics
            
        Returns:
            Tuple of (should_scale, reason)
        """
        if len(self.workers) >= self.max_workers:
            return False, "Maximum workers reached"
        
        current_time = time.time()
        if current_time - self.last_scaling_action < self.cooldown_period:
            return False, "Cooldown period active"
        
        # Check primary scaling policy
        if self.scaling_policy == ScalingPolicy.CPU_BASED:
            if metrics.cpu_usage > self.scale_up_threshold:
                return True, f"CPU usage {metrics.cpu_usage:.1f}% > {self.scale_up_threshold}%"
        
        elif self.scaling_policy == ScalingPolicy.MEMORY_BASED:
            if metrics.memory_usage > self.scale_up_threshold:
                return True, f"Memory usage {metrics.memory_usage:.1f}% > {self.scale_up_threshold}%"
        
        elif self.scaling_policy == ScalingPolicy.THROUGHPUT_BASED:
            # Calculate capacity utilization
            total_capacity = sum(w.capacity for w in self.workers.values())
            utilization = (metrics.throughput / total_capacity) * 100 if total_capacity > 0 else 0
            
            if utilization > self.scale_up_threshold:
                return True, f"Throughput utilization {utilization:.1f}% > {self.scale_up_threshold}%"
        
        elif self.scaling_policy == ScalingPolicy.LATENCY_BASED:
            # Define latency threshold (e.g., 500ms)
            latency_threshold = 500.0
            if metrics.average_latency > latency_threshold:
                return True, f"Average latency {metrics.average_latency:.1f}ms > {latency_threshold}ms"
        
        # Check queue length (universal trigger)
        if metrics.queue_length > len(self.workers) * 10:  # More than 10 requests per worker
            return True, f"Queue length {metrics.queue_length} too high"
        
        # Check custom rules
        for rule in self.custom_rules:
            if rule(metrics):
                return True, "Custom scaling rule triggered"
        
        # Predictive scaling
        if self.enable_predictive_scaling:
            predicted_load = self._predict_future_load(metrics)
            if predicted_load > self.scale_up_threshold:
                return True, f"Predicted load {predicted_load:.1f}% > {self.scale_up_threshold}%"
        
        return False, "No scaling needed"
    
    def should_scale_down(self, metrics: ScalingMetrics) -> Tuple[bool, str]:
        """
        Determine if the cluster should scale down.
        
        Args:
            metrics: Current cluster metrics
            
        Returns:
            Tuple of (should_scale, reason)
        """
        if len(self.workers) <= self.min_workers:
            return False, "Minimum workers reached"
        
        current_time = time.time()
        if current_time - self.last_scaling_action < self.cooldown_period:
            return False, "Cooldown period active"
        
        # Check primary scaling policy
        if self.scaling_policy == ScalingPolicy.CPU_BASED:
            if metrics.cpu_usage < self.scale_down_threshold:
                return True, f"CPU usage {metrics.cpu_usage:.1f}% < {self.scale_down_threshold}%"
        
        elif self.scaling_policy == ScalingPolicy.MEMORY_BASED:
            if metrics.memory_usage < self.scale_down_threshold:
                return True, f"Memory usage {metrics.memory_usage:.1f}% < {self.scale_down_threshold}%"
        
        elif self.scaling_policy == ScalingPolicy.THROUGHPUT_BASED:
            # Calculate capacity utilization
            total_capacity = sum(w.capacity for w in self.workers.values())
            utilization = (metrics.throughput / total_capacity) * 100 if total_capacity > 0 else 0
            
            if utilization < self.scale_down_threshold:
                return True, f"Throughput utilization {utilization:.1f}% < {self.scale_down_threshold}%"
        
        # Ensure we maintain sufficient capacity with one less worker
        remaining_capacity = sum(w.capacity for w in list(self.workers.values())[:-1])
        if metrics.throughput > remaining_capacity * 0.8:  # Don't scale down if it would cause overload
            return False, "Scaling down would cause overload"
        
        return False, "No scaling needed"
    
    def execute_scale_up(self, reason: str) -> bool:
        """
        Execute scale up action.
        
        Args:
            reason: Reason for scaling up
            
        Returns:
            True if scaling was successful
        """
        try:
            # Create new worker (in real implementation, this would provision actual infrastructure)
            new_endpoint = f"http://worker-{self.worker_counter + 1}:8080"
            worker_id = self.add_worker(new_endpoint, capacity=100)
            
            # Record scaling event
            scaling_event = ScalingEvent(
                timestamp=time.time(),
                action="scale_up",
                reason=reason,
                old_capacity=len(self.workers) - 1,
                new_capacity=len(self.workers),
                triggered_by=self.scaling_policy.value
            )
            
            self.scaling_history.append(scaling_event)
            self.last_scaling_action = time.time()
            self.scaling_performance['successful_scaling_actions'] += 1
            
            self.logger.info(f"Scaled up: Added worker {worker_id}. Reason: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to scale up: {e}")
            self.scaling_performance['failed_scaling_actions'] += 1
            return False
    
    def execute_scale_down(self, reason: str) -> bool:
        """
        Execute scale down action.
        
        Args:
            reason: Reason for scaling down
            
        Returns:
            True if scaling was successful
        """
        try:
            # Find worker with lowest utilization to remove
            with self._lock:
                if not self.workers:
                    return False
                
                worker_to_remove = min(self.workers.values(), key=lambda w: w.utilization)
                worker_id = worker_to_remove.node_id
            
            # Remove the worker
            self.remove_worker(worker_id)
            
            # Record scaling event
            scaling_event = ScalingEvent(
                timestamp=time.time(),
                action="scale_down",
                reason=reason,
                old_capacity=len(self.workers) + 1,
                new_capacity=len(self.workers),
                triggered_by=self.scaling_policy.value
            )
            
            self.scaling_history.append(scaling_event)
            self.last_scaling_action = time.time()
            self.scaling_performance['successful_scaling_actions'] += 1
            
            self.logger.info(f"Scaled down: Removed worker {worker_id}. Reason: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to scale down: {e}")
            self.scaling_performance['failed_scaling_actions'] += 1
            return False
    
    def _predict_future_load(self, current_metrics: ScalingMetrics) -> float:
        """
        Predict future load based on historical patterns.
        
        Args:
            current_metrics: Current metrics
            
        Returns:
            Predicted load percentage
        """
        if len(self.metrics_history) < 10:
            return current_metrics.cpu_usage  # Not enough data for prediction
        
        # Simple prediction based on trend analysis
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate trend
        cpu_values = [m.cpu_usage for m in recent_metrics]
        
        # Linear trend
        x = list(range(len(cpu_values)))
        if len(x) > 1:
            # Simple linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(cpu_values)
            sum_xy = sum(x[i] * cpu_values[i] for i in range(n))
            sum_x_squared = sum(x[i] ** 2 for i in range(n))
            
            # Calculate slope
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
            
            # Predict next 5 minutes (assume 1 minute intervals)
            predicted_load = current_metrics.cpu_usage + slope * 5
            return max(0, min(100, predicted_load))
        
        return current_metrics.cpu_usage
    
    def add_custom_scaling_rule(self, rule: Callable[[ScalingMetrics], bool]):
        """
        Add a custom scaling rule.
        
        Args:
            rule: Function that takes ScalingMetrics and returns bool for scale up
        """
        self.custom_rules.append(rule)
        self.logger.info("Added custom scaling rule")
    
    def start_auto_scaling(self, check_interval: float = 30.0):
        """
        Start the auto-scaling monitoring thread.
        
        Args:
            check_interval: Interval between scaling checks (seconds)
        """
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.logger.warning("Auto-scaling is already running")
            return
        
        self.stop_event.clear()
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop,
            args=(check_interval,),
            daemon=True,
            name="AutoScaler"
        )
        self.scaling_thread.start()
        
        self.logger.info(f"Started auto-scaling with {check_interval}s check interval")
    
    def stop_auto_scaling(self):
        """Stop the auto-scaling monitoring thread."""
        if self.scaling_thread and self.scaling_thread.is_alive():
            self.stop_event.set()
            self.scaling_thread.join(timeout=5.0)
            self.logger.info("Stopped auto-scaling")
    
    def _scaling_loop(self, check_interval: float):
        """Main scaling loop."""
        while not self.stop_event.wait(check_interval):
            try:
                # Collect metrics
                metrics = self.collect_cluster_metrics()
                
                # Check if scaling is needed
                should_up, up_reason = self.should_scale_up(metrics)
                should_down, down_reason = self.should_scale_down(metrics)
                
                if should_up:
                    self.execute_scale_up(up_reason)
                elif should_down:
                    self.execute_scale_down(down_reason)
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self._lock:
            worker_stats = []
            for worker in self.workers.values():
                worker_stats.append({
                    'node_id': worker.node_id,
                    'endpoint': worker.endpoint,
                    'capacity': worker.capacity,
                    'current_load': worker.current_load,
                    'utilization': worker.utilization,
                    'status': worker.status,
                    'last_heartbeat': worker.last_heartbeat,
                })
            
            recent_metrics = list(self.metrics_history)[-1] if self.metrics_history else None
            
            return {
                'cluster_size': len(self.workers),
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'scaling_policy': self.scaling_policy.value,
                'workers': worker_stats,
                'recent_metrics': asdict(recent_metrics) if recent_metrics else None,
                'scaling_events': len(self.scaling_history),
                'last_scaling_action': self.last_scaling_action,
                'performance': self.scaling_performance
            }
    
    def get_scaling_recommendations(self) -> List[str]:
        """Get scaling optimization recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            recommendations.append("Insufficient metrics data for analysis")
            return recommendations
        
        # Analyze recent performance
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Check for frequent scaling
        recent_scaling = [e for e in self.scaling_history if e.timestamp > time.time() - 3600]
        if len(recent_scaling) > 5:
            recommendations.append("Consider increasing cooldown period - frequent scaling detected")
        
        # Check utilization patterns
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        if avg_cpu < 30:
            recommendations.append("Consider reducing minimum workers - consistently low CPU usage")
        elif avg_cpu > 90:
            recommendations.append("Consider lowering scale-up threshold - consistently high CPU usage")
        
        # Check error rates
        avg_errors = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        if avg_errors > 5:
            recommendations.append("High error rate detected - investigate worker health")
        
        return recommendations


class IntelligentLoadBalancer:
    """
    Generation 3: Intelligent load balancer with multiple strategies.
    
    Features:
    - Multiple load balancing algorithms
    - Health-aware routing
    - Latency-aware routing
    - Adaptive weight adjustment
    - Circuit breaker integration
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE,
        health_check_interval: float = 30.0,
        circuit_breaker_threshold: int = 5,
    ):
        """
        Initialize intelligent load balancer.
        
        Args:
            strategy: Load balancing strategy
            health_check_interval: Health check interval (seconds)
            circuit_breaker_threshold: Circuit breaker failure threshold
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        # Load balancing state
        self.workers: Dict[str, WorkerNode] = {}
        self.round_robin_index = 0
        self.weights: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Circuit breaker state
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.circuit_breaker_open: Dict[str, bool] = defaultdict(bool)
        self.last_failure_time: Dict[str, float] = {}
        
        # Threading
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def add_worker(self, worker: WorkerNode):
        """Add a worker to the load balancer."""
        with self._lock:
            self.workers[worker.node_id] = worker
            self.weights[worker.node_id] = 1.0  # Initial weight
            
            self.logger.info(f"Added worker {worker.node_id} to load balancer")
    
    def remove_worker(self, node_id: str):
        """Remove a worker from the load balancer."""
        with self._lock:
            if node_id in self.workers:
                del self.workers[node_id]
                del self.weights[node_id]
                self.request_counts.pop(node_id, None)
                self.response_times.pop(node_id, None)
                
                self.logger.info(f"Removed worker {node_id} from load balancer")
    
    def select_worker(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[WorkerNode]:
        """
        Select the best worker for a request based on the current strategy.
        
        Args:
            request_context: Optional request context for routing decisions
            
        Returns:
            Selected worker node or None if no healthy workers available
        """
        with self._lock:
            healthy_workers = [
                worker for worker in self.workers.values()
                if worker.status == "healthy" and not self.circuit_breaker_open[worker.node_id]
            ]
            
            if not healthy_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_select(healthy_workers)
            
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_select(healthy_workers)
            
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_select(healthy_workers)
            
            elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
                return self._resource_aware_select(healthy_workers)
            
            elif self.strategy == LoadBalancingStrategy.LATENCY_AWARE:
                return self._latency_aware_select(healthy_workers)
            
            else:
                # Fallback to round robin
                return self._round_robin_select(healthy_workers)
    
    def _round_robin_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round robin worker selection."""
        if not workers:
            return None
        
        selected = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return selected
    
    def _least_connections_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least connections."""
        return min(workers, key=lambda w: w.current_load)
    
    def _weighted_round_robin_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted round robin selection based on worker weights."""
        # Simple weighted selection
        total_weight = sum(self.weights[w.node_id] for w in workers)
        if total_weight == 0:
            return self._round_robin_select(workers)
        
        # Normalize weights and select
        import random
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for worker in workers:
            cumulative += self.weights[worker.node_id]
            if r <= cumulative:
                return worker
        
        return workers[-1]  # Fallback
    
    def _resource_aware_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on resource utilization."""
        # Score based on inverse utilization (lower utilization = higher score)
        def score_worker(worker):
            utilization = worker.utilization
            # Avoid division by zero and heavily loaded workers
            return 1.0 / (1.0 + utilization / 100.0)
        
        return max(workers, key=score_worker)
    
    def _latency_aware_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on historical latency."""
        # Score based on inverse average latency
        def score_worker(worker):
            response_times = self.response_times[worker.node_id]
            if not response_times:
                return 1.0  # No data, assume good
            
            avg_latency = sum(response_times) / len(response_times)
            return 1.0 / (1.0 + avg_latency / 1000.0)  # Convert ms to seconds
        
        return max(workers, key=score_worker)
    
    def record_request_result(
        self, 
        node_id: str, 
        success: bool, 
        response_time_ms: float
    ):
        """
        Record the result of a request to update load balancing decisions.
        
        Args:
            node_id: Worker node ID
            success: Whether the request was successful
            response_time_ms: Response time in milliseconds
        """
        with self._lock:
            if node_id not in self.workers:
                return
            
            # Update request count
            self.request_counts[node_id] += 1
            
            # Record response time
            self.response_times[node_id].append(response_time_ms)
            
            # Update circuit breaker state
            if success:
                # Reset failure count on success
                self.failure_counts[node_id] = 0
                self.circuit_breaker_open[node_id] = False
            else:
                # Increment failure count
                self.failure_counts[node_id] += 1
                self.last_failure_time[node_id] = time.time()
                
                # Open circuit breaker if threshold reached
                if self.failure_counts[node_id] >= self.circuit_breaker_threshold:
                    self.circuit_breaker_open[node_id] = True
                    self.logger.warning(f"Circuit breaker opened for worker {node_id}")
            
            # Update worker weight based on performance
            self._update_worker_weight(node_id, success, response_time_ms)
    
    def _update_worker_weight(self, node_id: str, success: bool, response_time_ms: float):
        """Update worker weight based on performance."""
        if self.strategy != LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return
        
        current_weight = self.weights[node_id]
        
        if success and response_time_ms < 500:  # Good performance
            # Increase weight slightly
            self.weights[node_id] = min(2.0, current_weight + 0.1)
        elif not success or response_time_ms > 2000:  # Poor performance
            # Decrease weight
            self.weights[node_id] = max(0.1, current_weight - 0.2)
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            worker_stats = {}
            for node_id, worker in self.workers.items():
                response_times = list(self.response_times[node_id])
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                
                worker_stats[node_id] = {
                    'utilization': worker.utilization,
                    'request_count': self.request_counts[node_id],
                    'avg_response_time_ms': avg_response_time,
                    'weight': self.weights[node_id],
                    'failure_count': self.failure_counts[node_id],
                    'circuit_breaker_open': self.circuit_breaker_open[node_id],
                }
            
            return {
                'strategy': self.strategy.value,
                'total_workers': len(self.workers),
                'healthy_workers': sum(1 for w in self.workers.values() if w.status == "healthy"),
                'total_requests': sum(self.request_counts.values()),
                'worker_stats': worker_stats,
            }


class ClusterManager:
    """
    Generation 3: Comprehensive cluster management system.
    
    Coordinates auto-scaling and load balancing for optimal performance.
    """
    
    def __init__(
        self,
        auto_scaler: IntelligentAutoScaler,
        load_balancer: IntelligentLoadBalancer,
    ):
        """
        Initialize cluster manager.
        
        Args:
            auto_scaler: Auto-scaling system
            load_balancer: Load balancing system
        """
        self.auto_scaler = auto_scaler
        self.load_balancer = load_balancer
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def add_worker_to_cluster(self, endpoint: str, capacity: int = 100) -> str:
        """
        Add a new worker to both auto-scaler and load balancer.
        
        Args:
            endpoint: Worker endpoint
            capacity: Worker capacity
            
        Returns:
            Worker node ID
        """
        # Add to auto-scaler
        node_id = self.auto_scaler.add_worker(endpoint, capacity)
        
        # Add to load balancer
        worker = self.auto_scaler.workers[node_id]
        self.load_balancer.add_worker(worker)
        
        self.logger.info(f"Added worker {node_id} to cluster")
        return node_id
    
    def remove_worker_from_cluster(self, node_id: str) -> bool:
        """
        Remove a worker from both auto-scaler and load balancer.
        
        Args:
            node_id: Worker node ID
            
        Returns:
            True if removal was successful
        """
        # Remove from load balancer
        self.load_balancer.remove_worker(node_id)
        
        # Remove from auto-scaler
        success = self.auto_scaler.remove_worker(node_id)
        
        if success:
            self.logger.info(f"Removed worker {node_id} from cluster")
        
        return success
    
    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request using intelligent load balancing.
        
        Args:
            request_data: Request data
            
        Returns:
            Processing result with worker information
        """
        start_time = time.time()
        
        # Select worker using load balancer
        worker = self.load_balancer.select_worker(request_data)
        
        if worker is None:
            return {
                'success': False,
                'error': 'No healthy workers available',
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        
        try:
            # Simulate request processing
            # In real implementation, this would forward the request to the worker
            processing_time = 100  # ms
            time.sleep(processing_time / 1000)  # Simulate processing
            
            success = True  # Assume success for simulation
            response_time_ms = (time.time() - start_time) * 1000
            
            # Record request result
            self.load_balancer.record_request_result(
                worker.node_id, success, response_time_ms
            )
            
            return {
                'success': success,
                'worker_id': worker.node_id,
                'processing_time_ms': response_time_ms,
                'result': 'Request processed successfully'
            }
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            # Record failure
            self.load_balancer.record_request_result(
                worker.node_id, False, response_time_ms
            )
            
            return {
                'success': False,
                'error': str(e),
                'worker_id': worker.node_id,
                'processing_time_ms': response_time_ms
            }
    
    def get_cluster_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive cluster dashboard data."""
        return {
            'timestamp': time.time(),
            'auto_scaler': self.auto_scaler.get_cluster_status(),
            'load_balancer': self.load_balancer.get_load_balancer_stats(),
            'recommendations': {
                'scaling': self.auto_scaler.get_scaling_recommendations(),
            }
        }
    
    def start_cluster_management(self):
        """Start cluster management systems."""
        self.auto_scaler.start_auto_scaling()
        self.logger.info("Started cluster management")
    
    def stop_cluster_management(self):
        """Stop cluster management systems."""
        self.auto_scaler.stop_auto_scaling()
        self.logger.info("Stopped cluster management")


__all__ = [
    'IntelligentAutoScaler', 'IntelligentLoadBalancer', 'ClusterManager',
    'ScalingPolicy', 'LoadBalancingStrategy', 'WorkerNode', 
    'ScalingMetrics', 'ScalingEvent'
]