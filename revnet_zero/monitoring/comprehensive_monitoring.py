"""
Comprehensive Monitoring System with Health Checks and Alerting
"""

import torch
import torch.nn as nn
import psutil
import time
import threading
import logging
import json
import traceback
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from pathlib import Path
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    check_name: str
    status: str  # healthy, warning, unhealthy
    message: str
    timestamp: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

@dataclass 
class Alert:
    """System alert"""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: float
    source: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

class HealthCheck(ABC):
    """Base class for health checks"""
    
    def __init__(self, name: str, interval: int = 60):
        self.name = name
        self.interval = interval
        self.last_check = 0.0
        
    @abstractmethod
    def check(self, system_state: Dict[str, Any]) -> HealthCheckResult:
        """Perform the health check"""
        pass
        
    def is_due(self) -> bool:
        """Check if health check is due"""
        return time.time() - self.last_check >= self.interval

class SystemResourceHealthCheck(HealthCheck):
    """Health check for system resources"""
    
    def __init__(self, interval: int = 30):
        super().__init__("system_resources", interval)
        
        # Thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 85.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 85.0
        self.disk_critical_threshold = 95.0
        
    def check(self, system_state: Dict[str, Any]) -> HealthCheckResult:
        """Check system resource health"""
        
        self.last_check = time.time()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # GPU memory if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.max_memory_allocated()
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0
            gpu_info = {
                'gpu_memory_used_gb': gpu_memory_used / 1024**3,
                'gpu_memory_total_gb': gpu_memory_total / 1024**3,
                'gpu_memory_percent': gpu_memory_percent
            }
        
        # Determine status
        status = "healthy"
        messages = []
        suggestions = []
        
        if cpu_percent >= self.cpu_critical_threshold:
            status = "unhealthy"
            messages.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            suggestions.append("Reduce computational load or scale resources")
        elif cpu_percent >= self.cpu_warning_threshold:
            status = "warning"
            messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            suggestions.append("Monitor CPU usage trend")
            
        if memory_percent >= self.memory_critical_threshold:
            status = "unhealthy"
            messages.append(f"Critical memory usage: {memory_percent:.1f}%")
            suggestions.append("Free memory or increase system RAM")
        elif memory_percent >= self.memory_warning_threshold:
            status = "warning"
            messages.append(f"High memory usage: {memory_percent:.1f}%")
            suggestions.append("Monitor memory usage patterns")
            
        if disk_percent >= self.disk_critical_threshold:
            status = "unhealthy"
            messages.append(f"Critical disk usage: {disk_percent:.1f}%")
            suggestions.append("Free disk space immediately")
        elif disk_percent >= self.disk_warning_threshold:
            status = "warning"
            messages.append(f"High disk usage: {disk_percent:.1f}%")
            suggestions.append("Clean up old files and logs")
            
        if gpu_info and gpu_info.get('gpu_memory_percent', 0) > 90:
            status = "warning"
            messages.append(f"High GPU memory usage: {gpu_info['gpu_memory_percent']:.1f}%")
            suggestions.append("Optimize GPU memory usage")
            
        message = "; ".join(messages) if messages else "All system resources healthy"
        
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent,
            **gpu_info
        }
        
        return HealthCheckResult(
            check_name=self.name,
            status=status,
            message=message,
            timestamp=self.last_check,
            metrics=metrics,
            suggestions=suggestions
        )

class ModelHealthCheck(HealthCheck):
    """Health check for model training/inference"""
    
    def __init__(self, model, interval: int = 60):
        super().__init__("model_health", interval)
        self.model = model
        
        # Thresholds
        self.gradient_norm_threshold = 10.0
        self.loss_spike_threshold = 2.0  # 2x increase
        self.throughput_drop_threshold = 0.5  # 50% drop
        
        # History for trend analysis
        self.loss_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=50)
        
    def check(self, system_state: Dict[str, Any]) -> HealthCheckResult:
        """Check model health"""
        
        self.last_check = time.time()
        
        status = "healthy"
        messages = []
        suggestions = []
        metrics = {}
        
        # Gradient norm check
        gradient_norm = system_state.get('gradient_norm')
        if gradient_norm is not None:
            metrics['gradient_norm'] = gradient_norm
            if gradient_norm > self.gradient_norm_threshold:
                status = "warning"
                messages.append(f"High gradient norm: {gradient_norm:.3f}")
                suggestions.append("Enable gradient clipping or reduce learning rate")
                
        # Loss spike detection
        current_loss = system_state.get('loss')
        if current_loss is not None:
            self.loss_history.append(current_loss)
            metrics['current_loss'] = current_loss
            
            if len(self.loss_history) > 10:
                recent_avg = sum(list(self.loss_history)[-5:]) / 5
                older_avg = sum(list(self.loss_history)[-10:-5]) / 5
                
                if older_avg > 0 and recent_avg / older_avg > self.loss_spike_threshold:
                    status = "warning"
                    messages.append(f"Loss spike detected: {recent_avg:.4f} (was {older_avg:.4f})")
                    suggestions.append("Check for training instability or data issues")
                    
        # Throughput monitoring
        current_throughput = system_state.get('throughput')
        if current_throughput is not None:
            self.throughput_history.append(current_throughput)
            metrics['current_throughput'] = current_throughput
            
            if len(self.throughput_history) > 10:
                recent_avg = sum(list(self.throughput_history)[-5:]) / 5
                baseline_avg = sum(list(self.throughput_history)[:5]) / 5
                
                if baseline_avg > 0 and recent_avg / baseline_avg < self.throughput_drop_threshold:
                    status = "warning"
                    messages.append(f"Throughput drop: {recent_avg:.1f} (was {baseline_avg:.1f})")
                    suggestions.append("Check for performance degradation or resource constraints")
                    
        # Model-specific checks
        try:
            if hasattr(self.model, 'memory_scheduler'):
                memory_efficiency = system_state.get('memory_efficiency', 0)
                if memory_efficiency < 0.5:  # Less than 50% efficiency
                    status = "warning" 
                    messages.append(f"Low memory efficiency: {memory_efficiency:.2f}")
                    suggestions.append("Optimize memory scheduling strategy")
                    
            if hasattr(self.model, 'reversible_layers'):
                # Check reversible layer health
                for i, layer in enumerate(self.model.reversible_layers):
                    if hasattr(layer, 'coupling_residual') and layer.coupling_residual > 1e-3:
                        status = "warning"
                        messages.append(f"High coupling residual in layer {i}: {layer.coupling_residual:.6f}")
                        suggestions.append("Check reversible layer numerical stability")
                        
        except Exception as e:
            messages.append(f"Model inspection error: {str(e)}")
            
        message = "; ".join(messages) if messages else "Model health normal"
        
        return HealthCheckResult(
            check_name=self.name,
            status=status,
            message=message,
            timestamp=self.last_check,
            metrics=metrics,
            suggestions=suggestions
        )

class DataHealthCheck(HealthCheck):
    """Health check for data pipeline"""
    
    def __init__(self, interval: int = 120):
        super().__init__("data_health", interval)
        
    def check(self, system_state: Dict[str, Any]) -> HealthCheckResult:
        """Check data pipeline health"""
        
        self.last_check = time.time()
        
        status = "healthy"
        messages = []
        suggestions = []
        metrics = {}
        
        # Data loading speed
        data_loading_time = system_state.get('data_loading_time')
        if data_loading_time is not None:
            metrics['data_loading_time'] = data_loading_time
            if data_loading_time > 5.0:  # More than 5 seconds
                status = "warning"
                messages.append(f"Slow data loading: {data_loading_time:.2f}s")
                suggestions.append("Optimize data loading pipeline or increase workers")
                
        # Batch processing rate
        batch_processing_rate = system_state.get('batch_processing_rate')
        if batch_processing_rate is not None:
            metrics['batch_processing_rate'] = batch_processing_rate
            if batch_processing_rate < 1.0:  # Less than 1 batch/second
                status = "warning"
                messages.append(f"Low batch processing rate: {batch_processing_rate:.2f} batches/s")
                suggestions.append("Optimize batch processing or reduce batch size")
                
        # Data quality checks
        nan_count = system_state.get('nan_count', 0)
        if nan_count > 0:
            status = "warning"
            messages.append(f"NaN values detected: {nan_count}")
            suggestions.append("Check data preprocessing and handle missing values")
            
        message = "; ".join(messages) if messages else "Data pipeline healthy"
        
        return HealthCheckResult(
            check_name=self.name,
            status=status,
            message=message,
            timestamp=self.last_check,
            metrics=metrics,
            suggestions=suggestions
        )

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
        
        # Alert thresholds
        self.alert_cooldown = self.config.get('alert_cooldown', 300)  # 5 minutes
        self.max_alerts_per_hour = self.config.get('max_alerts_per_hour', 20)
        
        # Alert history for rate limiting
        self.alert_timestamps = deque(maxlen=100)
        
        self.logger = logging.getLogger(__name__)
        
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler"""
        self.alert_handlers.append(handler)
        
    def create_alert(self,
                    severity: AlertSeverity,
                    title: str,
                    message: str,
                    source: str,
                    metrics: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert"""
        
        # Rate limiting
        current_time = time.time()
        recent_alerts = [t for t in self.alert_timestamps if current_time - t < 3600]  # Last hour
        
        if len(recent_alerts) >= self.max_alerts_per_hour:
            self.logger.warning("Alert rate limit exceeded, dropping alert")
            return None
            
        # Create alert
        alert = Alert(
            id=f"alert_{int(current_time * 1000)}",
            severity=severity,
            title=title,
            message=message,
            timestamp=current_time,
            source=source,
            metrics=metrics or {}
        )
        
        self.alerts.append(alert)
        self.alert_timestamps.append(current_time)
        
        # Trigger handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
                
        return alert
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [a for a in self.alerts if not a.resolved]
        
    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical alerts"""
        return [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved]

class EmailAlertHandler:
    """Email alert handler"""
    
    def __init__(self, smtp_config: Dict[str, str]):
        self.smtp_config = smtp_config
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, alert: Alert) -> None:
        """Send email alert"""
        
        # Only send emails for warning and above
        if alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            try:
                self._send_email(alert)
            except Exception as e:
                self.logger.error(f"Failed to send email alert: {e}")
                
    def _send_email(self, alert: Alert) -> None:
        """Send email notification"""
        
        msg = MIMEMultipart()
        msg['From'] = self.smtp_config['from_email']
        msg['To'] = self.smtp_config['to_email']
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"""
Alert Details:
- Severity: {alert.severity.value.upper()}
- Source: {alert.source}
- Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}
- Message: {alert.message}

Metrics:
{json.dumps(alert.metrics, indent=2)}

Alert ID: {alert.id}
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port'])
        if self.smtp_config.get('use_tls'):
            server.starttls()
        if self.smtp_config.get('username'):
            server.login(self.smtp_config['username'], self.smtp_config['password'])
            
        server.send_message(msg)
        server.quit()

class ComprehensiveMonitor:
    """Comprehensive monitoring system"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        self.model = model
        self.config = config or {}
        
        # Components
        self.health_checks: List[HealthCheck] = []
        self.alert_manager = AlertManager(config.get('alerting', {}))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.system_state: Dict[str, Any] = {}
        
        # Health check results history
        self.health_history: Dict[str, List[HealthCheckResult]] = defaultdict(list)
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup alert handlers
        self._setup_alert_handlers()
        
        self.logger = logging.getLogger(__name__)
        
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks"""
        
        # System resources
        self.health_checks.append(SystemResourceHealthCheck(interval=30))
        
        # Model health
        self.health_checks.append(ModelHealthCheck(self.model, interval=60))
        
        # Data health
        self.health_checks.append(DataHealthCheck(interval=120))
        
    def _setup_alert_handlers(self) -> None:
        """Setup alert handlers"""
        
        # File logging handler
        def file_alert_handler(alert: Alert):
            log_entry = {
                'id': alert.id,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'source': alert.source,
                'metrics': alert.metrics
            }
            self.logger.warning(f"ALERT: {json.dumps(log_entry)}")
            
        self.alert_manager.add_alert_handler(file_alert_handler)
        
        # Email handler if configured
        email_config = self.config.get('email_alerts')
        if email_config:
            email_handler = EmailAlertHandler(email_config)
            self.alert_manager.add_alert_handler(email_handler)
            
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a custom health check"""
        self.health_checks.append(health_check)
        
    def update_system_state(self, **kwargs) -> None:
        """Update system state with new metrics"""
        self.system_state.update(kwargs)
        
    def start_monitoring(self) -> None:
        """Start comprehensive monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Started comprehensive monitoring")
        
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
        self.logger.info("Stopped comprehensive monitoring")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Run due health checks
                for health_check in self.health_checks:
                    if health_check.is_due():
                        result = health_check.check(self.system_state)
                        self.health_history[health_check.name].append(result)
                        
                        # Keep only last 1000 results per check
                        if len(self.health_history[health_check.name]) > 1000:
                            self.health_history[health_check.name] = self.health_history[health_check.name][-1000:]
                            
                        # Generate alerts based on health check results
                        self._process_health_check_result(result)
                        
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.logger.error(traceback.format_exc())
                
    def _process_health_check_result(self, result: HealthCheckResult) -> None:
        """Process health check result and generate alerts if needed"""
        
        if result.status == "unhealthy":
            self.alert_manager.create_alert(
                severity=AlertSeverity.ERROR,
                title=f"Health Check Failed: {result.check_name}",
                message=result.message,
                source=f"health_check_{result.check_name}",
                metrics=result.metrics
            )
        elif result.status == "warning":
            self.alert_manager.create_alert(
                severity=AlertSeverity.WARNING,
                title=f"Health Warning: {result.check_name}",
                message=result.message,
                source=f"health_check_{result.check_name}",
                metrics=result.metrics
            )
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Get latest health check results
        latest_health = {}
        for check_name, results in self.health_history.items():
            if results:
                latest_health[check_name] = {
                    'status': results[-1].status,
                    'message': results[-1].message,
                    'timestamp': results[-1].timestamp,
                    'metrics': results[-1].metrics
                }
                
        # Overall system health
        unhealthy_checks = [name for name, result in latest_health.items() if result['status'] == 'unhealthy']
        warning_checks = [name for name, result in latest_health.items() if result['status'] == 'warning']
        
        if unhealthy_checks:
            overall_status = 'unhealthy'
        elif warning_checks:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
            
        return {
            'overall_status': overall_status,
            'health_checks': latest_health,
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'critical_alerts': len(self.alert_manager.get_critical_alerts()),
            'monitoring_active': self.is_monitoring,
            'last_update': time.time()
        }
        
    def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        
        status = self.get_system_status()
        
        report_lines = [
            "# System Health Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Status: {status['overall_status'].upper()}",
            "",
            f"Active Alerts: {status['active_alerts']}",
            f"Critical Alerts: {status['critical_alerts']}",
            "",
            "## Health Check Details"
        ]
        
        for check_name, result in status['health_checks'].items():
            report_lines.extend([
                f"### {check_name.title()}",
                f"Status: {result['status'].upper()}",
                f"Message: {result['message']}",
                f"Last Check: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))}",
                ""
            ])
            
            if result['metrics']:
                report_lines.append("Metrics:")
                for metric, value in result['metrics'].items():
                    report_lines.append(f"- {metric}: {value}")
                report_lines.append("")
                
        # Recent alerts
        recent_alerts = [a for a in self.alert_manager.get_active_alerts() if time.time() - a.timestamp < 3600]
        if recent_alerts:
            report_lines.extend([
                "## Recent Alerts (Last Hour)",
                ""
            ])
            
            for alert in recent_alerts:
                report_lines.extend([
                    f"### [{alert.severity.value.upper()}] {alert.title}",
                    f"Message: {alert.message}",
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}",
                    f"Source: {alert.source}",
                    ""
                ])
                
        return "\n".join(report_lines)