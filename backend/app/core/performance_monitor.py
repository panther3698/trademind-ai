# ================================================================
# Performance Monitoring System for TradeMind AI
# ================================================================

"""
Performance Monitoring System
Tracks system efficiency, timing metrics, success rates, and system health

Features:
- Timing metrics for key operations
- Success rate tracking
- System health monitoring
- Performance alerting
- Metrics export for monitoring tools
"""

import time
import psutil
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
import json
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    TIMING = "timing"
    SUCCESS_RATE = "success_rate"
    SYSTEM_HEALTH = "system_health"
    COUNTER = "counter"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Performance alert"""
    severity: str  # "info", "warning", "error", "critical"
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        
        # Metrics storage
        self.metrics: List[PerformanceMetric] = []
        self.success_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.timing_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.system_health: Dict[str, Any] = {}
        
        # Alerts
        self.alerts: List[Alert] = []
        self.alert_thresholds = {
            "signal_generation_time": 5.0,  # seconds
            "news_analysis_time": 10.0,     # seconds
            "order_execution_time": 3.0,    # seconds
            "api_response_time": 2.0,       # seconds
            "memory_usage": 80.0,           # percentage
            "cpu_usage": 90.0,              # percentage
            "success_rate": 0.8,            # 80%
        }
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("✅ Performance Monitor initialized")
    
    def add_metric(self, name: str, value: float, metric_type: MetricType, 
                   tags: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a performance metric"""
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.metrics.append(metric)
            
            # Keep only recent metrics
            if len(self.metrics) > self.max_history_size:
                self.metrics.pop(0)
            
            # Store in specific collections
            if metric_type == MetricType.TIMING:
                self.timing_metrics[name].append(value)
            elif metric_type == MetricType.SUCCESS_RATE:
                self.success_rates[name].append(value)
            
            # Check for alerts
            self._check_alerts(metric)
    
    def _check_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts"""
        threshold = self.alert_thresholds.get(metric.name)
        if threshold is None:
            return
        
        is_alert = False
        if metric.metric_type == MetricType.TIMING:
            is_alert = metric.value > threshold
        elif metric.metric_type == MetricType.SUCCESS_RATE:
            is_alert = metric.value < threshold
        elif metric.metric_type == MetricType.SYSTEM_HEALTH:
            if "memory" in metric.name.lower() or "cpu" in metric.name.lower():
                is_alert = metric.value > threshold
        
        if is_alert:
            severity = "warning" if metric.value < threshold * 1.5 else "error"
            alert = Alert(
                severity=severity,
                message=f"{metric.name} exceeded threshold: {metric.value:.2f} > {threshold:.2f}",
                metric_name=metric.name,
                threshold=threshold,
                current_value=metric.value,
                timestamp=datetime.now()
            )
            self.alerts.append(alert)
            logger.warning(f"🚨 Performance Alert: {alert.message}")
    
    @contextmanager
    def timing(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.add_metric(
                name=f"{operation_name}_time",
                value=duration,
                metric_type=MetricType.TIMING,
                tags=tags or {}
            )
    
    @asynccontextmanager
    async def async_timing(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Async context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.add_metric(
                name=f"{operation_name}_time",
                value=duration,
                metric_type=MetricType.TIMING,
                tags=tags or {}
            )
    
    def track_success(self, operation_name: str, success: bool, tags: Optional[Dict[str, str]] = None):
        """Track success/failure of operations"""
        self.success_rates[operation_name].append(1.0 if success else 0.0)
        
        # Calculate success rate
        if self.success_rates[operation_name]:
            success_rate = sum(self.success_rates[operation_name]) / len(self.success_rates[operation_name])
            self.add_metric(
                name=f"{operation_name}_success_rate",
                value=success_rate,
                metric_type=MetricType.SUCCESS_RATE,
                tags=tags or {}
            )
    
    def increment_counter(self, counter_name: str, increment: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.add_metric(
            name=counter_name,
            value=increment,
            metric_type=MetricType.COUNTER,
            tags=tags or {}
        )
    
    async def update_system_health(self):
        """Update system health metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.add_metric(
                name="memory_usage_percent",
                value=memory.percent,
                metric_type=MetricType.SYSTEM_HEALTH,
                tags={"type": "memory"}
            )
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.add_metric(
                name="cpu_usage_percent",
                value=cpu_percent,
                metric_type=MetricType.SYSTEM_HEALTH,
                tags={"type": "cpu"}
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.add_metric(
                name="disk_usage_percent",
                value=(disk.used / disk.total) * 100,
                metric_type=MetricType.SYSTEM_HEALTH,
                tags={"type": "disk"}
            )
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.add_metric(
                name="network_bytes_sent",
                value=net_io.bytes_sent,
                metric_type=MetricType.SYSTEM_HEALTH,
                tags={"type": "network"}
            )
            self.add_metric(
                name="network_bytes_recv",
                value=net_io.bytes_recv,
                metric_type=MetricType.SYSTEM_HEALTH,
                tags={"type": "network"}
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to update system health: {e}")
    
    async def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("✅ Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                await self.update_system_health()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            # Calculate averages for timing metrics
            timing_averages = {}
            for name, values in self.timing_metrics.items():
                if values:
                    timing_averages[name] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
            
            # Calculate success rates
            success_rate_summary = {}
            for name, values in self.success_rates.items():
                if values:
                    success_rate_summary[name] = {
                        "current_rate": sum(values) / len(values),
                        "total_operations": len(values)
                    }
            
            # Get recent alerts
            recent_alerts = [alert for alert in self.alerts 
                           if alert.timestamp > datetime.now() - timedelta(hours=24)]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "timing_metrics": timing_averages,
                "success_rates": success_rate_summary,
                "system_health": self.system_health,
                "recent_alerts": [
                    {
                        "severity": alert.severity,
                        "message": alert.message,
                        "metric_name": alert.metric_name,
                        "threshold": alert.threshold,
                        "current_value": alert.current_value,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved
                    }
                    for alert in recent_alerts
                ],
                "total_metrics": len(self.metrics),
                "active_alerts": len([a for a in self.alerts if not a.resolved])
            }
    
    def export_metrics_for_grafana(self) -> List[Dict[str, Any]]:
        """Export metrics in Grafana-compatible format"""
        with self._lock:
            grafana_metrics = []
            
            for metric in self.metrics[-1000:]:  # Last 1000 metrics
                grafana_metric = {
                    "time": int(metric.timestamp.timestamp() * 1000),  # Unix timestamp in ms
                    "value": metric.value,
                    "metric": metric.name,
                    "type": metric.metric_type.value
                }
                
                # Add tags
                if metric.tags:
                    grafana_metric.update(metric.tags)
                
                grafana_metrics.append(grafana_metric)
            
            return grafana_metrics
    
    def clear_old_metrics(self, older_than_hours: int = 24):
        """Clear metrics older than specified hours"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            logger.info(f"🧹 Cleared metrics older than {older_than_hours} hours")

# Global performance monitor instance
performance_monitor = PerformanceMonitor() 