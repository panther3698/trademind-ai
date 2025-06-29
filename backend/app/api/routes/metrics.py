# ================================================================
# Performance Metrics API Routes
# ================================================================

"""
Performance Metrics API Routes
Provides comprehensive system monitoring and performance metrics

Endpoints:
- GET /api/system/metrics - Comprehensive metrics dashboard
- GET /api/system/metrics/grafana - Grafana-compatible metrics export
- GET /api/system/metrics/alerts - Recent performance alerts
- GET /api/system/metrics/health - System health overview
- POST /api/system/metrics/clear - Clear old metrics
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta

from app.core.performance_monitor import performance_monitor
from app.api.dependencies import get_service_manager
from app.core.services.service_manager import CorrectedServiceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["Performance Metrics"])

@router.get("/metrics")
async def get_system_metrics(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """
    Get comprehensive system performance metrics
    
    Returns:
        - Timing metrics for key operations
        - Success rates for various operations
        - System health metrics
        - Recent performance alerts
        - Overall system status
    """
    try:
        # Get performance metrics summary
        metrics_summary = performance_monitor.get_metrics_summary()
        
        # Get system health from service manager
        system_health = service_manager.get_system_health()
        
        # Get additional system information
        system_info = {
            "signal_generation_active": service_manager.signal_service.signal_generation_active,
            "interactive_trading_active": service_manager.interactive_trading_active,
            "news_monitoring_active": service_manager.news_monitoring_active,
            "regime_monitoring_active": service_manager.regime_monitoring_active,
            "current_market_status": service_manager.current_market_status,
            "current_trading_mode": service_manager.current_trading_mode,
            "current_regime": str(service_manager.current_regime),
            "regime_confidence": service_manager.regime_confidence
        }
        
        # Performance targets and current status
        performance_targets = {
            "signal_generation_time": {
                "target": "< 5 seconds",
                "current_avg": metrics_summary.get("timing_metrics", {}).get("signal_generation_time", {}).get("avg", 0),
                "status": "✅" if metrics_summary.get("timing_metrics", {}).get("signal_generation_time", {}).get("avg", 0) < 5 else "⚠️"
            },
            "news_analysis_time": {
                "target": "< 10 seconds",
                "current_avg": metrics_summary.get("timing_metrics", {}).get("news_analysis_time", {}).get("avg", 0),
                "status": "✅" if metrics_summary.get("timing_metrics", {}).get("news_analysis_time", {}).get("avg", 0) < 10 else "⚠️"
            },
            "order_execution_time": {
                "target": "< 3 seconds",
                "current_avg": metrics_summary.get("timing_metrics", {}).get("order_execution_time", {}).get("avg", 0),
                "status": "✅" if metrics_summary.get("timing_metrics", {}).get("order_execution_time", {}).get("avg", 0) < 3 else "⚠️"
            },
            "api_response_time": {
                "target": "< 2 seconds",
                "current_avg": metrics_summary.get("timing_metrics", {}).get("api_response_time", {}).get("avg", 0),
                "status": "✅" if metrics_summary.get("timing_metrics", {}).get("api_response_time", {}).get("avg", 0) < 2 else "⚠️"
            }
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "system_health": system_health,
            "performance_metrics": metrics_summary,
            "performance_targets": performance_targets,
            "monitoring_status": {
                "active": performance_monitor.monitoring_active,
                "total_metrics_collected": metrics_summary.get("total_metrics", 0),
                "active_alerts": metrics_summary.get("active_alerts", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/grafana")
async def get_grafana_metrics(
    hours: Optional[int] = Query(24, description="Number of hours of data to export")
):
    """
    Export metrics in Grafana-compatible format
    
    Args:
        hours: Number of hours of data to export (default: 24)
    
    Returns:
        List of metrics in Grafana time series format
    """
    try:
        # Get Grafana-compatible metrics
        grafana_metrics = performance_monitor.export_metrics_for_grafana()
        
        # Filter by time if specified
        if hours:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            grafana_metrics = [
                m for m in grafana_metrics 
                if m["time"] > int(cutoff_time.timestamp() * 1000)
            ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics_count": len(grafana_metrics),
            "time_range_hours": hours,
            "metrics": grafana_metrics
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to export Grafana metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/alerts")
async def get_performance_alerts(
    severity: Optional[str] = Query(None, description="Filter by alert severity (info, warning, error, critical)"),
    resolved: Optional[bool] = Query(False, description="Include resolved alerts")
):
    """
    Get recent performance alerts
    
    Args:
        severity: Filter by alert severity
        resolved: Include resolved alerts
    
    Returns:
        List of performance alerts
    """
    try:
        alerts_summary = performance_monitor.get_metrics_summary()
        alerts = alerts_summary.get("recent_alerts", [])
        
        # Apply filters
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        if not resolved:
            alerts = [a for a in alerts if not a["resolved"]]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(alerts),
            "alerts": alerts
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get performance alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/health")
async def get_system_health_overview(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """
    Get system health overview with performance indicators
    
    Returns:
        System health status with performance indicators
    """
    try:
        metrics_summary = performance_monitor.get_metrics_summary()
        system_health = service_manager.get_system_health()
        
        # Calculate overall health score
        health_score = 100
        
        # Check timing metrics
        timing_metrics = metrics_summary.get("timing_metrics", {})
        for metric_name, threshold in performance_monitor.alert_thresholds.items():
            if metric_name in timing_metrics:
                avg_time = timing_metrics[metric_name].get("avg", 0)
                if avg_time > threshold:
                    health_score -= 10
        
        # Check success rates
        success_rates = metrics_summary.get("success_rates", {})
        for rate_name, rate_data in success_rates.items():
            current_rate = rate_data.get("current_rate", 1.0)
            if current_rate < 0.8:  # 80% threshold
                health_score -= 15
        
        # Check active alerts
        active_alerts = metrics_summary.get("active_alerts", 0)
        health_score -= active_alerts * 5
        
        health_score = max(0, health_score)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health_score": health_score,
            "health_status": "🟢 Excellent" if health_score >= 90 else "🟡 Good" if health_score >= 70 else "🟠 Fair" if health_score >= 50 else "🔴 Poor",
            "system_health": system_health,
            "performance_indicators": {
                "active_alerts": active_alerts,
                "timing_violations": len([m for m in timing_metrics.values() if m.get("avg", 0) > 5]),
                "low_success_rates": len([r for r in success_rates.values() if r.get("current_rate", 1.0) < 0.8])
            },
            "recommendations": _generate_health_recommendations(health_score, metrics_summary)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get system health overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics/clear")
async def clear_old_metrics(
    hours: int = Query(24, description="Clear metrics older than this many hours")
):
    """
    Clear old performance metrics
    
    Args:
        hours: Clear metrics older than this many hours
    
    Returns:
        Confirmation of cleared metrics
    """
    try:
        old_count = len(performance_monitor.metrics)
        performance_monitor.clear_old_metrics(hours)
        new_count = len(performance_monitor.metrics)
        cleared_count = old_count - new_count
        
        return {
            "timestamp": datetime.now().isoformat(),
            "message": f"Cleared {cleared_count} metrics older than {hours} hours",
            "metrics_remaining": new_count,
            "cleared_count": cleared_count
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear old metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_health_recommendations(health_score: int, metrics_summary: Dict[str, Any]) -> List[str]:
    """Generate health recommendations based on metrics"""
    recommendations = []
    
    if health_score < 90:
        timing_metrics = metrics_summary.get("timing_metrics", {})
        success_rates = metrics_summary.get("success_rates", {})
        active_alerts = metrics_summary.get("active_alerts", 0)
        
        # Check for slow operations
        for metric_name, data in timing_metrics.items():
            avg_time = data.get("avg", 0)
            if avg_time > 5:
                recommendations.append(f"Optimize {metric_name} (avg: {avg_time:.2f}s)")
        
        # Check for low success rates
        for rate_name, rate_data in success_rates.items():
            current_rate = rate_data.get("current_rate", 1.0)
            if current_rate < 0.8:
                recommendations.append(f"Investigate {rate_name} low success rate ({current_rate:.1%})")
        
        # Check for alerts
        if active_alerts > 0:
            recommendations.append(f"Address {active_alerts} active performance alerts")
    
    if not recommendations:
        recommendations.append("System performance is optimal")
    
    return recommendations 