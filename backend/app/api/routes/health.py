from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import logging
from typing import Any

from app.api.dependencies import (
    get_service_manager, 
    get_analytics_service, 
    get_market_service,
    get_initialized_service_manager
)
from app.core.services.service_manager import CorrectedServiceManager
from app.core.services.analytics_service import CorrectedAnalytics
from app.services.enhanced_market_data_nifty100 import EnhancedMarketDataService
from app.core.availability import ENHANCED_MARKET_DATA_AVAILABLE

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/webhook/status")
async def webhook_status(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """Webhook status endpoint"""
    notification_service = service_manager.get_notification_service()
    webhook_handler = notification_service.get_webhook_handler() if notification_service else None
    order_engine = notification_service.get_order_engine() if notification_service else None
    
    return {
        "status": "active" if webhook_handler else "inactive",
        "interactive_trading": notification_service.is_interactive_trading_active() if notification_service else False,
        "order_engine": service_manager.system_health.get("order_execution", False),
        "webhook_configured": webhook_handler is not None,
        "news_intelligence": service_manager.system_health.get("news_intelligence", False),
        "news_monitoring": service_manager.news_monitoring_active,
        "news_signal_integration": service_manager.system_health.get("news_signal_integration", False),
        "production_mode": True,
        "demo_signals_disabled": True,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/api/system/status")
async def get_system_status(
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager),
    market_service: EnhancedMarketDataService = Depends(get_market_service)
):
    """Get detailed system status including fixed news-signal integration"""
    market_status = "UNKNOWN"
    trading_mode = "UNKNOWN"
    try:
        health = await market_service.get_service_health()
        market_status = health.get('market_status', 'UNKNOWN')
        trading_mode = health.get('trading_mode', 'UNKNOWN')
    except Exception:
        pass
    
    notification_service = service_manager.get_notification_service()
    order_engine = notification_service.get_order_engine() if notification_service else None
    order_engine_status = {}
    if order_engine:
        try:
            order_engine_status = await order_engine.get_connection_status()
        except Exception as e:
            logger.debug(f"Failed to get order engine status: {e}")
    
    return {
        "architecture": "enhanced_v5.1_interactive_trading_news_intelligence_production_fixed",
        "version": "5.1.0-production-fixed",
        "production_mode": True,
        "demo_signals_disabled": True,
        "initialized": service_manager.is_initialized(),
        "initialization_error": service_manager.get_initialization_error(),
        "system_health": service_manager.get_system_health(),
        "signal_generation_active": service_manager.signal_generation_active,
        "premarket_analysis_active": service_manager.premarket_analysis_active,
        "priority_trading_active": service_manager.priority_trading_active,
        "regime_monitoring_active": service_manager.regime_monitoring_active,
        "interactive_trading_active": service_manager.interactive_trading_active,
        "news_monitoring_active": service_manager.news_monitoring_active,
        "news_signal_integration_active": service_manager.system_health.get("news_signal_integration", False),
        "active_connections": len(service_manager.active_connections),
        "market_status": market_status,
        "trading_mode": trading_mode,
        "current_regime": str(service_manager.current_regime),
        "regime_confidence": service_manager.regime_confidence,
        "order_engine": order_engine_status,
        "features": {
            "enhanced_telegram": getattr(service_manager, 'ENHANCED_TELEGRAM_AVAILABLE', False),
            "order_execution": getattr(service_manager, 'ZERODHA_ORDER_ENGINE_AVAILABLE', False),
            "webhook_handler": getattr(service_manager, 'WEBHOOK_HANDLER_AVAILABLE', False),
            "enhanced_market_data": getattr(service_manager, 'ENHANCED_MARKET_DATA_AVAILABLE', False),
            "configuration": getattr(service_manager, 'CONFIG_AVAILABLE', False),
            "news_intelligence": getattr(service_manager, 'NEWS_INTELLIGENCE_AVAILABLE', False),
            "news_signal_integration": service_manager.system_health.get("news_signal_integration", False),
            "nifty100_universe": True,
            "premarket_analysis": True,
            "priority_trading": True,
            "regime_detection": getattr(service_manager, 'REGIME_DETECTOR_AVAILABLE', False),
            "backtesting": getattr(service_manager, 'BACKTEST_ENGINE_AVAILABLE', False),
            "institutional_grade": True,
            "production_only": True,
            "demo_signals": False
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/health")
async def health_check(
    service_manager: CorrectedServiceManager = Depends(get_service_manager),
    analytics_service: CorrectedAnalytics = Depends(get_analytics_service),
    market_service: EnhancedMarketDataService = Depends(get_market_service)
):
    """Enhanced health check with fixed news-signal integration status"""
    try:
        performance = analytics_service.get_performance_summary()
        market_health = {}
        try:
            market_health = await market_service.get_service_health()
        except Exception as e:
            logger.debug(f"Failed to get market health: {e}")
        
        notification_service = service_manager.get_notification_service()
        order_engine = notification_service.get_order_engine() if notification_service else None
        order_health = {}
        if order_engine:
            try:
                order_health = await order_engine.get_connection_status()
            except Exception as e:
                logger.debug(f"Failed to get order engine health: {e}")
        
        return {
            "status": "healthy" if service_manager.is_initialized() else "degraded",
            "architecture": "enhanced_v5.1_interactive_trading_news_intelligence_production_fixed",
            "production_mode": True,
            "demo_signals_disabled": True,
            "initialization_error": service_manager.get_initialization_error(),
            "system_health": service_manager.get_system_health(),
            "connections": len(service_manager.active_connections),
            "signal_generation_active": service_manager.signal_generation_active,
            "interactive_trading_active": service_manager.interactive_trading_active,
            "news_monitoring_active": service_manager.news_monitoring_active,
            "news_signal_integration_active": service_manager.system_health.get("news_signal_integration", False),
            "market_health": market_health,
            "order_engine_health": order_health,
            "performance": performance,
            "current_regime": str(service_manager.current_regime),
            "regime_confidence": service_manager.regime_confidence,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "production_mode": True,
            "demo_signals_disabled": True,
            "timestamp": datetime.now().isoformat()
        }

@router.get("/api/system/tasks")
async def get_task_health(
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager)
):
    """Get task health status from TaskManager"""
    try:
        task_health = service_manager.get_task_health()
        return {
            "task_health": task_health["task_health"],
            "market_hours": task_health["market_hours"],
            "active_tasks": task_health["active_tasks"],
            "uptime": service_manager.task_manager.get_uptime(),
            "restart_count": service_manager.task_manager.restart_count,
            "cache_stats": task_health["cache_stats"],
            "resource_usage": task_health["resource_usage"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Task health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 