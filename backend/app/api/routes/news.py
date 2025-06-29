from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import logging

from app.api.dependencies import (
    get_service_manager, 
    get_analytics_service,
    get_news_service
)
from app.core.services.service_manager import CorrectedServiceManager
from app.core.services.analytics_service import CorrectedAnalytics
from app.services.enhanced_news_intelligence import EnhancedNewsIntelligenceSystem

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/news/status")
async def get_news_status(
    service_manager: CorrectedServiceManager = Depends(get_service_manager),
    analytics_service: CorrectedAnalytics = Depends(get_analytics_service)
):
    """Get comprehensive news intelligence status"""
    try:
        stats = analytics_service.get_daily_stats()
        integration_stats = {}
        # Note: news_signal_integration is not available in current service manager
        # Using news_intelligence status instead
        return {
            "news_intelligence_active": service_manager.system_health.get("news_intelligence", False),
            "news_monitoring_active": service_manager.news_monitoring_active,
            "news_signal_integration_active": service_manager.system_health.get("news_signal_integration", False),
            "articles_processed_today": stats.get("news_articles_processed", 0),
            "breaking_news_alerts": stats.get("breaking_news_alerts", 0),
            "news_signals_generated": stats.get("news_signals_generated", 0),
            "news_triggered_signals": stats.get("news_triggered_signals", 0),
            "enhanced_ml_signals": stats.get("enhanced_ml_signals", 0),
            "avg_sentiment": stats.get("avg_news_sentiment", 0.0),
            "sources_active": stats.get("news_sources_active", 0),
            "last_update": stats.get("last_news_update"),
            "integration_stats": integration_stats,
            "system_health_flags": {
                "news_intelligence": service_manager.system_health.get("news_intelligence", False),
                "news_monitoring": service_manager.system_health.get("news_monitoring", False),
                "breaking_news_alerts": service_manager.system_health.get("breaking_news_alerts", False),
                "news_signal_integration": service_manager.system_health.get("news_signal_integration", False)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/news/refresh")
async def refresh_news(
    service_manager: CorrectedServiceManager = Depends(get_service_manager),
    news_service: EnhancedNewsIntelligenceSystem = Depends(get_news_service)
):
    """Manually refresh news intelligence"""
    try:
        news_intel = await news_service.get_comprehensive_news_intelligence(
            lookback_hours=6
        )
        return {
            "success": True,
            "articles_analyzed": news_intel.get("total_articles_analyzed", 0),
            "overall_sentiment": news_intel.get("overall_sentiment", 0.0),
            "sources_used": len(news_intel.get("news_sources_used", [])),
            "market_events": len(news_intel.get("market_events", [])),
            "news_signals": len(news_intel.get("news_signals", [])),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ News refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news/integration/stats")
async def get_news_integration_stats(
    service_manager: CorrectedServiceManager = Depends(get_service_manager),
    analytics_service: CorrectedAnalytics = Depends(get_analytics_service)
):
    """Get detailed news-signal integration statistics"""
    try:
        # Note: news_signal_integration is not available in current service manager
        # Using news_intelligence status instead
        daily_stats = analytics_service.get_daily_stats()
        return {
            "integration_active": service_manager.system_health.get("news_signal_integration", False),
            "integration_stats": {
                "news_intelligence_available": service_manager.news_intelligence is not None,
                "news_monitoring_active": service_manager.news_monitoring_active,
                "articles_processed": daily_stats.get("news_articles_processed", 0)
            },
            "daily_news_stats": {
                "news_triggered_signals": daily_stats.get("news_triggered_signals", 0),
                "enhanced_ml_signals": daily_stats.get("enhanced_ml_signals", 0),
                "breaking_news_alerts": daily_stats.get("breaking_news_alerts", 0),
                "news_articles_processed": daily_stats.get("news_articles_processed", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/news/trigger-analysis")
async def trigger_news_analysis(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """Manually trigger news analysis and signal generation"""
    try:
        if service_manager.news_intelligence and service_manager.system_health.get("news_intelligence", False):
            # Trigger news analysis using news_intelligence
            news_data = await service_manager.news_intelligence.get_comprehensive_news_intelligence(
                lookback_hours=6
            )
            return {
                "success": True,
                "message": "News analysis triggered successfully",
                "articles_analyzed": news_data.get("total_articles_analyzed", 0),
                "overall_sentiment": news_data.get("overall_sentiment", {}).get("adjusted_sentiment", 0.0),
                "sources_used": len(news_data.get("news_sources_used", [])),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="News intelligence not available")
    except Exception as e:
        logger.error(f"❌ Manual news analysis trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/news-signal-flow")
async def debug_news_signal_flow(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """Debug endpoint to check news-signal integration flow"""
    try:
        debug_info = {
            "news_intelligence_initialized": service_manager.news_intelligence is not None,
            "signal_generator_initialized": service_manager.signal_generator is not None,
            "news_signal_integration_initialized": False,  # Not available in current service manager
            "news_monitoring_active": service_manager.news_monitoring_active,
            "signal_generation_active": service_manager.signal_generation_active,
            "interactive_trading_active": service_manager.interactive_trading_active,
            "system_health": service_manager.system_health,
            "market_status": str(service_manager.current_market_status),
            "current_regime": str(service_manager.current_regime),
            "background_tasks": {
                "signal_generation_task": service_manager.signal_generation_task is not None,
                "market_monitor_task": service_manager.market_monitor_task is not None,
                "regime_monitor_task": service_manager.regime_monitor_task is not None,
                "news_integration_task": service_manager.news_integration_task is not None
            }
        }
        if service_manager.news_intelligence:
            debug_info["news_intelligence_details"] = {
                "news_intelligence_available": True,
                "last_news_check": str(service_manager.news_intelligence.last_update) if hasattr(service_manager.news_intelligence, 'last_update') else "Unknown"
            }
        return debug_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 