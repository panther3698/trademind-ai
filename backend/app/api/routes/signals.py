from fastapi import APIRouter, HTTPException, Depends, Request
from datetime import datetime
import logging

from app.api.dependencies import (
    get_service_manager, 
    get_analytics_service,
    get_signal_service,
    get_news_service
)
from app.core.services.service_manager import CorrectedServiceManager
from app.core.services.analytics_service import CorrectedAnalytics
from app.core.services.signal_service import SignalService
from app.services.enhanced_news_intelligence import EnhancedNewsIntelligenceSystem
from utils.profiling import profile_timing

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/analytics/performance")
async def get_performance(
    service_manager: CorrectedServiceManager = Depends(get_service_manager),
    analytics_service: CorrectedAnalytics = Depends(get_analytics_service),
    news_service: EnhancedNewsIntelligenceSystem = Depends(get_news_service)
):
    """Get comprehensive performance analytics including fixed news intelligence"""
    try:
        performance = analytics_service.get_performance_summary()
        
        # Add interactive trading specific metrics
        performance["interactive_trading"] = {
            "active": service_manager.interactive_trading_active,
            "signals_approved": performance["daily"]["signals_approved"],
            "signals_rejected": performance["daily"]["signals_rejected"],
            "orders_executed": performance["daily"]["orders_executed"],
            "approval_rate": performance["daily"]["approval_rate"],
            "order_success_rate": performance["daily"]["order_success_rate"],
            "trading_pnl": performance["daily"]["total_trading_pnl"]
        }
        
        # Add news intelligence metrics
        performance["news_intelligence"] = {
            "active": service_manager.system_health.get("news_intelligence", False),
            "integration_active": service_manager.system_health.get("news_signal_integration", False),  # FIXED
            "articles_processed": performance["daily"]["news_articles_processed"],
            "breaking_news_alerts": performance["daily"]["breaking_news_alerts"],
            "news_signals_generated": performance["daily"]["news_signals_generated"],
            "news_triggered_signals": performance["daily"]["news_triggered_signals"],  # FIXED
            "enhanced_ml_signals": performance["daily"]["enhanced_ml_signals"],  # FIXED
            "avg_sentiment": performance["daily"]["avg_news_sentiment"],
            "sources_active": performance["daily"]["news_sources_active"],
            "last_update": performance["daily"]["last_news_update"]
        }
        
        # Add integration statistics
        if service_manager.news_intelligence:
            performance["news_signal_integration"] = {
                "news_intelligence_available": True,
                "news_monitoring_active": service_manager.news_monitoring_active,
                "articles_processed": performance["daily"]["news_articles_processed"]
            }
        
        # Add production mode indicators
        performance["production_mode"] = True
        performance["demo_signals_disabled"] = True
        
        return performance
        
    except Exception as e:
        logger.error(f"❌ Performance analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/signals/generate")
async def generate_signals_manually(
    service_manager: CorrectedServiceManager = Depends(get_service_manager),
    signal_service: SignalService = Depends(get_signal_service)
):
    """Generate signals manually with fixed news-signal integration - PRODUCTION ONLY"""
    try:
        signals = await signal_service.generate_signals()
        
        for signal in signals:
            await signal_service.process_signal(signal, is_priority=False)
        
        return {
            "success": True,
            "signals_generated": len(signals),
            "signals": signals,
            "interactive_trading": service_manager.interactive_trading_active,
            "news_intelligence": service_manager.system_health.get("news_intelligence", False),
            "news_signal_integration": service_manager.system_health.get("news_signal_integration", False),  # FIXED
            "current_regime": str(service_manager.current_regime),
            "production_mode": True,
            "demo_signals_disabled": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Manual signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
@profile_timing("signal_generation_endpoint")
async def generate_signal(request: Request, service_manager=Depends(get_service_manager)):
    return await service_manager.signal_service.generate_signal(request) 