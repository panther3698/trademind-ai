# ================================================================
# API Dependencies for TradeMind AI
# Proper dependency injection using FastAPI Depends pattern
# ================================================================

"""
API Dependencies Module
Provides dependency injection for all API routes

This module replaces global service manager usage with proper
dependency injection, making the system more testable and maintainable.

Usage:
    @app.get("/health")
    async def health_check(
        analytics: CorrectedAnalytics = Depends(get_analytics_service),
        service_manager: CorrectedServiceManager = Depends(get_service_manager)
    ):
        return {"status": "healthy"}
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
import logging

from app.core.services.analytics_service import CorrectedAnalytics
from app.core.services.signal_service import SignalService
from app.core.services.news_service import NewsSignalIntegrationService
from app.core.services.notification_service import NotificationService
from app.core.services.service_manager import CorrectedServiceManager, get_global_service_manager
from app.services.enhanced_news_intelligence import EnhancedNewsIntelligenceSystem
from app.services.enhanced_market_data_nifty100 import EnhancedMarketDataService
from app.services.regime_detector import RegimeDetector
from app.services.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)

def get_service_manager() -> CorrectedServiceManager:
    """
    Get the global service manager instance
    
    Returns:
        CorrectedServiceManager: The initialized service manager instance
        
    Raises:
        HTTPException: If service manager is not available
    """
    try:
        service_manager = get_global_service_manager()
        if service_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service manager is not available"
            )
        return service_manager
    except Exception as e:
        logger.error(f"❌ Failed to get service manager: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service manager is not available"
        )

def get_analytics_service(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
) -> CorrectedAnalytics:
    """
    Get the analytics service
    
    Args:
        service_manager: The service manager instance
        
    Returns:
        CorrectedAnalytics: The analytics service
        
    Raises:
        HTTPException: If analytics service is not available
    """
    try:
        analytics_service = service_manager.get_analytics_service()
        if analytics_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Analytics service is not available"
            )
        return analytics_service
    except Exception as e:
        logger.error(f"❌ Failed to get analytics service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Analytics service is not available"
        )

def get_signal_service(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
) -> SignalService:
    """
    Get the signal service
    
    Args:
        service_manager: The service manager instance
        
    Returns:
        SignalService: The signal service
        
    Raises:
        HTTPException: If signal service is not available
    """
    try:
        signal_service = service_manager.get_signal_service()
        if signal_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Signal service is not available"
            )
        return signal_service
    except Exception as e:
        logger.error(f"❌ Failed to get signal service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Signal service is not available"
        )

def get_news_service(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
) -> EnhancedNewsIntelligenceSystem:
    """
    Get the news intelligence service
    
    Args:
        service_manager: The service manager instance
        
    Returns:
        EnhancedNewsIntelligenceSystem: The news intelligence service
        
    Raises:
        HTTPException: If news service is not available
    """
    try:
        news_service = service_manager.get_news_intelligence()
        if news_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="News service is not available"
            )
        return news_service
    except Exception as e:
        logger.error(f"❌ Failed to get news service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="News service is not available"
        )

def get_notification_service(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
) -> NotificationService:
    """
    Get the notification service
    
    Args:
        service_manager: The service manager instance
        
    Returns:
        NotificationService: The notification service
        
    Raises:
        HTTPException: If notification service is not available
    """
    try:
        notification_service = service_manager.get_notification_service()
        if notification_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Notification service is not available"
            )
        return notification_service
    except Exception as e:
        logger.error(f"❌ Failed to get notification service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Notification service is not available"
        )

def get_market_service(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
) -> EnhancedMarketDataService:
    """
    Get the market data service
    
    Args:
        service_manager: The service manager instance
        
    Returns:
        EnhancedMarketDataService: The market data service
        
    Raises:
        HTTPException: If market service is not available
    """
    try:
        market_service = service_manager.get_market_service()
        if market_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Market data service is not available"
            )
        return market_service
    except Exception as e:
        logger.error(f"❌ Failed to get market service: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Market data service is not available"
        )

def get_regime_detector(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
) -> RegimeDetector:
    """
    Get the regime detector service
    
    Args:
        service_manager: The service manager instance
        
    Returns:
        RegimeDetector: The regime detector service
        
    Raises:
        HTTPException: If regime detector is not available
    """
    try:
        regime_detector = service_manager.get_regime_detector()
        if regime_detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Regime detector is not available"
            )
        return regime_detector
    except Exception as e:
        logger.error(f"❌ Failed to get regime detector: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Regime detector is not available"
        )

def get_backtest_engine(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
) -> BacktestEngine:
    """
    Get the backtest engine service
    
    Args:
        service_manager: The service manager instance
        
    Returns:
        BacktestEngine: The backtest engine service
        
    Raises:
        HTTPException: If backtest engine is not available
    """
    try:
        backtest_engine = service_manager.get_backtest_engine()
        if backtest_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Backtest engine is not available"
            )
        return backtest_engine
    except Exception as e:
        logger.error(f"❌ Failed to get backtest engine: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backtest engine is not available"
        )

def get_initialized_service_manager(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
) -> CorrectedServiceManager:
    """
    Get the service manager only if it's initialized
    
    Args:
        service_manager: The service manager instance
        
    Returns:
        CorrectedServiceManager: The initialized service manager
        
    Raises:
        HTTPException: If service manager is not initialized
    """
    if not service_manager.is_initialized():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service manager is not initialized"
        )
    return service_manager

# ================================================================
# Dependency Aliases for Convenience
# ================================================================

# Common dependency combinations
def get_core_services(
    analytics: CorrectedAnalytics = Depends(get_analytics_service),
    signal: SignalService = Depends(get_signal_service),
    notification: NotificationService = Depends(get_notification_service)
):
    """
    Get core services (analytics, signal, notification)
    
    Returns:
        tuple: (analytics_service, signal_service, notification_service)
    """
    return analytics, signal, notification

def get_all_services(
    service_manager: CorrectedServiceManager = Depends(get_initialized_service_manager),
    analytics: CorrectedAnalytics = Depends(get_analytics_service),
    signal: SignalService = Depends(get_signal_service),
    notification: NotificationService = Depends(get_notification_service)
):
    """
    Get all services (for comprehensive endpoints)
    
    Returns:
        tuple: (service_manager, analytics_service, signal_service, notification_service)
    """
    return service_manager, analytics, signal, notification 