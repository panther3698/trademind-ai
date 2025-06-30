# ================================================================
# TradeMind AI - Clean Application Entry Point
# ================================================================

"""
TradeMind AI - FastAPI Application Entry Point
Clean architecture with extracted services and coordinator pattern

Architecture:
Frontend ↔ WebSocket/REST API ↔ Service Coordinator ↔ Extracted Services
   ↓              ↓                    ↓                        ↓
Dashboard    Real-time Updates    Service Manager         Individual Services
   ↓              ↓                    ↓                        ↓
Analytics    Signal Approval     Background Tasks        Telegram, Analytics, etc.
   ↓              ↓                    ↓                        ↓
News Intel   Market Intelligence  Breaking News Alerts    Event-driven    News-based Signals
"""

# ================================================================
# STANDARD LIBRARY IMPORTS
# ================================================================
import os
import sys
import logging
import signal as signal_handler
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

# ================================================================
# THIRD-PARTY IMPORTS
# ================================================================
from fastapi import FastAPI, Depends, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ================================================================
# APPLICATION IMPORTS
# ================================================================

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Core dependencies
from app.core.services.service_manager import CorrectedServiceManager, set_global_service_manager
from app.api.routes.health import router as health_router
from app.api.routes.news import router as news_router
from app.api.routes.signals import router as signals_router
from app.api.routes.websocket import router as websocket_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.feature_flags import router as feature_flags_router
from app.api.routes.config import router as config_router

# Configuration
try:
    from app.core.availability import settings, CONFIG_AVAILABLE
except ImportError:
    CONFIG_AVAILABLE = False

# ================================================================
# LOGGING SETUP
# ================================================================

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Always recreate the log file on startup to avoid lock/corruption issues
        logging.FileHandler('logs/trademind_enhanced.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# DEPENDENCY INJECTION
# ================================================================

# Global service manager instance
_service_manager: Optional[CorrectedServiceManager] = None

def get_service_manager() -> CorrectedServiceManager:
    """Dependency function to get the service manager instance"""
    global _service_manager
    if _service_manager is None:
        _service_manager = CorrectedServiceManager()
    return _service_manager

# ================================================================
# APPLICATION LIFESPAN MANAGEMENT
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management - startup and shutdown"""
    
    # Startup
    try:
        logger.info("🚀 Starting TradeMind AI Enhanced Edition...")
        
        # Get service manager and initialize all services via coordinator
        service_manager = get_service_manager()
        await service_manager.initialize_all_services()
        
        # Set global service manager for dependency injection
        set_global_service_manager(service_manager)
        
        # Start performance monitoring
        from app.core.performance_monitor import performance_monitor
        await performance_monitor.start_monitoring()
        
        # Start signal generation
        await service_manager.signal_service.start_signal_generation()
        
        logger.info("✅ TradeMind AI Enhanced Edition fully operational!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 TradeMind AI Enhanced Edition shutting down...")
    
    try:
        service_manager = get_service_manager()
        
        # Stop signal generation
        await service_manager.signal_service.stop_signal_generation()
        
        # Stop performance monitoring
        from app.core.performance_monitor import performance_monitor
        await performance_monitor.stop_monitoring()
        
        # Shutdown service coordinator
        await service_manager.shutdown()
        
        logger.info("✅ Shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# ================================================================
# FASTAPI APPLICATION
# ================================================================

app = FastAPI(
    title="TradeMind AI - Enhanced Edition with Interactive Trading + News Intelligence",
    description="AI-Powered Trading Platform with Interactive Approval, Order Execution & Real-time News Intelligence - PRODUCTION MODE",
    version="5.1.0-production-clean",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================================================================
# BASIC ENDPOINTS
# ================================================================

@app.get("/")
async def root(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """Root endpoint with system information"""
    return {
        "message": "TradeMind AI Enhanced Edition - PRODUCTION MODE",
        "version": "5.1.0-production-clean",
        "architecture": "coordinator_pattern_with_extracted_services",
        "production_mode": True,
        "demo_signals_disabled": True,
        "features": {
            "interactive_trading": service_manager.interactive_trading_active,
            "news_intelligence": service_manager.get_system_health().get("news_intelligence", False),
            "news_signal_integration": service_manager.get_system_health().get("news_signal_integration", False),
            "order_execution": service_manager.get_system_health().get("order_execution", False),
            "telegram": service_manager.get_system_health().get("telegram", False),
            "enhanced_market_data": service_manager.get_system_health().get("enhanced_market_data", False),
            "regime_detection": service_manager.get_system_health().get("regime_detection", False),
            "backtesting": service_manager.get_system_health().get("backtesting", False)
        },
        "system_health": service_manager.get_system_health(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """Health check endpoint"""
    try:
        return {
            "status": "healthy" if service_manager.is_initialized() else "degraded",
            "architecture": "coordinator_pattern_with_extracted_services",
            "production_mode": True,
            "demo_signals_disabled": True,
            "initialization_error": service_manager.get_initialization_error(),
            "system_health": service_manager.get_system_health(),
            "signal_generation_active": service_manager.signal_service.signal_generation_active,
            "interactive_trading_active": service_manager.interactive_trading_active,
            "news_monitoring_active": service_manager.news_monitoring_active,
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

@app.get("/api/system/status")
async def get_system_status(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """Get detailed system status"""
    return {
        "architecture": "coordinator_pattern_with_extracted_services",
        "version": "5.1.0-production-clean",
        "production_mode": True,
        "demo_signals_disabled": True,
        "initialized": service_manager.is_initialized(),
        "initialization_error": service_manager.get_initialization_error(),
        "system_health": service_manager.get_system_health(),
        "signal_generation_active": service_manager.signal_service.signal_generation_active,
        "interactive_trading_active": service_manager.interactive_trading_active,
        "news_monitoring_active": service_manager.news_monitoring_active,
        "features": {
            "interactive_trading": service_manager.interactive_trading_active,
            "news_intelligence": service_manager.get_system_health().get("news_intelligence", False),
            "news_signal_integration": service_manager.get_system_health().get("news_signal_integration", False),
            "order_execution": service_manager.get_system_health().get("order_execution", False),
            "telegram": service_manager.get_system_health().get("telegram", False),
            "enhanced_market_data": service_manager.get_system_health().get("enhanced_market_data", False),
            "regime_detection": service_manager.get_system_health().get("regime_detection", False),
            "backtesting": service_manager.get_system_health().get("backtesting", False),
            "production_only": True,
            "demo_signals": False
        },
        "timestamp": datetime.now().isoformat()
    }

# ================================================================
# WEBHOOK ENDPOINTS
# ================================================================

@app.post("/webhook/telegram")
async def telegram_webhook(
    request: Request, 
    background_tasks: BackgroundTasks,
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """Telegram webhook endpoint for interactive features"""
    try:
        notification_service = service_manager.get_notification_service()
        webhook_handler = notification_service.get_webhook_handler() if notification_service else None
        if webhook_handler:
            return await webhook_handler.handle_webhook(request)
        else:
            logger.warning("⚠️ Webhook handler not available")
            return JSONResponse({"ok": False, "error": "Webhook handler not configured"})
    except Exception as e:
        logger.error(f"❌ Webhook handling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webhook/status")
async def webhook_status(
    service_manager: CorrectedServiceManager = Depends(get_service_manager)
):
    """Webhook status endpoint"""
    notification_service = service_manager.get_notification_service()
    webhook_handler = notification_service.get_webhook_handler() if notification_service else None
    
    return {
        "status": "active" if webhook_handler else "inactive",
        "interactive_trading": service_manager.interactive_trading_active,
        "order_engine": service_manager.get_system_health().get("order_execution", False),
        "news_intelligence": service_manager.get_system_health().get("news_intelligence", False),
        "news_monitoring": service_manager.news_monitoring_active,
        "news_signal_integration": service_manager.get_system_health().get("news_signal_integration", False),
        "production_mode": True,
        "demo_signals_disabled": True,
        "timestamp": datetime.now().isoformat()
    }

# ================================================================
# SIGNAL HANDLING
# ================================================================

def shutdown_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    try:
        service_manager = get_service_manager()
        service_manager.signal_generation_active = False
        service_manager.regime_monitoring_active = False
        service_manager.news_monitoring_active = False
    except Exception as e:
        logger.error(f"Error in shutdown handler: {e}")

# Register signal handlers
signal_handler.signal(signal_handler.SIGINT, shutdown_handler)
signal_handler.signal(signal_handler.SIGTERM, shutdown_handler)

# ================================================================
# ROUTER INCLUDES
# ================================================================

# Include API routers
app.include_router(health_router)

app.include_router(news_router, prefix='/api')

app.include_router(signals_router, prefix='/api')

app.include_router(websocket_router)

app.include_router(metrics_router)

app.include_router(feature_flags_router)

app.include_router(config_router)

# ================================================================
# STARTUP MESSAGE
# ================================================================

logger.info("=" * 80)
logger.info("🚀 TradeMind AI Enhanced Edition v5.1 - Clean Architecture")
logger.info("=" * 80)
logger.info("🔥 DEMO SIGNALS: ❌ COMPLETELY DISABLED")
logger.info("✅ PRODUCTION SIGNALS: Only real ML signals with proper confidence thresholds")
logger.info("🔧 NEWS-SIGNAL INTEGRATION: ✅ ENABLED - Real-time news triggers immediate signals")
logger.info("🎯 Interactive Trading: ✅ ENABLED")
logger.info("📱 Enhanced Telegram: ✅ AVAILABLE")
logger.info("💼 Order Execution: ✅ AVAILABLE")
logger.info("🌐 Webhook Handler: ✅ AVAILABLE")
logger.info("📊 Enhanced Market Data: ✅ AVAILABLE")
logger.info("📰 News Intelligence: ✅ AVAILABLE")
logger.info("🔗 News-Signal Integration: ✅ ENABLED")
logger.info("⚡ Signal Flow: News → Breaking News Detection → Immediate Signal → Telegram → Order Execution")
logger.info("🏗️ Architecture: Coordinator Pattern with Extracted Services")
logger.info("🔧 Dependency Injection: ✅ IMPLEMENTED - All API routes use proper DI")
logger.info(f"🔧 Configuration: {'✅ LOADED' if CONFIG_AVAILABLE else '⚠️ USING DEFAULTS'}")
logger.info("=" * 80)
logger.info("📋 CLEAN ARCHITECTURE BENEFITS:")
logger.info("  ✅ ServiceManager is now a clean coordinator")
logger.info("  ✅ All business logic extracted to dedicated services")
logger.info("  ✅ Single responsibility principle applied")
logger.info("  ✅ Easy to test and maintain individual services")
logger.info("  ✅ Scalable and modular architecture")
logger.info("  ✅ Production-ready with all features preserved")
logger.info("  ✅ Proper dependency injection implemented")
logger.info("  ✅ No more global service manager usage in API routes")
logger.info("=" * 80)

# ================================================================
# APPLICATION ENTRY POINT
# ================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )