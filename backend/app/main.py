# ================================================================
# Updated backend/app/main.py - With Config System and Analytics
# ================================================================

"""
TradeMind AI - FastAPI Main Application with Professional Config and Analytics
Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
import logging
from datetime import datetime
from typing import List
import os

# Import our config and services
from app.core.config import settings
from app.services.telegram_service import TelegramService
from app.services.analytics_service import AnalyticsService

# Configure logging based on config
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
class ServiceManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.telegram_service: TelegramService = None
        self.analytics_service: AnalyticsService = None
        
        # Initialize services based on configuration
        if settings.is_telegram_configured:
            self.telegram_service = TelegramService(
                settings.telegram_bot_token, 
                settings.telegram_chat_id
            )
            logger.info("✅ Telegram service initialized")
        else:
            logger.warning("⚠️ Telegram not configured - add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")
        
        # Initialize analytics
        self.analytics_service = AnalyticsService(settings.database_url)
        logger.info("✅ Analytics service initialized")

    async def connect_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect_websocket(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast_to_dashboard(self, message: dict):
        """Broadcast message to all connected dashboard clients"""
        if self.active_connections:
            for connection in self.active_connections.copy():
                try:
                    await connection.send_json(message)
                except:
                    self.active_connections.remove(connection)

# Global service manager
service_manager = ServiceManager()

# Enhanced signal generation with analytics tracking
async def signal_generator_task():
    """Enhanced signal generator with analytics tracking"""
    counter = 1
    
    # Send test message on startup if Telegram is configured
    if service_manager.telegram_service:
        success = await service_manager.telegram_service.send_test_message()
        if success:
            logger.info("🚀 Telegram test message sent successfully")
        else:
            logger.error("❌ Telegram test message failed")
    
    while True:
        try:
            # Check if we've exceeded daily signal limit
            daily_stats = service_manager.analytics_service.get_daily_stats()
            if daily_stats["signals_generated"] >= settings.max_signals_per_day:
                logger.info(f"📊 Daily signal limit reached ({settings.max_signals_per_day}), waiting until tomorrow")
                await asyncio.sleep(3600)  # Wait 1 hour before checking again
                continue
            
            # Generate enhanced demo signal
            demo_signal = {
                "id": f"signal_{counter}",
                "symbol": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "LT"][counter % 7],
                "action": "BUY" if counter % 2 == 0 else "SELL",
                "entry_price": 2500 + (counter * 10),
                "target_price": 2550 + (counter * 10),
                "stop_loss": 2450 + (counter * 10),
                "confidence": max(settings.min_confidence_threshold, 0.65 + (counter % 4) * 0.05),
                "sentiment_score": 0.2 if counter % 2 == 0 else -0.1,
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "signal_type": "AI_GENERATED",
                "risk_level": "MEDIUM",
                "expected_duration": "INTRADAY"
            }
            
            # Track signal generation in analytics
            await service_manager.analytics_service.track_signal_generated(demo_signal)
            
            # Send to Telegram (if configured)
            telegram_success = False
            if service_manager.telegram_service:
                telegram_success = await service_manager.telegram_service.send_signal_notification(demo_signal)
                await service_manager.analytics_service.track_telegram_sent(telegram_success, demo_signal)
            
            # Broadcast to dashboard
            await service_manager.broadcast_to_dashboard({
                "type": "new_signal",
                "data": demo_signal
            })
            
            # Broadcast analytics update
            await service_manager.broadcast_to_dashboard({
                "type": "analytics_update",
                "data": service_manager.analytics_service.get_performance_summary()
            })
            
            logger.info(f"📡 Signal #{counter}: {demo_signal['symbol']} {demo_signal['action']} | "
                       f"Telegram: {'✅' if telegram_success else '❌'} | Dashboard: ✅")
            
            counter += 1
            
            # Wait based on configuration
            await asyncio.sleep(settings.signal_generation_interval)
            
        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
            await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 TradeMind AI Backend Starting...")
    logger.info(f"🔧 Environment: {settings.environment}")
    logger.info(f"📊 Analytics: {'Enabled' if settings.track_performance else 'Disabled'}")
    logger.info(f"📱 Telegram: {'Configured' if settings.is_telegram_configured else 'Not Configured'}")
    logger.info(f"💹 Zerodha: {'Configured' if settings.is_zerodha_configured else 'Not Configured'}")
    
    # Start background signal generator
    task = asyncio.create_task(signal_generator_task())
    
    yield
    
    # Shutdown
    task.cancel()
    logger.info("🛑 TradeMind AI Backend Shutting Down...")

# Create FastAPI app
app = FastAPI(
    title="TradeMind AI API",
    description="AI-Powered Indian Stock Trading Signals with Analytics",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# API Endpoints
# ================================================================

@app.get("/")
async def root():
    return {
        "message": "🇮🇳 TradeMind AI - Professional Trading Signals",
        "status": "active",
        "version": "2.0.0",
        "environment": settings.environment,
        "telegram_configured": settings.is_telegram_configured,
        "zerodha_configured": settings.is_zerodha_configured,
        "time": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check with service status"""
    performance = service_manager.analytics_service.get_performance_summary()
    
    return {
        "status": "healthy",
        "connections": len(service_manager.active_connections),
        "services": {
            "telegram": settings.is_telegram_configured,
            "zerodha": settings.is_zerodha_configured,
            "analytics": settings.track_performance
        },
        "performance": performance,
        "time": datetime.now().isoformat(),
        "uptime_hours": performance["daily"]["system_uptime_hours"]
    }

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    """Get comprehensive analytics for dashboard"""
    try:
        performance = service_manager.analytics_service.get_performance_summary()
        
        # Add configuration info
        performance["configuration"] = {
            "max_signals_per_day": settings.max_signals_per_day,
            "min_confidence_threshold": settings.min_confidence_threshold,
            "signal_interval_seconds": settings.signal_generation_interval,
            "telegram_enabled": settings.is_telegram_configured,
            "zerodha_enabled": settings.is_zerodha_configured
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"❌ Error getting dashboard analytics: {e}")
        return {"error": str(e)}

@app.get("/api/analytics/daily")
async def get_daily_analytics():
    """Get today's analytics"""
    try:
        return service_manager.analytics_service.get_daily_stats()
    except Exception as e:
        logger.error(f"❌ Error getting daily analytics: {e}")
        return {"error": str(e)}

@app.get("/api/signals/latest")
async def get_latest_signals():
    """Get latest signals with analytics"""
    daily_stats = service_manager.analytics_service.get_daily_stats()
    
    return {
        "signals": [
            {
                "id": "demo_latest",
                "symbol": "RELIANCE",
                "action": "BUY",
                "entry_price": 2485,
                "target_price": 2550,
                "stop_loss": 2440,
                "confidence": 0.78,
                "sentiment_score": 0.15,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }
        ],
        "count": 1,
        "daily_stats": daily_stats,
        "generated_at": datetime.now().isoformat()
    }

@app.post("/api/test-telegram")
async def test_telegram_notification():
    """Manual endpoint to test Telegram notifications"""
    if not service_manager.telegram_service:
        return {"error": "Telegram not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env file"}
    
    test_signal = {
        "id": "test_manual",
        "symbol": "TESTSTOCK",
        "action": "BUY",
        "entry_price": 1000,
        "target_price": 1050,
        "stop_loss": 950,
        "confidence": 0.85,
        "sentiment_score": 0.25,
        "created_at": datetime.now().isoformat(),
        "status": "test"
    }
    
    success = await service_manager.telegram_service.send_signal_notification(test_signal)
    
    if success:
        await service_manager.analytics_service.track_telegram_sent(True, test_signal)
    
    return {
        "success": success,
        "message": "Test notification sent successfully" if success else "Failed to send notification",
        "telegram_configured": settings.is_telegram_configured
    }

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    await service_manager.connect_websocket(websocket)
    
    try:
        # Send welcome message with current analytics
        performance = service_manager.analytics_service.get_performance_summary()
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to TradeMind AI Professional",
            "analytics": performance,
            "time": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Echo back any messages for testing
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "time": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        service_manager.disconnect_websocket(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)