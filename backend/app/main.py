# ================================================================
# TradeMind AI - Enhanced Main Application with Interactive Trading
# ================================================================

"""
TradeMind AI - Production-Ready FastAPI Application
Enhanced with Interactive Trading, Regime Detection & Backtesting

Architecture:
Frontend ↔ WebSocket/REST API ↔ Enhanced Service Manager ↔ Interactive Trading ↔ Zerodha
   ↓              ↓                    ↓                        ↓              ↓
Dashboard    Real-time Updates    Telegram Webhook         Order Engine    Live Execution
   ↓              ↓                    ↓                        ↓              ↓
Analytics    Signal Approval     Background Tasks        Risk Management   Position Tracking
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import json
import logging
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Optional, Any
import os
import sys
from pathlib import Path
import traceback
import signal as signal_handler
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trademind_enhanced.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Import configuration with fallback
try:
    from app.core.config import settings
    CONFIG_AVAILABLE = True
    logger.info("✅ Configuration loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Configuration not found: {e}")
    # Create minimal settings for fallback
    class MockSettings:
        environment = "development"
        log_level = "INFO"
        debug = True
        is_telegram_configured = False
        is_zerodha_configured = False
        max_signals_per_day = 10
        signal_generation_interval = 300
        min_confidence_threshold = 0.7
        track_performance = True
        database_url = "sqlite:///./trademind.db"
        redis_url = "redis://localhost:6379"
        telegram_bot_token = None
        telegram_chat_id = None
        zerodha_api_key = None
        zerodha_access_token = None
        is_production = False
    
    settings = MockSettings()
    CONFIG_AVAILABLE = False

# ================================================================
# ENHANCED IMPORTS - Interactive Trading Components
# ================================================================

# Enhanced Telegram Service (replaces old telegram_service)
try:
    from app.services.enhanced_telegram_service import (
        EnhancedTelegramService, 
        SignalStatus, 
        PendingSignal,
        create_telegram_service
    )
    ENHANCED_TELEGRAM_AVAILABLE = True
    logger.info("✅ Enhanced Telegram Service imported")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced Telegram Service not available: {e}")
    # Fallback to basic telegram service
    try:
        from app.services.telegram_service import TelegramService, create_telegram_service
        ENHANCED_TELEGRAM_AVAILABLE = False
        logger.info("⚠️ Using basic Telegram Service (no interactive features)")
    except ImportError:
        logger.error("❌ No Telegram service available")
        ENHANCED_TELEGRAM_AVAILABLE = False
        TelegramService = None

# Telegram Webhook Handler
try:
    from app.services.telegram_webhook_handler import (
        TelegramWebhookHandler,
        
        TradeMindTelegramIntegration,
        create_telegram_webhook_app
    )
    WEBHOOK_HANDLER_AVAILABLE = True
    logger.info("✅ Telegram Webhook Handler imported")
except ImportError as e:
    logger.warning(f"⚠️ Telegram Webhook Handler not available: {e}")
    WEBHOOK_HANDLER_AVAILABLE = False

# Zerodha Order Engine
try:
    from app.services.zerodha_order_engine import (
        ZerodhaOrderEngine,
        OrderType,
        ExchangeType,
        OrderResult,
        TradeOrder,
        OrderStatus
    )
    ZERODHA_ORDER_ENGINE_AVAILABLE = True
    logger.info("✅ Zerodha Order Engine imported")
except ImportError as e:
    logger.warning(f"⚠️ Zerodha Order Engine not available: {e}")
    ZERODHA_ORDER_ENGINE_AVAILABLE = False

# ================================================================
# EXISTING IMPORTS (Enhanced Services)
# ================================================================

# Enhanced Market Data Service
try:
    from app.services.enhanced_market_data_nifty100 import (
        EnhancedMarketDataService, MarketStatus, TradingMode, 
        Nifty100Universe, MarketTick
    )
    ENHANCED_MARKET_DATA_AVAILABLE = True
    logger.info("✅ Enhanced Market Data Service imported")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced Market Data Service not available: {e}")
    ENHANCED_MARKET_DATA_AVAILABLE = False

# Analytics Service
try:
    from app.services.analytics_service import AnalyticsService, create_analytics_service
    ANALYTICS_AVAILABLE = True
    logger.info("✅ Analytics Service imported")
except ImportError as e:
    logger.warning(f"⚠️ Analytics Service not available: {e}")
    ANALYTICS_AVAILABLE = False

# Signal Generator
try:
    from app.services.production_signal_generator import ProductionMLSignalGenerator
    SIGNAL_GENERATOR_AVAILABLE = True
    logger.info("✅ Production Signal Generator imported")
except ImportError as e:
    logger.warning(f"⚠️ Production Signal Generator not available: {e}")
    SIGNAL_GENERATOR_AVAILABLE = False

# Signal Logging
try:
    from app.core.signal_logger import (
        InstitutionalSignalLogger as SignalLogger, 
        SignalRecord
    )
    SIGNAL_LOGGING_AVAILABLE = True
    logger.info("✅ Signal Logging imported")
except ImportError as e:
    logger.warning(f"⚠️ Signal logging not available: {e}")
    SIGNAL_LOGGING_AVAILABLE = False

# Advanced Services (Regime Detection & Backtesting)
try:
    from app.services.regime_detector import RegimeDetector, MarketRegime
    REGIME_DETECTOR_AVAILABLE = True
    logger.info("✅ Regime Detector imported")
except ImportError as e:
    logger.warning(f"⚠️ Regime Detector not available: {e}")
    REGIME_DETECTOR_AVAILABLE = False
    
    # Create mock enum for compatibility
    class MarketRegime:
        BULLISH_TRENDING = "BULLISH_TRENDING"
        BEARISH_TRENDING = "BEARISH_TRENDING"
        SIDEWAYS_CHOPPY = "SIDEWAYS_CHOPPY"
        HIGH_VOLATILITY = "HIGH_VOLATILITY"

try:
    from app.services.backtest_engine import BacktestEngine
    BACKTEST_ENGINE_AVAILABLE = True
    logger.info("✅ Backtest Engine imported")
except ImportError as e:
    logger.warning(f"⚠️ Backtest Engine not available: {e}")
    BACKTEST_ENGINE_AVAILABLE = False

# Set aliases based on what's available
if ENHANCED_TELEGRAM_AVAILABLE:
    TelegramService = EnhancedTelegramService
    logger.info("✅ Using Enhanced Telegram Service with interactive features")
else:
    logger.warning("⚠️ Using basic Telegram Service (no interactive trading)")

# ================================================================
# Enhanced Analytics (Standalone Implementation)
# ================================================================

class CorrectedAnalytics:
    """Standalone analytics implementation with regime tracking"""
    
    def __init__(self):
        self.daily_stats = {
            "signals_generated": 0,
            "priority_signals": 0,
            "premarket_analyses": 0,
            "signals_sent": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "average_confidence": 0.0,
            "system_uptime_hours": 0,
            "last_signal_time": None,
            "last_premarket_analysis": None,
            "telegram_success": 0,
            "telegram_failures": 0,
            "nifty100_coverage": 100,
            "priority_stocks_tracked": 50,
            "status": "operational",
            "market_mode": "closed",
            "current_regime": "UNKNOWN",
            "regime_confidence": 0.0,
            "regime_changes": 0,
            "backtest_runs": 0,
            "last_backtest": None,
            # NEW: Interactive trading stats
            "signals_approved": 0,
            "signals_rejected": 0,
            "orders_executed": 0,
            "order_success_rate": 0.0,
            "total_trading_pnl": 0.0,
            "approval_rate": 0.0
        }
        self.start_time = datetime.now()
        self.premarket_opportunities = []
        self.priority_signals_today = []
        
    async def track_signal_generated(self, signal):
        """Track signal generation"""
        self.daily_stats["signals_generated"] += 1
        self.daily_stats["last_signal_time"] = datetime.now().isoformat()
        
        # Track priority signals
        current_time = datetime.now().time()
        if dt_time(9, 15) <= current_time <= dt_time(9, 45):
            self.daily_stats["priority_signals"] += 1
            self.priority_signals_today.append(signal)
        
        # Update average confidence
        current_avg = self.daily_stats["average_confidence"]
        count = self.daily_stats["signals_generated"]
        new_confidence = signal.get("confidence", 0.0)
        
        if count > 0:
            self.daily_stats["average_confidence"] = (
                (current_avg * (count - 1) + new_confidence) / count
            )
    
    async def track_signal_approval(self, approved: bool):
        """Track signal approval/rejection"""
        if approved:
            self.daily_stats["signals_approved"] += 1
        else:
            self.daily_stats["signals_rejected"] += 1
            
        # Update approval rate
        total_responses = self.daily_stats["signals_approved"] + self.daily_stats["signals_rejected"]
        if total_responses > 0:
            self.daily_stats["approval_rate"] = (
                self.daily_stats["signals_approved"] / total_responses * 100
            )
    
    async def track_order_execution(self, success: bool, pnl: float = 0.0):
        """Track order execution results"""
        if success:
            self.daily_stats["orders_executed"] += 1
            self.daily_stats["total_trading_pnl"] += pnl
            
        # Update success rate
        total_orders = self.daily_stats["orders_executed"]
        if total_orders > 0:
            self.daily_stats["order_success_rate"] = (
                self.daily_stats["orders_executed"] / 
                (self.daily_stats["signals_approved"] or 1) * 100
            )
    
    async def track_premarket_analysis(self, opportunities_count):
        """Track pre-market analysis"""
        self.daily_stats["premarket_analyses"] += 1
        self.daily_stats["last_premarket_analysis"] = datetime.now().isoformat()
        
    async def track_telegram_sent(self, success: bool, signal: Dict = None):
        """Track Telegram message success/failure"""
        if success:
            self.daily_stats["telegram_success"] += 1
            self.daily_stats["signals_sent"] += 1
        else:
            self.daily_stats["telegram_failures"] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        # Calculate uptime
        uptime_delta = datetime.now() - self.start_time
        self.daily_stats["system_uptime_hours"] = uptime_delta.total_seconds() / 3600
        
        return {
            "daily": self.daily_stats.copy(),
            "system": {
                "start_time": self.start_time.isoformat(),
                "uptime_hours": self.daily_stats["system_uptime_hours"],
                "is_operational": self.daily_stats["status"] == "operational"
            },
            "enhanced_features": {
                "interactive_trading": ENHANCED_TELEGRAM_AVAILABLE,
                "order_execution": ZERODHA_ORDER_ENGINE_AVAILABLE,
                "webhook_handler": WEBHOOK_HANDLER_AVAILABLE,
                "regime_detection": REGIME_DETECTOR_AVAILABLE,
                "backtesting": BACKTEST_ENGINE_AVAILABLE
            }
        }
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily statistics"""
        return self.daily_stats.copy()

# ================================================================
# ENHANCED SERVICE MANAGER WITH INTERACTIVE TRADING
# ================================================================

class CorrectedServiceManager:
    """Enhanced service manager with interactive trading capabilities"""
    
    def __init__(self):
        # Core services
        self.active_connections: List[WebSocket] = []
        self.telegram_service: Optional[TelegramService] = None
        self.analytics_service = CorrectedAnalytics()
        
        # Enhanced Market Data Service (No Legacy)
        self.enhanced_market_service = None
        
        # Optional services
        self.signal_generator: Optional[ProductionMLSignalGenerator] = None
        self.signal_logger: Optional[SignalLogger] = None
        
        # Advanced services
        self.regime_detector: Optional[RegimeDetector] = None
        self.backtest_engine: Optional[BacktestEngine] = None
        
        # NEW: Interactive Trading Components
        self.order_engine: Optional[ZerodhaOrderEngine] = None
        self.webhook_handler: Optional[TelegramWebhookHandler] = None
        self.telegram_integration: Optional[TradeMindTelegramIntegration] = None
        
        # Enhanced tracking
        self.current_market_status = MarketStatus.CLOSED if ENHANCED_MARKET_DATA_AVAILABLE else "CLOSED"
        self.current_trading_mode = TradingMode.PRE_MARKET_ANALYSIS if ENHANCED_MARKET_DATA_AVAILABLE else "PRE_MARKET_ANALYSIS"
        self.current_regime = MarketRegime.SIDEWAYS_CHOPPY
        self.regime_confidence = 0.5
        self.premarket_opportunities = []
        self.priority_signals_queue = []
        
        # System state
        self.is_initialized = False
        self.initialization_error = None
        self.system_health = {
            "telegram": False,
            "enhanced_market_data": False,
            "nifty100_universe": False,
            "premarket_analysis": False,
            "priority_trading": False,
            "signal_generation": False,
            "regime_detection": False,
            "backtesting": False,
            "analytics": True,
            # NEW: Interactive trading health
            "interactive_telegram": False,
            "order_execution": False,
            "webhook_handler": False,
            "zerodha_connection": False
        }
        
        # Signal generation control
        self.signal_generation_active = False
        self.premarket_analysis_active = False
        self.priority_trading_active = False
        self.regime_monitoring_active = False
        self.interactive_trading_active = False  # NEW
        
        # Background tasks
        self.signal_generation_task = None
        self.market_monitor_task = None
        self.regime_monitor_task = None
        
    async def initialize_all_services(self):
        """Initialize all services including interactive trading"""
        try:
            logger.info("🚀 Initializing TradeMind AI Enhanced Edition (with Interactive Trading)...")
            
            # 1. Analytics already initialized
            logger.info("✅ Analytics service ready")
            
            # 2. Initialize Enhanced Telegram Service
            await self._initialize_enhanced_telegram()
            
            # 3. Initialize Zerodha Order Engine
            await self._initialize_order_engine()
            
            # 4. Initialize Webhook Handler
            await self._initialize_webhook_handler()
            
            # 5. Initialize signal logging (optional)
            await self._initialize_signal_logging()
            
            # 6. Initialize enhanced market data service
            await self._initialize_enhanced_market_data()
            
            # 7. Initialize signal generator (optional)
            await self._initialize_signal_generator()
            
            # 8. Initialize regime detector
            await self._initialize_regime_detector()
            
            # 9. Initialize backtest engine
            await self._initialize_backtest_engine()
            
            # 10. Setup interactive trading integration
            await self._setup_interactive_trading()
            
            # 11. Start market monitoring
            await self._start_market_monitoring()
            
            # 12. Start regime monitoring
            await self._start_regime_monitoring()
            
            # 13. System health check
            await self._perform_health_check()
            
            self.is_initialized = True
            logger.info("✅ TradeMind AI Enhanced Edition with Interactive Trading initialized!")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Service initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _initialize_enhanced_telegram(self):
        """Initialize Enhanced Telegram Service"""
        try:
            if settings.is_telegram_configured and ENHANCED_TELEGRAM_AVAILABLE and TelegramService:
                self.telegram_service = TelegramService(
                    bot_token=settings.telegram_bot_token,
                    chat_id=settings.telegram_chat_id,
                    redis_url=settings.redis_url
                )
                
                if self.telegram_service.is_configured():
                    # Send enhanced startup notification
                    startup_message = (
                        "🚀 <b>TradeMind AI - Enhanced Edition v5.1</b>\n"
                        "🎯 <b>Interactive Trading System ACTIVE</b>\n\n"
                        
                        "📈 <b>Features:</b>\n"
                        f"• Enhanced Telegram: {'✅' if ENHANCED_TELEGRAM_AVAILABLE else '❌'}\n"
                        f"• Order Execution: {'✅' if ZERODHA_ORDER_ENGINE_AVAILABLE else '❌'}\n"
                        f"• Webhook Handler: {'✅' if WEBHOOK_HANDLER_AVAILABLE else '❌'}\n"
                        f"• Regime Detection: {'✅' if REGIME_DETECTOR_AVAILABLE else '❌'}\n"
                        f"• Backtesting: {'✅' if BACKTEST_ENGINE_AVAILABLE else '❌'}\n\n"
                        
                        "🎯 <b>Interactive Trading:</b>\n"
                        "• Approve/Reject buttons on signals\n"
                        "• Automatic order execution\n"
                        "• Real-time position tracking\n"
                        "• Risk management built-in\n\n"
                        
                        "⚡ <b>Ready for intelligent trading!</b>"
                    )
                    
                    success = await self.telegram_service.send_system_startup_notification()
                    if success:
                        await self.telegram_service.send_message(startup_message)
                        logger.info("📱 Enhanced startup notification sent")
                    
                    self.system_health["telegram"] = True
                    self.system_health["interactive_telegram"] = ENHANCED_TELEGRAM_AVAILABLE
                else:
                    logger.warning("⚠️ Enhanced Telegram not properly configured")
                    self.system_health["telegram"] = False
                    self.system_health["interactive_telegram"] = False
            else:
                logger.warning("⚠️ Enhanced Telegram not configured")
                self.system_health["telegram"] = False
                self.system_health["interactive_telegram"] = False
        except Exception as e:
            logger.error(f"❌ Enhanced Telegram initialization failed: {e}")
            self.system_health["telegram"] = False
            self.system_health["interactive_telegram"] = False
    
    async def _initialize_order_engine(self):
        """Initialize Zerodha Order Engine"""
        try:
            if (settings.is_zerodha_configured and 
                ZERODHA_ORDER_ENGINE_AVAILABLE and 
                settings.zerodha_api_key and 
                settings.zerodha_access_token):
                
                self.order_engine = ZerodhaOrderEngine(
                    api_key=settings.zerodha_api_key,
                    access_token=settings.zerodha_access_token,
                    enable_sandbox=not settings.is_production
                )
                
                # Test connection
                await asyncio.sleep(2)  # Wait for connection
                status = await self.order_engine.get_connection_status()
                
                if status.get("connected"):
                    logger.info(f"✅ Zerodha Order Engine connected: {status.get('user_name', 'Unknown')}")
                    self.system_health["order_execution"] = True
                    self.system_health["zerodha_connection"] = True
                else:
                    logger.warning("⚠️ Zerodha Order Engine connection failed")
                    self.system_health["order_execution"] = False
                    self.system_health["zerodha_connection"] = False
            else:
                logger.warning("⚠️ Zerodha Order Engine not configured")
                self.system_health["order_execution"] = False
                self.system_health["zerodha_connection"] = False
                
        except Exception as e:
            logger.error(f"❌ Zerodha Order Engine initialization failed: {e}")
            self.system_health["order_execution"] = False
            self.system_health["zerodha_connection"] = False
    
    async def _initialize_webhook_handler(self):
        """Initialize Telegram Webhook Handler"""
        try:
            if (WEBHOOK_HANDLER_AVAILABLE and 
                self.telegram_service and 
                self.order_engine):
                
                self.webhook_handler = TelegramWebhookHandler(
                    telegram_service=self.telegram_service,
                    order_engine=self.order_engine,
                    webhook_secret=getattr(settings, 'telegram_webhook_secret', None)
                )
                
                logger.info("✅ Telegram Webhook Handler initialized")
                self.system_health["webhook_handler"] = True
            else:
                logger.warning("⚠️ Webhook Handler not available or dependencies missing")
                self.system_health["webhook_handler"] = False
                
        except Exception as e:
            logger.error(f"❌ Webhook Handler initialization failed: {e}")
            self.system_health["webhook_handler"] = False
    
    async def _setup_interactive_trading(self):
        """Setup interactive trading integration"""
        try:
            if (self.telegram_service and 
                self.order_engine and 
                ENHANCED_TELEGRAM_AVAILABLE):
                
                # Set up approval/rejection callbacks
                self.telegram_service.set_approval_handlers(
                    approval_callback=self._handle_signal_approval,
                    rejection_callback=self._handle_signal_rejection
                )
                
                self.interactive_trading_active = True
                logger.info("✅ Interactive trading integration setup complete")
                
                # Send confirmation message
                if self.telegram_service.is_configured():
                    await self.telegram_service.send_message(
                        "🎯 <b>Interactive Trading System READY</b>\n\n"
                        "✅ Signal approval handlers configured\n"
                        "✅ Order execution engine connected\n"
                        "✅ Risk management active\n\n"
                        "📲 You will receive signals with Approve/Reject buttons during market hours."
                    )
            else:
                logger.warning("⚠️ Interactive trading setup incomplete - missing dependencies")
                self.interactive_trading_active = False
                
        except Exception as e:
            logger.error(f"❌ Interactive trading setup failed: {e}")
            self.interactive_trading_active = False
    
    async def _handle_signal_approval(self, signal_id: str, signal_data: Dict) -> Dict[str, Any]:
        """Handle signal approval and execute order"""
        try:
            logger.info(f"📈 Processing signal approval: {signal_id}")
            
            # Track approval in analytics
            await self.analytics_service.track_signal_approval(approved=True)
            
            if not self.order_engine:
                return {"success": False, "error": "Order engine not available"}
            
            # Extract order details
            symbol = signal_data.get("symbol", "UNKNOWN")
            action = signal_data.get("action", "BUY")
            entry_price = signal_data.get("entry_price", 0.0)
            quantity = signal_data.get("quantity", 1)
            stop_loss = signal_data.get("stop_loss")
            target_price = signal_data.get("target_price")
            
            # Place the order
            order_result = await self.order_engine.place_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=entry_price,
                order_type=OrderType.MARKET,  # Use market orders for immediate execution
                exchange=ExchangeType.NSE,
                product="CNC",  # Cash and Carry for delivery
                stop_loss=stop_loss,
                target=target_price
            )
            
            # Track order execution
            await self.analytics_service.track_order_execution(
                success=order_result.success,
                pnl=0.0  # Will be calculated later
            )
            
            # Broadcast to dashboard
            await self.broadcast_to_dashboard({
                "type": "order_executed",
                "signal_id": signal_id,
                "order_result": {
                    "success": order_result.success,
                    "order_id": order_result.order_id,
                    "status": order_result.status,
                    "message": order_result.message
                },
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": order_result.success,
                "order_id": order_result.order_id,
                "status": order_result.status,
                "message": order_result.message
            }
            
        except Exception as e:
            logger.error(f"❌ Signal approval handling failed: {e}")
            await self.analytics_service.track_order_execution(success=False)
            return {"success": False, "error": str(e)}
    
    async def _handle_signal_rejection(self, signal_id: str, signal_data: Dict):
        """Handle signal rejection"""
        try:
            logger.info(f"❌ Processing signal rejection: {signal_id}")
            
            # Track rejection in analytics
            await self.analytics_service.track_signal_approval(approved=False)
            
            # Broadcast to dashboard
            await self.broadcast_to_dashboard({
                "type": "signal_rejected",
                "signal_id": signal_id,
                "signal_data": signal_data,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Signal rejection handling failed: {e}")
    
    # ... [Rest of the existing methods remain the same] ...
    
    async def _initialize_signal_logging(self):
        """Initialize signal logging"""
        try:
            if SIGNAL_LOGGING_AVAILABLE and SignalLogger:
                self.signal_logger = SignalLogger("logs/enhanced")
                logger.info("✅ Signal logging initialized")
            else:
                logger.warning("⚠️ Using basic logging")
        except Exception as e:
            logger.error(f"❌ Signal logging initialization failed: {e}")
    
    async def _initialize_enhanced_market_data(self):
        """Initialize enhanced market data service"""
        try:
            if ENHANCED_MARKET_DATA_AVAILABLE:
                self.enhanced_market_service = EnhancedMarketDataService()
                await self.enhanced_market_service.initialize()
                
                self.system_health["enhanced_market_data"] = True
                self.system_health["nifty100_universe"] = True
                logger.info("✅ Enhanced Market Data Service initialized")
            else:
                logger.warning("⚠️ Enhanced Market Data Service not available")
                self.system_health["enhanced_market_data"] = False
                self.system_health["nifty100_universe"] = False
        except Exception as e:
            logger.error(f"❌ Enhanced Market Data initialization failed: {e}")
            self.system_health["enhanced_market_data"] = False
            self.system_health["nifty100_universe"] = False
    
    async def _initialize_signal_generator(self):
        """Initialize signal generator"""
        try:
            if (SIGNAL_GENERATOR_AVAILABLE and 
                self.enhanced_market_service and 
                self.signal_logger):
                
                self.signal_generator = ProductionMLSignalGenerator(
                    self.enhanced_market_service, 
                    self.signal_logger
                )
                self.system_health["signal_generation"] = True
                logger.info("✅ Production Signal Generator initialized")
            else:
                logger.warning("⚠️ Signal Generator not available")
                self.system_health["signal_generation"] = False
        except Exception as e:
            logger.error(f"❌ Signal Generator initialization failed: {e}")
            self.system_health["signal_generation"] = False
    
    async def _initialize_regime_detector(self):
        """Initialize regime detector"""
        try:
            if REGIME_DETECTOR_AVAILABLE and self.enhanced_market_service:
                self.regime_detector = RegimeDetector(self.enhanced_market_service)
                self.system_health["regime_detection"] = True
                logger.info("✅ Regime Detector initialized")
            else:
                logger.warning("⚠️ Regime Detector not available")
                self.system_health["regime_detection"] = False
        except Exception as e:
            logger.error(f"❌ Regime Detector initialization failed: {e}")
            self.system_health["regime_detection"] = False
    
    async def _initialize_backtest_engine(self):
        """Initialize backtest engine"""
        try:
            if BACKTEST_ENGINE_AVAILABLE and self.enhanced_market_service:
                self.backtest_engine = BacktestEngine(self.enhanced_market_service)
                self.system_health["backtesting"] = True
                logger.info("✅ Backtest Engine initialized")
            else:
                logger.warning("⚠️ Backtest Engine not available")
                self.system_health["backtesting"] = False
        except Exception as e:
            logger.error(f"❌ Backtest Engine initialization failed: {e}")
            self.system_health["backtesting"] = False
    
    async def _start_market_monitoring(self):
        """Start market monitoring task"""
        try:
            if self.enhanced_market_service:
                self.market_monitor_task = asyncio.create_task(self._market_monitor_loop())
                logger.info("✅ Market monitoring started")
            else:
                logger.warning("⚠️ Market monitoring not available")
        except Exception as e:
            logger.error(f"❌ Market monitoring failed to start: {e}")
    
    async def _start_regime_monitoring(self):
        """Start regime monitoring task"""
        try:
            if self.regime_detector:
                self.regime_monitor_task = asyncio.create_task(self._regime_monitor_loop())
                self.regime_monitoring_active = True
                logger.info("✅ Regime monitoring started")
            else:
                logger.warning("⚠️ Regime monitoring not available")
        except Exception as e:
            logger.error(f"❌ Regime monitoring failed to start: {e}")
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            logger.info("🔍 Performing system health check...")
            
            # Test Telegram connection
            if self.telegram_service and self.telegram_service.is_configured():
                self.system_health["telegram"] = True
            
            # Test order engine connection
            if self.order_engine:
                status = await self.order_engine.get_connection_status()
                self.system_health["zerodha_connection"] = status.get("connected", False)
            
            # Test market data
            if self.enhanced_market_service:
                try:
                    health = await self.enhanced_market_service.get_service_health()
                    self.system_health["enhanced_market_data"] = health.get("status") == "healthy"
                except:
                    pass
            
            logger.info("✅ Health check completed")
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
    
    # WebSocket methods
    async def connect_websocket(self, websocket: WebSocket):
        """Connect a WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send welcome message with enhanced features
        await websocket.send_json({
            "type": "connected",
            "enhanced_features": {
                "interactive_trading": self.interactive_trading_active,
                "order_execution": self.system_health["order_execution"],
                "regime_detection": self.system_health["regime_detection"],
                "backtesting": self.system_health["backtesting"]
            },
            "system_health": self.system_health,
            "timestamp": datetime.now().isoformat()
        })
    
    async def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass
    
    async def broadcast_to_dashboard(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients"""
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                await self.disconnect_websocket(conn)
    
    # Signal generation and processing
    async def start_signal_generation(self):
        """Start signal generation task"""
        if self.signal_generation_active:
            return
        
        self.signal_generation_active = True
        self.signal_generation_task = asyncio.create_task(self._signal_generation_loop())
        logger.info("✅ Signal generation started")
    
    async def stop_signal_generation(self):
        """Stop signal generation task"""
        self.signal_generation_active = False
        if self.signal_generation_task:
            self.signal_generation_task.cancel()
            try:
                await self.signal_generation_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Signal generation stopped")
    
    async def _signal_generation_loop(self):
        """Main signal generation loop"""
        while self.signal_generation_active:
            try:
                await self._check_premarket_analysis_trigger()
                await self._check_priority_trading_trigger()
                await self._check_regular_signal_generation()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Signal generation loop error: {e}")
                await asyncio.sleep(60)
    
    async def _market_monitor_loop(self):
        """Market monitoring loop"""
        while True:
            try:
                if self.enhanced_market_service:
                    # Get market status
                    health = await self.enhanced_market_service.get_service_health()
                    self.current_market_status = health.get('market_status', 'UNKNOWN')
                    self.current_trading_mode = health.get('trading_mode', 'UNKNOWN')
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Market monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _regime_monitor_loop(self):
        """Regime monitoring loop"""
        while self.regime_monitoring_active:
            try:
                if self.regime_detector:
                    # Detect current regime
                    regime_data = await self.regime_detector.detect_current_regime()
                    
                    # Check for regime change
                    new_regime = regime_data.get('regime', MarketRegime.SIDEWAYS_CHOPPY)
                    new_confidence = regime_data.get('confidence', 0.5)
                    
                    if new_regime != self.current_regime:
                        logger.info(f"🔄 Regime change: {self.current_regime} → {new_regime}")
                        self.current_regime = new_regime
                        self.regime_confidence = new_confidence
                        
                        # Broadcast regime change
                        await self.broadcast_to_dashboard({
                            "type": "regime_change",
                            "old_regime": str(self.current_regime),
                            "new_regime": str(new_regime),
                            "confidence": new_confidence,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Notify via Telegram
                        if self.telegram_service and self.telegram_service.is_configured():
                            regime_message = (
                                f"🔄 <b>Market Regime Change Detected</b>\n\n"
                                f"📊 <b>New Regime:</b> {new_regime}\n"
                                f"🎯 <b>Confidence:</b> {new_confidence:.1%}\n"
                                f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}\n\n"
                                f"💡 <i>Adapting trading strategies accordingly...</i>"
                            )
                            await self.telegram_service.send_message(regime_message)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Regime monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _generate_signals(self) -> List[Dict]:
        """Generate trading signals"""
        signals = []
        
        try:
            if self.signal_generator:
                # Use production ML signal generator
                signals = await self.signal_generator.generate_regime_aware_signals(
                    self.current_regime, 
                    self.regime_confidence
                )
            else:
                # Fallback to demo signals with regime awareness
                signals = await self._generate_demo_signals()
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return []
    
    async def _generate_demo_signals(self) -> List[Dict]:
        """Generate demo signals for testing"""
        import random
        
        nifty_stocks = [
            "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
            "ITC", "LT", "SBIN", "BHARTIARTL", "ASIANPAINT", "MARUTI", "HCLTECH",
            "KOTAKBANK", "BAJFINANCE", "WIPRO", "ULTRACEMCO", "TITAN", "TECHM",
            "SUNPHARMA"
        ]
        
        stock = random.choice(nifty_stocks)
        base_price = random.uniform(100, 3000)
        base_confidence = random.uniform(0.6, 0.95)
        
        signal = {
            "symbol": stock,
            "action": random.choice(["BUY", "SELL"]),
            "entry_price": round(base_price, 2),
            "target_price": 0,
            "stop_loss": 0,
            "confidence": round(base_confidence, 3),
            "sentiment_score": round(random.uniform(-0.2, 0.3), 3),
            "timestamp": datetime.now(),
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "signal_type": "ENHANCED_DEMO_REGIME_AWARE",
            "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "stock_universe": "NIFTY_100",
            "regime": str(self.current_regime),
            "regime_confidence": self.regime_confidence,
            "quantity": random.randint(1, 10)  # Add quantity for interactive trading
        }
        
        # Calculate targets with regime-based risk adjustment
        risk_adjustment = self._get_regime_risk_adjustment()
        if signal["action"] == "BUY":
            signal["target_price"] = round(signal["entry_price"] * (1 + 0.025 * risk_adjustment), 2)
            signal["stop_loss"] = round(signal["entry_price"] * (1 - 0.015 * risk_adjustment), 2)
        else:
            signal["target_price"] = round(signal["entry_price"] * (1 - 0.025 * risk_adjustment), 2)
            signal["stop_loss"] = round(signal["entry_price"] * (1 + 0.015 * risk_adjustment), 2)
        
        return [signal]
    
    def _get_regime_risk_adjustment(self) -> float:
        """Get risk adjustment factor based on current regime"""
        regime_adjustments = {
            MarketRegime.BULLISH_TRENDING: 1.2,
            MarketRegime.BEARISH_TRENDING: 0.8,
            MarketRegime.SIDEWAYS_CHOPPY: 1.0,
            MarketRegime.HIGH_VOLATILITY: 0.6
        }
        return regime_adjustments.get(self.current_regime, 1.0)
    
    async def _process_signal(self, signal: Dict, is_priority: bool = False):
        """Process signal with interactive trading capability"""
        try:
            # Add metadata
            signal["is_priority_signal"] = is_priority
            signal["processing_timestamp"] = datetime.now().isoformat()
            signal["current_regime"] = str(self.current_regime)
            signal["regime_confidence"] = self.regime_confidence
            
            # Track in analytics
            await self.analytics_service.track_signal_generated(signal)
            
            # Send for interactive approval or regular notification
            telegram_success = False
            if self.telegram_service and self.telegram_service.is_configured():
                if self.interactive_trading_active and ENHANCED_TELEGRAM_AVAILABLE:
                    # Send with interactive buttons
                    quantity = signal.get("quantity", 1)
                    telegram_success = await self.telegram_service.send_signal_with_approval(
                        signal, quantity
                    )
                else:
                    # Send regular notification
                    telegram_success = await self.telegram_service.send_signal_notification(signal)
                
                await self.analytics_service.track_telegram_sent(telegram_success, signal)
            
            # Broadcast to dashboard
            await self.broadcast_to_dashboard({
                "type": "new_signal",
                "data": signal,
                "is_priority": is_priority,
                "interactive_trading": self.interactive_trading_active,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"📈 Signal processed: {signal['symbol']} {signal['action']} "
                       f"@ ₹{signal['entry_price']} (Interactive: {self.interactive_trading_active})")
            
        except Exception as e:
            logger.error(f"❌ Signal processing failed: {e}")
    
    # ... [Rest of the existing methods for premarket analysis, priority trading, etc.] ...
    
    async def _check_premarket_analysis_trigger(self):
        """Check for pre-market analysis time"""
        now = datetime.now()
        current_time = now.time()
        
        if (now.weekday() < 5 and 
            dt_time(8, 0) <= current_time <= dt_time(9, 15) and
            not self.premarket_analysis_active and
            self.enhanced_market_service):
            
            await self._run_premarket_analysis()
    
    async def _check_priority_trading_trigger(self):
        """Check for priority trading time"""
        now = datetime.now()
        current_time = now.time()
        
        if (now.weekday() < 5 and 
            dt_time(9, 15) <= current_time <= dt_time(9, 45) and
            not self.priority_trading_active):
            
            await self._run_priority_trading()
    
    async def _check_regular_signal_generation(self):
        """Check for regular signal generation"""
        now = datetime.now()
        current_time = now.time()
        
        if (now.weekday() < 5 and 
            dt_time(9, 45) <= current_time <= dt_time(15, 30)):
            
            await self._run_regular_signal_generation()
    
    async def _run_premarket_analysis(self):
        """Run pre-market analysis"""
        try:
            self.premarket_analysis_active = True
            logger.info("🌅 Running pre-market analysis...")
            
            if self.enhanced_market_service:
                analysis_result = await self.enhanced_market_service.run_premarket_analysis()
                self.premarket_opportunities = analysis_result.get("top_opportunities", [])
                
                await self.analytics_service.track_premarket_analysis(len(self.premarket_opportunities))
                
                # Send summary
                if self.telegram_service and self.telegram_service.is_configured():
                    summary_message = (
                        f"🌅 <b>PRE-MARKET ANALYSIS COMPLETE</b>\n\n"
                        f"📊 Opportunities: {analysis_result.get('total_opportunities', 0)}\n"
                        f"🎯 Strong Buy: {analysis_result.get('strong_buy_count', 0)}\n"
                        f"📈 Buy: {analysis_result.get('buy_count', 0)}\n"
                        f"👀 Watch: {analysis_result.get('watch_count', 0)}\n\n"
                        f"🤖 Interactive trading mode: {'ACTIVE' if self.interactive_trading_active else 'DISABLED'}"
                    )
                    await self.telegram_service.send_message(summary_message)
            
            logger.info(f"✅ Pre-market analysis: {len(self.premarket_opportunities)} opportunities")
            
        except Exception as e:
            logger.error(f"❌ Pre-market analysis failed: {e}")
        finally:
            current_time = datetime.now().time()
            if current_time > dt_time(9, 15):
                self.premarket_analysis_active = False
    
    async def _run_priority_trading(self):
        """Run priority trading period"""
        try:
            self.priority_trading_active = True
            logger.info("⚡ Priority trading period active...")
            
            # Generate priority signals more frequently
            for _ in range(3):  # Generate 3 priority signals
                signals = await self._generate_signals()
                for signal in signals:
                    await self._process_signal(signal, is_priority=True)
                
                await asyncio.sleep(10)  # 10 seconds between priority signals
            
        except Exception as e:
            logger.error(f"❌ Priority trading failed: {e}")
        finally:
            current_time = datetime.now().time()
            if current_time > dt_time(9, 45):
                self.priority_trading_active = False
    
    async def _run_regular_signal_generation(self):
        """Run regular signal generation"""
        try:
            # Generate signals based on configured interval
            last_signal_time = getattr(self, '_last_signal_time', None)
            now = datetime.now()
            
            if (last_signal_time is None or 
                (now - last_signal_time).total_seconds() >= settings.signal_generation_interval):
                
                signals = await self._generate_signals()
                for signal in signals:
                    await self._process_signal(signal, is_priority=False)
                
                self._last_signal_time = now
                
        except Exception as e:
            logger.error(f"❌ Regular signal generation failed: {e}")


# ================================================================
# GLOBAL SERVICE MANAGER INSTANCE
# ================================================================

corrected_service_manager = CorrectedServiceManager()

# ================================================================
# FASTAPI APPLICATION LIFESPAN
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    try:
        logger.info("🚀 Starting TradeMind AI Enhanced Edition with Interactive Trading...")
        
        await corrected_service_manager.initialize_all_services()
        await corrected_service_manager.start_signal_generation()
        
        logger.info("✅ TradeMind AI Enhanced Edition fully operational!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 TradeMind AI Enhanced Edition shutting down...")
    
    if (corrected_service_manager.telegram_service and 
        corrected_service_manager.telegram_service.is_configured()):
        try:
            daily_stats = corrected_service_manager.analytics_service.get_daily_stats()
            shutdown_message = (
                "🛑 <b>TradeMind AI Enhanced - SHUTDOWN</b>\n\n"
                "📊 <b>Session Summary:</b>\n"
                f"• Signals generated: {daily_stats['signals_generated']}\n"
                f"• Signals approved: {daily_stats['signals_approved']}\n"
                f"• Signals rejected: {daily_stats['signals_rejected']}\n"
                f"• Orders executed: {daily_stats['orders_executed']}\n"
                f"• Trading P&L: ₹{daily_stats['total_trading_pnl']:.2f}\n"
                f"• Approval rate: {daily_stats['approval_rate']:.1f}%\n"
                f"• Order success rate: {daily_stats['order_success_rate']:.1f}%\n"
                f"• Uptime: {daily_stats['system_uptime_hours']:.1f} hours\n\n"
                "💤 <i>Interactive trading system going offline...</i>"
            )
            await corrected_service_manager.telegram_service.send_system_shutdown_notification()
            await corrected_service_manager.telegram_service.send_message(shutdown_message)
            
            # Close enhanced telegram service
            if hasattr(corrected_service_manager.telegram_service, 'close'):
                await corrected_service_manager.telegram_service.close()
                
            logger.info("📱 Shutdown notification sent")
        except Exception as e:
            logger.error(f"❌ Shutdown notification failed: {e}")
    
    await corrected_service_manager.stop_signal_generation()
    
    # Close other services
    if corrected_service_manager.order_engine:
        try:
            # No explicit close method in order engine, but clean up if needed
            pass
        except Exception as e:
            logger.error(f"Error closing order engine: {e}")
    
    if corrected_service_manager.enhanced_market_service:
        try:
            await corrected_service_manager.enhanced_market_service.close()
        except Exception as e:
            logger.error(f"Error closing enhanced service: {e}")
    
    logger.info("✅ Shutdown complete")

# ================================================================
# FASTAPI APPLICATION
# ================================================================

app = FastAPI(
    title="TradeMind AI - Enhanced Edition with Interactive Trading",
    description="AI-Powered Trading Platform with Interactive Approval & Order Execution",
    version="5.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# ================================================================
# TELEGRAM WEBHOOK ENDPOINTS
# ================================================================

@app.post("/webhook/telegram")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Telegram webhook endpoint for interactive features"""
    try:
        if corrected_service_manager.webhook_handler:
            return await corrected_service_manager.webhook_handler.handle_webhook(request)
        else:
            logger.warning("⚠️ Webhook handler not available")
            return JSONResponse({"ok": False, "error": "Webhook handler not configured"})
    except Exception as e:
        logger.error(f"❌ Webhook handling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webhook/status")
async def webhook_status():
    """Webhook status endpoint"""
    return {
        "status": "active" if corrected_service_manager.webhook_handler else "inactive",
        "interactive_trading": corrected_service_manager.interactive_trading_active,
        "order_engine": corrected_service_manager.system_health.get("order_execution", False),
        "webhook_configured": WEBHOOK_HANDLER_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

# ================================================================
# API ENDPOINTS
# ================================================================

@app.get("/")
async def root():
    """Root endpoint with enhanced system information"""
    return {
        "message": "TradeMind AI Enhanced Edition with Interactive Trading",
        "version": "5.1.0",
        "architecture": "enhanced_v5.1_interactive_trading",
        "features": {
            "interactive_trading": corrected_service_manager.interactive_trading_active,
            "enhanced_telegram": ENHANCED_TELEGRAM_AVAILABLE,
            "order_execution": ZERODHA_ORDER_ENGINE_AVAILABLE,
            "webhook_handler": WEBHOOK_HANDLER_AVAILABLE,
            "regime_detection": REGIME_DETECTOR_AVAILABLE,
            "backtesting": BACKTEST_ENGINE_AVAILABLE,
            "enhanced_market_data": ENHANCED_MARKET_DATA_AVAILABLE,
            "nifty100_universe": True,
            "premarket_analysis": True,
            "priority_trading": True,
            "institutional_grade": True
        },
        "system_health": corrected_service_manager.system_health,
        "configuration": {
            "environment": settings.environment,
            "debug": settings.debug,
            "telegram_configured": settings.is_telegram_configured,
            "zerodha_configured": settings.is_zerodha_configured,
            "max_signals_per_day": settings.max_signals_per_day,
            "signal_interval": settings.signal_generation_interval
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/analytics/performance")
async def get_performance():
    """Get comprehensive performance analytics"""
    try:
        performance = corrected_service_manager.analytics_service.get_performance_summary()
        
        # Add interactive trading specific metrics
        performance["interactive_trading"] = {
            "active": corrected_service_manager.interactive_trading_active,
            "signals_approved": performance["daily"]["signals_approved"],
            "signals_rejected": performance["daily"]["signals_rejected"],
            "orders_executed": performance["daily"]["orders_executed"],
            "approval_rate": performance["daily"]["approval_rate"],
            "order_success_rate": performance["daily"]["order_success_rate"],
            "trading_pnl": performance["daily"]["total_trading_pnl"]
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"❌ Performance analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signals/generate")
async def generate_signals_manually():
    """Generate signals manually with interactive trading"""
    try:
        signals = await corrected_service_manager._generate_signals()
        
        for signal in signals:
            await corrected_service_manager._process_signal(signal, is_priority=False)
        
        return {
            "success": True,
            "signals_generated": len(signals),
            "signals": signals,
            "interactive_trading": corrected_service_manager.interactive_trading_active,
            "current_regime": str(corrected_service_manager.current_regime),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Manual signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def get_system_status():
    """Get detailed system status including trading components"""
    market_status = "UNKNOWN"
    trading_mode = "UNKNOWN"
    
    try:
        if corrected_service_manager.enhanced_market_service:
            health = await corrected_service_manager.enhanced_market_service.get_service_health()
            market_status = health.get('market_status', 'UNKNOWN')
            trading_mode = health.get('trading_mode', 'UNKNOWN')
    except Exception:
        pass
    
    # Get order engine status
    order_engine_status = {}
    if corrected_service_manager.order_engine:
        try:
            order_engine_status = await corrected_service_manager.order_engine.get_connection_status()
        except Exception as e:
            logger.debug(f"Failed to get order engine status: {e}")
    
    return {
        "architecture": "enhanced_v5.1_interactive_trading",
        "version": "5.1.0",
        "initialized": corrected_service_manager.is_initialized,
        "initialization_error": corrected_service_manager.initialization_error,
        "system_health": corrected_service_manager.system_health,
        "signal_generation_active": corrected_service_manager.signal_generation_active,
        "premarket_analysis_active": corrected_service_manager.premarket_analysis_active,
        "priority_trading_active": corrected_service_manager.priority_trading_active,
        "regime_monitoring_active": corrected_service_manager.regime_monitoring_active,
        "interactive_trading_active": corrected_service_manager.interactive_trading_active,
        "active_connections": len(corrected_service_manager.active_connections),
        "market_status": market_status,
        "trading_mode": trading_mode,
        "current_regime": str(corrected_service_manager.current_regime),
        "regime_confidence": corrected_service_manager.regime_confidence,
        "order_engine": order_engine_status,
        "features": {
            "enhanced_telegram": ENHANCED_TELEGRAM_AVAILABLE,
            "order_execution": ZERODHA_ORDER_ENGINE_AVAILABLE,
            "webhook_handler": WEBHOOK_HANDLER_AVAILABLE,
            "enhanced_market_data": ENHANCED_MARKET_DATA_AVAILABLE,
            "configuration": CONFIG_AVAILABLE,
            "nifty100_universe": True,
            "premarket_analysis": True,
            "priority_trading": True,
            "regime_detection": REGIME_DETECTOR_AVAILABLE,
            "backtesting": BACKTEST_ENGINE_AVAILABLE,
            "institutional_grade": True
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with interactive trading status"""
    try:
        performance = corrected_service_manager.analytics_service.get_performance_summary()
        
        market_health = {}
        if corrected_service_manager.enhanced_market_service:
            try:
                market_health = await corrected_service_manager.enhanced_market_service.get_service_health()
            except Exception as e:
                logger.debug(f"Failed to get market health: {e}")
        
        order_health = {}
        if corrected_service_manager.order_engine:
            try:
                order_health = await corrected_service_manager.order_engine.get_connection_status()
            except Exception as e:
                logger.debug(f"Failed to get order engine health: {e}")
        
        return {
            "status": "healthy" if corrected_service_manager.is_initialized else "degraded",
            "architecture": "enhanced_v5.1_interactive_trading",
            "initialization_error": corrected_service_manager.initialization_error,
            "system_health": corrected_service_manager.system_health,
            "connections": len(corrected_service_manager.active_connections),
            "signal_generation_active": corrected_service_manager.signal_generation_active,
            "interactive_trading_active": corrected_service_manager.interactive_trading_active,
            "market_health": market_health,
            "order_engine_health": order_health,
            "performance": performance,
            "current_regime": str(corrected_service_manager.current_regime),
            "regime_confidence": corrected_service_manager.regime_confidence,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ================================================================
# WEBSOCKET ENDPOINTS
# ================================================================

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with interactive trading updates"""
    await corrected_service_manager.connect_websocket(websocket)
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "enhanced": True,
                        "interactive_trading": corrected_service_manager.interactive_trading_active,
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif message.get("type") == "get_status":
                    performance = corrected_service_manager.analytics_service.get_performance_summary()
                    await websocket.send_json({
                        "type": "status_update",
                        "system_health": corrected_service_manager.system_health,
                        "performance": performance,
                        "interactive_trading": corrected_service_manager.interactive_trading_active,
                        "current_regime": str(corrected_service_manager.current_regime),
                        "regime_confidence": corrected_service_manager.regime_confidence,
                        "timestamp": datetime.now().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await corrected_service_manager.disconnect_websocket(websocket)

# ================================================================
# SIGNAL HANDLING
# ================================================================

def shutdown_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    # Set flag for graceful shutdown
    corrected_service_manager.signal_generation_active = False
    corrected_service_manager.regime_monitoring_active = False

# Register signal handlers
signal_handler.signal(signal_handler.SIGINT, shutdown_handler)
signal_handler.signal(signal_handler.SIGTERM, shutdown_handler)

# ================================================================
# STARTUP MESSAGE
# ================================================================

logger.info("=" * 80)
logger.info("🚀 TradeMind AI Enhanced Edition v5.1 with Interactive Trading")
logger.info("=" * 80)
logger.info(f"🎯 Interactive Trading: {'✅ ENABLED' if ENHANCED_TELEGRAM_AVAILABLE else '❌ DISABLED'}")
logger.info(f"📱 Enhanced Telegram: {'✅ AVAILABLE' if ENHANCED_TELEGRAM_AVAILABLE else '❌ NOT AVAILABLE'}")
logger.info(f"💼 Order Execution: {'✅ AVAILABLE' if ZERODHA_ORDER_ENGINE_AVAILABLE else '❌ NOT AVAILABLE'}")
logger.info(f"🌐 Webhook Handler: {'✅ AVAILABLE' if WEBHOOK_HANDLER_AVAILABLE else '❌ NOT AVAILABLE'}")
logger.info(f"📊 Enhanced Market Data: {'✅ AVAILABLE' if ENHANCED_MARKET_DATA_AVAILABLE else '❌ NOT AVAILABLE'}")
logger.info(f"🔧 Configuration: {'✅ LOADED' if CONFIG_AVAILABLE else '⚠️ USING DEFAULTS'}")
logger.info("=" * 80)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
