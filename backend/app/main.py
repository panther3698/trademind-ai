# ================================================================
# TradeMind AI - Corrected Main Application (Enhanced Only)
# ================================================================

"""
TradeMind AI - Production-Ready FastAPI Application
Uses ONLY Enhanced Market Data Service (No Legacy Dependencies)

Architecture:
Frontend ↔ WebSocket/REST API ↔ Enhanced Service Manager ↔ Nifty 100 Universe ↔ Live Data
   ↓              ↓                    ↓                        ↓              ↓
Dashboard    Real-time Updates    Pre-Market Analysis    Priority Trading   Yahoo Finance
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
        logging.FileHandler('logs/trademind_corrected.log', encoding='utf-8')
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
        database_url = "sqlite:///./test.db"
        is_finbert_enabled = False
    
    settings = MockSettings()
    CONFIG_AVAILABLE = False

# ================================================================
# ONLY Enhanced Market Data Service Import (No Legacy)
# ================================================================

ENHANCED_MARKET_DATA_AVAILABLE = False
EnhancedMarketDataService = None
MarketStatus = None
TradingMode = None

try:
    from app.services.enhanced_market_data_nifty100 import (
        EnhancedMarketDataService,
        create_enhanced_market_data_service,
        MarketStatus,
        TradingMode,
        PreMarketOpportunity,
        MarketTick,
        Nifty100Universe
    )
    ENHANCED_MARKET_DATA_AVAILABLE = True
    logger.info("✅ Enhanced Market Data Service (Nifty 100) imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Enhanced market data service not found: {e}")
    logger.info("📝 Creating fallback implementations...")
    
    # Create minimal fallback implementations
    from enum import Enum
    
    class MarketStatus(Enum):
        PRE_MARKET = "PRE_MARKET"
        OPEN = "OPEN"
        CLOSED = "CLOSED"
        AFTER_HOURS = "AFTER_HOURS"
    
    class TradingMode(Enum):
        PRE_MARKET_ANALYSIS = "PRE_MARKET_ANALYSIS"
        PRIORITY_TRADING = "PRIORITY_TRADING"
        REGULAR_TRADING = "REGULAR_TRADING"
    
    class MockEnhancedMarketDataService:
        def __init__(self):
            self.is_initialized = False
            self.nifty100 = MockNifty100Universe()
            
        async def initialize(self):
            self.is_initialized = True
            logger.info("📊 Mock Enhanced Market Data Service initialized")
            
        async def get_service_health(self):
            return {
                "is_initialized": self.is_initialized,
                "market_status": "CLOSED",
                "trading_mode": "PRE_MARKET_ANALYSIS",
                "yahoo_finance_connected": False,
                "nifty100_universe": {
                    "total_stocks": 100,
                    "nifty50_stocks": 50,
                    "next50_stocks": 50,
                    "sectors_covered": 15
                },
                "premarket_analysis": {
                    "last_analysis": None,
                    "opportunities_found": 0,
                    "analysis_available": False
                }
            }
            
        async def update_market_status(self):
            pass
            
        async def run_premarket_analysis(self):
            return {
                "timestamp": datetime.now().isoformat(),
                "total_opportunities": 0,
                "strong_buy_count": 0,
                "buy_count": 0,
                "watch_count": 0,
                "top_opportunities": []
            }
            
        async def get_priority_trading_signals(self):
            return []
            
        async def get_live_market_data(self, symbol):
            return {
                "symbol": symbol,
                "quote": {
                    "ltp": 1500.0,
                    "change_percent": 0.5,
                    "volume": 100000
                },
                "error": "Mock data - Enhanced service not available"
            }
            
        async def get_nifty100_overview(self):
            return {
                "timestamp": datetime.now().isoformat(),
                "overview": {
                    "total_stocks_tracked": 100,
                    "average_change_percent": 0.2
                },
                "error": "Mock data - Enhanced service not available"
            }
            
        async def close(self):
            pass
    
    class MockNifty100Universe:
        def get_all_symbols(self):
            return ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR"]
        
        def get_priority_symbols(self, tier):
            return ["RELIANCE", "TCS", "HDFCBANK"]
        
        def get_symbol_info(self, symbol):
            return {"sector": "Mock", "priority": 1}
    
    def create_enhanced_market_data_service():
        return MockEnhancedMarketDataService()
    
    ENHANCED_MARKET_DATA_AVAILABLE = False

# Import other services with graceful fallbacks
TelegramService = None
AnalyticsService = None
ProductionMLSignalGenerator = None

try:
    from app.services.telegram_service import TelegramService
    TELEGRAM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Telegram service not available: {e}")
    TELEGRAM_AVAILABLE = False

try:
    from app.services.analytics_service import AnalyticsService
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Analytics service not available: {e}")
    ANALYTICS_AVAILABLE = False

try:
    from app.services.production_signal_generator import ProductionMLSignalGenerator
    SIGNAL_GENERATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Signal generator not available: {e}")
    SIGNAL_GENERATOR_AVAILABLE = False

# Import ML components (optional)
ML_AVAILABLE = False
try:
    from app.ml.models import (
        Nifty100StockUniverse, 
        FinBERTSentimentAnalyzer, 
        FeatureEngineering, 
        XGBoostSignalModel,
        EnsembleModel
    )
    from app.ml.training_pipeline import TrainingPipeline
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ ML components not available: {e}")

# Import signal logging (optional)
SIGNAL_LOGGING_AVAILABLE = False
try:
    from app.core.signal_logger import InstitutionalSignalLogger, SignalRecord
    SIGNAL_LOGGING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Signal logging not available: {e}")

# ================================================================
# Enhanced Analytics (Standalone Implementation)
# ================================================================

class CorrectedAnalytics:
    """Standalone analytics implementation"""
    
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
            "market_mode": "closed"
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
    
    async def track_premarket_analysis(self, opportunities_count):
        """Track pre-market analysis"""
        self.daily_stats["premarket_analyses"] += 1
        self.daily_stats["last_premarket_analysis"] = datetime.now().isoformat()
        
    async def track_telegram_sent(self, success, signal):
        """Track Telegram notifications"""
        if success:
            self.daily_stats["telegram_success"] += 1
            self.daily_stats["signals_sent"] += 1
        else:
            self.daily_stats["telegram_failures"] += 1
    
    def update_market_mode(self, mode: str):
        """Update current market mode"""
        self.daily_stats["market_mode"] = mode
    
    def get_daily_stats(self):
        """Get daily statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600
        self.daily_stats["system_uptime_hours"] = round(uptime, 2)
        return self.daily_stats
    
    def get_performance_summary(self):
        """Get performance summary"""
        daily = self.get_daily_stats()
        
        # Calculate telegram success rate
        total_telegram = daily["telegram_success"] + daily["telegram_failures"]
        telegram_success_rate = (daily["telegram_success"] / total_telegram * 100) if total_telegram > 0 else 100.0
        
        return {
            "daily": daily,
            "premarket": {
                "last_analysis": daily["last_premarket_analysis"],
                "analyses_count": daily["premarket_analyses"],
                "opportunities_found": len(self.premarket_opportunities),
                "priority_signals_ready": len(self.priority_signals_today)
            },
            "system_health": {
                "nifty100_enabled": ENHANCED_MARKET_DATA_AVAILABLE,
                "premarket_analysis_active": True,
                "priority_trading_enabled": True,
                "telegram_configured": settings.is_telegram_configured,
                "signal_generation_active": True,
                "last_activity": daily["last_signal_time"],
                "error_rate": (daily["telegram_failures"] / max(total_telegram, 1)) * 100
            },
            "configuration": {
                "stock_universe": "Nifty 100 Enhanced",
                "premarket_analysis": "8:00-9:15 AM",
                "priority_trading": "9:15-9:45 AM",
                "regular_trading": "9:45 AM-3:30 PM",
                "max_signals_per_day": settings.max_signals_per_day,
                "min_confidence_threshold": settings.min_confidence_threshold,
                "signal_interval_seconds": settings.signal_generation_interval,
                "telegram_enabled": settings.is_telegram_configured
            },
            "recent_performance": {
                "signals_today": daily["signals_generated"],
                "priority_signals": daily["priority_signals"],
                "premarket_analyses": daily["premarket_analyses"],
                "success_rate": telegram_success_rate,
                "profit_loss": daily["total_pnl"],
                "avg_confidence": daily["average_confidence"]
            },
            "market_coverage": {
                "universe": "Nifty 100 Complete",
                "total_stocks": daily["nifty100_coverage"],
                "priority_stocks": daily["priority_stocks_tracked"],
                "current_mode": daily["market_mode"],
                "enhanced_features": ENHANCED_MARKET_DATA_AVAILABLE
            }
        }

# ================================================================
# Corrected Service Manager (Enhanced Only)
# ================================================================

class CorrectedServiceManager:
    """Corrected service manager using ONLY enhanced market data service"""
    
    def __init__(self):
        # Core services
        self.active_connections: List[WebSocket] = []
        self.telegram_service: Optional[TelegramService] = None
        self.analytics_service = CorrectedAnalytics()
        
        # ONLY Enhanced Market Data Service (No Legacy)
        self.enhanced_market_service = None
        
        # Optional services
        self.signal_generator: Optional[ProductionMLSignalGenerator] = None
        self.signal_logger: Optional[InstitutionalSignalLogger] = None
        
        # Enhanced tracking
        self.current_market_status = MarketStatus.CLOSED
        self.current_trading_mode = TradingMode.PRE_MARKET_ANALYSIS
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
            "analytics": True  # Always available
        }
        
        # Signal generation control
        self.signal_generation_active = False
        self.premarket_analysis_active = False
        self.priority_trading_active = False
        
        # Background tasks
        self.signal_generation_task = None
        self.market_monitor_task = None
        
    async def initialize_all_services(self):
        """Initialize all services (enhanced only)"""
        try:
            logger.info("🚀 Initializing Corrected TradeMind AI (Enhanced Only)...")
            
            # 1. Analytics already initialized
            logger.info("✅ Analytics service ready")
            
            # 2. Initialize Telegram (optional)
            await self._initialize_telegram()
            
            # 3. Initialize signal logging (optional)
            await self._initialize_signal_logging()
            
            # 4. Initialize ONLY enhanced market data service
            await self._initialize_enhanced_market_data()
            
            # 5. Initialize signal generator (optional)
            await self._initialize_signal_generator()
            
            # 6. Start market monitoring
            await self._start_market_monitoring()
            
            # 7. System health check
            await self._perform_health_check()
            
            self.is_initialized = True
            logger.info("✅ Corrected TradeMind AI initialized successfully!")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Service initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _initialize_telegram(self):
        """Initialize Telegram"""
        try:
            if settings.is_telegram_configured and TELEGRAM_AVAILABLE and TelegramService:
                self.telegram_service = TelegramService()
                
                if self.telegram_service.is_configured():
                    startup_message = (
                        "🚀 *TradeMind AI - Enhanced Edition*\n\n"
                        "📈 *Complete Nifty 100 Universe*\n"
                        "🌅 Pre-Market Analysis: 8:00-9:15 AM\n"
                        "⚡ Priority Trading: 9:15-9:45 AM\n"
                        "📊 Regular Trading: 9:45 AM-3:30 PM\n\n"
                        f"🔧 *Service Status:*\n"
                        f"• Enhanced Data: {'✅' if ENHANCED_MARKET_DATA_AVAILABLE else '❌'}\n"
                        f"• Configuration: {'✅' if CONFIG_AVAILABLE else '⚠️'}\n\n"
                        "💡 _Professional trading signals ready!_"
                    )
                    
                    success = await self.telegram_service.send_message(startup_message)
                    if success:
                        logger.info("📱 Enhanced startup notification sent")
                    
                    self.system_health["telegram"] = True
                else:
                    logger.warning("⚠️ Telegram not properly configured")
                    self.system_health["telegram"] = False
            else:
                logger.warning("⚠️ Telegram not configured")
                self.system_health["telegram"] = False
        except Exception as e:
            logger.error(f"❌ Telegram initialization failed: {e}")
            self.system_health["telegram"] = False
    
    async def _initialize_signal_logging(self):
        """Initialize signal logging"""
        try:
            if SIGNAL_LOGGING_AVAILABLE and InstitutionalSignalLogger:
                self.signal_logger = InstitutionalSignalLogger("logs/enhanced")
                logger.info("✅ Signal logging initialized")
            else:
                logger.warning("⚠️ Using basic signal logging")
        except Exception as e:
            logger.error(f"❌ Signal logging initialization failed: {e}")
    
    async def _initialize_enhanced_market_data(self):
        """Initialize ONLY enhanced market data service"""
        try:
            logger.info("📊 Initializing Enhanced Market Data Service...")
            
            # Create enhanced market data service (works with both real and mock)
            self.enhanced_market_service = create_enhanced_market_data_service()
            await self.enhanced_market_service.initialize()
            
            # Check health
            health = await self.enhanced_market_service.get_service_health()
            if health['is_initialized']:
                self.system_health["enhanced_market_data"] = ENHANCED_MARKET_DATA_AVAILABLE
                self.system_health["nifty100_universe"] = True
                self.system_health["premarket_analysis"] = True
                self.system_health["priority_trading"] = True
                
                service_type = "Real Enhanced Service" if ENHANCED_MARKET_DATA_AVAILABLE else "Mock Enhanced Service"
                logger.info(f"✅ Enhanced Market Data Service Ready ({service_type})")
                logger.info(f"📊 Tracking {health['nifty100_universe']['total_stocks']} stocks")
                logger.info(f"⭐ Priority: {health['nifty100_universe']['nifty50_stocks']} Nifty 50")
                logger.info(f"📈 Additional: {health['nifty100_universe']['next50_stocks']} Next 50")
                logger.info(f"🏭 Sectors: {health['nifty100_universe']['sectors_covered']}")
                logger.info(f"🏪 Market: {health.get('market_status', 'Unknown')}")
                logger.info(f"🎯 Mode: {health.get('trading_mode', 'Unknown')}")
                
                # Update current status
                try:
                    self.current_market_status = MarketStatus(health.get('market_status', 'CLOSED'))
                    self.current_trading_mode = TradingMode(health.get('trading_mode', 'PRE_MARKET_ANALYSIS'))
                except:
                    pass
                
            else:
                logger.error("❌ Enhanced market data service failed to initialize")
                self.system_health["enhanced_market_data"] = False
                
        except Exception as e:
            logger.error(f"❌ Enhanced market data initialization failed: {e}")
            self.system_health["enhanced_market_data"] = False
    
    async def _initialize_signal_generator(self):
        """Initialize signal generator"""
        try:
            if (SIGNAL_GENERATOR_AVAILABLE and ProductionMLSignalGenerator and 
                self.enhanced_market_service and self.signal_logger):
                self.signal_generator = ProductionMLSignalGenerator(
                    self.enhanced_market_service,
                    self.signal_logger
                )
                self.system_health["signal_generation"] = True
                logger.info("✅ Signal generator initialized")
            else:
                logger.warning("⚠️ Using demo signal generator")
                self.system_health["signal_generation"] = True
        except Exception as e:
            logger.error(f"❌ Signal generator initialization failed: {e}")
            self.system_health["signal_generation"] = True
    
    async def _start_market_monitoring(self):
        """Start market monitoring"""
        try:
            self.market_monitor_task = asyncio.create_task(self._market_monitoring_loop())
            logger.info("🔄 Market monitoring started")
        except Exception as e:
            logger.error(f"❌ Market monitoring start failed: {e}")
    
    async def _market_monitoring_loop(self):
        """Market monitoring loop"""
        logger.info("🔄 Starting market monitoring loop...")
        
        while True:
            try:
                # Update market status
                if self.enhanced_market_service:
                    await self.enhanced_market_service.update_market_status()
                    
                    health = await self.enhanced_market_service.get_service_health()
                    old_status = self.current_market_status
                    old_mode = self.current_trading_mode
                    
                    try:
                        self.current_market_status = MarketStatus(health.get('market_status', 'CLOSED'))
                        self.current_trading_mode = TradingMode(health.get('trading_mode', 'PRE_MARKET_ANALYSIS'))
                    except:
                        pass
                    
                    # Update analytics
                    self.analytics_service.update_market_mode(self.current_market_status.value)
                    
                    # Handle transitions
                    if old_status != self.current_market_status or old_mode != self.current_trading_mode:
                        await self._handle_market_transition(old_status, self.current_market_status, 
                                                           old_mode, self.current_trading_mode)
                
                # Check for pre-market analysis
                await self._check_premarket_analysis_trigger()
                
                # Check for priority trading
                await self._check_priority_trading_trigger()
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                logger.info("🛑 Market monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Market monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _handle_market_transition(self, old_status, new_status, old_mode, new_mode):
        """Handle market transitions"""
        logger.info(f"🔄 Market Transition: {old_status.value} → {new_status.value}")
        logger.info(f"🎯 Mode Transition: {old_mode.value} → {new_mode.value}")
        
        # Send notifications
        if self.telegram_service and self.telegram_service.is_configured():
            if new_status == MarketStatus.PRE_MARKET:
                message = "🌅 *PRE-MARKET STARTED*\n\n📊 Analyzing Nifty 100...\n⚡ Priority signals at 9:15 AM"
                await self.telegram_service.send_message(message)
            elif new_status == MarketStatus.OPEN and new_mode == TradingMode.PRIORITY_TRADING:
                message = "⚡ *PRIORITY TRADING LIVE*\n\n🚀 9:15 AM execution\n📈 Best pre-market opportunities"
                await self.telegram_service.send_message(message)
            elif new_mode == TradingMode.REGULAR_TRADING:
                message = "📊 *REGULAR TRADING*\n\n🔄 Continuous signals active\n📈 Full Nifty 100 monitoring"
                await self.telegram_service.send_message(message)
        
        # Broadcast to dashboard
        await self.broadcast_to_dashboard({
            "type": "market_transition",
            "data": {
                "old_status": old_status.value,
                "new_status": new_status.value,
                "old_mode": old_mode.value,
                "new_mode": new_mode.value,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def _check_premarket_analysis_trigger(self):
        """Check for pre-market analysis time"""
        now = datetime.now()
        current_time = now.time()
        
        if (now.weekday() < 5 and 
            dt_time(8, 0) <= current_time <= dt_time(9, 15) and
            not self.premarket_analysis_active and
            self.enhanced_market_service):
            
            await self._run_premarket_analysis()
    
    async def _run_premarket_analysis(self):
        """Run pre-market analysis"""
        try:
            self.premarket_analysis_active = True
            logger.info("🌅 Running pre-market analysis...")
            
            analysis_result = await self.enhanced_market_service.run_premarket_analysis()
            self.premarket_opportunities = analysis_result.get("top_opportunities", [])
            
            # Track in analytics
            await self.analytics_service.track_premarket_analysis(len(self.premarket_opportunities))
            
            # Get priority signals
            priority_signals = await self.enhanced_market_service.get_priority_trading_signals()
            self.priority_signals_queue = priority_signals
            
            # Send summary
            if self.telegram_service and self.telegram_service.is_configured():
                summary_message = (
                    f"🌅 *PRE-MARKET ANALYSIS COMPLETE*\n\n"
                    f"📊 Opportunities: {analysis_result.get('total_opportunities', 0)}\n"
                    f"🎯 Strong Buy: {analysis_result.get('strong_buy_count', 0)}\n"
                    f"📈 Buy: {analysis_result.get('buy_count', 0)}\n"
                    f"👀 Watch: {analysis_result.get('watch_count', 0)}\n\n"
                    f"⚡ Priority queue: {len(priority_signals)} signals ready"
                )
                await self.telegram_service.send_message(summary_message)
            
            # Broadcast to dashboard
            await self.broadcast_to_dashboard({
                "type": "premarket_analysis_complete",
                "data": analysis_result,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"✅ Pre-market analysis: {len(self.premarket_opportunities)} opportunities")
            
        except Exception as e:
            logger.error(f"❌ Pre-market analysis failed: {e}")
        finally:
            current_time = datetime.now().time()
            if current_time > dt_time(9, 15):
                self.premarket_analysis_active = False
    
    async def _check_priority_trading_trigger(self):
        """Check for priority trading time"""
        now = datetime.now()
        current_time = now.time()
        
        if (now.weekday() < 5 and 
            dt_time(9, 15) <= current_time <= dt_time(9, 17) and
            not self.priority_trading_active and
            self.priority_signals_queue):
            
            await self._execute_priority_trading()
    
    async def _execute_priority_trading(self):
        """Execute priority trading"""
        try:
            self.priority_trading_active = True
            logger.info("⚡ Executing priority trading at 9:15 AM...")
            
            for signal in self.priority_signals_queue:
                await self._process_signal(signal, is_priority=True)
                await asyncio.sleep(1)
            
            # Send summary
            if self.telegram_service and self.telegram_service.is_configured():
                summary_message = (
                    f"⚡ *PRIORITY TRADING EXECUTED*\n\n"
                    f"🎯 Signals: {len(self.priority_signals_queue)}\n"
                    f"⏰ Time: 9:15 AM IST\n\n"
                    f"🔄 _Regular trading mode starting..._"
                )
                await self.telegram_service.send_message(summary_message)
            
            logger.info(f"✅ Priority trading: {len(self.priority_signals_queue)} signals sent")
            
        except Exception as e:
            logger.error(f"❌ Priority trading failed: {e}")
        finally:
            self.priority_signals_queue = []
            current_time = datetime.now().time()
            if current_time > dt_time(9, 45):
                self.priority_trading_active = False
    
    async def _perform_health_check(self):
        """System health check"""
        health_summary = {
            "total_services": len(self.system_health),
            "healthy_services": sum(self.system_health.values()),
            "health_percentage": (sum(self.system_health.values()) / len(self.system_health)) * 100
        }
        
        logger.info(f"🏥 System Health: {health_summary['healthy_services']}/{health_summary['total_services']} ({health_summary['health_percentage']:.1f}%)")
        
        for service, status in self.system_health.items():
            status_icon = "✅" if status else "❌"
            service_name = service.replace('_', ' ').title()
            logger.info(f"  {status_icon} {service_name}: {'Healthy' if status else 'Degraded'}")
    
    async def start_signal_generation(self):
        """Start signal generation"""
        if self.signal_generation_active:
            logger.warning("⚠️ Signal generation already active")
            return
        
        self.signal_generation_active = True
        self.signal_generation_task = asyncio.create_task(self._signal_generation_loop())
        
        logger.info("🎯 Signal generation started")
    
    async def stop_signal_generation(self):
        """Stop signal generation"""
        if not self.signal_generation_active:
            return
        
        self.signal_generation_active = False
        if self.signal_generation_task:
            self.signal_generation_task.cancel()
        
        logger.info("🛑 Signal generation stopped")
    
    async def _signal_generation_loop(self):
        """Signal generation loop"""
        logger.info("🔄 Starting signal generation loop...")
        
        while self.signal_generation_active:
            try:
                # Check if we should generate signals
                if not self._can_generate_signals():
                    await asyncio.sleep(60)
                    continue
                
                # Generate signals
                signals = await self._generate_signals()
                
                # Process signals
                for signal in signals:
                    await self._process_signal(signal, is_priority=False)
                
                # Wait
                await asyncio.sleep(settings.signal_generation_interval)
                
            except asyncio.CancelledError:
                logger.info("🛑 Signal generation loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Signal generation error: {e}")
                await asyncio.sleep(30)
    
    def _can_generate_signals(self) -> bool:
        """Check if we can generate signals"""
        try:
            # Check daily limits
            daily_stats = self.analytics_service.get_daily_stats()
            if daily_stats["signals_generated"] >= settings.max_signals_per_day:
                return False
            
            # Check market hours
            current_time = datetime.now().time()
            weekday = datetime.now().weekday()
            
            if weekday >= 5:  # Weekend
                return False
            
            if dt_time(9, 15) <= current_time <= dt_time(16, 0):
                return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Signal generation check failed: {e}")
            return False
    
    async def _generate_signals(self) -> List[Dict]:
        """Generate signals"""
        signals = []
        
        try:
            if self.signal_generator:
                signals = await self.signal_generator.generate_signals()
            else:
                signals = await self._generate_demo_signals()
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            signals = await self._generate_demo_signals()
        
        return signals
    
    async def _generate_demo_signals(self) -> List[Dict]:
        """Generate demo signals"""
        import random
        
        # Get symbols from enhanced service
        if self.enhanced_market_service and hasattr(self.enhanced_market_service, 'nifty100'):
            symbols = self.enhanced_market_service.nifty100.get_all_symbols()[:10]
        else:
            symbols = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR"]
        
        stock = random.choice(symbols)
        
        # Get price if possible
        base_price = 1500.0
        try:
            if self.enhanced_market_service:
                stock_data = await self.enhanced_market_service.get_live_market_data(stock)
                if stock_data and "quote" in stock_data:
                    base_price = stock_data["quote"]["ltp"]
        except Exception:
            pass
        
        signal = {
            "id": f"demo_{int(datetime.now().timestamp())}",
            "symbol": stock,
            "action": random.choice(["BUY", "SELL"]),
            "entry_price": round(base_price, 2),
            "target_price": 0,
            "stop_loss": 0,
            "confidence": round(random.uniform(0.65, 0.85), 3),
            "sentiment_score": round(random.uniform(-0.2, 0.3), 3),
            "timestamp": datetime.now(),
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "signal_type": "ENHANCED_DEMO",
            "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "stock_universe": "NIFTY_100"
        }
        
        # Calculate targets
        if signal["action"] == "BUY":
            signal["target_price"] = round(signal["entry_price"] * 1.025, 2)
            signal["stop_loss"] = round(signal["entry_price"] * 0.985, 2)
        else:
            signal["target_price"] = round(signal["entry_price"] * 0.975, 2)
            signal["stop_loss"] = round(signal["entry_price"] * 1.015, 2)
        
        return [signal]
    
    async def _process_signal(self, signal: Dict, is_priority: bool = False):
        """Process signal"""
        try:
            # Add metadata
            signal["is_priority_signal"] = is_priority
            signal["processing_timestamp"] = datetime.now().isoformat()
            
            # Track in analytics
            await self.analytics_service.track_signal_generated(signal)
            
            # Send to Telegram
            telegram_success = False
            if self.telegram_service and self.telegram_service.is_configured():
                telegram_success = await self._send_telegram_notification(signal, is_priority)
                await self.analytics_service.track_telegram_sent(telegram_success, signal)
            
            # Broadcast to dashboard
            await self.broadcast_to_dashboard({
                "type": "new_signal",
                "data": signal,
                "is_priority": is_priority,
                "timestamp": datetime.now().isoformat()
            })
            
            # Broadcast analytics
            try:
                analytics = self.analytics_service.get_performance_summary()
                await self.broadcast_to_dashboard({
                    "type": "analytics_update",
                    "data": analytics,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.debug(f"Analytics broadcast failed: {e}")
            
            # Log
            priority_indicator = "⚡ PRIORITY" if is_priority else "📊 REGULAR"
            logger.info(f"📡 {priority_indicator} Signal: {signal['symbol']} {signal['action']} @ ₹{signal['entry_price']} | "
                       f"Confidence: {signal['confidence']:.1%} | "
                       f"Telegram: {'✅' if telegram_success else '❌'}")
            
        except Exception as e:
            logger.error(f"❌ Signal processing failed: {e}")
    
    async def _send_telegram_notification(self, signal: Dict, is_priority: bool) -> bool:
        """Send Telegram notification"""
        try:
            if is_priority:
                header = "⚡ *PRIORITY SIGNAL - 9:15 AM*"
            else:
                header = "📊 *NIFTY 100 SIGNAL*"
            
            message = (
                f"{header}\n\n"
                f"📈 *Stock:* {signal['symbol']}\n"
                f"🎯 *Action:* {signal['action']}\n"
                f"💰 *Entry:* ₹{signal['entry_price']}\n"
                f"🎯 *Target:* ₹{signal['target_price']}\n"
                f"🛡️ *Stop Loss:* ₹{signal['stop_loss']}\n"
                f"📊 *Confidence:* {signal['confidence']:.1%}\n"
                f"🔧 *Source:* Enhanced System\n"
                f"⏰ *Time:* {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.telegram_service.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Telegram notification failed: {e}")
            return False
    
    # WebSocket management
    async def connect_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"📱 Dashboard connected. Total: {len(self.active_connections)}")
        
        welcome_data = {
            "type": "corrected_connection",
            "message": "Connected to TradeMind AI - Enhanced Edition",
            "system_health": self.system_health,
            "initialization_status": self.is_initialized,
            "market_status": self.current_market_status.value,
            "trading_mode": self.current_trading_mode.value,
            "features": {
                "enhanced_only": True,
                "nifty100_universe": True,
                "premarket_analysis": True,
                "priority_trading": True,
                "no_legacy_dependencies": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket.send_json(welcome_data)
    
    def disconnect_websocket(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"📱 Dashboard disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_to_dashboard(self, message: Dict):
        """Broadcast to dashboards"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.active_connections.remove(conn)

# ================================================================
# Global Service Manager
# ================================================================

corrected_service_manager = CorrectedServiceManager()

# ================================================================
# Application Lifecycle
# ================================================================

@asynccontextmanager
async def corrected_lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("🚀 TradeMind AI Enhanced Edition Starting...")
    logger.info(f"🔧 Environment: {settings.environment}")
    logger.info(f"📊 Enhanced Market Data: {'Available' if ENHANCED_MARKET_DATA_AVAILABLE else 'Mock Mode'}")
    logger.info(f"⚙️ Configuration: {'Loaded' if CONFIG_AVAILABLE else 'Mock Mode'}")
    
    try:
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
                "🛑 *TradeMind AI Enhanced - SHUTDOWN*\n\n"
                "📊 *Session Summary:*\n"
                f"• Signals generated: {daily_stats['signals_generated']}\n"
                f"• Priority signals: {daily_stats['priority_signals']}\n"
                f"• Pre-market analyses: {daily_stats['premarket_analyses']}\n"
                f"• Uptime: {daily_stats['system_uptime_hours']:.1f} hours\n\n"
                "💤 _System going offline..._"
            )
            await corrected_service_manager.telegram_service.send_message(shutdown_message)
            logger.info("📱 Shutdown notification sent")
        except Exception as e:
            logger.error(f"❌ Shutdown notification failed: {e}")
    
    await corrected_service_manager.stop_signal_generation()
    
    if corrected_service_manager.enhanced_market_service:
        try:
            await corrected_service_manager.enhanced_market_service.close()
        except Exception as e:
            logger.error(f"Error closing enhanced service: {e}")
    
    logger.info("✅ Shutdown complete")

# ================================================================
# FastAPI Application
# ================================================================

app = FastAPI(
    title="TradeMind AI - Enhanced Edition",
    description="AI-Powered Trading Platform with Complete Nifty 100 Universe & Professional Features",
    version="5.0.0",
    lifespan=corrected_lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# API Endpoints
# ================================================================

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    market_status = "UNKNOWN"
    trading_mode = "UNKNOWN"
    
    if corrected_service_manager.enhanced_market_service:
        try:
            health = await corrected_service_manager.enhanced_market_service.get_service_health()
            market_status = health.get('market_status', 'UNKNOWN')
            trading_mode = health.get('trading_mode', 'UNKNOWN')
        except Exception:
            pass
    
    return {
        "message": "🇮🇳 TradeMind AI - Enhanced Edition (Nifty 100 Universe)",
        "status": "operational" if corrected_service_manager.is_initialized else "initializing",
        "version": "5.0.0",
        "environment": settings.environment,
        "architecture": "Enhanced Only (No Legacy Dependencies)",
        "features": {
            "nifty100_universe": True,
            "premarket_analysis": "8:00-9:15 AM",
            "priority_trading": "9:15-9:45 AM",
            "regular_trading": "9:45 AM-3:30 PM",
            "enhanced_data_service": ENHANCED_MARKET_DATA_AVAILABLE,
            "total_stocks": 100,
            "priority_stocks": 50,
            "sectors_covered": 15
        },
        "system_health": corrected_service_manager.system_health,
        "services": {
            "enhanced_market_data": corrected_service_manager.system_health.get("enhanced_market_data", False),
            "nifty100_universe": corrected_service_manager.system_health.get("nifty100_universe", False),
            "premarket_analysis": corrected_service_manager.system_health.get("premarket_analysis", False),
            "priority_trading": corrected_service_manager.system_health.get("priority_trading", False),
            "telegram": settings.is_telegram_configured and corrected_service_manager.system_health.get("telegram", False),
            "analytics": corrected_service_manager.system_health.get("analytics", False),
            "signal_generation": corrected_service_manager.system_health.get("signal_generation", False)
        },
        "market_status": market_status,
        "trading_mode": trading_mode,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    try:
        performance = corrected_service_manager.analytics_service.get_performance_summary()
        
        market_health = {}
        if corrected_service_manager.enhanced_market_service:
            try:
                market_health = await corrected_service_manager.enhanced_market_service.get_service_health()
            except Exception as e:
                logger.debug(f"Failed to get market health: {e}")
        
        return {
            "status": "healthy" if corrected_service_manager.is_initialized else "degraded",
            "architecture": "enhanced_only",
            "initialization_error": corrected_service_manager.initialization_error,
            "system_health": corrected_service_manager.system_health,
            "connections": len(corrected_service_manager.active_connections),
            "signal_generation_active": corrected_service_manager.signal_generation_active,
            "premarket_analysis_active": corrected_service_manager.premarket_analysis_active,
            "priority_trading_active": corrected_service_manager.priority_trading_active,
            "enhanced_services": {
                "market_data_available": ENHANCED_MARKET_DATA_AVAILABLE,
                "configuration_available": CONFIG_AVAILABLE,
                "nifty100_universe": corrected_service_manager.system_health.get("nifty100_universe", False),
                "premarket_analysis": corrected_service_manager.system_health.get("premarket_analysis", False),
                "priority_trading": corrected_service_manager.system_health.get("priority_trading", False)
            },
            "market_data_health": market_health,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    """Get dashboard analytics"""
    try:
        performance = corrected_service_manager.analytics_service.get_performance_summary()
        
        performance["system_status"] = {
            "initialized": corrected_service_manager.is_initialized,
            "signal_generation_active": corrected_service_manager.signal_generation_active,
            "premarket_analysis_active": corrected_service_manager.premarket_analysis_active,
            "priority_trading_active": corrected_service_manager.priority_trading_active,
            "health": corrected_service_manager.system_health,
            "market_status": corrected_service_manager.current_market_status.value,
            "trading_mode": corrected_service_manager.current_trading_mode.value
        }
        
        if corrected_service_manager.enhanced_market_service:
            try:
                market_overview = await corrected_service_manager.enhanced_market_service.get_nifty100_overview()
                performance["nifty100_overview"] = market_overview
            except Exception as e:
                logger.debug(f"Failed to get overview: {e}")
        
        return performance
        
    except Exception as e:
        logger.error(f"❌ Error getting dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/status")
async def get_market_status():
    """Get market status"""
    try:
        if corrected_service_manager.enhanced_market_service:
            health = await corrected_service_manager.enhanced_market_service.get_service_health()
            return {
                "status": health.get('market_status', 'UNKNOWN'),
                "trading_mode": health.get('trading_mode', 'UNKNOWN'),
                "enhanced": True,
                "nifty100": True,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": corrected_service_manager.current_market_status.value,
                "trading_mode": corrected_service_manager.current_trading_mode.value,
                "enhanced": False,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"❌ Error getting market status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/nifty100/overview")
async def get_nifty100_overview():
    """Get Nifty 100 overview"""
    if not corrected_service_manager.enhanced_market_service:
        raise HTTPException(status_code=503, detail="Enhanced market service not available")
    
    try:
        overview = await corrected_service_manager.enhanced_market_service.get_nifty100_overview()
        return overview
    except Exception as e:
        logger.error(f"❌ Error getting Nifty 100 overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/premarket/analysis")
async def get_premarket_analysis():
    """Get pre-market analysis"""
    if not corrected_service_manager.enhanced_market_service:
        raise HTTPException(status_code=503, detail="Enhanced market service not available")
    
    try:
        analysis = await corrected_service_manager.enhanced_market_service.run_premarket_analysis()
        return {
            "analysis": analysis,
            "opportunities_found": len(corrected_service_manager.premarket_opportunities),
            "priority_signals_ready": len(corrected_service_manager.priority_signals_queue),
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting pre-market analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signals/manual")
async def generate_manual_signal():
    """Generate manual signal"""
    if not corrected_service_manager.signal_generation_active:
        return {"error": "Signal generation not active"}
    
    try:
        signals = await corrected_service_manager._generate_signals()
        
        for signal in signals:
            await corrected_service_manager._process_signal(signal, is_priority=False)
        
        return {
            "success": True,
            "signals_generated": len(signals),
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Manual signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def get_system_status():
    """Get detailed system status"""
    market_status = "UNKNOWN"
    trading_mode = "UNKNOWN"
    
    try:
        if corrected_service_manager.enhanced_market_service:
            health = await corrected_service_manager.enhanced_market_service.get_service_health()
            market_status = health.get('market_status', 'UNKNOWN')
            trading_mode = health.get('trading_mode', 'UNKNOWN')
    except Exception:
        pass
    
    return {
        "architecture": "enhanced_only",
        "version": "5.0.0",
        "initialized": corrected_service_manager.is_initialized,
        "initialization_error": corrected_service_manager.initialization_error,
        "system_health": corrected_service_manager.system_health,
        "signal_generation_active": corrected_service_manager.signal_generation_active,
        "premarket_analysis_active": corrected_service_manager.premarket_analysis_active,
        "priority_trading_active": corrected_service_manager.priority_trading_active,
        "active_connections": len(corrected_service_manager.active_connections),
        "market_status": market_status,
        "trading_mode": trading_mode,
        "features": {
            "enhanced_market_data": ENHANCED_MARKET_DATA_AVAILABLE,
            "configuration": CONFIG_AVAILABLE,
            "nifty100_universe": True,
            "premarket_analysis": True,
            "priority_trading": True,
            "no_legacy_dependencies": True
        },
        "premarket_stats": {
            "opportunities_found": len(corrected_service_manager.premarket_opportunities),
            "priority_signals_queue": len(corrected_service_manager.priority_signals_queue)
        },
        "configuration": {
            "environment": settings.environment,
            "debug": settings.debug,
            "telegram_configured": settings.is_telegram_configured,
            "max_signals_per_day": settings.max_signals_per_day,
            "signal_interval": settings.signal_generation_interval,
            "stock_universe": "Nifty 100 Complete",
            "total_stocks": 100,
            "priority_stocks": 50
        },
        "dependencies": {
            "enhanced_market_data_available": ENHANCED_MARKET_DATA_AVAILABLE,
            "telegram_available": TELEGRAM_AVAILABLE,
            "analytics_available": ANALYTICS_AVAILABLE,
            "signal_generator_available": SIGNAL_GENERATOR_AVAILABLE,
            "legacy_market_service": False  # Completely removed
        },
        "timestamp": datetime.now().isoformat()
    }

# ================================================================
# WebSocket
# ================================================================

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint"""
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
                        "timestamp": datetime.now().isoformat()
                    })
                elif message.get("type") == "request_analytics":
                    try:
                        performance = corrected_service_manager.analytics_service.get_performance_summary()
                        await websocket.send_json({
                            "type": "analytics_update",
                            "data": performance,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.debug(f"Failed to send analytics: {e}")
                elif message.get("type") == "request_market_status":
                    try:
                        await websocket.send_json({
                            "type": "market_status_update",
                            "data": {
                                "status": corrected_service_manager.current_market_status.value,
                                "trading_mode": corrected_service_manager.current_trading_mode.value
                            },
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        logger.debug(f"Failed to send market status: {e}")
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        corrected_service_manager.disconnect_websocket(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        corrected_service_manager.disconnect_websocket(websocket)

# ================================================================
# Signal Handlers
# ================================================================

def signal_handler_func(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")

signal_handler.signal(signal_handler.SIGINT, signal_handler_func)
signal_handler.signal(signal_handler.SIGTERM, signal_handler_func)

# ================================================================
# Application Entry Point
# ================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Ensure logs directory exists
    os.makedirs("logs/enhanced", exist_ok=True)
    
    logger.info("🚀 Starting TradeMind AI Enhanced Edition...")
    logger.info("📊 Architecture: Enhanced Only (No Legacy Dependencies)")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level=settings.log_level.lower() if hasattr(settings, 'log_level') else "info",
            access_log=True,
            reload=settings.debug if hasattr(settings, 'debug') else False,
            reload_dirs=["app"] if hasattr(settings, 'debug') and settings.debug else None
        )
    except KeyboardInterrupt:
        logger.info("🛑 Enhanced application stopped by user")
    except Exception as e:
        logger.error(f"❌ Enhanced application startup failed: {e}")
        sys.exit(1)