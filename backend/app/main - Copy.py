# ================================================================
# TradeMind AI - Enhanced Main Application with Nifty 100 Universe
# ================================================================

"""
TradeMind AI - Production-Ready FastAPI Application
Enhanced with Nifty 100 Universe, Pre-Market Analysis, and Priority Trading

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

# Import configuration
try:
    from app.core.config import settings
except ImportError:
    logging.error("❌ Configuration not found. Please ensure config.py exists.")
    sys.exit(1)

# Import enhanced market data service
try:
    from app.services.enhanced_market_data_nifty100 import (
        EnhancedMarketDataService,
        create_enhanced_market_data_service,
        MarketStatus,
        TradingMode,
        PreMarketOpportunity
    )
    ENHANCED_MARKET_DATA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Enhanced market data service not available: {e}")
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

# Import ML components
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

# Import signal logging
SIGNAL_LOGGING_AVAILABLE = False
try:
    from app.core.signal_logger import InstitutionalSignalLogger, SignalRecord
    SIGNAL_LOGGING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Signal logging not available: {e}")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure comprehensive logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trademind_enhanced.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# Enhanced Analytics Class for Better Tracking
# ================================================================

class EnhancedAnalytics:
    """Enhanced analytics implementation for Nifty 100 trading"""
    
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
        """Track signal generation with enhanced metrics"""
        self.daily_stats["signals_generated"] += 1
        self.daily_stats["last_signal_time"] = datetime.now().isoformat()
        
        # Track priority signals (9:15-9:45 AM)
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
    
    def get_premarket_summary(self):
        """Get pre-market analysis summary"""
        return {
            "last_analysis": self.daily_stats["last_premarket_analysis"],
            "analyses_count": self.daily_stats["premarket_analyses"],
            "opportunities_found": len(self.premarket_opportunities),
            "priority_signals_ready": len(self.priority_signals_today)
        }
    
    def get_performance_summary(self):
        """Get enhanced performance summary"""
        daily = self.get_daily_stats()
        
        # Calculate telegram success rate
        total_telegram = daily["telegram_success"] + daily["telegram_failures"]
        telegram_success_rate = (daily["telegram_success"] / total_telegram * 100) if total_telegram > 0 else 100.0
        
        return {
            "daily": daily,
            "premarket": self.get_premarket_summary(),
            "system_health": {
                "nifty100_enabled": True,
                "premarket_analysis_active": True,
                "priority_trading_enabled": True,
                "telegram_configured": settings.is_telegram_configured,
                "signal_generation_active": True,
                "last_activity": daily["last_signal_time"],
                "error_rate": (daily["telegram_failures"] / max(total_telegram, 1)) * 100
            },
            "configuration": {
                "stock_universe": "Nifty 100",
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
                "universe": "Nifty 100",
                "total_stocks": daily["nifty100_coverage"],
                "priority_stocks": daily["priority_stocks_tracked"],
                "current_mode": daily["market_mode"]
            }
        }

# ================================================================
# Enhanced Service Manager with Nifty 100 Integration
# ================================================================

class EnhancedTradeMindServiceManager:
    """Enhanced service manager with Nifty 100 universe and pre-market analysis"""
    
    def __init__(self):
        # Core services
        self.active_connections: List[WebSocket] = []
        self.telegram_service: Optional[TelegramService] = None
        self.analytics_service: Optional[EnhancedAnalytics] = None
        self.enhanced_market_service: Optional[EnhancedMarketDataService] = None
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
            "ml_models": False,
            "signal_generation": False,
            "analytics": False
        }
        
        # Enhanced signal generation control
        self.signal_generation_active = False
        self.premarket_analysis_active = False
        self.priority_trading_active = False
        
        # Background tasks
        self.signal_generation_task = None
        self.premarket_analysis_task = None
        self.market_monitor_task = None
        
    async def initialize_all_services(self):
        """Initialize all enhanced services"""
        try:
            logger.info("🚀 Initializing Enhanced TradeMind AI with Nifty 100 Universe...")
            
            # 1. Initialize enhanced analytics
            await self._initialize_enhanced_analytics()
            
            # 2. Initialize Telegram
            await self._initialize_telegram()
            
            # 3. Initialize signal logging
            await self._initialize_signal_logging()
            
            # 4. Initialize enhanced market data service (Nifty 100)
            await self._initialize_enhanced_market_data()
            
            # 5. Initialize ML components
            await self._initialize_ml_components()
            
            # 6. Initialize signal generator
            await self._initialize_signal_generator()
            
            # 7. Start market monitoring
            await self._start_market_monitoring()
            
            # 8. Final system check
            await self._perform_enhanced_health_check()
            
            self.is_initialized = True
            logger.info("✅ Enhanced TradeMind AI with Nifty 100 initialized successfully!")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Enhanced service initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _initialize_enhanced_analytics(self):
        """Initialize enhanced analytics"""
        try:
            if ANALYTICS_AVAILABLE and AnalyticsService:
                # Try institutional analytics first
                self.analytics_service = AnalyticsService(settings.database_url)
                logger.info("✅ Institutional AnalyticsService initialized")
            else:
                # Use enhanced analytics
                self.analytics_service = EnhancedAnalytics()
                logger.info("✅ Enhanced analytics initialized")
            
            self.system_health["analytics"] = True
            
        except Exception as e:
            logger.error(f"❌ Enhanced analytics initialization failed: {e}")
            self.analytics_service = EnhancedAnalytics()
            self.system_health["analytics"] = True
    
    async def _initialize_telegram(self):
        """Initialize Telegram with enhanced notifications"""
        try:
            if settings.is_telegram_configured and TELEGRAM_AVAILABLE and TelegramService:
                self.telegram_service = TelegramService()
                
                if self.telegram_service.is_configured():
                    # Send enhanced startup notification
                    startup_message = (
                        "🚀 *TradeMind AI Enhanced - LIVE*\n\n"
                        "📈 *Nifty 100 Universe Activated*\n"
                        "🌅 Pre-Market Analysis: 8:00-9:15 AM\n"
                        "⚡ Priority Trading: 9:15-9:45 AM\n"
                        "📊 Regular Trading: 9:45 AM-3:30 PM\n\n"
                        "💡 _Professional-grade signals incoming..._"
                    )
                    
                    success = await self.telegram_service.send_message(startup_message)
                    if success:
                        logger.info("📱 Enhanced Telegram startup notification sent")
                    
                    self.system_health["telegram"] = True
                    logger.info("✅ Telegram service with enhanced notifications ready")
                else:
                    logger.warning("⚠️ Telegram not properly configured")
                    self.system_health["telegram"] = False
            else:
                logger.warning("⚠️ Telegram not configured - enhanced signals will not be sent")
                self.system_health["telegram"] = False
        except Exception as e:
            logger.error(f"❌ Telegram initialization failed: {e}")
            self.system_health["telegram"] = False
    
    async def _initialize_signal_logging(self):
        """Initialize enhanced signal logging"""
        try:
            if SIGNAL_LOGGING_AVAILABLE and InstitutionalSignalLogger:
                self.signal_logger = InstitutionalSignalLogger("logs/enhanced")
                logger.info("✅ Enhanced signal logging initialized")
            else:
                logger.warning("⚠️ Using basic signal logging")
        except Exception as e:
            logger.error(f"❌ Signal logging initialization failed: {e}")
    
    async def _initialize_enhanced_market_data(self):
        """Initialize enhanced market data service with Nifty 100"""
        try:
            if ENHANCED_MARKET_DATA_AVAILABLE:
                self.enhanced_market_service = create_enhanced_market_data_service()
                await self.enhanced_market_service.initialize()
                
                # Check health
                health = await self.enhanced_market_service.get_service_health()
                if health['is_initialized']:
                    self.system_health["enhanced_market_data"] = True
                    self.system_health["nifty100_universe"] = True
                    self.system_health["premarket_analysis"] = True
                    self.system_health["priority_trading"] = True
                    
                    logger.info("✅ Enhanced Market Data Service (Nifty 100) initialized")
                    logger.info(f"📊 Tracking {health['nifty100_universe']['total_stocks']} Nifty 100 stocks")
                    logger.info(f"⭐ Priority stocks: {health['nifty100_universe']['nifty50_stocks']} (Nifty 50)")
                    logger.info(f"📈 Next 50 stocks: {health['nifty100_universe']['next50_stocks']}")
                    logger.info(f"🏭 Sectors covered: {health['nifty100_universe']['sectors_covered']}")
                    logger.info(f"🏪 Market Status: {health.get('market_status', 'Unknown')}")
                    logger.info(f"🎯 Trading Mode: {health.get('trading_mode', 'Unknown')}")
                else:
                    logger.error("❌ Enhanced market data service initialization failed")
                    self.system_health["enhanced_market_data"] = False
            else:
                logger.error("❌ Enhanced market data service not available")
                self.system_health["enhanced_market_data"] = False
                
        except Exception as e:
            logger.error(f"❌ Enhanced market data initialization failed: {e}")
            self.system_health["enhanced_market_data"] = False
    
    async def _initialize_ml_components(self):
        """Initialize ML components for enhanced analysis"""
        try:
            if ML_AVAILABLE:
                logger.info("✅ ML components available for enhanced analysis")
                self.system_health["ml_models"] = True
            else:
                logger.warning("⚠️ ML components not available - using enhanced demo signals")
                self.system_health["ml_models"] = False
        except Exception as e:
            logger.error(f"❌ ML initialization failed: {e}")
            self.system_health["ml_models"] = False
    
    async def _initialize_signal_generator(self):
        """Initialize enhanced signal generator"""
        try:
            if (SIGNAL_GENERATOR_AVAILABLE and ProductionMLSignalGenerator and 
                self.enhanced_market_service and self.signal_logger):
                self.signal_generator = ProductionMLSignalGenerator(
                    self.enhanced_market_service,
                    self.signal_logger
                )
                self.system_health["signal_generation"] = True
                logger.info("✅ Enhanced signal generator initialized")
            else:
                logger.warning("⚠️ Using enhanced demo signal generator")
                self.system_health["signal_generation"] = True
        except Exception as e:
            logger.error(f"❌ Signal generator initialization failed: {e}")
            self.system_health["signal_generation"] = True
    
    async def _start_market_monitoring(self):
        """Start enhanced market monitoring"""
        try:
            self.market_monitor_task = asyncio.create_task(self._market_monitoring_loop())
            logger.info("🔄 Enhanced market monitoring started")
        except Exception as e:
            logger.error(f"❌ Market monitoring start failed: {e}")
    
    async def _market_monitoring_loop(self):
        """Enhanced market monitoring loop"""
        logger.info("🔄 Starting enhanced market monitoring loop...")
        
        while True:
            try:
                # Update market status
                if self.enhanced_market_service:
                    await self.enhanced_market_service.update_market_status()
                    
                    health = await self.enhanced_market_service.get_service_health()
                    old_status = self.current_market_status
                    old_mode = self.current_trading_mode
                    
                    self.current_market_status = MarketStatus(health.get('market_status', 'CLOSED'))
                    self.current_trading_mode = TradingMode(health.get('trading_mode', 'PRE_MARKET_ANALYSIS'))
                    
                    # Update analytics
                    if self.analytics_service and hasattr(self.analytics_service, 'update_market_mode'):
                        self.analytics_service.update_market_mode(self.current_market_status.value)
                    
                    # Handle market status transitions
                    if old_status != self.current_market_status or old_mode != self.current_trading_mode:
                        await self._handle_market_transition(old_status, self.current_market_status, 
                                                           old_mode, self.current_trading_mode)
                
                # Check for pre-market analysis time
                await self._check_premarket_analysis_trigger()
                
                # Check for priority trading time
                await self._check_priority_trading_trigger()
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                logger.info("🛑 Market monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Market monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _handle_market_transition(self, old_status, new_status, old_mode, new_mode):
        """Handle market status and mode transitions"""
        logger.info(f"🔄 Market Transition: {old_status.value} → {new_status.value}")
        logger.info(f"🎯 Mode Transition: {old_mode.value} → {new_mode.value}")
        
        # Send notifications
        if self.telegram_service and self.telegram_service.is_configured():
            if new_status == MarketStatus.PRE_MARKET:
                message = "🌅 *PRE-MARKET PHASE STARTED*\n\n📊 Running Nifty 100 analysis...\n⚡ Priority signals incoming at 9:15 AM"
                await self.telegram_service.send_message(message)
            elif new_status == MarketStatus.OPEN and new_mode == TradingMode.PRIORITY_TRADING:
                message = "🚀 *PRIORITY TRADING ACTIVATED*\n\n⚡ 9:15 AM - Executing priority signals\n📈 Best opportunities from pre-market analysis"
                await self.telegram_service.send_message(message)
            elif new_mode == TradingMode.REGULAR_TRADING:
                message = "📊 *REGULAR TRADING MODE*\n\n🔄 Continuous signal generation active\n📈 Monitoring all Nifty 100 stocks"
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
        """Check if it's time to run pre-market analysis"""
        now = datetime.now()
        current_time = now.time()
        
        # Run pre-market analysis between 8:00-9:15 AM on weekdays
        if (now.weekday() < 5 and 
            dt_time(8, 0) <= current_time <= dt_time(9, 15) and
            not self.premarket_analysis_active and
            self.enhanced_market_service):
            
            await self._run_premarket_analysis()
    
    async def _run_premarket_analysis(self):
        """Run comprehensive pre-market analysis"""
        try:
            self.premarket_analysis_active = True
            logger.info("🌅 Starting pre-market analysis for Nifty 100...")
            
            if self.enhanced_market_service:
                # Run the enhanced pre-market analysis
                analysis_result = await self.enhanced_market_service.run_premarket_analysis()
                
                self.premarket_opportunities = analysis_result.get("top_opportunities", [])
                
                # Track in analytics
                if self.analytics_service and hasattr(self.analytics_service, 'track_premarket_analysis'):
                    await self.analytics_service.track_premarket_analysis(len(self.premarket_opportunities))
                
                # Prepare priority signals for 9:15 AM
                priority_signals = await self.enhanced_market_service.get_priority_trading_signals()
                self.priority_signals_queue = priority_signals
                
                # Send pre-market summary to Telegram
                if self.telegram_service and self.telegram_service.is_configured():
                    summary_message = (
                        f"🌅 *PRE-MARKET ANALYSIS COMPLETE*\n\n"
                        f"📊 *Nifty 100 Opportunities Found:* {analysis_result.get('total_opportunities', 0)}\n"
                        f"🎯 *Strong Buy Signals:* {analysis_result.get('strong_buy_count', 0)}\n"
                        f"📈 *Buy Signals:* {analysis_result.get('buy_count', 0)}\n"
                        f"👀 *Watch List:* {analysis_result.get('watch_count', 0)}\n\n"
                        f"⚡ *Priority signals ready for 9:15 AM*\n"
                        f"🎯 *{len(priority_signals)} high-priority trades prepared*"
                    )
                    await self.telegram_service.send_message(summary_message)
                
                # Broadcast to dashboard
                await self.broadcast_to_dashboard({
                    "type": "premarket_analysis_complete",
                    "data": analysis_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"✅ Pre-market analysis complete: {len(self.premarket_opportunities)} opportunities")
                
        except Exception as e:
            logger.error(f"❌ Pre-market analysis failed: {e}")
        finally:
            # Reset after 9:15 AM
            current_time = datetime.now().time()
            if current_time > dt_time(9, 15):
                self.premarket_analysis_active = False
    
    async def _check_priority_trading_trigger(self):
        """Check if it's time for priority trading (9:15 AM)"""
        now = datetime.now()
        current_time = now.time()
        
        # Trigger priority trading at 9:15 AM on weekdays
        if (now.weekday() < 5 and 
            dt_time(9, 15) <= current_time <= dt_time(9, 17) and  # 2-minute window
            not self.priority_trading_active and
            self.priority_signals_queue):
            
            await self._execute_priority_trading()
    
    async def _execute_priority_trading(self):
        """Execute priority trading signals at 9:15 AM"""
        try:
            self.priority_trading_active = True
            logger.info("⚡ Executing priority trading signals at 9:15 AM...")
            
            # Send priority signals
            for signal in self.priority_signals_queue:
                await self._process_enhanced_signal(signal, is_priority=True)
                await asyncio.sleep(1)  # Small delay between signals
            
            # Send priority trading summary
            if self.telegram_service and self.telegram_service.is_configured():
                summary_message = (
                    f"⚡ *PRIORITY TRADING EXECUTED*\n\n"
                    f"🎯 *Signals Sent:* {len(self.priority_signals_queue)}\n"
                    f"⏰ *Execution Time:* 9:15 AM IST\n"
                    f"📊 *Based on pre-market analysis*\n\n"
                    f"🔄 _Transitioning to regular trading mode..._"
                )
                await self.telegram_service.send_message(summary_message)
            
            logger.info(f"✅ Priority trading complete: {len(self.priority_signals_queue)} signals executed")
            
        except Exception as e:
            logger.error(f"❌ Priority trading execution failed: {e}")
        finally:
            # Reset after execution
            self.priority_signals_queue = []
            # Keep active until 9:45 AM
            current_time = datetime.now().time()
            if current_time > dt_time(9, 45):
                self.priority_trading_active = False
    
    async def _perform_enhanced_health_check(self):
        """Perform enhanced system health check"""
        health_summary = {
            "total_services": len(self.system_health),
            "healthy_services": sum(self.system_health.values()),
            "unhealthy_services": len(self.system_health) - sum(self.system_health.values()),
            "health_percentage": (sum(self.system_health.values()) / len(self.system_health)) * 100
        }
        
        logger.info(f"🏥 Enhanced System Health: {health_summary['healthy_services']}/{health_summary['total_services']} services ({health_summary['health_percentage']:.1f}%)")
        
        for service, status in self.system_health.items():
            status_icon = "✅" if status else "❌"
            service_name = service.replace('_', ' ').title()
            status_text = "Healthy" if status else "Unhealthy"
            logger.info(f"  {status_icon} {service_name}: {status_text}")
    
    async def start_enhanced_signal_generation(self):
        """Start enhanced signal generation with market modes"""
        if self.signal_generation_active:
            logger.warning("⚠️ Enhanced signal generation already active")
            return
        
        self.signal_generation_active = True
        self.signal_generation_task = asyncio.create_task(self._enhanced_signal_generation_loop())
        
        logger.info("🎯 Enhanced signal generation started (Nifty 100 + Pre-market + Priority)")
    
    async def stop_enhanced_signal_generation(self):
        """Stop enhanced signal generation"""
        if not self.signal_generation_active:
            return
        
        self.signal_generation_active = False
        if self.signal_generation_task:
            self.signal_generation_task.cancel()
        
        logger.info("🛑 Enhanced signal generation stopped")
    
    async def _enhanced_signal_generation_loop(self):
        """Enhanced signal generation loop with market awareness"""
        logger.info("🔄 Starting enhanced signal generation loop...")
        
        while self.signal_generation_active:
            try:
                # Check market status and mode
                if not self.enhanced_market_service:
                    await asyncio.sleep(60)
                    continue
                
                health = await self.enhanced_market_service.get_service_health()
                current_status = health.get('market_status', 'CLOSED')
                current_mode = health.get('trading_mode', 'PRE_MARKET_ANALYSIS')
                
                # Skip signal generation during pre-market analysis (let dedicated function handle it)
                if current_mode == 'PRE_MARKET_ANALYSIS':
                    await asyncio.sleep(60)
                    continue
                
                # Skip if priority trading is handling signals
                if current_mode == 'PRIORITY_TRADING' and self.priority_trading_active:
                    await asyncio.sleep(30)
                    continue
                
                # Check if we can generate signals
                if not self._can_generate_enhanced_signals():
                    await asyncio.sleep(60)
                    continue
                
                # Generate signals based on current mode
                signals = await self._generate_enhanced_signals(current_mode)
                
                # Process each signal
                for signal in signals:
                    await self._process_enhanced_signal(signal, is_priority=False)
                
                # Dynamic wait time based on market mode
                if current_mode == 'PRIORITY_TRADING':
                    await asyncio.sleep(30)  # Faster during priority hours
                elif current_mode == 'REGULAR_TRADING':
                    await asyncio.sleep(settings.signal_generation_interval)
                else:
                    await asyncio.sleep(300)  # 5 minutes during off-hours
                
            except asyncio.CancelledError:
                logger.info("🛑 Enhanced signal generation loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Enhanced signal generation error: {e}")
                await asyncio.sleep(30)
    
    def _can_generate_enhanced_signals(self) -> bool:
        """Check if we can generate enhanced signals"""
        try:
            # Check daily limits
            if self.analytics_service:
                daily_stats = self.analytics_service.get_daily_stats()
                if daily_stats["signals_generated"] >= settings.max_signals_per_day:
                    return False
            
            # Check market hours for regular signals
            current_time = datetime.now().time()
            weekday = datetime.now().weekday()
            
            if weekday >= 5:  # Weekend
                return False
            
            # Allow signals during market hours and slight extended hours
            if dt_time(9, 15) <= current_time <= dt_time(16, 0):
                return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Enhanced signal generation check failed: {e}")
            return False
    
    async def _generate_enhanced_signals(self, trading_mode: str) -> List[Dict]:
        """Generate enhanced signals based on trading mode"""
        signals = []
        
        try:
            if (self.signal_generator and self.system_health["signal_generation"] and 
                self.enhanced_market_service):
                # Use enhanced signal generator
                signals = await self.signal_generator.generate_signals()
            else:
                # Enhanced demo signal generation
                signals = await self._generate_enhanced_demo_signals(trading_mode)
            
        except Exception as e:
            logger.error(f"❌ Enhanced signal generation failed: {e}")
            signals = await self._generate_enhanced_demo_signals(trading_mode)
        
        return signals
    
    async def _generate_enhanced_demo_signals(self, trading_mode: str) -> List[Dict]:
        """Generate enhanced demo signals from Nifty 100"""
        import random
        
        # Nifty 50 priority stocks
        nifty50_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "INFY", "ITC", 
            "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", 
            "AXISBANK", "MARUTI", "BAJFINANCE", "TITAN", "NESTLEIND", "ULTRACEMCO", 
            "WIPRO", "ONGC", "NTPC", "POWERGRID", "SUNPHARMA", "TATAMOTORS"
        ]
        
        # Next 50 stocks
        next50_stocks = [
            "VEDL", "GODREJCP", "DABUR", "BIOCON", "MARICO", "SIEMENS", "BANKBARODA",
            "HDFCAMC", "TORNTPHARM", "BERGEPAINT", "BOSCHLTD", "MOTHERSON", "COLPAL",
            "LUPIN", "MCDOWELL-N", "GAIL", "DLF", "AMBUJACEM", "ADANIGREEN", "HAVELLS"
        ]
        
        # Choose stock based on trading mode
        if trading_mode == 'PRIORITY_TRADING':
            # Prioritize Nifty 50 during priority hours
            stock = random.choice(nifty50_stocks)
            confidence_boost = 0.1
        elif trading_mode == 'REGULAR_TRADING':
            # Mix of both during regular trading
            all_stocks = nifty50_stocks + next50_stocks
            stock = random.choice(all_stocks)
            confidence_boost = 0.05
        else:
            # Any stock during other times
            all_stocks = nifty50_stocks + next50_stocks
            stock = random.choice(all_stocks)
            confidence_boost = 0.0
        
        # Get realistic price data
        base_price = 1500.0
        try:
            if self.enhanced_market_service:
                stock_data = await self.enhanced_market_service.get_live_market_data(stock)
                if stock_data and "quote" in stock_data:
                    base_price = stock_data["quote"]["ltp"]
        except Exception as e:
            logger.debug(f"Failed to get live price for {stock}: {e}")
        
        # Generate signal with enhanced confidence based on mode
        base_confidence = random.uniform(0.65, 0.85) + confidence_boost
        
        signal = {
            "id": f"enhanced_{trading_mode.lower()}_{int(datetime.now().timestamp())}",
            "symbol": stock,
            "action": random.choice(["BUY", "SELL"]),
            "entry_price": round(base_price, 2),
            "target_price": 0,
            "stop_loss": 0,
            "confidence": round(min(0.95, base_confidence), 3),
            "sentiment_score": round(random.uniform(-0.2, 0.3), 3),
            "timestamp": datetime.now(),
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "signal_type": f"ENHANCED_{trading_mode}",
            "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "trading_mode": trading_mode,
            "stock_universe": "NIFTY_100",
            "sector": self._get_stock_sector(stock),
            "market_cap": "LARGE_CAP",
            "priority_tier": 1 if stock in nifty50_stocks else 2
        }
        
        # Calculate enhanced targets based on trading mode
        if signal["action"] == "BUY":
            if trading_mode == 'PRIORITY_TRADING':
                signal["target_price"] = round(signal["entry_price"] * 1.035, 2)  # Higher targets
                signal["stop_loss"] = round(signal["entry_price"] * 0.975, 2)
            else:
                signal["target_price"] = round(signal["entry_price"] * 1.025, 2)
                signal["stop_loss"] = round(signal["entry_price"] * 0.985, 2)
        else:
            if trading_mode == 'PRIORITY_TRADING':
                signal["target_price"] = round(signal["entry_price"] * 0.965, 2)
                signal["stop_loss"] = round(signal["entry_price"] * 1.025, 2)
            else:
                signal["target_price"] = round(signal["entry_price"] * 0.975, 2)
                signal["stop_loss"] = round(signal["entry_price"] * 1.015, 2)
        
        return [signal]
    
    def _get_stock_sector(self, symbol: str) -> str:
        """Get sector for stock symbol"""
        sector_mapping = {
            "RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Banking", "ICICIBANK": "Banking",
            "HINDUNILVR": "FMCG", "INFY": "IT", "ITC": "FMCG", "SBIN": "Banking",
            "BHARTIARTL": "Telecom", "KOTAKBANK": "Banking", "LT": "Infrastructure",
            "HCLTECH": "IT", "ASIANPAINT": "Paints", "AXISBANK": "Banking", "MARUTI": "Auto",
            "BAJFINANCE": "Financial Services", "TITAN": "Consumer Durables", "NESTLEIND": "FMCG",
            "ULTRACEMCO": "Cement", "WIPRO": "IT", "ONGC": "Energy", "NTPC": "Power",
            "POWERGRID": "Power", "SUNPHARMA": "Pharma", "TATAMOTORS": "Auto"
        }
        return sector_mapping.get(symbol, "Diversified")
    
    async def _process_enhanced_signal(self, signal: Dict, is_priority: bool = False):
        """Process enhanced signal with priority handling"""
        try:
            # Add priority flag
            signal["is_priority_signal"] = is_priority
            signal["processing_timestamp"] = datetime.now().isoformat()
            
            # Track in enhanced analytics
            if self.analytics_service:
                await self.analytics_service.track_signal_generated(signal)
            
            # Enhanced Telegram notification
            telegram_success = False
            if self.telegram_service and self.telegram_service.is_configured():
                telegram_success = await self._send_enhanced_telegram_notification(signal, is_priority)
                if self.analytics_service:
                    await self.analytics_service.track_telegram_sent(telegram_success, signal)
            
            # Enhanced signal logging
            if self.signal_logger:
                try:
                    logger.info(f"📝 Enhanced signal logged: {signal['id']}")
                except Exception as e:
                    logger.debug(f"Signal logging failed: {e}")
            
            # Broadcast to dashboard with enhanced data
            await self.broadcast_to_dashboard({
                "type": "new_enhanced_signal",
                "data": signal,
                "is_priority": is_priority,
                "timestamp": datetime.now().isoformat()
            })
            
            # Broadcast enhanced analytics
            if self.analytics_service:
                try:
                    analytics = self.analytics_service.get_performance_summary()
                    await self.broadcast_to_dashboard({
                        "type": "enhanced_analytics_update",
                        "data": analytics,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.debug(f"Enhanced analytics broadcast failed: {e}")
            
            # Enhanced logging with priority indicator
            priority_indicator = "⚡ PRIORITY" if is_priority else "📊 REGULAR"
            logger.info(f"📡 {priority_indicator} Enhanced Signal: {signal['symbol']} {signal['action']} @ ₹{signal['entry_price']} | "
                       f"Confidence: {signal['confidence']:.1%} | "
                       f"Mode: {signal.get('trading_mode', 'UNKNOWN')} | "
                       f"Tier: {signal.get('priority_tier', 'N/A')} | "
                       f"Telegram: {'✅' if telegram_success else '❌'}")
            
        except Exception as e:
            logger.error(f"❌ Enhanced signal processing failed: {e}")
    
    async def _send_enhanced_telegram_notification(self, signal: Dict, is_priority: bool) -> bool:
        """Send enhanced Telegram notification"""
        try:
            # Priority signal formatting
            if is_priority:
                header = "⚡ *PRIORITY SIGNAL - 9:15 AM*"
                mode_text = f"🎯 *Mode:* {signal.get('trading_mode', 'PRIORITY')}"
            else:
                header = f"📊 *{signal.get('trading_mode', 'REGULAR')} SIGNAL*"
                mode_text = f"🔄 *Mode:* {signal.get('trading_mode', 'REGULAR')}"
            
            # Enhanced message format
            message = (
                f"{header}\n\n"
                f"📈 *Stock:* {signal['symbol']}\n"
                f"🎯 *Action:* {signal['action']}\n"
                f"💰 *Entry:* ₹{signal['entry_price']}\n"
                f"🎯 *Target:* ₹{signal['target_price']}\n"
                f"🛡️ *Stop Loss:* ₹{signal['stop_loss']}\n"
                f"📊 *Confidence:* {signal['confidence']:.1%}\n"
                f"🏭 *Sector:* {signal.get('sector', 'N/A')}\n"
                f"⭐ *Tier:* Nifty {50 if signal.get('priority_tier') == 1 else 100}\n"
                f"{mode_text}\n"
                f"⏰ *Time:* {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return await self.telegram_service.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Enhanced Telegram notification failed: {e}")
            return False
    
    # WebSocket management (enhanced)
    async def connect_websocket(self, websocket: WebSocket):
        """Handle new WebSocket connection with enhanced data"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"📱 Dashboard connected. Total: {len(self.active_connections)}")
        
        # Send enhanced welcome message
        welcome_data = {
            "type": "enhanced_connection",
            "message": "Connected to TradeMind AI Professional - Nifty 100 Enhanced",
            "system_health": self.system_health,
            "initialization_status": self.is_initialized,
            "market_status": self.current_market_status.value if self.current_market_status else "UNKNOWN",
            "trading_mode": self.current_trading_mode.value if self.current_trading_mode else "UNKNOWN",
            "features": {
                "nifty100_universe": True,
                "premarket_analysis": True,
                "priority_trading": True,
                "enhanced_signals": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await websocket.send_json(welcome_data)
        
        # Send initial enhanced analytics if available
        if self.analytics_service:
            try:
                performance = self.analytics_service.get_performance_summary()
                await websocket.send_json({
                    "type": "initial_enhanced_analytics",
                    "data": performance,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.debug(f"Failed to send initial enhanced analytics: {e}")
    
    def disconnect_websocket(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"📱 Dashboard disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_to_dashboard(self, message: Dict):
        """Broadcast enhanced message to all connected dashboards"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

# ================================================================
# Global Enhanced Service Manager Instance
# ================================================================

enhanced_service_manager = EnhancedTradeMindServiceManager()

# ================================================================
# Enhanced Application Lifecycle Management
# ================================================================

@asynccontextmanager
async def enhanced_lifespan(app: FastAPI):
    """Enhanced application lifecycle management"""
    # Startup
    logger.info("🚀 TradeMind AI Enhanced Starting...")
    logger.info(f"🔧 Environment: {settings.environment}")
    logger.info(f"📊 Enhanced Analytics: {'Enabled' if settings.track_performance else 'Disabled'}")
    logger.info(f"📈 Nifty 100 Universe: {'Enabled' if ENHANCED_MARKET_DATA_AVAILABLE else 'Disabled'}")
    logger.info(f"🌅 Pre-Market Analysis: Enabled")
    logger.info(f"⚡ Priority Trading: Enabled")
    
    try:
        # Initialize all enhanced services
        await enhanced_service_manager.initialize_all_services()
        
        # Start enhanced signal generation
        await enhanced_service_manager.start_enhanced_signal_generation()
        
        logger.info("✅ TradeMind AI Enhanced fully operational!")
        
    except Exception as e:
        logger.error(f"❌ Enhanced startup failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 TradeMind AI Enhanced shutting down...")
    
    # Send enhanced shutdown notification
    if (enhanced_service_manager.telegram_service and 
        enhanced_service_manager.telegram_service.is_configured()):
        try:
            shutdown_message = (
                "🛑 *TradeMind AI Enhanced - SHUTDOWN*\n\n"
                "📊 *Session Summary:*\n"
                f"• Nifty 100 monitoring: Complete\n"
                f"• Pre-market analyses: {enhanced_service_manager.analytics_service.get_daily_stats()['premarket_analyses'] if enhanced_service_manager.analytics_service else 0}\n"
                f"• Priority signals: {enhanced_service_manager.analytics_service.get_daily_stats()['priority_signals'] if enhanced_service_manager.analytics_service else 0}\n"
                f"• Total signals: {enhanced_service_manager.analytics_service.get_daily_stats()['signals_generated'] if enhanced_service_manager.analytics_service else 0}\n\n"
                "💤 _System going offline..._"
            )
            await enhanced_service_manager.telegram_service.send_message(shutdown_message)
            logger.info("📱 Enhanced Telegram shutdown notification sent")
        except Exception as e:
            logger.error(f"❌ Failed to send enhanced shutdown notification: {e}")
    
    await enhanced_service_manager.stop_enhanced_signal_generation()
    
    # Close enhanced market data service
    if enhanced_service_manager.enhanced_market_service:
        try:
            await enhanced_service_manager.enhanced_market_service.close()
        except Exception as e:
            logger.error(f"Error closing enhanced market data service: {e}")
    
    logger.info("✅ Enhanced shutdown complete")

# ================================================================
# Enhanced FastAPI Application Setup
# ================================================================

app = FastAPI(
    title="TradeMind AI Professional - Nifty 100 Enhanced",
    description="AI-Powered Indian Stock Trading Platform with Nifty 100 Universe, Pre-Market Analysis & Priority Trading",
    version="4.0.0",
    lifespan=enhanced_lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# Enhanced API Endpoints
# ================================================================

@app.get("/")
async def enhanced_root():
    """Enhanced root endpoint with Nifty 100 system information"""
    market_status = "UNKNOWN"
    trading_mode = "UNKNOWN"
    
    if enhanced_service_manager.enhanced_market_service:
        try:
            health = await enhanced_service_manager.enhanced_market_service.get_service_health()
            market_status = health.get('market_status', 'UNKNOWN')
            trading_mode = health.get('trading_mode', 'UNKNOWN')
        except Exception:
            pass
    
    return {
        "message": "🇮🇳 TradeMind AI - Professional Trading Platform (Nifty 100 Enhanced)",
        "status": "operational" if enhanced_service_manager.is_initialized else "initializing",
        "version": "4.0.0",
        "environment": settings.environment,
        "enhanced_features": {
            "nifty100_universe": True,
            "premarket_analysis": "8:00-9:15 AM",
            "priority_trading": "9:15-9:45 AM",
            "regular_trading": "9:45 AM-3:30 PM",
            "total_stocks_tracked": 100,
            "priority_stocks": 50,
            "sectors_covered": 15
        },
        "system_health": enhanced_service_manager.system_health,
        "services": {
            "telegram": settings.is_telegram_configured and enhanced_service_manager.system_health.get("telegram", False),
            "enhanced_market_data": enhanced_service_manager.system_health.get("enhanced_market_data", False),
            "nifty100_universe": enhanced_service_manager.system_health.get("nifty100_universe", False),
            "premarket_analysis": enhanced_service_manager.system_health.get("premarket_analysis", False),
            "priority_trading": enhanced_service_manager.system_health.get("priority_trading", False),
            "ml_models": ML_AVAILABLE and enhanced_service_manager.system_health.get("ml_models", False),
            "analytics": enhanced_service_manager.system_health.get("analytics", False),
            "signal_generation": enhanced_service_manager.system_health.get("signal_generation", False)
        },
        "market_status": market_status,
        "trading_mode": trading_mode,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check endpoint"""
    if not enhanced_service_manager.analytics_service:
        return {"status": "initializing", "message": "Enhanced services still starting up"}
    
    try:
        performance = enhanced_service_manager.analytics_service.get_performance_summary()
        
        # Get enhanced market data health
        market_health = {}
        if enhanced_service_manager.enhanced_market_service:
            try:
                market_health = await enhanced_service_manager.enhanced_market_service.get_service_health()
            except Exception as e:
                logger.debug(f"Failed to get market health: {e}")
        
        return {
            "status": "healthy" if enhanced_service_manager.is_initialized else "degraded",
            "initialization_error": enhanced_service_manager.initialization_error,
            "system_health": enhanced_service_manager.system_health,
            "connections": len(enhanced_service_manager.active_connections),
            "signal_generation_active": enhanced_service_manager.signal_generation_active,
            "premarket_analysis_active": enhanced_service_manager.premarket_analysis_active,
            "priority_trading_active": enhanced_service_manager.priority_trading_active,
            "enhanced_services": {
                "nifty100_universe": enhanced_service_manager.system_health.get("nifty100_universe", False),
                "premarket_analysis": enhanced_service_manager.system_health.get("premarket_analysis", False),
                "priority_trading": enhanced_service_manager.system_health.get("priority_trading", False),
                "enhanced_market_data": enhanced_service_manager.system_health.get("enhanced_market_data", False)
            },
            "market_data_health": market_health,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/enhanced/dashboard")
async def get_enhanced_dashboard_analytics():
    """Get enhanced analytics for dashboard"""
    if not enhanced_service_manager.analytics_service:
        raise HTTPException(status_code=503, detail="Enhanced analytics service not available")
    
    try:
        performance = enhanced_service_manager.analytics_service.get_performance_summary()
        
        # Add enhanced system information
        performance["enhanced_system_status"] = {
            "initialized": enhanced_service_manager.is_initialized,
            "signal_generation_active": enhanced_service_manager.signal_generation_active,
            "premarket_analysis_active": enhanced_service_manager.premarket_analysis_active,
            "priority_trading_active": enhanced_service_manager.priority_trading_active,
            "health": enhanced_service_manager.system_health,
            "market_status": enhanced_service_manager.current_market_status.value if enhanced_service_manager.current_market_status else "UNKNOWN",
            "trading_mode": enhanced_service_manager.current_trading_mode.value if enhanced_service_manager.current_trading_mode else "UNKNOWN"
        }
        
        # Add market data if available
        if enhanced_service_manager.enhanced_market_service:
            try:
                market_overview = await enhanced_service_manager.enhanced_market_service.get_nifty100_overview()
                performance["nifty100_overview"] = market_overview
            except Exception as e:
                logger.debug(f"Failed to get Nifty 100 overview: {e}")
        
        return performance
        
    except Exception as e:
        logger.error(f"❌ Error getting enhanced dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced/premarket")
async def get_premarket_analysis():
    """Get pre-market analysis results"""
    if not enhanced_service_manager.enhanced_market_service:
        raise HTTPException(status_code=503, detail="Enhanced market data service not available")
    
    try:
        analysis = await enhanced_service_manager.enhanced_market_service.run_premarket_analysis()
        return {
            "analysis": analysis,
            "opportunities_found": len(enhanced_service_manager.premarket_opportunities),
            "priority_signals_ready": len(enhanced_service_manager.priority_signals_queue),
            "analysis_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting pre-market analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced/nifty100/overview")
async def get_nifty100_overview():
    """Get Nifty 100 market overview"""
    if not enhanced_service_manager.enhanced_market_service:
        raise HTTPException(status_code=503, detail="Enhanced market data service not available")
    
    try:
        overview = await enhanced_service_manager.enhanced_market_service.get_nifty100_overview()
        return overview
    except Exception as e:
        logger.error(f"❌ Error getting Nifty 100 overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced/market/data/{symbol}")
async def get_enhanced_stock_data(symbol: str):
    """Get enhanced live market data for a specific Nifty 100 stock"""
    if not enhanced_service_manager.enhanced_market_service:
        raise HTTPException(status_code=503, detail="Enhanced market data service not available")
    
    try:
        symbol = symbol.upper()
        data = await enhanced_service_manager.enhanced_market_service.get_live_market_data(symbol)
        return data
    except Exception as e:
        logger.error(f"❌ Error getting enhanced stock data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced/signals/priority")
async def get_priority_signals():
    """Get priority trading signals"""
    if not enhanced_service_manager.enhanced_market_service:
        raise HTTPException(status_code=503, detail="Enhanced market data service not available")
    
    try:
        signals = await enhanced_service_manager.enhanced_market_service.get_priority_trading_signals()
        return {
            "priority_signals": signals,
            "count": len(signals),
            "queue_ready": len(enhanced_service_manager.priority_signals_queue),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting priority signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced/premarket/run")
async def trigger_premarket_analysis():
    """Manually trigger pre-market analysis"""
    if not enhanced_service_manager.enhanced_market_service:
        raise HTTPException(status_code=503, detail="Enhanced market data service not available")
    
    try:
        await enhanced_service_manager._run_premarket_analysis()
        return {
            "success": True,
            "message": "Pre-market analysis triggered",
            "opportunities_found": len(enhanced_service_manager.premarket_opportunities),
            "priority_signals_prepared": len(enhanced_service_manager.priority_signals_queue),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Manual pre-market analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced/priority/execute")
async def trigger_priority_trading():
    """Manually trigger priority trading execution"""
    try:
        await enhanced_service_manager._execute_priority_trading()
        return {
            "success": True,
            "message": "Priority trading executed",
            "signals_sent": len(enhanced_service_manager.priority_signals_queue),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Manual priority trading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhanced/signals/manual")
async def generate_enhanced_manual_signal():
    """Manually trigger enhanced signal generation"""
    if not enhanced_service_manager.signal_generation_active:
        return {"error": "Enhanced signal generation not active"}
    
    try:
        # Get current trading mode
        trading_mode = "REGULAR_TRADING"
        if enhanced_service_manager.enhanced_market_service:
            health = await enhanced_service_manager.enhanced_market_service.get_service_health()
            trading_mode = health.get('trading_mode', 'REGULAR_TRADING')
        
        signals = await enhanced_service_manager._generate_enhanced_signals(trading_mode)
        
        for signal in signals:
            await enhanced_service_manager._process_enhanced_signal(signal, is_priority=False)
        
        return {
            "success": True,
            "signals_generated": len(signals),
            "signals": signals,
            "trading_mode": trading_mode,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced manual signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/enhanced/system/status")
async def get_enhanced_system_status():
    """Get detailed enhanced system status"""
    market_status = "UNKNOWN"
    trading_mode = "UNKNOWN"
    
    try:
        if enhanced_service_manager.enhanced_market_service:
            health = await enhanced_service_manager.enhanced_market_service.get_service_health()
            market_status = health.get('market_status', 'UNKNOWN')
            trading_mode = health.get('trading_mode', 'UNKNOWN')
    except Exception:
        pass
    
    return {
        "initialized": enhanced_service_manager.is_initialized,
        "initialization_error": enhanced_service_manager.initialization_error,
        "system_health": enhanced_service_manager.system_health,
        "signal_generation_active": enhanced_service_manager.signal_generation_active,
        "premarket_analysis_active": enhanced_service_manager.premarket_analysis_active,
        "priority_trading_active": enhanced_service_manager.priority_trading_active,
        "active_connections": len(enhanced_service_manager.active_connections),
        "market_status": market_status,
        "trading_mode": trading_mode,
        "enhanced_features": {
            "nifty100_universe": ENHANCED_MARKET_DATA_AVAILABLE,
            "premarket_analysis": True,
            "priority_trading": True,
            "enhanced_signals": True
        },
        "premarket_stats": {
            "opportunities_found": len(enhanced_service_manager.premarket_opportunities),
            "priority_signals_queue": len(enhanced_service_manager.priority_signals_queue)
        },
        "configuration": {
            "environment": settings.environment,
            "debug": settings.debug,
            "telegram_configured": settings.is_telegram_configured,
            "max_signals_per_day": settings.max_signals_per_day,
            "signal_interval": settings.signal_generation_interval,
            "stock_universe": "Nifty 100",
            "total_stocks": 100,
            "priority_stocks": 50
        },
        "timestamp": datetime.now().isoformat()
    }

# ================================================================
# Enhanced WebSocket Endpoint
# ================================================================

@app.websocket("/ws/enhanced")
async def enhanced_websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for real-time communication"""
    await enhanced_service_manager.connect_websocket(websocket)
    
    try:
        # Keep connection alive and handle enhanced messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle enhanced message types
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "enhanced": True,
                        "timestamp": datetime.now().isoformat()
                    })
                elif message.get("type") == "request_enhanced_analytics":
                    if enhanced_service_manager.analytics_service:
                        try:
                            performance = enhanced_service_manager.analytics_service.get_performance_summary()
                            await websocket.send_json({
                                "type": "enhanced_analytics_update",
                                "data": performance,
                                "timestamp": datetime.now().isoformat()
                            })
                        except Exception as e:
                            logger.debug(f"Failed to send enhanced analytics: {e}")
                elif message.get("type") == "request_nifty100_overview":
                    if enhanced_service_manager.enhanced_market_service:
                        try:
                            overview = await enhanced_service_manager.enhanced_market_service.get_nifty100_overview()
                            await websocket.send_json({
                                "type": "nifty100_overview_update",
                                "data": overview,
                                "timestamp": datetime.now().isoformat()
                            })
                        except Exception as e:
                            logger.debug(f"Failed to send Nifty 100 overview: {e}")
                elif message.get("type") == "request_premarket_status":
                    await websocket.send_json({
                        "type": "premarket_status_update",
                        "data": {
                            "analysis_active": enhanced_service_manager.premarket_analysis_active,
                            "opportunities_found": len(enhanced_service_manager.premarket_opportunities),
                            "priority_signals_ready": len(enhanced_service_manager.priority_signals_queue),
                            "priority_trading_active": enhanced_service_manager.priority_trading_active
                        },
                        "timestamp": datetime.now().isoformat()
                    })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        enhanced_service_manager.disconnect_websocket(websocket)
    except Exception as e:
        logger.error(f"❌ Enhanced WebSocket error: {e}")
        enhanced_service_manager.disconnect_websocket(websocket)

# ================================================================
# Legacy Compatibility Endpoints
# ================================================================

# Keep legacy endpoints for backward compatibility
@app.get("/api/analytics/dashboard")
async def legacy_dashboard():
    """Legacy dashboard endpoint"""
    return await get_enhanced_dashboard_analytics()

@app.get("/api/market/status")
async def legacy_market_status():
    """Legacy market status endpoint"""
    try:
        if enhanced_service_manager.enhanced_market_service:
            health = await enhanced_service_manager.enhanced_market_service.get_service_health()
            return {
                "status": health.get('market_status', 'UNKNOWN'),
                "trading_mode": health.get('trading_mode', 'UNKNOWN'),
                "enhanced": True,
                "nifty100": True,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "UNKNOWN",
                "enhanced": False,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"❌ Error getting legacy market status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/signals")
async def legacy_websocket_endpoint(websocket: WebSocket):
    """Legacy WebSocket endpoint"""
    await enhanced_websocket_endpoint(websocket)

# ================================================================
# Enhanced Signal Handlers
# ================================================================

def enhanced_signal_handler_func(signum, frame):
    """Handle shutdown signals for enhanced system"""
    logger.info(f"🛑 Enhanced system received signal {signum}, shutting down gracefully...")

signal_handler.signal(signal_handler.SIGINT, enhanced_signal_handler_func)
signal_handler.signal(signal_handler.SIGTERM, enhanced_signal_handler_func)

# ================================================================
# Enhanced Application Entry Point
# ================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Ensure enhanced logs directory exists
    os.makedirs("logs/enhanced", exist_ok=True)
    
    logger.info("🚀 Starting TradeMind AI Professional - Nifty 100 Enhanced...")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level=settings.log_level.lower(),
            access_log=True,
            reload=settings.debug,
            reload_dirs=["app"] if settings.debug else None
        )
    except KeyboardInterrupt:
        logger.info("🛑 Enhanced application stopped by user")
    except Exception as e:
        logger.error(f"❌ Enhanced application startup failed: {e}")
        sys.exit(1)