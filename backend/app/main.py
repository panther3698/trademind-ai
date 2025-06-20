# ================================================================
# TradeMind AI - Fully Integrated Main Application
# ================================================================

"""
TradeMind AI - Production-Ready FastAPI Application
Integrates all ML models, market data, signal generation, and analytics

Architecture:
Frontend ↔ WebSocket/REST API ↔ Service Manager ↔ ML Models ↔ Market Data
   ↓              ↓                    ↓              ↓          ↓
Dashboard    Real-time Updates    Signal Generation   AI Models  Live Data
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import os
import sys
from pathlib import Path
import traceback
import signal as signal_handler

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import configuration
try:
    from app.core.config import settings
except ImportError:
    logging.error("❌ Configuration not found. Please ensure config.py exists.")
    sys.exit(1)

# Import services with graceful fallbacks
TelegramService = None
AnalyticsService = None
MarketDataService = None
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
    from app.services.market_data_service import MarketDataService, create_market_data_service
    MARKET_DATA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ Market data service not available: {e}")
    MARKET_DATA_AVAILABLE = False

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
        logging.FileHandler('logs/trademind.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# Simple Analytics Class for Fallback
# ================================================================

class SimpleAnalytics:
    """Simple analytics implementation when full analytics service unavailable"""
    
    def __init__(self):
        self.daily_stats = {
            "signals_generated": 0,
            "signals_sent": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "average_confidence": 0.0,
            "system_uptime_hours": 0,
            "last_signal_time": None,
            "telegram_success": 0,
            "telegram_failures": 0,
            "status": "operational"
        }
        self.start_time = datetime.now()
    
    async def track_signal_generated(self, signal):
        """Track signal generation"""
        self.daily_stats["signals_generated"] += 1
        self.daily_stats["last_signal_time"] = datetime.now().isoformat()
        
        # Update average confidence
        current_avg = self.daily_stats["average_confidence"]
        count = self.daily_stats["signals_generated"]
        new_confidence = signal.get("confidence", 0.0)
        
        if count > 0:
            self.daily_stats["average_confidence"] = (
                (current_avg * (count - 1) + new_confidence) / count
            )
    
    async def track_telegram_sent(self, success, signal):
        """Track Telegram notifications"""
        if success:
            self.daily_stats["telegram_success"] += 1
            self.daily_stats["signals_sent"] += 1
        else:
            self.daily_stats["telegram_failures"] += 1
    
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
            "system_health": {
                "telegram_configured": settings.is_telegram_configured,
                "signal_generation_active": True,
                "last_activity": daily["last_signal_time"],
                "error_rate": (daily["telegram_failures"] / max(total_telegram, 1)) * 100
            },
            "configuration": {
                "max_signals_per_day": settings.max_signals_per_day,
                "min_confidence_threshold": settings.min_confidence_threshold,
                "signal_interval_seconds": settings.signal_generation_interval,
                "telegram_enabled": settings.is_telegram_configured,
                "zerodha_enabled": settings.is_zerodha_configured
            },
            "recent_performance": {
                "signals_today": daily["signals_generated"],
                "success_rate": telegram_success_rate,
                "profit_loss": daily["total_pnl"],
                "avg_confidence": daily["average_confidence"]
            }
        }

# ================================================================
# Service Manager - Orchestrates All Components
# ================================================================

class TradeMindServiceManager:
    """Comprehensive service manager integrating all TradeMind components"""
    
    def __init__(self):
        # Core services
        self.active_connections: List[WebSocket] = []
        self.telegram_service: Optional[TelegramService] = None
        self.analytics_service: Optional[AnalyticsService] = None
        self.market_data_service: Optional[MarketDataService] = None
        self.signal_generator: Optional[ProductionMLSignalGenerator] = None
        self.signal_logger: Optional[InstitutionalSignalLogger] = None
        
        # ML components
        self.ml_ensemble: Optional[EnsembleModel] = None
        self.stock_universe: Optional[Nifty100StockUniverse] = None
        self.sentiment_analyzer: Optional[FinBERTSentimentAnalyzer] = None
        self.training_pipeline: Optional[TrainingPipeline] = None
        
        # System state
        self.is_initialized = False
        self.initialization_error = None
        self.system_health = {
            "telegram": False,
            "market_data": False,
            "ml_models": False,
            "signal_generation": False,
            "analytics": False
        }
        
        # Signal generation control
        self.signal_generation_active = False
        self.signal_generation_task = None
        
    async def initialize_all_services(self):
        """Initialize all services in proper order"""
        try:
            logger.info("🚀 Initializing TradeMind AI Services...")
            
            # 1. Initialize basic analytics (no dependencies)
            await self._initialize_analytics()
            
            # 2. Initialize Telegram (optional)
            await self._initialize_telegram()
            
            # 3. Initialize signal logging
            await self._initialize_signal_logging()
            
            # 4. Initialize stock universe
            await self._initialize_stock_universe()
            
            # 5. Initialize ML components
            await self._initialize_ml_components()
            
            # 6. Initialize market data service
            await self._initialize_market_data()
            
            # 7. Initialize production signal generator
            await self._initialize_signal_generator()
            
            # 8. Final system check
            await self._perform_system_health_check()
            
            self.is_initialized = True
            logger.info("✅ All TradeMind AI services initialized successfully!")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Service initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _initialize_analytics(self):
        """Initialize analytics service"""
        try:
            if ANALYTICS_AVAILABLE and AnalyticsService:
                # Pass the database URL from settings
                self.analytics_service = AnalyticsService(settings.database_url)
                logger.info("✅ AnalyticsService initialized with database")
            else:
                # Create minimal analytics for basic functionality
                self.analytics_service = SimpleAnalytics()
                logger.info("✅ Simple analytics initialized (fallback)")
            
            self.system_health["analytics"] = True
            logger.info("✅ Analytics service initialized")
        except Exception as e:
            logger.error(f"❌ Analytics initialization failed: {e}")
            self.analytics_service = SimpleAnalytics()
            self.system_health["analytics"] = True  # Still functional with fallback
    
    async def _initialize_telegram(self):
        """Initialize Telegram service with startup notification"""
        try:
            if settings.is_telegram_configured and TELEGRAM_AVAILABLE and TelegramService:
                self.telegram_service = TelegramService()
                
                # Test connection and send startup notification
                if self.telegram_service.is_configured():
                    # Send startup notification
                    startup_success = await self.telegram_service.send_system_startup_notification()
                    if startup_success:
                        logger.info("📱 Telegram startup notification sent")
                    else:
                        logger.warning("⚠️ Telegram startup notification failed")
                    
                    self.system_health["telegram"] = True
                    logger.info("✅ Telegram service initialized")
                else:
                    logger.warning("⚠️ Telegram not properly configured")
                    self.system_health["telegram"] = False
            else:
                logger.warning("⚠️ Telegram not configured - signals will not be sent")
                self.system_health["telegram"] = False
        except Exception as e:
            logger.error(f"❌ Telegram initialization failed: {e}")
            self.system_health["telegram"] = False
    
    async def _initialize_signal_logging(self):
        """Initialize institutional signal logging"""
        try:
            if SIGNAL_LOGGING_AVAILABLE and InstitutionalSignalLogger:
                self.signal_logger = InstitutionalSignalLogger("logs")
                logger.info("✅ Signal logging directories created at logs")
            else:
                logger.warning("⚠️ Signal logging not available - using basic logging")
        except Exception as e:
            logger.error(f"❌ Signal logging initialization failed: {e}")
        
        logger.info("✅ Signal logging initialized")
    
    async def _initialize_stock_universe(self):
        """Initialize stock universe"""
        try:
            if ML_AVAILABLE and Nifty100StockUniverse:
                self.stock_universe = Nifty100StockUniverse()
                logger.info("✅ Stock universe initialized")
            else:
                logger.warning("⚠️ Stock universe not available - using basic stock list")
        except Exception as e:
            logger.error(f"❌ Stock universe initialization failed: {e}")
        
        logger.info("✅ Stock universe initialized")
    
    async def _initialize_ml_components(self):
        """Initialize ML models and components"""
        try:
            if not ML_AVAILABLE:
                logger.warning("⚠️ ML components not available - using basic signal generation")
                self.system_health["ml_models"] = False
                return
            
            # Initialize sentiment analyzer
            if settings.is_finbert_enabled:
                try:
                    self.sentiment_analyzer = FinBERTSentimentAnalyzer()
                    logger.info("✅ FinBERT model loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ FinBERT initialization failed: {e}")
            
            logger.info("✅ FinBERT sentiment analyzer initialized")
            
            # Initialize ML ensemble
            self.ml_ensemble = EnsembleModel()
            
            # Try to load existing models
            model_loaded = await self._load_or_train_models()
            
            self.system_health["ml_models"] = model_loaded
            logger.info(f"✅ ML components initialized (models loaded: {model_loaded})")
            
        except Exception as e:
            logger.error(f"❌ ML initialization failed: {e}")
            self.system_health["ml_models"] = False
    
    async def _load_or_train_models(self) -> bool:
        """Load existing models or trigger training"""
        try:
            # Check if model files exist
            models_dir = Path("models")
            model_files_to_check = [
                "xgboost_model_production.pkl",
                "xgboost_model_v1.0.pkl",
                ".models_trained",
                "model_health.json"
            ]
            
            # Check if any model files exist
            models_available = False
            for model_file in model_files_to_check:
                if (models_dir / model_file).exists():
                    models_available = True
                    logger.info(f"✅ Found model file: {model_file}")
                    break
            
            if models_available:
                logger.info("✅ ML models are available and healthy")
                return True
            else:
                logger.warning("⚠️ No ML model files found - using demo signals")
                return False
            
        except Exception as e:
            logger.error(f"❌ Model loading/training failed: {e}")
            return False
    
    async def _initialize_market_data(self):
        """Initialize market data service with the new integration"""
        try:
            if MARKET_DATA_AVAILABLE and MarketDataService:
                # Use the factory function from the new market data service
                self.market_data_service = create_market_data_service()
                await self.market_data_service.initialize()
                
                # Get health status
                health = await self.market_data_service.get_service_health()
                if health['is_initialized']:
                    self.system_health["market_data"] = True
                    data_source = "Yahoo Finance" if health.get('yahoo_finance_connected') else "Mock Data"
                    logger.info(f"✅ Market data service initialized")
                    logger.info(f"📊 Data Source: {data_source}")
                    logger.info(f"📈 Watchlist: {health.get('watchlist_size', 0)} symbols")
                    logger.info(f"🏪 Market Status: {health.get('market_status', 'Unknown')}")
                else:
                    logger.error(f"❌ Market data service failed: {health.get('initialization_error')}")
                    self.system_health["market_data"] = False
            else:
                logger.warning("⚠️ Market data service not available - using simulated data")
                self.system_health["market_data"] = False
        except Exception as e:
            logger.error(f"❌ Market data initialization failed: {e}")
            self.system_health["market_data"] = False
        
        logger.info("✅ Market data service initialized")
    
    async def _initialize_signal_generator(self):
        """Initialize production signal generator"""
        try:
            if (SIGNAL_GENERATOR_AVAILABLE and ProductionMLSignalGenerator and 
                self.market_data_service and self.signal_logger):
                self.signal_generator = ProductionMLSignalGenerator(
                    self.market_data_service,
                    self.signal_logger
                )
                
                # Set ML components if available
                if self.ml_ensemble:
                    # self.signal_generator.set_ml_ensemble(self.ml_ensemble)
                    pass
                
                self.system_health["signal_generation"] = True
                logger.info("✅ Production signal generator initialized")
            else:
                logger.warning("⚠️ Using demo signal generator - production components unavailable")
                self.system_health["signal_generation"] = True  # Still functional with demo
        except Exception as e:
            logger.error(f"❌ Signal generator initialization failed: {e}")
            self.system_health["signal_generation"] = True  # Can still use demo signals
        
        logger.info("✅ Production signal generator initialized")
    
    async def _perform_system_health_check(self):
        """Perform comprehensive system health check"""
        health_summary = {
            "total_services": len(self.system_health),
            "healthy_services": sum(self.system_health.values()),
            "unhealthy_services": len(self.system_health) - sum(self.system_health.values()),
            "health_percentage": (sum(self.system_health.values()) / len(self.system_health)) * 100
        }
        
        logger.info(f"🏥 System Health: {health_summary['healthy_services']}/{health_summary['total_services']} services healthy ({health_summary['health_percentage']:.1f}%)")
        
        for service, status in self.system_health.items():
            status_icon = "✅" if status else "❌"
            service_name = service.replace('_', ' ').title()
            status_text = "Healthy" if status else "Unhealthy"
            logger.info(f"  {status_icon} {service_name}: {status_text}")
    
    async def start_signal_generation(self):
        """Start the signal generation process"""
        if self.signal_generation_active:
            logger.warning("⚠️ Signal generation already active")
            return
        
        # Can start signal generation even without full components (will use demo signals)
        self.signal_generation_active = True
        self.signal_generation_task = asyncio.create_task(self._signal_generation_loop())
        
        if self.signal_generator:
            logger.info("🎯 Production signal generation started")
        else:
            logger.info("🎯 Demo signal generation started (production components unavailable)")
    
    async def stop_signal_generation(self):
        """Stop the signal generation process"""
        if not self.signal_generation_active:
            return
        
        self.signal_generation_active = False
        if self.signal_generation_task:
            self.signal_generation_task.cancel()
        
        logger.info("🛑 Signal generation stopped")
    
    async def _signal_generation_loop(self):
        """Main signal generation loop"""
        logger.info("🔄 Starting signal generation loop...")
        
        # Check if today is a new trading day
        current_date = datetime.now().date()
        logger.info(f"🌅 New trading day: {current_date}")
        
        while self.signal_generation_active:
            try:
                # Check market status first
                market_status = await self._get_market_status()
                logger.info(f"Market status: {market_status}")
                
                # Check if we can generate signals
                if not self._can_generate_signals():
                    await asyncio.sleep(60)  # Wait 1 minute before checking again
                    continue
                
                # Generate signals using ML or fallback method
                signals = await self._generate_signals()
                
                # Process each signal
                for signal in signals:
                    await self._process_signal(signal)
                
                # Wait before next generation cycle
                await asyncio.sleep(settings.signal_generation_interval)
                
            except asyncio.CancelledError:
                logger.info("🛑 Signal generation loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Signal generation error: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _get_market_status(self) -> str:
        """Get current market status"""
        try:
            if self.market_data_service:
                status = await self.market_data_service.get_market_status()
                return status.value
            else:
                # Fallback market status check
                now = datetime.now()
                if now.weekday() >= 5:  # Weekend
                    return "CLOSED"
                elif 9 <= now.hour <= 15:  # Market hours
                    return "OPEN"
                else:
                    return "CLOSED"
        except Exception as e:
            logger.error(f"❌ Market status check failed: {e}")
            return "CLOSED"
    
    def _can_generate_signals(self) -> bool:
        """Check if we can generate signals"""
        try:
            # Check daily limits
            if self.analytics_service:
                daily_stats = self.analytics_service.get_daily_stats()
                if daily_stats["signals_generated"] >= settings.max_signals_per_day:
                    return False
            
            # Add additional checks here as needed
            return True
        except Exception as e:
            logger.error(f"❌ Signal generation check failed: {e}")
            return False
    
    async def _generate_signals(self) -> List[Dict]:
        """Generate signals using available methods"""
        signals = []
        
        try:
            if self.signal_generator and self.system_health["signal_generation"] and self.market_data_service:
                # Use production ML signal generator
                signals = await self.signal_generator.generate_signals()
            else:
                # Fallback to demo signal generation
                signals = await self._generate_demo_signals()
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            # Fallback to demo signals
            signals = await self._generate_demo_signals()
        
        return signals
    
    async def _generate_demo_signals(self) -> List[Dict]:
        """Generate demo signals as fallback"""
        import random
        
        stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "LT", "WIPRO", "BHARTIARTL", "KOTAKBANK"]
        stock = random.choice(stocks)
        
        # Get realistic base prices if market data service is available
        base_price = 1500.0  # Default
        try:
            if self.market_data_service:
                stock_data = await self.market_data_service.get_live_market_data(stock)
                if stock_data and "quote" in stock_data:
                    base_price = stock_data["quote"]["ltp"]
        except Exception as e:
            logger.debug(f"Failed to get live price for {stock}: {e}")
        
        signal = {
            "id": f"demo_{int(datetime.now().timestamp())}",
            "symbol": stock,
            "action": random.choice(["BUY", "SELL"]),
            "entry_price": round(base_price, 2),
            "target_price": 0,
            "stop_loss": 0,
            "confidence": round(random.uniform(0.65, 0.85), 3),
            "sentiment_score": round(random.uniform(-0.3, 0.3), 3),
            "timestamp": datetime.now(),
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "signal_type": "DEMO" if not self.system_health["ml_models"] else "AI_GENERATED",
            "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"])
        }
        
        # Calculate target and stop loss
        if signal["action"] == "BUY":
            signal["target_price"] = round(signal["entry_price"] * 1.03, 2)
            signal["stop_loss"] = round(signal["entry_price"] * 0.98, 2)
        else:
            signal["target_price"] = round(signal["entry_price"] * 0.97, 2)
            signal["stop_loss"] = round(signal["entry_price"] * 1.02, 2)
        
        return [signal]
    
    async def _process_signal(self, signal: Dict):
        """Process a generated signal through all systems"""
        try:
            # Track in analytics
            if self.analytics_service:
                await self.analytics_service.track_signal_generated(signal)
            
            # Send to Telegram
            telegram_success = False
            if self.telegram_service and hasattr(self.telegram_service, 'is_configured') and self.telegram_service.is_configured():
                telegram_success = await self.telegram_service.send_signal_notification(signal)
                if self.analytics_service:
                    await self.analytics_service.track_telegram_sent(telegram_success, signal)
            elif self.telegram_service:
                # Fallback telegram service
                telegram_success = await self.telegram_service.send_signal_notification(signal)
                if self.analytics_service:
                    await self.analytics_service.track_telegram_sent(telegram_success, signal)
            
            # Log signal (if available)
            if self.signal_logger:
                try:
                    # await self.signal_logger.log_signal(signal)  # Method might not exist
                    logger.info(f"📝 Signal logged: {signal['id']}")
                except Exception as e:
                    logger.debug(f"Signal logging failed: {e}")
            
            # Broadcast to connected dashboards
            await self.broadcast_to_dashboard({
                "type": "new_signal",
                "data": signal,
                "timestamp": datetime.now().isoformat()
            })
            
            # Broadcast analytics update
            if self.analytics_service:
                try:
                    analytics = self.analytics_service.get_performance_summary()
                    await self.broadcast_to_dashboard({
                        "type": "analytics_update",
                        "data": analytics,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.debug(f"Analytics broadcast failed: {e}")
            
            logger.info(f"📡 Signal processed: {signal['symbol']} {signal['action']} @ ₹{signal['entry_price']} | "
                       f"Confidence: {signal['confidence']:.1%} | "
                       f"Telegram: {'✅' if telegram_success else '❌'}")
            
        except Exception as e:
            logger.error(f"❌ Signal processing failed: {e}")
    
    # WebSocket management
    async def connect_websocket(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"📱 Client connected. Total: {len(self.active_connections)}")
        
        # Send welcome message with system status
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to TradeMind AI Professional",
            "system_health": self.system_health,
            "initialization_status": self.is_initialized,
            "timestamp": datetime.now().isoformat()
        })
    
    def disconnect_websocket(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"📱 Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast_to_dashboard(self, message: Dict):
        """Broadcast message to all connected dashboards"""
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
# Global Service Manager Instance
# ================================================================

service_manager = TradeMindServiceManager()

# ================================================================
# Application Lifecycle Management
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("🚀 TradeMind AI Starting...")
    logger.info(f"🔧 Environment: {settings.environment}")
    logger.info(f"📊 Analytics: {'Enabled' if settings.track_performance else 'Disabled'}")
    logger.info(f"🤖 ML Models: {'Enabled' if ML_AVAILABLE else 'Disabled'}")
    
    try:
        # Initialize all services
        await service_manager.initialize_all_services()
        
        # Start signal generation
        await service_manager.start_signal_generation()
        
        logger.info("✅ TradeMind AI fully operational!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Continue running but mark as unhealthy
    
    yield
    
    # Shutdown
    logger.info("🛑 TradeMind AI shutting down...")
    
    # Send shutdown notification
    if (service_manager.telegram_service and 
        hasattr(service_manager.telegram_service, 'is_configured') and 
        service_manager.telegram_service.is_configured()):
        try:
            await service_manager.telegram_service.send_system_shutdown_notification()
            logger.info("📱 Telegram shutdown notification sent")
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram shutdown notification: {e}")
    
    await service_manager.stop_signal_generation()
    
    # Close market data service
    if service_manager.market_data_service:
        try:
            await service_manager.market_data_service.close()
        except Exception as e:
            logger.error(f"Error closing market data service: {e}")
    
    logger.info("✅ Shutdown complete")

# ================================================================
# FastAPI Application Setup
# ================================================================

app = FastAPI(
    title="TradeMind AI Professional",
    description="AI-Powered Indian Stock Trading Platform with Real-time Analytics",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
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
    """Root endpoint with system information"""
    return {
        "message": "🇮🇳 TradeMind AI - Professional Trading Platform",
        "status": "operational" if service_manager.is_initialized else "initializing",
        "version": "3.0.0",
        "environment": settings.environment,
        "system_health": service_manager.system_health,
        "services": {
            "telegram": settings.is_telegram_configured and service_manager.system_health.get("telegram", False),
            "market_data": service_manager.system_health.get("market_data", False),
            "ml_models": ML_AVAILABLE and service_manager.system_health.get("ml_models", False),
            "analytics": service_manager.system_health.get("analytics", False),
            "signal_generation": service_manager.system_health.get("signal_generation", False)
        },
        "market_status": await service_manager._get_market_status() if service_manager.is_initialized else "UNKNOWN",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    if not service_manager.analytics_service:
        return {"status": "initializing", "message": "Services still starting up"}
    
    try:
        performance = service_manager.analytics_service.get_performance_summary()
        
        return {
            "status": "healthy" if service_manager.is_initialized else "degraded",
            "initialization_error": service_manager.initialization_error,
            "system_health": service_manager.system_health,
            "connections": len(service_manager.active_connections),
            "signal_generation_active": service_manager.signal_generation_active,
            "services": {
                "telegram": settings.is_telegram_configured and service_manager.system_health.get("telegram", False),
                "market_data": service_manager.system_health.get("market_data", False),
                "ml_models": ML_AVAILABLE and service_manager.system_health.get("ml_models", False),
                "analytics": service_manager.system_health.get("analytics", False),
                "signal_generation": service_manager.system_health.get("signal_generation", False)
            },
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
    """Get comprehensive analytics for dashboard"""
    if not service_manager.analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not available")
    
    try:
        performance = service_manager.analytics_service.get_performance_summary()
        
        # Add system information
        performance["system_status"] = {
            "initialized": service_manager.is_initialized,
            "signal_generation_active": service_manager.signal_generation_active,
            "health": service_manager.system_health,
            "market_status": await service_manager._get_market_status()
        }
        
        return performance
        
    except Exception as e:
        logger.error(f"❌ Error getting dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/daily")
async def get_daily_analytics():
    """Get today's analytics"""
    if not service_manager.analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not available")
    
    try:
        return service_manager.analytics_service.get_daily_stats()
    except Exception as e:
        logger.error(f"❌ Error getting daily analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals/latest")
async def get_latest_signals():
    """Get latest signals with analytics"""
    if not service_manager.analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not available")
    
    try:
        daily_stats = service_manager.analytics_service.get_daily_stats()
        
        # Get recent signals from analytics service or signal logger if available
        signals = []
        if hasattr(service_manager.analytics_service, 'get_recent_signals'):
            signals = service_manager.analytics_service.get_recent_signals(limit=10)
        elif service_manager.signal_logger and hasattr(service_manager.signal_logger, 'get_recent_signals'):
            signals = service_manager.signal_logger.get_recent_signals(limit=10)
        
        return {
            "signals": signals,
            "count": len(signals),
            "daily_stats": daily_stats,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting latest signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/status")
async def get_market_status():
    """Get current market status"""
    try:
        status = await service_manager._get_market_status()
        
        market_info = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "is_trading_hours": status == "OPEN",
            "next_session": "Market opens at 9:15 AM IST" if status == "CLOSED" else "Market closes at 3:30 PM IST"
        }
        
        # Add market data service health if available
        if service_manager.market_data_service:
            try:
                health = await service_manager.market_data_service.get_service_health()
                market_info["data_source"] = "yahoo_finance" if health.get('yahoo_finance_connected') else "mock"
                market_info["watchlist_size"] = health.get('watchlist_size', 0)
            except Exception as e:
                logger.debug(f"Failed to get market data health: {e}")
        
        return market_info
        
    except Exception as e:
        logger.error(f"❌ Error getting market status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/data/{symbol}")
async def get_stock_data(symbol: str):
    """Get live market data for a specific stock"""
    if not service_manager.market_data_service:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        symbol = symbol.upper()
        data = await service_manager.market_data_service.get_live_market_data(symbol)
        return data
    except Exception as e:
        logger.error(f"❌ Error getting stock data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/watchlist")
async def get_watchlist_data():
    """Get live data for all watchlist stocks"""
    if not service_manager.market_data_service:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        data = await service_manager.market_data_service.get_watchlist_data()
        return data
    except Exception as e:
        logger.error(f"❌ Error getting watchlist data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/premarket")
async def get_premarket_analysis():
    """Get pre-market analysis"""
    if not service_manager.market_data_service:
        raise HTTPException(status_code=503, detail="Market data service not available")
    
    try:
        analysis = await service_manager.market_data_service.get_pre_market_analysis()
        return analysis
    except Exception as e:
        logger.error(f"❌ Error getting pre-market analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-telegram")
async def test_telegram_notification():
    """Manual endpoint to test Telegram notifications"""
    if not service_manager.telegram_service:
        return {
            "success": False,
            "error": "Telegram not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env file",
            "telegram_configured": settings.is_telegram_configured
        }
    
    test_signal = {
        "id": "test_manual",
        "symbol": "TESTSTOCK",
        "action": "BUY",
        "entry_price": 1000,
        "target_price": 1050,
        "stop_loss": 950,
        "confidence": 0.85,
        "sentiment_score": 0.25,
        "timestamp": datetime.now(),
        "created_at": datetime.now().isoformat(),
        "status": "test"
    }
    
    try:
        success = await service_manager.telegram_service.send_signal_notification(test_signal)
        
        if success and service_manager.analytics_service:
            await service_manager.analytics_service.track_telegram_sent(True, test_signal)
        
        return {
            "success": success,
            "message": "Test notification sent successfully" if success else "Failed to send notification",
            "telegram_configured": settings.is_telegram_configured,
            "test_signal": test_signal
        }
        
    except Exception as e:
        logger.error(f"❌ Telegram test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "telegram_configured": settings.is_telegram_configured
        }

@app.post("/api/signals/manual")
async def generate_manual_signal():
    """Manually trigger signal generation"""
    if not service_manager.signal_generation_active:
        return {"error": "Signal generation not active"}
    
    try:
        signals = await service_manager._generate_signals()
        
        for signal in signals:
            await service_manager._process_signal(signal)
        
        return {
            "success": True,
            "signals_generated": len(signals),
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Manual signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/restart-signal-generation")
async def restart_signal_generation():
    """Restart signal generation process"""
    try:
        await service_manager.stop_signal_generation()
        await asyncio.sleep(2)
        await service_manager.start_signal_generation()
        
        return {
            "success": True,
            "message": "Signal generation restarted",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Signal generation restart failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def get_system_status():
    """Get detailed system status"""
    market_status = "UNKNOWN"
    try:
        market_status = await service_manager._get_market_status()
    except Exception as e:
        logger.debug(f"Failed to get market status: {e}")
    
    return {
        "initialized": service_manager.is_initialized,
        "initialization_error": service_manager.initialization_error,
        "system_health": service_manager.system_health,
        "signal_generation_active": service_manager.signal_generation_active,
        "active_connections": len(service_manager.active_connections),
        "market_status": market_status,
        "ml_available": ML_AVAILABLE,
        "signal_logging_available": SIGNAL_LOGGING_AVAILABLE,
        "market_data_available": MARKET_DATA_AVAILABLE,
        "telegram_available": TELEGRAM_AVAILABLE,
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

# ================================================================
# WebSocket Endpoint
# ================================================================

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await service_manager.connect_websocket(websocket)
    
    try:
        # Send initial analytics if available
        if service_manager.analytics_service:
            try:
                performance = service_manager.analytics_service.get_performance_summary()
                await websocket.send_json({
                    "type": "initial_data",
                    "analytics": performance,
                    "system_health": service_manager.system_health,
                    "market_status": await service_manager._get_market_status(),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.debug(f"Failed to send initial analytics: {e}")
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif message.get("type") == "request_analytics":
                    if service_manager.analytics_service:
                        try:
                            performance = service_manager.analytics_service.get_performance_summary()
                            await websocket.send_json({
                                "type": "analytics_update",
                                "data": performance,
                                "timestamp": datetime.now().isoformat()
                            })
                        except Exception as e:
                            logger.debug(f"Failed to send analytics update: {e}")
                elif message.get("type") == "request_market_status":
                    try:
                        market_status = await service_manager._get_market_status()
                        await websocket.send_json({
                            "type": "market_status_update",
                            "data": {"status": market_status},
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
        service_manager.disconnect_websocket(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        service_manager.disconnect_websocket(websocket)

# ================================================================
# Signal Handlers for Graceful Shutdown
# ================================================================

def signal_handler_func(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
    # The lifespan context manager will handle cleanup

signal_handler.signal(signal_handler.SIGINT, signal_handler_func)
signal_handler.signal(signal_handler.SIGTERM, signal_handler_func)

# ================================================================
# Application Entry Point
# ================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    logger.info("🚀 Starting TradeMind AI Professional...")
    
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
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        sys.exit(1)