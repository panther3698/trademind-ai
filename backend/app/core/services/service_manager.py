import logging
import asyncio
import traceback
from typing import Dict, List, Optional, Any
# WebSocket type will be handled dynamically

from app.services.enhanced_market_data_nifty100 import (
    EnhancedMarketDataService, MarketStatus, TradingMode
)

# Import availability flags and settings
from app.core.availability import (
    ENHANCED_MARKET_DATA_AVAILABLE, SIGNAL_LOGGING_AVAILABLE,
    ADVANCED_ML_AVAILABLE, NEWS_INTELLIGENCE_AVAILABLE,
    TELEGRAM_AVAILABLE, ZERODHA_AVAILABLE, REGIME_DETECTION_AVAILABLE,
    BACKTESTING_AVAILABLE, INTERACTIVE_TRADING_AVAILABLE,
    NEWS_SIGNAL_INTEGRATION_AVAILABLE, settings
)

# Import feature flags
from app.core.config.feature_flags import get_feature_flags_manager, is_feature_enabled

# Import services
from app.core.services.analytics_service import CorrectedAnalytics
from app.core.services.notification_service import NotificationService
from app.core.services.signal_service import SignalService
from app.core.services.news_service import NewsSignalIntegrationService
from app.core.signal_logger import InstitutionalSignalLogger
from app.services.production_signal_generator import ProductionMLSignalGenerator
from app.services.regime_detector import RegimeDetector, MarketRegime
from app.services.backtest_engine import BacktestEngine
from app.services.enhanced_news_intelligence import EnhancedNewsIntelligenceSystem

# Import new task and cache managers
from app.core.services.task_manager import TaskManager
from app.core.services.cache_manager import CacheManager

# Import ML components
from sklearn.ensemble import RandomForestClassifier

SIGNAL_GENERATOR_AVAILABLE = True
REGIME_DETECTOR_AVAILABLE = True
BACKTEST_ENGINE_AVAILABLE = True

logger = logging.getLogger(__name__)

# ================================================================
# SERVICE MANAGER CLASS
# ================================================================

class CorrectedServiceManager:
    """Service Coordinator - Manages and coordinates all trading system services"""
    
    def __init__(self):
        # Core services
        self.active_connections: List[Any] = []  # WebSocket connections
        self.analytics_service = CorrectedAnalytics()
        
        # Extracted services
        self.notification_service = NotificationService(self.analytics_service)
        
        # Market and data services
        self.enhanced_market_service = None
        self.signal_generator: Optional[ProductionMLSignalGenerator] = None
        self.signal_logger: Optional[InstitutionalSignalLogger] = None
        
        # Advanced services
        self.regime_detector: Optional[RegimeDetector] = None
        self.backtest_engine: Optional[BacktestEngine] = None
        
        # News intelligence components
        self.news_intelligence: Optional[EnhancedNewsIntelligenceSystem] = None
        self.news_cache = {}
        
        # System state tracking
        self.current_market_status = MarketStatus.CLOSED if ENHANCED_MARKET_DATA_AVAILABLE else "CLOSED"
        self.current_trading_mode = TradingMode.PRE_MARKET_ANALYSIS if ENHANCED_MARKET_DATA_AVAILABLE else "PRE_MARKET_ANALYSIS"
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.premarket_opportunities = []
        self.priority_signals_queue = []
        
        # System state
        self._is_initialized = False
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
            "interactive_telegram": False,
            "order_execution": False,
            "webhook_handler": False,
            "zerodha_connection": False,
            "news_intelligence": False,
            "news_monitoring": False,
            "breaking_news_alerts": False,
            "news_signal_integration": False
        }
        
        # Service control flags
        self.signal_generation_active = False
        self.premarket_analysis_active = False
        self.priority_trading_active = False
        self.regime_monitoring_active = False
        self.interactive_trading_active = False
        self.news_monitoring_active = False
        
        # Initialize new task and cache managers
        self.task_manager = TaskManager()
        self.cache_manager = CacheManager()
        
        # Initialize signal service (will be updated after all services are initialized)
        self.signal_service = self._create_signal_service()
    
    def _create_signal_service(self) -> SignalService:
        """Create signal service with current service references"""
        return SignalService(
            news_intelligence=self.news_intelligence,
            signal_generator=self.signal_generator,
            analytics_service=self.analytics_service,
            signal_logger=self.signal_logger,
            telegram_service=self.notification_service.get_telegram_service(),
            regime_detector=self.regime_detector,
            backtest_engine=self.backtest_engine,
            order_engine=self.notification_service.get_order_engine(),
            webhook_handler=self.notification_service.get_webhook_handler(),
            telegram_integration=None,
            enhanced_market_service=self.enhanced_market_service,
            system_health=self.system_health,
            current_regime=self.current_regime,
            regime_confidence=self.regime_confidence,
            premarket_opportunities=self.premarket_opportunities,
            priority_signals_queue=self.priority_signals_queue,
            interactive_trading_active=self.notification_service.is_interactive_trading_active(),
            signal_generation_active=self.signal_generation_active,
            premarket_analysis_active=self.premarket_analysis_active,
            priority_trading_active=self.priority_trading_active,
            news_monitoring_active=self.news_monitoring_active
        )
    
    def _update_signal_service(self):
        """Update signal service with latest service references"""
        try:
            self.signal_service = self._create_signal_service()
            logger.info("✅ Signal service updated with latest service references")
        except Exception as e:
            logger.error(f"❌ Failed to update signal service: {e}")
    
    async def initialize_all_services(self):
        """Initialize all services in the correct order"""
        try:
            logger.info("🚀 Initializing TradeMind AI Service Coordinator...")
            
            # 1. Core services (already initialized)
            logger.info("✅ Analytics service ready")
            
            # 2. Initialize signal logging
            await self._initialize_signal_logging()
            
            # 3. Initialize market data service
            await self._initialize_enhanced_market_data()
            
            # 4. Initialize news intelligence
            await self._initialize_news_intelligence()
            
            # 5. Initialize signal generator
            await self._initialize_signal_generator()
            
            # 6. Initialize notification services
            await self._initialize_notification_services()
            
            # 7. Initialize advanced services
            await self._initialize_advanced_services()
            
            # 8. Update signal service with all components
            self._update_signal_service()
            
            # 9. Start background monitoring
            await self._start_background_monitoring()
            
            # 10. Perform health check
            await self._perform_health_check()
            
            self._is_initialized = True
            logger.info("✅ TradeMind AI Service Coordinator initialized successfully!")
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ Service initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    # ================================================================
    # SERVICE INITIALIZATION METHODS
    # ================================================================
    
    async def _initialize_signal_logging(self):
        """Initialize signal logging service"""
        try:
            if SIGNAL_LOGGING_AVAILABLE and InstitutionalSignalLogger:
                self.signal_logger = InstitutionalSignalLogger("logs/enhanced")
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
    
    async def _initialize_news_intelligence(self):
        """Initialize news intelligence service"""
        try:
            if not NEWS_INTELLIGENCE_AVAILABLE:
                logger.warning("⚠️ News intelligence not available")
                return
            
            # Check feature flag
            if not is_feature_enabled("news_intelligence_enabled"):
                logger.info("ℹ️ News intelligence disabled by feature flag")
                return
            
            # Create config for news intelligence
            config = {
                "polygon_api_key": getattr(settings, 'polygon_api_key', None),
                "enable_hindi_analysis": getattr(settings, 'enable_hindi_analysis', False),
                "max_articles_per_source": 100,
                "sentiment_threshold": getattr(settings, 'news_sentiment_threshold', 0.3),
                "working_sources_count": 8,
                "failed_apis_removed": ["alpha_vantage", "eodhd", "finnhub", "newsapi"]
            }
            
            self.news_intelligence = EnhancedNewsIntelligenceSystem(config)
            await self.news_intelligence.initialize()
            
            self.system_health["news_intelligence"] = True
            self.system_health["breaking_news_alerts"] = True
            
            logger.info("✅ News intelligence service initialized")
            
        except Exception as e:
            logger.error(f"❌ News intelligence initialization failed: {e}")
            self.system_health["news_intelligence"] = False
    
    async def _initialize_signal_generator(self):
        """Initialize signal generator"""
        try:
            if not SIGNAL_GENERATOR_AVAILABLE:
                logger.warning("⚠️ Signal generator not available")
                return
            
            # Check feature flags
            if not is_feature_enabled("signal_generation_enabled"):
                logger.info("ℹ️ Signal generation disabled by feature flag")
                return
            
            if not is_feature_enabled("ml_signal_generation"):
                logger.info("ℹ️ ML signal generation disabled by feature flag")
                return
            
            if self.enhanced_market_service and self.signal_logger:
                self.signal_generator = ProductionMLSignalGenerator(
                    market_data_service=self.enhanced_market_service,
                    signal_logger=self.signal_logger
                )
                self.system_health["signal_generation"] = True
                logger.info("✅ Signal generator initialized")
            else:
                logger.warning("⚠️ Signal generator dependencies not available")
                
        except Exception as e:
            logger.error(f"❌ Signal generator initialization failed: {e}")
            self.system_health["signal_generation"] = False
    
    async def _initialize_notification_services(self):
        """Initialize notification services (Telegram, order execution, webhooks)"""
        try:
            # Initialize notification service (Telegram)
            if TELEGRAM_AVAILABLE:
                await self.notification_service.initialize_enhanced_telegram()
                self.system_health["telegram"] = True
                logger.info("✅ Notification service initialized")
            else:
                logger.warning("⚠️ Telegram not available")
            
            # Initialize interactive trading if enabled
            if is_feature_enabled("interactive_trading_enabled"):
                self.interactive_trading_active = True
                self.system_health["interactive_telegram"] = True
                logger.info("✅ Interactive trading enabled")
            else:
                logger.info("ℹ️ Interactive trading disabled by feature flag")
            
            # Initialize order execution if enabled
            if ZERODHA_AVAILABLE and is_feature_enabled("order_execution_enabled"):
                order_engine_initialized = await self.notification_service.initialize_order_engine()
                if order_engine_initialized:
                    self.system_health["order_execution"] = True
                    logger.info("✅ Order execution initialized")
                else:
                    logger.warning("⚠️ Order engine not available")
            else:
                logger.info("ℹ️ Order execution disabled by feature flag or not available")
            
            # Initialize webhook handler if enabled
            if is_feature_enabled("webhook_handler_enabled"):
                webhook_handler_initialized = await self.notification_service.initialize_webhook_handler()
                if webhook_handler_initialized:
                    self.system_health["webhook_handler"] = True
                    logger.info("✅ Webhook handler initialized")
                else:
                    logger.warning("⚠️ Webhook handler not available")
            else:
                logger.info("ℹ️ Webhook handler disabled by feature flag")
                
        except Exception as e:
            logger.error(f"❌ Notification services initialization failed: {e}")
            self.system_health["telegram"] = False
            self.system_health["interactive_telegram"] = False
            self.system_health["order_execution"] = False
            self.system_health["webhook_handler"] = False
    
    async def _initialize_advanced_services(self):
        """Initialize advanced services (regime detection, backtesting)"""
        try:
            # Initialize regime detector
            if REGIME_DETECTION_AVAILABLE and is_feature_enabled("regime_detection_enabled"):
                if self.enhanced_market_service:
                    self.regime_detector = RegimeDetector(self.enhanced_market_service)
                    self.system_health["regime_detection"] = True
                    logger.info("✅ Regime detector initialized")
                else:
                    logger.warning("⚠️ Regime detector requires market data service")
            else:
                logger.info("ℹ️ Regime detection disabled by feature flag or not available")
            
            # Initialize backtest engine
            if BACKTEST_ENGINE_AVAILABLE:
                if self.enhanced_market_service:
                    self.backtest_engine = BacktestEngine(self.enhanced_market_service)
                    self.system_health["backtesting"] = True
                    logger.info("✅ Backtest engine initialized")
                else:
                    logger.warning("⚠️ Backtest engine requires market data service")
            else:
                logger.warning("⚠️ Backtest engine not available")
                
        except Exception as e:
            logger.error(f"❌ Advanced services initialization failed: {e}")
            self.system_health["regime_detection"] = False
            self.system_health["backtesting"] = False
    
    # ================================================================
    # BACKGROUND MONITORING METHODS
    # ================================================================
    
    async def _start_background_monitoring(self):
        """Start all background monitoring tasks via TaskManager"""
        try:
            await self._start_task_manager()
            logger.info("✅ Background monitoring tasks started via TaskManager")
        except Exception as e:
            logger.error(f"❌ Background monitoring startup failed: {e}")

    async def _start_task_manager(self):
        """Start centralized task manager with all background tasks"""
        try:
            # Register all background tasks with TaskManager
            self.task_manager.register_task(
                name="market_monitoring",
                coro=self._market_monitor_loop,
                priority="critical",
                market_hours_only=False
            )
            
            self.task_manager.register_task(
                name="regime_monitoring", 
                coro=self._regime_monitor_loop,
                priority="important",
                market_hours_only=False
            )
            
            self.task_manager.register_task(
                name="news_integration",
                coro=self._news_integration_loop,
                priority="nice_to_have", 
                market_hours_only=True
            )
            
            self.task_manager.register_task(
                name="signal_generation",
                coro=self._signal_generation_loop,
                priority="critical",
                market_hours_only=True
            )
            
            # Start the task manager
            await self.task_manager.start()
            logger.info("✅ TaskManager started with all background tasks")
            
        except Exception as e:
            logger.error(f"❌ TaskManager startup failed: {e}")
            raise
    
    # ================================================================
    # MONITORING LOOPS
    # ================================================================
    
    async def _market_monitor_loop(self):
        """Market monitoring loop with caching"""
        try:
            if self.enhanced_market_service:
                # Check cache first
                cached_health = self.cache_manager.get("market_health", "market")
                if not cached_health:
                    # Get fresh data and cache it
                    health = await self.enhanced_market_service.get_service_health()
                    self.cache_manager.set("market_health", health, "market")
                    
                    self.current_market_status = health.get('market_status', 'UNKNOWN')
                    self.current_trading_mode = health.get('trading_mode', 'UNKNOWN')
                else:
                    # Use cached data
                    self.current_market_status = cached_health.get('market_status', 'UNKNOWN')
                    self.current_trading_mode = cached_health.get('trading_mode', 'UNKNOWN')
                    
        except Exception as e:
            logger.error(f"❌ Market monitoring error: {e}")

    async def _regime_monitor_loop(self):
        """Regime monitoring loop with caching"""
        try:
            if self.regime_detector:
                # Check cache first
                cached_regime = self.cache_manager.get("current_regime", "regime")
                if not cached_regime:
                    # Detect current regime and cache it
                    regime_data = await self.regime_detector.detect_current_regime()
                    self.cache_manager.set("current_regime", regime_data, "regime")
                    
                    new_regime = regime_data.get('regime', MarketRegime.SIDEWAYS)
                    new_confidence = regime_data.get('confidence', 0.5)
                    
                    if new_regime != self.current_regime:
                        logger.info(f"🔄 Regime change: {self.current_regime} → {new_regime}")
                        self.current_regime = new_regime
                        self.regime_confidence = new_confidence
                        
                        # Notify via notification service
                        await self.notification_service.send_regime_change_notification(
                            str(self.current_regime), 
                            str(new_regime), 
                            new_confidence
                        )
            
        except Exception as e:
            logger.error(f"❌ Regime monitoring error: {e}")

    async def _news_integration_loop(self):
        """News integration loop wrapper for TaskManager"""
        try:
            if self.news_intelligence:
                # Use existing news integration logic but with caching
                # This is a placeholder - implement based on actual news integration
                self.news_monitoring_active = True
                self.system_health["news_monitoring"] = True
                self.system_health["breaking_news_alerts"] = True
                logger.debug("News integration loop running")
        except Exception as e:
            logger.error(f"❌ News integration error: {e}")

    async def _signal_generation_loop(self):
        """Signal generation loop wrapper for TaskManager"""
        try:
            if self.signal_service and self.signal_generation_active:
                # Use existing signal generation logic
                await self.signal_service.generate_signals()
                logger.debug("Signal generation loop running")
        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
    
    # ================================================================
    # SYSTEM HEALTH AND COORDINATION
    # ================================================================
    
    async def _perform_health_check(self):
        """Perform comprehensive system health check"""
        try:
            # Get notification service health
            notification_health = self.notification_service.get_system_health()
            
            # Update system health status
            health_status = {
                "telegram": notification_health.get("telegram", False),
                "enhanced_market_data": self.enhanced_market_service is not None,
                "nifty100_universe": self.enhanced_market_service is not None,
                "premarket_analysis": self.premarket_analysis_active,
                "priority_trading": self.priority_trading_active,
                "signal_generation": self.signal_generator is not None,
                "regime_detection": self.regime_detector is not None,
                "backtesting": self.backtest_engine is not None,
                "analytics": True,
                "interactive_telegram": notification_health.get("interactive_telegram", False),
                "order_execution": notification_health.get("order_execution", False),
                "webhook_handler": notification_health.get("webhook_handler", False),
                "zerodha_connection": notification_health.get("zerodha_connection", False),
                "news_intelligence": self.news_intelligence is not None,
                "news_monitoring": self.news_monitoring_active,
                "breaking_news_alerts": self.news_intelligence is not None,
                "news_signal_integration": False
            }
            
            self.system_health.update(health_status)
            logger.info("✅ System health check completed")
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
    
    # ================================================================
    # PUBLIC INTERFACE METHODS
    # ================================================================
    
    def get_system_health(self) -> Dict[str, bool]:
        """Get current system health status"""
        return self.system_health.copy()
    
    def get_analytics_service(self) -> CorrectedAnalytics:
        """Get analytics service"""
        return self.analytics_service
    
    def get_notification_service(self) -> NotificationService:
        """Get notification service"""
        return self.notification_service
    
    def get_signal_service(self) -> SignalService:
        """Get signal service"""
        return self.signal_service
    
    def get_market_service(self) -> Optional[EnhancedMarketDataService]:
        """Get market data service"""
        return self.enhanced_market_service
    
    def get_news_intelligence(self) -> Optional[EnhancedNewsIntelligenceSystem]:
        """Get news intelligence service"""
        return self.news_intelligence
    
    def get_regime_detector(self) -> Optional[RegimeDetector]:
        """Get regime detector service"""
        return self.regime_detector
    
    def get_backtest_engine(self) -> Optional[BacktestEngine]:
        """Get backtest engine service"""
        return self.backtest_engine
    
    def is_initialized(self) -> bool:
        """Check if all services are initialized"""
        return self._is_initialized
    
    def get_initialization_error(self) -> Optional[str]:
        """Get initialization error if any"""
        return self.initialization_error
    
    async def shutdown(self):
        """Shutdown all services gracefully"""
        try:
            logger.info("🛑 Shutting down Service Coordinator...")
            
            # Stop TaskManager (handles all background tasks)
            await self.task_manager.shutdown()
            
            # Stop signal generation
            if self.signal_service:
                await self.signal_service.stop_signal_generation()
            
            logger.info("✅ Service Coordinator shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

    def get_task_health(self) -> Dict[str, Any]:
        """Get health status of all background tasks"""
        return {
            "task_health": self.task_manager.get_health(),
            "cache_stats": self.cache_manager.get_metrics(),
            "market_hours": self.task_manager.is_market_hours(),
            "active_tasks": self.task_manager.get_active_tasks(),
            "resource_usage": self.task_manager.get_resource_usage()
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return self.cache_manager.get_metrics()

# ================================================================
# GLOBAL SERVICE MANAGER INSTANCE (for dependency injection)
# ================================================================

# Global service manager instance for dependency injection
_global_service_manager: Optional[CorrectedServiceManager] = None

def get_global_service_manager() -> Optional[CorrectedServiceManager]:
    """Get the global service manager instance"""
    return _global_service_manager

def set_global_service_manager(service_manager: CorrectedServiceManager) -> None:
    """Set the global service manager instance"""
    global _global_service_manager
    _global_service_manager = service_manager