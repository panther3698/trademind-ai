import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, Callable
import logging

from app.core.config import settings
from app.core.services.analytics_service import CorrectedAnalytics
from app.services.enhanced_telegram_service import TelegramService
from app.services.zerodha_order_engine import ZerodhaOrderEngine, OrderType, ExchangeType, OrderResult
from app.services.telegram_webhook_handler import TelegramWebhookHandler
from utils.profiling import profile_timing

logger = logging.getLogger(__name__)

class NotificationService:
    """Notification service for handling telegram notifications and interactive trading"""
    
    def __init__(self, analytics_service: CorrectedAnalytics):
        self.analytics_service = analytics_service
        
        # Telegram and notification components
        self.telegram_service: Optional[TelegramService] = None
        self.order_engine: Optional[ZerodhaOrderEngine] = None
        self.webhook_handler: Optional[TelegramWebhookHandler] = None
        
        # System health tracking
        self.system_health: Dict[str, bool] = {
            "telegram": False,
            "interactive_telegram": False,
            "order_execution": False,
            "webhook_handler": False,
            "zerodha_connection": False,
            "news_intelligence": False,
            "news_monitoring": False,
            "breaking_news_alerts": False,
            "news_signal_integration": False
        }
        
        # Interactive trading state
        self.interactive_trading_active = False
    
    async def initialize_enhanced_telegram(self) -> bool:
        """Initialize Enhanced Telegram Service"""
        try:
            if settings.is_telegram_configured and TelegramService:
                self.telegram_service = TelegramService(
                    bot_token=settings.telegram_bot_token,
                    chat_id=settings.telegram_chat_id,
                    redis_url=settings.redis_url
                )
                
                if self.telegram_service.is_configured():
                    # Send enhanced startup notification
                    startup_message = (
                        "🚀 <b>TradeMind AI - Enhanced Edition v5.1 PRODUCTION</b>\n"
                        "🎯 <b>Interactive Trading + News Intelligence ACTIVE</b>\n\n"
                        
                        "📈 <b>Features:</b>\n"
                        f"• Enhanced Telegram: {'✅' if TelegramService else '❌'}\n"
                        f"• Order Execution: {'✅' if ZerodhaOrderEngine else '❌'}\n"
                        f"• Webhook Handler: {'✅' if TelegramWebhookHandler else '❌'}\n"
                        f"• News Intelligence: {'✅' if self.system_health.get('news_signal_integration') else '❌'}\n\n"
                        
                        "🎯 <b>Interactive Trading:</b>\n"
                        "• Approve/Reject buttons on signals\n"
                        "• Automatic order execution\n"
                        "• Real-time position tracking\n"
                        "• Risk management built-in\n\n"
                        
                        "📰 <b>News Intelligence:</b>\n"
                        "• Real-time news monitoring (30s)\n"
                        "• Breaking news immediate signals\n"
                        "• Sentiment-enhanced ML signals\n"
                        "• Multi-source analysis\n\n"
                        
                        "🔥 <b>PRODUCTION MODE - Only Real ML Signals!</b>\n"
                        "⚡ <b>Ready for intelligent trading!</b>"
                    )
                    
                    success = await self.telegram_service.send_system_startup_notification()
                    if success:
                        await self.telegram_service.send_message(startup_message)
                        logger.info("📱 Enhanced startup notification sent")
                    
                    self.system_health["telegram"] = True
                    self.system_health["interactive_telegram"] = True
                    return True
                else:
                    logger.warning("⚠️ Enhanced Telegram not properly configured")
                    self.system_health["telegram"] = False
                    self.system_health["interactive_telegram"] = False
                    return False
            else:
                logger.warning("⚠️ Enhanced Telegram not configured")
                self.system_health["telegram"] = False
                self.system_health["interactive_telegram"] = False
                return False
        except Exception as e:
            logger.error(f"❌ Enhanced Telegram initialization failed: {e}")
            self.system_health["telegram"] = False
            self.system_health["interactive_telegram"] = False
            return False
    
    async def initialize_order_engine(self) -> bool:
        """Initialize Zerodha Order Engine"""
        try:
            logger.info(f"🔧 Attempting to initialize Zerodha Order Engine with: ")
            logger.info(f"  is_zerodha_configured: {settings.is_zerodha_configured}")
            logger.info(f"  zerodha_api_key: {settings.zerodha_api_key[:6] + '...' if settings.zerodha_api_key else 'NOT SET'}")
            logger.info(f"  zerodha_access_token: {settings.zerodha_access_token[:6] + '...' if settings.zerodha_access_token else 'NOT SET'}")
            logger.info(f"  zerodha_secret: {settings.zerodha_secret[:6] + '...' if getattr(settings, 'zerodha_secret', None) else 'NOT SET'}")
            logger.info(f"  is_production: {settings.is_production}")

            if (settings.is_zerodha_configured and 
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
                
                logger.info(f"🔧 Zerodha connection status: {status}")
                
                if status and status.get("connected"):
                    logger.info(f"✅ Zerodha Order Engine connected: {status.get('user_name', 'Unknown')}")
                    self.system_health["order_execution"] = True
                    self.system_health["zerodha_connection"] = True
                    return True
                else:
                    logger.warning("⚠️ Zerodha Order Engine connection failed")
                    self.system_health["order_execution"] = False
                    self.system_health["zerodha_connection"] = False
                    return False
            else:
                logger.warning("⚠️ Zerodha Order Engine not configured (missing keys or not marked as configured)")
                self.system_health["order_execution"] = False
                self.system_health["zerodha_connection"] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Zerodha Order Engine initialization failed: {e}")
            self.system_health["order_execution"] = False
            self.system_health["zerodha_connection"] = False
            return False
    
    async def initialize_webhook_handler(self) -> bool:
        """Initialize Telegram Webhook Handler"""
        try:
            logger.info(f"🔧 Attempting to initialize Telegram Webhook Handler:")
            logger.info(f"  TelegramWebhookHandler: {'AVAILABLE' if TelegramWebhookHandler else 'NOT AVAILABLE'}")
            logger.info(f"  telegram_service: {'SET' if self.telegram_service else 'NOT SET'}")
            logger.info(f"  order_engine: {'SET' if self.order_engine else 'NOT SET'}")
            if (TelegramWebhookHandler and 
                self.telegram_service and 
                self.order_engine):
                
                self.webhook_handler = TelegramWebhookHandler(
                    telegram_service=self.telegram_service,
                    order_engine=self.order_engine,
                    webhook_secret=getattr(settings, 'telegram_webhook_secret', None)
                )
                
                logger.info("✅ Telegram Webhook Handler initialized")
                self.system_health["webhook_handler"] = True
                return True
            else:
                logger.warning("⚠️ Webhook Handler not available or dependencies missing")
                self.system_health["webhook_handler"] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Webhook Handler initialization failed: {e}")
            self.system_health["webhook_handler"] = False
            return False
    
    async def setup_interactive_trading(self) -> bool:
        """Setup interactive trading integration"""
        try:
            if (self.telegram_service and 
                self.order_engine):
                
                # Set up approval/rejection callbacks with wrapper functions
                self.telegram_service.set_approval_handlers(
                    approval_callback=self._handle_signal_approval_wrapper,
                    rejection_callback=self._handle_signal_rejection_wrapper
                )
                
                self.interactive_trading_active = True
                logger.info("✅ Interactive trading integration setup complete")
                
                # Send confirmation message
                if self.telegram_service.is_configured():
                    integration_status = "✅ ACTIVE" if self.system_health.get("news_signal_integration") else "❌ DISABLED"
                    await self.telegram_service.send_message(
                        "🎯 <b>Interactive Trading + News Intelligence READY (PRODUCTION)</b>\n\n"
                        "✅ Signal approval handlers configured\n"
                        "✅ Order execution engine connected\n"
                        "✅ Risk management active\n"
                        f"✅ News-Signal Integration: {integration_status}\n"
                        "🔥 Demo signals DISABLED - Only real ML signals\n\n"
                        "📲 You will receive high-confidence signals with Approve/Reject buttons during market hours.\n"
                        "📰 Breaking news will trigger immediate signals."
                    )
                return True
            else:
                logger.warning("⚠️ Interactive trading setup incomplete - missing dependencies")
                self.interactive_trading_active = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Interactive trading setup failed: {e}")
            self.interactive_trading_active = False
            return False
    
    def _handle_signal_approval_wrapper(self, signal_id: str, signal_data: Dict) -> None:
        """Wrapper for signal approval callback to match expected signature"""
        asyncio.create_task(self._handle_signal_approval(signal_id, signal_data))
    
    def _handle_signal_rejection_wrapper(self, signal_id: str, signal_data: Dict) -> None:
        """Wrapper for signal rejection callback to match expected signature"""
        asyncio.create_task(self._handle_signal_rejection(signal_id, signal_data))
    
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
            
            # Send notification about order execution
            if self.telegram_service and self.telegram_service.is_configured():
                if order_result.success:
                    execution_message = (
                        f"✅ <b>Order Executed Successfully</b>\n\n"
                        f"📊 <b>Signal ID:</b> {signal_id}\n"
                        f"🎯 <b>Symbol:</b> {symbol}\n"
                        f"📈 <b>Action:</b> {action}\n"
                        f"💰 <b>Quantity:</b> {quantity}\n"
                        f"💵 <b>Entry Price:</b> ₹{entry_price:.2f}\n"
                        f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}\n\n"
                        f"🎉 <i>Position opened successfully!</i>"
                    )
                else:
                    execution_message = (
                        f"❌ <b>Order Execution Failed</b>\n\n"
                        f"📊 <b>Signal ID:</b> {signal_id}\n"
                        f"🎯 <b>Symbol:</b> {symbol}\n"
                        f"📈 <b>Action:</b> {action}\n"
                        f"❌ <b>Error:</b> {order_result.error or 'Unknown error'}\n"
                        f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}\n\n"
                        f"⚠️ <i>Please check your account and try again.</i>"
                    )
                
                await self.telegram_service.send_message(execution_message)
            
            # Convert OrderResult to dict for return
            return {
                "success": order_result.success,
                "order_id": order_result.order_id,
                "status": order_result.status,
                "message": order_result.message,
                "error": order_result.error
            }
            
        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")
            error_result = {"success": False, "error": str(e)}
            
            # Send error notification
            if self.telegram_service and self.telegram_service.is_configured():
                error_message = (
                    f"❌ <b>Order Execution Error</b>\n\n"
                    f"📊 <b>Signal ID:</b> {signal_id}\n"
                    f"❌ <b>Error:</b> {str(e)}\n"
                    f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}\n\n"
                    f"⚠️ <i>Please contact support if this persists.</i>"
                )
                await self.telegram_service.send_message(error_message)
            
            return error_result
    
    async def _handle_signal_rejection(self, signal_id: str, signal_data: Dict):
        """Handle signal rejection"""
        try:
            # Log rejection
            logger.info(f"❌ Signal {signal_id} rejected by user")
            
            # Update analytics - use track_signal_approval with approved=False
            await self.analytics_service.track_signal_approval(approved=False)
            
            # Send rejection notification
            if self.telegram_service and self.telegram_service.is_configured():
                rejection_message = (
                    f"❌ <b>Signal Rejected</b>\n\n"
                    f"📊 <b>Signal ID:</b> {signal_id}\n"
                    f"🎯 <b>Symbol:</b> {signal_data.get('symbol', 'UNKNOWN')}\n"
                    f"📈 <b>Action:</b> {signal_data.get('action', 'UNKNOWN')}\n"
                    f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}\n\n"
                    f"💡 <i>Signal rejected. Waiting for next opportunity...</i>"
                )
                await self.telegram_service.send_message(rejection_message)
            
        except Exception as e:
            logger.error(f"❌ Signal rejection handling failed: {e}")
    
    async def send_regime_change_notification(self, old_regime: str, new_regime: str, confidence: float):
        """Send regime change notification"""
        try:
            if self.telegram_service and self.telegram_service.is_configured():
                regime_message = (
                    f"🔄 <b>Market Regime Change Detected</b>\n\n"
                    f"📊 <b>New Regime:</b> {new_regime}\n"
                    f"🎯 <b>Confidence:</b> {confidence:.1%}\n"
                    f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}\n\n"
                    f"💡 <i>Adapting trading strategies accordingly...</i>"
                )
                await self.telegram_service.send_message(regime_message)
        except Exception as e:
            logger.error(f"❌ Regime change notification failed: {e}")
    
    async def send_news_intelligence_status(self, news_intelligence_status: bool, news_monitoring_status: bool):
        """Send news intelligence status update"""
        try:
            if self.telegram_service and self.telegram_service.is_configured():
                # Update system health
                self.system_health["news_intelligence"] = news_intelligence_status
                self.system_health["news_monitoring"] = news_monitoring_status
                self.system_health["breaking_news_alerts"] = news_intelligence_status
                
                status_message = (
                    f"📰 <b>News Intelligence Status Update</b>\n\n"
                    f"🧠 <b>News Intelligence:</b> {'✅ ACTIVE' if news_intelligence_status else '❌ DISABLED'}\n"
                    f"📡 <b>News Monitoring:</b> {'✅ ACTIVE' if news_monitoring_status else '❌ DISABLED'}\n"
                    f"🚨 <b>Breaking News Alerts:</b> {'✅ ACTIVE' if news_intelligence_status else '❌ DISABLED'}\n"
                    f"⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}\n\n"
                    f"💡 <i>News-enhanced signals are {'enabled' if news_intelligence_status else 'disabled'}.</i>"
                )
                await self.telegram_service.send_message(status_message)
        except Exception as e:
            logger.error(f"❌ News intelligence status notification failed: {e}")
    
    def get_system_health(self) -> Dict[str, bool]:
        """Get current system health status"""
        return self.system_health.copy()
    
    def is_interactive_trading_active(self) -> bool:
        """Check if interactive trading is active"""
        return self.interactive_trading_active
    
    def get_telegram_service(self) -> Optional[TelegramService]:
        """Get telegram service instance"""
        return self.telegram_service
    
    def get_order_engine(self) -> Optional[ZerodhaOrderEngine]:
        """Get order engine instance"""
        return self.order_engine
    
    def get_webhook_handler(self) -> Optional[TelegramWebhookHandler]:
        """Get webhook handler instance"""
        return self.webhook_handler
    
    def set_telegram_service(self, telegram_service: TelegramService):
        """Set telegram service instance"""
        self.telegram_service = telegram_service
        if telegram_service and telegram_service.is_configured():
            self.system_health["telegram"] = True
            self.system_health["interactive_telegram"] = True
    
    def set_order_engine(self, order_engine: ZerodhaOrderEngine):
        """Set order engine instance"""
        self.order_engine = order_engine
        if order_engine:
            self.system_health["order_execution"] = True
            self.system_health["zerodha_connection"] = True
    
    def set_webhook_handler(self, webhook_handler: TelegramWebhookHandler):
        """Set webhook handler instance"""
        self.webhook_handler = webhook_handler
        if webhook_handler:
            self.system_health["webhook_handler"] = True

    @profile_timing("order_execution_pipeline")
    async def place_order(self, order_request):
        valid = await self._validate_order(order_request)
        if not valid:
            raise Exception("Order validation failed")
        result = await self._send_order_to_zerodha(order_request)
        await self._update_order_status(result)
        return result

    @profile_timing("validate_order")
    async def _validate_order(self, order_request):
        # ... validation logic ...
        pass

    @profile_timing("send_order_to_zerodha")
    async def _send_order_to_zerodha(self, order_request):
        # ... Zerodha API call ...
        pass

    @profile_timing("update_order_status")
    async def _update_order_status(self, result):
        # ... DB/status update ...
        pass 