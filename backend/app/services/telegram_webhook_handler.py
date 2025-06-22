# backend/app/services/telegram_webhook_handler.py
"""
TradeMind AI - Complete Telegram Webhook Handler & System Integration
Production-ready webhook handling with proper security and integration
"""

import asyncio
import logging
import json
import hmac
import hashlib
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

from .enhanced_telegram_service import EnhancedTelegramService
from .zerodha_order_engine import ZerodhaOrderEngine, OrderType, ExchangeType
from ..core.config import settings

logger = logging.getLogger(__name__)

class TelegramWebhookHandler:
    """
    Production-ready Telegram webhook handler
    Integrates with TradeMind AI system for order execution
    """
    
    def __init__(self, 
                 telegram_service: EnhancedTelegramService,
                 order_engine: ZerodhaOrderEngine,
                 webhook_secret: Optional[str] = None):
        self.telegram_service = telegram_service
        self.order_engine = order_engine
        self.webhook_secret = webhook_secret
        
        # Service manager not needed - direct integration
        
        logger.info("✅ Telegram webhook handler initialized")
    
    def verify_webhook_signature(self, body: bytes, signature: str) -> bool:
        """Verify webhook signature for security"""
        if not self.webhook_secret:
            return True  # Skip verification if no secret is set
        
        try:
            expected_signature = hmac.new(
                self.webhook_secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(f"sha256={expected_signature}", signature)
        except Exception as e:
            logger.error(f"❌ Signature verification failed: {e}")
            return False
    
    async def handle_webhook(self, request: Request) -> JSONResponse:
        """
        Handle incoming webhook from Telegram
        CRITICAL: Must respond quickly to avoid timeout
        """
        try:
            # Get request body and headers
            body = await request.body()
            signature = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            
            # Verify signature if secret is configured
            if self.webhook_secret and not self.verify_webhook_signature(body, signature):
                logger.warning("❌ Invalid webhook signature")
                raise HTTPException(status_code=403, detail="Invalid signature")
            
            # Parse update
            try:
                update = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON in webhook: {e}")
                raise HTTPException(status_code=400, detail="Invalid JSON")
            
            # Process update asynchronously to avoid blocking
            asyncio.create_task(self._process_update(update))
            
            # Return immediate response to Telegram
            return JSONResponse({"ok": True})
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Webhook handling failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _process_update(self, update: Dict[str, Any]):
        """Process Telegram update asynchronously"""
        try:
            update_id = update.get("update_id")
            logger.debug(f"📥 Processing update {update_id}")
            
            # Handle callback queries (button presses)
            if "callback_query" in update:
                await self._handle_callback_query(update)
            
            # Handle regular messages
            elif "message" in update:
                await self._handle_message(update)
            
            # Handle edited messages
            elif "edited_message" in update:
                await self._handle_edited_message(update)
            
            else:
                logger.debug(f"⏭️ Unsupported update type: {list(update.keys())}")
                
        except Exception as e:
            logger.error(f"❌ Update processing failed: {e}")
    
    async def _handle_callback_query(self, update: Dict[str, Any]):
        """Handle callback queries from inline keyboards"""
        try:
            success = await self.telegram_service.handle_callback_query(update)
            if success:
                logger.info("✅ Callback query handled successfully")
            else:
                logger.warning("⚠️ Callback query handling failed")
                
        except Exception as e:
            logger.error(f"❌ Callback query handling error: {e}")
    
    async def _handle_message(self, update: Dict[str, Any]):
        """Handle regular messages"""
        try:
            message = update.get("message", {})
            chat_id = message.get("chat", {}).get("id")
            text = message.get("text", "")
            
            # Only respond to messages from authorized chat
            if str(chat_id) != self.telegram_service.chat_id:
                logger.warning(f"⚠️ Unauthorized message from chat {chat_id}")
                return
            
            # Handle commands
            if text.startswith("/"):
                await self._handle_command(text, chat_id)
            else:
                # Log the message but don't respond to avoid spam
                logger.debug(f"📝 Received message: {text[:50]}...")
                
        except Exception as e:
            logger.error(f"❌ Message handling error: {e}")
    
    async def _handle_edited_message(self, update: Dict[str, Any]):
        """Handle edited messages"""
        logger.debug("📝 Message edited - ignoring")
    
    async def _handle_command(self, command: str, chat_id: str):
        """Handle bot commands"""
        try:
            command = command.lower().strip()
            
            if command == "/start":
                await self._send_welcome_message()
            
            elif command == "/status":
                await self._send_status_message()
            
            elif command == "/orders":
                await self._send_orders_status()
            
            elif command == "/positions":
                await self._send_positions_status()
            
            elif command == "/help":
                await self._send_help_message()
            
            else:
                await self.telegram_service.send_message(
                    f"❓ Unknown command: {command}\nUse /help for available commands"
                )
                
        except Exception as e:
            logger.error(f"❌ Command handling error: {e}")
    
    async def _send_welcome_message(self):
        """Send welcome message"""
        message = """
🤖 <b>TradeMind AI - Interactive Trading Bot</b>

✅ <b>Connection Status:</b> Active
🔄 <b>Mode:</b> Interactive Order Approval

<b>📋 Available Commands:</b>
/status - System status
/orders - Recent orders
/positions - Current positions
/help - Show this help

🎯 <b>How it works:</b>
1. I'll send you trading signals with Approve/Reject buttons
2. Click ✅ <b>APPROVE</b> to execute the trade automatically
3. Click ❌ <b>REJECT</b> to skip the trade
4. All orders are placed directly in your Zerodha account

⚠️ <b>Important:</b> You have 5 minutes to respond to each signal before it expires.

🚀 <b>Ready to start intelligent trading!</b>
"""
        await self.telegram_service.send_message(message)
    
    async def _send_status_message(self):
        """Send system status"""
        try:
            # Get Zerodha connection status
            zerodha_status = await self.order_engine.get_connection_status()
            order_summary = self.order_engine.get_order_summary()
            
            status_emoji = "✅" if zerodha_status.get("connected") else "❌"
            market_emoji = "🟢" if zerodha_status.get("market_open") else "🔴"
            
            message = f"""
📊 <b>TradeMind AI System Status</b>

🔗 <b>Zerodha Connection:</b> {status_emoji} {'Connected' if zerodha_status.get("connected") else 'Disconnected'}
{market_emoji} <b>Market Status:</b> {'Open' if zerodha_status.get("market_open") else 'Closed'}

👤 <b>Account:</b> {zerodha_status.get("user_name", "Unknown")}
💰 <b>Available Cash:</b> ₹{zerodha_status.get("available_cash", 0):,.2f}

📈 <b>Today's Activity:</b>
• Orders Placed: {order_summary.get("orders_today", 0)}
• Pending Orders: {order_summary.get("pending_orders", 0)}
• Rate Limit: {zerodha_status.get("rate_limit_remaining", 0)} remaining

⏰ <b>Last Updated:</b> {datetime.now().strftime("%H:%M:%S")}
"""
            
            await self.telegram_service.send_message(message)
            
        except Exception as e:
            await self.telegram_service.send_message(f"❌ Failed to get status: {e}")
    
    async def _send_orders_status(self):
        """Send recent orders status"""
        try:
            orders = await self.order_engine.get_order_history(days=1)
            
            if not orders:
                await self.telegram_service.send_message("📭 No orders found for today")
                return
            
            message = "<b>📋 Recent Orders (Today)</b>\n\n"
            
            for order in orders[-10:]:  # Last 10 orders
                symbol = order.get("tradingsymbol", "Unknown")
                action = order.get("transaction_type", "")
                quantity = order.get("quantity", 0)
                status = order.get("status", "Unknown")
                price = order.get("price", 0) or order.get("average_price", 0)
                
                status_emoji = {
                    "COMPLETE": "✅",
                    "OPEN": "⏳",
                    "CANCELLED": "❌",
                    "REJECTED": "🚫"
                }.get(status, "❓")
                
                message += f"{status_emoji} <b>{symbol}</b> {action} {quantity} @ ₹{price:.2f}\n"
            
            await self.telegram_service.send_message(message)
            
        except Exception as e:
            await self.telegram_service.send_message(f"❌ Failed to get orders: {e}")
    
    async def _send_positions_status(self):
        """Send current positions"""
        try:
            positions = await self.order_engine.get_positions()
            
            if not positions:
                await self.telegram_service.send_message("📭 No open positions")
                return
            
            message = "<b>📊 Current Positions</b>\n\n"
            
            for position in positions:
                if position.get("quantity", 0) != 0:  # Only show non-zero positions
                    symbol = position.get("tradingsymbol", "Unknown")
                    quantity = position.get("quantity", 0)
                    pnl = position.get("pnl", 0)
                    pnl_emoji = "📈" if pnl >= 0 else "📉"
                    
                    message += f"{pnl_emoji} <b>{symbol}</b> Qty: {quantity} P&L: ₹{pnl:,.2f}\n"
            
            await self.telegram_service.send_message(message)
            
        except Exception as e:
            await self.telegram_service.send_message(f"❌ Failed to get positions: {e}")
    
    async def _send_help_message(self):
        """Send help message"""
        message = """
🆘 <b>TradeMind AI Help</b>

<b>📋 Available Commands:</b>
/start - Welcome message and setup
/status - Check system and account status
/orders - View recent order history
/positions - View current positions
/help - Show this help message

<b>🎯 Trading Workflow:</b>
1. <b>Signal Generation:</b> AI analyzes market and finds opportunities
2. <b>Approval Request:</b> You receive signal with Approve/Reject buttons
3. <b>Order Execution:</b> Approved signals are executed automatically
4. <b>Confirmation:</b> You get instant confirmation of order placement

<b>⚠️ Important Notes:</b>
• Signals expire after 5 minutes
• Only trade during market hours (9:15 AM - 3:30 PM)
• Ensure sufficient funds in your Zerodha account
• All trades are executed at market prices

<b>🔧 Support:</b>
For technical issues, contact your system administrator.

<b>⚖️ Disclaimer:</b>
Trading involves risk. Trade responsibly and within your risk tolerance.
"""
        await self.telegram_service.send_message(message)


# FastAPI integration
def create_telegram_webhook_app(webhook_handler: TelegramWebhookHandler) -> FastAPI:
    """Create FastAPI app with telegram webhook endpoint"""
    
    app = FastAPI(
        title="TradeMind AI Telegram Webhook",
        description="Webhook handler for Telegram bot integration",
        version="1.0.0"
    )
    
    @app.post("/webhook/telegram")
    async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
        """Telegram webhook endpoint"""
        return await webhook_handler.handle_webhook(request)
    
    @app.get("/webhook/status")
    async def webhook_status():
        """Webhook status endpoint"""
        return {
            "status": "active",
            "webhook_configured": True,
            "timestamp": datetime.now().isoformat()
        }
    
    return app


# Integration with existing TradeMind system
class TradeMindTelegramIntegration:
    """
    Complete integration class for TradeMind AI system
    Connects Telegram service with existing signal generation
    """
    
    def __init__(self):
        # Initialize services
        self.telegram_service = None
        self.order_engine = None
        self.webhook_handler = None
        self.webhook_app = None
        
        # Configuration
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize all services"""
        try:
            # Check if Telegram is configured
            if not settings.is_telegram_configured:
                logger.warning("⚠️ Telegram not configured - interactive trading disabled")
                return False
            
            # Check if Zerodha is configured
            if not settings.is_zerodha_configured:
                logger.warning("⚠️ Zerodha not configured - order execution disabled")
                return False
            
            # Initialize Telegram service
            self.telegram_service = EnhancedTelegramService(
                bot_token=settings.telegram_bot_token,
                chat_id=settings.telegram_chat_id,
                redis_url=settings.redis_url
            )
            
            # Initialize order engine
            self.order_engine = ZerodhaOrderEngine(
                api_key=settings.zerodha_api_key,
                access_token=settings.zerodha_access_token,
                enable_sandbox=not settings.is_production
            )
            
            # Wait for connections to establish
            await asyncio.sleep(3)
            
            # Initialize webhook handler
            self.webhook_handler = TelegramWebhookHandler(
                telegram_service=self.telegram_service,
                order_engine=self.order_engine,
                webhook_secret=getattr(settings, 'telegram_webhook_secret', None)
            )
            
            # Create webhook app
            self.webhook_app = create_telegram_webhook_app(self.webhook_handler)
            
            self.is_initialized = True
            logger.info("✅ TradeMind Telegram integration initialized successfully")
            
            # Send startup notification
            await self.telegram_service.send_message(
                "🚀 <b>TradeMind AI Started</b>\n\n"
                "✅ Interactive trading is now active!\n"
                "🎯 Ready to receive and execute trading signals.\n\n"
                "Use /status to check system status."
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ TradeMind Telegram integration failed: {e}")
            self.is_initialized = False
            return False
    
    async def send_signal_for_approval(self, signal: Dict[str, Any], quantity: int = 1) -> bool:
        """
        Send trading signal for user approval
        This method should be called from your existing signal generation system
        """
        if not self.is_initialized:
            logger.warning("⚠️ Telegram integration not initialized")
            return False
        
        try:
            return await self.telegram_service.send_signal_with_approval(signal, quantity)
        except Exception as e:
            logger.error(f"❌ Failed to send signal for approval: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        if not self.is_initialized:
            return {"initialized": False}
        
        try:
            telegram_status = self.telegram_service.is_configured()
            zerodha_status = await self.order_engine.get_connection_status()
            order_summary = self.order_engine.get_order_summary()
            
            return {
                "initialized": True,
                "telegram": {
                    "configured": telegram_status,
                    "connected": True  # If service is initialized, it's connected
                },
                "zerodha": zerodha_status,
                "orders": order_summary,
                "market_open": self.order_engine.is_market_open(),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get system status: {e}")
            return {"initialized": True, "error": str(e)}
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            if self.telegram_service:
                await self.telegram_service.send_message(
                    "🛑 <b>TradeMind AI Stopping</b>\n\n"
                    "💤 Interactive trading temporarily disabled.\n"
                    "System will resume on next startup."
                )
                await self.telegram_service.close()
            
            logger.info("✅ TradeMind Telegram integration shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


# Global integration instance
telegram_integration = TradeMindTelegramIntegration()


# Usage example for integration with existing main.py
async def integrate_with_existing_main():
    """
    Example integration with your existing main.py
    Add this to your existing TradeMind system
    """
    
    # Initialize the integration
    success = await telegram_integration.initialize()
    
    if success:
        # Example: Send a signal for approval (call this from your signal generation code)
        test_signal = {
            'symbol': 'RELIANCE',
            'action': 'BUY',
            'entry_price': 2450.50,
            'target_price': 2580.00,
            'stop_loss': 2380.00,
            'confidence': 87.5
        }
        
        # This replaces your existing telegram notification
        await telegram_integration.send_signal_for_approval(test_signal, quantity=1)
        
        # Get system status
        status = await telegram_integration.get_system_status()
        print(f"System status: {status}")


if __name__ == "__main__":
    # For testing the webhook handler standalone
    asyncio.run(integrate_with_existing_main())
