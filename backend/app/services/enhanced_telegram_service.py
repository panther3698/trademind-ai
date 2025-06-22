# backend/app/services/enhanced_telegram_service.py
"""
TradeMind AI - Enhanced Interactive Telegram Service
Replaces the old telegram_service.py with interactive approval capabilities
"""

import asyncio
import aiohttp
import logging
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
from dataclasses import dataclass

# Redis import with fallback
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("⚠️ Redis not available. Install with: pip install redis aioredis")

logger = logging.getLogger(__name__)

class SignalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved" 
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class PendingSignal:
    signal_id: str
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    quantity: int
    status: SignalStatus
    created_at: datetime
    expires_at: datetime
    message_id: Optional[int] = None

class EnhancedTelegramService:
    """
    Enhanced Telegram service - backward compatible with old TelegramService
    Adds interactive approval capabilities while maintaining existing functionality
    """
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None, redis_url: str = "redis://localhost:6379"):
        # Get credentials from environment if not provided
        import os
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        self.enabled = bool(self.bot_token and self.chat_id)
        
        # Redis for storing pending signals (with fallback)
        self.redis_client = None
        self.redis_available = False
        self.pending_signals = {}  # Fallback to memory storage
        
        # Initialize Redis connection if available
        if REDIS_AVAILABLE:
            asyncio.create_task(self._init_redis(redis_url))
        
        # Callback handlers for interactive features
        self.approval_callback: Optional[Callable] = None
        self.rejection_callback: Optional[Callable] = None
        
        # Rate limiting
        self.last_message_time = 0
        self.min_message_interval = 1.0  # 1 second between messages
        
        # Signal expiry time (5 minutes)
        self.signal_expiry_seconds = 300
        
        # HTTP session for requests
        self.session = None
        
        if self.enabled:
            logger.info("✅ Enhanced Telegram service initialized")
        else:
            logger.warning("⚠️ Telegram not configured - service disabled")
    
    async def _init_redis(self, redis_url: str):
        """Initialize Redis connection asynchronously"""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis connection established for Telegram service")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available, using memory storage: {e}")
            self.redis_available = False
    
    async def _get_session(self):
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=aiohttp.TCPConnector(limit=10)
            )
        return self.session
    
    async def close(self):
        """Clean up resources"""
        if self.session and not self.session.closed:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
    
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured (backward compatibility)"""
        return self.enabled
    
    # ================================================================
    # BACKWARD COMPATIBILITY METHODS (same interface as old service)
    # ================================================================
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a simple text message (backward compatible)"""
        if not self.enabled:
            return False
            
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_message_time
            if time_since_last < self.min_message_interval:
                await asyncio.sleep(self.min_message_interval - time_since_last)
            
            session = await self._get_session()
            
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            async with session.post(f"{self.base_url}/sendMessage", data=payload) as response:
                self.last_message_time = time.time()
                return response.status == 200
                
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return False
    
    async def send_signal_notification(self, signal: Dict[str, Any]) -> bool:
        """Send signal notification (backward compatible with old service)"""
        if not self.enabled:
            return False
            
        try:
            # Extract signal data with defaults
            symbol = signal.get("symbol", "UNKNOWN")
            action = signal.get("action", "UNKNOWN").upper()
            entry_price = signal.get("entry_price", 0.0)
            target_price = signal.get("target_price", 0.0)
            stop_loss = signal.get("stop_loss", 0.0)
            confidence = signal.get("confidence", 0.0)
            timestamp = signal.get("timestamp", datetime.now())
            
            # Determine emoji based on action
            action_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
            
            # Calculate potential profit/loss percentages
            profit_pct = ((target_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            risk_pct = ((entry_price - stop_loss) / entry_price * 100) if entry_price > 0 else 0
            
            # Format the message with HTML
            message = f"""
🚀 <b>TradeMind AI Signal</b> {action_emoji}

📊 <b>Stock:</b> {symbol}
🎯 <b>Action:</b> {action}
💰 <b>Entry:</b> ₹{entry_price:.2f}
🎯 <b>Target:</b> ₹{target_price:.2f} ({profit_pct:+.1f}%)
🛡️ <b>Stop Loss:</b> ₹{stop_loss:.2f} ({risk_pct:-.1f}%)
🎲 <b>Confidence:</b> {confidence:.1f}%

⏰ <b>Time:</b> {timestamp.strftime("%d-%m-%Y %H:%M:%S") if hasattr(timestamp, 'strftime') else str(timestamp)}

💡 <i>Risk-Reward Ratio:</i> {(profit_pct/risk_pct if risk_pct > 0 else 0):.2f}:1

⚠️ <i>Trade at your own risk. This is algorithmic analysis, not financial advice.</i>
""".strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Signal notification formatting error: {e}")
            return False
    
    async def send_system_startup_notification(self) -> bool:
        """Send system startup notification (backward compatible)"""
        if not self.enabled:
            return False
            
        try:
            current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
            
            message = f"""
🤖 <b>TradeMind AI Enhanced - System Started</b>

✅ <b>Status:</b> Fully Operational
🕐 <b>Start Time:</b> {current_time}
📊 <b>Market:</b> Indian Stock Market
🎯 <b>Signal Mode:</b> Enhanced with Interactive Approval
🔄 <b>Auto-Trading:</b> Ready (with manual approval)

🚨 <b>NEW Features Active:</b>
• Interactive signal approval buttons
• Automatic order execution (Zerodha)
• Real-time position tracking
• Risk management with position sizing

📱 You will receive trading signals with Approve/Reject buttons during market hours (9:15 AM - 3:30 PM IST)

💼 <i>Ready to analyze 100+ Nifty stocks with interactive trading!</i>
""".strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Startup notification error: {e}")
            return False
    
    async def send_system_shutdown_notification(self) -> bool:
        """Send system shutdown notification (backward compatible)"""
        if not self.enabled:
            return False
            
        try:
            current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
            
            message = f"""
🛑 <b>TradeMind AI Enhanced - System Shutdown</b>

📴 <b>Status:</b> System Stopped
🕐 <b>Shutdown Time:</b> {current_time}

💤 <i>Interactive trading signals paused. System will resume on next startup.</i>
""".strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Shutdown notification error: {e}")
            return False
    
    # ================================================================
    # NEW INTERACTIVE FEATURES
    # ================================================================
    
    def set_approval_handlers(self, 
                            approval_callback: Callable[[str, Dict], None],
                            rejection_callback: Callable[[str, Dict], None]):
        """Set callbacks for signal approval/rejection"""
        self.approval_callback = approval_callback
        self.rejection_callback = rejection_callback
        logger.info("✅ Interactive approval handlers configured")
    
    async def send_signal_with_approval(self, signal: Dict[str, Any], quantity: int = 1) -> bool:
        """Send trading signal with interactive approve/reject buttons"""
        if not self.enabled:
            logger.warning("Telegram not configured")
            return False
            
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_message_time
            if time_since_last < self.min_message_interval:
                await asyncio.sleep(self.min_message_interval - time_since_last)
            
            # Generate unique signal ID
            signal_id = str(uuid.uuid4())[:8]
            
            # Create pending signal object
            pending_signal = PendingSignal(
                signal_id=signal_id,
                symbol=signal.get('symbol', 'UNKNOWN'),
                action=signal.get('action', 'BUY').upper(),
                entry_price=signal.get('entry_price', 0.0),
                target_price=signal.get('target_price', 0.0),
                stop_loss=signal.get('stop_loss', 0.0),
                confidence=signal.get('confidence', 0.0),
                quantity=quantity,
                status=SignalStatus.PENDING,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=self.signal_expiry_seconds)
            )
            
            # Store signal data
            await self._store_pending_signal(pending_signal)
            
            # Create message with buttons
            message_text = self._format_signal_message(pending_signal)
            inline_keyboard = self._create_approval_keyboard(signal_id)
            
            # Send message with inline keyboard
            message_id = await self._send_message_with_keyboard(message_text, inline_keyboard)
            
            if message_id:
                # Update stored signal with message ID
                pending_signal.message_id = message_id
                await self._store_pending_signal(pending_signal)
                
                logger.info(f"📤 Interactive signal sent: {pending_signal.symbol} (ID: {signal_id})")
                
                # Schedule expiry cleanup
                asyncio.create_task(self._schedule_signal_expiry(signal_id))
                
                self.last_message_time = time.time()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to send interactive signal: {e}")
            return False
    
    async def handle_callback_query(self, update: Dict[str, Any]) -> bool:
        """
        Handle callback queries from inline keyboard buttons
        CRITICAL: This MUST call answerCallbackQuery within 15 seconds
        """
        callback_query = update.get('callback_query', {})
        callback_query_id = callback_query.get('id')
        
        if not callback_query_id:
            return False
        
        try:
            callback_data = callback_query.get('data', '')
            message_id = callback_query.get('message', {}).get('message_id')
            
            # CRITICAL: Answer callback query immediately to prevent loading spinner
            await self._answer_callback_query(callback_query_id, "Processing...")
            
            if not callback_data.startswith(('approve_', 'reject_')):
                return False
            
            action, signal_id = callback_data.split('_', 1)
            
            # Get signal data
            pending_signal = await self._get_pending_signal(signal_id)
            if not pending_signal:
                await self._edit_message(
                    message_id,
                    "❌ Signal expired or not found",
                    None
                )
                return False
            
            # Process approval/rejection
            if action == 'approve':
                success = await self._handle_approval(pending_signal, message_id)
                result_message = "✅ Signal approved and executed!" if success else "❌ Order execution failed"
            elif action == 'reject':
                await self._handle_rejection(pending_signal, message_id)
                result_message = "❌ Signal rejected"
            else:
                result_message = "❓ Unknown action"
            
            # Send final callback answer with result
            await self._answer_callback_query(
                callback_query_id,
                result_message,
                show_alert=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Callback query handling failed: {e}")
            # Still answer the callback query to prevent loading spinner
            await self._answer_callback_query(
                callback_query_id,
                "❌ Error processing request",
                show_alert=True
            )
            return False
    
    # ================================================================
    # PRIVATE METHODS FOR INTERACTIVE FEATURES
    # ================================================================
    
    async def _store_pending_signal(self, pending_signal: PendingSignal):
        """Store pending signal with expiry"""
        try:
            signal_data = {
                'signal_id': pending_signal.signal_id,
                'symbol': pending_signal.symbol,
                'action': pending_signal.action,
                'entry_price': pending_signal.entry_price,
                'target_price': pending_signal.target_price,
                'stop_loss': pending_signal.stop_loss,
                'confidence': pending_signal.confidence,
                'quantity': pending_signal.quantity,
                'status': pending_signal.status.value,
                'created_at': pending_signal.created_at.isoformat(),
                'expires_at': pending_signal.expires_at.isoformat(),
                'message_id': pending_signal.message_id
            }
            
            if self.redis_available and self.redis_client:
                await self.redis_client.setex(
                    f"signal:{pending_signal.signal_id}",
                    self.signal_expiry_seconds,
                    json.dumps(signal_data)
                )
            else:
                self.pending_signals[pending_signal.signal_id] = signal_data
                
        except Exception as e:
            logger.error(f"❌ Failed to store signal {pending_signal.signal_id}: {e}")
    
    async def _get_pending_signal(self, signal_id: str) -> Optional[PendingSignal]:
        """Retrieve pending signal data"""
        try:
            signal_data = None
            
            if self.redis_available and self.redis_client:
                data = await self.redis_client.get(f"signal:{signal_id}")
                if data:
                    signal_data = json.loads(data)
            else:
                signal_data = self.pending_signals.get(signal_id)
            
            if signal_data:
                return PendingSignal(
                    signal_id=signal_data['signal_id'],
                    symbol=signal_data['symbol'],
                    action=signal_data['action'],
                    entry_price=signal_data['entry_price'],
                    target_price=signal_data['target_price'],
                    stop_loss=signal_data['stop_loss'],
                    confidence=signal_data['confidence'],
                    quantity=signal_data['quantity'],
                    status=SignalStatus(signal_data['status']),
                    created_at=datetime.fromisoformat(signal_data['created_at']),
                    expires_at=datetime.fromisoformat(signal_data['expires_at']),
                    message_id=signal_data.get('message_id')
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get signal {signal_id}: {e}")
            return None
    
    async def _remove_pending_signal(self, signal_id: str):
        """Remove signal from pending storage"""
        try:
            if self.redis_available and self.redis_client:
                await self.redis_client.delete(f"signal:{signal_id}")
            else:
                self.pending_signals.pop(signal_id, None)
        except Exception as e:
            logger.error(f"❌ Failed to remove signal {signal_id}: {e}")
    
    def _format_signal_message(self, pending_signal: PendingSignal) -> str:
        """Format signal message with approval request"""
        action_emoji = "🟢" if pending_signal.action == "BUY" else "🔴"
        
        profit_pct = ((pending_signal.target_price - pending_signal.entry_price) / 
                     pending_signal.entry_price * 100) if pending_signal.entry_price > 0 else 0
        risk_pct = ((pending_signal.entry_price - pending_signal.stop_loss) / 
                   pending_signal.entry_price * 100) if pending_signal.entry_price > 0 else 0
        
        risk_reward = profit_pct / risk_pct if risk_pct > 0 else 0
        
        message = f"""
🤖 <b>TradeMind AI - TRADE APPROVAL REQUIRED</b> {action_emoji}

📊 <b>Stock:</b> {pending_signal.symbol}
🎯 <b>Action:</b> {pending_signal.action}
💰 <b>Entry:</b> ₹{pending_signal.entry_price:.2f}
🎯 <b>Target:</b> ₹{pending_signal.target_price:.2f} ({profit_pct:+.1f}%)
🛡️ <b>Stop Loss:</b> ₹{pending_signal.stop_loss:.2f} ({risk_pct:-.1f}%)
📊 <b>Quantity:</b> {pending_signal.quantity} shares
🎲 <b>Confidence:</b> {pending_signal.confidence:.1f}%

💡 <b>Risk-Reward:</b> {risk_reward:.2f}:1
🆔 <b>Signal ID:</b> {pending_signal.signal_id}

⏰ <b>Expires in:</b> 5 minutes
⚠️ <i>Please approve or reject this trade within 5 minutes</i>

🔄 <b>Status:</b> Awaiting your approval...
""".strip()
        
        return message
    
    def _create_approval_keyboard(self, signal_id: str) -> Dict[str, Any]:
        """Create inline keyboard with approve/reject buttons"""
        return {
            "inline_keyboard": [
                [
                    {
                        "text": "✅ APPROVE & EXECUTE",
                        "callback_data": f"approve_{signal_id}"
                    },
                    {
                        "text": "❌ REJECT",
                        "callback_data": f"reject_{signal_id}"
                    }
                ]
            ]
        }
    
    async def _send_message_with_keyboard(self, text: str, keyboard: Dict[str, Any]) -> Optional[int]:
        """Send message with inline keyboard and return message ID"""
        try:
            session = await self._get_session()
            
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'HTML',
                'reply_markup': json.dumps(keyboard)
            }
            
            async with session.post(f"{self.base_url}/sendMessage", data=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('ok'):
                        return result['result']['message_id']
                else:
                    logger.error(f"❌ Telegram API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to send message with keyboard: {e}")
        
        return None
    
    async def _answer_callback_query(self, callback_query_id: str, text: str = "", show_alert: bool = False):
        """
        CRITICAL: Answer callback query to remove loading spinner
        Must be called within 15 seconds of receiving the callback
        """
        try:
            session = await self._get_session()
            
            payload = {
                'callback_query_id': callback_query_id,
                'text': text,
                'show_alert': show_alert
            }
            
            async with session.post(f"{self.base_url}/answerCallbackQuery", data=payload) as response:
                if response.status != 200:
                    logger.error(f"❌ Failed to answer callback query: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error answering callback query: {e}")
    
    async def _edit_message(self, message_id: int, text: str, keyboard: Optional[Dict[str, Any]] = None):
        """Edit existing message"""
        try:
            session = await self._get_session()
            
            payload = {
                'chat_id': self.chat_id,
                'message_id': message_id,
                'text': text,
                'parse_mode': 'HTML'
            }
            
            if keyboard:
                payload['reply_markup'] = json.dumps(keyboard)
            
            async with session.post(f"{self.base_url}/editMessageText", data=payload) as response:
                if response.status != 200:
                    logger.error(f"❌ Failed to edit message: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error editing message: {e}")
    
    async def _handle_approval(self, pending_signal: PendingSignal, message_id: int) -> bool:
        """Handle signal approval and execute order"""
        try:
            # Update signal status
            pending_signal.status = SignalStatus.APPROVED
            await self._store_pending_signal(pending_signal)
            
            # Execute order via callback (if available)
            order_success = False
            order_id = None
            
            if self.approval_callback:
                try:
                    # Call the order execution callback
                    result = await self.approval_callback(pending_signal.signal_id, pending_signal.__dict__)
                    order_success = result.get('success', False)
                    order_id = result.get('order_id')
                except Exception as e:
                    logger.error(f"❌ Order execution failed: {e}")
                    order_success = False
            
            # Update message with result
            if order_success:
                pending_signal.status = SignalStatus.EXECUTED
                success_text = f"""
✅ <b>TRADE EXECUTED SUCCESSFULLY</b>

📊 <b>Stock:</b> {pending_signal.symbol}
🎯 <b>Action:</b> {pending_signal.action}
💰 <b>Entry:</b> ₹{pending_signal.entry_price:.2f}
📊 <b>Quantity:</b> {pending_signal.quantity} shares
🔢 <b>Order ID:</b> {order_id or 'Generated'}

✅ <b>Status:</b> Order placed successfully
⏰ <b>Executed at:</b> {datetime.now().strftime('%H:%M:%S')}

🎉 <i>Your order has been submitted to the exchange!</i>
"""
            else:
                pending_signal.status = SignalStatus.FAILED
                success_text = f"""
❌ <b>ORDER EXECUTION FAILED</b>

📊 <b>Stock:</b> {pending_signal.symbol}
🎯 <b>Action:</b> {pending_signal.action}
💰 <b>Entry:</b> ₹{pending_signal.entry_price:.2f}

❌ <b>Status:</b> Order placement failed
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

🔄 <i>Please check your Zerodha account or try manual execution</i>
"""
            
            await self._edit_message(message_id, success_text)
            await self._store_pending_signal(pending_signal)
            
            # Clean up after execution
            await self._remove_pending_signal(pending_signal.signal_id)
            
            return order_success
            
        except Exception as e:
            logger.error(f"❌ Approval handling failed: {e}")
            return False
    
    async def _handle_rejection(self, pending_signal: PendingSignal, message_id: int):
        """Handle signal rejection"""
        try:
            # Update signal status
            pending_signal.status = SignalStatus.REJECTED
            await self._store_pending_signal(pending_signal)
            
            # Call rejection callback if available
            if self.rejection_callback:
                try:
                    await self.rejection_callback(pending_signal.signal_id, pending_signal.__dict__)
                except Exception as e:
                    logger.error(f"❌ Rejection callback failed: {e}")
            
            # Update message
            reject_text = f"""
❌ <b>TRADE REJECTED</b>

📊 <b>Stock:</b> {pending_signal.symbol}
🎯 <b>Action:</b> {pending_signal.action}
💰 <b>Entry:</b> ₹{pending_signal.entry_price:.2f}

❌ <b>Status:</b> Manually rejected by user
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

💡 <i>No order was placed. Signal discarded.</i>
"""
            
            await self._edit_message(message_id, reject_text)
            
            # Clean up
            await self._remove_pending_signal(pending_signal.signal_id)
            
        except Exception as e:
            logger.error(f"❌ Rejection handling failed: {e}")
    
    async def _schedule_signal_expiry(self, signal_id: str):
        """Schedule automatic signal expiry"""
        try:
            await asyncio.sleep(self.signal_expiry_seconds)
            
            # Check if signal still exists and is pending
            pending_signal = await self._get_pending_signal(signal_id)
            if pending_signal and pending_signal.status == SignalStatus.PENDING:
                # Expire the signal
                pending_signal.status = SignalStatus.EXPIRED
                
                if pending_signal.message_id:
                    expire_text = f"""
⏰ <b>SIGNAL EXPIRED</b>

📊 <b>Stock:</b> {pending_signal.symbol}
🎯 <b>Action:</b> {pending_signal.action}
💰 <b>Entry:</b> ₹{pending_signal.entry_price:.2f}

⏰ <b>Status:</b> Signal expired (5 minutes timeout)
❌ <b>Result:</b> No action taken

💡 <i>Signal was not approved within the time limit</i>
"""
                    await self._edit_message(pending_signal.message_id, expire_text)
                
                # Clean up
                await self._remove_pending_signal(signal_id)
                
                logger.info(f"⏰ Signal {signal_id} expired automatically")
                
        except Exception as e:
            logger.error(f"❌ Signal expiry handling failed: {e}")


# ================================================================
# FACTORY FUNCTION (backward compatibility)
# ================================================================

def create_telegram_service() -> EnhancedTelegramService:
    """Factory function to create enhanced telegram service (backward compatibility)"""
    return EnhancedTelegramService()

# Alias for backward compatibility
TelegramService = EnhancedTelegramService
