# backend/app/services/telegram_service.py
"""
TradeMind AI - Production Telegram Service
Real Telegram Bot integration with proper error handling and formatting
"""

import asyncio
import aiohttp
import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from urllib.parse import quote
from dotenv import load_dotenv

# Explicitly load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TelegramService:
    """Production Telegram notification service for trading signals"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        # Get credentials from environment variables
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token:
            logger.warning("⚠️ TELEGRAM_BOT_TOKEN not configured - Telegram disabled")
            self.enabled = False
            return
            
        if not self.chat_id:
            logger.warning("⚠️ TELEGRAM_CHAT_ID not configured - Telegram disabled")
            self.enabled = False
            return
            
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.enabled = True
        
        logger.info("✅ Telegram service initialized successfully")
    
    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a text message to Telegram"""
        if not self.enabled:
            logger.debug("Telegram disabled - message not sent")
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            
            # Prepare the payload
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            # Send the message using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            logger.info("✅ Telegram message sent successfully")
                            return True
                        else:
                            logger.error(f"❌ Telegram API error: {result.get('description')}")
                            return False
                    else:
                        logger.error(f"❌ Telegram HTTP error: {response.status}")
                        return False
                        
        except aiohttp.ClientError as e:
            logger.error(f"❌ Telegram network error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Telegram unexpected error: {e}")
            return False
    
    async def send_signal_notification(self, signal: Dict[str, Any]) -> bool:
        """Send beautifully formatted trading signal notification"""
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

⏰ <b>Time:</b> {timestamp.strftime("%d-%m-%Y %H:%M:%S")}

💡 <i>Risk-Reward Ratio:</i> {(profit_pct/risk_pct if risk_pct > 0 else 0):.2f}:1

⚠️ <i>Trade at your own risk. This is algorithmic analysis, not financial advice.</i>
""".strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Signal notification formatting error: {e}")
            return False
    
    async def send_system_startup_notification(self) -> bool:
        """Send system startup notification"""
        if not self.enabled:
            return False
            
        try:
            current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
            
            message = f"""
🤖 <b>TradeMind AI System Started</b>

✅ <b>Status:</b> Fully Operational
🕐 <b>Start Time:</b> {current_time}
📊 <b>Market:</b> Indian Stock Market
🎯 <b>Signal Mode:</b> Production
🔄 <b>Auto-Trading:</b> Ready

🚨 <b>Features Active:</b>
• Real-time signal generation
• ML-powered analysis (65%+ accuracy)
• Live market data monitoring
• WebSocket dashboard updates

📱 You will receive trading signals during market hours (9:15 AM - 3:30 PM IST)

💼 <i>Ready to analyze 100+ Nifty stocks</i>
""".strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Startup notification error: {e}")
            return False
    
    async def send_system_shutdown_notification(self) -> bool:
        """Send system shutdown notification"""
        if not self.enabled:
            return False
            
        try:
            current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")
            
            message = f"""
🛑 <b>TradeMind AI System Shutdown</b>

📴 <b>Status:</b> System Stopped
🕐 <b>Shutdown Time:</b> {current_time}

💤 <i>Trading signals paused. System will resume on next startup.</i>
""".strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Shutdown notification error: {e}")
            return False
    
    async def send_daily_summary(self, summary: Dict[str, Any]) -> bool:
        """Send daily trading summary"""
        if not self.enabled:
            return False
            
        try:
            signals_count = summary.get("signals_generated", 0)
            win_rate = summary.get("win_rate", 0.0)
            total_pnl = summary.get("total_pnl", 0.0)
            best_signal = summary.get("best_signal", {})
            
            pnl_emoji = "📈" if total_pnl >= 0 else "📉"
            
            message = f"""
📊 <b>TradeMind AI Daily Summary</b>

🎯 <b>Signals Generated:</b> {signals_count}
🏆 <b>Win Rate:</b> {win_rate:.1f}%
{pnl_emoji} <b>Total P&L:</b> ₹{total_pnl:+,.2f}

🌟 <b>Best Signal:</b> {best_signal.get('symbol', 'N/A')}
💰 <b>Best Return:</b> {best_signal.get('return_pct', 0):+.1f}%

📅 <b>Date:</b> {datetime.now().strftime("%d-%m-%Y")}

🚀 <i>Keep following for tomorrow's signals!</i>
""".strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Daily summary error: {e}")
            return False
    
    async def send_market_alert(self, alert_type: str, message_text: str) -> bool:
        """Send market alerts (volatility, news, etc.)"""
        if not self.enabled:
            return False
            
        try:
            alert_emojis = {
                "volatility": "⚡",
                "news": "📰",
                "error": "🚨",
                "warning": "⚠️",
                "info": "ℹ️"
            }
            
            emoji = alert_emojis.get(alert_type, "🔔")
            
            message = f"""
{emoji} <b>TradeMind AI Alert</b>

{message_text}

🕐 <b>Time:</b> {datetime.now().strftime("%H:%M:%S IST")}
""".strip()
            
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"❌ Market alert error: {e}")
            return False
    
    async def send_test_message(self) -> bool:
        """Send a test message to verify Telegram connection"""
        if not self.enabled:
            logger.warning("Telegram not configured - test message skipped")
            return False
            
        try:
            test_message = f"""
🧪 <b>TradeMind AI Test Message</b>

✅ Telegram integration is working!
🕐 <b>Test Time:</b> {datetime.now().strftime("%d-%m-%Y %H:%M:%S IST")}

🎯 Your bot is ready to receive trading signals.
""".strip()
            
            result = await self.send_message(test_message)
            
            if result:
                logger.info("✅ Telegram test message sent successfully")
            else:
                logger.error("❌ Telegram test message failed")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Telegram test message error: {e}")
            return False
    
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return self.enabled
    
    async def get_bot_info(self) -> Optional[Dict]:
        """Get bot information for verification"""
        if not self.enabled:
            return None
            
        try:
            url = f"{self.base_url}/getMe"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            return result.get("result")
                            
        except Exception as e:
            logger.error(f"❌ Get bot info error: {e}")
            
        return None


# ================================================================
# Factory function for easy integration
# ================================================================

def create_telegram_service() -> TelegramService:
    """Factory function to create and configure Telegram service"""
    return TelegramService()


# ================================================================
# Example usage and testing
# ================================================================

async def main():
    """Test the Telegram service"""
    telegram = create_telegram_service()
    
    if telegram.is_configured():
        print("Testing Telegram service...")
        
        # Test basic message
        await telegram.send_test_message()
        
        # Test startup notification
        await telegram.send_system_startup_notification()
        
        # Test signal notification
        sample_signal = {
            "symbol": "RELIANCE",
            "action": "BUY",
            "entry_price": 2450.50,
            "target_price": 2580.00,
            "stop_loss": 2380.00,
            "confidence": 87.5,
            "timestamp": datetime.now()
        }
        await telegram.send_signal_notification(sample_signal)
        
    else:
        print("❌ Telegram not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")

if __name__ == "__main__":
    # For testing purposes
    asyncio.run(main())