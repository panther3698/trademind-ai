import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service for tracking and analyzing trading performance"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # In-memory stats for quick access
        self.daily_stats = {
            "signals_generated": 0,
            "signals_sent": 0,
            "telegram_success": 0,
            "telegram_failures": 0,
            "total_profit_loss": 0.0,
            "successful_signals": 0,
            "failed_signals": 0,
            "average_confidence": 0.0,
            "uptime_start": datetime.now(),
            "last_signal_time": None
        }
        
        self.monthly_stats = {
            "total_signals": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "best_signal": None,
            "worst_signal": None,
            "active_users": 0
        }
    
    async def track_signal_generated(self, signal: Dict) -> None:
        """Track when a signal is generated"""
        try:
            self.daily_stats["signals_generated"] += 1
            self.daily_stats["last_signal_time"] = datetime.now()
            
            # Update average confidence
            current_avg = self.daily_stats["average_confidence"]
            count = self.daily_stats["signals_generated"]
            new_confidence = signal.get("confidence", 0.0)
            
            self.daily_stats["average_confidence"] = (
                (current_avg * (count - 1) + new_confidence) / count
            )
            
            logger.info(f"📊 Signal tracked: {signal['symbol']} {signal['action']} (Confidence: {new_confidence:.1%})")
            
        except Exception as e:
            logger.error(f"❌ Error tracking signal: {e}")
    
    async def track_telegram_sent(self, success: bool, signal: Dict) -> None:
        """Track Telegram notification success/failure"""
        try:
            if success:
                self.daily_stats["telegram_success"] += 1
                self.daily_stats["signals_sent"] += 1
                logger.info(f"📱 Telegram sent: {signal['symbol']} {signal['action']}")
            else:
                self.daily_stats["telegram_failures"] += 1
                logger.warning(f"📱 Telegram failed: {signal['symbol']} {signal['action']}")
                
        except Exception as e:
            logger.error(f"❌ Error tracking Telegram: {e}")
    
    async def track_signal_outcome(self, signal_id: str, outcome: str, pnl: float) -> None:
        """Track the outcome of a signal (profit/loss)"""
        try:
            self.daily_stats["total_profit_loss"] += pnl
            
            if outcome == "profit":
                self.daily_stats["successful_signals"] += 1
            else:
                self.daily_stats["failed_signals"] += 1
            
            logger.info(f"💰 Signal outcome: {signal_id} -> {outcome} (₹{pnl:,.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error tracking outcome: {e}")
    
    def get_daily_stats(self) -> Dict:
        """Get current daily statistics"""
        try:
            total_signals = self.daily_stats["successful_signals"] + self.daily_stats["failed_signals"]
            win_rate = (self.daily_stats["successful_signals"] / total_signals * 100) if total_signals > 0 else 0
            
            telegram_total = self.daily_stats["telegram_success"] + self.daily_stats["telegram_failures"]
            telegram_success_rate = (self.daily_stats["telegram_success"] / telegram_total * 100) if telegram_total > 0 else 0
            
            uptime_seconds = (datetime.now() - self.daily_stats["uptime_start"]).total_seconds()
            uptime_hours = uptime_seconds / 3600
            
            return {
                "signals_generated": self.daily_stats["signals_generated"],
                "signals_sent": self.daily_stats["signals_sent"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(self.daily_stats["total_profit_loss"], 2),
                "average_confidence": round(self.daily_stats["average_confidence"] * 100, 1),
                "telegram_success_rate": round(telegram_success_rate, 1),
                "system_uptime_hours": round(uptime_hours, 1),
                "last_signal_time": self.daily_stats["last_signal_time"].isoformat() if self.daily_stats["last_signal_time"] else None,
                "status": self._get_system_status()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting daily stats: {e}")
            return {"error": str(e)}
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for dashboard"""
        try:
            daily = self.get_daily_stats()
            
            return {
                "daily": daily,
                "system_health": {
                    "telegram_configured": bool(self.daily_stats.get("telegram_success", 0) > 0),
                    "signal_generation_active": bool(self.daily_stats.get("signals_generated", 0) > 0),
                    "last_activity": daily.get("last_signal_time"),
                    "error_rate": self._calculate_error_rate()
                },
                "recent_performance": {
                    "signals_today": daily["signals_generated"],
                    "success_rate": daily["win_rate"],
                    "profit_loss": daily["total_pnl"],
                    "avg_confidence": daily["average_confidence"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def _get_system_status(self) -> str:
        """Determine current system status"""
        try:
            if self.daily_stats["signals_generated"] == 0:
                return "STARTING"
            
            telegram_total = self.daily_stats["telegram_success"] + self.daily_stats["telegram_failures"]
            if telegram_total > 0:
                telegram_rate = self.daily_stats["telegram_success"] / telegram_total
                if telegram_rate > 0.9:
                    return "EXCELLENT"
                elif telegram_rate > 0.7:
                    return "GOOD"
                else:
                    return "ISSUES"
            
            return "ACTIVE"
            
        except Exception as e:
            return "ERROR"
    
    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate"""
        try:
            total_operations = (
                self.daily_stats["signals_generated"] + 
                self.daily_stats["telegram_success"] + 
                self.daily_stats["telegram_failures"]
            )
            
            errors = self.daily_stats["telegram_failures"]
            
            return round((errors / total_operations * 100), 2) if total_operations > 0 else 0.0
            
        except Exception as e:
            return 0.0
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (called at midnight)"""
        try:
            self.daily_stats = {
                "signals_generated": 0,
                "signals_sent": 0,
                "telegram_success": 0,
                "telegram_failures": 0,
                "total_profit_loss": 0.0,
                "successful_signals": 0,
                "failed_signals": 0,
                "average_confidence": 0.0,
                "uptime_start": datetime.now(),
                "last_signal_time": None
            }
            
            logger.info("🔄 Daily statistics reset")
            
        except Exception as e:
            logger.error(f"❌ Error resetting daily stats: {e}")