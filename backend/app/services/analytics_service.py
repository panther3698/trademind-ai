import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import threading

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service for tracking and analyzing trading performance - CORRECTED VERSION"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Thread lock for safe concurrent access
        self._stats_lock = threading.Lock()
        
        # CORRECTED: Complete initialization of all possible keys
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
            "last_signal_time": None,
            # Additional keys that may be accessed
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "system_uptime_hours": 0.0,
            "status": "STARTING",
            "error_count": 0,
            "last_error_time": None
        }
        
        self.monthly_stats = {
            "total_signals": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "best_signal": None,
            "worst_signal": None,
            "active_users": 0
        }
        
        logger.info("✅ AnalyticsService initialized with all keys")
    
    def _ensure_key_exists(self, key: str, default_value=0):
        """Ensure a key exists in daily_stats with a default value"""
        with self._stats_lock:
            if key not in self.daily_stats:
                self.daily_stats[key] = default_value
                logger.debug(f"Added missing key '{key}' with default value: {default_value}")
    
    async def track_signal_generated(self, signal: Dict) -> None:
        """Track when a signal is generated - SAFE VERSION"""
        try:
            with self._stats_lock:
                # Ensure keys exist
                self._ensure_key_exists("signals_generated", 0)
                self._ensure_key_exists("average_confidence", 0.0)
                
                self.daily_stats["signals_generated"] += 1
                self.daily_stats["last_signal_time"] = datetime.now()
                
                # Update average confidence safely
                current_avg = self.daily_stats.get("average_confidence", 0.0)
                count = self.daily_stats.get("signals_generated", 1)
                new_confidence = signal.get("confidence", 0.0)
                
                if count > 0:
                    self.daily_stats["average_confidence"] = (
                        (current_avg * (count - 1) + new_confidence) / count
                    )
                else:
                    self.daily_stats["average_confidence"] = new_confidence
            
            symbol = signal.get("symbol", "UNKNOWN")
            action = signal.get("action", "UNKNOWN")
            logger.info(f"📊 Signal tracked: {symbol} {action} (Confidence: {new_confidence:.1%})")
            
        except Exception as e:
            logger.error(f"❌ Error tracking signal: {e}")
            self._increment_error_count()
    
    async def track_telegram_sent(self, success: bool, signal: Dict) -> None:
        """Track Telegram notification success/failure - SAFE VERSION"""
        try:
            with self._stats_lock:
                # Ensure keys exist
                self._ensure_key_exists("telegram_success", 0)
                self._ensure_key_exists("telegram_failures", 0)
                self._ensure_key_exists("signals_sent", 0)
                
                if success:
                    self.daily_stats["telegram_success"] += 1
                    self.daily_stats["signals_sent"] += 1
                    symbol = signal.get("symbol", "UNKNOWN")
                    action = signal.get("action", "UNKNOWN")
                    logger.info(f"📱 Telegram sent: {symbol} {action}")
                else:
                    self.daily_stats["telegram_failures"] += 1
                    symbol = signal.get("symbol", "UNKNOWN")
                    action = signal.get("action", "UNKNOWN")
                    logger.warning(f"📱 Telegram failed: {symbol} {action}")
                
        except Exception as e:
            logger.error(f"❌ Error tracking Telegram: {e}")
            self._increment_error_count()
    
    async def track_signal_outcome(self, signal_id: str, outcome: str, pnl: float) -> None:
        """Track the outcome of a signal (profit/loss) - SAFE VERSION"""
        try:
            with self._stats_lock:
                # Ensure keys exist
                self._ensure_key_exists("total_profit_loss", 0.0)
                self._ensure_key_exists("successful_signals", 0)
                self._ensure_key_exists("failed_signals", 0)
                
                self.daily_stats["total_profit_loss"] += pnl
                
                if outcome == "profit":
                    self.daily_stats["successful_signals"] += 1
                else:
                    self.daily_stats["failed_signals"] += 1
            
            logger.info(f"💰 Signal outcome: {signal_id} -> {outcome} (₹{pnl:,.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error tracking outcome: {e}")
            self._increment_error_count()
    
    def get_daily_stats(self) -> Dict:
        """Get current daily statistics - COMPLETELY SAFE VERSION"""
        try:
            with self._stats_lock:
                # Ensure all keys exist with safe defaults
                safe_stats = {
                    "signals_generated": self.daily_stats.get("signals_generated", 0),
                    "signals_sent": self.daily_stats.get("signals_sent", 0),
                    "telegram_success": self.daily_stats.get("telegram_success", 0),
                    "telegram_failures": self.daily_stats.get("telegram_failures", 0),
                    "total_profit_loss": self.daily_stats.get("total_profit_loss", 0.0),
                    "successful_signals": self.daily_stats.get("successful_signals", 0),
                    "failed_signals": self.daily_stats.get("failed_signals", 0),
                    "average_confidence": self.daily_stats.get("average_confidence", 0.0),
                    "uptime_start": self.daily_stats.get("uptime_start", datetime.now()),
                    "last_signal_time": self.daily_stats.get("last_signal_time"),
                    "error_count": self.daily_stats.get("error_count", 0)
                }
                
                # Safe calculations
                total_signals = safe_stats["successful_signals"] + safe_stats["failed_signals"]
                win_rate = (safe_stats["successful_signals"] / max(total_signals, 1)) * 100
                
                telegram_total = safe_stats["telegram_success"] + safe_stats["telegram_failures"]
                telegram_success_rate = (safe_stats["telegram_success"] / max(telegram_total, 1)) * 100
                
                uptime_seconds = (datetime.now() - safe_stats["uptime_start"]).total_seconds()
                uptime_hours = uptime_seconds / 3600
                
                # Return comprehensive safe statistics
                result = {
                    "signals_generated": safe_stats["signals_generated"],
                    "signals_sent": safe_stats["signals_sent"],
                    "successful_signals": safe_stats["successful_signals"],
                    "failed_signals": safe_stats["failed_signals"],
                    "win_rate": round(win_rate, 1),
                    "total_pnl": round(safe_stats["total_profit_loss"], 2),
                    "average_confidence": round(safe_stats["average_confidence"] * 100, 1),
                    "telegram_success": safe_stats["telegram_success"],
                    "telegram_failures": safe_stats["telegram_failures"],
                    "telegram_success_rate": round(telegram_success_rate, 1),
                    "system_uptime_hours": round(uptime_hours, 1),
                    "last_signal_time": safe_stats["last_signal_time"].isoformat() if safe_stats["last_signal_time"] else None,
                    "status": self._get_system_status_safe(safe_stats),
                    "error_count": safe_stats["error_count"],
                    "error_rate": self._calculate_error_rate_safe(safe_stats)
                }
                
                return result
            
        except Exception as e:
            logger.error(f"❌ Error getting daily stats: {e}")
            # Return safe defaults if everything fails
            return {
                "signals_generated": 0,
                "signals_sent": 0,
                "successful_signals": 0,
                "failed_signals": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_confidence": 0.0,
                "telegram_success": 0,
                "telegram_failures": 0,
                "telegram_success_rate": 0.0,
                "system_uptime_hours": 0.0,
                "last_signal_time": None,
                "status": "ERROR",
                "error_count": 1,
                "error_rate": 0.0
            }
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary for dashboard - SAFE VERSION"""
        try:
            daily = self.get_daily_stats()
            
            return {
                "daily": daily,
                "system_health": {
                    "telegram_configured": bool(daily.get("telegram_success", 0) > 0 or daily.get("telegram_failures", 0) > 0),
                    "signal_generation_active": bool(daily.get("signals_generated", 0) > 0),
                    "last_activity": daily.get("last_signal_time"),
                    "error_rate": daily.get("error_rate", 0.0),
                    "uptime_hours": daily.get("system_uptime_hours", 0.0)
                },
                "recent_performance": {
                    "signals_today": daily.get("signals_generated", 0),
                    "success_rate": daily.get("win_rate", 0.0),
                    "profit_loss": daily.get("total_pnl", 0.0),
                    "avg_confidence": daily.get("average_confidence", 0.0)
                },
                "telegram_stats": {
                    "messages_sent": daily.get("telegram_success", 0),
                    "messages_failed": daily.get("telegram_failures", 0),
                    "success_rate": daily.get("telegram_success_rate", 0.0)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {
                "error": str(e),
                "daily": self.get_daily_stats(),
                "system_health": {"status": "ERROR"},
                "recent_performance": {"signals_today": 0},
                "telegram_stats": {"messages_sent": 0}
            }
    
    def _get_system_status_safe(self, safe_stats: Dict) -> str:
        """Determine current system status - SAFE VERSION"""
        try:
            signals_generated = safe_stats.get("signals_generated", 0)
            
            if signals_generated == 0:
                return "STARTING"
            
            telegram_total = safe_stats.get("telegram_success", 0) + safe_stats.get("telegram_failures", 0)
            error_count = safe_stats.get("error_count", 0)
            
            # Check for high error rate
            if error_count > 10:
                return "ERROR"
            
            if telegram_total > 0:
                telegram_rate = safe_stats.get("telegram_success", 0) / telegram_total
                if telegram_rate > 0.9:
                    return "EXCELLENT"
                elif telegram_rate > 0.7:
                    return "GOOD"
                else:
                    return "ISSUES"
            
            return "ACTIVE"
            
        except Exception as e:
            logger.error(f"Error determining system status: {e}")
            return "UNKNOWN"
    
    def _calculate_error_rate_safe(self, safe_stats: Dict) -> float:
        """Calculate overall error rate - SAFE VERSION"""
        try:
            total_operations = (
                safe_stats.get("signals_generated", 0) + 
                safe_stats.get("telegram_success", 0) + 
                safe_stats.get("telegram_failures", 0)
            )
            
            errors = safe_stats.get("telegram_failures", 0) + safe_stats.get("error_count", 0)
            
            return round((errors / max(total_operations, 1) * 100), 2)
            
        except Exception as e:
            logger.error(f"Error calculating error rate: {e}")
            return 0.0
    
    def _increment_error_count(self):
        """Safely increment error count"""
        try:
            with self._stats_lock:
                self._ensure_key_exists("error_count", 0)
                self._ensure_key_exists("last_error_time", None)
                
                self.daily_stats["error_count"] += 1
                self.daily_stats["last_error_time"] = datetime.now()
                
        except Exception as e:
            logger.error(f"Error incrementing error count: {e}")
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (called at midnight) - SAFE VERSION"""
        try:
            with self._stats_lock:
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
                    "last_signal_time": None,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "system_uptime_hours": 0.0,
                    "status": "STARTING",
                    "error_count": 0,
                    "last_error_time": None
                }
            
            logger.info("🔄 Daily statistics reset successfully")
            
        except Exception as e:
            logger.error(f"❌ Error resetting daily stats: {e}")
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health information"""
        try:
            daily = self.get_daily_stats()
            
            return {
                "overall_status": daily.get("status", "UNKNOWN"),
                "uptime_hours": daily.get("system_uptime_hours", 0.0),
                "signals_generated_today": daily.get("signals_generated", 0),
                "win_rate_percentage": daily.get("win_rate", 0.0),
                "telegram_operational": daily.get("telegram_success_rate", 0.0) > 50.0,
                "error_rate_percentage": daily.get("error_rate", 0.0),
                "last_signal_time": daily.get("last_signal_time"),
                "total_pnl_today": daily.get("total_pnl", 0.0),
                "average_confidence": daily.get("average_confidence", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system health: {e}")
            return {
                "overall_status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_quick_stats(self) -> Dict:
        """Get quick stats for frequent polling"""
        try:
            with self._stats_lock:
                return {
                    "signals_today": self.daily_stats.get("signals_generated", 0),
                    "last_signal": self.daily_stats.get("last_signal_time").isoformat() if self.daily_stats.get("last_signal_time") else None,
                    "status": self._get_system_status_safe(self.daily_stats),
                    "uptime_hours": round((datetime.now() - self.daily_stats.get("uptime_start", datetime.now())).total_seconds() / 3600, 1)
                }
        except Exception as e:
            logger.error(f"❌ Error getting quick stats: {e}")
            return {
                "signals_today": 0,
                "last_signal": None,
                "status": "ERROR",
                "uptime_hours": 0.0
            }