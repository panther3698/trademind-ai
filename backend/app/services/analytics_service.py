# backend/app/services/analytics_service.py
"""
TradeMind AI - Production Analytics Service
Complete performance tracking, signal analytics, and system monitoring
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import os
from contextlib import asynccontextmanager
import aiosqlite

logger = logging.getLogger(__name__)

@dataclass
class SignalPerformance:
    """Signal performance tracking"""
    signal_id: str
    symbol: str
    action: str
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    created_at: datetime
    status: str  # active, completed, stopped
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    success: Optional[bool] = None

@dataclass
class DailyStats:
    """Daily system statistics"""
    date: str
    signals_generated: int
    signals_sent: int
    signals_successful: int
    signals_failed: int
    total_pnl: float
    win_rate: float
    average_confidence: float
    telegram_success_rate: float
    system_uptime_hours: float
    last_signal_time: Optional[str]
    status: str

@dataclass
class SystemHealth:
    """System health metrics"""
    telegram_configured: bool
    signal_generation_active: bool
    market_data_connected: bool
    ml_models_loaded: bool
    last_activity: Optional[str]
    error_rate: float
    uptime_hours: float

class AnalyticsService:
    """
    Production Analytics Service for TradeMind AI
    Tracks signals, performance, system health, and user analytics
    """
    
    def __init__(self, database_url: str = None):
        # Database configuration
        self.database_url = database_url or "sqlite:///./analytics.db"
        self.db_path = self._extract_db_path(self.database_url)
        
        # Ensure database directory exists (only if there's a directory part)
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # In-memory state for fast access
        self.daily_stats = DailyStats(
            date=datetime.now().strftime("%Y-%m-%d"),
            signals_generated=0,
            signals_sent=0,
            signals_successful=0,
            signals_failed=0,
            total_pnl=0.0,
            win_rate=0.0,
            average_confidence=0.0,
            telegram_success_rate=100.0,
            system_uptime_hours=0.0,
            last_signal_time=None,
            status="operational"
        )
        
        # System tracking
        self.start_time = datetime.now()
        self.telegram_sent = 0
        self.telegram_failed = 0
        self.signal_performances: Dict[str, SignalPerformance] = {}
        
        # Initialize database
        asyncio.create_task(self._initialize_database())
        
        logger.info("✅ AnalyticsService initialized with all keys")
    
    def _extract_db_path(self, database_url: str) -> str:
        """Extract database file path from URL"""
        if database_url.startswith("sqlite:///"):
            path = database_url.replace("sqlite:///", "")
            # Handle relative paths on Windows
            if not os.path.isabs(path):
                # Convert relative path to absolute path in current directory
                path = os.path.join(os.getcwd(), path)
        elif database_url.startswith("sqlite://"):
            path = database_url.replace("sqlite://", "")
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
        else:
            # Default fallback - create in current directory
            path = os.path.join(os.getcwd(), "analytics.db")
        
        # Ensure the path uses proper separators for the OS
        return os.path.normpath(path)
    
    async def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create signals table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        target_price REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        confidence REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        exit_price REAL,
                        exit_time TEXT,
                        pnl REAL,
                        pnl_percentage REAL,
                        success BOOLEAN
                    )
                """)
                
                # Create daily_stats table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS daily_stats (
                        date TEXT PRIMARY KEY,
                        signals_generated INTEGER DEFAULT 0,
                        signals_sent INTEGER DEFAULT 0,
                        signals_successful INTEGER DEFAULT 0,
                        signals_failed INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0.0,
                        win_rate REAL DEFAULT 0.0,
                        average_confidence REAL DEFAULT 0.0,
                        telegram_success_rate REAL DEFAULT 100.0,
                        system_uptime_hours REAL DEFAULT 0.0,
                        last_signal_time TEXT,
                        status TEXT DEFAULT 'operational'
                    )
                """)
                
                # Create system_events table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        event_data TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                
                await db.commit()
                logger.info("✅ Analytics database initialized")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    async def track_signal_generated(self, signal: Dict[str, Any]):
        """Track a new signal generation"""
        try:
            # Update in-memory stats
            self.daily_stats.signals_generated += 1
            self.daily_stats.last_signal_time = datetime.now().isoformat()
            
            # Update average confidence
            if self.daily_stats.signals_generated > 0:
                current_avg = self.daily_stats.average_confidence
                new_confidence = signal.get("confidence", 0.0) * 100  # Convert to percentage
                count = self.daily_stats.signals_generated
                
                self.daily_stats.average_confidence = (
                    (current_avg * (count - 1) + new_confidence) / count
                )
            
            # Create signal performance record
            signal_perf = SignalPerformance(
                signal_id=signal.get("id", f"signal_{int(datetime.now().timestamp())}"),
                symbol=signal.get("symbol", "UNKNOWN"),
                action=signal.get("action", "BUY"),
                entry_price=signal.get("entry_price", 0.0),
                target_price=signal.get("target_price", 0.0),
                stop_loss=signal.get("stop_loss", 0.0),
                confidence=signal.get("confidence", 0.0),
                created_at=datetime.now(),
                status="active"
            )
            
            self.signal_performances[signal_perf.signal_id] = signal_perf
            
            # Store in database
            await self._store_signal_in_db(signal_perf)
            await self._update_daily_stats_in_db()
            
            # Log system event
            await self._log_system_event("signal_generated", {
                "signal_id": signal_perf.signal_id,
                "symbol": signal_perf.symbol,
                "action": signal_perf.action,
                "confidence": signal_perf.confidence
            })
            
            logger.info(f"📊 Signal tracked: {signal_perf.symbol} {signal_perf.action} (ID: {signal_perf.signal_id})")
            
        except Exception as e:
            logger.error(f"❌ Signal tracking failed: {e}")
    
    async def track_telegram_sent(self, success: bool, signal: Dict[str, Any]):
        """Track Telegram notification success/failure"""
        try:
            if success:
                self.telegram_sent += 1
                self.daily_stats.signals_sent += 1
            else:
                self.telegram_failed += 1
            
            # Update success rate
            total_attempts = self.telegram_sent + self.telegram_failed
            if total_attempts > 0:
                self.daily_stats.telegram_success_rate = (self.telegram_sent / total_attempts) * 100
            
            # Update database
            await self._update_daily_stats_in_db()
            
            # Log event
            await self._log_system_event("telegram_notification", {
                "success": success,
                "signal_id": signal.get("id"),
                "symbol": signal.get("symbol")
            })
            
            logger.info(f"📱 Telegram tracked: {'✅ Success' if success else '❌ Failed'} | Rate: {self.daily_stats.telegram_success_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Telegram tracking failed: {e}")
    
    async def track_signal_completion(self, signal_id: str, exit_price: float, success: bool):
        """Track signal completion (hit target or stop loss)"""
        try:
            if signal_id not in self.signal_performances:
                logger.warning(f"Signal {signal_id} not found for completion tracking")
                return
            
            signal_perf = self.signal_performances[signal_id]
            signal_perf.exit_price = exit_price
            signal_perf.exit_time = datetime.now()
            signal_perf.success = success
            signal_perf.status = "completed"
            
            # Calculate P&L
            if signal_perf.action.upper() == "BUY":
                signal_perf.pnl = exit_price - signal_perf.entry_price
            else:  # SELL
                signal_perf.pnl = signal_perf.entry_price - exit_price
            
            signal_perf.pnl_percentage = (signal_perf.pnl / signal_perf.entry_price) * 100
            
            # Update daily stats
            if success:
                self.daily_stats.signals_successful += 1
            else:
                self.daily_stats.signals_failed += 1
            
            self.daily_stats.total_pnl += signal_perf.pnl
            
            # Update win rate
            total_completed = self.daily_stats.signals_successful + self.daily_stats.signals_failed
            if total_completed > 0:
                self.daily_stats.win_rate = (self.daily_stats.signals_successful / total_completed) * 100
            
            # Update database
            await self._update_signal_in_db(signal_perf)
            await self._update_daily_stats_in_db()
            
            logger.info(f"🎯 Signal completed: {signal_id} | P&L: ₹{signal_perf.pnl:.2f} ({signal_perf.pnl_percentage:+.1f}%) | {'✅ Success' if success else '❌ Failed'}")
            
        except Exception as e:
            logger.error(f"❌ Signal completion tracking failed: {e}")
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get current daily statistics"""
        try:
            # Update uptime
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600
            self.daily_stats.system_uptime_hours = round(uptime, 2)
            
            return asdict(self.daily_stats)
            
        except Exception as e:
            logger.error(f"❌ Get daily stats failed: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            daily = self.get_daily_stats()
            
            # Calculate error rate
            total_telegram = self.telegram_sent + self.telegram_failed
            error_rate = (self.telegram_failed / max(total_telegram, 1)) * 100
            
            return {
                "daily": daily,
                "system_health": {
                    "telegram_configured": True,  # Will be updated by main.py
                    "signal_generation_active": True,
                    "market_data_connected": True,
                    "ml_models_loaded": True,
                    "last_activity": daily["last_signal_time"],
                    "error_rate": round(error_rate, 2),
                    "uptime_hours": daily["system_uptime_hours"]
                },
                "configuration": {
                    "max_signals_per_day": 3,  # Default, will be overridden
                    "min_confidence_threshold": 0.65,
                    "signal_interval_seconds": 45,
                    "telegram_enabled": True,
                    "zerodha_enabled": False
                },
                "recent_performance": {
                    "signals_today": daily["signals_generated"],
                    "success_rate": daily["telegram_success_rate"],
                    "profit_loss": daily["total_pnl"],
                    "avg_confidence": daily["average_confidence"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Performance summary failed: {e}")
            return {
                "daily": asdict(self.daily_stats),
                "system_health": {"error": str(e)},
                "configuration": {},
                "recent_performance": {}
            }
    
    async def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent signals from database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ) as cursor:
                    rows = await cursor.fetchall()
                    
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Get recent signals failed: {e}")
            return []
    
    async def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get complete analytics dashboard data"""
        try:
            performance = self.get_performance_summary()
            recent_signals = await self.get_recent_signals(5)
            
            return {
                "performance": performance,
                "recent_signals": recent_signals,
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0"
            }
            
        except Exception as e:
            logger.error(f"❌ Analytics dashboard failed: {e}")
            return {"error": str(e)}
    
    async def _store_signal_in_db(self, signal_perf: SignalPerformance):
        """Store signal performance in database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO signals 
                    (id, symbol, action, entry_price, target_price, stop_loss, confidence, 
                     created_at, status, exit_price, exit_time, pnl, pnl_percentage, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_perf.signal_id,
                    signal_perf.symbol,
                    signal_perf.action,
                    signal_perf.entry_price,
                    signal_perf.target_price,
                    signal_perf.stop_loss,
                    signal_perf.confidence,
                    signal_perf.created_at.isoformat(),
                    signal_perf.status,
                    signal_perf.exit_price,
                    signal_perf.exit_time.isoformat() if signal_perf.exit_time else None,
                    signal_perf.pnl,
                    signal_perf.pnl_percentage,
                    signal_perf.success
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"❌ Store signal in DB failed: {e}")
    
    async def _update_signal_in_db(self, signal_perf: SignalPerformance):
        """Update existing signal in database"""
        await self._store_signal_in_db(signal_perf)  # Same operation for SQLite
    
    async def _update_daily_stats_in_db(self):
        """Update daily stats in database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO daily_stats
                    (date, signals_generated, signals_sent, signals_successful, signals_failed,
                     total_pnl, win_rate, average_confidence, telegram_success_rate,
                     system_uptime_hours, last_signal_time, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.daily_stats.date,
                    self.daily_stats.signals_generated,
                    self.daily_stats.signals_sent,
                    self.daily_stats.signals_successful,
                    self.daily_stats.signals_failed,
                    self.daily_stats.total_pnl,
                    self.daily_stats.win_rate,
                    self.daily_stats.average_confidence,
                    self.daily_stats.telegram_success_rate,
                    self.daily_stats.system_uptime_hours,
                    self.daily_stats.last_signal_time,
                    self.daily_stats.status
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"❌ Update daily stats in DB failed: {e}")
    
    async def _log_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log system events to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO system_events (event_type, event_data, timestamp)
                    VALUES (?, ?, ?)
                """, (
                    event_type,
                    json.dumps(event_data),
                    datetime.now().isoformat()
                ))
                await db.commit()
                
        except Exception as e:
            logger.debug(f"System event logging failed: {e}")
    
    async def get_historical_performance(self, days: int = 30) -> Dict[str, Any]:
        """Get historical performance data"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get daily stats for the last N days
                async with db.execute("""
                    SELECT * FROM daily_stats 
                    WHERE date >= date('now', '-{} days')
                    ORDER BY date DESC
                """.format(days)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    daily_history = [dict(zip(columns, row)) for row in rows]
                
                # Get completed signals
                async with db.execute("""
                    SELECT * FROM signals 
                    WHERE status = 'completed' 
                    AND created_at >= datetime('now', '-{} days')
                    ORDER BY created_at DESC
                """.format(days)) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    completed_signals = [dict(zip(columns, row)) for row in rows]
                
                return {
                    "daily_history": daily_history,
                    "completed_signals": completed_signals,
                    "period_days": days,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"❌ Historical performance failed: {e}")
            return {"error": str(e)}
    
    async def reset_daily_stats(self):
        """Reset daily stats for new trading day"""
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            if self.daily_stats.date != current_date:
                logger.info(f"🌅 New trading day: {current_date}")
                
                # Archive current stats
                await self._update_daily_stats_in_db()
                
                # Reset for new day
                self.daily_stats = DailyStats(
                    date=current_date,
                    signals_generated=0,
                    signals_sent=0,
                    signals_successful=0,
                    signals_failed=0,
                    total_pnl=0.0,
                    win_rate=0.0,
                    average_confidence=0.0,
                    telegram_success_rate=100.0,
                    system_uptime_hours=0.0,
                    last_signal_time=None,
                    status="operational"
                )
                
                # Reset counters
                self.telegram_sent = 0
                self.telegram_failed = 0
                self.signal_performances.clear()
                self.start_time = datetime.now()
                
        except Exception as e:
            logger.error(f"❌ Daily stats reset failed: {e}")
    
    async def close(self):
        """Close analytics service and save final stats"""
        try:
            await self._update_daily_stats_in_db()
            await self._log_system_event("analytics_service_shutdown", {
                "final_stats": asdict(self.daily_stats)
            })
            logger.info("✅ Analytics service closed gracefully")
            
        except Exception as e:
            logger.error(f"Error closing analytics service: {e}")

# ================================================================
# Factory Function
# ================================================================

def create_analytics_service(database_url: str = None) -> AnalyticsService:
    """Factory function to create analytics service"""
    return AnalyticsService(database_url)

# ================================================================
# Testing
# ================================================================

async def test_analytics_service():
    """Test the analytics service"""
    print("🧪 Testing Analytics Service...")
    
    analytics = create_analytics_service("sqlite:///test_analytics.db")
    
    try:
        # Test signal tracking
        test_signal = {
            "id": "test_signal_001",
            "symbol": "TESTSTOCK",
            "action": "BUY",
            "entry_price": 1000.0,
            "target_price": 1050.0,
            "stop_loss": 950.0,
            "confidence": 0.85
        }
        
        await analytics.track_signal_generated(test_signal)
        await analytics.track_telegram_sent(True, test_signal)
        
        # Get performance summary
        performance = analytics.get_performance_summary()
        print(f"✅ Signals Generated: {performance['daily']['signals_generated']}")
        print(f"✅ Signals Sent: {performance['daily']['signals_sent']}")
        print(f"✅ Telegram Success Rate: {performance['daily']['telegram_success_rate']:.1f}%")
        
        print("🎉 Analytics service test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await analytics.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_analytics_service())