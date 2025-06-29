from datetime import datetime, time as dt_time
from typing import Dict, Any, Optional

class CorrectedAnalytics:
    """
    Standalone analytics implementation with regime tracking + news intelligence
    Tracks daily statistics, performance metrics, and news intelligence integration.
    Provides methods to record and summarize analytics for signals, trading, Telegram, and news events.
    Used by the enhanced service manager to monitor system health, trading performance, and news-driven features.
    """
    
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
            "market_mode": "closed",
            "current_regime": "UNKNOWN",
            "regime_confidence": 0.0,
            "regime_changes": 0,
            "backtest_runs": 0,
            "last_backtest": None,
            # Interactive trading stats
            "signals_approved": 0,
            "signals_rejected": 0,
            "orders_executed": 0,
            "order_success_rate": 0.0,
            "total_trading_pnl": 0.0,
            "approval_rate": 0.0,
            # NEWS INTELLIGENCE STATS
            "news_articles_processed": 0,
            "breaking_news_alerts": 0,
            "news_signals_generated": 0,
            "avg_news_sentiment": 0.0,
            "last_news_update": None,
            "news_sources_active": 0,
            # FIXED: NEWS-SIGNAL INTEGRATION STATS
            "news_triggered_signals": 0,
            "enhanced_ml_signals": 0,
            "news_integration_active": False
        }
        self.start_time = datetime.now()
        self.premarket_opportunities = []
        self.priority_signals_today = []
        
    async def track_news_processed(self, articles_count: int, sentiment_avg: float, sources_count: int):
        """Track news intelligence processing"""
        self.daily_stats["news_articles_processed"] += articles_count
        self.daily_stats["avg_news_sentiment"] = sentiment_avg
        self.daily_stats["news_sources_active"] = sources_count
        self.daily_stats["last_news_update"] = datetime.now().isoformat()
        
    async def track_breaking_news(self):
        """Track breaking news alert"""
        self.daily_stats["breaking_news_alerts"] += 1
        
    async def track_news_signal(self):
        """Track news-driven signal"""
        self.daily_stats["news_signals_generated"] += 1
        
    async def track_news_triggered_signal(self):
        """Track news-triggered signal"""
        self.daily_stats["news_triggered_signals"] += 1
        
    async def track_enhanced_ml_signal(self):
        """Track news-enhanced ML signal"""
        self.daily_stats["enhanced_ml_signals"] += 1
        
    async def track_signal_generated(self, signal: Optional[Dict]):
        """Track signal generation"""
        self.daily_stats["signals_generated"] += 1
        self.daily_stats["last_signal_time"] = datetime.now().isoformat()
        
        # Track priority signals
        current_time = datetime.now().time()
        if dt_time(9, 15) <= current_time <= dt_time(9, 45):
            self.daily_stats["priority_signals"] += 1
            self.priority_signals_today.append(signal)
        
        # Update average confidence
        current_avg = self.daily_stats["average_confidence"]
        count = self.daily_stats["signals_generated"]
        new_confidence = signal.get("confidence", 0.0) if signal else 0.0
        
        if count > 0:
            self.daily_stats["average_confidence"] = (
                (current_avg * (count - 1) + new_confidence) / count
            )
    
    async def track_signal_approval(self, approved: bool):
        """Track signal approval/rejection"""
        if approved:
            self.daily_stats["signals_approved"] += 1
        else:
            self.daily_stats["signals_rejected"] += 1
            
        # Update approval rate
        total_responses = self.daily_stats["signals_approved"] + self.daily_stats["signals_rejected"]
        if total_responses > 0:
            self.daily_stats["approval_rate"] = (
                self.daily_stats["signals_approved"] / total_responses * 100
            )
    
    async def track_order_execution(self, success: bool, pnl: float = 0.0):
        """Track order execution results"""
        if success:
            self.daily_stats["orders_executed"] += 1
            self.daily_stats["total_trading_pnl"] += pnl
            
        # Update success rate
        total_orders = self.daily_stats["orders_executed"]
        if total_orders > 0:
            self.daily_stats["order_success_rate"] = (
                self.daily_stats["orders_executed"] / 
                (self.daily_stats["signals_approved"] or 1) * 100
            )
    
    async def track_premarket_analysis(self, opportunities_count):
        """Track pre-market analysis"""
        self.daily_stats["premarket_analyses"] += 1
        self.daily_stats["last_premarket_analysis"] = datetime.now().isoformat()
        
    async def track_telegram_sent(self, success: bool, signal: Optional[Dict] = None):
        """Track Telegram message success/failure"""
        if success:
            self.daily_stats["telegram_success"] += 1
            self.daily_stats["signals_sent"] += 1
        else:
            self.daily_stats["telegram_failures"] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        # Calculate uptime
        uptime_delta = datetime.now() - self.start_time
        self.daily_stats["system_uptime_hours"] = uptime_delta.total_seconds() / 3600
        
        return {
            "daily": self.daily_stats.copy(),
            "system": {
                "start_time": self.start_time.isoformat(),
                "uptime_hours": self.daily_stats["system_uptime_hours"],
                "is_operational": self.daily_stats["status"] == "operational"
            },
            "enhanced_features": {
                "interactive_trading": False,  # Set by main.py
                "order_execution": False,
                "webhook_handler": False,
                "regime_detection": False,
                "backtesting": False,
                "news_intelligence": False,
                "news_signal_integration": self.daily_stats["news_integration_active"]
            }
        }
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get daily statistics"""
        return self.daily_stats.copy() 