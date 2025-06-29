# ================================================================
# Analytics Domain Models
# ================================================================

"""
Analytics domain models for TradeMind AI

This module defines the core analytics-related data structures
used throughout the application.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# ================================================================
# MODELS
# ================================================================

class TradeOutcome(BaseModel):
    """Trade outcome model"""
    
    order_id: str = Field(..., description="Order ID")
    symbol: str = Field(..., description="Stock symbol")
    entry_price: float = Field(..., gt=0, description="Entry price")
    exit_price: float = Field(..., gt=0, description="Exit price")
    quantity: int = Field(..., gt=0, description="Quantity traded")
    pnl: float = Field(..., description="Profit/Loss")
    status: str = Field(..., description="Trade status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Trade timestamp")

class DailyStats(BaseModel):
    """Daily statistics model"""
    
    signals_generated: int = Field(default=0, ge=0, description="Number of signals generated")
    signals_executed: int = Field(default=0, ge=0, description="Number of signals executed")
    successful_trades: int = Field(default=0, ge=0, description="Number of successful trades")
    total_pnl: float = Field(default=0.0, description="Total profit/loss")
    win_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Win rate")
    news_articles_processed: int = Field(default=0, ge=0, description="News articles processed")
    breaking_news_alerts: int = Field(default=0, ge=0, description="Breaking news alerts")

class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    
    total_trades: int = Field(default=0, ge=0, description="Total number of trades")
    winning_trades: int = Field(default=0, ge=0, description="Number of winning trades")
    losing_trades: int = Field(default=0, ge=0, description="Number of losing trades")
    total_pnl: float = Field(default=0.0, description="Total profit/loss")
    average_win: float = Field(default=0.0, description="Average winning trade")
    average_loss: float = Field(default=0.0, description="Average losing trade")
    max_drawdown: float = Field(default=0.0, description="Maximum drawdown")
    sharpe_ratio: float = Field(default=0.0, description="Sharpe ratio")

class NewsAnalytics(BaseModel):
    """News analytics model"""
    
    articles_processed: int = Field(default=0, ge=0, description="Articles processed")
    average_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0, description="Average sentiment")
    breaking_news_alerts: int = Field(default=0, ge=0, description="Breaking news alerts")
    sentiment_distribution: Dict[str, int] = Field(default_factory=dict, description="Sentiment distribution")
    top_sources: List[str] = Field(default_factory=list, description="Top news sources")

class AnalyticsSummary(BaseModel):
    """Analytics summary model"""
    
    daily_stats: DailyStats = Field(..., description="Daily statistics")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics")
    news_analytics: NewsAnalytics = Field(..., description="News analytics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Summary timestamp")

# ================================================================
# VALIDATION
# ================================================================

def validate_trade_data(data: Dict[str, Any]) -> bool:
    """Validate trade data"""
    required_fields = ["order_id", "symbol", "entry_price", "exit_price", "quantity", "pnl", "status"]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required trade data: {field}")
    
    if data["entry_price"] <= 0 or data["exit_price"] <= 0:
        raise ValueError("Prices must be positive")
    
    if data["quantity"] <= 0:
        raise ValueError("Quantity must be positive")
    
    return True

def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """Calculate win rate from trades"""
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
    return winning_trades / len(trades)

def calculate_total_pnl(trades: List[Dict[str, Any]]) -> float:
    """Calculate total PnL from trades"""
    return sum(trade.get("pnl", 0) for trade in trades) 