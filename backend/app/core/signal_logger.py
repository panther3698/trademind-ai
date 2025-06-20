# backend/app/core/signal_logger.py
"""
Institutional-Grade Signal Logger for TradeMind AI
Captures every signal with complete market context and performance tracking.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import csv
import os

logger = logging.getLogger(__name__)

class TradeOutcome(Enum):
    PENDING = "PENDING"
    HIT_TARGET = "HIT_TARGET"
    HIT_STOP_LOSS = "HIT_STOP_LOSS"
    TIMED_EXIT = "TIMED_EXIT"
    MANUAL_EXIT = "MANUAL_EXIT"
    CANCELLED = "CANCELLED"

class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class TechnicalIndicators:
    """Technical indicators snapshot at signal generation"""
    rsi_14: float
    rsi_21: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    vwap: float
    adx: float
    atr_14: float
    stoch_k: float
    stoch_d: float
    cci: float
    williams_r: float
    momentum_10: float
    roc_10: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    volume_sma_20: float
    volume_ratio: float  # Current volume / 20-day average

@dataclass
class MarketContext:
    """Market context and macro factors"""
    nifty_price: float
    nifty_change_pct: float
    nifty_volume_ratio: float
    sector_vs_nifty: float  # Sector performance vs Nifty
    vix_level: float
    fii_flow: Optional[float] = None
    dii_flow: Optional[float] = None
    sgx_nifty: Optional[float] = None
    crude_oil: Optional[float] = None
    dollar_index: Optional[float] = None
    us_futures: Optional[float] = None

@dataclass
class PreMarketData:
    """Pre-market analysis data"""
    gap_percentage: float
    pre_market_volume: float
    news_sentiment_score: float  # -1 to +1
    news_count: int
    social_sentiment: Optional[float] = None
    analyst_rating_change: Optional[str] = None
    pre_open_price: Optional[float] = None

@dataclass
class SignalRecord:
    """Complete signal record with all context"""
    # Basic Signal Info
    signal_id: str
    timestamp: datetime
    ticker: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    target_price: float
    
    # AI/ML Scores
    ml_confidence: float  # 0-1 probability score
    technical_score: float  # -1 to +1
    sentiment_score: float  # -1 to +1
    macro_score: float  # -1 to +1
    final_score: float  # Combined weighted score
    
    # Technical Indicators
    indicators: TechnicalIndicators
    
    # Market Context
    market_context: MarketContext
    
    # Pre-market Data
    pre_market: PreMarketData
    
    # Risk Metrics
    risk_reward_ratio: float
    position_size_suggested: float
    capital_at_risk: float
    
    # Execution Tracking
    execution_lag_seconds: Optional[float] = None
    actual_entry_price: Optional[float] = None
    actual_exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    outcome: TradeOutcome = TradeOutcome.PENDING
    pnl_points: Optional[float] = None
    pnl_percentage: Optional[float] = None
    pnl_rupees: Optional[float] = None
    
    # Additional Metadata
    model_version: str = "v1.0"
    signal_source: str = "AI_MODEL"
    notes: Optional[str] = None

class InstitutionalSignalLogger:
    """
    Enterprise-grade signal logging system for quantitative trading.
    
    Features:
    - Comprehensive signal recording with full market context
    - Trade outcome tracking and P&L calculation
    - Daily performance summaries
    - Risk monitoring and alerts
    - Compliance audit trails
    """
    
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = Path(base_log_dir)
        self.setup_directories()
        
        # Active signals tracking
        self.active_signals: Dict[str, SignalRecord] = {}
        
        # Daily stats
        self.daily_stats = {
            "signals_generated": 0,
            "signals_executed": 0,
            "winners": 0,
            "losers": 0,
            "total_pnl": 0.0,
            "avg_execution_lag": 0.0
        }
        
    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.base_log_dir / "trades",
            self.base_log_dir / "daily_summaries", 
            self.base_log_dir / "compliance_logs",
            self.base_log_dir / "performance_analytics",
            self.base_log_dir / "risk_events"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"✅ Signal logging directories created at {self.base_log_dir}")
    
    def log_signal(self, signal: SignalRecord) -> bool:
        """
        Log a new trading signal with complete context
        
        Args:
            signal: Complete signal record with all market context
            
        Returns:
            bool: Success status
        """
        try:
            # Add to active tracking
            self.active_signals[signal.signal_id] = signal
            
            # Write to daily CSV
            self._write_to_daily_csv(signal)
            
            # Write detailed JSON for analysis
            self._write_detailed_json(signal)
            
            # Update daily stats
            self.daily_stats["signals_generated"] += 1
            
            # Log compliance event
            self._log_compliance_event("SIGNAL_GENERATED", {
                "signal_id": signal.signal_id,
                "ticker": signal.ticker,
                "confidence": signal.ml_confidence,
                "risk_amount": signal.capital_at_risk
            })
            
            logger.info(f"📊 Signal logged: {signal.ticker} {signal.direction.value} "
                       f"@ ₹{signal.entry_price} (Confidence: {signal.ml_confidence:.1%})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to log signal {signal.signal_id}: {e}")
            return False
    
    def update_signal_outcome(self, 
                            signal_id: str,
                            outcome: TradeOutcome,
                            exit_price: float,
                            exit_timestamp: datetime,
                            execution_lag: Optional[float] = None) -> bool:
        """
        Update signal with trade outcome and P&L calculation
        
        Args:
            signal_id: Unique signal identifier
            outcome: Trade outcome (HIT_TARGET, HIT_STOP_LOSS, etc.)
            exit_price: Actual exit price
            exit_timestamp: When trade was closed
            execution_lag: Seconds between signal and execution
            
        Returns:
            bool: Success status
        """
        try:
            if signal_id not in self.active_signals:
                logger.warning(f"⚠️ Signal {signal_id} not found in active signals")
                return False
                
            signal = self.active_signals[signal_id]
            
            # Update signal record
            signal.outcome = outcome
            signal.actual_exit_price = exit_price
            signal.exit_timestamp = exit_timestamp
            signal.execution_lag_seconds = execution_lag
            
            # Calculate P&L
            self._calculate_pnl(signal)
            
            # Write updated record
            self._write_to_daily_csv(signal, update=True)
            self._write_detailed_json(signal)
            
            # Update daily stats
            self._update_daily_stats(signal)
            
            # Log compliance event
            self._log_compliance_event("TRADE_CLOSED", {
                "signal_id": signal_id,
                "outcome": outcome.value,
                "pnl_percentage": signal.pnl_percentage,
                "pnl_rupees": signal.pnl_rupees
            })
            
            # Remove from active tracking if closed
            if outcome != TradeOutcome.PENDING:
                del self.active_signals[signal_id]
            
            logger.info(f"📈 Trade closed: {signal.ticker} {outcome.value} "
                       f"P&L: ₹{signal.pnl_rupees:.2f} ({signal.pnl_percentage:.2f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update signal outcome: {e}")
            return False
    
    def _calculate_pnl(self, signal: SignalRecord):
        """Calculate P&L metrics for completed trade"""
        entry_price = signal.actual_entry_price or signal.entry_price
        exit_price = signal.actual_exit_price
        
        if signal.direction == SignalDirection.BUY:
            pnl_points = exit_price - entry_price
        else:  # SELL
            pnl_points = entry_price - exit_price
            
        pnl_percentage = (pnl_points / entry_price) * 100
        pnl_rupees = pnl_points * signal.position_size_suggested
        
        signal.pnl_points = pnl_points
        signal.pnl_percentage = pnl_percentage
        signal.pnl_rupees = pnl_rupees
    
    def _write_to_daily_csv(self, signal: SignalRecord, update: bool = False):
        """Write signal to daily CSV file"""
        today = signal.timestamp.strftime("%Y-%m-%d")
        csv_path = self.base_log_dir / "trades" / f"{today}.csv"
        
        # Flatten signal data for CSV
        csv_data = self._flatten_signal_for_csv(signal)
        
        file_exists = csv_path.exists()
        mode = 'a' if file_exists and not update else 'w'
        
        with open(csv_path, mode, newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_data.keys())
            
            if not file_exists or mode == 'w':
                writer.writeheader()
                
            writer.writerow(csv_data)
    
    def _write_detailed_json(self, signal: SignalRecord):
        """Write detailed signal data as JSON for analysis"""
        today = signal.timestamp.strftime("%Y-%m-%d")
        json_dir = self.base_log_dir / "trades" / "detailed" / today
        json_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = json_dir / f"{signal.signal_id}.json"
        
        # Convert dataclass to dict, handling datetime serialization
        signal_dict = asdict(signal)
        signal_dict['timestamp'] = signal.timestamp.isoformat()
        if signal.exit_timestamp:
            signal_dict['exit_timestamp'] = signal.exit_timestamp.isoformat()
        
        with open(json_path, 'w') as f:
            json.dump(signal_dict, f, indent=2, default=str)
    
    def _flatten_signal_for_csv(self, signal: SignalRecord) -> Dict:
        """Flatten nested signal structure for CSV export"""
        flat_data = {
            # Basic info
            'signal_id': signal.signal_id,
            'timestamp': signal.timestamp.isoformat(),
            'ticker': signal.ticker,
            'direction': signal.direction.value,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'target_price': signal.target_price,
            
            # AI scores
            'ml_confidence': signal.ml_confidence,
            'technical_score': signal.technical_score,
            'sentiment_score': signal.sentiment_score,
            'macro_score': signal.macro_score,
            'final_score': signal.final_score,
            
            # Technical indicators (flattened)
            'rsi_14': signal.indicators.rsi_14,
            'rsi_21': signal.indicators.rsi_21,
            'macd_line': signal.indicators.macd_line,
            'macd_signal': signal.indicators.macd_signal,
            'vwap': signal.indicators.vwap,
            'adx': signal.indicators.adx,
            'atr_14': signal.indicators.atr_14,
            'bb_width': signal.indicators.bb_width,
            'volume_ratio': signal.indicators.volume_ratio,
            
            # Market context
            'nifty_price': signal.market_context.nifty_price,
            'nifty_change_pct': signal.market_context.nifty_change_pct,
            'vix_level': signal.market_context.vix_level,
            'sector_vs_nifty': signal.market_context.sector_vs_nifty,
            
            # Pre-market
            'gap_percentage': signal.pre_market.gap_percentage,
            'news_sentiment_score': signal.pre_market.news_sentiment_score,
            'news_count': signal.pre_market.news_count,
            
            # Risk metrics
            'risk_reward_ratio': signal.risk_reward_ratio,
            'position_size_suggested': signal.position_size_suggested,
            'capital_at_risk': signal.capital_at_risk,
            
            # Execution tracking
            'execution_lag_seconds': signal.execution_lag_seconds,
            'actual_entry_price': signal.actual_entry_price,
            'actual_exit_price': signal.actual_exit_price,
            'exit_timestamp': signal.exit_timestamp.isoformat() if signal.exit_timestamp else None,
            'outcome': signal.outcome.value,
            'pnl_points': signal.pnl_points,
            'pnl_percentage': signal.pnl_percentage,
            'pnl_rupees': signal.pnl_rupees,
            
            # Metadata
            'model_version': signal.model_version,
            'signal_source': signal.signal_source,
            'notes': signal.notes
        }
        
        return flat_data
    
    def _update_daily_stats(self, signal: SignalRecord):
        """Update daily performance statistics"""
        self.daily_stats["signals_executed"] += 1
        
        if signal.pnl_percentage and signal.pnl_percentage > 0:
            self.daily_stats["winners"] += 1
        elif signal.pnl_percentage and signal.pnl_percentage < 0:
            self.daily_stats["losers"] += 1
            
        if signal.pnl_rupees:
            self.daily_stats["total_pnl"] += signal.pnl_rupees
            
        if signal.execution_lag_seconds:
            # Update average execution lag
            current_avg = self.daily_stats["avg_execution_lag"]
            count = self.daily_stats["signals_executed"]
            new_avg = ((current_avg * (count - 1)) + signal.execution_lag_seconds) / count
            self.daily_stats["avg_execution_lag"] = new_avg
    
    def _log_compliance_event(self, event_type: str, data: Dict):
        """Log compliance and audit events"""
        today = datetime.now().strftime("%Y-%m-%d")
        compliance_path = self.base_log_dir / "compliance_logs" / f"{today}.json"
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        # Append to daily compliance log
        with open(compliance_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict:
        """
        Get comprehensive daily performance summary
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dict with daily performance metrics
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        csv_path = self.base_log_dir / "trades" / f"{date}.csv"
        
        if not csv_path.exists():
            return {"error": f"No data found for {date}"}
        
        # Load daily data
        df = pd.read_csv(csv_path)
        
        # Calculate metrics
        total_signals = len(df)
        completed_trades = len(df[df['outcome'] != 'PENDING'])
        
        if completed_trades > 0:
            winners = len(df[df['pnl_percentage'] > 0])
            losers = len(df[df['pnl_percentage'] < 0])
            win_rate = (winners / completed_trades) * 100
            
            total_pnl = df['pnl_rupees'].sum()
            avg_winner = df[df['pnl_percentage'] > 0]['pnl_percentage'].mean()
            avg_loser = df[df['pnl_percentage'] < 0]['pnl_percentage'].mean()
            
            avg_confidence = df['ml_confidence'].mean()
            avg_execution_lag = df['execution_lag_seconds'].mean()
        else:
            winners = losers = win_rate = total_pnl = 0
            avg_winner = avg_loser = avg_confidence = avg_execution_lag = 0
        
        summary = {
            "date": date,
            "total_signals": total_signals,
            "completed_trades": completed_trades,
            "pending_trades": total_signals - completed_trades,
            "winners": winners,
            "losers": losers,
            "win_rate_pct": round(win_rate, 2),
            "total_pnl_rupees": round(total_pnl, 2),
            "avg_winner_pct": round(avg_winner, 2) if avg_winner else 0,
            "avg_loser_pct": round(avg_loser, 2) if avg_loser else 0,
            "avg_confidence": round(avg_confidence, 3),
            "avg_execution_lag_sec": round(avg_execution_lag, 2),
            "risk_reward_analysis": self._analyze_risk_reward(df)
        }
        
        return summary
    
    def _analyze_risk_reward(self, df: pd.DataFrame) -> Dict:
        """Analyze risk-reward patterns"""
        if len(df) == 0:
            return {}
            
        return {
            "avg_risk_reward_ratio": round(df['risk_reward_ratio'].mean(), 2),
            "avg_capital_at_risk": round(df['capital_at_risk'].mean(), 2),
            "max_capital_at_risk": round(df['capital_at_risk'].max(), 2),
            "high_confidence_signals": len(df[df['ml_confidence'] > 0.7]),
            "medium_confidence_signals": len(df[(df['ml_confidence'] > 0.5) & (df['ml_confidence'] <= 0.7)]),
            "low_confidence_signals": len(df[df['ml_confidence'] <= 0.5])
        }
    
    def get_active_signals(self) -> List[Dict]:
        """Get all currently active signals"""
        return [asdict(signal) for signal in self.active_signals.values()]
    
    def check_risk_limits(self, new_signal: SignalRecord) -> Dict[str, Any]:
        """
        Check if new signal violates risk limits
        
        Returns:
            Dict with risk check results and recommendations
        """
        today = datetime.now().strftime("%Y-%m-%d")
        summary = self.get_daily_summary(today)
        
        # Current day metrics
        current_signals = summary.get("total_signals", 0)
        current_capital_at_risk = sum(s.capital_at_risk for s in self.active_signals.values())
        
        # Risk checks
        risk_checks = {
            "max_signals_per_day": {
                "current": current_signals,
                "limit": 5,  # Configurable
                "violated": current_signals >= 5
            },
            "max_capital_at_risk": {
                "current": current_capital_at_risk,
                "new_total": current_capital_at_risk + new_signal.capital_at_risk,
                "limit": 50000,  # ₹50,000 max exposure
                "violated": (current_capital_at_risk + new_signal.capital_at_risk) > 50000
            },
            "consecutive_losses": {
                "count": self._count_consecutive_losses(),
                "limit": 3,
                "violated": self._count_consecutive_losses() >= 3
            },
            "minimum_confidence": {
                "signal_confidence": new_signal.ml_confidence,
                "limit": 0.6,
                "violated": new_signal.ml_confidence < 0.6
            }
        }
        
        overall_risk_ok = not any(check["violated"] for check in risk_checks.values())
        
        return {
            "risk_checks": risk_checks,
            "overall_risk_ok": overall_risk_ok,
            "recommendation": "PROCEED" if overall_risk_ok else "BLOCK_SIGNAL"
        }
    
    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing trades"""
        today = datetime.now().strftime("%Y-%m-%d")
        csv_path = self.base_log_dir / "trades" / f"{today}.csv"
        
        if not csv_path.exists():
            return 0
            
        df = pd.read_csv(csv_path)
        completed = df[df['outcome'] != 'PENDING'].sort_values('timestamp')
        
        if len(completed) == 0:
            return 0
            
        consecutive_losses = 0
        for _, row in completed.iloc[::-1].iterrows():  # Reverse order (latest first)
            if row['pnl_percentage'] < 0:
                consecutive_losses += 1
            else:
                break
                
        return consecutive_losses


# Example usage and integration
def create_sample_signal() -> SignalRecord:
    """Create a sample signal for testing"""
    
    indicators = TechnicalIndicators(
        rsi_14=65.5,
        rsi_21=62.3,
        macd_line=0.45,
        macd_signal=0.38,
        macd_histogram=0.07,
        bb_upper=1250.5,
        bb_middle=1230.0,
        bb_lower=1209.5,
        bb_width=41.0,
        vwap=1235.7,
        adx=28.5,
        atr_14=25.8,
        stoch_k=78.2,
        stoch_d=75.1,
        cci=85.3,
        williams_r=-22.5,
        momentum_10=2.3,
        roc_10=1.8,
        sma_20=1228.5,
        sma_50=1215.3,
        ema_12=1238.2,
        ema_26=1225.1,
        volume_sma_20=1500000,
        volume_ratio=1.8
    )
    
    market_context = MarketContext(
        nifty_price=18750.25,
        nifty_change_pct=0.75,
        nifty_volume_ratio=1.2,
        sector_vs_nifty=1.5,
        vix_level=12.8,
        fii_flow=250.5,
        dii_flow=-180.2
    )
    
    pre_market = PreMarketData(
        gap_percentage=2.1,
        pre_market_volume=50000,
        news_sentiment_score=0.65,
        news_count=3
    )
    
    return SignalRecord(
        signal_id="TM_20241219_001",
        timestamp=datetime.now(),
        ticker="RELIANCE",
        direction=SignalDirection.BUY,
        entry_price=1240.0,
        stop_loss=1220.0,
        target_price=1280.0,
        ml_confidence=0.78,
        technical_score=0.65,
        sentiment_score=0.72,
        macro_score=0.45,
        final_score=0.68,
        indicators=indicators,
        market_context=market_context,
        pre_market=pre_market,
        risk_reward_ratio=2.0,
        position_size_suggested=100,
        capital_at_risk=2000.0,
        model_version="v1.0",
        signal_source="AI_MODEL"
    )


if __name__ == "__main__":
    # Test the logger
    logger = InstitutionalSignalLogger("logs")
    
    # Create and log a sample signal
    sample_signal = create_sample_signal()
    success = logger.log_signal(sample_signal)
    
    if success:
        print("✅ Signal logged successfully")
        
        # Simulate trade completion after 30 minutes
        import time
        time.sleep(2)  # Simulate some time passing
        
        logger.update_signal_outcome(
            sample_signal.signal_id,
            TradeOutcome.HIT_TARGET,
            1275.0,  # Exit price
            datetime.now(),
            45.0  # 45 seconds execution lag
        )
        
        # Get daily summary
        summary = logger.get_daily_summary()
        print("\n📊 Daily Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        print("❌ Failed to log signal")