# backend/app/services/backtest_engine.py
"""
TradeMind AI - Professional Backtesting Engine
Simulates historical 1-minute OHLCV data with realistic slippage and execution logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Import our services
from app.core.signal_logger import SignalRecord, TradeOutcome, SignalDirection
from app.services.enhanced_market_data_nifty100 import EnhancedMarketDataService as MarketDataService

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Individual backtest trade result"""
    signal_id: str
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    target_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str  # "STOP_LOSS", "TARGET", "TIME_EXIT", "SLIPPAGE_FAIL"
    pnl_points: float
    pnl_percentage: float
    pnl_rupees: float
    confidence: float
    execution_lag_seconds: int
    slippage_points: float
    
@dataclass
class BacktestSummary:
    """Comprehensive backtest performance summary"""
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_pnl_rupees: float
    total_pnl_percentage: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    profit_factor: float  # Gross profit / Gross loss
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    avg_trade_duration_minutes: int
    avg_execution_lag_seconds: float
    avg_slippage_points: float
    risk_reward_ratio: float
    calmar_ratio: float  # Annual return / Max drawdown
    
    # Confidence-based metrics
    high_confidence_trades: int  # >0.8 confidence
    high_confidence_win_rate: float
    low_confidence_trades: int   # <0.6 confidence
    low_confidence_win_rate: float

class BacktestEngine:
    """
    Professional backtesting engine with realistic simulation
    
    Features:
    - 1-minute OHLCV historical data simulation
    - Realistic slippage modeling
    - Proper stop loss/target execution
    - Comprehensive performance metrics
    - Confidence-based analysis
    """
    
    def __init__(self, market_data_service: MarketDataService):
        self.market_data = market_data_service
        
        # Backtest parameters
        self.slippage_bps = 2  # 2 basis points (0.02%) slippage
        self.execution_delay_seconds = 60  # 1-minute execution delay
        self.max_trade_duration_days = 5  # Auto-exit after 5 days
        self.commission_per_trade = 20  # ₹20 commission per trade
        
        # Results storage
        self.backtest_results = []
        self.trades_log = []
        
    async def run_backtest(self, 
                          signals: List[SignalRecord],
                          start_date: datetime,
                          end_date: datetime) -> BacktestSummary:
        """
        Run comprehensive backtest on historical signals
        
        Args:
            signals: List of historical signals to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestSummary with all performance metrics
        """
        try:
            logger.info(f"🔄 Starting backtest: {len(signals)} signals from {start_date.date()} to {end_date.date()}")
            
            # Reset results
            self.backtest_results = []
            self.trades_log = []
            
            # Process each signal
            for i, signal in enumerate(signals):
                try:
                    trade_result = await self._simulate_trade(signal)
                    if trade_result:
                        self.backtest_results.append(trade_result)
                        
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(signals)} signals...")
                        
                except Exception as e:
                    logger.error(f"Error simulating trade for signal {signal.signal_id}: {e}")
                    continue
            
            # Calculate comprehensive summary
            summary = self._calculate_backtest_summary(start_date, end_date)
            
            # Save results
            await self._save_backtest_results(summary)
            
            logger.info(f"✅ Backtest complete: {summary.total_trades} trades, "
                       f"{summary.win_rate_pct:.1f}% win rate, "
                       f"₹{summary.total_pnl_rupees:.0f} total P&L")
            
            return summary
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return BacktestSummary(
                start_date=start_date, end_date=end_date, total_trades=0,
                winning_trades=0, losing_trades=0, win_rate_pct=0.0,
                total_pnl_rupees=0.0, total_pnl_percentage=0.0,
                avg_win_pct=0.0, avg_loss_pct=0.0, largest_win_pct=0.0, largest_loss_pct=0.0,
                profit_factor=0.0, sharpe_ratio=0.0, max_drawdown_pct=0.0,
                max_drawdown_duration_days=0, avg_trade_duration_minutes=0,
                avg_execution_lag_seconds=0.0, avg_slippage_points=0.0,
                risk_reward_ratio=0.0, calmar_ratio=0.0,
                high_confidence_trades=0, high_confidence_win_rate=0.0,
                low_confidence_trades=0, low_confidence_win_rate=0.0
            )
    
    async def _simulate_trade(self, signal: SignalRecord) -> Optional[BacktestTrade]:
        """
        Simulate individual trade with realistic execution
        
        Args:
            signal: Signal to simulate
            
        Returns:
            BacktestTrade result or None if failed
        """
        try:
            # Get historical data for the signal period
            start_date = signal.timestamp
            end_date = start_date + timedelta(days=self.max_trade_duration_days)
            
            historical_data = await self.market_data.zerodha.get_historical_data(
                signal.ticker, "minute", start_date, end_date
            )
            
            if len(historical_data) < 10:  # Need sufficient data
                return None
                
            # Convert to DataFrame for easier processing
            df = pd.DataFrame([{
                'timestamp': h.timestamp,
                'open': h.open,
                'high': h.high,
                'low': h.low,
                'close': h.close,
                'volume': h.volume
            } for h in historical_data])
            
            df = df.sort_values('timestamp')
            
            # Find entry point (with execution delay)
            signal_time = signal.timestamp
            entry_time = signal_time + timedelta(seconds=self.execution_delay_seconds)
            
            # Find the first bar after entry time
            entry_bars = df[df['timestamp'] >= entry_time]
            if len(entry_bars) == 0:
                return None
                
            entry_bar = entry_bars.iloc[0]
            
            # Calculate slippage
            slippage_points = self._calculate_slippage(signal.entry_price, entry_bar, signal.direction)
            actual_entry_price = signal.entry_price + slippage_points
            
            # Check if entry is possible (within the bar range)
            if signal.direction == SignalDirection.BUY:
                if actual_entry_price > entry_bar['high']:
                    # Cannot execute - price gapped up beyond reach
                    return BacktestTrade(
                        signal_id=signal.signal_id,
                        symbol=signal.ticker,
                        direction=signal.direction.value,
                        entry_time=entry_time,
                        entry_price=actual_entry_price,
                        stop_loss=signal.stop_loss,
                        target_price=signal.target_price,
                        exit_time=entry_time,
                        exit_price=actual_entry_price,
                        exit_reason="SLIPPAGE_FAIL",
                        pnl_points=0.0,
                        pnl_percentage=0.0,
                        pnl_rupees=-self.commission_per_trade,
                        confidence=signal.ml_confidence,
                        execution_lag_seconds=self.execution_delay_seconds,
                        slippage_points=slippage_points
                    )
            else:  # SELL
                if actual_entry_price < entry_bar['low']:
                    # Cannot execute - price gapped down beyond reach
                    return BacktestTrade(
                        signal_id=signal.signal_id,
                        symbol=signal.ticker,
                        direction=signal.direction.value,
                        entry_time=entry_time,
                        entry_price=actual_entry_price,
                        stop_loss=signal.stop_loss,
                        target_price=signal.target_price,
                        exit_time=entry_time,
                        exit_price=actual_entry_price,
                        exit_reason="SLIPPAGE_FAIL",
                        pnl_points=0.0,
                        pnl_percentage=0.0,
                        pnl_rupees=-self.commission_per_trade,
                        confidence=signal.ml_confidence,
                        execution_lag_seconds=self.execution_delay_seconds,
                        slippage_points=slippage_points
                    )
            
            # Simulate trade execution from entry point
            remaining_bars = df[df['timestamp'] > entry_time]
            
            for _, bar in remaining_bars.iterrows():
                # Check for stop loss or target hit
                exit_result = self._check_exit_conditions(
                    bar, actual_entry_price, signal.stop_loss, 
                    signal.target_price, signal.direction
                )
                
                if exit_result:
                    exit_price, exit_reason = exit_result
                    exit_time = bar['timestamp']
                    
                    # Calculate P&L
                    pnl_points, pnl_percentage, pnl_rupees = self._calculate_pnl(
                        actual_entry_price, exit_price, signal.direction, 
                        signal.position_size_suggested
                    )
                    
                    return BacktestTrade(
                        signal_id=signal.signal_id,
                        symbol=signal.ticker,
                        direction=signal.direction.value,
                        entry_time=entry_time,
                        entry_price=actual_entry_price,
                        stop_loss=signal.stop_loss,
                        target_price=signal.target_price,
                        exit_time=exit_time,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        pnl_points=pnl_points,
                        pnl_percentage=pnl_percentage,
                        pnl_rupees=pnl_rupees,
                        confidence=signal.ml_confidence,
                        execution_lag_seconds=self.execution_delay_seconds,
                        slippage_points=slippage_points
                    )
            
            # If we reach here, trade timed out
            final_bar = remaining_bars.iloc[-1] if len(remaining_bars) > 0 else entry_bar
            exit_price = final_bar['close']
            
            pnl_points, pnl_percentage, pnl_rupees = self._calculate_pnl(
                actual_entry_price, exit_price, signal.direction, 
                signal.position_size_suggested
            )
            
            return BacktestTrade(
                signal_id=signal.signal_id,
                symbol=signal.ticker,
                direction=signal.direction.value,
                entry_time=entry_time,
                entry_price=actual_entry_price,
                stop_loss=signal.stop_loss,
                target_price=signal.target_price,
                exit_time=final_bar['timestamp'],
                exit_price=exit_price,
                exit_reason="TIME_EXIT",
                pnl_points=pnl_points,
                pnl_percentage=pnl_percentage,
                pnl_rupees=pnl_rupees,
                confidence=signal.ml_confidence,
                execution_lag_seconds=self.execution_delay_seconds,
                slippage_points=slippage_points
            )
            
        except Exception as e:
            logger.error(f"Trade simulation failed for {signal.signal_id}: {e}")
            return None
    
    def _calculate_slippage(self, 
                           expected_price: float, 
                           entry_bar: pd.Series, 
                           direction: SignalDirection) -> float:
        """Calculate realistic slippage based on market conditions"""
        try:
            # Base slippage
            base_slippage = expected_price * (self.slippage_bps / 10000)
            
            # Adjust for volatility (bar range)
            bar_range = (entry_bar['high'] - entry_bar['low']) / expected_price
            volatility_multiplier = min(3.0, max(0.5, bar_range * 50))  # Scale volatility
            
            # Adjust for volume (mock - in real system use actual volume data)
            volume_multiplier = 1.0  # Assume normal volume
            
            total_slippage = base_slippage * volatility_multiplier * volume_multiplier
            
            # Apply slippage direction
            if direction == SignalDirection.BUY:
                return total_slippage  # Pay more for buying
            else:
                return -total_slippage  # Receive less for selling
                
        except Exception as e:
            logger.error(f"Slippage calculation failed: {e}")
            return 0.0
    
    def _check_exit_conditions(self, 
                              bar: pd.Series,
                              entry_price: float,
                              stop_loss: float,
                              target_price: float,
                              direction: SignalDirection) -> Optional[Tuple[float, str]]:
        """Check if stop loss or target is hit in this bar"""
        try:
            if direction == SignalDirection.BUY:
                # Check stop loss (sell at stop)
                if bar['low'] <= stop_loss:
                    return stop_loss, "STOP_LOSS"
                
                # Check target (sell at target)
                if bar['high'] >= target_price:
                    return target_price, "TARGET"
                    
            else:  # SELL
                # Check stop loss (buy back at stop)
                if bar['high'] >= stop_loss:
                    return stop_loss, "STOP_LOSS"
                
                # Check target (buy back at target)
                if bar['low'] <= target_price:
                    return target_price, "TARGET"
            
            return None
            
        except Exception as e:
            logger.error(f"Exit condition check failed: {e}")
            return None
    
    def _calculate_pnl(self, 
                      entry_price: float,
                      exit_price: float,
                      direction: SignalDirection,
                      position_size: float) -> Tuple[float, float, float]:
        """Calculate P&L metrics"""
        try:
            if direction == SignalDirection.BUY:
                pnl_points = exit_price - entry_price
            else:
                pnl_points = entry_price - exit_price
            
            pnl_percentage = (pnl_points / entry_price) * 100
            pnl_rupees = (pnl_points * position_size) - self.commission_per_trade
            
            return pnl_points, pnl_percentage, pnl_rupees
            
        except Exception as e:
            logger.error(f"P&L calculation failed: {e}")
            return 0.0, 0.0, -self.commission_per_trade
    
    def _calculate_backtest_summary(self, start_date: datetime, end_date: datetime) -> BacktestSummary:
        """Calculate comprehensive backtest performance metrics"""
        try:
            if not self.backtest_results:
                return BacktestSummary(
                    start_date=start_date, end_date=end_date, total_trades=0,
                    winning_trades=0, losing_trades=0, win_rate_pct=0.0,
                    total_pnl_rupees=0.0, total_pnl_percentage=0.0,
                    avg_win_pct=0.0, avg_loss_pct=0.0, largest_win_pct=0.0, largest_loss_pct=0.0,
                    profit_factor=0.0, sharpe_ratio=0.0, max_drawdown_pct=0.0,
                    max_drawdown_duration_days=0, avg_trade_duration_minutes=0,
                    avg_execution_lag_seconds=0.0, avg_slippage_points=0.0,
                    risk_reward_ratio=0.0, calmar_ratio=0.0,
                    high_confidence_trades=0, high_confidence_win_rate=0.0,
                    low_confidence_trades=0, low_confidence_win_rate=0.0
                )
            
            trades_df = pd.DataFrame([asdict(trade) for trade in self.backtest_results])
            
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl_percentage'] > 0])
            losing_trades = len(trades_df[trades_df['pnl_percentage'] < 0])
            win_rate_pct = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl_rupees = trades_df['pnl_rupees'].sum()
            total_pnl_percentage = trades_df['pnl_percentage'].mean()
            
            winners = trades_df[trades_df['pnl_percentage'] > 0]
            losers = trades_df[trades_df['pnl_percentage'] < 0]
            
            avg_win_pct = winners['pnl_percentage'].mean() if len(winners) > 0 else 0
            avg_loss_pct = losers['pnl_percentage'].mean() if len(losers) > 0 else 0
            largest_win_pct = winners['pnl_percentage'].max() if len(winners) > 0 else 0
            largest_loss_pct = losers['pnl_percentage'].min() if len(losers) > 0 else 0
            
            # Profit factor
            gross_profit = winners['pnl_rupees'].sum() if len(winners) > 0 else 0
            gross_loss = abs(losers['pnl_rupees'].sum()) if len(losers) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Sharpe ratio (simplified)
            daily_returns = trades_df['pnl_percentage']
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            
            # Maximum drawdown
            cumulative_pnl = trades_df['pnl_percentage'].cumsum()
            rolling_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - rolling_max)
            max_drawdown_pct = abs(drawdown.min())
            
            # Drawdown duration (simplified)
            max_drawdown_duration_days = 0  # Would need more complex calculation
            
            # Trade duration
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df['duration_minutes'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
            avg_trade_duration_minutes = trades_df['duration_minutes'].mean()
            
            # Execution metrics
            avg_execution_lag_seconds = trades_df['execution_lag_seconds'].mean()
            avg_slippage_points = trades_df['slippage_points'].mean()
            
            # Risk-reward ratio
            if avg_loss_pct < 0:
                risk_reward_ratio = avg_win_pct / abs(avg_loss_pct)
            else:
                risk_reward_ratio = 0
            
            # Calmar ratio
            annual_return = (total_pnl_percentage / len(trades_df)) * 252 if len(trades_df) > 0 else 0
            calmar_ratio = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
            
            # Confidence-based metrics
            high_confidence = trades_df[trades_df['confidence'] > 0.8]
            low_confidence = trades_df[trades_df['confidence'] < 0.6]
            
            high_confidence_trades = len(high_confidence)
            high_confidence_win_rate = (len(high_confidence[high_confidence['pnl_percentage'] > 0]) / len(high_confidence) * 100) if len(high_confidence) > 0 else 0
            
            low_confidence_trades = len(low_confidence)
            low_confidence_win_rate = (len(low_confidence[low_confidence['pnl_percentage'] > 0]) / len(low_confidence) * 100) if len(low_confidence) > 0 else 0
            
            return BacktestSummary(
                start_date=start_date,
                end_date=end_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate_pct=win_rate_pct,
                total_pnl_rupees=total_pnl_rupees,
                total_pnl_percentage=total_pnl_percentage,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                largest_win_pct=largest_win_pct,
                largest_loss_pct=largest_loss_pct,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_pct=max_drawdown_pct,
                max_drawdown_duration_days=max_drawdown_duration_days,
                avg_trade_duration_minutes=avg_trade_duration_minutes,
                avg_execution_lag_seconds=avg_execution_lag_seconds,
                avg_slippage_points=avg_slippage_points,
                risk_reward_ratio=risk_reward_ratio,
                calmar_ratio=calmar_ratio,
                high_confidence_trades=high_confidence_trades,
                high_confidence_win_rate=high_confidence_win_rate,
                low_confidence_trades=low_confidence_trades,
                low_confidence_win_rate=low_confidence_win_rate
            )
            
        except Exception as e:
            logger.error(f"Backtest summary calculation failed: {e}")
            return BacktestSummary(
                start_date=start_date, end_date=end_date, total_trades=0,
                winning_trades=0, losing_trades=0, win_rate_pct=0.0,
                total_pnl_rupees=0.0, total_pnl_percentage=0.0,
                avg_win_pct=0.0, avg_loss_pct=0.0, largest_win_pct=0.0, largest_loss_pct=0.0,
                profit_factor=0.0, sharpe_ratio=0.0, max_drawdown_pct=0.0,
                max_drawdown_duration_days=0, avg_trade_duration_minutes=0,
                avg_execution_lag_seconds=0.0, avg_slippage_points=0.0,
                risk_reward_ratio=0.0, calmar_ratio=0.0,
                high_confidence_trades=0, high_confidence_win_rate=0.0,
                low_confidence_trades=0, low_confidence_win_rate=0.0
            )
    
    async def _save_backtest_results(self, summary: BacktestSummary):
        """Save backtest results to files"""
        try:
            # Create backtest results directory
            results_dir = Path("logs/backtest_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed trades
            trades_file = results_dir / f"backtest_trades_{timestamp}.json"
            with open(trades_file, 'w') as f:
                trades_data = [asdict(trade) for trade in self.backtest_results]
                json.dump(trades_data, f, indent=2, default=str)
            
            # Save summary
            summary_file = results_dir / f"backtest_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(asdict(summary), f, indent=2, default=str)
            
            logger.info(f"✅ Backtest results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")

# ================================================================
# backend/app/services/regime_detector.py
# Market Regime Detection and Engine Switching
# ================================================================

class MarketRegime:
    """Market regime classification - simplified"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"

@dataclass
class RegimeAnalysis:
    """Market regime analysis result"""
    regime: str
    confidence: float
    nifty_trend_strength: float
    sector_dispersion: float
    volatility_level: float
    volume_profile: str
    trading_strategy: str  # "MOMENTUM", "MEAN_REVERSION", "BREAKOUT"
    risk_adjustment: float  # Multiplier for position sizing

class RegimeDetector:
    """
    Market regime detection system that runs at 9:30 AM
    Classifies market conditions and switches trading strategies
    """
    
    def __init__(self, market_data_service: MarketDataService):
        self.market_data = market_data_service
        
        # Regime detection parameters
        self.trend_threshold = 0.5  # % movement to confirm trend
        self.volatility_threshold = 15.0  # VIX equivalent
        self.sector_dispersion_threshold = 2.0  # Sector spread
        
        # Current regime state
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.last_regime_update = None
        
    async def detect_market_regime(self) -> RegimeAnalysis:
        """
        Detect current market regime at 9:30 AM
        
        Returns:
            RegimeAnalysis with regime classification and strategy
        """
        try:
            logger.info("🔍 Detecting market regime...")
            
            # Get Nifty data for trend analysis
            nifty_analysis = await self._analyze_nifty_trend()
            
            # Get sector dispersion
            sector_analysis = await self._analyze_sector_dispersion()
            
            # Get volatility measures
            volatility_analysis = await self._analyze_market_volatility()
            
            # Get volume profile
            volume_analysis = await self._analyze_volume_profile()
            
            # Classify regime
            regime_result = self._classify_regime(
                nifty_analysis, sector_analysis, volatility_analysis, volume_analysis
            )
            
            # Update current state
            self.current_regime = regime_result.regime
            self.regime_confidence = regime_result.confidence
            self.last_regime_update = datetime.now()
            
            logger.info(f"📊 Market Regime: {regime_result.regime} "
                       f"(Confidence: {regime_result.confidence:.1%}) "
                       f"Strategy: {regime_result.trading_strategy}")
            
            return regime_result
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return RegimeAnalysis(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                nifty_trend_strength=0.0,
                sector_dispersion=1.0,
                volatility_level=15.0,
                volume_profile="NORMAL",
                trading_strategy="MEAN_REVERSION",
                risk_adjustment=1.0
            )
    
    async def _analyze_nifty_trend(self) -> Dict[str, float]:
        """Analyze Nifty trend strength and direction"""
        try:
            # Get Nifty data
            nifty_data = await self.market_data.get_live_market_data("NIFTY50")
            if not nifty_data or not nifty_data.get("quote"):
                return {"trend_strength": 0.0, "direction": 0.0, "gap_pct": 0.0}
            
            quote = nifty_data["quote"]
            
            # Calculate gap from previous close
            gap_pct = ((quote["ltp"] - quote["prev_close"]) / quote["prev_close"]) * 100
            
            # Get first 15 minutes movement (mock for now)
            # In production, collect actual 15-minute data
            current_time = datetime.now()
            market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            
            if current_time >= market_open + timedelta(minutes=15):
                # Mock 15-minute movement calculation
                movement_15min = gap_pct * 0.6  # Assume some movement from gap
            else:
                movement_15min = gap_pct
            
            # Calculate trend strength
            trend_strength = abs(movement_15min)
            direction = 1.0 if movement_15min > 0 else -1.0
            
            return {
                "trend_strength": trend_strength,
                "direction": direction,
                "gap_pct": gap_pct,
                "movement_15min": movement_15min
            }
            
        except Exception as e:
            logger.error(f"Nifty trend analysis failed: {e}")
            return {"trend_strength": 0.0, "direction": 0.0, "gap_pct": 0.0}
    
    async def _analyze_sector_dispersion(self) -> Dict[str, float]:
        """Analyze sector performance dispersion"""
        try:
            # Get sector leaders for analysis
            sector_stocks = {
                "BANKING": ["HDFCBANK", "ICICIBANK", "KOTAKBANK"],
                "IT": ["TCS", "INFY", "HCLTECH"],
                "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA"],
                "AUTO": ["MARUTI", "TATAMOTORS", "M&M"],
                "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND"]
            }
            
            sector_performances = {}
            
            for sector, stocks in sector_stocks.items():
                sector_performance = []
                
                for stock in stocks:
                    try:
                        stock_data = await self.market_data.get_live_market_data(stock)
                        if stock_data and stock_data.get("quote"):
                            quote = stock_data["quote"]
                            change_pct = ((quote["ltp"] - quote["prev_close"]) / quote["prev_close"]) * 100
                            sector_performance.append(change_pct)
                            
                        # Rate limiting
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.debug(f"Failed to get data for {stock}: {e}")
                        continue
                
                if sector_performance:
                    sector_performances[sector] = np.mean(sector_performance)
            
            # Calculate dispersion
            if len(sector_performances) > 0:
                performances = list(sector_performances.values())
                sector_dispersion = np.std(performances)
                sector_range = max(performances) - min(performances)
            else:
                sector_dispersion = 1.0
                sector_range = 2.0
            
            return {
                "sector_dispersion": sector_dispersion,
                "sector_range": sector_range,
                "sector_performances": sector_performances
            }
            
        except Exception as e:
            logger.error(f"Sector dispersion analysis failed: {e}")
            return {"sector_dispersion": 1.0, "sector_range": 2.0, "sector_performances": {}}
    
    async def _analyze_market_volatility(self) -> Dict[str, float]:
        """Analyze market volatility levels"""
        try:
            # In production, get actual VIX data
            # For now, estimate volatility from Nifty movement
            
            nifty_data = await self.market_data.get_live_market_data("NIFTY50")
            if not nifty_data or not nifty_data.get("quote"):
                return {"volatility_level": 15.0, "volatility_regime": "NORMAL"}
            
            quote = nifty_data["quote"]
            
            # Calculate intraday range as volatility proxy
            if quote.get("high") and quote.get("low"):
                intraday_range_pct = ((quote["high"] - quote["low"]) / quote["ltp"]) * 100
                
                # Estimate VIX equivalent
                estimated_vix = intraday_range_pct * 3  # Rough conversion
            else:
                estimated_vix = 15.0
            
            # Classify volatility regime
            if estimated_vix > 25:
                volatility_regime = "HIGH"
            elif estimated_vix < 12:
                volatility_regime = "LOW"
            else:
                volatility_regime = "NORMAL"
            
            return {
                "volatility_level": estimated_vix,
                "volatility_regime": volatility_regime,
                "intraday_range_pct": intraday_range_pct if 'intraday_range_pct' in locals() else 1.0
            }
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {"volatility_level": 15.0, "volatility_regime": "NORMAL"}
    
    async def _analyze_volume_profile(self) -> Dict[str, Any]:
        """Analyze market volume profile"""
        try:
            # Sample high-volume stocks for market volume analysis
            volume_stocks = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]
            
            volume_ratios = []
            
            for stock in volume_stocks:
                try:
                    stock_data = await self.market_data.get_live_market_data(stock)
                    if stock_data and stock_data.get("technical_indicators"):
                        indicators = stock_data["technical_indicators"]
                        volume_ratio = indicators.get("volume_ratio", 1.0)
                        volume_ratios.append(volume_ratio)
                        
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"Volume analysis failed for {stock}: {e}")
                    continue
            
            if volume_ratios:
                avg_volume_ratio = np.mean(volume_ratios)
                
                if avg_volume_ratio > 1.5:
                    volume_profile = "HIGH"
                elif avg_volume_ratio < 0.8:
                    volume_profile = "LOW"
                else:
                    volume_profile = "NORMAL"
            else:
                avg_volume_ratio = 1.0
                volume_profile = "NORMAL"
            
            return {
                "avg_volume_ratio": avg_volume_ratio,
                "volume_profile": volume_profile,
                "volume_stocks_analyzed": len(volume_ratios)
            }
            
        except Exception as e:
            logger.error(f"Volume profile analysis failed: {e}")
            return {"avg_volume_ratio": 1.0, "volume_profile": "NORMAL"}
    
    def _classify_regime(self, 
                        nifty_analysis: Dict,
                        sector_analysis: Dict,
                        volatility_analysis: Dict,
                        volume_analysis: Dict) -> RegimeAnalysis:
        """Classify market regime based on all analyses"""
        try:
            trend_strength = nifty_analysis.get("trend_strength", 0.0)
            gap_pct = abs(nifty_analysis.get("gap_pct", 0.0))
            sector_dispersion = sector_analysis.get("sector_dispersion", 1.0)
            volatility_level = volatility_analysis.get("volatility_level", 15.0)
            volume_profile = volume_analysis.get("volume_profile", "NORMAL")
            
            # Classification logic
            regime = MarketRegime.SIDEWAYS  # Default
            trading_strategy = "MEAN_REVERSION"    # Default
            confidence = 0.5
            risk_adjustment = 1.0
            
            # Gap day detection
            if gap_pct > 1.0:  # >1% gap
                regime = MarketRegime.SIDEWAYS
                trading_strategy = "BREAKOUT"
                confidence = 0.8
                risk_adjustment = 1.2  # Increase risk in gap days
                
            # Trending market detection
            elif trend_strength > 0.75 and sector_dispersion < 2.0:
                if nifty_analysis.get("direction", 0) > 0:
                    regime = MarketRegime.BULLISH
                else:
                    regime = MarketRegime.BEARISH
                trading_strategy = "MOMENTUM"
                confidence = 0.75
                risk_adjustment = 1.1
                
            # High volatility detection
            elif volatility_level > 20:
                regime = MarketRegime.SIDEWAYS
                trading_strategy = "MEAN_REVERSION"
                confidence = 0.7
                risk_adjustment = 0.8  # Reduce risk in high volatility
                
            # Low volatility detection
            elif volatility_level < 12 and sector_dispersion < 1.5:
                regime = MarketRegime.SIDEWAYS
                trading_strategy = "BREAKOUT"
                confidence = 0.6
                risk_adjustment = 1.0
                
            # Choppy/sideways (default)
            else:
                regime = MarketRegime.SIDEWAYS
                trading_strategy = "MEAN_REVERSION"
                confidence = 0.6
                risk_adjustment = 0.9
            
            return RegimeAnalysis(
                regime=regime,
                confidence=confidence,
                nifty_trend_strength=trend_strength,
                sector_dispersion=sector_dispersion,
                volatility_level=volatility_level,
                volume_profile=volume_profile,
                trading_strategy=trading_strategy,
                risk_adjustment=risk_adjustment
            )
            
        except Exception as e:
            logger.error(f"Regime classification failed: {e}")
            return RegimeAnalysis(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                nifty_trend_strength=0.0,
                sector_dispersion=1.0,
                volatility_level=15.0,
                volume_profile="NORMAL",
                trading_strategy="MEAN_REVERSION",
                risk_adjustment=1.0
            )
    
    def get_strategy_for_regime(self, regime: str) -> Dict[str, Any]:
        """Get trading strategy parameters for current regime"""
        strategy_configs = {
            MarketRegime.BULLISH: {
                "strategy": "MOMENTUM",
                "confidence_threshold": 0.65,
                "risk_reward_ratio": 2.5,
                "max_trades_per_day": 4,
                "position_size_multiplier": 1.1
            },
            MarketRegime.BEARISH: {
                "strategy": "MOMENTUM_SHORT",
                "confidence_threshold": 0.70,
                "risk_reward_ratio": 2.0,
                "max_trades_per_day": 3,
                "position_size_multiplier": 1.0
            },
            MarketRegime.SIDEWAYS: {
                "strategy": "MEAN_REVERSION",
                "confidence_threshold": 0.75,
                "risk_reward_ratio": 1.8,
                "max_trades_per_day": 2,
                "position_size_multiplier": 0.9
            },
            MarketRegime.SIDEWAYS: {
                "strategy": "BREAKOUT",
                "confidence_threshold": 0.80,
                "risk_reward_ratio": 3.0,
                "max_trades_per_day": 2,
                "position_size_multiplier": 1.2
            },
            MarketRegime.SIDEWAYS: {
                "strategy": "MEAN_REVERSION",
                "confidence_threshold": 0.80,
                "risk_reward_ratio": 1.5,
                "max_trades_per_day": 1,
                "position_size_multiplier": 0.7
            }
        }
        
        return strategy_configs.get(regime, strategy_configs[MarketRegime.SIDEWAYS])

# ================================================================
# backend/app/services/pre_market_scanner.py
# Enhanced Pre-Market Module with precise timing
# ================================================================

class PreMarketScanner:
    """
    Pre-market scanner that runs from 8:30-9:15 AM
    Provides ranked stock opportunities before market open
    """
    
    def __init__(self, market_data_service: MarketDataService):
        self.market_data = market_data_service
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        
        # Scanner parameters
        self.scan_start_time = dt_time(8, 30)  # 8:30 AM
        self.scan_end_time = dt_time(9, 15)    # 9:15 AM
        self.signal_trigger_time = dt_time(9, 15)  # Exact signal time
        
        # Global cues (mock data - in production, fetch real data)
        self.global_cues = {
            "sgx_nifty": 0.0,
            "nasdaq_futures": 0.0,
            "crude_oil": 0.0,
            "dollar_index": 0.0,
            "asian_markets": 0.0
        }
    
    async def run_pre_market_scan(self) -> Dict[str, Any]:
        """
        Run complete pre-market analysis
        Triggered exactly at 9:15 AM
        """
        try:
            current_time = datetime.now().time()
            
            # Check if we're in pre-market window
            if not (self.scan_start_time <= current_time <= self.scan_end_time):
                logger.warning(f"Pre-market scan called outside window: {current_time}")
                return {"error": "Outside pre-market window"}
            
            logger.info("🌅 Starting pre-market scan at 9:15 AM...")
            
            # Get global cues
            global_analysis = await self._analyze_global_cues()
            
            # Scan all Nifty 100 stocks
            stock_rankings = await self._scan_nifty_100_stocks()
            
            # Generate final signal list
            final_signals = self._generate_pre_market_signals(stock_rankings, global_analysis)
            
            # Prepare result
            result = {
                "timestamp": datetime.now().isoformat(),
                "global_cues": global_analysis,
                "top_opportunities": final_signals[:10],
                "total_stocks_scanned": len(stock_rankings),
                "market_sentiment": self._calculate_overall_sentiment(stock_rankings),
                "ready_for_trading": True
            }
            
            logger.info(f"✅ Pre-market scan complete: {len(final_signals)} opportunities identified")
            
            return result
            
        except Exception as e:
            logger.error(f"Pre-market scan failed: {e}")
            return {"error": str(e), "ready_for_trading": False}
    
    async def _analyze_global_cues(self) -> Dict[str, float]:
        """Analyze global market cues"""
        try:
            # In production, fetch real data from APIs
            # For now, mock the global cues
            
            global_cues = {
                "sgx_nifty_change": np.random.normal(0, 0.5),  # Mock SGX Nifty
                "nasdaq_futures_change": np.random.normal(0, 0.8),  # Mock Nasdaq futures
                "crude_oil_change": np.random.normal(0, 1.2),  # Mock crude oil
                "dollar_index_change": np.random.normal(0, 0.3),  # Mock DXY
                "asian_markets_avg": np.random.normal(0, 0.6),  # Mock Asian markets
                "global_sentiment_score": np.random.uniform(-0.5, 0.5)
            }
            
            # Calculate overall global bias
            global_bias = (
                global_cues["sgx_nifty_change"] * 0.4 +
                global_cues["nasdaq_futures_change"] * 0.2 +
                global_cues["asian_markets_avg"] * 0.2 +
                global_cues["global_sentiment_score"] * 0.2
            )
            
            global_cues["overall_bias"] = global_bias
            global_cues["bias_direction"] = "BULLISH" if global_bias > 0.2 else "BEARISH" if global_bias < -0.2 else "NEUTRAL"
            
            logger.info(f"🌍 Global cues: {global_cues['bias_direction']} bias ({global_bias:.2f})")
            
            return global_cues
            
        except Exception as e:
            logger.error(f"Global cues analysis failed: {e}")
            return {"overall_bias": 0.0, "bias_direction": "NEUTRAL"}
    
    async def _scan_nifty_100_stocks(self) -> List[Dict]:
        """Scan all Nifty 100 stocks for pre-market opportunities"""
        try:
            stock_universe = Nifty100StockUniverse()
            all_stocks = stock_universe.get_all_stocks()
            
            opportunities = []
            processed = 0
            
            logger.info(f"📊 Scanning {len(all_stocks)} Nifty 100 stocks...")
            
            for symbol in all_stocks:
                try:
                    # Get market data
                    market_data = await self.market_data.get_live_market_data(symbol)
                    if not market_data or not market_data.get("quote"):
                        continue
                    
                    quote = market_data["quote"]
                    
                    # Calculate pre-market metrics
                    opportunity = await self._analyze_stock_pre_market(symbol, quote, market_data)
                    if opportunity:
                        opportunities.append(opportunity)
                    
                    processed += 1
                    if processed % 25 == 0:
                        logger.info(f"Pre-market scan progress: {processed}/{len(all_stocks)}")
                    
                    # Rate limiting
                    await asyncio.sleep(0.05)
                    
                except Exception as e:
                    logger.debug(f"Pre-market analysis failed for {symbol}: {e}")
                    continue
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
            
            logger.info(f"✅ Pre-market scan complete: {len(opportunities)} opportunities")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Nifty 100 scan failed: {e}")
            return []
    
    async def _analyze_stock_pre_market(self, symbol: str, quote: Dict, market_data: Dict) -> Optional[Dict]:
        """Analyze individual stock for pre-market opportunity"""
        try:
            # Gap analysis
            gap_pct = ((quote["ltp"] - quote["prev_close"]) / quote["prev_close"]) * 100
            
            # Volume analysis
            technical_indicators = market_data.get("technical_indicators", {})
            volume_ratio = technical_indicators.get("volume_ratio", 1.0)
            
            # Sentiment analysis
            sentiment_data = market_data.get("sentiment", {})
            news_sentiment = sentiment_data.get("sentiment_score", 0.0)
            news_count = sentiment_data.get("news_count", 0)
            
            # Enhanced FinBERT sentiment for important news
            if news_count > 0 and abs(news_sentiment) > 0.3:
                relevant_news = sentiment_data.get("relevant_news", [])
                if relevant_news:
                    news_text = relevant_news[0].get("headline", "")
                    finbert_result = self.sentiment_analyzer.analyze_sentiment(news_text)
                    enhanced_sentiment = finbert_result.get("finbert_score", news_sentiment)
                else:
                    enhanced_sentiment = news_sentiment
            else:
                enhanced_sentiment = news_sentiment
            
            # Calculate opportunity score
            opportunity_score = self._calculate_pre_market_score(
                gap_pct, volume_ratio, enhanced_sentiment, news_count
            )
            
            # Get sector info
            stock_universe = Nifty100StockUniverse()
            sector = stock_universe.get_sector(symbol)
            
            return {
                "symbol": symbol,
                "sector": sector,
                "current_price": quote["ltp"],
                "prev_close": quote["prev_close"],
                "gap_percentage": gap_pct,
                "volume_ratio": volume_ratio,
                "news_sentiment": enhanced_sentiment,
                "news_count": news_count,
                "opportunity_score": opportunity_score,
                "signal_strength": self._classify_signal_strength(opportunity_score),
                "recommended_action": self._get_recommended_action(gap_pct, enhanced_sentiment),
                "risk_level": self._assess_risk_level(abs(gap_pct), volume_ratio),
                "market_data": market_data
            }
            
        except Exception as e:
            logger.error(f"Stock pre-market analysis failed for {symbol}: {e}")
            return None
    
    def _calculate_pre_market_score(self, gap_pct: float, volume_ratio: float, sentiment: float, news_count: int) -> float:
        """Calculate comprehensive pre-market opportunity score"""
        try:
            # Weights for different factors
            gap_weight = 0.35
            volume_weight = 0.25
            sentiment_weight = 0.25
            news_weight = 0.15
            
            # Normalize and score each factor
            gap_score = min(abs(gap_pct) / 3.0, 1.0)  # Normalize to 3% gap
            volume_score = min(volume_ratio / 2.0, 1.0)  # Normalize to 2x volume
            sentiment_score = abs(sentiment)  # Sentiment strength
            news_score = min(news_count / 5.0, 1.0)  # Normalize to 5 news items
            
            # Calculate weighted score
            total_score = (
                gap_score * gap_weight +
                volume_score * volume_weight +
                sentiment_score * sentiment_weight +
                news_score * news_weight
            )
            
            # Apply direction multiplier
            direction_multiplier = 1.0
            if gap_pct > 0 and sentiment > 0:  # Both positive
                direction_multiplier = 1.2
            elif gap_pct < 0 and sentiment < 0:  # Both negative
                direction_multiplier = 1.1
            elif (gap_pct > 0 and sentiment < -0.3) or (gap_pct < 0 and sentiment > 0.3):  # Contradictory
                direction_multiplier = 0.8
            
            final_score = total_score * direction_multiplier
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Pre-market score calculation failed: {e}")
            return 0.0
    
    def _classify_signal_strength(self, score: float) -> str:
        """Classify signal strength based on opportunity score"""
        if score > 0.8:
            return "VERY_STRONG"
        elif score > 0.6:
            return "STRONG"
        elif score > 0.4:
            return "MODERATE"
        elif score > 0.2:
            return "WEAK"
        else:
            return "VERY_WEAK"
    
    def _get_recommended_action(self, gap_pct: float, sentiment: float) -> str:
        """Get recommended trading action"""
        if gap_pct > 1.5 and sentiment > 0.3:
            return "STRONG_BUY"
        elif gap_pct > 0.5 and sentiment > 0.1:
            return "BUY"
        elif gap_pct < -1.5 and sentiment < -0.3:
            return "STRONG_SELL"
        elif gap_pct < -0.5 and sentiment < -0.1:
            return "SELL"
        else:
            return "WATCH"
    
    def _assess_risk_level(self, abs_gap: float, volume_ratio: float) -> str:
        """Assess risk level for the opportunity"""
        risk_score = (abs_gap * 0.6) + (max(0, volume_ratio - 1) * 0.4)
        
        if risk_score > 3.0:
            return "VERY_HIGH"
        elif risk_score > 2.0:
            return "HIGH"
        elif risk_score > 1.0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_pre_market_signals(self, stock_rankings: List[Dict], global_analysis: Dict) -> List[Dict]:
        """Generate final pre-market signals for 9:15 AM"""
        try:
            # Filter for high-quality opportunities
            strong_opportunities = [
                stock for stock in stock_rankings 
                if stock["opportunity_score"] > 0.6 and stock["signal_strength"] in ["STRONG", "VERY_STRONG"]
            ]
            
            # Apply global bias filter
            global_bias = global_analysis.get("overall_bias", 0.0)
            
            if global_bias > 0.3:  # Strong positive global bias
                # Favor bullish signals
                filtered_signals = [
                    stock for stock in strong_opportunities
                    if stock["gap_percentage"] > 0 or stock["news_sentiment"] > 0.2
                ]
            elif global_bias < -0.3:  # Strong negative global bias
                # Favor bearish signals
                filtered_signals = [
                    stock for stock in strong_opportunities  
                    if stock["gap_percentage"] < 0 or stock["news_sentiment"] < -0.2
                ]
            else:  # Neutral global bias
                filtered_signals = strong_opportunities
            
            # Limit to top 5 signals
            final_signals = filtered_signals[:5]
            
            return final_signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return []
    
    def _calculate_overall_sentiment(self, stock_rankings: List[Dict]) -> Dict[str, Any]:
        """Calculate overall market sentiment from stock analysis"""
        try:
            if not stock_rankings:
                return {"sentiment": "NEUTRAL", "score": 0.0, "confidence": 0.0}
            
            # Aggregate sentiment scores
            sentiment_scores = [stock["news_sentiment"] for stock in stock_rankings]
            gap_percentages = [stock["gap_percentage"] for stock in stock_rankings]
            
            avg_sentiment = np.mean(sentiment_scores)
            avg_gap = np.mean(gap_percentages)
            
            # Calculate overall sentiment
            overall_score = (avg_sentiment * 0.6) + (avg_gap / 100 * 0.4)
            
            if overall_score > 0.2:
                sentiment = "BULLISH"
            elif overall_score < -0.2:
                sentiment = "BEARISH"
            else:
                sentiment = "NEUTRAL"
            
            # Calculate confidence based on consistency
            sentiment_std = np.std(sentiment_scores)
            confidence = max(0.0, 1.0 - (sentiment_std / 0.5))
            
            return {
                "sentiment": sentiment,
                "score": overall_score,
                "confidence": confidence,
                "stocks_positive": len([s for s in sentiment_scores if s > 0.1]),
                "stocks_negative": len([s for s in sentiment_scores if s < -0.1]),
                "avg_gap_percentage": avg_gap
            }
            
        except Exception as e:
            logger.error(f"Overall sentiment calculation failed: {e}")
            return {"sentiment": "NEUTRAL", "score": 0.0, "confidence": 0.0}


# ================================================================
# Updated API endpoints for new components
# ================================================================

"""
Add these endpoints to your main.py:

@app.post("/api/backtest/run")
async def run_backtest(start_date: str, end_date: str, signal_ids: List[str] = None):
    '''Run backtest on historical signals'''
    try:
        # Load signals from signal logger
        signals = []  # Load from signal_logger based on date range and IDs
        
        backtest_engine = BacktestEngine(service_manager.market_data_service)
        summary = await backtest_engine.run_backtest(
            signals, 
            datetime.fromisoformat(start_date),
            datetime.fromisoformat(end_date)
        )
        
        return asdict(summary)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/regime/current")
async def get_current_regime():
    '''Get current market regime analysis'''
    try:
        regime_detector = RegimeDetector(service_manager.market_data_service)
        regime_analysis = await regime_detector.detect_market_regime()
        return asdict(regime_analysis)
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/premarket/scan")
async def run_premarket_scan():
    '''Run pre-market scan (8:30-9:15 AM)'''
    try:
        scanner = PreMarketScanner(service_manager.market_data_service)
        scan_result = await scanner.run_pre_market_scan()
        return scan_result
    except Exception as e:
        return {"error": str(e)}
"""