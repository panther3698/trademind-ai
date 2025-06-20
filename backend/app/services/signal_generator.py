# backend/app/services/signal_generator.py
"""
Real Signal Generator integrating Market Data with AI Models
Replaces the dummy signal generation with actual market analysis
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import asdict

# Import our services
from app.core.signal_logger import (
    InstitutionalSignalLogger, SignalRecord, TechnicalIndicators, 
    MarketContext, PreMarketData, SignalDirection, TradeOutcome
)
from app.services.market_data_service import MarketDataService
from app.core.config import settings

logger = logging.getLogger(__name__)

class AISignalGenerator:
    """
    Production AI Signal Generator using real market data
    
    Features:
    - Real market data integration
    - Multi-factor signal scoring
    - Risk management integration
    - Institutional-grade logging
    """
    
    def __init__(self, market_data_service: MarketDataService, signal_logger: InstitutionalSignalLogger):
        self.market_data = market_data_service
        self.signal_logger = signal_logger
        
        # Signal generation parameters
        self.min_confidence_threshold = 0.65
        self.max_signals_per_day = 3
        self.max_position_size = 100000  # ₹1 lakh per trade
        
        # Watchlist for signal generation
        self.signal_watchlist = [
            "RELIANCE", "TCS", "HDFC", "INFY", "HDFCBANK", "ICICIBANK",
            "KOTAKBANK", "HINDUNILVR", "SBIN", "BHARTIARTL"
        ]
        
        # Model weights for different factors
        self.model_weights = {
            "technical": 0.40,
            "sentiment": 0.25,
            "volume": 0.20,
            "macro": 0.15
        }
        
        # Track daily signal count
        self.daily_signal_count = 0
        self.last_signal_date = None
        
    async def generate_signals(self) -> List[SignalRecord]:
        """
        Main signal generation function
        
        Returns:
            List of high-confidence trading signals
        """
        try:
            # Reset daily counter if new day
            today = datetime.now().date()
            if self.last_signal_date != today:
                self.daily_signal_count = 0
                self.last_signal_date = today
            
            # Check if we've reached daily limit
            if self.daily_signal_count >= self.max_signals_per_day:
                logger.info(f"Daily signal limit reached ({self.max_signals_per_day})")
                return []
            
            # Get market status
            await self.market_data.update_market_status()
            if self.market_data.market_status.value != "OPEN":
                logger.info(f"Market not open: {self.market_data.market_status.value}")
                return []
            
            signals = []
            
            # Analyze each symbol in watchlist
            for symbol in self.signal_watchlist:
                try:
                    signal = await self._analyze_symbol_for_signal(symbol)
                    if signal:
                        # Check risk limits before adding
                        risk_check = self.signal_logger.check_risk_limits(signal)
                        
                        if risk_check["overall_risk_ok"]:
                            signals.append(signal)
                            self.daily_signal_count += 1
                            
                            # Log the signal
                            success = self.signal_logger.log_signal(signal)
                            if success:
                                logger.info(f"🎯 Signal generated: {signal.ticker} {signal.direction.value} "
                                          f"Confidence: {signal.ml_confidence:.1%}")
                            
                            # Stop if we hit daily limit
                            if self.daily_signal_count >= self.max_signals_per_day:
                                break
                        else:
                            logger.warning(f"⚠️ Signal blocked by risk limits: {symbol}")
                            
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return []
    
    async def _analyze_symbol_for_signal(self, symbol: str) -> Optional[SignalRecord]:
        """
        Analyze individual symbol for trading signal
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            SignalRecord if signal found, None otherwise
        """
        try:
            # Get comprehensive market data
            market_data = await self.market_data.get_live_market_data(symbol)
            if not market_data:
                return None
            
            quote = market_data["quote"]
            indicators = market_data["technical_indicators"]
            sentiment_data = market_data["sentiment"]
            
            # Calculate individual scores
            technical_score = self._calculate_technical_score(indicators, quote)
            sentiment_score = sentiment_data.get("sentiment_score", 0.0)
            volume_score = self._calculate_volume_score(indicators, quote)
            macro_score = await self._calculate_macro_score()
            
            # Calculate final weighted score
            final_score = (
                technical_score * self.model_weights["technical"] +
                sentiment_score * self.model_weights["sentiment"] + 
                volume_score * self.model_weights["volume"] +
                macro_score * self.model_weights["macro"]
            )
            
            # Convert to confidence (0-1)
            ml_confidence = self._score_to_confidence(final_score)
            
            # Check if signal meets threshold
            if ml_confidence < self.min_confidence_threshold:
                return None
            
            # Determine signal direction
            direction = SignalDirection.BUY if final_score > 0 else SignalDirection.SELL
            
            # Calculate entry, target, and stop loss
            entry_price = quote["ltp"]
            stop_loss, target_price = self._calculate_price_levels(
                entry_price, indicators, direction
            )
            
            # Calculate position sizing
            position_size = self._calculate_position_size(
                entry_price, stop_loss, ml_confidence
            )
            
            # Create signal record with full context
            signal = SignalRecord(
                signal_id=f"TM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}",
                timestamp=datetime.now(),
                ticker=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                ml_confidence=ml_confidence,
                technical_score=technical_score,
                sentiment_score=sentiment_score,
                macro_score=macro_score,
                final_score=final_score,
                indicators=self._create_technical_indicators_object(indicators),
                market_context=await self._create_market_context(),
                pre_market=self._create_pre_market_data(sentiment_data),
                risk_reward_ratio=(target_price - entry_price) / (entry_price - stop_loss) if direction == SignalDirection.BUY else (entry_price - target_price) / (stop_loss - entry_price),
                position_size_suggested=position_size,
                capital_at_risk=abs(entry_price - stop_loss) * position_size,
                model_version="v1.0_production",
                signal_source="AI_REAL_DATA"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Symbol analysis failed for {symbol}: {e}")
            return None
    
    def _calculate_technical_score(self, indicators: Dict, quote: Dict) -> float:
        """
        Calculate technical analysis score (-1 to +1)
        
        Args:
            indicators: Technical indicators dictionary
            quote: Current quote data
            
        Returns:
            float: Technical score
        """
        if not indicators:
            return 0.0
        
        try:
            score = 0.0
            weight_sum = 0.0
            
            # RSI analysis (30% weight)
            rsi_14 = indicators.get("rsi_14", 50)
            if rsi_14 < 30:  # Oversold
                score += 0.7 * 0.30
            elif rsi_14 > 70:  # Overbought
                score -= 0.7 * 0.30
            elif 45 <= rsi_14 <= 55:  # Neutral
                score += 0.1 * 0.30
            weight_sum += 0.30
            
            # MACD analysis (25% weight)
            macd_line = indicators.get("macd_line", 0)
            macd_signal = indicators.get("macd_signal", 0)
            macd_histogram = indicators.get("macd_histogram", 0)
            
            if macd_line > macd_signal and macd_histogram > 0:  # Bullish
                score += 0.6 * 0.25
            elif macd_line < macd_signal and macd_histogram < 0:  # Bearish
                score -= 0.6 * 0.25
            weight_sum += 0.25
            
            # Bollinger Bands analysis (20% weight)
            current_price = quote.get("ltp", 0)
            bb_upper = indicators.get("bb_upper", current_price)
            bb_lower = indicators.get("bb_lower", current_price)
            bb_middle = indicators.get("bb_middle", current_price)
            
            if current_price < bb_lower:  # Below lower band - potential buy
                score += 0.5 * 0.20
            elif current_price > bb_upper:  # Above upper band - potential sell
                score -= 0.5 * 0.20
            weight_sum += 0.20
            
            # ADX trend strength (15% weight)
            adx = indicators.get("adx", 0)
            if adx > 25:  # Strong trend
                # Use price vs moving averages to determine direction
                sma_20 = indicators.get("sma_20", current_price)
                if current_price > sma_20:
                    score += 0.4 * 0.15
                else:
                    score -= 0.4 * 0.15
            weight_sum += 0.15
            
            # Volume confirmation (10% weight)
            volume_ratio = indicators.get("volume_ratio", 1.0)
            if volume_ratio > 1.5:  # High volume confirms signal
                score += 0.3 * 0.10
            elif volume_ratio < 0.8:  # Low volume weakens signal
                score -= 0.2 * 0.10
            weight_sum += 0.10
            
            # Normalize score
            if weight_sum > 0:
                score = score / weight_sum
            
            return max(-1.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Technical score calculation failed: {e}")
            return 0.0
    
    def _calculate_volume_score(self, indicators: Dict, quote: Dict) -> float:
        """Calculate volume-based score"""
        try:
            volume_ratio = indicators.get("volume_ratio", 1.0)
            
            if volume_ratio > 2.0:  # Very high volume
                return 0.8
            elif volume_ratio > 1.5:  # High volume
                return 0.5
            elif volume_ratio > 1.2:  # Above average volume
                return 0.3
            elif volume_ratio < 0.8:  # Low volume
                return -0.3
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Volume score calculation failed: {e}")
            return 0.0
    
    async def _calculate_macro_score(self) -> float:
        """Calculate macro market score"""
        try:
            # Get overall market sentiment
            market_sentiment = await self.market_data.news_service.get_market_sentiment()
            
            # Simple macro score based on market sentiment and VIX equivalent
            base_score = market_sentiment.overall_score
            
            # Adjust based on fear/greed index
            fear_greed = market_sentiment.fear_greed_index
            if fear_greed > 70:  # Greedy market
                base_score *= 0.8  # Reduce confidence
            elif fear_greed < 30:  # Fearful market
                base_score *= 1.2  # Opportunity in fear
            
            return max(-1.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"Macro score calculation failed: {e}")
            return 0.0
    
    def _score_to_confidence(self, score: float) -> float:
        """Convert combined score to ML confidence probability"""
        # Use sigmoid function to convert score to 0-1 probability
        try:
            confidence = 1 / (1 + np.exp(-score * 3))  # Amplify with factor of 3
            return max(0.0, min(1.0, confidence))
        except:
            return 0.5
    
    def _calculate_price_levels(self, entry_price: float, indicators: Dict, direction: SignalDirection) -> Tuple[float, float]:
        """
        Calculate stop loss and target price levels
        
        Args:
            entry_price: Entry price
            indicators: Technical indicators
            direction: Signal direction
            
        Returns:
            Tuple of (stop_loss, target_price)
        """
        try:
            # Use ATR for dynamic stop loss
            atr = indicators.get("atr_14", entry_price * 0.02)  # Default 2% if no ATR
            
            # Calculate stop loss (1.5x ATR from entry)
            stop_distance = atr * 1.5
            
            # Calculate target (2x stop loss distance for 2:1 R/R)
            target_distance = stop_distance * 2
            
            if direction == SignalDirection.BUY:
                stop_loss = entry_price - stop_distance
                target_price = entry_price + target_distance
            else:  # SELL
                stop_loss = entry_price + stop_distance
                target_price = entry_price - target_distance
            
            # Round to nearest 0.05
            stop_loss = round(stop_loss * 20) / 20
            target_price = round(target_price * 20) / 20
            
            return stop_loss, target_price
            
        except Exception as e:
            logger.error(f"Price level calculation failed: {e}")
            # Fallback to percentage-based levels
            if direction == SignalDirection.BUY:
                return entry_price * 0.985, entry_price * 1.03  # 1.5% SL, 3% target
            else:
                return entry_price * 1.015, entry_price * 0.97
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float, confidence: float) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            confidence: ML confidence
            
        Returns:
            float: Suggested position size (number of shares)
        """
        try:
            # Maximum risk per trade (₹2000)
            max_risk_per_trade = 2000
            
            # Risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            # Base position size
            base_position_size = max_risk_per_trade / risk_per_share
            
            # Adjust based on confidence
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
            adjusted_position_size = base_position_size * confidence_multiplier
            
            # Ensure maximum position value doesn't exceed limit
            max_position_value = min(self.max_position_size, adjusted_position_size * entry_price)
            final_position_size = max_position_value / entry_price
            
            return max(1, int(final_position_size))  # Minimum 1 share
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 10  # Default position size
    
    def _create_technical_indicators_object(self, indicators: Dict) -> TechnicalIndicators:
        """Create TechnicalIndicators object from dictionary"""
        return TechnicalIndicators(
            rsi_14=indicators.get("rsi_14", 50.0),
            rsi_21=indicators.get("rsi_21", 50.0),
            macd_line=indicators.get("macd_line", 0.0),
            macd_signal=indicators.get("macd_signal", 0.0),
            macd_histogram=indicators.get("macd_histogram", 0.0),
            bb_upper=indicators.get("bb_upper", 0.0),
            bb_middle=indicators.get("bb_middle", 0.0),
            bb_lower=indicators.get("bb_lower", 0.0),
            bb_width=indicators.get("bb_width", 0.0),
            vwap=indicators.get("vwap", 0.0),
            adx=indicators.get("adx", 0.0),
            atr_14=indicators.get("atr_14", 0.0),
            stoch_k=indicators.get("stoch_k", 50.0),
            stoch_d=indicators.get("stoch_d", 50.0),
            cci=indicators.get("cci", 0.0),
            williams_r=indicators.get("williams_r", -50.0),
            momentum_10=indicators.get("momentum_10", 0.0),
            roc_10=indicators.get("roc_10", 0.0),
            sma_20=indicators.get("sma_20", 0.0),
            sma_50=indicators.get("sma_50", 0.0),
            ema_12=indicators.get("ema_12", 0.0),
            ema_26=indicators.get("ema_26", 0.0),
            volume_sma_20=indicators.get("volume_sma_20", 0.0),
            volume_ratio=indicators.get("volume_ratio", 1.0)
        )
    
    async def _create_market_context(self) -> MarketContext:
        """Create market context object"""
        try:
            # Get Nifty data (assuming NIFTY50 or similar)
            nifty_data = await self.market_data.get_live_market_data("NIFTY50")
            
            if nifty_data and nifty_data.get("quote"):
                nifty_quote = nifty_data["quote"]
                return MarketContext(
                    nifty_price=nifty_quote.get("ltp", 0.0),
                    nifty_change_pct=nifty_quote.get("change_percent", 0.0),
                    nifty_volume_ratio=1.0,  # Would need historical Nifty volume
                    sector_vs_nifty=0.0,  # Would need sector classification
                    vix_level=15.0,  # Would need VIX data
                    fii_flow=None,
                    dii_flow=None
                )
            else:
                # Fallback values
                return MarketContext(
                    nifty_price=18500.0,
                    nifty_change_pct=0.0,
                    nifty_volume_ratio=1.0,
                    sector_vs_nifty=0.0,
                    vix_level=15.0
                )
                
        except Exception as e:
            logger.error(f"Market context creation failed: {e}")
            return MarketContext(
                nifty_price=18500.0,
                nifty_change_pct=0.0,
                nifty_volume_ratio=1.0,
                sector_vs_nifty=0.0,
                vix_level=15.0
            )
    
    def _create_pre_market_data(self, sentiment_data: Dict) -> PreMarketData:
        """Create pre-market data object"""
        return PreMarketData(
            gap_percentage=0.0,  # Would need previous close comparison
            pre_market_volume=0.0,  # Would need pre-market volume data
            news_sentiment_score=sentiment_data.get("sentiment_score", 0.0),
            news_count=sentiment_data.get("news_count", 0),
            social_sentiment=None,
            analyst_rating_change=None
        )


# Integration with FastAPI main.py
class ProductionSignalGeneratorTask:
    """
    Production signal generator task for FastAPI integration
    """
    
    def __init__(self):
        self.market_data_service = None
        self.signal_logger = None
        self.signal_generator = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize all services"""
        try:
            # Initialize market data service
            self.market_data_service = MarketDataService(
                settings.zerodha_api_key,
                settings.zerodha_access_token
            )
            await self.market_data_service.initialize()
            
            # Initialize signal logger
            self.signal_logger = InstitutionalSignalLogger("logs")
            
            # Initialize signal generator
            self.signal_generator = AISignalGenerator(
                self.market_data_service,
                self.signal_logger
            )
            
            logger.info("🚀 Production Signal Generator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize signal generator: {e}")
            raise
    
    async def run_signal_generation_loop(self):
        """Main signal generation loop"""
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    # Check if market is open
                    await self.market_data_service.update_market_status()
                    market_status = self.market_data_service.market_status
                    
                    if market_status.value == "OPEN":
                        # Generate signals
                        signals = await self.signal_generator.generate_signals()
                        
                        if signals:
                            logger.info(f"🎯 Generated {len(signals)} new signals")
                            
                            # Broadcast signals via WebSocket (integrate with existing)
                            for signal in signals:
                                signal_dict = asdict(signal)
                                # Convert datetime objects to ISO strings
                                signal_dict['timestamp'] = signal.timestamp.isoformat()
                                
                                # Send to dashboard and Telegram
                                # (This would integrate with existing service_manager)
                                
                        # Wait 60 seconds before next check (during market hours)
                        await asyncio.sleep(60)
                    else:
                        # Market closed - wait longer
                        logger.info(f"Market status: {market_status.value} - waiting...")
                        await asyncio.sleep(300)  # 5 minutes
                        
                except Exception as e:
                    logger.error(f"Signal generation loop error: {e}")
                    await asyncio.sleep(30)  # Wait before retry
                    
        except asyncio.CancelledError:
            logger.info("Signal generation loop cancelled")
        finally:
            self.is_running = False
            if self.market_data_service:
                await self.market_data_service.close()
    
    def stop(self):
        """Stop the signal generation loop"""
        self.is_running = False


# Replace the dummy signal_generator_task in main.py with this:
production_task = ProductionSignalGeneratorTask()

async def production_signal_generator_task():
    """Production signal generator task for main.py"""
    try:
        await production_task.initialize()
        await production_task.run_signal_generation_loop()
    except Exception as e:
        logger.error(f"Production signal generator task failed: {e}")
    finally:
        production_task.stop()


# Example usage for testing
async def test_signal_generation():
    """Test the production signal generator"""
    
    # Mock environment variables for testing
    import os
    os.environ["ZERODHA_API_KEY"] = "your_api_key"
    os.environ["ZERODHA_ACCESS_TOKEN"] = "your_access_token"
    
    try:
        task = ProductionSignalGeneratorTask()
        await task.initialize()
        
        # Generate test signals
        signals = await task.signal_generator.generate_signals()
        
        print(f"Generated {len(signals)} signals:")
        for signal in signals:
            print(f"- {signal.ticker} {signal.direction.value} @ ₹{signal.entry_price} "
                  f"(Confidence: {signal.ml_confidence:.1%})")
        
        # Get daily summary
        summary = task.signal_logger.get_daily_summary()
        print(f"\nDaily Summary: {summary}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        if task.market_data_service:
            await task.market_data_service.close()

if __name__ == "__main__":
    # Run test
    asyncio.run(test_signal_generation())