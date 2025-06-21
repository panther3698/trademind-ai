# ================================================================
# backend/app/services/production_signal_generator.py
# Complete Enhanced ML Signal Generator with Regime Detection Integration
# ================================================================

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import asdict

# Import our ML components
from app.ml.models import (
    Nifty100StockUniverse, 
    FinBERTSentimentAnalyzer, 
    FeatureEngineering, 
    XGBoostSignalModel,
    TrainingDataCollector,
    EnsembleModel
)

# ADD THESE ADVANCED IMPORTS:
from app.ml.advanced_models import (
    LightGBMModel,
    CatBoostModel, 
    LSTMModel,
    EnsembleModel as AdvancedEnsembleModel
)
from app.ml.advanced_sentiment import (
    ComprehensiveSentimentEngine
)

# Import existing components
from app.core.signal_logger import (
    InstitutionalSignalLogger, SignalRecord, TechnicalIndicators, 
    MarketContext, PreMarketData, SignalDirection, TradeOutcome
)
from app.services.enhanced_market_data_nifty100 import EnhancedMarketDataService as MarketDataService

logger = logging.getLogger(__name__)

class ProductionMLSignalGenerator:
    """
    Production ML-based Signal Generator with Regime Detection Integration:
    - Nifty 100 stock universe
    - FinBERT sentiment analysis
    - XGBoost prediction models
    - Comprehensive feature engineering
    - Market regime awareness for adaptive strategies
    """
    
    def __init__(self, market_data_service: MarketDataService, signal_logger: InstitutionalSignalLogger):
        self.market_data = market_data_service
        self.signal_logger = signal_logger
        
        # Initialize ML components (Enhanced)
        self.stock_universe = Nifty100StockUniverse()
        self.sentiment_analyzer = ComprehensiveSentimentEngine()  # UPGRADED
        self.feature_engineer = FeatureEngineering(self.stock_universe)
        self.ml_model = EnsembleModel()  # UPGRADED - Enhanced with auto-detection
        
        # Signal generation parameters (base values)
        self.base_min_ml_confidence = 0.70  # Base 70% ML confidence threshold
        self.base_max_signals_per_day = 3
        self.max_capital_per_trade = 100000  # ₹1 lakh per trade
        
        # Current adjusted parameters (will be updated by regime)
        self.min_ml_confidence = self.base_min_ml_confidence
        self.max_signals_per_day = self.base_max_signals_per_day
        
        # Track daily activity
        self.daily_signal_count = 0
        self.last_signal_date = None
        
        # Performance tracking
        self.model_performance = {
            "total_predictions": 0,
            "high_confidence_predictions": 0,
            "accuracy_tracking": [],
            "regime_performance": {}  # Track performance by regime
        }
        
        # Regime awareness
        self.regime_context = {
            "regime": "SIDEWAYS_CHOPPY",  # Default
            "confidence": 0.5,
            "risk_adjustment": 1.0,
            "trading_strategy": "MEAN_REVERSION"
        }
        
        # Regime-specific parameters
        self.regime_configs = {
            "TRENDING_BULLISH": {
                "confidence_adjustment": -0.05,  # Lower threshold (easier to trigger)
                "max_signals_multiplier": 1.3,   # More signals allowed
                "risk_multiplier": 1.1,          # Slightly higher risk
                "rr_ratio_bonus": 0.5,           # Better risk-reward
                "analysis_limit_multiplier": 1.5
            },
            "TRENDING_BEARISH": {
                "confidence_adjustment": -0.03,
                "max_signals_multiplier": 1.1,
                "risk_multiplier": 1.0,
                "rr_ratio_bonus": 0.3,
                "analysis_limit_multiplier": 1.3
            },
            "SIDEWAYS_CHOPPY": {
                "confidence_adjustment": +0.10,  # Higher threshold (harder to trigger)
                "max_signals_multiplier": 0.7,   # Fewer signals
                "risk_multiplier": 0.8,          # Lower risk
                "rr_ratio_bonus": 0.0,           # Standard risk-reward
                "analysis_limit_multiplier": 1.0
            },
            "GAP_DAY": {
                "confidence_adjustment": -0.05,
                "max_signals_multiplier": 1.5,
                "risk_multiplier": 1.2,
                "rr_ratio_bonus": 0.8,
                "analysis_limit_multiplier": 1.8
            },
            "HIGH_VOLATILITY": {
                "confidence_adjustment": +0.15,  # Much higher threshold
                "max_signals_multiplier": 0.5,   # Much fewer signals
                "risk_multiplier": 0.6,          # Much lower risk
                "rr_ratio_bonus": -0.2,          # Conservative targets
                "analysis_limit_multiplier": 0.6
            },
            "LOW_VOLATILITY": {
                "confidence_adjustment": -0.02,
                "max_signals_multiplier": 1.2,
                "risk_multiplier": 1.0,
                "rr_ratio_bonus": 0.2,
                "analysis_limit_multiplier": 1.2
            }
        }
        
        # Initialize model (load if exists, train if needed)
        self._initialize_ml_model()
    
    def set_regime_context(self, regime_context: Dict):
        """
        Set regime context for adaptive signal generation
        Called by main.py service manager
        
        Args:
            regime_context: Dict with regime, confidence, risk_adjustment, trading_strategy
        """
        try:
            old_regime = self.regime_context.get("regime", "UNKNOWN")
            self.regime_context.update(regime_context)
            new_regime = regime_context.get("regime", "SIDEWAYS_CHOPPY")
            
            if old_regime != new_regime:
                logger.info(f"🎯 Signal generator regime update: {old_regime} → {new_regime}")
                logger.info(f"📊 Regime confidence: {regime_context.get('confidence', 0.5):.1%}")
                logger.info(f"📈 Trading strategy: {regime_context.get('trading_strategy', 'MEAN_REVERSION')}")
                
                # Update signal generation parameters
                self._update_regime_parameters()
                
        except Exception as e:
            logger.error(f"❌ Failed to set regime context: {e}")
    
    def _update_regime_parameters(self):
        """Update signal generation parameters based on current regime"""
        try:
            current_regime = self.regime_context.get("regime", "SIDEWAYS_CHOPPY")
            config = self.regime_configs.get(current_regime, self.regime_configs["SIDEWAYS_CHOPPY"])
            
            # Update confidence threshold
            self.min_ml_confidence = max(0.5, min(0.9, 
                self.base_min_ml_confidence + config["confidence_adjustment"]
            ))
            
            # Update max signals per day
            self.max_signals_per_day = max(1, int(
                self.base_max_signals_per_day * config["max_signals_multiplier"]
            ))
            
            logger.info(f"🔧 Regime-adjusted parameters:")
            logger.info(f"  • Min confidence: {self.min_ml_confidence:.1%}")
            logger.info(f"  • Max signals/day: {self.max_signals_per_day}")
            logger.info(f"  • Risk multiplier: {config['risk_multiplier']:.1f}x")
            logger.info(f"  • R:R bonus: {config['rr_ratio_bonus']:+.1f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update regime parameters: {e}")
    
    def _initialize_ml_model(self):
        """Initialize or load ML model"""
        try:
            # Try to load existing model
            if self.ml_model.load_model():
                logger.info("✅ Loaded existing XGBoost model")
            else:
                logger.warning("⚠️ No trained model found. Using fallback scoring method.")
                # In production, you'd want to train immediately
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
    
    async def generate_signals(self) -> List[SignalRecord]:
        """
        Generate regime-aware trading signals using ML pipeline for Nifty 100 stocks
        
        Returns:
            List of high-confidence ML-generated signals adapted to market regime
        """
        try:
            # Reset daily counter if new day
            today = datetime.now().date()
            if self.last_signal_date != today:
                self.daily_signal_count = 0
                self.last_signal_date = today
                logger.info(f"🌅 New trading day: {today}")
                logger.info(f"🎯 Current regime: {self.regime_context.get('regime', 'UNKNOWN')}")
            
            # Check daily limits (regime-adjusted)
            if self.daily_signal_count >= self.max_signals_per_day:
                logger.info(f"Daily signal limit reached ({self.max_signals_per_day}) for regime {self.regime_context.get('regime')}")
                return []
            
            # Check market status
            await self.market_data.update_market_status()
            if self.market_data.market_status.value != "OPEN":
                logger.info(f"Market closed: {self.market_data.market_status.value}")
                return []
            
            current_regime = self.regime_context.get("regime", "SIDEWAYS_CHOPPY")
            logger.info(f"🔍 Scanning Nifty 100 stocks for ML signals (Regime: {current_regime})...")
            
            # Get top opportunity stocks using pre-market analysis
            opportunity_stocks = await self._get_top_opportunity_stocks()
            
            signals = []
            
            # Analyze top opportunity stocks with ML (regime-aware)
            analysis_limit = self._get_regime_analysis_limit()
            for stock_data in opportunity_stocks[:analysis_limit]:
                try:
                    symbol = stock_data["symbol"]
                    signal = await self._analyze_stock_with_ml(symbol, stock_data)
                    
                    if signal:
                        # Validate signal with regime-aware risk management
                        risk_check = self.signal_logger.check_risk_limits(signal)
                        
                        if risk_check["overall_risk_ok"]:
                            # Add regime metadata to signal
                            signal = self._enhance_signal_with_regime_data(signal)
                            
                            signals.append(signal)
                            self.daily_signal_count += 1
                            
                            # Log the signal with regime info
                            success = self.signal_logger.log_signal(signal)
                            if success:
                                logger.info(f"🎯 ML Signal ({current_regime}): {signal.ticker} {signal.direction.value} "
                                          f"ML Confidence: {signal.ml_confidence:.1%} | "
                                          f"Entry: ₹{signal.entry_price} | Risk Adj: {self.regime_context.get('risk_adjustment', 1.0):.1f}x")
                            
                            # Track regime-specific performance
                            self._track_regime_signal(signal, current_regime)
                            
                            # Stop if daily limit reached
                            if self.daily_signal_count >= self.max_signals_per_day:
                                break
                        else:
                            logger.warning(f"⚠️ Signal blocked by regime-aware risk limits: {symbol}")
                
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
            
            if signals:
                logger.info(f"🚀 Generated {len(signals)} regime-aware ML signals for {today}")
            else:
                logger.info(f"📊 No ML signals met regime-adjusted confidence threshold ({self.min_ml_confidence:.1%}) today")
            
            return signals
            
        except Exception as e:
            logger.error(f"Regime-aware ML signal generation failed: {e}")
            return []
    
    def _get_regime_analysis_limit(self) -> int:
        """Get number of stocks to analyze based on regime"""
        current_regime = self.regime_context.get("regime", "SIDEWAYS_CHOPPY")
        config = self.regime_configs.get(current_regime, self.regime_configs["SIDEWAYS_CHOPPY"])
        
        base_limit = 20
        multiplier = config["analysis_limit_multiplier"]
        return max(10, int(base_limit * multiplier))
    
    def _enhance_signal_with_regime_data(self, signal: SignalRecord) -> SignalRecord:
        """Add regime-specific metadata to signal"""
        try:
            # Add regime information to notes
            regime_info = (
                f"Regime: {self.regime_context.get('regime', 'UNKNOWN')} "
                f"(Conf: {self.regime_context.get('confidence', 0.5):.1%}), "
                f"Strategy: {self.regime_context.get('trading_strategy', 'MEAN_REVERSION')}, "
                f"Risk Adj: {self.regime_context.get('risk_adjustment', 1.0):.1f}x"
            )
            
            if signal.notes:
                signal.notes = f"{signal.notes} | {regime_info}"
            else:
                signal.notes = regime_info
            
            # Update signal source to indicate regime awareness
            signal.signal_source = f"XGBOOST_FINBERT_REGIME_{self.regime_context.get('regime', 'UNKNOWN')}"
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to enhance signal with regime data: {e}")
            return signal
    
    def _track_regime_signal(self, signal: SignalRecord, regime: str):
        """Track signal generation by regime for performance analysis"""
        try:
            if regime not in self.model_performance["regime_performance"]:
                self.model_performance["regime_performance"][regime] = {
                    "signals_generated": 0,
                    "avg_confidence": 0.0,
                    "confidence_sum": 0.0,
                    "avg_final_score": 0.0,
                    "final_score_sum": 0.0
                }
            
            regime_stats = self.model_performance["regime_performance"][regime]
            regime_stats["signals_generated"] += 1
            regime_stats["confidence_sum"] += signal.ml_confidence
            regime_stats["avg_confidence"] = regime_stats["confidence_sum"] / regime_stats["signals_generated"]
            
            if signal.final_score:
                regime_stats["final_score_sum"] += signal.final_score
                regime_stats["avg_final_score"] = regime_stats["final_score_sum"] / regime_stats["signals_generated"]
            
        except Exception as e:
            logger.error(f"Failed to track regime signal: {e}")
    
    async def _get_top_opportunity_stocks(self) -> List[Dict]:
        """
        Get top opportunity stocks from Nifty 100 using regime-aware pre-market analysis
        
        Returns:
            List of stock opportunities sorted by ML potential with regime considerations
        """
        try:
            current_regime = self.regime_context.get("regime", "SIDEWAYS_CHOPPY")
            logger.info(f"📋 Running regime-aware pre-market analysis on Nifty 100 (Regime: {current_regime})...")
            
            # Get all Nifty 100 stocks
            all_stocks = self.stock_universe.get_all_stocks()
            
            opportunities = []
            processed = 0
            
            for symbol in all_stocks:
                try:
                    # Get basic market data
                    market_data = await self.market_data.get_live_market_data(symbol)
                    if not market_data or not market_data.get("quote"):
                        continue
                    
                    quote = market_data["quote"]
                    
                    # Calculate basic opportunity metrics
                    gap_pct = ((quote["ltp"] - quote["prev_close"]) / quote["prev_close"]) * 100
                    volume_ratio = quote.get("volume", 1) / 1000000  # Normalize volume
                    
                    # Get news sentiment
                    sentiment_data = market_data.get("sentiment", {})
                    sentiment_score = sentiment_data.get("sentiment_score", 0.0)
                    
                    # Regime-aware opportunity scoring
                    opportunity_score = self._calculate_regime_aware_opportunity_score(
                        gap_pct, volume_ratio, sentiment_score, current_regime
                    )
                    
                    opportunities.append({
                        "symbol": symbol,
                        "opportunity_score": opportunity_score,
                        "gap_percentage": gap_pct,
                        "volume_ratio": volume_ratio,
                        "sentiment_score": sentiment_score,
                        "current_price": quote["ltp"],
                        "sector": self.stock_universe.get_sector(symbol),
                        "market_data": market_data,
                        "regime_score": opportunity_score
                    })
                    
                    processed += 1
                    if processed % 20 == 0:
                        logger.info(f"Analyzed {processed}/{len(all_stocks)} stocks...")
                    
                    # Rate limiting
                    await asyncio.sleep(0.05)  # 50ms delay
                    
                except Exception as e:
                    logger.debug(f"Opportunity analysis failed for {symbol}: {e}")
                    continue
            
            # Sort by regime-aware opportunity score
            opportunities.sort(key=lambda x: x["regime_score"], reverse=True)
            
            logger.info(f"✅ Analyzed {len(opportunities)} stocks, top 10 regime-aware opportunities:")
            for i, opp in enumerate(opportunities[:10]):
                logger.info(f"  {i+1}. {opp['symbol']} - Regime Score: {opp['regime_score']:.2f} "
                          f"Gap: {opp['gap_percentage']:.1f}% Sentiment: {opp['sentiment_score']:.2f}")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Regime-aware opportunity analysis failed: {e}")
            return []
    
    def _calculate_regime_aware_opportunity_score(self, gap_pct: float, volume_ratio: float, 
                                                sentiment_score: float, regime: str) -> float:
        """Calculate opportunity score with regime-specific weightings"""
        try:
            # Base scoring weights
            gap_weight = 0.4
            volume_weight = 0.3
            sentiment_weight = 0.3
            
            # Regime-specific adjustments
            if regime == "TRENDING_BULLISH":
                gap_weight = 0.5    # Higher weight on momentum
                volume_weight = 0.3
                sentiment_weight = 0.2
            elif regime == "TRENDING_BEARISH":
                gap_weight = 0.5
                volume_weight = 0.3
                sentiment_weight = 0.2
            elif regime == "SIDEWAYS_CHOPPY":
                gap_weight = 0.2    # Lower weight on momentum
                volume_weight = 0.4  # Higher weight on volume
                sentiment_weight = 0.4  # Higher weight on sentiment
            elif regime == "GAP_DAY":
                gap_weight = 0.6    # Much higher weight on gaps
                volume_weight = 0.25
                sentiment_weight = 0.15
            elif regime == "HIGH_VOLATILITY":
                gap_weight = 0.3
                volume_weight = 0.4
                sentiment_weight = 0.3
            elif regime == "LOW_VOLATILITY":
                gap_weight = 0.3
                volume_weight = 0.3
                sentiment_weight = 0.4
            
            # Calculate weighted score
            opportunity_score = (
                abs(gap_pct) * gap_weight +
                min(volume_ratio, 3.0) * volume_weight +
                abs(sentiment_score) * sentiment_weight
            )
            
            # Apply regime confidence multiplier
            regime_confidence = self.regime_context.get("confidence", 0.5)
            confidence_multiplier = 0.8 + (regime_confidence * 0.4)  # 0.8 to 1.2
            
            return opportunity_score * confidence_multiplier
            
        except Exception as e:
            logger.error(f"Regime-aware scoring failed: {e}")
            return abs(gap_pct) * 0.4 + min(volume_ratio, 3.0) * 0.3 + abs(sentiment_score) * 0.3
    
    async def _analyze_stock_with_ml(self, symbol: str, stock_data: Dict) -> Optional[SignalRecord]:
        """
        Analyze individual stock using regime-aware ML pipeline
        
        Args:
            symbol: Stock symbol
            stock_data: Pre-computed stock data from opportunity analysis
            
        Returns:
            SignalRecord if regime-adjusted ML confidence is high enough
        """
        try:
            current_regime = self.regime_context.get("regime", "SIDEWAYS_CHOPPY")
            logger.debug(f"🤖 Regime-aware ML analysis for {symbol} (Regime: {current_regime})...")
            
            market_data = stock_data["market_data"]
            quote = market_data["quote"]
            
            # Get historical data for technical analysis
            try:
                historical_data = await self.market_data.zerodha.get_historical_data(
                    symbol, "minute", datetime.now() - timedelta(days=30)
                )
            except Exception as e:
                logger.debug(f"Historical data unavailable for {symbol}: {e}")
                historical_data = []
            
            if len(historical_data) < 50:
                logger.debug(f"Insufficient historical data for {symbol}")
                # Use fallback analysis
                return await self._fallback_signal_analysis(symbol, stock_data)
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': h.timestamp,
                'open': h.open,
                'high': h.high,
                'low': h.low,
                'close': h.close,
                'volume': h.volume
            } for h in historical_data])
            
            # Engineer features
            current_quote = {
                'ltp': quote["ltp"],
                'prev_close': quote["prev_close"],
                'volume': quote.get("volume", 0),
                'change_percent': stock_data["gap_percentage"]
            }
            
            # Enhanced sentiment analysis using FinBERT
            news_sentiment = await self._get_enhanced_sentiment(symbol)
            
            # Get Nifty data for correlation
            nifty_data = await self._get_nifty_correlation_data()
            
            # Engineer comprehensive features with regime context
            features = self.feature_engineer.engineer_features(
                symbol, df, current_quote, news_sentiment, nifty_data
            )
            
            if not features:
                logger.debug(f"Feature engineering failed for {symbol}")
                return await self._fallback_signal_analysis(symbol, stock_data)
            
            # Add regime-specific features
            features = self._add_regime_features(features, current_regime)
            
            # ML prediction with regime considerations
            ml_probability, prediction_details = self.ml_model.predict_signal_probability(features)
            
            # Apply regime-specific confidence adjustment
            adjusted_probability = self._apply_regime_confidence_adjustment(ml_probability, current_regime)
            
            # Track prediction
            self.model_performance["total_predictions"] += 1
            if adjusted_probability > self.min_ml_confidence:
                self.model_performance["high_confidence_predictions"] += 1
            
            # Check regime-adjusted ML confidence threshold
            if adjusted_probability < self.min_ml_confidence:
                logger.debug(f"{symbol} Regime-adjusted ML confidence {adjusted_probability:.1%} below threshold {self.min_ml_confidence:.1%}")
                return None
            
            # Determine signal direction based on features and ML output
            direction = self._determine_signal_direction(features, adjusted_probability, current_regime)
            
            # Calculate price levels using regime-enhanced logic
            entry_price = quote["ltp"]
            stop_loss, target_price = self._calculate_regime_aware_price_levels(
                entry_price, features, direction, adjusted_probability, current_regime
            )
            
            # Calculate position sizing based on ML confidence and regime
            position_size = self._calculate_regime_aware_position_size(
                entry_price, stop_loss, adjusted_probability, features, current_regime
            )
            
            # Create comprehensive signal record with regime awareness
            signal = SignalRecord(
                signal_id=f"ML_{current_regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}",
                timestamp=datetime.now(),
                ticker=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                ml_confidence=adjusted_probability,  # Use regime-adjusted confidence
                technical_score=features.get("technical_composite_score", 0.0),
                sentiment_score=news_sentiment.get("finbert_score", 0.0),
                macro_score=features.get("nifty_performance", 0.0),
                final_score=adjusted_probability,  # Regime-adjusted ML probability is our final score
                indicators=self._create_technical_indicators_from_features(features),
                market_context=await self._create_enhanced_market_context(features),
                pre_market=self._create_pre_market_data_from_features(features, news_sentiment),
                risk_reward_ratio=self._calculate_risk_reward_ratio(entry_price, stop_loss, target_price, direction),
                position_size_suggested=position_size,
                capital_at_risk=abs(entry_price - stop_loss) * position_size,
                model_version="v1.1_ML_regime_aware",
                signal_source=f"XGBOOST_FINBERT_REGIME_{current_regime}",
                notes=f"Regime: {current_regime}, Original ML: {ml_probability:.1%}, Adjusted: {adjusted_probability:.1%}, Top features: {list(prediction_details.get('feature_contributions', {}).keys())[:3]}"
            )
            
            logger.debug(f"✅ {symbol} Regime-aware ML Signal: {direction.value} Confidence: {adjusted_probability:.1%}")
            return signal
            
        except Exception as e:
            logger.error(f"Regime-aware ML analysis failed for {symbol}: {e}")
            return None
    
    async def _fallback_signal_analysis(self, symbol: str, stock_data: Dict) -> Optional[SignalRecord]:
        """Fallback signal analysis when ML features are unavailable"""
        try:
            logger.debug(f"🔄 Using fallback analysis for {symbol}")
            
            quote = stock_data["market_data"]["quote"]
            gap_pct = stock_data["gap_percentage"]
            sentiment_score = stock_data["sentiment_score"]
            
            # Simple scoring without ML
            momentum_score = abs(gap_pct) / 5.0  # Normalize gap percentage
            sentiment_strength = abs(sentiment_score)
            volume_score = min(stock_data["volume_ratio"], 2.0) / 2.0
            
            # Combine scores
            fallback_score = (momentum_score * 0.5 + sentiment_strength * 0.3 + volume_score * 0.2)
            
            # Apply regime adjustment
            current_regime = self.regime_context.get("regime", "SIDEWAYS_CHOPPY")
            regime_multiplier = self._get_regime_fallback_multiplier(current_regime)
            adjusted_score = fallback_score * regime_multiplier
            
            # Check threshold
            if adjusted_score < 0.6:  # Lower threshold for fallback
                return None
            
            # Create simple signal
            direction = SignalDirection.BUY if gap_pct > 0 and sentiment_score > 0 else SignalDirection.SELL
            entry_price = quote["ltp"]
            
            # Simple price levels
            if direction == SignalDirection.BUY:
                stop_loss = entry_price * 0.985
                target_price = entry_price * 1.04
            else:
                stop_loss = entry_price * 1.015
                target_price = entry_price * 0.96
            
            # Simple position sizing
            position_size = min(50, int(5000 / abs(entry_price - stop_loss)))
            
            signal = SignalRecord(
                signal_id=f"FALLBACK_{current_regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}",
                timestamp=datetime.now(),
                ticker=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                ml_confidence=adjusted_score,
                technical_score=momentum_score,
                sentiment_score=sentiment_score,
                macro_score=0.0,
                final_score=adjusted_score,
                indicators=self._create_fallback_technical_indicators(quote),
                market_context=await self._create_fallback_market_context(),
                pre_market=self._create_fallback_pre_market_data(stock_data),
                risk_reward_ratio=self._calculate_risk_reward_ratio(entry_price, stop_loss, target_price, direction),
                position_size_suggested=position_size,
                capital_at_risk=abs(entry_price - stop_loss) * position_size,
                model_version="v1.1_fallback_regime_aware",
                signal_source=f"FALLBACK_REGIME_{current_regime}",
                notes=f"Fallback analysis - Regime: {current_regime}, Score: {adjusted_score:.1%}"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Fallback analysis failed for {symbol}: {e}")
            return None
    
    def _get_regime_fallback_multiplier(self, regime: str) -> float:
        """Get fallback score multiplier for regime"""
        multipliers = {
            "TRENDING_BULLISH": 1.2,
            "TRENDING_BEARISH": 1.1,
            "SIDEWAYS_CHOPPY": 0.8,
            "GAP_DAY": 1.3,
            "HIGH_VOLATILITY": 0.6,
            "LOW_VOLATILITY": 1.0
        }
        return multipliers.get(regime, 1.0)
    
    def _add_regime_features(self, features: Dict, regime: str) -> Dict:
        """Add regime-specific features to the feature set"""
        try:
            # Add regime as categorical features
            regime_features = {
                f"regime_{regime.lower()}": 1.0,
                "regime_confidence": self.regime_context.get("confidence", 0.5),
                "regime_risk_adjustment": self.regime_context.get("risk_adjustment", 1.0)
            }
            
            # Set all other regime indicators to 0
            all_regimes = ["TRENDING_BULLISH", "TRENDING_BEARISH", "SIDEWAYS_CHOPPY", "GAP_DAY", "HIGH_VOLATILITY", "LOW_VOLATILITY"]
            for r in all_regimes:
                if r != regime:
                    regime_features[f"regime_{r.lower()}"] = 0.0
            
            features.update(regime_features)
            return features
            
        except Exception as e:
            logger.error(f"Failed to add regime features: {e}")
            return features
    
    def _apply_regime_confidence_adjustment(self, ml_probability: float, regime: str) -> float:
        """Apply regime-specific confidence adjustments to ML probability"""
        try:
            # Get regime confidence from context
            regime_confidence = self.regime_context.get("confidence", 0.5)
            
            # Apply regime-specific adjustments
            if regime in ["TRENDING_BULLISH", "TRENDING_BEARISH", "GAP_DAY"]:
                # In trending/gap markets, boost ML confidence
                confidence_boost = 0.05 * regime_confidence
                adjusted_probability = ml_probability + confidence_boost
            elif regime == "HIGH_VOLATILITY":
                # In volatile markets, be more conservative
                confidence_penalty = 0.1 * regime_confidence
                adjusted_probability = ml_probability - confidence_penalty
            elif regime == "LOW_VOLATILITY":
                # In low volatility, slight boost
                confidence_boost = 0.02 * regime_confidence
                adjusted_probability = ml_probability + confidence_boost
            else:  # SIDEWAYS_CHOPPY
                # In choppy markets, slight penalty
                confidence_penalty = 0.03 * regime_confidence
                adjusted_probability = ml_probability - confidence_penalty
            
            # Ensure probability stays within bounds
            return max(0.0, min(1.0, adjusted_probability))
            
        except Exception as e:
            logger.error(f"Failed to apply regime confidence adjustment: {e}")
            return ml_probability
    
    def _determine_signal_direction(self, features: Dict, ml_probability: float, regime: str) -> SignalDirection:
        """Determine signal direction based on features, ML output, and regime"""
        try:
            # Primary signal from ML probability (>0.5 = BUY, <0.5 = SELL)
            if ml_probability > 0.5:
                base_direction = SignalDirection.BUY
            else:
                base_direction = SignalDirection.SELL
                
            # Confirm with technical indicators
            technical_bullish_signals = (
                features.get("above_sma20", 0) +
                features.get("macd_bullish", 0) +
                features.get("high_volume", 0) +
                features.get("positive_sentiment", 0)
            )
            
            # Regime-specific direction adjustments
            regime_strategy = self.regime_context.get("trading_strategy", "MEAN_REVERSION")
            
            if regime == "TRENDING_BEARISH" and regime_strategy == "MOMENTUM":
                # In bearish trends with momentum strategy, bias towards SELL
                if base_direction == SignalDirection.BUY and technical_bullish_signals < 3:
                    logger.debug(f"Bearish regime suggests caution for BUY signal")
            elif regime == "TRENDING_BULLISH" and regime_strategy == "MOMENTUM":
                # In bullish trends with momentum strategy, bias towards BUY
                if base_direction == SignalDirection.SELL and technical_bullish_signals > 1:
                    logger.debug(f"Bullish regime suggests caution for SELL signal")
            elif regime == "HIGH_VOLATILITY":
                # In high volatility, require stronger confirmation
                if base_direction == SignalDirection.BUY and technical_bullish_signals < 3:
                    logger.debug("High volatility regime requires stronger confirmation for BUY")
                elif base_direction == SignalDirection.SELL and technical_bullish_signals > 1:
                    logger.debug("High volatility regime requires stronger confirmation for SELL")
            
            return base_direction
            
        except Exception as e:
            logger.error(f"Direction determination failed: {e}")
            return SignalDirection.BUY  # Default
    
    def _calculate_regime_aware_price_levels(self, 
                                           entry_price: float, 
                                           features: Dict, 
                                           direction: SignalDirection,
                                           ml_confidence: float,
                                           regime: str) -> Tuple[float, float]:
        """Calculate stop loss and target using regime-aware ML-enhanced logic"""
        try:
            # Base ATR for stop loss
            atr_percent = features.get("atr_percent", 2.0)
            
            # Get regime configuration
            config = self.regime_configs.get(regime, self.regime_configs["SIDEWAYS_CHOPPY"])
            
            # Adjust stop loss based on ML confidence, volatility, and regime
            confidence_multiplier = 0.8 + (ml_confidence * 0.4)  # 0.8 to 1.2
            volatility_multiplier = min(2.0, max(0.5, atr_percent / 2.0))
            regime_risk_multiplier = config["risk_multiplier"]
            
            stop_distance_percent = 1.5 * confidence_multiplier * volatility_multiplier * regime_risk_multiplier
            
            # Calculate target with regime-enhanced risk-reward
            base_rr_ratio = 2.0
            confidence_rr_bonus = (ml_confidence - 0.5) * 2  # Extra reward for high confidence
            regime_rr_bonus = config["rr_ratio_bonus"]
            target_rr_ratio = base_rr_ratio + confidence_rr_bonus + regime_rr_bonus
            
            # Regime-specific adjustments
            if regime == "HIGH_VOLATILITY":
                # Tighter stops and more conservative targets in volatile markets
                stop_distance_percent *= 0.8
                target_rr_ratio *= 0.9
            elif regime in ["TRENDING_BULLISH", "TRENDING_BEARISH"]:
                # Wider targets in trending markets
                target_rr_ratio *= 1.2
            elif regime == "GAP_DAY":
                # Quick profits on gap days
                target_rr_ratio *= 1.4
                
            if direction == SignalDirection.BUY:
                stop_loss = entry_price * (1 - stop_distance_percent / 100)
                target_price = entry_price * (1 + (stop_distance_percent * target_rr_ratio) / 100)
            else:
                stop_loss = entry_price * (1 + stop_distance_percent / 100)
                target_price = entry_price * (1 - (stop_distance_percent * target_rr_ratio) / 100)
            
            # Round to nearest 0.05
            stop_loss = round(stop_loss * 20) / 20
            target_price = round(target_price * 20) / 20
            
            return stop_loss, target_price
            
        except Exception as e:
            logger.error(f"Regime-aware ML price levels calculation failed: {e}")
            # Fallback to simple percentage
            if direction == SignalDirection.BUY:
                return entry_price * 0.985, entry_price * 1.04
            else:
                return entry_price * 1.015, entry_price * 0.96
    
    def _calculate_regime_aware_position_size(self, 
                                            entry_price: float, 
                                            stop_loss: float, 
                                            ml_confidence: float,
                                            features: Dict,
                                            regime: str) -> float:
        """Calculate position size based on ML confidence, risk, and regime"""
        try:
            # Base risk per trade
            base_risk = 2500  # ₹2500 base risk
            
            # Get regime configuration
            config = self.regime_configs.get(regime, self.regime_configs["SIDEWAYS_CHOPPY"])
            
            # Adjust risk based on ML confidence
            confidence_risk_multiplier = 0.6 + (ml_confidence * 0.8)  # 0.6 to 1.4
            
            # Apply regime risk adjustment
            regime_risk_multiplier = config["risk_multiplier"]
            regime_confidence_multiplier = 0.8 + (self.regime_context.get("confidence", 0.5) * 0.4)
            
            adjusted_risk = base_risk * confidence_risk_multiplier * regime_risk_multiplier * regime_confidence_multiplier
            
            # Adjust for volatility
            volatility = features.get("atr_percent", 2.0)
            volatility_multiplier = min(1.5, max(0.5, 2.0 / volatility))
            final_risk = adjusted_risk * volatility_multiplier
            
            # Special regime adjustments
            if regime == "HIGH_VOLATILITY":
                final_risk *= 0.7  # Even more conservative in high volatility
            elif regime == "GAP_DAY":
                final_risk *= 1.3  # Slightly more aggressive on gap days
            
            # Calculate position size
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share <= 0:
                return 20  # Default position size
                
            position_size = final_risk / risk_per_share
            
            # Cap position value
            max_position_value = min(self.max_capital_per_trade, position_size * entry_price)
            final_position_size = max_position_value / entry_price
            
            return max(1, int(final_position_size))
            
        except Exception as e:
            logger.error(f"Regime-aware ML position sizing failed: {e}")
            return 20  # Default
    
    # ================================================================
    # Enhanced Helper Methods
    # ================================================================
    
    async def _get_enhanced_sentiment(self, symbol: str) -> Dict:
        """Get enhanced sentiment analysis using FinBERT"""
        try:
            # Get news sentiment from market data service
            sentiment_data = await self.market_data.news_service.get_symbol_sentiment(symbol)
            
            # Enhance with FinBERT analysis
            relevant_news = sentiment_data.get("relevant_news", [])
            
            if relevant_news:
                # Analyze with FinBERT
                news_texts = [news.get("headline", "") + " " + news.get("content", "") 
                             for news in relevant_news[:5]]
                
                finbert_results = self.sentiment_analyzer.analyze_news_batch(news_texts)
                
                # Aggregate FinBERT scores
                finbert_scores = [r["finbert_score"] for r in finbert_results if "finbert_score" in r]
                
                if finbert_scores:
                    avg_finbert_score = np.mean(finbert_scores)
                    finbert_confidence = np.mean([abs(score) for score in finbert_scores])
                else:
                    avg_finbert_score = sentiment_data.get("sentiment_score", 0.0)
                    finbert_confidence = 0.5
            else:
                avg_finbert_score = sentiment_data.get("sentiment_score", 0.0)
                finbert_confidence = 0.5
            
            return {
                "finbert_score": avg_finbert_score,
                "finbert_confidence": finbert_confidence,
                "news_count": sentiment_data.get("news_count", 0),
                "latest_headline": sentiment_data.get("latest_headline"),
                "sentiment_score": avg_finbert_score  # For compatibility
            }
            
        except Exception as e:
            logger.error(f"Enhanced sentiment analysis failed for {symbol}: {e}")
            return {"finbert_score": 0.0, "finbert_confidence": 0.0, "news_count": 0, "sentiment_score": 0.0}
    
    async def _get_nifty_correlation_data(self) -> pd.DataFrame:
        """Get Nifty data for correlation analysis"""
        try:
            # Try to get actual Nifty data
            nifty_data = await self.market_data.get_live_market_data("NIFTY50")
            
            if nifty_data and nifty_data.get("quote"):
                # Create simple DataFrame with current Nifty info
                nifty_quote = nifty_data["quote"]
                nifty_df = pd.DataFrame({
                    'close': [nifty_quote.get("prev_close", 18500), nifty_quote.get("ltp", 18500)],
                    'timestamp': [datetime.now() - timedelta(days=1), datetime.now()]
                })
                return nifty_df
            else:
                # Fallback mock data
                return pd.DataFrame({
                    'close': [18500, 18525],
                    'timestamp': [datetime.now() - timedelta(days=1), datetime.now()]
                })
                
        except Exception as e:
            logger.error(f"Nifty correlation data failed: {e}")
            return pd.DataFrame({'close': [18500], 'timestamp': [datetime.now()]})
    
    def _calculate_risk_reward_ratio(self, entry: float, stop: float, target: float, direction: SignalDirection) -> float:
        """Calculate risk-reward ratio"""
        try:
            if direction == SignalDirection.BUY:
                risk = entry - stop
                reward = target - entry
            else:
                risk = stop - entry
                reward = entry - target
            
            return reward / risk if risk > 0 else 2.0
        except:
            return 2.0
    
    def _create_technical_indicators_from_features(self, features: Dict) -> TechnicalIndicators:
        """Create TechnicalIndicators object from features"""
        return TechnicalIndicators(
            rsi_14=features.get("rsi_14", 50.0),
            rsi_21=features.get("rsi_14", 50.0),  # Use same as rsi_14 if not available
            macd_line=features.get("macd_line", 0.0),
            macd_signal=features.get("macd_signal", 0.0),
            macd_histogram=features.get("macd_line", 0.0) - features.get("macd_signal", 0.0),
            bb_upper=features.get("price_level", 0.0) * 1.02,
            bb_middle=features.get("price_level", 0.0),
            bb_lower=features.get("price_level", 0.0) * 0.98,
            bb_width=features.get("bb_position", 0.1) * features.get("price_level", 0.0),
            vwap=features.get("price_level", 0.0),
            adx=features.get("adx", 25.0),
            atr_14=features.get("atr_percent", 2.0) * features.get("price_level", 0.0) / 100,
            stoch_k=50.0,
            stoch_d=50.0,
            cci=0.0,
            williams_r=-50.0,
            momentum_10=features.get("momentum_10d", 0.0),
            roc_10=features.get("momentum_10d", 0.0),
            sma_20=features.get("price_level", 0.0) * (1 + features.get("price_vs_sma20", 0.0)),
            sma_50=features.get("price_level", 0.0) * (1 + features.get("price_vs_sma50", 0.0)),
            ema_12=features.get("price_level", 0.0),
            ema_26=features.get("price_level", 0.0),
            volume_sma_20=1000000.0,
            volume_ratio=features.get("volume_ratio", 1.0)
        )
    
    def _create_fallback_technical_indicators(self, quote: Dict) -> TechnicalIndicators:
        """Create fallback technical indicators when ML features are unavailable"""
        ltp = quote.get("ltp", 0.0)
        return TechnicalIndicators(
            rsi_14=50.0, rsi_21=50.0, macd_line=0.0, macd_signal=0.0, macd_histogram=0.0,
            bb_upper=ltp * 1.02, bb_middle=ltp, bb_lower=ltp * 0.98, bb_width=ltp * 0.04,
            vwap=ltp, adx=25.0, atr_14=ltp * 0.02, stoch_k=50.0, stoch_d=50.0,
            cci=0.0, williams_r=-50.0, momentum_10=0.0, roc_10=0.0,
            sma_20=ltp, sma_50=ltp, ema_12=ltp, ema_26=ltp,
            volume_sma_20=1000000.0, volume_ratio=1.0
        )
    
    async def _create_enhanced_market_context(self, features: Dict) -> MarketContext:
        """Create enhanced market context"""
        return MarketContext(
            nifty_price=features.get("nifty_performance", 0.0) * 18500 + 18500,
            nifty_change_pct=features.get("nifty_performance", 0.0) * 100,
            nifty_volume_ratio=1.0,
            sector_vs_nifty=features.get("relative_to_nifty", 0.0) * 100,
            vix_level=15.0 + features.get("sentiment_strength", 0.0) * 10,
            fii_flow=None,
            dii_flow=None
        )
    
    async def _create_fallback_market_context(self) -> MarketContext:
        """Create fallback market context"""
        return MarketContext(
            nifty_price=18500.0, nifty_change_pct=0.0, nifty_volume_ratio=1.0,
            sector_vs_nifty=0.0, vix_level=15.0, fii_flow=None, dii_flow=None
        )
    
    def _create_pre_market_data_from_features(self, features: Dict, sentiment: Dict) -> PreMarketData:
        """Create pre-market data from features"""
        return PreMarketData(
            gap_percentage=features.get("price_gap_percent", 0.0),
            pre_market_volume=features.get("volume_ratio", 1.0) * 1000000,
            news_sentiment_score=sentiment.get("finbert_score", 0.0),
            news_count=sentiment.get("news_count", 0),
            social_sentiment=None,
            analyst_rating_change=None
        )
    
    def _create_fallback_pre_market_data(self, stock_data: Dict) -> PreMarketData:
        """Create fallback pre-market data"""
        return PreMarketData(
            gap_percentage=stock_data.get("gap_percentage", 0.0),
            pre_market_volume=stock_data.get("volume_ratio", 1.0) * 1000000,
            news_sentiment_score=stock_data.get("sentiment_score", 0.0),
            news_count=0,
            social_sentiment=None,
            analyst_rating_change=None
        )
    
    # ================================================================
    # Model Management and Performance Tracking
    # ================================================================
    
    async def retrain_model_if_needed(self):
        """Retrain ML model if performance degrades (regime-aware)"""
        try:
            # Check if we have enough new data for retraining
            performance_window = self.model_performance["accuracy_tracking"][-10:]
            
            if len(performance_window) >= 10:
                recent_accuracy = np.mean(performance_window)
                if recent_accuracy < 0.6:  # Below 60% accuracy
                    logger.warning(f"⚠️ Model performance degraded: {recent_accuracy:.1%}")
                    logger.info("🔄 Initiating regime-aware model retraining...")
                    
                    # Collect new training data with regime information
                    data_collector = TrainingDataCollector(self.market_data, self.signal_logger)
                    training_data = await data_collector.collect_training_data(lookback_days=30)
                    
                    if len(training_data) > 100:
                        # Add regime information to training data
                        training_data = self._add_regime_training_features(training_data)
                        
                        # Retrain model
                        feature_columns = [col for col in training_data.columns 
                                         if col not in ['symbol', 'date', 'entry_price', 'max_future_gain', 'profitable']]
                        
                        results = self.ml_model.train_model(training_data[feature_columns + ['profitable']])
                        logger.info(f"✅ Regime-aware model retrained - New accuracy: {results['accuracy']:.1%}")
                    
        except Exception as e:
            logger.error(f"Regime-aware model retraining failed: {e}")
    
    def _add_regime_training_features(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Add regime features to training data"""
        try:
            # Add regime features based on historical data (simplified for now)
            all_regimes = ["TRENDING_BULLISH", "TRENDING_BEARISH", "SIDEWAYS_CHOPPY", "GAP_DAY", "HIGH_VOLATILITY", "LOW_VOLATILITY"]
            
            for regime in all_regimes:
                training_data[f"regime_{regime.lower()}"] = 0.0  # Default to 0
            
            # For now, set current regime to 1 (in production, you'd classify historical regimes)
            current_regime = self.regime_context.get("regime", "SIDEWAYS_CHOPPY")
            training_data[f"regime_{current_regime.lower()}"] = 1.0
            training_data["regime_confidence"] = self.regime_context.get("confidence", 0.5)
            training_data["regime_risk_adjustment"] = self.regime_context.get("risk_adjustment", 1.0)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to add regime training features: {e}")
            return training_data
    
    def get_regime_performance_summary(self) -> Dict:
        """Get performance summary by regime"""
        try:
            return {
                "current_regime": self.regime_context.get("regime", "UNKNOWN"),
                "regime_confidence": self.regime_context.get("confidence", 0.5),
                "trading_strategy": self.regime_context.get("trading_strategy", "MEAN_REVERSION"),
                "regime_adjustments": self.regime_configs.get(
                    self.regime_context.get("regime", "SIDEWAYS_CHOPPY"), 
                    self.regime_configs["SIDEWAYS_CHOPPY"]
                ),
                "regime_performance": self.model_performance["regime_performance"],
                "current_parameters": {
                    "min_ml_confidence": self.min_ml_confidence,
                    "max_signals_per_day": self.max_signals_per_day,
                    "risk_adjustment": self.regime_context.get("risk_adjustment", 1.0)
                },
                "daily_stats": {
                    "signals_generated_today": self.daily_signal_count,
                    "signals_remaining": max(0, self.max_signals_per_day - self.daily_signal_count),
                    "last_signal_date": self.last_signal_date.isoformat() if self.last_signal_date else None
                },
                "model_performance": {
                    "total_predictions": self.model_performance["total_predictions"],
                    "high_confidence_predictions": self.model_performance["high_confidence_predictions"],
                    "confidence_rate": (
                        self.model_performance["high_confidence_predictions"] / 
                        max(1, self.model_performance["total_predictions"])
                    )
                }
            }
        except Exception as e:
            logger.error(f"Failed to get regime performance summary: {e}")
            return {}
    
    def update_signal_outcome(self, signal_id: str, outcome: TradeOutcome):
        """Update signal outcome for performance tracking"""
        try:
            # Extract regime from signal_id
            if "REGIME_" in signal_id:
                regime = signal_id.split("REGIME_")[1].split("_")[0]
                
                if regime in self.model_performance["regime_performance"]:
                    regime_stats = self.model_performance["regime_performance"][regime]
                    
                    # Add outcome tracking
                    if "outcomes" not in regime_stats:
                        regime_stats["outcomes"] = []
                    
                    regime_stats["outcomes"].append({
                        "signal_id": signal_id,
                        "profitable": outcome.profitable,
                        "pnl": outcome.realized_pnl,
                        "timestamp": datetime.now()
                    })
                    
                    # Update accuracy tracking
                    recent_outcomes = regime_stats["outcomes"][-20:]  # Last 20 outcomes
                    if len(recent_outcomes) >= 5:
                        accuracy = sum(1 for o in recent_outcomes if o["profitable"]) / len(recent_outcomes)
                        self.model_performance["accuracy_tracking"].append(accuracy)
                        
                        logger.info(f"📊 Regime {regime} accuracy: {accuracy:.1%} (last {len(recent_outcomes)} signals)")
            
        except Exception as e:
            logger.error(f"Failed to update signal outcome: {e}")
    
    # ================================================================
    # Public Interface for Service Manager
    # ================================================================
    
    def get_current_regime_context(self) -> Dict:
        """Get current regime context (for service manager)"""
        return self.regime_context.copy()
    
    async def health_check(self) -> Dict:
        """Health check for service manager"""
        try:
            return {
                "status": "healthy",
                "regime": self.regime_context.get("regime", "UNKNOWN"),
                "ml_model_loaded": hasattr(self.ml_model, 'model') and self.ml_model.model is not None,
                "daily_signals_generated": self.daily_signal_count,
                "max_daily_signals": self.max_signals_per_day,
                "confidence_threshold": self.min_ml_confidence,
                "total_predictions": self.model_performance["total_predictions"],
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}