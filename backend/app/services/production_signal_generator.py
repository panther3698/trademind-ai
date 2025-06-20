# ================================================================
# backend/app/services/production_signal_generator.py
# Updated Signal Generator with ML Models
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
    TrainingDataCollector
)

# Import existing components
from app.core.signal_logger import (
    InstitutionalSignalLogger, SignalRecord, TechnicalIndicators, 
    MarketContext, PreMarketData, SignalDirection, TradeOutcome
)
from app.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

class ProductionMLSignalGenerator:
    """
    Production ML-based Signal Generator using:
    - Nifty 100 stock universe
    - FinBERT sentiment analysis
    - XGBoost prediction models
    - Comprehensive feature engineering
    """
    
    def __init__(self, market_data_service: MarketDataService, signal_logger: InstitutionalSignalLogger):
        self.market_data = market_data_service
        self.signal_logger = signal_logger
        
        # Initialize ML components
        self.stock_universe = Nifty100StockUniverse()
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        self.feature_engineer = FeatureEngineering(self.stock_universe)
        self.ml_model = XGBoostSignalModel()
        
        # Signal generation parameters
        self.min_ml_confidence = 0.70  # 70% ML confidence threshold
        self.max_signals_per_day = 3
        self.max_capital_per_trade = 100000  # ₹1 lakh per trade
        
        # Track daily activity
        self.daily_signal_count = 0
        self.last_signal_date = None
        
        # Performance tracking
        self.model_performance = {
            "total_predictions": 0,
            "high_confidence_predictions": 0,
            "accuracy_tracking": []
        }
        
        # Initialize model (load if exists, train if needed)
        self._initialize_ml_model()
    
    def _initialize_ml_model(self):
        """Initialize or load ML model"""
        try:
            # Try to load existing model
            if self.ml_model.load_model():
                logger.info("✅ Loaded existing XGBoost model")
            else:
                logger.warning("⚠️ No trained model found. Model will be trained on first run.")
                # In production, you'd want to train immediately
                # For now, we'll use a fallback scoring method
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
    
    async def generate_signals(self) -> List[SignalRecord]:
        """
        Generate trading signals using ML pipeline for Nifty 100 stocks
        
        Returns:
            List of high-confidence ML-generated signals
        """
        try:
            # Reset daily counter if new day
            today = datetime.now().date()
            if self.last_signal_date != today:
                self.daily_signal_count = 0
                self.last_signal_date = today
                logger.info(f"🌅 New trading day: {today}")
            
            # Check daily limits
            if self.daily_signal_count >= self.max_signals_per_day:
                logger.info(f"Daily signal limit reached ({self.max_signals_per_day})")
                return []
            
            # Check market status
            await self.market_data.update_market_status()
            if self.market_data.market_status.value != "OPEN":
                logger.info(f"Market closed: {self.market_data.market_status.value}")
                return []
            
            logger.info("🔍 Scanning Nifty 100 stocks for ML signals...")
            
            # Get top opportunity stocks using pre-market analysis
            opportunity_stocks = await self._get_top_opportunity_stocks()
            
            signals = []
            
            # Analyze top opportunity stocks with ML
            for stock_data in opportunity_stocks[:20]:  # Top 20 opportunities
                try:
                    symbol = stock_data["symbol"]
                    signal = await self._analyze_stock_with_ml(symbol, stock_data)
                    
                    if signal:
                        # Validate signal with risk management
                        risk_check = self.signal_logger.check_risk_limits(signal)
                        
                        if risk_check["overall_risk_ok"]:
                            signals.append(signal)
                            self.daily_signal_count += 1
                            
                            # Log the signal
                            success = self.signal_logger.log_signal(signal)
                            if success:
                                logger.info(f"🎯 ML Signal: {signal.ticker} {signal.direction.value} "
                                          f"ML Confidence: {signal.ml_confidence:.1%} | "
                                          f"Entry: ₹{signal.entry_price}")
                            
                            # Stop if daily limit reached
                            if self.daily_signal_count >= self.max_signals_per_day:
                                break
                        else:
                            logger.warning(f"⚠️ Signal blocked by risk limits: {symbol}")
                
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
            
            if signals:
                logger.info(f"🚀 Generated {len(signals)} ML signals for {today}")
            else:
                logger.info("📊 No ML signals met confidence threshold today")
            
            return signals
            
        except Exception as e:
            logger.error(f"ML signal generation failed: {e}")
            return []
    
    async def _get_top_opportunity_stocks(self) -> List[Dict]:
        """
        Get top opportunity stocks from Nifty 100 using pre-market analysis
        
        Returns:
            List of stock opportunities sorted by ML potential
        """
        try:
            logger.info("📋 Running pre-market analysis on Nifty 100...")
            
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
                    
                    # Calculate basic opportunity score
                    gap_pct = ((quote["ltp"] - quote["prev_close"]) / quote["prev_close"]) * 100
                    volume_ratio = quote.get("volume", 1) / 1000000  # Normalize volume
                    
                    # Get news sentiment
                    sentiment_data = market_data.get("sentiment", {})
                    sentiment_score = sentiment_data.get("sentiment_score", 0.0)
                    
                    # Simple opportunity scoring
                    opportunity_score = (
                        abs(gap_pct) * 0.4 +  # Price movement
                        min(volume_ratio, 3.0) * 0.3 +  # Volume (capped at 3x)
                        abs(sentiment_score) * 0.3  # Sentiment strength
                    )
                    
                    opportunities.append({
                        "symbol": symbol,
                        "opportunity_score": opportunity_score,
                        "gap_percentage": gap_pct,
                        "volume_ratio": volume_ratio,
                        "sentiment_score": sentiment_score,
                        "current_price": quote["ltp"],
                        "sector": self.stock_universe.get_sector(symbol),
                        "market_data": market_data
                    })
                    
                    processed += 1
                    if processed % 20 == 0:
                        logger.info(f"Analyzed {processed}/{len(all_stocks)} stocks...")
                    
                    # Rate limiting
                    await asyncio.sleep(0.05)  # 50ms delay
                    
                except Exception as e:
                    logger.debug(f"Opportunity analysis failed for {symbol}: {e}")
                    continue
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
            
            logger.info(f"✅ Analyzed {len(opportunities)} stocks, top 10 opportunities:")
            for i, opp in enumerate(opportunities[:10]):
                logger.info(f"  {i+1}. {opp['symbol']} - Score: {opp['opportunity_score']:.2f} "
                          f"Gap: {opp['gap_percentage']:.1f}% Sentiment: {opp['sentiment_score']:.2f}")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Opportunity analysis failed: {e}")
            return []
    
    async def _analyze_stock_with_ml(self, symbol: str, stock_data: Dict) -> Optional[SignalRecord]:
        """
        Analyze individual stock using ML pipeline
        
        Args:
            symbol: Stock symbol
            stock_data: Pre-computed stock data from opportunity analysis
            
        Returns:
            SignalRecord if ML confidence is high enough
        """
        try:
            logger.debug(f"🤖 ML analysis for {symbol}...")
            
            market_data = stock_data["market_data"]
            quote = market_data["quote"]
            
            # Get historical data for technical analysis
            historical_data = await self.market_data.zerodha.get_historical_data(
                symbol, "minute", datetime.now() - timedelta(days=30)
            )
            
            if len(historical_data) < 50:
                logger.debug(f"Insufficient historical data for {symbol}")
                return None
            
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
            
            # Engineer comprehensive features
            features = self.feature_engineer.engineer_features(
                symbol, df, current_quote, news_sentiment, nifty_data
            )
            
            if not features:
                logger.debug(f"Feature engineering failed for {symbol}")
                return None
            
            # ML prediction
            ml_probability, prediction_details = self.ml_model.predict_signal_probability(features)
            
            # Track prediction
            self.model_performance["total_predictions"] += 1
            if ml_probability > self.min_ml_confidence:
                self.model_performance["high_confidence_predictions"] += 1
            
            # Check ML confidence threshold
            if ml_probability < self.min_ml_confidence:
                logger.debug(f"{symbol} ML confidence {ml_probability:.1%} below threshold {self.min_ml_confidence:.1%}")
                return None
            
            # Determine signal direction based on features and ML output
            direction = self._determine_signal_direction(features, ml_probability)
            
            # Calculate price levels using enhanced logic
            entry_price = quote["ltp"]
            stop_loss, target_price = self._calculate_ml_price_levels(
                entry_price, features, direction, ml_probability
            )
            
            # Calculate position sizing based on ML confidence
            position_size = self._calculate_ml_position_size(
                entry_price, stop_loss, ml_probability, features
            )
            
            # Create comprehensive signal record
            signal = SignalRecord(
                signal_id=f"ML_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}",
                timestamp=datetime.now(),
                ticker=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                ml_confidence=ml_probability,
                technical_score=features.get("technical_composite_score", 0.0),
                sentiment_score=news_sentiment.get("finbert_score", 0.0),
                macro_score=features.get("nifty_performance", 0.0),
                final_score=ml_probability,  # ML probability is our final score
                indicators=self._create_technical_indicators_from_features(features),
                market_context=await self._create_enhanced_market_context(features),
                pre_market=self._create_pre_market_data_from_features(features, news_sentiment),
                risk_reward_ratio=self._calculate_risk_reward_ratio(entry_price, stop_loss, target_price, direction),
                position_size_suggested=position_size,
                capital_at_risk=abs(entry_price - stop_loss) * position_size,
                model_version="v1.0_ML_production",
                signal_source="XGBOOST_FINBERT",
                notes=f"Top features: {list(prediction_details.get('feature_contributions', {}).keys())[:3]}"
            )
            
            logger.debug(f"✅ {symbol} ML Signal: {direction.value} Confidence: {ml_probability:.1%}")
            return signal
            
        except Exception as e:
            logger.error(f"ML analysis failed for {symbol}: {e}")
            return None
    
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
            return {"finbert_score": 0.0, "finbert_confidence": 0.0, "news_count": 0}
    
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
    
    def _determine_signal_direction(self, features: Dict, ml_probability: float) -> SignalDirection:
        """Determine signal direction based on features and ML output"""
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
            
            # If technical signals strongly contradict ML, be cautious
            if base_direction == SignalDirection.BUY and technical_bullish_signals < 2:
                logger.debug("Technical indicators suggest caution for BUY signal")
            elif base_direction == SignalDirection.SELL and technical_bullish_signals > 2:
                logger.debug("Technical indicators suggest caution for SELL signal")
            
            return base_direction
            
        except Exception as e:
            logger.error(f"Direction determination failed: {e}")
            return SignalDirection.BUY  # Default
    
    def _calculate_ml_price_levels(self, 
                                  entry_price: float, 
                                  features: Dict, 
                                  direction: SignalDirection,
                                  ml_confidence: float) -> Tuple[float, float]:
        """Calculate stop loss and target using ML-enhanced logic"""
        try:
            # Base ATR for stop loss
            atr_percent = features.get("atr_percent", 2.0)
            
            # Adjust stop loss based on ML confidence and volatility
            confidence_multiplier = 0.8 + (ml_confidence * 0.4)  # 0.8 to 1.2
            volatility_multiplier = min(2.0, max(0.5, atr_percent / 2.0))
            
            stop_distance_percent = 1.5 * confidence_multiplier * volatility_multiplier
            
            # Calculate target with enhanced risk-reward
            base_rr_ratio = 2.0
            confidence_rr_bonus = (ml_confidence - 0.5) * 2  # Extra reward for high confidence
            target_rr_ratio = base_rr_ratio + confidence_rr_bonus
            
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
            logger.error(f"ML price levels calculation failed: {e}")
            # Fallback to simple percentage
            if direction == SignalDirection.BUY:
                return entry_price * 0.985, entry_price * 1.04
            else:
                return entry_price * 1.015, entry_price * 0.96
    
    def _calculate_ml_position_size(self, 
                                   entry_price: float, 
                                   stop_loss: float, 
                                   ml_confidence: float,
                                   features: Dict) -> float:
        """Calculate position size based on ML confidence and risk"""
        try:
            # Base risk per trade
            base_risk = 2500  # ₹2500 base risk
            
            # Adjust risk based on ML confidence
            confidence_risk_multiplier = 0.6 + (ml_confidence * 0.8)  # 0.6 to 1.4
            adjusted_risk = base_risk * confidence_risk_multiplier
            
            # Adjust for volatility
            volatility = features.get("atr_percent", 2.0)
            volatility_multiplier = min(1.5, max(0.5, 2.0 / volatility))
            final_risk = adjusted_risk * volatility_multiplier
            
            # Calculate position size
            risk_per_share = abs(entry_price - stop_loss)
            position_size = final_risk / risk_per_share
            
            # Cap position value
            max_position_value = min(self.max_capital_per_trade, position_size * entry_price)
            final_position_size = max_position_value / entry_price
            
            return max(1, int(final_position_size))
            
        except Exception as e:
            logger.error(f"ML position sizing failed: {e}")
            return 20  # Default
    
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
    
    async def retrain_model_if_needed(self):
        """Retrain ML model if performance degrades"""
        try:
            # Check if we have enough new data for retraining
            performance_window = self.model_performance["accuracy_tracking"][-10:]
            
            if len(performance_window) >= 10:
                recent_accuracy = np.mean(performance_window)
                if recent_accuracy < 0.6:  # Below 60% accuracy
                    logger.warning(f"⚠️ Model performance degraded: {recent_accuracy:.1%}")
                    logger.info("🔄 Initiating model retraining...")
                    
                    # Collect new training data
                    data_collector = TrainingDataCollector(self.market_data, self.signal_logger)
                    training_data = await data_collector.collect_training_data(lookback_days=30)
                    
                    if len(training_data) > 100:
                        # Retrain model
                        feature_columns = [col for col in training_data.columns 
                                         if col not in ['symbol', 'date', 'entry_price', 'max_future_gain', 'profitable']]
                        
                        results = self.ml_model.train_model(training_data[feature_columns + ['profitable']])
                        logger.info(f"✅ Model retrained - New accuracy: {results['accuracy']:.1%}")
                    
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
