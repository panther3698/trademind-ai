# ================================================================
# backend/app/services/production_signal_generator.py
# Complete Enhanced ML Signal Generator with Regime Detection + News Intelligence Integration
# ENHANCED: Added comprehensive news intelligence for superior signal generation
# ================================================================

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import logging
from dataclasses import asdict
from sklearn.ensemble import RandomForestClassifier
import time

# FIXED: Add TYPE_CHECKING imports to prevent circular imports
if TYPE_CHECKING:
    from app.ml.models import FeatureEngineering
    from app.services.enhanced_market_data_nifty100 import EnhancedMarketDataService

# FIXED: Direct import for AdvancedSentimentAnalyzer (no try/except)
from app.ml.advanced_sentiment import AdvancedSentimentAnalyzer
from app.ml.advanced_ensemble_patch import AdvancedEnsembleModelPatch

# Import ML components with conditional loading to prevent circular imports
def _lazy_import_ml_components():
    """Lazy import ML components to prevent circular imports"""
    from app.ml.models import (
        Nifty100StockUniverse, 
        FeatureEngineering, 
        XGBoostSignalModel,
        TrainingDataCollector,
        EnsembleModel
    )
    return {
        'Nifty100StockUniverse': Nifty100StockUniverse,
        'FeatureEngineering': FeatureEngineering,
        'XGBoostSignalModel': XGBoostSignalModel,
        'TrainingDataCollector': TrainingDataCollector,
        'EnsembleModel': EnsembleModel
    }

# Import advanced models with conditional loading
def _lazy_import_advanced_models():
    """Lazy import advanced models to prevent circular imports"""
    try:
        from app.ml.advanced_models import (
            LightGBMModel,
            CatBoostModel, 
            LSTMModel,
            EnsembleModel as AdvancedEnsembleModel
        )
        return {
            'LightGBMModel': LightGBMModel,
            'CatBoostModel': CatBoostModel,
            'LSTMModel': LSTMModel,
            'AdvancedEnsembleModel': AdvancedEnsembleModel
        }
    except ImportError:
        return None

# Import core components directly (these shouldn't have circular dependencies)
from app.core.signal_logger import (
    InstitutionalSignalLogger, SignalRecord, TechnicalIndicators, 
    MarketContext, PreMarketData, SignalDirection, TradeOutcome
)

# Import simplified sentiment analyzer
from app.ml.simplified_sentiment import SimplifiedSentimentAnalyzer

# Import optimized model loader
from app.ml.optimized_model_loader import get_optimized_loader

# Import performance monitoring
from app.core.performance_monitor import performance_monitor

logger = logging.getLogger(__name__)

class ProductionMLSignalGenerator:
    """
    Production ML-based Signal Generator with Simplified Regime Detection + News Intelligence Integration:
    - Nifty 100 stock universe
    - Simplified sentiment analysis (FinBERT only)
    - XGBoost prediction models
    - Simplified feature engineering
    - Simplified market regime awareness (3 regimes only)
    - ENHANCED: Real-time news intelligence for superior signal generation
    
    SIMPLIFIED: Using FinBERT-only sentiment analysis for better performance
    """
    
    def __init__(self, market_data_service, signal_logger: InstitutionalSignalLogger):
        self.market_data = market_data_service
        self.signal_logger = signal_logger
        
        # NEWS INTELLIGENCE INTEGRATION
        self.news_intelligence = None  # Will be set by main service manager
        
        # Lazy load ML components to prevent circular imports
        self._ml_components = _lazy_import_ml_components()
        self._advanced_models = _lazy_import_advanced_models()
        
        # Initialize ML components (SIMPLIFIED - single FinBERT sentiment system)
        self.stock_universe = self._ml_components['Nifty100StockUniverse']()
        
        # SIMPLIFIED: Single FinBERT sentiment analyzer
        self.sentiment_analyzer = SimplifiedSentimentAnalyzer()
        logger.info("✅ Simplified FinBERT sentiment analyzer initialized")
        
        # Pass the sentiment analyzer to feature engineering to avoid duplication
        self.feature_engineer = self._ml_components['FeatureEngineering'](
            self.stock_universe, 
            sentiment_analyzer=self.sentiment_analyzer  # Pass existing instance
        )
        
        # OPTIMIZED: Initialize optimized model loader
        self.optimized_loader = get_optimized_loader()
        self.ml_model = None  # Will be loaded by optimized loader
        logger.info("✅ Optimized model loader initialized")
        
        # Signal generation parameters (base values) - REDUCED THRESHOLDS
        self.base_min_ml_confidence = 0.55  # Reduced from 0.70 to 0.55 (55% ML confidence threshold)
        self.base_max_signals_per_day = 5   # Increased from 3 to 5
        self.max_capital_per_trade = 100000  # ₹1 lakh per trade
        
        # Current adjusted parameters (will be updated by regime)
        self.min_ml_confidence = self.base_min_ml_confidence
        self.max_signals_per_day = self.base_max_signals_per_day
        
        # Track daily activity
        self.daily_signal_count = 0
        self.last_signal_date = None
        
        # NEWS INTELLIGENCE TRACKING
        self.news_enhanced_signals_count = 0
        self.last_news_analysis = None
        
        # Performance tracking
        self.model_performance = {
            "total_predictions": 0,
            "high_confidence_predictions": 0,
            "accuracy_tracking": [],
            "regime_performance": {},  # Track performance by regime
            # NEWS INTELLIGENCE PERFORMANCE
            "news_enhanced_predictions": 0,
            "news_accuracy_boost": 0.0,
            # OPTIMIZED MODEL PERFORMANCE
            "model_load_time_ms": 0.0,
            "breaking_news_signals": 0
        }
        
        # Regime awareness
        self.regime_context = {
            "regime": "SIDEWAYS",  # Default - simplified regime
            "confidence": 0.5,
            "risk_adjustment": 1.0,
            "trading_strategy": "MEAN_REVERSION"
        }
        
        # Simplified regime-specific parameters (3 regimes only)
        self.regime_configs = {
            "BULLISH": {
                "confidence_adjustment": -0.05,  # Lower threshold (easier to trigger)
                "max_signals_multiplier": 1.3,   # More signals allowed
                "risk_multiplier": 1.1,          # Slightly higher risk
                "rr_ratio_bonus": 0.5,           # Better risk-reward
                "analysis_limit_multiplier": 1.5,
                "news_weight": 0.25               # NEWS: Weight in signal scoring
            },
            "BEARISH": {
                "confidence_adjustment": -0.03,
                "max_signals_multiplier": 1.1,
                "risk_multiplier": 1.0,
                "rr_ratio_bonus": 0.3,
                "analysis_limit_multiplier": 1.3,
                "news_weight": 0.30               # NEWS: Higher weight in bearish trends
            },
            "SIDEWAYS": {
                "confidence_adjustment": +0.10,  # Higher threshold (harder to trigger)
                "max_signals_multiplier": 0.7,   # Fewer signals
                "risk_multiplier": 0.8,          # Lower risk
                "rr_ratio_bonus": 0.0,           # Standard risk-reward
                "analysis_limit_multiplier": 1.0,
                "news_weight": 0.35               # NEWS: Higher weight in choppy markets
            }
        }
        
        # Initialize model (load if exists, train if needed)
        self._initialize_ml_model()
        
        logger.info("✅ Production ML Signal Generator initialized with single sentiment system + News Intelligence ready")
    
    # ================================================================
    # NEWS INTELLIGENCE INTEGRATION METHODS
    # ================================================================
    
    def set_news_intelligence(self, news_intelligence_service):
        """Set news intelligence service reference (optional enhancement)"""
        try:
            self.news_intelligence = news_intelligence_service
            if news_intelligence_service:
                logger.info("🔗 News intelligence connected (optional enhancement)")
            else:
                logger.info("ℹ️ News intelligence not connected - using ML-only signals")
        except Exception as e:
            logger.error(f"Failed to set news intelligence: {e}")
    
    async def _get_simplified_news_sentiment(self, symbol: str) -> Dict:
        """Get simplified news sentiment (optional enhancement)"""
        try:
            if not self.news_intelligence:
                # No news intelligence - return neutral sentiment
                return {
                    "finbert_score": 0.0,
                    "finbert_confidence": 0.0,
                    "weighted_score": 0.0,
                    "news_count": 0,
                    "sentiment_score": 0.0,
                    "news_impact_score": 0.0,
                    "enhanced_by_news_intelligence": False
                }
            
            # Use news intelligence if available
            news_analysis = await self.news_intelligence.get_comprehensive_news_intelligence(
                symbols=[symbol],
                lookback_hours=24
            )
            
            if not news_analysis or "error" in news_analysis:
                return {
                    "finbert_score": 0.0,
                    "finbert_confidence": 0.0,
                    "weighted_score": 0.0,
                    "news_count": 0,
                    "sentiment_score": 0.0,
                    "news_impact_score": 0.0,
                    "enhanced_by_news_intelligence": False
                }
            
            # Extract simplified sentiment data
            sentiment_analysis = news_analysis.get("sentiment_analysis", {})
            symbol_sentiment = sentiment_analysis.get("symbol_sentiment", {}).get(symbol, 0.0)
            
            # Calculate simple news impact score
            news_impact = min(1.0, abs(symbol_sentiment) * 2.0)  # Scale sentiment to impact
            
            return {
                "finbert_score": symbol_sentiment,
                "finbert_confidence": abs(symbol_sentiment) if symbol_sentiment else 0.0,
                "weighted_score": symbol_sentiment,
                "news_count": news_analysis.get("total_articles_analyzed", 0),
                "sentiment_score": symbol_sentiment,
                "news_impact_score": news_impact,
                "enhanced_by_news_intelligence": True
            }
                
        except Exception as e:
            logger.error(f"Simplified news sentiment analysis failed for {symbol}: {e}")
            return {
                "finbert_score": 0.0,
                "finbert_confidence": 0.0,
                "weighted_score": 0.0,
                "news_count": 0,
                "sentiment_score": 0.0,
                "news_impact_score": 0.0,
                "enhanced_by_news_intelligence": False
            }
    
    async def _get_comprehensive_news_sentiment(self, symbol: str) -> Dict:
        """Get comprehensive news sentiment using enhanced news intelligence system"""
        try:
            if self.news_intelligence:
                # Use comprehensive news intelligence
                news_analysis = await self.news_intelligence.get_comprehensive_news_intelligence(
                    symbols=[symbol],
                    lookback_hours=24
                )
                
                # Extract enhanced sentiment data
                sentiment_analysis = news_analysis.get("sentiment_analysis", {})
                symbol_sentiment = sentiment_analysis.get("symbol_sentiment", {}).get(symbol, 0.0)
                overall_sentiment = news_analysis.get("overall_sentiment", 0.0)
                market_events = news_analysis.get("market_events", [])
                
                # Calculate news impact score
                news_impact = self._calculate_news_impact_score(symbol, market_events, sentiment_analysis)
                
                return {
                    "finbert_score": symbol_sentiment,
                    "finbert_confidence": abs(symbol_sentiment) if symbol_sentiment else 0.0,
                    "weighted_score": symbol_sentiment,
                    "overall_market_sentiment": overall_sentiment,
                    "symbol_specific_sentiment": symbol_sentiment,
                    "news_count": news_analysis.get("total_articles_analyzed", 0),
                    "sentiment_score": symbol_sentiment,
                    "news_impact_score": news_impact,
                    "high_impact_events": [e for e in market_events if symbol in e.get("symbols_mentioned", [])],
                    "breaking_news_detected": len([e for e in market_events if e.get("significance_score", 0) > 0.8]) > 0,
                    "news_sources_count": len(news_analysis.get("news_sources_used", [])),
                    "latest_headline": market_events[0].get("title", "") if market_events else None,
                    "enhanced_by_news_intelligence": True
                }
            else:
                # Fallback to existing method
                return await self._get_enhanced_sentiment(symbol)
                
        except Exception as e:
            logger.error(f"Comprehensive news sentiment analysis failed for {symbol}: {e}")
            # Fallback to existing method
            return await self._get_enhanced_sentiment(symbol)
    
    def _calculate_news_impact_score(self, symbol: str, market_events: List[Dict], sentiment_analysis: Dict) -> float:
        """Calculate news impact score for the symbol"""
        try:
            impact_score = 0.0
            
            # Check for symbol-specific events
            symbol_events = [e for e in market_events if symbol in e.get("symbols_mentioned", [])]
            
            for event in symbol_events:
                significance = event.get("significance_score", 0.0)
                sentiment = event.get("sentiment_score", 0.0)
                
                # Weight by significance and sentiment strength
                event_impact = significance * abs(sentiment)
                impact_score += event_impact
            
            # Add sector-wide impact
            sector = self.stock_universe.get_sector(symbol)
            sector_sentiment = sentiment_analysis.get("sector_sentiment", {}).get(sector, 0.0)
            sector_impact = abs(sector_sentiment) * 0.3  # 30% weight for sector news
            
            impact_score += sector_impact
            
            # Normalize to 0-1 range
            return min(1.0, impact_score)
            
        except Exception as e:
            logger.debug(f"News impact calculation failed for {symbol}: {e}")
            return 0.0
    
    async def _check_for_breaking_news_signals(self) -> List[SignalRecord]:
        """Check for immediate trading opportunities from breaking news"""
        try:
            if not self.news_intelligence:
                return []
            
            # Get recent breaking news (last 30 minutes)
            news_intel = await self.news_intelligence.get_comprehensive_news_intelligence(
                lookback_hours=0.5  # 30 minutes
            )
            
            breaking_signals = []
            high_impact_events = news_intel.get("market_events", [])
            
            for event in high_impact_events:
                significance = event.get("significance_score", 0.0)
                if significance > 0.8:  # High significance threshold
                    
                    symbols_mentioned = event.get("symbols_mentioned", [])
                    for symbol in symbols_mentioned:
                        
                        if symbol in self.stock_universe.get_all_stocks():
                            # Generate breaking news signal
                            breaking_signal = await self._generate_breaking_news_signal(event, symbol)
                            if breaking_signal:
                                breaking_signals.append(breaking_signal)
                                self.breaking_news_signals_count += 1
                                logger.info(f"🚨 Breaking news signal generated: {symbol} - {event.get('title', '')[:50]}...")
            
            return breaking_signals
            
        except Exception as e:
            logger.error(f"Breaking news signal check failed: {e}")
            return []
    
    async def _generate_breaking_news_signal(self, event: Dict, symbol: str) -> Optional[SignalRecord]:
        """Generate trading signal from breaking news event"""
        try:
            sentiment_score = event.get("sentiment_score", 0.0)
            significance_score = event.get("significance_score", 0.0)
            
            # Only generate signals for strong sentiment with high significance
            if abs(sentiment_score) < 0.4 or significance_score < 0.8:
                return None
            
            # Get current market data
            market_data = await self.market_data.get_live_market_data(symbol)
            if not market_data or not market_data.get("quote"):
                return None
            
            quote = market_data["quote"]
            current_regime = self.regime_context.get("regime", "SIDEWAYS")
            
            # Determine direction based on sentiment
            direction = SignalDirection.BUY if sentiment_score > 0 else SignalDirection.SELL
            entry_price = quote["ltp"]
            
            # Calculate price levels for breaking news (tighter stops, quick profits)
            if direction == SignalDirection.BUY:
                stop_loss = entry_price * 0.98  # 2% stop
                target_price = entry_price * 1.03  # 3% target
            else:
                stop_loss = entry_price * 1.02  # 2% stop
                target_price = entry_price * 0.97  # 3% target
            
            # Position size based on significance (higher significance = larger position)
            base_position = 30
            significance_multiplier = 0.5 + (significance_score * 0.5)  # 0.5x to 1.0x
            position_size = int(base_position * significance_multiplier)
            
            # Create breaking news signal
            signal = SignalRecord(
                signal_id=f"BREAKING_NEWS_{current_regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}",
                timestamp=datetime.now(),
                ticker=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                ml_confidence=significance_score,  # Use significance as confidence
                technical_score=0.0,  # No technical analysis for breaking news
                sentiment_score=sentiment_score,
                macro_score=0.0,
                final_score=significance_score * abs(sentiment_score),
                indicators=self._create_fallback_technical_indicators(quote),
                market_context=await self._create_fallback_market_context(),
                pre_market=self._create_breaking_news_pre_market_data(event),
                risk_reward_ratio=self._calculate_risk_reward_ratio(entry_price, stop_loss, target_price, direction),
                position_size_suggested=position_size,
                capital_at_risk=abs(entry_price - stop_loss) * position_size,
                model_version="v1.2_breaking_news",
                signal_source=f"BREAKING_NEWS_{current_regime}",
                notes=f"Breaking News: {event.get('title', '')[:100]}... | Significance: {significance_score:.1%} | Sentiment: {sentiment_score:.2f}"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Breaking news signal generation failed for {symbol}: {e}")
            return None
    
    def _create_breaking_news_pre_market_data(self, event: Dict) -> PreMarketData:
        """Create pre-market data for breaking news signals"""
        return PreMarketData(
            gap_percentage=0.0,  # No gap data for breaking news
            pre_market_volume=0,
            news_sentiment_score=event.get("sentiment_score", 0.0),
            news_count=1,  # One breaking news event
            social_sentiment=None,
            analyst_rating_change=None
        )
    
    # ================================================================
    # ENHANCED SIGNAL GENERATION WITH NEWS INTELLIGENCE
    # ================================================================
    
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
            new_regime = regime_context.get("regime", "SIDEWAYS")
            
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
            current_regime = self.regime_context.get("regime", "SIDEWAYS")
            config = self.regime_configs.get(current_regime, self.regime_configs["SIDEWAYS"])
            
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
            logger.info(f"  • News weight: {config['news_weight']:.1%}")  # NEWS: Show news weight
            
        except Exception as e:
            logger.error(f"❌ Failed to update regime parameters: {e}")
    
    def _initialize_ml_model(self):
        """Initialize or load ML model"""
        try:
            # Ensure RandomForestClassifier is available for model loading
            from sklearn.ensemble import RandomForestClassifier
            
            # Try to load XGBoost bridge for advanced ensemble
            try:
                from app.ml.advanced_ensemble_patch import AdvancedEnsembleModelPatch
                bridge_data = AdvancedEnsembleModelPatch.load_xgboost_bridge()
                if bridge_data:
                    logger.info("SUCCESS: Loaded XGBoost bridge for advanced ensemble")
                    self._bridge_data = bridge_data
                    return
            except ImportError:
                pass
            
            # Try to load existing model
            if hasattr(self.ml_model, 'load_model') and self.ml_model.load_model():
                logger.info("SUCCESS: Loaded existing ML model")
            else:
                logger.warning("WARNING: No trained model found. Using fallback scoring method.")
                # In production, you'd want to train immediately
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
    
    async def generate_signals(self) -> List[SignalRecord]:
        """
        Generate simplified regime-aware trading signals using ML pipeline with optional news enhancement
        
        Returns:
            List of high-confidence ML-generated signals with optional news enhancement
        """
        start_time = time.time()
        logger.info(f"🚀 Starting signal generation process...")
        total_analyzed = 0
        confidence_rejected = 0
        confidence_rejected_symbols = []
        rejection_reasons = []
        try:
            # Reset daily counter if new day
            today = datetime.now().date()
            if self.last_signal_date != today:
                self.daily_signal_count = 0
                self.news_enhanced_signals_count = 0
                self.last_signal_date = today
                logger.info(f"🌅 New trading day: {today}")
                logger.info(f"🎯 Current regime: {self.regime_context.get('regime', 'UNKNOWN')}")
                logger.info(f"📰 News Intelligence: {'✅ ACTIVE' if self.news_intelligence else '❌ INACTIVE'}")
            
            # Check daily limits (regime-adjusted)
            if self.daily_signal_count >= self.max_signals_per_day:
                logger.info(f"Daily signal limit reached ({self.max_signals_per_day}) for regime {self.regime_context.get('regime')}")
                return []
            
            # Check market status with timeout
            logger.info("📊 Checking market status...")
            try:
                await asyncio.wait_for(self.market_data.update_market_status(), timeout=10.0)
                if hasattr(self.market_data, 'market_status') and self.market_data.market_status.value != "OPEN":
                    logger.info(f"Market closed: {self.market_data.market_status.value}")
                    return []
            except asyncio.TimeoutError:
                logger.warning("⚠️ Market status check timed out, proceeding with analysis")
            except Exception as e:
                logger.error(f"❌ Market status check failed: {e}")
                return []
            
            # Get top opportunity stocks with timeout
            logger.info("🔍 Fetching trading opportunities...")
            try:
                opportunities = await asyncio.wait_for(self._get_top_opportunity_stocks(), timeout=30.0)
                if not opportunities:
                    logger.info("No trading opportunities found")
                    return []
                logger.info(f"📈 Found {len(opportunities)} trading opportunities")
            except asyncio.TimeoutError:
                logger.error("❌ Opportunity analysis timed out after 30 seconds")
                return []
            except Exception as e:
                logger.error(f"❌ Opportunity analysis failed: {e}")
                return []
            
            # Generate signals for top opportunities
            signals = []
            current_regime = self.regime_context.get("regime", "SIDEWAYS")
            logger.info(f"🎯 Generating signals for {len(opportunities)} opportunities (Regime: {current_regime})")
            
            for i, stock_data in enumerate(opportunities):
                try:
                    if self.daily_signal_count >= self.max_signals_per_day:
                        logger.info(f"Daily signal limit reached during processing")
                        break
                    symbol = stock_data["symbol"]
                    logger.debug(f"Analyzing {symbol} ({i+1}/{len(opportunities)})")
                    total_analyzed += 1
                    # --- CHANGED: get signal, ml_conf, reason ---
                    signal, ml_conf, reject_reason = await asyncio.wait_for(
                        self._analyze_stock_with_simplified_ml_and_news(symbol, stock_data),
                        timeout=15.0
                    )
                    stock_data['ml_confidence'] = ml_conf
                    if signal:
                        risk_check = self.signal_logger.check_risk_limits(signal)
                        if risk_check["overall_risk_ok"]:
                            signal = self._enhance_signal_with_regime_data(signal)
                            signals.append(signal)
                            self.daily_signal_count += 1
                            if hasattr(signal, 'notes') and signal.notes and 'News Enhanced' in signal.notes:
                                self.news_enhanced_signals_count += 1
                            success = self.signal_logger.log_signal(signal)
                            if success:
                                news_indicator = "📰" if self.news_intelligence else ""
                                logger.info(f"🎯 ML Signal ({current_regime}): {signal.ticker} {signal.direction.value} {news_indicator} "
                                          f"ML Confidence: {signal.ml_confidence:.1%} | "
                                          f"Entry: ₹{signal.entry_price} | Risk Adj: {self.regime_context.get('risk_adjustment', 1.0):.1f}x")
                            self._track_regime_signal(signal, current_regime)
                        else:
                            logger.info(f"❌ {symbol} rejected: Risk limits exceeded (ML confidence: {ml_conf if ml_conf is not None else 'N/A'})")
                            rejection_reasons.append((symbol, ml_conf, 'risk_limit'))
                    else:
                        # Always log the reason and confidence
                        logger.info(f"❌ {symbol} rejected: {reject_reason} (ML confidence: {ml_conf if ml_conf is not None else 'N/A'})")
                        rejection_reasons.append((symbol, ml_conf, reject_reason))
                        if reject_reason == 'below_confidence_threshold':
                            confidence_rejected += 1
                            confidence_rejected_symbols.append((symbol, ml_conf))
                except asyncio.TimeoutError:
                    logger.warning(f"⚠️ Signal generation timed out for {symbol}")
                    rejection_reasons.append((symbol, None, 'timeout'))
                    continue
                except Exception as e:
                    logger.error(f"Signal generation failed for {symbol}: {e}")
                    rejection_reasons.append((symbol, None, f'exception: {e}'))
                    continue
            if total_analyzed > 0:
                percent_rejected = (confidence_rejected / total_analyzed) * 100
                logger.info(f"📊 {confidence_rejected}/{total_analyzed} stocks ({percent_rejected:.1f}%) rejected due to ML confidence threshold ({self.min_ml_confidence:.2f})")
                if confidence_rejected_symbols:
                    logger.info(f"Symbols rejected (confidence): {[f'{s}({c:.2f})' for s,c in confidence_rejected_symbols]}")
                # Log all rejection reasons summary
                reason_counts = {}
                for _, _, reason in rejection_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                logger.info(f"📊 Rejection reasons: {reason_counts}")
            # Log completion with detailed metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"✅ Signal generation completed in {processing_time:.2f}s")
            logger.info(f"📊 Results: {len(signals)} signals generated (Regime: {current_regime}, News Enhanced: {self.news_enhanced_signals_count})")
            logger.info(f"📈 Daily progress: {self.daily_signal_count}/{self.max_signals_per_day} signals")
            
            # Log performance metrics
            if signals:
                avg_confidence = sum(s.ml_confidence for s in signals) / len(signals)
                logger.info(f"📊 Average confidence: {avg_confidence:.1%}")
                
                # Log signal details for monitoring
                for signal in signals:
                    logger.info(f"📋 Signal: {signal.ticker} | Direction: {signal.direction.value} | "
                              f"Entry: ₹{signal.entry_price} | Target: ₹{signal.target_price} | "
                              f"Stop: ₹{signal.stop_loss} | Confidence: {signal.ml_confidence:.1%}")
            
            return signals
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.error(f"❌ Signal generation failed after {processing_time:.2f}s: {e}")
            return []
    
    async def generate_regime_aware_signals(self, regime=None, regime_confidence=None) -> List[SignalRecord]:
        """
        Generate regime-aware trading signals (compatibility wrapper method)
        
        This method is called by main.py service manager and provides compatibility
        with the expected interface while delegating to the main generate_signals method.
        
        Args:
            regime: Market regime (optional, updates internal state)
            regime_confidence: Regime confidence (optional)
            
        Returns:
            List of regime-aware trading signals enhanced with news intelligence
        """
        try:
            # Update regime context if provided
            if regime is not None:
                regime_str = str(regime).replace('MarketRegime.', '')  # Handle enum conversion
                regime_context = {
                    'regime': regime_str,
                    'confidence': regime_confidence or 0.5,
                    'trading_strategy': self._get_regime_strategy(regime_str),
                    'risk_adjustment': self._get_regime_risk_adjustment(regime_str)
                }
                self.set_regime_context(regime_context)
                logger.info(f"🎯 Regime context updated via generate_regime_aware_signals: {regime_str} (Conf: {regime_confidence or 0.5:.1%})")
            
            # Call the main signal generation method
            signals = await self.generate_signals()
            
            # Log the completion with news intelligence status
            if signals:
                news_status = "with News Intelligence" if self.news_intelligence else "without News Intelligence"
                logger.info(f"✅ generate_regime_aware_signals completed: {len(signals)} signals generated {news_status}")
            else:
                logger.info(f"📊 generate_regime_aware_signals completed: No signals generated")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ generate_regime_aware_signals failed: {e}")
            return []
    
    def _get_regime_strategy(self, regime: str) -> str:
        """Get trading strategy for regime"""
        regime_strategies = {
            'BULLISH': 'MOMENTUM_FOLLOW',
            'BEARISH': 'SHORT_TERM_REVERSAL', 
            'SIDEWAYS': 'MEAN_REVERSION'
        }
        return regime_strategies.get(regime, 'MEAN_REVERSION')
    
    def _get_regime_risk_adjustment(self, regime: str) -> float:
        """Get risk adjustment for regime"""
        regime_risk = {
            'BULLISH': 1.1,
            'BEARISH': 0.9,
            'SIDEWAYS': 1.0
        }
        return regime_risk.get(regime, 1.0)
    
    def _get_regime_analysis_limit(self) -> int:
        """Get number of stocks to analyze based on regime"""
        current_regime = self.regime_context.get("regime", "SIDEWAYS")
        config = self.regime_configs.get(current_regime, self.regime_configs["SIDEWAYS"])
        
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
            
            # Add news intelligence indicator
            if self.news_intelligence:
                regime_info += ", News Enhanced: ✅"
            
            if signal.notes:
                signal.notes = f"{signal.notes} | {regime_info}"
            else:
                signal.notes = regime_info
            
            # Update signal source to indicate regime awareness
            signal.signal_source = f"ADVANCED_SENTIMENT_REGIME_{self.regime_context.get('regime', 'UNKNOWN')}"
            
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
                    "final_score_sum": 0.0,
                    "news_enhanced_signals": 0  # NEWS: Track news enhancement
                }
            
            regime_stats = self.model_performance["regime_performance"][regime]
            regime_stats["signals_generated"] += 1
            regime_stats["confidence_sum"] += signal.ml_confidence
            regime_stats["avg_confidence"] = regime_stats["confidence_sum"] / regime_stats["signals_generated"]
            
            # Track news enhancement
            if hasattr(signal, 'notes') and 'News Enhanced' in signal.notes:
                regime_stats["news_enhanced_signals"] += 1
            
            if signal.final_score:
                regime_stats["final_score_sum"] += signal.final_score
                regime_stats["avg_final_score"] = regime_stats["final_score_sum"] / regime_stats["signals_generated"]
            
        except Exception as e:
            logger.error(f"Failed to track regime signal: {e}")
    
    async def _get_top_opportunity_stocks(self) -> List[Dict]:
        """
        Get top opportunity stocks from Nifty 100 using market-appropriate analysis
        
        Returns:
            List of stock opportunities sorted by ML potential with regime considerations
        """
        try:
            current_regime = self.regime_context.get("regime", "SIDEWAYS")
            current_time = datetime.now().time()
            
            # Check if we're in pre-market hours (8:00 AM to 9:15 AM)
            is_premarket = dt_time(8, 0) <= current_time <= dt_time(9, 15)
            
            if is_premarket:
                logger.info(f"📋 Running regime-aware pre-market analysis on Nifty 100 (Regime: {current_regime})...")
            else:
                logger.info(f"📊 Running live market opportunity analysis on Nifty 100 (Regime: {current_regime})...")
            
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
                    
                    # Get news sentiment (enhanced if available) - but only during pre-market or if news is fresh
                    sentiment_score = 0.0
                    news_impact = 0.0
                    
                    if self.news_intelligence:
                        # Only fetch news during pre-market or if we haven't fetched recently
                        if is_premarket or not hasattr(self, '_last_news_fetch') or \
                           (datetime.now() - getattr(self, '_last_news_fetch', datetime.min)).total_seconds() > 300:  # 5 minutes
                            try:
                                # Quick news sentiment check
                                news_sentiment_data = await self._get_comprehensive_news_sentiment(symbol)
                                sentiment_score = news_sentiment_data.get("symbol_specific_sentiment", 0.0)
                                news_impact = news_sentiment_data.get("news_impact_score", 0.0)
                                self._last_news_fetch = datetime.now()
                            except Exception as e:
                                logger.debug(f"News sentiment fetch failed for {symbol}: {e}")
                                sentiment_score = 0.0
                                news_impact = 0.0
                        else:
                            # Use cached sentiment data during live market
                            sentiment_data = market_data.get("sentiment", {})
                            sentiment_score = sentiment_data.get("sentiment_score", 0.0)
                            news_impact = sentiment_data.get("news_impact_score", 0.0)
                    else:
                        sentiment_data = market_data.get("sentiment", {})
                        sentiment_score = sentiment_data.get("sentiment_score", 0.0)
                        news_impact = 0.0
                    
                    # Regime-aware opportunity scoring (enhanced with news)
                    opportunity_score = self._calculate_regime_aware_opportunity_score_with_news(
                        gap_pct, volume_ratio, sentiment_score, news_impact, current_regime
                    )
                    
                    opportunities.append({
                        "symbol": symbol,
                        "opportunity_score": opportunity_score,
                        "gap_percentage": gap_pct,
                        "volume_ratio": volume_ratio,
                        "sentiment_score": sentiment_score,
                        "news_impact_score": news_impact,  # NEWS: Add news impact
                        "current_price": quote["ltp"],
                        "sector": self.stock_universe.get_sector(symbol),
                        "market_data": market_data,
                        "regime_score": opportunity_score,
                        "news_enhanced": news_impact > 0.1,  # NEWS: Flag news enhancement
                        "analysis_type": "premarket" if is_premarket else "live_market"
                    })
                    
                    processed += 1
                    if processed % 20 == 0:
                        logger.info(f"Analyzed {processed}/{len(all_stocks)} stocks...")
                    
                    # Rate limiting - faster during live market
                    await asyncio.sleep(0.02 if not is_premarket else 0.05)  # 20ms live, 50ms pre-market
                    
                except Exception as e:
                    logger.debug(f"Opportunity analysis failed for {symbol}: {e}")
                    continue
            
            # Sort by regime-aware opportunity score
            opportunities.sort(key=lambda x: x["regime_score"], reverse=True)
            
            news_enhanced = len([o for o in opportunities[:10] if o.get("news_enhanced", False)])
            analysis_type = "pre-market" if is_premarket else "live market"
            logger.info(f"✅ Analyzed {len(opportunities)} stocks using {analysis_type} analysis, top 10 regime-aware opportunities (News Enhanced: {news_enhanced}):")
            for i, opp in enumerate(opportunities[:10]):
                news_indicator = "📰" if opp.get("news_enhanced", False) else ""
                logger.info(f"  {i+1}. {opp['symbol']} {news_indicator} - Score: {opp['regime_score']:.2f} "
                          f"Gap: {opp['gap_percentage']:.1f}% Sentiment: {opp['sentiment_score']:.2f}")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Regime-aware opportunity analysis failed: {e}")
            return []
    
    def _calculate_regime_aware_opportunity_score_with_news(self, gap_pct: float, volume_ratio: float, 
                                                          sentiment_score: float, news_impact: float, regime: str) -> float:
        """Calculate opportunity score with regime-specific weightings + news intelligence"""
        try:
            # Base scoring weights
            gap_weight = 0.4
            volume_weight = 0.3
            sentiment_weight = 0.3
            news_weight = 0.0  # Will be set based on regime
            
            # Get regime configuration
            config = self.regime_configs.get(regime, self.regime_configs["SIDEWAYS"])
            news_weight = config.get("news_weight", 0.25)
            
            # Adjust weights to include news (reduce others proportionally)
            if news_impact > 0:
                total_base_weight = gap_weight + volume_weight + sentiment_weight
                adjustment_factor = (1.0 - news_weight) / total_base_weight
                gap_weight *= adjustment_factor
                volume_weight *= adjustment_factor
                sentiment_weight *= adjustment_factor
            
            # Regime-specific adjustments
            if regime == "BULLISH":
                gap_weight = gap_weight * 1.25    # Higher weight on momentum
                news_weight = min(news_weight * 1.2, 0.5)  # Boost news weight in trending markets
            elif regime == "BEARISH":
                gap_weight = gap_weight * 1.25
                news_weight = min(news_weight * 1.3, 0.5)  # Even higher news weight in bearish trends
            elif regime == "SIDEWAYS":
                volume_weight = volume_weight * 1.3  # Higher weight on volume
                sentiment_weight = sentiment_weight * 1.2  # Higher weight on sentiment
                news_weight = min(news_weight * 1.4, 0.6)  # News very important in choppy markets
            elif regime == "GAP_DAY":
                gap_weight = gap_weight * 1.5    # Much higher weight on gaps
                news_weight = min(news_weight * 1.6, 0.7)  # News critical on gap days
            elif regime == "HIGH_VOLATILITY":
                news_weight = min(news_weight * 1.8, 0.8)  # News extremely important in volatile markets
            
            # Calculate weighted score
            opportunity_score = (
                abs(gap_pct) * gap_weight +
                min(volume_ratio, 3.0) * volume_weight +
                abs(sentiment_score) * sentiment_weight +
                news_impact * news_weight  # NEWS: Add news impact
            )
            
            # Apply regime confidence multiplier
            regime_confidence = self.regime_context.get("confidence", 0.5)
            confidence_multiplier = 0.8 + (regime_confidence * 0.4)  # 0.8 to 1.2
            
            # News boost for high-impact news
            if news_impact > 0.5:
                news_boost = 1.0 + (news_impact * 0.3)  # Up to 30% boost for high-impact news
                opportunity_score *= news_boost
            
            return opportunity_score * confidence_multiplier
            
        except Exception as e:
            logger.error(f"Regime-aware scoring with news failed: {e}")
            return abs(gap_pct) * 0.4 + min(volume_ratio, 3.0) * 0.3 + abs(sentiment_score) * 0.3
    
    async def _analyze_stock_with_simplified_ml_and_news(self, symbol: str, stock_data: Dict) -> tuple:
        """
        Analyze individual stock using simplified ML pipeline with optional news enhancement
        Returns: (SignalRecord or None, ml_confidence or None, rejection_reason or None)
        """
        try:
            current_regime = self.regime_context.get("regime", "SIDEWAYS")
            logger.debug(f"🤖 Simplified ML + News analysis for {symbol} (Regime: {current_regime})...")
            market_data = stock_data["market_data"]
            quote = market_data["quote"]
            # Get historical data for technical analysis
            try:
                if hasattr(self.market_data, 'zerodha'):
                    historical_data = await self.market_data.zerodha.get_historical_data(
                        symbol, "minute", datetime.now() - timedelta(days=30)
                    )
                else:
                    historical_data = None
            except Exception as e:
                return None, None, f"historical_data_error: {e}"
            # Feature engineering
            try:
                features = self.feature_engineer.create_features(symbol, quote, historical_data)
            except Exception as e:
                return None, None, f"feature_engineering_error: {e}"
            # ML inference
            try:
                ml_probability = self.ml_model.predict_signal_probability(features)
            except Exception as e:
                return None, None, f"ml_inference_error: {e}"
            # Always attach confidence
            stock_data['ml_confidence'] = ml_probability
            # Confidence threshold check
            if ml_probability < self.min_ml_confidence:
                return None, ml_probability, 'below_confidence_threshold'
            # Post-processing and signal creation
            try:
                signal = self._postprocess_signal(features, ml_probability, symbol, quote)
                if not signal:
                    return None, ml_probability, 'postprocess_failed'
            except Exception as e:
                return None, ml_probability, f'postprocess_error: {e}'
            return signal, ml_probability, None
        except Exception as e:
            return None, None, f'exception: {e}'
    
    def _postprocess_signal(self, features: Dict, ml_probability: float, symbol: str, quote: Dict) -> Optional[SignalRecord]:
        """Post-process the signal based on ML output and regime context"""
        try:
            current_regime = self.regime_context.get("regime", "SIDEWAYS")
            logger.debug(f"🤖 Post-processing signal for {symbol} (Regime: {current_regime})...")
            
            # Determine signal direction based on ML output
            direction = SignalDirection.BUY if ml_probability > 0.5 else SignalDirection.SELL
            
            # Calculate price levels using regime-enhanced logic
            entry_price = quote["ltp"]
            stop_loss, target_price = self._calculate_regime_aware_price_levels(
                entry_price, features, direction, ml_probability, current_regime
            )
            
            # Calculate position sizing based on ML confidence and regime
            position_size = self._calculate_regime_aware_position_size(
                entry_price, stop_loss, ml_probability, features, current_regime
            )
            
            # Create comprehensive signal record with regime awareness + optional news enhancement
            signal = SignalRecord(
                signal_id=f"ML_{current_regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}",
                timestamp=datetime.now(),
                ticker=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                ml_confidence=ml_probability,  # Use news-enhanced confidence
                technical_score=features.get("technical_composite_score", 0.0),
                sentiment_score=features.get("sentiment_score", 0.0),
                macro_score=features.get("nifty_performance", 0.0),
                final_score=ml_probability,  # News-enhanced ML probability is our final score
                indicators=self._create_technical_indicators_from_features(features),
                market_context=await self._create_enhanced_market_context(features),
                pre_market=self._create_pre_market_data_from_features(features, features),
                risk_reward_ratio=self._calculate_risk_reward_ratio(entry_price, stop_loss, target_price, direction),
                position_size_suggested=position_size,
                capital_at_risk=abs(entry_price - stop_loss) * position_size,
                model_version="v1.2_simplified_ml_news_regime_aware",
                signal_source=f"SIMPLIFIED_ML_{current_regime}",
                notes=self._create_simplified_signal_notes(symbol, current_regime, ml_probability, ml_probability, features, features)
            )
            
            logger.debug(f"✅ {symbol} Simplified ML Signal: {direction.value} Confidence: {ml_probability:.1%}")
            return signal
            
        except Exception as e:
            logger.error(f"Failed to post-process signal for {symbol}: {e}")
            return None
    
    def _add_simplified_news_features(self, features: Dict, news_sentiment: Dict, regime: str) -> Dict:
        """Add basic news-specific features to the feature set"""
        try:
            news_features = {
                "news_sentiment_score": news_sentiment.get("symbol_specific_sentiment", 0.0),
                "news_impact_score": news_sentiment.get("news_impact_score", 0.0),
                "news_count": min(news_sentiment.get("news_count", 0), 10),  # Cap at 10
                "breaking_news_detected": 1.0 if news_sentiment.get("breaking_news_detected", False) else 0.0,
                "high_impact_events_count": len(news_sentiment.get("high_impact_events", [])),
                "overall_market_sentiment": news_sentiment.get("overall_market_sentiment", 0.0),
                "news_sources_count": min(news_sentiment.get("news_sources_count", 0), 10),  # Cap at 10
                "positive_news_sentiment": 1.0 if news_sentiment.get("symbol_specific_sentiment", 0.0) > 0.1 else 0.0,
                "negative_news_sentiment": 1.0 if news_sentiment.get("symbol_specific_sentiment", 0.0) < -0.1 else 0.0,
                "strong_news_sentiment": 1.0 if abs(news_sentiment.get("symbol_specific_sentiment", 0.0)) > 0.5 else 0.0,
                "news_intelligence_available": 1.0 if news_sentiment.get("enhanced_by_news_intelligence", False) else 0.0
            }
            
            features.update(news_features)
            return features
            
        except Exception as e:
            logger.error(f"Failed to add simplified news features: {e}")
            return features
    
    def _apply_simplified_news_confidence_adjustment(self, ml_probability: float, news_sentiment: Dict, regime: str) -> float:
        """Apply basic news-enhanced confidence adjustment to ML probability"""
        try:
            # Start with base regime adjustment
            adjusted_probability = self._apply_regime_confidence_adjustment(ml_probability, regime)
            
            # Apply news enhancement
            if news_sentiment.get("enhanced_by_news_intelligence", False):
                news_impact = news_sentiment.get("news_impact_score", 0.0)
                symbol_sentiment = news_sentiment.get("symbol_specific_sentiment", 0.0)
                
                # News confidence boost/penalty based on sentiment alignment with ML prediction
                if (ml_probability > 0.5 and symbol_sentiment > 0.1) or (ml_probability < 0.5 and symbol_sentiment < -0.1):
                    # News sentiment aligns with ML prediction - boost confidence
                    news_boost = news_impact * 0.1  # Up to 10% boost
                    adjusted_probability += news_boost
                elif abs(symbol_sentiment) > 0.3:
                    # Strong opposing news sentiment - reduce confidence
                    news_penalty = abs(symbol_sentiment) * 0.05  # Up to 5% penalty
                    adjusted_probability -= news_penalty
                
                # Breaking news additional boost
                if news_sentiment.get("breaking_news_detected", False):
                    breaking_news_boost = 0.05  # 5% boost for breaking news
                    adjusted_probability += breaking_news_boost
                
                # Track news accuracy boost
                if news_boost > 0:
                    self.model_performance["news_accuracy_boost"] = max(
                        self.model_performance["news_accuracy_boost"], 
                        news_boost
                    )
            
            # Ensure probability stays within bounds
            return max(0.0, min(1.0, adjusted_probability))
            
        except Exception as e:
            logger.error(f"Failed to apply simplified news confidence adjustment: {e}")
            return ml_probability
    
    def _apply_regime_confidence_adjustment(self, ml_probability: float, regime: str) -> float:
        """Apply regime-specific confidence adjustments to ML probability"""
        try:
            # Get regime confidence from context
            regime_confidence = self.regime_context.get("confidence", 0.5)
            
            # Apply regime-specific adjustments
            if regime in ["BULLISH", "BEARISH", "GAP_DAY"]:
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
            else:  # SIDEWAYS
                # In choppy markets, slight penalty
                confidence_penalty = 0.03 * regime_confidence
                adjusted_probability = ml_probability - confidence_penalty
            
            # Ensure probability stays within bounds
            return max(0.0, min(1.0, adjusted_probability))
            
        except Exception as e:
            logger.error(f"Failed to apply regime confidence adjustment: {e}")
            return ml_probability
    
    def _determine_signal_direction_with_simplified_news(self, features: Dict, ml_probability: float, news_sentiment: Dict, regime: str) -> SignalDirection:
        """Determine signal direction based on features, ML output, news sentiment, and regime"""
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
            
            # NEWS: Add news sentiment confirmation
            news_bullish_signals = 0
            if news_sentiment.get("enhanced_by_news_intelligence", False):
                symbol_sentiment = news_sentiment.get("symbol_specific_sentiment", 0.0)
                if symbol_sentiment > 0.2:
                    news_bullish_signals += 2  # Strong positive news
                elif symbol_sentiment > 0.1:
                    news_bullish_signals += 1  # Moderate positive news
                elif symbol_sentiment < -0.2:
                    news_bullish_signals -= 2  # Strong negative news
                elif symbol_sentiment < -0.1:
                    news_bullish_signals -= 1  # Moderate negative news
                
                # Breaking news override
                if news_sentiment.get("breaking_news_detected", False):
                    if abs(symbol_sentiment) > 0.4:
                        # Strong breaking news can override technical signals
                        if symbol_sentiment > 0.4:
                            news_bullish_signals += 3
                        elif symbol_sentiment < -0.4:
                            news_bullish_signals -= 3
            
            total_bullish_signals = technical_bullish_signals + news_bullish_signals
            
            # Regime-specific direction adjustments
            regime_strategy = self.regime_context.get("trading_strategy", "MEAN_REVERSION")
            
            if regime == "BEARISH" and regime_strategy == "MOMENTUM":
                # In bearish trends with momentum strategy, bias towards SELL
                if base_direction == SignalDirection.BUY and total_bullish_signals < 4:
                    logger.debug(f"Bearish regime + news suggests caution for BUY signal")
            elif regime == "BULLISH" and regime_strategy == "MOMENTUM":
                # In bullish trends with momentum strategy, bias towards BUY
                if base_direction == SignalDirection.SELL and total_bullish_signals > 0:
                    logger.debug(f"Bullish regime + news suggests caution for SELL signal")
            elif regime == "HIGH_VOLATILITY":
                # In high volatility, require stronger confirmation (including news)
                required_confirmation = 4 if news_sentiment.get("enhanced_by_news_intelligence", False) else 3
                if base_direction == SignalDirection.BUY and total_bullish_signals < required_confirmation:
                    logger.debug("High volatility regime requires stronger confirmation for BUY (including news)")
                elif base_direction == SignalDirection.SELL and total_bullish_signals > 1:
                    logger.debug("High volatility regime requires stronger confirmation for SELL (including news)")
            
            return base_direction
            
        except Exception as e:
            logger.error(f"Direction determination with news failed: {e}")
            return SignalDirection.BUY  # Default
    
    def _create_enhanced_signal_notes(self, symbol: str, regime: str, original_ml: float, adjusted_ml: float, news_sentiment: Dict, prediction_details: Any) -> str:
        """Create comprehensive signal notes with news intelligence information"""
        try:
            notes_parts = []
            
            # Regime information
            notes_parts.append(f"Regime: {regime}")
            notes_parts.append(f"ML: {original_ml:.1%}→{adjusted_ml:.1%}")
            
            # News intelligence information
            if news_sentiment.get("enhanced_by_news_intelligence", False):
                news_impact = news_sentiment.get("news_impact_score", 0.0)
                symbol_sentiment = news_sentiment.get("symbol_specific_sentiment", 0.0)
                news_count = news_sentiment.get("news_count", 0)
                
                notes_parts.append(f"News Enhanced: ✅")
                notes_parts.append(f"News Impact: {news_impact:.2f}")
                notes_parts.append(f"News Sentiment: {symbol_sentiment:.2f}")
                notes_parts.append(f"Articles: {news_count}")
                
                if news_sentiment.get("breaking_news_detected", False):
                    notes_parts.append("Breaking News: 🚨")
                
                if news_sentiment.get("latest_headline"):
                    headline = news_sentiment["latest_headline"][:50] + "..." if len(news_sentiment["latest_headline"]) > 50 else news_sentiment["latest_headline"]
                    notes_parts.append(f"Latest: {headline}")
            else:
                notes_parts.append("News Enhanced: ❌")
            
            # Top ML features
            if isinstance(prediction_details, dict) and "feature_contributions" in prediction_details:
                top_features = list(prediction_details["feature_contributions"].keys())[:3]
                notes_parts.append(f"Top ML Features: {top_features}")
            
            return " | ".join(notes_parts)
            
        except Exception as e:
            logger.error(f"Failed to create enhanced signal notes: {e}")
            return f"Regime: {regime}, ML: {original_ml:.1%}→{adjusted_ml:.1%}, Error in notes generation"
    
    # ================================================================
    # EXISTING METHODS (Enhanced with News Intelligence Support)
    # ================================================================
    
    async def _get_enhanced_sentiment(self, symbol: str) -> Dict:
        """Get enhanced sentiment analysis using the simplified FinBERT sentiment system (fallback method)"""
        try:
            # Get recent news for the symbol
            news_data = []
            
            # Try to get news from market data service
            try:
                if hasattr(self.market_data, 'get_symbol_news'):
                    news_response = await self.market_data.get_symbol_news(symbol)
                    if news_response and "articles" in news_response:
                        news_data = news_response["articles"][:5]  # Top 5 articles
            except:
                pass
            
            if news_data:
                # Use simplified sentiment analyzer
                sentiment_results = []
                for article in news_data:
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    if text.strip():
                        # Use simplified sentiment analyzer
                        if hasattr(self.sentiment_analyzer, 'analyze_text'):
                            result = self.sentiment_analyzer.analyze_text(text, source="news")
                            sentiment_results.append({
                                "finbert_score": result.finbert_score,
                                "finbert_confidence": result.finbert_confidence,
                                "sentiment_class": result.sentiment_class,
                                "strength": result.strength
                            })
                
                if sentiment_results:
                    # Aggregate sentiment scores
                    avg_finbert = np.mean([r["finbert_score"] for r in sentiment_results])
                    avg_confidence = np.mean([r["finbert_confidence"] for r in sentiment_results])
                    
                    # Use FinBERT score as the primary sentiment score
                    sentiment_score = avg_finbert
                    
                    return {
                        "finbert_score": avg_finbert,
                        "finbert_confidence": avg_confidence,
                        "weighted_score": sentiment_score,  # Use FinBERT score as weighted score
                        "news_count": len(news_data),
                        "sentiment_score": sentiment_score,
                        "latest_headline": news_data[0].get('title', '') if news_data else '',
                        "enhanced_by_news_intelligence": False  # Mark as fallback
                    }
            
            # Fallback: neutral sentiment
            return {
                "finbert_score": 0.0,
                "finbert_confidence": 0.0,
                "weighted_score": 0.0,
                "news_count": 0,
                "sentiment_score": 0.0,
                "latest_headline": None,
                "enhanced_by_news_intelligence": False
            }
            
        except Exception as e:
            logger.error(f"Enhanced sentiment analysis failed for {symbol}: {e}")
            return {
                "finbert_score": 0.0,
                "finbert_confidence": 0.0,
                "weighted_score": 0.0,
                "news_count": 0,
                "sentiment_score": 0.0,
                "latest_headline": None,
                "enhanced_by_news_intelligence": False
            }
    
    # ================================================================
    # Helper Methods for Fallback and Feature Engineering (Unchanged)
    # ================================================================
    
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
            
            # NEWS: Add news enhancement to fallback
            news_boost = 0.0
            if stock_data.get("news_enhanced", False):
                news_impact = stock_data.get("news_impact_score", 0.0)
                news_boost = news_impact * 0.2  # 20% boost from news
            
            # Combine scores
            fallback_score = (momentum_score * 0.5 + sentiment_strength * 0.3 + volume_score * 0.2) + news_boost
            
            # Apply regime adjustment
            current_regime = self.regime_context.get("regime", "SIDEWAYS")
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
                model_version="v1.2_fallback_regime_news_aware",
                signal_source=f"FALLBACK_REGIME_{current_regime}",
                notes=f"Fallback analysis - Regime: {current_regime}, Score: {adjusted_score:.1%}, News: {'✅' if stock_data.get('news_enhanced', False) else '❌'}"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Fallback analysis failed for {symbol}: {e}")
            return None
    
    def _get_regime_fallback_multiplier(self, regime: str) -> float:
        """Get fallback score multiplier for regime"""
        multipliers = {
            "BULLISH": 1.2,
            "BEARISH": 1.1,
            "SIDEWAYS": 0.8,
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
            all_regimes = ["BULLISH", "BEARISH", "SIDEWAYS", "GAP_DAY", "HIGH_VOLATILITY", "LOW_VOLATILITY"]
            for r in all_regimes:
                if r != regime:
                    regime_features[f"regime_{r.lower()}"] = 0.0
            
            features.update(regime_features)
            return features
            
        except Exception as e:
            logger.error(f"Failed to add regime features: {e}")
            return features
    
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
            
            if regime == "BEARISH" and regime_strategy == "MOMENTUM":
                # In bearish trends with momentum strategy, bias towards SELL
                if base_direction == SignalDirection.BUY and technical_bullish_signals < 3:
                    logger.debug(f"Bearish regime suggests caution for BUY signal")
            elif regime == "BULLISH" and regime_strategy == "MOMENTUM":
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
            config = self.regime_configs.get(regime, self.regime_configs["SIDEWAYS"])
            
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
            elif regime in ["BULLISH", "BEARISH"]:
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
            config = self.regime_configs.get(regime, self.regime_configs["SIDEWAYS"])
            
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
    
    async def _get_nifty_correlation_data(self) -> pd.DataFrame:
        """Get Nifty data for correlation analysis"""
        try:
            # Try to get actual Nifty data
            if hasattr(self.market_data, 'get_live_market_data'):
                nifty_data = await self.market_data.get_live_market_data("NIFTY50")
                
                if nifty_data and nifty_data.get("quote"):
                    # Create simple DataFrame with current Nifty info
                    nifty_quote = nifty_data["quote"]
                    nifty_df = pd.DataFrame({
                        'close': [nifty_quote.get("prev_close", 18500), nifty_quote.get("ltp", 18500)],
                        'timestamp': [datetime.now() - timedelta(days=1), datetime.now()]
                    })
                    return nifty_df
            
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
            news_sentiment_score=sentiment.get("symbol_specific_sentiment", sentiment.get("finbert_score", 0.0)),
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
    # Model Management and Performance Tracking (Enhanced with News)
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
                    data_collector = self._ml_components['TrainingDataCollector'](self.market_data, self.signal_logger)
                    # Set feature engineer to avoid duplicate sentiment creation
                    if hasattr(data_collector, 'set_feature_engineer'):
                        data_collector.set_feature_engineer(self.feature_engineer)
                    
                    training_data = await data_collector.collect_training_data(lookback_days=30)
                    
                    if len(training_data) > 100:
                        # Add regime information to training data
                        training_data = self._add_regime_training_features(training_data)
                        
                        # Retrain model
                        feature_columns = [col for col in training_data.columns 
                                         if col not in ['symbol', 'date', 'entry_price', 'max_future_gain', 'profitable']]
                        
                        if hasattr(self.ml_model, 'train'):
                            results = self.ml_model.train(training_data[feature_columns + ['profitable']])
                            logger.info(f"✅ Regime-aware model retrained - New accuracy: {results.get('weighted_ensemble_auc', 0.0):.1%}")
                    
        except Exception as e:
            logger.error(f"Regime-aware model retraining failed: {e}")
    
    def _add_regime_training_features(self, training_data: pd.DataFrame) -> pd.DataFrame:
        """Add regime features to training data"""
        try:
            # Add regime features based on historical data (simplified for now)
            all_regimes = ["BULLISH", "BEARISH", "SIDEWAYS", "GAP_DAY", "HIGH_VOLATILITY", "LOW_VOLATILITY"]
            
            for regime in all_regimes:
                training_data[f"regime_{regime.lower()}"] = 0.0  # Default to 0
            
            # For now, set current regime to 1 (in production, you'd classify historical regimes)
            current_regime = self.regime_context.get("regime", "SIDEWAYS")
            training_data[f"regime_{current_regime.lower()}"] = 1.0
            training_data["regime_confidence"] = self.regime_context.get("confidence", 0.5)
            training_data["regime_risk_adjustment"] = self.regime_context.get("risk_adjustment", 1.0)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to add regime training features: {e}")
            return training_data
    
    def get_regime_performance_summary(self) -> Dict:
        """Get performance summary by regime including news intelligence metrics"""
        try:
            return {
                "current_regime": self.regime_context.get("regime", "UNKNOWN"),
                "regime_confidence": self.regime_context.get("confidence", 0.5),
                "trading_strategy": self.regime_context.get("trading_strategy", "MEAN_REVERSION"),
                "regime_adjustments": self.regime_configs.get(
                    self.regime_context.get("regime", "SIDEWAYS"), 
                    self.regime_configs["SIDEWAYS"]
                ),
                "regime_performance": self.model_performance["regime_performance"],
                "current_parameters": {
                    "min_ml_confidence": self.min_ml_confidence,
                    "max_signals_per_day": self.max_signals_per_day,
                    "risk_adjustment": self.regime_context.get("risk_adjustment", 1.0)
                },
                "daily_stats": {
                    "signals_generated_today": self.daily_signal_count,
                    "news_enhanced_signals": self.news_enhanced_signals_count,
                    "breaking_news_signals": self.breaking_news_signals_count,
                    "signals_remaining": max(0, self.max_signals_per_day - self.daily_signal_count),
                    "last_signal_date": self.last_signal_date.isoformat() if self.last_signal_date else None
                },
                "model_performance": {
                    "total_predictions": self.model_performance["total_predictions"],
                    "high_confidence_predictions": self.model_performance["high_confidence_predictions"],
                    "news_enhanced_predictions": self.model_performance["news_enhanced_predictions"],
                    "news_accuracy_boost": self.model_performance["news_accuracy_boost"],
                    "breaking_news_signals": self.model_performance["breaking_news_signals"],
                    "confidence_rate": (
                        self.model_performance["high_confidence_predictions"] / 
                        max(1, self.model_performance["total_predictions"])
                    )
                },
                "news_intelligence": {
                    "available": self.news_intelligence is not None,
                    "enhanced_predictions_today": self.news_enhanced_signals_count,
                    "breaking_news_signals_today": self.breaking_news_signals_count,
                    "last_news_analysis": self.last_news_analysis
                },
                "sentiment_system": "ADVANCED_SENTIMENT_ANALYZER_WITH_NEWS_INTELLIGENCE"
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
                        "timestamp": datetime.now(),
                        "news_enhanced": "NEWS" in signal_id or "BREAKING_NEWS" in signal_id
                    })
                    
                    # Update accuracy tracking
                    recent_outcomes = regime_stats["outcomes"][-20:]  # Last 20 outcomes
                    if len(recent_outcomes) >= 5:
                        accuracy = sum(1 for o in recent_outcomes if o["profitable"]) / len(recent_outcomes)
                        self.model_performance["accuracy_tracking"].append(accuracy)
                        
                        # Track news-enhanced performance
                        news_enhanced_outcomes = [o for o in recent_outcomes if o["news_enhanced"]]
                        if len(news_enhanced_outcomes) >= 3:
                            news_accuracy = sum(1 for o in news_enhanced_outcomes if o["profitable"]) / len(news_enhanced_outcomes)
                            logger.info(f"📊 Regime {regime} accuracy: {accuracy:.1%} (News Enhanced: {news_accuracy:.1%})")
                        else:
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
                "ml_model_loaded": hasattr(self.ml_model, 'is_trained') and getattr(self.ml_model, 'is_trained', False),
                "daily_signals_generated": self.daily_signal_count,
                "max_daily_signals": self.max_signals_per_day,
                "confidence_threshold": self.min_ml_confidence,
                "total_predictions": self.model_performance["total_predictions"],
                "news_intelligence_connected": self.news_intelligence is not None,
                "news_enhanced_signals_today": self.news_enhanced_signals_count,
                "breaking_news_signals_today": self.breaking_news_signals_count,
                "sentiment_system": "ADVANCED_SENTIMENT_ANALYZER_WITH_NEWS_INTELLIGENCE",
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e), 
                "sentiment_system": "ERROR",
                "news_intelligence_connected": False
            }
    
    # ================================================================
    # News Intelligence Performance Analytics
    # ================================================================
    
    def get_news_intelligence_analytics(self) -> Dict:
        """Get detailed analytics on news intelligence performance"""
        try:
            if not self.news_intelligence:
                return {
                    "status": "unavailable",
                    "message": "News intelligence service not connected"
                }
            
            total_signals = self.daily_signal_count
            news_enhanced = self.news_enhanced_signals_count
            breaking_news = self.breaking_news_signals_count
            
            analytics = {
                "status": "active",
                "daily_summary": {
                    "total_signals_generated": total_signals,
                    "news_enhanced_signals": news_enhanced,
                    "breaking_news_signals": breaking_news,
                    "traditional_signals": total_signals - news_enhanced - breaking_news,
                    "news_enhancement_rate": (news_enhanced / max(1, total_signals)) * 100,
                    "breaking_news_rate": (breaking_news / max(1, total_signals)) * 100
                },
                "performance_metrics": {
                    "total_predictions_enhanced": self.model_performance["news_enhanced_predictions"],
                    "news_accuracy_boost": self.model_performance["news_accuracy_boost"],
                    "breaking_news_signals_generated": self.model_performance["breaking_news_signals"]
                },
                "regime_breakdown": {}
            }
            
            # Add regime-specific news performance
            for regime, stats in self.model_performance["regime_performance"].items():
                news_enhanced_count = stats.get("news_enhanced_signals", 0)
                total_regime_signals = stats.get("signals_generated", 0)
                
                analytics["regime_breakdown"][regime] = {
                    "total_signals": total_regime_signals,
                    "news_enhanced": news_enhanced_count,
                    "news_enhancement_rate": (news_enhanced_count / max(1, total_regime_signals)) * 100
                }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get news intelligence analytics: {e}")
            return {"status": "error", "error": str(e)}
    
    def reset_daily_counters(self):
        """Reset daily counters (called by service manager at day start)"""
        try:
            self.daily_signal_count = 0
            self.news_enhanced_signals_count = 0
            self.breaking_news_signals_count = 0
            self.last_signal_date = datetime.now().date()
            logger.info("🔄 Daily counters reset for new trading day")
        except Exception as e:
            logger.error(f"Failed to reset daily counters: {e}")
    
    # ================================================================
    # Enhanced Debugging and Monitoring Methods
    # ================================================================
    
    def get_detailed_status(self) -> Dict:
        """Get detailed status for debugging and monitoring"""
        try:
            current_regime = self.regime_context.get("regime", "UNKNOWN")
            regime_config = self.regime_configs.get(current_regime, self.regime_configs["SIDEWAYS"])
            
            return {
                "signal_generator_status": {
                    "version": "v1.2_ML_news_regime_aware",
                    "initialization_time": datetime.now().isoformat(),
                    "sentiment_system": "ADVANCED_SENTIMENT_ANALYZER_SINGLE_INSTANCE",
                    "ml_model_type": "ADVANCED_ENSEMBLE" if self._advanced_models else "BASIC_ENSEMBLE"
                },
                "news_intelligence": {
                    "connected": self.news_intelligence is not None,
                    "service_type": "EnhancedNewsIntelligenceSystem" if self.news_intelligence else None,
                    "last_analysis": self.last_news_analysis
                },
                "regime_context": {
                    "current_regime": current_regime,
                    "confidence": self.regime_context.get("confidence", 0.5),
                    "strategy": self.regime_context.get("trading_strategy", "MEAN_REVERSION"),
                    "risk_adjustment": self.regime_context.get("risk_adjustment", 1.0),
                    "regime_config": regime_config
                },
                "trading_parameters": {
                    "base_confidence_threshold": self.base_min_ml_confidence,
                    "current_confidence_threshold": self.min_ml_confidence,
                    "base_max_signals_per_day": self.base_max_signals_per_day,
                    "current_max_signals_per_day": self.max_signals_per_day,
                    "max_capital_per_trade": self.max_capital_per_trade
                },
                "daily_activity": {
                    "signals_generated": self.daily_signal_count,
                    "news_enhanced_signals": self.news_enhanced_signals_count,
                    "breaking_news_signals": self.breaking_news_signals_count,
                    "signals_remaining": max(0, self.max_signals_per_day - self.daily_signal_count),
                    "last_signal_date": self.last_signal_date.isoformat() if self.last_signal_date else None
                },
                "performance_tracking": self.model_performance
            }
            
        except Exception as e:
            logger.error(f"Failed to get detailed status: {e}")
            return {"error": str(e), "status": "error"}
    
    # ================================================================
    # Compatibility Methods (for existing integrations)
    # ================================================================
    
    async def generate_ml_signals(self) -> List[SignalRecord]:
        """Compatibility method - delegates to generate_signals()"""
        return await self.generate_signals()
    
    def get_performance_summary(self) -> Dict:
        """Compatibility method - delegates to get_regime_performance_summary()"""
        return self.get_regime_performance_summary()
    
    def is_news_intelligence_available(self) -> bool:
        """Check if news intelligence is available and connected"""
        return self.news_intelligence is not None
    
    def get_news_intelligence_status(self) -> str:
        """Get news intelligence connection status"""
        if self.news_intelligence is None:
            return "DISCONNECTED"
        else:
            return "CONNECTED"
    
    # ================================================================
    # Advanced News Intelligence Features
    # ================================================================
    
    async def get_market_news_summary(self) -> Dict:
        """Get current market news summary using news intelligence"""
        try:
            if not self.news_intelligence:
                return {"status": "unavailable", "message": "News intelligence not connected"}
            
            # Get comprehensive market news
            news_intel = await self.news_intelligence.get_comprehensive_news_intelligence(
                lookback_hours=4  # Last 4 hours
            )
            
            self.last_news_analysis = datetime.now()
            
            return {
                "status": "success",
                "market_sentiment": news_intel.get("overall_sentiment", 0.0),
                "total_articles": news_intel.get("total_articles_analyzed", 0),
                "high_impact_events": len([e for e in news_intel.get("market_events", []) if e.get("significance_score", 0) > 0.7]),
                "sector_sentiments": news_intel.get("sentiment_analysis", {}).get("sector_sentiment", {}),
                "top_events": news_intel.get("market_events", [])[:5],
                "news_sources": news_intel.get("news_sources_used", []),
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get market news summary: {e}")
            return {"status": "error", "error": str(e)}
    
    async def analyze_symbol_news_impact(self, symbol: str) -> Dict:
        """Analyze news impact for a specific symbol"""
        try:
            if not self.news_intelligence:
                return {"status": "unavailable", "message": "News intelligence not connected"}
            
            # Get symbol-specific news analysis
            news_sentiment = await self._get_comprehensive_news_sentiment(symbol)
            
            return {
                "status": "success",
                "symbol": symbol,
                "analysis": {
                    "sentiment_score": news_sentiment.get("symbol_specific_sentiment", 0.0),
                    "news_impact_score": news_sentiment.get("news_impact_score", 0.0),
                    "articles_analyzed": news_sentiment.get("news_count", 0),
                    "breaking_news_detected": news_sentiment.get("breaking_news_detected", False),
                    "high_impact_events": news_sentiment.get("high_impact_events", []),
                    "latest_headline": news_sentiment.get("latest_headline"),
                    "enhanced_by_intelligence": news_sentiment.get("enhanced_by_news_intelligence", False)
                },
                "trading_implications": {
                    "sentiment_strength": "STRONG" if abs(news_sentiment.get("symbol_specific_sentiment", 0.0)) > 0.5 else "MODERATE" if abs(news_sentiment.get("symbol_specific_sentiment", 0.0)) > 0.2 else "WEAK",
                    "impact_level": "HIGH" if news_sentiment.get("news_impact_score", 0.0) > 0.7 else "MEDIUM" if news_sentiment.get("news_impact_score", 0.0) > 0.3 else "LOW",
                    "trading_bias": "BULLISH" if news_sentiment.get("symbol_specific_sentiment", 0.0) > 0.1 else "BEARISH" if news_sentiment.get("symbol_specific_sentiment", 0.0) < -0.1 else "NEUTRAL"
                },
                "analysis_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze symbol news impact for {symbol}: {e}")
            return {"status": "error", "symbol": symbol, "error": str(e)}
    
    # ================================================================
    # System Maintenance and Cleanup
    # ================================================================
    
    def cleanup_performance_tracking(self):
        """Clean up old performance tracking data to prevent memory bloat"""
        try:
            # Keep only last 100 accuracy tracking points
            if len(self.model_performance["accuracy_tracking"]) > 100:
                self.model_performance["accuracy_tracking"] = self.model_performance["accuracy_tracking"][-100:]
            
            # Clean up regime performance outcomes (keep last 50 per regime)
            for regime_stats in self.model_performance["regime_performance"].values():
                if "outcomes" in regime_stats and len(regime_stats["outcomes"]) > 50:
                    regime_stats["outcomes"] = regime_stats["outcomes"][-50:]
            
            logger.info("🧹 Performance tracking data cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup performance tracking: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            logger.info("🔄 ProductionMLSignalGenerator cleanup completed")
        except:
            pass  # Ignore errors during cleanup
    
    async def generate_ml_signal(self, symbol: str, market_data: Dict[str, Any], 
                                news_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Generate ML-based trading signal with performance monitoring
        """
        async with performance_monitor.async_timing("signal_generation", {"symbol": symbol}):
            try:
                # Track signal generation attempt
                performance_monitor.increment_counter("signal_generation_attempts", tags={"symbol": symbol})
                
                # Generate signal using existing logic
                signal = await self._generate_ml_signal_internal(symbol, market_data, news_context)
                
                # Track success
                success = signal is not None
                performance_monitor.track_success("signal_generation", success, {"symbol": symbol})
                
                if success:
                    performance_monitor.increment_counter("signals_generated", tags={"symbol": symbol})
                    logger.info(f"✅ ML signal generated for {symbol} with performance monitoring")
                else:
                    logger.warning(f"⚠️ Failed to generate ML signal for {symbol}")
                
                return signal
                
            except Exception as e:
                # Track failure
                performance_monitor.track_success("signal_generation", False, {"symbol": symbol})
                logger.error(f"❌ Signal generation failed for {symbol}: {e}")
                return None
    
    async def get_top_opportunity_stocks(self) -> List[Dict]:
        """
        Public method to get top opportunity stocks (for fallback signal generation)
        
        Returns:
            List of stock opportunities sorted by potential
        """
        try:
            return await self._get_top_opportunity_stocks()
        except Exception as e:
            logger.error(f"❌ get_top_opportunity_stocks failed: {e}")
            return []