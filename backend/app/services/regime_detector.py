# ================================================================
# backend/app/services/regime_detector.py
# Simplified Market Regime Detection System
# ================================================================

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

# Import market data service
from app.services.enhanced_market_data_nifty100 import EnhancedMarketDataService as MarketDataService

logger = logging.getLogger(__name__)

class MarketRegime:
    """Simplified market regime classification - only 3 regimes"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"

@dataclass
class RegimeAnalysis:
    """Simplified market regime analysis result"""
    regime: str
    confidence: float
    trend_strength: float
    volatility_level: float
    trading_strategy: str  # "MOMENTUM", "MEAN_REVERSION"
    risk_adjustment: float  # Multiplier for position sizing

class RegimeDetector:
    """
    Simplified market regime detection system
    Only 3 regimes: BULLISH, BEARISH, SIDEWAYS
    Reduced complexity for better performance
    """
    
    def __init__(self, market_data_service: MarketDataService):
        self.market_data = market_data_service
        
        # Simplified regime detection parameters
        self.trend_threshold = 0.5  # % movement to confirm trend
        self.volatility_threshold = 15.0  # VIX equivalent
        
        # Current regime state
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.last_regime_update = None
        
    async def detect_market_regime(self) -> RegimeAnalysis:
        """
        Detect current market regime with simplified logic
        
        Returns:
            RegimeAnalysis with simplified regime classification
        """
        try:
            logger.info("🔍 Detecting simplified market regime...")
            
            # Get Nifty data for trend analysis
            nifty_analysis = await self._analyze_nifty_trend()
            
            # Get volatility measures
            volatility_analysis = await self._analyze_market_volatility()
            
            # Classify regime with simplified logic
            regime_result = self._classify_regime_simplified(
                nifty_analysis, volatility_analysis
            )
            
            # Update current state
            self.current_regime = regime_result.regime
            self.regime_confidence = regime_result.confidence
            self.last_regime_update = datetime.now()
            
            logger.info(f"📊 Simplified Market Regime: {regime_result.regime} "
                       f"(Confidence: {regime_result.confidence:.1%}) "
                       f"Strategy: {regime_result.trading_strategy}")
            
            return regime_result
            
        except Exception as e:
            logger.error(f"Simplified regime detection failed: {e}")
            return RegimeAnalysis(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                trend_strength=0.0,
                volatility_level=15.0,
                trading_strategy="MEAN_REVERSION",
                risk_adjustment=1.0
            )
    
    async def _analyze_nifty_trend(self) -> Dict[str, float]:
        """Analyze Nifty trend strength and direction - simplified"""
        try:
            # Get Nifty data
            nifty_data = await self.market_data.get_live_market_data("NIFTY50")
            if not nifty_data or not nifty_data.get("quote"):
                return {"trend_strength": 0.0, "direction": 0.0, "gap_pct": 0.0}
            
            quote = nifty_data["quote"]
            
            # Calculate gap from previous close
            gap_pct = ((quote["ltp"] - quote["prev_close"]) / quote["prev_close"]) * 100
            
            # Simplified trend calculation
            trend_strength = abs(gap_pct)
            direction = 1.0 if gap_pct > 0 else -1.0
            
            return {
                "trend_strength": trend_strength,
                "direction": direction,
                "gap_pct": gap_pct
            }
            
        except Exception as e:
            logger.error(f"Nifty trend analysis failed: {e}")
            return {"trend_strength": 0.0, "direction": 0.0, "gap_pct": 0.0}
    
    async def _analyze_market_volatility(self) -> Dict[str, Any]:
        """Analyze market volatility - simplified"""
        try:
            # Sample high-volume stocks for volatility analysis
            volatility_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
            volatility_measures = []
            for stock in volatility_stocks:
                try:
                    stock_data = await self.market_data.get_live_market_data(stock)
                    if stock_data and stock_data.get("quote"):
                        quote = stock_data["quote"]
                        change_pct = abs((quote["ltp"] - quote["prev_close"]) / quote["prev_close"]) * 100
                        volatility_measures.append(change_pct)
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.debug(f"Volatility analysis failed for {stock}: {e}")
                    continue
            if volatility_measures:
                avg_volatility = np.mean(volatility_measures)
                volatility_level = min(30.0, max(5.0, float(avg_volatility) * 2))  # Scale and cap
            else:
                volatility_level = 15.0
            return {
                "volatility_level": volatility_level,
                "volatility_regime": "HIGH" if volatility_level > 20 else "NORMAL" if volatility_level > 10 else "LOW"
            }
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {"volatility_level": 15.0, "volatility_regime": "NORMAL"}
    
    def _classify_regime_simplified(self, 
                        nifty_analysis: Dict,
                                  volatility_analysis: Dict) -> RegimeAnalysis:
        """Simplified regime classification - only 3 regimes"""
        try:
            trend_strength = nifty_analysis.get("trend_strength", 0.0)
            direction = nifty_analysis.get("direction", 0.0)
            volatility_level = volatility_analysis.get("volatility_level", 15.0)
            
            # Simplified classification logic
            regime = MarketRegime.SIDEWAYS  # Default
            trading_strategy = "MEAN_REVERSION"  # Default
            confidence = 0.5
            risk_adjustment = 1.0
            
            # Bullish regime: strong positive trend
            if trend_strength > 0.75 and direction > 0:
                regime = MarketRegime.BULLISH
                trading_strategy = "MOMENTUM"
                confidence = 0.7
                risk_adjustment = 1.1
                
            # Bearish regime: strong negative trend
            elif trend_strength > 0.75 and direction < 0:
                regime = MarketRegime.BEARISH
                trading_strategy = "MOMENTUM"
                confidence = 0.7
                risk_adjustment = 0.9
                
            # Sideways regime: weak trend or high volatility
            else:
                regime = MarketRegime.SIDEWAYS
                trading_strategy = "MEAN_REVERSION"
                confidence = 0.6
                risk_adjustment = 1.0
            
            return RegimeAnalysis(
                regime=regime,
                confidence=confidence,
                trend_strength=trend_strength,
                volatility_level=volatility_level,
                trading_strategy=trading_strategy,
                risk_adjustment=risk_adjustment
            )
            
        except Exception as e:
            logger.error(f"Simplified regime classification failed: {e}")
            return RegimeAnalysis(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                trend_strength=0.0,
                volatility_level=15.0,
                trading_strategy="MEAN_REVERSION",
                risk_adjustment=1.0
            )
    
    def get_strategy_for_regime(self, regime: str) -> Dict[str, Any]:
        """Get simplified trading strategy parameters for current regime"""
        strategy_configs = {
            MarketRegime.BULLISH: {
                "strategy": "MOMENTUM",
                "confidence_threshold": 0.65,
                "risk_reward_ratio": 2.0,
                "max_trades_per_day": 3,
                "position_size_multiplier": 1.1
            },
            MarketRegime.BEARISH: {
                "strategy": "MOMENTUM",
                "confidence_threshold": 0.70,
                "risk_reward_ratio": 2.0,
                "max_trades_per_day": 3,
                "position_size_multiplier": 0.9
            },
            MarketRegime.SIDEWAYS: {
                "strategy": "MEAN_REVERSION",
                "confidence_threshold": 0.75,
                "risk_reward_ratio": 1.8,
                "max_trades_per_day": 2,
                "position_size_multiplier": 1.0
            }
        }
        
        return strategy_configs.get(regime, strategy_configs[MarketRegime.SIDEWAYS])
    
    async def detect_current_regime(self):
        """
        Detect current market regime - simplified compatibility method
        """
        try:
            regime_analysis = await self.detect_market_regime()
            return {
                'regime': regime_analysis.regime,
                'confidence': regime_analysis.confidence,
                'timestamp': datetime.now().isoformat(),
                'analysis': f'Simplified regime detection: {regime_analysis.regime}',
                'source': 'simplified_regime_detector',
                'trading_strategy': regime_analysis.trading_strategy,
                'risk_adjustment': regime_analysis.risk_adjustment
            }
        except Exception as e:
            # Error fallback
            return {
                'regime': MarketRegime.SIDEWAYS,
                'confidence': 0.5,
                'timestamp': datetime.now().isoformat(),
                'analysis': f'Error in simplified regime detection: {e}',
                'source': 'error_fallback'
            }
