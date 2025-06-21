# ================================================================
# backend/app/services/regime_detector.py
# Market Regime Detection and Engine Switching
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
    """Market regime classification"""
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    SIDEWAYS_CHOPPY = "SIDEWAYS_CHOPPY" 
    GAP_DAY = "GAP_DAY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

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
        self.current_regime = MarketRegime.SIDEWAYS_CHOPPY
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
                regime=MarketRegime.SIDEWAYS_CHOPPY,
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
            regime = MarketRegime.SIDEWAYS_CHOPPY  # Default
            trading_strategy = "MEAN_REVERSION"    # Default
            confidence = 0.5
            risk_adjustment = 1.0
            
            # Gap day detection
            if gap_pct > 1.0:  # >1% gap
                regime = MarketRegime.GAP_DAY
                trading_strategy = "BREAKOUT"
                confidence = 0.8
                risk_adjustment = 1.2  # Increase risk in gap days
                
            # Trending market detection
            elif trend_strength > 0.75 and sector_dispersion < 2.0:
                if nifty_analysis.get("direction", 0) > 0:
                    regime = MarketRegime.TRENDING_BULLISH
                else:
                    regime = MarketRegime.TRENDING_BEARISH
                trading_strategy = "MOMENTUM"
                confidence = 0.75
                risk_adjustment = 1.1
                
            # High volatility detection
            elif volatility_level > 20:
                regime = MarketRegime.HIGH_VOLATILITY
                trading_strategy = "MEAN_REVERSION"
                confidence = 0.7
                risk_adjustment = 0.8  # Reduce risk in high volatility
                
            # Low volatility detection
            elif volatility_level < 12 and sector_dispersion < 1.5:
                regime = MarketRegime.LOW_VOLATILITY
                trading_strategy = "BREAKOUT"
                confidence = 0.6
                risk_adjustment = 1.0
                
            # Choppy/sideways (default)
            else:
                regime = MarketRegime.SIDEWAYS_CHOPPY
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
                regime=MarketRegime.SIDEWAYS_CHOPPY,
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
            MarketRegime.TRENDING_BULLISH: {
                "strategy": "MOMENTUM",
                "confidence_threshold": 0.65,
                "risk_reward_ratio": 2.5,
                "max_trades_per_day": 4,
                "position_size_multiplier": 1.1
            },
            MarketRegime.TRENDING_BEARISH: {
                "strategy": "MOMENTUM_SHORT",
                "confidence_threshold": 0.70,
                "risk_reward_ratio": 2.0,
                "max_trades_per_day": 3,
                "position_size_multiplier": 1.0
            },
            MarketRegime.SIDEWAYS_CHOPPY: {
                "strategy": "MEAN_REVERSION",
                "confidence_threshold": 0.75,
                "risk_reward_ratio": 1.8,
                "max_trades_per_day": 2,
                "position_size_multiplier": 0.9
            },
            MarketRegime.GAP_DAY: {
                "strategy": "BREAKOUT",
                "confidence_threshold": 0.80,
                "risk_reward_ratio": 3.0,
                "max_trades_per_day": 2,
                "position_size_multiplier": 1.2
            },
            MarketRegime.HIGH_VOLATILITY: {
                "strategy": "MEAN_REVERSION",
                "confidence_threshold": 0.80,
                "risk_reward_ratio": 1.5,
                "max_trades_per_day": 1,
                "position_size_multiplier": 0.7
            },
            MarketRegime.LOW_VOLATILITY: {
                "strategy": "BREAKOUT",
                "confidence_threshold": 0.65,
                "risk_reward_ratio": 2.2,
                "max_trades_per_day": 3,
                "position_size_multiplier": 1.0
            }
        }
        
        return strategy_configs.get(regime, strategy_configs[MarketRegime.SIDEWAYS_CHOPPY])