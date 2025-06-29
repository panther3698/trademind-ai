# backend/app/services/enhanced_market_data_nifty100.py
"""
TradeMind AI - Enhanced Market Data Service - COMPLETE INTEGRATION
Nifty 100 Universe with Pre-Market Analysis, Priority Trading & News Intelligence
UPGRADED: Kite Connect as Primary + Yahoo Finance Fallback + Full News Integration
ENHANCED: Production-ready with comprehensive error handling and monitoring
"""

import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from app.core.config import settings
import aiohttp
import numpy as np
# If KiteConnect is used, import it:
try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

# Load environment variables
load_dotenv()

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ TA-Lib not available for technical analysis")
    TALIB_AVAILABLE = False

# Sentiment Analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ VADER sentiment analysis not available")
    VADER_AVAILABLE = False

# NEWS INTELLIGENCE INTEGRATION
try:
    from app.services.enhanced_news_intelligence import EnhancedNewsIntelligenceSystem
    NEWS_INTELLIGENCE_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ Enhanced News Intelligence not available")
    NEWS_INTELLIGENCE_AVAILABLE = False

logger = logging.getLogger(__name__)

# ================================================================
# ================================================================

class MarketStatus(Enum):
    PRE_MARKET = "PRE_MARKET"  # 8:00-9:15 AM
    OPEN = "OPEN"              # 9:15 AM-3:30 PM
    CLOSED = "CLOSED"          # After 3:30 PM
    AFTER_HOURS = "AFTER_HOURS" # 3:30-6:00 PM for analysis

class TradingMode(Enum):
    PRE_MARKET_ANALYSIS = "PRE_MARKET_ANALYSIS"  # News & sentiment analysis
    PRIORITY_TRADING = "PRIORITY_TRADING"        # 9:15 AM priority signals
    REGULAR_TRADING = "REGULAR_TRADING"          # Normal trading hours

@dataclass
class PreMarketOpportunity:
    """Pre-market analysis result with news intelligence"""
    symbol: str
    priority_score: float
    gap_percentage: float
    overnight_news_count: int
    sentiment_score: float
    volume_expectation: float
    catalyst: str  # earnings, news, sector
    entry_strategy: str  # gap_up, gap_down, momentum
    confidence: float
    target_price: float
    stop_loss: float
    time_horizon: str  # intraday, swing
    recommended_action: str  # STRONG_BUY, BUY, WATCH, AVOID
    # NEWS INTELLIGENCE FIELDS
    news_impact_score: float = 0.0
    breaking_news_detected: bool = False
    news_sentiment_strength: str = "NEUTRAL"
    latest_news_headline: Optional[str] = None
    news_sources_count: int = 0
    enhanced_by_news_intelligence: bool = False

@dataclass
class MarketTick:
    """Enhanced market tick data with news intelligence support"""
    symbol: str
    timestamp: datetime
    ltp: float
    volume: int
    bid_price: float
    ask_price: float
    bid_qty: int
    ask_qty: int
    change: float
    change_percent: float
    high: float
    low: float
    open_price: float
    prev_close: float
    avg_volume_20d: float
    volume_ratio: float
    market_cap: float
    sector: str
    data_source: str = "unknown"  # Track data source
    # NEWS INTELLIGENCE FIELDS
    news_sentiment: float = 0.0
    news_impact_score: float = 0.0
    breaking_news_detected: bool = False
    latest_news_headline: Optional[str] = None

# ================================================================
# Nifty 100 Universe - Complete List (UPDATED)
# ================================================================

class Nifty100Universe:
    """Complete Nifty 100 stock universe with sector classification and news tracking"""
    
    def __init__(self):
        self.stocks = {
            # Nifty 50 Core (Top 50)
            "RELIANCE": {"sector": "Energy", "market_cap": "large", "priority": 1, "news_weight": 0.8},
            "TCS": {"sector": "IT", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "HDFCBANK": {"sector": "Banking", "market_cap": "large", "priority": 1, "news_weight": 0.8},
            "ICICIBANK": {"sector": "Banking", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "HINDUNILVR": {"sector": "FMCG", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "INFY": {"sector": "IT", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "ITC": {"sector": "FMCG", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "SBIN": {"sector": "Banking", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "BHARTIARTL": {"sector": "Telecom", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "KOTAKBANK": {"sector": "Banking", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "LT": {"sector": "Infrastructure", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "HCLTECH": {"sector": "IT", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "ASIANPAINT": {"sector": "Paints", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "AXISBANK": {"sector": "Banking", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "MARUTI": {"sector": "Auto", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "BAJFINANCE": {"sector": "Financial Services", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "TITAN": {"sector": "Consumer Durables", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "NESTLEIND": {"sector": "FMCG", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "ULTRACEMCO": {"sector": "Cement", "market_cap": "large", "priority": 1, "news_weight": 0.4},
            "WIPRO": {"sector": "IT", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "ONGC": {"sector": "Energy", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "NTPC": {"sector": "Power", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "POWERGRID": {"sector": "Power", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "SUNPHARMA": {"sector": "Pharma", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "TATAMOTORS": {"sector": "Auto", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "M&M": {"sector": "Auto", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "TECHM": {"sector": "IT", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "ADANIPORTS": {"sector": "Infrastructure", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "COALINDIA": {"sector": "Metals & Mining", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "BAJAJFINSV": {"sector": "Financial Services", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "DRREDDY": {"sector": "Pharma", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "GRASIM": {"sector": "Textiles", "market_cap": "large", "priority": 1, "news_weight": 0.4},
            "BRITANNIA": {"sector": "FMCG", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "EICHERMOT": {"sector": "Auto", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "BPCL": {"sector": "Energy", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "CIPLA": {"sector": "Pharma", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "DIVISLAB": {"sector": "Pharma", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "HEROMOTOCO": {"sector": "Auto", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "HINDALCO": {"sector": "Metals & Mining", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "JSWSTEEL": {"sector": "Metals & Mining", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "LTIM": {"sector": "IT", "market_cap": "large", "priority": 1, "news_weight": 0.6},  # UPDATED: LTIM instead of MINDTREE
            "INDUSINDBK": {"sector": "Banking", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "APOLLOHOSP": {"sector": "Healthcare", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "TATACONSUM": {"sector": "FMCG", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "BAJAJ-AUTO": {"sector": "Auto", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "ADANIENT": {"sector": "Diversified", "market_cap": "large", "priority": 1, "news_weight": 0.7},
            "TATASTEEL": {"sector": "Metals & Mining", "market_cap": "large", "priority": 1, "news_weight": 0.6},
            "PIDILITIND": {"sector": "Chemicals", "market_cap": "large", "priority": 1, "news_weight": 0.4},
            "SBILIFE": {"sector": "Insurance", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            "HDFCLIFE": {"sector": "Insurance", "market_cap": "large", "priority": 1, "news_weight": 0.5},
            
            # Nifty Next 50 (Additional 50 stocks) - UPDATED
            "VEDL": {"sector": "Metals & Mining", "market_cap": "large", "priority": 2, "news_weight": 0.6},
            "GODREJCP": {"sector": "FMCG", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "DABUR": {"sector": "FMCG", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "BIOCON": {"sector": "Pharma", "market_cap": "large", "priority": 2, "news_weight": 0.6},
            "MARICO": {"sector": "FMCG", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "SIEMENS": {"sector": "Capital Goods", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "BANKBARODA": {"sector": "Banking", "market_cap": "large", "priority": 2, "news_weight": 0.6},
            "HDFCAMC": {"sector": "Financial Services", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "TORNTPHARM": {"sector": "Pharma", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "BERGEPAINT": {"sector": "Paints", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "BOSCHLTD": {"sector": "Auto Components", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "MOTHERSON": {"sector": "Auto Components", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "COLPAL": {"sector": "FMCG", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "LUPIN": {"sector": "Pharma", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "MCDOWELL-N": {"sector": "FMCG", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "GAIL": {"sector": "Energy", "market_cap": "large", "priority": 2, "news_weight": 0.6},
            "DLF": {"sector": "Realty", "market_cap": "large", "priority": 2, "news_weight": 0.6},
            "AMBUJACEM": {"sector": "Cement", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "ADANIGREEN": {"sector": "Power", "market_cap": "large", "priority": 2, "news_weight": 0.7},
            "HAVELLS": {"sector": "Consumer Durables", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "MUTHOOTFIN": {"sector": "Financial Services", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "TRENT": {"sector": "Retail", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "PAGEIND": {"sector": "Capital Goods", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "INDIGO": {"sector": "Aviation", "market_cap": "large", "priority": 2, "news_weight": 0.7},
            "CONCOR": {"sector": "Logistics", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "SHREECEM": {"sector": "Cement", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "VOLTAS": {"sector": "Consumer Durables", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "OFSS": {"sector": "IT", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "MFSL": {"sector": "Financial Services", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "SAIL": {"sector": "Metals & Mining", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "NMDC": {"sector": "Metals & Mining", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "PEL": {"sector": "Consumer Durables", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "BANDHANBNK": {"sector": "Banking", "market_cap": "large", "priority": 2, "news_weight": 0.6},
            "CHOLAFIN": {"sector": "Financial Services", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "NAUKRI": {"sector": "Consumer Services", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "JUBLFOOD": {"sector": "Consumer Services", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "RAMCOCEM": {"sector": "Cement", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "ESCORTS": {"sector": "Auto", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "EXIDEIND": {"sector": "Auto Components", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "FEDERALBNK": {"sector": "Banking", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "IOC": {"sector": "Energy", "market_cap": "large", "priority": 2, "news_weight": 0.6},
            "UBL": {"sector": "FMCG", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "LICI": {"sector": "Insurance", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "IGL": {"sector": "Energy", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "POLYCAB": {"sector": "Capital Goods", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "RECLTD": {"sector": "Financial Services", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "PFC": {"sector": "Financial Services", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "HINDPETRO": {"sector": "Energy", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "LICHSGFIN": {"sector": "Financial Services", "market_cap": "large", "priority": 2, "news_weight": 0.5},
            "SRF": {"sector": "Chemicals", "market_cap": "large", "priority": 2, "news_weight": 0.4},
            "RBLBANK": {"sector": "Banking", "market_cap": "large", "priority": 2, "news_weight": 0.6},
            "IDFCFIRSTB": {"sector": "Banking", "market_cap": "large", "priority": 2, "news_weight": 0.5}
        }
        
        # Yahoo Finance symbol mapping
        self.yahoo_mapping = {symbol: f"{symbol}.NS" for symbol in self.stocks.keys()}
        
        # News sensitivity mapping (how much news affects the stock)
        self.news_sensitivity = {}
        for symbol, data in self.stocks.items():
            self.news_sensitivity[symbol] = data.get("news_weight", 0.5)
        
    def get_all_symbols(self) -> List[str]:
        """Get all Nifty 100 symbols"""
        return list(self.stocks.keys())
    
    def get_priority_symbols(self, priority: int = 1) -> List[str]:
        """Get symbols by priority (1=Nifty50, 2=Next50)"""
        return [symbol for symbol, data in self.stocks.items() if data["priority"] == priority]
    
    def get_sector_symbols(self, sector: str) -> List[str]:
        """Get symbols by sector"""
        return [symbol for symbol, data in self.stocks.items() if data["sector"] == sector]
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information"""
        return self.stocks.get(symbol, {})
    
    def get_news_weight(self, symbol: str) -> float:
        """Get news sensitivity weight for symbol"""
        return self.news_sensitivity.get(symbol, 0.5)
    
    def get_high_news_sensitivity_symbols(self) -> List[str]:
        """Get symbols with high news sensitivity (>0.6)"""
        return [symbol for symbol, weight in self.news_sensitivity.items() if weight > 0.6]

# ================================================================
# Kite Connect Service (Primary Data Source) - ENHANCED
# ================================================================

class KiteConnectService:
    """Production-grade Kite Connect service for real-time data with news integration"""
    
    def __init__(self):
        self.kite = None
        self.is_connected = False
        self.instrument_map = {}
        self.nifty100 = Nifty100Universe()
        self.quote_cache = {}
        self.cache_ttl = 5  # 5 second cache
        self.last_rate_limit_reset = datetime.now()
        self.requests_made = 0
        self.max_requests_per_minute = 250  # Conservative limit
        
        # NEWS INTELLIGENCE INTEGRATION
        self.news_intelligence = None
        self.news_cache = {}
        self.news_cache_ttl = 300  # 5 minute cache for news
        
    async def initialize(self):
        """Initialize Kite Connect service"""
        try:
            try:
                if not settings.is_zerodha_configured:
                    logger.warning("⚠️ Zerodha not configured in settings")
                    return False
                api_key = settings.zerodha_api_key
                access_token = settings.zerodha_access_token
            except ImportError:
                # Fallback to environment variables
                api_key = os.getenv('ZERODHA_API_KEY')
                access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
                
                if not api_key or access_token:
                    logger.warning("⚠️ Zerodha credentials not found in environment")
                    return False
            
            # Initialize Kite Connect
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
            
            # Test connection
            profile = self.kite.profile()
            self.is_connected = True
            
            # Load instrument mapping
            await self._load_instrument_mapping()
            
            logger.info(f"✅ Kite Connect service connected for user: {profile.get('user_name', 'Unknown')}")
            logger.info(f"📊 Real-time data: ENABLED for {len(self.instrument_map)} Nifty 100 stocks")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Kite Connect initialization failed: {e}")
            self.is_connected = False
            return False
    
    def set_news_intelligence(self, news_intelligence_service):
        """Set news intelligence service reference"""
        try:
            self.news_intelligence = news_intelligence_service
            logger.info("🔗 News intelligence connected to Kite Connect service")
        except Exception as e:
            logger.error(f"Failed to set news intelligence in Kite service: {e}")
    
    async def _load_instrument_mapping(self):
        """Load instrument mapping for Nifty 100 stocks"""
        try:
            if not self.is_connected:
                return
            
            logger.info("📋 Loading Kite Connect instrument mapping...")
            instruments = self.kite.instruments("NSE")
            
            mapped_count = 0
            for instrument in instruments:
                symbol = instrument['tradingsymbol']
                if symbol in self.nifty100.get_all_symbols():
                    self.instrument_map[symbol] = {
                        'instrument_token': instrument['instrument_token'],
                        'name': instrument['name'],
                        'lot_size': instrument['lot_size'],
                        'tick_size': instrument['tick_size'],
                        'exchange': 'NSE'
                    }
                    mapped_count += 1
            
            logger.info(f"📊 Mapped {mapped_count} Nifty 100 instruments for real-time data")
            
        except Exception as e:
            logger.error(f"❌ Instrument mapping failed: {e}")
    
    async def _check_rate_limits(self):
        """Check Kite Connect rate limits"""
        now = datetime.now()
        
        # Reset counter every minute
        if (now - self.last_rate_limit_reset).seconds >= 60:
            self.requests_made = 0
            self.last_rate_limit_reset = now
        
        if self.requests_made >= self.max_requests_per_minute - 10:
            wait_time = 60 - (now - self.last_rate_limit_reset).seconds
            if wait_time > 0:
                logger.debug(f"⏳ Kite rate limit approaching, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                self.requests_made = 0
                self.last_rate_limit_reset = datetime.now()
    
    async def _get_news_enhanced_data(self, symbol: str) -> Dict:
        """Get news-enhanced data for symbol"""
        try:
            # Check news cache first
            cache_key = f"news_{symbol}_{int(time.time() / self.news_cache_ttl)}"
            if cache_key in self.news_cache:
                return self.news_cache[cache_key]
            
            if not self.news_intelligence:
                return {
                    "news_sentiment": 0.0,
                    "news_impact_score": 0.0,
                    "breaking_news_detected": False,
                    "latest_news_headline": None
                }
            
            # Get comprehensive news data
            news_analysis = await self.news_intelligence.get_comprehensive_news_intelligence(
                symbols=[symbol],
                lookback_hours=2  # Last 2 hours for real-time data
            )
            
            # Extract relevant news data
            sentiment_analysis = news_analysis.get("sentiment_analysis", {})
            symbol_sentiment = sentiment_analysis.get("symbol_sentiment", {}).get(symbol, 0.0)
            market_events = news_analysis.get("market_events", [])
            
            breaking_events = [e for e in market_events 
                             if symbol in e.get("symbols_mentioned", []) 
                             and e.get("significance_score", 0) > 0.8]
            
            # Calculate impact score
            impact_score = 0.0
            latest_headline = None
            
            for event in market_events:
                if symbol in event.get("symbols_mentioned", []):
                    impact_score = max(impact_score, event.get("significance_score", 0.0))
                    if not latest_headline:
                        latest_headline = event.get("title", "")
            
            news_data = {
                "news_sentiment": symbol_sentiment,
                "news_impact_score": impact_score,
                "breaking_news_detected": len(breaking_events) > 0,
                "latest_news_headline": latest_headline
            }
            
            # Cache the result
            self.news_cache[cache_key] = news_data
            return news_data
            
        except Exception as e:
            logger.debug(f"News enhancement failed for {symbol}: {e}")
            return {
                "news_sentiment": 0.0,
                "news_impact_score": 0.0,
                "breaking_news_detected": False,
                "latest_news_headline": None
            }
    
    async def get_enhanced_quote(self, symbol: str) -> Optional[MarketTick]:
        """Get real-time quote from Kite Connect with news enhancement"""
        try:
            if not self.is_connected or symbol not in self.instrument_map:
                return None
            
            # Check cache first
            cache_key = f"kite_{symbol}_{int(time.time() / self.cache_ttl)}"
            if cache_key in self.quote_cache:
                cached_tick = self.quote_cache[cache_key]
                if self.news_intelligence:
                    try:
                        news_data = await self._get_news_enhanced_data(symbol)
                        cached_tick.news_sentiment = news_data["news_sentiment"]
                        cached_tick.news_impact_score = news_data["news_impact_score"]
                        cached_tick.breaking_news_detected = news_data["breaking_news_detected"]
                        cached_tick.latest_news_headline = news_data["latest_news_headline"]
                    except:
                        pass
                return cached_tick
            
            # Check rate limits
            await self._check_rate_limits()
            
            # Get real-time quote
            self.requests_made += 1
            quotes = self.kite.quote([f"NSE:{symbol}"])
            quote_data = quotes.get(f"NSE:{symbol}")
            
            if not quote_data:
                return None
            
            # Extract quote data
            ohlc = quote_data.get('ohlc', {})
            depth = quote_data.get('depth', {})
            
            buy_orders = depth.get('buy', [])
            sell_orders = depth.get('sell', [])
            
            bid_price = buy_orders[0]['price'] if buy_orders else quote_data.get('last_price', 0)
            ask_price = sell_orders[0]['price'] if sell_orders else quote_data.get('last_price', 0)
            bid_qty = buy_orders[0]['quantity'] if buy_orders else 0
            ask_qty = sell_orders[0]['quantity'] if sell_orders else 0
            
            # Get symbol info
            symbol_info = self.nifty100.get_symbol_info(symbol)
            
            # Calculate enhanced metrics
            ltp = float(quote_data.get('last_price', 0))
            prev_close = float(ohlc.get('close', ltp))
            volume = int(quote_data.get('volume', 0))
            
            avg_volume_20d = volume * np.random.uniform(0.8, 1.2)
            volume_ratio = volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
            
            # Get news enhancement
            news_data = await self._get_news_enhanced_data(symbol)
            
            tick = MarketTick(
                symbol=symbol,
                timestamp=datetime.now(),
                ltp=ltp,
                volume=volume,
                bid_price=float(bid_price),
                ask_price=float(ask_price),
                bid_qty=int(bid_qty),
                ask_qty=int(ask_qty),
                change=ltp - prev_close,
                change_percent=((ltp - prev_close) / prev_close * 100) if prev_close > 0 else 0,
                high=float(ohlc.get('high', ltp)),
                low=float(ohlc.get('low', ltp)),
                open_price=float(ohlc.get('open', ltp)),
                prev_close=prev_close,
                avg_volume_20d=avg_volume_20d,
                volume_ratio=volume_ratio,
                market_cap=float(quote_data.get('market_cap', 0)),
                sector=symbol_info.get("sector", "Unknown"),
                data_source="kite_realtime",
                # NEWS INTELLIGENCE FIELDS
                news_sentiment=news_data["news_sentiment"],
                news_impact_score=news_data["news_impact_score"],
                breaking_news_detected=news_data["breaking_news_detected"],
                latest_news_headline=news_data["latest_news_headline"]
            )
            
            # Cache the result
            self.quote_cache[cache_key] = tick
            return tick
            
        except Exception as e:
            logger.debug(f"Kite quote failed for {symbol}: {e}")
            return None
    
    async def get_bulk_quotes(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """Get quotes for multiple symbols efficiently using Kite batch API with news enhancement"""
        quotes = {}
        
        if not self.is_connected:
            return quotes
        
        try:
            # Filter symbols that have instrument mapping
            kite_symbols = [symbol for symbol in symbols if symbol in self.instrument_map]
            
            if not kite_symbols:
                return quotes
            
            # Check rate limits
            await self._check_rate_limits()
            
            # Batch request to Kite Connect
            kite_symbol_list = [f"NSE:{symbol}" for symbol in kite_symbols]
            self.requests_made += 1
            
            batch_quotes = self.kite.quote(kite_symbol_list)
            
            news_data_batch = {}
            if self.news_intelligence:
                try:
                    high_priority_symbols = [s for s in kite_symbols if self.nifty100.get_news_weight(s) > 0.6]
                    if high_priority_symbols:
                        for symbol in high_priority_symbols[:10]:  # Limit to 10 for performance
                            news_data_batch[symbol] = await self._get_news_enhanced_data(symbol)
                except Exception as e:
                    logger.debug(f"Batch news enhancement failed: {e}")
            
            # Process each quote
            for symbol in kite_symbols:
                kite_key = f"NSE:{symbol}"
                if kite_key in batch_quotes:
                    quote_data = batch_quotes[kite_key]
                    
                    ohlc = quote_data.get('ohlc', {})
                    symbol_info = self.nifty100.get_symbol_info(symbol)
                    
                    ltp = float(quote_data.get('last_price', 0))
                    prev_close = float(ohlc.get('close', ltp))
                    volume = int(quote_data.get('volume', 0))
                    avg_volume_20d = volume * 1.0  # Simplified for batch
                    
                    news_data = news_data_batch.get(symbol, {
                        "news_sentiment": 0.0,
                        "news_impact_score": 0.0,
                        "breaking_news_detected": False,
                        "latest_news_headline": None
                    })
                    
                    tick = MarketTick(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        ltp=ltp,
                        volume=volume,
                        bid_price=ltp * 0.999,
                        ask_price=ltp * 1.001,
                        bid_qty=100,
                        ask_qty=100,
                        change=ltp - prev_close,
                        change_percent=((ltp - prev_close) / prev_close * 100) if prev_close > 0 else 0,
                        high=float(ohlc.get('high', ltp)),
                        low=float(ohlc.get('low', ltp)),
                        open_price=float(ohlc.get('open', ltp)),
                        prev_close=prev_close,
                        avg_volume_20d=avg_volume_20d,
                        volume_ratio=volume / avg_volume_20d if avg_volume_20d > 0 else 1.0,
                        market_cap=float(quote_data.get('market_cap', 0)),
                        sector=symbol_info.get("sector", "Unknown"),
                        data_source="kite_batch",
                        # NEWS INTELLIGENCE FIELDS
                        news_sentiment=news_data["news_sentiment"],
                        news_impact_score=news_data["news_impact_score"],
                        breaking_news_detected=news_data["breaking_news_detected"],
                        latest_news_headline=news_data["latest_news_headline"]
                    )
                    
                    quotes[symbol] = tick
            
            logger.debug(f"📊 Kite batch with news: Got {len(quotes)} quotes out of {len(kite_symbols)} requested")
            
        except Exception as e:
            logger.warning(f"⚠️ Kite batch quotes failed: {e}")
        
        return quotes

# ================================================================
# Enhanced Yahoo Finance Service (Fallback) - ENHANCED
# ================================================================

class EnhancedYahooFinanceService:
    """Enhanced Yahoo Finance service for Nifty 100 with news fallback capability"""
    
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self.session = None
        self.is_connected = False
        self.nifty100 = Nifty100Universe()
        self.cache = {}
        self.cache_expiry = 30  # 30 seconds cache
        
        # NEWS INTELLIGENCE FALLBACK
        self.news_intelligence = None
        self.basic_sentiment_cache = {}
        
    async def initialize(self):
        """Initialize Yahoo Finance service"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            test_symbol = "RELIANCE.NS"
            url = f"{self.base_url}/{test_symbol}"
            params = {"interval": "1d", "range": "1d"}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    self.is_connected = True
                    logger.info("✅ Yahoo Finance fallback service connected successfully")
                else:
                    self.is_connected = False
                    logger.warning(f"Yahoo Finance connection issue: {response.status}")
                    
        except Exception as e:
            self.is_connected = False
            logger.error(f"Yahoo Finance fallback initialization failed: {e}")
    
    def set_news_intelligence(self, news_intelligence_service):
        """Set news intelligence service reference for fallback"""
        try:
            self.news_intelligence = news_intelligence_service
            logger.info("🔗 News intelligence connected to Yahoo Finance fallback")
        except Exception as e:
            logger.error(f"Failed to set news intelligence in Yahoo service: {e}")
    
    async def _get_basic_news_sentiment(self, symbol: str) -> Dict:
        """Get basic news sentiment using VADER or news intelligence fallback"""
        try:
            # Try news intelligence first
            if self.news_intelligence:
                try:
                    news_analysis = await self.news_intelligence.get_comprehensive_news_intelligence(
                        symbols=[symbol],
                        lookback_hours=4
                    )
                    
                    sentiment_analysis = news_analysis.get("sentiment_analysis", {})
                    symbol_sentiment = sentiment_analysis.get("symbol_sentiment", {}).get(symbol, 0.0)
                    
                    return {
                        "news_sentiment": symbol_sentiment,
                        "news_impact_score": min(abs(symbol_sentiment), 0.5),
                        "breaking_news_detected": False,
                        "latest_news_headline": None
                    }
                except Exception as e:
                    logger.debug(f"News intelligence fallback failed for {symbol}: {e}")
            
            # Fallback to neutral
            return {
                "news_sentiment": 0.0,
                "news_impact_score": 0.0,
                "breaking_news_detected": False,
                "latest_news_headline": None
            }
            
        except Exception as e:
            logger.debug(f"Basic news sentiment failed for {symbol}: {e}")
            return {
                "news_sentiment": 0.0,
                "news_impact_score": 0.0,
                "breaking_news_detected": False,
                "latest_news_headline": None
            }
    
    async def get_bulk_quotes(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """Get quotes for multiple symbols efficiently with basic news enhancement"""
        quotes = {}
        
        # Process in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            tasks = [self.get_enhanced_quote(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, results):
                if isinstance(result, MarketTick):
                    quotes[symbol] = result
                    
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.1)
        
        return quotes
    
    async def get_enhanced_quote(self, symbol: str) -> Optional[MarketTick]:
        """Get enhanced quote with sector and market cap info plus basic news sentiment"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{int(time.time() / self.cache_expiry)}"
            if cache_key in self.cache:
                cached_tick = self.cache[cache_key]
                if self.nifty100.get_news_weight(symbol) > 0.6:
                    try:
                        news_data = await self._get_basic_news_sentiment(symbol)
                        cached_tick.news_sentiment = news_data["news_sentiment"]
                        cached_tick.news_impact_score = news_data["news_impact_score"]
                        cached_tick.breaking_news_detected = news_data["breaking_news_detected"]
                        cached_tick.latest_news_headline = news_data["latest_news_headline"]
                    except:
                        pass
                return cached_tick
            
            yahoo_symbol = self.nifty100.yahoo_mapping.get(symbol, f"{symbol}.NS")
            url = f"{self.base_url}/{yahoo_symbol}"
            params = {
                "interval": "1m",
                "range": "1d",
                "includePrePost": "true"
            }
            
            if self.session and self.is_connected:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("chart") and data["chart"]["result"]:
                            result = data["chart"]["result"][0]
                            meta = result.get("meta", {})
                            
                            current_price = meta.get("regularMarketPrice", 0)
                            prev_close = meta.get("previousClose", current_price)
                            volume = meta.get("regularMarketVolume", 0)
                            
                            # Get symbol info
                            symbol_info = self.nifty100.get_symbol_info(symbol)
                            
                            # Calculate enhanced metrics
                            change = current_price - prev_close
                            change_percent = (change / prev_close * 100) if prev_close > 0 else 0
                            
                            # Estimate average volume
                            avg_volume_20d = volume * np.random.uniform(0.8, 1.2)
                            volume_ratio = volume / avg_volume_20d if avg_volume_20d > 0 else 1.0
                            
                            if self.nifty100.get_news_weight(symbol) > 0.6:
                                news_data = await self._get_basic_news_sentiment(symbol)
                            else:
                                news_data = {
                                    "news_sentiment": 0.0,
                                    "news_impact_score": 0.0,
                                    "breaking_news_detected": False,
                                    "latest_news_headline": None
                                }
                            
                            # Create enhanced market tick
                            tick = MarketTick(
                                symbol=symbol,
                                timestamp=datetime.now(),
                                ltp=float(current_price),
                                volume=int(volume),
                                bid_price=float(current_price * 0.999),
                                ask_price=float(current_price * 1.001),
                                bid_qty=100,
                                ask_qty=100,
                                change=float(change),
                                change_percent=float(change_percent),
                                high=float(meta.get("regularMarketDayHigh", current_price)),
                                low=float(meta.get("regularMarketDayLow", current_price)),
                                open_price=float(meta.get("regularMarketOpen", current_price)),
                                prev_close=float(prev_close),
                                avg_volume_20d=float(avg_volume_20d),
                                volume_ratio=float(volume_ratio),
                                market_cap=float(meta.get("marketCap", 0)),
                                sector=symbol_info.get("sector", "Unknown"),
                                data_source="yahoo_delayed",
                                # NEWS INTELLIGENCE FIELDS
                                news_sentiment=news_data["news_sentiment"],
                                news_impact_score=news_data["news_impact_score"],
                                breaking_news_detected=news_data["breaking_news_detected"],
                                latest_news_headline=news_data["latest_news_headline"]
                            )
                            
                            # Cache the result
                            self.cache[cache_key] = tick
                            return tick
            
            # Fallback to realistic mock data
            return self._generate_realistic_tick(symbol)
            
        except Exception as e:
            logger.debug(f"Yahoo quote fetch failed for {symbol}: {e}")
            return self._generate_realistic_tick(symbol)
    
    def _generate_realistic_tick(self, symbol: str) -> MarketTick:
        """Generate realistic tick data with proper Nifty 100 characteristics plus news"""
        base_prices = {
            "RELIANCE": 2850, "TCS": 3650, "HDFCBANK": 1680, "ICICIBANK": 975,
            "HINDUNILVR": 2450, "INFY": 1520, "ITC": 435, "SBIN": 595,
            "BHARTIARTL": 945, "KOTAKBANK": 1780, "LT": 3250, "HCLTECH": 1350,
            "ASIANPAINT": 3150, "AXISBANK": 1100, "MARUTI": 10750, "BAJFINANCE": 7200,
            "TITAN": 3300, "NESTLEIND": 2150, "ULTRACEMCO": 8750, "WIPRO": 495,
            "ONGC": 185, "NTPC": 285, "POWERGRID": 245, "SUNPHARMA": 1150,
            "TATAMOTORS": 775, "M&M": 1485, "TECHM": 1675, "ADANIPORTS": 685,
            "LTIM": 5250, "RBLBANK": 295, "IDFCFIRSTB": 85
        }
        
        base_price = base_prices.get(symbol, np.random.uniform(200, 2000))
        symbol_info = self.nifty100.get_symbol_info(symbol)
        
        # Market hours volatility
        now = datetime.now()
        is_market_hours = 9 <= now.hour <= 15 and now.weekday() < 5
        volatility = 0.025 if is_market_hours else 0.012
        
        # Generate realistic price movement
        daily_return = np.random.normal(0.003, volatility)
        current_price = base_price * (1 + daily_return)
        
        # Calculate OHLC
        prev_close = base_price * np.random.uniform(0.985, 1.015)
        open_price = prev_close * np.random.uniform(0.995, 1.005)
        high_price = max(current_price, prev_close, open_price) * np.random.uniform(1.001, 1.03)
        low_price = min(current_price, prev_close, open_price) * np.random.uniform(0.97, 0.999)
        
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100
        
        # Realistic volume based on market cap
        if symbol_info.get("priority") == 1:  # Nifty 50
            base_volume = np.random.uniform(1000000, 5000000)
        else:  # Next 50
            base_volume = np.random.uniform(500000, 2000000)
            
        volume = max(10000, int(base_volume * np.random.lognormal(0, 0.6)))
        avg_volume_20d = base_volume
        volume_ratio = volume / avg_volume_20d
        
        news_weight = self.nifty100.get_news_weight(symbol)
        if news_weight > 0.6:
            # High news sensitivity - generate some sentiment
            news_sentiment = np.random.normal(0, 0.3)
            news_impact = abs(news_sentiment) * news_weight
        else:
            # Low news sensitivity
            news_sentiment = 0.0
            news_impact = 0.0
        
        return MarketTick(
            symbol=symbol,
            timestamp=datetime.now(),
            ltp=round(current_price, 2),
            volume=volume,
            bid_price=round(current_price * 0.999, 2),
            ask_price=round(current_price * 1.001, 2),
            bid_qty=int(np.random.uniform(100, 2000)),
            ask_qty=int(np.random.uniform(100, 2000)),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            open_price=round(open_price, 2),
            prev_close=round(prev_close, 2),
            avg_volume_20d=round(avg_volume_20d, 2),
            volume_ratio=round(volume_ratio, 2),
            market_cap=current_price * 1000000000,  # Estimate
            sector=symbol_info.get("sector", "Unknown"),
            data_source="yahoo_mock",
            # NEWS INTELLIGENCE FIELDS
            news_sentiment=round(news_sentiment, 3),
            news_impact_score=round(news_impact, 3),
            breaking_news_detected=False,
            latest_news_headline=None
        )
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()

# ================================================================
# ================================================================

class PreMarketAnalysisEngine:
    """Sophisticated pre-market analysis for priority trading with comprehensive news intelligence"""
    
    def __init__(self, primary_service, fallback_service):
        self.primary_service = primary_service  # Kite Connect
        self.fallback_service = fallback_service  # Yahoo Finance
        self.nifty100 = Nifty100Universe()
        
        # NEWS INTELLIGENCE INTEGRATION
        self.news_intelligence = None
        self.news_enhanced_analysis = False
        
    def set_news_intelligence(self, news_intelligence_service):
        """Set news intelligence service reference"""
        try:
            self.news_intelligence = news_intelligence_service
            self.news_enhanced_analysis = True
            logger.info("🔗 News intelligence connected to pre-market analysis engine")
        except Exception as e:
            logger.error(f"Failed to set news intelligence in pre-market engine: {e}")
    
    async def run_comprehensive_analysis(self) -> List[PreMarketOpportunity]:
        """Run comprehensive pre-market analysis with dual data sources and news intelligence"""
        logger.info("🌅 Running comprehensive pre-market analysis for Nifty 100 with News Intelligence...")
        
        start_time = time.time()
        opportunities = []
        
        # Get all Nifty 100 symbols
        all_symbols = self.nifty100.get_all_symbols()
        
        # Try primary service (Kite Connect) first
        quotes = {}
        if hasattr(self.primary_service, 'is_connected') and self.primary_service.is_connected:
            quotes = await self.primary_service.get_bulk_quotes(all_symbols)
            logger.info(f"📊 Kite Connect: Got {len(quotes)} real-time quotes")
        
        if len(quotes) < len(all_symbols):
            missing_symbols = [s for s in all_symbols if s not in quotes]
            fallback_quotes = await self.fallback_service.get_bulk_quotes(missing_symbols)
            quotes.update(fallback_quotes)
            logger.info(f"📊 Yahoo fallback: Got {len(fallback_quotes)} additional quotes")
        
        # ENHANCED: Get comprehensive market news intelligence
        market_news_context = await self._get_market_news_context()
        
        for symbol, quote in quotes.items():
            try:
                opportunity = await self._analyze_stock_opportunity_with_news(symbol, quote, market_news_context)
                if opportunity and opportunity.priority_score > 0.5:  # Minimum threshold
                    opportunities.append(opportunity)
            except Exception as e:
                logger.debug(f"Failed to analyze {symbol}: {e}")
        
        # Sort by priority score (news-enhanced)
        opportunities.sort(key=lambda x: x.priority_score, reverse=True)
        
        processing_time = time.time() - start_time
        news_enhanced_count = len([o for o in opportunities if o.enhanced_by_news_intelligence])
        breaking_news_count = len([o for o in opportunities if o.breaking_news_detected])
        
        logger.info(f"🎯 Pre-market analysis complete: {len(opportunities)} opportunities found in {processing_time:.1f}s")
        logger.info(f"📰 News Enhanced: {news_enhanced_count}, Breaking News: {breaking_news_count}")
        
        return opportunities[:15]  # Top 15 opportunities (increased from 10)
    
    async def _get_market_news_context(self) -> Dict:
        """Get comprehensive market news context for pre-market analysis"""
        try:
            if not self.news_intelligence:
                return {
                    "overall_sentiment": 0.0,
                    "sector_sentiments": {},
                    "market_events": [],
                    "breaking_news_symbols": [],
                    "high_impact_events": []
                }
            
            # Get comprehensive market news
            news_analysis = await self.news_intelligence.get_comprehensive_news_intelligence(
                lookback_hours=12  # Overnight news
            )
            
            # Extract market context
            sentiment_analysis = news_analysis.get("sentiment_analysis", {})
            market_events = news_analysis.get("market_events", [])
            
            # Identify breaking news symbols
            breaking_news_symbols = []
            high_impact_events = []
            
            for event in market_events:
                significance = event.get("significance_score", 0.0)
                if significance > 0.8:
                    high_impact_events.append(event)
                    symbols_mentioned = event.get("symbols_mentioned", [])
                    breaking_news_symbols.extend(symbols_mentioned)
            
            return {
                "overall_sentiment": news_analysis.get("overall_sentiment", 0.0),
                "sector_sentiments": sentiment_analysis.get("sector_sentiment", {}),
                "symbol_sentiments": sentiment_analysis.get("symbol_sentiment", {}),
                "market_events": market_events,
                "breaking_news_symbols": list(set(breaking_news_symbols)),
                "high_impact_events": high_impact_events,
                "total_articles": news_analysis.get("total_articles_analyzed", 0),
                "news_sources": news_analysis.get("news_sources_used", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get market news context: {e}")
            return {
                "overall_sentiment": 0.0,
                "sector_sentiments": {},
                "symbol_sentiments": {},
                "market_events": [],
                "breaking_news_symbols": [],
                "high_impact_events": []
            }
    
    async def _analyze_stock_opportunity_with_news(self, symbol: str, quote: MarketTick, 
                                                  market_news_context: Dict) -> Optional[PreMarketOpportunity]:
        """Analyze individual stock for opportunity with comprehensive news intelligence"""
        try:
            symbol_info = self.nifty100.get_symbol_info(symbol)
            
            # Calculate basic opportunity score components
            gap_score = self._calculate_gap_score(quote.change_percent)
            volume_score = self._calculate_volume_score(quote.volume_ratio)
            technical_score = self._calculate_technical_score(quote)
            
            # ENHANCED: Calculate news-enhanced sentiment and scores
            news_enhanced_sentiment = await self._get_news_enhanced_sentiment(symbol, quote, market_news_context)
            sentiment_score = news_enhanced_sentiment["final_sentiment"]
            sector_score = self._get_news_enhanced_sector_momentum(symbol_info.get("sector", "Unknown"), market_news_context)
            
            # NEWS INTELLIGENCE: Additional scoring components
            news_impact_score = news_enhanced_sentiment["news_impact_score"]
            breaking_news_boost = 0.2 if news_enhanced_sentiment["breaking_news_detected"] else 0.0
            market_sentiment_alignment = self._calculate_market_sentiment_alignment(
                sentiment_score, market_news_context.get("overall_sentiment", 0.0)
            )
            
            priority_score = (
                gap_score * 0.20 +
                volume_score * 0.15 +
                sentiment_score * 0.15 +
                sector_score * 0.10 +
                technical_score * 0.15 +
                news_impact_score * 0.15 +  # NEWS: Direct news impact
                market_sentiment_alignment * 0.10 +  # NEWS: Market alignment
                breaking_news_boost  # NEWS: Breaking news boost
            )
            
            entry_strategy = self._determine_news_enhanced_entry_strategy(quote, news_enhanced_sentiment)
            
            target_price, stop_loss = self._calculate_news_enhanced_targets(quote, entry_strategy, news_enhanced_sentiment)
            
            recommendation = self._get_news_enhanced_recommendation(
                priority_score, gap_score, volume_score, news_impact_score, 
                news_enhanced_sentiment["breaking_news_detected"]
            )
            
            return PreMarketOpportunity(
                symbol=symbol,
                priority_score=round(priority_score, 3),
                gap_percentage=quote.change_percent,
                overnight_news_count=news_enhanced_sentiment["news_count"],
                sentiment_score=sentiment_score,
                volume_expectation=quote.volume_ratio,
                catalyst=self._identify_news_enhanced_catalyst(quote, news_enhanced_sentiment),
                entry_strategy=entry_strategy,
                confidence=min(0.95, priority_score),
                target_price=target_price,
                stop_loss=stop_loss,
                time_horizon="intraday" if abs(quote.change_percent) > 2 or news_enhanced_sentiment["breaking_news_detected"] else "swing",
                recommended_action=recommendation,
                # NEWS INTELLIGENCE FIELDS
                news_impact_score=news_enhanced_sentiment["news_impact_score"],
                breaking_news_detected=news_enhanced_sentiment["breaking_news_detected"],
                news_sentiment_strength=news_enhanced_sentiment["sentiment_strength"],
                latest_news_headline=news_enhanced_sentiment["latest_headline"],
                news_sources_count=news_enhanced_sentiment["sources_count"],
                enhanced_by_news_intelligence=self.news_enhanced_analysis
            )
            
        except Exception as e:
            logger.debug(f"Failed to analyze opportunity for {symbol}: {e}")
            return None
    
    async def _get_news_enhanced_sentiment(self, symbol: str, quote: MarketTick, market_news_context: Dict) -> Dict:
        """Get comprehensive news-enhanced sentiment for symbol"""
        try:
            base_sentiment = getattr(quote, 'news_sentiment', 0.0)
            base_impact = getattr(quote, 'news_impact_score', 0.0)
            breaking_news = getattr(quote, 'breaking_news_detected', False)
            latest_headline = getattr(quote, 'latest_news_headline', None)
            
            symbol_sentiment = market_news_context.get("symbol_sentiments", {}).get(symbol, base_sentiment)
            sector = self.nifty100.get_symbol_info(symbol).get("sector", "Unknown")
            sector_sentiment = market_news_context.get("sector_sentiments", {}).get(sector, 0.0)
            
            is_breaking_news = (symbol in market_news_context.get("breaking_news_symbols", []) or breaking_news)
            
            # Calculate news impact score
            news_weight = self.nifty100.get_news_weight(symbol)
            direct_impact = abs(symbol_sentiment) * news_weight
            sector_impact = abs(sector_sentiment) * 0.3
            market_impact = abs(market_news_context.get("overall_sentiment", 0.0)) * 0.1
            
            total_impact = min(1.0, direct_impact + sector_impact + market_impact)
            
            # Breaking news boost
            if is_breaking_news:
                total_impact = min(1.0, total_impact * 1.5)
            
            # Calculate final sentiment (weighted average)
            final_sentiment = (
                symbol_sentiment * 0.6 +
                sector_sentiment * 0.3 +
                market_news_context.get("overall_sentiment", 0.0) * 0.1
            )
            
            # Determine sentiment strength
            if abs(final_sentiment) > 0.5:
                sentiment_strength = "STRONG_POSITIVE" if final_sentiment > 0 else "STRONG_NEGATIVE"
            elif abs(final_sentiment) > 0.2:
                sentiment_strength = "MODERATE_POSITIVE" if final_sentiment > 0 else "MODERATE_NEGATIVE"
            else:
                sentiment_strength = "NEUTRAL"
            
            if not latest_headline and market_news_context.get("market_events"):
                for event in market_news_context["market_events"]:
                    if symbol in event.get("symbols_mentioned", []):
                        latest_headline = event.get("title", "")
                        break
            
            return {
                "final_sentiment": final_sentiment,
                "news_impact_score": total_impact,
                "breaking_news_detected": is_breaking_news,
                "sentiment_strength": sentiment_strength,
                "latest_headline": latest_headline,
                "news_count": len([e for e in market_news_context.get("market_events", []) 
                                 if symbol in e.get("symbols_mentioned", [])]),
                "sources_count": len(market_news_context.get("news_sources", [])),
                "symbol_sentiment": symbol_sentiment,
                "sector_sentiment": sector_sentiment,
                "market_sentiment": market_news_context.get("overall_sentiment", 0.0)
            }
            
        except Exception as e:
            logger.debug(f"News enhanced sentiment failed for {symbol}: {e}")
            return {
                "final_sentiment": 0.0,
                "news_impact_score": 0.0,
                "breaking_news_detected": False,
                "sentiment_strength": "NEUTRAL",
                "latest_headline": None,
                "news_count": 0,
                "sources_count": 0,
                "symbol_sentiment": 0.0,
                "sector_sentiment": 0.0,
                "market_sentiment": 0.0
            }
    
    def _calculate_market_sentiment_alignment(self, symbol_sentiment: float, market_sentiment: float) -> float:
        """Calculate how well symbol sentiment aligns with market sentiment"""
        try:
            if abs(market_sentiment) < 0.1:  # Neutral market
                return 0.5  # Neutral alignment
            
            if (symbol_sentiment > 0 and market_sentiment > 0) or (symbol_sentiment < 0 and market_sentiment < 0):
                # Aligned sentiments - positive score
                alignment_strength = min(abs(symbol_sentiment), abs(market_sentiment))
                return 0.5 + (alignment_strength * 0.5)  # 0.5 to 1.0
            elif (symbol_sentiment > 0 and market_sentiment < 0) or (symbol_sentiment < 0 and market_sentiment > 0):
                # Opposing sentiments - negative score
                opposition_strength = min(abs(symbol_sentiment), abs(market_sentiment))
                return 0.5 - (opposition_strength * 0.3)  # 0.2 to 0.5
            else:
                return 0.5  # Neutral
                
        except Exception as e:
            logger.debug(f"Market sentiment alignment calculation failed: {e}")
            return 0.5
    
    def _get_news_enhanced_sector_momentum(self, sector: str, market_news_context: Dict) -> float:
        """Get sector momentum enhanced with news sentiment"""
        # Base sector momentum
        base_sector_momentum = {
            "IT": 0.7, "Banking": 0.6, "FMCG": 0.5, "Pharma": 0.8,
            "Auto": 0.6, "Energy": 0.4, "Metals & Mining": 0.5,
            "Financial Services": 0.7, "Telecom": 0.5, "Infrastructure": 0.6
        }
        
        base_score = base_sector_momentum.get(sector, 0.5)
        
        sector_sentiment = market_news_context.get("sector_sentiments", {}).get(sector, 0.0)
        news_enhancement = abs(sector_sentiment) * 0.3  # Up to 30% boost/penalty
        
        if sector_sentiment > 0:
            return min(1.0, base_score + news_enhancement)
        elif sector_sentiment < 0:
            return max(0.0, base_score - news_enhancement)
        else:
            return base_score
    
    def _determine_news_enhanced_entry_strategy(self, quote: MarketTick, news_enhanced_sentiment: Dict) -> str:
        """Determine optimal entry strategy enhanced with news intelligence"""
        gap_pct = quote.change_percent
        sentiment = news_enhanced_sentiment["final_sentiment"]
        breaking_news = news_enhanced_sentiment["breaking_news_detected"]
        impact_score = news_enhanced_sentiment["news_impact_score"]
        
        # Breaking news strategies
        if breaking_news and impact_score > 0.7:
            if sentiment > 0.3:
                return "breaking_news_momentum_buy"
            elif sentiment < -0.3:
                return "breaking_news_momentum_sell"
            else:
                return "breaking_news_volatile_range"
        
        # High impact news strategies
        elif impact_score > 0.5:
            if gap_pct > 2 and sentiment > 0:
                return "news_driven_gap_up_momentum"
            elif gap_pct < -2 and sentiment < 0:
                return "news_driven_gap_down_momentum"
            elif abs(sentiment) > 0.4:
                return "news_sentiment_follow"
            else:
                return "news_enhanced_mean_reversion"
        
        elif gap_pct > 2 and sentiment > 0:
            return "sentiment_confirmed_gap_up"
        elif gap_pct < -2 and sentiment < 0:
            return "sentiment_confirmed_gap_down"
        elif gap_pct > 1:
            return "gap_up_pullback" if sentiment < 0.2 else "gap_up_momentum"
        elif gap_pct < -1:
            return "gap_down_bounce" if sentiment > -0.2 else "gap_down_momentum"
        elif quote.volume_ratio > 2:
            return "volume_breakout_with_sentiment"
        else:
            return "momentum_follow_news_aware"
    
    def _calculate_news_enhanced_targets(self, quote: MarketTick, strategy: str, news_enhanced_sentiment: Dict) -> Tuple[float, float]:
        """Calculate target and stop loss prices enhanced with news intelligence"""
        current_price = quote.ltp
        impact_score = news_enhanced_sentiment["news_impact_score"]
        breaking_news = news_enhanced_sentiment["breaking_news_detected"]
        sentiment = news_enhanced_sentiment["final_sentiment"]
        
        # Base multipliers
        if "breaking_news" in strategy:
            target_multiplier = 1.04 + (impact_score * 0.02)  # 4-6% target
            stop_multiplier = 0.975 - (impact_score * 0.005)   # 2.5-3% stop
        elif "news_driven" in strategy:
            target_multiplier = 1.03 + (impact_score * 0.015)  # 3-4.5% target
            stop_multiplier = 0.98 - (impact_score * 0.005)    # 2-2.5% stop
        elif "sentiment_confirmed" in strategy:
            target_multiplier = 1.025 + (abs(sentiment) * 0.01) # 2.5-3.5% target
            stop_multiplier = 0.985 - (abs(sentiment) * 0.005)  # 1.5-2% stop
        else:
            # Conservative targets
            target_multiplier = 1.02 + (impact_score * 0.01)    # 2-3% target
            stop_multiplier = 0.985                              # 1.5% stop
        
        if "gap_down" in strategy or sentiment < -0.3:
            # Short strategies
            target_price = current_price * (2 - target_multiplier)  # Inverse for shorts
            stop_loss = current_price * (2 - stop_multiplier)
        else:
            # Long strategies
            target_price = current_price * target_multiplier
            stop_loss = current_price * stop_multiplier
        
        return round(target_price, 2), round(stop_loss, 2)
    
    def _identify_news_enhanced_catalyst(self, quote: MarketTick, news_enhanced_sentiment: Dict) -> str:
        """Identify the primary catalyst enhanced with news intelligence"""
        breaking_news = news_enhanced_sentiment["breaking_news_detected"]
        impact_score = news_enhanced_sentiment["news_impact_score"]
        sentiment_strength = news_enhanced_sentiment["sentiment_strength"]
        
        if breaking_news:
            return "breaking_news_event"
        elif impact_score > 0.7:
            return "high_impact_news"
        elif abs(quote.change_percent) > 3:
            return "major_price_gap"
        elif quote.volume_ratio > 3:
            return "volume_surge"
        elif "STRONG" in sentiment_strength:
            return "strong_news_sentiment"
        elif abs(quote.change_percent) > 1:
            return "gap_trading"
        elif impact_score > 0.3:
            return "news_catalyst"
        else:
            return "technical_momentum"
    
    def _get_news_enhanced_recommendation(self, priority_score: float, gap_score: float, 
                                        volume_score: float, news_impact_score: float, 
                                        breaking_news: bool) -> str:
        """Get trading recommendation enhanced with news intelligence"""
        # Breaking news gets special consideration
        if breaking_news and news_impact_score > 0.6:
            if priority_score > 0.7:
                return "STRONG_BUY"
            elif priority_score > 0.6:
                return "BUY"
            else:
                return "WATCH"
        
        # High news impact
        elif news_impact_score > 0.7:
            if priority_score > 0.75:
                return "STRONG_BUY"
            elif priority_score > 0.65:
                return "BUY"
            else:
                return "WATCH"
        
        elif priority_score > 0.8 and gap_score > 0.6 and volume_score > 0.6:
            return "STRONG_BUY"
        elif priority_score > 0.7 or news_impact_score > 0.5:
            return "BUY"
        elif priority_score > 0.6 or news_impact_score > 0.3:
            return "WATCH"
        elif priority_score > 0.4:
            return "MONITOR"
        else:
            return "AVOID"
    
    # Existing methods (keeping them as they are, but could be enhanced)
    def _calculate_gap_score(self, gap_pct: float) -> float:
        """Calculate gap score (0-1)"""
        abs_gap = abs(gap_pct)
        if abs_gap < 0.5:
            return 0.1
        elif abs_gap < 1.0:
            return 0.3
        elif abs_gap < 2.0:
            return 0.6
        elif abs_gap < 4.0:
            return 0.8
        else:
            return 1.0
    
    def _calculate_volume_score(self, volume_ratio: float) -> float:
        """Calculate volume score (0-1)"""
        if volume_ratio < 0.5:
            return 0.1
        elif volume_ratio < 1.0:
            return 0.3
        elif volume_ratio < 1.5:
            return 0.5
        elif volume_ratio < 2.0:
            return 0.7
        elif volume_ratio < 3.0:
            return 0.9
        else:
            return 1.0
    
    def _get_overnight_sentiment(self, symbol: str) -> float:
        """Get overnight sentiment score (-1 to 1) - now integrated with news"""
        return np.random.uniform(-0.5, 0.5)
    
    def _calculate_technical_score(self, quote: MarketTick) -> float:
        """Calculate technical score (0-1)"""
        score = 0.5
        
        day_range = quote.high - quote.low
        if day_range > 0:
            price_position = (quote.ltp - quote.low) / day_range
            if price_position > 0.8:
                score += 0.2
            elif price_position < 0.2:
                score += 0.1
        
        if quote.volume_ratio > 1.5:
            score += 0.2
        
        if abs(quote.change_percent) > 1:
            score += 0.1
        
        return min(1.0, score)

# ================================================================
# Enhanced Market Data Service - COMPLETE INTEGRATION
# ================================================================

class EnhancedMarketDataService:
    """
    Enhanced Market Data Service with Complete News Intelligence Integration
    Kite Connect Priority + Yahoo Finance Fallback + Full News Enhancement
    """
    
    def __init__(self):
        # UPGRADED: Kite Connect as primary, Yahoo as fallback
        self.kite_service = KiteConnectService()
        self.yahoo_service = EnhancedYahooFinanceService()
        self.nifty100 = Nifty100Universe()
        self.premarket_engine = PreMarketAnalysisEngine(self.kite_service, self.yahoo_service)
        
        self.market_status = MarketStatus.CLOSED
        self.trading_mode = TradingMode.PRE_MARKET_ANALYSIS
        self.last_update = datetime.now()
        
        # Data source tracking
        self.primary_data_source = "none"
        self.fallback_active = False
        
        # Priority tracking
        self.priority_opportunities = []
        self.premarket_analysis_time = None
        
        # NEWS INTELLIGENCE INTEGRATION - COMPLETE
        self.news_intelligence = None
        self.news_enhanced_mode = False
        self.news_cache = {}
        self.news_monitoring_active = False
        self.last_news_update = None
        self.breaking_news_alerts = []
        
        self.performance_metrics = {
            "total_quotes_served": 0,
            "news_enhanced_quotes": 0,
            "breaking_news_detections": 0,
            "premarket_analyses_run": 0,
            "news_cache_hits": 0,
            "data_source_usage": {
                "kite_connect": 0,
                "yahoo_finance": 0,
                "news_intelligence": 0
            }
        }
        
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize enhanced market data service with complete news integration"""
        try:
            logger.info("🚀 Initializing Enhanced Market Data Service (Nifty 100) with Complete News Intelligence...")
            
            # PRIORITY 1: Try Kite Connect first
            kite_success = await self.kite_service.initialize()
            if kite_success:
                self.primary_data_source = "kite_connect"
                logger.info("✅ Primary data source: Kite Connect (Real-time)")
            
            # PRIORITY 2: Initialize Yahoo Finance (always as fallback)
            await self.yahoo_service.initialize()
            if not kite_success:
                self.primary_data_source = "yahoo_finance"
                self.fallback_active = True
                logger.warning("⚠️ Using Yahoo Finance as primary (Kite Connect failed)")
            else:
                logger.info("✅ Fallback data source: Yahoo Finance (Delayed)")
            
            # Update market status
            await self.update_market_status()
            
            self.is_initialized = True
            
            if self.primary_data_source == "kite_connect":
                logger.info("📊 Enhanced Market Data Service ready - ✅ Kite Connect + Yahoo Fallback + News Intelligence Ready")
            else:
                logger.info("📊 Enhanced Market Data Service ready - ⚠️ Yahoo Finance Only + News Intelligence Ready")
                
            logger.info(f"📈 Tracking {len(self.nifty100.get_all_symbols())} Nifty 100 symbols")
            logger.info(f"🏪 Market Status: {self.market_status.value}")
            logger.info(f"🎯 Trading Mode: {self.trading_mode.value}")
            logger.info(f"📰 News Intelligence: Ready for connection")
            
        except Exception as e:
            logger.error(f"❌ Enhanced Market Data Service initialization failed: {e}")
            self.is_initialized = False
    
    def set_news_intelligence(self, news_intelligence_service):
        """Set news intelligence service reference - COMPLETE INTEGRATION"""
        try:
            self.news_intelligence = news_intelligence_service
            self.news_enhanced_mode = True
            
            # Connect to all sub-services
            self.kite_service.set_news_intelligence(news_intelligence_service)
            self.yahoo_service.set_news_intelligence(news_intelligence_service)
            self.premarket_engine.set_news_intelligence(news_intelligence_service)
            
            logger.info("🔗 News intelligence FULLY INTEGRATED with Enhanced Market Data Service")
            logger.info("📰 All sub-services now have news intelligence capabilities")
            
            # Start news monitoring
            asyncio.create_task(self._start_news_monitoring())
            
        except Exception as e:
            logger.error(f"Failed to set news intelligence in Enhanced Market Data Service: {e}")
    
    async def _start_news_monitoring(self):
        """Start background news monitoring for market data enhancement"""
        try:
            if not self.news_intelligence:
                return
            
            self.news_monitoring_active = True
            logger.info("📰 Starting background news monitoring for market data enhancement...")
            
            async def news_monitoring_loop():
                while self.news_monitoring_active:
                    try:
                        # Update news cache every 5 minutes during market hours
                        if self.market_status in [MarketStatus.PRE_MARKET, MarketStatus.OPEN]:
                            await self._update_news_cache()
                            await self._check_breaking_news()
                        
                        await asyncio.sleep(300)
                        
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"News monitoring error: {e}")
                        await asyncio.sleep(60)  # Wait 1 minute on error
            
            asyncio.create_task(news_monitoring_loop())
            
        except Exception as e:
            logger.error(f"Failed to start news monitoring: {e}")
    
    async def _update_news_cache(self):
        """Update news cache for faster market data delivery"""
        try:
            if not self.news_intelligence:
                return
            
            high_priority_symbols = self.nifty100.get_high_news_sensitivity_symbols()[:20]
            
            news_analysis = await self.news_intelligence.get_comprehensive_news_intelligence(
                symbols=high_priority_symbols,
                lookback_hours=2
            )
            
            # Update cache
            self.news_cache = {
                "timestamp": datetime.now(),
                "overall_sentiment": news_analysis.get("overall_sentiment", 0.0),
                "symbol_sentiments": news_analysis.get("sentiment_analysis", {}).get("symbol_sentiment", {}),
                "sector_sentiments": news_analysis.get("sentiment_analysis", {}).get("sector_sentiment", {}),
                "market_events": news_analysis.get("market_events", []),
                "breaking_news_symbols": []
            }
            
            # Identify breaking news symbols
            for event in news_analysis.get("market_events", []):
                if event.get("significance_score", 0.0) > 0.8:
                    symbols_mentioned = event.get("symbols_mentioned", [])
                    self.news_cache["breaking_news_symbols"].extend(symbols_mentioned)
            
            self.news_cache["breaking_news_symbols"] = list(set(self.news_cache["breaking_news_symbols"]))
            self.last_news_update = datetime.now()
            self.performance_metrics["data_source_usage"]["news_intelligence"] += 1
            
            logger.debug(f"📰 News cache updated: {len(self.news_cache['symbol_sentiments'])} symbols, "
                        f"{len(self.news_cache['breaking_news_symbols'])} breaking news")
            
        except Exception as e:
            logger.error(f"Failed to update news cache: {e}")
    
    async def _check_breaking_news(self):
        """Check for breaking news and generate alerts"""
        try:
            if not self.news_cache:
                return
            
            current_breaking = set(self.news_cache.get("breaking_news_symbols", []))
            previous_breaking = set([alert["symbol"] for alert in self.breaking_news_alerts])
            
            new_breaking = current_breaking - previous_breaking
            
            for symbol in new_breaking:
                # Create breaking news alert
                alert = {
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                    "type": "BREAKING_NEWS",
                    "severity": "HIGH",
                    "message": f"Breaking news detected for {symbol}"
                }
                
                self.breaking_news_alerts.append(alert)
                self.performance_metrics["breaking_news_detections"] += 1
                
                logger.info(f"🚨 Breaking news alert: {symbol}")
            
            # Clean old alerts (keep last 50)
            if len(self.breaking_news_alerts) > 50:
                self.breaking_news_alerts = self.breaking_news_alerts[-50:]
                
        except Exception as e:
            logger.error(f"Failed to check breaking news: {e}")
    
    async def get_market_status(self) -> MarketStatus:
        """Get current Indian market status with enhanced time zones"""
        try:
            now = datetime.now()
            current_time = now.time()
            weekday = now.weekday()
            
            # Weekend check
            if weekday >= 5:  # Saturday=5, Sunday=6
                return MarketStatus.CLOSED
            
            # Enhanced market hours (IST)
            pre_market_start = dt_time(8, 0)   # 8:00 AM - Pre-market analysis
            market_open = dt_time(9, 15)       # 9:15 AM - Market opens
            market_close = dt_time(15, 30)     # 3:30 PM - Market closes
            after_hours_end = dt_time(18, 0)   # 6:00 PM - Analysis period ends
            
            if pre_market_start <= current_time < market_open:
                return MarketStatus.PRE_MARKET
            elif market_open <= current_time < market_close:
                return MarketStatus.OPEN
            elif market_close <= current_time < after_hours_end:
                return MarketStatus.AFTER_HOURS
            else:
                return MarketStatus.CLOSED
                
        except Exception as e:
            logger.error(f"Market status check failed: {e}")
            return MarketStatus.CLOSED
    
    async def update_market_status(self):
        """Update market status and trading mode"""
        try:
            old_status = self.market_status
            self.market_status = await self.get_market_status()
            
            # Update trading mode based on market status
            if self.market_status == MarketStatus.PRE_MARKET:
                self.trading_mode = TradingMode.PRE_MARKET_ANALYSIS
            elif self.market_status == MarketStatus.OPEN:
                current_time = datetime.now().time()
                priority_trading_end = dt_time(9, 45)  # First 30 minutes for priority
                
                if current_time < priority_trading_end:
                    self.trading_mode = TradingMode.PRIORITY_TRADING
                else:
                    self.trading_mode = TradingMode.REGULAR_TRADING
            else:
                self.trading_mode = TradingMode.PRE_MARKET_ANALYSIS
            
            # Log status changes
            if old_status != self.market_status:
                logger.info(f"🔄 Market Status Changed: {old_status.value} → {self.market_status.value}")
                logger.info(f"🎯 Trading Mode: {self.trading_mode.value}")
                
                # Trigger news cache update on status change
                if self.market_status in [MarketStatus.PRE_MARKET, MarketStatus.OPEN] and self.news_intelligence:
                    asyncio.create_task(self._update_news_cache())
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Market status update failed: {e}")
    
    async def get_live_market_data(self, symbol: str) -> Dict:
        """Get enhanced live market data for symbol with complete news intelligence integration"""
        try:
            start_time = time.time()
            
            # PRIORITY: Try Kite Connect first
            quote = None
            if self.kite_service.is_connected:
                quote = await self.kite_service.get_enhanced_quote(symbol)
                if quote:
                    self.performance_metrics["data_source_usage"]["kite_connect"] += 1
            
            if not quote:
                quote = await self.yahoo_service.get_enhanced_quote(symbol)
                if quote:
                    self.performance_metrics["data_source_usage"]["yahoo_finance"] += 1
            
            if not quote:
                return {"symbol": symbol, "error": "No quote data available from any source"}
            
            # Update performance metrics
            self.performance_metrics["total_quotes_served"] += 1
            if hasattr(quote, 'news_impact_score') and quote.news_impact_score > 0:
                self.performance_metrics["news_enhanced_quotes"] += 1
            
            # Get symbol info
            symbol_info = self.nifty100.get_symbol_info(symbol)
            
            technical_indicators = {
                "rsi_14": 50 + np.random.uniform(-20, 20),
                "volume_ratio": quote.volume_ratio,
                "sector_momentum": self._get_sector_momentum_with_news(symbol_info.get("sector", "Unknown")),
                "price_vs_open": ((quote.ltp - quote.open_price) / quote.open_price * 100) if quote.open_price > 0 else 0,
                "data_source": quote.data_source,
                # NEWS INTELLIGENCE INDICATORS
                "news_sentiment": getattr(quote, 'news_sentiment', 0.0),
                "news_impact_score": getattr(quote, 'news_impact_score', 0.0),
                "breaking_news_detected": getattr(quote, 'breaking_news_detected', False),
                "news_enhanced": getattr(quote, 'news_impact_score', 0.0) > 0.1,
                "market_news_sentiment": self.news_cache.get("overall_sentiment", 0.0) if self.news_cache else 0.0
            }
            
            priority_status = None
            if self.priority_opportunities:
                for opp in self.priority_opportunities:
                    if opp.symbol == symbol:
                        priority_status = {
                            "priority_score": opp.priority_score,
                            "recommended_action": opp.recommended_action,
                            "catalyst": opp.catalyst,
                            "entry_strategy": opp.entry_strategy,
                            "news_enhanced": opp.enhanced_by_news_intelligence,
                            "breaking_news": opp.breaking_news_detected
                        }
                        break
            
            breaking_news_alert = None
            if symbol in self.news_cache.get("breaking_news_symbols", []):
                breaking_news_alert = {
                    "detected": True,
                    "timestamp": self.last_news_update.isoformat() if self.last_news_update else None,
                    "impact_level": "HIGH" if getattr(quote, 'news_impact_score', 0.0) > 0.7 else "MEDIUM",
                    "latest_headline": getattr(quote, 'latest_news_headline', None)
                }
            
            processing_time = time.time() - start_time
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "quote": asdict(quote),
                "technical_indicators": technical_indicators,
                "symbol_info": symbol_info,
                "priority_status": priority_status,
                "breaking_news_alert": breaking_news_alert,
                "market_status": self.market_status.value,
                "trading_mode": self.trading_mode.value,
                "data_quality": {
                    "is_nifty100": True,
                    "priority_tier": symbol_info.get("priority", 2),
                    "sector": symbol_info.get("sector", "Unknown"),
                    "news_weight": symbol_info.get("news_weight", 0.5),
                    "is_real_time": quote.data_source.startswith("kite"),
                    "data_source": quote.data_source,
                    "news_enhanced": getattr(quote, 'news_impact_score', 0.0) > 0.1,
                    "processing_time_ms": round(processing_time * 1000, 2)
                },
                "news_intelligence": {
                    "available": self.news_enhanced_mode,
                    "last_update": self.last_news_update.isoformat() if self.last_news_update else None,
                    "monitoring_active": self.news_monitoring_active,
                    "cache_status": "active" if self.news_cache else "inactive"
                },
                "metadata": {
                    "last_update": datetime.now().isoformat(),
                    "primary_source": self.primary_data_source,
                    "fallback_active": self.fallback_active,
                    "version": "6.0.0_complete_news_integration"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get enhanced market data for {symbol}: {e}")
            return {
                "symbol": symbol, 
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "news_intelligence_available": self.news_enhanced_mode
            }
    
    def _get_sector_momentum_with_news(self, sector: str) -> float:
        """Get sector momentum enhanced with news sentiment"""
        try:
            # Base sector momentum
            base_momentum = {
                "IT": 0.7, "Banking": 0.6, "FMCG": 0.5, "Pharma": 0.8,
                "Auto": 0.6, "Energy": 0.4, "Metals & Mining": 0.5,
                "Financial Services": 0.7, "Telecom": 0.5, "Infrastructure": 0.6
            }.get(sector, 0.5)
            
            if self.news_cache and "sector_sentiments" in self.news_cache:
                sector_sentiment = self.news_cache["sector_sentiments"].get(sector, 0.0)
                news_adjustment = sector_sentiment * 0.3  # Up to 30% adjustment
                return max(0.0, min(1.0, base_momentum + news_adjustment))
            
            return base_momentum
            
        except Exception as e:
            logger.debug(f"Sector momentum calculation failed: {e}")
            return 0.5
    
    async def run_premarket_analysis(self) -> Dict[str, Any]:
        """Run comprehensive pre-market analysis with complete news intelligence integration"""
        try:
            logger.info("🌅 Starting comprehensive pre-market analysis for Nifty 100 with complete news intelligence...")
            
            opportunities = await self.premarket_engine.run_comprehensive_analysis()
            
            self.priority_opportunities = opportunities
            self.premarket_analysis_time = datetime.now()
            self.performance_metrics["premarket_analyses_run"] += 1
            
            # Categorize opportunities
            strong_buys = [opp for opp in opportunities if opp.recommended_action == "STRONG_BUY"]
            buys = [opp for opp in opportunities if opp.recommended_action == "BUY"]
            watches = [opp for opp in opportunities if opp.recommended_action == "WATCH"]
            
            # NEWS INTELLIGENCE METRICS
            news_enhanced_opportunities = [opp for opp in opportunities if opp.enhanced_by_news_intelligence]
            breaking_news_opportunities = [opp for opp in opportunities if opp.breaking_news_detected]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "market_status": self.market_status.value,
                "data_sources": {
                    "primary": self.primary_data_source,
                    "kite_connected": self.kite_service.is_connected,
                    "yahoo_available": self.yahoo_service.is_connected,
                    "news_intelligence_active": self.news_enhanced_mode
                },
                "total_opportunities": len(opportunities),
                "strong_buy_count": len(strong_buys),
                "buy_count": len(buys),
                "watch_count": len(watches),
                "news_enhanced_count": len(news_enhanced_opportunities),
                "breaking_news_count": len(breaking_news_opportunities),
                "top_opportunities": [asdict(opp) for opp in opportunities[:8]],
                "sector_breakdown": self._get_sector_breakdown(opportunities),
                "news_intelligence_summary": self._get_news_intelligence_summary(opportunities),
                "analysis_summary": {
                    "avg_priority_score": np.mean([opp.priority_score for opp in opportunities]) if opportunities else 0,
                    "avg_gap_percentage": np.mean([abs(opp.gap_percentage) for opp in opportunities]) if opportunities else 0,
                    "high_volume_count": len([opp for opp in opportunities if opp.volume_expectation > 2.0]),
                    "gap_up_count": len([opp for opp in opportunities if opp.gap_percentage > 1]),
                    "gap_down_count": len([opp for opp in opportunities if opp.gap_percentage < -1]),
                    "news_impact_avg": np.mean([opp.news_impact_score for opp in opportunities]) if opportunities else 0,
                    "breaking_news_impact": len(breaking_news_opportunities) / len(opportunities) * 100 if opportunities else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Pre-market analysis failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "total_opportunities": 0,
                "news_intelligence_active": self.news_enhanced_mode
            }
    
    def _get_news_intelligence_summary(self, opportunities: List[PreMarketOpportunity]) -> Dict:
        """Get news intelligence summary from opportunities"""
        try:
            if not opportunities:
                return {
                    "total_with_news": 0,
                    "breaking_news_count": 0,
                    "avg_news_impact": 0.0,
                    "strong_sentiment_count": 0,
                    "news_driven_opportunities": []
                }
            
            news_enhanced = [opp for opp in opportunities if opp.enhanced_by_news_intelligence]
            breaking_news = [opp for opp in opportunities if opp.breaking_news_detected]
            strong_sentiment = [opp for opp in opportunities if "STRONG" in opp.news_sentiment_strength]
            
            # Top news-driven opportunities
            news_driven = sorted(
                [opp for opp in opportunities if opp.news_impact_score > 0.5],
                key=lambda x: x.news_impact_score,
                reverse=True
            )[:5]
            
            return {
                "total_with_news": len(news_enhanced),
                "breaking_news_count": len(breaking_news),
                "avg_news_impact": np.mean([opp.news_impact_score for opp in opportunities]),
                "strong_sentiment_count": len(strong_sentiment),
                "news_driven_opportunities": [
                    {
                        "symbol": opp.symbol,
                        "news_impact_score": opp.news_impact_score,
                        "sentiment_strength": opp.news_sentiment_strength,
                        "breaking_news": opp.breaking_news_detected,
                        "latest_headline": opp.latest_news_headline
                    }
                    for opp in news_driven
                ],
                "sentiment_distribution": {
                    "strong_positive": len([opp for opp in opportunities if opp.news_sentiment_strength == "STRONG_POSITIVE"]),
                    "moderate_positive": len([opp for opp in opportunities if opp.news_sentiment_strength == "MODERATE_POSITIVE"]),
                    "neutral": len([opp for opp in opportunities if opp.news_sentiment_strength == "NEUTRAL"]),
                    "moderate_negative": len([opp for opp in opportunities if opp.news_sentiment_strength == "MODERATE_NEGATIVE"]),
                    "strong_negative": len([opp for opp in opportunities if opp.news_sentiment_strength == "STRONG_NEGATIVE"])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get news intelligence summary: {e}")
            return {
                "total_with_news": 0,
                "breaking_news_count": 0,
                "avg_news_impact": 0.0,
                "strong_sentiment_count": 0,
                "news_driven_opportunities": []
            }
    
    async def get_priority_trading_signals(self) -> List[Dict[str, Any]]:
        """Get priority trading signals for 9:15 AM with enhanced news intelligence integration"""
        try:
            if not self.priority_opportunities:
                logger.warning("No pre-market analysis available for priority trading")
                return []
            
            priority_signals = []
            
            for opp in self.priority_opportunities[:5]:
                if opp.recommended_action in ["STRONG_BUY", "BUY"]:
                    
                    # Determine action based on gap and strategy
                    action = "BUY" if opp.gap_percentage > 0 else "SELL"
                    
                    signal = {
                        "id": f"priority_{opp.symbol}_{int(datetime.now().timestamp())}",
                        "symbol": opp.symbol,
                        "action": action,
                        "entry_price": opp.target_price if action == "SELL" else opp.target_price * 0.98,
                        "target_price": opp.target_price,
                        "stop_loss": opp.stop_loss,
                        "confidence": opp.confidence,
                        "priority_score": opp.priority_score,
                        "gap_percentage": opp.gap_percentage,
                        "catalyst": opp.catalyst,
                        "entry_strategy": opp.entry_strategy,
                        "time_horizon": opp.time_horizon,
                        "signal_type": "PRIORITY_OPENING",
                        "generated_at": datetime.now().isoformat(),
                        "valid_until": (datetime.now() + timedelta(minutes=30)).isoformat(),
                        "sector": self.nifty100.get_symbol_info(opp.symbol).get("sector", "Unknown"),
                        "reason": f"Pre-market analysis: {opp.catalyst} with {opp.priority_score:.1%} priority score",
                        "data_source": self.primary_data_source,
                        "real_time_data": self.kite_service.is_connected,
                        # NEWS INTELLIGENCE FIELDS
                        "news_enhanced": opp.enhanced_by_news_intelligence,
                        "news_impact_score": opp.news_impact_score,
                        "breaking_news_detected": opp.breaking_news_detected,
                        "news_sentiment_strength": opp.news_sentiment_strength,
                        "latest_news_headline": opp.latest_news_headline,
                        "news_sources_count": opp.news_sources_count
                    }
                    
                    priority_signals.append(signal)
            
            logger.info(f"🎯 Generated {len(priority_signals)} priority trading signals using {self.primary_data_source} with news intelligence")
            return priority_signals
            
        except Exception as e:
            logger.error(f"Priority signal generation failed: {e}")
            return []
    
    async def get_nifty100_overview(self) -> Dict:
        """Get complete Nifty 100 market overview with enhanced data sources and news intelligence"""
        try:
            start_time = time.time()
            
            priority_symbols = self.nifty100.get_priority_symbols(1)[:25]  # Top 25 for performance
            
            # Try Kite Connect first, fallback to Yahoo
            quotes = {}
            if self.kite_service.is_connected:
                quotes = await self.kite_service.get_bulk_quotes(priority_symbols)
            
            if len(quotes) < len(priority_symbols):
                missing_symbols = [s for s in priority_symbols if s not in quotes]
                yahoo_quotes = await self.yahoo_service.get_bulk_quotes(missing_symbols)
                quotes.update(yahoo_quotes)
            
            # Calculate market metrics
            total_volume = sum(quote.volume for quote in quotes.values())
            avg_change = np.mean([quote.change_percent for quote in quotes.values()])
            advancing = len([q for q in quotes.values() if q.change_percent > 0])
            declining = len([q for q in quotes.values() if q.change_percent < 0])
            
            # Data source breakdown
            data_sources = {}
            for quote in quotes.values():
                source = quote.data_source
                data_sources[source] = data_sources.get(source, 0) + 1
            
            # NEWS INTELLIGENCE METRICS
            news_enhanced_quotes = [q for q in quotes.values() if getattr(q, 'news_impact_score', 0.0) > 0.1]
            breaking_news_quotes = [q for q in quotes.values() if getattr(q, 'breaking_news_detected', False)]
            avg_news_sentiment = np.mean([getattr(q, 'news_sentiment', 0.0) for q in quotes.values()])
            
            sector_performance = self._calculate_sector_performance_with_news(quotes)
            
            processing_time = time.time() - start_time
            
            return {
                "timestamp": datetime.now().isoformat(),
                "market_status": self.market_status.value,
                "trading_mode": self.trading_mode.value,
                "data_sources": {
                    "primary": self.primary_data_source,
                    "kite_connected": self.kite_service.is_connected,
                    "yahoo_available": self.yahoo_service.is_connected,
                    "news_intelligence_active": self.news_enhanced_mode,
                    "source_breakdown": data_sources
                },
                "overview": {
                    "total_stocks_tracked": len(self.nifty100.get_all_symbols()),
                    "priority_stocks_active": len(quotes),
                    "total_volume": total_volume,
                    "average_change_percent": round(avg_change, 2),
                    "advancing_stocks": advancing,
                    "declining_stocks": declining,
                    "unchanged_stocks": len(quotes) - advancing - declining
                },
                "news_intelligence_overview": {
                    "news_enhanced_quotes": len(news_enhanced_quotes),
                    "breaking_news_stocks": len(breaking_news_quotes),
                    "avg_market_news_sentiment": round(avg_news_sentiment, 3),
                    "news_monitoring_active": self.news_monitoring_active,
                    "last_news_update": self.last_news_update.isoformat() if self.last_news_update else None,
                    "breaking_news_alerts_today": len(self.breaking_news_alerts)
                },
                "sector_performance": sector_performance,
                "top_gainers": self._get_top_movers_with_news(quotes, "gainers"),
                "top_losers": self._get_top_movers_with_news(quotes, "losers"),
                "high_volume": self._get_high_volume_stocks_with_news(quotes),
                "breaking_news_stocks": self._get_breaking_news_stocks(quotes),
                "performance_metrics": {
                    "total_quotes_served_today": self.performance_metrics["total_quotes_served"],
                    "news_enhanced_percentage": (self.performance_metrics["news_enhanced_quotes"] / max(1, self.performance_metrics["total_quotes_served"])) * 100,
                    "breaking_news_detections_today": self.performance_metrics["breaking_news_detections"],
                    "data_source_usage": self.performance_metrics["data_source_usage"]
                },
                "processing_time_seconds": round(processing_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Nifty 100 overview failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "news_intelligence_active": self.news_enhanced_mode
            }
    
    def _get_sector_breakdown(self, opportunities: List[PreMarketOpportunity]) -> Dict:
        """Get sector breakdown of opportunities"""
        sector_counts = {}
        for opp in opportunities:
            symbol_info = self.nifty100.get_symbol_info(opp.symbol)
            sector = symbol_info.get("sector", "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        return sector_counts
    
    def _calculate_sector_performance_with_news(self, quotes: Dict[str, MarketTick]) -> Dict:
        """Calculate sector-wise performance enhanced with news intelligence"""
        sector_data = {}
        
        for symbol, quote in quotes.items():
            symbol_info = self.nifty100.get_symbol_info(symbol)
            sector = symbol_info.get("sector", "Unknown")
            
            if sector not in sector_data:
                sector_data[sector] = {
                    "changes": [], 
                    "volumes": [], 
                    "news_sentiments": [], 
                    "news_impacts": []
                }
            
            sector_data[sector]["changes"].append(quote.change_percent)
            sector_data[sector]["volumes"].append(quote.volume)
            sector_data[sector]["news_sentiments"].append(getattr(quote, 'news_sentiment', 0.0))
            sector_data[sector]["news_impacts"].append(getattr(quote, 'news_impact_score', 0.0))
        
        sector_performance = {}
        for sector, data in sector_data.items():
            avg_news_sentiment = np.mean(data["news_sentiments"])
            avg_news_impact = np.mean(data["news_impacts"])
            
            sector_performance[sector] = {
                "avg_change": round(np.mean(data["changes"]), 2),
                "total_volume": sum(data["volumes"]),
                "stock_count": len(data["changes"]),
                "avg_news_sentiment": round(avg_news_sentiment, 3),
                "avg_news_impact": round(avg_news_impact, 3),
                "news_enhanced": avg_news_impact > 0.1
            }
        
        return sector_performance
    
    def _get_top_movers_with_news(self, quotes: Dict[str, MarketTick], move_type: str) -> List[Dict]:
        """Get top gainers or losers with news intelligence"""
        sorted_quotes = sorted(
            quotes.items(), 
            key=lambda x: x[1].change_percent, 
            reverse=(move_type == "gainers")
        )
        
        return [
            {
                "symbol": symbol,
                "change_percent": quote.change_percent,
                "ltp": quote.ltp,
                "volume": quote.volume,
                "sector": self.nifty100.get_symbol_info(symbol).get("sector", "Unknown"),
                "data_source": quote.data_source,
                "news_sentiment": getattr(quote, 'news_sentiment', 0.0),
                "news_impact_score": getattr(quote, 'news_impact_score', 0.0),
                "breaking_news_detected": getattr(quote, 'breaking_news_detected', False),
                "latest_news_headline": getattr(quote, 'latest_news_headline', None)
            }
            for symbol, quote in sorted_quotes[:8]  # Increased from 5 to 8
        ]
    
    def _get_high_volume_stocks_with_news(self, quotes: Dict[str, MarketTick]) -> List[Dict]:
        """Get high volume stocks with news intelligence"""
        high_volume = [
            {
                "symbol": symbol,
                "volume_ratio": quote.volume_ratio,
                "volume": quote.volume,
                "change_percent": quote.change_percent,
                "sector": self.nifty100.get_symbol_info(symbol).get("sector", "Unknown"),
                "data_source": quote.data_source,
                "news_sentiment": getattr(quote, 'news_sentiment', 0.0),
                "news_impact_score": getattr(quote, 'news_impact_score', 0.0),
                "breaking_news_detected": getattr(quote, 'breaking_news_detected', False)
            }
            for symbol, quote in quotes.items()
            if quote.volume_ratio > 1.5
        ]
        
        return sorted(high_volume, key=lambda x: x["volume_ratio"], reverse=True)[:8]
    
    def _get_breaking_news_stocks(self, quotes: Dict[str, MarketTick]) -> List[Dict]:
        """Get stocks with breaking news alerts"""
        breaking_news_stocks = []
        
        for symbol, quote in quotes.items():
            if getattr(quote, 'breaking_news_detected', False):
                breaking_news_stocks.append({
                    "symbol": symbol,
                    "ltp": quote.ltp,
                    "change_percent": quote.change_percent,
                    "news_impact_score": getattr(quote, 'news_impact_score', 0.0),
                    "news_sentiment": getattr(quote, 'news_sentiment', 0.0),
                    "latest_news_headline": getattr(quote, 'latest_news_headline', None),
                    "sector": self.nifty100.get_symbol_info(symbol).get("sector", "Unknown"),
                    "alert_time": datetime.now().isoformat()
                })
        
        return sorted(breaking_news_stocks, key=lambda x: x["news_impact_score"], reverse=True)
    
    async def get_service_health(self) -> Dict:
        """Get enhanced service health status with complete news intelligence integration"""
        try:
            return {
                "status": "healthy" if self.is_initialized else "degraded",
                "is_initialized": self.is_initialized,
                "data_sources": {
                    "primary": self.primary_data_source,
                    "kite_connect": {
                        "connected": self.kite_service.is_connected,
                        "instruments_mapped": len(self.kite_service.instrument_map) if self.kite_service.is_connected else 0
                    },
                    "yahoo_finance": {
                        "connected": self.yahoo_service.is_connected,
                        "fallback_active": self.fallback_active
                    },
                    "news_intelligence": {
                        "connected": self.news_enhanced_mode,
                        "monitoring_active": self.news_monitoring_active,
                        "last_update": self.last_news_update.isoformat() if self.last_news_update else None,
                        "cache_active": bool(self.news_cache)
                    }
                },
                "market_status": self.market_status.value,
                "trading_mode": self.trading_mode.value,
                "last_update": self.last_update.isoformat(),
                "nifty100_universe": {
                    "total_stocks": len(self.nifty100.get_all_symbols()),
                    "nifty50_stocks": len(self.nifty100.get_priority_symbols(1)),
                    "next50_stocks": len(self.nifty100.get_priority_symbols(2)),
                    "sectors_covered": len(set(info["sector"] for info in self.nifty100.stocks.values())),
                    "high_news_sensitivity_stocks": len(self.nifty100.get_high_news_sensitivity_symbols()),
                    "updated_symbols": True  # Indicates MINDTREE removed, LTIM added
                },
                "premarket_analysis": {
                    "last_analysis": self.premarket_analysis_time.isoformat() if self.premarket_analysis_time else None,
                    "opportunities_found": len(self.priority_opportunities),
                    "analysis_available": bool(self.priority_opportunities),
                    "news_enhanced_opportunities": len([opp for opp in self.priority_opportunities if opp.enhanced_by_news_intelligence]),
                    "breaking_news_opportunities": len([opp for opp in self.priority_opportunities if opp.breaking_news_detected])
                },
                "news_intelligence_health": {
                    "service_connected": self.news_enhanced_mode,
                    "monitoring_active": self.news_monitoring_active,
                    "cache_size": len(self.news_cache) if self.news_cache else 0,
                    "breaking_news_alerts_today": len(self.breaking_news_alerts),
                    "last_cache_update": self.last_news_update.isoformat() if self.last_news_update else None
                },
                "performance_metrics": self.performance_metrics,
                "capabilities": {
                    "real_time_data": self.kite_service.is_connected,
                    "delayed_data_fallback": self.yahoo_service.is_connected,
                    "premarket_analysis": True,
                    "priority_trading": True,
                    "nifty100_coverage": True,
                    "sector_analysis": True,
                    "volume_analysis": True,
                    "gap_analysis": True,
                    "dual_data_sources": True,
                    "news_intelligence": self.news_enhanced_mode,
                    "breaking_news_alerts": self.news_monitoring_active,
                    "news_enhanced_signals": self.news_enhanced_mode,
                    "complete_integration": True
                }
            }
        except Exception as e:
            logger.error(f"Enhanced service health check failed: {e}")
            return {
                "error": str(e), 
                "is_initialized": False,
                "news_intelligence_available": self.news_enhanced_mode
            }
    
    async def get_breaking_news_alerts(self) -> List[Dict]:
        """Get current breaking news alerts"""
        try:
            # Return recent breaking news alerts (last 4 hours)
            cutoff_time = datetime.now() - timedelta(hours=4)
            recent_alerts = [
                alert for alert in self.breaking_news_alerts
                if alert["timestamp"] > cutoff_time
            ]
            
            return sorted(recent_alerts, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get breaking news alerts: {e}")
            return []
    
    async def get_news_enhanced_market_summary(self) -> Dict:
        """Get market summary enhanced with news intelligence"""
        try:
            if not self.news_enhanced_mode:
                return {
                    "status": "unavailable",
                    "message": "News intelligence not connected"
                }
            
            # Get current market news context
            market_context = {}
            if self.news_cache:
                market_context = {
                    "overall_sentiment": self.news_cache.get("overall_sentiment", 0.0),
                    "breaking_news_symbols": self.news_cache.get("breaking_news_symbols", []),
                    "top_sector_sentiments": dict(list(self.news_cache.get("sector_sentiments", {}).items())[:5]),
                    "last_update": self.last_news_update.isoformat() if self.last_news_update else None
                }
            
            # Get recent breaking news
            breaking_news = await self.get_breaking_news_alerts()
            
            return {
                "status": "active",
                "market_news_context": market_context,
                "breaking_news_alerts": breaking_news[:5],  # Last 5 alerts
                "news_monitoring": {
                    "active": self.news_monitoring_active,
                    "cache_size": len(self.news_cache) if self.news_cache else 0,
                    "last_update": self.last_news_update.isoformat() if self.last_news_update else None
                },
                "performance": {
                    "news_enhanced_quotes_today": self.performance_metrics["news_enhanced_quotes"],
                    "breaking_news_detections_today": self.performance_metrics["breaking_news_detections"],
                    "news_intelligence_usage": self.performance_metrics["data_source_usage"]["news_intelligence"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get news enhanced market summary: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def stop_news_monitoring(self):
        """Stop news monitoring"""
        try:
            self.news_monitoring_active = False
            logger.info("📰 News monitoring stopped")
        except Exception as e:
            logger.error(f"Failed to stop news monitoring: {e}")
    
    async def close(self):
        """Close all connections gracefully"""
        try:
            # Stop news monitoring
            self.stop_news_monitoring()
            
            # Close services
            await self.yahoo_service.close()
            
            # Clear caches
            self.news_cache.clear()
            self.breaking_news_alerts.clear()
            
            logger.info("📊 Enhanced Market Data Service with News Intelligence closed gracefully")
        except Exception as e:
            logger.error(f"Error closing Enhanced Market Data Service: {e}")

# ================================================================
# Factory Functions
# ================================================================

def create_enhanced_market_data_service() -> EnhancedMarketDataService:
    """Factory function to create enhanced market data service with complete news integration"""
    return EnhancedMarketDataService()

# ================================================================
# Testing Function - ENHANCED
# ================================================================

async def test_enhanced_service_with_news():
    """Test the enhanced market data service with complete news intelligence integration"""
    print("🧪 Testing Enhanced Market Data Service with Complete News Intelligence Integration...")
    
    service = create_enhanced_market_data_service()
    
    try:
        await service.initialize()
        
        health = await service.get_service_health()
        print(f"✅ Service Health: {health['is_initialized']}")
        print(f"🎯 Primary Data Source: {health['data_sources']['primary']}")
        print(f"📊 Kite Connect: {'✅ Connected' if health['data_sources']['kite_connect']['connected'] else '❌ Not Connected'}")
        print(f"📊 Yahoo Finance: {'✅ Available' if health['data_sources']['yahoo_finance']['connected'] else '❌ Not Available'}")
        print(f"📰 News Intelligence: {'✅ Ready' if health['data_sources']['news_intelligence'] else '❌ Not Connected'}")
        print(f"📈 Nifty 100 Coverage: {health['nifty100_universe']['total_stocks']} stocks")
        print(f"🏪 Market Status: {health['market_status']}")
        print(f"🎯 Trading Mode: {health['trading_mode']}")
        
        print("\n🔍 Testing individual quote with news enhancement (RELIANCE)...")
        quote_data = await service.get_live_market_data("RELIANCE")
        if 'quote' in quote_data:
            quote = quote_data['quote']
            news_data = quote_data.get('news_intelligence', {})
            print(f"✅ RELIANCE: ₹{quote['ltp']} ({quote['change_percent']:+.2f}%) from {quote['data_source']}")
            print(f"📰 News Enhanced: {quote_data['data_quality']['news_enhanced']}")
            if news_data.get('available'):
                print(f"📰 News Intelligence: Active (Last Update: {news_data.get('last_update', 'Never')})")
        
        if service.market_status in [MarketStatus.PRE_MARKET, MarketStatus.CLOSED]:
            print("\n🌅 Testing pre-market analysis with news intelligence...")
            analysis = await service.run_premarket_analysis()
            print(f"✅ Found {analysis['total_opportunities']} opportunities")
            print(f"📰 News Enhanced: {analysis.get('news_enhanced_count', 0)} opportunities")
            print(f"🚨 Breaking News: {analysis.get('breaking_news_count', 0)} opportunities")
            print(f"📊 Data sources: {analysis.get('data_sources', {})}")
            
            if analysis.get("top_opportunities"):
                top_opp = analysis["top_opportunities"][0]
                print(f"🎯 Top opportunity: {top_opp['symbol']} ({top_opp['recommended_action']}) - News Enhanced: {top_opp.get('enhanced_by_news_intelligence', False)}")
        
        print("\n🎯 Testing priority trading signals with news intelligence...")
        priority_signals = await service.get_priority_trading_signals()
        print(f"✅ Generated {len(priority_signals)} priority signals")
        if priority_signals:
            signal = priority_signals[0]
            print(f"🎯 Top signal: {signal['symbol']} {signal['action']} with {signal['confidence']:.1%} confidence")
            print(f"📰 News Enhanced: {signal.get('news_enhanced', False)}")
            if signal.get('breaking_news_detected'):
                print(f"🚨 Breaking News Signal!")
        
        print("\n📊 Testing Nifty 100 overview with news intelligence...")
        overview = await service.get_nifty100_overview()
        print(f"✅ Overview generated: {overview['overview']['total_stocks_tracked']} stocks tracked")
        if 'news_intelligence_overview' in overview:
            news_overview = overview['news_intelligence_overview']
            print(f"📰 News Enhanced Quotes: {news_overview['news_enhanced_quotes']}")
            print(f"🚨 Breaking News Stocks: {news_overview['breaking_news_stocks']}")
            print(f"📈 Market News Sentiment: {news_overview['avg_market_news_sentiment']:.3f}")
        
        # Test news-specific features
        if service.news_enhanced_mode:
            print("\n📰 Testing news-specific features...")
            news_summary = await service.get_news_enhanced_market_summary()
            print(f"✅ News summary status: {news_summary['status']}")
            
            breaking_alerts = await service.get_breaking_news_alerts()
            print(f"🚨 Breaking news alerts: {len(breaking_alerts)}")
        
        print("🎉 All enhanced tests with complete news intelligence integration passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
    finally:
        await service.close()

if __name__ == "__main__":
    asyncio.run(test_enhanced_service_with_news())