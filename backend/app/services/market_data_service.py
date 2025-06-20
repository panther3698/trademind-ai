# backend/app/services/market_data_service.py
"""
Institutional Market Data Service for TradeMind AI
Integrates Zerodha Kite Connect, news feeds, and real-time market data
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import aiohttp
import feedparser
import re
from dataclasses import dataclass, asdict
from enum import Enum
import ta  # Technical Analysis library
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time

# For sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️ VADER Sentiment not available. Install with: pip install vaderSentiment")

logger = logging.getLogger(__name__)

class MarketStatus(Enum):
    PRE_OPEN = "PRE_OPEN"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    BREAK = "BREAK"

@dataclass
class MarketTick:
    """Real-time market tick data"""
    symbol: str
    timestamp: datetime
    ltp: float  # Last traded price
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

@dataclass
class OHLCV:
    """OHLCV data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class NewsItem:
    """Structured news data"""
    headline: str
    content: str
    source: str
    timestamp: datetime
    url: str
    sentiment_score: float
    symbols_mentioned: List[str]
    category: str  # earnings, policy, sector, etc.

@dataclass
class MarketSentiment:
    """Market sentiment aggregation"""
    overall_score: float  # -1 to +1
    news_count: int
    positive_news: int
    negative_news: int
    neutral_news: int
    trending_topics: List[str]
    fear_greed_index: float

class ZerodhaDataService:
    """
    Zerodha Kite Connect integration for live market data
    """
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.base_url = "https://api.kite.trade"
        self.session = None
        
        # Cache for instrument mapping
        self.instrument_cache = {}
        self.last_cache_update = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    async def initialize(self):
        """Initialize the Zerodha service"""
        self.session = aiohttp.ClientSession(
            headers={
                "X-Kite-Version": "3",
                "Authorization": f"token {self.api_key}:{self.access_token}"
            }
        )
        
        # Load instrument cache
        await self.update_instrument_cache()
        logger.info("✅ Zerodha Data Service initialized")
    
    async def _rate_limited_request(self, url: str, params: Dict = None) -> Dict:
        """Make rate-limited requests to Kite API"""
        # Ensure minimum interval between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Kite API error: {response.status} - {await response.text()}")
                    return {}
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {}
    
    async def update_instrument_cache(self):
        """Update instrument mapping cache"""
        try:
            url = f"{self.base_url}/instruments"
            data = await self._rate_limited_request(url)
            
            if data:
                # Convert to symbol -> instrument_token mapping
                for instrument in data.get('data', []):
                    symbol = instrument.get('trading_symbol', '')
                    if symbol:
                        self.instrument_cache[symbol] = instrument
                
                self.last_cache_update = datetime.now()
                logger.info(f"📊 Updated {len(self.instrument_cache)} instruments")
                
        except Exception as e:
            logger.error(f"Failed to update instrument cache: {e}")
    
    def get_instrument_token(self, symbol: str) -> Optional[str]:
        """Get instrument token for symbol"""
        return self.instrument_cache.get(symbol, {}).get('instrument_token')
    
    async def get_quote(self, symbol: str) -> Optional[MarketTick]:
        """Get real-time quote for symbol"""
        try:
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                logger.warning(f"Instrument token not found for {symbol}")
                return None
            
            url = f"{self.base_url}/quote"
            params = {"i": f"NSE:{symbol}"}
            
            data = await self._rate_limited_request(url, params)
            quote_data = data.get('data', {}).get(f"NSE:{symbol}", {})
            
            if quote_data:
                return MarketTick(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    ltp=float(quote_data.get('last_price', 0)),
                    volume=int(quote_data.get('volume', 0)),
                    bid_price=float(quote_data.get('depth', {}).get('buy', [{}])[0].get('price', 0)),
                    ask_price=float(quote_data.get('depth', {}).get('sell', [{}])[0].get('price', 0)),
                    bid_qty=int(quote_data.get('depth', {}).get('buy', [{}])[0].get('quantity', 0)),
                    ask_qty=int(quote_data.get('depth', {}).get('sell', [{}])[0].get('quantity', 0)),
                    change=float(quote_data.get('net_change', 0)),
                    change_percent=float(quote_data.get('net_change', 0) / quote_data.get('last_price', 1) * 100),
                    high=float(quote_data.get('ohlc', {}).get('high', 0)),
                    low=float(quote_data.get('ohlc', {}).get('low', 0)),
                    open_price=float(quote_data.get('ohlc', {}).get('open', 0)),
                    prev_close=float(quote_data.get('ohlc', {}).get('close', 0))
                )
            
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, 
                                 symbol: str, 
                                 interval: str = "minute",
                                 from_date: datetime = None,
                                 to_date: datetime = None) -> List[OHLCV]:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading symbol
            interval: minute, 5minute, 15minute, day
            from_date: Start date
            to_date: End date
        """
        try:
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                return []
            
            if not from_date:
                from_date = datetime.now() - timedelta(days=30)
            if not to_date:
                to_date = datetime.now()
            
            url = f"{self.base_url}/instruments/historical/{instrument_token}/{interval}"
            params = {
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d")
            }
            
            data = await self._rate_limited_request(url, params)
            candles = data.get('data', {}).get('candles', [])
            
            ohlcv_data = []
            for candle in candles:
                ohlcv_data.append(OHLCV(
                    timestamp=datetime.fromisoformat(candle[0].replace('Z', '+00:00')),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=int(candle[5])
                ))
            
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    async def get_market_status(self) -> MarketStatus:
        """Get current market status"""
        try:
            now = datetime.now().time()
            
            # NSE trading hours (simplified)
            pre_open = dt_time(9, 0)
            market_open = dt_time(9, 15)
            market_close = dt_time(15, 30)
            
            if pre_open <= now < market_open:
                return MarketStatus.PRE_OPEN
            elif market_open <= now < market_close:
                return MarketStatus.OPEN
            else:
                return MarketStatus.CLOSED
                
        except Exception as e:
            logger.error(f"Failed to get market status: {e}")
            return MarketStatus.CLOSED
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class NewsAndSentimentService:
    """
    News aggregation and sentiment analysis service
    """
    
    def __init__(self):
        self.news_sources = {
            "economic_times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "moneycontrol": "https://www.moneycontrol.com/rss/buzzingstocks.xml",
            "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
            "livemint": "https://www.livemint.com/rss/markets",
            "financial_express": "https://www.financialexpress.com/market/rss/"
        }
        
        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
            logger.warning("⚠️ Sentiment analysis disabled - install vaderSentiment")
        
        # Cache to avoid duplicate news
        self.processed_news = set()
        self.news_cache = []
        self.cache_expiry = 300  # 5 minutes
        self.last_cache_update = 0
        
        # Stock symbol patterns
        self.symbol_patterns = self._compile_symbol_patterns()
    
    def _compile_symbol_patterns(self) -> Dict:
        """Compile regex patterns for stock symbol detection"""
        # Top 100 NSE stocks
        top_stocks = [
            "RELIANCE", "TCS", "HDFC", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK",
            "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "ASIANPAINT", "LT", "AXISBANK",
            "MARUTI", "BAJFINANCE", "TITAN", "NESTLEIND", "HCLTECH", "WIPRO", "ULTRACEMCO",
            "ADANIPORTS", "ONGC", "TATAMOTORS", "SUNPHARMA", "NTPC", "POWERGRID", "M&M",
            "TECHM", "COALINDIA", "DRREDDY", "GRASIM", "BAJAJFINSV", "BRITANNIA", "CIPLA"
        ]
        
        patterns = {}
        for stock in top_stocks:
            # Create patterns like "RELIANCE", "Reliance Industries", etc.
            patterns[stock] = [
                re.compile(rf'\b{stock}\b', re.IGNORECASE),
                re.compile(rf'\b{stock.lower()}\b'),
                re.compile(rf'\b{stock.title()}\b')
            ]
        
        return patterns
    
    async def fetch_news_feed(self, source_name: str, url: str) -> List[Dict]:
        """Fetch and parse RSS news feed"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse RSS feed
                        feed = feedparser.parse(content)
                        
                        news_items = []
                        for entry in feed.entries[:20]:  # Limit to latest 20 items
                            # Create unique hash for duplicate detection
                            content_hash = hashlib.md5(
                                (entry.title + entry.link).encode()
                            ).hexdigest()
                            
                            if content_hash not in self.processed_news:
                                news_items.append({
                                    "headline": entry.title,
                                    "content": entry.get('description', ''),
                                    "source": source_name,
                                    "url": entry.link,
                                    "timestamp": datetime.now(),
                                    "hash": content_hash
                                })
                                self.processed_news.add(content_hash)
                        
                        return news_items
                        
        except Exception as e:
            logger.error(f"Failed to fetch news from {source_name}: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text
        
        Returns:
            float: Sentiment score between -1 (negative) and +1 (positive)
        """
        if not self.sentiment_analyzer:
            return 0.0
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            # Return compound score which is normalized between -1 and +1
            return scores['compound']
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0
    
    def extract_mentioned_symbols(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text"""
        mentioned_symbols = []
        
        for symbol, patterns in self.symbol_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    mentioned_symbols.append(symbol)
                    break  # Avoid duplicates
        
        return mentioned_symbols
    
    def categorize_news(self, headline: str, content: str) -> str:
        """Categorize news item"""
        text = (headline + " " + content).lower()
        
        categories = {
            "earnings": ["earnings", "quarterly", "results", "profit", "revenue"],
            "policy": ["rbi", "policy", "rate", "monetary", "fiscal", "budget"],
            "merger": ["merger", "acquisition", "takeover", "deal"],
            "sector": ["banking", "pharma", "it", "auto", "steel", "cement"],
            "global": ["fed", "china", "us", "europe", "global", "international"],
            "market": ["sensex", "nifty", "market", "trading", "index"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return "general"
    
    async def get_latest_news(self, force_refresh: bool = False) -> List[NewsItem]:
        """Get latest aggregated news with sentiment analysis"""
        current_time = time.time()
        
        # Use cache if recent and not forcing refresh
        if (not force_refresh and 
            self.news_cache and 
            current_time - self.last_cache_update < self.cache_expiry):
            return self.news_cache
        
        all_news = []
        
        # Fetch from all sources
        tasks = []
        for source_name, url in self.news_sources.items():
            tasks.append(self.fetch_news_feed(source_name, url))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for source_idx, result in enumerate(results):
            if isinstance(result, list):
                all_news.extend(result)
        
        # Process and enrich news items
        processed_news = []
        for news_item in all_news[-50:]:  # Latest 50 items
            try:
                sentiment_score = self.analyze_sentiment(
                    news_item["headline"] + " " + news_item["content"]
                )
                
                symbols_mentioned = self.extract_mentioned_symbols(
                    news_item["headline"] + " " + news_item["content"]
                )
                
                category = self.categorize_news(
                    news_item["headline"], 
                    news_item["content"]
                )
                
                processed_news.append(NewsItem(
                    headline=news_item["headline"],
                    content=news_item["content"],
                    source=news_item["source"],
                    timestamp=news_item["timestamp"],
                    url=news_item["url"],
                    sentiment_score=sentiment_score,
                    symbols_mentioned=symbols_mentioned,
                    category=category
                ))
                
            except Exception as e:
                logger.error(f"Failed to process news item: {e}")
        
        # Update cache
        self.news_cache = processed_news
        self.last_cache_update = current_time
        
        logger.info(f"📰 Processed {len(processed_news)} news items")
        return processed_news
    
    async def get_symbol_sentiment(self, symbol: str) -> Dict:
        """Get sentiment analysis for specific symbol"""
        news_items = await self.get_latest_news()
        
        relevant_news = [
            news for news in news_items 
            if symbol in news.symbols_mentioned
        ]
        
        if not relevant_news:
            return {
                "symbol": symbol,
                "sentiment_score": 0.0,
                "news_count": 0,
                "latest_headline": None
            }
        
        # Aggregate sentiment
        total_sentiment = sum(news.sentiment_score for news in relevant_news)
        avg_sentiment = total_sentiment / len(relevant_news)
        
        # Weight by recency (more recent news has higher weight)
        now = datetime.now()
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for news in relevant_news:
            hours_old = (now - news.timestamp).total_seconds() / 3600
            weight = max(0.1, 1.0 - (hours_old / 24))  # Decay over 24 hours
            weighted_sentiment += news.sentiment_score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_sentiment = weighted_sentiment / total_weight
        else:
            final_sentiment = avg_sentiment
        
        return {
            "symbol": symbol,
            "sentiment_score": final_sentiment,
            "news_count": len(relevant_news),
            "latest_headline": relevant_news[0].headline if relevant_news else None,
            "relevant_news": [asdict(news) for news in relevant_news[:5]]
        }
    
    async def get_market_sentiment(self) -> MarketSentiment:
        """Get overall market sentiment"""
        news_items = await self.get_latest_news()
        
        if not news_items:
            return MarketSentiment(
                overall_score=0.0,
                news_count=0,
                positive_news=0,
                negative_news=0,
                neutral_news=0,
                trending_topics=[],
                fear_greed_index=50.0
            )
        
        # Categorize news by sentiment
        positive_news = len([n for n in news_items if n.sentiment_score > 0.1])
        negative_news = len([n for n in news_items if n.sentiment_score < -0.1])
        neutral_news = len(news_items) - positive_news - negative_news
        
        # Calculate overall sentiment
        if news_items:
            overall_score = sum(n.sentiment_score for n in news_items) / len(news_items)
        else:
            overall_score = 0.0
        
        # Extract trending topics
        all_symbols = []
        for news in news_items:
            all_symbols.extend(news.symbols_mentioned)
        
        from collections import Counter
        trending_topics = [symbol for symbol, count in Counter(all_symbols).most_common(10)]
        
        # Simple fear/greed index (0-100)
        fear_greed_index = max(0, min(100, 50 + (overall_score * 50)))
        
        return MarketSentiment(
            overall_score=overall_score,
            news_count=len(news_items),
            positive_news=positive_news,
            negative_news=negative_news,
            neutral_news=neutral_news,
            trending_topics=trending_topics,
            fear_greed_index=fear_greed_index
        )

class TechnicalAnalysisService:
    """
    Technical analysis calculations using real market data
    """
    
    @staticmethod
    def calculate_indicators(ohlcv_data: List[OHLCV]) -> Dict:
        """
        Calculate comprehensive technical indicators
        
        Args:
            ohlcv_data: List of OHLCV data points
            
        Returns:
            Dict with all calculated indicators
        """
        if len(ohlcv_data) < 50:  # Need sufficient data
            logger.warning("Insufficient data for technical analysis")
            return {}
        
        # Convert to pandas DataFrame
        df = pd.DataFrame([
            {
                'timestamp': ohlcv.timestamp,
                'open': ohlcv.open,
                'high': ohlcv.high,
                'low': ohlcv.low,
                'close': ohlcv.close,
                'volume': ohlcv.volume
            }
            for ohlcv in ohlcv_data
        ])
        
        df = df.sort_values('timestamp')
        
        try:
            indicators = {}
            
            # RSI
            indicators['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[-1]
            indicators['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi().iloc[-1]
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            indicators['macd_line'] = macd.macd().iloc[-1]
            indicators['macd_signal'] = macd.macd_signal().iloc[-1]
            indicators['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
            indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
            indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
            indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
            
            # VWAP
            indicators['vwap'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend().iloc[-1]
            
            # ADX
            indicators['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx().iloc[-1]
            
            # ATR
            indicators['atr_14'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range().iloc[-1]
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            indicators['stoch_k'] = stoch.stoch().iloc[-1]
            indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]
            
            # CCI
            indicators['cci'] = ta.momentum.CCIIndicator(df['high'], df['low'], df['close']).cci().iloc[-1]
            
            # Williams %R
            indicators['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r().iloc[-1]
            
            # Moving Averages
            indicators['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator().iloc[-1]
            indicators['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator().iloc[-1]
            indicators['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator().iloc[-1]
            indicators['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator().iloc[-1]
            
            # Volume indicators
            indicators['volume_sma_20'] = df['volume'].rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma_20']
            
            # Momentum
            indicators['momentum_10'] = df['close'].iloc[-1] / df['close'].iloc[-10] - 1
            indicators['roc_10'] = ta.momentum.ROCIndicator(df['close'], window=10).roc().iloc[-1]
            
            # Replace any NaN values with 0
            for key, value in indicators.items():
                if pd.isna(value):
                    indicators[key] = 0.0
                    
            return indicators
            
        except Exception as e:
            logger.error(f"Technical analysis calculation failed: {e}")
            return {}

class MarketDataService:
    """
    Main market data service coordinating all data sources
    """
    
    def __init__(self, zerodha_api_key: str, zerodha_access_token: str):
        self.zerodha = ZerodhaDataService(zerodha_api_key, zerodha_access_token)
        self.news_service = NewsAndSentimentService()
        self.technical_service = TechnicalAnalysisService()
        
        # Market state
        self.market_status = MarketStatus.CLOSED
        self.last_update = datetime.now()
        
        # Watchlist (configurable)
        self.watchlist = [
            "RELIANCE", "TCS", "HDFC", "INFY", "HDFCBANK", "ICICIBANK", 
            "KOTAKBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", 
            "ASIANPAINT", "LT", "AXISBANK", "MARUTI", "BAJFINANCE"
        ]
    
    async def initialize(self):
        """Initialize all data services"""
        await self.zerodha.initialize()
        logger.info("🚀 Market Data Service initialized")
    
    async def get_live_market_data(self, symbol: str) -> Dict:
        """
        Get comprehensive live market data for a symbol
        
        Returns:
            Dict containing tick data, technical indicators, and sentiment
        """
        try:
            # Get real-time quote
            quote = await self.zerodha.get_quote(symbol)
            if not quote:
                return {}
            
            # Get historical data for technical analysis
            historical_data = await self.zerodha.get_historical_data(
                symbol, 
                interval="minute",
                from_date=datetime.now() - timedelta(days=10)
            )
            
            # Calculate technical indicators
            technical_indicators = {}
            if historical_data:
                technical_indicators = self.technical_service.calculate_indicators(historical_data)
            
            # Get sentiment analysis
            sentiment_data = await self.news_service.get_symbol_sentiment(symbol)
            
            # Combine all data
            market_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "quote": asdict(quote),
                "technical_indicators": technical_indicators,
                "sentiment": sentiment_data,
                "market_status": self.market_status.value
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to get live market data for {symbol}: {e}")
            return {}
    
    async def get_pre_market_analysis(self) -> Dict:
        """
        Get pre-market analysis for all watchlist stocks
        
        This runs between 8:30-9:15 AM to prepare for market open
        """
        try:
            pre_market_data = {}
            
            # Get market sentiment
            market_sentiment = await self.news_service.get_market_sentiment()
            
            # Analyze each symbol in watchlist
            for symbol in self.watchlist:
                try:
                    # Get latest quote
                    quote = await self.zerodha.get_quote(symbol)
                    if not quote:
                        continue
                    
                    # Calculate gap percentage
                    gap_percentage = ((quote.ltp - quote.prev_close) / quote.prev_close) * 100
                    
                    # Get sentiment
                    sentiment = await self.news_service.get_symbol_sentiment(symbol)
                    
                    # Get historical data for volatility calculation
                    historical = await self.zerodha.get_historical_data(
                        symbol, 
                        interval="day",
                        from_date=datetime.now() - timedelta(days=30)
                    )
                    
                    avg_volume = 0
                    if historical:
                        avg_volume = np.mean([h.volume for h in historical[-20:]])
                    
                    volume_spike = (quote.volume / avg_volume) if avg_volume > 0 else 0
                    
                    pre_market_data[symbol] = {
                        "symbol": symbol,
                        "current_price": quote.ltp,
                        "prev_close": quote.prev_close,
                        "gap_percentage": gap_percentage,
                        "volume_spike": volume_spike,
                        "sentiment_score": sentiment["sentiment_score"],
                        "news_count": sentiment["news_count"],
                        "latest_headline": sentiment.get("latest_headline"),
                        "pre_market_score": self._calculate_pre_market_score(
                            gap_percentage, volume_spike, sentiment["sentiment_score"]
                        )
                    }
                    
                except Exception as e:
                    logger.error(f"Pre-market analysis failed for {symbol}: {e}")
            
            # Sort by pre-market score
            sorted_symbols = sorted(
                pre_market_data.items(),
                key=lambda x: abs(x[1]["pre_market_score"]),
                reverse=True
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "market_sentiment": asdict(market_sentiment),
                "top_movers": dict(sorted_symbols[:10]),
                "analysis_summary": {
                    "total_symbols_analyzed": len(pre_market_data),
                    "positive_gaps": len([s for s in pre_market_data.values() if s["gap_percentage"] > 0]),
                    "negative_gaps": len([s for s in pre_market_data.values() if s["gap_percentage"] < 0]),
                    "high_volume_symbols": len([s for s in pre_market_data.values() if s["volume_spike"] > 1.5])
                }
            }
            
        except Exception as e:
            logger.error(f"Pre-market analysis failed: {e}")
            return {}
    
    def _calculate_pre_market_score(self, gap_pct: float, volume_spike: float, sentiment: float) -> float:
        """
        Calculate pre-market opportunity score
        
        Args:
            gap_pct: Gap percentage from previous close
            volume_spike: Volume relative to average
            sentiment: News sentiment score
            
        Returns:
            float: Pre-market score (higher = more interesting)
        """
        # Weight factors
        gap_weight = 0.4
        volume_weight = 0.3
        sentiment_weight = 0.3
        
        # Normalize inputs
        gap_score = min(abs(gap_pct) / 5.0, 1.0)  # Normalize to 5% gap
        volume_score = min(volume_spike / 3.0, 1.0)  # Normalize to 3x volume
        sentiment_score = abs(sentiment)  # Absolute sentiment strength
        
        total_score = (gap_score * gap_weight + 
                      volume_score * volume_weight + 
                      sentiment_score * sentiment_weight)
        
        # Apply gap direction (positive for up gaps, negative for down gaps)
        if gap_pct < 0:
            total_score *= -1
            
        return total_score
    
    async def update_market_status(self):
        """Update current market status"""
        self.market_status = await self.zerodha.get_market_status()
        self.last_update = datetime.now()
    
    async def close(self):
        """Close all connections"""
        await self.zerodha.close()
        logger.info("Market Data Service closed")

# Example usage and testing
async def test_market_data_service():
    """Test the market data service"""
    
    # These would come from environment variables
    API_KEY = "your_zerodha_api_key"
    ACCESS_TOKEN = "your_zerodha_access_token"
    
    service = MarketDataService(API_KEY, ACCESS_TOKEN)
    
    try:
        await service.initialize()
        
        # Test live data
        print("Testing live market data...")
        live_data = await service.get_live_market_data("RELIANCE")
        if live_data:
            print(f"✅ RELIANCE data: {live_data['quote']['ltp']}")
        
        # Test pre-market analysis
        print("\nTesting pre-market analysis...")
        pre_market = await service.get_pre_market_analysis()
        if pre_market:
            print(f"✅ Pre-market analysis: {len(pre_market.get('top_movers', {}))} symbols")
        
        # Test news sentiment
        print("\nTesting news sentiment...")
        news_items = await service.news_service.get_latest_news()
        print(f"✅ Fetched {len(news_items)} news items")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        await service.close()

if __name__ == "__main__":
    # Run test
    asyncio.run(test_market_data_service())