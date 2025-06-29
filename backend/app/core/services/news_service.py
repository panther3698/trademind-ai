import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class NewsSignalIntegrationService:
    """
    FIXED: Proper integration between news intelligence and signal generation
    This service bridges the gap between news collection and trading signals
    Handles breaking news, full news analysis, and signal enhancement logic.
    """
    
    def __init__(self, news_intelligence, signal_generator, market_data_service, analytics_service):
        self.news_intelligence = news_intelligence
        self.signal_generator = signal_generator
        self.market_data = market_data_service
        self.analytics = analytics_service
        
        # Integration settings
        self.news_check_interval = 30  # Check news every 30 seconds
        self.breaking_news_threshold = 0.7  # Immediate signal trigger
        self.high_sentiment_threshold = 0.5  # High sentiment trigger
        self.news_signal_cooldown = 300  # 5 minutes between news signals for same symbol
        
        # State tracking
        self.last_news_check = None
        self.last_breaking_news_check = None
        self.news_triggered_signals = {}  # Symbol -> last trigger time
        self.current_news_data = {}
        
        # Performance tracking
        self.integration_stats = {
            "news_checks_completed": 0,
            "breaking_news_detected": 0,
            "news_triggered_signals": 0,
            "high_sentiment_signals": 0,
            "news_enhanced_ml_signals": 0
        }
        
        logger.info("✅ News-Signal Integration Service initialized")
    
    async def start_integrated_news_monitoring(self):
        """Start integrated news monitoring with immediate signal triggers"""
        logger.info("🚀 Starting integrated news-signal monitoring...")
        return asyncio.create_task(self._integrated_news_loop())
    
    async def _integrated_news_loop(self):
        """FIXED: Main integrated news monitoring loop"""
        while True:
            try:
                # Quick news check for breaking news (every 30 seconds)
                await self._quick_breaking_news_check()
                
                # Full news analysis (every 5 minutes)
                if self._should_run_full_news_analysis():
                    await self._full_news_analysis_with_signals()
                
                await asyncio.sleep(self.news_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Integrated news monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _quick_breaking_news_check(self):
        """Quick check for breaking news that requires immediate action"""
        try:
            # Get only high-impact recent news
            news_data = await self.news_intelligence.get_comprehensive_news_intelligence(
                lookback_hours=6,  # Last 6 hours for RSS feeds
                symbols=["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"]  # Top stocks only
            )
            
            if not news_data or "error" in news_data:
                return
            
            # Check for breaking news
            breaking_events = self._detect_breaking_news(news_data)
            
            for event in breaking_events:
                await self._handle_breaking_news_event(event)
                
            self.integration_stats["news_checks_completed"] += 1
            
        except Exception as e:
            logger.debug(f"Quick news check error: {e}")
    
    async def _full_news_analysis_with_signals(self):
        """Full news analysis with enhanced signal generation"""
        try:
            logger.info("📰 Running full news analysis with signal integration...")
            
            # Get comprehensive news intelligence
            news_data = await self.news_intelligence.get_comprehensive_news_intelligence(
                lookback_hours=24,  # Last 24 hours to capture RSS articles
                symbols=None,  # All symbols
                sectors=None   # All sectors
            )
            
            if not news_data or "error" in news_data:
                return
            
            # Update current news data
            self.current_news_data = news_data
            self.last_news_check = datetime.now()
            
            # Process news for signal enhancement
            await self._enhance_signal_generation_with_news(news_data)
            
            # Update analytics
            articles_count = news_data.get("total_articles_analyzed", 0)
            sentiment_avg = news_data.get("overall_sentiment", {}).get("adjusted_sentiment", 0.0)
            sources_count = len(news_data.get("news_sources_used", []))
            
            await self.analytics.track_news_processed(articles_count, sentiment_avg, sources_count)
            
            logger.info(f"📊 News analysis complete: {articles_count} articles, sentiment: {sentiment_avg:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Full news analysis failed: {e}")
    
    def _detect_breaking_news(self, news_data: Dict) -> List[Dict]:
        """Detect breaking news events"""
        breaking_events = []
        
        try:
            # Check market events
            market_events = news_data.get("market_events", [])
            for event in market_events:
                significance = event.get("significance_score", 0.0)
                if significance > self.breaking_news_threshold:
                    breaking_events.append(event)
            
            # Check high impact news
            high_impact_news = news_data.get("high_impact_news", [])
            for news in high_impact_news:
                if news.get("market_impact") == "HIGH":
                    breaking_events.append({
                        "event_type": "HIGH_IMPACT_NEWS",
                        "title": news.get("title", ""),
                        "symbols_mentioned": news.get("symbols_mentioned", []),
                        "sentiment_score": news.get("sentiment_score", 0.0),
                        "significance_score": 0.8,  # High impact
                        "source": news.get("source", ""),
                        "published_at": news.get("published_at", datetime.now())
                    })
                    
        except Exception as e:
            logger.debug(f"Breaking news detection error: {e}")
        
        return breaking_events
    
    async def _handle_breaking_news_event(self, event: Dict):
        """Handle breaking news event with immediate signal generation"""
        try:
            symbols = event.get("symbols_mentioned", [])
            sentiment = event.get("sentiment_score", 0.0)
            significance = event.get("significance_score", 0.0)
            
            self.integration_stats["breaking_news_detected"] += 1
            
            # Process each affected symbol
            for symbol in symbols:
                # Check cooldown
                if self._is_symbol_in_cooldown(symbol):
                    continue
                
                # Generate immediate news-triggered signal
                if abs(sentiment) > 0.4 and significance > self.breaking_news_threshold:
                    await self._generate_news_triggered_signal(symbol, event, sentiment, significance)
                    
        except Exception as e:
            logger.error(f"❌ Breaking news handling failed: {e}")
    
    async def _generate_news_triggered_signal(self, symbol: str, event: Dict, sentiment: float, significance: float):
        """Generate immediate trading signal from breaking news"""
        try:
            # Create news-driven signal
            news_signal = {
                "signal_id": f"NEWS_{symbol}_{int(datetime.now().timestamp())}",
                "symbol": symbol,
                "action": "BUY" if sentiment > 0 else "SELL",
                "entry_price": 0.0,  # Will be filled by market data
                "confidence": significance,
                "sentiment_score": sentiment,
                "signal_source": "BREAKING_NEWS",
                "notes": f"Breaking news signal: {event.get('title', '')[:100]}...",
                "timestamp": datetime.now(),
                "created_at": datetime.now().isoformat(),
                "is_news_signal": True,
                "is_breaking_news": True,
                "event_type": event.get("event_type", "UNKNOWN"),
                "news_significance": significance
            }
            
            # Mark symbol in cooldown
            self.news_triggered_signals[symbol] = datetime.now()
            
            # Track stats
            self.integration_stats["news_triggered_signals"] += 1
            await self.analytics.track_news_signal()
            
            logger.info(f"🚨 Breaking news signal generated: {symbol} - {sentiment:.2f} sentiment, {significance:.2f} significance")
            
            return news_signal
            
        except Exception as e:
            logger.error(f"❌ News signal generation failed: {e}")
            return None
    
    async def _enhance_signal_generation_with_news(self, news_data: Dict):
        """Enhance ML signal generation with news intelligence"""
        try:
            # Get current news sentiment by symbol
            symbol_sentiment = news_data.get("sentiment_analysis", {}).get("symbol_sentiment", {})
            sector_impact = news_data.get("sector_impact", {})
            
            # Check for high sentiment symbols that should trigger signals
            for symbol, sentiment in symbol_sentiment.items():
                if abs(sentiment) > self.high_sentiment_threshold:
                    if not self._is_symbol_in_cooldown(symbol):
                        # Trigger enhanced ML signal generation for this symbol
                        await self._trigger_enhanced_ml_signal(symbol, sentiment, news_data)
                        
            self.integration_stats["news_enhanced_ml_signals"] += 1
            
        except Exception as e:
            logger.error(f"❌ Signal enhancement failed: {e}")
    
    async def _trigger_enhanced_ml_signal(self, symbol: str, sentiment: float, news_data: Dict):
        """Trigger enhanced ML signal generation for specific symbol"""
        try:
            if not self.signal_generator:
                return
            
            # Generate ML signal with news enhancement
            signals = await self.signal_generator.generate_signals()
            
            # Find signals for our symbol
            for signal in signals:
                if hasattr(signal, 'ticker') and signal.ticker == symbol:
                    # Enhance signal with news data
                    enhanced_signal = self._enhance_signal_with_news(signal, sentiment, news_data)
                    
                    # Track enhancement
                    self.integration_stats["high_sentiment_signals"] += 1
                    
                    logger.info(f"📈 Enhanced ML signal with news: {symbol} - sentiment {sentiment:.2f}")
                    return enhanced_signal
                    
        except Exception as e:
            logger.error(f"❌ Enhanced ML signal generation failed: {e}")
    
    def _enhance_signal_with_news(self, signal, sentiment: float, news_data: Dict):
        """Enhance existing signal with news intelligence"""
        try:
            # Add news context to signal
            if hasattr(signal, 'notes'):
                signal.notes += f" | News sentiment: {sentiment:.2f}"
            
            # Adjust confidence based on news
            if hasattr(signal, 'ml_confidence'):
                news_boost = min(0.1, abs(sentiment) * 0.2)  # Max 10% boost
                if sentiment > 0 and getattr(signal.direction, 'value', None) == "BUY":
                    signal.ml_confidence += news_boost
                elif sentiment < 0 and getattr(signal.direction, 'value', None) == "SELL":
                    signal.ml_confidence += news_boost
                    
            return signal
            
        except Exception as e:
            logger.debug(f"Signal enhancement error: {e}")
            return signal
    
    def _is_symbol_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        if symbol not in self.news_triggered_signals:
            return False
            
        last_trigger = self.news_triggered_signals[symbol]
        return (datetime.now() - last_trigger).total_seconds() < self.news_signal_cooldown
    
    def _should_run_full_news_analysis(self) -> bool:
        """Check if full news analysis should run"""
        if not self.last_news_check:
            return True
            
        return (datetime.now() - self.last_news_check).total_seconds() > 300  # Every 5 minutes
    
    def get_integration_stats(self) -> Dict:
        """Get integration statistics"""
        return self.integration_stats.copy() 