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