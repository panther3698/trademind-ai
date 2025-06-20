# backend/app/ml/advanced_sentiment.py
"""
TradeMind AI - Advanced Sentiment Analysis and Filtering System
Multiple sentiment models and alternative data sources for maximum accuracy
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import json
import re
from pathlib import Path
import pickle
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Basic sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Advanced NLP libraries
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, 
        pipeline, BertTokenizer, BertForSequenceClassification
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers torch")

# Text processing
try:
    import spacy
    import nltk
    from textblob import TextBlob
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    print("⚠️ Advanced NLP libraries not available. Install with: pip install spacy nltk textblob")

# Financial data APIs
try:
    import yfinance as yf
    import pandas_datareader as pdr
    FINANCIAL_APIS_AVAILABLE = True
except ImportError:
    FINANCIAL_APIS_AVAILABLE = False
    print("⚠️ Financial APIs not available. Install with: pip install yfinance pandas-datareader")

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Comprehensive sentiment analysis result"""
    text: str
    timestamp: datetime
    
    # Basic sentiment scores
    vader_score: float
    textblob_score: float
    
    # Advanced model scores
    finbert_score: float
    finbert_confidence: float
    roberta_score: float
    
    # Ensemble scores
    weighted_score: float
    confidence: float
    
    # Classification
    sentiment_class: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    strength: str  # "WEAK", "MODERATE", "STRONG"
    
    # Metadata
    source: str
    symbols_mentioned: List[str]
    keywords: List[str]

class AdvancedSentimentAnalyzer:
    """
    Multi-model sentiment analyzer combining:
    - VADER (rule-based)
    - FinBERT (financial domain-specific)
    - RoBERTa (general purpose transformer)
    - Custom ensemble weighting
    """
    
    def __init__(self):
        # Initialize basic analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize transformer models
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.roberta_pipeline = None
        
        # Model weights for ensemble
        self.model_weights = {
            'vader': 0.2,
            'textblob': 0.15,
            'finbert': 0.4,
            'roberta': 0.25
        }
        
        # Initialize models
        asyncio.create_task(self._initialize_models())
        
        # Stock symbol patterns
        self.stock_patterns = self._compile_stock_patterns()
        
    async def _initialize_models(self):
        """Initialize transformer models asynchronously"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using basic sentiment only")
            return
        
        try:
            # Initialize FinBERT
            logger.info("Loading FinBERT model...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            
            # Initialize RoBERTa
            logger.info("Loading RoBERTa sentiment model...")
            self.roberta_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            logger.info("✅ Advanced sentiment models loaded")
            
        except Exception as e:
            logger.error(f"Failed to load transformer models: {e}")
            self.finbert_model = None
            self.roberta_pipeline = None
    
    def _compile_stock_patterns(self) -> Dict:
        """Compile patterns for detecting stock mentions"""
        # Extended Nifty 100 + major global stocks
        stocks = [
            # Indian stocks
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "INFY", "ITC",
            "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", "AXISBANK",
            "MARUTI", "BAJFINANCE", "TITAN", "NESTLEIND", "ULTRACEMCO", "WIPRO",
            
            # Global stocks (for sentiment spillover)
            "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX"
        ]
        
        patterns = {}
        for stock in stocks:
            patterns[stock] = [
                re.compile(rf'\b{stock}\b', re.IGNORECASE),
                re.compile(rf'\b{stock.lower()}\b'),
                re.compile(rf'\${stock}\b', re.IGNORECASE)  # $SYMBOL format
            ]
        
        return patterns
    
    def analyze_text(self, text: str, source: str = "unknown") -> SentimentResult:
        """
        Comprehensive sentiment analysis using multiple models
        
        Args:
            text: Text to analyze
            source: Source of the text (news, twitter, etc.)
            
        Returns:
            SentimentResult with comprehensive sentiment scores
        """
        try:
            # Basic sentiment analysis
            vader_result = self.vader.polarity_scores(text)
            vader_score = vader_result['compound']
            
            textblob_score = 0.0
            if NLP_AVAILABLE:
                try:
                    blob = TextBlob(text)
                    textblob_score = blob.sentiment.polarity
                except:
                    pass
            
            # Advanced sentiment analysis
            finbert_score, finbert_confidence = self._analyze_finbert(text)
            roberta_score = self._analyze_roberta(text)
            
            # Calculate ensemble score
            weighted_score = (
                vader_score * self.model_weights['vader'] +
                textblob_score * self.model_weights['textblob'] +
                finbert_score * self.model_weights['finbert'] +
                roberta_score * self.model_weights['roberta']
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence([
                abs(vader_score), abs(textblob_score), 
                finbert_confidence, abs(roberta_score)
            ])
            
            # Classify sentiment
            sentiment_class = self._classify_sentiment(weighted_score)
            strength = self._classify_strength(abs(weighted_score), confidence)
            
            # Extract mentioned symbols and keywords
            symbols_mentioned = self._extract_symbols(text)
            keywords = self._extract_keywords(text)
            
            return SentimentResult(
                text=text[:200] + "..." if len(text) > 200 else text,
                timestamp=datetime.now(),
                vader_score=vader_score,
                textblob_score=textblob_score,
                finbert_score=finbert_score,
                finbert_confidence=finbert_confidence,
                roberta_score=roberta_score,
                weighted_score=weighted_score,
                confidence=confidence,
                sentiment_class=sentiment_class,
                strength=strength,
                source=source,
                symbols_mentioned=symbols_mentioned,
                keywords=keywords
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._create_default_result(text, source)
    
    def _analyze_finbert(self, text: str) -> Tuple[float, float]:
        """Analyze sentiment using FinBERT"""
        if not self.finbert_model or not self.finbert_tokenizer:
            return 0.0, 0.0
        
        try:
            # Tokenize and predict
            inputs = self.finbert_tokenizer(text, return_tensors="pt", 
                                          truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [negative, neutral, positive]
            negative, neutral, positive = predictions[0].tolist()
            
            # Convert to -1 to +1 scale
            sentiment_score = positive - negative
            confidence = max(negative, neutral, positive)
            
            return sentiment_score, confidence
            
        except Exception as e:
            logger.debug(f"FinBERT analysis failed: {e}")
            return 0.0, 0.0
    
    def _analyze_roberta(self, text: str) -> float:
        """Analyze sentiment using RoBERTa"""
        if not self.roberta_pipeline:
            return 0.0
        
        try:
            result = self.roberta_pipeline(text[:512])  # Truncate for model limits
            
            label = result[0]['label']
            score = result[0]['score']
            
            # Convert to -1 to +1 scale
            if label == 'LABEL_2':  # Positive
                return score
            elif label == 'LABEL_0':  # Negative
                return -score
            else:  # Neutral
                return 0.0
                
        except Exception as e:
            logger.debug(f"RoBERTa analysis failed: {e}")
            return 0.0
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate overall confidence from individual model scores"""
        try:
            # Confidence based on score agreement and magnitude
            avg_magnitude = np.mean(scores)
            score_std = np.std(scores)
            
            # Higher confidence when scores are high and agree
            confidence = avg_magnitude * (1 - min(score_std, 1.0))
            return min(confidence, 1.0)
            
        except:
            return 0.5
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment into positive/negative/neutral"""
        if score > 0.1:
            return "POSITIVE"
        elif score < -0.1:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _classify_strength(self, magnitude: float, confidence: float) -> str:
        """Classify sentiment strength"""
        strength_score = magnitude * confidence
        
        if strength_score > 0.6:
            return "STRONG"
        elif strength_score > 0.3:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract mentioned stock symbols"""
        mentioned = []
        
        for symbol, patterns in self.stock_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    mentioned.append(symbol)
                    break
        
        return list(set(mentioned))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract financial keywords"""
        financial_keywords = [
            "earnings", "revenue", "profit", "loss", "growth", "dividend",
            "acquisition", "merger", "ipo", "results", "guidance", "outlook",
            "bullish", "bearish", "buy", "sell", "upgrade", "downgrade"
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _create_default_result(self, text: str, source: str) -> SentimentResult:
        """Create default result when analysis fails"""
        return SentimentResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            timestamp=datetime.now(),
            vader_score=0.0,
            textblob_score=0.0,
            finbert_score=0.0,
            finbert_confidence=0.0,
            roberta_score=0.0,
            weighted_score=0.0,
            confidence=0.0,
            sentiment_class="NEUTRAL",
            strength="WEAK",
            source=source,
            symbols_mentioned=[],
            keywords=[]
        )

class AlternativeDataCollector:
    """
    Collect alternative data sources for enhanced market intelligence
    - Google Trends
    - Social media sentiment
    - Economic indicators
    - Sector rotation data
    """
    
    def __init__(self):
        self.session = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession()
    
    async def get_google_trends(self, keywords: List[str], timeframe: str = "today 1-m") -> Dict[str, Any]:
        """
        Get Google Trends data for financial keywords
        (Requires pytrends: pip install pytrends)
        """
        try:
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360)
            
            # Build payload
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='IN')
            
            # Get interest over time
            interest_over_time = pytrends.interest_over_time()
            
            if interest_over_time.empty:
                return {"error": "No trends data available"}
            
            # Calculate trend scores
            trend_scores = {}
            for keyword in keywords:
                if keyword in interest_over_time.columns:
                    recent_values = interest_over_time[keyword].tail(7)  # Last 7 days
                    trend_scores[keyword] = {
                        "current_interest": recent_values.iloc[-1],
                        "trend_direction": "up" if recent_values.iloc[-1] > recent_values.iloc[0] else "down",
                        "volatility": recent_values.std(),
                        "avg_interest": recent_values.mean()
                    }
            
            return {
                "trends": trend_scores,
                "timestamp": datetime.now().isoformat(),
                "timeframe": timeframe
            }
            
        except Exception as e:
            logger.error(f"Google Trends collection failed: {e}")
            return {"error": str(e)}
    
    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """
        Get Fear & Greed Index (mock implementation - replace with real API)
        """
        try:
            # Mock calculation based on multiple factors
            # In production, use CNN Fear & Greed Index API or similar
            
            vix_equivalent = np.random.uniform(10, 35)  # Mock VIX
            put_call_ratio = np.random.uniform(0.7, 1.3)  # Mock put/call ratio
            market_momentum = np.random.uniform(-5, 5)  # Mock momentum
            
            # Calculate composite fear/greed score
            fear_greed_score = 50  # Neutral baseline
            
            # Adjust based on factors
            fear_greed_score -= (vix_equivalent - 15) * 2  # Higher VIX = more fear
            fear_greed_score -= (put_call_ratio - 1.0) * 30  # Higher put/call = more fear
            fear_greed_score += market_momentum * 5  # Positive momentum = less fear
            
            # Clamp to 0-100
            fear_greed_score = max(0, min(100, fear_greed_score))
            
            # Classify
            if fear_greed_score > 75:
                classification = "Extreme Greed"
            elif fear_greed_score > 55:
                classification = "Greed"
            elif fear_greed_score > 45:
                classification = "Neutral"
            elif fear_greed_score > 25:
                classification = "Fear"
            else:
                classification = "Extreme Fear"
            
            return {
                "fear_greed_score": fear_greed_score,
                "classification": classification,
                "factors": {
                    "vix_equivalent": vix_equivalent,
                    "put_call_ratio": put_call_ratio,
                    "market_momentum": market_momentum
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fear & Greed index calculation failed: {e}")
            return {"error": str(e)}
    
    async def get_sector_rotation_data(self) -> Dict[str, Any]:
        """
        Analyze sector rotation patterns
        """
        try:
            # Mock sector performance data
            # In production, fetch real sector ETF data
            
            sectors = {
                "Technology": np.random.uniform(-3, 5),
                "Banking": np.random.uniform(-2, 4),
                "Pharmaceutical": np.random.uniform(-1, 3),
                "FMCG": np.random.uniform(-1, 2),
                "Auto": np.random.uniform(-4, 6),
                "Energy": np.random.uniform(-5, 8),
                "Real Estate": np.random.uniform(-3, 4),
                "Metals": np.random.uniform(-6, 10)
            }
            
            # Identify rotation patterns
            best_performers = sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:3]
            worst_performers = sorted(sectors.items(), key=lambda x: x[1])[:3]
            
            # Calculate sector dispersion
            sector_returns = list(sectors.values())
            sector_dispersion = np.std(sector_returns)
            
            return {
                "sector_performance": sectors,
                "best_performers": best_performers,
                "worst_performers": worst_performers,
                "sector_dispersion": sector_dispersion,
                "rotation_strength": "High" if sector_dispersion > 3 else "Low",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sector rotation analysis failed: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

class SocialSentimentMonitor:
    """
    Monitor social media sentiment from Twitter, Reddit, etc.
    (Requires Twitter API, Reddit API access)
    """
    
    def __init__(self):
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        
        # Social media keywords for financial sentiment
        self.financial_keywords = [
            "#nifty", "#sensex", "#stocks", "#trading", "#investing",
            "#bulls", "#bears", "#market", "#earnings", "#ipo"
        ]
    
    async def get_twitter_sentiment(self, keywords: List[str] = None) -> Dict[str, Any]:
        """
        Get Twitter sentiment for financial keywords
        (Requires Twitter API v2 access)
        """
        try:
            # Mock implementation - replace with real Twitter API
            if keywords is None:
                keywords = self.financial_keywords
            
            # Mock sentiment data
            sentiment_data = {}
            
            for keyword in keywords:
                # Mock tweet analysis
                positive_tweets = np.random.randint(10, 100)
                negative_tweets = np.random.randint(5, 50)
                neutral_tweets = np.random.randint(20, 150)
                
                total_tweets = positive_tweets + negative_tweets + neutral_tweets
                
                sentiment_score = (positive_tweets - negative_tweets) / total_tweets
                
                sentiment_data[keyword] = {
                    "sentiment_score": sentiment_score,
                    "total_tweets": total_tweets,
                    "positive_tweets": positive_tweets,
                    "negative_tweets": negative_tweets,
                    "neutral_tweets": neutral_tweets,
                    "engagement_rate": np.random.uniform(0.05, 0.15)
                }
            
            # Calculate overall social sentiment
            overall_sentiment = np.mean([data["sentiment_score"] for data in sentiment_data.values()])
            
            return {
                "keyword_sentiment": sentiment_data,
                "overall_sentiment": overall_sentiment,
                "sentiment_strength": "Strong" if abs(overall_sentiment) > 0.3 else "Weak",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Twitter sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    async def get_reddit_sentiment(self, subreddits: List[str] = None) -> Dict[str, Any]:
        """
        Get Reddit sentiment from financial subreddits
        """
        try:
            if subreddits is None:
                subreddits = ["IndiaInvestments", "SecurityAnalysis", "stocks", "investing"]
            
            # Mock Reddit sentiment analysis
            subreddit_sentiment = {}
            
            for subreddit in subreddits:
                # Mock post analysis
                posts_analyzed = np.random.randint(20, 100)
                avg_sentiment = np.random.uniform(-0.5, 0.5)
                sentiment_std = np.random.uniform(0.1, 0.4)
                
                subreddit_sentiment[subreddit] = {
                    "avg_sentiment": avg_sentiment,
                    "sentiment_volatility": sentiment_std,
                    "posts_analyzed": posts_analyzed,
                    "trending_topics": ["earnings", "market_outlook", "stock_picks"]
                }
            
            # Calculate overall Reddit sentiment
            overall_reddit_sentiment = np.mean([data["avg_sentiment"] for data in subreddit_sentiment.values()])
            
            return {
                "subreddit_sentiment": subreddit_sentiment,
                "overall_sentiment": overall_reddit_sentiment,
                "confidence": 1.0 - np.mean([data["sentiment_volatility"] for data in subreddit_sentiment.values()]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Reddit sentiment analysis failed: {e}")
            return {"error": str(e)}

class EconomicIndicatorCollector:
    """
    Collect economic indicators that impact market sentiment
    """
    
    def __init__(self):
        self.indicators = [
            'GDP_GROWTH', 'INFLATION_CPI', 'REPO_RATE', 'FII_FLOW', 
            'DII_FLOW', 'USD_INR', 'CRUDE_OIL', 'GOLD_PRICE'
        ]
    
    async def get_economic_indicators(self) -> Dict[str, Any]:
        """
        Get latest economic indicators
        (Mock implementation - replace with real data sources)
        """
        try:
            # Mock economic data
            indicators_data = {
                'GDP_GROWTH': {
                    'value': np.random.uniform(5.5, 7.5),
                    'previous': 6.8,
                    'change': 'positive',
                    'impact': 'bullish'
                },
                'INFLATION_CPI': {
                    'value': np.random.uniform(4.0, 6.5),
                    'previous': 5.2,
                    'change': 'negative',
                    'impact': 'bearish'
                },
                'REPO_RATE': {
                    'value': 6.5,
                    'previous': 6.25,
                    'change': 'positive',
                    'impact': 'bearish'
                },
                'FII_FLOW': {
                    'value': np.random.uniform(-500, 1000),  # Crores
                    'previous': 250,
                    'change': 'positive',
                    'impact': 'bullish'
                },
                'USD_INR': {
                    'value': np.random.uniform(82, 84),
                    'previous': 83.2,
                    'change': 'negative',
                    'impact': 'bullish'
                }
            }
            
            # Calculate composite economic sentiment
            bullish_indicators = sum(1 for data in indicators_data.values() if data['impact'] == 'bullish')
            bearish_indicators = sum(1 for data in indicators_data.values() if data['impact'] == 'bearish')
            
            economic_sentiment = (bullish_indicators - bearish_indicators) / len(indicators_data)
            
            return {
                'indicators': indicators_data,
                'economic_sentiment': economic_sentiment,
                'outlook': 'positive' if economic_sentiment > 0.2 else 'negative' if economic_sentiment < -0.2 else 'neutral',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Economic indicators collection failed: {e}")
            return {"error": str(e)}

class ComprehensiveSentimentEngine:
    """
    Main engine combining all sentiment and alternative data sources
    """
    
    def __init__(self):
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.alt_data_collector = AlternativeDataCollector()
        self.social_monitor = SocialSentimentMonitor()
        self.economic_collector = EconomicIndicatorCollector()
        
    async def initialize(self):
        """Initialize all components"""
        await self.alt_data_collector.initialize()
        logger.info("✅ Comprehensive Sentiment Engine initialized")
    
    async def get_comprehensive_sentiment(self, 
                                        news_texts: List[str] = None,
                                        symbols: List[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis combining all sources
        
        Args:
            news_texts: List of news texts to analyze
            symbols: List of symbols to analyze
            
        Returns:
            Comprehensive sentiment analysis
        """
        try:
            logger.info("🔍 Running comprehensive sentiment analysis...")
            
            results = {}
            
            # 1. News sentiment analysis
            if news_texts:
                news_sentiment = []
                for text in news_texts:
                    sentiment_result = self.sentiment_analyzer.analyze_text(text, "news")
                    news_sentiment.append(sentiment_result)
                
                # Aggregate news sentiment
                avg_news_sentiment = np.mean([s.weighted_score for s in news_sentiment])
                news_confidence = np.mean([s.confidence for s in news_sentiment])
                
                results['news_sentiment'] = {
                    'avg_sentiment': avg_news_sentiment,
                    'confidence': news_confidence,
                    'analyzed_articles': len(news_sentiment),
                    'positive_articles': sum(1 for s in news_sentiment if s.sentiment_class == "POSITIVE"),
                    'negative_articles': sum(1 for s in news_sentiment if s.sentiment_class == "NEGATIVE")
                }
            
            # 2. Social media sentiment
            social_sentiment = await self.social_monitor.get_twitter_sentiment()
            results['social_sentiment'] = social_sentiment
            
            # 3. Alternative data
            fear_greed = await self.alt_data_collector.get_fear_greed_index()
            results['fear_greed_index'] = fear_greed
            
            sector_rotation = await self.alt_data_collector.get_sector_rotation_data()
            results['sector_rotation'] = sector_rotation
            
            # 4. Economic indicators
            economic_data = await self.economic_collector.get_economic_indicators()
            results['economic_indicators'] = economic_data
            
            # 5. Calculate composite sentiment
            composite_sentiment = self._calculate_composite_sentiment(results)
            results['composite_sentiment'] = composite_sentiment
            
            logger.info(f"✅ Comprehensive analysis complete - Composite sentiment: {composite_sentiment['score']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_composite_sentiment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite sentiment from all sources"""
        try:
            sentiment_components = []
            weights = []
            
            # News sentiment (weight: 0.3)
            if 'news_sentiment' in results and not results['news_sentiment'].get('error'):
                sentiment_components.append(results['news_sentiment']['avg_sentiment'])
                weights.append(0.3)
            
            # Social sentiment (weight: 0.2)
            if 'social_sentiment' in results and not results['social_sentiment'].get('error'):
                sentiment_components.append(results['social_sentiment']['overall_sentiment'])
                weights.append(0.2)
            
            # Fear & Greed (weight: 0.2)
            if 'fear_greed_index' in results and not results['fear_greed_index'].get('error'):
                fear_greed_score = results['fear_greed_index']['fear_greed_score']
                # Convert 0-100 to -1 to +1
                normalized_fg = (fear_greed_score - 50) / 50
                sentiment_components.append(normalized_fg)
                weights.append(0.2)
            
            # Economic indicators (weight: 0.2)
            if 'economic_indicators' in results and not results['economic_indicators'].get('error'):
                sentiment_components.append(results['economic_indicators']['economic_sentiment'])
                weights.append(0.2)
            
            # Sector rotation (weight: 0.1)
            if 'sector_rotation' in results and not results['sector_rotation'].get('error'):
                sector_sentiment = 0.1 if results['sector_rotation']['rotation_strength'] == "High" else 0.0
                sentiment_components.append(sector_sentiment)
                weights.append(0.1)
            
            if not sentiment_components:
                return {"score": 0.0, "confidence": 0.0, "classification": "NEUTRAL"}
            
            # Calculate weighted average
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            composite_score = np.average(sentiment_components, weights=weights)
            
            # Calculate confidence based on agreement between sources
            if len(sentiment_components) > 1:
                sentiment_std = np.std(sentiment_components)
                confidence = max(0, 1.0 - sentiment_std)
            else:
                confidence = 0.5
            
            # Classify composite sentiment
            if composite_score > 0.2:
                classification = "BULLISH"
            elif composite_score < -0.2:
                classification = "BEARISH"
            else:
                classification = "NEUTRAL"
            
            return {
                "score": composite_score,
                "confidence": confidence,
                "classification": classification,
                "components_count": len(sentiment_components),
                "component_scores": sentiment_components,
                "weights_used": weights.tolist()
            }
            
        except Exception as e:
            logger.error(f"Composite sentiment calculation failed: {e}")
            return {"score": 0.0, "confidence": 0.0, "classification": "NEUTRAL", "error": str(e)}
    
    async def close(self):
        """Close all connections"""
        await self.alt_data_collector.close()


# Factory for easy instantiation
class SentimentFactory:
    """Factory for creating sentiment analysis components"""
    
    @staticmethod
    def create_analyzer(analyzer_type: str):
        """Create sentiment analyzer by type"""
        analyzers = {
            'basic': lambda: SentimentIntensityAnalyzer(),
            'advanced': AdvancedSentimentAnalyzer,
            'comprehensive': ComprehensiveSentimentEngine
        }
        
        if analyzer_type not in analyzers:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")
        
        return analyzers[analyzer_type]()
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available sentiment models"""
        models = ['vader', 'textblob']
        
        if TRANSFORMERS_AVAILABLE:
            models.extend(['finbert', 'roberta'])
        
        return models


if __name__ == "__main__":
    # Example usage
    async def test_sentiment():
        engine = ComprehensiveSentimentEngine()
        await engine.initialize()
        
        # Test news sentiment
        news_texts = [
            "Reliance Industries reports strong quarterly earnings with 15% growth",
            "Market volatility increases amid global uncertainty",
            "TCS announces major deal with US client"
        ]
        
        result = await engine.get_comprehensive_sentiment(news_texts=news_texts)
        print("Sentiment Analysis Results:")
        print(json.dumps(result, indent=2, default=str))
        
        await engine.close()
    
    # Run test
    print("TradeMind AI - Advanced Sentiment Analysis")
    print("Available models:")
    for model in SentimentFactory.get_available_models():
        print(f"  ✅ {model}")
