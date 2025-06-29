# ================================================================
# backend/app/ml/simplified_sentiment.py
# Simplified Sentiment Analysis - FinBERT Only
# ================================================================

import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for transformers availability
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers torch")

@dataclass
class SimplifiedSentimentResult:
    """Simplified sentiment analysis result - FinBERT only"""
    text: str
    timestamp: datetime
    
    # FinBERT scores
    finbert_score: float  # -1 to +1 scale
    finbert_confidence: float  # 0 to 1 scale
    
    # Classification
    sentiment_class: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    strength: str  # "WEAK", "MODERATE", "STRONG"
    
    # Metadata
    source: str
    symbols_mentioned: List[str]
    keywords: List[str]

class SimplifiedSentimentAnalyzer:
    """
    Simplified sentiment analyzer using only FinBERT
    Financial domain-specific analysis for trading signals
    """
    
    def __init__(self):
        # Initialize FinBERT model
        self.finbert_tokenizer = None
        self.finbert_model = None
        
        # Stock symbol patterns for Indian market
        self.stock_patterns = self._compile_stock_patterns()
        
        # Initialize model asynchronously
        asyncio.create_task(self._initialize_finbert())
        
        logger.info("✅ Simplified sentiment analyzer initialized (FinBERT only)")
    
    async def _initialize_finbert(self):
        """Initialize FinBERT model asynchronously"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using fallback sentiment")
            return
        
        try:
            logger.info("Loading FinBERT model for simplified sentiment analysis...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            
            logger.info("✅ FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.finbert_model = None
            self.finbert_tokenizer = None
    
    def _compile_stock_patterns(self) -> Dict:
        """Compile patterns for detecting Indian stock mentions"""
        # Nifty 100 stocks + major global stocks
        stocks = [
            # Indian stocks (Nifty 100)
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "INFY", "ITC",
            "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", "AXISBANK",
            "MARUTI", "BAJFINANCE", "TITAN", "NESTLEIND", "ULTRACEMCO", "WIPRO",
            "SUNPHARMA", "DRREDDY", "CIPLA", "TATAMOTORS", "M&M", "POWERGRID",
            "NTPC", "ONGC", "COALINDIA", "TECHM", "ADANIENT", "ADANIPORTS",
            
            # Global stocks
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
    
    def analyze_text(self, text: str, source: str = "unknown") -> SimplifiedSentimentResult:
        """
        Simplified sentiment analysis using only FinBERT
        
        Args:
            text: Text to analyze
            source: Source of the text (news, twitter, etc.)
            
        Returns:
            SimplifiedSentimentResult with FinBERT sentiment scores
        """
        try:
            # FinBERT analysis
            finbert_score, finbert_confidence = self._analyze_finbert(text)
            
            # Classify sentiment
            sentiment_class = self._classify_sentiment(finbert_score)
            strength = self._classify_strength(abs(finbert_score), finbert_confidence)
            
            # Extract mentioned symbols and keywords
            symbols_mentioned = self._extract_symbols(text)
            keywords = self._extract_keywords(text)
            
            return SimplifiedSentimentResult(
                text=text[:200] + "..." if len(text) > 200 else text,
                timestamp=datetime.now(),
                finbert_score=finbert_score,
                finbert_confidence=finbert_confidence,
                sentiment_class=sentiment_class,
                strength=strength,
                source=source,
                symbols_mentioned=symbols_mentioned,
                keywords=keywords
            )
            
        except Exception as e:
            logger.error(f"Simplified sentiment analysis failed: {e}")
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
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment based on FinBERT score"""
        if score > 0.1:
            return "POSITIVE"
        elif score < -0.1:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _classify_strength(self, magnitude: float, confidence: float) -> str:
        """Classify sentiment strength"""
        if magnitude > 0.5 and confidence > 0.7:
            return "STRONG"
        elif magnitude > 0.2 and confidence > 0.5:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract mentioned stock symbols from text"""
        symbols = []
        
        for symbol, patterns in self.stock_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    symbols.append(symbol)
                    break
        
        return list(set(symbols))  # Remove duplicates
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract financial keywords from text"""
        financial_keywords = [
            "earnings", "revenue", "profit", "loss", "growth", "decline",
            "bullish", "bearish", "rally", "crash", "volatility", "trend",
            "dividend", "buyback", "merger", "acquisition", "ipo", "listing",
            "quarterly", "annual", "guidance", "forecast", "upgrade", "downgrade"
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _create_default_result(self, text: str, source: str) -> SimplifiedSentimentResult:
        """Create default result when analysis fails"""
        return SimplifiedSentimentResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            timestamp=datetime.now(),
            finbert_score=0.0,
            finbert_confidence=0.0,
            sentiment_class="NEUTRAL",
            strength="WEAK",
            source=source,
            symbols_mentioned=[],
            keywords=[]
        )
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about the simplified analyzer"""
        return {
            "analyzer_type": "SIMPLIFIED_FINBERT_ONLY",
            "models_used": ["FinBERT"],
            "complexity": "LOW",
            "performance_impact": "MINIMAL",
            "finbert_loaded": self.finbert_model is not None,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "stock_patterns_count": len(self.stock_patterns)
        }

# Factory function for compatibility
def create_simplified_sentiment_analyzer():
    """Create simplified sentiment analyzer"""
    return SimplifiedSentimentAnalyzer() 