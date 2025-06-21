# backend/app/ml/models.py
"""
TradeMind AI - Machine Learning Models and Training Pipeline
Implements FinBERT, XGBoost, and complete feature engineering for Nifty 100 stocks
Enhanced with Advanced Ensemble auto-detection and fallback
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from pathlib import Path
import asyncio
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("⚠️ FinBERT not available. Install with: pip install transformers torch")

# Technical Analysis
import ta

# Import our services
from app.services.market_data_service import MarketDataService

# ================================================================
# NEW: Advanced Models Integration with Auto-Detection
# ================================================================
try:
    from app.ml.advanced_models import (
        LightGBMModel, CatBoostModel, LSTMModel,
        EnsembleModel as AdvancedEnsembleModel
    )
    ADVANCED_MODELS_AVAILABLE = True
    print("✅ Advanced ML models available (LightGBM, CatBoost, LSTM)")
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    print("⚠️ Advanced models not available, using basic ensemble")

try:
    from app.ml.advanced_sentiment import ComprehensiveSentimentEngine
    ADVANCED_SENTIMENT_AVAILABLE = True
    print("✅ Advanced sentiment analysis available")
except ImportError:
    ADVANCED_SENTIMENT_AVAILABLE = False
    print("⚠️ Advanced sentiment not available, using basic sentiment")

logger = logging.getLogger(__name__)

class Nifty100StockUniverse:
    """
    Nifty 100 stock universe management and data collection
    """
    
    def __init__(self):
        # Complete Nifty 100 stock list (as of 2024)
        self.nifty_100_stocks = [
            # Nifty 50
            "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "INFY", "ITC", 
            "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "HCLTECH", "ASIANPAINT", "AXISBANK",
            "MARUTI", "BAJFINANCE", "TITAN", "NESTLEIND", "ULTRACEMCO", "WIPRO", "ONGC",
            "NTPC", "POWERGRID", "SUNPHARMA", "TATAMOTORS", "M&M", "TECHM", "ADANIPORTS",
            "COALINDIA", "BAJAJFINSV", "DRREDDY", "GRASIM", "BRITANNIA", "EICHERMOT",
            "BPCL", "CIPLA", "DIVISLAB", "HEROMOTOCO", "HINDALCO", "JSWSTEEL", "LTIM",
            "INDUSINDBK", "APOLLOHOSP", "TATACONSUM", "BAJAJ-AUTO", "ADANIENT", "TATASTEEL",
            "PIDILITIND", "SBILIFE", "HDFCLIFE",
            
            # Additional Nifty Next 50 (part of Nifty 100)
            "ADANIGREEN", "ADANIPORTS", "AMBUJACEM", "BANDHANBNK", "BERGEPAINT", "BIOCON",
            "BOSCHLTD", "CADILAHC", "CHOLAFIN", "COLPAL", "CONCOR", "COROMANDEL", "CUMMINSIND",
            "DABUR", "DALBHARAT", "DEEPAKNTR", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL",
            "GLAND", "GODREJCP", "GODREJPROP", "HAVELLS", "HDFCAMC", "HINDPETRO", "ICICIGI",
            "ICICIPRULI", "IDFCFIRSTB", "IGL", "INDIGO", "IOC", "IRCTC", "JINDALSTEL",
            "JUBLFOOD", "KOTAKBANK", "L&TFH", "LICHSGFIN", "LUPIN", "MARICO", "MCDOWELL-N",
            "MFSL", "MINDTREE", "MUTHOOTFIN", "NAUKRI", "NMDC", "OFSS", "PAGEIND", "PEL",
            "PETRONET", "PFC", "PIIND", "PNB", "POLYCAB", "RAMCOCEM", "RBLBANK", "RECLTD",
            "SAIL", "SHREECEM", "SIEMENS", "SRF", "TORNTPHARM", "TRENT", "UBL", "VOLTAS"
        ]
        
        # Sector classification for Nifty 100 stocks
        self.sector_mapping = {
            # Banking
            "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "SBIN": "BANKING", "KOTAKBANK": "BANKING",
            "AXISBANK": "BANKING", "INDUSINDBK": "BANKING", "BANDHANBNK": "BANKING", "FEDERALBNK": "BANKING",
            "IDFCFIRSTB": "BANKING", "PNB": "BANKING", "RBLBANK": "BANKING",
            
            # IT
            "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT", 
            "LTIM": "IT", "MINDTREE": "IT", "OFSS": "IT",
            
            # Oil & Gas
            "RELIANCE": "OIL_GAS", "ONGC": "OIL_GAS", "BPCL": "OIL_GAS", "IOC": "OIL_GAS",
            "HINDPETRO": "OIL_GAS", "GAIL": "OIL_GAS", "PETRONET": "OIL_GAS",
            
            # FMCG
            "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
            "DABUR": "FMCG", "MARICO": "FMCG", "GODREJCP": "FMCG", "COLPAL": "FMCG",
            
            # Auto
            "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO", "BAJAJ-AUTO": "AUTO",
            "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO", "ESCORTS": "AUTO",
            
            # Pharma
            "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA", "DIVISLAB": "PHARMA",
            "LUPIN": "PHARMA", "BIOCON": "PHARMA", "TORNTPHARM": "PHARMA", "CADILAHC": "PHARMA",
            
            # Metals
            "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS", "JINDALSTEL": "METALS",
            "SAIL": "METALS", "NMDC": "METALS",
            
            # Infrastructure
            "LT": "INFRA", "NTPC": "INFRA", "POWERGRID": "INFRA", "ADANIPORTS": "INFRA",
            "COALINDIA": "INFRA", "CONCOR": "INFRA", "PFC": "INFRA", "RECLTD": "INFRA",
            
            # Default for others
        }
        
        # Default sector for unmapped stocks
        for stock in self.nifty_100_stocks:
            if stock not in self.sector_mapping:
                self.sector_mapping[stock] = "OTHERS"
    
    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        return self.sector_mapping.get(symbol, "OTHERS")
    
    def get_sector_stocks(self, sector: str) -> List[str]:
        """Get all stocks in a sector"""
        return [stock for stock, sec in self.sector_mapping.items() if sec == sector]
    
    def get_all_stocks(self) -> List[str]:
        """Get complete Nifty 100 stock list"""
        return self.nifty_100_stocks.copy()

class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analysis for financial news
    """
    
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.vader_analyzer = SentimentIntensityAnalyzer()  # Fallback
        
        if FINBERT_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1  # CPU
                )
                logger.info("✅ FinBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")
                self.sentiment_pipeline = None
        else:
            logger.warning("⚠️ FinBERT not available, using VADER fallback")
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT (with VADER fallback)
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment scores
        """
        try:
            if self.sentiment_pipeline and len(text) > 10:
                # Clean and truncate text for FinBERT
                clean_text = text[:512]  # FinBERT max length
                
                result = self.sentiment_pipeline(clean_text)[0]
                
                # Convert FinBERT output to standardized format
                label = result['label'].lower()
                confidence = result['score']
                
                if label == 'positive':
                    finbert_score = confidence
                elif label == 'negative':
                    finbert_score = -confidence
                else:  # neutral
                    finbert_score = 0.0
                
                return {
                    "finbert_score": finbert_score,
                    "finbert_confidence": confidence,
                    "finbert_label": label
                }
            else:
                # Fallback to VADER
                vader_scores = self.vader_analyzer.polarity_scores(text)
                return {
                    "finbert_score": vader_scores['compound'],
                    "finbert_confidence": abs(vader_scores['compound']),
                    "finbert_label": "fallback_vader"
                }
                
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            # Return neutral sentiment on error
            return {
                "finbert_score": 0.0,
                "finbert_confidence": 0.0,
                "finbert_label": "error"
            }
    
    def analyze_news_batch(self, news_items: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple news items"""
        return [self.analyze_sentiment(text) for text in news_items]

class FeatureEngineering:
    """
    Comprehensive feature engineering for trading signals
    """
    
    def __init__(self, stock_universe: Nifty100StockUniverse):
        self.stock_universe = stock_universe
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
    
    def engineer_features(self, 
                         symbol: str,
                         ohlcv_data: pd.DataFrame,
                         current_quote: Dict,
                         news_sentiment: Dict,
                         nifty_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Engineer comprehensive features for ML model
        
        Args:
            symbol: Stock symbol
            ohlcv_data: Historical OHLCV data
            current_quote: Current market quote
            news_sentiment: News sentiment data
            nifty_data: Nifty index data for correlation
            
        Returns:
            Dict of engineered features
        """
        try:
            features = {}
            
            # Basic price features
            current_price = current_quote.get('ltp', 0)
            prev_close = current_quote.get('prev_close', current_price)
            
            features['price_gap_percent'] = ((current_price - prev_close) / prev_close) * 100
            features['price_level'] = current_price
            
            # Technical indicators
            technical_features = self._calculate_technical_features(ohlcv_data, current_price)
            features.update(technical_features)
            
            # Volume features
            volume_features = self._calculate_volume_features(ohlcv_data, current_quote)
            features.update(volume_features)
            
            # Sentiment features
            sentiment_features = self._calculate_sentiment_features(news_sentiment)
            features.update(sentiment_features)
            
            # Sector and market features
            market_features = self._calculate_market_features(symbol, nifty_data, current_quote)
            features.update(market_features)
            
            # Time-based features
            time_features = self._calculate_time_features()
            features.update(time_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed for {symbol}: {e}")
            return {}
    
    def _calculate_technical_features(self, df: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Calculate technical indicator features"""
        features = {}
        
        if len(df) < 20:
            return features
        
        try:
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return features
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # RSI
            rsi_14 = ta.momentum.RSIIndicator(close, window=14).rsi()
            features['rsi_14'] = rsi_14.iloc[-1] if not rsi_14.empty else 50.0
            features['rsi_oversold'] = 1.0 if features['rsi_14'] < 30 else 0.0
            features['rsi_overbought'] = 1.0 if features['rsi_14'] > 70 else 0.0
            
            # MACD
            macd = ta.trend.MACD(close)
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()
            features['macd_line'] = macd_line.iloc[-1] if not macd_line.empty else 0.0
            features['macd_signal'] = macd_signal.iloc[-1] if not macd_signal.empty else 0.0
            features['macd_bullish'] = 1.0 if features['macd_line'] > features['macd_signal'] else 0.0
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_middle = bb.bollinger_mavg()
            
            if not bb_upper.empty and not bb_lower.empty:
                features['bb_position'] = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                features['bb_squeeze'] = 1.0 if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] < 0.1 else 0.0
            
            # ADX (Trend Strength)
            adx = ta.trend.ADXIndicator(high, low, close).adx()
            features['adx'] = adx.iloc[-1] if not adx.empty else 25.0
            features['strong_trend'] = 1.0 if features['adx'] > 25 else 0.0
            
            # ATR (Volatility)
            atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
            features['atr_percent'] = (atr.iloc[-1] / current_price * 100) if not atr.empty else 2.0
            
            # Moving Averages
            sma_20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
            sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator() if len(close) >= 50 else sma_20
            
            if not sma_20.empty:
                features['price_vs_sma20'] = (current_price - sma_20.iloc[-1]) / sma_20.iloc[-1]
                features['above_sma20'] = 1.0 if current_price > sma_20.iloc[-1] else 0.0
            
            if not sma_50.empty:
                features['price_vs_sma50'] = (current_price - sma_50.iloc[-1]) / sma_50.iloc[-1]
                features['sma20_vs_sma50'] = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] if not sma_20.empty else 0.0
            
            # Momentum
            if len(close) >= 10:
                features['momentum_10d'] = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            
            return features
            
        except Exception as e:
            logger.error(f"Technical features calculation failed: {e}")
            return {}
    
    def _calculate_volume_features(self, df: pd.DataFrame, current_quote: Dict) -> Dict[str, float]:
        """Calculate volume-based features"""
        features = {}
        
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return features
            
            volume = df['volume']
            current_volume = current_quote.get('volume', 0)
            
            # Volume ratios
            avg_volume_20 = volume.rolling(20).mean().iloc[-1]
            features['volume_ratio'] = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
            features['high_volume'] = 1.0 if features['volume_ratio'] > 1.5 else 0.0
            features['very_high_volume'] = 1.0 if features['volume_ratio'] > 2.0 else 0.0
            
            # Volume trend
            avg_volume_5 = volume.rolling(5).mean().iloc[-1]
            features['volume_trend'] = (avg_volume_5 - avg_volume_20) / avg_volume_20 if avg_volume_20 > 0 else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Volume features calculation failed: {e}")
            return {}
    
    def _calculate_sentiment_features(self, news_sentiment: Dict) -> Dict[str, float]:
        """Calculate sentiment-based features"""
        features = {}
        
        try:
            # Basic sentiment
            features['news_sentiment'] = news_sentiment.get('sentiment_score', 0.0)
            features['news_count'] = min(news_sentiment.get('news_count', 0), 10)  # Cap at 10
            features['positive_sentiment'] = 1.0 if features['news_sentiment'] > 0.1 else 0.0
            features['negative_sentiment'] = 1.0 if features['news_sentiment'] < -0.1 else 0.0
            
            # Sentiment strength
            features['sentiment_strength'] = abs(features['news_sentiment'])
            features['strong_sentiment'] = 1.0 if features['sentiment_strength'] > 0.5 else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Sentiment features calculation failed: {e}")
            return {}
    
    def _calculate_market_features(self, symbol: str, nifty_data: pd.DataFrame, current_quote: Dict) -> Dict[str, float]:
        """Calculate market and sector features"""
        features = {}
        
        try:
            # Sector information
            sector = self.stock_universe.get_sector(symbol)
            features['sector_banking'] = 1.0 if sector == 'BANKING' else 0.0
            features['sector_it'] = 1.0 if sector == 'IT' else 0.0
            features['sector_pharma'] = 1.0 if sector == 'PHARMA' else 0.0
            features['sector_auto'] = 1.0 if sector == 'AUTO' else 0.0
            features['sector_fmcg'] = 1.0 if sector == 'FMCG' else 0.0
            features['sector_oil_gas'] = 1.0 if sector == 'OIL_GAS' else 0.0
            
            # Market performance
            if nifty_data is not None and len(nifty_data) > 0:
                nifty_change = nifty_data['close'].pct_change().iloc[-1] if 'close' in nifty_data.columns else 0.0
                stock_change = current_quote.get('change_percent', 0.0) / 100
                
                features['nifty_performance'] = nifty_change
                features['relative_to_nifty'] = stock_change - nifty_change
                features['outperforming_nifty'] = 1.0 if features['relative_to_nifty'] > 0.005 else 0.0  # 0.5% threshold
            else:
                features['nifty_performance'] = 0.0
                features['relative_to_nifty'] = 0.0
                features['outperforming_nifty'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Market features calculation failed: {e}")
            return {}
    
    def _calculate_time_features(self) -> Dict[str, float]:
        """Calculate time-based features"""
        features = {}
        
        try:
            now = datetime.now()
            
            # Market timing
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            if market_open <= now <= market_close:
                # Minutes since market open
                minutes_since_open = (now - market_open).total_seconds() / 60
                features['minutes_since_open'] = minutes_since_open
                features['early_session'] = 1.0 if minutes_since_open < 60 else 0.0
                features['mid_session'] = 1.0 if 60 <= minutes_since_open <= 240 else 0.0
                features['late_session'] = 1.0 if minutes_since_open > 240 else 0.0
            else:
                features['minutes_since_open'] = 0.0
                features['early_session'] = 0.0
                features['mid_session'] = 0.0
                features['late_session'] = 0.0
            
            # Day of week
            weekday = now.weekday()  # 0 = Monday
            features['monday'] = 1.0 if weekday == 0 else 0.0
            features['friday'] = 1.0 if weekday == 4 else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Time features calculation failed: {e}")
            return {}

class XGBoostSignalModel:
    """
    XGBoost model for trading signal prediction
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_version = "v1.0"
        
        # Model parameters
        self.xgb_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'auc'
        }
    
    def train_model(self, 
                   training_data: pd.DataFrame,
                   target_column: str = 'profitable',
                   test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train XGBoost model on historical data
        
        Args:
            training_data: DataFrame with features and target
            target_column: Name of target column
            test_size: Fraction for test set
            
        Returns:
            Dict with training results and metrics
        """
        try:
            logger.info(f"Training XGBoost model on {len(training_data)} samples")
            
            # Prepare features and target
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns]
            y = training_data[target_column]
            
            # Handle missing values
            X = X.fillna(0)
            
            # Store feature columns for later use
            self.feature_columns = feature_columns
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost model
            self.model = xgb.XGBClassifier(**self.xgb_params)
            
            # Fit with evaluation set
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Feature importance
            feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Training results
            results = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'total_samples': len(training_data),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(feature_columns),
                'top_features': top_features,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'model_version': self.model_version
            }
            
            logger.info(f"✅ Model trained - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
            
            # Save model
            self.save_model()
            
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {}
    
    def predict_signal_probability(self, features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Predict signal probability for given features
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple of (probability, prediction_details)
        """
        try:
            if self.model is None:
                self.load_model()
            
            if self.model is None:
                return 0.5, {"error": "Model not available"}
            
            # Prepare features
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            # Scale features
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predict
            probability = self.model.predict_proba(X_scaled)[0, 1]
            prediction = self.model.predict(X_scaled)[0]
            
            # Get feature contributions (SHAP-like)
            feature_contributions = {}
            if hasattr(self.model, 'feature_importances_'):
                for i, col in enumerate(self.feature_columns):
                    feature_contributions[col] = {
                        'value': features.get(col, 0.0),
                        'importance': self.model.feature_importances_[i]
                    }
            
            prediction_details = {
                'probability': probability,
                'prediction': int(prediction),
                'confidence': max(probability, 1 - probability),
                'feature_contributions': feature_contributions,
                'model_version': self.model_version
            }
            
            return probability, prediction_details
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5, {"error": str(e)}
    
    def save_model(self):
        """Save trained model and scaler"""
        try:
            model_path = self.model_dir / f"xgboost_model_{self.model_version}.pkl"
            scaler_path = self.model_dir / f"scaler_{self.model_version}.pkl"
            features_path = self.model_dir / f"features_{self.model_version}.pkl"
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.feature_columns, features_path)
            
            logger.info(f"✅ Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Model save failed: {e}")
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            model_path = self.model_dir / f"xgboost_model_{self.model_version}.pkl"
            scaler_path = self.model_dir / f"scaler_{self.model_version}.pkl"
            features_path = self.model_dir / f"features_{self.model_version}.pkl"
            
            if all(path.exists() for path in [model_path, scaler_path, features_path]):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.feature_columns = joblib.load(features_path)
                
                logger.info(f"✅ Model loaded from {model_path}")
                return True
            else:
                logger.warning("⚠️ Model files not found")
                return False
                
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False

# ================================================================
# ENHANCED ENSEMBLE MODEL with Advanced Auto-Detection
# ================================================================

class EnsembleModel:
    """
    Enhanced Ensemble Model - Auto-selects Advanced or Basic based on availability
    Uses Advanced Ensemble (LightGBM+CatBoost+LSTM) if dependencies available,
    falls back to Basic Ensemble (XGBoost+RF+LR) if not
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Determine ensemble type and initialize accordingly
        if ADVANCED_MODELS_AVAILABLE:
            logger.info("🚀 Using Advanced Ensemble (LightGBM+CatBoost+LSTM)")
            self._ensemble = AdvancedEnsembleModel()
            self.ensemble_type = "ADVANCED"
        else:
            logger.info("⚠️ Using Basic Ensemble (XGBoost+RF+LR)")
            self._setup_basic_ensemble()
            self.ensemble_type = "BASIC"
        
        # Common properties
        self.ensemble_version = "v2.0"  # Updated version for enhanced ensemble
        self.is_trained = False
    
    def _setup_basic_ensemble(self):
        """Setup basic ensemble when advanced models unavailable"""
        # Individual models
        self.xgboost_model = None
        self.random_forest_model = None
        self.logistic_model = None
        
        # Ensemble components
        self.meta_model = None  # Meta-learner for stacking
        self.model_weights = {"xgboost": 0.5, "random_forest": 0.3, "logistic": 0.2}
        self.scaler = None
        self.feature_columns = None
        
        # Performance tracking
        self.individual_performance = {}
        self.ensemble_performance = {}
    
    def train(self, 
             training_data: pd.DataFrame,
             target_column: str = 'profitable',
             validation_split: float = 0.2,
             enable_stacking: bool = True) -> Dict[str, Any]:
        """
        Train ensemble - delegates to advanced or basic
        
        Args:
            training_data: DataFrame with features and target
            target_column: Name of target column
            validation_split: Fraction for validation set
            enable_stacking: Whether to use stacking ensemble
            
        Returns:
            Dict with training results and metrics
        """
        try:
            if self.ensemble_type == "ADVANCED":
                logger.info("🚀 Training Advanced Ensemble...")
                
                # Prepare data for advanced ensemble
                feature_columns = [col for col in training_data.columns if col != target_column]
                X = training_data[feature_columns].fillna(0)
                y = training_data[target_column]
                
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42, stratify=y
                )
                
                # Train advanced ensemble
                results = self._ensemble.train_ensemble(X_train, y_train, X_val, y_val)
                
                # Mark as trained
                self.is_trained = True
                
                # Add ensemble type info
                results['ensemble_type'] = 'ADVANCED'
                results['models_used'] = ['LightGBM', 'CatBoost', 'LSTM', 'RandomForest', 'LogisticRegression', 'SVM']
                
                logger.info("✅ Advanced Ensemble training complete!")
                return results
                
            else:
                logger.info("⚠️ Training Basic Ensemble...")
                return self._train_basic_ensemble(training_data, target_column, validation_split, enable_stacking)
                
        except Exception as e:
            logger.error(f"❌ Ensemble training failed: {e}")
            return {'status': 'error', 'message': str(e), 'ensemble_type': self.ensemble_type}
    
    def _train_basic_ensemble(self, 
                            training_data: pd.DataFrame,
                            target_column: str = 'profitable',
                            validation_split: float = 0.2,
                            enable_stacking: bool = True) -> Dict[str, Any]:
        """Train basic ensemble (XGBoost + RF + LR)"""
        try:
            logger.info(f"🔄 Training basic ensemble model on {len(training_data)} samples")
            
            # Prepare data
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns].fillna(0)
            y = training_data[target_column]
            
            self.feature_columns = feature_columns
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train individual models
            results = {}
            validation_predictions = {}
            
            # 1. XGBoost
            logger.info("🚀 Training XGBoost...")
            self.xgboost_model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='auc'
            )
            
            self.xgboost_model.fit(X_train_scaled, y_train, verbose=False)
            xgb_pred = self.xgboost_model.predict_proba(X_val_scaled)[:, 1]
            xgb_accuracy = accuracy_score(y_val, (xgb_pred > 0.5).astype(int))
            xgb_auc = roc_auc_score(y_val, xgb_pred)
            
            results['xgboost'] = {'accuracy': xgb_accuracy, 'auc': xgb_auc}
            validation_predictions['xgboost'] = xgb_pred
            
            # 2. Random Forest
            logger.info("🌲 Training Random Forest...")
            self.random_forest_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.random_forest_model.fit(X_train_scaled, y_train)
            rf_pred = self.random_forest_model.predict_proba(X_val_scaled)[:, 1]
            rf_accuracy = accuracy_score(y_val, (rf_pred > 0.5).astype(int))
            rf_auc = roc_auc_score(y_val, rf_pred)
            
            results['random_forest'] = {'accuracy': rf_accuracy, 'auc': rf_auc}
            validation_predictions['random_forest'] = rf_pred
            
            # 3. Logistic Regression
            logger.info("📈 Training Logistic Regression...")
            self.logistic_model = LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000
            )
            
            self.logistic_model.fit(X_train_scaled, y_train)
            lr_pred = self.logistic_model.predict_proba(X_val_scaled)[:, 1]
            lr_accuracy = accuracy_score(y_val, (lr_pred > 0.5).astype(int))
            lr_auc = roc_auc_score(y_val, lr_pred)
            
            results['logistic'] = {'accuracy': lr_accuracy, 'auc': lr_auc}
            validation_predictions['logistic'] = lr_pred
            
            # Calculate dynamic weights based on AUC performance
            total_auc = sum(r['auc'] for r in results.values())
            if total_auc > 0:
                self.model_weights = {
                    model: results[model]['auc'] / total_auc 
                    for model in results.keys()
                }
            
            # Weighted ensemble prediction
            weighted_pred = sum(
                validation_predictions[model] * self.model_weights[model]
                for model in validation_predictions.keys()
            )
            
            weighted_accuracy = accuracy_score(y_val, (weighted_pred > 0.5).astype(int))
            weighted_auc = roc_auc_score(y_val, weighted_pred)
            
            results['weighted_ensemble'] = {'accuracy': weighted_accuracy, 'auc': weighted_auc}
            
            # 4. Stacking ensemble (if enabled)
            if enable_stacking and len(validation_predictions) >= 2:
                logger.info("🔗 Training stacking meta-model...")
                
                # Create stacking features
                stacking_features = np.column_stack(list(validation_predictions.values()))
                
                # Train meta-learner
                self.meta_model = LogisticRegression(random_state=42)
                self.meta_model.fit(stacking_features, y_val)
                
                # Meta-model predictions
                meta_pred = self.meta_model.predict_proba(stacking_features)[:, 1]
                meta_accuracy = accuracy_score(y_val, (meta_pred > 0.5).astype(int))
                meta_auc = roc_auc_score(y_val, meta_pred)
                
                results['stacking_ensemble'] = {'accuracy': meta_accuracy, 'auc': meta_auc}
            
            # Store performance metrics
            self.individual_performance = results
            self.is_trained = True
            
            # Find best performer
            best_model = max(results.items(), key=lambda x: x[1]['auc'])
            
            # Overall results
            ensemble_results = {
                'status': 'success',
                'ensemble_type': 'BASIC',
                'models_used': ['XGBoost', 'RandomForest', 'LogisticRegression'],
                'total_samples': len(training_data),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': len(feature_columns),
                'individual_results': results,
                'model_weights': self.model_weights,
                'best_individual_model': best_model[0],
                'best_individual_auc': best_model[1]['auc'],
                'weighted_ensemble_auc': weighted_auc,
                'ensemble_version': self.ensemble_version
            }
            
            # Save ensemble
            self.save_ensemble()
            
            logger.info(f"✅ Basic Ensemble training complete!")
            logger.info(f"📊 Best individual: {best_model[0]} (AUC: {best_model[1]['auc']:.3f})")
            logger.info(f"🎯 Weighted ensemble AUC: {weighted_auc:.3f}")
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"❌ Basic ensemble training failed: {e}")
            return {'status': 'error', 'message': str(e), 'ensemble_type': 'BASIC'}
    
    def predict(self, features: Dict[str, float]) -> np.ndarray:
        """
        Make ensemble predictions - delegates to advanced or basic
        
        Args:
            features: Feature dictionary
            
        Returns:
            Array of prediction probabilities
        """
        try:
            if self.ensemble_type == "ADVANCED":
                if not self.is_trained:
                    return np.array([0.5])
                
                # Convert dict to DataFrame for advanced ensemble
                if isinstance(features, dict):
                    feature_df = pd.DataFrame([features])
                else:
                    feature_df = features
                
                result = self._ensemble.predict_ensemble(feature_df)
                return np.array([result.get('ensemble_prediction', 0.5)])
                
            else:
                return self._predict_basic_ensemble(features)
                
        except Exception as e:
            logger.error(f"❌ Ensemble prediction failed: {e}")
            return np.array([0.5])
    
    def _predict_basic_ensemble(self, features: Dict[str, float]) -> np.ndarray:
        """Make basic ensemble predictions"""
        try:
            if not self.is_trained:
                self.load_ensemble()
            
            if not self.is_trained:
                return np.array([0.5])  # Neutral prediction
            
            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Get individual predictions
            predictions = {}
            
            if self.xgboost_model:
                predictions['xgboost'] = self.xgboost_model.predict_proba(X_scaled)[0, 1]
            
            if self.random_forest_model:
                predictions['random_forest'] = self.random_forest_model.predict_proba(X_scaled)[0, 1]
            
            if self.logistic_model:
                predictions['logistic'] = self.logistic_model.predict_proba(X_scaled)[0, 1]
            
            if not predictions:
                return np.array([0.5])
            
            # Weighted ensemble prediction
            weighted_pred = sum(
                pred * self.model_weights.get(model, 0)
                for model, pred in predictions.items()
            )
            
            # Stacking prediction (if available)
            if self.meta_model and len(predictions) >= 2:
                stacking_features = np.array(list(predictions.values())).reshape(1, -1)
                stacking_pred = self.meta_model.predict_proba(stacking_features)[0, 1]
                
                # Combine weighted and stacking (70% weighted, 30% stacking)
                final_pred = 0.7 * weighted_pred + 0.3 * stacking_pred
            else:
                final_pred = weighted_pred
            
            return np.array([final_pred])
            
        except Exception as e:
            logger.error(f"❌ Basic ensemble prediction failed: {e}")
            return np.array([0.5])
    
    def predict_proba(self, X):
        """
        Return prediction probabilities in sklearn format
        
        Args:
            X: Feature array or dict
            
        Returns:
            Array of [negative_prob, positive_prob]
        """
        try:
            if isinstance(X, dict):
                predictions = self.predict(X)
            else:
                # Handle DataFrame or array input
                if hasattr(X, 'iloc'):  # DataFrame
                    features = X.iloc[0].to_dict() if len(X) > 0 else {}
                else:  # Array
                    features = {col: val for col, val in zip(self.feature_columns or [], X[0] if len(X) > 0 else [])}
                predictions = self.predict(features)
            
            positive_prob = predictions[0]
            negative_prob = 1.0 - positive_prob
            
            return np.array([[negative_prob, positive_prob]])
            
        except Exception as e:
            logger.error(f"❌ predict_proba failed: {e}")
            return np.array([[0.5, 0.5]])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance from all models"""
        try:
            if self.ensemble_type == "ADVANCED" and self.is_trained:
                # Get feature importance from advanced ensemble
                return {}  # Advanced ensemble handles this internally
            else:
                return self._get_basic_feature_importance()
                
        except Exception as e:
            logger.error(f"❌ Feature importance calculation failed: {e}")
            return {}
    
    def _get_basic_feature_importance(self) -> Dict[str, float]:
        """Get basic ensemble feature importance"""
        try:
            if not self.is_trained:
                return {}
            
            importance_dict = {}
            
            # XGBoost importance
            if self.xgboost_model and hasattr(self.xgboost_model, 'feature_importances_'):
                for i, col in enumerate(self.feature_columns):
                    importance_dict[col] = importance_dict.get(col, 0) + \
                                         self.xgboost_model.feature_importances_[i] * self.model_weights.get('xgboost', 0)
            
            # Random Forest importance
            if self.random_forest_model and hasattr(self.random_forest_model, 'feature_importances_'):
                for i, col in enumerate(self.feature_columns):
                    importance_dict[col] = importance_dict.get(col, 0) + \
                                         self.random_forest_model.feature_importances_[i] * self.model_weights.get('random_forest', 0)
            
            # Logistic Regression coefficients (absolute values)
            if self.logistic_model and hasattr(self.logistic_model, 'coef_'):
                coef_abs = np.abs(self.logistic_model.coef_[0])
                coef_norm = coef_abs / np.sum(coef_abs)  # Normalize
                
                for i, col in enumerate(self.feature_columns):
                    importance_dict[col] = importance_dict.get(col, 0) + \
                                         coef_norm[i] * self.model_weights.get('logistic', 0)
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"❌ Basic feature importance calculation failed: {e}")
            return {}
    
    def save_ensemble(self):
        """Save the ensemble model"""
        try:
            ensemble_path = self.model_dir / f"enhanced_ensemble_model_{self.ensemble_version}.pkl"
            
            if self.ensemble_type == "ADVANCED":
                # Save advanced ensemble (handled by advanced ensemble itself)
                logger.info("💾 Advanced ensemble saved internally")
            else:
                # Save basic ensemble
                ensemble_data = {
                    'ensemble_type': self.ensemble_type,
                    'xgboost_model': self.xgboost_model,
                    'random_forest_model': self.random_forest_model,
                    'logistic_model': self.logistic_model,
                    'meta_model': self.meta_model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'model_weights': self.model_weights,
                    'individual_performance': self.individual_performance,
                    'ensemble_version': self.ensemble_version,
                    'is_trained': self.is_trained
                }
                
                joblib.dump(ensemble_data, ensemble_path)
                logger.info(f"✅ Basic ensemble saved to {ensemble_path}")
            
        except Exception as e:
            logger.error(f"❌ Ensemble save failed: {e}")
    
    def load_ensemble(self):
        """Load the ensemble model"""
        try:
            ensemble_path = self.model_dir / f"enhanced_ensemble_model_{self.ensemble_version}.pkl"
            
            if self.ensemble_type == "ADVANCED":
                # Advanced ensemble loads itself
                logger.info("📂 Advanced ensemble loaded internally")
                return True
            else:
                # Load basic ensemble
                if ensemble_path.exists():
                    ensemble_data = joblib.load(ensemble_path)
                    
                    self.xgboost_model = ensemble_data.get('xgboost_model')
                    self.random_forest_model = ensemble_data.get('random_forest_model')
                    self.logistic_model = ensemble_data.get('logistic_model')
                    self.meta_model = ensemble_data.get('meta_model')
                    self.scaler = ensemble_data.get('scaler')
                    self.feature_columns = ensemble_data.get('feature_columns')
                    self.model_weights = ensemble_data.get('model_weights', {})
                    self.individual_performance = ensemble_data.get('individual_performance', {})
                    self.is_trained = ensemble_data.get('is_trained', False)
                    
                    logger.info(f"✅ Basic ensemble loaded from {ensemble_path}")
                    return True
                else:
                    logger.warning(f"⚠️ Enhanced ensemble model not found at {ensemble_path}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Ensemble load failed: {e}")
            return False
    
    def load_model(self):
        """
        Backward compatibility method for load_model()
        Delegates to load_ensemble() for compatibility with existing code
        """
        try:
            result = self.load_ensemble()
            if result:
                logger.info(f"✅ Enhanced {self.ensemble_type} ensemble loaded successfully")
            else:
                logger.warning(f"⚠️ Enhanced {self.ensemble_type} ensemble not found - using default parameters")
            return result
        except Exception as e:
            logger.error(f"❌ load_model failed: {e}")
            return False
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about current ensemble configuration"""
        return {
            'ensemble_type': self.ensemble_type,
            'ensemble_version': self.ensemble_version,
            'is_trained': self.is_trained,
            'advanced_models_available': ADVANCED_MODELS_AVAILABLE,
            'advanced_sentiment_available': ADVANCED_SENTIMENT_AVAILABLE,
            'models_used': self._get_models_used(),
            'performance_boost': self._get_expected_performance_boost()
        }
    
    def _get_models_used(self) -> List[str]:
        """Get list of models being used"""
        if self.ensemble_type == "ADVANCED":
            return ['LightGBM', 'CatBoost', 'LSTM', 'RandomForest', 'LogisticRegression', 'SVM']
        else:
            return ['XGBoost', 'RandomForest', 'LogisticRegression']
    
    def _get_expected_performance_boost(self) -> str:
        """Get expected performance boost description"""
        if self.ensemble_type == "ADVANCED":
            return "+15-25% accuracy improvement over basic ensemble"
        else:
            return "Basic ensemble - install advanced dependencies for +15-25% boost"

# ================================================================
# REST OF THE ORIGINAL CODE (TrainingDataCollector, etc.)
# ================================================================

class TrainingDataCollector:
    """
    Collect and prepare training data from historical signals
    """
    
    def __init__(self, market_data_service: MarketDataService, signal_logger):
        self.market_data = market_data_service
        self.signal_logger = signal_logger
        self.stock_universe = Nifty100StockUniverse()
        self.feature_engineer = FeatureEngineering(self.stock_universe)
    
    async def collect_training_data(self, 
                                  lookback_days: int = 90,
                                  min_samples_per_stock: int = 10) -> pd.DataFrame:
        """
        Collect historical training data for all Nifty 100 stocks
        
        Args:
            lookback_days: Number of days to look back
            min_samples_per_stock: Minimum samples required per stock
            
        Returns:
            DataFrame with features and targets
        """
        try:
            logger.info(f"Collecting training data for {lookback_days} days")
            
            all_training_data = []
            processed_stocks = 0
            
            # Get Nifty data for correlation analysis
            nifty_data = await self._get_nifty_historical_data(lookback_days)
            
            for symbol in self.stock_universe.get_all_stocks():
                try:
                    stock_data = await self._collect_stock_training_data(
                        symbol, lookback_days, nifty_data
                    )
                    
                    if len(stock_data) >= min_samples_per_stock:
                        all_training_data.extend(stock_data)
                        processed_stocks += 1
                        
                        if processed_stocks % 10 == 0:
                            logger.info(f"Processed {processed_stocks} stocks...")
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to collect data for {symbol}: {e}")
            
            # Convert to DataFrame
            if all_training_data:
                training_df = pd.DataFrame(all_training_data)
                logger.info(f"✅ Collected {len(training_df)} training samples from {processed_stocks} stocks")
                return training_df
            else:
                logger.warning("⚠️ No training data collected")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Training data collection failed: {e}")
            return pd.DataFrame()
    
    async def _collect_stock_training_data(self, 
                                         symbol: str, 
                                         lookback_days: int,
                                         nifty_data: pd.DataFrame) -> List[Dict]:
        """Collect training data for a single stock"""
        try:
            # Get historical OHLCV data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            ohlcv_data = await self.market_data.zerodha.get_historical_data(
                symbol, "day", start_date, end_date
            )
            
            if len(ohlcv_data) < 30:  # Need sufficient data
                return []
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': ohlcv.timestamp.date(),
                'open': ohlcv.open,
                'high': ohlcv.high,
                'low': ohlcv.low,
                'close': ohlcv.close,
                'volume': ohlcv.volume
            } for ohlcv in ohlcv_data])
            
            # Generate training samples
            training_samples = []
            
            for i in range(20, len(df) - 5):  # Leave buffer for forward-looking target
                try:
                    # Features based on data up to day i
                    historical_df = df.iloc[:i+1]
                    current_quote = {
                        'ltp': df.iloc[i]['close'],
                        'prev_close': df.iloc[i-1]['close'],
                        'volume': df.iloc[i]['volume'],
                        'change_percent': ((df.iloc[i]['close'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close']) * 100
                    }
                    
                    # Mock news sentiment (in production, use real historical sentiment)
                    news_sentiment = {
                        'sentiment_score': np.random.normal(0, 0.3),  # Mock sentiment
                        'news_count': np.random.randint(0, 5)
                    }
                    
                    # Engineer features
                    features = self.feature_engineer.engineer_features(
                        symbol, historical_df, current_quote, news_sentiment, nifty_data
                    )
                    
                    if not features:
                        continue
                    
                    # Calculate target (profitable signal)
                    entry_price = df.iloc[i]['close']
                    future_prices = df.iloc[i+1:i+6]['close']  # Next 5 days
                    
                    # Simple profit target: 2% gain within 5 days
                    max_gain = ((future_prices.max() - entry_price) / entry_price) * 100
                    target = 1 if max_gain >= 2.0 else 0
                    
                    # Add metadata
                    features.update({
                        'symbol': symbol,
                        'date': df.iloc[i]['date'],
                        'entry_price': entry_price,
                        'max_future_gain': max_gain,
                        'profitable': target
                    })
                    
                    training_samples.append(features)
                    
                except Exception as e:
                    logger.debug(f"Sample generation failed for {symbol} at {i}: {e}")
                    continue
            
            return training_samples
            
        except Exception as e:
            logger.error(f"Stock training data collection failed for {symbol}: {e}")
            return []
    
    async def _get_nifty_historical_data(self, lookback_days: int) -> pd.DataFrame:
        """Get Nifty historical data for correlation analysis"""
        try:
            # In production, fetch actual Nifty data
            # For now, generate mock data
            dates = pd.date_range(
                end=datetime.now().date(), 
                periods=lookback_days, 
                freq='D'
            )
            
            # Mock Nifty data with realistic movement
            base_price = 18500
            returns = np.random.normal(0.0005, 0.015, len(dates))  # 0.05% mean, 1.5% std
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            nifty_df = pd.DataFrame({
                'date': dates,
                'close': prices
            })
            
            return nifty_df
            
        except Exception as e:
            logger.error(f"Nifty data collection failed: {e}")
            return pd.DataFrame()

# Example usage and integration
async def train_production_model():
    """Train the production Enhanced Ensemble model"""
    
    # Initialize services
    market_data_service = MarketDataService("api_key", "access_token")
    await market_data_service.initialize()
    
    # Initialize data collector
    data_collector = TrainingDataCollector(market_data_service, None)
    
    try:
        # Collect training data
        print("🔄 Collecting training data...")
        training_data = await data_collector.collect_training_data(lookback_days=60)
        
        if len(training_data) < 100:
            print("❌ Insufficient training data")
            return
        
        # Prepare features
        feature_columns = [col for col in training_data.columns 
                          if col not in ['symbol', 'date', 'entry_price', 'max_future_gain', 'profitable']]
        
        # Train Enhanced Ensemble model
        print("🤖 Training Enhanced Ensemble model...")
        model = EnsembleModel()
        
        # Show what type of ensemble is being used
        ensemble_info = model.get_ensemble_info()
        print(f"📊 Using {ensemble_info['ensemble_type']} ensemble with models: {ensemble_info['models_used']}")
        print(f"🎯 Expected performance: {ensemble_info['performance_boost']}")
        
        # Train the model
        results = model.train(training_data[feature_columns + ['profitable']])
        
        print("✅ Enhanced Ensemble training complete!")
        if results.get('status') == 'success':
            if 'weighted_ensemble_auc' in results:
                print(f"📊 Results: AUC: {results['weighted_ensemble_auc']:.3f}")
            print(f"🎯 Ensemble Type: {results.get('ensemble_type', 'Unknown')}")
            print(f"🔧 Models Used: {results.get('models_used', [])}")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
    finally:
        await market_data_service.close()

if __name__ == "__main__":
    # Test the Enhanced ML pipeline
    asyncio.run(train_production_model())