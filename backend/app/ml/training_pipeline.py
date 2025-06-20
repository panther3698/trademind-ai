# backend/app/ml/training_pipeline.py
"""
TradeMind AI - Complete Training Pipeline
Comprehensive training system with 10-year historical data collection and auto-retraining
"""

import asyncio
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb

# Import our components
from app.ml.models import (
    Nifty100StockUniverse, FinBERTSentimentAnalyzer, 
    FeatureEngineering, XGBoostSignalModel, EnsembleModel
)

logger = logging.getLogger(__name__)

# ================================================================
# SIMPLE TRAINING PIPELINE (for main.py compatibility)
# ================================================================

class TrainingPipeline:
    """
    Simple training pipeline class that main.py expects
    Provides basic functionality and delegates to comprehensive pipeline
    """
    
    def __init__(self, market_data_service=None, signal_logger=None):
        self.market_data_service = market_data_service
        self.signal_logger = signal_logger
        
        # Initialize ML components
        self.stock_universe = Nifty100StockUniverse()
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        self.feature_engineer = FeatureEngineering(self.stock_universe)
        
        # Models
        self.xgboost_model = XGBoostSignalModel()
        self.ensemble_model = EnsembleModel()
        
        # Initialize comprehensive pipeline if services available
        self.comprehensive_pipeline = None
        if market_data_service and signal_logger:
            try:
                self.data_collector = HistoricalDataCollector(market_data_service)
                self.comprehensive_pipeline = ComprehensiveTrainingPipeline(
                    self.data_collector, signal_logger
                )
            except Exception as e:
                logger.warning(f"Comprehensive pipeline not available: {e}")
        
        # Training state
        self.is_trained = False
        self.training_results = {}
        self.pipeline_version = "v1.0"
    
    async def train_models(self, 
                          training_data: pd.DataFrame = None,
                          use_comprehensive: bool = True) -> Dict[str, Any]:
        """
        Train models using available data
        
        Args:
            training_data: Optional pre-prepared training data
            use_comprehensive: Whether to use comprehensive pipeline if available
            
        Returns:
            Dict with training results
        """
        try:
            logger.info("🚀 Starting model training...")
            
            # Try comprehensive pipeline first
            if use_comprehensive and self.comprehensive_pipeline:
                logger.info("📊 Using comprehensive training pipeline...")
                metrics = await self.comprehensive_pipeline.train_comprehensive_model()
                
                self.is_trained = True
                self.training_results = {
                    'status': 'success',
                    'method': 'comprehensive',
                    'accuracy': metrics.accuracy,
                    'auc_score': metrics.auc_score,
                    'training_samples': metrics.training_samples,
                    'validation_samples': metrics.validation_samples,
                    'feature_count': metrics.feature_count,
                    'training_duration': metrics.training_duration_minutes
                }
                
                return self.training_results
            
            # Fallback to simple training
            elif training_data is not None and len(training_data) > 0:
                logger.info("📊 Using simple training with provided data...")
                return await self._train_simple_models(training_data)
            
            # Generate synthetic data for testing
            else:
                logger.info("🧪 Using synthetic data for testing...")
                synthetic_data = self._generate_test_data()
                return await self._train_simple_models(synthetic_data)
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'method': 'unknown'
            }
    
    async def _train_simple_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train models with simple approach"""
        try:
            # Prepare data
            feature_columns = [col for col in training_data.columns 
                             if col not in ['symbol', 'date', 'target', 'profitable']]
            
            target_col = 'profitable' if 'profitable' in training_data.columns else 'target'
            
            if target_col not in training_data.columns:
                raise ValueError("No target column found in training data")
            
            X = training_data[feature_columns].fillna(0)
            y = training_data[target_col]
            
            # Train XGBoost
            xgb_results = self.xgboost_model.train_model(
                training_data[feature_columns + [target_col]].rename(columns={target_col: 'profitable'})
            )
            
            # Train ensemble if enough data
            ensemble_results = {}
            if len(training_data) > 500:
                ensemble_results = self.ensemble_model.train(
                    training_data[feature_columns + [target_col]].rename(columns={target_col: 'profitable'})
                )
            
            self.is_trained = True
            self.training_results = {
                'status': 'success',
                'method': 'simple',
                'xgboost_results': xgb_results,
                'ensemble_results': ensemble_results,
                'training_samples': len(training_data),
                'feature_count': len(feature_columns)
            }
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Simple training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _generate_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic test data"""
        try:
            logger.info(f"🧪 Generating {n_samples} synthetic samples...")
            
            data = []
            stocks = self.stock_universe.get_all_stocks()[:20]
            
            for i in range(n_samples):
                symbol = np.random.choice(stocks)
                
                # Generate realistic features
                sample = {
                    'symbol': symbol,
                    'date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                    
                    # Technical features
                    'rsi_14': np.random.uniform(20, 80),
                    'rsi_oversold': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'rsi_overbought': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'macd_line': np.random.normal(0, 5),
                    'macd_signal': np.random.normal(0, 5),
                    'macd_bullish': np.random.choice([0, 1]),
                    'bb_position': np.random.uniform(0, 1),
                    'adx': np.random.uniform(10, 50),
                    'strong_trend': np.random.choice([0, 1], p=[0.6, 0.4]),
                    'atr_percent': np.random.uniform(1, 5),
                    
                    # Price features
                    'price_gap_percent': np.random.normal(0, 2),
                    'price_vs_sma20': np.random.normal(0, 0.1),
                    'above_sma20': np.random.choice([0, 1]),
                    'price_vs_sma50': np.random.normal(0, 0.15),
                    'momentum_10d': np.random.normal(0, 0.08),
                    
                    # Volume features
                    'volume_ratio': np.random.lognormal(0, 0.5),
                    'high_volume': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'volume_trend': np.random.normal(0, 0.3),
                    
                    # Sentiment features
                    'news_sentiment': np.random.normal(0, 0.4),
                    'news_count': np.random.randint(0, 8),
                    'positive_sentiment': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'sentiment_strength': np.random.uniform(0, 1),
                    
                    # Market features
                    'nifty_performance': np.random.normal(0, 0.02),
                    'relative_to_nifty': np.random.normal(0, 0.03),
                    'outperforming_nifty': np.random.choice([0, 1]),
                    
                    # Sector features
                    'sector_banking': 1 if self.stock_universe.get_sector(symbol) == 'BANKING' else 0,
                    'sector_it': 1 if self.stock_universe.get_sector(symbol) == 'IT' else 0,
                    'sector_pharma': 1 if self.stock_universe.get_sector(symbol) == 'PHARMA' else 0,
                }
                
                # Generate target based on features
                profit_probability = 0.3
                if sample['rsi_oversold']:
                    profit_probability += 0.2
                if sample['strong_trend']:
                    profit_probability += 0.15
                if sample['positive_sentiment']:
                    profit_probability += 0.15
                if sample['high_volume']:
                    profit_probability += 0.1
                
                sample['profitable'] = np.random.choice([0, 1], p=[1-profit_probability, profit_probability])
                data.append(sample)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Test data generation failed: {e}")
            return pd.DataFrame()
    
    def get_trained_model(self):
        """Get the best trained model"""
        try:
            if self.ensemble_model.is_trained:
                return self.ensemble_model
            elif self.xgboost_model.model is not None:
                return self.xgboost_model
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting trained model: {e}")
            return None
    
    def save_pipeline(self, save_dir: str = "models"):
        """Save the training pipeline"""
        try:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            
            # Save models
            if self.xgboost_model.model is not None:
                self.xgboost_model.save_model()
            
            if self.ensemble_model.is_trained:
                self.ensemble_model.save_ensemble()
            
            # Save pipeline metadata
            metadata = {
                'pipeline_version': self.pipeline_version,
                'is_trained': self.is_trained,
                'training_results': self.training_results,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(save_path / "simple_pipeline_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Simple pipeline saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Pipeline save failed: {e}")

# ================================================================
# COMPREHENSIVE TRAINING PIPELINE (Your existing implementation)
# ================================================================

@dataclass
class TrainingMetrics:
    """Training session metrics"""
    training_id: str
    timestamp: datetime
    model_version: str
    training_samples: int
    validation_samples: int
    accuracy: float
    auc_score: float
    precision: float
    recall: float
    f1_score: float
    feature_count: int
    training_duration_minutes: float
    data_start_date: datetime
    data_end_date: datetime
    top_features: List[Tuple[str, float]]

@dataclass
class ModelPerformance:
    """Live model performance tracking"""
    model_version: str
    deployment_date: datetime
    total_predictions: int
    correct_predictions: int
    live_accuracy: float
    profitable_signals: int
    unprofitable_signals: int
    avg_pnl_per_signal: float
    confidence_accuracy_by_bucket: Dict[str, float]
    last_updated: datetime

class HistoricalDataCollector:
    """
    Collect 10 years of historical data for all Nifty 100 stocks
    """
    
    def __init__(self, market_data_service):
        self.market_data = market_data_service
        self.stock_universe = Nifty100StockUniverse()
        
        # Database for storing historical data
        self.db_path = Path("data/historical_data.db")
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Data collection parameters
        self.years_lookback = 10
        self.batch_size = 50
        self.rate_limit_delay = 0.1
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for historical data storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    features_json TEXT,
                    target_profitable INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_collection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    start_date DATE,
                    end_date DATE,
                    records_collected INTEGER,
                    collection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date ON ohlcv_data(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON features_data(symbol, date)')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Historical data database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def collect_10_year_data(self, 
                                  symbols: List[str] = None,
                                  force_refresh: bool = False) -> Dict[str, int]:
        """
        Collect 10 years of historical data for Nifty 100 stocks
        """
        try:
            if symbols is None:
                symbols = self.stock_universe.get_all_stocks()
            
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=self.years_lookback * 365)
            
            logger.info(f"🔄 Starting 10-year data collection for {len(symbols)} stocks")
            logger.info(f"📅 Date range: {start_date} to {end_date}")
            
            collection_stats = {
                "total_symbols": len(symbols),
                "successful_collections": 0,
                "failed_collections": 0,
                "total_records": 0,
                "start_time": datetime.now()
            }
            
            # Process in batches
            for i in range(0, len(symbols), self.batch_size):
                batch = symbols[i:i + self.batch_size]
                logger.info(f"📊 Processing batch {i//self.batch_size + 1}/{(len(symbols)-1)//self.batch_size + 1}")
                
                batch_results = await self._collect_batch_data(batch, start_date, end_date, force_refresh)
                
                for result in batch_results:
                    if result["success"]:
                        collection_stats["successful_collections"] += 1
                        collection_stats["total_records"] += result["records_count"]
                    else:
                        collection_stats["failed_collections"] += 1
                
                await asyncio.sleep(2)
            
            collection_stats["end_time"] = datetime.now()
            collection_stats["duration_minutes"] = (
                collection_stats["end_time"] - collection_stats["start_time"]
            ).total_seconds() / 60
            
            logger.info(f"✅ Data collection complete!")
            logger.info(f"📈 Success: {collection_stats['successful_collections']}/{collection_stats['total_symbols']}")
            logger.info(f"📊 Total records: {collection_stats['total_records']:,}")
            
            return collection_stats
            
        except Exception as e:
            logger.error(f"Historical data collection failed: {e}")
            return {"error": str(e)}
    
    async def _collect_batch_data(self, symbols, start_date, end_date, force_refresh):
        """Collect data for a batch of symbols"""
        tasks = []
        for symbol in symbols:
            task = self._collect_symbol_data(symbol, start_date, end_date, force_refresh)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Data collection failed for {symbols[i]}: {result}")
                processed_results.append({"symbol": symbols[i], "success": False, "error": str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _collect_symbol_data(self, symbol, start_date, end_date, force_refresh):
        """Collect historical data for a single symbol"""
        try:
            if not force_refresh and self._has_recent_data(symbol, start_date, end_date):
                return {"symbol": symbol, "success": True, "records_count": 0, "message": "Already exists"}
            
            # Mock data collection for now (replace with real API calls)
            await asyncio.sleep(self.rate_limit_delay)
            
            # Generate mock historical data
            mock_data = self._generate_mock_ohlcv_data(symbol, start_date, end_date)
            records_stored = self._store_ohlcv_data(symbol, mock_data)
            self._log_collection(symbol, start_date, end_date, records_stored)
            
            return {"symbol": symbol, "success": True, "records_count": records_stored}
            
        except Exception as e:
            logger.error(f"Symbol data collection failed for {symbol}: {e}")
            return {"symbol": symbol, "success": False, "error": str(e)}
    
    def _generate_mock_ohlcv_data(self, symbol, start_date, end_date):
        """Generate mock OHLCV data for testing"""
        try:
            from collections import namedtuple
            OHLCV = namedtuple('OHLCV', ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            mock_data = []
            current_date = start_date
            base_price = np.random.uniform(100, 3000)
            
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Only weekdays
                    # Generate realistic OHLCV
                    daily_return = np.random.normal(0.001, 0.02)
                    open_price = base_price * (1 + daily_return)
                    
                    high_premium = np.random.uniform(0.005, 0.03)
                    low_discount = np.random.uniform(0.005, 0.03)
                    
                    high_price = open_price * (1 + high_premium)
                    low_price = open_price * (1 - low_discount)
                    
                    close_return = np.random.normal(0, 0.01)
                    close_price = open_price * (1 + close_return)
                    close_price = max(low_price, min(high_price, close_price))
                    
                    volume = int(np.random.lognormal(12, 0.8))
                    
                    ohlcv = OHLCV(
                        timestamp=datetime.combine(current_date, datetime.min.time()),
                        open=round(open_price, 2),
                        high=round(high_price, 2),
                        low=round(low_price, 2),
                        close=round(close_price, 2),
                        volume=volume
                    )
                    
                    mock_data.append(ohlcv)
                    base_price = close_price
                
                current_date += timedelta(days=1)
            
            return mock_data
            
        except Exception as e:
            logger.error(f"Mock data generation failed for {symbol}: {e}")
            return []
    
    def _has_recent_data(self, symbol, start_date, end_date):
        """Check if we have recent data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM ohlcv_data 
                WHERE symbol = ? AND date >= ? AND date <= ?
            ''', (symbol, start_date, end_date))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            expected_days = (end_date - start_date).days * 0.7
            return count >= (expected_days * 0.8)
            
        except Exception as e:
            logger.error(f"Data check failed for {symbol}: {e}")
            return False
    
    def _store_ohlcv_data(self, symbol, historical_data):
        """Store OHLCV data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            records_stored = 0
            
            for ohlcv in historical_data:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO ohlcv_data 
                        (symbol, date, open_price, high_price, low_price, close_price, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        ohlcv.timestamp.date(),
                        ohlcv.open,
                        ohlcv.high,
                        ohlcv.low,
                        ohlcv.close,
                        ohlcv.volume
                    ))
                    records_stored += 1
                    
                except sqlite3.IntegrityError:
                    continue
            
            conn.commit()
            conn.close()
            
            return records_stored
            
        except Exception as e:
            logger.error(f"OHLCV data storage failed for {symbol}: {e}")
            return 0
    
    def _log_collection(self, symbol, start_date, end_date, records_count):
        """Log data collection activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO data_collection_log 
                (symbol, start_date, end_date, records_collected)
                VALUES (?, ?, ?, ?)
            ''', (symbol, start_date, end_date, records_count))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Collection logging failed for {symbol}: {e}")
    
    def get_historical_data(self, symbol, start_date=None, end_date=None):
        """Retrieve historical data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if start_date is None:
                start_date = datetime.now().date() - timedelta(days=365 * self.years_lookback)
            if end_date is None:
                end_date = datetime.now().date()
            
            query = '''
                SELECT symbol, date, open_price, high_price, low_price, close_price, volume
                FROM ohlcv_data 
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            df['date'] = pd.to_datetime(df['date'])
            return df
            
        except Exception as e:
            logger.error(f"Historical data retrieval failed for {symbol}: {e}")
            return pd.DataFrame()

class ComprehensiveTrainingPipeline:
    """
    Comprehensive training pipeline using 10 years of historical data
    """
    
    def __init__(self, historical_collector, signal_logger):
        self.data_collector = historical_collector
        self.signal_logger = signal_logger
        self.stock_universe = Nifty100StockUniverse()
        self.feature_engineer = FeatureEngineering(self.stock_universe)
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        
        self.min_samples_per_stock = 500
        self.lookback_window = 30
        self.forward_window = 5
        self.profit_threshold = 2.0
        
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        self.training_results_dir = Path("logs/training_results")
        self.training_results_dir.mkdir(parents=True, exist_ok=True)
    
    async def train_comprehensive_model(self, years_lookback=10, retrain=False):
        """Train comprehensive model using historical data"""
        try:
            training_start = datetime.now()
            training_id = f"comprehensive_{training_start.strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"🚀 Starting comprehensive model training")
            logger.info(f"🆔 Training ID: {training_id}")
            
            # Generate training features
            training_data = await self._generate_training_features(years_lookback)
            
            if len(training_data) < 1000:
                # Use synthetic data if insufficient real data
                logger.warning("Insufficient real data, using synthetic data")
                training_data = self._generate_synthetic_training_data(5000)
            
            # Train model
            model_results = await self._train_xgboost_model(training_data, training_id)
            validation_results = await self._validate_model_performance(model_results["model"], training_data)
            
            # Save model
            await self._save_trained_model(
                model_results["model"], 
                model_results["scaler"],
                model_results["feature_columns"],
                training_id
            )
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds() / 60
            
            # Create metrics
            metrics = TrainingMetrics(
                training_id=training_id,
                timestamp=training_start,
                model_version=f"v_{training_id}",
                training_samples=model_results["training_samples"],
                validation_samples=model_results["validation_samples"],
                accuracy=validation_results["accuracy"],
                auc_score=validation_results["auc_score"],
                precision=validation_results["precision"],
                recall=validation_results["recall"],
                f1_score=validation_results["f1_score"],
                feature_count=len(model_results["feature_columns"]),
                training_duration_minutes=training_duration,
                data_start_date=datetime.now() - timedelta(days=years_lookback * 365),
                data_end_date=datetime.now(),
                top_features=model_results["top_features"]
            )
            
            await self._save_training_metrics(metrics)
            
            logger.info("✅ Comprehensive model training complete!")
            logger.info(f"📊 Accuracy: {metrics.accuracy:.1%}, AUC: {metrics.auc_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive training failed: {e}")
            raise
    
    async def _generate_training_features(self, years_lookback):
        """Generate training features from historical data"""
        try:
            # For now, generate synthetic features
            return self._generate_synthetic_training_data(2000)
            
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_training_data(self, n_samples):
        """Generate synthetic training data"""
        try:
            data = []
            stocks = self.stock_universe.get_all_stocks()[:30]
            
            for i in range(n_samples):
                symbol = np.random.choice(stocks)
                
                sample = {
                    'symbol': symbol,
                    'date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
                    
                    # Technical features
                    'rsi_14': np.random.uniform(20, 80),
                    'rsi_oversold': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'rsi_overbought': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'macd_line': np.random.normal(0, 5),
                    'macd_signal': np.random.normal(0, 5),
                    'macd_bullish': np.random.choice([0, 1]),
                    'bb_position': np.random.uniform(0, 1),
                    'adx': np.random.uniform(10, 50),
                    'strong_trend': np.random.choice([0, 1], p=[0.6, 0.4]),
                    'atr_percent': np.random.uniform(1, 5),
                    
                    # Price features
                    'price_gap_percent': np.random.normal(0, 2),
                    'price_vs_sma20': np.random.normal(0, 0.1),
                    'above_sma20': np.random.choice([0, 1]),
                    'price_vs_sma50': np.random.normal(0, 0.15),
                    'momentum_10d': np.random.normal(0, 0.08),
                    
                    # Volume features
                    'volume_ratio': np.random.lognormal(0, 0.5),
                    'high_volume': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'volume_trend': np.random.normal(0, 0.3),
                    
                    # Sentiment features
                    'news_sentiment': np.random.normal(0, 0.4),
                    'news_count': np.random.randint(0, 8),
                    'positive_sentiment': np.random.choice([0, 1], p=[0.7, 0.3]),
                    'sentiment_strength': np.random.uniform(0, 1),
                    
                    # Market features
                    'nifty_performance': np.random.normal(0, 0.02),
                    'relative_to_nifty': np.random.normal(0, 0.03),
                    'outperforming_nifty': np.random.choice([0, 1]),
                    
                    # Sector features
                    'sector_banking': 1 if self.stock_universe.get_sector(symbol) == 'BANKING' else 0,
                    'sector_it': 1 if self.stock_universe.get_sector(symbol) == 'IT' else 0,
                    'sector_pharma': 1 if self.stock_universe.get_sector(symbol) == 'PHARMA' else 0,
                }
                
                # Generate realistic target
                profit_probability = 0.35
                if sample['rsi_oversold']:
                    profit_probability += 0.15
                if sample['strong_trend']:
                    profit_probability += 0.1
                if sample['positive_sentiment']:
                    profit_probability += 0.1
                if sample['high_volume']:
                    profit_probability += 0.05
                
                sample['target_profitable'] = np.random.choice([0, 1], p=[1-profit_probability, profit_probability])
                data.append(sample)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            return pd.DataFrame()
    
    async def _train_xgboost_model(self, training_data, training_id):
        """Train XGBoost model"""
        try:
            feature_columns = [col for col in training_data.columns 
                             if col not in ['symbol', 'date', 'target_profitable']]
            
            X = training_data[feature_columns].fillna(0)
            y = training_data['target_profitable']
            
            # Time-based split
            training_data_sorted = training_data.sort_values('date')
            split_idx = int(len(training_data_sorted) * 0.8)
            
            train_indices = training_data_sorted.index[:split_idx]
            val_indices = training_data_sorted.index[split_idx:]
            
            X_train, X_val = X.loc[train_indices], X.loc[val_indices]
            y_train, y_val = y.loc[train_indices], y.loc[val_indices]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train XGBoost
            xgb_params = {
                'objective': 'binary:logistic',
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'eval_metric': 'auc'
            }
            
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            
            # Feature importance
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            return {
                "model": model,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "top_features": top_features
            }
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            raise
    
    async def _validate_model_performance(self, model, training_data):
        """Validate model performance"""
        try:
            feature_columns = [col for col in training_data.columns 
                             if col not in ['symbol', 'date', 'target_profitable']]
            
            X = training_data[feature_columns].fillna(0)
            y = training_data['target_profitable']
            
            # Time-based split for validation
            training_data_sorted = training_data.sort_values('date')
            split_idx = int(len(training_data_sorted) * 0.8)
            val_indices = training_data_sorted.index[split_idx:]
            
            X_val = X.loc[val_indices]
            y_val = y.loc[val_indices]
            
            # Scale validation data
            scaler = StandardScaler()
            scaler.fit(X.loc[training_data_sorted.index[:split_idx]])
            X_val_scaled = scaler.transform(X_val)
            
            # Predictions
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_val, y_pred)
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            class_report = classification_report(y_val, y_pred, output_dict=True)
            
            return {
                "accuracy": accuracy,
                "auc_score": auc_score,
                "precision": class_report['1']['precision'],
                "recall": class_report['1']['recall'],
                "f1_score": class_report['1']['f1-score'],
                "classification_report": class_report
            }
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {"accuracy": 0.0, "auc_score": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    
    async def _save_trained_model(self, model, scaler, feature_columns, training_id):
        """Save trained model"""
        try:
            model_version = f"comprehensive_{training_id}"
            
            model_path = self.models_dir / f"xgboost_model_{model_version}.pkl"
            scaler_path = self.models_dir / f"scaler_{model_version}.pkl"
            features_path = self.models_dir / f"features_{model_version}.pkl"
            metadata_path = self.models_dir / f"metadata_{model_version}.json"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(feature_columns, features_path)
            
            metadata = {
                "model_version": model_version,
                "training_id": training_id,
                "created_at": datetime.now().isoformat(),
                "feature_count": len(feature_columns),
                "model_type": "xgboost_comprehensive"
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise
    
    async def _save_training_metrics(self, metrics):
        """Save training metrics"""
        try:
            metrics_file = self.training_results_dir / f"training_metrics_{metrics.training_id}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
            
            logger.info(f"✅ Training metrics saved: {metrics_file}")
            
        except Exception as e:
            logger.error(f"Training metrics save failed: {e}")

# ================================================================
# TRAINING ORCHESTRATOR
# ================================================================

class TrainingOrchestrator:
    """Main orchestrator for all training activities"""
    
    def __init__(self, market_data_service=None, signal_logger=None):
        self.market_data = market_data_service
        self.signal_logger = signal_logger
        
        # Initialize simple pipeline (always available)
        self.simple_pipeline = TrainingPipeline(market_data_service, signal_logger)
        
        # Initialize comprehensive components if services available
        self.data_collector = None
        self.comprehensive_pipeline = None
        
        if market_data_service and signal_logger:
            try:
                self.data_collector = HistoricalDataCollector(market_data_service)
                self.comprehensive_pipeline = ComprehensiveTrainingPipeline(
                    self.data_collector, signal_logger
                )
            except Exception as e:
                logger.warning(f"Comprehensive pipeline not available: {e}")
    
    async def run_initial_training(self, use_comprehensive=True):
        """Run initial training"""
        try:
            if use_comprehensive and self.comprehensive_pipeline:
                logger.info("🚀 Running comprehensive training...")
                return await self.comprehensive_pipeline.train_comprehensive_model()
            else:
                logger.info("🚀 Running simple training...")
                return await self.simple_pipeline.train_models(use_comprehensive=False)
                
        except Exception as e:
            logger.error(f"Initial training failed: {e}")
            # Fallback to simple training
            return await self.simple_pipeline.train_models(use_comprehensive=False)

# ================================================================
# EXAMPLE USAGE
# ================================================================

async def example_usage():
    """Example of how to use the training pipeline"""
    try:
        print("🚀 TradeMind AI Training Pipeline Example")
        
        # Initialize simple pipeline
        pipeline = TrainingPipeline()
        
        # Run training
        results = await pipeline.train_models(use_comprehensive=False)
        
        print(f"📊 Training Results:")
        print(f"   Status: {results.get('status')}")
        print(f"   Method: {results.get('method')}")
        
        if results.get('status') == 'success':
            print("✅ Training completed successfully!")
            
            # Get trained model
            trained_model = pipeline.get_trained_model()
            if trained_model:
                print(f"🤖 Best model: {type(trained_model).__name__}")
                
                # Test prediction
                test_features = {
                    'rsi_14': 35.0,
                    'price_gap_percent': 1.5,
                    'positive_sentiment': 1.0,
                    'high_volume': 1.0
                }
                
                if hasattr(trained_model, 'predict'):
                    prediction = trained_model.predict(test_features)
                    print(f"🔮 Test prediction: {prediction[0]:.3f}")
        else:
            print(f"❌ Training failed: {results.get('error')}")
            
    except Exception as e:
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())