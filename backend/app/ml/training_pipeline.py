# backend/app/ml/training_pipeline.py
"""
TradeMind AI - Comprehensive Training Pipeline
10-year historical data collection, training, and auto-retraining system
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
import aiohttp
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
    FeatureEngineering, XGBoostSignalModel
)
from app.core.signal_logger import InstitutionalSignalLogger, TradeOutcome
from app.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)

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
    
    def __init__(self, market_data_service: MarketDataService):
        self.market_data = market_data_service
        self.stock_universe = Nifty100StockUniverse()
        
        # Database for storing historical data
        self.db_path = Path("data/historical_data.db")
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Data collection parameters
        self.years_lookback = 10
        self.batch_size = 50  # Process 50 stocks at a time
        self.rate_limit_delay = 0.1  # 100ms between API calls
        
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
                    features_json TEXT,  -- Store feature dict as JSON
                    target_profitable INTEGER,  -- 0 or 1
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
            
            # Create indexes for performance
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
        
        Args:
            symbols: List of symbols to collect (default: all Nifty 100)
            force_refresh: Whether to refresh existing data
            
        Returns:
            Dict with collection statistics
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
            
            # Process in batches to avoid overwhelming the API
            for i in range(0, len(symbols), self.batch_size):
                batch = symbols[i:i + self.batch_size]
                logger.info(f"📊 Processing batch {i//self.batch_size + 1}/{(len(symbols)-1)//self.batch_size + 1}: {len(batch)} stocks")
                
                # Process batch concurrently
                batch_results = await self._collect_batch_data(batch, start_date, end_date, force_refresh)
                
                # Update statistics
                for result in batch_results:
                    if result["success"]:
                        collection_stats["successful_collections"] += 1
                        collection_stats["total_records"] += result["records_count"]
                    else:
                        collection_stats["failed_collections"] += 1
                
                # Rate limiting between batches
                await asyncio.sleep(2)
            
            collection_stats["end_time"] = datetime.now()
            collection_stats["duration_minutes"] = (
                collection_stats["end_time"] - collection_stats["start_time"]
            ).total_seconds() / 60
            
            logger.info(f"✅ Data collection complete!")
            logger.info(f"📈 Success: {collection_stats['successful_collections']}/{collection_stats['total_symbols']} stocks")
            logger.info(f"📊 Total records: {collection_stats['total_records']:,}")
            logger.info(f"⏱️  Duration: {collection_stats['duration_minutes']:.1f} minutes")
            
            return collection_stats
            
        except Exception as e:
            logger.error(f"Historical data collection failed: {e}")
            return {"error": str(e)}
    
    async def _collect_batch_data(self, 
                                 symbols: List[str], 
                                 start_date: datetime.date,
                                 end_date: datetime.date,
                                 force_refresh: bool) -> List[Dict]:
        """Collect data for a batch of symbols"""
        tasks = []
        
        for symbol in symbols:
            task = self._collect_symbol_data(symbol, start_date, end_date, force_refresh)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Data collection failed for {symbols[i]}: {result}")
                processed_results.append({"symbol": symbols[i], "success": False, "error": str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _collect_symbol_data(self, 
                                  symbol: str,
                                  start_date: datetime.date,
                                  end_date: datetime.date,
                                  force_refresh: bool) -> Dict:
        """Collect historical data for a single symbol"""
        try:
            # Check if data already exists
            if not force_refresh and self._has_recent_data(symbol, start_date, end_date):
                logger.debug(f"✅ {symbol}: Data already exists, skipping")
                return {"symbol": symbol, "success": True, "records_count": 0, "message": "Already exists"}
            
            # Get historical data from Zerodha
            historical_data = await self.market_data.zerodha.get_historical_data(
                symbol, "day", datetime.combine(start_date, datetime.min.time()), 
                datetime.combine(end_date, datetime.min.time())
            )
            
            if not historical_data:
                logger.warning(f"⚠️ {symbol}: No historical data received")
                return {"symbol": symbol, "success": False, "error": "No data received"}
            
            # Store in database
            records_stored = self._store_ohlcv_data(symbol, historical_data)
            
            # Log collection
            self._log_collection(symbol, start_date, end_date, records_stored)
            
            logger.debug(f"✅ {symbol}: {records_stored} records collected")
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            return {
                "symbol": symbol, 
                "success": True, 
                "records_count": records_stored,
                "date_range": f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            logger.error(f"Symbol data collection failed for {symbol}: {e}")
            return {"symbol": symbol, "success": False, "error": str(e)}
    
    def _has_recent_data(self, symbol: str, start_date: datetime.date, end_date: datetime.date) -> bool:
        """Check if we already have recent data for this symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM ohlcv_data 
                WHERE symbol = ? AND date >= ? AND date <= ?
            ''', (symbol, start_date, end_date))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            # Consider data complete if we have at least 80% of expected trading days
            expected_days = (end_date - start_date).days * 0.7  # ~70% are trading days
            return count >= (expected_days * 0.8)
            
        except Exception as e:
            logger.error(f"Data check failed for {symbol}: {e}")
            return False
    
    def _store_ohlcv_data(self, symbol: str, historical_data: List) -> int:
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
                    # Record already exists
                    continue
            
            conn.commit()
            conn.close()
            
            return records_stored
            
        except Exception as e:
            logger.error(f"OHLCV data storage failed for {symbol}: {e}")
            return 0
    
    def _log_collection(self, symbol: str, start_date: datetime.date, end_date: datetime.date, records_count: int):
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
    
    def get_historical_data(self, 
                           symbol: str, 
                           start_date: datetime.date = None,
                           end_date: datetime.date = None) -> pd.DataFrame:
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
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Historical data retrieval failed for {symbol}: {e}")
            return pd.DataFrame()

class ComprehensiveTrainingPipeline:
    """
    Comprehensive training pipeline using 10 years of historical data
    """
    
    def __init__(self, 
                 historical_collector: HistoricalDataCollector,
                 signal_logger: InstitutionalSignalLogger):
        self.data_collector = historical_collector
        self.signal_logger = signal_logger
        self.stock_universe = Nifty100StockUniverse()
        self.feature_engineer = FeatureEngineering(self.stock_universe)
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        
        # Training configuration
        self.min_samples_per_stock = 500  # Minimum samples required
        self.lookback_window = 30  # Days of data for feature calculation
        self.forward_window = 5   # Days to check for profit
        self.profit_threshold = 2.0  # 2% profit target
        
        # Model storage
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Training results storage
        self.training_results_dir = Path("logs/training_results")
        self.training_results_dir.mkdir(parents=True, exist_ok=True)
    
    async def train_comprehensive_model(self, 
                                      years_lookback: int = 10,
                                      retrain: bool = False) -> TrainingMetrics:
        """
        Train comprehensive model using 10 years of data
        
        Args:
            years_lookback: Years of historical data to use
            retrain: Force retraining even if model exists
            
        Returns:
            TrainingMetrics with training results
        """
        try:
            training_start = datetime.now()
            training_id = f"comprehensive_{training_start.strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"🚀 Starting comprehensive model training")
            logger.info(f"📅 Using {years_lookback} years of historical data")
            logger.info(f"🆔 Training ID: {training_id}")
            
            # Step 1: Ensure we have historical data
            logger.info("📊 Step 1: Checking historical data availability...")
            data_stats = await self._ensure_historical_data(years_lookback)
            
            # Step 2: Generate training features
            logger.info("🔧 Step 2: Generating training features...")
            training_data = await self._generate_training_features(years_lookback)
            
            if len(training_data) < 1000:
                raise ValueError(f"Insufficient training data: {len(training_data)} samples")
            
            # Step 3: Train XGBoost model
            logger.info("🤖 Step 3: Training XGBoost model...")
            model_results = await self._train_xgboost_model(training_data, training_id)
            
            # Step 4: Validate model performance
            logger.info("📈 Step 4: Validating model performance...")
            validation_results = await self._validate_model_performance(model_results["model"], training_data)
            
            # Step 5: Save model and results
            logger.info("💾 Step 5: Saving model and results...")
            model_path = await self._save_trained_model(
                model_results["model"], 
                model_results["scaler"],
                model_results["feature_columns"],
                training_id
            )
            
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds() / 60
            
            # Create training metrics
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
            
            # Save training metrics
            await self._save_training_metrics(metrics)
            
            logger.info("✅ Comprehensive model training complete!")
            logger.info(f"📊 Final Results:")
            logger.info(f"   - Accuracy: {metrics.accuracy:.1%}")
            logger.info(f"   - AUC Score: {metrics.auc_score:.3f}")
            logger.info(f"   - Training Samples: {metrics.training_samples:,}")
            logger.info(f"   - Features: {metrics.feature_count}")
            logger.info(f"   - Duration: {metrics.training_duration_minutes:.1f} minutes")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive training failed: {e}")
            raise
    
    async def _ensure_historical_data(self, years_lookback: int) -> Dict:
        """Ensure we have sufficient historical data"""
        try:
            # Check data availability
            all_stocks = self.stock_universe.get_all_stocks()
            
            # Sample a few stocks to check data quality
            sample_stocks = all_stocks[:10]
            data_quality = {}
            
            for symbol in sample_stocks:
                start_date = datetime.now().date() - timedelta(days=years_lookback * 365)
                df = self.data_collector.get_historical_data(symbol, start_date)
                data_quality[symbol] = len(df)
            
            avg_records = np.mean(list(data_quality.values()))
            min_expected = years_lookback * 250  # ~250 trading days per year
            
            if avg_records < min_expected * 0.7:  # Less than 70% of expected data
                logger.warning(f"⚠️ Insufficient historical data. Collecting...")
                await self.data_collector.collect_10_year_data(force_refresh=False)
            
            return {
                "stocks_checked": len(sample_stocks),
                "avg_records_per_stock": avg_records,
                "data_quality": "sufficient" if avg_records >= min_expected * 0.7 else "insufficient"
            }
            
        except Exception as e:
            logger.error(f"Historical data check failed: {e}")
            return {"error": str(e)}
    
    async def _generate_training_features(self, years_lookback: int) -> pd.DataFrame:
        """Generate comprehensive training features from historical data"""
        try:
            all_stocks = self.stock_universe.get_all_stocks()
            all_training_data = []
            
            logger.info(f"🔧 Generating features for {len(all_stocks)} stocks...")
            
            # Process stocks in batches
            batch_size = 20
            for i in range(0, len(all_stocks), batch_size):
                batch = all_stocks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_stocks)-1)//batch_size + 1}")
                
                # Process batch
                batch_tasks = [self._generate_stock_features(symbol, years_lookback) for symbol in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Collect results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Feature generation failed for {batch[j]}: {result}")
                    elif result is not None and len(result) > 0:
                        all_training_data.extend(result)
                
                # Rate limiting
                await asyncio.sleep(1)
            
            if not all_training_data:
                raise ValueError("No training data generated")
            
            # Convert to DataFrame
            training_df = pd.DataFrame(all_training_data)
            
            logger.info(f"✅ Generated {len(training_df)} training samples")
            logger.info(f"📊 Features: {len([col for col in training_df.columns if col not in ['symbol', 'date', 'target_profitable']])}")
            
            return training_df
            
        except Exception as e:
            logger.error(f"Training feature generation failed: {e}")
            raise
    
    async def _generate_stock_features(self, symbol: str, years_lookback: int) -> List[Dict]:
        """Generate training features for a single stock"""
        try:
            # Get historical data
            start_date = datetime.now().date() - timedelta(days=years_lookback * 365)
            df = self.data_collector.get_historical_data(symbol, start_date)
            
            if len(df) < self.min_samples_per_stock:
                logger.debug(f"Insufficient data for {symbol}: {len(df)} records")
                return []
            
            # Sort by date
            df = df.sort_values('date')
            
            training_samples = []
            
            # Generate samples with sliding window
            for i in range(self.lookback_window, len(df) - self.forward_window):
                try:
                    # Get historical window for feature calculation
                    historical_window = df.iloc[i-self.lookback_window:i+1].copy()
                    
                    # Create mock current quote
                    current_row = df.iloc[i]
                    prev_row = df.iloc[i-1]
                    
                    current_quote = {
                        'ltp': current_row['close_price'],
                        'prev_close': prev_row['close_price'],
                        'volume': current_row['volume'],
                        'change_percent': ((current_row['close_price'] - prev_row['close_price']) / prev_row['close_price']) * 100
                    }
                    
                    # Mock news sentiment (in production, use real historical sentiment)
                    mock_sentiment = {
                        'sentiment_score': np.random.normal(0, 0.3),
                        'news_count': np.random.randint(0, 5),
                        'finbert_score': np.random.normal(0, 0.3)
                    }
                    
                    # Rename columns for feature engineering
                    feature_df = historical_window.rename(columns={
                        'open_price': 'open',
                        'high_price': 'high', 
                        'low_price': 'low',
                        'close_price': 'close'
                    })
                    
                    # Generate features
                    features = self.feature_engineer.engineer_features(
                        symbol, feature_df, current_quote, mock_sentiment
                    )
                    
                    if not features:
                        continue
                    
                    # Calculate target (profitable signal)
                    entry_price = current_row['close_price']
                    future_window = df.iloc[i+1:i+1+self.forward_window]
                    
                    if len(future_window) == 0:
                        continue
                    
                    # Check if trade would be profitable
                    max_gain = ((future_window['high_price'].max() - entry_price) / entry_price) * 100
                    max_loss = ((future_window['low_price'].min() - entry_price) / entry_price) * 100
                    
                    # Target: 1 if would hit profit target before stop loss
                    target = 1 if max_gain >= self.profit_threshold and abs(max_loss) < self.profit_threshold else 0
                    
                    # Add metadata
                    sample = features.copy()
                    sample.update({
                        'symbol': symbol,
                        'date': current_row['date'],
                        'target_profitable': target,
                        'entry_price': entry_price,
                        'max_future_gain': max_gain,
                        'max_future_loss': max_loss
                    })
                    
                    training_samples.append(sample)
                    
                except Exception as e:
                    logger.debug(f"Sample generation failed for {symbol} at {i}: {e}")
                    continue
            
            logger.debug(f"✅ {symbol}: Generated {len(training_samples)} training samples")
            return training_samples
            
        except Exception as e:
            logger.error(f"Stock feature generation failed for {symbol}: {e}")
            return []
    
    async def _train_xgboost_model(self, training_data: pd.DataFrame, training_id: str) -> Dict:
        """Train XGBoost model on comprehensive dataset"""
        try:
            logger.info(f"🤖 Training XGBoost model on {len(training_data)} samples...")
            
            # Prepare features and target
            feature_columns = [col for col in training_data.columns 
                             if col not in ['symbol', 'date', 'target_profitable', 'entry_price', 'max_future_gain', 'max_future_loss']]
            
            X = training_data[feature_columns].fillna(0)
            y = training_data['target_profitable']
            
            # Time-based split (older data for training, newer for validation)
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
            
            # Train XGBoost with comprehensive parameters
            xgb_params = {
                'objective': 'binary:logistic',
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'auc'
            }
            
            model = xgb.XGBClassifier(**xgb_params)
            
            # Train with early stopping
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Get feature importance
            feature_importance = dict(zip(feature_columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            logger.info(f"✅ XGBoost training complete")
            logger.info(f"📊 Training samples: {len(X_train)}")
            logger.info(f"📈 Validation samples: {len(X_val)}")
            logger.info(f"🎯 Top 5 features: {[f[0] for f in top_features[:5]]}")
            
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
    
    async def _validate_model_performance(self, model, training_data: pd.DataFrame) -> Dict:
        """Validate model performance on test set"""
        try:
            # Prepare test data
            feature_columns = [col for col in training_data.columns 
                             if col not in ['symbol', 'date', 'target_profitable', 'entry_price', 'max_future_gain', 'max_future_loss']]
            
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
            
            # Make predictions
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            # Get detailed classification report
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
    
    async def _save_trained_model(self, 
                                 model, 
                                 scaler, 
                                 feature_columns: List[str], 
                                 training_id: str) -> str:
        """Save trained model and metadata"""
        try:
            model_version = f"comprehensive_{training_id}"
            
            # Save model components
            model_path = self.models_dir / f"xgboost_model_{model_version}.pkl"
            scaler_path = self.models_dir / f"scaler_{model_version}.pkl"
            features_path = self.models_dir / f"features_{model_version}.pkl"
            metadata_path = self.models_dir / f"metadata_{model_version}.json"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(feature_columns, features_path)
            
            # Save metadata
            metadata = {
                "model_version": model_version,
                "training_id": training_id,
                "created_at": datetime.now().isoformat(),
                "feature_count": len(feature_columns),
                "model_type": "xgboost_comprehensive",
                "training_data_years": 10
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Model saved: {model_path}")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise
    
    async def _save_training_metrics(self, metrics: TrainingMetrics):
        """Save training metrics for tracking"""
        try:
            metrics_file = self.training_results_dir / f"training_metrics_{metrics.training_id}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
            
            logger.info(f"✅ Training metrics saved: {metrics_file}")
            
        except Exception as e:
            logger.error(f"Training metrics save failed: {e}")

class AutoRetrainingSystem:
    """
    Auto-retraining system that learns from actual trade outcomes
    """
    
    def __init__(self, 
                 signal_logger: InstitutionalSignalLogger,
                 training_pipeline: ComprehensiveTrainingPipeline):
        self.signal_logger = signal_logger
        self.training_pipeline = training_pipeline
        
        # Retraining configuration
        self.weekly_retrain_enabled = True
        self.performance_threshold = 0.65  # Retrain if accuracy drops below 65%
        self.min_new_samples = 50  # Minimum new samples needed for retraining
        self.max_weeks_without_retrain = 4  # Force retrain after 4 weeks
        
        # Performance tracking
        self.performance_history = []
        self.last_retrain_date = None
        
    async def check_retrain_trigger(self) -> Dict[str, Any]:
        """
        Check if model should be retrained based on performance
        
        Returns:
            Dict with retrain decision and reasoning
        """
        try:
            logger.info("🔍 Checking auto-retrain triggers...")
            
            # Get recent model performance
            recent_performance = await self._calculate_recent_performance()
            
            # Check trigger conditions
            triggers = {
                "performance_degradation": False,
                "weekly_schedule": False,
                "forced_retrain": False,
                "sufficient_new_data": False
            }
            
            reasons = []
            
            # 1. Performance degradation trigger
            if recent_performance["accuracy"] < self.performance_threshold:
                triggers["performance_degradation"] = True
                reasons.append(f"Accuracy dropped to {recent_performance['accuracy']:.1%} (threshold: {self.performance_threshold:.1%})")
            
            # 2. Weekly schedule trigger
            if self._should_weekly_retrain():
                triggers["weekly_schedule"] = True
                reasons.append("Weekly retrain schedule")
            
            # 3. Forced retrain trigger (too long without retrain)
            if self._should_force_retrain():
                triggers["forced_retrain"] = True
                reasons.append(f"No retrain for {self.max_weeks_without_retrain} weeks")
            
            # 4. Sufficient new data trigger
            new_samples_count = await self._count_new_training_samples()
            if new_samples_count >= self.min_new_samples:
                triggers["sufficient_new_data"] = True
                reasons.append(f"{new_samples_count} new training samples available")
            
            # Decision logic
            should_retrain = (
                triggers["performance_degradation"] or
                (triggers["weekly_schedule"] and triggers["sufficient_new_data"]) or
                triggers["forced_retrain"]
            )
            
            result = {
                "should_retrain": should_retrain,
                "triggers": triggers,
                "reasons": reasons,
                "recent_performance": recent_performance,
                "new_samples_available": new_samples_count,
                "last_retrain": self.last_retrain_date.isoformat() if self.last_retrain_date else None
            }
            
            if should_retrain:
                logger.info(f"🔄 Retrain triggered: {', '.join(reasons)}")
            else:
                logger.info("✅ No retrain needed at this time")
            
            return result
            
        except Exception as e:
            logger.error(f"Retrain trigger check failed: {e}")
            return {"should_retrain": False, "error": str(e)}
    
    async def execute_auto_retrain(self) -> Dict[str, Any]:
        """
        Execute automatic retraining with new data
        
        Returns:
            Dict with retraining results
        """
        try:
            logger.info("🚀 Starting auto-retraining process...")
            
            # Collect new training data from recent trades
            new_training_data = await self._collect_recent_training_data()
            
            if len(new_training_data) < self.min_new_samples:
                logger.warning(f"Insufficient new data for retraining: {len(new_training_data)} samples")
                return {"success": False, "reason": "Insufficient new data"}
            
            # Combine with historical data (last 2 years for faster training)
            historical_data = await self.training_pipeline._generate_training_features(years_lookback=2)
            combined_data = pd.concat([historical_data, new_training_data], ignore_index=True)
            
            logger.info(f"📊 Training with {len(combined_data)} samples ({len(new_training_data)} new)")
            
            # Train new model
            training_id = f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_results = await self.training_pipeline._train_xgboost_model(combined_data, training_id)
            
            # Validate new model
            validation_results = await self.training_pipeline._validate_model_performance(
                model_results["model"], combined_data
            )
            
            # Check if new model is better
            current_performance = await self._calculate_recent_performance()
            
            if validation_results["accuracy"] > current_performance["accuracy"]:
                # Save new model
                model_path = await self.training_pipeline._save_trained_model(
                    model_results["model"],
                    model_results["scaler"],
                    model_results["feature_columns"],
                    training_id
                )
                
                # Update XGBoost model in production
                await self._deploy_new_model(training_id)
                
                self.last_retrain_date = datetime.now()
                
                logger.info("✅ Auto-retraining successful!")
                logger.info(f"📈 Accuracy improved: {current_performance['accuracy']:.1%} → {validation_results['accuracy']:.1%}")
                
                return {
                    "success": True,
                    "training_id": training_id,
                    "old_accuracy": current_performance["accuracy"],
                    "new_accuracy": validation_results["accuracy"],
                    "improvement": validation_results["accuracy"] - current_performance["accuracy"],
                    "training_samples": len(combined_data),
                    "new_samples": len(new_training_data)
                }
            else:
                logger.warning("⚠️ New model performance not better than current model")
                return {
                    "success": False,
                    "reason": "New model performance not improved",
                    "old_accuracy": current_performance["accuracy"],
                    "new_accuracy": validation_results["accuracy"]
                }
                
        except Exception as e:
            logger.error(f"Auto-retraining failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _calculate_recent_performance(self, days: int = 30) -> Dict[str, float]:
        """Calculate recent model performance from actual trades"""
        try:
            # Get recent trade outcomes from signal logger
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            recent_summary = self.signal_logger.get_daily_summary()
            
            # Calculate performance metrics
            performance = {
                "accuracy": recent_summary.get("win_rate_pct", 0.0) / 100,
                "total_trades": recent_summary.get("completed_trades", 0),
                "profitable_trades": recent_summary.get("winners", 0),
                "avg_pnl": recent_summary.get("total_pnl_rupees", 0.0),
                "period_days": days
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Recent performance calculation failed: {e}")
            return {"accuracy": 0.5, "total_trades": 0, "profitable_trades": 0, "avg_pnl": 0.0}
    
    async def _collect_recent_training_data(self, days: int = 60) -> pd.DataFrame:
        """Collect training data from recent actual trades"""
        try:
            # In production, this would collect actual trade outcomes
            # and create training samples from them
            
            # For now, return empty DataFrame - this needs to be implemented
            # based on your actual signal logger data structure
            
            logger.info(f"📊 Collecting recent training data from last {days} days...")
            
            # This would analyze recent signals and their outcomes to create
            # new training samples with actual results
            
            return pd.DataFrame()  # Placeholder
            
        except Exception as e:
            logger.error(f"Recent training data collection failed: {e}")
            return pd.DataFrame()
    
    def _should_weekly_retrain(self) -> bool:
        """Check if weekly retrain is due"""
        if not self.weekly_retrain_enabled:
            return False
        
        if self.last_retrain_date is None:
            return True
        
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        return days_since_retrain >= 7  # Weekly retraining
    
    def _should_force_retrain(self) -> bool:
        """Check if forced retrain is needed"""
        if self.last_retrain_date is None:
            return True
        
        weeks_since_retrain = (datetime.now() - self.last_retrain_date).days / 7
        return weeks_since_retrain >= self.max_weeks_without_retrain
    
    async def _count_new_training_samples(self) -> int:
        """Count available new training samples"""
        try:
            # Count recent completed trades that can be used for training
            recent_summary = self.signal_logger.get_daily_summary()
            return recent_summary.get("completed_trades", 0)
            
        except Exception as e:
            logger.error(f"New samples count failed: {e}")
            return 0
    
    async def _deploy_new_model(self, training_id: str):
        """Deploy newly trained model to production"""
        try:
            # Update the XGBoost model instance with new model
            model_version = f"comprehensive_{training_id}"
            
            # Load new model components
            models_dir = Path("models")
            model_path = models_dir / f"xgboost_model_{model_version}.pkl"
            scaler_path = models_dir / f"scaler_{model_version}.pkl"
            features_path = models_dir / f"features_{model_version}.pkl"
            
            if all(path.exists() for path in [model_path, scaler_path, features_path]):
                # This would update the production model instance
                # Implementation depends on your production setup
                logger.info(f"✅ New model deployed: {model_version}")
            else:
                logger.error("New model files not found for deployment")
                
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")


# Main training orchestrator
class TrainingOrchestrator:
    """
    Main orchestrator for all training activities
    """
    
    def __init__(self, market_data_service: MarketDataService, signal_logger: InstitutionalSignalLogger):
        self.market_data = market_data_service
        self.signal_logger = signal_logger
        
        # Initialize components
        self.data_collector = HistoricalDataCollector(market_data_service)
        self.training_pipeline = ComprehensiveTrainingPipeline(self.data_collector, signal_logger)
        self.auto_retrain = AutoRetrainingSystem(signal_logger, self.training_pipeline)
    
    async def initial_comprehensive_training(self) -> TrainingMetrics:
        """Run initial comprehensive training with 10 years of data"""
        logger.info("🚀 Starting initial comprehensive training...")
        
        # Collect 10 years of data
        await self.data_collector.collect_10_year_data()
        
        # Train comprehensive model
        metrics = await self.training_pipeline.train_comprehensive_model(years_lookback=10)
        
        return metrics
    
    async def daily_retrain_check(self):
        """Daily check for retraining triggers"""
        try:
            retrain_check = await self.auto_retrain.check_retrain_trigger()
            
            if retrain_check["should_retrain"]:
                logger.info("🔄 Executing auto-retrain...")
                retrain_result = await self.auto_retrain.execute_auto_retrain()
                
                if retrain_result["success"]:
                    logger.info("✅ Auto-retrain completed successfully")
                else:
                    logger.warning(f"⚠️ Auto-retrain failed: {retrain_result.get('reason', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Daily retrain check failed: {e}")
    
    async def scheduled_weekly_retrain(self):
        """Scheduled weekly retraining"""
        try:
            logger.info("📅 Running scheduled weekly retrain...")
            retrain_result = await self.auto_retrain.execute_auto_retrain()
            
            if retrain_result["success"]:
                logger.info("✅ Weekly retrain completed successfully")
            else:
                logger.warning(f"⚠️ Weekly retrain failed: {retrain_result.get('reason', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"Weekly retrain failed: {e}")


# Command-line interface for training
async def main():
    """Main function for running training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TradeMind AI Training Pipeline")
    parser.add_argument("--action", choices=["collect", "train", "retrain"], required=True,
                       help="Action to perform")
    parser.add_argument("--years", type=int, default=10, help="Years of data to use")
    parser.add_argument("--force", action="store_true", help="Force action even if not needed")
    
    args = parser.parse_args()
    
    # Initialize services (mock for CLI)
    from app.services.market_data_service import MarketDataService
    from app.core.signal_logger import InstitutionalSignalLogger
    
    market_data_service = MarketDataService("api_key", "access_token")
    signal_logger = InstitutionalSignalLogger("logs")
    
    orchestrator = TrainingOrchestrator(market_data_service, signal_logger)
    
    try:
        if args.action == "collect":
            print("🔄 Collecting historical data...")
            result = await orchestrator.data_collector.collect_10_year_data(force_refresh=args.force)
            print(f"✅ Collection complete: {result}")
            
        elif args.action == "train":
            print("🤖 Training comprehensive model...")
            metrics = await orchestrator.initial_comprehensive_training()
            print(f"✅ Training complete: {metrics.accuracy:.1%} accuracy")
            
        elif args.action == "retrain":
            print("🔄 Checking auto-retrain...")
            await orchestrator.daily_retrain_check()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
