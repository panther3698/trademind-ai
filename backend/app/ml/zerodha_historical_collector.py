# backend/app/ml/zerodha_historical_collector.py
"""
TradeMind AI - Zerodha Kite Connect Historical Data Collector
FIXED: Now handles 2000-day Kite Connect limit by chunking requests
"""

import asyncio
import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from collections import namedtuple
import yfinance as yf
import json

# Import your existing components
from app.services.zerodha_order_engine import ZerodhaOrderEngine
from app.ml.models import Nifty100StockUniverse
from app.core.config import settings

logger = logging.getLogger(__name__)

# OHLCV data structure
OHLCV = namedtuple('OHLCV', ['timestamp', 'open', 'high', 'low', 'close', 'volume'])

class ZerodhaHistoricalDataCollector:
    """
    Production-ready historical data collector using Zerodha Kite Connect API
    FIXED: Handles 2000-day Kite Connect limit by chunking requests
    """
    
    def __init__(self, zerodha_engine: ZerodhaOrderEngine = None):
        # Use existing Zerodha connection or create new one
        if zerodha_engine and zerodha_engine.is_connected:
            self.kite = zerodha_engine.kite
            self.is_connected = True
            logger.info("✅ Using existing Zerodha connection for data collection")
        else:
            # Create new connection for data collection
            self._initialize_kite_connection()
        
        self.stock_universe = Nifty100StockUniverse()
        
        # Database setup
        self.db_path = Path("data/historical_data.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        # Collection parameters
        self.years_lookback = 10
        self.batch_size = 20  # Kite allows good rate limits
        self.rate_limit_delay = 0.2  # 200ms between requests
        self.max_retries = 3
        
        # FIXED: Kite Connect limits
        self.kite_rate_limit = 300  # 300 requests per minute
        self.kite_max_days_per_request = 1800  # FIXED: Conservative limit to stay under 2000
        self.request_count = 0
        self.rate_limit_reset_time = datetime.now()
        
        # Symbol mapping for Kite Connect
        self.kite_symbol_map = self._create_kite_symbol_mapping()
    
    def _initialize_kite_connection(self):
        """Initialize Kite Connect for data collection"""
        try:
            if not settings.zerodha_api_key or not settings.zerodha_access_token:
                logger.info("⚠️ Zerodha credentials not configured - will use Yahoo Finance")
                self.is_connected = False
                self.kite = None
                return
            
            from kiteconnect import KiteConnect
            
            self.kite = KiteConnect(api_key=settings.zerodha_api_key)
            self.kite.set_access_token(settings.zerodha_access_token)
            
            # Test connection
            profile = self.kite.profile()
            self.is_connected = True
            
            logger.info(f"✅ Kite Connect initialized for data collection")
            logger.info(f"📊 User: {profile.get('user_name', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"⚠️ Kite Connect initialization failed: {e}")
            logger.info("🔄 Will use Yahoo Finance as data source")
            self.is_connected = False
            self.kite = None
    
    def _create_kite_symbol_mapping(self) -> Dict[str, str]:
        """Create mapping from stock symbols to Kite Connect instrument tokens"""
        symbol_map = {}
        
        # Common Nifty 100 symbols mapping
        nifty_symbols = {
            'RELIANCE': 'RELIANCE',
            'TCS': 'TCS', 
            'HDFCBANK': 'HDFCBANK',
            'BHARTIARTL': 'BHARTIARTL',
            'ICICIBANK': 'ICICIBANK',
            'SBIN': 'SBIN',
            'LICI': 'LICI',
            'ITC': 'ITC',
            'LT': 'LT',
            'KOTAKBANK': 'KOTAKBANK',
            'HINDUNILVR': 'HINDUNILVR',
            'BAJFINANCE': 'BAJFINANCE',
            'MARUTI': 'MARUTI',
            'ASIANPAINT': 'ASIANPAINT',
            'AXISBANK': 'AXISBANK',
            'NESTLEIND': 'NESTLEIND',
            'ULTRACEMCO': 'ULTRACEMCO',
            'DMART': 'DMART',
            'TITAN': 'TITAN',
            'WIPRO': 'WIPRO',
            'TECHM': 'TECHM',
            'SUNPHARMA': 'SUNPHARMA',
            'POWERGRID': 'POWERGRID',
            'NTPC': 'NTPC',
            'JSWSTEEL': 'JSWSTEEL',
            'TATAMOTORS': 'TATAMOTORS',
            'ONGC': 'ONGC',
            'COALINDIA': 'COALINDIA',
            'HCLTECH': 'HCLTECH',
            'BPCL': 'BPCL',
            'GRASIM': 'GRASIM',
            'INFY': 'INFY',
            'ADANIENT': 'ADANIENT',
            'TATACONSUM': 'TATACONSUM',
            'TATASTEEL': 'TATASTEEL',
            'CIPLA': 'CIPLA',
            'DRREDDY': 'DRREDDY',
            'APOLLOHOSP': 'APOLLOHOSP',
            'EICHERMOT': 'EICHERMOT',
            'BAJAJFINSV': 'BAJAJFINSV',
            'INDUSINDBK': 'INDUSINDBK',
            'BRITANNIA': 'BRITANNIA',
            'DIVISLAB': 'DIVISLAB',
            'HEROMOTOCO': 'HEROMOTOCO',
            'SHRIRAMFIN': 'SHRIRAMFIN',
            'ADANIPORTS': 'ADANIPORTS'
        }
        
        return nifty_symbols
    
    def _init_database(self):
        """Initialize database with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced OHLCV table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    data_source TEXT DEFAULT 'ZERODHA',
                    is_adjusted BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # Data quality tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    collection_date DATE NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    total_days_expected INTEGER,
                    actual_records_collected INTEGER,
                    data_completeness_pct REAL,
                    data_source TEXT,
                    collection_duration_seconds REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_date ON ohlcv_data(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_data(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv_data(date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality_symbol ON data_quality_log(symbol)')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Database initialized with enhanced schema")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def collect_10_year_data(self, 
                                 symbols: List[str] = None,
                                 force_refresh: bool = False,
                                 start_date: date = None,
                                 end_date: date = None) -> Dict[str, Any]:
        """
        Collect 10 years of historical data for specified symbols
        FIXED: Now handles Kite Connect 2000-day limit
        """
        try:
            # Set default parameters
            if symbols is None:
                symbols = self.stock_universe.get_all_stocks()  # ALL NIFTY 100 STOCKS
                logger.info(f"📊 Collecting data for ALL Nifty 100 stocks: {len(symbols)} symbols")
            
            if end_date is None:
                end_date = (datetime.now() - timedelta(days=1)).date()  # Yesterday
            
            if start_date is None:
                start_date = end_date - timedelta(days=self.years_lookback * 365)
            
            logger.info(f"🚀 Starting 10-year data collection with Kite limit handling")
            logger.info(f"📊 Symbols: {len(symbols)} stocks")
            logger.info(f"📅 Date range: {start_date} to {end_date}")
            logger.info(f"🔄 Force refresh: {force_refresh}")
            logger.info(f"⚠️ Kite limit: {self.kite_max_days_per_request} days per request")
            
            # Initialize collection stats
            collection_stats = {
                "start_time": datetime.now(),
                "symbols_total": len(symbols),
                "symbols_successful": 0,
                "symbols_failed": 0,
                "total_records_collected": 0,
                "data_sources_used": {"zerodha": 0, "yahoo": 0, "cached": 0, "failed": 0},
                "collection_errors": [],
                "data_quality_summary": {}
            }
            
            # Process symbols in batches
            for i in range(0, len(symbols), self.batch_size):
                batch = symbols[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (len(symbols) - 1) // self.batch_size + 1
                
                logger.info(f"📦 Processing batch {batch_num}/{total_batches}: {batch}")
                
                # Process batch
                batch_results = await self._process_batch(batch, start_date, end_date, force_refresh)
                
                # Update stats
                for result in batch_results:
                    if result["success"]:
                        collection_stats["symbols_successful"] += 1
                        collection_stats["total_records_collected"] += result["records_collected"]
                        collection_stats["data_sources_used"][result["data_source"]] += 1
                    else:
                        collection_stats["symbols_failed"] += 1
                        collection_stats["data_sources_used"]["failed"] += 1
                        collection_stats["collection_errors"].append({
                            "symbol": result["symbol"],
                            "error": result.get("error", "Unknown error")
                        })
                
                # Rate limiting between batches
                await asyncio.sleep(2)
                
                # Progress update
                progress = ((i + len(batch)) / len(symbols)) * 100
                logger.info(f"📈 Progress: {progress:.1f}% complete")
            
            # Finalize stats
            collection_stats["end_time"] = datetime.now()
            collection_stats["duration_minutes"] = (
                collection_stats["end_time"] - collection_stats["start_time"]
            ).total_seconds() / 60
            
            # Generate data quality summary
            collection_stats["data_quality_summary"] = await self._generate_quality_summary(symbols, start_date, end_date)
            
            # Log final results
            logger.info("🎉 Data collection completed!")
            logger.info(f"✅ Successful: {collection_stats['symbols_successful']}/{collection_stats['symbols_total']}")
            logger.info(f"📊 Total records: {collection_stats['total_records_collected']:,}")
            logger.info(f"⏱️ Duration: {collection_stats['duration_minutes']:.1f} minutes")
            logger.info(f"🔗 Data sources: Zerodha: {collection_stats['data_sources_used']['zerodha']}, Yahoo: {collection_stats['data_sources_used']['yahoo']}")
            
            if collection_stats["collection_errors"]:
                logger.warning(f"⚠️ {len(collection_stats['collection_errors'])} symbols failed")
                for error in collection_stats["collection_errors"][:5]:  # Show first 5 errors
                    logger.warning(f"   - {error['symbol']}: {error['error']}")
            
            return collection_stats
            
        except Exception as e:
            logger.error(f"❌ Data collection failed: {e}")
            raise
    
    async def _process_batch(self, 
                           symbols: List[str], 
                           start_date: date, 
                           end_date: date, 
                           force_refresh: bool) -> List[Dict[str, Any]]:
        """Process a batch of symbols"""
        batch_results = []
        
        for symbol in symbols:
            try:
                result = await self._collect_symbol_data(symbol, start_date, end_date, force_refresh)
                batch_results.append(result)
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"❌ Batch processing failed for {symbol}: {e}")
                batch_results.append({
                    "symbol": symbol,
                    "success": False,
                    "error": str(e),
                    "records_collected": 0,
                    "data_source": "failed"
                })
        
        return batch_results
    
    async def _collect_symbol_data(self, 
                                 symbol: str, 
                                 start_date: date, 
                                 end_date: date, 
                                 force_refresh: bool) -> Dict[str, Any]:
        """
        Collect historical data for a single symbol
        FIXED: Handles Kite Connect 2000-day limit by chunking requests
        """
        collection_start = datetime.now()
        
        try:
            # Check if we already have this data
            if not force_refresh:
                existing_data_info = self._check_existing_data(symbol, start_date, end_date)
                if existing_data_info["is_complete"]:
                    logger.info(f"📁 {symbol}: Using existing data ({existing_data_info['record_count']} records)")
                    return {
                        "symbol": symbol,
                        "success": True,
                        "records_collected": 0,  # No new records
                        "data_source": "cached",
                        "collection_duration": 0
                    }
            
            # Method 1: Try Zerodha Kite Connect (PRIMARY) with chunking
            historical_data = await self._fetch_kite_data_chunked(symbol, start_date, end_date)
            data_source = "zerodha"
            
            # Method 2: Fallback to Yahoo Finance
            if not historical_data and self._should_use_yahoo_fallback(symbol):
                historical_data = await self._fetch_yahoo_data(symbol, start_date, end_date)
                data_source = "yahoo"
            
            if not historical_data:
                raise ValueError(f"No data available from any source for {symbol}")
            
            # Validate and clean data
            validated_data = self._validate_ohlcv_data(historical_data, symbol)
            
            if not validated_data:
                raise ValueError(f"All data failed validation for {symbol}")
            
            # Store in database
            records_stored = await self._store_historical_data(symbol, validated_data, data_source)
            
            # Log quality metrics
            await self._log_data_quality(
                symbol, start_date, end_date, 
                len(validated_data), records_stored, 
                data_source, collection_start, True
            )
            
            collection_duration = (datetime.now() - collection_start).total_seconds()
            
            logger.info(f"✅ {symbol}: {records_stored} records from {data_source} ({collection_duration:.1f}s)")
            
            return {
                "symbol": symbol,
                "success": True,
                "records_collected": records_stored,
                "data_source": data_source,
                "collection_duration": collection_duration
            }
            
        except Exception as e:
            # Log failed collection
            await self._log_data_quality(
                symbol, start_date, end_date, 
                0, 0, "failed", collection_start, False, str(e)
            )
            
            logger.error(f"❌ {symbol}: Collection failed - {e}")
            
            return {
                "symbol": symbol,
                "success": False,
                "error": str(e),
                "records_collected": 0,
                "data_source": "failed",
                "collection_duration": (datetime.now() - collection_start).total_seconds()
            }
    
    async def _fetch_kite_data_chunked(self, symbol: str, start_date: date, end_date: date) -> List[OHLCV]:
        """
        FIXED: Fetch data from Zerodha Kite Connect API with chunking to handle 2000-day limit
        """
        if not self.is_connected:
            return []
        
        try:
            # Calculate total days
            total_days = (end_date - start_date).days
            
            if total_days <= self.kite_max_days_per_request:
                # Single request if within limit
                return await self._fetch_kite_data_single(symbol, start_date, end_date)
            else:
                # FIXED: Multiple chunked requests for longer periods
                logger.info(f"📊 {symbol}: Chunking {total_days} days into smaller requests...")
                
                all_data = []
                current_start = start_date
                
                while current_start < end_date:
                    # Calculate chunk end date
                    chunk_end = min(
                        current_start + timedelta(days=self.kite_max_days_per_request),
                        end_date
                    )
                    
                    logger.debug(f"📦 {symbol}: Fetching chunk {current_start} to {chunk_end}")
                    
                    # Fetch chunk data
                    chunk_data = await self._fetch_kite_data_single(symbol, current_start, chunk_end)
                    
                    if chunk_data:
                        all_data.extend(chunk_data)
                        logger.debug(f"✅ {symbol}: Got {len(chunk_data)} records for chunk")
                    else:
                        logger.warning(f"⚠️ {symbol}: No data for chunk {current_start} to {chunk_end}")
                    
                    # Move to next chunk
                    current_start = chunk_end + timedelta(days=1)
                    
                    # Rate limiting between chunks
                    await asyncio.sleep(self.rate_limit_delay)
                
                logger.info(f"📊 {symbol}: Collected {len(all_data)} total records from {total_days//self.kite_max_days_per_request + 1} chunks")
                return all_data
                
        except Exception as e:
            logger.warning(f"⚠️ Kite chunked data fetch failed for {symbol}: {e}")
            return []
    
    async def _fetch_kite_data_single(self, symbol: str, start_date: date, end_date: date) -> List[OHLCV]:
        """Fetch data from Kite Connect for a single date range (under 2000 days)"""
        try:
            # Check rate limits
            await self._check_rate_limits()
            
            # Get Kite symbol
            kite_symbol = self.kite_symbol_map.get(symbol, symbol)
            
            logger.debug(f"📡 Fetching Kite data for {symbol} ({kite_symbol}) from {start_date} to {end_date}")
            
            # Get instrument token
            try:
                # Get all instruments (cached by kite)
                instruments = self.kite.instruments("NSE")
                
                # Find the instrument token for our symbol
                instrument_token = None
                for instrument in instruments:
                    if instrument['tradingsymbol'] == symbol:
                        instrument_token = instrument['instrument_token']
                        break
                
                if not instrument_token:
                    logger.warning(f"⚠️ Instrument token not found for {symbol}")
                    return []
                
                # Fetch historical data using proper instrument token
                historical_records = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=start_date,
                    to_date=end_date,
                    interval="day"
                )
                
            except Exception as kite_error:
                logger.warning(f"⚠️ Kite API call failed for {symbol}: {kite_error}")
                return []
            
            # Convert to OHLCV format
            ohlcv_data = []
            for record in historical_records:
                try:
                    ohlcv = OHLCV(
                        timestamp=pd.to_datetime(record['date']),
                        open=float(record['open']),
                        high=float(record['high']),
                        low=float(record['low']),
                        close=float(record['close']),
                        volume=int(record['volume'])
                    )
                    ohlcv_data.append(ohlcv)
                except (KeyError, ValueError) as e:
                    logger.warning(f"⚠️ Invalid Kite record for {symbol}: {e}")
                    continue
            
            self.request_count += 1
            logger.debug(f"📊 Kite: {len(ohlcv_data)} records for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            logger.warning(f"⚠️ Kite single data fetch failed for {symbol}: {e}")
            return []
    
    async def _fetch_yahoo_data(self, symbol: str, start_date: date, end_date: date) -> List[OHLCV]:
        """Fetch data from Yahoo Finance as fallback"""
        try:
            # Convert to Yahoo symbol format
            yahoo_symbol = f"{symbol}.NS"  # NSE format
            
            logger.debug(f"📡 Fetching Yahoo data for {symbol} ({yahoo_symbol})")
            
            # Download data
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                # Try BSE format
                yahoo_symbol = f"{symbol}.BO"
                ticker = yf.Ticker(yahoo_symbol)
                data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                return []
            
            # Convert to OHLCV format
            ohlcv_data = []
            for date_idx, row in data.iterrows():
                try:
                    ohlcv = OHLCV(
                        timestamp=pd.to_datetime(date_idx),
                        open=round(float(row['Open']), 2),
                        high=round(float(row['High']), 2),
                        low=round(float(row['Low']), 2),
                        close=round(float(row['Close']), 2),
                        volume=int(row['Volume'])
                    )
                    ohlcv_data.append(ohlcv)
                except (ValueError, KeyError) as e:
                    logger.warning(f"⚠️ Invalid Yahoo record for {symbol}: {e}")
                    continue
            
            logger.info(f"📊 Yahoo: {len(ohlcv_data)} records for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            logger.warning(f"⚠️ Yahoo data fetch failed for {symbol}: {e}")
            return []
    
    def _should_use_yahoo_fallback(self, symbol: str) -> bool:
        """Determine if Yahoo Finance should be used as fallback"""
        # Use Yahoo for all symbols if Kite fails
        return True
    
    async def _check_rate_limits(self):
        """Check and enforce Kite Connect rate limits"""
        now = datetime.now()
        
        # Reset counter every minute
        if now - self.rate_limit_reset_time > timedelta(minutes=1):
            self.request_count = 0
            self.rate_limit_reset_time = now
        
        # Wait if we're approaching rate limit
        if self.request_count >= self.kite_rate_limit - 10:
            wait_time = 60 - (now - self.rate_limit_reset_time).seconds
            if wait_time > 0:
                logger.info(f"⏳ Rate limit reached, waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.rate_limit_reset_time = datetime.now()
    
    def _check_existing_data(self, symbol: str, start_date: date, end_date: date) -> Dict[str, Any]:
        """Check if we already have data for this symbol and date range"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*), MIN(date), MAX(date)
                FROM ohlcv_data 
                WHERE symbol = ? AND date >= ? AND date <= ?
            ''', (symbol, start_date, end_date))
            
            result = cursor.fetchone()
            record_count = result[0] if result else 0
            min_date = result[1] if result else None
            max_date = result[2] if result else None
            
            conn.close()
            
            # Calculate expected trading days (rough estimate: 250 days per year)
            expected_days = (end_date - start_date).days * 0.7  # Account for weekends/holidays
            completeness = (record_count / expected_days) if expected_days > 0 else 0
            
            return {
                "record_count": record_count,
                "min_date": min_date,
                "max_date": max_date,
                "completeness": completeness,
                "is_complete": completeness >= 0.8  # 80% completeness threshold
            }
            
        except Exception as e:
            logger.error(f"❌ Error checking existing data for {symbol}: {e}")
            return {"record_count": 0, "is_complete": False, "completeness": 0}
    
    def _validate_ohlcv_data(self, ohlcv_data: List[OHLCV], symbol: str) -> List[OHLCV]:
        """Validate and clean OHLCV data"""
        validated_data = []
        
        for i, ohlcv in enumerate(ohlcv_data):
            try:
                # Basic validation
                if any(price <= 0 for price in [ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close]):
                    logger.debug(f"⚠️ {symbol}: Invalid prices on {ohlcv.timestamp.date()}")
                    continue
                
                if ohlcv.volume < 0:
                    logger.debug(f"⚠️ {symbol}: Negative volume on {ohlcv.timestamp.date()}")
                    continue
                
                # OHLC relationship validation
                if not (ohlcv.low <= ohlcv.open <= ohlcv.high and 
                       ohlcv.low <= ohlcv.close <= ohlcv.high):
                    logger.debug(f"⚠️ {symbol}: Invalid OHLC relationship on {ohlcv.timestamp.date()}")
                    continue
                
                # Price spike detection (compared to previous day)
                if validated_data:
                    prev_close = validated_data[-1].close
                    price_change = abs(ohlcv.open - prev_close) / prev_close
                    
                    if price_change > 0.5:  # More than 50% change
                        logger.info(f"📊 {symbol}: Large price change on {ohlcv.timestamp.date()}: {price_change:.1%}")
                        # Don't skip - might be legitimate (stock split, etc.)
                
                validated_data.append(ohlcv)
                
            except Exception as e:
                logger.warning(f"⚠️ {symbol}: Validation error for record {i}: {e}")
                continue
        
        validation_rate = len(validated_data) / len(ohlcv_data) if ohlcv_data else 0
        
        if validation_rate < 0.8:
            logger.warning(f"⚠️ {symbol}: Low validation rate: {validation_rate:.1%}")
        else:
            logger.debug(f"✅ {symbol}: Validation rate: {validation_rate:.1%}")
        
        return validated_data
    
    async def _store_historical_data(self, symbol: str, ohlcv_data: List[OHLCV], data_source: str) -> int:
        """Store validated OHLCV data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            records_stored = 0
            
            for ohlcv in ohlcv_data:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO ohlcv_data 
                        (symbol, date, open_price, high_price, low_price, close_price, volume, data_source, is_adjusted)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        ohlcv.timestamp.date(),
                        ohlcv.open,
                        ohlcv.high,
                        ohlcv.low,
                        ohlcv.close,
                        ohlcv.volume,
                        data_source,
                        1  # Zerodha data is pre-adjusted
                    ))
                    records_stored += 1
                    
                except sqlite3.Error as e:
                    logger.warning(f"⚠️ {symbol}: Database error for {ohlcv.timestamp.date()}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            return records_stored
            
        except Exception as e:
            logger.error(f"❌ {symbol}: Data storage failed: {e}")
            return 0
    
    async def _log_data_quality(self, symbol: str, start_date: date, end_date: date, 
                              expected_records: int, actual_records: int, 
                              data_source: str, collection_start: datetime, 
                              success: bool, error_message: str = None):
        """Log data quality metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            expected_days = (end_date - start_date).days
            completeness = (actual_records / expected_days) * 100 if expected_days > 0 else 0
            duration = (datetime.now() - collection_start).total_seconds()
            
            cursor.execute('''
                INSERT INTO data_quality_log 
                (symbol, collection_date, start_date, end_date, total_days_expected, 
                 actual_records_collected, data_completeness_pct, data_source, 
                 collection_duration_seconds, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, datetime.now().date(), start_date, end_date, expected_days,
                actual_records, completeness, data_source, duration, success, error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.debug(f"Quality logging failed for {symbol}: {e}")
    
    async def _generate_quality_summary(self, symbols: List[str], start_date: date, end_date: date) -> Dict[str, Any]:
        """Generate data quality summary report"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Overall statistics
            overall_stats = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as total_symbols,
                    AVG(data_completeness_pct) as avg_completeness,
                    SUM(actual_records_collected) as total_records,
                    AVG(collection_duration_seconds) as avg_collection_time,
                    data_source,
                    COUNT(*) as source_count
                FROM data_quality_log 
                WHERE collection_date = date('now')
                GROUP BY data_source
            ''', conn)
            
            # Symbol-level quality
            symbol_quality = pd.read_sql_query('''
                SELECT symbol, data_completeness_pct, actual_records_collected, data_source
                FROM data_quality_log 
                WHERE collection_date = date('now')
                ORDER BY data_completeness_pct DESC
            ''', conn)
            
            conn.close()
            
            return {
                "overall_stats": overall_stats.to_dict('records') if not overall_stats.empty else [],
                "top_quality_symbols": symbol_quality.head(10).to_dict('records') if not symbol_quality.empty else [],
                "low_quality_symbols": symbol_quality.tail(5).to_dict('records') if not symbol_quality.empty else [],
                "avg_completeness": symbol_quality['data_completeness_pct'].mean() if not symbol_quality.empty else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Quality summary generation failed: {e}")
            return {"error": str(e)}
    
    def get_available_data_summary(self) -> Dict[str, Any]:
        """Get summary of currently available data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            summary = pd.read_sql_query('''
                SELECT 
                    symbol,
                    COUNT(*) as record_count,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    data_source,
                    MAX(created_at) as last_updated
                FROM ohlcv_data 
                GROUP BY symbol
                ORDER BY record_count DESC
            ''', conn)
            
            total_records = pd.read_sql_query('SELECT COUNT(*) as total FROM ohlcv_data', conn)
            
            conn.close()
            
            return {
                "total_records": total_records.iloc[0]['total'] if not total_records.empty else 0,
                "unique_symbols": len(summary),
                "symbol_summary": summary.to_dict('records') if not summary.empty else [],
                "date_range": {
                    "earliest": summary['start_date'].min() if not summary.empty else None,
                    "latest": summary['end_date'].max() if not summary.empty else None
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Data summary generation failed: {e}")
            return {"error": str(e)}

    async def get_historical_data(self, symbol: str, interval: str, start_date: Optional[date], end_date: Optional[date]):
        """Fetch historical OHLCV data for a symbol from the DB, with enhanced debug logging."""
        import pandas as pd
        import sqlite3
        db_path = str(self.db_path)
        logger.info(f"[DEBUG] get_historical_data: symbol={symbol}, interval={interval}, start_date={start_date}, end_date={end_date}, db_path={db_path}")
        try:
            conn = sqlite3.connect(db_path)
            query = '''
                SELECT date, open_price, high_price, low_price, close_price, volume
                FROM ohlcv_data
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            '''
            df = pd.read_sql_query(query, conn, params=(symbol, str(start_date), str(end_date)))
            conn.close()
            logger.info(f"[DEBUG] get_historical_data: {symbol} returned type={type(df)}, len={len(df)}")
            if not df.empty:
                logger.info(f"[DEBUG] get_historical_data: {symbol} head=\n{df.head(3)}")
                return df
            else:
                # Log available symbols in DB for diagnosis
                conn = sqlite3.connect(db_path)
                symbols_df = pd.read_sql_query('SELECT DISTINCT symbol FROM ohlcv_data', conn)
                conn.close()
                logger.warning(f"[DEBUG] get_historical_data: No data for {symbol}. Available symbols in DB: {symbols_df['symbol'].tolist()}")
                return None
        except Exception as e:
            logger.error(f"[DEBUG] get_historical_data: Exception for {symbol}: {e}")
            return None


# ================================================================
# INTEGRATION WITH EXISTING TRAINING PIPELINE
# ================================================================

async def replace_mock_data_with_real_data():
    """
    This function replaces the mock data generation in training_pipeline.py
    Run this to start collecting real data with proper chunking
    """
    try:
        logger.info("🚀 Starting real data collection with Kite limit handling...")
        
        # Initialize collector
        collector = ZerodhaHistoricalDataCollector()
        
        if not collector.is_connected:
            logger.warning("⚠️ Cannot connect to Zerodha - using Yahoo Finance only")
        
        # Start with a small subset for testing
        test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'SBIN']
        
        logger.info(f"📊 Testing with {len(test_symbols)} symbols first...")
        
        # Collect data with chunking
        results = await collector.collect_10_year_data(
            symbols=test_symbols,
            force_refresh=False  # Don't refresh if data exists
        )
        
        # Report results
        if results["symbols_successful"] > 0:
            logger.info("✅ Real data collection successful with chunking!")
            logger.info(f"📊 Collected {results['total_records_collected']:,} records")
            
            # Show data summary
            summary = collector.get_available_data_summary()
            logger.info(f"📈 Database now contains {summary['total_records']:,} total records")
            logger.info(f"🎯 Covering {summary['unique_symbols']} unique symbols")
            
            return True
        else:
            logger.error("❌ Real data collection failed for all symbols")
            return False
            
    except Exception as e:
        logger.error(f"❌ Real data collection setup failed: {e}")
        return False


# ================================================================
# USAGE EXAMPLE
# ================================================================

if __name__ == "__main__":
    async def main():
        # Replace mock data with real Zerodha data (with chunking)
        success = await replace_mock_data_with_real_data()
        
        if success:
            print("🎉 Successfully collected real historical data with Kite limit handling!")
            print("📊 Your models can now train on actual market data from the last 10 years!")
            print("🚀 Next: Run your training pipeline to see the difference!")
        else:
            print("❌ Data collection failed. Using Yahoo Finance fallback.")
    
    asyncio.run(main())