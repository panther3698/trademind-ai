# ================================================================
# Pytest Configuration and Fixtures for TradeMind AI Tests
# ================================================================

"""
Pytest configuration and shared fixtures for TradeMind AI unit tests

This file provides:
- Test configuration
- Mock fixtures for external dependencies
- Sample data fixtures
- Service setup fixtures
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.config import Settings
from app.core.services.analytics_service import CorrectedAnalytics
from app.core.services.signal_service import SignalService
from app.core.services.news_service import NewsSignalIntegrationService
from app.core.services.notification_service import NotificationService
from app.services.production_signal_generator import ProductionMLSignalGenerator
from app.services.enhanced_news_intelligence import EnhancedNewsIntelligenceSystem
from app.services.zerodha_order_engine import ZerodhaOrderEngine
from app.core.performance_monitor import PerformanceMonitor

# ================================================================
# TEST CONFIGURATION
# ================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_settings():
    """Test configuration settings"""
    return Settings(
        environment="test",
        debug=True,
        secret_key="test-secret-key",
        database_url="sqlite:///./test.db",
        redis_url="redis://localhost:6379",
        telegram_bot_token="test-bot-token",
        telegram_chat_id="test-chat-id",
        zerodha_api_key="test-api-key",
        zerodha_secret="test-secret",
        zerodha_access_token="test-access-token",
        news_api_key="test-news-api-key",
        polygon_api_key="test-polygon-api-key",
        alpha_vantage_api_key="test-alpha-vantage-key",
        finnhub_api_key="test-finnhub-key"
    )

# ================================================================
# MOCK FIXTURES
# ================================================================

@pytest.fixture
def mock_kite_connect():
    """Mock Kite Connect API"""
    mock_kite = Mock()
    mock_kite.margins = Mock(return_value={
        "equity": {
            "available": {
                "cash": 50000.0,
                "net": 50000.0
            }
        }
    })
    mock_kite.place_order = Mock(return_value="test-order-id")
    mock_kite.orders = Mock(return_value=[{
        "order_id": "test-order-id",
        "status": "COMPLETE",
        "tradingsymbol": "RELIANCE",
        "quantity": 10,
        "price": 2500.0
    }])
    mock_kite.cancel_order = Mock(return_value=True)
    mock_kite.modify_order = Mock(return_value=True)
    return mock_kite

@pytest.fixture
def mock_telegram_bot():
    """Mock Telegram Bot API"""
    mock_bot = Mock()
    mock_bot.send_message = AsyncMock(return_value={"message_id": 123})
    mock_bot.send_photo = AsyncMock(return_value={"message_id": 124})
    mock_bot.answer_callback_query = AsyncMock(return_value=True)
    return mock_bot

@pytest.fixture
def mock_news_api():
    """Mock News API responses"""
    return {
        "status": "ok",
        "totalResults": 10,
        "articles": [
            {
                "source": {"name": "Test News"},
                "author": "Test Author",
                "title": "Test News Title",
                "description": "Test news description",
                "url": "https://test.com/news",
                "urlToImage": "https://test.com/image.jpg",
                "publishedAt": "2025-06-29T10:00:00Z",
                "content": "Test news content for sentiment analysis."
            }
        ]
    }

@pytest.fixture
def mock_polygon_api():
    """Mock Polygon API responses"""
    return {
        "results": [
            {
                "id": "test-article-id",
                "publisher": {"name": "Test Publisher"},
                "title": "Test Polygon News",
                "author": "Test Author",
                "published_utc": "2025-06-29T10:00:00Z",
                "article_url": "https://test.com/polygon-news",
                "tickers": ["RELIANCE", "TCS"],
                "description": "Test polygon news content",
                "content": "Test polygon news content for analysis."
            }
        ]
    }

@pytest.fixture
def mock_database():
    """Mock database operations"""
    mock_db = Mock()
    mock_db.execute = AsyncMock()
    mock_db.fetch_all = AsyncMock(return_value=[])
    mock_db.fetch_one = AsyncMock(return_value=None)
    mock_db.commit = AsyncMock()
    return mock_db

# ================================================================
# SAMPLE DATA FIXTURES
# ================================================================

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return {
        "symbol": "RELIANCE",
        "price": 2500.0,
        "volume": 1000000,
        "high": 2550.0,
        "low": 2450.0,
        "open": 2480.0,
        "close": 2500.0,
        "change": 20.0,
        "change_percent": 0.8,
        "timestamp": datetime.now().isoformat(),
        "indicators": {
            "rsi": 65.5,
            "macd": 15.2,
            "bollinger_upper": 2600.0,
            "bollinger_lower": 2400.0,
            "sma_20": 2480.0,
            "sma_50": 2450.0
        }
    }

@pytest.fixture
def sample_news_data():
    """Sample news data for testing"""
    return {
        "articles": [
            {
                "title": "Reliance Industries Reports Strong Q4 Results",
                "description": "Reliance Industries reported better-than-expected quarterly results",
                "content": "Reliance Industries has reported strong quarterly results with revenue growth of 15%",
                "sentiment": 0.8,
                "relevance": 0.9,
                "published_at": datetime.now().isoformat(),
                "source": "Economic Times"
            },
            {
                "title": "Market Volatility Expected Due to Global Events",
                "description": "Global market volatility expected to impact Indian markets",
                "content": "Global economic uncertainty may lead to increased market volatility",
                "sentiment": -0.3,
                "relevance": 0.7,
                "published_at": datetime.now().isoformat(),
                "source": "Business Standard"
            }
        ],
        "overall_sentiment": 0.25,
        "breaking_news_count": 1,
        "total_articles": 2
    }

@pytest.fixture
def sample_signal_data():
    """Sample signal data for testing"""
    return {
        "symbol": "RELIANCE",
        "signal_type": "BUY",
        "confidence": 0.85,
        "price": 2500.0,
        "target_price": 2600.0,
        "stop_loss": 2400.0,
        "quantity": 10,
        "reasoning": "Strong technical indicators and positive news sentiment",
        "timestamp": datetime.now().isoformat(),
        "source": "ML_MODEL",
        "metadata": {
            "model_accuracy": 0.75,
            "news_sentiment": 0.8,
            "technical_score": 0.9
        }
    }

@pytest.fixture
def sample_order_data():
    """Sample order data for testing"""
    return {
        "symbol": "RELIANCE",
        "order_type": "BUY",
        "quantity": 10,
        "price": 2500.0,
        "order_id": "test-order-123",
        "status": "PENDING",
        "timestamp": datetime.now().isoformat(),
        "signal_id": "signal-456"
    }

# ================================================================
# SERVICE FIXTURES
# ================================================================

@pytest.fixture
def mock_analytics_service():
    """Mock analytics service"""
    mock_analytics = Mock(spec=CorrectedAnalytics)
    mock_analytics.get_daily_stats = Mock(return_value={
        "signals_generated": 25,
        "signals_executed": 20,
        "successful_trades": 15,
        "total_pnl": 5000.0,
        "win_rate": 0.75,
        "news_articles_processed": 100,
        "breaking_news_alerts": 5
    })
    mock_analytics.track_signal = AsyncMock()
    mock_analytics.track_order = AsyncMock()
    mock_analytics.track_trade_outcome = AsyncMock()
    mock_analytics.track_signal_generated = AsyncMock()
    mock_analytics.track_telegram_sent = AsyncMock()
    mock_analytics.track_news_triggered_signal = AsyncMock()
    mock_analytics.track_enhanced_ml_signal = AsyncMock()
    mock_analytics.track_signal_approval = AsyncMock()
    mock_analytics.track_order_execution = AsyncMock()
    mock_analytics.track_news_processed = AsyncMock()
    mock_analytics.track_breaking_news = AsyncMock()
    mock_analytics.track_news_signal = AsyncMock()
    mock_analytics.track_premarket_analysis = AsyncMock()
    mock_analytics.get_performance_summary = Mock(return_value={
        "daily": {"signals_generated": 25, "win_rate": 0.75},
        "system": {"uptime_hours": 24.5, "is_operational": True},
        "enhanced_features": {"news_signal_integration": True}
    })
    return mock_analytics

@pytest.fixture
def mock_signal_service():
    """Mock signal service"""
    mock_signal = Mock(spec=SignalService)
    mock_signal.generate_signals = AsyncMock(return_value=[])
    mock_signal.process_signal = AsyncMock()
    mock_signal.start_signal_generation = AsyncMock()
    mock_signal.stop_signal_generation = AsyncMock()
    mock_signal.convert_signal_record_to_dict = Mock(return_value={
        "symbol": "RELIANCE",
        "action": "BUY",
        "entry_price": 2500.0,
        "target_price": 2600.0,
        "stop_loss": 2400.0,
        "confidence": 0.85,
        "timestamp": datetime.now().isoformat()
    })
    return mock_signal

@pytest.fixture
def mock_news_service():
    """Mock news service"""
    mock_news = Mock(spec=EnhancedNewsIntelligenceSystem)
    mock_news.analyze_news = AsyncMock(return_value={
        "sentiment": 0.8,
        "articles": [],
        "breaking_news": []
    })
    mock_news.start_monitoring = AsyncMock()
    mock_news.stop_monitoring = AsyncMock()
    mock_news.get_news_summary = Mock(return_value={
        "total_articles": 100,
        "average_sentiment": 0.6,
        "breaking_news_count": 5
    })
    return mock_news

@pytest.fixture
def mock_notification_service():
    """Mock notification service"""
    mock_notification = Mock(spec=NotificationService)
    mock_notification.send_signal_notification = AsyncMock(return_value=True)
    mock_notification.send_order_notification = AsyncMock(return_value=True)
    mock_notification.send_telegram_message = AsyncMock(return_value=True)
    mock_notification.send_breaking_news_notification = AsyncMock(return_value=True)
    mock_notification.send_system_alert_notification = AsyncMock(return_value=True)
    mock_notification.format_signal_message = Mock(return_value="Test signal message")
    mock_notification.format_order_message = Mock(return_value="Test order message")
    mock_notification.format_news_message = Mock(return_value="Test news message")
    mock_notification.get_telegram_service = Mock(return_value=Mock())
    mock_notification.get_order_engine = Mock(return_value=Mock())
    mock_notification.is_interactive_trading_active = Mock(return_value=False)
    return mock_notification

@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor"""
    mock_monitor = Mock(spec=PerformanceMonitor)
    mock_monitor.timing = Mock()
    mock_monitor.async_timing = Mock()
    mock_monitor.track_success = Mock()
    mock_monitor.increment_counter = Mock()
    mock_monitor.add_metric = Mock()
    return mock_monitor

# ================================================================
# INTEGRATION TEST FIXTURES
# ================================================================

@pytest.fixture
def mock_service_manager():
    """Mock service manager for integration tests"""
    mock_manager = Mock()
    mock_manager.analytics_service = mock_analytics_service()
    mock_manager.signal_service = mock_signal_service()
    mock_manager.news_intelligence = mock_news_service()
    mock_manager.notification_service = mock_notification_service()
    mock_manager.get_system_health = Mock(return_value={
        "signal_generation": True,
        "news_intelligence": True,
        "telegram": True,
        "order_execution": True
    })
    return mock_manager

# ================================================================
# TEST UTILITIES
# ================================================================

def create_mock_response(status_code: int, data: Dict[str, Any]) -> Mock:
    """Create a mock HTTP response"""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json = Mock(return_value=data)
    mock_response.text = Mock(return_value=json.dumps(data))
    return mock_response

def create_mock_exception(exception_type: type, message: str) -> Exception:
    """Create a mock exception"""
    return exception_type(message)

# ================================================================
# TEST CONFIGURATION
# ================================================================

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        if "test_" in item.nodeid:
            item.add_marker(pytest.mark.unit) 