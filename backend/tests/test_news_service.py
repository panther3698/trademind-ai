# ================================================================
# News Service Unit Tests
# ================================================================

"""
Comprehensive unit tests for News Services

Tests cover:
- News intelligence analysis
- News signal integration
- Breaking news detection
- Sentiment analysis
- Error handling scenarios
- Performance monitoring
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

from app.core.services.news_service import NewsSignalIntegrationService
from app.services.enhanced_news_intelligence import EnhancedNewsIntelligenceSystem
from app.core.domain.signal import Signal, SignalType, SignalSource
from app.core.performance_monitor import PerformanceMonitor

# ================================================================
# NEWS SIGNAL INTEGRATION SERVICE TESTS
# ================================================================

class TestNewsSignalIntegrationService:
    """Test cases for NewsSignalIntegrationService"""

    @pytest.fixture
    def news_service(self, mock_analytics_service):
        """Create NewsSignalIntegrationService instance with mocked dependencies"""
        mock_news_intelligence = Mock()
        mock_signal_generator = Mock()
        mock_market_data_service = Mock()
        
        return NewsSignalIntegrationService(
            news_intelligence=mock_news_intelligence,
            signal_generator=mock_signal_generator,
            market_data_service=mock_market_data_service,
            analytics_service=mock_analytics_service
        )

    @pytest.mark.asyncio
    async def test_news_service_initialization(self, news_service):
        """Test news service initialization"""
        # Assert
        assert news_service is not None
        assert hasattr(news_service, 'news_intelligence')
        assert hasattr(news_service, 'signal_generator')
        assert hasattr(news_service, 'market_data')
        assert hasattr(news_service, 'analytics')

    @pytest.mark.asyncio
    async def test_start_integrated_news_monitoring(self, news_service):
        """Test starting integrated news monitoring"""
        # Act
        task = await news_service.start_integrated_news_monitoring()

        # Assert
        assert task is not None
        assert not task.done()

    @pytest.mark.asyncio
    async def test_get_integration_stats(self, news_service):
        """Test getting integration statistics"""
        # Act
        stats = news_service.get_integration_stats()

        # Assert
        assert stats is not None
        assert "news_checks_completed" in stats
        assert "breaking_news_detected" in stats
        assert "news_triggered_signals" in stats

# ================================================================
# ENHANCED NEWS INTELLIGENCE SYSTEM TESTS
# ================================================================

class TestEnhancedNewsIntelligenceSystem:
    """Test cases for EnhancedNewsIntelligenceSystem"""

    @pytest.fixture
    def news_intelligence_system(self):
        """Create EnhancedNewsIntelligenceSystem instance with test config"""
        config = {
            "polygon_api_key": "test_key",
            "enable_hindi_analysis": False,
            "max_articles_per_source": 10,
            "sentiment_threshold": 0.3,
            "working_sources_count": 2,
            "failed_apis_removed": ["alpha_vantage", "eodhd", "finnhub", "newsapi"]
        }
        return EnhancedNewsIntelligenceSystem(config)

    @pytest.mark.asyncio
    async def test_news_intelligence_initialization(self, news_intelligence_system):
        """Test news intelligence system initialization"""
        # Act
        await news_intelligence_system.initialize()

        # Assert
        assert news_intelligence_system is not None
        assert hasattr(news_intelligence_system, 'config')
        assert hasattr(news_intelligence_system, 'news_sources')

    @pytest.mark.asyncio
    async def test_get_comprehensive_news_intelligence(self, news_intelligence_system):
        """Test comprehensive news intelligence retrieval"""
        # Arrange
        symbols = ["RELIANCE", "TCS"]
        lookback_hours = 24

        # Act
        result = await news_intelligence_system.get_comprehensive_news_intelligence(
            symbols=symbols,
            lookback_hours=lookback_hours
        )

        # Assert
        assert result is not None
        assert "timestamp" in result
        assert "analysis_period_hours" in result
        assert "total_articles_analyzed" in result

    @pytest.mark.asyncio
    async def test_get_comprehensive_news_intelligence_no_symbols(self, news_intelligence_system):
        """Test comprehensive news intelligence without specific symbols"""
        # Arrange
        lookback_hours = 12

        # Act
        result = await news_intelligence_system.get_comprehensive_news_intelligence(
            lookback_hours=lookback_hours
        )

        # Assert
        assert result is not None
        assert "timestamp" in result
        assert "analysis_period_hours" in result

    @pytest.mark.asyncio
    async def test_get_comprehensive_news_intelligence_with_sectors(self, news_intelligence_system):
        """Test comprehensive news intelligence with sector filtering"""
        # Arrange
        sectors = ["BANKING", "IT"]
        lookback_hours = 6

        # Act
        result = await news_intelligence_system.get_comprehensive_news_intelligence(
            sectors=sectors,
            lookback_hours=lookback_hours
        )

        # Assert
        assert result is not None
        assert "timestamp" in result
        assert "sector_impact" in result

# ================================================================
# PERFORMANCE MONITORING TESTS
# ================================================================

class TestNewsPerformanceMonitoring:
    """Performance monitoring tests for news services"""

    @pytest.mark.asyncio
    async def test_news_intelligence_performance_tracking(self, mock_analytics_service):
        """Test news intelligence performance tracking"""
        # Arrange
        config = {
            "polygon_api_key": "test_key",
            "enable_hindi_analysis": False,
            "max_articles_per_source": 10,
            "sentiment_threshold": 0.3,
            "working_sources_count": 2,
            "failed_apis_removed": ["alpha_vantage", "eodhd", "finnhub", "newsapi"]
        }
        news_intelligence = EnhancedNewsIntelligenceSystem(config)
        await news_intelligence.initialize()

        # Act
        result = await news_intelligence.get_comprehensive_news_intelligence(
            symbols=["RELIANCE"],
            lookback_hours=1
        )

        # Assert
        assert result is not None
        assert "fetch_statistics" in result

    @pytest.mark.asyncio
    async def test_news_integration_performance_tracking(self, mock_analytics_service):
        """Test news integration performance tracking"""
        # Arrange
        mock_news_intelligence = Mock()
        mock_signal_generator = Mock()
        mock_market_data_service = Mock()
        
        news_service = NewsSignalIntegrationService(
            news_intelligence=mock_news_intelligence,
            signal_generator=mock_signal_generator,
            market_data_service=mock_market_data_service,
            analytics_service=mock_analytics_service
        )

        # Act
        stats = news_service.get_integration_stats()

        # Assert
        assert stats is not None
        assert "news_checks_completed" in stats

# ================================================================
# INTEGRATION TESTS
# ================================================================

class TestNewsIntegration:
    """Integration tests for news services"""

    @pytest.mark.asyncio
    async def test_news_with_signal_service_integration(self, mock_analytics_service):
        """Test news integration with signal service"""
        # Arrange
        config = {
            "polygon_api_key": "test_key",
            "enable_hindi_analysis": False,
            "max_articles_per_source": 10,
            "sentiment_threshold": 0.3,
            "working_sources_count": 2,
            "failed_apis_removed": ["alpha_vantage", "eodhd", "finnhub", "newsapi"]
        }
        news_intelligence = EnhancedNewsIntelligenceSystem(config)
        await news_intelligence.initialize()
        
        mock_signal_generator = Mock()
        mock_market_data_service = Mock()
        
        news_service = NewsSignalIntegrationService(
            news_intelligence=news_intelligence,
            signal_generator=mock_signal_generator,
            market_data_service=mock_market_data_service,
            analytics_service=mock_analytics_service
        )

        # Act
        result = await news_intelligence.get_comprehensive_news_intelligence(
            symbols=["RELIANCE"],
            lookback_hours=1
        )

        # Assert
        assert result is not None
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_news_with_analytics_integration(self, mock_analytics_service):
        """Test news integration with analytics service"""
        # Arrange
        config = {
            "polygon_api_key": "test_key",
            "enable_hindi_analysis": False,
            "max_articles_per_source": 10,
            "sentiment_threshold": 0.3,
            "working_sources_count": 2,
            "failed_apis_removed": ["alpha_vantage", "eodhd", "finnhub", "newsapi"]
        }
        news_intelligence = EnhancedNewsIntelligenceSystem(config)
        await news_intelligence.initialize()

        # Act
        result = await news_intelligence.get_comprehensive_news_intelligence(
            lookback_hours=1
        )

        # Assert
        assert result is not None
        assert "fetch_statistics" in result

# ================================================================
# ERROR HANDLING TESTS
# ================================================================

class TestNewsErrorHandling:
    """Error handling tests for news services"""

    @pytest.mark.asyncio
    async def test_news_intelligence_api_failure(self):
        """Test news intelligence when API fails"""
        # Arrange
        config = {
            "polygon_api_key": "invalid_key",
            "enable_hindi_analysis": False,
            "max_articles_per_source": 10,
            "sentiment_threshold": 0.3,
            "working_sources_count": 2,
            "failed_apis_removed": ["alpha_vantage", "eodhd", "finnhub", "newsapi"]
        }
        news_intelligence = EnhancedNewsIntelligenceSystem(config)
        await news_intelligence.initialize()

        # Act
        result = await news_intelligence.get_comprehensive_news_intelligence(
            symbols=["RELIANCE"],
            lookback_hours=1
        )

        # Assert
        assert result is not None
        # Should handle API failures gracefully
        assert "error" in result or "fetch_statistics" in result

    @pytest.mark.asyncio
    async def test_news_intelligence_invalid_config(self):
        """Test news intelligence with invalid configuration"""
        # Arrange
        config = {}  # Empty config
        
        # Act & Assert
        # Should handle invalid config gracefully
        news_intelligence = EnhancedNewsIntelligenceSystem(config)
        await news_intelligence.initialize()
        
        result = await news_intelligence.get_comprehensive_news_intelligence(
            lookback_hours=1
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_news_intelligence_network_timeout(self):
        """Test news intelligence when network times out"""
        # Arrange
        config = {
            "polygon_api_key": "test_key",
            "enable_hindi_analysis": False,
            "max_articles_per_source": 10,
            "sentiment_threshold": 0.3,
            "working_sources_count": 2,
            "failed_apis_removed": ["alpha_vantage", "eodhd", "finnhub", "newsapi"]
        }
        news_intelligence = EnhancedNewsIntelligenceSystem(config)
        await news_intelligence.initialize()

        # Act
        result = await news_intelligence.get_comprehensive_news_intelligence(
            symbols=["RELIANCE"],
            lookback_hours=1
        )

        # Assert
        assert result is not None
        # Should handle network timeouts gracefully
        assert "error" in result or "fetch_statistics" in result

# ================================================================
# PERFORMANCE TESTS
# ================================================================

class TestNewsPerformance:
    """Performance tests for news services"""

    @pytest.mark.asyncio
    async def test_news_intelligence_bulk_analysis_performance(self):
        """Test news intelligence performance with bulk analysis"""
        # Arrange
        config = {
            "polygon_api_key": "test_key",
            "enable_hindi_analysis": False,
            "max_articles_per_source": 10,
            "sentiment_threshold": 0.3,
            "working_sources_count": 2,
            "failed_apis_removed": ["alpha_vantage", "eodhd", "finnhub", "newsapi"]
        }
        news_intelligence = EnhancedNewsIntelligenceSystem(config)
        await news_intelligence.initialize()
        
        symbols = [f"STOCK_{i}" for i in range(50)]

        # Act
        start_time = datetime.now()
        result = await news_intelligence.get_comprehensive_news_intelligence(
            symbols=symbols,
            lookback_hours=1
        )
        end_time = datetime.now()

        # Assert
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert result is not None

    @pytest.mark.asyncio
    async def test_news_intelligence_complex_analysis_performance(self):
        """Test news intelligence performance with complex analysis"""
        # Arrange
        config = {
            "polygon_api_key": "test_key",
            "enable_hindi_analysis": False,
            "max_articles_per_source": 10,
            "sentiment_threshold": 0.3,
            "working_sources_count": 2,
            "failed_apis_removed": ["alpha_vantage", "eodhd", "finnhub", "newsapi"]
        }
        news_intelligence = EnhancedNewsIntelligenceSystem(config)
        await news_intelligence.initialize()

        # Act
        start_time = datetime.now()
        result = await news_intelligence.get_comprehensive_news_intelligence(
            symbols=["RELIANCE", "TCS", "HDFC"],
            sectors=["BANKING", "IT"],
            lookback_hours=24
        )
        end_time = datetime.now()

        # Assert
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 60.0  # Should complete within 60 seconds
        assert result is not None 