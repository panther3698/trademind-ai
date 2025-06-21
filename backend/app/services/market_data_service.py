# backend/app/services/market_data_service.py
"""
TradeMind AI - Compatibility Layer for Legacy Imports
Redirects legacy MarketDataService imports to Enhanced Service
"""

import logging
from typing import Dict, List, Optional, Any

# Import the enhanced service and alias it
from app.services.enhanced_market_data_nifty100 import (
    EnhancedMarketDataService,
    MarketTick,
    MarketStatus,
    TradingMode,
    Nifty100Universe
)

logger = logging.getLogger(__name__)

# Compatibility alias - redirect legacy imports to enhanced service
MarketDataService = EnhancedMarketDataService

# Export everything that legacy code might expect
__all__ = [
    'MarketDataService',
    'MarketTick', 
    'MarketStatus',
    'TradingMode',
    'Nifty100Universe'
]

logger.info("✅ Legacy MarketDataService imports redirected to Enhanced Service")
