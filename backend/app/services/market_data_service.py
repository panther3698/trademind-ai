# backend/app/services/market_data_service.py
"""
TradeMind AI - Compatibility Layer for Legacy Imports
Redirects legacy MarketDataService imports to Enhanced Service
FIXED: Added proper import guards to prevent reimport loops
"""

import sys
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

# FIXED: Prevent reimport loops with proper import guards
if TYPE_CHECKING:
    # Type-only imports for IDE support
    from app.services.enhanced_market_data_nifty100 import (
        EnhancedMarketDataService,
        MarketTick,
        MarketStatus,
        TradingMode,
        Nifty100Universe
    )

# FIXED: Runtime import with loop prevention
try:
    # Check if enhanced module is already imported to prevent loops
    if 'app.services.enhanced_market_data_nifty100' not in sys.modules:
        # First-time import - safe to proceed
        from app.services.enhanced_market_data_nifty100 import (
            EnhancedMarketDataService,
            MarketTick,
            MarketStatus,
            TradingMode,
            Nifty100Universe
        )
        logger.info("✅ Enhanced market data service imported successfully")
    else:
        # Module already loaded - get from sys.modules to prevent reimport
        enhanced_module = sys.modules['app.services.enhanced_market_data_nifty100']
        EnhancedMarketDataService = getattr(enhanced_module, 'EnhancedMarketDataService')
        MarketTick = getattr(enhanced_module, 'MarketTick')
        MarketStatus = getattr(enhanced_module, 'MarketStatus')
        TradingMode = getattr(enhanced_module, 'TradingMode')
        Nifty100Universe = getattr(enhanced_module, 'Nifty100Universe')
        logger.info("✅ Enhanced market data service loaded from existing module")
    
    # Compatibility alias - redirect legacy imports to enhanced service
    MarketDataService = EnhancedMarketDataService
    
    # FIXED: Mark successful import
    _IMPORT_SUCCESS = True
    
except ImportError as e:
    logger.error(f"❌ Failed to import enhanced market data service: {e}")
    logger.warning("⚠️ Creating fallback placeholder for compatibility")
    
    # FIXED: Create fallback classes to prevent import failures
    class _FallbackMarketDataService:
        """Fallback service when enhanced service unavailable"""
        def __init__(self, *args, **kwargs):
            logger.warning("⚠️ Using fallback MarketDataService - enhanced service not available")
            raise ImportError("Enhanced market data service not available")
    
    class _FallbackEnum:
        """Fallback enum for compatibility"""
        pass
    
    # Assign fallbacks
    MarketDataService = _FallbackMarketDataService
    EnhancedMarketDataService = _FallbackMarketDataService
    MarketTick = _FallbackEnum
    MarketStatus = _FallbackEnum
    TradingMode = _FallbackEnum
    Nifty100Universe = _FallbackEnum
    
    _IMPORT_SUCCESS = False

# FIXED: Export everything that legacy code might expect
__all__ = [
    'MarketDataService',
    'EnhancedMarketDataService',  # Also export enhanced version directly
    'MarketTick', 
    'MarketStatus',
    'TradingMode',
    'Nifty100Universe'
]

# FIXED: Add import status for debugging
def get_import_status() -> Dict[str, Any]:
    """Get import status for debugging"""
    return {
        'import_success': _IMPORT_SUCCESS,
        'enhanced_module_loaded': 'app.services.enhanced_market_data_nifty100' in sys.modules,
        'compatibility_layer_active': True,
        'available_classes': [name for name in __all__ if globals().get(name) is not None]
    }

# FIXED: Log final status
if _IMPORT_SUCCESS:
    logger.info("✅ Legacy MarketDataService imports redirected to Enhanced Service with import guards")
else:
    logger.warning("⚠️ Legacy MarketDataService using fallback mode - enhanced service unavailable")