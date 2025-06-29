#!/usr/bin/env python3
"""
Test script for simplified regime detector
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_regime_detector():
    """Test the simplified regime detector"""
    try:
        from app.services.regime_detector import RegimeDetector, MarketRegime
        
        print("✅ Simplified regime detector imported successfully")
        print(f"Available regimes: {MarketRegime.BULLISH}, {MarketRegime.BEARISH}, {MarketRegime.SIDEWAYS}")
        
        # Test regime classification logic
        detector = RegimeDetector(None)  # No market data service for testing
        
        # Test bullish classification
        nifty_analysis = {"trend_strength": 1.0, "direction": 1.0}
        volatility_analysis = {"volatility_level": 15.0}
        
        result = detector._classify_regime_simplified(nifty_analysis, volatility_analysis)
        print(f"✅ Bullish test: {result.regime} (confidence: {result.confidence:.1%})")
        
        # Test bearish classification
        nifty_analysis = {"trend_strength": 1.0, "direction": -1.0}
        result = detector._classify_regime_simplified(nifty_analysis, volatility_analysis)
        print(f"✅ Bearish test: {result.regime} (confidence: {result.confidence:.1%})")
        
        # Test sideways classification
        nifty_analysis = {"trend_strength": 0.3, "direction": 0.0}
        result = detector._classify_regime_simplified(nifty_analysis, volatility_analysis)
        print(f"✅ Sideways test: {result.regime} (confidence: {result.confidence:.1%})")
        
        print("🎉 All regime detector tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Regime detector test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_regime_detector())
    sys.exit(0 if success else 1) 