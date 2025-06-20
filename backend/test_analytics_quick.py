# test_analytics.py
"""
Quick test script for Analytics Service
"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.analytics_service import AnalyticsService

async def quick_test():
    """Quick test of analytics service"""
    print("🧪 Quick Analytics Test...")
    
    try:
        # Create analytics service with explicit path
        db_path = os.path.join(os.getcwd(), "test_analytics.db")
        analytics = AnalyticsService(f"sqlite:///{db_path}")
        
        print(f"✅ Analytics service created")
        print(f"📁 Database path: {analytics.db_path}")
        
        # Wait a moment for database initialization
        await asyncio.sleep(1)
        
        # Test signal tracking
        test_signal = {
            "id": "test_001",
            "symbol": "TESTSTOCK",
            "action": "BUY",
            "entry_price": 1000.0,
            "target_price": 1050.0,
            "stop_loss": 950.0,
            "confidence": 0.85
        }
        
        await analytics.track_signal_generated(test_signal)
        print("✅ Signal tracking works")
        
        await analytics.track_telegram_sent(True, test_signal)
        print("✅ Telegram tracking works")
        
        # Get stats
        stats = analytics.get_daily_stats()
        print(f"✅ Daily stats: {stats['signals_generated']} signals generated")
        
        performance = analytics.get_performance_summary()
        print(f"✅ Performance summary generated")
        
        await analytics.close()
        print("✅ Analytics service closed")
        
        print("🎉 All tests passed!")
        
        # Clean up test database
        if os.path.exists(db_path):
            os.remove(db_path)
            print("🧹 Test database cleaned up")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())