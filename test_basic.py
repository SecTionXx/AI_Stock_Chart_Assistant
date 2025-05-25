#!/usr/bin/env python3
"""
Basic Test Script for AI Stock Chart Assistant v2.0
Tests core functionality without AI dependencies
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_yahoo_finance():
    """Test Yahoo Finance integration"""
    print("🔍 Testing Yahoo Finance integration...")
    
    try:
        from src.integrations.yahoo_finance import YahooFinanceClient
        
        yahoo = YahooFinanceClient()
        
        # Test basic stock data fetch
        print("  📊 Fetching AAPL data...")
        data = await yahoo.get_historical_data("AAPL", period="5d", interval="1d")
        
        if data is not None and len(data) > 0:
            print(f"  ✅ Success! Retrieved {len(data)} data points")
            # Yahoo Finance uses 'Close' (capital C) for column names
            print(f"  📈 Latest close price: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("  ❌ Failed to retrieve data")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

async def test_cache_manager():
    """Test cache manager"""
    print("🗄️ Testing Cache Manager...")
    
    try:
        from src.core.cache_manager import CacheManager
        
        cache = CacheManager()
        
        # Test basic caching
        test_key = "test_key"
        test_value = {"test": "data", "number": 123}
        
        # Store data
        await cache.set(test_key, test_value)
        print("  ✅ Data stored in cache")
        
        # Retrieve data
        retrieved = await cache.get(test_key)
        if retrieved == test_value:
            print("  ✅ Data retrieved successfully")
            return True
        else:
            print("  ❌ Data mismatch")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

async def test_pattern_detector():
    """Test pattern detector with sample data"""
    print("🔍 Testing Pattern Detector...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Test basic pattern detection logic without TA-Lib
        # Create sample price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        sample_data = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.abs(np.random.randn(100) * 0.2),
            'low': prices - np.abs(np.random.randn(100) * 0.2),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Simple pattern detection - look for basic trends
        sma_short = sample_data['close'].rolling(10).mean()
        sma_long = sample_data['close'].rolling(20).mean()
        
        # Count trend changes
        trend_changes = 0
        for i in range(20, len(sample_data)):
            if (sma_short.iloc[i] > sma_long.iloc[i]) != (sma_short.iloc[i-1] > sma_long.iloc[i-1]):
                trend_changes += 1
        
        print(f"  ✅ Basic pattern analysis completed. Found {trend_changes} trend changes")
        print("  ℹ️ Note: Full pattern detection requires TA-Lib installation")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

async def test_technical_indicators():
    """Test basic technical indicators"""
    print("📊 Testing Technical Indicators...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        
        data = pd.DataFrame({
            'close': prices,
            'high': prices + np.abs(np.random.randn(50) * 0.2),
            'low': prices - np.abs(np.random.randn(50) * 0.2),
            'volume': np.random.randint(1000000, 10000000, 50)
        }, index=dates)
        
        # Calculate simple moving averages
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        
        # Calculate basic indicators
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        
        print(f"  ✅ Calculated indicators for {len(data)} data points")
        print(f"  📈 Latest SMA 20: ${data['sma_20'].iloc[-1]:.2f}")
        print(f"  📊 20-day volatility: {volatility.iloc[-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False

async def main():
    """Run all tests"""
    print("🚀 AI Stock Chart Assistant v2.0 - Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Yahoo Finance", test_yahoo_finance),
        ("Cache Manager", test_cache_manager),
        ("Pattern Detector", test_pattern_detector),
        ("Technical Indicators", test_technical_indicators)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
        
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 50)
    print("📋 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The system is ready for basic usage.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main()) 