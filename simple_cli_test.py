#!/usr/bin/env python3
"""
Simple CLI Test for AI Stock Chart Assistant v2.0
Tests basic CLI functionality without complex AI dependencies
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def simple_stock_analysis(symbol: str):
    """Perform a simple stock analysis"""
    print(f"🔍 Analyzing {symbol}...")
    
    try:
        from src.integrations.yahoo_finance import YahooFinanceClient
        
        # Initialize Yahoo Finance client
        yahoo = YahooFinanceClient()
        
        # Get current stock data
        print("📊 Fetching current stock data...")
        stock_data = await yahoo.get_stock_data(symbol)
        
        if stock_data:
            print(f"✅ Current Stock Data for {symbol}:")
            print(f"   💰 Price: ${stock_data.current_price:.2f}")
            print(f"   📈 Change: {stock_data.change:+.2f} ({stock_data.change_percent:+.2f}%)")
            print(f"   📊 Volume: {stock_data.volume:,}")
            if stock_data.market_cap:
                print(f"   🏢 Market Cap: ${stock_data.market_cap:,}")
        
        # Get historical data
        print("\n📈 Fetching historical data...")
        hist_data = await yahoo.get_historical_data(symbol, period="1mo", interval="1d")
        
        if hist_data is not None and len(hist_data) > 0:
            print(f"✅ Retrieved {len(hist_data)} days of historical data")
            
            # Calculate basic technical indicators
            close_prices = hist_data['Close']
            
            # Simple moving averages
            sma_5 = close_prices.rolling(5).mean().iloc[-1]
            sma_20 = close_prices.rolling(20).mean().iloc[-1]
            
            # Basic volatility
            returns = close_prices.pct_change().dropna()
            volatility = returns.std() * 100
            
            # Price position
            high_52w = close_prices.max()
            low_52w = close_prices.min()
            current_price = close_prices.iloc[-1]
            position = (current_price - low_52w) / (high_52w - low_52w) * 100
            
            print(f"\n📊 Technical Analysis:")
            print(f"   📈 SMA 5: ${sma_5:.2f}")
            print(f"   📈 SMA 20: ${sma_20:.2f}")
            print(f"   📊 Volatility: {volatility:.2f}%")
            print(f"   🎯 Position in range: {position:.1f}%")
            
            # Simple trend analysis
            if sma_5 > sma_20:
                trend = "🟢 Bullish (Short MA > Long MA)"
            else:
                trend = "🔴 Bearish (Short MA < Long MA)"
            
            print(f"   📈 Trend: {trend}")
        
        # Get technical indicators
        print("\n🔧 Calculating technical indicators...")
        tech_indicators = await yahoo.calculate_technical_indicators(symbol)
        
        if tech_indicators:
            print(f"✅ Technical Indicators:")
            print(f"   📊 RSI: {tech_indicators.rsi:.2f}")
            print(f"   📈 MACD: {tech_indicators.macd:.4f}")
            print(f"   🎯 Support: ${tech_indicators.support_level:.2f}")
            print(f"   🎯 Resistance: ${tech_indicators.resistance_level:.2f}")
        
        # Get news
        print("\n📰 Fetching latest news...")
        news = await yahoo.get_market_news(symbol, limit=3)
        
        if news:
            print(f"✅ Latest News ({len(news)} articles):")
            for i, article in enumerate(news[:3], 1):
                title = article.get('title', 'No title')[:60] + "..." if len(article.get('title', '')) > 60 else article.get('title', 'No title')
                print(f"   {i}. {title}")
        
        print(f"\n🎉 Analysis complete for {symbol}!")
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return False

async def main():
    """Main function"""
    print("🚀 AI Stock Chart Assistant v2.0 - Simple CLI Test")
    print("=" * 60)
    
    # Test with a few popular stocks
    symbols = ["AAPL", "GOOGL", "TSLA"]
    
    for symbol in symbols:
        success = await simple_stock_analysis(symbol)
        if success:
            print(f"✅ {symbol} analysis completed successfully")
        else:
            print(f"❌ {symbol} analysis failed")
        print("-" * 60)
    
    print("\n🎯 CLI test completed!")

if __name__ == "__main__":
    asyncio.run(main()) 