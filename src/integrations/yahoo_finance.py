"""
Yahoo Finance Integration for AI Stock Chart Assistant v2.0

Provides real-time stock data including:
- Current prices and volume
- Historical data
- Technical indicators
- Market news and sentiment
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

import yfinance as yf
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class StockData:
    """Stock data container"""
    symbol: str
    current_price: float
    previous_close: float
    change: float
    change_percent: float
    volume: int
    avg_volume: int
    market_cap: Optional[int]
    pe_ratio: Optional[float]
    timestamp: datetime


@dataclass
class TechnicalIndicators:
    """Technical indicators container"""
    sma_20: float
    sma_50: float
    sma_200: float
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    support_level: float
    resistance_level: float


@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    volume_weighted_price: float
    high_volume_levels: List[float]
    low_volume_levels: List[float]
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    unusual_volume: bool


class YahooFinanceClient:
    """
    Yahoo Finance API client with caching and error handling
    """
    
    def __init__(self, cache_ttl: int = 300):  # 5 minutes cache
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_stock_data(self, symbol: str) -> Optional[StockData]:
        """
        Get current stock data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            StockData object or None if failed
        """
        # Check cache first
        cache_key = f"stock_data_{symbol}"
        if self._is_cached(cache_key):
            cached_data = self.cache[cache_key]['data']
            return StockData(**cached_data)
        
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(ticker.info.get)
            hist = await asyncio.to_thread(
                ticker.history, 
                period="2d"
            )
            
            if hist.empty:
                self.logger.warning(f"No data available for symbol: {symbol}")
                return None
            
            # Get current and previous data
            current = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else current
            
            # Calculate change
            change = current['Close'] - previous['Close']
            change_percent = (change / previous['Close']) * 100
            
            # Create StockData object
            stock_data = StockData(
                symbol=symbol.upper(),
                current_price=float(current['Close']),
                previous_close=float(previous['Close']),
                change=float(change),
                change_percent=float(change_percent),
                volume=int(current['Volume']),
                avg_volume=info.get('averageVolume', 0),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                timestamp=datetime.now()
            )
            
            # Cache the result
            self._cache_data(cache_key, stock_data.__dict__)
            
            return stock_data
            
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_historical_data(self, 
                                symbol: str, 
                                period: str = "1y",
                                interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get historical stock data
        
        Args:
            symbol: Stock symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"historical_{symbol}_{period}_{interval}"
        
        # Check cache
        if self._is_cached(cache_key):
            return pd.DataFrame(self.cache[cache_key]['data'])
        
        try:
            ticker = yf.Ticker(symbol)
            hist = await asyncio.to_thread(
                ticker.history,
                period=period,
                interval=interval
            )
            
            if hist.empty:
                self.logger.warning(f"No historical data for {symbol}")
                return None
            
            # Cache the result
            self._cache_data(cache_key, hist.to_dict())
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    async def calculate_technical_indicators(self, 
                                           symbol: str,
                                           period: str = "6mo") -> Optional[TechnicalIndicators]:
        """
        Calculate technical indicators for a stock
        
        Args:
            symbol: Stock symbol
            period: Data period for calculations
            
        Returns:
            TechnicalIndicators object
        """
        try:
            # Get historical data
            hist = await self.get_historical_data(symbol, period=period)
            if hist is None or hist.empty:
                return None
            
            # Calculate indicators
            close_prices = hist['Close']
            
            # Simple Moving Averages
            sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
            sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
            sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
            
            # RSI
            rsi = self._calculate_rsi(close_prices)
            
            # MACD
            macd, macd_signal = self._calculate_macd(close_prices)
            
            # Bollinger Bands
            bollinger_upper, bollinger_lower = self._calculate_bollinger_bands(close_prices)
            
            # Support and Resistance
            support, resistance = self._calculate_support_resistance(hist)
            
            return TechnicalIndicators(
                sma_20=float(sma_20) if not pd.isna(sma_20) else 0.0,
                sma_50=float(sma_50) if not pd.isna(sma_50) else 0.0,
                sma_200=float(sma_200) if not pd.isna(sma_200) else 0.0,
                rsi=float(rsi),
                macd=float(macd),
                macd_signal=float(macd_signal),
                bollinger_upper=float(bollinger_upper),
                bollinger_lower=float(bollinger_lower),
                support_level=float(support),
                resistance_level=float(resistance)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return None
    
    async def analyze_volume_profile(self, 
                                   symbol: str,
                                   period: str = "1mo") -> Optional[VolumeProfile]:
        """
        Analyze volume profile for a stock
        
        Args:
            symbol: Stock symbol
            period: Analysis period
            
        Returns:
            VolumeProfile object
        """
        try:
            hist = await self.get_historical_data(symbol, period=period)
            if hist is None or hist.empty:
                return None
            
            # Calculate volume-weighted average price
            vwap = (hist['Close'] * hist['Volume']).sum() / hist['Volume'].sum()
            
            # Find high and low volume price levels
            price_volume = hist.groupby(hist['Close'].round(2))['Volume'].sum()
            sorted_pv = price_volume.sort_values(ascending=False)
            
            high_volume_levels = sorted_pv.head(5).index.tolist()
            low_volume_levels = sorted_pv.tail(5).index.tolist()
            
            # Analyze volume trend
            recent_volume = hist['Volume'].tail(10).mean()
            historical_volume = hist['Volume'].mean()
            
            if recent_volume > historical_volume * 1.2:
                volume_trend = "increasing"
            elif recent_volume < historical_volume * 0.8:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
            
            # Check for unusual volume
            latest_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            unusual_volume = latest_volume > avg_volume * 2
            
            return VolumeProfile(
                volume_weighted_price=float(vwap),
                high_volume_levels=high_volume_levels,
                low_volume_levels=low_volume_levels,
                volume_trend=volume_trend,
                unusual_volume=unusual_volume
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile for {symbol}: {e}")
            return None
    
    async def get_market_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent news for a stock symbol
        
        Args:
            symbol: Stock symbol
            limit: Number of news items to return
            
        Returns:
            List of news items
        """
        try:
            ticker = yf.Ticker(symbol)
            news = await asyncio.to_thread(lambda: ticker.news)
            
            # Format news items
            formatted_news = []
            for item in news[:limit]:
                formatted_news.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'source': item.get('publisher', '')
                })
            
            return formatted_news
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    async def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[StockData]]:
        """
        Get data for multiple stocks concurrently
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to StockData
        """
        tasks = [self.get_stock_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stock_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching data for {symbol}: {result}")
                stock_data[symbol] = None
            else:
                stock_data[symbol] = result
        
        return stock_data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD and signal line"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        
        return (
            macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0,
            signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0.0
        )
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        return (
            upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else 0.0,
            lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else 0.0
        )
    
    def _calculate_support_resistance(self, hist: pd.DataFrame) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        try:
            # Simple method: use recent lows and highs
            recent_data = hist.tail(50)  # Last 50 periods
            
            # Support: recent low
            support = recent_data['Low'].min()
            
            # Resistance: recent high
            resistance = recent_data['High'].max()
            
            return float(support), float(resistance)
            
        except Exception:
            current_price = hist['Close'].iloc[-1]
            return float(current_price * 0.95), float(current_price * 1.05)
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and not expired"""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        return (time.time() - cache_time) < self.cache_ttl
    
    def _cache_data(self, key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.logger.info("Cache cleared")


class MarketDataAnalyzer:
    """
    Advanced market data analysis utilities
    """
    
    def __init__(self, yahoo_client: YahooFinanceClient):
        self.client = yahoo_client
        self.logger = logging.getLogger(__name__)
    
    async def analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze market sentiment for a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sentiment analysis results
        """
        try:
            # Get stock data and technical indicators
            stock_data = await self.client.get_stock_data(symbol)
            tech_indicators = await self.client.calculate_technical_indicators(symbol)
            volume_profile = await self.client.analyze_volume_profile(symbol)
            
            if not all([stock_data, tech_indicators, volume_profile]):
                return {'sentiment': 'neutral', 'confidence': 0.0}
            
            # Calculate sentiment score
            sentiment_score = 0.0
            factors = []
            
            # Price momentum
            if stock_data.change_percent > 2:
                sentiment_score += 0.3
                factors.append("Strong positive price momentum")
            elif stock_data.change_percent < -2:
                sentiment_score -= 0.3
                factors.append("Strong negative price momentum")
            
            # Moving average position
            if stock_data.current_price > tech_indicators.sma_20:
                sentiment_score += 0.2
                factors.append("Price above 20-day SMA")
            else:
                sentiment_score -= 0.2
                factors.append("Price below 20-day SMA")
            
            # RSI levels
            if tech_indicators.rsi > 70:
                sentiment_score -= 0.2
                factors.append("RSI indicates overbought")
            elif tech_indicators.rsi < 30:
                sentiment_score += 0.2
                factors.append("RSI indicates oversold")
            
            # Volume analysis
            if volume_profile.unusual_volume and stock_data.change_percent > 0:
                sentiment_score += 0.3
                factors.append("Unusual volume with positive price action")
            elif volume_profile.unusual_volume and stock_data.change_percent < 0:
                sentiment_score -= 0.3
                factors.append("Unusual volume with negative price action")
            
            # Normalize sentiment score
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Determine sentiment label
            if sentiment_score > 0.3:
                sentiment = "bullish"
            elif sentiment_score < -0.3:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return {
                'sentiment': sentiment,
                'score': sentiment_score,
                'confidence': abs(sentiment_score),
                'factors': factors,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment for {symbol}: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
    
    async def compare_with_market(self, symbol: str, market_symbols: List[str] = None) -> Dict[str, Any]:
        """
        Compare stock performance with market indices
        
        Args:
            symbol: Stock symbol to compare
            market_symbols: Market indices to compare against
            
        Returns:
            Comparison results
        """
        if market_symbols is None:
            market_symbols = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow, NASDAQ
        
        try:
            # Get data for stock and market indices
            all_symbols = [symbol] + market_symbols
            stock_data = await self.client.get_multiple_stocks(all_symbols)
            
            if not stock_data.get(symbol):
                return {'error': f'No data available for {symbol}'}
            
            target_stock = stock_data[symbol]
            comparisons = {}
            
            for market_symbol in market_symbols:
                market_data = stock_data.get(market_symbol)
                if market_data:
                    relative_performance = (
                        target_stock.change_percent - market_data.change_percent
                    )
                    
                    comparisons[market_symbol] = {
                        'market_change': market_data.change_percent,
                        'stock_change': target_stock.change_percent,
                        'relative_performance': relative_performance,
                        'outperforming': relative_performance > 0
                    }
            
            return {
                'symbol': symbol,
                'comparisons': comparisons,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing {symbol} with market: {e}")
            return {'error': str(e)} 