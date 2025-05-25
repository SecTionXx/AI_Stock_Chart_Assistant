"""
Advanced Trend Analysis System for Stock Charts

This module implements sophisticated trend analysis algorithms
for identifying trend direction, strength, and potential reversals.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from scipy import signal
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import talib


class TrendDirection(Enum):
    """Enumeration of trend directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class TrendStrength(Enum):
    """Enumeration of trend strength levels"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalType(Enum):
    """Enumeration of signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class TrendSignal:
    """Represents a trend analysis signal"""
    signal_type: SignalType
    confidence: float
    trend_direction: TrendDirection
    trend_strength: TrendStrength
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: str = "1D"
    timestamp: datetime = field(default_factory=datetime.now)
    indicators: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    risk_reward_ratio: Optional[float] = None
    probability: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrendAnalyzer:
    """
    Advanced trend analysis system using multiple technical indicators
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.lookback_period = self.config.get('lookback_period', 50)
        self.short_ma_period = self.config.get('short_ma_period', 10)
        self.long_ma_period = self.config.get('long_ma_period', 50)
        
        # Technical indicator parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        
        self.logger.info("TrendAnalyzer initialized")
    
    async def analyze_trend(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> TrendSignal:
        """
        Perform comprehensive trend analysis
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_data: Optional volume data
            
        Returns:
            TrendSignal with analysis results
        """
        try:
            self.logger.info("Starting trend analysis")
            
            if len(price_data) < self.lookback_period:
                self.logger.warning(f"Insufficient data: {len(price_data)} < {self.lookback_period}")
                return self._create_default_signal()
            
            # Calculate technical indicators
            indicators = await self._calculate_indicators(price_data, volume_data)
            
            # Analyze trend direction and strength
            trend_direction = await self._determine_trend_direction(price_data, indicators)
            trend_strength = await self._determine_trend_strength(price_data, indicators)
            
            # Generate trading signal
            signal_type = await self._generate_signal(indicators, trend_direction, trend_strength)
            
            # Calculate confidence
            confidence = await self._calculate_confidence(indicators, trend_direction, trend_strength)
            
            # Calculate price targets and stop loss
            price_target, stop_loss = await self._calculate_targets(
                price_data, trend_direction, trend_strength
            )
            
            # Calculate risk-reward ratio
            current_price = price_data['close'].iloc[-1]
            risk_reward_ratio = None
            if price_target and stop_loss:
                potential_profit = abs(price_target - current_price)
                potential_loss = abs(current_price - stop_loss)
                if potential_loss > 0:
                    risk_reward_ratio = potential_profit / potential_loss
            
            # Generate description
            description = self._generate_description(
                signal_type, trend_direction, trend_strength, confidence
            )
            
            # Calculate probability
            probability = await self._calculate_probability(indicators, trend_direction)
            
            signal = TrendSignal(
                signal_type=signal_type,
                confidence=confidence,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                price_target=price_target,
                stop_loss=stop_loss,
                indicators=indicators,
                description=description,
                risk_reward_ratio=risk_reward_ratio,
                probability=probability,
                metadata={
                    'analysis_timestamp': datetime.now(),
                    'data_points': len(price_data),
                    'current_price': current_price
                }
            )
            
            self.logger.info(f"Trend analysis complete: {signal_type.value} signal")
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return self._create_default_signal()
    
    async def _calculate_indicators(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            close = price_data['close'].values
            high = price_data['high'].values if 'high' in price_data.columns else close
            low = price_data['low'].values if 'low' in price_data.columns else close
            volume = volume_data['volume'].values if volume_data is not None else None
            
            # Moving Averages
            indicators['sma_short'] = talib.SMA(close, timeperiod=self.short_ma_period)[-1]
            indicators['sma_long'] = talib.SMA(close, timeperiod=self.long_ma_period)[-1]
            indicators['ema_short'] = talib.EMA(close, timeperiod=self.short_ma_period)[-1]
            indicators['ema_long'] = talib.EMA(close, timeperiod=self.long_ma_period)[-1]
            
            # RSI
            rsi = talib.RSI(close, timeperiod=self.rsi_period)
            indicators['rsi'] = rsi[-1]
            indicators['rsi_avg'] = np.mean(rsi[-5:])  # 5-period average
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                close, 
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow, 
                signalperiod=self.macd_signal
            )
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macd_signal[-1]
            indicators['macd_histogram'] = macd_hist[-1]
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                close, 
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std
            )
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]
            indicators['bb_position'] = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            indicators['stoch_k'] = stoch_k[-1]
            indicators['stoch_d'] = stoch_d[-1]
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)[-1]
            
            # Average True Range (ATR)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            # Commodity Channel Index (CCI)
            indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)[-1]
            
            # Money Flow Index (if volume available)
            if volume is not None:
                indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)[-1]
                
                # On Balance Volume
                indicators['obv'] = talib.OBV(close, volume)[-1]
                
                # Volume Rate of Change
                indicators['volume_roc'] = talib.ROC(volume.astype(float), timeperiod=10)[-1]
            
            # Price Rate of Change
            indicators['price_roc'] = talib.ROC(close, timeperiod=10)[-1]
            
            # Linear regression slope (trend strength)
            x = np.arange(len(close[-20:]))
            slope, _, r_value, _, _ = linregress(x, close[-20:])
            indicators['linear_slope'] = slope
            indicators['linear_r_squared'] = r_value ** 2
            
            # Momentum
            indicators['momentum'] = talib.MOM(close, timeperiod=10)[-1]
            
            # Current price relative to recent high/low
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])
            indicators['price_position'] = (close[-1] - recent_low) / (recent_high - recent_low)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
        
        return indicators
    
    async def _determine_trend_direction(
        self, 
        price_data: pd.DataFrame, 
        indicators: Dict[str, float]
    ) -> TrendDirection:
        """Determine overall trend direction"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            # Moving Average signals
            if indicators.get('sma_short', 0) > indicators.get('sma_long', 0):
                bullish_signals += 2
            else:
                bearish_signals += 2
            total_signals += 2
            
            if indicators.get('ema_short', 0) > indicators.get('ema_long', 0):
                bullish_signals += 2
            else:
                bearish_signals += 2
            total_signals += 2
            
            # MACD signals
            if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                bullish_signals += 1
            else:
                bearish_signals += 1
            total_signals += 1
            
            if indicators.get('macd_histogram', 0) > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            total_signals += 1
            
            # Linear regression slope
            slope = indicators.get('linear_slope', 0)
            if slope > 0.1:
                bullish_signals += 2
            elif slope < -0.1:
                bearish_signals += 2
            else:
                # Sideways
                pass
            total_signals += 2
            
            # Price momentum
            if indicators.get('momentum', 0) > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            total_signals += 1
            
            # Price rate of change
            if indicators.get('price_roc', 0) > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            total_signals += 1
            
            # Determine trend based on signal strength
            bullish_ratio = bullish_signals / total_signals
            bearish_ratio = bearish_signals / total_signals
            
            if bullish_ratio > 0.6:
                return TrendDirection.BULLISH
            elif bearish_ratio > 0.6:
                return TrendDirection.BEARISH
            elif abs(bullish_ratio - bearish_ratio) < 0.2:
                return TrendDirection.SIDEWAYS
            else:
                return TrendDirection.UNKNOWN
                
        except Exception as e:
            self.logger.error(f"Error determining trend direction: {str(e)}")
            return TrendDirection.UNKNOWN
    
    async def _determine_trend_strength(
        self, 
        price_data: pd.DataFrame, 
        indicators: Dict[str, float]
    ) -> TrendStrength:
        """Determine trend strength"""
        try:
            strength_score = 0
            max_score = 0
            
            # Linear regression R-squared (trend consistency)
            r_squared = indicators.get('linear_r_squared', 0)
            strength_score += r_squared * 3
            max_score += 3
            
            # ATR relative to price (volatility)
            atr = indicators.get('atr', 0)
            current_price = price_data['close'].iloc[-1]
            if current_price > 0:
                atr_ratio = atr / current_price
                # Lower volatility = stronger trend
                volatility_score = max(0, 1 - atr_ratio * 10)
                strength_score += volatility_score * 2
            max_score += 2
            
            # Moving average separation
            sma_short = indicators.get('sma_short', 0)
            sma_long = indicators.get('sma_long', 0)
            if sma_long > 0:
                ma_separation = abs(sma_short - sma_long) / sma_long
                separation_score = min(1, ma_separation * 20)
                strength_score += separation_score * 2
            max_score += 2
            
            # MACD histogram strength
            macd_hist = abs(indicators.get('macd_histogram', 0))
            macd_score = min(1, macd_hist * 100)  # Normalize
            strength_score += macd_score * 1
            max_score += 1
            
            # Price position in recent range
            price_position = indicators.get('price_position', 0.5)
            # Extreme positions indicate strong trends
            position_score = max(abs(price_position - 0.5) * 2, 0)
            strength_score += position_score * 1
            max_score += 1
            
            # Normalize strength score
            if max_score > 0:
                normalized_strength = strength_score / max_score
            else:
                normalized_strength = 0
            
            # Classify strength
            if normalized_strength >= 0.8:
                return TrendStrength.VERY_STRONG
            elif normalized_strength >= 0.6:
                return TrendStrength.STRONG
            elif normalized_strength >= 0.4:
                return TrendStrength.MODERATE
            elif normalized_strength >= 0.2:
                return TrendStrength.WEAK
            else:
                return TrendStrength.VERY_WEAK
                
        except Exception as e:
            self.logger.error(f"Error determining trend strength: {str(e)}")
            return TrendStrength.WEAK
    
    async def _generate_signal(
        self, 
        indicators: Dict[str, float],
        trend_direction: TrendDirection,
        trend_strength: TrendStrength
    ) -> SignalType:
        """Generate trading signal based on analysis"""
        try:
            buy_signals = 0
            sell_signals = 0
            
            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi < 30:  # Oversold
                buy_signals += 2
            elif rsi > 70:  # Overbought
                sell_signals += 2
            elif rsi < 40:
                buy_signals += 1
            elif rsi > 60:
                sell_signals += 1
            
            # Stochastic signals
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            if stoch_k < 20 and stoch_d < 20:  # Oversold
                buy_signals += 1
            elif stoch_k > 80 and stoch_d > 80:  # Overbought
                sell_signals += 1
            
            # MACD signals
            if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                buy_signals += 1
            else:
                sell_signals += 1
            
            # Bollinger Bands signals
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position < 0.1:  # Near lower band
                buy_signals += 1
            elif bb_position > 0.9:  # Near upper band
                sell_signals += 1
            
            # Williams %R signals
            williams_r = indicators.get('williams_r', -50)
            if williams_r < -80:  # Oversold
                buy_signals += 1
            elif williams_r > -20:  # Overbought
                sell_signals += 1
            
            # Trend direction influence
            if trend_direction == TrendDirection.BULLISH:
                buy_signals += 2
            elif trend_direction == TrendDirection.BEARISH:
                sell_signals += 2
            
            # Trend strength influence
            strength_multiplier = {
                TrendStrength.VERY_STRONG: 2,
                TrendStrength.STRONG: 1.5,
                TrendStrength.MODERATE: 1,
                TrendStrength.WEAK: 0.5,
                TrendStrength.VERY_WEAK: 0.2
            }.get(trend_strength, 1)
            
            buy_signals *= strength_multiplier
            sell_signals *= strength_multiplier
            
            # Generate signal
            signal_diff = buy_signals - sell_signals
            
            if signal_diff > 4:
                return SignalType.STRONG_BUY
            elif signal_diff > 2:
                return SignalType.BUY
            elif signal_diff < -4:
                return SignalType.STRONG_SELL
            elif signal_diff < -2:
                return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return SignalType.HOLD
    
    async def _calculate_confidence(
        self, 
        indicators: Dict[str, float],
        trend_direction: TrendDirection,
        trend_strength: TrendStrength
    ) -> float:
        """Calculate confidence in the analysis"""
        try:
            confidence_factors = []
            
            # Trend consistency (R-squared)
            r_squared = indicators.get('linear_r_squared', 0)
            confidence_factors.append(r_squared)
            
            # Indicator agreement
            rsi = indicators.get('rsi', 50)
            stoch_k = indicators.get('stoch_k', 50)
            williams_r = indicators.get('williams_r', -50)
            
            # Check if momentum indicators agree
            momentum_agreement = 0
            if trend_direction == TrendDirection.BULLISH:
                if rsi > 50:
                    momentum_agreement += 1
                if stoch_k > 50:
                    momentum_agreement += 1
                if williams_r > -50:
                    momentum_agreement += 1
            elif trend_direction == TrendDirection.BEARISH:
                if rsi < 50:
                    momentum_agreement += 1
                if stoch_k < 50:
                    momentum_agreement += 1
                if williams_r < -50:
                    momentum_agreement += 1
            
            confidence_factors.append(momentum_agreement / 3)
            
            # MACD confirmation
            macd_confirmation = 0
            if (indicators.get('macd', 0) > indicators.get('macd_signal', 0) and 
                trend_direction == TrendDirection.BULLISH):
                macd_confirmation = 1
            elif (indicators.get('macd', 0) < indicators.get('macd_signal', 0) and 
                  trend_direction == TrendDirection.BEARISH):
                macd_confirmation = 1
            
            confidence_factors.append(macd_confirmation)
            
            # Trend strength contribution
            strength_confidence = {
                TrendStrength.VERY_STRONG: 0.9,
                TrendStrength.STRONG: 0.8,
                TrendStrength.MODERATE: 0.6,
                TrendStrength.WEAK: 0.4,
                TrendStrength.VERY_WEAK: 0.2
            }.get(trend_strength, 0.5)
            
            confidence_factors.append(strength_confidence)
            
            # Volume confirmation (if available)
            if 'mfi' in indicators:
                mfi = indicators['mfi']
                if trend_direction == TrendDirection.BULLISH and mfi > 50:
                    confidence_factors.append(0.8)
                elif trend_direction == TrendDirection.BEARISH and mfi < 50:
                    confidence_factors.append(0.8)
                else:
                    confidence_factors.append(0.4)
            
            # Calculate weighted average
            confidence = np.mean(confidence_factors)
            
            # Ensure confidence is within bounds
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    async def _calculate_targets(
        self, 
        price_data: pd.DataFrame,
        trend_direction: TrendDirection,
        trend_strength: TrendStrength
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate price targets and stop loss levels"""
        try:
            current_price = price_data['close'].iloc[-1]
            
            # Calculate ATR for volatility-based targets
            high = price_data['high'].values if 'high' in price_data.columns else price_data['close'].values
            low = price_data['low'].values if 'low' in price_data.columns else price_data['close'].values
            close = price_data['close'].values
            
            atr = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            # Strength multipliers
            strength_multipliers = {
                TrendStrength.VERY_STRONG: 3.0,
                TrendStrength.STRONG: 2.5,
                TrendStrength.MODERATE: 2.0,
                TrendStrength.WEAK: 1.5,
                TrendStrength.VERY_WEAK: 1.0
            }
            
            multiplier = strength_multipliers.get(trend_strength, 2.0)
            
            # Calculate targets based on trend direction
            if trend_direction == TrendDirection.BULLISH:
                price_target = current_price + (atr * multiplier)
                stop_loss = current_price - (atr * 1.5)
            elif trend_direction == TrendDirection.BEARISH:
                price_target = current_price - (atr * multiplier)
                stop_loss = current_price + (atr * 1.5)
            else:
                # For sideways trends, use smaller targets
                price_target = current_price + (atr * 0.5)
                stop_loss = current_price - (atr * 0.5)
            
            return price_target, stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating targets: {str(e)}")
            return None, None
    
    async def _calculate_probability(
        self, 
        indicators: Dict[str, float],
        trend_direction: TrendDirection
    ) -> float:
        """Calculate probability of trend continuation"""
        try:
            probability_factors = []
            
            # Trend strength factor
            r_squared = indicators.get('linear_r_squared', 0)
            probability_factors.append(0.3 + r_squared * 0.4)  # 0.3 to 0.7 range
            
            # Momentum factor
            rsi = indicators.get('rsi', 50)
            if trend_direction == TrendDirection.BULLISH:
                momentum_prob = min(1.0, (rsi - 30) / 40)  # Higher RSI = higher prob
            elif trend_direction == TrendDirection.BEARISH:
                momentum_prob = min(1.0, (70 - rsi) / 40)  # Lower RSI = higher prob
            else:
                momentum_prob = 0.5
            
            probability_factors.append(momentum_prob)
            
            # MACD factor
            macd_hist = indicators.get('macd_histogram', 0)
            if trend_direction == TrendDirection.BULLISH and macd_hist > 0:
                probability_factors.append(0.7)
            elif trend_direction == TrendDirection.BEARISH and macd_hist < 0:
                probability_factors.append(0.7)
            else:
                probability_factors.append(0.4)
            
            # Volume factor (if available)
            if 'mfi' in indicators:
                mfi = indicators['mfi']
                if trend_direction == TrendDirection.BULLISH and mfi > 50:
                    probability_factors.append(0.7)
                elif trend_direction == TrendDirection.BEARISH and mfi < 50:
                    probability_factors.append(0.7)
                else:
                    probability_factors.append(0.4)
            
            # Calculate weighted probability
            probability = np.mean(probability_factors)
            
            # Ensure probability is within reasonable bounds
            return max(0.2, min(0.9, probability))
            
        except Exception as e:
            self.logger.error(f"Error calculating probability: {str(e)}")
            return 0.5
    
    def _generate_description(
        self, 
        signal_type: SignalType,
        trend_direction: TrendDirection,
        trend_strength: TrendStrength,
        confidence: float
    ) -> str:
        """Generate human-readable description"""
        try:
            strength_desc = {
                TrendStrength.VERY_STRONG: "very strong",
                TrendStrength.STRONG: "strong", 
                TrendStrength.MODERATE: "moderate",
                TrendStrength.WEAK: "weak",
                TrendStrength.VERY_WEAK: "very weak"
            }.get(trend_strength, "moderate")
            
            direction_desc = {
                TrendDirection.BULLISH: "bullish",
                TrendDirection.BEARISH: "bearish",
                TrendDirection.SIDEWAYS: "sideways",
                TrendDirection.UNKNOWN: "unclear"
            }.get(trend_direction, "unclear")
            
            signal_desc = {
                SignalType.STRONG_BUY: "Strong Buy",
                SignalType.BUY: "Buy",
                SignalType.HOLD: "Hold",
                SignalType.SELL: "Sell",
                SignalType.STRONG_SELL: "Strong Sell"
            }.get(signal_type, "Hold")
            
            confidence_desc = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "low"
            
            return (f"{signal_desc} signal with {confidence_desc} confidence. "
                   f"Trend is {direction_desc} with {strength_desc} momentum.")
            
        except Exception as e:
            self.logger.error(f"Error generating description: {str(e)}")
            return "Analysis completed with mixed signals."
    
    def _create_default_signal(self) -> TrendSignal:
        """Create a default signal for error cases"""
        return TrendSignal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            trend_direction=TrendDirection.UNKNOWN,
            trend_strength=TrendStrength.WEAK,
            description="Insufficient data for reliable analysis"
        )
    
    async def get_support_resistance_levels(
        self, 
        price_data: pd.DataFrame,
        num_levels: int = 5
    ) -> Dict[str, List[float]]:
        """
        Identify key support and resistance levels
        
        Args:
            price_data: DataFrame with OHLCV data
            num_levels: Number of levels to identify
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            high = price_data['high'].values if 'high' in price_data.columns else price_data['close'].values
            low = price_data['low'].values if 'low' in price_data.columns else price_data['close'].values
            
            # Find peaks and valleys
            peaks, _ = signal.find_peaks(high, prominence=np.std(high) * 0.02)
            valleys, _ = signal.find_peaks(-low, prominence=np.std(low) * 0.02)
            
            # Get resistance levels (peaks)
            resistance_levels = high[peaks]
            resistance_levels = sorted(resistance_levels, reverse=True)[:num_levels]
            
            # Get support levels (valleys)
            support_levels = low[valleys]
            support_levels = sorted(support_levels)[:num_levels]
            
            return {
                'resistance': resistance_levels,
                'support': support_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error finding support/resistance levels: {str(e)}")
            return {'resistance': [], 'support': []}
    
    async def analyze_volume_profile(
        self, 
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze volume profile for additional insights
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_data: DataFrame with volume data
            
        Returns:
            Volume profile analysis
        """
        try:
            if len(volume_data) == 0:
                return {}
            
            volume = volume_data['volume'].values
            close = price_data['close'].values
            
            # Volume-weighted average price (VWAP)
            vwap = np.cumsum(close * volume) / np.cumsum(volume)
            
            # Volume rate of change
            volume_roc = talib.ROC(volume.astype(float), timeperiod=10)
            
            # Average volume
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            
            # Volume trend
            volume_trend = "increasing" if np.mean(volume[-5:]) > np.mean(volume[-10:-5]) else "decreasing"
            
            return {
                'vwap': vwap[-1],
                'volume_roc': volume_roc[-1],
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                'volume_trend': volume_trend,
                'unusual_volume': current_volume > avg_volume * 2
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {str(e)}")
            return {} 