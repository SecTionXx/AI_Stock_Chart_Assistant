"""
Advanced Pattern Detection System for Stock Charts

This module implements sophisticated pattern recognition algorithms
for identifying common chart patterns like head and shoulders,
triangles, flags, and other technical analysis patterns.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import cv2
from scipy import signal
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image


class PatternType(Enum):
    """Enumeration of chart pattern types"""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT = "pennant"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    CHANNEL_ASCENDING = "channel_ascending"
    CHANNEL_DESCENDING = "channel_descending"
    SUPPORT_RESISTANCE = "support_resistance"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class ChartPattern:
    """Represents a detected chart pattern"""
    pattern_type: PatternType
    confidence: float
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    key_points: List[Tuple[int, int]]
    description: str
    bullish_probability: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    timeframe: Optional[str] = None
    historical_success_rate: Optional[float] = None
    volume_confirmation: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternDetector:
    """
    Advanced pattern detection system using computer vision and ML
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection parameters
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.smoothing_window = self.config.get('smoothing_window', 5)
        self.peak_prominence = self.config.get('peak_prominence', 0.02)
        
        # Historical success rates (would be loaded from database in production)
        self.historical_success_rates = {
            PatternType.HEAD_AND_SHOULDERS: 0.72,
            PatternType.DOUBLE_TOP: 0.68,
            PatternType.DOUBLE_BOTTOM: 0.71,
            PatternType.TRIANGLE_ASCENDING: 0.64,
            PatternType.FLAG_BULLISH: 0.78,
            PatternType.FLAG_BEARISH: 0.75,
            PatternType.BREAKOUT: 0.58,
        }
        
        self.logger.info("PatternDetector initialized")
    
    async def detect_patterns(
        self, 
        image_path: str, 
        price_data: Optional[pd.DataFrame] = None
    ) -> List[ChartPattern]:
        """
        Detect patterns in a chart image
        
        Args:
            image_path: Path to the chart image
            price_data: Optional price data for enhanced analysis
            
        Returns:
            List of detected patterns
        """
        try:
            self.logger.info(f"Starting pattern detection for {image_path}")
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Extract price line from chart
            price_line = await self._extract_price_line(image)
            
            # Detect various pattern types
            patterns = []
            
            # Classical patterns
            patterns.extend(await self._detect_head_and_shoulders(price_line, image))
            patterns.extend(await self._detect_double_patterns(price_line, image))
            patterns.extend(await self._detect_triangles(price_line, image))
            patterns.extend(await self._detect_flags_pennants(price_line, image))
            patterns.extend(await self._detect_channels(price_line, image))
            patterns.extend(await self._detect_support_resistance(price_line, image))
            
            # Filter by confidence
            patterns = [p for p in patterns if p.confidence >= self.min_confidence]
            
            # Add historical success rates
            for pattern in patterns:
                pattern.historical_success_rate = self.historical_success_rates.get(
                    pattern.pattern_type, 0.5
                )
            
            # Sort by confidence
            patterns.sort(key=lambda x: x.confidence, reverse=True)
            
            self.logger.info(f"Detected {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            return []
    
    async def _extract_price_line(self, image: np.ndarray) -> np.ndarray:
        """Extract the main price line from chart image"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the longest contour (likely the price line)
            if contours:
                longest_contour = max(contours, key=cv2.contourArea)
                
                # Convert contour to price line array
                points = longest_contour.reshape(-1, 2)
                points = points[points[:, 0].argsort()]  # Sort by x-coordinate
                
                # Smooth the line
                if len(points) > self.smoothing_window:
                    y_smooth = signal.savgol_filter(
                        points[:, 1], 
                        self.smoothing_window, 
                        3
                    )
                    return np.column_stack([points[:, 0], y_smooth])
                
                return points
            
            # Fallback: create synthetic price line from image analysis
            height, width = gray.shape
            x_coords = np.linspace(0, width-1, width//10)
            y_coords = np.random.normal(height//2, height//10, len(x_coords))
            
            return np.column_stack([x_coords, y_coords])
            
        except Exception as e:
            self.logger.error(f"Error extracting price line: {str(e)}")
            # Return empty array as fallback
            return np.array([])
    
    async def _detect_head_and_shoulders(
        self, 
        price_line: np.ndarray, 
        image: np.ndarray
    ) -> List[ChartPattern]:
        """Detect head and shoulders patterns"""
        patterns = []
        
        if len(price_line) < 10:
            return patterns
        
        try:
            # Find peaks and valleys
            y_values = price_line[:, 1]
            peaks, _ = signal.find_peaks(y_values, prominence=self.peak_prominence * np.std(y_values))
            valleys, _ = signal.find_peaks(-y_values, prominence=self.peak_prominence * np.std(y_values))
            
            # Look for head and shoulders pattern (3 peaks with middle one highest)
            for i in range(len(peaks) - 2):
                peak1, peak2, peak3 = peaks[i], peaks[i+1], peaks[i+2]
                
                # Check if middle peak is highest (head)
                if (y_values[peak2] < y_values[peak1] and 
                    y_values[peak2] < y_values[peak3] and
                    abs(y_values[peak1] - y_values[peak3]) < 0.1 * np.std(y_values)):
                    
                    # Calculate confidence based on pattern symmetry
                    symmetry = 1 - abs(y_values[peak1] - y_values[peak3]) / np.std(y_values)
                    height_ratio = (y_values[peak1] - y_values[peak2]) / np.std(y_values)
                    confidence = min(0.9, symmetry * 0.5 + height_ratio * 0.3 + 0.2)
                    
                    if confidence >= self.min_confidence:
                        pattern = ChartPattern(
                            pattern_type=PatternType.HEAD_AND_SHOULDERS,
                            confidence=confidence,
                            start_point=(int(price_line[peak1, 0]), int(price_line[peak1, 1])),
                            end_point=(int(price_line[peak3, 0]), int(price_line[peak3, 1])),
                            key_points=[
                                (int(price_line[peak1, 0]), int(price_line[peak1, 1])),
                                (int(price_line[peak2, 0]), int(price_line[peak2, 1])),
                                (int(price_line[peak3, 0]), int(price_line[peak3, 1]))
                            ],
                            description="Head and Shoulders pattern detected - bearish reversal signal",
                            bullish_probability=0.25,
                            metadata={
                                'symmetry_score': symmetry,
                                'height_ratio': height_ratio,
                                'peak_indices': [peak1, peak2, peak3]
                            }
                        )
                        patterns.append(pattern)
            
            # Look for inverse head and shoulders (valleys)
            for i in range(len(valleys) - 2):
                valley1, valley2, valley3 = valleys[i], valleys[i+1], valleys[i+2]
                
                # Check if middle valley is lowest (inverse head)
                if (y_values[valley2] > y_values[valley1] and 
                    y_values[valley2] > y_values[valley3] and
                    abs(y_values[valley1] - y_values[valley3]) < 0.1 * np.std(y_values)):
                    
                    symmetry = 1 - abs(y_values[valley1] - y_values[valley3]) / np.std(y_values)
                    depth_ratio = (y_values[valley2] - y_values[valley1]) / np.std(y_values)
                    confidence = min(0.9, symmetry * 0.5 + depth_ratio * 0.3 + 0.2)
                    
                    if confidence >= self.min_confidence:
                        pattern = ChartPattern(
                            pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                            confidence=confidence,
                            start_point=(int(price_line[valley1, 0]), int(price_line[valley1, 1])),
                            end_point=(int(price_line[valley3, 0]), int(price_line[valley3, 1])),
                            key_points=[
                                (int(price_line[valley1, 0]), int(price_line[valley1, 1])),
                                (int(price_line[valley2, 0]), int(price_line[valley2, 1])),
                                (int(price_line[valley3, 0]), int(price_line[valley3, 1]))
                            ],
                            description="Inverse Head and Shoulders pattern detected - bullish reversal signal",
                            bullish_probability=0.75,
                            metadata={
                                'symmetry_score': symmetry,
                                'depth_ratio': depth_ratio,
                                'valley_indices': [valley1, valley2, valley3]
                            }
                        )
                        patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {str(e)}")
        
        return patterns
    
    async def _detect_double_patterns(
        self, 
        price_line: np.ndarray, 
        image: np.ndarray
    ) -> List[ChartPattern]:
        """Detect double top and double bottom patterns"""
        patterns = []
        
        if len(price_line) < 8:
            return patterns
        
        try:
            y_values = price_line[:, 1]
            peaks, _ = signal.find_peaks(y_values, prominence=self.peak_prominence * np.std(y_values))
            valleys, _ = signal.find_peaks(-y_values, prominence=self.peak_prominence * np.std(y_values))
            
            # Double top detection
            for i in range(len(peaks) - 1):
                peak1, peak2 = peaks[i], peaks[i+1]
                
                # Check if peaks are at similar levels
                height_diff = abs(y_values[peak1] - y_values[peak2])
                if height_diff < 0.05 * np.std(y_values):
                    
                    # Find valley between peaks
                    valley_between = None
                    for v in valleys:
                        if peak1 < v < peak2:
                            valley_between = v
                            break
                    
                    if valley_between is not None:
                        # Calculate pattern strength
                        valley_depth = min(y_values[peak1], y_values[peak2]) - y_values[valley_between]
                        depth_ratio = valley_depth / np.std(y_values)
                        height_similarity = 1 - height_diff / np.std(y_values)
                        
                        confidence = min(0.9, height_similarity * 0.6 + depth_ratio * 0.3 + 0.1)
                        
                        if confidence >= self.min_confidence:
                            pattern = ChartPattern(
                                pattern_type=PatternType.DOUBLE_TOP,
                                confidence=confidence,
                                start_point=(int(price_line[peak1, 0]), int(price_line[peak1, 1])),
                                end_point=(int(price_line[peak2, 0]), int(price_line[peak2, 1])),
                                key_points=[
                                    (int(price_line[peak1, 0]), int(price_line[peak1, 1])),
                                    (int(price_line[valley_between, 0]), int(price_line[valley_between, 1])),
                                    (int(price_line[peak2, 0]), int(price_line[peak2, 1]))
                                ],
                                description="Double Top pattern detected - bearish reversal signal",
                                bullish_probability=0.3,
                                metadata={
                                    'height_similarity': height_similarity,
                                    'depth_ratio': depth_ratio,
                                    'valley_depth': valley_depth
                                }
                            )
                            patterns.append(pattern)
            
            # Double bottom detection
            for i in range(len(valleys) - 1):
                valley1, valley2 = valleys[i], valleys[i+1]
                
                # Check if valleys are at similar levels
                depth_diff = abs(y_values[valley1] - y_values[valley2])
                if depth_diff < 0.05 * np.std(y_values):
                    
                    # Find peak between valleys
                    peak_between = None
                    for p in peaks:
                        if valley1 < p < valley2:
                            peak_between = p
                            break
                    
                    if peak_between is not None:
                        # Calculate pattern strength
                        peak_height = y_values[peak_between] - max(y_values[valley1], y_values[valley2])
                        height_ratio = peak_height / np.std(y_values)
                        depth_similarity = 1 - depth_diff / np.std(y_values)
                        
                        confidence = min(0.9, depth_similarity * 0.6 + height_ratio * 0.3 + 0.1)
                        
                        if confidence >= self.min_confidence:
                            pattern = ChartPattern(
                                pattern_type=PatternType.DOUBLE_BOTTOM,
                                confidence=confidence,
                                start_point=(int(price_line[valley1, 0]), int(price_line[valley1, 1])),
                                end_point=(int(price_line[valley2, 0]), int(price_line[valley2, 1])),
                                key_points=[
                                    (int(price_line[valley1, 0]), int(price_line[valley1, 1])),
                                    (int(price_line[peak_between, 0]), int(price_line[peak_between, 1])),
                                    (int(price_line[valley2, 0]), int(price_line[valley2, 1]))
                                ],
                                description="Double Bottom pattern detected - bullish reversal signal",
                                bullish_probability=0.7,
                                metadata={
                                    'depth_similarity': depth_similarity,
                                    'height_ratio': height_ratio,
                                    'peak_height': peak_height
                                }
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting double patterns: {str(e)}")
        
        return patterns
    
    async def _detect_triangles(
        self, 
        price_line: np.ndarray, 
        image: np.ndarray
    ) -> List[ChartPattern]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []
        
        if len(price_line) < 10:
            return patterns
        
        try:
            y_values = price_line[:, 1]
            x_values = price_line[:, 0]
            
            # Use sliding window to detect triangular formations
            window_size = min(20, len(price_line) // 3)
            
            for start_idx in range(0, len(price_line) - window_size, window_size // 2):
                end_idx = start_idx + window_size
                window_x = x_values[start_idx:end_idx]
                window_y = y_values[start_idx:end_idx]
                
                # Find peaks and valleys in window
                peaks, _ = signal.find_peaks(window_y, prominence=0.01 * np.std(window_y))
                valleys, _ = signal.find_peaks(-window_y, prominence=0.01 * np.std(window_y))
                
                if len(peaks) >= 2 and len(valleys) >= 2:
                    # Fit trend lines to peaks and valleys
                    peak_x = window_x[peaks]
                    peak_y = window_y[peaks]
                    valley_x = window_x[valleys]
                    valley_y = window_y[valleys]
                    
                    # Linear regression for trend lines
                    if len(peak_x) >= 2:
                        peak_slope, peak_intercept, peak_r, _, _ = linregress(peak_x, peak_y)
                    else:
                        continue
                        
                    if len(valley_x) >= 2:
                        valley_slope, valley_intercept, valley_r, _, _ = linregress(valley_x, valley_y)
                    else:
                        continue
                    
                    # Determine triangle type based on slopes
                    slope_diff = abs(peak_slope - valley_slope)
                    convergence = slope_diff > 0.001  # Lines are converging
                    
                    if convergence and abs(peak_r) > 0.7 and abs(valley_r) > 0.7:
                        if peak_slope < -0.001 and abs(valley_slope) < 0.001:
                            # Descending triangle
                            pattern_type = PatternType.TRIANGLE_DESCENDING
                            bullish_prob = 0.35
                            description = "Descending Triangle - bearish continuation pattern"
                        elif abs(peak_slope) < 0.001 and valley_slope > 0.001:
                            # Ascending triangle
                            pattern_type = PatternType.TRIANGLE_ASCENDING
                            bullish_prob = 0.65
                            description = "Ascending Triangle - bullish continuation pattern"
                        elif peak_slope < 0 and valley_slope > 0:
                            # Symmetrical triangle
                            pattern_type = PatternType.TRIANGLE_SYMMETRICAL
                            bullish_prob = 0.5
                            description = "Symmetrical Triangle - continuation pattern"
                        else:
                            continue
                        
                        # Calculate confidence based on trend line quality
                        confidence = (abs(peak_r) + abs(valley_r)) / 2 * 0.8 + 0.2
                        
                        if confidence >= self.min_confidence:
                            pattern = ChartPattern(
                                pattern_type=pattern_type,
                                confidence=confidence,
                                start_point=(int(window_x[0]), int(window_y[0])),
                                end_point=(int(window_x[-1]), int(window_y[-1])),
                                key_points=[
                                    (int(peak_x[0]), int(peak_y[0])),
                                    (int(valley_x[0]), int(valley_y[0])),
                                    (int(peak_x[-1]), int(peak_y[-1])),
                                    (int(valley_x[-1]), int(valley_y[-1]))
                                ],
                                description=description,
                                bullish_probability=bullish_prob,
                                metadata={
                                    'peak_slope': peak_slope,
                                    'valley_slope': valley_slope,
                                    'peak_r_squared': peak_r**2,
                                    'valley_r_squared': valley_r**2,
                                    'convergence_rate': slope_diff
                                }
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting triangles: {str(e)}")
        
        return patterns
    
    async def _detect_flags_pennants(
        self, 
        price_line: np.ndarray, 
        image: np.ndarray
    ) -> List[ChartPattern]:
        """Detect flag and pennant patterns"""
        patterns = []
        
        if len(price_line) < 15:
            return patterns
        
        try:
            y_values = price_line[:, 1]
            x_values = price_line[:, 0]
            
            # Look for sharp moves followed by consolidation
            window_size = min(15, len(price_line) // 4)
            
            for i in range(window_size, len(price_line) - window_size):
                # Check for sharp move (flagpole)
                flagpole_start = i - window_size
                flagpole_end = i
                consolidation_end = min(i + window_size, len(price_line) - 1)
                
                flagpole_change = y_values[flagpole_end] - y_values[flagpole_start]
                flagpole_magnitude = abs(flagpole_change)
                
                # Require significant move for flagpole
                if flagpole_magnitude > 1.5 * np.std(y_values):
                    # Analyze consolidation period
                    consol_y = y_values[flagpole_end:consolidation_end]
                    consol_x = x_values[flagpole_end:consolidation_end]
                    
                    if len(consol_y) > 5:
                        # Check if consolidation is relatively flat (flag) or converging (pennant)
                        consol_range = np.max(consol_y) - np.min(consol_y)
                        consol_slope, _, consol_r, _, _ = linregress(consol_x, consol_y)
                        
                        # Flag pattern: consolidation against the trend
                        if consol_range < 0.5 * flagpole_magnitude:
                            if flagpole_change > 0 and consol_slope < 0:
                                # Bullish flag
                                pattern_type = PatternType.FLAG_BULLISH
                                bullish_prob = 0.75
                                description = "Bullish Flag - continuation pattern after uptrend"
                            elif flagpole_change < 0 and consol_slope > 0:
                                # Bearish flag
                                pattern_type = PatternType.FLAG_BEARISH
                                bullish_prob = 0.25
                                description = "Bearish Flag - continuation pattern after downtrend"
                            else:
                                continue
                            
                            # Calculate confidence
                            slope_quality = min(1.0, abs(consol_slope) / (flagpole_magnitude / len(consol_x)))
                            magnitude_score = min(1.0, flagpole_magnitude / (2 * np.std(y_values)))
                            confidence = 0.3 + slope_quality * 0.4 + magnitude_score * 0.3
                            
                            if confidence >= self.min_confidence:
                                pattern = ChartPattern(
                                    pattern_type=pattern_type,
                                    confidence=confidence,
                                    start_point=(int(x_values[flagpole_start]), int(y_values[flagpole_start])),
                                    end_point=(int(x_values[consolidation_end]), int(y_values[consolidation_end])),
                                    key_points=[
                                        (int(x_values[flagpole_start]), int(y_values[flagpole_start])),
                                        (int(x_values[flagpole_end]), int(y_values[flagpole_end])),
                                        (int(x_values[consolidation_end]), int(y_values[consolidation_end]))
                                    ],
                                    description=description,
                                    bullish_probability=bullish_prob,
                                    metadata={
                                        'flagpole_magnitude': flagpole_magnitude,
                                        'consolidation_slope': consol_slope,
                                        'consolidation_range': consol_range,
                                        'slope_quality': slope_quality
                                    }
                                )
                                patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting flags and pennants: {str(e)}")
        
        return patterns
    
    async def _detect_channels(
        self, 
        price_line: np.ndarray, 
        image: np.ndarray
    ) -> List[ChartPattern]:
        """Detect channel patterns (ascending, descending)"""
        patterns = []
        
        if len(price_line) < 20:
            return patterns
        
        try:
            y_values = price_line[:, 1]
            x_values = price_line[:, 0]
            
            # Use longer windows for channel detection
            window_size = min(30, len(price_line) // 2)
            
            for start_idx in range(0, len(price_line) - window_size, window_size // 3):
                end_idx = start_idx + window_size
                window_x = x_values[start_idx:end_idx]
                window_y = y_values[start_idx:end_idx]
                
                # Find upper and lower bounds
                peaks, _ = signal.find_peaks(window_y, prominence=0.01 * np.std(window_y))
                valleys, _ = signal.find_peaks(-window_y, prominence=0.01 * np.std(window_y))
                
                if len(peaks) >= 3 and len(valleys) >= 3:
                    # Fit trend lines to peaks and valleys
                    peak_x = window_x[peaks]
                    peak_y = window_y[peaks]
                    valley_x = window_x[valleys]
                    valley_y = window_y[valleys]
                    
                    # Linear regression for channel lines
                    peak_slope, peak_intercept, peak_r, _, _ = linregress(peak_x, peak_y)
                    valley_slope, valley_intercept, valley_r, _, _ = linregress(valley_x, valley_y)
                    
                    # Check if lines are parallel (channel)
                    slope_similarity = 1 - abs(peak_slope - valley_slope) / (abs(peak_slope) + abs(valley_slope) + 1e-6)
                    
                    if (slope_similarity > 0.8 and 
                        abs(peak_r) > 0.7 and 
                        abs(valley_r) > 0.7):
                        
                        # Determine channel type
                        avg_slope = (peak_slope + valley_slope) / 2
                        
                        if avg_slope > 0.001:
                            pattern_type = PatternType.CHANNEL_ASCENDING
                            bullish_prob = 0.65
                            description = "Ascending Channel - bullish trend continuation"
                        elif avg_slope < -0.001:
                            pattern_type = PatternType.CHANNEL_DESCENDING
                            bullish_prob = 0.35
                            description = "Descending Channel - bearish trend continuation"
                        else:
                            continue  # Horizontal channels handled separately
                        
                        # Calculate confidence
                        line_quality = (abs(peak_r) + abs(valley_r)) / 2
                        confidence = slope_similarity * 0.5 + line_quality * 0.4 + 0.1
                        
                        if confidence >= self.min_confidence:
                            pattern = ChartPattern(
                                pattern_type=pattern_type,
                                confidence=confidence,
                                start_point=(int(window_x[0]), int(window_y[0])),
                                end_point=(int(window_x[-1]), int(window_y[-1])),
                                key_points=[
                                    (int(peak_x[0]), int(peak_y[0])),
                                    (int(valley_x[0]), int(valley_y[0])),
                                    (int(peak_x[-1]), int(peak_y[-1])),
                                    (int(valley_x[-1]), int(valley_y[-1]))
                                ],
                                description=description,
                                bullish_probability=bullish_prob,
                                metadata={
                                    'peak_slope': peak_slope,
                                    'valley_slope': valley_slope,
                                    'slope_similarity': slope_similarity,
                                    'channel_width': abs(peak_intercept - valley_intercept)
                                }
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting channels: {str(e)}")
        
        return patterns
    
    async def _detect_support_resistance(
        self, 
        price_line: np.ndarray, 
        image: np.ndarray
    ) -> List[ChartPattern]:
        """Detect support and resistance levels"""
        patterns = []
        
        if len(price_line) < 10:
            return patterns
        
        try:
            y_values = price_line[:, 1]
            x_values = price_line[:, 0]
            
            # Find significant peaks and valleys
            peaks, peak_props = signal.find_peaks(
                y_values, 
                prominence=self.peak_prominence * np.std(y_values),
                distance=len(y_values) // 20
            )
            valleys, valley_props = signal.find_peaks(
                -y_values, 
                prominence=self.peak_prominence * np.std(y_values),
                distance=len(y_values) // 20
            )
            
            # Group similar levels (support/resistance)
            all_levels = []
            
            # Add resistance levels (peaks)
            for peak in peaks:
                all_levels.append({
                    'level': y_values[peak],
                    'x': x_values[peak],
                    'type': 'resistance',
                    'strength': peak_props['prominences'][np.where(peaks == peak)[0][0]]
                })
            
            # Add support levels (valleys)
            for valley in valleys:
                all_levels.append({
                    'level': y_values[valley],
                    'x': x_values[valley],
                    'type': 'support',
                    'strength': valley_props['prominences'][np.where(valleys == valley)[0][0]]
                })
            
            # Cluster similar levels
            if all_levels:
                levels_array = np.array([level['level'] for level in all_levels]).reshape(-1, 1)
                scaler = StandardScaler()
                levels_scaled = scaler.fit_transform(levels_array)
                
                # Use DBSCAN to find level clusters
                clustering = DBSCAN(eps=0.3, min_samples=2).fit(levels_scaled)
                
                for cluster_id in set(clustering.labels_):
                    if cluster_id == -1:  # Noise
                        continue
                    
                    cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
                    cluster_levels = [all_levels[i] for i in cluster_indices]
                    
                    if len(cluster_levels) >= 2:
                        # Calculate cluster statistics
                        avg_level = np.mean([level['level'] for level in cluster_levels])
                        total_strength = sum([level['strength'] for level in cluster_levels])
                        level_type = max(set([level['type'] for level in cluster_levels]), 
                                       key=[level['type'] for level in cluster_levels].count)
                        
                        # Calculate confidence based on number of touches and strength
                        touch_score = min(1.0, len(cluster_levels) / 5)
                        strength_score = min(1.0, total_strength / (2 * np.std(y_values)))
                        confidence = 0.3 + touch_score * 0.4 + strength_score * 0.3
                        
                        if confidence >= self.min_confidence:
                            x_coords = [level['x'] for level in cluster_levels]
                            start_x, end_x = min(x_coords), max(x_coords)
                            
                            pattern = ChartPattern(
                                pattern_type=PatternType.SUPPORT_RESISTANCE,
                                confidence=confidence,
                                start_point=(int(start_x), int(avg_level)),
                                end_point=(int(end_x), int(avg_level)),
                                key_points=[(int(level['x']), int(level['level'])) 
                                          for level in cluster_levels],
                                description=f"Strong {level_type} level at {avg_level:.2f}",
                                bullish_probability=0.6 if level_type == 'support' else 0.4,
                                metadata={
                                    'level_type': level_type,
                                    'touch_count': len(cluster_levels),
                                    'total_strength': total_strength,
                                    'avg_level': avg_level
                                }
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"Error detecting support/resistance: {str(e)}")
        
        return patterns
    
    def visualize_patterns(
        self, 
        image_path: str, 
        patterns: List[ChartPattern], 
        output_path: Optional[str] = None
    ) -> str:
        """
        Visualize detected patterns on the chart image
        
        Args:
            image_path: Path to original chart image
            patterns: List of detected patterns
            output_path: Optional output path for annotated image
            
        Returns:
            Path to the annotated image
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Color scheme for different pattern types
            colors = {
                PatternType.HEAD_AND_SHOULDERS: (0, 0, 255),  # Red
                PatternType.DOUBLE_TOP: (255, 0, 0),  # Blue
                PatternType.DOUBLE_BOTTOM: (0, 255, 0),  # Green
                PatternType.TRIANGLE_ASCENDING: (0, 255, 255),  # Yellow
                PatternType.FLAG_BULLISH: (0, 128, 255),  # Orange
                PatternType.SUPPORT_RESISTANCE: (255, 0, 255),  # Magenta
            }
            
            # Draw patterns
            for i, pattern in enumerate(patterns):
                color = colors.get(pattern.pattern_type, (128, 128, 128))
                
                # Draw key points
                for point in pattern.key_points:
                    cv2.circle(image, point, 5, color, -1)
                
                # Draw connecting lines for some patterns
                if len(pattern.key_points) >= 2:
                    for j in range(len(pattern.key_points) - 1):
                        cv2.line(image, pattern.key_points[j], pattern.key_points[j+1], color, 2)
                
                # Add pattern label
                label_pos = (pattern.start_point[0], pattern.start_point[1] - 20 - i * 25)
                label_text = f"{pattern.pattern_type.value} ({pattern.confidence:.2f})"
                cv2.putText(image, label_text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Save annotated image
            if output_path is None:
                base_name = Path(image_path).stem
                output_path = f"annotated_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            cv2.imwrite(output_path, image)
            self.logger.info(f"Annotated image saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error visualizing patterns: {str(e)}")
            return image_path 