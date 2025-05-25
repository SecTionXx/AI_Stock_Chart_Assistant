"""
Machine Learning modules for pattern recognition and analysis
"""

from .pattern_detector import PatternDetector, ChartPattern
from .trend_analyzer import TrendAnalyzer, TrendSignal
from .ml_models import MLModelManager, PatternClassifier

__all__ = [
    'PatternDetector',
    'ChartPattern', 
    'TrendAnalyzer',
    'TrendSignal',
    'MLModelManager',
    'PatternClassifier'
] 