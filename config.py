#!/usr/bin/env python3
"""
Configuration module for AI Stock Chart Assistant
Handles application settings and constants
"""

import os
from typing import Dict, Any

class AppConfig:
    """Application configuration and constants"""
    
    # Application metadata
    APP_NAME = "AI Stock Chart Assistant"
    APP_VERSION = "1.0 MVP"
    APP_AUTHOR = "Boworn Treesinsub"
    
    # GUI Settings
    WINDOW_SIZE = "1000x700"
    MIN_WINDOW_SIZE = (800, 600)
    THEME_MODE = "dark"  # "System", "Dark", "Light"
    COLOR_THEME = "blue"  # "blue", "green", "dark-blue"
    
    # Image Settings
    SUPPORTED_FORMATS = [
        ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp"),
        ("PNG files", "*.png"),
        ("JPEG files", "*.jpg *.jpeg"),
        ("All files", "*.*")
    ]
    PREVIEW_SIZE = (180, 120)
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
    
    # API Settings
    GEMINI_MODEL = "gemini-1.5-flash"
    API_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    
    # Default prompts
    DEFAULT_QUESTION = ("Analyze this stock chart and identify any notable patterns, "
                       "trends, or technical indicators.")
    
    ANALYSIS_PROMPT_TEMPLATE = """You are an expert technical analyst. Please analyze the stock chart image and provide insights about:

1. Overall trend direction (bullish, bearish, or sideways)
2. Key support and resistance levels visible
3. Any chart patterns (triangles, head and shoulders, double tops/bottoms, etc.)
4. Technical indicators if visible (moving averages, RSI, MACD, etc.)
5. Volume patterns if shown
6. Potential price targets or levels to watch

User's specific question: {question}

Please provide a clear, structured analysis that would be helpful for educational purposes. Remember that this is for informational use only and not financial advice."""
    
    # UI Text
    DISCLAIMER_TEXT = ("⚠️ DISCLAIMER: This tool is for informational and educational purposes only. "
                      "The AI analysis should not be considered as financial advice. "
                      "Always consult with qualified financial advisors before making investment decisions.")
    
    ERROR_MESSAGES = {
        'no_image': "Please upload an image first.",
        'invalid_image': "Failed to load image. Please check the file format and try again.",
        'api_error': "Failed to connect to AI service. Please check your internet connection and API key.",
        'analysis_failed': "Analysis failed. Please try again or contact support.",
        'file_too_large': f"Image file is too large. Maximum size is {MAX_IMAGE_SIZE // (1024*1024)}MB.",
        'invalid_api_key': "Invalid API key. Please check your Google Gemini API key.",
        'rate_limit': "API rate limit exceeded. Please wait a moment and try again."
    }
    
    @classmethod
    def get_api_key(cls) -> str:
        """Get API key from environment variable"""
        return os.getenv('GEMINI_API_KEY', '')
    
    @classmethod
    def validate_image_size(cls, file_path: str) -> bool:
        """Check if image file size is within limits"""
        try:
            file_size = os.path.getsize(file_path)
            return file_size <= cls.MAX_IMAGE_SIZE
        except OSError:
            return False
    
    @classmethod
    def get_app_info(cls) -> Dict[str, Any]:
        """Get application information dictionary"""
        return {
            'name': cls.APP_NAME,
            'version': cls.APP_VERSION,
            'author': cls.APP_AUTHOR,
            'supported_formats': cls.SUPPORTED_FORMATS,
            'max_image_size_mb': cls.MAX_IMAGE_SIZE // (1024*1024)
        }
