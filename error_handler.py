#!/usr/bin/env python3
"""
Error handling utilities for AI Stock Chart Assistant
Provides consistent error handling and user feedback
"""

import logging
import traceback
from functools import wraps
from typing import Callable, Any
from tkinter import messagebox
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analyzer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockAnalyzerError(Exception):
    """Base exception for Stock Analyzer application"""
    pass

class ImageError(StockAnalyzerError):
    """Raised when there's an issue with image processing"""
    pass

class APIError(StockAnalyzerError):
    """Raised when there's an issue with API calls"""
    pass

class ConfigurationError(StockAnalyzerError):
    """Raised when there's a configuration issue"""
    pass

def handle_exceptions(show_message: bool = True):
    """Decorator for handling exceptions with user feedback"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = get_user_friendly_error(e)
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                
                if show_message:
                    messagebox.showerror("Error", error_msg)
                
                return None
        return wrapper
    return decorator

def get_user_friendly_error(exception: Exception) -> str:
    """Convert technical exceptions to user-friendly messages"""
    
    # Handle specific exception types
    if isinstance(exception, FileNotFoundError):
        return "The selected file could not be found. Please try selecting the file again."
    
    elif isinstance(exception, PermissionError):
        return "Permission denied accessing the file. Please check file permissions."
    
    elif isinstance(exception, (IOError, OSError)):
        return "File system error. Please check the file and try again."
    
    # Handle PIL/Image errors
    elif "PIL" in str(type(exception)) or "Image" in str(type(exception)):
        return "Invalid or corrupted image file. Please select a valid image file."
    
    # Handle Google API errors
    elif isinstance(exception, genai.types.GenerationError):
        return "AI analysis service encountered an error. Please try again."
    
    elif "quota" in str(exception).lower() or "rate" in str(exception).lower():
        return "API rate limit exceeded. Please wait a moment and try again."
    
    elif "key" in str(exception).lower() and "invalid" in str(exception).lower():
        return "Invalid API key. Please check your Google Gemini API key."
    
    elif "network" in str(exception).lower() or "connection" in str(exception).lower():
        return "Network connection error. Please check your internet connection."
    
    # Handle custom application errors
    elif isinstance(exception, ImageError):
        return f"Image processing error: {str(exception)}"
    
    elif isinstance(exception, APIError):
        return f"API service error: {str(exception)}"
    
    elif isinstance(exception, ConfigurationError):
        return f"Configuration error: {str(exception)}"
    
    # Generic error for unknown exceptions
    else:
        return f"An unexpected error occurred: {str(exception)}"

def log_error(error: Exception, context: str = ""):
    """Log error with context information"""
    error_details = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'traceback': traceback.format_exc()
    }
    
    logger.error(f"Error occurred - {error_details}")

def validate_api_response(response) -> bool:
    """Validate API response and raise appropriate errors"""
    if not response:
        raise APIError("Empty response from API service")
    
    if not hasattr(response, 'text') or not response.text:
        raise APIError("Invalid response format from API service")
    
    return True

def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Any]:
    """Safely execute a function and return success status with result"""
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        log_error(e, f"Safe execution of {func.__name__}")
        return False, get_user_friendly_error(e)

# Context manager for error handling
class ErrorContext:
    """Context manager for consistent error handling"""
    
    def __init__(self, operation_name: str, show_errors: bool = True):
        self.operation_name = operation_name
        self.show_errors = show_errors
    
    def __enter__(self):
        logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = get_user_friendly_error(exc_val)
            log_error(exc_val, self.operation_name)
            
            if self.show_errors:
                messagebox.showerror("Error", f"{self.operation_name} failed: {error_msg}")
            
            return False  # Don't suppress the exception
        else:
            logger.info(f"Completed operation: {self.operation_name}")
            return True
