#!/usr/bin/env python3
"""
Enhanced Error Handler with Retry Mechanisms and Recovery
Provides robust error handling, automatic retries, and graceful degradation
"""

import logging
import traceback
import time
import json
import os
from functools import wraps
from typing import Callable, Any, Optional, Dict
from tkinter import messagebox
import google.generativeai as genai
from datetime import datetime, timedelta

# Enhanced logging configuration
class ColoredFormatter(logging.Formatter):
    """Colored log formatter for better visibility"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

# Configure enhanced logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler('stock_analyzer_detailed.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s'
))

# Console handler with colors
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(
    '%(asctime)s | %(levelname)s | %(message)s'
))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

class RetryConfig:
    """Configuration for retry mechanisms"""
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds
    MAX_DELAY = 30.0  # seconds
    BACKOFF_MULTIPLIER = 2.0
    
    # Retry-able error patterns
    RETRYABLE_ERRORS = [
        'network', 'connection', 'timeout', 'temporary', 'rate limit',
        'service unavailable', 'internal server error', 'bad gateway'
    ]

class ErrorRecoveryManager:
    """Manages error recovery and application state"""
    
    def __init__(self):
        self.error_counts = {}
        self.last_errors = {}
        self.recovery_actions = {}
        self.app_state_file = 'app_recovery_state.json'
        
    def save_app_state(self, state_data: Dict):
        """Save application state for recovery"""
        try:
            with open(self.app_state_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'state': state_data
                }, f, indent=2)
            logger.info("App state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save app state: {e}")
    
    def load_app_state(self) -> Optional[Dict]:
        """Load saved application state"""
        try:
            if os.path.exists(self.app_state_file):
                with open(self.app_state_file, 'r') as f:
                    data = json.load(f)
                    # Check if state is recent (within 24 hours)
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - timestamp < timedelta(hours=24):
                        logger.info("Recovered app state from previous session")
                        return data['state']
            return None
        except Exception as e:
            logger.error(f"Failed to load app state: {e}")
            return None
    
    def clear_app_state(self):
        """Clear saved application state"""
        try:
            if os.path.exists(self.app_state_file):
                os.remove(self.app_state_file)
        except Exception as e:
            logger.error(f"Failed to clear app state: {e}")
    
    def record_error(self, error_type: str, error_msg: str):
        """Record error for pattern analysis"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_errors[error_type] = {
            'message': error_msg,
            'timestamp': datetime.now().isoformat(),
            'count': self.error_counts[error_type]
        }
        
        # Log error patterns
        if self.error_counts[error_type] > 3:
            logger.warning(f"Recurring error pattern detected: {error_type} ({self.error_counts[error_type]} times)")
    
    def suggest_recovery_action(self, error_type: str) -> str:
        """Suggest recovery actions based on error patterns"""
        suggestions = {
            'api_error': "Try checking your API key and internet connection",
            'image_error': "Try using a different image format (PNG or JPG recommended)",
            'network_error': "Check your internet connection and try again",
            'rate_limit': "Wait a few minutes before trying again",
            'file_error': "Check file permissions and available disk space"
        }
        return suggestions.get(error_type, "Try restarting the application")

# Global error recovery manager
recovery_manager = ErrorRecoveryManager()

def retry_on_failure(
    max_retries: int = RetryConfig.MAX_RETRIES,
    base_delay: float = RetryConfig.BASE_DELAY,
    max_delay: float = RetryConfig.MAX_DELAY,
    backoff_multiplier: float = RetryConfig.BACKOFF_MULTIPLIER,
    exceptions: tuple = (Exception,)
):
    """Decorator for automatic retry with exponential backoff"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"✅ {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Check if error is retryable
                    is_retryable = any(pattern in error_msg for pattern in RetryConfig.RETRYABLE_ERRORS)
                    
                    if attempt < max_retries and is_retryable:
                        logger.warning(f"⚠️ {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        logger.info(f"🔄 Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        delay = min(delay * backoff_multiplier, max_delay)
                    else:
                        logger.error(f"❌ {func.__name__} failed permanently: {e}")
                        break
            
            # Record error for analysis
            error_type = type(last_exception).__name__
            recovery_manager.record_error(error_type, str(last_exception))
            raise last_exception
            
        return wrapper
    return decorator

class EnhancedErrorHandler:
    """Enhanced error handling with recovery mechanisms"""
    
    @staticmethod
    def get_error_category(exception: Exception) -> str:
        """Categorize errors for better handling"""
        error_str = str(exception).lower()
        error_type = type(exception).__name__.lower()
        
        if 'network' in error_str or 'connection' in error_str:
            return 'network_error'
        elif 'quota' in error_str or 'rate' in error_str:
            return 'rate_limit'
        elif 'key' in error_str and 'invalid' in error_str:
            return 'api_key_error'
        elif 'permission' in error_str or 'access' in error_str:
            return 'permission_error'
        elif 'file' in error_type or 'io' in error_type:
            return 'file_error'
        elif 'pil' in error_type or 'image' in error_type:
            return 'image_error'
        elif 'generationerror' in error_type:
            return 'api_error'
        else:
            return 'unknown_error'
    
    @staticmethod
    def get_user_friendly_message(exception: Exception) -> Dict[str, str]:
        """Get user-friendly error messages with recovery suggestions"""
        category = EnhancedErrorHandler.get_error_category(exception)
        
        messages = {
            'network_error': {
                'title': 'Connection Issue',
                'message': 'Unable to connect to the AI service. Please check your internet connection.',
                'suggestion': 'Try again in a few moments, or check your network settings.'
            },
            'rate_limit': {
                'title': 'Usage Limit Reached',
                'message': 'You\'ve reached the API usage limit for now.',
                'suggestion': 'Please wait 5-10 minutes before trying again. The free tier has rate limits.'
            },
            'api_key_error': {
                'title': 'API Key Issue',
                'message': 'There\'s a problem with your API key.',
                'suggestion': 'Please check your Google Gemini API key and make sure it\'s valid.'
            },
            'permission_error': {
                'title': 'Permission Denied',
                'message': 'The application doesn\'t have permission to access this file or folder.',
                'suggestion': 'Try running as administrator or check file permissions.'
            },
            'file_error': {
                'title': 'File Access Problem',
                'message': 'Cannot access or process the selected file.',
                'suggestion': 'Make sure the file exists and isn\'t being used by another program.'
            },
            'image_error': {
                'title': 'Image Format Issue',
                'message': 'The selected image cannot be processed.',
                'suggestion': 'Try using a PNG or JPG image, and ensure it\'s not corrupted.'
            },
            'api_error': {
                'title': 'AI Service Error',
                'message': 'The AI analysis service encountered an error.',
                'suggestion': 'This is usually temporary. Please try again in a few moments.'
            },
            'unknown_error': {
                'title': 'Unexpected Error',
                'message': f'An unexpected error occurred: {str(exception)}',
                'suggestion': 'Please try restarting the application. If the problem persists, check the log file.'
            }
        }
        
        return messages.get(category, messages['unknown_error'])

def enhanced_exception_handler(show_message: bool = True, save_state: bool = False):
    """Enhanced exception handler with recovery features"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                # Log detailed error information
                logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
                
                # Get error details
                error_info = EnhancedErrorHandler.get_user_friendly_message(e)
                category = EnhancedErrorHandler.get_error_category(e)
                
                # Record error
                recovery_manager.record_error(category, str(e))
                
                # Save application state if requested
                if save_state and hasattr(args[0], 'get_app_state'):
                    try:
                        state = args[0].get_app_state()
                        recovery_manager.save_app_state(state)
                    except:
                        pass
                
                # Show user-friendly message
                if show_message:
                    suggestion = recovery_manager.suggest_recovery_action(category)
                    full_message = f"{error_info['message']}\n\n💡 Suggestion: {error_info['suggestion']}"
                    
                    messagebox.showerror(error_info['title'], full_message)
                
                return None
            
        return wrapper
    return decorator

class ConnectionManager:
    """Manages API connections with health checks and fallbacks"""
    
    def __init__(self):
        self.connection_status = {}
        self.last_health_check = {}
        self.health_check_interval = 300  # 5 minutes
    
    @retry_on_failure(max_retries=2, base_delay=0.5)
    def test_api_connection(self, api_key: str) -> bool:
        """Test API connection with retry"""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Quick test with minimal content
            response = model.generate_content("Hello")
            
            if response and hasattr(response, 'text'):
                logger.info("✅ API connection test successful")
                self.connection_status['gemini'] = True
                self.last_health_check['gemini'] = datetime.now()
                return True
            else:
                raise Exception("Empty response from API")
                
        except Exception as e:
            logger.error(f"❌ API connection test failed: {e}")
            self.connection_status['gemini'] = False
            raise
    
    def is_connection_healthy(self, service: str = 'gemini') -> bool:
        """Check if connection is healthy"""
        if service not in self.connection_status:
            return False
        
        last_check = self.last_health_check.get(service)
        if not last_check:
            return False
        
        # Check if health check is recent
        time_since_check = datetime.now() - last_check
        return (self.connection_status[service] and 
                time_since_check.total_seconds() < self.health_check_interval)

# Global connection manager
connection_manager = ConnectionManager()

class OfflineMode:
    """Handles offline functionality and graceful degradation"""
    
    @staticmethod
    def is_online() -> bool:
        """Check if internet connection is available"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    @staticmethod
    def get_offline_message() -> str:
        """Get offline mode message"""
        return """🔌 **Offline Mode**

The AI analysis feature requires an internet connection. Here's what you can still do:

✅ **Available offline:**
• Upload and preview chart images
• View previously saved analyses
• Export saved results

❌ **Requires internet:**
• AI chart analysis
• API-based features

💡 **To get back online:**
• Check your internet connection
• Verify your API key is working
• Try again once connected"""

# Context managers for enhanced error handling
class SafeOperation:
    """Context manager for safe operations with automatic recovery"""
    
    def __init__(self, operation_name: str, critical: bool = False, save_state: bool = False):
        self.operation_name = operation_name
        self.critical = critical
        self.save_state = save_state
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"🚀 Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        
        if exc_type is None:
            logger.info(f"✅ Operation completed: {self.operation_name} ({duration.total_seconds():.2f}s)")
            return True
        else:
            logger.error(f"❌ Operation failed: {self.operation_name} - {exc_val}")
            
            # Handle critical errors differently
            if self.critical:
                error_info = EnhancedErrorHandler.get_user_friendly_message(exc_val)
                messagebox.showerror(
                    f"Critical Error in {self.operation_name}",
                    f"{error_info['message']}\n\nThe application may need to restart."
                )
            
            return False  # Don't suppress exceptions

# Utility functions for error recovery
def safe_file_operation(func: Callable, *args, **kwargs) -> tuple[bool, Any]:
    """Safely perform file operations with error handling"""
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return False, str(e)

def validate_system_requirements() -> Dict[str, bool]:
    """Validate system requirements and dependencies"""
    requirements = {}
    
    # Check Python version
    import sys
    requirements['python_version'] = sys.version_info >= (3, 8)
    
    # Check required modules
    required_modules = ['customtkinter', 'PIL', 'google.generativeai']
    for module in required_modules:
        try:
            __import__(module)
            requirements[f'module_{module}'] = True
        except ImportError:
            requirements[f'module_{module}'] = False
    
    # Check disk space (require at least 100MB)
    try:
        import shutil
        free_space = shutil.disk_usage('.').free
        requirements['disk_space'] = free_space > 100 * 1024 * 1024
    except:
        requirements['disk_space'] = True  # Assume OK if can't check
    
    # Check write permissions
    try:
        test_file = 'test_write_permission.tmp'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        requirements['write_permission'] = True
    except:
        requirements['write_permission'] = False
    
    return requirements

# Export the enhanced error handling tools
__all__ = [
    'enhanced_exception_handler',
    'retry_on_failure', 
    'SafeOperation',
    'ConnectionManager',
    'OfflineMode',
    'ErrorRecoveryManager',
    'recovery_manager',
    'connection_manager',
    'validate_system_requirements',
    'safe_file_operation'
]
