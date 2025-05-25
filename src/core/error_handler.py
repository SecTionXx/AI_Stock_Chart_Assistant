"""
AI Stock Chart Assistant - Advanced Error Handling Module
Production-ready error handling with retry mechanisms, session recovery, and user guidance.
"""

import time
import json
import logging
import traceback
import threading
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Union, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)

from .config import get_config, get_logger

logger = get_logger(__name__)

class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    NETWORK = "network"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTHENTICATION = "api_authentication"
    API_QUOTA = "api_quota"
    FILE_VALIDATION = "file_validation"
    FILE_SIZE = "file_size"
    FILE_FORMAT = "file_format"
    SYSTEM_RESOURCE = "system_resource"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION = "configuration"
    USER_INPUT = "user_input"
    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    suggested_action: str
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    can_retry: bool = True
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class SessionState:
    """Session state management for recovery."""
    
    def __init__(self):
        self.config = get_config()
        self.session_file = Path(self.config.storage.session_directory) / "current_session.json"
        self.backup_files = []
        self._lock = threading.Lock()
    
    def save_state(self, state_data: Dict[str, Any]) -> bool:
        """Save current session state."""
        try:
            with self._lock:
                # Add timestamp
                state_data["timestamp"] = datetime.now().isoformat()
                state_data["version"] = self.config.version
                
                # Create backup of current session
                if self.session_file.exists():
                    backup_file = self._create_backup()
                    if backup_file:
                        self.backup_files.append(backup_file)
                
                # Save new state
                with open(self.session_file, 'w') as f:
                    json.dump(state_data, f, indent=2, default=str)
                
                # Cleanup old backups
                self._cleanup_backups()
                
                logger.info("Session state saved successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            return False
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load saved session state."""
        try:
            if not self.session_file.exists():
                return None
            
            with open(self.session_file, 'r') as f:
                state_data = json.load(f)
            
            # Validate state data
            if self._validate_state(state_data):
                logger.info("Session state loaded successfully")
                return state_data
            else:
                logger.warning("Invalid session state, attempting recovery from backup")
                return self._recover_from_backup()
                
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return self._recover_from_backup()
    
    def _create_backup(self) -> Optional[Path]:
        """Create backup of current session."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.session_file.parent / f"session_backup_{timestamp}.json"
            
            import shutil
            shutil.copy2(self.session_file, backup_file)
            
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to create session backup: {e}")
            return None
    
    def _cleanup_backups(self) -> None:
        """Remove old backup files."""
        try:
            max_backups = self.config.error_handling.session_backup_count
            
            # Get all backup files sorted by creation time
            backup_pattern = self.session_file.parent / "session_backup_*.json"
            backup_files = sorted(
                Path(self.session_file.parent).glob("session_backup_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Remove excess backups
            for backup_file in backup_files[max_backups:]:
                backup_file.unlink()
                logger.debug(f"Removed old backup: {backup_file}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup backups: {e}")
    
    def _validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate session state data."""
        required_fields = ["timestamp", "version"]
        
        for field in required_fields:
            if field not in state_data:
                return False
        
        # Check if state is not too old
        try:
            timestamp = datetime.fromisoformat(state_data["timestamp"])
            max_age = timedelta(days=7)  # Sessions older than 7 days are invalid
            
            if datetime.now() - timestamp > max_age:
                return False
                
        except (ValueError, TypeError):
            return False
        
        return True
    
    def _recover_from_backup(self) -> Optional[Dict[str, Any]]:
        """Attempt to recover from backup files."""
        backup_pattern = self.session_file.parent / "session_backup_*.json"
        backup_files = sorted(
            Path(self.session_file.parent).glob("session_backup_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for backup_file in backup_files:
            try:
                with open(backup_file, 'r') as f:
                    state_data = json.load(f)
                
                if self._validate_state(state_data):
                    logger.info(f"Recovered session from backup: {backup_file}")
                    return state_data
                    
            except Exception as e:
                logger.warning(f"Failed to recover from backup {backup_file}: {e}")
                continue
        
        logger.warning("No valid backup found for session recovery")
        return None

class ErrorHandler:
    """Advanced error handling with retry mechanisms and user guidance."""
    
    def __init__(self):
        self.config = get_config()
        self.session_state = SessionState()
        self.error_history: List[ErrorInfo] = []
        self.connection_status = {"api": True, "network": True}
        self._monitoring_active = False
        
        # Error classification patterns
        self.error_patterns = {
            ErrorCategory.NETWORK: [
                "connection", "timeout", "network", "unreachable", "dns"
            ],
            ErrorCategory.API_RATE_LIMIT: [
                "rate limit", "too many requests", "quota exceeded"
            ],
            ErrorCategory.API_AUTHENTICATION: [
                "authentication", "unauthorized", "invalid api key", "forbidden"
            ],
            ErrorCategory.FILE_VALIDATION: [
                "invalid file", "corrupted", "unsupported format"
            ],
            ErrorCategory.FILE_SIZE: [
                "file too large", "size limit", "maximum size"
            ],
            ErrorCategory.SYSTEM_RESOURCE: [
                "memory", "disk space", "resource", "out of space"
            ]
        }
    
    def classify_error(self, error: Exception, context: str = "") -> ErrorCategory:
        """Classify error based on type and message."""
        error_text = f"{str(error)} {context}".lower()
        
        for category, patterns in self.error_patterns.items():
            if any(pattern in error_text for pattern in patterns):
                return category
        
        # Check exception types
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, PermissionError):
            return ErrorCategory.SYSTEM_RESOURCE
        elif isinstance(error, ValueError):
            return ErrorCategory.USER_INPUT
        
        return ErrorCategory.UNKNOWN
    
    def get_error_severity(self, category: ErrorCategory, retry_count: int = 0) -> ErrorSeverity:
        """Determine error severity based on category and retry count."""
        if retry_count >= 3:
            return ErrorSeverity.HIGH
        
        severity_map = {
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.API_RATE_LIMIT: ErrorSeverity.LOW,
            ErrorCategory.API_AUTHENTICATION: ErrorSeverity.HIGH,
            ErrorCategory.API_QUOTA: ErrorSeverity.HIGH,
            ErrorCategory.FILE_VALIDATION: ErrorSeverity.LOW,
            ErrorCategory.FILE_SIZE: ErrorSeverity.LOW,
            ErrorCategory.SYSTEM_RESOURCE: ErrorSeverity.HIGH,
            ErrorCategory.CONFIGURATION: ErrorSeverity.CRITICAL,
            ErrorCategory.USER_INPUT: ErrorSeverity.LOW,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM
        }
        
        return severity_map.get(category, ErrorSeverity.MEDIUM)
    
    def get_user_friendly_message(self, category: ErrorCategory, error: Exception) -> tuple:
        """Get user-friendly error message and suggested action."""
        messages = {
            ErrorCategory.NETWORK: (
                "Connection issue detected. Please check your internet connection.",
                "Verify your internet connection and try again. The app will automatically retry."
            ),
            ErrorCategory.API_RATE_LIMIT: (
                "API rate limit reached. Please wait a moment.",
                "The service is temporarily busy. Please wait 30 seconds and try again."
            ),
            ErrorCategory.API_AUTHENTICATION: (
                "API authentication failed. Please check your API key.",
                "Go to Settings and verify your Gemini API key is correct."
            ),
            ErrorCategory.FILE_VALIDATION: (
                "The selected file appears to be invalid or corrupted.",
                "Please select a different image file (PNG, JPG, or JPEG format)."
            ),
            ErrorCategory.FILE_SIZE: (
                "The selected file is too large.",
                f"Please select an image smaller than {self.config.ui.max_file_size_mb}MB."
            ),
            ErrorCategory.SYSTEM_RESOURCE: (
                "System resource issue detected.",
                "Close other applications to free up memory and try again."
            ),
            ErrorCategory.CONFIGURATION: (
                "Configuration error detected.",
                "Please check your settings and restart the application."
            ),
            ErrorCategory.USER_INPUT: (
                "Invalid input provided.",
                "Please check your input and try again."
            ),
            ErrorCategory.UNKNOWN: (
                "An unexpected error occurred.",
                "Please try again. If the problem persists, restart the application."
            )
        }
        
        return messages.get(category, messages[ErrorCategory.UNKNOWN])
    
    def handle_error(self, error: Exception, context: str = "", 
                    retry_count: int = 0) -> ErrorInfo:
        """Handle error with classification and user guidance."""
        category = self.classify_error(error, context)
        severity = self.get_error_severity(category, retry_count)
        user_message, suggested_action = self.get_user_friendly_message(category, error)
        
        error_info = ErrorInfo(
            category=category,
            severity=severity,
            message=str(error),
            user_message=user_message,
            suggested_action=suggested_action,
            timestamp=datetime.now(),
            retry_count=retry_count,
            max_retries=self.config.error_handling.max_retry_attempts,
            can_retry=self._can_retry(category, retry_count),
            context={"original_context": context, "traceback": traceback.format_exc()}
        )
        
        # Log error
        self._log_error(error_info)
        
        # Add to history
        self.error_history.append(error_info)
        
        # Update connection status
        self._update_connection_status(category)
        
        return error_info
    
    def _can_retry(self, category: ErrorCategory, retry_count: int) -> bool:
        """Determine if error can be retried."""
        if retry_count >= self.config.error_handling.max_retry_attempts:
            return False
        
        non_retryable = {
            ErrorCategory.API_AUTHENTICATION,
            ErrorCategory.FILE_VALIDATION,
            ErrorCategory.FILE_SIZE,
            ErrorCategory.CONFIGURATION,
            ErrorCategory.USER_INPUT
        }
        
        return category not in non_retryable
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error with appropriate level."""
        log_data = {
            "category": error_info.category.value,
            "severity": error_info.severity.value,
            "message": error_info.message,
            "retry_count": error_info.retry_count,
            "can_retry": error_info.can_retry
        }
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", extra=log_data)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error("High severity error", extra=log_data)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error", extra=log_data)
        else:
            logger.info("Low severity error", extra=log_data)
    
    def _update_connection_status(self, category: ErrorCategory) -> None:
        """Update connection status based on error category."""
        if category in [ErrorCategory.NETWORK, ErrorCategory.API_RATE_LIMIT]:
            self.connection_status["network"] = False
            self.connection_status["api"] = False
        elif category in [ErrorCategory.API_AUTHENTICATION, ErrorCategory.API_QUOTA]:
            self.connection_status["api"] = False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Retry attempt failed: {e}")
            raise
    
    def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitor():
            while self._monitoring_active:
                try:
                    # Check API connectivity
                    self._check_api_health()
                    
                    # Check system resources
                    self._check_system_health()
                    
                    time.sleep(self.config.error_handling.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(30)  # Wait before retrying
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logger.info("Health monitoring started")
    
    def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._monitoring_active = False
        logger.info("Health monitoring stopped")
    
    def _check_api_health(self) -> None:
        """Check API connectivity."""
        try:
            # Simple connectivity check (implement based on your API)
            import requests
            response = requests.get("https://www.google.com", timeout=5)
            
            if response.status_code == 200:
                self.connection_status["network"] = True
                self.connection_status["api"] = True
            else:
                self.connection_status["network"] = False
                
        except Exception:
            self.connection_status["network"] = False
            self.connection_status["api"] = False
    
    def _check_system_health(self) -> None:
        """Check system resource health."""
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                logger.warning(f"Low disk space: {disk.percent}% used")
                
        except ImportError:
            logger.debug("psutil not available for system monitoring")
        except Exception as e:
            logger.error(f"System health check failed: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        recent_errors = [
            error for error in self.error_history
            if datetime.now() - error.timestamp < timedelta(hours=1)
        ]
        
        category_counts = {}
        for error in recent_errors:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "category_breakdown": category_counts,
            "connection_status": self.connection_status,
            "last_error": recent_errors[-1].message if recent_errors else None
        }
    
    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        logger.info("Error history cleared")
    
    def cleanup(self) -> None:
        """Cleanup error handler resources."""
        self.stop_health_monitoring()
        self.clear_error_history()
        logger.info("Error handler cleanup completed")

# Global error handler instance
error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return error_handler

def handle_exception(func: Callable) -> Callable:
    """Decorator for automatic exception handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_info = error_handler.handle_error(e, f"Function: {func.__name__}")
            logger.error(f"Exception in {func.__name__}: {error_info.user_message}")
            raise
    
    return wrapper

if __name__ == "__main__":
    # Error handler test
    print("AI Stock Chart Assistant - Error Handler Test")
    print("=" * 50)
    
    # Test error classification
    test_errors = [
        ConnectionError("Network connection failed"),
        ValueError("Invalid input provided"),
        PermissionError("Access denied"),
        Exception("Unknown error occurred")
    ]
    
    for error in test_errors:
        error_info = error_handler.handle_error(error, "Test context")
        print(f"Error: {error}")
        print(f"Category: {error_info.category.value}")
        print(f"Severity: {error_info.severity.value}")
        print(f"User Message: {error_info.user_message}")
        print(f"Can Retry: {error_info.can_retry}")
        print("-" * 30) 