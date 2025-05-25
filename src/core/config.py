"""
AI Stock Chart Assistant - Configuration Module
Production-ready configuration with environment variables and validation.
"""

import os
import logging
import structlog
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """API configuration settings."""
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = "gemini-1.5-flash"
    max_retries: int = 3
    timeout_seconds: int = 30
    rate_limit_delay: float = 1.0
    
    def __post_init__(self):
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

@dataclass
class UIConfig:
    """User interface configuration."""
    window_title: str = "AI Stock Chart Assistant"
    window_width: int = 1400
    window_height: int = 900
    min_width: int = 1000
    min_height: int = 700
    theme: str = "dark"
    color_theme: str = "blue"
    
    # Layout settings
    sidebar_width: int = 300
    preview_width: int = 400
    status_bar_height: int = 30
    
    # File upload settings
    max_file_size_mb: int = 20
    supported_formats: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

@dataclass
class ErrorHandlingConfig:
    """Error handling and recovery configuration."""
    auto_retry_enabled: bool = True
    max_retry_attempts: int = 3
    retry_delay_base: float = 1.0
    retry_delay_max: float = 60.0
    exponential_backoff: bool = True
    
    # Session recovery
    auto_save_enabled: bool = True
    auto_save_interval: int = 30  # seconds
    session_backup_count: int = 5
    
    # Connection monitoring
    health_check_interval: int = 60  # seconds
    connection_timeout: int = 10

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Structured logging
    structured_logging: bool = True
    log_file_path: str = "logs/app.log"

@dataclass
class StorageConfig:
    """Data storage and persistence configuration."""
    data_directory: str = "data"
    session_directory: str = "data/sessions"
    history_directory: str = "data/history"
    export_directory: str = "exports"
    
    # History settings
    max_history_entries: int = 100
    history_retention_days: int = 30
    
    # Export settings
    default_export_format: str = "pdf"
    include_metadata: bool = True
    
    @property
    def data_dir(self) -> Path:
        """Get data directory as Path object."""
        return Path(self.data_directory)
    
    @property
    def sessions_dir(self) -> Path:
        """Get sessions directory as Path object."""
        return Path(self.session_directory)
    
    @property
    def history_dir(self) -> Path:
        """Get history directory as Path object."""
        return Path(self.history_directory)
    
    @property
    def exports_dir(self) -> Path:
        """Get exports directory as Path object."""
        return Path(self.export_directory)
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory as Path object."""
        return Path("logs")

class AppConfig:
    """Main application configuration manager."""
    
    def __init__(self):
        self.api = APIConfig()
        self.ui = UIConfig()
        self.error_handling = ErrorHandlingConfig()
        self.logging = LoggingConfig()
        self.storage = StorageConfig()
        
        # Application metadata
        self.app_name = "AI Stock Chart Assistant"
        self.version = "1.0.0"
        self.author = "AI Assistant"
        self.description = "Production-ready AI-powered stock chart analysis tool"
        
        # Initialize directories
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _create_directories(self) -> None:
        """Create necessary application directories."""
        directories = [
            self.storage.data_directory,
            self.storage.session_directory,
            self.storage.history_directory,
            self.storage.export_directory,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Configure application logging."""
        # Create logs directory
        log_path = Path(self.logging.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure structured logging
        if self.logging.structured_logging:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        
        # Configure standard logging
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=self._get_log_handlers()
        )
    
    def _get_log_handlers(self) -> list:
        """Get configured log handlers."""
        handlers = []
        
        if self.logging.console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter(self.logging.format)
            )
            handlers.append(console_handler)
        
        if self.logging.file_enabled:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.log_file_path,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(
                logging.Formatter(self.logging.format)
            )
            handlers.append(file_handler)
        
        return handlers
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration as dictionary."""
        return {
            "api_key": self.api.gemini_api_key,
            "model": self.api.gemini_model,
            "max_retries": self.api.max_retries,
            "timeout": self.api.timeout_seconds,
            "rate_limit_delay": self.api.rate_limit_delay
        }
    
    def validate_configuration(self) -> bool:
        """Validate all configuration settings."""
        try:
            # Validate API key
            if not self.api.gemini_api_key:
                raise ValueError("Gemini API key is required")
            
            # Validate file size limits
            if self.ui.max_file_size_mb <= 0:
                raise ValueError("Max file size must be positive")
            
            # Validate retry settings
            if self.error_handling.max_retry_attempts < 0:
                raise ValueError("Max retry attempts cannot be negative")
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get environment information for debugging."""
        return {
            "python_version": os.sys.version,
            "platform": os.name,
            "working_directory": os.getcwd(),
            "config_valid": str(self.validate_configuration()),
            "api_key_configured": str(bool(self.api.gemini_api_key)),
        }

# Global configuration instance
config = AppConfig()

# Convenience functions
def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config

def setup_logging() -> None:
    """Setup application logging (convenience function)."""
    # Logging is already set up in AppConfig.__init__
    pass

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    if config.logging.structured_logging:
        return structlog.get_logger(name)
    return logging.getLogger(name)

# Environment validation
def check_environment() -> bool:
    """Check if the environment is properly configured."""
    required_env_vars = ["GEMINI_API_KEY"]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required variables.")
        return False
    
    return True

if __name__ == "__main__":
    # Configuration test
    print("AI Stock Chart Assistant - Configuration Test")
    print("=" * 50)
    
    if check_environment():
        print("✓ Environment variables configured")
        
        if config.validate_configuration():
            print("✓ Configuration validation passed")
            print(f"✓ Application: {config.app_name} v{config.version}")
            print(f"✓ Theme: {config.ui.theme}")
            print(f"✓ Logging level: {config.logging.level}")
        else:
            print("✗ Configuration validation failed")
    else:
        print("✗ Environment configuration incomplete") 