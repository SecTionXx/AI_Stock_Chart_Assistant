#!/usr/bin/env python3
"""
AI Stock Chart Assistant - Main Application Entry Point

A production-ready AI-powered stock chart analysis tool with enhanced error handling,
session recovery, and professional user interface.

Features:
- Google Gemini Vision API integration for chart analysis
- Professional CustomTkinter GUI with three-column layout
- Comprehensive error handling and retry mechanisms
- Session auto-save and recovery
- Analysis history and export capabilities
- Real-time status monitoring and health checks

Author: AI Assistant
Version: 1.0.0
"""

import sys
import os
import logging
from pathlib import Path
import traceback

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.gui.main_window import MainWindow
    from src.core.config import get_config, setup_logging
    from src.core.error_handler import get_error_handler, ErrorCategory
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements_new.txt")
    sys.exit(1)


def setup_environment():
    """Setup the application environment and check requirements."""
    try:
        # Initialize configuration
        config = get_config()
        
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting AI Stock Chart Assistant v1.0.0")
        
        # Check for required environment variables
        required_env_vars = ["GEMINI_API_KEY"]
        missing_vars = []
        
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            print("\n❌ Missing Required Environment Variables:")
            print("Please create a .env file in the project root with:")
            for var in missing_vars:
                print(f"  {var}=your_api_key_here")
            print("\nExample .env file:")
            print("GEMINI_API_KEY=your_google_gemini_api_key")
            print("\nGet your API key from: https://makersuite.google.com/app/apikey")
            return False
            
        # Check if directories exist (they should be created by config)
        required_dirs = [
            config.storage.data_dir,
            config.storage.sessions_dir,
            config.storage.history_dir,
            config.storage.exports_dir,
            config.storage.logs_dir
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                logger.warning(f"Directory does not exist: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
                
        logger.info("Environment setup completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        ("customtkinter", "CustomTkinter"),
        ("google.generativeai", "Google Generative AI"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("tenacity", "Tenacity"),
        ("structlog", "Structlog"),
        ("dotenv", "Python Dotenv")
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
            
    if missing_packages:
        print("\n❌ Missing Required Dependencies:")
        print("Please install the missing packages:")
        print("pip install -r requirements_new.txt")
        print(f"\nMissing: {', '.join(missing_packages)}")
        return False
        
    return True


def main():
    """Main application entry point."""
    print("🚀 AI Stock Chart Assistant v1.0.0")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ Dependencies OK")
    
    # Setup environment
    print("⚙️ Setting up environment...")
    if not setup_environment():
        sys.exit(1)
    print("✅ Environment OK")
    
    try:
        # Initialize error handler
        error_handler = get_error_handler()
        
        # Create and run the main application
        print("🎨 Initializing GUI...")
        app = MainWindow()
        
        print("✅ Application ready!")
        print("📊 Starting AI Stock Chart Assistant...")
        print("\nKeyboard shortcuts:")
        print("  Ctrl+O: Open image file")
        print("  Ctrl+S: Save session")
        print("  F5: Refresh analysis")
        print("  Ctrl+Q: Quit application")
        print("\n" + "=" * 50)
        
        # Start the application
        app.run()
        
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        
        # Log the error if possible
        try:
            logger = logging.getLogger(__name__)
            logger.error(f"Fatal application error: {e}", exc_info=True)
            
            # Try to use error handler
            error_handler = get_error_handler()
            error_handler.handle_error(
                e, "Fatal application startup error"
            )
        except:
            pass  # Don't let error handling crash the error reporting
            
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup
        try:
            print("🧹 Cleaning up...")
            error_handler = get_error_handler()
            error_handler.cleanup()
        except:
            pass


if __name__ == "__main__":
    main() 