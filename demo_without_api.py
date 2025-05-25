#!/usr/bin/env python3
"""
AI Stock Chart Assistant - Demo Mode (No API Key Required)

This demo shows the GUI interface without requiring a Gemini API key.
Perfect for testing the interface and understanding the workflow.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set demo mode environment variable
os.environ["DEMO_MODE"] = "true"
os.environ["GEMINI_API_KEY"] = "demo_key_for_testing"

try:
    import customtkinter as ctk
    from src.gui.main_window import MainWindow
    from src.core.config import get_config
    
    print("🎨 AI Stock Chart Assistant - Demo Mode")
    print("=" * 50)
    print("This demo shows the interface without requiring an API key.")
    print("You can upload images and see the GUI, but AI analysis won't work.")
    print("To enable full functionality, get a Gemini API key and set it in .env")
    print("=" * 50)
    
    # Create and run the demo application
    app = MainWindow()
    app.run()
    
except ImportError as e:
    print(f"Error: Missing dependencies - {e}")
    print("Please install dependencies: pip install -r requirements_new.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting demo: {e}")
    sys.exit(1) 