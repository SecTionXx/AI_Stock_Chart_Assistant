#!/usr/bin/env python3
"""
AI Stock Chart Assistant - Installation Test Script

This script verifies that all dependencies are installed correctly
and the basic functionality is working.

Run this script after installation to ensure everything is set up properly.
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version compatibility."""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        ("customtkinter", "CustomTkinter"),
        ("google.generativeai", "Google Generative AI"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("tenacity", "Tenacity"),
        ("structlog", "Structlog"),
        ("dotenv", "Python Dotenv")
    ]
    
    all_good = True
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} - OK")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            all_good = False
    
    return all_good

def test_environment():
    """Test environment configuration."""
    print("\n⚙️ Testing environment...")
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️ .env file not found (you can create one from env_example.txt)")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        if api_key.startswith("your_") or len(api_key) < 10:
            print("⚠️ GEMINI_API_KEY found but appears to be placeholder")
        else:
            print("✅ GEMINI_API_KEY configured")
    else:
        print("⚠️ GEMINI_API_KEY not found in environment")
    
    return True

def test_project_structure():
    """Test project structure."""
    print("\n📁 Testing project structure...")
    
    required_paths = [
        "src/",
        "src/core/",
        "src/gui/",
        "src/utils/",
        "src/core/config.py",
        "src/core/analyzer.py",
        "src/core/error_handler.py",
        "src/gui/main_window.py",
        "src/utils/image_handler.py",
        "main.py",
        "requirements_new.txt"
    ]
    
    all_good = True
    
    for path in required_paths:
        if Path(path).exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - MISSING")
            all_good = False
    
    return all_good

def test_imports():
    """Test importing main modules."""
    print("\n🔧 Testing module imports...")
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from src.core.config import get_config
        print("✅ Config module import - OK")
        
        from src.core.error_handler import get_error_handler
        print("✅ Error handler module import - OK")
        
        from src.utils.image_handler import get_image_processor
        print("✅ Image handler module import - OK")
        
        # Test config initialization
        config = get_config()
        print("✅ Config initialization - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        return False

def test_gui_basic():
    """Test basic GUI functionality."""
    print("\n🎨 Testing GUI basics...")
    
    try:
        import customtkinter as ctk
        
        # Test basic CustomTkinter functionality
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create a test window (don't show it)
        root = ctk.CTk()
        root.withdraw()  # Hide the window
        
        # Test creating basic widgets
        label = ctk.CTkLabel(root, text="Test")
        button = ctk.CTkButton(root, text="Test")
        
        # Clean up
        root.destroy()
        
        print("✅ GUI framework - OK")
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 AI Stock Chart Assistant - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("Environment", test_environment),
        ("Project Structure", test_project_structure),
        ("Module Imports", test_imports),
        ("GUI Framework", test_gui_basic)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("You can now run the application with: python main.py")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the issues above.")
        print("Common solutions:")
        print("  - Install dependencies: pip install -r requirements_new.txt")
        print("  - Create .env file from env_example.txt")
        print("  - Set your GEMINI_API_KEY in the .env file")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 