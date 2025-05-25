#!/usr/bin/env python3
"""
Test script for AI Stock Chart Assistant
Tests basic functionality without requiring GUI interaction
"""

import os
import sys
from PIL import Image
import io
import base64

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("🧪 Testing dependencies...")
    
    required_modules = [
        'customtkinter',
        'PIL',
        'google.generativeai'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError:
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies satisfied!")
        return True

def test_api_key():
    """Test if API key is available"""
    print("\n🔑 Testing API key...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("✅ GEMINI_API_KEY found in environment")
        print(f"   Key preview: {api_key[:8]}...{api_key[-4:]}")
        return True
    else:
        print("⚠️ GEMINI_API_KEY not found in environment")
        print("   The app will prompt for API key at runtime")
        return False

def create_test_image():
    """Create a simple test image for testing"""
    print("\n🖼️ Creating test image...")
    
    try:
        # Create a simple test chart-like image
        from PIL import Image, ImageDraw
        
        # Create a 400x300 image with white background
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple line chart
        points = [(50, 250), (100, 200), (150, 180), (200, 220), (250, 160), (300, 140), (350, 120)]
        draw.line(points, fill='blue', width=3)
        
        # Add some grid lines
        for i in range(5):
            y = 50 + i * 50
            draw.line([(30, y), (370, y)], fill='lightgray', width=1)
        
        for i in range(8):
            x = 50 + i * 40
            draw.line([(x, 30), (x, 270)], fill='lightgray', width=1)
        
        # Add title
        draw.text((150, 10), "Test Stock Chart", fill='black')
        
        # Save test image
        test_image_path = "test_chart.png"
        img.save(test_image_path)
        print(f"✅ Test image created: {test_image_path}")
        
        return test_image_path
        
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

def test_image_loading(image_path):
    """Test image loading functionality"""
    print(f"\n📁 Testing image loading...")
    
    try:
        from PIL import Image
        
        # Load image
        img = Image.open(image_path)
        print(f"✅ Image loaded successfully")
        print(f"   Size: {img.size}")
        print(f"   Mode: {img.mode}")
        print(f"   Format: {img.format}")
        
        # Test thumbnail creation
        thumbnail = img.copy()
        thumbnail.thumbnail((180, 120), Image.Resampling.LANCZOS)
        print(f"✅ Thumbnail created: {thumbnail.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image loading failed: {e}")
        return False

def test_config_import():
    """Test configuration import"""
    print("\n⚙️ Testing configuration import...")
    
    try:
        from config import AppConfig
        print("✅ Config module imported successfully")
        print(f"   App Name: {AppConfig.APP_NAME}")
        print(f"   Version: {AppConfig.APP_VERSION}")
        return True
    except ImportError:
        print("⚠️ Config module not found - using fallback configuration")
        return False

def test_api_connection():
    """Test API connection (if API key is available)"""
    print("\n🌐 Testing API connection...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️ Skipping API test - no API key available")
        return False
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test with a simple text prompt
        response = model.generate_content("Hello! Please respond with 'API test successful'")
        
        if response and response.text:
            print("✅ API connection successful")
            print(f"   Response preview: {response.text[:50]}...")
            return True
        else:
            print("❌ API responded but with empty content")
            return False
            
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting AI Stock Chart Assistant Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Dependencies
    if test_dependencies():
        tests_passed += 1
    
    # Test 2: API Key
    if test_api_key():
        tests_passed += 1
    
    # Test 3: Config Import
    if test_config_import():
        tests_passed += 1
    
    # Test 4: Create Test Image
    test_image_path = create_test_image()
    if test_image_path:
        tests_passed += 1
        
        # Test 5: Image Loading
        if test_image_loading(test_image_path):
            tests_passed += 1
    
    # Test 6: API Connection
    if test_api_connection():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The application should work correctly.")
    elif tests_passed >= 4:
        print("⚠️ Most tests passed. The application should work with minor issues.")
    else:
        print("❌ Several tests failed. Please fix the issues before running the application.")
    
    print("\n📋 Next Steps:")
    if tests_passed < total_tests:
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Set up your API key: set GEMINI_API_KEY=your_key_here")
    print("3. Run the application: python stock_chart_analyzer.py")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
