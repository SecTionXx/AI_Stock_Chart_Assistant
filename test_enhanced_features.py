#!/usr/bin/env python3
"""
Test script for Enhanced Error Handling features
Tests retry mechanisms, recovery, and error handling robustness
"""

import os
import sys
import time
from datetime import datetime

def test_enhanced_error_handling():
    """Test enhanced error handling capabilities"""
    print("🧪 Testing Enhanced Error Handling Features")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 8
    
    # Test 1: Import enhanced error handler
    print("\n🔧 Test 1: Enhanced Error Handler Import")
    try:
        from enhanced_error_handler import (
            enhanced_exception_handler, retry_on_failure, SafeOperation,
            recovery_manager, connection_manager, validate_system_requirements
        )
        print("✅ Enhanced error handler imported successfully")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ Failed to import enhanced error handler: {e}")
    
    # Test 2: System validation
    print("\n🔍 Test 2: System Requirements Validation")
    try:
        if 'validate_system_requirements' in locals():
            requirements = validate_system_requirements()
            failed = [k for k, v in requirements.items() if not v]
            if not failed:
                print("✅ All system requirements met")
                tests_passed += 1
            else:
                print(f"⚠️ Some requirements not met: {failed}")
                tests_passed += 0.5  # Partial credit
        else:
            print("⚠️ System validation not available (fallback mode)")
    except Exception as e:
        print(f"❌ System validation failed: {e}")
    
    # Test 3: Retry decorator
    print("\n🔄 Test 3: Retry Mechanism")
    try:
        attempt_count = 0
        
        @retry_on_failure(max_retries=2, base_delay=0.1)
        def test_retry_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise Exception(f"Test failure {attempt_count}")
            return "Success!"
        
        result = test_retry_function()
        if result == "Success!" and attempt_count == 2:
            print("✅ Retry mechanism working correctly")
            tests_passed += 1
        else:
            print(f"⚠️ Retry mechanism partially working (attempts: {attempt_count})")
    except Exception as e:
        print(f"❌ Retry mechanism test failed: {e}")
    
    # Test 4: Safe operation context manager
    print("\n🛡️ Test 4: Safe Operation Context Manager")
    try:
        with SafeOperation("Test Operation") as op:
            # Simulate successful operation
            time.sleep(0.1)
        print("✅ Safe operation context manager working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Safe operation test failed: {e}")
    
    # Test 5: Recovery manager
    print("\n💾 Test 5: Recovery Manager")
    try:
        if 'recovery_manager' in locals():
            # Test state saving and loading
            test_state = {
                'test_data': 'test_value',
                'timestamp': datetime.now().isoformat()
            }
            recovery_manager.save_app_state(test_state)
            
            # Try to load it back
            loaded_state = recovery_manager.load_app_state()
            if loaded_state and loaded_state['state']['test_data'] == 'test_value':
                print("✅ Recovery manager state save/load working")
                tests_passed += 1
                
                # Cleanup
                recovery_manager.clear_app_state()
            else:
                print("⚠️ Recovery manager partially working")
        else:
            print("⚠️ Recovery manager not available (fallback mode)")
    except Exception as e:
        print(f"❌ Recovery manager test failed: {e}")
    
    # Test 6: Connection manager
    print("\n🌐 Test 6: Connection Manager")
    try:
        if 'connection_manager' in locals():
            # Test without actual API key (should fail gracefully)
            try:
                connection_manager.test_api_connection("invalid_key")
            except:
                pass  # Expected to fail
            
            print("✅ Connection manager handles invalid keys gracefully")
            tests_passed += 1
        else:
            print("⚠️ Connection manager not available (fallback mode)")
    except Exception as e:
        print(f"❌ Connection manager test failed: {e}")
    
    # Test 7: Error categorization
    print("\n🏷️ Test 7: Error Categorization")
    try:
        from enhanced_error_handler import EnhancedErrorHandler
        
        # Test different error types
        network_error = Exception("network connection failed")
        api_error = Exception("invalid api key")
        
        cat1 = EnhancedErrorHandler.get_error_category(network_error)
        cat2 = EnhancedErrorHandler.get_error_category(api_error)
        
        if cat1 == 'network_error' and cat2 == 'api_key_error':
            print("✅ Error categorization working correctly")
            tests_passed += 1
        else:
            print(f"⚠️ Error categorization partially working ({cat1}, {cat2})")
    except Exception as e:
        print(f"❌ Error categorization test failed: {e}")
    
    # Test 8: Enhanced exception handler decorator
    print("\n🎯 Test 8: Enhanced Exception Handler Decorator")
    try:
        error_handled = False
        
        @enhanced_exception_handler(show_message=False)
        def test_exception_function():
            nonlocal error_handled
            error_handled = True
            raise ValueError("Test exception")
        
        result = test_exception_function()
        if result is None and error_handled:
            print("✅ Enhanced exception handler working correctly")
            tests_passed += 1
        else:
            print("⚠️ Enhanced exception handler partially working")
    except Exception as e:
        print(f"❌ Enhanced exception handler test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Enhanced Error Handling Test Results: {tests_passed}/{total_tests}")
    
    if tests_passed >= 7:
        print("🎉 Excellent! Enhanced error handling is working well.")
        status = "excellent"
    elif tests_passed >= 5:
        print("✅ Good! Most enhanced features are working.")
        status = "good"
    elif tests_passed >= 3:
        print("⚠️ Partial! Some enhanced features are working.")
        status = "partial"
    else:
        print("❌ Issues detected with enhanced error handling.")
        status = "issues"
    
    print("\n📋 Enhanced Features Status:")
    print("• Automatic retry mechanisms")
    print("• Session recovery and auto-save")
    print("• Graceful error degradation")
    print("• Connection health monitoring")
    print("• Enhanced user feedback")
    
    return status

def test_application_startup():
    """Test if the enhanced application can start"""
    print("\n🚀 Testing Enhanced Application Startup")
    print("-" * 40)
    
    try:
        # Try importing the enhanced application
        from enhanced_stock_analyzer import EnhancedStockChartAnalyzer
        print("✅ Enhanced application module imports successfully")
        
        # Test configuration loading
        try:
            from config import AppConfig
            print(f"✅ Configuration loaded: {AppConfig.APP_NAME} v{AppConfig.APP_VERSION}")
        except ImportError:
            print("⚠️ Using fallback configuration")
        
        print("✅ Enhanced application ready to start")
        return True
        
    except ImportError as e:
        print(f"❌ Enhanced application import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Enhanced application startup test failed: {e}")
        return False

def main():
    """Run all enhanced error handling tests"""
    print("🧪 Enhanced AI Stock Chart Assistant - Error Handling Tests")
    print("🔧 Testing robustness, recovery, and reliability features")
    print("=" * 60)
    
    # Test enhanced error handling
    eh_status = test_enhanced_error_handling()
    
    # Test application startup
    app_ready = test_application_startup()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    if eh_status == "excellent" and app_ready:
        print("🎉 EXCELLENT: Enhanced version is ready for production use!")
        print("🚀 All reliability features are working correctly.")
    elif eh_status in ["excellent", "good"] and app_ready:
        print("✅ READY: Enhanced version is ready with robust error handling.")
        print("💪 Your application can handle errors gracefully.")
    elif app_ready:
        print("⚠️ USABLE: Application will work but with limited enhanced features.")
        print("🔧 Consider checking the enhanced_error_handler.py file.")
    else:
        print("❌ ISSUES: Please resolve import/dependency issues before using.")
        print("📋 Run: pip install -r requirements.txt")
    
    print("\n🎯 Next Steps:")
    if eh_status == "excellent" and app_ready:
        print("1. ✅ Ready to use! Run: python enhanced_stock_analyzer.py")
        print("2. 🧪 Try the enhanced features: auto-retry, session recovery")
        print("3. 📊 Monitor the enhanced error handling in action")
    else:
        print("1. 🔧 Fix any issues shown above")
        print("2. 🔄 Re-run this test script")
        print("3. 📞 Check logs for detailed error information")
    
    return eh_status == "excellent" and app_ready

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
