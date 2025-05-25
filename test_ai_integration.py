#!/usr/bin/env python3
"""
AI Integration Test Script

This script tests the AI integration capabilities including:
1. Chart image generation
2. Multi-model consensus engine
3. Cost controls and caching
4. API key validation

Run this after setting up your API keys in .env file.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append('src')

from src.integrations.yahoo_finance import YahooFinanceIntegration
from src.core.chart_generator import ChartGenerator
from src.core.multi_model_engine import MultiModelEngine
from src.core.cache_manager import CacheManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIIntegrationTester:
    """Test AI integration capabilities"""
    
    def __init__(self):
        self.yahoo = YahooFinanceIntegration()
        self.chart_gen = ChartGenerator()
        self.cache_manager = CacheManager()
        self.ai_engine = None
        
    async def test_environment_setup(self):
        """Test if environment variables are properly set"""
        logger.info("🔧 Testing Environment Setup...")
        
        # Check for .env file
        env_file = Path('.env')
        if not env_file.exists():
            logger.warning("❌ .env file not found. Creating template...")
            await self.create_env_template()
            return False
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.error("❌ python-dotenv not installed. Run: pip install python-dotenv")
            return False
        
        # Check API keys
        openai_key = os.getenv('OPENAI_API_KEY', '')
        gemini_key = os.getenv('GOOGLE_API_KEY', '')
        
        if openai_key == 'your-openai-key-here' or not openai_key:
            logger.warning("⚠️  OpenAI API key not set in .env file")
            return False
            
        if gemini_key == 'your-gemini-key-here' or not gemini_key:
            logger.warning("⚠️  Google Gemini API key not set in .env file")
            return False
        
        logger.info("✅ Environment setup complete")
        return True
    
    async def create_env_template(self):
        """Create .env template file"""
        template = '''# AI Stock Chart Assistant v2.0 - Environment Configuration
# Fill in your actual API keys below

# =============================================================================
# AI MODEL API KEYS (Required for AI features)
# =============================================================================

# OpenAI API Key (for GPT-4V)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-openai-key-here

# Google Gemini API Key (for Gemini Vision)
# Get from: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your-gemini-key-here

# =============================================================================
# AI MODEL CONFIGURATION
# =============================================================================

# OpenAI Model Selection
OPENAI_MODEL=gpt-4-vision-preview

# Gemini Model Selection
GEMINI_MODEL=gemini-pro-vision

# Enable/Disable AI Consensus (set to false to use basic analysis only)
ENABLE_AI_CONSENSUS=true

# =============================================================================
# COST CONTROLS & LIMITS
# =============================================================================

# Maximum daily spending on AI API calls (in USD)
MAX_DAILY_AI_COST=10.00

# Maximum cost per single AI request (in USD)
MAX_REQUEST_COST=0.50

# Enable AI response caching (recommended)
CACHE_AI_RESPONSES=true

# AI cache time-to-live in seconds (24 hours = 86400)
AI_CACHE_TTL=86400

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Maximum number of concurrent AI requests
MAX_CONCURRENT_AI_REQUESTS=3

# Request timeout in seconds
AI_REQUEST_TIMEOUT=30

# Enable request batching for multiple stocks
ENABLE_AI_BATCHING=true

# =============================================================================
# LOGGING & MONITORING
# =============================================================================

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Enable AI usage monitoring
MONITOR_AI_USAGE=true

# Enable performance metrics
ENABLE_METRICS=true
'''
        
        with open('.env', 'w') as f:
            f.write(template)
        
        logger.info("📝 Created .env template file. Please fill in your API keys.")
    
    async def test_chart_generation(self, symbol: str = "AAPL"):
        """Test chart image generation"""
        logger.info(f"📊 Testing Chart Generation for {symbol}...")
        
        try:
            # Get stock data
            data = await self.yahoo.get_stock_data(symbol, period="1mo")
            if data is None or data.empty:
                logger.error(f"❌ Failed to get data for {symbol}")
                return False
            
            # Add technical indicators
            data = await self.yahoo.add_technical_indicators(data)
            
            # Generate comprehensive chart
            chart_bytes = await self.chart_gen.generate_comprehensive_chart(
                symbol=symbol,
                data=data,
                timeframe="1D",
                chart_type="candlestick"
            )
            
            # Save chart for AI analysis
            chart_path = await self.chart_gen.save_chart_for_ai(
                chart_bytes, 
                f"test_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            logger.info(f"✅ Chart generated successfully: {chart_path}")
            logger.info(f"📏 Chart size: {len(chart_bytes):,} bytes")
            
            return chart_path
            
        except Exception as e:
            logger.error(f"❌ Chart generation failed: {e}")
            return False
    
    async def test_ai_models(self, chart_path: str, symbol: str = "AAPL"):
        """Test AI model integration"""
        logger.info("🤖 Testing AI Model Integration...")
        
        # Check if API keys are set
        env_ready = await self.test_environment_setup()
        if not env_ready:
            logger.warning("⚠️  Skipping AI model test - API keys not configured")
            return False
        
        try:
            # Initialize AI engine
            self.ai_engine = MultiModelEngine()
            
            # Test individual models
            await self.test_openai_model(chart_path, symbol)
            await self.test_gemini_model(chart_path, symbol)
            
            # Test consensus analysis
            await self.test_consensus_analysis(chart_path, symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ AI model test failed: {e}")
            return False
    
    async def test_openai_model(self, chart_path: str, symbol: str):
        """Test OpenAI GPT-4V model"""
        logger.info("🔍 Testing OpenAI GPT-4V...")
        
        try:
            # Encode image
            image_b64 = self.chart_gen.encode_image_for_api(chart_path)
            
            # Test analysis
            result = await self.ai_engine.analyze_with_openai(
                image_data=image_b64,
                symbol=symbol,
                prompt="Analyze this stock chart and provide technical analysis insights."
            )
            
            if result and result.get('success'):
                logger.info("✅ OpenAI analysis successful")
                logger.info(f"💰 Cost: ${result.get('cost', 0):.4f}")
                logger.info(f"📝 Analysis preview: {result.get('analysis', '')[:100]}...")
            else:
                logger.warning("⚠️  OpenAI analysis returned no results")
                
        except Exception as e:
            logger.error(f"❌ OpenAI test failed: {e}")
    
    async def test_gemini_model(self, chart_path: str, symbol: str):
        """Test Google Gemini model"""
        logger.info("🔍 Testing Google Gemini...")
        
        try:
            # Test analysis
            result = await self.ai_engine.analyze_with_gemini(
                image_path=chart_path,
                symbol=symbol,
                prompt="Analyze this stock chart and provide technical analysis insights."
            )
            
            if result and result.get('success'):
                logger.info("✅ Gemini analysis successful")
                logger.info(f"💰 Cost: ${result.get('cost', 0):.4f}")
                logger.info(f"📝 Analysis preview: {result.get('analysis', '')[:100]}...")
            else:
                logger.warning("⚠️  Gemini analysis returned no results")
                
        except Exception as e:
            logger.error(f"❌ Gemini test failed: {e}")
    
    async def test_consensus_analysis(self, chart_path: str, symbol: str):
        """Test multi-model consensus analysis"""
        logger.info("🎯 Testing Multi-Model Consensus...")
        
        try:
            # Run consensus analysis
            result = await self.ai_engine.get_consensus_analysis(
                image_path=chart_path,
                symbol=symbol,
                prompt="Provide comprehensive technical analysis of this stock chart."
            )
            
            if result and result.get('success'):
                logger.info("✅ Consensus analysis successful")
                logger.info(f"🎯 Consensus Score: {result.get('consensus_score', 0):.2f}")
                logger.info(f"🤝 Agreement Level: {result.get('agreement_level', 'Unknown')}")
                logger.info(f"💰 Total Cost: ${result.get('total_cost', 0):.4f}")
                logger.info(f"📊 Models Used: {len(result.get('individual_results', []))}")
                
                # Show consensus insights
                consensus = result.get('consensus_analysis', {})
                if consensus:
                    logger.info(f"📈 Trend: {consensus.get('trend', 'Unknown')}")
                    logger.info(f"🎯 Confidence: {consensus.get('confidence', 'Unknown')}")
                    logger.info(f"💡 Key Insights: {consensus.get('key_insights', 'None')[:100]}...")
            else:
                logger.warning("⚠️  Consensus analysis returned no results")
                
        except Exception as e:
            logger.error(f"❌ Consensus analysis failed: {e}")
    
    async def test_caching_system(self):
        """Test AI response caching"""
        logger.info("💾 Testing Caching System...")
        
        try:
            # Test cache operations
            test_key = "test_ai_response"
            test_data = {
                "analysis": "Test analysis result",
                "timestamp": datetime.now().isoformat(),
                "cost": 0.05
            }
            
            # Store in cache
            await self.cache_manager.set(test_key, test_data, ttl=3600)
            
            # Retrieve from cache
            cached_data = await self.cache_manager.get(test_key)
            
            if cached_data and cached_data.get('analysis') == test_data['analysis']:
                logger.info("✅ Caching system working correctly")
                
                # Test cache stats
                stats = await self.cache_manager.get_stats()
                logger.info(f"📊 Cache Stats: {stats}")
                
                return True
            else:
                logger.error("❌ Caching system failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cache test failed: {e}")
            return False
    
    async def run_comprehensive_test(self):
        """Run all AI integration tests"""
        logger.info("🚀 Starting AI Integration Test Suite...")
        logger.info("=" * 60)
        
        results = {
            'environment': False,
            'chart_generation': False,
            'ai_models': False,
            'caching': False
        }
        
        # Test 1: Environment Setup
        results['environment'] = await self.test_environment_setup()
        
        # Test 2: Chart Generation
        chart_path = await self.test_chart_generation("AAPL")
        results['chart_generation'] = bool(chart_path)
        
        # Test 3: Caching System
        results['caching'] = await self.test_caching_system()
        
        # Test 4: AI Models (only if environment is ready)
        if results['environment'] and chart_path:
            results['ai_models'] = await self.test_ai_models(chart_path, "AAPL")
        
        # Print results summary
        logger.info("=" * 60)
        logger.info("🎯 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        total_passed = sum(results.values())
        total_tests = len(results)
        
        logger.info(f"\n📊 Overall: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            logger.info("🎉 All tests passed! AI integration is ready!")
        elif results['environment']:
            logger.info("⚠️  Some tests failed, but basic functionality works")
        else:
            logger.info("❌ Environment setup required. Please configure API keys in .env file")
        
        return results

async def main():
    """Main test function"""
    tester = AIIntegrationTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main()) 