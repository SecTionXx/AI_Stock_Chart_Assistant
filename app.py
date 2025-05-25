#!/usr/bin/env python3
"""
AI Stock Chart Assistant v2.0 - Main Application Entry Point

This is the main entry point for the AI Stock Chart Assistant v2.0.
It provides both command-line interface and web dashboard options.

Usage:
    python app.py --web                    # Launch web dashboard
    python app.py --cli AAPL              # CLI analysis for AAPL
    python app.py --batch symbols.txt     # Batch analysis
    python app.py --train data.csv        # Train ML models
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Core imports
from src.core.multi_model_engine import MultiModelEngine, ConsensusResult
from src.core.cache_manager import CacheManager
from src.core.performance_optimizer import PerformanceOptimizer
from src.integrations.yahoo_finance import YahooFinanceIntegration
from src.ml.pattern_detector import PatternDetector
from src.ml.trend_analyzer import TrendAnalyzer
from src.ml.ml_models import MLModelManager
from src.ui.dashboard import StockDashboard


class AIStockAssistant:
    """
    Main application class for AI Stock Chart Assistant v2.0
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize components
        self.cache_manager = None
        self.performance_optimizer = None
        self.yahoo_finance = None
        self.multi_model_engine = None
        self.pattern_detector = None
        self.trend_analyzer = None
        self.ml_manager = None
        
        self.logger.info("AI Stock Chart Assistant v2.0 initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('ai_stock_assistant.log')
            ]
        )
        
        return logging.getLogger(__name__)
    
    async def initialize_components(self):
        """Initialize all application components"""
        try:
            self.logger.info("Initializing components...")
            
            # Core components
            self.cache_manager = CacheManager(self.config.get('cache', {}))
            self.performance_optimizer = PerformanceOptimizer(self.config.get('performance', {}))
            self.yahoo_finance = YahooFinanceIntegration(self.config.get('yahoo_finance', {}))
            
            # AI components
            self.multi_model_engine = MultiModelEngine(self.config.get('ai_models', {}))
            self.pattern_detector = PatternDetector(self.config.get('pattern_detection', {}))
            self.trend_analyzer = TrendAnalyzer(self.config.get('trend_analysis', {}))
            self.ml_manager = MLModelManager(self.config.get('ml_models', {}))
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    async def analyze_symbol(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d",
        full_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a stock symbol
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            full_analysis: Whether to run full AI analysis
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            self.logger.info(f"Starting analysis for {symbol}")
            
            # Fetch stock data
            stock_data = await self.yahoo_finance.get_stock_data(
                symbol, period=period, interval=interval
            )
            
            if stock_data is None or len(stock_data) == 0:
                raise ValueError(f"No data available for symbol {symbol}")
            
            analysis_results = {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(stock_data)
            }
            
            # Basic technical analysis
            technical_indicators = await self._calculate_technical_indicators(stock_data)
            analysis_results['technical_indicators'] = technical_indicators
            
            if full_analysis:
                # Pattern detection
                patterns = await self.pattern_detector.detect_patterns(stock_data)
                analysis_results['patterns'] = [
                    {
                        'type': p.pattern_type.value,
                        'confidence': p.confidence,
                        'bullish_probability': p.bullish_probability,
                        'start_time': str(p.start_time),
                        'end_time': str(p.end_time),
                        'key_points': len(p.key_points)
                    }
                    for p in patterns
                ]
                
                # Trend analysis
                trend_analysis = await self.trend_analyzer.analyze_trends(stock_data)
                analysis_results['trend_analysis'] = trend_analysis
                
                # ML predictions
                features = await self.ml_manager.extract_features_from_data(
                    stock_data, patterns=patterns
                )
                
                if features:
                    trend_prediction = await self.ml_manager.predict_trend(features)
                    price_prediction = await self.ml_manager.predict_price_direction(features)
                    
                    analysis_results['ml_predictions'] = {
                        'trend': {
                            'prediction': trend_prediction.prediction,
                            'confidence': trend_prediction.confidence,
                            'model': trend_prediction.model_name
                        },
                        'price_direction': {
                            'prediction': price_prediction.prediction,
                            'confidence': price_prediction.confidence,
                            'model': price_prediction.model_name
                        }
                    }
                
                # Multi-model AI consensus (if configured)
                if self.multi_model_engine and self.config.get('enable_ai_consensus', False):
                    try:
                        # Create chart image for AI analysis
                        chart_image = await self.performance_optimizer.create_chart_image(stock_data)
                        
                        # Get AI consensus
                        consensus = await self.multi_model_engine.analyze_chart_consensus(chart_image)
                        
                        analysis_results['ai_consensus'] = {
                            'direction': consensus.consensus_direction,
                            'confidence': consensus.consensus_confidence,
                            'agreement': consensus.model_agreement,
                            'individual_results': {
                                model: {
                                    'analysis': result.analysis,
                                    'confidence': result.confidence,
                                    'reasoning': result.reasoning
                                }
                                for model, result in consensus.individual_results.items()
                            }
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"AI consensus analysis failed: {str(e)}")
                        analysis_results['ai_consensus'] = {'error': str(e)}
            
            # Market context
            try:
                news = await self.yahoo_finance.get_news(symbol)
                analysis_results['news'] = news[:5]  # Top 5 news items
                
                market_data = await self.yahoo_finance.get_market_summary()
                analysis_results['market_context'] = market_data
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch market context: {str(e)}")
            
            # Performance metrics
            current_price = stock_data['close'].iloc[-1]
            prev_price = stock_data['close'].iloc[-2] if len(stock_data) > 1 else current_price
            
            analysis_results['performance'] = {
                'current_price': float(current_price),
                'previous_price': float(prev_price),
                'change': float(current_price - prev_price),
                'change_percent': float((current_price - prev_price) / prev_price * 100),
                'volume': int(stock_data['volume'].iloc[-1]),
                'high_52w': float(stock_data['high'].rolling(252).max().iloc[-1]),
                'low_52w': float(stock_data['low'].rolling(252).min().iloc[-1])
            }
            
            self.logger.info(f"Analysis completed for {symbol}")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _calculate_technical_indicators(self, data) -> Dict[str, Any]:
        """Calculate basic technical indicators"""
        try:
            indicators = {}
            
            # Simple moving averages
            indicators['sma_20'] = float(data['close'].rolling(20).mean().iloc[-1])
            indicators['sma_50'] = float(data['close'].rolling(50).mean().iloc[-1])
            
            # Price position relative to moving averages
            current_price = data['close'].iloc[-1]
            indicators['above_sma_20'] = current_price > indicators['sma_20']
            indicators['above_sma_50'] = current_price > indicators['sma_50']
            
            # Volatility
            returns = data['close'].pct_change().dropna()
            indicators['volatility_20d'] = float(returns.rolling(20).std().iloc[-1] * 100)
            
            # Volume analysis
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            indicators['volume_ratio'] = float(current_volume / avg_volume)
            
            # Price momentum
            if len(data) >= 10:
                momentum = (current_price - data['close'].iloc[-10]) / data['close'].iloc[-10] * 100
                indicators['momentum_10d'] = float(momentum)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}
    
    async def batch_analysis(self, symbols: List[str], output_file: str = None) -> List[Dict[str, Any]]:
        """
        Perform batch analysis on multiple symbols
        
        Args:
            symbols: List of stock symbols to analyze
            output_file: Optional output file path
            
        Returns:
            List of analysis results
        """
        self.logger.info(f"Starting batch analysis for {len(symbols)} symbols")
        
        results = []
        
        for i, symbol in enumerate(symbols):
            try:
                self.logger.info(f"Analyzing {symbol} ({i+1}/{len(symbols)})")
                
                result = await self.analyze_symbol(symbol, full_analysis=False)
                results.append(result)
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {symbol}: {str(e)}")
                results.append({
                    'symbol': symbol,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save results if output file specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                self.logger.info(f"Results saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save results: {str(e)}")
        
        return results
    
    async def train_models(self, training_data_file: str):
        """
        Train ML models with provided data
        
        Args:
            training_data_file: Path to training data CSV file
        """
        try:
            import pandas as pd
            
            self.logger.info(f"Training models with data from {training_data_file}")
            
            # Load training data
            training_data = pd.read_csv(training_data_file)
            
            # Train models
            performance = await self.ml_manager.train_trend_predictor(
                training_data.to_dict('records')
            )
            
            self.logger.info(f"Model training completed - Accuracy: {performance.accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise
    
    def launch_web_dashboard(self):
        """Launch the Streamlit web dashboard"""
        try:
            self.logger.info("Launching web dashboard...")
            
            # Import and run dashboard
            dashboard = StockDashboard()
            dashboard.render_dashboard()
            
        except Exception as e:
            self.logger.error(f"Failed to launch web dashboard: {str(e)}")
            raise


def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file"""
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config file {config_file}: {e}")
    
    # Default configuration
    return {
        'log_level': 'INFO',
        'cache': {
            'memory_cache_size': 1000,
            'disk_cache_ttl': 3600
        },
        'yahoo_finance': {
            'rate_limit_delay': 1.0
        },
        'ai_models': {
            'openai_model': 'gpt-4-vision-preview',
            'google_model': 'gemini-pro-vision'
        },
        'enable_ai_consensus': False  # Disabled by default due to API costs
    }


def load_symbols_from_file(file_path: str) -> List[str]:
    """Load stock symbols from text file"""
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip()]
        return symbols
    except Exception as e:
        print(f"Error loading symbols from {file_path}: {e}")
        return []


async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="AI Stock Chart Assistant v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --web                           # Launch web dashboard
  python app.py --cli AAPL                     # Analyze AAPL
  python app.py --cli AAPL --period 6mo        # Analyze AAPL with 6-month data
  python app.py --batch symbols.txt            # Batch analysis from file
  python app.py --batch AAPL,GOOGL,TSLA        # Batch analysis from list
  python app.py --train training_data.csv      # Train ML models
        """
    )
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--web', action='store_true', help='Launch web dashboard')
    group.add_argument('--cli', type=str, help='Run CLI analysis for symbol')
    group.add_argument('--batch', type=str, help='Batch analysis (file path or comma-separated symbols)')
    group.add_argument('--train', type=str, help='Train ML models with data file')
    
    # Analysis options
    parser.add_argument('--period', type=str, default='1y',
                       choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
                       help='Data period (default: 1y)')
    parser.add_argument('--interval', type=str, default='1d',
                       choices=['1m', '5m', '15m', '30m', '1h', '1d'],
                       help='Data interval (default: 1d)')
    parser.add_argument('--full', action='store_true',
                       help='Run full analysis including AI models')
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON format)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize application
    app = AIStockAssistant(config)
    
    try:
        if args.web:
            # Launch web dashboard
            app.launch_web_dashboard()
            
        else:
            # Initialize components for CLI operations
            await app.initialize_components()
            
            if args.cli:
                # Single symbol analysis
                print(f"Analyzing {args.cli}...")
                
                result = await app.analyze_symbol(
                    args.cli,
                    period=args.period,
                    interval=args.interval,
                    full_analysis=args.full
                )
                
                # Display results
                print(f"\n=== Analysis Results for {args.cli} ===")
                print(f"Current Price: ${result.get('performance', {}).get('current_price', 'N/A')}")
                print(f"Change: {result.get('performance', {}).get('change_percent', 'N/A'):.2f}%")
                
                if 'technical_indicators' in result:
                    ti = result['technical_indicators']
                    print(f"SMA 20: ${ti.get('sma_20', 'N/A'):.2f}")
                    print(f"SMA 50: ${ti.get('sma_50', 'N/A'):.2f}")
                    print(f"Volatility (20d): {ti.get('volatility_20d', 'N/A'):.2f}%")
                
                if 'patterns' in result:
                    patterns = result['patterns']
                    print(f"Patterns Detected: {len(patterns)}")
                    for pattern in patterns[:3]:  # Show top 3
                        print(f"  - {pattern['type']}: {pattern['confidence']:.1%} confidence")
                
                if 'ml_predictions' in result:
                    ml = result['ml_predictions']
                    if 'trend' in ml:
                        print(f"Trend Prediction: {ml['trend']['prediction']} ({ml['trend']['confidence']:.1%})")
                    if 'price_direction' in ml:
                        print(f"Price Direction: {ml['price_direction']['prediction']} ({ml['price_direction']['confidence']:.1%})")
                
                # Save to file if requested
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"\nResults saved to {args.output}")
            
            elif args.batch:
                # Batch analysis
                if ',' in args.batch:
                    # Comma-separated symbols
                    symbols = [s.strip().upper() for s in args.batch.split(',')]
                else:
                    # File path
                    symbols = load_symbols_from_file(args.batch)
                
                if not symbols:
                    print("No symbols to analyze")
                    return
                
                print(f"Starting batch analysis for {len(symbols)} symbols...")
                
                results = await app.batch_analysis(symbols, args.output)
                
                # Summary
                successful = len([r for r in results if 'error' not in r])
                print(f"\nBatch analysis completed: {successful}/{len(symbols)} successful")
                
                if args.output:
                    print(f"Results saved to {args.output}")
            
            elif args.train:
                # Train models
                print(f"Training models with data from {args.train}...")
                await app.train_models(args.train)
                print("Model training completed")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Handle both sync and async execution
    if sys.platform.startswith('win'):
        # Windows-specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main()) 