"""
Professional Dashboard Interface for AI Stock Chart Assistant v2.0

This module provides:
- Real-time stock data dashboard
- Interactive chart visualization
- AI analysis results display
- Performance metrics and alerts
- Portfolio management interface
"""

import asyncio
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import asdict

# Import our core modules
from ..core.multi_model_engine import MultiModelEngine, ConsensusResult
from ..core.cache_manager import CacheManager
from ..core.performance_optimizer import PerformanceOptimizer
from ..integrations.yahoo_finance import YahooFinanceIntegration
from ..ml.pattern_detector import PatternDetector, ChartPattern
from ..ml.trend_analyzer import TrendAnalyzer, TrendDirection
from ..ml.ml_models import MLModelManager, PredictionResult


class DashboardConfig:
    """Dashboard configuration settings"""
    
    def __init__(self):
        self.page_title = "AI Stock Chart Assistant v2.0"
        self.page_icon = "📈"
        self.layout = "wide"
        self.sidebar_state = "expanded"
        
        # Chart settings
        self.chart_height = 600
        self.chart_theme = "plotly_dark"
        
        # Update intervals (seconds)
        self.real_time_update_interval = 30
        self.cache_refresh_interval = 300
        
        # Display settings
        self.max_patterns_display = 10
        self.max_news_items = 5
        self.confidence_threshold = 0.7


class StockDashboard:
    """
    Main dashboard class for the AI Stock Chart Assistant
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.multi_model_engine = None
        self.cache_manager = None
        self.performance_optimizer = None
        self.yahoo_finance = None
        self.pattern_detector = None
        self.trend_analyzer = None
        self.ml_manager = None
        
        # Dashboard state
        self.current_symbol = "AAPL"
        self.current_timeframe = "1d"
        self.current_period = "1y"
        self.auto_refresh = False
        
        # Initialize session state
        self._init_session_state()
        
        self.logger.info("StockDashboard initialized")
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.symbol = self.current_symbol
            st.session_state.timeframe = self.current_timeframe
            st.session_state.period = self.current_period
            st.session_state.auto_refresh = self.auto_refresh
            st.session_state.last_update = None
            st.session_state.analysis_cache = {}
            st.session_state.alerts = []
    
    async def initialize_components(self):
        """Initialize all dashboard components"""
        try:
            # Initialize core components
            self.cache_manager = CacheManager()
            self.performance_optimizer = PerformanceOptimizer()
            self.yahoo_finance = YahooFinanceIntegration()
            
            # Initialize AI components
            self.multi_model_engine = MultiModelEngine()
            self.pattern_detector = PatternDetector()
            self.trend_analyzer = TrendAnalyzer()
            self.ml_manager = MLModelManager()
            
            self.logger.info("Dashboard components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing dashboard components: {str(e)}")
            st.error(f"Failed to initialize dashboard: {str(e)}")
    
    def render_dashboard(self):
        """Render the main dashboard interface"""
        try:
            # Configure page
            st.set_page_config(
                page_title=self.config.page_title,
                page_icon=self.config.page_icon,
                layout=self.config.layout,
                initial_sidebar_state=self.config.sidebar_state
            )
            
            # Custom CSS
            self._inject_custom_css()
            
            # Header
            self._render_header()
            
            # Sidebar
            self._render_sidebar()
            
            # Main content
            if st.session_state.get('components_initialized', False):
                self._render_main_content()
            else:
                self._render_initialization_screen()
                
        except Exception as e:
            self.logger.error(f"Error rendering dashboard: {str(e)}")
            st.error(f"Dashboard error: {str(e)}")
    
    def _inject_custom_css(self):
        """Inject custom CSS for styling"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            margin-bottom: 1rem;
        }
        
        .alert-success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        
        .alert-warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        
        .alert-danger {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 0.75rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        
        .pattern-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        .confidence-high { color: #28a745; font-weight: bold; }
        .confidence-medium { color: #ffc107; font-weight: bold; }
        .confidence-low { color: #dc3545; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">🤖 AI Stock Chart Assistant v2.0</h1>
            <p style="color: white; margin: 0; opacity: 0.8;">
                Advanced AI-powered stock analysis with multi-model consensus and real-time insights
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("📊 Analysis Controls")
            
            # Stock symbol input
            symbol = st.text_input(
                "Stock Symbol",
                value=st.session_state.symbol,
                help="Enter stock ticker symbol (e.g., AAPL, GOOGL, TSLA)"
            ).upper()
            
            # Timeframe selection
            timeframe = st.selectbox(
                "Timeframe",
                options=["1m", "5m", "15m", "30m", "1h", "1d"],
                index=5,  # Default to 1d
                help="Chart timeframe for analysis"
            )
            
            # Period selection
            period = st.selectbox(
                "Period",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=5,  # Default to 1y
                help="Historical data period"
            )
            
            # Auto-refresh toggle
            auto_refresh = st.checkbox(
                "Auto Refresh",
                value=st.session_state.auto_refresh,
                help="Automatically refresh data every 30 seconds"
            )
            
            # Update session state
            if (symbol != st.session_state.symbol or 
                timeframe != st.session_state.timeframe or 
                period != st.session_state.period):
                st.session_state.symbol = symbol
                st.session_state.timeframe = timeframe
                st.session_state.period = period
                st.session_state.analysis_cache = {}  # Clear cache on change
            
            st.session_state.auto_refresh = auto_refresh
            
            # Analysis buttons
            st.markdown("---")
            st.subheader("🔍 Analysis Actions")
            
            if st.button("🚀 Run Full Analysis", type="primary"):
                st.session_state.run_analysis = True
            
            if st.button("📈 Quick Analysis"):
                st.session_state.run_quick_analysis = True
            
            if st.button("🔄 Refresh Data"):
                st.session_state.refresh_data = True
            
            # Model settings
            st.markdown("---")
            st.subheader("🤖 AI Model Settings")
            
            use_consensus = st.checkbox(
                "Multi-Model Consensus",
                value=True,
                help="Use multiple AI models for consensus analysis"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=self.config.confidence_threshold,
                step=0.1,
                help="Minimum confidence for displaying predictions"
            )
            
            # Performance metrics
            st.markdown("---")
            st.subheader("📊 Performance")
            
            if st.session_state.get('last_update'):
                st.write(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
            
            # Component status
            self._render_component_status()
    
    def _render_component_status(self):
        """Render component initialization status"""
        st.markdown("**Component Status:**")
        
        components = {
            "Cache Manager": self.cache_manager is not None,
            "Yahoo Finance": self.yahoo_finance is not None,
            "Multi-Model AI": self.multi_model_engine is not None,
            "Pattern Detector": self.pattern_detector is not None,
            "ML Models": self.ml_manager is not None
        }
        
        for component, status in components.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {component}")
    
    def _render_initialization_screen(self):
        """Render initialization screen"""
        st.info("🔄 Initializing AI components...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize components
        if st.button("Initialize Dashboard"):
            asyncio.run(self._initialize_with_progress(progress_bar, status_text))
    
    async def _initialize_with_progress(self, progress_bar, status_text):
        """Initialize components with progress display"""
        try:
            components = [
                ("Cache Manager", lambda: CacheManager()),
                ("Performance Optimizer", lambda: PerformanceOptimizer()),
                ("Yahoo Finance", lambda: YahooFinanceIntegration()),
                ("Multi-Model Engine", lambda: MultiModelEngine()),
                ("Pattern Detector", lambda: PatternDetector()),
                ("Trend Analyzer", lambda: TrendAnalyzer()),
                ("ML Manager", lambda: MLModelManager())
            ]
            
            for i, (name, initializer) in enumerate(components):
                status_text.text(f"Initializing {name}...")
                
                if name == "Cache Manager":
                    self.cache_manager = initializer()
                elif name == "Performance Optimizer":
                    self.performance_optimizer = initializer()
                elif name == "Yahoo Finance":
                    self.yahoo_finance = initializer()
                elif name == "Multi-Model Engine":
                    self.multi_model_engine = initializer()
                elif name == "Pattern Detector":
                    self.pattern_detector = initializer()
                elif name == "Trend Analyzer":
                    self.trend_analyzer = initializer()
                elif name == "ML Manager":
                    self.ml_manager = initializer()
                
                progress_bar.progress((i + 1) / len(components))
                await asyncio.sleep(0.1)  # Small delay for visual effect
            
            st.session_state.components_initialized = True
            status_text.text("✅ All components initialized successfully!")
            st.success("Dashboard ready! Please refresh the page.")
            
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
    
    def _render_main_content(self):
        """Render main dashboard content"""
        # Check for analysis triggers
        if st.session_state.get('run_analysis'):
            asyncio.run(self._run_full_analysis())
            st.session_state.run_analysis = False
        
        if st.session_state.get('run_quick_analysis'):
            asyncio.run(self._run_quick_analysis())
            st.session_state.run_quick_analysis = False
        
        if st.session_state.get('refresh_data'):
            asyncio.run(self._refresh_data())
            st.session_state.refresh_data = False
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            self._handle_auto_refresh()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Chart Analysis", 
            "🤖 AI Insights", 
            "📊 Technical Analysis",
            "📰 Market Intelligence",
            "⚙️ Settings"
        ])
        
        with tab1:
            self._render_chart_analysis()
        
        with tab2:
            self._render_ai_insights()
        
        with tab3:
            self._render_technical_analysis()
        
        with tab4:
            self._render_market_intelligence()
        
        with tab5:
            self._render_settings()
    
    def _render_chart_analysis(self):
        """Render chart analysis tab"""
        st.header(f"📈 {st.session_state.symbol} Chart Analysis")
        
        # Get cached data or fetch new
        chart_data = st.session_state.analysis_cache.get('chart_data')
        
        if chart_data is None:
            with st.spinner("Loading chart data..."):
                chart_data = asyncio.run(self._fetch_chart_data())
                st.session_state.analysis_cache['chart_data'] = chart_data
        
        if chart_data is not None:
            # Create interactive chart
            fig = self._create_interactive_chart(chart_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Chart metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = chart_data['close'].iloc[-1]
                prev_price = chart_data['close'].iloc[-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
                
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{change:+.2f} ({change_pct:+.2f}%)"
                )
            
            with col2:
                volume = chart_data['volume'].iloc[-1]
                avg_volume = chart_data['volume'].rolling(20).mean().iloc[-1]
                volume_ratio = volume / avg_volume
                
                st.metric(
                    "Volume",
                    f"{volume:,.0f}",
                    f"{volume_ratio:.2f}x avg"
                )
            
            with col3:
                high_52w = chart_data['high'].rolling(252).max().iloc[-1]
                low_52w = chart_data['low'].rolling(252).min().iloc[-1]
                position = (current_price - low_52w) / (high_52w - low_52w) * 100
                
                st.metric(
                    "52W Position",
                    f"{position:.1f}%",
                    f"${low_52w:.2f} - ${high_52w:.2f}"
                )
            
            with col4:
                volatility = chart_data['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
                st.metric(
                    "Volatility (20d)",
                    f"{volatility:.1f}%"
                )
        else:
            st.error("Failed to load chart data. Please check the symbol and try again.")
    
    def _render_ai_insights(self):
        """Render AI insights tab"""
        st.header("🤖 AI-Powered Insights")
        
        # Get AI analysis results
        ai_results = st.session_state.analysis_cache.get('ai_analysis')
        
        if ai_results is None:
            st.info("Run analysis to see AI insights")
            return
        
        # Consensus analysis
        if 'consensus' in ai_results:
            self._render_consensus_analysis(ai_results['consensus'])
        
        # Pattern recognition
        if 'patterns' in ai_results:
            self._render_pattern_analysis(ai_results['patterns'])
        
        # Trend analysis
        if 'trends' in ai_results:
            self._render_trend_analysis(ai_results['trends'])
        
        # ML predictions
        if 'ml_predictions' in ai_results:
            self._render_ml_predictions(ai_results['ml_predictions'])
    
    def _render_consensus_analysis(self, consensus: ConsensusResult):
        """Render multi-model consensus analysis"""
        st.subheader("🎯 Multi-Model Consensus")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Consensus summary
            confidence_class = self._get_confidence_class(consensus.consensus_confidence)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Overall Assessment</h4>
                <p><strong>Direction:</strong> {consensus.consensus_direction}</p>
                <p><strong>Confidence:</strong> <span class="{confidence_class}">{consensus.consensus_confidence:.1%}</span></p>
                <p><strong>Agreement:</strong> {consensus.model_agreement:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual model results
            st.markdown("**Individual Model Results:**")
            for model_name, result in consensus.individual_results.items():
                confidence_class = self._get_confidence_class(result.confidence)
                st.markdown(f"""
                - **{model_name}**: {result.analysis} 
                  <span class="{confidence_class}">({result.confidence:.1%})</span>
                """, unsafe_allow_html=True)
        
        with col2:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = consensus.consensus_confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Consensus Confidence"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_pattern_analysis(self, patterns: List[ChartPattern]):
        """Render pattern analysis results"""
        st.subheader("🔍 Pattern Recognition")
        
        if not patterns:
            st.info("No significant patterns detected")
            return
        
        # Filter patterns by confidence
        high_conf_patterns = [p for p in patterns if p.confidence > 0.7]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            for pattern in high_conf_patterns[:self.config.max_patterns_display]:
                confidence_class = self._get_confidence_class(pattern.confidence)
                
                st.markdown(f"""
                <div class="pattern-card">
                    <h5>{pattern.pattern_type.value.replace('_', ' ').title()}</h5>
                    <p><strong>Confidence:</strong> <span class="{confidence_class}">{pattern.confidence:.1%}</span></p>
                    <p><strong>Bullish Probability:</strong> {pattern.bullish_probability:.1%}</p>
                    <p><strong>Time Range:</strong> {pattern.start_time} to {pattern.end_time}</p>
                    <p><strong>Key Points:</strong> {len(pattern.key_points)} identified</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Pattern distribution chart
            pattern_counts = {}
            for pattern in patterns:
                pattern_type = pattern.pattern_type.value
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            if pattern_counts:
                fig = px.pie(
                    values=list(pattern_counts.values()),
                    names=list(pattern_counts.keys()),
                    title="Pattern Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_trend_analysis(self, trends: Dict[str, Any]):
        """Render trend analysis results"""
        st.subheader("📈 Trend Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            short_trend = trends.get('short_term', {})
            direction = short_trend.get('direction', 'Unknown')
            strength = short_trend.get('strength', 0)
            
            st.markdown(f"""
            <div class="metric-card">
                <h5>Short Term (5-20 days)</h5>
                <p><strong>Direction:</strong> {direction}</p>
                <p><strong>Strength:</strong> {strength:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            medium_trend = trends.get('medium_term', {})
            direction = medium_trend.get('direction', 'Unknown')
            strength = medium_trend.get('strength', 0)
            
            st.markdown(f"""
            <div class="metric-card">
                <h5>Medium Term (20-50 days)</h5>
                <p><strong>Direction:</strong> {direction}</p>
                <p><strong>Strength:</strong> {strength:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            long_trend = trends.get('long_term', {})
            direction = long_trend.get('direction', 'Unknown')
            strength = long_trend.get('strength', 0)
            
            st.markdown(f"""
            <div class="metric-card">
                <h5>Long Term (50+ days)</h5>
                <p><strong>Direction:</strong> {direction}</p>
                <p><strong>Strength:</strong> {strength:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_ml_predictions(self, predictions: Dict[str, PredictionResult]):
        """Render ML model predictions"""
        st.subheader("🧠 Machine Learning Predictions")
        
        for pred_type, prediction in predictions.items():
            confidence_class = self._get_confidence_class(prediction.confidence)
            
            st.markdown(f"""
            <div class="metric-card">
                <h5>{pred_type.replace('_', ' ').title()}</h5>
                <p><strong>Prediction:</strong> {prediction.prediction}</p>
                <p><strong>Confidence:</strong> <span class="{confidence_class}">{prediction.confidence:.1%}</span></p>
                <p><strong>Model:</strong> {prediction.model_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probabilities if available
            if prediction.probabilities:
                prob_df = pd.DataFrame(
                    list(prediction.probabilities.items()),
                    columns=['Outcome', 'Probability']
                )
                prob_df['Probability'] = prob_df['Probability'] * 100
                
                fig = px.bar(
                    prob_df,
                    x='Outcome',
                    y='Probability',
                    title=f"{pred_type} Probabilities"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_technical_analysis(self):
        """Render technical analysis tab"""
        st.header("📊 Technical Analysis")
        
        # Get technical indicators
        chart_data = st.session_state.analysis_cache.get('chart_data')
        
        if chart_data is None:
            st.info("Load chart data first to see technical analysis")
            return
        
        # Calculate technical indicators
        indicators = self._calculate_technical_indicators(chart_data)
        
        # Display indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Momentum Indicators")
            
            # RSI
            rsi = indicators.get('rsi', [])
            if len(rsi) > 0:
                current_rsi = rsi[-1]
                rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
                
                st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)
                
                # RSI chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='blue')
                ))
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig.update_layout(title="RSI (14)", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # MACD
            macd_data = indicators.get('macd', {})
            if macd_data:
                macd = macd_data.get('macd', [])
                signal = macd_data.get('signal', [])
                histogram = macd_data.get('histogram', [])
                
                if len(macd) > 0:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                      subplot_titles=('MACD Line', 'MACD Histogram'))
                    
                    fig.add_trace(go.Scatter(y=macd, name='MACD', line=dict(color='blue')), row=1, col=1)
                    fig.add_trace(go.Scatter(y=signal, name='Signal', line=dict(color='red')), row=1, col=1)
                    fig.add_trace(go.Bar(y=histogram, name='Histogram'), row=2, col=1)
                    
                    fig.update_layout(title="MACD", height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Trend Indicators")
            
            # Moving Averages
            sma_20 = indicators.get('sma_20', [])
            sma_50 = indicators.get('sma_50', [])
            
            if len(sma_20) > 0 and len(sma_50) > 0:
                current_price = chart_data['close'].iloc[-1]
                
                st.metric("SMA 20", f"${sma_20[-1]:.2f}", 
                         "Above" if current_price > sma_20[-1] else "Below")
                st.metric("SMA 50", f"${sma_50[-1]:.2f}", 
                         "Above" if current_price > sma_50[-1] else "Below")
                
                # MA crossover signal
                if sma_20[-1] > sma_50[-1]:
                    st.success("🟢 Bullish MA Crossover")
                else:
                    st.error("🔴 Bearish MA Crossover")
            
            # Bollinger Bands
            bb_data = indicators.get('bollinger_bands', {})
            if bb_data:
                upper = bb_data.get('upper', [])
                middle = bb_data.get('middle', [])
                lower = bb_data.get('lower', [])
                
                if len(upper) > 0:
                    current_price = chart_data['close'].iloc[-1]
                    bb_position = (current_price - lower[-1]) / (upper[-1] - lower[-1])
                    
                    st.metric("BB Position", f"{bb_position:.1%}")
                    
                    if bb_position > 0.8:
                        st.warning("⚠️ Near Upper Band")
                    elif bb_position < 0.2:
                        st.warning("⚠️ Near Lower Band")
                    else:
                        st.info("ℹ️ Within Normal Range")
    
    def _render_market_intelligence(self):
        """Render market intelligence tab"""
        st.header("📰 Market Intelligence")
        
        # Market news
        news_data = st.session_state.analysis_cache.get('news_data')
        
        if news_data is None:
            with st.spinner("Loading market news..."):
                news_data = asyncio.run(self._fetch_market_news())
                st.session_state.analysis_cache['news_data'] = news_data
        
        if news_data:
            st.subheader("📰 Latest News")
            
            for i, article in enumerate(news_data[:self.config.max_news_items]):
                with st.expander(f"{article.get('title', 'No Title')}"):
                    st.write(f"**Source:** {article.get('source', 'Unknown')}")
                    st.write(f"**Published:** {article.get('published', 'Unknown')}")
                    st.write(article.get('summary', 'No summary available'))
                    
                    if article.get('url'):
                        st.markdown(f"[Read Full Article]({article['url']})")
        
        # Market sentiment
        st.subheader("📊 Market Sentiment")
        
        sentiment_data = st.session_state.analysis_cache.get('sentiment_data')
        
        if sentiment_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Sentiment", sentiment_data.get('overall', 'Neutral'))
            
            with col2:
                st.metric("News Sentiment", sentiment_data.get('news', 'Neutral'))
            
            with col3:
                st.metric("Social Sentiment", sentiment_data.get('social', 'Neutral'))
    
    def _render_settings(self):
        """Render settings tab"""
        st.header("⚙️ Dashboard Settings")
        
        # API Configuration
        st.subheader("🔑 API Configuration")
        
        with st.expander("AI Model Settings"):
            st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
            st.text_input("Google API Key", type="password", help="Enter your Google API key")
            
            st.selectbox("Primary Model", ["gpt-4-vision-preview", "gemini-pro-vision"])
            st.selectbox("Fallback Model", ["gpt-3.5-turbo", "gemini-pro"])
        
        # Performance Settings
        st.subheader("⚡ Performance Settings")
        
        cache_ttl = st.slider("Cache TTL (minutes)", 1, 60, 15)
        max_concurrent = st.slider("Max Concurrent Requests", 1, 10, 3)
        
        # Display Settings
        st.subheader("🎨 Display Settings")
        
        chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_dark", "plotly_white"])
        chart_height = st.slider("Chart Height", 400, 800, 600)
        
        # Export Settings
        st.subheader("💾 Export Settings")
        
        if st.button("Export Analysis Results"):
            self._export_analysis_results()
        
        if st.button("Export Chart Data"):
            self._export_chart_data()
    
    def _create_interactive_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create interactive candlestick chart with indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume', 'RSI'),
            row_width=[0.2, 0.1, 0.1]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=data.index, y=data['volume'], name='Volume'),
            row=2, col=1
        )
        
        # RSI
        if 'rsi' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['rsi'], name='RSI'),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title=f"{st.session_state.symbol} - {st.session_state.timeframe}",
            height=self.config.chart_height,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            import talib
            
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # RSI
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd'] = {
                'macd': macd,
                'signal': macd_signal,
                'histogram': macd_hist
            }
            
            # Moving Averages
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            indicators['bollinger_bands'] = {
                'upper': bb_upper,
                'middle': bb_middle,
                'lower': bb_lower
            }
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            indicators['stochastic'] = {
                'k': stoch_k,
                'd': stoch_d
            }
            
        except ImportError:
            self.logger.warning("TA-Lib not available, using simplified indicators")
            # Simplified indicators without TA-Lib
            indicators['sma_20'] = data['close'].rolling(20).mean()
            indicators['sma_50'] = data['close'].rolling(50).mean()
        
        return indicators
    
    def _get_confidence_class(self, confidence: float) -> str:
        """Get CSS class for confidence level"""
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.6:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def _handle_auto_refresh(self):
        """Handle auto-refresh functionality"""
        if st.session_state.get('last_update'):
            time_since_update = datetime.now() - st.session_state.last_update
            if time_since_update.total_seconds() > self.config.real_time_update_interval:
                st.session_state.refresh_data = True
    
    async def _fetch_chart_data(self) -> Optional[pd.DataFrame]:
        """Fetch chart data for current symbol"""
        try:
            if self.yahoo_finance:
                data = await self.yahoo_finance.get_stock_data(
                    st.session_state.symbol,
                    period=st.session_state.period,
                    interval=st.session_state.timeframe
                )
                return data
        except Exception as e:
            self.logger.error(f"Error fetching chart data: {str(e)}")
        return None
    
    async def _fetch_market_news(self) -> List[Dict[str, Any]]:
        """Fetch market news for current symbol"""
        try:
            if self.yahoo_finance:
                news = await self.yahoo_finance.get_news(st.session_state.symbol)
                return news
        except Exception as e:
            self.logger.error(f"Error fetching news: {str(e)}")
        return []
    
    async def _run_full_analysis(self):
        """Run comprehensive analysis"""
        try:
            with st.spinner("Running full AI analysis..."):
                # Get chart data
                chart_data = await self._fetch_chart_data()
                if chart_data is None:
                    st.error("Failed to fetch chart data")
                    return
                
                st.session_state.analysis_cache['chart_data'] = chart_data
                
                # Run AI analysis
                ai_results = {}
                
                # Multi-model consensus
                if self.multi_model_engine:
                    # Convert chart to image for AI analysis
                    chart_image = await self.performance_optimizer.create_chart_image(chart_data)
                    consensus = await self.multi_model_engine.analyze_chart_consensus(chart_image)
                    ai_results['consensus'] = consensus
                
                # Pattern detection
                if self.pattern_detector:
                    patterns = await self.pattern_detector.detect_patterns(chart_data)
                    ai_results['patterns'] = patterns
                
                # Trend analysis
                if self.trend_analyzer:
                    trends = await self.trend_analyzer.analyze_trends(chart_data)
                    ai_results['trends'] = trends
                
                # ML predictions
                if self.ml_manager:
                    features = await self.ml_manager.extract_features_from_data(chart_data)
                    
                    predictions = {}
                    predictions['trend'] = await self.ml_manager.predict_trend(features)
                    predictions['price_direction'] = await self.ml_manager.predict_price_direction(features)
                    
                    ai_results['ml_predictions'] = predictions
                
                st.session_state.analysis_cache['ai_analysis'] = ai_results
                st.session_state.last_update = datetime.now()
                
                st.success("✅ Full analysis completed!")
                
        except Exception as e:
            self.logger.error(f"Error in full analysis: {str(e)}")
            st.error(f"Analysis failed: {str(e)}")
    
    async def _run_quick_analysis(self):
        """Run quick analysis"""
        try:
            with st.spinner("Running quick analysis..."):
                # Get chart data
                chart_data = await self._fetch_chart_data()
                if chart_data is None:
                    st.error("Failed to fetch chart data")
                    return
                
                st.session_state.analysis_cache['chart_data'] = chart_data
                
                # Quick technical analysis
                indicators = self._calculate_technical_indicators(chart_data)
                
                # Simple trend analysis
                sma_20 = indicators.get('sma_20')
                sma_50 = indicators.get('sma_50')
                
                if sma_20 is not None and sma_50 is not None:
                    current_price = chart_data['close'].iloc[-1]
                    
                    trend_signal = "Bullish" if current_price > sma_20.iloc[-1] > sma_50.iloc[-1] else "Bearish"
                    
                    st.success(f"✅ Quick analysis: {trend_signal} trend detected")
                
                st.session_state.last_update = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error in quick analysis: {str(e)}")
            st.error(f"Quick analysis failed: {str(e)}")
    
    async def _refresh_data(self):
        """Refresh all cached data"""
        try:
            with st.spinner("Refreshing data..."):
                # Clear cache
                st.session_state.analysis_cache = {}
                
                # Refresh chart data
                chart_data = await self._fetch_chart_data()
                if chart_data is not None:
                    st.session_state.analysis_cache['chart_data'] = chart_data
                
                # Refresh news
                news_data = await self._fetch_market_news()
                if news_data:
                    st.session_state.analysis_cache['news_data'] = news_data
                
                st.session_state.last_update = datetime.now()
                st.success("✅ Data refreshed!")
                
        except Exception as e:
            self.logger.error(f"Error refreshing data: {str(e)}")
            st.error(f"Data refresh failed: {str(e)}")
    
    def _export_analysis_results(self):
        """Export analysis results to JSON"""
        try:
            analysis_data = st.session_state.analysis_cache.get('ai_analysis', {})
            
            if analysis_data:
                # Convert to JSON-serializable format
                export_data = {}
                for key, value in analysis_data.items():
                    if hasattr(value, '__dict__'):
                        export_data[key] = asdict(value)
                    else:
                        export_data[key] = value
                
                json_str = json.dumps(export_data, indent=2, default=str)
                
                st.download_button(
                    label="Download Analysis Results",
                    data=json_str,
                    file_name=f"{st.session_state.symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No analysis results to export")
                
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    def _export_chart_data(self):
        """Export chart data to CSV"""
        try:
            chart_data = st.session_state.analysis_cache.get('chart_data')
            
            if chart_data is not None:
                csv_str = chart_data.to_csv()
                
                st.download_button(
                    label="Download Chart Data",
                    data=csv_str,
                    file_name=f"{st.session_state.symbol}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No chart data to export")
                
        except Exception as e:
            st.error(f"Export failed: {str(e)}")


def main():
    """Main function to run the dashboard"""
    dashboard = StockDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main() 