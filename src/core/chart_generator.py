"""
Chart Image Generator for AI Analysis

This module creates high-quality chart images that can be analyzed by AI models.
It generates professional-looking stock charts with technical indicators,
annotations, and proper formatting for optimal AI interpretation.
"""

import asyncio
import logging
import io
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import tempfile

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set up matplotlib and seaborn styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChartGenerator:
    """
    Professional chart generator for AI analysis
    """
    
    def __init__(self, 
                 width: int = 1920, 
                 height: int = 1080,
                 dpi: int = 150,
                 style: str = "professional"):
        self.width = width
        self.height = height
        self.dpi = dpi
        self.style = style
        self.logger = logging.getLogger(__name__)
        
        # Chart styling
        self.colors = {
            'background': '#1e1e1e',
            'grid': '#333333',
            'text': '#ffffff',
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'volume': '#888888',
            'ma_short': '#ffaa00',
            'ma_long': '#00aaff',
            'rsi': '#ff00ff',
            'macd': '#00ffff'
        }
        
        # Configure plotly
        pio.kaleido.scope.mathjax = None
    
    async def generate_comprehensive_chart(self, 
                                         symbol: str,
                                         data: pd.DataFrame,
                                         technical_indicators: Optional[Dict] = None,
                                         timeframe: str = "1D",
                                         chart_type: str = "candlestick") -> bytes:
        """
        Generate a comprehensive chart with multiple panels for AI analysis
        
        Args:
            symbol: Stock symbol
            data: OHLCV data
            technical_indicators: Technical indicators data
            timeframe: Chart timeframe
            chart_type: Type of chart (candlestick, line, etc.)
            
        Returns:
            Chart image as bytes
        """
        try:
            # Create subplot figure with multiple panels
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f'{symbol} - {timeframe} Chart',
                    'Volume',
                    'RSI',
                    'MACD'
                ),
                row_heights=[0.5, 0.2, 0.15, 0.15]
            )
            
            # Main price chart
            if chart_type == "candlestick":
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=symbol,
                        increasing_line_color=self.colors['bullish'],
                        decreasing_line_color=self.colors['bearish']
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name=f'{symbol} Close',
                        line=dict(color=self.colors['bullish'], width=2)
                    ),
                    row=1, col=1
                )
            
            # Add moving averages if available
            if 'SMA_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color=self.colors['ma_short'], width=1)
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_50'],
                        mode='lines',
                        name='SMA 50',
                        line=dict(color=self.colors['ma_long'], width=1)
                    ),
                    row=1, col=1
                )
            
            # Volume chart
            colors = [self.colors['bullish'] if close >= open_price 
                     else self.colors['bearish'] 
                     for close, open_price in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # RSI chart
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color=self.colors['rsi'], width=2)
                    ),
                    row=3, col=1
                )
                
                # RSI overbought/oversold lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold", row=3, col=1)
            
            # MACD chart
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color=self.colors['macd'], width=2)
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACD_Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='orange', width=1)
                    ),
                    row=4, col=1
                )
                
                # MACD histogram
                if 'MACD_Histogram' in data.columns:
                    colors = [self.colors['bullish'] if val >= 0 else self.colors['bearish'] 
                             for val in data['MACD_Histogram']]
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['MACD_Histogram'],
                            name='Histogram',
                            marker_color=colors,
                            opacity=0.6
                        ),
                        row=4, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} Technical Analysis - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                template='plotly_dark',
                height=self.height,
                width=self.width,
                showlegend=True,
                font=dict(color=self.colors['text'], size=12),
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background']
            )
            
            # Update x-axes
            fig.update_xaxes(
                gridcolor=self.colors['grid'],
                showgrid=True,
                rangeslider_visible=False
            )
            
            # Update y-axes
            fig.update_yaxes(
                gridcolor=self.colors['grid'],
                showgrid=True
            )
            
            # Add annotations with key information
            latest_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            
            annotation_text = (
                f"Latest: ${latest_price:.2f}<br>"
                f"Change: {price_change:+.2f} ({price_change_pct:+.2f}%)<br>"
                f"Volume: {data['Volume'].iloc[-1]:,.0f}"
            )
            
            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=14, color=self.colors['text']),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor=self.colors['grid'],
                borderwidth=1
            )
            
            # Convert to image bytes
            img_bytes = fig.to_image(format="png", engine="kaleido")
            
            self.logger.info(f"Generated comprehensive chart for {symbol}")
            return img_bytes
            
        except Exception as e:
            self.logger.error(f"Error generating chart for {symbol}: {e}")
            raise
    
    async def generate_pattern_analysis_chart(self, 
                                            symbol: str,
                                            data: pd.DataFrame,
                                            patterns: List[Dict] = None) -> bytes:
        """
        Generate a chart focused on pattern analysis with annotations
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Set dark theme
            fig.patch.set_facecolor(self.colors['background'])
            ax1.set_facecolor(self.colors['background'])
            ax2.set_facecolor(self.colors['background'])
            
            # Main price chart with candlesticks
            self._plot_candlesticks(ax1, data)
            
            # Add pattern annotations
            if patterns:
                for pattern in patterns:
                    self._add_pattern_annotation(ax1, pattern, data)
            
            # Volume chart
            self._plot_volume(ax2, data)
            
            # Styling
            self._apply_chart_styling(ax1, ax2, symbol)
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, 
                       facecolor=self.colors['background'],
                       bbox_inches='tight')
            buf.seek(0)
            img_bytes = buf.getvalue()
            buf.close()
            plt.close()
            
            return img_bytes
            
        except Exception as e:
            self.logger.error(f"Error generating pattern chart for {symbol}: {e}")
            raise
    
    def _plot_candlesticks(self, ax, data):
        """Plot candlestick chart"""
        for i, (idx, row) in enumerate(data.iterrows()):
            color = self.colors['bullish'] if row['Close'] >= row['Open'] else self.colors['bearish']
            
            # Candlestick body
            body_height = abs(row['Close'] - row['Open'])
            body_bottom = min(row['Close'], row['Open'])
            
            ax.bar(i, body_height, bottom=body_bottom, 
                  color=color, alpha=0.8, width=0.6)
            
            # Wicks
            ax.plot([i, i], [row['Low'], row['High']], 
                   color=color, linewidth=1, alpha=0.8)
    
    def _plot_volume(self, ax, data):
        """Plot volume bars"""
        colors = [self.colors['bullish'] if close >= open_price 
                 else self.colors['bearish'] 
                 for close, open_price in zip(data['Close'], data['Open'])]
        
        ax.bar(range(len(data)), data['Volume'], color=colors, alpha=0.7)
    
    def _add_pattern_annotation(self, ax, pattern, data):
        """Add pattern annotation to chart"""
        # This would add pattern-specific annotations
        # Implementation depends on pattern structure
        pass
    
    def _apply_chart_styling(self, ax1, ax2, symbol):
        """Apply consistent styling to charts"""
        for ax in [ax1, ax2]:
            ax.grid(True, color=self.colors['grid'], alpha=0.3)
            ax.tick_params(colors=self.colors['text'])
            for spine in ax.spines.values():
                spine.set_color(self.colors['grid'])
        
        ax1.set_title(f'{symbol} - Pattern Analysis', 
                     color=self.colors['text'], fontsize=16, fontweight='bold')
        ax2.set_xlabel('Time', color=self.colors['text'])
        ax1.set_ylabel('Price', color=self.colors['text'])
        ax2.set_ylabel('Volume', color=self.colors['text'])
    
    async def generate_comparison_chart(self, 
                                      symbols: List[str],
                                      data_dict: Dict[str, pd.DataFrame]) -> bytes:
        """
        Generate a comparison chart for multiple stocks
        """
        try:
            fig = go.Figure()
            
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
            
            for i, symbol in enumerate(symbols):
                if symbol in data_dict:
                    data = data_dict[symbol]
                    # Normalize to percentage change from first value
                    normalized = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=normalized,
                            mode='lines',
                            name=symbol,
                            line=dict(color=colors[i % len(colors)], width=2)
                        )
                    )
            
            fig.update_layout(
                title='Stock Performance Comparison (% Change)',
                template='plotly_dark',
                height=600,
                width=self.width,
                yaxis_title='Percentage Change (%)',
                xaxis_title='Date',
                font=dict(color=self.colors['text']),
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background']
            )
            
            # Convert to bytes
            img_bytes = fig.to_image(format="png", engine="kaleido")
            return img_bytes
            
        except Exception as e:
            self.logger.error(f"Error generating comparison chart: {e}")
            raise
    
    async def save_chart_for_ai(self, 
                               chart_bytes: bytes, 
                               filename: Optional[str] = None) -> str:
        """
        Save chart image optimized for AI analysis
        
        Returns:
            Path to saved image file
        """
        try:
            if filename is None:
                filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Create charts directory if it doesn't exist
            charts_dir = Path("charts")
            charts_dir.mkdir(exist_ok=True)
            
            filepath = charts_dir / filename
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(chart_bytes)
            
            # Optimize for AI analysis (ensure good contrast, size, etc.)
            await self._optimize_for_ai(filepath)
            
            self.logger.info(f"Chart saved for AI analysis: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving chart: {e}")
            raise
    
    async def _optimize_for_ai(self, filepath: Path):
        """
        Optimize image for AI analysis
        """
        try:
            with Image.open(filepath) as img:
                # Ensure RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (AI models have size limits)
                max_size = (1920, 1080)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Enhance contrast slightly for better AI recognition
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.1)
                
                # Save optimized image
                img.save(filepath, 'PNG', optimize=True, quality=95)
                
        except Exception as e:
            self.logger.error(f"Error optimizing image for AI: {e}")
    
    def encode_image_for_api(self, image_path: str) -> str:
        """
        Encode image to base64 for API transmission
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error encoding image: {e}")
            raise 