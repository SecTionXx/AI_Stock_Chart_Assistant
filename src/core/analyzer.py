"""
AI Stock Chart Assistant - AI Analyzer Module
Production-ready AI analysis with Google Gemini Vision API integration.
"""

import base64
import asyncio
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from PIL import Image
import requests

from .config import get_config, get_logger
from .error_handler import get_error_handler, handle_exception, ErrorCategory

logger = get_logger(__name__)
error_handler = get_error_handler()

@dataclass
class AnalysisResult:
    """Structured analysis result."""
    success: bool
    analysis_text: str
    confidence_score: float
    technical_indicators: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: str
    timestamp: datetime
    processing_time: float
    image_path: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class StockChartAnalyzer:
    """AI-powered stock chart analyzer using Google Gemini Vision."""
    
    def __init__(self):
        self.config = get_config()
        self.api_config = self.config.get_api_config()
        self._initialize_api()
        
        # Analysis templates
        self.analysis_prompt = self._get_analysis_prompt()
        self.technical_prompt = self._get_technical_prompt()
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_window = 60
    
    def _initialize_api(self) -> None:
        """Initialize Google Gemini API."""
        try:
            if not self.api_config["api_key"]:
                raise ValueError("Gemini API key not configured")
            
            genai.configure(api_key=self.api_config["api_key"])
            
            # Initialize the model
            self.model = genai.GenerativeModel(self.api_config["model"])
            
            logger.info(f"Gemini API initialized with model: {self.api_config['model']}")
            
        except Exception as e:
            error_info = error_handler.handle_error(e, "API initialization")
            logger.error(f"Failed to initialize Gemini API: {error_info.user_message}")
            raise
    
    def _get_analysis_prompt(self) -> str:
        """Get the main analysis prompt template."""
        return """
        You are an expert financial analyst specializing in technical analysis of stock charts. 
        Analyze the provided stock chart image and provide a comprehensive analysis.
        
        Please provide your analysis in the following structured format:
        
        ## CHART OVERVIEW
        - Chart type and timeframe
        - Stock symbol (if visible)
        - Current price level and recent price action
        
        ## TECHNICAL ANALYSIS
        - Trend analysis (short, medium, long-term)
        - Support and resistance levels
        - Chart patterns identified
        - Volume analysis (if visible)
        - Key technical indicators visible
        
        ## MARKET SIGNALS
        - Bullish signals identified
        - Bearish signals identified
        - Neutral/consolidation patterns
        
        ## RECOMMENDATIONS
        - Short-term outlook (1-5 days)
        - Medium-term outlook (1-4 weeks)
        - Key levels to watch
        - Risk management suggestions
        
        ## CONFIDENCE ASSESSMENT
        - Analysis confidence level (1-10)
        - Factors affecting confidence
        - Additional data needed for better analysis
        
        Be specific, actionable, and professional. Focus on what can be clearly observed in the chart.
        If certain elements are unclear or not visible, mention this limitation.
        """
    
    def _get_technical_prompt(self) -> str:
        """Get the technical indicators prompt template."""
        return """
        Focus specifically on technical indicators and patterns in this stock chart:
        
        1. TREND INDICATORS:
           - Moving averages (if visible)
           - Trend lines
           - Channel patterns
        
        2. MOMENTUM INDICATORS:
           - RSI levels (if visible)
           - MACD signals (if visible)
           - Stochastic readings (if visible)
        
        3. VOLUME ANALYSIS:
           - Volume patterns
           - Volume-price relationships
           - Volume breakouts or divergences
        
        4. CHART PATTERNS:
           - Classical patterns (triangles, flags, head & shoulders, etc.)
           - Candlestick patterns (if applicable)
           - Breakout or breakdown patterns
        
        5. SUPPORT/RESISTANCE:
           - Key horizontal levels
           - Dynamic support/resistance
           - Fibonacci levels (if applicable)
        
        Provide specific price levels where possible and rate the strength of each signal.
        """
    
    @handle_exception
    def analyze_chart(self, image_path: str, analysis_type: str = "comprehensive") -> AnalysisResult:
        """
        Analyze a stock chart image.
        
        Args:
            image_path: Path to the chart image
            analysis_type: Type of analysis ("comprehensive", "technical", "quick")
        
        Returns:
            AnalysisResult object with analysis details
        """
        start_time = time.time()
        
        try:
            # Validate image
            self._validate_image(image_path)
            
            # Check rate limits
            self._check_rate_limits()
            
            # Prepare image for API
            image_data = self._prepare_image(image_path)
            
            # Select appropriate prompt
            prompt = self._get_prompt_for_analysis_type(analysis_type)
            
            # Perform analysis
            analysis_text = self._call_gemini_api(image_data, prompt)
            
            # Process and structure the response
            result = self._process_analysis_response(
                analysis_text, image_path, start_time
            )
            
            logger.info(f"Chart analysis completed successfully in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_info = error_handler.handle_error(e, f"Chart analysis: {image_path}")
            
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                success=False,
                analysis_text="",
                confidence_score=0.0,
                technical_indicators={},
                recommendations=[],
                risk_assessment="Analysis failed",
                timestamp=datetime.now(),
                processing_time=processing_time,
                image_path=image_path,
                error_message=error_info.user_message
            )
    
    def _validate_image(self, image_path: str) -> None:
        """Validate image file."""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not path.suffix.lower() in self.config.ui.supported_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.ui.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB (max: {self.config.ui.max_file_size_mb}MB)")
        
        # Validate image can be opened
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {e}")
    
    def _check_rate_limits(self) -> None:
        """Check and enforce rate limits."""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_request_time > self.rate_limit_window:
            self.request_count = 0
        
        # Check if we're within limits
        if self.request_count >= self.max_requests_per_window:
            wait_time = self.rate_limit_window - (current_time - self.last_request_time)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.request_count = 0
        
        self.request_count += 1
        self.last_request_time = current_time
    
    def _prepare_image(self, image_path: str) -> bytes:
        """Prepare image for API submission."""
        try:
            # Open and potentially resize image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (API limits)
                max_dimension = 2048
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to {new_size}")
                
                # Save to bytes
                import io
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=85)
                return img_bytes.getvalue()
                
        except Exception as e:
            raise ValueError(f"Failed to prepare image: {e}")
    
    def _get_prompt_for_analysis_type(self, analysis_type: str) -> str:
        """Get appropriate prompt based on analysis type."""
        if analysis_type == "technical":
            return self.technical_prompt
        elif analysis_type == "quick":
            return """
            Provide a quick analysis of this stock chart focusing on:
            1. Current trend direction
            2. Key support/resistance levels
            3. Short-term outlook
            4. Main risk factors
            
            Keep the analysis concise but actionable.
            """
        else:  # comprehensive
            return self.analysis_prompt
    
    def _call_gemini_api(self, image_data: bytes, prompt: str) -> str:
        """Call Gemini Vision API with retry logic."""
        def api_call():
            try:
                # Prepare the image for Gemini
                import io
                image = Image.open(io.BytesIO(image_data))
                
                # Generate content
                response = self.model.generate_content([prompt, image])
                
                if not response.text:
                    raise ValueError("Empty response from Gemini API")
                
                return response.text
                
            except Exception as e:
                if "quota" in str(e).lower():
                    raise Exception("API quota exceeded") from e
                elif "rate" in str(e).lower():
                    raise Exception("Rate limit exceeded") from e
                elif "authentication" in str(e).lower():
                    raise Exception("Authentication failed") from e
                else:
                    raise
        
        # Use error handler's retry mechanism
        try:
            return error_handler.retry_with_backoff(api_call)
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
    
    def _process_analysis_response(self, analysis_text: str, image_path: str, start_time: float) -> AnalysisResult:
        """Process and structure the analysis response."""
        processing_time = time.time() - start_time
        
        # Extract structured information from the response
        technical_indicators = self._extract_technical_indicators(analysis_text)
        recommendations = self._extract_recommendations(analysis_text)
        confidence_score = self._extract_confidence_score(analysis_text)
        risk_assessment = self._extract_risk_assessment(analysis_text)
        
        return AnalysisResult(
            success=True,
            analysis_text=analysis_text,
            confidence_score=confidence_score,
            technical_indicators=technical_indicators,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            timestamp=datetime.now(),
            processing_time=processing_time,
            image_path=image_path
        )
    
    def _extract_technical_indicators(self, text: str) -> Dict[str, Any]:
        """Extract technical indicators from analysis text."""
        indicators = {}
        
        # Look for common patterns
        text_lower = text.lower()
        
        # Trend analysis
        if "bullish" in text_lower:
            indicators["trend"] = "bullish"
        elif "bearish" in text_lower:
            indicators["trend"] = "bearish"
        else:
            indicators["trend"] = "neutral"
        
        # Support/Resistance levels
        import re
        
        # Look for price levels (basic pattern matching)
        price_pattern = r'\$?(\d+\.?\d*)'
        prices = re.findall(price_pattern, text)
        if prices:
            indicators["price_levels"] = [float(p) for p in prices[:5]]  # Top 5 levels
        
        # Volume analysis
        if "volume" in text_lower:
            if "high volume" in text_lower or "strong volume" in text_lower:
                indicators["volume"] = "high"
            elif "low volume" in text_lower or "weak volume" in text_lower:
                indicators["volume"] = "low"
            else:
                indicators["volume"] = "normal"
        
        return indicators
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from analysis text."""
        recommendations = []
        
        # Look for recommendation sections
        lines = text.split('\n')
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            
            if "recommendation" in line.lower() or "outlook" in line.lower():
                in_recommendations = True
                continue
            
            if in_recommendations and line:
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    recommendations.append(line[1:].strip())
                elif line and not line.startswith('#'):
                    recommendations.append(line)
                
                # Stop at next section
                if line.startswith('#') and len(recommendations) > 0:
                    break
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _extract_confidence_score(self, text: str) -> float:
        """Extract confidence score from analysis text."""
        import re
        
        # Look for confidence patterns
        confidence_patterns = [
            r'confidence.*?(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10',
            r'(\d+)%\s*confidence'
        ]
        
        for pattern in confidence_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    score = float(matches[0])
                    # Normalize to 0-1 scale
                    if score > 1:
                        score = score / 10 if score <= 10 else score / 100
                    return min(max(score, 0.0), 1.0)
                except ValueError:
                    continue
        
        # Default confidence based on text quality
        if len(text) > 500:
            return 0.7  # Good detailed analysis
        elif len(text) > 200:
            return 0.5  # Moderate analysis
        else:
            return 0.3  # Brief analysis
    
    def _extract_risk_assessment(self, text: str) -> str:
        """Extract risk assessment from analysis text."""
        text_lower = text.lower()
        
        # Look for risk-related keywords
        high_risk_keywords = ["high risk", "risky", "volatile", "uncertain", "caution"]
        medium_risk_keywords = ["moderate risk", "some risk", "watch", "monitor"]
        low_risk_keywords = ["low risk", "stable", "safe", "conservative"]
        
        high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in text_lower)
        medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in text_lower)
        low_risk_count = sum(1 for keyword in low_risk_keywords if keyword in text_lower)
        
        if high_risk_count > medium_risk_count and high_risk_count > low_risk_count:
            return "High Risk"
        elif low_risk_count > medium_risk_count and low_risk_count > high_risk_count:
            return "Low Risk"
        else:
            return "Medium Risk"
    
    def test_api_connection(self) -> Tuple[bool, str]:
        """Test API connection and configuration."""
        try:
            # Simple test with a minimal request
            test_prompt = "Respond with 'API connection successful' if you can see this message."
            
            # Create a simple test image
            from PIL import Image
            import io
            
            test_image = Image.new('RGB', (100, 100), color='white')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Test the API
            response = self.model.generate_content([test_prompt, test_image])
            
            if response.text:
                logger.info("API connection test successful")
                return True, "API connection successful"
            else:
                return False, "Empty response from API"
                
        except Exception as e:
            error_info = error_handler.handle_error(e, "API connection test")
            return False, error_info.user_message
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis history (placeholder for future implementation)."""
        # This would typically load from a database or file
        # For now, return empty list
        return []
    
    def export_analysis(self, result: AnalysisResult, format: str = "pdf") -> str:
        """Export analysis result to file (placeholder for future implementation)."""
        # This would generate PDF or other format exports
        # For now, return a simple text export
        
        export_dir = Path(self.config.storage.export_directory)
        export_dir.mkdir(exist_ok=True)
        
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}.txt"
        filepath = export_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Stock Chart Analysis Report\n")
            f.write(f"Generated: {result.timestamp}\n")
            f.write(f"Processing Time: {result.processing_time:.2f}s\n")
            f.write(f"Confidence Score: {result.confidence_score:.2f}\n")
            f.write(f"Risk Assessment: {result.risk_assessment}\n")
            f.write(f"\n{'-'*50}\n\n")
            f.write(result.analysis_text)
            
            if result.recommendations:
                f.write(f"\n\nKey Recommendations:\n")
                for i, rec in enumerate(result.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
        
        return str(filepath)

# Global analyzer instance
analyzer = StockChartAnalyzer()

def get_analyzer() -> StockChartAnalyzer:
    """Get the global analyzer instance."""
    return analyzer

if __name__ == "__main__":
    # Analyzer test
    print("AI Stock Chart Assistant - Analyzer Test")
    print("=" * 50)
    
    # Test API connection
    success, message = analyzer.test_api_connection()
    if success:
        print("✓ API connection test passed")
    else:
        print(f"✗ API connection test failed: {message}")
    
    print(f"✓ Analyzer initialized with model: {analyzer.api_config['model']}")
    print(f"✓ Rate limit: {analyzer.max_requests_per_window} requests per {analyzer.rate_limit_window}s") 