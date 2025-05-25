"""
Multi-Model Analysis Engine for AI Stock Chart Assistant v2.0

This module provides consensus analysis across multiple AI models including:
- Google Gemini Vision
- OpenAI GPT-4V
- Anthropic Claude (future)

Features:
- Consensus scoring across models
- Performance tracking
- Fallback chains for reliability
- Cost optimization
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics

import google.generativeai as genai
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

from .cache_manager import CacheManager
from .performance_optimizer import PerformanceOptimizer


class ModelProvider(Enum):
    """Supported AI model providers"""
    GEMINI = "gemini"
    GPT4V = "gpt4v"
    CLAUDE = "claude"  # Future implementation


@dataclass
class ModelResponse:
    """Response from a single AI model"""
    provider: ModelProvider
    analysis: str
    confidence: float  # 0.0 to 1.0
    processing_time: float
    token_usage: int
    cost_estimate: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ConsensusAnalysis:
    """Final consensus analysis from multiple models"""
    final_analysis: str
    consensus_confidence: float
    individual_responses: List[ModelResponse]
    agreement_score: float  # How much models agree (0.0 to 1.0)
    total_cost: float
    total_time: float
    primary_model: ModelProvider
    fallback_used: bool


class MultiModelEngine:
    """
    Multi-model analysis engine with consensus scoring and fallback chains
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = CacheManager()
        self.optimizer = PerformanceOptimizer()
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.model_configs = {
            ModelProvider.GEMINI: {
                "model_name": "gemini-pro-vision",
                "cost_per_1k_tokens": 0.00025,  # Approximate
                "max_tokens": 2048,
                "temperature": 0.1
            },
            ModelProvider.GPT4V: {
                "model_name": "gpt-4-vision-preview",
                "cost_per_1k_tokens": 0.01,  # Approximate
                "max_tokens": 2048,
                "temperature": 0.1
            }
        }
        
        # Performance tracking
        self.model_performance = {
            provider: {
                "total_requests": 0,
                "successful_requests": 0,
                "average_confidence": 0.0,
                "average_response_time": 0.0,
                "total_cost": 0.0
            }
            for provider in ModelProvider
        }
        
        # Initialize API clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients for different providers"""
        try:
            # Gemini
            if self.config.get("gemini_api_key"):
                genai.configure(api_key=self.config["gemini_api_key"])
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
            
            # OpenAI GPT-4V
            if self.config.get("openai_api_key"):
                self.openai_client = openai.OpenAI(api_key=self.config["openai_api_key"])
            
            # Token encoder for cost calculation
            try:
                self.token_encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                # Fallback if tiktoken is not available
                self.token_encoder = None
                self.logger.warning("tiktoken not available, token counting disabled")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API clients: {e}")
    
    async def analyze_chart(
        self, 
        image_path: str, 
        prompt: str,
        models: Optional[List[ModelProvider]] = None,
        use_cache: bool = True
    ) -> ConsensusAnalysis:
        """
        Analyze chart using multiple AI models and return consensus
        
        Args:
            image_path: Path to the chart image
            prompt: Analysis prompt
            models: List of models to use (default: all available)
            use_cache: Whether to use cached results
            
        Returns:
            ConsensusAnalysis with combined results
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(image_path, prompt)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.logger.info("Returning cached consensus analysis")
                return ConsensusAnalysis(**cached_result)
        
        # Determine which models to use
        if models is None:
            models = self._get_available_models()
        
        # Run analysis on all models concurrently
        tasks = []
        for model in models:
            task = self._analyze_with_model(model, image_path, prompt)
            tasks.append(task)
        
        # Wait for all models to complete
        model_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful responses
        successful_responses = []
        for response in model_responses:
            if isinstance(response, ModelResponse) and response.success:
                successful_responses.append(response)
                self._update_performance_metrics(response)
        
        if not successful_responses:
            raise Exception("All models failed to analyze the chart")
        
        # Generate consensus analysis
        consensus = self._generate_consensus(successful_responses)
        consensus.total_time = time.time() - start_time
        
        # Cache the result
        if use_cache:
            await self.cache.set(cache_key, asdict(consensus), ttl=3600)  # 1 hour
        
        return consensus
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _analyze_with_model(
        self, 
        provider: ModelProvider, 
        image_path: str, 
        prompt: str
    ) -> ModelResponse:
        """Analyze chart with a specific model"""
        start_time = time.time()
        
        try:
            if provider == ModelProvider.GEMINI:
                return await self._analyze_with_gemini(image_path, prompt, start_time)
            elif provider == ModelProvider.GPT4V:
                return await self._analyze_with_gpt4v(image_path, prompt, start_time)
            else:
                raise NotImplementedError(f"Model {provider} not implemented yet")
                
        except Exception as e:
            self.logger.error(f"Error analyzing with {provider}: {e}")
            return ModelResponse(
                provider=provider,
                analysis="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                token_usage=0,
                cost_estimate=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _analyze_with_gemini(self, image_path: str, prompt: str, start_time: float) -> ModelResponse:
        """Analyze with Google Gemini Vision"""
        from PIL import Image
        
        # Load and optimize image
        image = Image.open(image_path)
        optimized_image = self.optimizer.optimize_image_for_api(image)
        
        # Enhanced prompt for better analysis
        enhanced_prompt = self._enhance_prompt_for_gemini(prompt)
        
        # Make API call
        response = await asyncio.to_thread(
            self.gemini_model.generate_content,
            [enhanced_prompt, optimized_image]
        )
        
        # Extract analysis and calculate confidence
        analysis_text = response.text
        confidence = self._calculate_confidence_gemini(response)
        
        # Calculate token usage and cost
        token_count = len(self.token_encoder.encode(analysis_text))
        cost = (token_count / 1000) * self.model_configs[ModelProvider.GEMINI]["cost_per_1k_tokens"]
        
        return ModelResponse(
            provider=ModelProvider.GEMINI,
            analysis=analysis_text,
            confidence=confidence,
            processing_time=time.time() - start_time,
            token_usage=token_count,
            cost_estimate=cost,
            success=True
        )
    
    async def _analyze_with_gpt4v(self, image_path: str, prompt: str, start_time: float) -> ModelResponse:
        """Analyze with OpenAI GPT-4V"""
        import base64
        
        # Encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Enhanced prompt
        enhanced_prompt = self._enhance_prompt_for_gpt4v(prompt)
        
        # Make API call
        response = await asyncio.to_thread(
            self.openai_client.chat.completions.create,
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2048,
            temperature=0.1
        )
        
        # Extract analysis
        analysis_text = response.choices[0].message.content
        confidence = self._calculate_confidence_gpt4v(response)
        
        # Calculate cost
        token_usage = response.usage.total_tokens
        cost = (token_usage / 1000) * self.model_configs[ModelProvider.GPT4V]["cost_per_1k_tokens"]
        
        return ModelResponse(
            provider=ModelProvider.GPT4V,
            analysis=analysis_text,
            confidence=confidence,
            processing_time=time.time() - start_time,
            token_usage=token_usage,
            cost_estimate=cost,
            success=True
        )
    
    def _generate_consensus(self, responses: List[ModelResponse]) -> ConsensusAnalysis:
        """Generate consensus analysis from multiple model responses"""
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement_score(responses)
        
        # Determine primary model (highest confidence)
        primary_response = max(responses, key=lambda r: r.confidence)
        
        # Generate weighted consensus
        consensus_text = self._generate_weighted_consensus(responses)
        
        # Calculate overall confidence
        confidences = [r.confidence for r in responses]
        consensus_confidence = statistics.mean(confidences) * agreement_score
        
        # Calculate totals
        total_cost = sum(r.cost_estimate for r in responses)
        total_time = max(r.processing_time for r in responses)
        
        return ConsensusAnalysis(
            final_analysis=consensus_text,
            consensus_confidence=consensus_confidence,
            individual_responses=responses,
            agreement_score=agreement_score,
            total_cost=total_cost,
            total_time=total_time,
            primary_model=primary_response.provider,
            fallback_used=len(responses) < len(self._get_available_models())
        )
    
    def _calculate_agreement_score(self, responses: List[ModelResponse]) -> float:
        """Calculate how much the models agree with each other"""
        if len(responses) < 2:
            return 1.0
        
        # Simple implementation: compare key phrases and sentiment
        # In production, this could use more sophisticated NLP
        
        analyses = [r.analysis.lower() for r in responses]
        
        # Key financial terms to check for agreement
        key_terms = [
            'bullish', 'bearish', 'uptrend', 'downtrend', 'support', 'resistance',
            'breakout', 'breakdown', 'consolidation', 'reversal', 'continuation'
        ]
        
        agreement_scores = []
        
        for term in key_terms:
            term_mentions = [term in analysis for analysis in analyses]
            if any(term_mentions):
                # Calculate agreement on this term
                positive_mentions = sum(term_mentions)
                agreement = positive_mentions / len(responses)
                agreement_scores.append(agreement)
        
        if not agreement_scores:
            return 0.5  # Neutral if no key terms found
        
        return statistics.mean(agreement_scores)
    
    def _generate_weighted_consensus(self, responses: List[ModelResponse]) -> str:
        """Generate consensus text weighted by confidence scores"""
        
        # Weight responses by confidence
        total_confidence = sum(r.confidence for r in responses)
        
        if total_confidence == 0:
            return "Unable to generate reliable analysis."
        
        # For now, use the highest confidence response as primary
        # In production, this could be more sophisticated
        primary_response = max(responses, key=lambda r: r.confidence)
        
        # Add consensus metadata
        consensus_text = f"""
CONSENSUS ANALYSIS (Confidence: {primary_response.confidence:.1%})

{primary_response.analysis}

---
Analysis Details:
- Models Used: {', '.join([r.provider.value for r in responses])}
- Agreement Score: {self._calculate_agreement_score(responses):.1%}
- Total Cost: ${sum(r.cost_estimate for r in responses):.4f}
"""
        
        return consensus_text.strip()
    
    def _enhance_prompt_for_gemini(self, base_prompt: str) -> str:
        """Enhance prompt specifically for Gemini model"""
        return f"""
You are an expert financial analyst specializing in technical chart analysis. 
Analyze this stock chart image with high precision and provide detailed insights.

{base_prompt}

Please provide:
1. Technical pattern identification
2. Support and resistance levels
3. Trend analysis
4. Volume analysis (if visible)
5. Risk assessment
6. Trading recommendations

Be specific with price levels and timeframes. Rate your confidence in the analysis.
"""
    
    def _enhance_prompt_for_gpt4v(self, base_prompt: str) -> str:
        """Enhance prompt specifically for GPT-4V model"""
        return f"""
As a professional technical analyst, examine this stock chart and provide comprehensive analysis.

{base_prompt}

Focus on:
- Chart patterns and formations
- Key support/resistance levels
- Trend direction and strength
- Volume patterns
- Entry/exit points
- Risk management

Provide confidence level for each observation.
"""
    
    def _calculate_confidence_gemini(self, response) -> float:
        """Calculate confidence score for Gemini response"""
        # Placeholder implementation
        # In production, this could analyze response metadata
        text = response.text.lower()
        
        confidence_indicators = {
            'clear': 0.1, 'obvious': 0.1, 'strong': 0.1,
            'likely': 0.05, 'probable': 0.05, 'confident': 0.1
        }
        
        uncertainty_indicators = {
            'might': -0.1, 'could': -0.05, 'possibly': -0.1,
            'uncertain': -0.2, 'unclear': -0.15
        }
        
        base_confidence = 0.7
        
        for indicator, weight in confidence_indicators.items():
            if indicator in text:
                base_confidence += weight
        
        for indicator, weight in uncertainty_indicators.items():
            if indicator in text:
                base_confidence += weight
        
        return max(0.1, min(1.0, base_confidence))
    
    def _calculate_confidence_gpt4v(self, response) -> float:
        """Calculate confidence score for GPT-4V response"""
        # Similar to Gemini but could use different heuristics
        return self._calculate_confidence_gemini(response)
    
    def _get_available_models(self) -> List[ModelProvider]:
        """Get list of available/configured models"""
        available = []
        
        if self.config.get("gemini_api_key"):
            available.append(ModelProvider.GEMINI)
        
        if self.config.get("openai_api_key"):
            available.append(ModelProvider.GPT4V)
        
        return available
    
    def _generate_cache_key(self, image_path: str, prompt: str) -> str:
        """Generate cache key for image and prompt combination"""
        import hashlib
        
        # Include image hash and prompt hash
        with open(image_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        return f"consensus_{image_hash}_{prompt_hash}"
    
    def _update_performance_metrics(self, response: ModelResponse):
        """Update performance tracking for a model"""
        provider = response.provider
        metrics = self.model_performance[provider]
        
        metrics["total_requests"] += 1
        if response.success:
            metrics["successful_requests"] += 1
        
        # Update running averages
        total = metrics["total_requests"]
        metrics["average_confidence"] = (
            (metrics["average_confidence"] * (total - 1) + response.confidence) / total
        )
        metrics["average_response_time"] = (
            (metrics["average_response_time"] * (total - 1) + response.processing_time) / total
        )
        metrics["total_cost"] += response.cost_estimate
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all models"""
        return {
            "model_performance": self.model_performance,
            "cache_stats": self.cache.get_stats(),
            "optimizer_stats": self.optimizer.get_stats()
        } 