#!/usr/bin/env python3
"""
Advanced AI Stock Chart Assistant with Multi-Model Integration
Enhanced accuracy through consensus analysis and intelligent caching
Version: 2.0 - Advanced Features
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import google.generativeai as genai
import openai
import os
import threading
import json
import hashlib
import time
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import sqlite3
import requests
from dataclasses import dataclass
import yfinance as yf
import numpy as np

# Enhanced configuration
class AdvancedConfig:
    APP_NAME = "AI Stock Chart Assistant Pro"
    APP_VERSION = "2.0 Advanced"
    WINDOW_SIZE = "1400x900"
    
    # API Models
    GEMINI_MODEL = "gemini-1.5-flash"
    GPT_MODEL = "gpt-4-vision-preview"
    
    # Cache settings
    CACHE_DB = "analysis_cache.db"
    CACHE_EXPIRY_HOURS = 24
    MAX_CACHE_SIZE_MB = 100
    
    # Analysis settings
    CONFIDENCE_THRESHOLD = 0.7
    MAX_CONCURRENT_ANALYSES = 3
    
    @classmethod
    def get_api_keys(cls):
        return {
            'gemini': os.getenv('GEMINI_API_KEY', ''),
            'openai': os.getenv('OPENAI_API_KEY', '')
        }

@dataclass
class AnalysisResult:
    model: str
    analysis: str
    confidence: float
    timestamp: datetime
    processing_time: float
    patterns_detected: List[str]
    key_levels: Dict[str, float]

class CacheManager:
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite cache database"""
        conn = sqlite3.connect(AdvancedConfig.CACHE_DB)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_cache (
                image_hash TEXT PRIMARY KEY,
                prompt_hash TEXT,
                model TEXT,
                result TEXT,
                confidence REAL,
                timestamp TEXT,
                access_count INTEGER DEFAULT 1
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT PRIMARY KEY,
                data TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def get_image_hash(self, image_path: str) -> str:
        """Generate hash for image file"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get_prompt_hash(self, prompt: str) -> str:
        """Generate hash for prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get_cached_analysis(self, image_hash: str, prompt_hash: str, model: str) -> Optional[Dict]:
        """Retrieve cached analysis"""
        conn = sqlite3.connect(AdvancedConfig.CACHE_DB)
        cursor = conn.cursor()
        
        # Check if cache exists and not expired
        expiry_time = datetime.now() - timedelta(hours=AdvancedConfig.CACHE_EXPIRY_HOURS)
        cursor.execute('''
            SELECT result, confidence, timestamp FROM analysis_cache 
            WHERE image_hash=? AND prompt_hash=? AND model=? AND timestamp>?
        ''', (image_hash, prompt_hash, model, expiry_time.isoformat()))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'analysis': result[0],
                'confidence': result[1],
                'timestamp': result[2]
            }
        return None
    
    def cache_analysis(self, image_hash: str, prompt_hash: str, model: str, 
                      result: str, confidence: float):
        """Cache analysis result"""
        conn = sqlite3.connect(AdvancedConfig.CACHE_DB)
        conn.execute('''
            INSERT OR REPLACE INTO analysis_cache 
            (image_hash, prompt_hash, model, result, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_hash, prompt_hash, model, result, confidence, datetime.now().isoformat()))
        conn.commit()
        conn.close()

class MultiModelAnalyzer:
    def __init__(self):
        self.api_keys = AdvancedConfig.get_api_keys()
        self.cache = CacheManager()
        self.models = {}
        self.init_models()
    
    def init_models(self):
        """Initialize AI models"""
        # Gemini setup
        if self.api_keys['gemini']:
            try:
                genai.configure(api_key=self.api_keys['gemini'])
                self.models['gemini'] = genai.GenerativeModel(AdvancedConfig.GEMINI_MODEL)
            except Exception as e:
                print(f"Gemini setup failed: {e}")
        
        # OpenAI setup
        if self.api_keys['openai']:
            try:
                openai.api_key = self.api_keys['openai']
                self.models['openai'] = True
            except Exception as e:
                print(f"OpenAI setup failed: {e}")
    
    def analyze_with_gemini(self, image, prompt: str) -> AnalysisResult:
        """Analyze with Gemini"""
        start_time = time.time()
        
        try:
            response = self.models['gemini'].generate_content([prompt, image])
            processing_time = time.time() - start_time
            
            # Extract patterns and levels (simplified)
            patterns = self.extract_patterns(response.text)
            levels = self.extract_key_levels(response.text)
            
            return AnalysisResult(
                model="Gemini",
                analysis=response.text,
                confidence=0.85,  # Base confidence
                timestamp=datetime.now(),
                processing_time=processing_time,
                patterns_detected=patterns,
                key_levels=levels
            )
        except Exception as e:
            raise Exception(f"Gemini analysis failed: {str(e)}")
    
    def analyze_with_openai(self, image_path: str, prompt: str) -> AnalysisResult:
        """Analyze with OpenAI GPT-4V"""
        start_time = time.time()
        
        try:
            # Convert image to base64
            import base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = openai.ChatCompletion.create(
                model=AdvancedConfig.GPT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            processing_time = time.time() - start_time
            analysis_text = response.choices[0].message.content
            
            patterns = self.extract_patterns(analysis_text)
            levels = self.extract_key_levels(analysis_text)
            
            return AnalysisResult(
                model="GPT-4V",
                analysis=analysis_text,
                confidence=0.88,
                timestamp=datetime.now(),
                processing_time=processing_time,
                patterns_detected=patterns,
                key_levels=levels
            )
        except Exception as e:
            raise Exception(f"OpenAI analysis failed: {str(e)}")
    
    def extract_patterns(self, text: str) -> List[str]:
        """Extract chart patterns from analysis text"""
        patterns = []
        pattern_keywords = [
            'head and shoulders', 'double top', 'double bottom', 'triangle',
            'flag', 'pennant', 'wedge', 'channel', 'support', 'resistance'
        ]
        
        text_lower = text.lower()
        for pattern in pattern_keywords:
            if pattern in text_lower:
                patterns.append(pattern.title())
        
        return patterns
    
    def extract_key_levels(self, text: str) -> Dict[str, float]:
        """Extract key price levels from analysis"""
        levels = {}
        # Simplified extraction - in production, use regex for price patterns
        lines = text.split('\n')
        for line in lines:
            if 'support' in line.lower() and '$' in line:
                # Extract price after $ (simplified)
                try:
                    price = float(line.split('$')[1].split()[0].replace(',', ''))
                    levels['support'] = price
                except:
                    pass
            elif 'resistance' in line.lower() and '$' in line:
                try:
                    price = float(line.split('$')[1].split()[0].replace(',', ''))
                    levels['resistance'] = price
                except:
                    pass
        
        return levels
    
    def consensus_analysis(self, image, image_path: str, prompt: str) -> Dict:
        """Run consensus analysis across multiple models"""
        results = []
        
        # Get cached results first
        image_hash = self.cache.get_image_hash(image_path)
        prompt_hash = self.cache.get_prompt_hash(prompt)
        
        # Try each available model
        for model_name in self.models:
            cached = self.cache.get_cached_analysis(image_hash, prompt_hash, model_name)
            if cached:
                print(f"Using cached result for {model_name}")
                results.append(AnalysisResult(
                    model=model_name.title(),
                    analysis=cached['analysis'],
                    confidence=cached['confidence'],
                    timestamp=datetime.fromisoformat(cached['timestamp']),
                    processing_time=0.0,
                    patterns_detected=[],
                    key_levels={}
                ))
                continue
            
            try:
                if model_name == 'gemini':
                    result = self.analyze_with_gemini(image, prompt)
                elif model_name == 'openai':
                    result = self.analyze_with_openai(image_path, prompt)
                
                results.append(result)
                
                # Cache the result
                self.cache.cache_analysis(
                    image_hash, prompt_hash, model_name,
                    result.analysis, result.confidence
                )
                
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue
        
        return self.create_consensus(results)
    
    def create_consensus(self, results: List[AnalysisResult]) -> Dict:
        """Create consensus analysis from multiple results"""
        if not results:
            raise Exception("No analysis results available")
        
        # Calculate consensus confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # Combine patterns
        all_patterns = []
        for result in results:
            all_patterns.extend(result.patterns_detected)
        unique_patterns = list(set(all_patterns))
        
        # Create combined analysis
        consensus_text = "**CONSENSUS ANALYSIS**\n\n"
        consensus_text += f"**Models Used:** {', '.join(r.model for r in results)}\n"
        consensus_text += f"**Consensus Confidence:** {avg_confidence:.1%}\n\n"
        
        if len(results) > 1:
            consensus_text += "**Individual Analyses:**\n\n"
            for i, result in enumerate(results, 1):
                consensus_text += f"**{result.model} Analysis (Confidence: {result.confidence:.1%}):**\n"
                consensus_text += f"{result.analysis}\n\n"
        else:
            consensus_text += results[0].analysis
        
        if unique_patterns:
            consensus_text += f"\n**Detected Patterns:** {', '.join(unique_patterns)}\n"
        
        return {
            'analysis': consensus_text,
            'confidence': avg_confidence,
            'models_used': len(results),
            'patterns': unique_patterns,
            'processing_time': sum(r.processing_time for r in results)
        }

class MarketDataProvider:
    def __init__(self):
        self.cache = CacheManager()
    
    def get_stock_data(self, symbol: str) -> Dict:
        """Get current stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            info = stock.info
            
            return {
                'symbol': symbol,
                'current_price': hist['Close'].iloc