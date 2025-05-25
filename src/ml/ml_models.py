"""
Machine Learning Models for Advanced Pattern Recognition and Prediction

This module implements ML models for:
- Chart pattern classification
- Trend prediction
- Price direction forecasting
- Model performance tracking and retraining
"""

import asyncio
import logging
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import talib

from .pattern_detector import ChartPattern, PatternType
from .trend_analyzer import TrendDirection, TrendStrength


@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    training_date: datetime
    validation_samples: int
    feature_importance: Optional[Dict[str, float]] = None
    cross_val_scores: Optional[List[float]] = None


@dataclass
class PredictionResult:
    """Result from ML model prediction"""
    prediction: Union[str, int, float]
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    feature_values: Optional[Dict[str, float]] = None
    model_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class PatternClassifier:
    """
    ML classifier for chart pattern recognition
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.performance_history = {}
        
        # Model directory
        self.model_dir = Path(self.config.get('model_dir', 'models'))
        self.model_dir.mkdir(exist_ok=True)
        
        self.logger.info("PatternClassifier initialized")
    
    async def train_pattern_classifier(
        self, 
        training_data: pd.DataFrame,
        target_column: str = 'pattern_type'
    ) -> ModelPerformance:
        """
        Train pattern classification model
        
        Args:
            training_data: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            ModelPerformance metrics
        """
        try:
            self.logger.info("Training pattern classifier")
            
            # Prepare features and target
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns]
            y = training_data[target_column]
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            self.label_encoders['pattern_classifier'] = label_encoder
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Create and train model pipeline
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y_encoded, cv=5)
            
            # Feature importance
            feature_importance = {}
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                importances = model.named_steps['classifier'].feature_importances_
                feature_importance = dict(zip(feature_columns, importances))
            
            # Store model
            self.models['pattern_classifier'] = model
            self.scalers['pattern_classifier'] = model.named_steps['scaler']
            
            # Create performance object
            performance = ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=conf_matrix,
                training_date=datetime.now(),
                validation_samples=len(X_test),
                feature_importance=feature_importance,
                cross_val_scores=cv_scores.tolist()
            )
            
            self.performance_history['pattern_classifier'] = performance
            
            # Save model
            await self._save_model('pattern_classifier')
            
            self.logger.info(f"Pattern classifier trained - Accuracy: {accuracy:.3f}")
            return performance
            
        except Exception as e:
            self.logger.error(f"Error training pattern classifier: {str(e)}")
            raise
    
    async def predict_pattern(
        self, 
        features: Dict[str, float],
        target_type: str = 'pattern_type'
    ) -> PredictionResult:
        """
        Predict chart pattern from features
        
        Args:
            features: Dictionary of feature values
            target_type: Type of prediction target
            
        Returns:
            PredictionResult with prediction and confidence
        """
        try:
            model_name = 'pattern_classifier'
            
            # Load model if not in memory
            if model_name not in self.models:
                await self._load_model(model_name)
            
            model = self.models[model_name]
            label_encoder = self.label_encoders[model_name]
            
            # Prepare features
            feature_df = pd.DataFrame([features])
            
            # Make prediction
            prediction_encoded = model.predict(feature_df)[0]
            prediction_proba = model.predict_proba(feature_df)[0]
            
            # Decode prediction
            prediction = label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = np.max(prediction_proba)
            
            # Create probability dictionary
            class_labels = label_encoder.classes_
            probabilities = dict(zip(class_labels, prediction_proba))
            
            return PredictionResult(
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                feature_values=features,
                model_name=model_name
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting pattern: {str(e)}")
            return PredictionResult(
                prediction="unknown",
                confidence=0.0,
                model_name=model_name
            )
    
    def _create_model(self, model_type: str):
        """Create ML model based on type"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=1000
            )
        }
        
        return models.get(model_type, models['random_forest'])
    
    async def _save_model(self, model_name: str):
        """Save model to disk"""
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            
            model_data = {
                'model': self.models[model_name],
                'label_encoder': self.label_encoders.get(model_name),
                'performance': self.performance_history.get(model_name),
                'timestamp': datetime.now()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
    
    async def _load_model(self, model_name: str):
        """Load model from disk"""
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models[model_name] = model_data['model']
                if model_data.get('label_encoder'):
                    self.label_encoders[model_name] = model_data['label_encoder']
                if model_data.get('performance'):
                    self.performance_history[model_name] = model_data['performance']
                
                self.logger.info(f"Model loaded: {model_path}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")


class MLModelManager:
    """
    Manager for all ML models and feature engineering
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize classifiers
        self.pattern_classifier = PatternClassifier(config)
        
        # Feature engineering parameters
        self.feature_windows = self.config.get('feature_windows', [5, 10, 20])
        self.technical_indicators = self.config.get('technical_indicators', True)
        
        self.logger.info("MLModelManager initialized")
    
    async def extract_features_from_data(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        patterns: Optional[List[ChartPattern]] = None
    ) -> Dict[str, float]:
        """
        Extract comprehensive features from price and volume data
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_data: Optional volume data
            patterns: Optional detected patterns
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            if len(price_data) < max(self.feature_windows):
                self.logger.warning("Insufficient data for feature extraction")
                return features
            
            close = price_data['close'].values
            high = price_data['high'].values if 'high' in price_data.columns else close
            low = price_data['low'].values if 'low' in price_data.columns else close
            
            # Price-based features
            features.update(self._extract_price_features(close, high, low))
            
            # Technical indicator features
            if self.technical_indicators:
                features.update(self._extract_technical_features(close, high, low))
            
            # Volume features
            if volume_data is not None and len(volume_data) > 0:
                volume = volume_data['volume'].values
                features.update(self._extract_volume_features(close, volume))
            
            # Pattern-based features
            if patterns:
                features.update(self._extract_pattern_features(patterns))
            
            # Statistical features
            features.update(self._extract_statistical_features(close))
            
            # Volatility features
            features.update(self._extract_volatility_features(close, high, low))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return {}
    
    def _extract_price_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Extract price-based features"""
        features = {}
        
        try:
            # Price returns for different windows
            for window in self.feature_windows:
                if len(close) > window:
                    returns = (close[-1] - close[-window]) / close[-window]
                    features[f'return_{window}d'] = returns
                    
                    # High-low range
                    hl_range = (np.max(high[-window:]) - np.min(low[-window:])) / close[-1]
                    features[f'hl_range_{window}d'] = hl_range
            
            # Current price position in recent range
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])
            if recent_high > recent_low:
                features['price_position'] = (close[-1] - recent_low) / (recent_high - recent_low)
            
            # Price momentum
            if len(close) >= 10:
                momentum = (close[-1] - close[-10]) / close[-10]
                features['momentum_10d'] = momentum
            
        except Exception as e:
            self.logger.error(f"Error extracting price features: {str(e)}")
        
        return features
    
    def _extract_technical_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Extract technical indicator features"""
        features = {}
        
        try:
            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            if len(rsi) > 0 and not np.isnan(rsi[-1]):
                features['rsi'] = rsi[-1]
                features['rsi_oversold'] = 1 if rsi[-1] < 30 else 0
                features['rsi_overbought'] = 1 if rsi[-1] > 70 else 0
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            if len(macd) > 0 and not np.isnan(macd[-1]):
                features['macd'] = macd[-1]
                features['macd_signal'] = macd_signal[-1]
                features['macd_histogram'] = macd_hist[-1]
                features['macd_bullish'] = 1 if macd[-1] > macd_signal[-1] else 0
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]):
                bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                features['bb_position'] = bb_position
                features['bb_squeeze'] = 1 if (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] < 0.1 else 0
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            if len(stoch_k) > 0 and not np.isnan(stoch_k[-1]):
                features['stoch_k'] = stoch_k[-1]
                features['stoch_d'] = stoch_d[-1]
                features['stoch_oversold'] = 1 if stoch_k[-1] < 20 else 0
                features['stoch_overbought'] = 1 if stoch_k[-1] > 80 else 0
            
            # Williams %R
            williams_r = talib.WILLR(high, low, close, timeperiod=14)
            if len(williams_r) > 0 and not np.isnan(williams_r[-1]):
                features['williams_r'] = williams_r[-1]
            
            # ATR (Average True Range)
            atr = talib.ATR(high, low, close, timeperiod=14)
            if len(atr) > 0 and not np.isnan(atr[-1]):
                features['atr'] = atr[-1] / close[-1]  # Normalized ATR
            
            # Moving averages
            for period in [10, 20, 50]:
                if len(close) >= period:
                    sma = talib.SMA(close, timeperiod=period)
                    if len(sma) > 0 and not np.isnan(sma[-1]):
                        features[f'sma_{period}_ratio'] = close[-1] / sma[-1]
                        features[f'above_sma_{period}'] = 1 if close[-1] > sma[-1] else 0
            
        except Exception as e:
            self.logger.error(f"Error extracting technical features: {str(e)}")
        
        return features
    
    def _extract_volume_features(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """Extract volume-based features"""
        features = {}
        
        try:
            # Volume ratios
            if len(volume) >= 20:
                avg_volume = np.mean(volume[-20:])
                if avg_volume > 0:
                    features['volume_ratio'] = volume[-1] / avg_volume
                    features['high_volume'] = 1 if volume[-1] > avg_volume * 2 else 0
            
            # Volume trend
            if len(volume) >= 10:
                recent_volume = np.mean(volume[-5:])
                older_volume = np.mean(volume[-10:-5])
                if older_volume > 0:
                    features['volume_trend'] = recent_volume / older_volume
            
            # On Balance Volume
            if len(close) == len(volume):
                obv = talib.OBV(close, volume)
                if len(obv) >= 10:
                    obv_trend = (obv[-1] - obv[-10]) / abs(obv[-10]) if obv[-10] != 0 else 0
                    features['obv_trend'] = obv_trend
            
            # Volume-Price Trend
            if len(close) == len(volume) and len(close) >= 2:
                price_change = close[-1] - close[-2]
                volume_change = volume[-1] - volume[-2] if len(volume) >= 2 else 0
                if price_change != 0:
                    features['volume_price_correlation'] = volume_change / abs(price_change)
            
        except Exception as e:
            self.logger.error(f"Error extracting volume features: {str(e)}")
        
        return features
    
    def _extract_pattern_features(self, patterns: List[ChartPattern]) -> Dict[str, float]:
        """Extract pattern-based features"""
        features = {}
        
        try:
            # Pattern counts by type
            pattern_counts = {}
            total_confidence = 0
            
            for pattern in patterns:
                pattern_type = pattern.pattern_type.value
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
                total_confidence += pattern.confidence
            
            # Pattern features
            features['pattern_count'] = len(patterns)
            features['avg_pattern_confidence'] = total_confidence / len(patterns) if patterns else 0
            
            # Specific pattern indicators
            bullish_patterns = ['triangle_ascending', 'flag_bullish', 'double_bottom', 'inverse_head_and_shoulders']
            bearish_patterns = ['triangle_descending', 'flag_bearish', 'double_top', 'head_and_shoulders']
            
            bullish_count = sum(pattern_counts.get(p, 0) for p in bullish_patterns)
            bearish_count = sum(pattern_counts.get(p, 0) for p in bearish_patterns)
            
            features['bullish_patterns'] = bullish_count
            features['bearish_patterns'] = bearish_count
            features['pattern_bias'] = (bullish_count - bearish_count) / max(1, bullish_count + bearish_count)
            
            # Highest confidence pattern
            if patterns:
                highest_conf_pattern = max(patterns, key=lambda p: p.confidence)
                features['max_pattern_confidence'] = highest_conf_pattern.confidence
                features['max_pattern_bullish_prob'] = highest_conf_pattern.bullish_probability
            
        except Exception as e:
            self.logger.error(f"Error extracting pattern features: {str(e)}")
        
        return features
    
    def _extract_statistical_features(self, close: np.ndarray) -> Dict[str, float]:
        """Extract statistical features"""
        features = {}
        
        try:
            for window in self.feature_windows:
                if len(close) >= window:
                    window_data = close[-window:]
                    
                    # Basic statistics
                    features[f'mean_{window}d'] = np.mean(window_data)
                    features[f'std_{window}d'] = np.std(window_data)
                    features[f'skew_{window}d'] = self._calculate_skewness(window_data)
                    features[f'kurtosis_{window}d'] = self._calculate_kurtosis(window_data)
                    
                    # Percentile features
                    features[f'current_percentile_{window}d'] = np.percentile(window_data, 
                                                                            (window_data == close[-1]).sum() / len(window_data) * 100)
            
        except Exception as e:
            self.logger.error(f"Error extracting statistical features: {str(e)}")
        
        return features
    
    def _extract_volatility_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Extract volatility-based features"""
        features = {}
        
        try:
            # Historical volatility
            if len(close) >= 20:
                returns = np.diff(np.log(close))
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                features['historical_volatility'] = volatility
            
            # True Range based volatility
            if len(high) >= 14:
                tr = talib.TRANGE(high, low, close)
                atr = talib.ATR(high, low, close, timeperiod=14)
                if len(atr) > 0 and not np.isnan(atr[-1]):
                    features['atr_volatility'] = atr[-1] / close[-1]
            
            # Volatility regime
            if len(close) >= 50:
                short_vol = np.std(close[-10:])
                long_vol = np.std(close[-50:])
                if long_vol > 0:
                    features['volatility_regime'] = short_vol / long_vol
            
        except Exception as e:
            self.logger.error(f"Error extracting volatility features: {str(e)}")
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return 0
    
    async def train_trend_predictor(
        self, 
        historical_data: List[Dict[str, Any]]
    ) -> ModelPerformance:
        """Train trend prediction model"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Train pattern classifier
            performance = await self.pattern_classifier.train_pattern_classifier(df)
            
            self.logger.info("Trend predictor training completed")
            return performance
            
        except Exception as e:
            self.logger.error(f"Error training trend predictor: {str(e)}")
            raise
    
    async def predict_trend(
        self, 
        features: Dict[str, float]
    ) -> PredictionResult:
        """Predict trend using trained models"""
        try:
            # Use pattern classifier for trend prediction
            result = await self.pattern_classifier.predict_pattern(features, 'trend_direction')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting trend: {str(e)}")
            return PredictionResult(prediction="unknown", confidence=0.0)
    
    async def train_price_predictor(
        self, 
        price_data: pd.DataFrame,
        target_horizon: int = 5
    ) -> ModelPerformance:
        """
        Train price direction prediction model
        
        Args:
            price_data: Historical price data
            target_horizon: Days ahead to predict
            
        Returns:
            ModelPerformance metrics
        """
        try:
            self.logger.info(f"Training price predictor for {target_horizon}-day horizon")
            
            # Create features and targets
            features_list = []
            targets = []
            
            for i in range(len(price_data) - target_horizon):
                # Extract features for current window
                window_data = price_data.iloc[i:i+50]  # 50-day window
                if len(window_data) < 50:
                    continue
                
                features = await self.extract_features_from_data(window_data)
                
                # Calculate target (price direction)
                current_price = price_data['close'].iloc[i+49]
                future_price = price_data['close'].iloc[i+49+target_horizon]
                direction = 1 if future_price > current_price else 0
                
                features_list.append(features)
                targets.append(direction)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            features_df['target'] = targets
            
            # Train classifier
            performance = await self.pattern_classifier.train_pattern_classifier(
                features_df, 'target'
            )
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error training price predictor: {str(e)}")
            raise
    
    async def predict_price_direction(
        self, 
        features: Dict[str, float]
    ) -> PredictionResult:
        """Predict price direction"""
        try:
            result = await self.pattern_classifier.predict_pattern(features, 'price_direction')
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting price direction: {str(e)}")
            return PredictionResult(prediction=0, confidence=0.0)
    
    async def evaluate_model_performance(
        self, 
        model_name: str,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        try:
            # Extract features from test data
            features_list = []
            for i in range(len(test_data)):
                window_data = test_data.iloc[max(0, i-49):i+1]
                if len(window_data) >= 10:  # Minimum window
                    features = await self.extract_features_from_data(window_data)
                    features_list.append(features)
            
            # Make predictions
            predictions = []
            for features in features_list:
                result = await self.pattern_classifier.predict_pattern(features)
                predictions.append(result)
            
            # Calculate performance metrics
            confidences = [p.confidence for p in predictions]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                'model_name': model_name,
                'test_samples': len(predictions),
                'average_confidence': avg_confidence,
                'predictions': len([p for p in predictions if p.confidence > 0.7]),
                'evaluation_date': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model performance: {str(e)}")
            return {}
    
    async def retrain_models(
        self, 
        new_data: pd.DataFrame,
        retrain_threshold: float = 0.1
    ):
        """Retrain models if performance degrades"""
        try:
            # Evaluate current performance
            current_performance = await self.evaluate_model_performance('pattern_classifier', new_data)
            
            # Check if retraining is needed
            avg_confidence = current_performance.get('average_confidence', 0)
            
            if avg_confidence < retrain_threshold:
                self.logger.info("Model performance below threshold, retraining...")
                
                # Retrain with new data
                await self.train_trend_predictor(new_data.to_dict('records'))
                
                self.logger.info("Model retraining completed")
            else:
                self.logger.info("Model performance acceptable, no retraining needed")
                
        except Exception as e:
            self.logger.error(f"Error in model retraining: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'pattern_classifier': {
                'loaded': 'pattern_classifier' in self.pattern_classifier.models,
                'performance': self.pattern_classifier.performance_history.get('pattern_classifier'),
                'last_training': getattr(
                    self.pattern_classifier.performance_history.get('pattern_classifier'), 
                    'training_date', 
                    None
                )
            },
            'feature_windows': self.feature_windows,
            'technical_indicators': self.technical_indicators
        } 