"""
Simplified ML Brain for Trading Platform
Designed for reliability and future extensibility
"""

import numpy as np
from collections import deque
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import sklearn, but continue without it if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available - ML features will be limited")
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None

class SimpleMLBrain:
    """
    Simplified ML system - single model, clear extension points
    Easy to enhance with new features when proven stable
    """
    
    def __init__(self):
        # Core model - start simple
        self.model = None
        self.is_trained = False
        
        # Data storage
        self.trade_history = deque(maxlen=1000)
        self.min_samples_to_train = 30  # Low threshold to start
        
        # Performance tracking
        self.predictions_correct = 0
        self.predictions_total = 0
        self.last_train_time = datetime.now()
        
        # EXTENSION POINT 1: Additional models can be added here
        # self.advanced_models = {}  # Uncomment when ready
        
        # EXTENSION POINT 2: Feature sets for future use
        self.feature_version = "v1"  # Track feature evolution
        
        # Load any existing model
        self.load_model()
    
    def extract_features(self, market_data: Dict) -> List[float]:
        """
        Extract features - EASY TO EXTEND
        Just add new features to the list
        """
        features = []
        
        # === BASIC FEATURES (v1) ===
        
        # 1. Price position (0-1 scale)
        if 'current_price' in market_data and 'day_high' in market_data and 'day_low' in market_data:
            price_range = market_data['day_high'] - market_data['day_low']
            if price_range > 0:
                price_position = (market_data['current_price'] - market_data['day_low']) / price_range
                features.append(min(max(price_position, 0), 1))
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # 2. Volume ratio (capped at 3)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        features.append(min(volume_ratio, 3.0))
        
        # 3. RSI normalized (0-1)
        rsi = market_data.get('rsi', 50)
        features.append(rsi / 100.0)
        
        # 4. Price change 5 min (capped at ±5%)
        change_5min = market_data.get('change_5min', 0)
        features.append(max(min(change_5min, 0.05), -0.05) + 0.05)
        
        # 5. Hour of day (0-1)
        hour = datetime.now().hour
        market_hour = max(0, min(hour - 9, 7)) / 7
        features.append(market_hour)
        
        # 6. Day of week (Monday=0, Friday=1)
        day = datetime.now().weekday()
        features.append(min(day / 4, 1))
        
        # === EXTENSION POINT: Add v2 features here when ready ===
        # if self.feature_version == "v2":
        #     features.append(market_data.get('vwap_distance', 0))
        #     features.append(market_data.get('sector_strength', 0))
        #     features.append(market_data.get('market_regime', 0))
        
        return features
    
    def predict(self, market_data: Dict) -> Dict:
        """
        Make prediction - EXTENSIBLE
        Can add confidence adjustments, multi-model voting, etc.
        """
        
        # If sklearn not available, use rule-based fallback
        if not SKLEARN_AVAILABLE:
            # Simple rule-based logic
            rsi = market_data.get('rsi', 50)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            action = 'NEUTRAL'
            confidence = 0.5
            
            if rsi < 30 and volume_ratio > 1.5:
                action = 'BUY'
                confidence = 0.65
            elif rsi > 70 and volume_ratio > 1.5:
                action = 'SELL'
                confidence = 0.65
            
            return {
                'action': action,
                'confidence': confidence,
                'ml_active': False,
                'win_rate': 0.5,
                'reason': f'Rule-based (no sklearn): RSI={rsi:.0f}',
                'feature_version': self.feature_version
            }
        
        # Not trained yet - return neutral
        if not self.is_trained:
            return {
                'action': 'NEUTRAL',
                'confidence': 0.0,
                'ml_active': False,
                'reason': f'Learning... ({len(self.trade_history)}/{self.min_samples_to_train} samples)'
            }
        
        try:
            # Extract features
            features = self.extract_features(market_data)
            
            # Basic prediction
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            confidence = max(probabilities)
            
            # Convert to action
            if prediction == 1:
                action = 'BUY'
            elif prediction == -1:
                action = 'SELL'
            else:
                action = 'NEUTRAL'
            
            # === EXTENSION POINT: Advanced logic ===
            # Could add:
            # - Multi-timeframe confirmation
            # - Risk adjustment based on market conditions
            # - Ensemble voting if multiple models
            
            # Calculate win rate
            win_rate = self.predictions_correct / max(self.predictions_total, 1)
            
            return {
                'action': action,
                'confidence': float(confidence),
                'ml_active': True,
                'win_rate': win_rate,
                'reason': f'ML: {action} ({confidence:.1%} conf, {win_rate:.1%} accuracy)',
                'feature_version': self.feature_version
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {
                'action': 'NEUTRAL',
                'confidence': 0.0,
                'ml_active': False,
                'reason': 'ML error - using strategies only'
            }
    
    def record_trade(self, market_data: Dict, action: str, result: Optional[float] = None):
        """
        Record trade for learning - EXTENSIBLE
        Can add more sophisticated outcome tracking
        """
        try:
            features = self.extract_features(market_data)
            
            # Determine label
            if result is not None:
                if result > 0.5:  # Profitable
                    label = 1 if action == 'BUY' else -1
                elif result < -0.5:  # Loss
                    label = -1 if action == 'BUY' else 1
                else:  # Neutral
                    label = 0
            else:
                label = 0  # No result yet
            
            # Store trade
            self.trade_history.append({
                'features': features,
                'label': label,
                'action': action,
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'feature_version': self.feature_version
            })
            
            # === EXTENSION POINT: Advanced tracking ===
            # Could add:
            # - Market condition at time of trade
            # - Correlated assets performance
            # - News sentiment score
            
            # Retrain periodically
            if len(self.trade_history) >= self.min_samples_to_train:
                time_since_train = (datetime.now() - self.last_train_time).seconds
                if time_since_train > 1800:  # 30 minutes
                    self.train()
                    
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def train(self):
        """
        Train model - EXTENSIBLE
        Easy to add new algorithms or ensemble methods
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Training skipped - scikit-learn not available")
            return False
            
        try:
            # Need minimum samples
            if len(self.trade_history) < self.min_samples_to_train:
                return False
            
            # Get trades with results
            trades_with_results = [t for t in self.trade_history if t.get('result') is not None]
            
            if len(trades_with_results) < 20:
                logger.info(f"Only {len(trades_with_results)} trades with results")
                return False
            
            # Prepare data
            X = np.array([t['features'] for t in trades_with_results])
            y = np.array([t['label'] for t in trades_with_results])
            
            # Train Random Forest (simple and robust)
            self.model = RandomForestClassifier(
                n_estimators=50,  # Moderate number of trees
                max_depth=5,      # Shallow to avoid overfitting
                min_samples_split=10,
                random_state=42
            )
            
            self.model.fit(X, y)
            
            # === EXTENSION POINT: Advanced models ===
            # When ready, can add:
            # - XGBoost for better accuracy
            # - Neural network for complex patterns
            # - Ensemble voting between models
            
            # Calculate training accuracy
            predictions = self.model.predict(X)
            accuracy = np.mean(predictions == y)
            
            self.is_trained = True
            self.last_train_time = datetime.now()
            
            logger.info(f"ML trained on {len(X)} samples, accuracy: {accuracy:.1%}")
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def evaluate_prediction(self, prediction: Dict, actual_result: float):
        """
        Evaluate prediction accuracy - EXTENSIBLE
        Can add more sophisticated metrics
        """
        self.predictions_total += 1
        
        # Simple evaluation
        if prediction.get('action') == 'BUY' and actual_result > 0:
            self.predictions_correct += 1
        elif prediction.get('action') == 'SELL' and actual_result < 0:
            self.predictions_correct += 1
        elif prediction.get('action') == 'NEUTRAL' and abs(actual_result) < 0.5:
            self.predictions_correct += 1
        
        # === EXTENSION POINT: Advanced metrics ===
        # Could track:
        # - Profit factor
        # - Sharpe ratio contribution
        # - Maximum drawdown impact
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'is_trained': self.is_trained,
            'total_trades': len(self.trade_history),
            'trades_with_results': len([t for t in self.trade_history if t.get('result') is not None]),
            'accuracy': self.predictions_correct / max(self.predictions_total, 1),
            'predictions_made': self.predictions_total,
            'model_type': 'RandomForest' if SKLEARN_AVAILABLE else 'RuleBased',
            'feature_version': self.feature_version,
            'last_train': self.last_train_time.isoformat() if self.last_train_time else None,
            'extensible': True,
            'sklearn_available': SKLEARN_AVAILABLE
        }
    
    def save_model(self):
        """Save model and stats"""
        try:
            os.makedirs('ml_models', exist_ok=True)
            
            # Only save model if sklearn is available and model exists
            if SKLEARN_AVAILABLE and self.model:
                try:
                    import joblib
                    joblib.dump(self.model, 'ml_models/simple_model.pkl')
                except ImportError:
                    logger.warning("joblib not available - model not saved")
            
            # Save stats and metadata
            stats = {
                'is_trained': self.is_trained,
                'trades_recorded': len(self.trade_history),
                'accuracy': self.predictions_correct / max(self.predictions_total, 1),
                'feature_version': self.feature_version,
                'saved_at': datetime.now().isoformat(),
                'sklearn_available': SKLEARN_AVAILABLE
            }
            
            with open('ml_models/stats.json', 'w') as f:
                json.dump(stats, f)
                
            logger.info("Model saved")
            
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def load_model(self):
        """Load existing model"""
        try:
            if SKLEARN_AVAILABLE and os.path.exists('ml_models/simple_model.pkl'):
                try:
                    import joblib
                    self.model = joblib.load('ml_models/simple_model.pkl')
                    self.is_trained = True
                except ImportError:
                    logger.warning("joblib not available - cannot load model")
                
                if os.path.exists('ml_models/stats.json'):
                    with open('ml_models/stats.json', 'r') as f:
                        stats = json.load(f)
                        self.feature_version = stats.get('feature_version', 'v1')
                        logger.info(f"Loaded model: {stats['accuracy']:.1%} accuracy, version {self.feature_version}")
                
                return True
                
        except Exception as e:
            logger.error(f"Load error (starting fresh): {e}")
            
        return False
    
    # === FUTURE EXTENSION METHODS ===
    # Easy to add when ready:
    
    def add_advanced_model(self, name: str, model):
        """Add additional models for ensemble - FUTURE"""
        pass
    
    def add_feature_extractor(self, name: str, extractor):
        """Add custom feature extractors - FUTURE"""
        pass
    
    def enable_deep_learning(self):
        """Enable neural network predictions - FUTURE"""
        pass
    
    def enable_reinforcement_learning(self):
        """Enable RL for strategy optimization - FUTURE"""
        pass

# Global instance
ml_brain = SimpleMLBrain()