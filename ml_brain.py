"""
Advanced Machine Learning Brain for AI Trading Platform
This module provides self-improving ML capabilities
Location: Save as ml_brain.py in your project root
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class AdvancedMLBrain:
    """Self-improving ML system that learns from market data"""
    
    def __init__(self):
        self.models = {
            'random_forest': None,
            'gradient_boost': None,
            'neural_network': None,
            'xgboost': None
        }
        
        self.model_weights = {
            'random_forest': 0.25,
            'gradient_boost': 0.25,
            'neural_network': 0.25,
            'xgboost': 0.25
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = []
        self.prediction_history = []
        self.accuracy_scores = {model: [] for model in self.models}
        
        # Auto-load saved models if they exist
        self.load_models()
        
    def prepare_features(self, market_data: Dict) -> np.ndarray:
        """Convert market data into ML features"""
        features = []
        
        # Price features
        features.append(market_data.get('rsi', 50) / 100)
        features.append(market_data.get('macd', 0))
        features.append(market_data.get('bb_position', 0.5))
        features.append(market_data.get('volume_ratio', 1))
        
        # Momentum features
        features.append(market_data.get('momentum_5', 0))
        features.append(market_data.get('momentum_10', 0))
        
        # Trend features
        features.append(1 if market_data.get('sma_20', 0) > market_data.get('sma_50', 0) else 0)
        features.append(market_data.get('adx', 25) / 100)
        
        # Volatility
        features.append(market_data.get('atr', 0))
        features.append(market_data.get('volatility', 0))
        
        # Time features (market session, day of week)
        now = datetime.now()
        features.append(now.hour / 24)  # Normalized hour
        features.append(now.weekday() / 7)  # Normalized day
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, force_retrain: bool = False):
        """Train all ML models on collected data"""
        if len(self.training_data) < 100 and not force_retrain:
            logger.info(f"Not enough data to train: {len(self.training_data)} samples")
            return False
        
        logger.info(f"Training models with {len(self.training_data)} samples...")
        
        # Prepare training data
        X = np.array([d['features'] for d in self.training_data])
        y = np.array([d['label'] for d in self.training_data])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.models['random_forest'].fit(X_train_scaled, y_train)
        rf_score = self.models['random_forest'].score(X_test_scaled, y_test)
        
        # Train Gradient Boosting
        self.models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.models['gradient_boost'].fit(X_train_scaled, y_train)
        gb_score = self.models['gradient_boost'].score(X_test_scaled, y_test)
        
        # Train Neural Network
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        self.models['neural_network'].fit(X_train_scaled, y_train)
        nn_score = self.models['neural_network'].score(X_test_scaled, y_test)
        
        # Train XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.models['xgboost'].fit(X_train_scaled, y_train)
        xgb_score = self.models['xgboost'].score(X_test_scaled, y_test)
        
        # Update model weights based on performance
        scores = {
            'random_forest': rf_score,
            'gradient_boost': gb_score,
            'neural_network': nn_score,
            'xgboost': xgb_score
        }
        
        # Normalize weights based on accuracy
        total_score = sum(scores.values())
        for model, score in scores.items():
            self.model_weights[model] = score / total_score
            self.accuracy_scores[model].append(score)
        
        self.is_trained = True
        logger.info(f"Training complete! Scores: {scores}")
        
        # Save models
        self.save_models()
        
        return True
    
    def predict(self, market_data: Dict) -> Dict:
        """Make ensemble prediction using all models"""
        if not self.is_trained:
            # Return basic prediction if models not trained
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'ml_powered': False,
                'reason': 'ML models still learning'
            }
        
        # Prepare features
        features = self.prepare_features(market_data)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            if model is not None:
                pred = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0]
                predictions[model_name] = pred
                probabilities[model_name] = max(prob)
        
        # Weighted ensemble voting
        buy_score = 0
        sell_score = 0
        hold_score = 0
        
        for model_name, pred in predictions.items():
            weight = self.model_weights[model_name]
            confidence = probabilities[model_name]
            
            if pred == 1:  # Buy
                buy_score += weight * confidence
            elif pred == -1:  # Sell
                sell_score += weight * confidence
            else:  # Hold
                hold_score += weight * confidence
        
        # Determine final action
        max_score = max(buy_score, sell_score, hold_score)
        
        if max_score == buy_score and buy_score > 0.6:
            action = 'BUY'
            confidence = buy_score
        elif max_score == sell_score and sell_score > 0.6:
            action = 'SELL'
            confidence = sell_score
        else:
            action = 'HOLD'
            confidence = hold_score
        
        # Record prediction for learning
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'features': features.tolist(),
            'prediction': action,
            'confidence': confidence,
            'model_scores': predictions,
            'market_data': market_data
        }
        self.prediction_history.append(prediction_record)
        
        return {
            'action': action,
            'confidence': min(confidence, 0.95),
            'ml_powered': True,
            'models_agree': len(set(predictions.values())) == 1,
            'best_model': max(self.accuracy_scores, key=lambda k: self.accuracy_scores[k][-1] if self.accuracy_scores[k] else 0),
            'reason': self.generate_explanation(action, predictions, market_data)
        }
    
    def generate_explanation(self, action: str, predictions: Dict, market_data: Dict) -> str:
        """Generate human-readable explanation for prediction"""
        explanations = []
        
        if action == 'BUY':
            if market_data.get('rsi', 50) < 30:
                explanations.append("Oversold conditions detected")
            if market_data.get('momentum_5', 0) > 0.02:
                explanations.append("Strong positive momentum")
            if len(set(predictions.values())) == 1:
                explanations.append("All models agree on buy signal")
        elif action == 'SELL':
            if market_data.get('rsi', 50) > 70:
                explanations.append("Overbought conditions detected")
            if market_data.get('momentum_5', 0) < -0.02:
                explanations.append("Strong negative momentum")
        else:
            explanations.append("No clear signal from ML models")
        
        models_voting_buy = sum(1 for p in predictions.values() if p == 1)
        models_voting_sell = sum(1 for p in predictions.values() if p == -1)
        explanations.append(f"{models_voting_buy} models vote BUY, {models_voting_sell} vote SELL")
        
        return " | ".join(explanations)
    
    def add_training_sample(self, features: np.ndarray, outcome: str):
        """Add new training sample from actual market outcome"""
        # Convert outcome to label
        if outcome == 'profitable':
            label = 1  # Buy was correct
        elif outcome == 'loss':
            label = -1  # Sell was correct
        else:
            label = 0  # Hold was correct
        
        self.training_data.append({
            'features': features.flatten(),
            'label': label,
            'timestamp': datetime.now().isoformat()
        })
        
        # Retrain if we have enough new samples
        if len(self.training_data) % 50 == 0:
            logger.info(f"Auto-retraining with {len(self.training_data)} samples")
            self.train_models()
    
    def evaluate_prediction(self, prediction_id: str, actual_outcome: str):
        """Evaluate how well the prediction performed"""
        # Find the prediction
        for pred in self.prediction_history[-100:]:  # Check last 100 predictions
            if pred['timestamp'] == prediction_id:
                # Determine if prediction was correct
                correct = False
                if pred['prediction'] == 'BUY' and actual_outcome == 'profitable':
                    correct = True
                elif pred['prediction'] == 'SELL' and actual_outcome == 'profitable':
                    correct = True
                elif pred['prediction'] == 'HOLD' and actual_outcome == 'neutral':
                    correct = True
                
                # Add training sample
                self.add_training_sample(
                    np.array(pred['features']),
                    actual_outcome
                )
                
                return correct
        return False
    
    def get_model_stats(self) -> Dict:
        """Get current model performance statistics"""
        stats = {
            'is_trained': self.is_trained,
            'training_samples': len(self.training_data),
            'predictions_made': len(self.prediction_history),
            'model_weights': self.model_weights,
            'model_accuracies': {}
        }
        
        for model, scores in self.accuracy_scores.items():
            if scores:
                stats['model_accuracies'][model] = {
                    'current': scores[-1],
                    'average': np.mean(scores),
                    'improving': scores[-1] > scores[0] if len(scores) > 1 else False
                }
        
        return stats
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save each model
            for name, model in self.models.items():
                if model is not None:
                    joblib.dump(model, f'models/{name}.pkl')
            
            # Save scaler
            joblib.dump(self.scaler, 'models/scaler.pkl')
            
            # Save metadata
            metadata = {
                'model_weights': self.model_weights,
                'is_trained': self.is_trained,
                'training_samples': len(self.training_data),
                'accuracy_scores': {k: v[-10:] for k, v in self.accuracy_scores.items()},
                'saved_at': datetime.now().isoformat()
            }
            
            with open('models/metadata.json', 'w') as f:
                json.dump(metadata, f)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load previously trained models"""
        try:
            if not os.path.exists('models'):
                logger.info("No saved models found")
                return
            
            # Load models
            for name in self.models.keys():
                model_path = f'models/{name}.pkl'
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    logger.info(f"Loaded {name} model")
            
            # Load scaler
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
            
            # Load metadata
            if os.path.exists('models/metadata.json'):
                with open('models/metadata.json', 'r') as f:
                    metadata = json.load(f)
                    self.model_weights = metadata['model_weights']
                    self.is_trained = metadata['is_trained']
                    self.accuracy_scores = metadata.get('accuracy_scores', {})
                    logger.info(f"Loaded ML brain: {metadata['training_samples']} training samples")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def continuous_learning_loop(self):
        """Background task for continuous model improvement"""
        while True:
            await asyncio.sleep(3600)  # Every hour
            
            # Retrain if we have new data
            if len(self.training_data) >= 100:
                logger.info("Hourly retraining triggered")
                self.train_models()
            
            # Clean old prediction history
            if len(self.prediction_history) > 10000:
                self.prediction_history = self.prediction_history[-5000:]
            
            # Log stats
            stats = self.get_model_stats()
            logger.info(f"ML Brain Stats: {stats}")


# Singleton instance
ml_brain = AdvancedMLBrain()


async def generate_ml_prediction(market_data: Dict) -> Dict:
    """Generate prediction using ML brain"""
    return ml_brain.predict(market_data)


async def train_ml_models(data: List[Dict]):
    """Train ML models with historical data"""
    for sample in data:
        ml_brain.training_data.append(sample)
    
    return ml_brain.train_models()


def get_ml_stats() -> Dict:
    """Get ML system statistics"""
    return ml_brain.get_model_stats()