"""
Personal ML Engine - Advanced Machine Learning with Continuous Improvement
COMPLETELY ISOLATED from customer system - no shared signals or state
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import pickle
import json
from datetime import datetime, timedelta
from collections import deque
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

class PersonalMLEngine:
    """
    Isolated ML system for personal trading
    - Continuous learning from market data
    - Online model updates
    - Performance-weighted ensemble
    """
    
    def __init__(self):
        # Core models ensemble
        self.models = {
            'xgboost': None,
            'random_forest': None,
            'gradient_boost': None,
            'neural_network': None,
            'extra_trees': None
        }
        
        # Model performance tracking
        self.model_performance = {
            name: {
                'accuracy': deque(maxlen=100),
                'predictions': deque(maxlen=1000),
                'last_train': None,
                'total_predictions': 0,
                'successful_predictions': 0
            } for name in self.models.keys()
        }
        
        # Training data management
        self.training_buffer = deque(maxlen=10000)  # Rolling window of training data
        self.feature_names = None
        self.scaler = RobustScaler()
        self.is_trained = False
        
        # Continuous learning parameters
        self.min_samples_to_train = 500
        self.retrain_interval = 3600  # Retrain every hour
        self.online_learning_rate = 0.1
        
        # Performance metrics
        self.prediction_history = deque(maxlen=5000)
        self.feature_importance = {}
        
        # Isolation flag - ensures no cross-contamination
        self.system_type = "PERSONAL_ML_ONLY"
        
    async def initialize_training(self):
        """Initialize ML models with historical data"""
        logger.info("Initializing Personal ML Engine...")
        
        # Generate synthetic training data if no history
        if len(self.training_buffer) < self.min_samples_to_train:
            await self._generate_synthetic_training_data()
        
        # Train initial models
        self.train_models()
        
        # Start continuous learning loop
        asyncio.create_task(self.continuous_learning_loop())
        
        logger.info("Personal ML Engine initialized and learning")
    
    async def _generate_synthetic_training_data(self):
        """Generate synthetic data for initial training"""
        logger.info("Generating synthetic training data...")
        
        for _ in range(1000):
            # Create realistic market features
            features = self._generate_synthetic_features()
            
            # Generate label based on realistic patterns
            label = self._generate_synthetic_label(features)
            
            self.training_buffer.append({
                'features': features,
                'label': label,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 365))
            })
    
    def _generate_synthetic_features(self) -> np.ndarray:
        """Generate realistic market features"""
        features = []
        
        # Technical indicators (normalized)
        features.append(np.random.uniform(0, 100) / 100)  # RSI
        features.append(np.random.normal(0, 0.1))  # MACD
        features.append(np.random.uniform(0, 1))  # BB Position
        features.append(np.random.uniform(0.5, 2))  # Volume Ratio
        features.append(np.random.normal(0, 0.05))  # ROC
        features.append(np.random.uniform(0, 100) / 100)  # MFI
        features.append(np.random.uniform(0.01, 0.05))  # Volatility
        
        # Price patterns
        features.append(np.random.choice([0, 1]))  # Above SMA20
        features.append(np.random.choice([0, 1]))  # Above SMA50
        features.append(np.random.uniform(0, 1))  # Support/Resistance position
        
        # Market regime
        features.append(np.random.uniform(-0.1, 0.1))  # Trend strength
        features.append(np.random.uniform(0, 1))  # Market volatility regime
        
        return np.array(features)
    
    def _generate_synthetic_label(self, features: np.ndarray) -> int:
        """Generate label based on features (simulating market behavior)"""
        # Simple rules to create somewhat realistic labels
        rsi = features[0] * 100
        macd = features[1]
        trend = features[10]
        
        if rsi < 30 and macd < -0.05:
            return 1  # Buy signal likely profitable
        elif rsi > 70 and macd > 0.05:
            return -1  # Sell signal likely profitable
        elif trend > 0.05 and features[7] == 1:  # Uptrend with price above SMA20
            return 1
        elif trend < -0.05 and features[7] == 0:  # Downtrend with price below SMA20
            return -1
        else:
            return 0  # Hold
    
    def prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features from market data"""
        if market_data.empty or len(market_data) < 50:
            return None
        
        latest = market_data.iloc[-1]
        features = []
        
        # Technical indicators
        features.append(latest.get('RSI', 50) / 100)
        features.append(latest.get('MACD', 0))
        features.append(latest.get('BB_Position', 0.5))
        features.append(latest.get('Volume_Ratio', 1))
        features.append(latest.get('ROC_10', 0))
        features.append(latest.get('MFI', 50) / 100)
        features.append(latest.get('Volatility', 0.02))
        
        # Price patterns
        features.append(1 if latest.get('Close', 0) > latest.get('SMA_20', 0) else 0)
        features.append(1 if latest.get('SMA_20', 0) > latest.get('SMA_50', 0) else 0)
        sr_pos = 0.5
        if latest.get('Resistance', 0) > 0 and latest.get('Support', 0) > 0:
            sr_pos = (latest.get('Close', 0) - latest.get('Support', 0)) / (latest.get('Resistance', 0) - latest.get('Support', 0))
        features.append(np.clip(sr_pos, 0, 1))
        
        # Market regime indicators
        features.append(latest.get('Momentum_5', 0) if 'Momentum_5' in market_data.columns else 0)
        features.append(latest.get('ATR', 0.02) / latest.get('Close', 1))
        
        self.feature_names = [
            'RSI_norm', 'MACD', 'BB_Position', 'Volume_Ratio', 'ROC_10', 'MFI_norm',
            'Volatility', 'Above_SMA20', 'SMA20_above_SMA50', 'SR_Position',
            'Momentum_5', 'ATR_Ratio'
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_models(self, incremental: bool = False):
        """Train or update all models"""
        if len(self.training_buffer) < self.min_samples_to_train:
            logger.warning(f"Not enough training data: {len(self.training_buffer)} samples")
            return False
        
        logger.info(f"Training models with {len(self.training_buffer)} samples...")
        
        # Prepare training data
        X = np.array([sample['features'] for sample in self.training_buffer])
        y = np.array([sample['label'] for sample in self.training_buffer])
        
        # Remove any NaN or infinite values
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            return False
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train each model
        models_config = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        }
        
        for name, model in models_config.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Validate
                score = model.score(X_val_scaled, y_val)
                
                # Store model and performance
                self.models[name] = model
                self.model_performance[name]['accuracy'].append(score)
                self.model_performance[name]['last_train'] = datetime.now()
                
                logger.info(f"  {name}: {score:.3f} accuracy")
                
                # Extract feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        self.is_trained = True
        
        # Save models
        self.save_models()
        
        return True
    
    def predict(self, market_data: pd.DataFrame) -> Dict:
        """Make ensemble prediction"""
        if not self.is_trained:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'ml_active': False,
                'reason': 'ML models still training'
            }
        
        # Prepare features
        features = self.prepare_features(market_data)
        if features is None:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'ml_active': False,
                'reason': 'Insufficient data'
            }
        
        # Scale features
        try:
            features_scaled = self.scaler.transform(features)
        except:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'ml_active': False,
                'reason': 'Feature scaling error'
            }
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            if model is None:
                continue
            
            try:
                # Get prediction and probability
                pred = model.predict(features_scaled)[0]
                
                # Get probability estimates
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    # Get confidence for the predicted class
                    if len(proba) == 3:  # 3 classes: sell, hold, buy
                        max_prob = np.max(proba)
                    else:
                        max_prob = np.max(proba)
                else:
                    max_prob = 0.6  # Default confidence if no probability
                
                predictions[name] = pred
                probabilities[name] = max_prob
                
            except Exception as e:
                logger.error(f"Prediction error for {name}: {e}")
                continue
        
        if not predictions:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'ml_active': False,
                'reason': 'No model predictions available'
            }
        
        # Weighted ensemble voting based on model performance
        action_scores = {-1: 0, 0: 0, 1: 0}  # sell, hold, buy
        
        for name, pred in predictions.items():
            # Get model weight based on recent accuracy
            recent_accuracy = np.mean(self.model_performance[name]['accuracy']) if self.model_performance[name]['accuracy'] else 0.5
            weight = recent_accuracy * probabilities[name]
            
            action_scores[pred] += weight
        
        # Determine final action
        best_action = max(action_scores, key=action_scores.get)
        total_weight = sum(action_scores.values())
        confidence = action_scores[best_action] / total_weight if total_weight > 0 else 0
        
        # Map to action string
        action_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        action = action_map[best_action]
        
        # Record prediction for learning
        self.record_prediction({
            'timestamp': datetime.now(),
            'features': features.tolist(),
            'prediction': best_action,
            'confidence': confidence,
            'model_predictions': predictions,
            'model_probabilities': probabilities
        })
        
        return {
            'action': action,
            'confidence': min(confidence, 0.95),
            'ml_active': True,
            'models_agree': len(set(predictions.values())) == 1,
            'dominant_model': max(predictions.items(), key=lambda x: probabilities.get(x[0], 0))[0],
            'reason': self._generate_reason(action, predictions, features[0])
        }
    
    def _generate_reason(self, action: str, predictions: Dict, features: np.ndarray) -> str:
        """Generate explanation for ML prediction"""
        reasons = []
        
        # Count model votes
        votes = {-1: 0, 0: 0, 1: 0}
        for pred in predictions.values():
            votes[pred] += 1
        
        if votes[1] > votes[-1]:
            reasons.append(f"{votes[1]}/{len(predictions)} models bullish")
        elif votes[-1] > votes[1]:
            reasons.append(f"{votes[-1]}/{len(predictions)} models bearish")
        
        # Add feature-based reasoning
        try:
            rsi = features[0] * 100
            if rsi < 30:
                reasons.append("Oversold conditions")
            elif rsi > 70:
                reasons.append("Overbought conditions")
        except:
            pass
        
        return " | ".join(reasons) if reasons else f"ML consensus: {action}"
    
    def record_prediction(self, prediction: Dict):
        """Record prediction for future learning"""
        self.prediction_history.append(prediction)
        
        # Update model prediction counts
        for model_name, pred in prediction['model_predictions'].items():
            self.model_performance[model_name]['total_predictions'] += 1
    
    def update_with_result(self, prediction_id: datetime, actual_outcome: str):
        """Update models with actual outcome"""
        # Find the prediction
        prediction = None
        for p in self.prediction_history:
            if p['timestamp'] == prediction_id:
                prediction = p
                break
        
        if not prediction:
            return
        
        # Determine if prediction was correct
        actual_label = 1 if actual_outcome == 'profitable' else -1 if actual_outcome == 'loss' else 0
        predicted_label = prediction['prediction']
        
        # Update model performance
        for model_name, model_pred in prediction['model_predictions'].items():
            if model_pred == actual_label:
                self.model_performance[model_name]['successful_predictions'] += 1
        
        # Add to training buffer for continuous learning
        self.training_buffer.append({
            'features': prediction['features'],
            'label': actual_label,
            'timestamp': datetime.now()
        })
    
    async def continuous_learning_loop(self):
        """Continuously retrain models with new data"""
        while True:
            await asyncio.sleep(self.retrain_interval)
            
            try:
                # Check if we have enough new data
                if len(self.training_buffer) >= self.min_samples_to_train:
                    logger.info("Continuous learning: Retraining models...")
                    
                    # Retrain models
                    success = self.train_models(incremental=True)
                    
                    if success:
                        logger.info("Models successfully retrained")
                        
                        # Adjust model weights based on recent performance
                        self._update_model_weights()
                    
                # Clean old predictions
                if len(self.prediction_history) > 5000:
                    self.prediction_history = deque(list(self.prediction_history)[-2500:], maxlen=5000)
                
            except Exception as e:
                logger.error(f"Error in continuous learning: {e}")
    
    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        for name in self.models.keys():
            perf = self.model_performance[name]
            if perf['total_predictions'] > 0:
                success_rate = perf['successful_predictions'] / perf['total_predictions']
                # Exponential moving average of performance
                if perf['accuracy']:
                    perf['accuracy'].append(success_rate * 0.3 + np.mean(perf['accuracy']) * 0.7)
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs('models/personal', exist_ok=True)
            
            # Save each model
            for name, model in self.models.items():
                if model is not None:
                    with open(f'models/personal/{name}.pkl', 'wb') as f:
                        pickle.dump(model, f)
            
            # Save scaler
            with open('models/personal/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save metadata
            metadata = {
                'is_trained': self.is_trained,
                'training_samples': len(self.training_buffer),
                'model_performance': {
                    name: {
                        'accuracy': list(perf['accuracy'])[-10:],
                        'total_predictions': perf['total_predictions'],
                        'successful_predictions': perf['successful_predictions']
                    } for name, perf in self.model_performance.items()
                },
                'feature_names': self.feature_names,
                'saved_at': datetime.now().isoformat()
            }
            
            with open('models/personal/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load previously trained models"""
        try:
            model_dir = 'models/personal'
            if not os.path.exists(model_dir):
                return False
            
            # Load models
            for name in self.models.keys():
                model_path = f'{model_dir}/{name}.pkl'
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            # Load scaler
            scaler_path = f'{model_dir}/scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load metadata
            metadata_path = f'{model_dir}/metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.is_trained = metadata.get('is_trained', False)
                    self.feature_names = metadata.get('feature_names')
                    
                    # Restore performance metrics
                    for name, perf in metadata.get('model_performance', {}).items():
                        if name in self.model_performance:
                            self.model_performance[name]['accuracy'] = deque(perf.get('accuracy', []), maxlen=100)
                            self.model_performance[name]['total_predictions'] = perf.get('total_predictions', 0)
                            self.model_performance[name]['successful_predictions'] = perf.get('successful_predictions', 0)
            
            logger.info(f"Loaded personal ML models from {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """Get current ML performance statistics"""
        stats = {
            'is_trained': self.is_trained,
            'training_samples': len(self.training_buffer),
            'total_predictions': len(self.prediction_history),
            'models': {}
        }
        
        for name, perf in self.model_performance.items():
            recent_accuracy = np.mean(perf['accuracy']) if perf['accuracy'] else 0
            success_rate = perf['successful_predictions'] / perf['total_predictions'] if perf['total_predictions'] > 0 else 0
            
            stats['models'][name] = {
                'recent_accuracy': recent_accuracy,
                'success_rate': success_rate,
                'total_predictions': perf['total_predictions'],
                'last_trained': perf['last_train'].isoformat() if perf['last_train'] else None
            }
        
        # Best performing model
        if stats['models']:
            best_model = max(stats['models'].items(), key=lambda x: x[1]['recent_accuracy'])
            stats['best_model'] = best_model[0]
            stats['best_accuracy'] = best_model[1]['recent_accuracy']
        
        return stats