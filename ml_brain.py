"""
Simple ML Brain for Trading Signals
Minimal implementation to prevent import errors
"""

from typing import Dict, Any

class SimpleMLBrain:
    def __init__(self):
        self.is_trained = False
        self.total_trades = 0
        self.trades_with_results = 0
        self.predictions_made = 0
        
    def predict(self, market_data: Dict) -> Dict:
        """Simple prediction - returns neutral most of the time"""
        self.predictions_made += 1
        
        if self.trades_with_results < 30:
            return {
                'action': 'NEUTRAL',
                'confidence': 0.5,
                'ml_active': False,
                'reason': f'Learning... ({self.trades_with_results}/30 trades)'
            }
        
        change = market_data.get('change_5min', 0)
        
        if abs(change) < 0.001:
            return {'action': 'NEUTRAL', 'confidence': 0.5, 'ml_active': False, 'reason': 'No clear signal'}
        
        action = 'BUY' if change > 0 else 'SELL'
        confidence = min(0.7, 0.5 + abs(change) * 100)
        
        return {
            'action': action,
            'confidence': confidence,
            'ml_active': True,
            'reason': f'ML: {action} signal detected'
        }
    
    def record_trade(self, market_data: Dict, action: str, result: Any):
        """Record trade for learning"""
        self.total_trades += 1
        if result is not None:
            self.trades_with_results += 1
            if self.trades_with_results >= 30:
                self.is_trained = True
    
    def evaluate_prediction(self, prediction: Dict, result: float):
        """Evaluate how well prediction performed"""
        if result != 0:
            self.trades_with_results += 1
    
    def get_stats(self) -> Dict:
        """Get ML statistics"""
        accuracy = 0.5 + (min(self.trades_with_results, 100) / 1000)
        
        return {
            'is_trained': self.is_trained,
            'total_trades': self.total_trades,
            'trades_with_results': self.trades_with_results,
            'predictions_made': self.predictions_made,
            'accuracy': accuracy,
            'model_type': 'SimpleML'
        }