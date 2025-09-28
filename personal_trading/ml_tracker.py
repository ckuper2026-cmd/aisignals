"""
ML Progress Tracker - Monitor your ML performance over time
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class MLProgressTracker:
    def __init__(self, data_dir: str = "ml_tracking"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Tracking files
        self.predictions_file = self.data_dir / "predictions.json"
        self.metrics_file = self.data_dir / "metrics.json"
        self.model_performance_file = self.data_dir / "model_performance.json"
        
        # Load existing data
        self.predictions = self._load_json(self.predictions_file, default=[])
        self.metrics = self._load_json(self.metrics_file, default={})
        self.model_performance = self._load_json(self.model_performance_file, default={})
    
    def _load_json(self, filepath: Path, default=None):
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return default if default is not None else {}
    
    def _save_json(self, data, filepath: Path):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def record_prediction(self, prediction_data: Dict):
        """Record each ML prediction"""
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'symbol': prediction_data.get('symbol'),
            'action': prediction_data.get('action'),
            'confidence': prediction_data.get('confidence'),
            'models_agree': prediction_data.get('models_agree'),
            'dominant_model': prediction_data.get('dominant_model'),
            'price': prediction_data.get('price'),
            'outcome': None,  # Updated when trade closes
            'pnl': None
        }
        
        self.predictions.append(prediction)
        self._save_json(self.predictions, self.predictions_file)
        
        return len(self.predictions) - 1  # Return index for later update
    
    def update_outcome(self, prediction_idx: int, outcome: str, pnl: float):
        """Update prediction with actual outcome"""
        if 0 <= prediction_idx < len(self.predictions):
            self.predictions[prediction_idx]['outcome'] = outcome
            self.predictions[prediction_idx]['pnl'] = pnl
            self._save_json(self.predictions, self.predictions_file)
    
    def calculate_metrics(self) -> Dict:
        """Calculate current ML performance metrics"""
        if not self.predictions:
            return {
                'total_predictions': 0,
                'accuracy': 0,
                'avg_confidence': 0,
                'profit_factor': 0
            }
        
        # Filter completed predictions
        completed = [p for p in self.predictions if p['outcome'] is not None]
        
        if not completed:
            return {
                'total_predictions': len(self.predictions),
                'completed_predictions': 0,
                'accuracy': 0,
                'avg_confidence': np.mean([p['confidence'] for p in self.predictions])
            }
        
        # Calculate metrics
        correct = sum(1 for p in completed if p['outcome'] == 'profitable')
        accuracy = correct / len(completed)
        
        # Profit metrics
        profits = [p['pnl'] for p in completed if p['pnl'] and p['pnl'] > 0]
        losses = [abs(p['pnl']) for p in completed if p['pnl'] and p['pnl'] < 0]
        
        profit_factor = sum(profits) / sum(losses) if losses else float('inf')
        
        # Confidence correlation
        conf_correct = [p['confidence'] for p in completed if p['outcome'] == 'profitable']
        conf_incorrect = [p['confidence'] for p in completed if p['outcome'] != 'profitable']
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(self.predictions),
            'completed_predictions': len(completed),
            'accuracy': accuracy,
            'avg_confidence': np.mean([p['confidence'] for p in self.predictions]),
            'avg_confidence_correct': np.mean(conf_correct) if conf_correct else 0,
            'avg_confidence_incorrect': np.mean(conf_incorrect) if conf_incorrect else 0,
            'profit_factor': profit_factor,
            'total_pnl': sum(p['pnl'] for p in completed if p['pnl']),
            'win_rate': accuracy,
            'avg_win': np.mean(profits) if profits else 0,
            'avg_loss': np.mean(losses) if losses else 0
        }
        
        # Save metrics history
        date_key = datetime.now().strftime("%Y-%m-%d")
        if date_key not in self.metrics:
            self.metrics[date_key] = []
        self.metrics[date_key].append(metrics)
        self._save_json(self.metrics, self.metrics_file)
        
        return metrics
    
    def track_model_performance(self, model_stats: Dict):
        """Track individual model performance"""
        timestamp = datetime.now().isoformat()
        
        for model_name, stats in model_stats.items():
            if model_name not in self.model_performance:
                self.model_performance[model_name] = []
            
            self.model_performance[model_name].append({
                'timestamp': timestamp,
                'accuracy': stats.get('accuracy'),
                'predictions': stats.get('total_predictions'),
                'successful': stats.get('successful_predictions')
            })
        
        # Keep only last 100 entries per model
        for model_name in self.model_performance:
            self.model_performance[model_name] = self.model_performance[model_name][-100:]
        
        self._save_json(self.model_performance, self.model_performance_file)
    
    def get_progress_report(self) -> str:
        """Generate progress report"""
        metrics = self.calculate_metrics()
        
        # Calculate trends
        accuracy_trend = self._calculate_trend('accuracy')
        confidence_trend = self._calculate_trend('avg_confidence')
        
        report = f"""
ML PROGRESS REPORT - {datetime.now().strftime("%Y-%m-%d %H:%M")}
{'='*60}

OVERALL METRICS:
  Total Predictions: {metrics['total_predictions']}
  Completed Trades: {metrics['completed_predictions']}
  Current Accuracy: {metrics['accuracy']:.1%}
  Accuracy Trend: {accuracy_trend}
  
CONFIDENCE ANALYSIS:
  Avg Confidence: {metrics['avg_confidence']:.1%}
  Confidence (Correct): {metrics['avg_confidence_correct']:.1%}
  Confidence (Wrong): {metrics['avg_confidence_incorrect']:.1%}
  Confidence Trend: {confidence_trend}
  
PROFIT METRICS:
  Profit Factor: {metrics['profit_factor']:.2f}
  Total P&L: ${metrics['total_pnl']:.2f}
  Avg Win: ${metrics['avg_win']:.2f}
  Avg Loss: ${metrics['avg_loss']:.2f}

MODEL PERFORMANCE:
"""
        
        # Add model-specific stats
        for model_name, history in self.model_performance.items():
            if history:
                recent = history[-1]
                report += f"  {model_name}: {recent.get('accuracy', 0):.1%} accuracy ({recent.get('predictions', 0)} predictions)\n"
        
        return report
    
    def _calculate_trend(self, metric: str, days: int = 7) -> str:
        """Calculate trend for a metric"""
        start_date = datetime.now() - timedelta(days=days)
        
        values = []
        for date_str, day_metrics in self.metrics.items():
            date = datetime.fromisoformat(date_str)
            if date >= start_date:
                for m in day_metrics:
                    if metric in m:
                        values.append(m[metric])
        
        if len(values) < 2:
            return "→ Stable"
        
        # Simple linear trend
        trend = (values[-1] - values[0]) / len(values)
        
        if trend > 0.01:
            return "↑ Improving"
        elif trend < -0.01:
            return "↓ Declining"
        else:
            return "→ Stable"
    
    def plot_progress(self, save_path: str = None):
        """Generate progress plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Accuracy over time
        accuracy_data = []
        for date_str, day_metrics in self.metrics.items():
            for m in day_metrics:
                accuracy_data.append({
                    'date': datetime.fromisoformat(date_str),
                    'accuracy': m['accuracy']
                })
        
        if accuracy_data:
            df = pd.DataFrame(accuracy_data)
            axes[0, 0].plot(df['date'], df['accuracy'] * 100, marker='o')
            axes[0, 0].set_title('Accuracy Over Time')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence vs Accuracy
        completed = [p for p in self.predictions if p['outcome'] is not None]
        if completed:
            correct = [p['confidence'] for p in completed if p['outcome'] == 'profitable']
            incorrect = [p['confidence'] for p in completed if p['outcome'] != 'profitable']
            
            axes[0, 1].hist([correct, incorrect], label=['Correct', 'Incorrect'], alpha=0.7, bins=20)
            axes[0, 1].set_title('Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].legend()
        
        # 3. Model Performance Comparison
        model_names = []
        model_accuracies = []
        
        for model_name, history in self.model_performance.items():
            if history:
                model_names.append(model_name)
                recent_acc = [h['accuracy'] for h in history[-10:] if h.get('accuracy')]
                model_accuracies.append(np.mean(recent_acc) if recent_acc else 0)
        
        if model_names:
            axes[1, 0].bar(model_names, model_accuracies)
            axes[1, 0].set_title('Model Performance Comparison')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_xticklabels(model_names, rotation=45)
        
        # 4. P&L Curve
        pnl_cumulative = []
        cumsum = 0
        for p in self.predictions:
            if p['pnl'] is not None:
                cumsum += p['pnl']
                pnl_cumulative.append(cumsum)
        
        if pnl_cumulative:
            axes[1, 1].plot(pnl_cumulative)
            axes[1, 1].set_title('Cumulative P&L')
            axes[1, 1].set_ylabel('P&L ($)')
            axes[1, 1].set_xlabel('Trade #')
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        return fig


def integrate_tracker(trading_system):
    """Integrate tracker with trading system"""
    tracker = MLProgressTracker()
    
    # Monkey patch to add tracking
    original_execute = trading_system.execute_signals
    
    async def tracked_execute(signals, paper_trade=True):
        results = await original_execute(signals, paper_trade)
        
        # Track each execution
        for i, result in enumerate(results):
            if result['success'] and i < len(signals):
                signal = signals[i]
                pred_idx = tracker.record_prediction({
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'price': signal.entry_price,
                    'models_agree': True  # Update based on actual ML
                })
                
                # Store index for later outcome update
                result['prediction_idx'] = pred_idx
        
        return results
    
    trading_system.execute_signals = tracked_execute
    trading_system.ml_tracker = tracker
    
    return tracker


# CLI for checking progress
if __name__ == "__main__":
    import sys
    
    tracker = MLProgressTracker()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "report":
            print(tracker.get_progress_report())
        
        elif command == "plot":
            save_path = sys.argv[2] if len(sys.argv) > 2 else "ml_progress.png"
            tracker.plot_progress(save_path)
            print(f"Plots saved to {save_path}")
        
        elif command == "metrics":
            metrics = tracker.calculate_metrics()
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
        
        else:
            print("Usage: python ml_tracker.py [report|plot|metrics]")
    else:
        # Default: show report
        print(tracker.get_progress_report())