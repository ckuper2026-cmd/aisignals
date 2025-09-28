#!/usr/bin/env python3
"""
Track ML Progress - Simple commands to monitor your ML performance
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Quick command to check ML status
def check_ml_status():
    """Check current ML training status and performance"""
    
    # Check if models exist
    model_dir = Path("models/personal")
    
    if not model_dir.exists():
        print("No models trained yet")
        return
    
    # Load metadata
    metadata_file = model_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"ML Status:")
        print(f"  Trained: {metadata.get('is_trained', False)}")
        print(f"  Training Samples: {metadata.get('training_samples', 0)}")
        print(f"  Last Updated: {metadata.get('saved_at', 'Never')}")
        
        # Model performance
        if 'model_performance' in metadata:
            print(f"\nModel Accuracies:")
            for model, perf in metadata['model_performance'].items():
                acc = perf.get('accuracy', [])
                if acc:
                    print(f"  {model}: {acc[-1]:.1%}")
    else:
        print("No model metadata found")

# Quick command to see recent predictions
def check_recent_predictions():
    """Show recent ML predictions and outcomes"""
    
    predictions_file = Path("ml_tracking/predictions.json")
    
    if not predictions_file.exists():
        print("No predictions recorded yet")
        return
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Show last 10 predictions
    recent = predictions[-10:] if len(predictions) > 10 else predictions
    
    print(f"\nRecent Predictions ({len(recent)} shown):")
    print("-" * 60)
    
    correct = 0
    total_with_outcome = 0
    
    for pred in recent:
        timestamp = pred['timestamp'].split('T')[0]
        symbol = pred.get('symbol', 'N/A')
        action = pred.get('action', 'N/A')
        confidence = pred.get('confidence', 0) * 100
        outcome = pred.get('outcome', 'pending')
        
        print(f"{timestamp} | {symbol:6} | {action:4} | Conf: {confidence:5.1f}% | {outcome}")
        
        if outcome and outcome != 'pending':
            total_with_outcome += 1
            if outcome == 'profitable':
                correct += 1
    
    if total_with_outcome > 0:
        accuracy = (correct / total_with_outcome) * 100
        print("-" * 60)
        print(f"Recent Accuracy: {accuracy:.1f}% ({correct}/{total_with_outcome} correct)")

# Track ML improvement over time
def track_improvement():
    """Show ML improvement metrics over time"""
    
    metrics_file = Path("ml_tracking/metrics.json")
    
    if not metrics_file.exists():
        print("No metrics history yet")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Calculate weekly averages
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    two_weeks_ago = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    
    this_week = []
    last_week = []
    
    for date_str, day_metrics in metrics.items():
        if date_str >= week_ago:
            this_week.extend(day_metrics)
        elif date_str >= two_weeks_ago:
            last_week.extend(day_metrics)
    
    if this_week:
        avg_accuracy_now = sum(m['accuracy'] for m in this_week) / len(this_week)
        avg_confidence_now = sum(m['avg_confidence'] for m in this_week) / len(this_week)
        print(f"\nThis Week:")
        print(f"  Avg Accuracy: {avg_accuracy_now:.1%}")
        print(f"  Avg Confidence: {avg_confidence_now:.1%}")
    
    if last_week:
        avg_accuracy_last = sum(m['accuracy'] for m in last_week) / len(last_week)
        avg_confidence_last = sum(m['avg_confidence'] for m in last_week) / len(last_week)
        print(f"\nLast Week:")
        print(f"  Avg Accuracy: {avg_accuracy_last:.1%}")
        print(f"  Avg Confidence: {avg_confidence_last:.1%}")
        
        if this_week:
            acc_change = (avg_accuracy_now - avg_accuracy_last) * 100
            print(f"\nWeekly Change:")
            print(f"  Accuracy: {'+' if acc_change > 0 else ''}{acc_change:.1f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            check_ml_status()
        elif command == "predictions":
            check_recent_predictions()
        elif command == "improvement":
            track_improvement()
        else:
            print("Commands: status, predictions, improvement")
    else:
        # Show all
        check_ml_status()
        check_recent_predictions()
        track_improvement()