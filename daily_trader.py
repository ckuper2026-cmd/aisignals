#!/usr/bin/env python3
"""
Daily Performance Tracker
Generates daily and weekly performance reports
"""

import json
import pandas as pd
from datetime import datetime, timedelta
import os
import httpx
import asyncio

class DailyTracker:
    def __init__(self):
        self.data_file = "daily_performance.json"
        self.load_data()
        
    def load_data(self):
        """Load existing performance data"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                'days': [],
                'summary': {}
            }
    
    def save_data(self):
        """Save performance data"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    async def collect_daily_metrics(self):
        """Collect end-of-day metrics"""
        metrics = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'day_of_week': datetime.now().strftime('%A'),
            'signals_generated': 0,
            'trades_executed': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'best_trade': None,
            'worst_trade': None,
            'api_calls': {
                'yfinance': 0,
                'polygon': 0
            },
            'cache_stats': {
                'hit_rate': 0.0,
                'size': 0
            },
            'volatility_zones': {
                'opening_bell': {'signals': 0, 'trades': 0},
                'power_hour': {'signals': 0, 'trades': 0},
                'closing_cross': {'signals': 0, 'trades': 0}
            },
            'errors': [],
            'notes': ""
        }
        
        # Try to get actual metrics from API
        try:
            async with httpx.AsyncClient() as client:
                # Get rate limits
                resp = await client.get("http://localhost:8001/api/rate-limits")
                if resp.status_code == 200:
                    data = resp.json()
                    metrics['api_calls']['yfinance'] = data['yfinance']['used']
                    metrics['api_calls']['polygon'] = data['polygon']['used']
                    metrics['cache_stats']['size'] = data['cache_stats']['entries']
                
                # Get signals
                resp = await client.get("http://localhost:8001/api/signals")
                if resp.status_code == 200:
                    data = resp.json()
                    metrics['signals_generated'] = data.get('count', 0)
        except:
            pass
        
        return metrics
    
    def add_manual_metrics(self, metrics):
        """Add manual observations"""
        print("\n📝 MANUAL DATA ENTRY")
        print("-" * 40)
        
        # Get user input
        try:
            trades = input("Trades executed today (default 0): ").strip()
            metrics['trades_executed'] = int(trades) if trades else 0
            
            wins = input("Winning trades (default 0): ").strip()
            metrics['winning_trades'] = int(wins) if wins else 0
            
            losses = input("Losing trades (default 0): ").strip()
            metrics['losing_trades'] = int(losses) if losses else 0
            
            pnl = input("Total P&L in $ (e.g., 250.50 or -125.75): ").strip()
            metrics['total_pnl'] = float(pnl) if pnl else 0.0
            
            # Calculate win rate
            total = metrics['winning_trades'] + metrics['losing_trades']
            if total > 0:
                metrics['win_rate'] = (metrics['winning_trades'] / total) * 100
            
            notes = input("Notes/Observations: ").strip()
            metrics['notes'] = notes
            
        except ValueError:
            print("Invalid input, using defaults")
        
        return metrics
    
    def generate_daily_report(self, metrics):
        """Generate daily performance report"""
        print("\n" + "="*60)
        print(f"    DAILY PERFORMANCE REPORT - {metrics['date']}")
        print("="*60)
        
        print(f"\n📊 TRADING METRICS")
        print("-" * 40)
        print(f"  Signals Generated:  {metrics['signals_generated']}")
        print(f"  Trades Executed:    {metrics['trades_executed']}")
        print(f"  Winning Trades:     {metrics['winning_trades']}")
        print(f"  Losing Trades:      {metrics['losing_trades']}")
        print(f"  Win Rate:           {metrics['win_rate']:.1f}%")
        print(f"  Total P&L:          ${metrics['total_pnl']:+.2f}")
        
        print(f"\n🌐 API USAGE")
        print("-" * 40)
        print(f"  YFinance Calls:     {metrics['api_calls']['yfinance']}")
        print(f"  Polygon Calls:      {metrics['api_calls']['polygon']}")
        print(f"  Cache Size:         {metrics['cache_stats']['size']} entries")
        
        if metrics['notes']:
            print(f"\n📝 NOTES")
            print("-" * 40)
            print(f"  {metrics['notes']}")
        
        # Performance rating
        print(f"\n⭐ DAILY RATING")
        print("-" * 40)
        
        if metrics['total_pnl'] > 1000:
            rating = "EXCELLENT"
            stars = "⭐⭐⭐⭐⭐"
        elif metrics['total_pnl'] > 500:
            rating = "GOOD"
            stars = "⭐⭐⭐⭐"
        elif metrics['total_pnl'] > 0:
            rating = "POSITIVE"
            stars = "⭐⭐⭐"
        elif metrics['total_pnl'] > -500:
            rating = "BREAK-EVEN"
            stars = "⭐⭐"
        else:
            rating = "NEEDS IMPROVEMENT"
            stars = "⭐"
        
        print(f"  Rating: {rating} {stars}")
        
        print("\n" + "="*60)
    
    def generate_weekly_summary(self):
        """Generate weekly performance summary"""
        if len(self.data['days']) < 1:
            print("No data to summarize")
            return
        
        # Get last 5 trading days
        recent_days = self.data['days'][-5:]
        
        print("\n" + "="*60)
        print("         WEEKLY PERFORMANCE SUMMARY")
        print("="*60)
        
        # Calculate totals
        total_signals = sum(d['signals_generated'] for d in recent_days)
        total_trades = sum(d['trades_executed'] for d in recent_days)
        total_wins = sum(d['winning_trades'] for d in recent_days)
        total_losses = sum(d['losing_trades'] for d in recent_days)
        total_pnl = sum(d['total_pnl'] for d in recent_days)
        total_api_calls = sum(d['api_calls']['yfinance'] + d['api_calls']['polygon'] for d in recent_days)
        
        print(f"\n📊 WEEK TOTALS")
        print("-" * 40)
        print(f"  Trading Days:       {len(recent_days)}")
        print(f"  Total Signals:      {total_signals}")
        print(f"  Total Trades:       {total_trades}")
        print(f"  Total Wins:         {total_wins}")
        print(f"  Total Losses:       {total_losses}")
        
        if total_wins + total_losses > 0:
            week_win_rate = (total_wins / (total_wins + total_losses)) * 100
            print(f"  Week Win Rate:      {week_win_rate:.1f}%")
        
        print(f"  Week P&L:           ${total_pnl:+.2f}")
        
        print(f"\n📈 DAILY BREAKDOWN")
        print("-" * 40)
        for day in recent_days:
            print(f"  {day['date']}: ${day['total_pnl']:+8.2f} ({day['winning_trades']}W/{day['losing_trades']}L)")
        
        print(f"\n🌐 API USAGE (WEEK)")
        print("-" * 40)
        print(f"  Total API Calls:    {total_api_calls}")
        print(f"  Daily Average:      {total_api_calls / len(recent_days):.0f}")
        
        # Best and worst days
        if recent_days:
            best_day = max(recent_days, key=lambda x: x['total_pnl'])
            worst_day = min(recent_days, key=lambda x: x['total_pnl'])
            
            print(f"\n🏆 HIGHLIGHTS")
            print("-" * 40)
            print(f"  Best Day:  {best_day['date']} (${best_day['total_pnl']:+.2f})")
            print(f"  Worst Day: {worst_day['date']} (${worst_day['total_pnl']:+.2f})")
        
        # Calculate daily average
        if len(recent_days) > 0:
            avg_daily_pnl = total_pnl / len(recent_days)
            print(f"  Daily Avg P&L:      ${avg_daily_pnl:+.2f}")
        
        # Projection
        if len(recent_days) >= 3:
            # Simple projection based on average
            monthly_projection = avg_daily_pnl * 20  # 20 trading days
            annual_projection = avg_daily_pnl * 252  # 252 trading days
            
            print(f"\n🔮 PROJECTIONS (if current rate continues)")
            print("-" * 40)
            print(f"  Monthly (20 days):  ${monthly_projection:+,.2f}")
            print(f"  Annual (252 days):  ${annual_projection:+,.2f}")
            
            # On $100k account
            monthly_return = (monthly_projection / 100000) * 100
            annual_return = (annual_projection / 100000) * 100
            print(f"  Monthly Return:     {monthly_return:+.1f}%")
            print(f"  Annual Return:      {annual_return:+.1f}%")
        
        print("\n" + "="*60)
    
    async def run_daily_tracking(self):
        """Run daily tracking routine"""
        print("="*60)
        print("       DAILY PERFORMANCE TRACKER")
        print("="*60)
        
        # Collect metrics
        print("\nCollecting automated metrics...")
        metrics = await self.collect_daily_metrics()
        
        print(f"Date: {metrics['date']}")
        print(f"Day: {metrics['day_of_week']}")
        
        # Get manual input
        metrics = self.add_manual_metrics(metrics)
        
        # Generate report
        self.generate_daily_report(metrics)
        
        # Save to file
        self.data['days'].append(metrics)
        self.save_data()
        
        # Ask if user wants weekly summary
        if len(self.data['days']) >= 5:
            show_weekly = input("\nShow weekly summary? (y/n): ").lower()
            if show_weekly == 'y':
                self.generate_weekly_summary()
        
        print("\n✅ Daily metrics saved to daily_performance.json")

async def main():
    tracker = DailyTracker()
    await tracker.run_daily_tracking()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DAILY PERFORMANCE TRACKER")
    print("="*60)
    print("\nRun this at the end of each trading day (after 4 PM)")
    print("It will collect metrics and generate reports\n")
    
    asyncio.run(main())