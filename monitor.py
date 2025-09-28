"""
Performance Monitoring Dashboard
Real-time tracking of trading system performance
"""

import asyncio
import httpx
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import os
from collections import defaultdict

class PerformanceMonitor:
    """Monitor and analyze trading performance"""
    
    def __init__(self, trading_url="http://localhost:8001", infra_url="http://localhost:8000"):
        self.trading_url = trading_url
        self.infra_url = infra_url
        self.metrics = {
            'signals_generated': 0,
            'trades_executed': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'best_trade': {'symbol': None, 'pnl': 0},
            'worst_trade': {'symbol': None, 'pnl': 0},
            'api_calls': defaultdict(int),
            'cache_hits': 0,
            'cache_misses': 0,
            'volatility_zone_performance': defaultdict(lambda: {'trades': 0, 'pnl': 0}),
            'strategy_performance': defaultdict(lambda: {'signals': 0, 'success_rate': 0}),
            'hourly_performance': defaultdict(lambda: {'signals': 0, 'trades': 0, 'pnl': 0})
        }
        
        self.trades = []
        self.signals = []
        self.daily_pnl = []
        
    async def collect_metrics(self):
        """Collect current performance metrics"""
        async with httpx.AsyncClient() as client:
            try:
                # Get system status
                resp = await client.get(f"{self.trading_url}/")
                if resp.status_code == 200:
                    data = resp.json()
                    
                    # Track API usage
                    self.metrics['api_calls']['yfinance'] = data['data_sources']['yfinance'].get('wait_time', 0)
                    self.metrics['api_calls']['polygon'] = data['data_sources']['polygon'].get('wait_time', 0)
                    self.metrics['cache_size'] = data.get('cache_size', 0)
                    
                    # Current volatility state
                    self.metrics['current_volatility'] = data['volatility']
                
                # Get signals
                resp = await client.get(f"{self.trading_url}/api/signals")
                if resp.status_code == 200:
                    data = resp.json()
                    new_signals = data.get('signals', [])
                    
                    for signal in new_signals:
                        if signal not in self.signals:
                            self.signals.append(signal)
                            self.metrics['signals_generated'] += 1
                            
                            # Track by strategy
                            strategy = signal.get('strategy', 'unknown')
                            self.metrics['strategy_performance'][strategy]['signals'] += 1
                            
                            # Track by hour
                            hour = datetime.now().hour
                            self.metrics['hourly_performance'][hour]['signals'] += 1
                
                # Get rate limits
                resp = await client.get(f"{self.trading_url}/api/rate-limits")
                if resp.status_code == 200:
                    data = resp.json()
                    self.metrics['rate_limits'] = data
                    
            except Exception as e:
                print(f"Error collecting metrics: {e}")
    
    def calculate_performance(self):
        """Calculate performance statistics"""
        if self.trades:
            # Win rate
            profitable = sum(1 for t in self.trades if t['pnl'] > 0)
            self.metrics['profitable_trades'] = profitable
            self.metrics['losing_trades'] = len(self.trades) - profitable
            self.metrics['win_rate'] = profitable / len(self.trades) * 100
            
            # Average win/loss
            wins = [t['pnl'] for t in self.trades if t['pnl'] > 0]
            losses = [t['pnl'] for t in self.trades if t['pnl'] < 0]
            
            if wins:
                self.metrics['average_win'] = sum(wins) / len(wins)
            if losses:
                self.metrics['average_loss'] = sum(losses) / len(losses)
            
            # Best/worst trades
            if self.trades:
                best = max(self.trades, key=lambda x: x['pnl'])
                worst = min(self.trades, key=lambda x: x['pnl'])
                self.metrics['best_trade'] = {'symbol': best['symbol'], 'pnl': best['pnl']}
                self.metrics['worst_trade'] = {'symbol': worst['symbol'], 'pnl': worst['pnl']}
            
            # Calculate Sharpe ratio (simplified)
            if self.daily_pnl and len(self.daily_pnl) > 1:
                returns = pd.Series(self.daily_pnl).pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    self.metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * (252 ** 0.5)
    
    def print_dashboard(self):
        """Print performance dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("                    TRADING PERFORMANCE DASHBOARD")
        print("=" * 80)
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Trading Performance
        print("\n📊 TRADING PERFORMANCE")
        print("-" * 40)
        print(f"  Signals Generated:    {self.metrics['signals_generated']}")
        print(f"  Trades Executed:      {self.metrics['trades_executed']}")
        print(f"  Win Rate:            {self.metrics['win_rate']:.1f}%")
        print(f"  Profitable Trades:    {self.metrics['profitable_trades']}")
        print(f"  Losing Trades:       {self.metrics['losing_trades']}")
        print(f"  Total P&L:           ${self.metrics['total_pnl']:.2f}")
        
        # Best/Worst Trades
        print("\n💰 BEST/WORST TRADES")
        print("-" * 40)
        if self.metrics['best_trade']['symbol']:
            print(f"  Best Trade:  {self.metrics['best_trade']['symbol']} +${self.metrics['best_trade']['pnl']:.2f}")
        if self.metrics['worst_trade']['symbol']:
            print(f"  Worst Trade: {self.metrics['worst_trade']['symbol']} -${abs(self.metrics['worst_trade']['pnl']):.2f}")
        
        # Risk Metrics
        print("\n⚠️ RISK METRICS")
        print("-" * 40)
        print(f"  Sharpe Ratio:        {self.metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        {self.metrics['max_drawdown']:.1f}%")
        print(f"  Average Win:         ${self.metrics['average_win']:.2f}")
        print(f"  Average Loss:        ${abs(self.metrics['average_loss']):.2f}")
        
        # API Usage
        print("\n🌐 API USAGE")
        print("-" * 40)
        if 'rate_limits' in self.metrics:
            rl = self.metrics['rate_limits']
            yf = rl.get('yfinance', {})
            pg = rl.get('polygon', {})
            print(f"  YFinance:   {yf.get('used', 0)}/{yf.get('limit', 2000)} (wait: {yf.get('wait_time', 0)}s)")
            print(f"  Polygon:    {pg.get('used', 0)}/{pg.get('limit', 5)} (wait: {pg.get('wait_time', 0)}s)")
            print(f"  Cache Size: {self.metrics.get('cache_size', 0)} entries")
        
        # Volatility Zone Performance
        print("\n🔥 VOLATILITY ZONES")
        print("-" * 40)
        if 'current_volatility' in self.metrics:
            v = self.metrics['current_volatility']
            print(f"  Current Zone: {'ACTIVE' if v.get('active') else 'NORMAL'}")
            print(f"  Multiplier:   {v.get('multiplier', 1.0)}x")
            print(f"  Threshold:    {v.get('threshold', 0.45) * 100:.0f}%")
        
        for zone, perf in self.metrics['volatility_zone_performance'].items():
            if perf['trades'] > 0:
                print(f"  {zone}: {perf['trades']} trades, P&L: ${perf['pnl']:.2f}")
        
        # Strategy Performance
        print("\n📈 STRATEGY PERFORMANCE")
        print("-" * 40)
        for strategy, perf in self.metrics['strategy_performance'].items():
            if perf['signals'] > 0:
                print(f"  {strategy}: {perf['signals']} signals")
        
        # Hourly Activity
        print("\n⏰ HOURLY ACTIVITY")
        print("-" * 40)
        current_hour = datetime.now().hour
        for hour in range(9, 17):  # Market hours
            perf = self.metrics['hourly_performance'][hour]
            marker = " ←" if hour == current_hour else ""
            if perf['signals'] > 0:
                print(f"  {hour:02d}:00 - Signals: {perf['signals']}, Trades: {perf['trades']}{marker}")
        
        print("\n" + "=" * 80)
        print("  Press Ctrl+C to stop monitoring")
        print("=" * 80)
    
    async def monitor_loop(self, refresh_interval=30):
        """Main monitoring loop"""
        print("Starting performance monitoring...")
        print(f"Refresh interval: {refresh_interval} seconds")
        
        while True:
            try:
                await self.collect_metrics()
                self.calculate_performance()
                self.print_dashboard()
                await asyncio.sleep(refresh_interval)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(refresh_interval)

    def save_metrics(self, filename="performance_metrics.json"):
        """Save metrics to file"""
        with open(filename, 'w') as f:
            # Convert defaultdicts to regular dicts for JSON serialization
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    k: dict(v) if isinstance(v, defaultdict) else v
                    for k, v in self.metrics.items()
                },
                'trades': self.trades,
                'signals': self.signals
            }
            json.dump(save_data, f, indent=2, default=str)
        print(f"Metrics saved to {filename}")

async def main():
    """Run performance monitor"""
    monitor = PerformanceMonitor()
    
    # Check if services are running
    print("Checking services...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8001/")
            if resp.status_code != 200:
                print("Error: Trading engine not running on port 8001")
                print("Start it with: python app_production.py")
                return
    except:
        print("Error: Cannot connect to trading engine")
        print("Start it with: python app_production.py")
        return
    
    # Start monitoring
    await monitor.monitor_loop(refresh_interval=30)
    
    # Save metrics on exit
    monitor.save_metrics()

if __name__ == "__main__":
    print("="*60)
    print("PERFORMANCE MONITORING DASHBOARD")
    print("="*60)
    print("\nThis will monitor your trading system in real-time")
    print("Updates every 30 seconds")
    print("\nPress Ctrl+C to stop\n")
    
    asyncio.run(main())