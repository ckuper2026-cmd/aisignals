#!/usr/bin/env python3
"""
Remote Performance Monitor
Check trading system performance from anywhere
"""

import httpx
import asyncio
import json
from datetime import datetime
import sys

# Change this to your Railway URL after deployment
API_URL = "https://your-app.up.railway.app"  # UPDATE THIS!

# For local testing
if len(sys.argv) > 1 and sys.argv[1] == "local":
    API_URL = "http://localhost:8000"

async def check_performance():
    """Check trading performance remotely"""
    
    print("="*60)
    print("    REMOTE TRADING MONITOR")
    print("="*60)
    print(f"API: {API_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # 1. System Status
            resp = await client.get(f"{API_URL}/")
            if resp.status_code == 200:
                data = resp.json()
                print("\n📊 SYSTEM STATUS")
                print(f"  Status: {data['status']}")
                print(f"  Market: {'OPEN' if data['market_open'] else 'CLOSED'}")
                print(f"  Uptime: {data['uptime']}")
                print(f"  Scans: {data['scan_count']}")
                print(f"  Positions: {data['positions']}")
                
                portfolio = data['portfolio']
                print(f"\n💰 PORTFOLIO")
                print(f"  Value: ${portfolio['total_value']:,.2f}")
                print(f"  Cash: ${portfolio['cash']:,.2f}")
                print(f"  Return: {portfolio['total_return_pct']:.2f}%")
                
                if data['volatility']['zone']:
                    print(f"\n🔥 VOLATILITY ZONE: {data['volatility']['zone']}")
                    print(f"  Multiplier: {data['volatility']['multiplier']}x")
            
            # 2. Performance Metrics
            resp = await client.get(f"{API_URL}/api/performance")
            if resp.status_code == 200:
                data = resp.json()
                metrics = data['overall']
                
                print(f"\n📈 PERFORMANCE")
                print(f"  Total Trades: {metrics['total_trades']}")
                print(f"  Win Rate: {metrics['win_rate']:.1f}%")
                print(f"  Realized P&L: ${metrics['realized_pnl']:.2f}")
                print(f"  Unrealized P&L: ${metrics['unrealized_pnl']:.2f}")
                print(f"  Best Trade: ${metrics['best_trade']:.2f}")
                print(f"  Worst Trade: ${metrics['worst_trade']:.2f}")
                print(f"  Max Drawdown: {metrics['max_drawdown']:.1f}%")
                
                # Zone performance
                zones = data['by_zone']
                if zones:
                    print(f"\n⏰ ZONE PERFORMANCE")
                    for zone, perf in zones.items():
                        if perf['trades'] > 0:
                            print(f"  {zone}: {perf['trades']} trades, P&L: ${perf['pnl']:.2f}")
                
                # Recent trades
                if data['trades']['last_10']:
                    print(f"\n📝 RECENT TRADES")
                    for trade in data['trades']['last_10'][-3:]:  # Last 3
                        print(f"  {trade['symbol']}: ${trade['pnl']:.2f} ({trade['exit_reason']})")
            
            # 3. Current Positions
            resp = await client.get(f"{API_URL}/api/portfolio")
            if resp.status_code == 200:
                data = resp.json()
                positions = data['positions']
                
                if positions:
                    print(f"\n📊 OPEN POSITIONS")
                    for pos in positions:
                        print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['entry_price']:.2f}")
                        print(f"    Current: ${pos['current_price']:.2f} ({pos['pnl_pct']:+.1f}%)")
                        print(f"    P&L: ${pos['pnl']:+.2f}")
            
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            print("Check that the trading system is running")

async def continuous_monitor():
    """Monitor continuously"""
    while True:
        await check_performance()
        print("\n⏱️ Next update in 60 seconds... (Ctrl+C to stop)")
        await asyncio.sleep(60)

async def get_daily_summary():
    """Get daily performance summary"""
    print("\n" + "="*60)
    print("    DAILY SUMMARY")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        try:
            # Get performance data
            resp = await client.get(f"{API_URL}/api/performance")
            if resp.status_code == 200:
                data = resp.json()
                
                # Calculate daily stats
                by_hour = data['by_hour']
                total_signals = sum(h['signals'] for h in by_hour.values())
                total_trades = sum(h['trades'] for h in by_hour.values())
                total_pnl = sum(h['pnl'] for h in by_hour.values())
                
                print(f"\n📊 TODAY'S ACTIVITY")
                print(f"  Signals: {total_signals}")
                print(f"  Trades: {total_trades}")
                print(f"  P&L: ${total_pnl:.2f}")
                
                # Hourly breakdown
                print(f"\n⏰ HOURLY BREAKDOWN")
                for hour in range(9, 17):
                    if str(hour) in by_hour:
                        h = by_hour[str(hour)]
                        if h['trades'] > 0:
                            print(f"  {hour:02d}:00 - S:{h['signals']} T:{h['trades']} P&L:${h['pnl']:.2f}")
            
            # Get all trades
            resp = await client.get(f"{API_URL}/api/trades")
            if resp.status_code == 200:
                data = resp.json()
                trades = data['trades']
                
                if trades:
                    # Today's trades
                    today = datetime.now().date().isoformat()
                    today_trades = [t for t in trades if t['timestamp'].startswith(today)]
                    
                    if today_trades:
                        wins = [t for t in today_trades if t['pnl'] > 0]
                        losses = [t for t in today_trades if t['pnl'] < 0]
                        
                        print(f"\n📈 TODAY'S TRADES")
                        print(f"  Total: {len(today_trades)}")
                        print(f"  Wins: {len(wins)}")
                        print(f"  Losses: {len(losses)}")
                        if today_trades:
                            win_rate = (len(wins) / len(today_trades)) * 100
                            print(f"  Win Rate: {win_rate:.1f}%")
                        
                        total_pnl = sum(t['pnl'] for t in today_trades)
                        print(f"  Total P&L: ${total_pnl:.2f}")
            
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    print("\n📱 REMOTE TRADING MONITOR")
    print("-"*40)
    print("1. Check once")
    print("2. Monitor continuously")
    print("3. Daily summary")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        asyncio.run(check_performance())
    elif choice == "2":
        asyncio.run(continuous_monitor())
    elif choice == "3":
        asyncio.run(get_daily_summary())
    else:
        print("Exiting...")