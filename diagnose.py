#!/usr/bin/env python3
"""
Trading System Diagnostics
Quick health check and troubleshooting
"""

import httpx
import asyncio
import json
import sys
from datetime import datetime
import pytz

async def run_diagnostics():
    """Run complete system diagnostics"""
    
    print("="*60)
    print("       TRADING SYSTEM DIAGNOSTICS")
    print("="*60)
    print(f"\nTimestamp: {datetime.now()}")
    print("-"*60)
    
    results = {
        'infrastructure': False,
        'trading_engine': False,
        'data_feeds': False,
        'rate_limits_ok': False,
        'cache_working': False,
        'market_hours': False,
        'volatility_zone': None,
        'errors': []
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        
        # 1. Check Infrastructure
        print("\n1. INFRASTRUCTURE CHECK")
        print("-"*30)
        try:
            resp = await client.get("http://localhost:8000/")
            if resp.status_code == 200:
                print("✅ Infrastructure running on port 8000")
                results['infrastructure'] = True
                
                # Try health endpoint
                resp = await client.get("http://localhost:8000/health")
                if resp.status_code == 200:
                    print("✅ Infrastructure health check passed")
            else:
                print(f"❌ Infrastructure returned status {resp.status_code}")
        except Exception as e:
            print(f"❌ Infrastructure not responding: {e}")
            print("   Fix: python platform_infrastructure.py")
            results['errors'].append("Infrastructure not running")
        
        # 2. Check Trading Engine
        print("\n2. TRADING ENGINE CHECK")
        print("-"*30)
        try:
            resp = await client.get("http://localhost:8001/")
            if resp.status_code == 200:
                data = resp.json()
                print(f"✅ Trading engine running (v{data['version']})")
                results['trading_engine'] = True
                
                # Check cache
                cache_size = data.get('cache_size', 0)
                print(f"✅ Cache active: {cache_size} entries")
                results['cache_working'] = cache_size > 0
                
                # Check volatility
                vol = data.get('volatility', {})
                if vol.get('active'):
                    print(f"🔥 Volatility zone ACTIVE: {vol.get('multiplier')}x")
                    results['volatility_zone'] = 'active'
                else:
                    print(f"📊 Normal trading mode")
                    results['volatility_zone'] = 'normal'
                    
            else:
                print(f"❌ Trading engine returned status {resp.status_code}")
        except Exception as e:
            print(f"❌ Trading engine not responding: {e}")
            print("   Fix: python app_production.py")
            results['errors'].append("Trading engine not running")
        
        # 3. Check Data Feeds
        print("\n3. DATA FEED CHECK")
        print("-"*30)
        if results['trading_engine']:
            try:
                resp = await client.get("http://localhost:8001/api/quotes/SPY")
                if resp.status_code == 200:
                    data = resp.json()
                    if 'quotes' in data and 'SPY' in data['quotes']:
                        price = data['quotes']['SPY'].get('price', 0)
                        if price > 0:
                            print(f"✅ Data feeds working (SPY: ${price:.2f})")
                            results['data_feeds'] = True
                        else:
                            print("⚠️ Data feeds returning zero prices")
                    else:
                        print("❌ No quote data returned")
                else:
                    print(f"❌ Quotes endpoint returned {resp.status_code}")
            except Exception as e:
                print(f"❌ Cannot fetch quotes: {e}")
                results['errors'].append("Data feeds not working")
        else:
            print("⏭️ Skipped (trading engine not running)")
        
        # 4. Check Rate Limits
        print("\n4. RATE LIMIT CHECK")
        print("-"*30)
        if results['trading_engine']:
            try:
                resp = await client.get("http://localhost:8001/api/rate-limits")
                if resp.status_code == 200:
                    data = resp.json()
                    
                    yf = data.get('yfinance', {})
                    yf_used = yf.get('used', 0)
                    yf_limit = yf.get('limit', 2000)
                    yf_pct = (yf_used / yf_limit * 100) if yf_limit > 0 else 0
                    
                    if yf_pct < 80:
                        print(f"✅ YFinance: {yf_used}/{yf_limit} ({yf_pct:.1f}%)")
                        results['rate_limits_ok'] = True
                    else:
                        print(f"⚠️ YFinance: {yf_used}/{yf_limit} ({yf_pct:.1f}% - HIGH)")
                    
                    pg = data.get('polygon', {})
                    pg_used = pg.get('used', 0)
                    pg_limit = pg.get('limit', 5)
                    
                    if pg_used > 0:
                        print(f"📊 Polygon: {pg_used}/{pg_limit} (backup API in use)")
                    else:
                        print(f"✅ Polygon: Not needed (YFinance working)")
                    
                    # Check wait times
                    yf_wait = yf.get('wait_time', 0)
                    if yf_wait > 0:
                        print(f"⏱️ YFinance cooldown: {yf_wait}s")
                        results['errors'].append(f"Rate limit cooldown: {yf_wait}s")
                        
                else:
                    print(f"❌ Rate limits endpoint returned {resp.status_code}")
            except Exception as e:
                print(f"❌ Cannot check rate limits: {e}")
        else:
            print("⏭️ Skipped (trading engine not running)")
        
        # 5. Check Market Hours
        print("\n5. MARKET HOURS CHECK")
        print("-"*30)
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        current_time = now_et.strftime("%H:%M")
        day_of_week = now_et.weekday()
        
        if day_of_week >= 5:
            print(f"🚫 Weekend - Market closed")
            results['market_hours'] = False
        elif "09:30" <= current_time <= "16:00":
            print(f"✅ Market OPEN (ET: {current_time})")
            results['market_hours'] = True
            
            # Check which volatility zone
            zones = {
                ("09:30", "10:00"): "Opening Bell (1.8x)",
                ("10:00", "10:30"): "Morning Reversal (1.3x)",
                ("15:00", "15:50"): "Power Hour (1.5x)",
                ("15:50", "16:00"): "Closing Cross (2.0x)"
            }
            
            for (start, end), zone_name in zones.items():
                if start <= current_time <= end:
                    print(f"🔥 Active Zone: {zone_name}")
                    break
        else:
            print(f"🌙 Market closed (ET: {current_time})")
            results['market_hours'] = False
        
        # 6. Check Signals
        print("\n6. SIGNAL GENERATION CHECK")
        print("-"*30)
        if results['trading_engine'] and results['data_feeds']:
            try:
                resp = await client.get("http://localhost:8001/api/signals")
                if resp.status_code == 200:
                    data = resp.json()
                    signal_count = len(data.get('signals', []))
                    
                    if signal_count > 0:
                        print(f"✅ Generating signals: {signal_count} active")
                        
                        # Show first signal
                        first_signal = data['signals'][0]
                        print(f"   Example: {first_signal['action']} {first_signal['symbol']} @ ${first_signal['price']:.2f}")
                        print(f"   Confidence: {first_signal['confidence']:.1%}")
                    else:
                        if results['market_hours']:
                            print("⚠️ No signals generated (check thresholds)")
                            results['errors'].append("No signals despite market open")
                        else:
                            print("📊 No signals (normal - market closed)")
                else:
                    print(f"❌ Signals endpoint returned {resp.status_code}")
            except Exception as e:
                print(f"❌ Cannot fetch signals: {e}")
        else:
            print("⏭️ Skipped (prerequisites not met)")
    
    # 7. System Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    # Calculate health score
    health_score = sum([
        results['infrastructure'] * 20,
        results['trading_engine'] * 20,
        results['data_feeds'] * 20,
        results['rate_limits_ok'] * 20,
        results['cache_working'] * 20
    ])
    
    print(f"\n🏥 SYSTEM HEALTH: {health_score}%")
    
    if health_score == 100:
        print("✅ All systems operational")
    elif health_score >= 80:
        print("⚠️ Minor issues detected")
    elif health_score >= 60:
        print("⚠️ Some components need attention")
    else:
        print("❌ Critical issues - immediate attention required")
    
    # Show errors
    if results['errors']:
        print("\n🚨 ISSUES TO FIX:")
        print("-"*30)
        for i, error in enumerate(results['errors'], 1):
            print(f"{i}. {error}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-"*30)
    
    if not results['infrastructure']:
        print("1. Start infrastructure: python platform_infrastructure.py")
    
    if not results['trading_engine']:
        print("2. Start trading engine: python app_production.py")
    
    if not results['data_feeds']:
        print("3. Check internet connection and YFinance availability")
    
    if not results['rate_limits_ok']:
        print("4. Wait for rate limits to reset or increase cache TTL")
    
    if not results['cache_working'] and results['trading_engine']:
        print("5. Cache is empty - will populate after first queries")
    
    if results['market_hours'] and len(results['errors']) == 0:
        print("✅ System ready for trading!")
    
    print("\n" + "="*60)
    
    return health_score

if __name__ == "__main__":
    print("\nRunning system diagnostics...")
    print("This will check all components and identify issues\n")
    
    score = asyncio.run(run_diagnostics())
    
    # Exit code based on health
    if score == 100:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Issues found