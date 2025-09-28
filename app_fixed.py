"""
AI Trading Platform - REALISTIC WORKING VERSION
Fixed all critical issues identified in review
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yfinance as yf
import pytz
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Trading Platform - Realistic Version",
    description="Working trading platform with realistic strategies",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# REALISTIC CONFIGURATION
# ============================================

# Simplified stock universe - just the most liquid
STOCK_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT"]

# Realistic parameters
SCAN_INTERVAL = 300  # 5 minutes between scans (respect rate limits)
DATA_CACHE_TTL = 300  # Cache data for 5 minutes
MIN_CONFIDENCE = 0.65  # Only trade high confidence signals
MAX_POSITIONS = 3  # Maximum concurrent positions
POSITION_SIZE_PCT = 0.20  # 20% of portfolio per position
STOP_LOSS_PCT = 0.03  # 3% stop loss
TAKE_PROFIT_PCT = 0.06  # 6% take profit

# ============================================
# DATA CACHE TO PREVENT API OVERLOAD
# ============================================

class DataCache:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
    
    def get(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached data if fresh"""
        if symbol in self.cache:
            age = time.time() - self.last_update.get(symbol, 0)
            if age < DATA_CACHE_TTL:
                return self.cache[symbol]
        return None
    
    def set(self, symbol: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        self.cache[symbol] = data
        self.last_update[symbol] = time.time()
    
    def clear_old(self):
        """Remove stale entries"""
        current = time.time()
        stale = [s for s, t in self.last_update.items() 
                if current - t > DATA_CACHE_TTL * 2]
        for symbol in stale:
            self.cache.pop(symbol, None)
            self.last_update.pop(symbol, None)

# ============================================
# SIMPLIFIED SIGNAL
# ============================================

@dataclass
class Signal:
    symbol: str
    action: str  # BUY or SELL
    price: float
    confidence: float
    strategy: str
    stop_loss: float
    take_profit: float
    timestamp: str
    
    def to_dict(self):
        return asdict(self)

# ============================================
# REALISTIC PORTFOLIO MANAGER
# ============================================

class Portfolio:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = deque(maxlen=100)
        self.total_trades = 0
        self.winning_trades = 0
        
    def can_buy(self) -> bool:
        """Check if we can open a new position"""
        return len(self.positions) < MAX_POSITIONS and self.cash > 10000
    
    def execute_trade(self, signal: Signal) -> Dict:
        """Execute a trade with proper validation"""
        try:
            if signal.action == "BUY":
                if not self.can_buy():
                    return {"success": False, "error": "Position limit reached or insufficient funds"}
                
                if signal.symbol in self.positions:
                    return {"success": False, "error": "Already have position"}
                
                # Calculate position size
                position_value = self.cash * POSITION_SIZE_PCT
                shares = int(position_value / signal.price)
                
                if shares < 1:
                    return {"success": False, "error": "Insufficient funds for minimum position"}
                
                cost = shares * signal.price
                
                self.cash -= cost
                self.positions[signal.symbol] = {
                    "shares": shares,
                    "entry_price": signal.price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "entry_time": datetime.now().isoformat()
                }
                
                self.total_trades += 1
                
                trade = {
                    "type": "BUY",
                    "symbol": signal.symbol,
                    "shares": shares,
                    "price": signal.price,
                    "timestamp": datetime.now().isoformat()
                }
                self.trades.append(trade)
                
                logger.info(f"BOUGHT {shares} {signal.symbol} @ ${signal.price:.2f}")
                return {"success": True, "trade": trade}
                
            else:  # SELL
                if signal.symbol not in self.positions:
                    return {"success": False, "error": "No position to sell"}
                
                position = self.positions[signal.symbol]
                shares = position["shares"]
                entry_price = position["entry_price"]
                
                revenue = shares * signal.price
                self.cash += revenue
                
                # Track performance
                profit = (signal.price - entry_price) * shares
                if profit > 0:
                    self.winning_trades += 1
                
                del self.positions[signal.symbol]
                
                trade = {
                    "type": "SELL",
                    "symbol": signal.symbol,
                    "shares": shares,
                    "price": signal.price,
                    "profit": profit,
                    "profit_pct": (signal.price - entry_price) / entry_price * 100,
                    "timestamp": datetime.now().isoformat()
                }
                self.trades.append(trade)
                
                logger.info(f"SOLD {shares} {signal.symbol} @ ${signal.price:.2f} "
                          f"(P&L: ${profit:.2f})")
                return {"success": True, "trade": trade}
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def check_stops(self, current_prices: Dict) -> List[Signal]:
        """Check for stop loss and take profit conditions"""
        exit_signals = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            entry_price = position["entry_price"]
            
            # Check stop loss
            if current_price <= position["stop_loss"]:
                exit_signals.append(Signal(
                    symbol=symbol,
                    action="SELL",
                    price=current_price,
                    confidence=1.0,
                    strategy="stop_loss",
                    stop_loss=0,
                    take_profit=0,
                    timestamp=datetime.now().isoformat()
                ))
                logger.warning(f"STOP LOSS triggered for {symbol}")
                
            # Check take profit
            elif current_price >= position["take_profit"]:
                exit_signals.append(Signal(
                    symbol=symbol,
                    action="SELL",
                    price=current_price,
                    confidence=1.0,
                    strategy="take_profit",
                    stop_loss=0,
                    take_profit=0,
                    timestamp=datetime.now().isoformat()
                ))
                logger.info(f"TAKE PROFIT triggered for {symbol}")
        
        return exit_signals
    
    def get_value(self) -> float:
        """Calculate total portfolio value"""
        return self.cash  # Simplified - would need current prices for positions

# ============================================
# WORKING SIGNAL GENERATOR
# ============================================

class SignalGenerator:
    def __init__(self):
        self.portfolio = Portfolio()
        self.data_cache = DataCache()
        self.last_scan = 0
        
    def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data with caching to prevent API overload"""
        # Check cache first
        cached = self.data_cache.get(symbol)
        if cached is not None:
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            # Use 15-minute data for more reliable signals
            data = ticker.history(period="5d", interval="15m")
            
            if not data.empty and len(data) > 50:
                self.data_cache.set(symbol, data)
                return data
                
        except Exception as e:
            logger.error(f"Data fetch error for {symbol}: {e}")
        
        return None
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate WORKING technical indicators"""
        try:
            # Simple Moving Averages
            data['SMA_20'] = data['Close'].rolling(20, min_periods=1).mean()
            data['SMA_50'] = data['Close'].rolling(50, min_periods=1).mean()
            
            # RSI - Simplified calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            
            rs = gain / loss.replace(0, 1)
            rsi = 100 - (100 / (1 + rs))
            
            # Volume
            volume_avg = data['Volume'].rolling(20, min_periods=1).mean()
            volume_ratio = data['Volume'] / volume_avg.replace(0, 1)
            
            current_price = data['Close'].iloc[-1]
            sma20 = data['SMA_20'].iloc[-1]
            sma50 = data['SMA_50'].iloc[-1]
            
            return {
                'price': float(current_price),
                'sma20': float(sma20),
                'sma50': float(sma50),
                'rsi': float(rsi.iloc[-1]) if not rsi.empty else 50,
                'volume_ratio': float(volume_ratio.iloc[-1]) if not volume_ratio.empty else 1,
                'trend': 'up' if sma20 > sma50 else 'down'
            }
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return {}
    
    def generate_signal(self, symbol: str) -> Optional[Signal]:
        """Generate realistic trading signal"""
        data = self.fetch_data(symbol)
        if data is None or len(data) < 50:
            return None
        
        indicators = self.calculate_indicators(data)
        if not indicators:
            return None
        
        price = indicators['price']
        rsi = indicators['rsi']
        volume_ratio = indicators['volume_ratio']
        trend = indicators['trend']
        
        # REALISTIC SIGNAL LOGIC
        action = None
        confidence = 0.0
        strategy = ""
        
        # Only generate signals at extremes with confirmation
        
        # Strong oversold bounce
        if rsi < 25 and volume_ratio > 1.5 and trend == 'up':
            action = "BUY"
            confidence = 0.75
            strategy = "oversold_bounce"
        
        # Strong overbought reversal
        elif rsi > 75 and volume_ratio > 1.5 and trend == 'down':
            action = "SELL"
            confidence = 0.70
            strategy = "overbought_reversal"
        
        # Trend following with volume
        elif trend == 'up' and price > indicators['sma20'] * 1.01 and volume_ratio > 2.0:
            action = "BUY"
            confidence = 0.65
            strategy = "trend_breakout"
        
        # Don't trade choppy markets
        else:
            return None
        
        if confidence < MIN_CONFIDENCE:
            return None
        
        # Calculate stops
        if action == "BUY":
            stop_loss = price * (1 - STOP_LOSS_PCT)
            take_profit = price * (1 + TAKE_PROFIT_PCT)
        else:
            stop_loss = price * (1 + STOP_LOSS_PCT)
            take_profit = price * (1 - TAKE_PROFIT_PCT)
        
        return Signal(
            symbol=symbol,
            action=action,
            price=price,
            confidence=confidence,
            strategy=strategy,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            timestamp=datetime.now().isoformat()
        )
    
    def scan_market(self) -> List[Signal]:
        """Scan market with rate limiting"""
        # Rate limit scanning
        if time.time() - self.last_scan < SCAN_INTERVAL:
            return []
        
        self.last_scan = time.time()
        signals = []
        
        # Clear old cache entries
        self.data_cache.clear_old()
        
        for symbol in STOCK_UNIVERSE:
            # Skip if we already have a position
            if symbol in self.portfolio.positions:
                continue
            
            signal = self.generate_signal(symbol)
            if signal:
                signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Only return best signal if we can't buy multiple
        if not self.portfolio.can_buy():
            return []
        
        return signals[:1]  # One signal at a time

# ============================================
# GLOBAL INSTANCE
# ============================================

signal_generator = SignalGenerator()

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "platform": "AI Trading Platform - Realistic Version",
        "status": "operational",
        "portfolio_value": signal_generator.portfolio.get_value(),
        "positions": len(signal_generator.portfolio.positions),
        "max_positions": MAX_POSITIONS,
        "scan_interval": f"{SCAN_INTERVAL}s"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/signals")
async def get_signals():
    """Get current signals"""
    signals = signal_generator.scan_market()
    
    return {
        "signals": [s.to_dict() for s in signals],
        "can_trade": signal_generator.portfolio.can_buy(),
        "positions_held": len(signal_generator.portfolio.positions)
    }

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio status"""
    portfolio = signal_generator.portfolio
    
    win_rate = 0
    if portfolio.total_trades > 0:
        win_rate = portfolio.winning_trades / portfolio.total_trades * 100
    
    return {
        "cash": portfolio.cash,
        "positions": portfolio.positions,
        "total_value": portfolio.get_value(),
        "total_trades": portfolio.total_trades,
        "win_rate": win_rate,
        "recent_trades": list(portfolio.trades)[-10:]
    }

@app.post("/api/execute")
async def execute_trade(signal_json: dict):
    """Execute a trade"""
    try:
        signal = Signal(**signal_json)
        result = signal_generator.portfolio.execute_trade(signal)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/check-stops")
async def check_stops():
    """Manually check stop losses"""
    # Get current prices
    current_prices = {}
    for symbol in signal_generator.portfolio.positions.keys():
        data = signal_generator.fetch_data(symbol)
        if data is not None and not data.empty:
            current_prices[symbol] = data['Close'].iloc[-1]
    
    # Check stops
    exit_signals = signal_generator.portfolio.check_stops(current_prices)
    
    # Execute exits
    results = []
    for signal in exit_signals:
        result = signal_generator.portfolio.execute_trade(signal)
        results.append(result)
    
    return {"exits_triggered": len(results), "results": results}

# ============================================
# BACKGROUND TASK
# ============================================

async def trading_loop():
    """Simple trading loop that actually works"""
    while True:
        try:
            # Only scan during market hours
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et)
            
            # Check if market is open (9:30 AM - 4:00 PM ET, weekdays)
            if now.weekday() < 5:  # Monday = 0, Friday = 4
                market_open = now.replace(hour=9, minute=30, second=0)
                market_close = now.replace(hour=16, minute=0, second=0)
                
                if market_open <= now <= market_close:
                    # Get signals
                    signals = signal_generator.scan_market()
                    
                    # Execute best signal if any
                    if signals and signal_generator.portfolio.can_buy():
                        best_signal = signals[0]
                        if best_signal.confidence >= MIN_CONFIDENCE:
                            result = signal_generator.portfolio.execute_trade(best_signal)
                            if result.get("success"):
                                logger.info(f"Auto-executed: {best_signal.symbol} "
                                          f"{best_signal.action} @ ${best_signal.price}")
                    
                    # Check stops
                    current_prices = {}
                    for symbol in signal_generator.portfolio.positions.keys():
                        data = signal_generator.fetch_data(symbol)
                        if data is not None and not data.empty:
                            current_prices[symbol] = data['Close'].iloc[-1]
                    
                    exit_signals = signal_generator.portfolio.check_stops(current_prices)
                    for signal in exit_signals:
                        signal_generator.portfolio.execute_trade(signal)
            
            # Wait before next iteration
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup():
    """Start background tasks"""
    asyncio.create_task(trading_loop())
    logger.info("Trading platform started - REALISTIC VERSION")
    logger.info(f"Scanning {len(STOCK_UNIVERSE)} stocks every {SCAN_INTERVAL}s")
    logger.info(f"Max positions: {MAX_POSITIONS}, Position size: {POSITION_SIZE_PCT*100}%")
    logger.info(f"Stop loss: {STOP_LOSS_PCT*100}%, Take profit: {TAKE_PROFIT_PCT*100}%")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)