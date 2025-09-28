"""
Standalone Trading System v5.0
For remote monitoring via Railway - no infrastructure needed
Self-contained portfolio and performance tracking
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import yfinance as yf
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Standalone Trading System",
    version="5.0.0",
    description="Remote monitoring via Railway"
)

# CORS - allow all origins for remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CONFIGURATION
# ============================================

# Portfolio settings
INITIAL_CAPITAL = 100000
POSITION_SIZE_PCT = 0.1  # 10% per position
MAX_POSITIONS = 5

# Stock universe
STOCK_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

# Volatility zones
VOLATILITY_ZONES = {
    "opening_bell": {"start": "09:30", "end": "10:00", "multiplier": 1.8, "threshold": 0.3},
    "morning_reversal": {"start": "10:00", "end": "10:30", "multiplier": 1.3, "threshold": 0.35},
    "midday": {"start": "10:30", "end": "14:00", "multiplier": 1.1, "threshold": 0.4},
    "pre_power": {"start": "14:00", "end": "15:00", "multiplier": 1.3, "threshold": 0.35},
    "power_hour": {"start": "15:00", "end": "15:50", "multiplier": 1.5, "threshold": 0.3},
    "closing_cross": {"start": "15:50", "end": "16:00", "multiplier": 2.0, "threshold": 0.25}
}

# Risk management
STOP_LOSS_PCT = 0.02  # 2%
TAKE_PROFIT_PCT = 0.03  # 3%

# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: str
    stop_loss: float
    take_profit: float
    entry_zone: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def update_pnl(self):
        """Update P&L calculations"""
        self.pnl = (self.current_price - self.entry_price) * self.quantity
        self.pnl_pct = ((self.current_price - self.entry_price) / self.entry_price) * 100

@dataclass
class Trade:
    symbol: str
    action: str
    quantity: int
    price: float
    timestamp: str
    strategy: str
    confidence: float
    pnl: float = 0.0
    exit_reason: Optional[str] = None

@dataclass
class Signal:
    symbol: str
    action: str
    price: float
    confidence: float
    strategy: str
    stop_loss: float
    take_profit: float
    timestamp: str
    zone: Optional[str] = None

# ============================================
# PORTFOLIO MANAGER
# ============================================

class Portfolio:
    """Self-contained portfolio management"""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.all_signals: List[Signal] = []
        self.performance_history = []
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'realized_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'win_rate': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'max_drawdown': 0.0,
            'peak_value': initial_capital,
            'signals_generated': 0,
            'positions_opened': 0,
            'positions_closed': 0
        }
        
        # Track by time period
        self.hourly_performance = defaultdict(lambda: {'signals': 0, 'trades': 0, 'pnl': 0})
        self.zone_performance = defaultdict(lambda: {'signals': 0, 'trades': 0, 'pnl': 0})
        self.strategy_performance = defaultdict(lambda: {'signals': 0, 'wins': 0, 'losses': 0})
        
    def get_total_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.current_price * pos.quantity for pos in self.positions.values())
        return self.cash + positions_value
    
    def can_trade(self) -> bool:
        """Check if we can open new positions"""
        return len(self.positions) < MAX_POSITIONS and self.cash > 10000
    
    def open_position(self, signal: Signal) -> bool:
        """Open a new position"""
        try:
            # Calculate position size
            position_value = min(self.cash * POSITION_SIZE_PCT, self.cash - 1000)
            quantity = int(position_value / signal.price)
            
            if quantity < 1:
                return False
            
            cost = quantity * signal.price
            if cost > self.cash:
                return False
            
            # Create position
            position = Position(
                symbol=signal.symbol,
                quantity=quantity,
                entry_price=signal.price,
                current_price=signal.price,
                entry_time=signal.timestamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_zone=signal.zone
            )
            
            # Update portfolio
            self.cash -= cost
            self.positions[signal.symbol] = position
            self.metrics['positions_opened'] += 1
            
            # Track by hour and zone
            hour = datetime.fromisoformat(signal.timestamp).hour
            self.hourly_performance[hour]['trades'] += 1
            if signal.zone:
                self.zone_performance[signal.zone]['trades'] += 1
            
            logger.info(f"OPENED: {quantity} {signal.symbol} @ ${signal.price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def close_position(self, symbol: str, price: float, reason: str = "manual") -> Optional[Trade]:
        """Close an existing position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.current_price = price
        position.update_pnl()
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            action="SELL",
            quantity=position.quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            strategy="exit",
            confidence=1.0,
            pnl=position.pnl,
            exit_reason=reason
        )
        
        # Update portfolio
        self.cash += position.quantity * price
        self.closed_trades.append(trade)
        
        # Update metrics
        self.metrics['total_trades'] += 1
        self.metrics['positions_closed'] += 1
        self.metrics['realized_pnl'] += position.pnl
        self.metrics['total_pnl'] = self.metrics['realized_pnl']
        
        if position.pnl > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['best_trade'] = max(self.metrics['best_trade'], position.pnl)
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['worst_trade'] = min(self.metrics['worst_trade'], position.pnl)
        
        # Update win rate
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (self.metrics['winning_trades'] / self.metrics['total_trades']) * 100
        
        # Track by hour and zone
        hour = datetime.now().hour
        self.hourly_performance[hour]['pnl'] += position.pnl
        if position.entry_zone:
            self.zone_performance[position.entry_zone]['pnl'] += position.pnl
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"CLOSED: {symbol} @ ${price:.2f} - P&L: ${position.pnl:.2f} ({position.pnl_pct:.1f}%) - Reason: {reason}")
        return trade
    
    def update_positions(self, quotes: Dict[str, float]):
        """Update current prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in quotes:
                position.current_price = quotes[symbol]
                position.update_pnl()
        
        # Calculate unrealized P&L
        self.metrics['unrealized_pnl'] = sum(pos.pnl for pos in self.positions.values())
        
        # Update max drawdown
        current_value = self.get_total_value()
        if current_value > self.metrics['peak_value']:
            self.metrics['peak_value'] = current_value
        
        drawdown = ((self.metrics['peak_value'] - current_value) / self.metrics['peak_value']) * 100
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
    
    def check_exits(self) -> List[Trade]:
        """Check all positions for exit conditions"""
        exits = []
        current_zone = get_current_zone()
        
        for symbol, position in list(self.positions.items()):
            exit_reason = None
            
            # Check stop loss
            if position.current_price <= position.stop_loss:
                exit_reason = "stop_loss"
            # Check take profit
            elif position.current_price >= position.take_profit:
                exit_reason = "take_profit"
            # Zone exit - close when volatile zone ends
            elif position.entry_zone and position.entry_zone != current_zone:
                exit_reason = f"zone_end_{position.entry_zone}"
            # EOD liquidation
            elif is_near_market_close():
                exit_reason = "eod_liquidation"
            
            if exit_reason:
                trade = self.close_position(symbol, position.current_price, exit_reason)
                if trade:
                    exits.append(trade)
        
        return exits

# ============================================
# SIGNAL GENERATOR
# ============================================

class SignalGenerator:
    """Generate trading signals"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.last_signals = {}
        self.cache = {}
        
    async def generate_signals(self) -> List[Signal]:
        """Generate signals for all stocks"""
        signals = []
        mult, threshold = get_volatility_multiplier()
        current_zone = get_current_zone()
        
        for symbol in STOCK_UNIVERSE:
            try:
                # Get data (with caching)
                cache_key = f"{symbol}_{datetime.now().minute}"
                if cache_key in self.cache:
                    df = self.cache[cache_key]
                else:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period="5d", interval="5m")
                    if not df.empty:
                        self.cache[cache_key] = df
                
                if df is None or len(df) < 20:
                    continue
                
                # Calculate indicators
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['RSI'] = calculate_rsi(df['Close'])
                
                latest = df.iloc[-1]
                current_price = latest['Close']
                
                # Generate signal
                action = None
                confidence = 0
                strategy = ""
                
                if latest['RSI'] < 35:
                    action = "BUY"
                    confidence = 0.65
                    strategy = "RSI_oversold"
                elif latest['RSI'] > 65:
                    action = "SELL"
                    confidence = 0.65
                    strategy = "RSI_overbought"
                elif current_price > latest['SMA_20'] * 1.01:
                    action = "BUY"
                    confidence = 0.55
                    strategy = "trend_breakout"
                
                # Apply volatility boost
                if mult > 1.0 and confidence > 0:
                    confidence = min(0.95, confidence * mult)
                
                # Check threshold
                if action and confidence >= threshold:
                    # Set stops
                    if action == "BUY":
                        stop_loss = current_price * (1 - STOP_LOSS_PCT)
                        take_profit = current_price * (1 + TAKE_PROFIT_PCT)
                    else:
                        continue  # Skip sells for now
                    
                    signal = Signal(
                        symbol=symbol,
                        action=action,
                        price=current_price,
                        confidence=confidence,
                        strategy=strategy,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        timestamp=datetime.now().isoformat(),
                        zone=current_zone
                    )
                    
                    signals.append(signal)
                    
                    # Track signal generation
                    self.portfolio.metrics['signals_generated'] += 1
                    hour = datetime.now().hour
                    self.portfolio.hourly_performance[hour]['signals'] += 1
                    if current_zone:
                        self.portfolio.zone_performance[current_zone]['signals'] += 1
                    self.portfolio.strategy_performance[strategy]['signals'] += 1
                    
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        # Store all signals
        self.portfolio.all_signals.extend(signals)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals[:3]  # Return top 3

# ============================================
# TRADING ENGINE
# ============================================

class TradingEngine:
    """Main trading engine"""
    
    def __init__(self):
        self.portfolio = Portfolio()
        self.signal_generator = SignalGenerator(self.portfolio)
        self.is_running = False
        self.last_scan = None
        self.scan_count = 0
        self.start_time = datetime.now()
        
    async def update_quotes(self) -> Dict[str, float]:
        """Get current quotes for all positions"""
        quotes = {}
        for symbol in list(self.portfolio.positions.keys()) + STOCK_UNIVERSE:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    quotes[symbol] = hist['Close'].iloc[-1]
            except:
                pass
        return quotes
    
    async def trading_loop(self):
        """Main trading loop"""
        self.is_running = True
        
        while self.is_running:
            try:
                if not is_market_open():
                    logger.info("Market closed, waiting...")
                    await asyncio.sleep(300)
                    continue
                
                # Update quotes
                quotes = await self.update_quotes()
                self.portfolio.update_positions(quotes)
                
                # Check exits first
                exits = self.portfolio.check_exits()
                
                # Generate new signals
                if self.portfolio.can_trade():
                    signals = await self.signal_generator.generate_signals()
                    
                    # Execute best signal
                    for signal in signals:
                        if signal.symbol not in self.portfolio.positions:
                            if self.portfolio.open_position(signal):
                                break
                
                # Update scan count
                self.scan_count += 1
                self.last_scan = datetime.now()
                
                # Record performance snapshot
                self.portfolio.performance_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'total_value': self.portfolio.get_total_value(),
                    'positions': len(self.portfolio.positions),
                    'cash': self.portfolio.cash,
                    'unrealized_pnl': self.portfolio.metrics['unrealized_pnl'],
                    'realized_pnl': self.portfolio.metrics['realized_pnl']
                })
                
                # Determine scan interval based on volatility
                mult, _ = get_volatility_multiplier()
                if mult >= 1.5:
                    interval = 30  # 30 seconds during high volatility
                elif mult > 1.0:
                    interval = 60  # 60 seconds during medium volatility
                else:
                    interval = 120  # 2 minutes normally
                
                logger.info(f"Scan #{self.scan_count} complete. Next in {interval}s")
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)

# ============================================
# GLOBAL INSTANCE
# ============================================

trading_engine = TradingEngine()

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_volatility_multiplier() -> tuple:
    """Get current volatility settings"""
    try:
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        current_time = now.strftime("%H:%M")
        
        for zone, config in VOLATILITY_ZONES.items():
            if config["start"] <= current_time <= config["end"]:
                return config["multiplier"], config["threshold"]
        return 1.0, 0.45
    except:
        return 1.0, 0.45

def get_current_zone() -> Optional[str]:
    """Get current volatility zone"""
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    current_time = now.strftime("%H:%M")
    
    for zone, config in VOLATILITY_ZONES.items():
        if config["start"] <= current_time <= config["end"]:
            return zone
    return None

def is_market_open() -> bool:
    """Check if market is open"""
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    
    if now.weekday() >= 5:
        return False
    
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    
    return market_open <= now <= market_close

def is_near_market_close() -> bool:
    """Check if near market close"""
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    current_time = now.strftime("%H:%M")
    return current_time >= "15:55"

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1)
    return 100 - (100 / (1 + rs))

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """System status and overview"""
    portfolio = trading_engine.portfolio
    mult, threshold = get_volatility_multiplier()
    
    return {
        "status": "running" if trading_engine.is_running else "stopped",
        "version": "5.0.0",
        "market_open": is_market_open(),
        "uptime": str(datetime.now() - trading_engine.start_time),
        "scan_count": trading_engine.scan_count,
        "last_scan": trading_engine.last_scan.isoformat() if trading_engine.last_scan else None,
        "portfolio": {
            "total_value": portfolio.get_total_value(),
            "cash": portfolio.cash,
            "initial_capital": portfolio.initial_capital,
            "total_return_pct": ((portfolio.get_total_value() - portfolio.initial_capital) / portfolio.initial_capital) * 100
        },
        "positions": len(portfolio.positions),
        "volatility": {
            "zone": get_current_zone(),
            "multiplier": mult,
            "threshold": threshold
        }
    }

@app.get("/api/portfolio")
async def get_portfolio():
    """Detailed portfolio information"""
    portfolio = trading_engine.portfolio
    
    return {
        "summary": {
            "total_value": portfolio.get_total_value(),
            "cash": portfolio.cash,
            "initial_capital": portfolio.initial_capital,
            "total_return": portfolio.get_total_value() - portfolio.initial_capital,
            "total_return_pct": ((portfolio.get_total_value() - portfolio.initial_capital) / portfolio.initial_capital) * 100
        },
        "positions": [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "pnl": pos.pnl,
                "pnl_pct": pos.pnl_pct,
                "entry_time": pos.entry_time,
                "entry_zone": pos.entry_zone
            }
            for pos in portfolio.positions.values()
        ],
        "metrics": portfolio.metrics
    }

@app.get("/api/performance")
async def get_performance():
    """Performance metrics and analytics"""
    portfolio = trading_engine.portfolio
    
    return {
        "overall": portfolio.metrics,
        "by_hour": dict(portfolio.hourly_performance),
        "by_zone": dict(portfolio.zone_performance),
        "by_strategy": dict(portfolio.strategy_performance),
        "trades": {
            "total": len(portfolio.closed_trades),
            "last_10": [
                {
                    "symbol": t.symbol,
                    "action": t.action,
                    "price": t.price,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                    "timestamp": t.timestamp
                }
                for t in portfolio.closed_trades[-10:]
            ]
        }
    }

@app.get("/api/signals")
async def get_signals():
    """Recent signals generated"""
    portfolio = trading_engine.portfolio
    
    return {
        "total_generated": portfolio.metrics['signals_generated'],
        "recent": [
            {
                "symbol": s.symbol,
                "action": s.action,
                "price": s.price,
                "confidence": s.confidence,
                "strategy": s.strategy,
                "zone": s.zone,
                "timestamp": s.timestamp
            }
            for s in portfolio.all_signals[-20:]
        ]
    }

@app.get("/api/history")
async def get_history():
    """Performance history for charting"""
    portfolio = trading_engine.portfolio
    
    # Get last 100 snapshots
    history = portfolio.performance_history[-100:]
    
    return {
        "snapshots": history,
        "summary": {
            "start_value": portfolio.initial_capital,
            "current_value": portfolio.get_total_value(),
            "high_water_mark": portfolio.metrics['peak_value'],
            "max_drawdown": portfolio.metrics['max_drawdown']
        }
    }

@app.get("/api/trades")
async def get_trades():
    """All closed trades"""
    portfolio = trading_engine.portfolio
    
    return {
        "count": len(portfolio.closed_trades),
        "trades": [
            {
                "symbol": t.symbol,
                "action": t.action,
                "quantity": t.quantity,
                "price": t.price,
                "pnl": t.pnl,
                "strategy": t.strategy,
                "confidence": t.confidence,
                "exit_reason": t.exit_reason,
                "timestamp": t.timestamp
            }
            for t in portfolio.closed_trades
        ]
    }

@app.post("/api/start")
async def start_trading():
    """Start the trading engine"""
    if not trading_engine.is_running:
        asyncio.create_task(trading_engine.trading_loop())
        return {"message": "Trading started"}
    return {"message": "Already running"}

@app.post("/api/stop")
async def stop_trading():
    """Stop the trading engine"""
    trading_engine.is_running = False
    return {"message": "Trading stopped"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup():
    """Start trading engine on startup"""
    logger.info("="*60)
    logger.info("STANDALONE TRADING SYSTEM v5.0")
    logger.info("Remote monitoring via Railway")
    logger.info("="*60)
    
    # Auto-start trading
    asyncio.create_task(trading_engine.trading_loop())
    logger.info("Trading engine started")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)