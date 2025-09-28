# CENTRALIZED PORTFOLIO VERSION - FIXES ALL ISSUES
# Key changes:
# 1. Single centralized portfolio (no user accounts)
# 2. Only generate SELL signals for stocks we own
# 3. Track all positions properly
# 4. Lower confidence threshold for testing
# 5. Better logging to see what's happening

import os
import asyncio
import json
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pydantic import BaseModel

# FastAPI and web framework imports
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# External libraries
import pandas as pd
import numpy as np
import yfinance as yf
from cryptography.fernet import Fernet
import pytz
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Photon AI Trading Platform",
    description="AI-powered trading signals with automated execution",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CENTRALIZED PORTFOLIO SYSTEM
# ============================================
PORTFOLIO = {
    "cash": 100000.00,  # Starting with $100k
    "positions": {},    # {symbol: {"qty": 0, "avg_price": 0}}
    "trades": [],       # History of all trades
    "total_value": 100000.00,
    "daily_trades": 0,
    "last_reset": datetime.now().date()
}

# Signal dataclass
@dataclass
class Signal:
    symbol: str
    action: str
    price: float
    confidence: float
    risk_score: float
    strategy: str
    explanation: str
    rsi: float
    volume_ratio: float
    momentum: float
    timestamp: str
    potential_return: float
    stop_loss: float
    take_profit: float
    
    def to_dict(self):
        return asdict(self)

# Global state
active_websockets = set()
current_signals = []
executed_signals = set()  # Track which signals we've executed

# Configuration
MAX_POSITION_SIZE = 10000  # Max $10k per position
MIN_CONFIDENCE = 0.65  # Lowered for more trades
MAX_DAILY_TRADES = 20
STOCK_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "SPY", "QQQ"]

# ============================================
# PERMANENT TRADE LOGGING
# ============================================

TRADES_LOG_FILE = "photon_trades_log.json"
SIGNALS_LOG_FILE = "photon_signals_log.json"

def save_trade_to_file(trade: Dict):
    """Save trade to permanent JSON file"""
    try:
        # Load existing trades
        if os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE, 'r') as f:
                trades = json.load(f)
        else:
            trades = []
        
        # Add new trade
        trades.append(trade)
        
        # Save back
        with open(TRADES_LOG_FILE, 'w') as f:
            json.dump(trades, f, indent=2)
        
        logger.info(f"📝 Trade logged to {TRADES_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to save trade log: {e}")

def save_signal_to_file(signal: Signal, executed: bool = False):
    """Save signal to permanent JSON file"""
    try:
        # Load existing signals
        if os.path.exists(SIGNALS_LOG_FILE):
            with open(SIGNALS_LOG_FILE, 'r') as f:
                signals = json.load(f)
        else:
            signals = []
        
        # Add signal with execution status
        signal_data = signal.to_dict()
        signal_data['executed'] = executed
        signal_data['logged_at'] = datetime.now().isoformat()
        signals.append(signal_data)
        
        # Save back
        with open(SIGNALS_LOG_FILE, 'w') as f:
            json.dump(signals, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to save signal log: {e}")

def load_trade_history() -> List[Dict]:
    """Load all historical trades from file"""
    try:
        if os.path.exists(TRADES_LOG_FILE):
            with open(TRADES_LOG_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return []

# ============================================
# PORTFOLIO MANAGEMENT
# ============================================

def get_portfolio_value() -> float:
    """Calculate total portfolio value"""
    positions_value = 0
    for symbol, position in PORTFOLIO["positions"].items():
        if position["qty"] > 0:
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.info.get('currentPrice', position["avg_price"])
                positions_value += position["qty"] * current_price
            except:
                positions_value += position["qty"] * position["avg_price"]
    
    PORTFOLIO["total_value"] = PORTFOLIO["cash"] + positions_value
    return PORTFOLIO["total_value"]

def can_buy(symbol: str, price: float, quantity: int) -> bool:
    """Check if we can afford to buy"""
    cost = price * quantity
    return PORTFOLIO["cash"] >= cost and cost <= MAX_POSITION_SIZE

def can_sell(symbol: str, quantity: int) -> bool:
    """Check if we have enough shares to sell"""
    if symbol not in PORTFOLIO["positions"]:
        return False
    return PORTFOLIO["positions"][symbol]["qty"] >= quantity

def execute_portfolio_trade(symbol: str, action: str, price: float, quantity: int) -> Dict:
    """Execute a trade on the centralized portfolio"""
    
    # Reset daily trade counter if new day
    if datetime.now().date() > PORTFOLIO["last_reset"]:
        PORTFOLIO["daily_trades"] = 0
        PORTFOLIO["last_reset"] = datetime.now().date()
    
    # Check daily trade limit
    if PORTFOLIO["daily_trades"] >= MAX_DAILY_TRADES:
        logger.warning(f"Daily trade limit reached ({MAX_DAILY_TRADES})")
        return {"success": False, "error": "Daily trade limit reached"}
    
    if action == "BUY":
        if not can_buy(symbol, price, quantity):
            return {"success": False, "error": "Insufficient funds or position too large"}
        
        # Execute buy
        cost = price * quantity
        PORTFOLIO["cash"] -= cost
        
        if symbol not in PORTFOLIO["positions"]:
            PORTFOLIO["positions"][symbol] = {"qty": 0, "avg_price": 0}
        
        position = PORTFOLIO["positions"][symbol]
        total_cost = (position["qty"] * position["avg_price"]) + cost
        position["qty"] += quantity
        position["avg_price"] = total_cost / position["qty"] if position["qty"] > 0 else price
        
        trade_type = "BUY"
        
    else:  # SELL
        if not can_sell(symbol, quantity):
            return {"success": False, "error": f"Don't own {quantity} shares of {symbol}"}
        
        # Execute sell
        revenue = price * quantity
        PORTFOLIO["cash"] += revenue
        PORTFOLIO["positions"][symbol]["qty"] -= quantity
        
        if PORTFOLIO["positions"][symbol]["qty"] == 0:
            del PORTFOLIO["positions"][symbol]
        
        trade_type = "SELL"
    
    # Record trade
    trade = {
        "id": f"trade_{secrets.token_urlsafe(8)}",
        "symbol": symbol,
        "action": trade_type,
        "quantity": quantity,
        "price": price,
        "total": price * quantity,
        "timestamp": datetime.now().isoformat(),
        "portfolio_value": get_portfolio_value()
    }
    
    PORTFOLIO["trades"].append(trade)
    PORTFOLIO["daily_trades"] += 1
    
    # PERMANENT LOGGING
    save_trade_to_file(trade)
    
    logger.info(f"✅ EXECUTED: {trade_type} {quantity} {symbol} @ ${price:.2f}")
    logger.info(f"Portfolio: Cash=${PORTFOLIO['cash']:.2f}, Total=${get_portfolio_value():.2f}")
    
    return {"success": True, "trade": trade}

# ============================================
# MARKET HOURS AND DATA VALIDATION
# ============================================

def is_market_open() -> bool:
    """Check if US stock market is currently open"""
    now = datetime.now()
    et = pytz.timezone('US/Eastern')
    et_now = datetime.now(et)
    
    if et_now.weekday() >= 5:  # Weekend
        return False
    
    market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_open = market_open <= et_now <= market_close
    
    if not is_open:
        logger.info(f"Market closed. ET time: {et_now.strftime('%H:%M')}")
    
    return is_open

def validate_signal(signal: Signal) -> bool:
    """Validate signal before execution"""
    if signal.price <= 0:
        return False
    if signal.confidence < 0 or signal.confidence > 1:
        return False
    if signal.stop_loss <= 0:
        return False
    return True

# ============================================
# SIGNAL GENERATION
# ============================================

def generate_signal(symbol: str = None) -> Optional[Signal]:
    """Generate trading signal with ownership check"""
    if not symbol:
        symbol = np.random.choice(STOCK_UNIVERSE)
    
    # Get market data with retries
    hist = None
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1h")
            if not hist.empty:
                break
        except Exception as e:
            logger.warning(f"Yahoo attempt {attempt+1} failed for {symbol}: {e}")
            if attempt < 2:
                time.sleep(2)
    
    if hist is None or hist.empty:
        logger.warning(f"No market data for {symbol}")
        return None
    
    # Get current price
    current_price = hist['Close'].iloc[-1]
    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
    
    # Calculate indicators
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
    current_volume = hist['Volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    momentum = (current_price - prev_close) / prev_close if prev_close > 0 else 0
    
    # ATR for stop loss
    high_low = hist['High'] - hist['Low']
    high_close = np.abs(hist['High'] - hist['Close'].shift())
    low_close = np.abs(hist['Low'] - hist['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=14).mean().iloc[-1]
    
    # Generate signal
    action = "HOLD"
    confidence = 0.0
    strategy = "no_signal"
profit_pct = 0  # Initialize to fix UnboundLocalError
    
    # CHECK OWNERSHIP FOR SELL SIGNALS
    owns_stock = symbol in PORTFOLIO["positions"] and PORTFOLIO["positions"][symbol]["qty"] > 0
    
    # BUY signals (no ownership required)
    if rsi < 30 and momentum < -0.01:
        action = "BUY"
        confidence = 0.7 + (30 - rsi) / 100
        strategy = "oversold_reversal"
    elif volume_ratio > 1.5 and momentum > 0.005:
        action = "BUY"
        confidence = min(0.65 + volume_ratio / 10, 0.85)
        strategy = "volume_breakout"
    # SELL signals (ONLY if we own the stock)
    elif owns_stock:
        if rsi > 70 and momentum > 0.01:
            action = "SELL"
            confidence = 0.7 + (rsi - 70) / 100
            strategy = "overbought_reversal"
        elif volume_ratio > 1.5 and momentum < -0.005:
            action = "SELL"
            confidence = min(0.65 + volume_ratio / 10, 0.85)
            strategy = "volume_breakdown"
        # Take profit check
        elif PORTFOLIO["positions"][symbol]["avg_price"] > 0:
            profit_pct = (current_price - PORTFOLIO["positions"][symbol]["avg_price"]) / PORTFOLIO["positions"][symbol]["avg_price"]
            if profit_pct > 0.05:  # 5% profit
                action = "SELL"
                confidence = 0.75
                strategy = "take_profit"
    else:
        return None  # No signal
    
    # Calculate stop loss and take profit
    if action == "BUY":
        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + (atr * 2.5)
        potential_return = ((take_profit - current_price) / current_price) * 100
    else:
        stop_loss = current_price + (atr * 1.5)
        take_profit = current_price - (atr * 2.5)
        potential_return = ((current_price - take_profit) / current_price) * 100
    
    volatility = atr / current_price if current_price > 0 else 0
    risk_score = min(volatility * 10, 1.0)
    
    explanations = {
        "oversold_reversal": f"RSI at {rsi:.1f} - oversold bounce likely",
        "overbought_reversal": f"RSI at {rsi:.1f} - overbought pullback likely",
        "volume_breakout": f"Volume {volume_ratio:.1f}x average - breakout",
        "volume_breakdown": f"Volume {volume_ratio:.1f}x average - breakdown",
        "take_profit": f"Position up {profit_pct*100:.1f}% - taking profit",
        "no_signal": "No clear signal"
    }
    
    signal = Signal(
        symbol=symbol,
        action=action,
        price=round(current_price, 2),
        confidence=round(confidence, 3),
        risk_score=round(risk_score, 3),
        strategy=strategy,
        explanation=explanations.get(strategy, "Technical signal"),
        rsi=round(rsi, 1),
        volume_ratio=round(volume_ratio, 2),
        momentum=round(momentum * 100, 2),
        timestamp=datetime.now().isoformat(),
        potential_return=round(abs(potential_return), 2),
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2)
    )
    
    if not validate_signal(signal):
        return None
    
    logger.info(f"📊 SIGNAL: {action} {symbol} @ ${current_price:.2f} (Confidence: {confidence:.2f}, Owns: {owns_stock})")
    
    return signal

async def scan_markets() -> List[Signal]:
    """Scan for trading opportunities"""
    if not is_market_open():
        logger.warning("Market is closed")
        return []
    
    signals = []
    
    # First scan owned positions for sell opportunities
    for symbol in PORTFOLIO["positions"]:
        if PORTFOLIO["positions"][symbol]["qty"] > 0:
            signal = generate_signal(symbol)
            if signal and signal.action == "SELL":
                signals.append(signal)
    
    # Then scan universe for buy opportunities
    for symbol in STOCK_UNIVERSE:
        if symbol not in PORTFOLIO["positions"] or PORTFOLIO["positions"][symbol]["qty"] == 0:
            signal = generate_signal(symbol)
            if signal and signal.action == "BUY":
                signals.append(signal)
    
    signals.sort(key=lambda x: x.confidence, reverse=True)
    # NO LIMIT - return all signals for testing
    
    logger.info(f"Generated {len(signals)} signals")
    return signals

# ============================================
# AUTO-TRADING ENGINE
# ============================================

async def process_auto_trades(signals: List[Signal]):
    """Process signals for auto-trading"""
    for signal in signals:
        # Log all signals
        save_signal_to_file(signal, executed=False)
        
        # Check if already executed
        signal_key = f"{signal.symbol}_{signal.timestamp}"
        if signal_key in executed_signals:
            continue
        
        # Check confidence threshold
        if signal.confidence < MIN_CONFIDENCE:
            logger.info(f"Signal below threshold: {signal.symbol} {signal.confidence:.2f} < {MIN_CONFIDENCE}")
            continue
        
        # Calculate position size
        if signal.action == "BUY":
            # Use 10% of available cash or MAX_POSITION_SIZE, whichever is less
            position_size = min(PORTFOLIO["cash"] * 0.1, MAX_POSITION_SIZE)
            quantity = int(position_size / signal.price)
            
            if quantity < 1:
                continue
        else:  # SELL
            # Sell 50% of position
            if signal.symbol in PORTFOLIO["positions"]:
                quantity = max(1, PORTFOLIO["positions"][signal.symbol]["qty"] // 2)
            else:
                continue
        
        # Execute trade
        result = execute_portfolio_trade(signal.symbol, signal.action, signal.price, quantity)
        
        if result["success"]:
            executed_signals.add(signal_key)
            # Mark signal as executed in log
            save_signal_to_file(signal, executed=True)
            
            # Broadcast to websockets
            await broadcast_trade(result["trade"], signal)

async def broadcast_trade(trade: Dict, signal: Signal):
    """Broadcast trade to all websockets"""
    message = {
        "type": "trade_executed",
        "trade": trade,
        "signal": signal.to_dict(),
        "portfolio": {
            "cash": PORTFOLIO["cash"],
            "total_value": get_portfolio_value(),
            "positions": len(PORTFOLIO["positions"]),
            "daily_trades": PORTFOLIO["daily_trades"]
        }
    }
    
    disconnected = set()
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except:
            disconnected.add(ws)
    
    active_websockets.difference_update(disconnected)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "platform": "Photon AI Trading - Centralized Portfolio",
        "portfolio_value": get_portfolio_value(),
        "cash": PORTFOLIO["cash"],
        "positions": len(PORTFOLIO["positions"]),
        "daily_trades": PORTFOLIO["daily_trades"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "photon"}

@app.get("/api/portfolio")
async def get_portfolio():
    """Get complete portfolio status"""
    positions_detail = []
    for symbol, position in PORTFOLIO["positions"].items():
        if position["qty"] > 0:
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.info.get('currentPrice', position["avg_price"])
            except:
                current_price = position["avg_price"]
            
            market_value = position["qty"] * current_price
            unrealized_pl = market_value - (position["qty"] * position["avg_price"])
            unrealized_plpc = (unrealized_pl / (position["qty"] * position["avg_price"]) * 100) if position["avg_price"] > 0 else 0
            
            positions_detail.append({
                "symbol": symbol,
                "qty": position["qty"],
                "avg_price": round(position["avg_price"], 2),
                "current_price": round(current_price, 2),
                "market_value": round(market_value, 2),
                "unrealized_pl": round(unrealized_pl, 2),
                "unrealized_plpc": round(unrealized_plpc, 2)
            })
    
    return {
        "cash": round(PORTFOLIO["cash"], 2),
        "positions": positions_detail,
        "total_value": round(get_portfolio_value(), 2),
        "positions_value": round(get_portfolio_value() - PORTFOLIO["cash"], 2),
        "daily_trades": PORTFOLIO["daily_trades"],
        "max_daily_trades": MAX_DAILY_TRADES,
        "total_trades": len(PORTFOLIO["trades"])
    }

@app.get("/api/trades")
async def get_trades(limit: int = 20):
    """Get recent trades"""
    return {
        "trades": PORTFOLIO["trades"][-limit:],
        "total": len(PORTFOLIO["trades"])
    }

@app.get("/api/trade-log")
async def get_trade_log():
    """Get complete trade history from permanent log"""
    trades = load_trade_history()
    return {
        "trades": trades,
        "total": len(trades),
        "log_file": TRADES_LOG_FILE
    }

@app.get("/api/signal-log") 
async def get_signal_log(limit: int = 100):
    """Get signal history from permanent log"""
    try:
        if os.path.exists(SIGNALS_LOG_FILE):
            with open(SIGNALS_LOG_FILE, 'r') as f:
                signals = json.load(f)
                return {
                    "signals": signals[-limit:],
                    "total": len(signals),
                    "executed": sum(1 for s in signals if s.get('executed')),
                    "log_file": SIGNALS_LOG_FILE
                }
    except:
        pass
    return {"signals": [], "total": 0, "executed": 0}

@app.get("/api/signals")
async def get_signals():
    """Get current signals"""
    global current_signals
    
    if not current_signals:
        current_signals = await scan_markets()
    
    return {
        "signals": [s.to_dict() for s in current_signals],
        "count": len(current_signals),
        "market_open": is_market_open(),
        "portfolio_value": get_portfolio_value()
    }

@app.get("/api/platform-status")
async def platform_status():
    """Complete platform status"""
    et = pytz.timezone('US/Eastern')
    et_now = datetime.now(et)
    
    # Test Yahoo Finance
    data_ok = False
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1d")
        data_ok = not hist.empty
    except:
        data_ok = False
    
    return {
        "platform": "Photon Trading - Centralized",
        "mode": "PAPER_TRADING",
        "timestamp": datetime.now().isoformat(),
        "market": {
            "is_open": is_market_open(),
            "current_time_et": et_now.strftime("%H:%M:%S ET"),
            "trading_hours": "9:30 AM - 4:00 PM ET"
        },
        "data_source": {
            "provider": "Yahoo Finance",
            "status": "OK" if data_ok else "ERROR",
            "real_time": data_ok
        },
        "portfolio": {
            "value": get_portfolio_value(),
            "cash": PORTFOLIO["cash"],
            "positions": len(PORTFOLIO["positions"]),
            "daily_trades": PORTFOLIO["daily_trades"],
            "max_daily_trades": MAX_DAILY_TRADES
        },
        "signals": {
            "active": len(current_signals),
            "min_confidence": MIN_CONFIDENCE
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_websockets.add(websocket)
    logger.info(f"WebSocket connected. Total: {len(active_websockets)}")
    
    try:
        # Send initial portfolio
        await websocket.send_json({
            "type": "connection",
            "portfolio": await get_portfolio(),
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep alive
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({
                "type": "ping",
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)

# ============================================
# BACKGROUND TASKS
# ============================================

async def signal_generator_loop():
    """Generate signals and execute trades"""
    while True:
        try:
            if not is_market_open():
                await asyncio.sleep(300)  # Wait 5 minutes
                continue
            
            # Generate signals
            global current_signals
            current_signals = await scan_markets()
            
            # Process for auto-trading
            await process_auto_trades(current_signals)
            
            # Broadcast signals
            if active_websockets:
                message = {
                    "type": "signals_update",
                    "signals": [s.to_dict() for s in current_signals],
                    "portfolio": await get_portfolio(),
                    "timestamp": datetime.now().isoformat()
                }
                
                disconnected = set()
                for ws in active_websockets:
                    try:
                        await ws.send_json(message)
                    except:
                        disconnected.add(ws)
                
                active_websockets.difference_update(disconnected)
            
            # Wait before next scan
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Signal generator error: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup():
    """Start background tasks"""
    asyncio.create_task(signal_generator_loop())
    
    logger.info("="*50)
    logger.info("PHOTON TRADING - CENTRALIZED PORTFOLIO")
    logger.info(f"Starting Balance: ${PORTFOLIO['cash']:,.2f}")
    logger.info(f"Max Position Size: ${MAX_POSITION_SIZE:,.2f}")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE:.0%}")
    logger.info(f"Max Daily Trades: {MAX_DAILY_TRADES}")
    logger.info("="*50)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")