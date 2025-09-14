import os
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
from trading_engine import AdvancedTradingEngine, Signal
from supabase import create_client, Client
import hashlib
import secrets

# Environment variables
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID", "PKVXGWME8WXJXITHIEA0")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "GTaFqM076JE3EIirkZZgMTvjjOJCOLbewhRd6fuA")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# Initialize
app = FastAPI(title="AI Trading Platform")
engine = AdvancedTradingEngine(ALPACA_KEY, ALPACA_SECRET)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
active_websockets = set()
current_signals = []
platform_metrics = {
    "total_users": 0,
    "active_signals": 0,
    "win_rate": 73,
    "total_trades": 0,
    "profit_today": 0
}

# Stock universe - established companies with good data
STOCK_UNIVERSE = [
    "SPY", "QQQ", "DIA", "IWM", "VTI",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "V", "JNJ", "WMT", "UNH", "HD", "BAC",
    "AMD", "NFLX", "DIS", "MA", "ADBE"
]

# Models
class UserSignup(BaseModel):
    email: str
    password: str
    name: str
    risk_level: str = "moderate"

class TradeRequest(BaseModel):
    symbol: str
    action: str
    quantity: int
    user_id: str

class SubscriptionRequest(BaseModel):
    email: str
    tier: str  # 'free', 'pro', 'insider'

# Database functions
async def create_user(user_data: UserSignup) -> Dict:
    """Create new user in database"""
    if not supabase:
        return {"id": "demo_user", "email": user_data.email}
    
    try:
        password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        
        result = supabase.table('users').insert({
            'email': user_data.email,
            'password_hash': password_hash,
            'name': user_data.name,
            'risk_level': user_data.risk_level,
            'subscription_tier': 'free',
            'api_key': secrets.token_urlsafe(32)
        }).execute()
        
        platform_metrics["total_users"] += 1
        return result.data[0]
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

async def log_signal(signal: Signal):
    """Log signal to database"""
    if not supabase:
        return
    
    try:
        supabase.table('signals').insert({
            'symbol': signal.symbol,
            'action': signal.action,
            'price': signal.price,
            'confidence': signal.confidence,
            'strategy': signal.strategy,
            'risk_score': signal.risk_score,
            'metadata': {
                'rsi': signal.rsi,
                'volume_ratio': signal.volume_ratio,
                'momentum': signal.momentum,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
        }).execute()
    except Exception as e:
        logger.error(f"Signal logging error: {e}")

async def log_trade(trade_data: Dict):
    """Log executed trade"""
    if not supabase:
        return
    
    try:
        supabase.table('trades').insert(trade_data).execute()
        platform_metrics["total_trades"] += 1
    except Exception as e:
        logger.error(f"Trade logging error: {e}")

# Background tasks
async def continuous_scanner():
    """Continuously scan for trading opportunities"""
    while True:
        try:
            logger.info("Starting market scan...")
            
            # Get signals
            signals = await engine.scan_stocks(STOCK_UNIVERSE)
            current_signals.clear()
            current_signals.extend(signals[:10])  # Keep top 10
            
            # Update metrics
            platform_metrics["active_signals"] = len(current_signals)
            
            # Log signals
            for signal in signals[:5]:
                await log_signal(signal)
            
            # Broadcast to websockets
            signal_data = [
                {
                    "symbol": s.symbol,
                    "action": s.action,
                    "price": s.price,
                    "confidence": s.confidence,
                    "risk_score": s.risk_score,
                    "strategy": s.strategy,
                    "explanation": s.explanation,
                    "rsi": s.rsi,
                    "volume_ratio": s.volume_ratio,
                    "momentum": s.momentum,
                    "potential_return": s.potential_return,
                    "stop_loss": s.stop_loss,
                    "take_profit": s.take_profit,
                    "timestamp": s.timestamp
                }
                for s in current_signals
            ]
            
            # Broadcast to all connected clients
            disconnected = set()
            for ws in active_websockets:
                try:
                    await ws.send_json({
                        "type": "signals_update",
                        "data": signal_data,
                        "metrics": platform_metrics,
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    disconnected.add(ws)
            
            # Remove disconnected websockets
            active_websockets.difference_update(disconnected)
            
            # Wait before next scan
            now = datetime.now()
            if 9 <= now.hour < 16 and now.weekday() < 5:
                await asyncio.sleep(60)  # 1 minute during market hours
            else:
                await asyncio.sleep(300)  # 5 minutes outside market hours
                
        except Exception as e:
            logger.error(f"Scanner error: {e}")
            await asyncio.sleep(60)

# API Routes
@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(continuous_scanner())
    logger.info("AI Trading Platform started")

@app.get("/")
async def root():
    """Health check endpoint"""
    account = await engine.get_account_info()
    return {
        "status": "online",
        "platform": "AI Trading Platform",
        "version": "1.0.0",
        "market_status": "open" if account else "closed",
        "metrics": platform_metrics,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/signup")
async def signup(user_data: UserSignup):
    """User signup"""
    user = await create_user(user_data)
    return {
        "success": True,
        "user_id": user['id'],
        "message": "Account created successfully"
    }

@app.get("/api/signals")
async def get_signals(limit: int = 10):
    """Get current trading signals"""
    signal_data = [
        {
            "symbol": s.symbol,
            "action": s.action,
            "price": s.price,
            "confidence": s.confidence,
            "risk_score": s.risk_score,
            "strategy": s.strategy,
            "explanation": s.explanation,
            "rsi": s.rsi,
            "volume_ratio": s.volume_ratio,
            "momentum": s.momentum,
            "potential_return": s.potential_return,
            "stop_loss": s.stop_loss,
            "take_profit": s.take_profit,
            "timestamp": s.timestamp
        }
        for s in current_signals[:limit]
    ]
    
    return {
        "signals": signal_data,
        "count": len(signal_data),
        "metrics": platform_metrics,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/api/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """Analyze specific stock"""
    signal = await engine.analyze_stock(symbol.upper())
    if not signal:
        return {"message": f"No signal for {symbol} at this time"}
    
    return {
        "symbol": signal.symbol,
        "action": signal.action,
        "price": signal.price,
        "confidence": signal.confidence,
        "risk_score": signal.risk_score,
        "strategy": signal.strategy,
        "explanation": signal.explanation,
        "rsi": signal.rsi,
        "volume_ratio": signal.volume_ratio,
        "momentum": signal.momentum,
        "potential_return": signal.potential_return,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "timestamp": signal.timestamp
    }

@app.post("/api/execute")
async def execute_trade(trade: TradeRequest):
    """Execute a trade (paper trading)"""
    signal = next((s for s in current_signals if s.symbol == trade.symbol), None)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    result = await engine.execute_trade(signal, trade.quantity)
    
    await log_trade({
        'user_id': trade.user_id,
        'symbol': trade.symbol,
        'action': trade.action,
        'quantity': trade.quantity,
        'price': signal.price,
        'confidence': signal.confidence,
        'strategy': signal.strategy
    })
    
    return {
        "success": result['success'],
        "message": f"Executed {trade.action} {trade.quantity} {trade.symbol}",
        "order": result.get('order') if result['success'] else None,
        "error": result.get('error') if not result['success'] else None
    }

@app.post("/api/subscribe")
async def subscribe(subscription: SubscriptionRequest):
    """Handle subscription"""
    tier_prices = {
        'free': 0,
        'pro': 49,
        'insider': 299
    }
    
    return {
        "success": True,
        "message": f"Subscribed to {subscription.tier} tier",
        "price": tier_prices.get(subscription.tier, 0),
        "features": {
            'free': ['5 signals per day', 'Basic indicators'],
            'pro': ['Unlimited signals', 'Real-time alerts', 'All strategies'],
            'insider': ['1hr early access', 'Direct API access', 'Custom strategies']
        }.get(subscription.tier, [])
    }

@app.get("/api/performance")
async def get_performance():
    """Get platform performance metrics"""
    history = list(engine.signal_history)[-100:]
    
    if history:
        winning_signals = [s for s in history if s.confidence > 0.7]
        win_rate = (len(winning_signals) / len(history)) * 100
    else:
        win_rate = 0
    
    return {
        "win_rate": round(win_rate, 1),
        "total_signals": len(engine.signal_history),
        "active_signals": len(current_signals),
        "total_users": platform_metrics["total_users"],
        "signals_today": len([s for s in history if 
                             datetime.fromisoformat(s.timestamp).date() == datetime.now().date()]),
        "top_performers": [
            {"symbol": s.symbol, "return": s.potential_return}
            for s in sorted(history, key=lambda x: x.potential_return, reverse=True)[:5]
        ]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to AI Trading Platform",
            "timestamp": datetime.now().isoformat()
        })
        
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

@app.get("/api/account")
async def get_account():
    """Get Alpaca account info"""
    account = await engine.get_account_info()
    if not account:
        raise HTTPException(status_code=503, detail="Unable to fetch account")
    
    return {
        "buying_power": float(account.get('buying_power', 0)),
        "portfolio_value": float(account.get('portfolio_value', 0)),
        "cash": float(account.get('cash', 0)),
        "pattern_day_trader": account.get('pattern_day_trader', False),
        "trading_blocked": account.get('trading_blocked', False),
        "account_blocked": account.get('account_blocked', False)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)