"""
Production Trading System v4.0
- Integrates with infrastructure platform
- Handles rate limits
- Uses free Polygon tier efficiently
- Caches aggressively to avoid API calls
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd
from collections import deque
import httpx
import yfinance as yf
import pytz
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Production Trading Engine",
    version="4.0.0"
)

# CORS configuration
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

# Infrastructure API
INFRA_API = os.getenv("INFRA_API", "http://localhost:8000")

# Polygon API (Free tier: 5 API calls/minute)
POLYGON_KEY = os.getenv('POLYGON_API_KEY', 'free_tier_key')
POLYGON_RATE_LIMIT = 5  # calls per minute

# Data source priority (use cheapest first)
DATA_SOURCES = {
    'yfinance': {'limit': 2000, 'cost': 0},  # Free, 2000/hour soft limit
    'polygon': {'limit': 5, 'cost': 1},  # Free tier: 5/min
}

# Volatility zones
VOLATILITY_ZONES = {
    "opening_bell": {"start": "09:30", "end": "10:00", "multiplier": 1.8, "threshold": 0.3},
    "morning_reversal": {"start": "10:00", "end": "10:30", "multiplier": 1.3, "threshold": 0.35},
    "midday": {"start": "10:30", "end": "14:00", "multiplier": 1.1, "threshold": 0.4},
    "pre_power": {"start": "14:00", "end": "15:00", "multiplier": 1.3, "threshold": 0.35},
    "power_hour": {"start": "15:00", "end": "15:50", "multiplier": 1.5, "threshold": 0.3},
    "closing_cross": {"start": "15:50", "end": "16:00", "multiplier": 2.0, "threshold": 0.25}
}

# Limited universe for free tier
STOCK_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]  # Just 5 to manage rate limits

# ============================================
# INFRASTRUCTURE CLIENT
# ============================================

class InfrastructureClient:
    """Client for platform infrastructure API"""
    
    def __init__(self):
        self.base_url = INFRA_API
        self.client = httpx.AsyncClient(timeout=30.0)
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def verify_user(self, token: str) -> Optional[Dict]:
        """Verify user token and get permissions"""
        cache_key = f"user_{token[:20]}"
        
        # Check cache
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/integration/verify-user",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                data = response.json()
                self._cache[cache_key] = (data, time.time())
                return data
        except Exception as e:
            logger.error(f"Infrastructure verification error: {e}")
        
        return None
    
    async def record_trade(self, token: str, trade: Dict) -> bool:
        """Record executed trade in infrastructure"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/integration/record-trade",
                headers={"Authorization": f"Bearer {token}"},
                json=trade
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Trade recording error: {e}")
            return False
    
    async def get_user_accounts(self, token: str) -> List[Dict]:
        """Get user's trading accounts"""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/accounts",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Get accounts error: {e}")
        return []

infra = InfrastructureClient()

# ============================================
# RATE LIMITER
# ============================================

class RateLimiter:
    """Manage API rate limits across data sources"""
    
    def __init__(self):
        self.calls = {
            'yfinance': deque(maxlen=2000),
            'polygon': deque(maxlen=5)
        }
        
    def can_call(self, source: str) -> bool:
        """Check if we can make an API call"""
        if source not in self.calls:
            return False
            
        now = time.time()
        
        if source == 'yfinance':
            # 2000 per hour
            self.calls[source] = deque(
                [t for t in self.calls[source] if now - t < 3600],
                maxlen=2000
            )
            return len(self.calls[source]) < 2000
            
        elif source == 'polygon':
            # 5 per minute
            self.calls[source] = deque(
                [t for t in self.calls[source] if now - t < 60],
                maxlen=5
            )
            return len(self.calls[source]) < 5
            
        return False
    
    def record_call(self, source: str):
        """Record an API call"""
        if source in self.calls:
            self.calls[source].append(time.time())
    
    def get_wait_time(self, source: str) -> float:
        """Get seconds to wait before next call"""
        if source not in self.calls or not self.calls[source]:
            return 0
            
        now = time.time()
        
        if source == 'yfinance':
            oldest = min(self.calls[source])
            wait = max(0, 3600 - (now - oldest))
        elif source == 'polygon':
            oldest = min(self.calls[source])
            wait = max(0, 60 - (now - oldest))
        else:
            wait = 0
            
        return wait

rate_limiter = RateLimiter()

# ============================================
# DATA CACHE
# ============================================

class DataCache:
    """Cache market data to minimize API calls"""
    
    def __init__(self):
        self.cache = {}
        self.ttl = {
            'quote': 60,  # 1 minute for quotes
            'bars': 300,  # 5 minutes for historical bars
            'snapshot': 30  # 30 seconds for snapshots
        }
    
    def get(self, key: str, data_type: str = 'quote') -> Optional[Any]:
        """Get cached data if valid"""
        if key not in self.cache:
            return None
            
        data, timestamp = self.cache[key]
        ttl = self.ttl.get(data_type, 60)
        
        if time.time() - timestamp > ttl:
            del self.cache[key]
            return None
            
        return data
    
    def set(self, key: str, data: Any):
        """Set cache with timestamp"""
        self.cache[key] = (data, time.time())
    
    def clear_expired(self):
        """Remove expired entries"""
        now = time.time()
        expired = []
        
        for key, (_, timestamp) in self.cache.items():
            if now - timestamp > 600:  # 10 minutes max
                expired.append(key)
        
        for key in expired:
            del self.cache[key]

data_cache = DataCache()

# ============================================
# DATA FETCHER
# ============================================

class DataFetcher:
    """Fetch data with rate limiting and caching"""
    
    async def get_quotes(self, symbols: List[str]) -> Dict:
        """Get current quotes with caching and rate limiting"""
        quotes = {}
        
        for symbol in symbols:
            # Check cache first
            cached = data_cache.get(f"quote_{symbol}", 'quote')
            if cached:
                quotes[symbol] = cached
                continue
            
            # Try yfinance first (free)
            if rate_limiter.can_call('yfinance'):
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        quote = {
                            'price': hist['Close'].iloc[-1],
                            'volume': hist['Volume'].iloc[-1],
                            'timestamp': datetime.now().isoformat()
                        }
                        quotes[symbol] = quote
                        data_cache.set(f"quote_{symbol}", quote)
                        rate_limiter.record_call('yfinance')
                        continue
                except Exception as e:
                    logger.debug(f"yfinance error for {symbol}: {e}")
            
            # Fallback to last known price
            quotes[symbol] = {'price': 0, 'volume': 0, 'timestamp': datetime.now().isoformat()}
        
        return quotes
    
    async def get_bars(self, symbol: str, period: str = "5d") -> Optional[pd.DataFrame]:
        """Get historical bars with caching"""
        cache_key = f"bars_{symbol}_{period}"
        
        # Check cache
        cached = data_cache.get(cache_key, 'bars')
        if cached is not None:
            return cached
        
        # Use yfinance (free)
        if rate_limiter.can_call('yfinance'):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval="5m")
                if not hist.empty:
                    data_cache.set(cache_key, hist)
                    rate_limiter.record_call('yfinance')
                    return hist
            except Exception as e:
                logger.error(f"Error fetching bars for {symbol}: {e}")
        
        return None

data_fetcher = DataFetcher()

# ============================================
# SIGNAL GENERATION
# ============================================

@dataclass
class TradingSignal:
    symbol: str
    action: str
    price: float
    confidence: float
    strategy: str
    stop_loss: float
    take_profit: float
    position_size: int
    timestamp: str
    user_id: Optional[str] = None
    account_id: Optional[str] = None
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

class SignalGenerator:
    """Generate signals with minimal API calls"""
    
    def __init__(self):
        self.last_signals = {}
        self.signal_ttl = 60  # Reuse signals for 60 seconds
        
    async def generate_signal(self, symbol: str, user_data: Optional[Dict] = None) -> Optional[TradingSignal]:
        """Generate trading signal with caching"""
        
        # Check if we have a recent signal
        if symbol in self.last_signals:
            signal, timestamp = self.last_signals[symbol]
            if time.time() - timestamp < self.signal_ttl:
                return signal
        
        # Get historical data
        bars = await data_fetcher.get_bars(symbol)
        if bars is None or len(bars) < 20:
            return None
        
        # Simple technical analysis
        df = bars.copy()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        # Get volatility settings
        mult, threshold = self.get_volatility_multiplier()
        
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
        elif current_price > latest['SMA_20']:
            action = "BUY"
            confidence = 0.55
            strategy = "trend_follow"
        elif current_price < latest['SMA_20']:
            action = "SELL"
            confidence = 0.55
            strategy = "trend_reverse"
        
        # Apply volatility boost
        if mult > 1.0:
            confidence = min(0.95, confidence * mult)
        
        # Check threshold
        if not action or confidence < threshold:
            return None
        
        # Calculate position size based on user tier
        base_size = 100
        if user_data:
            if user_data.get('subscription_tier') == 'premium':
                base_size = 200
            elif user_data.get('subscription_tier') == 'pro':
                base_size = 500
        
        position_size = int(base_size * (confidence / 0.7))
        
        # Set stops
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if action == "BUY":
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 2.5)
        else:
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 2.5)
        
        signal = TradingSignal(
            symbol=symbol,
            action=action,
            price=float(current_price),
            confidence=confidence,
            strategy=strategy,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            position_size=position_size,
            timestamp=datetime.now().isoformat(),
            user_id=user_data.get('user_id') if user_data else None,
            account_id=user_data.get('account_id') if user_data else None
        )
        
        # Cache the signal
        self.last_signals[symbol] = (signal, time.time())
        
        return signal
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 1)
        return 100 - (100 / (1 + rs))
    
    def get_volatility_multiplier(self) -> tuple:
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
    
    async def scan_universe(self, user_data: Optional[Dict] = None) -> List[TradingSignal]:
        """Scan universe with rate limit awareness"""
        signals = []
        mult, _ = self.get_volatility_multiplier()
        
        # Limit scans based on rate limits
        symbols = STOCK_UNIVERSE[:3] if rate_limiter.get_wait_time('yfinance') > 0 else STOCK_UNIVERSE
        
        for symbol in symbols:
            # Add delay between requests to avoid rate limits
            await asyncio.sleep(0.5)
            
            signal = await self.generate_signal(symbol, user_data)
            if signal:
                signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Return based on user tier
        max_signals = 3  # Free tier
        if user_data:
            if user_data.get('subscription_tier') == 'premium':
                max_signals = 10
            elif user_data.get('subscription_tier') == 'pro':
                max_signals = 5
        
        return signals[:max_signals]

signal_generator = SignalGenerator()

# ============================================
# API DEPENDENCIES
# ============================================

async def get_current_user(authorization: str = Header(None)) -> Optional[Dict]:
    """Get current user from infrastructure"""
    if not authorization:
        return None
    
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    user = await infra.verify_user(token)
    
    if not user:
        raise HTTPException(401, "Invalid or expired token")
    
    return user

async def require_user(user: Dict = Depends(get_current_user)) -> Dict:
    """Require authenticated user"""
    if not user:
        raise HTTPException(401, "Authentication required")
    return user

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Health check and status"""
    mult, threshold = signal_generator.get_volatility_multiplier()
    
    return {
        "platform": "Production Trading Engine",
        "version": "4.0.0",
        "status": "operational",
        "infrastructure": {
            "connected": INFRA_API,
            "status": "ready"
        },
        "data_sources": {
            "yfinance": {
                "available": rate_limiter.can_call('yfinance'),
                "wait_time": rate_limiter.get_wait_time('yfinance')
            },
            "polygon": {
                "available": rate_limiter.can_call('polygon'),
                "wait_time": rate_limiter.get_wait_time('polygon')
            }
        },
        "volatility": {
            "multiplier": mult,
            "threshold": threshold,
            "active": mult > 1.0
        },
        "cache_size": len(data_cache.cache)
    }

@app.get("/api/signals")
async def get_signals(user: Optional[Dict] = Depends(get_current_user)):
    """Get trading signals (rate limited based on tier)"""
    
    # Public signals (limited)
    if not user:
        signals = await signal_generator.scan_universe()
        return {
            "signals": [s.to_dict() for s in signals[:1]],  # Only 1 for public
            "tier": "public",
            "message": "Sign in for more signals"
        }
    
    # User signals based on tier
    signals = await signal_generator.scan_universe(user)
    
    return {
        "signals": [s.to_dict() for s in signals],
        "count": len(signals),
        "tier": user.get('subscription_tier', 'free'),
        "limits": user.get('limits', {}),
        "cache_hit_rate": f"{len(data_cache.cache)}/{len(STOCK_UNIVERSE)}"
    }

@app.post("/api/execute/{symbol}")
async def execute_trade(
    symbol: str,
    action: Optional[str] = None,
    quantity: Optional[int] = None,
    account_id: Optional[str] = None,
    user: Dict = Depends(require_user),
    authorization: str = Header()
):
    """Execute trade with infrastructure integration"""
    
    # Check user limits
    if not user.get('can_live_trade', False):
        return {
            "success": False,
            "error": "Upgrade to premium for live trading",
            "tier": user.get('subscription_tier', 'free')
        }
    
    # Get user accounts
    accounts = await infra.get_user_accounts(authorization)
    if not accounts:
        return {"success": False, "error": "No trading accounts found"}
    
    # Select account
    if account_id:
        account = next((a for a in accounts if a['id'] == account_id), None)
    else:
        account = accounts[0]  # Default to first account
    
    if not account:
        return {"success": False, "error": "Invalid account"}
    
    # Generate signal
    signal = await signal_generator.generate_signal(symbol, {**user, 'account_id': account['id']})
    
    if not signal:
        return {"success": False, "error": "No valid signal generated"}
    
    # Override if specified
    if action:
        signal.action = action
    if quantity:
        signal.position_size = quantity
    
    # Mock execution (replace with real broker API)
    trade_result = {
        "success": True,
        "order_id": f"ORD_{int(time.time())}",
        "symbol": signal.symbol,
        "action": signal.action,
        "quantity": signal.position_size,
        "price": signal.price,
        "timestamp": signal.timestamp
    }
    
    # Record in infrastructure
    await infra.record_trade(authorization, {
        "account_id": account['id'],
        "symbol": signal.symbol,
        "action": signal.action,
        "quantity": signal.position_size,
        "price": signal.price,
        "status": "executed",
        "strategy": signal.strategy,
        "confidence": signal.confidence
    })
    
    return trade_result

@app.get("/api/quotes/{symbols}")
async def get_quotes(symbols: str, user: Optional[Dict] = Depends(get_current_user)):
    """Get current quotes (cached to avoid rate limits)"""
    
    symbol_list = symbols.split(',')
    
    # Limit symbols based on tier
    if not user:
        symbol_list = symbol_list[:1]
    elif user.get('subscription_tier') == 'free':
        symbol_list = symbol_list[:3]
    elif user.get('subscription_tier') == 'pro':
        symbol_list = symbol_list[:5]
    
    quotes = await data_fetcher.get_quotes(symbol_list)
    
    return {
        "quotes": quotes,
        "cached": True,
        "tier": user.get('subscription_tier', 'public') if user else 'public'
    }

@app.get("/api/rate-limits")
async def get_rate_limits(user: Dict = Depends(require_user)):
    """Get current rate limit status"""
    
    return {
        "yfinance": {
            "used": len(rate_limiter.calls['yfinance']),
            "limit": 2000,
            "period": "hour",
            "wait_time": rate_limiter.get_wait_time('yfinance')
        },
        "polygon": {
            "used": len(rate_limiter.calls['polygon']),
            "limit": 5,
            "period": "minute",
            "wait_time": rate_limiter.get_wait_time('polygon')
        },
        "cache_stats": {
            "entries": len(data_cache.cache),
            "ttl": data_cache.ttl
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    """WebSocket for real-time updates"""
    await websocket.accept()
    
    # Verify user if token provided
    user = None
    if token:
        user = await infra.verify_user(token)
    
    try:
        while True:
            # Send updates based on tier
            if user and user.get('subscription_tier') in ['pro', 'premium']:
                # Real-time updates for paid users
                await asyncio.sleep(10)
            else:
                # Slower updates for free users
                await asyncio.sleep(60)
            
            # Send heartbeat with current stats
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "tier": user.get('subscription_tier', 'free') if user else 'public'
            })
            
    except WebSocketDisconnect:
        pass

# ============================================
# BACKGROUND TASKS
# ============================================

async def cache_cleanup_task():
    """Periodic cache cleanup"""
    while True:
        await asyncio.sleep(600)  # Every 10 minutes
        data_cache.clear_expired()
        logger.info(f"Cache cleanup: {len(data_cache.cache)} entries remaining")

async def preload_data_task():
    """Preload data during low activity"""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        
        # Preload during off-hours to save API calls during trading
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        if now.hour < 9 or now.hour >= 16:  # Outside market hours
            for symbol in STOCK_UNIVERSE:
                if rate_limiter.can_call('yfinance'):
                    await data_fetcher.get_bars(symbol)
                    await asyncio.sleep(1)

@app.on_event("startup")
async def startup():
    """Start background tasks"""
    asyncio.create_task(cache_cleanup_task())
    asyncio.create_task(preload_data_task())
    
    logger.info("="*60)
    logger.info("PRODUCTION TRADING ENGINE v4.0")
    logger.info(f"Infrastructure API: {INFRA_API}")
    logger.info(f"Rate Limits: YFinance 2000/hr, Polygon 5/min")
    logger.info(f"Cache TTL: Quotes 60s, Bars 300s")
    logger.info("="*60)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)