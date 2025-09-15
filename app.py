import os
import asyncio
import json
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel

# FastAPI and web framework imports
from fastapi import FastAPI, WebSocket, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# External libraries
import pandas as pd
import numpy as np
import yfinance as yf
import stripe
import alpaca_trade_api as tradeapi
from cryptography.fernet import Fernet

# Your local imports
from trading_engine import AdvancedTradingEngine, Signal
from ml_brain import ml_brain, generate_ml_prediction, get_ml_stats

# Optional imports with error handling
try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
except:
    supabase = None
    print("Supabase not configured - running without database")

# Initialize FastAPI app FIRST
app = FastAPI(title="AI Trading Platform")

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

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Environment variables
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

# Initialize trading engine
engine = AdvancedTradingEngine(ALPACA_KEY, ALPACA_SECRET) if ALPACA_KEY else None

# Generate encryption key for storing API keys
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY or ENCRYPTION_KEY == "generate_with_fernet" or ENCRYPTION_KEY == "generate-random-string-here":
    # Generate a valid key if placeholder or missing
    ENCRYPTION_KEY = Fernet.generate_key()
    logger.warning("Generated new encryption key - set ENCRYPTION_KEY env var in production")
else:
    # Ensure the key is in bytes format
    try:
        if isinstance(ENCRYPTION_KEY, str):
            ENCRYPTION_KEY = ENCRYPTION_KEY.encode()
    except:
        ENCRYPTION_KEY = Fernet.generate_key()
        logger.warning("Invalid encryption key format - generated new one")

try:
    cipher = Fernet(ENCRYPTION_KEY)
except Exception as e:
    logger.error(f"Encryption key error: {e}")
    ENCRYPTION_KEY = Fernet.generate_key()
    cipher = Fernet(ENCRYPTION_KEY)

# Global state
active_websockets = set()
current_signals = []
user_daily_signals = {}
platform_metrics = {
    "total_users": 0,
    "active_signals": 0,
    "win_rate": 73,
    "total_trades": 0,
    "profit_today": 0
}

# Stock universe
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

# Health check endpoint - MUST be first
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "photon"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "platform": "AI Trading Platform",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

class UserAlpacaManager:
    """Manage individual user Alpaca connections"""
    
    def __init__(self):
        self.user_connections = {}
    
    def encrypt_credentials(self, api_key: str, secret_key: str) -> Dict:
        """Encrypt Alpaca credentials for storage"""
        encrypted_key = cipher.encrypt(api_key.encode()).decode()
        encrypted_secret = cipher.encrypt(secret_key.encode()).decode()
        return {
            'api_key_encrypted': encrypted_key,
            'secret_key_encrypted': encrypted_secret
        }
    
    def decrypt_credentials(self, encrypted_data: Dict) -> Dict:
        """Decrypt stored Alpaca credentials"""
        api_key = cipher.decrypt(encrypted_data['api_key_encrypted'].encode()).decode()
        secret_key = cipher.decrypt(encrypted_data['secret_key_encrypted'].encode()).decode()
        return {'api_key': api_key, 'secret_key': secret_key}
    
    async def link_alpaca_account(self, user_id: str, api_key: str, secret_key: str, paper: bool = True) -> Dict:
        """Link user's Alpaca account"""
        try:
            base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
            api = tradeapi.REST(api_key, secret_key, base_url=base_url, api_version='v2')
            
            account = api.get_account()
            encrypted = self.encrypt_credentials(api_key, secret_key)
            
            if supabase:
                supabase.table('alpaca_accounts').upsert({
                    'user_id': user_id,
                    'api_key_encrypted': encrypted['api_key_encrypted'],
                    'secret_key_encrypted': encrypted['secret_key_encrypted'],
                    'paper_trading': paper,
                    'account_number': account.account_number,
                    'buying_power': float(account.buying_power),
                    'portfolio_value': float(account.portfolio_value),
                    'linked_at': datetime.now().isoformat()
                }).execute()
            
            self.user_connections[user_id] = api
            
            return {
                'success': True,
                'account_number': account.account_number,
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'paper_trading': paper
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Invalid Alpaca credentials or API error'
            }
    
    async def get_user_api(self, user_id: str) -> tradeapi.REST:
        """Get user's Alpaca API connection"""
        if user_id in self.user_connections:
            return self.user_connections[user_id]
        
        if supabase:
            result = supabase.table('alpaca_accounts').select('*').eq('user_id', user_id).execute()
            if result.data:
                account_data = result.data[0]
                creds = self.decrypt_credentials(account_data)
                
                base_url = 'https://paper-api.alpaca.markets' if account_data['paper_trading'] else 'https://api.alpaca.markets'
                api = tradeapi.REST(creds['api_key'], creds['secret_key'], base_url=base_url, api_version='v2')
                
                self.user_connections[user_id] = api
                return api
        
        return None
    
    async def execute_user_trade(self, user_id: str, signal: Signal, shares: int) -> Dict:
        """Execute trade on user's Alpaca account"""
        api = await self.get_user_api(user_id)
        if not api:
            return {'success': False, 'error': 'No Alpaca account linked'}
        
        try:
            order = api.submit_order(
                symbol=signal.symbol,
                qty=shares,
                side='buy' if signal.action == 'BUY' else 'sell',
                type='limit',
                limit_price=signal.price,
                time_in_force='day',
                order_class='bracket',
                stop_loss={'stop_price': signal.stop_loss},
                take_profit={'limit_price': signal.take_profit}
            )
            
            return {
                'success': True,
                'order_id': order.id,
                'status': order.status,
                'filled_qty': order.filled_qty,
                'filled_price': order.filled_avg_price
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_user_positions(self, user_id: str) -> List[Dict]:
        """Get user's current positions"""
        api = await self.get_user_api(user_id)
        if not api:
            return []
        
        try:
            positions = api.list_positions()
            return [{
                'symbol': p.symbol,
                'qty': int(p.qty),
                'side': p.side,
                'avg_entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price) if hasattr(p, 'current_price') else 0,
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc) if hasattr(p, 'unrealized_plpc') else 0
            } for p in positions]
        except:
            return []

# Initialize manager
alpaca_manager = UserAlpacaManager()

class CommissionTracker:
    """Track and calculate 5% commission on profitable trades"""
    
    def __init__(self, commission_rate=0.05):
        self.commission_rate = commission_rate
        self.user_positions = {}
        self.user_commissions = {}
        
    async def record_entry(self, user_id: str, trade_data: Dict):
        """Record when user enters a position"""
        if user_id not in self.user_positions:
            self.user_positions[user_id] = []
            
        position = {
            'symbol': trade_data['symbol'],
            'entry_price': trade_data['price'],
            'quantity': trade_data['quantity'],
            'action': trade_data['action'],
            'timestamp': datetime.now().isoformat(),
            'position_id': secrets.token_urlsafe(8)
        }
        
        self.user_positions[user_id].append(position)
        
        if supabase:
            supabase.table('positions').insert({
                'user_id': user_id,
                'position_id': position['position_id'],
                'symbol': position['symbol'],
                'entry_price': position['entry_price'],
                'quantity': position['quantity'],
                'action': position['action'],
                'status': 'open'
            }).execute()
            
        return position['position_id']
    
    async def record_exit(self, user_id: str, exit_data: Dict):
        """Record when user exits a position and calculate commission"""
        if user_id not in self.user_positions:
            return None
            
        position = None
        for pos in self.user_positions[user_id]:
            if pos['symbol'] == exit_data['symbol'] and pos['action'] != exit_data['action']:
                position = pos
                break
                
        if not position:
            return None
            
        if position['action'] == 'BUY':
            profit = (exit_data['price'] - position['entry_price']) * position['quantity']
        else:
            profit = (position['entry_price'] - exit_data['price']) * position['quantity']
            
        commission = 0
        if profit > 0:
            commission = profit * self.commission_rate
            
            if user_id not in self.user_commissions:
                self.user_commissions[user_id] = 0
            self.user_commissions[user_id] += commission
            
            if supabase:
                supabase.table('commissions').insert({
                    'user_id': user_id,
                    'position_id': position['position_id'],
                    'profit': profit,
                    'commission': commission,
                    'timestamp': datetime.now().isoformat()
                }).execute()
        
        self.user_positions[user_id].remove(position)
        
        return {
            'profit': profit,
            'commission': commission,
            'net_profit': profit - commission
        }

# Initialize commission tracker
commission_tracker = CommissionTracker(commission_rate=0.05)

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
            if not engine:
                await asyncio.sleep(60)
                continue
                
            logger.info("Starting market scan...")
            
            signals = await engine.scan_stocks(STOCK_UNIVERSE)
            current_signals.clear()
            current_signals.extend(signals[:10])
            
            platform_metrics["active_signals"] = len(current_signals)
            
            for signal in signals[:5]:
                await log_signal(signal)
            
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
            
            active_websockets.difference_update(disconnected)
            
            now = datetime.now()
            if 9 <= now.hour < 16 and now.weekday() < 5:
                await asyncio.sleep(60)
            else:
                await asyncio.sleep(300)
                
        except Exception as e:
            logger.error(f"Scanner error: {e}")
            await asyncio.sleep(60)

async def generate_initial_training_data():
    """Generate synthetic training data for initial ML training"""
    await asyncio.sleep(10)
    
    import random
    for _ in range(200):
        features = np.random.randn(12)
        outcome = random.choice(['profitable', 'loss', 'neutral'])
        ml_brain.add_training_sample(features, outcome)
    
    logger.info("Starting initial ML training...")
    ml_brain.train_models(force_retrain=True)
    logger.info("Initial ML training complete!")

# API Routes
@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(continuous_scanner())
    asyncio.create_task(generate_initial_training_data())
    logger.info("AI Trading Platform started")

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

@app.post("/api/execute")
async def execute_trade(trade: TradeRequest):
    """Execute a trade"""
    signal = next((s for s in current_signals if s.symbol == trade.symbol), None)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    result = await alpaca_manager.execute_user_trade(trade.user_id, signal, trade.quantity)
    
    if result['success']:
        await log_trade({
            'user_id': trade.user_id,
            'symbol': trade.symbol,
            'action': trade.action,
            'quantity': trade.quantity,
            'price': signal.price,
            'confidence': signal.confidence,
            'strategy': signal.strategy
        })
    
    return result

@app.post("/api/link-alpaca")
async def link_alpaca_account(request: Dict):
    """Link user's Alpaca account"""
    user_id = request['user_id']
    api_key = request['api_key']
    secret_key = request['secret_key']
    paper = request.get('paper_trading', True)
    
    result = await alpaca_manager.link_alpaca_account(user_id, api_key, secret_key, paper)
    return result

@app.get("/api/alpaca-account/{user_id}")
async def get_alpaca_account(user_id: str):
    """Get user's Alpaca account info"""
    api = await alpaca_manager.get_user_api(user_id)
    if not api:
        return {'error': 'No Alpaca account linked'}
    
    try:
        account = api.get_account()
        return {
            'account_number': account.account_number,
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked
        }
    except Exception as e:
        return {'error': str(e)}

@app.get("/api/user-positions/{user_id}")
async def get_user_positions(user_id: str):
    """Get user's current positions"""
    positions = await alpaca_manager.get_user_positions(user_id)
    return {'positions': positions, 'count': len(positions)}

@app.post("/api/create-checkout")
async def create_checkout_session(request: Dict):
    """Create Stripe checkout session for subscription"""
    logger.info(f"Checkout request: {request}")
    
    user_id = request.get('user_id')
    price_id = request.get('price_id')
    tier = request.get('tier')
    
    price_mapping = {
        'price_1S6KlkPuxT6s7WvF1Pn0dChn': 'pro',
        'price_1S6KlvPuxT6s7WvFAuxjJQA7': 'insider'
    }
    
    if price_id not in price_mapping:
        raise HTTPException(status_code=400, detail="Invalid price ID")
    
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https://ai-trading-platform-p93x6ap4t-charlie-kupers-projects.vercel.app/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='https://ai-trading-platform-p93x6ap4t-charlie-kupers-projects.vercel.app/',
            customer_email=request.get('email'),
            metadata={
                'user_id': user_id,
                'tier': tier
            },
            subscription_data={
                'metadata': {
                    'user_id': user_id,
                    'tier': tier
                }
            }
        )
        
        return {
            'session_id': checkout_session.id,
            'url': checkout_session.url
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Checkout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Server error creating checkout")

@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        
        user_id = session['metadata'].get('user_id')
        tier = session['metadata'].get('tier')
        
        if user_id and tier and supabase:
            supabase.table('users').update({
                'subscription_tier': tier,
                'stripe_customer_id': session['customer'],
                'stripe_subscription_id': session['subscription'],
                'subscription_started': datetime.now().isoformat()
            }).eq('id', user_id).execute()
            
            logger.info(f"User {user_id} upgraded to {tier}")
            
            for key in list(user_daily_signals.keys()):
                if key.startswith(user_id):
                    del user_daily_signals[key]
    
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        
        if supabase:
            result = supabase.table('users').select('id').eq(
                'stripe_subscription_id', subscription['id']
            ).execute()
            
            if result.data:
                user_id = result.data[0]['id']
                supabase.table('users').update({
                    'subscription_tier': 'free',
                    'subscription_ended': datetime.now().isoformat()
                }).eq('id', user_id).execute()
                
                logger.info(f"User {user_id} downgraded to free tier")
    
    return {'received': True}

@app.get("/api/commission-balance/{user_id}")
async def get_commission_balance(user_id: str):
    """Get user's commission balance"""
    balance = 0
    if supabase:
        result = supabase.table('commissions').select('commission').eq('user_id', user_id).execute()
        balance = sum(row['commission'] for row in result.data)
    
    return {
        "user_id": user_id,
        "commission_owed": balance,
        "commission_rate": "5%",
        "message": f"You owe ${balance:.2f} in commission on profitable trades"
    }

@app.get("/api/ml-stats")
async def ml_stats():
    """Get ML system statistics"""
    stats = get_ml_stats()
    return {
        "ml_status": "active" if stats['is_trained'] else "learning",
        "models_trained": stats['is_trained'],
        "training_samples": stats['training_samples'],
        "predictions_made": stats['predictions_made'],
        "model_accuracies": stats.get('model_accuracies', {}),
        "best_model": max(stats['model_weights'], key=stats['model_weights'].get) if stats['model_weights'] else "none"
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
    if not engine:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")
    
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

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Photon Trading Platform on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )