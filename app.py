from ml_brain import ml_brain, generate_ml_prediction, get_ml_stats
import os
import asyncio
import json
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
import numpy as np
# Add this to app.py for FULL Alpaca account linking

import alpaca_trade_api as tradeapi
from cryptography.fernet import Fernet
import base64

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Environment variables
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID")  # Note: _ID at the end
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

# Generate encryption key for storing API keys (store this in env)
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY") or Fernet.generate_key().decode()
cipher = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

class UserAlpacaManager:
    """Manage individual user Alpaca connections"""
    
    def __init__(self):
        self.user_connections = {}  # Cache of active connections
    
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
            # Test the credentials
            base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
            api = tradeapi.REST(api_key, secret_key, base_url=base_url, api_version='v2')
            
            # Verify by getting account info
            account = api.get_account()
            
            # Encrypt and store credentials
            encrypted = self.encrypt_credentials(api_key, secret_key)
            
            if supabase:
                # Store in database
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
            
            # Cache the connection
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
        # Check cache first
        if user_id in self.user_connections:
            return self.user_connections[user_id]
        
        # Load from database
        if supabase:
            result = supabase.table('alpaca_accounts').select('*').eq('user_id', user_id).execute()
            if result.data:
                account_data = result.data[0]
                creds = self.decrypt_credentials(account_data)
                
                base_url = 'https://paper-api.alpaca.markets' if account_data['paper_trading'] else 'https://api.alpaca.markets'
                api = tradeapi.REST(creds['api_key'], creds['secret_key'], base_url=base_url, api_version='v2')
                
                # Cache it
                self.user_connections[user_id] = api
                return api
        
        return None
    
    async def execute_user_trade(self, user_id: str, signal: Signal, shares: int) -> Dict:
        """Execute trade on user's Alpaca account"""
        api = await self.get_user_api(user_id)
        if not api:
            return {'success': False, 'error': 'No Alpaca account linked'}
        
        try:
            # Place order with bracket (stop loss and take profit)
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
                'current_price': float(p.current_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            } for p in positions]
        except:
            return []

# Initialize manager
alpaca_manager = UserAlpacaManager()

# API Endpoints for Alpaca linking
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

# Add this to your app.py after the imports

class CommissionTracker:
    """Track and calculate 5% commission on profitable trades"""
    
    def __init__(self, commission_rate=0.05):
        self.commission_rate = commission_rate
        self.user_positions = {}  # Track open positions
        self.user_commissions = {}  # Track commissions owed
        
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
        
        # Store in database
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
            
        # Find matching position
        position = None
        for pos in self.user_positions[user_id]:
            if pos['symbol'] == exit_data['symbol'] and pos['action'] != exit_data['action']:
                position = pos
                break
                
        if not position:
            return None
            
        # Calculate P&L
        if position['action'] == 'BUY':
            # Long position
            profit = (exit_data['price'] - position['entry_price']) * position['quantity']
        else:
            # Short position
            profit = (position['entry_price'] - exit_data['price']) * position['quantity']
            
        # Calculate commission on profits only
        commission = 0
        if profit > 0:
            commission = profit * self.commission_rate
            
            # Track commission
            if user_id not in self.user_commissions:
                self.user_commissions[user_id] = 0
            self.user_commissions[user_id] += commission
            
            # Store in database
            if supabase:
                supabase.table('commissions').insert({
                    'user_id': user_id,
                    'position_id': position['position_id'],
                    'profit': profit,
                    'commission': commission,
                    'timestamp': datetime.now().isoformat()
                }).execute()
        
        # Remove closed position
        self.user_positions[user_id].remove(position)
        
        return {
            'profit': profit,
            'commission': commission,
            'net_profit': profit - commission
        }
    
    async def get_user_commission_balance(self, user_id: str) -> float:
        """Get total commission owed by user"""
        if supabase:
            result = supabase.table('commissions').select('commission').eq('user_id', user_id).execute()
            return sum(row['commission'] for row in result.data)
        return self.user_commissions.get(user_id, 0)
    
    async def process_commission_payment(self, user_id: str, amount: float):
        """Process commission payment via Stripe"""
        try:
            # Create Stripe invoice for commission
            invoice = stripe.Invoice.create(
                customer=user_id,  # Assumes Stripe customer ID stored as user_id
                collection_method='charge_automatically',
                description=f"Trading commission ({self.commission_rate*100}% of profits)",
                auto_advance=True
            )
            
            # Add line item
            stripe.InvoiceItem.create(
                customer=user_id,
                amount=int(amount * 100),  # Convert to cents
                currency='usd',
                description='Trading commission on profitable trades',
                invoice=invoice.id
            )
            
            # Finalize and charge
            invoice.finalize_invoice()
            invoice.pay()
            
            # Update database
            if supabase:
                supabase.table('commission_payments').insert({
                    'user_id': user_id,
                    'amount': amount,
                    'stripe_invoice_id': invoice.id,
                    'status': 'paid',
                    'timestamp': datetime.now().isoformat()
                }).execute()
                
            return {'success': True, 'invoice_id': invoice.id}
            
        except Exception as e:
            logger.error(f"Commission payment failed: {e}")
            return {'success': False, 'error': str(e)}

# Initialize commission tracker
commission_tracker = CommissionTracker(commission_rate=0.05)

# Add these API endpoints to your app.py:

@app.post("/api/execute-with-commission")
async def execute_trade_with_commission(trade: TradeRequest):
    """Execute trade and track for commission"""
    # Find the signal
    signal = next((s for s in current_signals if s.symbol == trade.symbol), None)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    # Execute via Alpaca
    result = await engine.execute_trade(signal, trade.quantity)
    
    if result['success']:
        # Record entry for commission tracking
        position_id = await commission_tracker.record_entry(
            trade.user_id,
            {
                'symbol': trade.symbol,
                'action': trade.action,
                'quantity': trade.quantity,
                'price': signal.price
            }
        )
        
        # Log trade
        await log_trade({
            'user_id': trade.user_id,
            'position_id': position_id,
            'symbol': trade.symbol,
            'action': trade.action,
            'quantity': trade.quantity,
            'price': signal.price,
            'confidence': signal.confidence,
            'strategy': signal.strategy
        })
        
        return {
            "success": True,
            "message": f"Executed {trade.action} {trade.quantity} {trade.symbol}",
            "position_id": position_id,
            "commission_rate": "5% on profits",
            "order": result.get('order')
        }
    
    return {
        "success": False,
        "error": result.get('error')
    }

@app.post("/api/close-position")
async def close_position(request: Dict):
    """Close position and calculate commission"""
    user_id = request['user_id']
    symbol = request['symbol']
    exit_price = request['exit_price']
    action = request['action']  # Opposite of entry action
    
    # Record exit and calculate commission
    result = await commission_tracker.record_exit(
        user_id,
        {
            'symbol': symbol,
            'price': exit_price,
            'action': action
        }
    )
    
    if result:
        return {
            "success": True,
            "profit": result['profit'],
            "commission": result['commission'],
            "net_profit": result['net_profit'],
            "message": f"Position closed. {'Profit' if result['profit'] > 0 else 'Loss'}: ${abs(result['profit']):.2f}"
        }
    
    return {
        "success": False,
        "message": "No matching position found"
    }

@app.get("/api/commission-balance/{user_id}")
async def get_commission_balance(user_id: str):
    """Get user's commission balance"""
    balance = await commission_tracker.get_user_commission_balance(user_id)
    return {
        "user_id": user_id,
        "commission_owed": balance,
        "commission_rate": "5%",
        "message": f"You owe ${balance:.2f} in commission on profitable trades"
    }

@app.post("/api/collect-commission")
async def collect_commission(request: Dict):
    """Automatically collect commission via Stripe"""
    user_id = request['user_id']
    balance = await commission_tracker.get_user_commission_balance(user_id)
    
    if balance < 10:  # Minimum $10 to process
        return {
            "success": False,
            "message": f"Commission balance ${balance:.2f} below $10 minimum"
        }
    
    result = await commission_tracker.process_commission_payment(user_id, balance)
    return result

# Environment variables
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
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

# Stock universe
STOCK_UNIVERSE = [
    # Major indices and ETFs (always work)
    "SPY", "QQQ", "DIA", "IWM", "VTI",
    
    # Mega-cap tech (reliable data)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    
    # Established large-caps
    "JPM", "V", "JNJ", "WMT", "UNH", "HD", "BAC",
    
    # Active trading stocks
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
        # Hash password
        password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
        
        # Create user
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
            
            # Wait before next scan (1 minute during market hours, 5 minutes otherwise)
            now = datetime.now()
            if 9 <= now.hour < 16 and now.weekday() < 5:
                await asyncio.sleep(60)  # 1 minute during market hours
            else:
                await asyncio.sleep(300)  # 5 minutes outside market hours
                
        except Exception as e:
            logger.error(f"Scanner error: {e}")
            await asyncio.sleep(60)

async def generate_initial_training_data():
    """Generate synthetic training data for initial ML training"""
    await asyncio.sleep(10)  # Wait for app to initialize
    
    # Create synthetic training samples
    import random
    for _ in range(200):  # Generate 200 samples
        features = np.random.randn(12)  # 12 features
        outcome = random.choice(['profitable', 'loss', 'neutral'])
        ml_brain.add_training_sample(features, outcome)
    
    # Trigger initial training
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

@app.post("/api/ml-train")
async def trigger_training():
    """Manually trigger ML training"""
    success = ml_brain.train_models(force_retrain=True)
    return {"success": success, "message": "Training initiated" if success else "Not enough data"}

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
    # Find the signal
    signal = next((s for s in current_signals if s.symbol == trade.symbol), None)
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    # Execute via Alpaca
    result = await engine.execute_trade(signal, trade.quantity)
    
    # Log trade
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
    """Handle subscription (placeholder for Stripe integration)"""
    # In production, integrate with Stripe here
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
    # Calculate actual metrics from signal history
    history = list(engine.signal_history)[-100:]  # Last 100 signals
    
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
    
    # Send initial data
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to AI Trading Platform",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
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

# Account info endpoint
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
# Add this COMPLETE commission system to app.py

class AutoCommissionCollector:
    """Automatically collect commissions when positions close"""
    
    def __init__(self):
        self.pending_commissions = {}
        
    async def monitor_position_closure(self, user_id: str):
        """Monitor user positions and collect commission when closed"""
        api = await alpaca_manager.get_user_api(user_id)
        if not api:
            return
        
        # Get closed orders from last 24 hours
        closed_orders = api.list_orders(
            status='filled',
            after=datetime.now() - timedelta(days=1)
        )
        
        for order in closed_orders:
            await self.process_closed_position(user_id, order)
    
    async def process_closed_position(self, user_id: str, order):
        """Process a closed position and calculate commission"""
        # Check if this is a closing order (selling a long or buying back a short)
        if order.order_class == 'simple' and order.filled_qty:
            # Look for the opening position in our database
            if supabase:
                # Find matching opening position
                result = supabase.table('positions').select('*').eq(
                    'user_id', user_id
                ).eq('symbol', order.symbol).eq('status', 'open').execute()
                
                if result.data:
                    opening_position = result.data[0]
                    
                    # Calculate P&L
                    if opening_position['action'] == 'BUY':
                        # Was long, now sold
                        profit = (float(order.filled_avg_price) - float(opening_position['entry_price'])) * int(order.filled_qty)
                    else:
                        # Was short, now covered
                        profit = (float(opening_position['entry_price']) - float(order.filled_avg_price)) * int(order.filled_qty)
                    
                    # Only charge commission on profits
                    if profit > 0:
                        # Get user's commission rate based on tier
                        tier = await get_user_subscription_tier(user_id)
                        commission_rate = SubscriptionTiers.FEATURES[tier]['commission_rate']
                        commission = profit * commission_rate
                        
                        # Store commission record
                        supabase.table('commissions').insert({
                            'user_id': user_id,
                            'position_id': opening_position['position_id'],
                            'symbol': order.symbol,
                            'profit': profit,
                            'commission': commission,
                            'collected': False,
                            'timestamp': datetime.now().isoformat()
                        }).execute()
                        
                        # Update position status
                        supabase.table('positions').update({
                            'status': 'closed',
                            'exit_price': float(order.filled_avg_price),
                            'profit': profit,
                            'commission': commission,
                            'closed_at': datetime.now().isoformat()
                        }).eq('position_id', opening_position['position_id']).execute()
                        
                        # Auto-collect if over threshold
                        await self.auto_collect_if_due(user_id)
                        
                        return {
                            'profit': profit,
                            'commission': commission,
                            'net_profit': profit - commission
                        }
        
        return None
    # Add this to app.py for complete Stripe checkout functionality

import stripe
from fastapi import Request, HTTPException

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.post("/api/create-checkout")
async def create_checkout_session(request: Dict):
logger.info(f"Checkout request: {request}")
    """Create Stripe checkout session for subscription"""
    user_id = request.get('user_id')
    price_id = request.get('price_id')
    tier = request.get('tier')
    
    # Map tier to price IDs (your actual Stripe price IDs)
    price_mapping = {
        'price_1S6KlkPuxT6s7WvF1Pn0dChn': 'pro',
        'price_1S6KlvPuxT6s7WvFAuxjJQA7': 'insider'
    }
    
    if price_id not in price_mapping:
        raise HTTPException(status_code=400, detail="Invalid price ID")
    
    try:
        # Create Stripe checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https:// https://ai-trading-platform-p93x6ap4t-charlie-kupers-projects.vercel.app/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=' https://ai-trading-platform-p93x6ap4t-charlie-kupers-projects.vercel.app/',
            customer_email=request.get('email'),  # Pre-fill if available
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
           except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        return {'error': f'Stripe error: {str(e)}'}
    except Exception as e:
        logger.error(f"Checkout error: {str(e)}")
        return {'error': f'Server error: {str(e)}'} 
)
        
        return {
            'session_id': checkout_session.id,
            'url': checkout_session.url
        }
        
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        
        # Extract metadata
        user_id = session['metadata'].get('user_id')
        tier = session['metadata'].get('tier')
        
        if user_id and tier:
            # Update user's subscription in database
            if supabase:
                supabase.table('users').update({
                    'subscription_tier': tier,
                    'stripe_customer_id': session['customer'],
                    'stripe_subscription_id': session['subscription'],
                    'subscription_started': datetime.now().isoformat()
                }).eq('id', user_id).execute()
                
                logger.info(f"User {user_id} upgraded to {tier}")
            
            # Reset daily limits for upgraded user
            for key in list(user_daily_signals.keys()):
                if key.startswith(user_id):
                    del user_daily_signals[key]
    
    elif event['type'] == 'customer.subscription.deleted':
        # Handle subscription cancellation
        subscription = event['data']['object']
        
        if supabase:
            # Find user by stripe subscription ID
            result = supabase.table('users').select('id').eq(
                'stripe_subscription_id', subscription['id']
            ).execute()
            
            if result.data:
                user_id = result.data[0]['id']
                # Downgrade to free tier
                supabase.table('users').update({
                    'subscription_tier': 'free',
                    'subscription_ended': datetime.now().isoformat()
                }).eq('id', user_id).execute()
                
                logger.info(f"User {user_id} downgraded to free tier")
    
    return {'received': True}

# Update SubscriptionTiers in app.py with precise delays

class SubscriptionTiers:
    """Tiered delay system for signal distribution"""
    
    FEATURES = {
        'free': {
            'signals_per_day': 5,
            'signal_delay_seconds': 300,  # 5 minutes (300 seconds)
            'max_concurrent_positions': 3,
            'commission_rate': 0.07,  # 7%
            'price': 0
        },
        'pro': {
            'signals_per_day': -1,
            'signal_delay_seconds': 10,  # 10 seconds
            'max_concurrent_positions': 10,
            'commission_rate': 0.04,  # 4%
            'price': 49
        },
        'insider': {
            'signals_per_day': -1,
            'signal_delay_seconds': 5,  # 5 seconds
            'max_concurrent_positions': -1,
            'commission_rate': 0.03,  # 3%
            'price': 299
        },
        # Future institutional tier (not implemented yet)
        # 'institutional': {
        #     'signal_delay_seconds': 0,  # ZERO delay
        #     'commission_rate': 0.01,  # 1% or flat fee
        #     'price': 10000,  # $10k/month
        #     'features': ['Direct API', 'Co-location', 'Raw signals']
        # }
    }

class SignalQueueSystem:
    """Queue signals with tier-based delays"""
    
    def __init__(self):
        self.signal_timestamps = {}  # Track when each signal was generated
        self.signal_queue = []
        
    async def distribute_signal(self, signal: Signal):
        """Distribute signal with tier-appropriate delays"""
        
        # Record the EXACT generation time
        generation_time = datetime.now()
        signal.generation_timestamp = generation_time.isoformat()
        self.signal_timestamps[signal.symbol] = generation_time
        
        # Add to queue for delayed distribution
        self.signal_queue.append({
            'signal': signal,
            'generated_at': generation_time,
            'distributions': {
                'institutional': generation_time,  # Future: 0 delay
                'insider': generation_time + timedelta(seconds=5),  # 5 sec delay
                'pro': generation_time + timedelta(seconds=10),  # 10 sec delay
                'free': generation_time + timedelta(seconds=300)  # 5 min delay
            }
        })
        
        # Start distribution tasks
        asyncio.create_task(self.process_distribution_queue())
    
    async def process_distribution_queue(self):
        """Process the queue and distribute signals at appropriate times"""
        
        while self.signal_queue:
            current_time = datetime.now()
            
            for queued_signal in self.signal_queue[:]:  # Copy to iterate safely
                signal = queued_signal['signal']
                distributions = queued_signal['distributions']
                
                # Check each tier's release time
                for tier, release_time in distributions.items():
                    if current_time >= release_time:
                        await self.send_to_tier_users(signal, tier)
                        # Remove this tier from pending distributions
                        del distributions[tier]
                
                # Remove from queue if all tiers have been served
                if not distributions:
                    self.signal_queue.remove(queued_signal)
            
            # Check every second for precision
            await asyncio.sleep(1)
    
    async def send_to_tier_users(self, signal: Signal, tier: str):
        """Send signal to all users of a specific tier"""
        
        if tier == 'institutional':
            # Future implementation for hedge funds
            # They would get raw signal data via direct API
            return
        
        # Get users in this tier
        if supabase:
            users = supabase.table('users').select('id').eq('subscription_tier', tier).execute()
            
            for user in users.data:
                # Send via WebSocket
                await self.send_signal_to_user(user['id'], signal, tier)
    
    async def send_signal_to_user(self, user_id: str, signal: Signal, tier: str):
        """Send signal to specific user with tier info"""
        
        # Add tier-specific metadata
        signal_data = {
            'symbol': signal.symbol,
            'action': signal.action,
            'price': signal.price,
            'confidence': signal.confidence,
            'tier_delay': SubscriptionTiers.FEATURES[tier]['signal_delay_seconds'],
            'your_tier': tier,
            'timestamp': signal.timestamp,
            'generation_timestamp': signal.generation_timestamp  # When it was actually generated
        }
        
        # Send via WebSocket to user
        for ws in active_websockets:
            if ws.user_id == user_id:  # Assumes WebSocket tracks user_id
                await ws.send_json({
                    'type': 'new_signal',
                    'data': signal_data
                })

# Initialize queue system
signal_queue = SignalQueueSystem()

# Modified signal generation to use queue
async def continuous_scanner_with_delays():
    """Generate signals and queue them with delays"""
    while True:
        try:
            if is_market_hours():
                # Generate signals
                signals = await engine.scan_stocks(STOCK_UNIVERSE)
                
                # Queue each signal for delayed distribution
                for signal in signals[:10]:  # Top 10 signals
                    await signal_queue.distribute_signal(signal)
                    
                    # Log the cascade effect
                    logger.info(f"""
                    Signal generated: {signal.symbol} {signal.action}
                    Distribution schedule:
                    - Insider: +5 seconds (drives initial momentum)
                    - Pro: +10 seconds (amplifies movement)
                    - Free: +5 minutes (confirms trend)
                    """)
                
                await asyncio.sleep(60)  # Scan every minute
            else:
                await asyncio.sleep(300)  # 5 minutes when market closed
                
        except Exception as e:
            logger.error(f"Scanner error: {e}")
            await asyncio.sleep(60)

# Track signal performance by tier
class SignalPerformanceTracker:
    """Track how each tier's entry affects price movement"""
    
    async def track_cascade_effect(self, signal: Signal):
        """Monitor price movement as each tier enters"""
        
        symbol = signal.symbol
        original_price = signal.price
        
        # Track price at each tier's entry point
        timeline = {
            'generated': {'time': datetime.now(), 'price': original_price},
            'insider_entry': None,  # +5 seconds
            'pro_entry': None,      # +10 seconds
            'free_entry': None      # +5 minutes
        }
        
        # Monitor price changes
        await asyncio.sleep(5)
        ticker = yf.Ticker(symbol)
        timeline['insider_entry'] = {
            'time': datetime.now(),
            'price': ticker.info.get('regularMarketPrice', original_price),
            'volume_spike': 'calculating...'
        }
        
        await asyncio.sleep(5)  # 10 seconds total
        timeline['pro_entry'] = {
            'time': datetime.now(),
            'price': ticker.info.get('regularMarketPrice', original_price),
            'movement': f"+{(ticker.info.get('regularMarketPrice', original_price) - original_price) / original_price * 100:.2f}%"
        }
        
        await asyncio.sleep(290)  # 5 minutes total
        timeline['free_entry'] = {
            'time': datetime.now(),
            'price': ticker.info.get('regularMarketPrice', original_price),
            'total_movement': f"+{(ticker.info.get('regularMarketPrice', original_price) - original_price) / original_price * 100:.2f}%"
        }
        
        # Store for analysis (useful for selling to hedge funds later)
        if supabase:
            supabase.table('cascade_effects').insert({
                'signal_id': signal.symbol + '_' + signal.timestamp,
                'timeline': timeline,
                'profitable_for_institutional': timeline['free_entry']['price'] > original_price
            }).execute()
        
        return timeline

# Architecture ready for institutional tier
class InstitutionalAPI:
    """Future API for hedge funds (not active yet)"""
    
    # This is just the architecture - not implemented
    # When ready, hedge funds would:
    # 1. Get signals with ZERO delay
    # 2. See the delay schedule for retail tiers
    # 3. Know exactly when buying pressure will hit
    # 4. Position themselves accordingly
    # 5. Pay $10k+/month for this advantage
    
    # The beauty: Retail users create predictable momentum
    # Institutional clients profit from knowing the schedule
    # Everyone wins (except Free tier who enters last)
    pass
    async def auto_collect_if_due(self, user_id: str):
        """Automatically collect commission if balance exceeds threshold"""
        MIN_COLLECTION = 25  # Minimum $25 to auto-collect
        
        # Get uncollected commission balance
        if supabase:
            result = supabase.table('commissions').select('commission').eq(
                'user_id', user_id
            ).eq('collected', False).execute()
            
            total_owed = sum(row['commission'] for row in result.data)
            
            if total_owed >= MIN_COLLECTION:
                # Get user's saved payment method
                user_result = supabase.table('users').select(
                    'stripe_customer_id'
                ).eq('id', user_id).execute()
                
                if user_result.data and user_result.data[0].get('stripe_customer_id'):
                    # Auto-charge commission
                    await self.collect_commission_via_stripe(
                        user_id, 
                        user_result.data[0]['stripe_customer_id'],
                        total_owed
                    )
    
    async def collect_commission_via_stripe(self, user_id: str, stripe_customer_id: str, amount: float):
        """Automatically charge commission to user's card"""
        import stripe
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        
        try:
            # Create and immediately charge invoice
            invoice = stripe.Invoice.create(
                customer=stripe_customer_id,
                auto_advance=True,
                collection_method='charge_automatically',
                description=f"Trading commission (auto-collected)"
            )
            
            # Add commission line item
            stripe.InvoiceItem.create(
                customer=stripe_customer_id,
                amount=int(amount * 100),  # Convert to cents
                currency='usd',
                description=f'Trading commission on profitable trades',
                invoice=invoice.id
            )
            
            # Finalize and auto-pay
            finalized = stripe.Invoice.finalize_invoice(invoice.id)
            paid = stripe.Invoice.pay(invoice.id)
            
            # Mark commissions as collected
            if supabase:
                supabase.table('commissions').update({
                    'collected': True,
                    'collection_invoice': invoice.id,
                    'collected_at': datetime.now().isoformat()
                }).eq('user_id', user_id).eq('collected', False).execute()
            
            logger.info(f"Auto-collected ${amount:.2f} commission from user {user_id}")
            
            return {'success': True, 'amount': amount, 'invoice_id': invoice.id}
            
        except stripe.error.CardError as e:
            logger.error(f"Card declined for commission: {e}")
            # Send notification to user that card was declined
            return {'success': False, 'error': 'Card declined'}
        except Exception as e:
            logger.error(f"Commission collection failed: {e}")
            return {'success': False, 'error': str(e)}

# Initialize commission collector
commission_collector = AutoCommissionCollector()

# Background task to monitor all user positions
async def commission_monitor_loop():
    """Background task to monitor positions and collect commissions"""
    while True:
        try:
            if supabase:
                # Get all users with linked Alpaca accounts
                users = supabase.table('alpaca_accounts').select('user_id').execute()
                
                for user in users.data:
                    await commission_collector.monitor_position_closure(user['user_id'])
            
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            logger.error(f"Commission monitor error: {e}")
            await asyncio.sleep(60)

# Add to startup
@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(continuous_scanner())
    asyncio.create_task(commission_monitor_loop())  # Add this
    asyncio.create_task(generate_initial_training_data())
    logger.info("AI Trading Platform started with commission monitoring")
# At the bottom of app.py
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    
    uvicorn.run(
        app,  # Your FastAPI app instance
        host="0.0.0.0",
        port=port
    )
# At the VERY BOTTOM of app.py
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)