"""
AI Edge Trading Platform - Enhanced Testing Version
Integrates multiple strategies that work together, not replace each other
"""

import os
import asyncio
import json
import csv
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI imports
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# External libraries
import yfinance as yf
import pytz
from collections import deque

# Import ML brain for learning
ML_ENABLED = False
ml_brain = None

try:
    from ml_brain import SimpleMLBrain
    ml_brain = SimpleMLBrain()  # Create instance explicitly
    ML_ENABLED = True
    logger.info("Simple ML Brain loaded successfully")
except ImportError as e:
    logger.warning(f"ML Brain import error: {e}")
except Exception as e:
    logger.warning(f"ML Brain initialization error: {e}")

# Initialize FastAPI
app = FastAPI(
    title="Photon AI Edge Trading Platform - Enhanced",
    description="Multi-strategy AI trading with adaptive intelligence",
    version="2.1.0"
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
# CONFIGURATION FROM ENVIRONMENT
# ============================================

# Get configuration from environment
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.45"))  # Lower for learning
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PERCENTAGE", "2.0"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "20"))
DEFAULT_POSITION_SIZE = int(os.getenv("DEFAULT_POSITION_SIZE", "5000"))
SIGNAL_INTERVAL = int(os.getenv("SIGNAL_INTERVAL", "60"))
AUTO_TRADING_ENABLED = os.getenv("AUTO_TRADING_DEFAULT_ENABLED", "true").lower() == "true"
AUTO_TRADE_MIN_CONF = float(os.getenv("AUTO_TRADE_MIN_CONFIDENCE", "0.50"))  # Lower for learning

# Alpaca configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
PAPER_TRADING = os.getenv("PAPER_TRADING_DEFAULT", "true").lower() == "true"

# Stock universe from environment
STOCK_UNIVERSE = os.getenv("STOCK_UNIVERSE", "SPY,QQQ,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD").split(",")

# Platform info
PLATFORM_NAME = os.getenv("PLATFORM_NAME", "Photon AI Trading Platform")
PLATFORM_VERSION = os.getenv("PLATFORM_VERSION", "2.1.0")

# Trading zones with strategy preferences
MARKET_ZONES = {
    "pre_market": {
        "start": "09:00", "end": "09:30", 
        "strategies": ["news_catalyst", "gap_trading"],
        "multiplier": 1.2,
        "threshold": 0.35  # Lowered from default
    },
    "opening_bell": {
        "start": "09:30", "end": "10:00",
        "strategies": ["opening_range", "momentum", "volatility_breakout"],
        "multiplier": 1.8,
        "threshold": 0.25  # Lowered from 0.3
    },
    "morning_trend": {
        "start": "10:00", "end": "11:30",
        "strategies": ["trend_following", "vwap_bounce", "momentum"],
        "multiplier": 1.3,
        "threshold": 0.30  # Lowered from 0.35
    },
    "lunch_hour": {
        "start": "11:30", "end": "13:30",
        "strategies": ["mean_reversion", "range_trading", "scalping"],
        "multiplier": 1.1,
        "threshold": 0.35  # Lowered from 0.4
    },
    "afternoon_trend": {
        "start": "13:30", "end": "15:00",
        "strategies": ["trend_following", "sector_rotation", "pairs_trading"],
        "multiplier": 1.2,
        "threshold": 0.33  # Lowered from 0.38
    },
    "power_hour": {
        "start": "15:00", "end": "15:50",
        "strategies": ["momentum", "breakout", "trend_following"],
        "multiplier": 1.5,
        "threshold": 0.28  # Lowered from 0.32
    },
    "closing_cross": {
        "start": "15:50", "end": "16:00",
        "strategies": ["closing_reversal", "momentum", "scalping"],
        "multiplier": 2.0,
        "threshold": 0.20  # Lowered from 0.25
    }
}

def get_current_market_zone() -> Tuple[str, Dict]:
    """Get current market zone and its configuration"""
    try:
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        current_time = now.strftime("%H:%M")
        
        for zone_name, config in MARKET_ZONES.items():
            if config["start"] <= current_time <= config["end"]:
                logger.info(f"📍 Active Zone: {zone_name.upper()} - Strategies: {config.get('strategies', [])}")
                return zone_name, config
        
        return "after_hours", {"multiplier": 0.5, "threshold": 0.75, "strategies": []}
    except:
        return "unknown", {"multiplier": 1.0, "threshold": 0.45, "strategies": []}

# Expanded universe for testing
STOCK_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"
]

# Risk configuration
RISK_CONFIG = {
    "max_portfolio_risk": 0.20,
    "max_position_risk": 0.02,
    "correlation_threshold": 0.7,
    "var_confidence": 0.95,
    "max_leverage": 1.5,
    "stop_loss_multiplier": 1.5,
    "take_profit_multiplier": 2.5,
    "max_concurrent_positions": 8,
    "min_trade_interval": 30,  # seconds between trades on same symbol
}

# ============================================
# ENHANCED SIGNAL WITH MULTI-STRATEGY
# ============================================

@dataclass
class MultiStrategySignal:
    symbol: str
    primary_action: str  # Main action (BUY/SELL/HOLD)
    strategies_voting: Dict[str, str]  # Each strategy's vote
    combined_confidence: float
    price: float
    zone: str
    zone_multiplier: float
    technical_indicators: Dict
    risk_score: float
    expected_return: float
    stop_loss: float
    take_profit: float
    position_size: int
    timestamp: str
    execution_priority: int
    explanations: List[str]
    
    def to_dict(self):
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                d[key] = 0.0
        return d

# ============================================
# STRATEGY AGGREGATOR
# ============================================

class StrategyAggregator:
    """Combines multiple trading strategies intelligently"""
    
    def __init__(self):
        self.strategy_performance = {
            'opening_range': deque(maxlen=50),
            'mean_reversion': deque(maxlen=50),
            'trend_following': deque(maxlen=50),
            'momentum': deque(maxlen=50),
            'vwap_bounce': deque(maxlen=50),
            'range_trading': deque(maxlen=50),
            'scalping': deque(maxlen=50),
            'breakout': deque(maxlen=50),
        }
        self.last_trade_time = {}
        
    def can_trade_symbol(self, symbol: str) -> bool:
        """Check if enough time has passed since last trade"""
        if symbol not in self.last_trade_time:
            return True
        return time.time() - self.last_trade_time[symbol] > RISK_CONFIG["min_trade_interval"]
    
    def mark_trade(self, symbol: str):
        """Mark that a trade was executed"""
        self.last_trade_time[symbol] = time.time()
    
    async def evaluate_strategies(self, symbol: str, data: pd.DataFrame, zone_config: Dict) -> Dict:
        """Run multiple strategies and aggregate results"""
        results = {}
        
        # Run each strategy
        if "opening_range" in zone_config.get("strategies", []):
            results["opening_range"] = await self.opening_range_strategy(symbol, data)
        
        if "mean_reversion" in zone_config.get("strategies", []):
            results["mean_reversion"] = await self.mean_reversion_strategy(symbol, data)
        
        if "trend_following" in zone_config.get("strategies", []):
            results["trend_following"] = await self.trend_following_strategy(symbol, data)
        
        if "momentum" in zone_config.get("strategies", []):
            results["momentum"] = await self.momentum_strategy(symbol, data)
        
        if "vwap_bounce" in zone_config.get("strategies", []):
            results["vwap_bounce"] = await self.vwap_strategy(symbol, data)
        
        if "range_trading" in zone_config.get("strategies", []):
            results["range_trading"] = await self.range_strategy(symbol, data)
        
        if "scalping" in zone_config.get("strategies", []):
            results["scalping"] = await self.scalping_strategy(symbol, data)
        
        if "breakout" in zone_config.get("strategies", []):
            results["breakout"] = await self.breakout_strategy(symbol, data)
        
        return results
    
    async def opening_range_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Opening range breakout"""
        if len(data) < 30:
            return {"action": "HOLD", "confidence": 0}
        
        try:
            # Get first 30 minutes of trading
            first_30min = data.iloc[:30] if len(data) > 30 else data
            opening_high = first_30min['High'].max()
            opening_low = first_30min['Low'].min()
            current = data['Close'].iloc[-1]
            
            if current > opening_high * 1.001:
                return {"action": "BUY", "confidence": 0.75, "reason": "ORB breakout"}
            elif current < opening_low * 0.999:
                return {"action": "SELL", "confidence": 0.75, "reason": "ORB breakdown"}
        except:
            pass
        
        return {"action": "HOLD", "confidence": 0}
    
    async def mean_reversion_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Mean reversion based on Bollinger Bands"""
        if len(data) < 20:
            return {"action": "HOLD", "confidence": 0}
        
        try:
            sma20 = data['Close'].rolling(20).mean().iloc[-1]
            std20 = data['Close'].rolling(20).std().iloc[-1]
            current = data['Close'].iloc[-1]
            
            upper = sma20 + (2 * std20)
            lower = sma20 - (2 * std20)
            
            # Calculate RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain.iloc[-1] / (loss.iloc[-1] + 0.001)
            rsi = 100 - (100 / (1 + rs))
            
            # MORE AGGRESSIVE THRESHOLDS
            if current < lower and rsi < 40:  # Was 35
                return {"action": "BUY", "confidence": 0.75, "reason": "Oversold bounce"}
            elif current > upper and rsi > 60:  # Was 65
                return {"action": "SELL", "confidence": 0.75, "reason": "Overbought reversal"}
            # Add medium confidence signals
            elif current < sma20 and rsi < 45:
                return {"action": "BUY", "confidence": 0.60, "reason": "Below average"}
            elif current > sma20 and rsi > 55:
                return {"action": "SELL", "confidence": 0.60, "reason": "Above average"}
        except:
            pass
        
        return {"action": "HOLD", "confidence": 0}
    
    async def trend_following_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Follow the trend with moving averages"""
        if len(data) < 50:
            return {"action": "HOLD", "confidence": 0}
        
        try:
            sma10 = data['Close'].rolling(10).mean().iloc[-1]
            sma20 = data['Close'].rolling(20).mean().iloc[-1]
            sma50 = data['Close'].rolling(50).mean().iloc[-1]
            current = data['Close'].iloc[-1]
            
            # MACD
            ema12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema26 = data['Close'].ewm(span=26).mean().iloc[-1]
            macd = ema12 - ema26
            
            # Strong signals
            if sma10 > sma20 > sma50 and macd > 0:
                momentum = (current - data['Close'].iloc[-10]) / data['Close'].iloc[-10]
                conf = min(0.65 + momentum * 5, 0.85)  # Higher base
                return {"action": "BUY", "confidence": conf, "reason": "Uptrend confirmed"}
            elif sma10 < sma20 < sma50 and macd < 0:
                momentum = (data['Close'].iloc[-10] - current) / data['Close'].iloc[-10]
                conf = min(0.65 + momentum * 5, 0.85)
                return {"action": "SELL", "confidence": conf, "reason": "Downtrend confirmed"}
            
            # Medium signals - simple MA crossovers
            elif sma10 > sma20 and current > sma20:
                return {"action": "BUY", "confidence": 0.60, "reason": "Bullish MA cross"}
            elif sma10 < sma20 and current < sma20:
                return {"action": "SELL", "confidence": 0.60, "reason": "Bearish MA cross"}
            
            # Weak signals - just price vs moving average
            elif current > sma50:
                return {"action": "BUY", "confidence": 0.50, "reason": "Above long MA"}
            elif current < sma50:
                return {"action": "SELL", "confidence": 0.50, "reason": "Below long MA"}
        except:
            pass
        
        return {"action": "HOLD", "confidence": 0}
    
    async def momentum_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Momentum continuation"""
        if len(data) < 20:
            return {"action": "HOLD", "confidence": 0}
        
        try:
            roc_5 = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
            roc_10 = (data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].iloc[-10]
            
            volume_surge = data['Volume'].iloc[-5:].mean() / data['Volume'].mean()
            
            # MORE AGGRESSIVE THRESHOLDS
            if roc_5 > 0.001 and roc_10 > 0.002 and volume_surge > 1.1:  # Lowered all thresholds
                conf = min(0.65 + roc_5 * 20, 0.80)  # Higher base confidence
                return {"action": "BUY", "confidence": conf, "reason": "Strong momentum"}
            elif roc_5 < -0.001 and roc_10 < -0.002 and volume_surge > 1.1:  # Lowered
                conf = min(0.65 + abs(roc_5) * 20, 0.80)
                return {"action": "SELL", "confidence": conf, "reason": "Negative momentum"}
            # Add medium momentum signals
            elif roc_5 > 0.0005:  # Very small positive momentum
                return {"action": "BUY", "confidence": 0.55, "reason": "Mild upward momentum"}
            elif roc_5 < -0.0005:  # Very small negative momentum
                return {"action": "SELL", "confidence": 0.55, "reason": "Mild downward momentum"}
        except:
            pass
        
        return {"action": "HOLD", "confidence": 0}
    
    async def vwap_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """VWAP bounce trading"""
        if len(data) < 20:
            return {"action": "HOLD", "confidence": 0}
        
        try:
            typical = (data['High'] + data['Low'] + data['Close']) / 3
            vwap = (typical * data['Volume']).cumsum() / data['Volume'].cumsum()
            current = data['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            distance = (current - current_vwap) / current_vwap
            
            if abs(distance) < 0.002:
                # At VWAP, check direction
                if data['Close'].iloc[-1] > data['Close'].iloc[-5]:
                    return {"action": "BUY", "confidence": 0.65, "reason": "VWAP support"}
                else:
                    return {"action": "SELL", "confidence": 0.65, "reason": "VWAP resistance"}
        except:
            pass
        
        return {"action": "HOLD", "confidence": 0}
    
    async def range_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Range trading for sideways markets"""
        if len(data) < 50:
            return {"action": "HOLD", "confidence": 0}
        
        try:
            recent = data.iloc[-50:]
            resistance = recent['High'].rolling(20).max().iloc[-1]
            support = recent['Low'].rolling(20).min().iloc[-1]
            current = data['Close'].iloc[-1]
            
            range_size = (resistance - support) / support
            if 0.01 < range_size < 0.03:  # 1-3% range
                position = (current - support) / (resistance - support)
                
                if position < 0.2:
                    return {"action": "BUY", "confidence": 0.70, "reason": "Near support"}
                elif position > 0.8:
                    return {"action": "SELL", "confidence": 0.70, "reason": "Near resistance"}
        except:
            pass
        
        return {"action": "HOLD", "confidence": 0}
    
    async def scalping_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Quick scalps on micro movements"""
        if len(data) < 15:
            return {"action": "HOLD", "confidence": 0}
        
        try:
            volatility = data['Close'].pct_change().iloc[-15:].std()
            if volatility < 0.003:  # Low volatility
                recent_high = data['High'].iloc[-15:].max()
                recent_low = data['Low'].iloc[-15:].min()
                current = data['Close'].iloc[-1]
                
                if current > recent_high:
                    return {"action": "BUY", "confidence": 0.60, "reason": "Micro breakout"}
                elif current < recent_low:
                    return {"action": "SELL", "confidence": 0.60, "reason": "Micro breakdown"}
        except:
            pass
        
        return {"action": "HOLD", "confidence": 0}
    
    async def breakout_strategy(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Volatility breakout"""
        if len(data) < 20:
            return {"action": "HOLD", "confidence": 0}
        
        try:
            atr = data['High'].sub(data['Low']).rolling(14).mean().iloc[-1]
            current = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2]
            
            if current > prev_close + (atr * 0.5):
                return {"action": "BUY", "confidence": 0.72, "reason": "Volatility breakout"}
            elif current < prev_close - (atr * 0.5):
                return {"action": "SELL", "confidence": 0.72, "reason": "Volatility breakdown"}
        except:
            pass
        
        return {"action": "HOLD", "confidence": 0}

# ============================================
# ENHANCED PORTFOLIO MANAGER
# ============================================

class EnhancedPortfolio:
    """SIMULATED PORTFOLIO - Testing AI performance with real data, no actual trades"""
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # Simulated positions only
        self.trades = deque(maxlen=1000)
        self.pending_orders = {}
        self.strategy_aggregator = StrategyAggregator()
        self.last_trade_timestamp = time.time()  # Track last trade time
        self.trades_today_count = 0  # Track daily trades
        self.is_simulated = True  # Flag for simulated trading
        
        # Initialize trade logger
        self.init_trade_logger()
        logger.info("=" * 60)
        logger.info("SIMULATED TRADING MODE - No real money involved")
        logger.info("Testing AI signals with paper portfolio")
        logger.info("=" * 60)
        
    def init_trade_logger(self):
        """Initialize trade logging to CSV for analysis"""
        try:
            os.makedirs('trade_logs', exist_ok=True)
            self.trade_log_file = f'trade_logs/trades_{datetime.now().strftime("%Y%m%d")}.csv'
            
            # Create CSV header if file doesn't exist
            if not os.path.exists(self.trade_log_file):
                import csv
                with open(self.trade_log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'symbol', 'action', 'quantity', 'price',
                        'total_value', 'confidence', 'strategy', 'zone',
                        'cash_before', 'cash_after', 'portfolio_value',
                        'strategies_voting', 'ml_prediction', 'pnl'
                    ])
            logger.info(f"Trade logging initialized: {self.trade_log_file}")
        except Exception as e:
            logger.error(f"Failed to initialize trade logger: {e}")
            self.trade_log_file = None
    
    def log_trade(self, trade_data: Dict):
        """Log trade to CSV file"""
        if not self.trade_log_file:
            return
            
        try:
            import csv
            with open(self.trade_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade_data.get('timestamp', datetime.now().isoformat()),
                    trade_data.get('symbol', ''),
                    trade_data.get('action', ''),
                    trade_data.get('quantity', 0),
                    trade_data.get('price', 0),
                    trade_data.get('total_value', 0),
                    trade_data.get('confidence', 0),
                    trade_data.get('strategy', ''),
                    trade_data.get('zone', ''),
                    trade_data.get('cash_before', self.cash),
                    trade_data.get('cash_after', self.cash),
                    trade_data.get('portfolio_value', self.get_total_value()),
                    json.dumps(trade_data.get('strategies_voting', {})),
                    trade_data.get('ml_prediction', ''),
                    trade_data.get('pnl', 0)
                ])
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    def track_trade_for_learning(self, trade: Dict):
        """Track trade for ML learning"""
        if ML_ENABLED and ml_brain:
            # Simple tracking - just note the trade happened
            try:
                market_data = {
                    'current_price': trade.get('price', 0),
                    'day_high': trade.get('price', 0) * 1.02,  # Estimate
                    'day_low': trade.get('price', 0) * 0.98,   # Estimate
                    'volume_ratio': 1.0,
                    'rsi': 50,
                    'change_5min': 0
                }
                ml_brain.record_trade(market_data, trade.get('action', 'BUY'), None)
            except:
                pass
    
    def get_total_value(self) -> float:
        try:
            positions_value = sum(
                pos.get("qty", 0) * pos.get("current_price", pos.get("avg_price", 0))
                for pos in self.positions.values()
            )
            return self.cash + positions_value
        except:
            return self.cash
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        return len(self.positions) < RISK_CONFIG["max_concurrent_positions"]
    
    def execute_multi_strategy_trade(self, signal: MultiStrategySignal) -> Dict:
        """Execute trade based on multi-strategy signal"""
        try:
            # Check if we can trade this symbol
            if not self.strategy_aggregator.can_trade_symbol(signal.symbol):
                return {"success": False, "error": "Too soon to trade this symbol again"}
            
            # Check position limits
            if signal.primary_action == "BUY" and not self.can_open_position():
                return {"success": False, "error": "Max positions reached"}
            
            # Calculate quantity
            if signal.primary_action == "BUY":
                max_position_value = self.cash * RISK_CONFIG["max_position_risk"]
                quantity = int(max_position_value / signal.price) if signal.price > 0 else 0
                
                if quantity < 1:
                    return {"success": False, "error": "Insufficient funds"}
                
                cost = signal.price * quantity
                self.cash -= cost
                
                if signal.symbol not in self.positions:
                    self.positions[signal.symbol] = {
                        "qty": 0,
                        "avg_price": 0,
                        "current_price": signal.price
                    }
                
                pos = self.positions[signal.symbol]
                total_cost = (pos["qty"] * pos["avg_price"]) + cost
                pos["qty"] += quantity
                pos["avg_price"] = total_cost / pos["qty"] if pos["qty"] > 0 else signal.price
                pos["current_price"] = signal.price
                pos["stop_loss"] = signal.stop_loss
                pos["take_profit"] = signal.take_profit
                pos["entry_zone"] = signal.zone
                
            else:  # SELL
                if signal.symbol not in self.positions:
                    return {"success": False, "error": "No position to sell"}
                
                quantity = self.positions[signal.symbol]["qty"]
                if quantity < 1:
                    return {"success": False, "error": "No shares to sell"}
                
                self.cash += signal.price * quantity
                del self.positions[signal.symbol]
            
            # Mark trade time
            self.strategy_aggregator.mark_trade(signal.symbol)
            self.last_trade_timestamp = time.time()  # Update last trade time
            self.trades_today_count += 1
            
            # Record trade
            trade = {
                "id": f"trade_{datetime.now().timestamp()}",
                "symbol": signal.symbol,
                "action": signal.primary_action,
                "quantity": quantity,
                "price": signal.price,
                "timestamp": datetime.now().isoformat(),
                "zone": signal.zone,
                "strategies": signal.strategies_voting,
                "confidence": signal.combined_confidence
            }
            
            self.trades.append(trade)
            
            # Log trade to CSV
            self.log_trade({
                **trade,
                'total_value': quantity * signal.price,
                'strategy': list(signal.strategies_voting.keys())[0] if signal.strategies_voting else 'unknown',
                'cash_before': self.cash + (quantity * signal.price if signal.primary_action == "BUY" else 0),
                'cash_after': self.cash,
                'portfolio_value': self.get_total_value(),
                'strategies_voting': signal.strategies_voting,
                'ml_prediction': 'ML' if 'ml_brain' in signal.strategies_voting else 'None'
            })
            
            logger.info(f"Multi-strategy trade: {signal.primary_action} {quantity} {signal.symbol} @ ${signal.price:.2f}")
            
            return {"success": True, "trade": trade}
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {"success": False, "error": str(e)}

# ============================================
# SIGNAL GENERATOR WITH MULTI-STRATEGY
# ============================================

class MultiStrategySignalGenerator:
    def __init__(self):
        self.portfolio = EnhancedPortfolio()
        self.signal_history = deque(maxlen=1000)
        
    async def generate_multi_strategy_signal(self, symbol: str) -> Optional[MultiStrategySignal]:
        """Generate signal using multiple strategies for current zone"""
        
        # Get current market zone
        zone_name, zone_config = get_current_market_zone()
        
        if not zone_config.get("strategies"):
            return None
        
        # Get market data
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty or len(hist) < 20:
                return None
        except:
            return None
        
        # Run strategies for this zone
        strategy_results = await self.portfolio.strategy_aggregator.evaluate_strategies(
            symbol, hist, zone_config
        )
        
        if not strategy_results:
            return None
        
        # Get ML prediction if available
        ml_vote = None
        ml_confidence = 0
        if ML_ENABLED and ml_brain:
            try:
                # Prepare simple market data for ML
                current_price = hist['Close'].iloc[-1]
                market_data = {
                    'current_price': current_price,
                    'day_high': hist['High'].max(),
                    'day_low': hist['Low'].min(),
                    'volume_ratio': hist['Volume'].iloc[-5:].mean() / hist['Volume'].mean() if len(hist) > 5 else 1,
                    'rsi': 50,  # Simple RSI placeholder
                    'change_5min': (current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] if len(hist) > 5 else 0
                }
                
                ml_prediction = ml_brain.predict(market_data)
                
                # Add ML vote only if confident enough
                if ml_prediction.get('ml_active') and ml_prediction.get('confidence', 0) > 0.6:
                    ml_vote = ml_prediction['action']
                    ml_confidence = ml_prediction['confidence']
                    
                    if ml_vote != 'NEUTRAL':
                        strategy_results['ml_brain'] = {
                            'action': 'BUY' if ml_vote == 'BUY' else 'SELL',
                            'confidence': ml_confidence,
                            'reason': ml_prediction.get('reason', 'ML prediction')
                        }
                        logger.info(f"ML: {symbol} {ml_vote} ({ml_confidence:.1%})")
                
                # Record this evaluation for learning
                ml_brain.record_trade(market_data, 'EVAL', None)
                    
            except Exception as e:
                logger.debug(f"ML prediction skipped: {e}")
        
        # Aggregate votes
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        total_confidence = 0
        explanations = []
        
        for strategy_name, result in strategy_results.items():
            if result["confidence"] > 0:
                if result["action"] == "BUY":
                    buy_votes += result["confidence"]
                    explanations.append(f"{strategy_name}: {result.get('reason', 'BUY')}")
                elif result["action"] == "SELL":
                    sell_votes += result["confidence"]
                    explanations.append(f"{strategy_name}: {result.get('reason', 'SELL')}")
                else:
                    hold_votes += result["confidence"]
                
                total_confidence += result["confidence"]
        
        # Determine primary action
        if buy_votes > sell_votes and buy_votes > hold_votes:
            primary_action = "BUY"
            combined_confidence = buy_votes / max(len(strategy_results), 1)
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            primary_action = "SELL"
            combined_confidence = sell_votes / max(len(strategy_results), 1)
        else:
            # If close, pick the stronger one
            if abs(buy_votes - sell_votes) < 0.1 and max(buy_votes, sell_votes) > 0:
                primary_action = "BUY" if buy_votes >= sell_votes else "SELL"
                combined_confidence = max(buy_votes, sell_votes) / max(len(strategy_results), 1)
            else:
                primary_action = "HOLD"
                combined_confidence = hold_votes / max(len(strategy_results), 1)
        
        # Apply zone multiplier
        zone_multiplier = zone_config.get("multiplier", 1.0)
        combined_confidence = min(0.95, combined_confidence * zone_multiplier)
        
        # Ensure minimum confidence if we have any votes
        if primary_action != "HOLD" and combined_confidence < 0.4:
            combined_confidence = 0.4  # Minimum 40% for any actionable signal
        
        # Don't generate HOLD signals
        if primary_action == "HOLD":
            return None
        
        # Calculate stop/target
        current_price = hist['Close'].iloc[-1]
        atr = hist['High'].sub(hist['Low']).rolling(14).mean().iloc[-1] if len(hist) > 14 else current_price * 0.02
        
        if primary_action == "BUY":
            stop_loss = current_price - (atr * RISK_CONFIG["stop_loss_multiplier"])
            take_profit = current_price + (atr * RISK_CONFIG["take_profit_multiplier"])
        else:
            stop_loss = current_price + (atr * RISK_CONFIG["stop_loss_multiplier"])
            take_profit = current_price - (atr * RISK_CONFIG["take_profit_multiplier"])
        
        # Calculate expected return
        expected_return = abs((take_profit - current_price) / current_price)
        
        # Create signal
        signal = MultiStrategySignal(
            symbol=symbol,
            primary_action=primary_action,
            strategies_voting=strategy_results,
            combined_confidence=combined_confidence,
            price=float(current_price),
            zone=zone_name,
            zone_multiplier=zone_multiplier,
            technical_indicators={},
            risk_score=min(atr / current_price * 10, 1.0) if current_price > 0 else 0.5,
            expected_return=expected_return,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            position_size=0,
            timestamp=datetime.now().isoformat(),
            execution_priority=int(combined_confidence * 10),
            explanations=explanations
        )
        
        self.signal_history.append(signal)
        
        return signal
    
    async def scan_universe(self) -> List[MultiStrategySignal]:
        """Scan all stocks for multi-strategy signals"""
        signals = []
        zone_name, zone_config = get_current_market_zone()
        
        # Skip if market closed
        if zone_name == "after_hours":
            return []
        
        for symbol in STOCK_UNIVERSE:
            try:
                signal = await self.generate_multi_strategy_signal(symbol)
                # Accept signals with reasonable confidence
                min_threshold = zone_config.get("threshold", MIN_CONFIDENCE) * 0.8  # 20% below zone threshold
                if signal and signal.combined_confidence >= min_threshold:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort by priority
        signals.sort(key=lambda x: x.execution_priority, reverse=True)
        
        # Return top signals based on zone
        max_signals = 15 if zone_config.get("multiplier", 1) >= 1.5 else 10
        return signals[:max_signals]

# ============================================
# POSITION MANAGER WITH ZONE AWARENESS
# ============================================

class ZoneAwareExitManager:
    def __init__(self, portfolio: EnhancedPortfolio):
        self.portfolio = portfolio
        
    async def check_exits(self) -> List[Dict]:
        """Check positions for zone-specific exit conditions"""
        exits = []
        zone_name, zone_config = get_current_market_zone()
        
        for symbol, position in list(self.portfolio.positions.items()):
            try:
                ticker = yf.Ticker(symbol)
                recent = ticker.history(period="1d", interval="1m")
                
                if recent.empty:
                    continue
                
                current_price = recent['Close'].iloc[-1]
                entry_price = position['avg_price']
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                
                exit_reason = None
                
                # Zone-specific exits
                if position.get("entry_zone") != zone_name:
                    # Exit if we've moved to a different zone
                    if pnl_percent > 0.5:
                        exit_reason = "ZONE_CHANGE_PROFIT"
                    elif pnl_percent < -1:
                        exit_reason = "ZONE_CHANGE_LOSS"
                
                # Stop loss and take profit
                if current_price <= position.get("stop_loss", 0):
                    exit_reason = "STOP_LOSS"
                elif current_price >= position.get("take_profit", float('inf')):
                    exit_reason = "TAKE_PROFIT"
                
                # Aggressive zone exits
                if zone_config.get("multiplier", 1) >= 1.5:
                    # Tighter exits during volatile zones
                    if pnl_percent <= -1:
                        exit_reason = "VOLATILE_STOP"
                    elif pnl_percent >= 2:
                        exit_reason = "VOLATILE_PROFIT"
                
                # End of day exit
                et = pytz.timezone('US/Eastern')
                current_time = datetime.now(et).strftime("%H:%M")
                if current_time >= "15:55":
                    exit_reason = "END_OF_DAY"
                
                if exit_reason:
                    # Create exit signal
                    exit_signal = MultiStrategySignal(
                        symbol=symbol,
                        primary_action="SELL",
                        strategies_voting={"exit_manager": {"action": "SELL", "confidence": 1.0}},
                        combined_confidence=1.0,
                        price=current_price,
                        zone=zone_name,
                        zone_multiplier=1.0,
                        technical_indicators={},
                        risk_score=0,
                        expected_return=0,
                        stop_loss=0,
                        take_profit=0,
                        position_size=position["qty"],
                        timestamp=datetime.now().isoformat(),
                        execution_priority=10,
                        explanations=[f"EXIT: {exit_reason} (P&L: {pnl_percent:.1f}%)"]
                    )
                    
                    result = self.portfolio.execute_multi_strategy_trade(exit_signal)
                    if result["success"]:
                        exits.append({
                            "symbol": symbol,
                            "reason": exit_reason,
                            "pnl_percent": pnl_percent,
                            "exit_price": current_price
                        })
                        logger.info(f"Exit: {symbol} - {exit_reason} @ ${current_price:.2f} (P&L: {pnl_percent:.1f}%)")
            
            except Exception as e:
                logger.error(f"Exit check error for {symbol}: {e}")
        
        return exits

# ============================================
# GLOBAL INSTANCES
# ============================================

signal_generator = MultiStrategySignalGenerator()
exit_manager = ZoneAwareExitManager(signal_generator.portfolio)
active_websockets = set()
current_signals = []
last_signal_time = time.time()

# ============================================
# API ENDPOINTS
# ============================================

def is_market_open() -> bool:
    """Check if US market is open"""
    try:
        et = pytz.timezone('US/Eastern')
        et_now = datetime.now(et)
        
        if et_now.weekday() >= 5:
            return False
        
        market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = et_now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= et_now <= market_close
    except:
        return False

@app.get("/")
async def root():
    zone_name, zone_config = get_current_market_zone()
    return {
        "platform": "Photon AI Trading - Simulated Portfolio",
        "version": "2.1.0",
        "status": "operational",
        "mode": "SIMULATED - Testing AI signals without real trades",
        "market_open": is_market_open(),
        "portfolio": {
            "type": "Simulated",
            "value": signal_generator.portfolio.get_total_value(),
            "initial": 100000,
            "is_real": False
        },
        "current_zone": {
            "name": zone_name,
            "strategies": zone_config.get("strategies", []),
            "multiplier": zone_config.get("multiplier", 1.0),
            "threshold": zone_config.get("threshold", 0.5)
        },
        "data_source": "Yahoo Finance (real market data)",
        "railway_app": "ai-signals.up.railway.app"
    }

@app.get("/api/signals")
async def get_signals():
    """Get current multi-strategy signals"""
    global current_signals, last_signal_time
    
    zone_name, zone_config = get_current_market_zone()
    
    # Refresh signals based on zone activity
    refresh_interval = 30 if zone_config.get("multiplier", 1) >= 1.5 else 60
    
    if time.time() - last_signal_time > refresh_interval or not current_signals:
        if is_market_open():
            current_signals = await signal_generator.scan_universe()
            last_signal_time = time.time()
    
    return {
        "signals": [s.to_dict() for s in current_signals],
        "count": len(current_signals),
        "zone": zone_name,
        "zone_config": zone_config,
        "portfolio": {
            "value": signal_generator.portfolio.get_total_value(),
            "cash": signal_generator.portfolio.cash,
            "positions": len(signal_generator.portfolio.positions),
            "max_positions": RISK_CONFIG["max_concurrent_positions"]
        }
    }

@app.post("/api/execute")
async def execute_trade(symbol: str, action: str = None):
    """Execute a multi-strategy trade"""
    try:
        # Find existing signal or generate new one
        signal = None
        for s in current_signals:
            if s.symbol == symbol:
                signal = s
                break
        
        if not signal:
            signal = await signal_generator.generate_multi_strategy_signal(symbol)
        
        if not signal:
            return {"success": False, "error": "Could not generate signal"}
        
        # Override action if specified
        if action:
            signal.primary_action = action
        
        result = signal_generator.portfolio.execute_multi_strategy_trade(signal)
        
        if result.get("success"):
            await broadcast_to_websockets({
                "type": "trade_executed",
                "trade": result["trade"],
                "portfolio_value": signal_generator.portfolio.get_total_value()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Execute error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/portfolio")
async def get_portfolio():
    """Get enhanced portfolio details"""
    try:
        portfolio = signal_generator.portfolio
        
        positions_detail = []
        for symbol, position in portfolio.positions.items():
            positions_detail.append({
                "symbol": symbol,
                "quantity": position.get("qty", 0),
                "avg_price": position.get("avg_price", 0),
                "current_price": position.get("current_price", 0),
                "entry_zone": position.get("entry_zone", "unknown"),
                "stop_loss": position.get("stop_loss", 0),
                "take_profit": position.get("take_profit", 0)
            })
        
        return {
            "total_value": portfolio.get_total_value(),
            "cash": portfolio.cash,
            "positions": positions_detail,
            "trades_today": len([t for t in portfolio.trades if t["timestamp"][:10] == datetime.now().date().isoformat()]),
            "strategy_performance": dict(portfolio.strategy_aggregator.strategy_performance)
        }
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        return {"error": str(e)}

@app.get("/api/ml-stats")
async def get_ml_stats():
    """Get ML brain statistics"""
    if not ML_ENABLED or not ml_brain:
        return {
            "enabled": False,
            "message": "ML brain not available"
        }
    
    stats = ml_brain.get_stats()
    return {
        "enabled": True,
        "is_trained": stats['is_trained'],
        "total_trades": stats['total_trades'],
        "trades_with_results": stats['trades_with_results'],
        "accuracy": round(stats['accuracy'] * 100, 1),
        "predictions_made": stats['predictions_made'],
        "model_type": stats['model_type'],
        "status": "Active" if stats['is_trained'] else f"Learning ({stats['trades_with_results']}/30)"
    }

@app.get("/mobile")
async def mobile_dashboard():
    """Mobile-friendly status page"""
    try:
        portfolio = signal_generator.portfolio
        zone_name, zone_config = get_current_market_zone()
        
        # Calculate daily P&L
        daily_pnl = portfolio.get_total_value() - portfolio.initial_capital
        daily_pnl_pct = (daily_pnl / portfolio.initial_capital) * 100
        
        # Get recent trades
        recent_trades = list(portfolio.trades)[-5:] if portfolio.trades else []
        
        return {
            "status": "🟢 Active" if is_market_open() else "🔴 Closed",
            "portfolio": {
                "value": f"${portfolio.get_total_value():,.0f}",
                "daily_pnl": f"{'+'if daily_pnl>=0 else ''}{daily_pnl:,.0f}",
                "daily_pct": f"{'+'if daily_pnl_pct>=0 else ''}{daily_pnl_pct:.1f}%",
                "cash": f"${portfolio.cash:,.0f}",
                "positions": len(portfolio.positions)
            },
            "trading": {
                "zone": zone_name.replace('_', ' ').title(),
                "multiplier": f"{zone_config.get('multiplier', 1)}x",
                "trades_today": len([t for t in portfolio.trades if t["timestamp"][:10] == datetime.now().date().isoformat()]),
                "last_trade": recent_trades[-1]["timestamp"][:16] if recent_trades else "None"
            },
            "ml": {
                "status": "Active" if ML_ENABLED and ml_brain and ml_brain.is_trained else "Learning",
                "accuracy": f"{ml_brain.get_stats()['accuracy']*100:.0f}%" if ML_ENABLED and ml_brain else "N/A"
            },
            "recent": [
                {
                    "time": t["timestamp"][11:16],
                    "symbol": t["symbol"],
                    "action": "🟢" if t["action"] == "BUY" else "🔴",
                    "price": f"${t['price']:.2f}"
                }
                for t in recent_trades
            ]
        }
    except Exception as e:
        logger.error(f"Mobile dashboard error: {e}")
        return {"status": "Error", "message": str(e)}

@app.get("/quick")
async def quick_status():
    """Ultra-simple status for phone widgets"""
    try:
        portfolio = signal_generator.portfolio
        value = portfolio.get_total_value()
        pnl = value - portfolio.initial_capital
        
        return {
            "value": round(value),
            "pnl": round(pnl),
            "pct": round((pnl / portfolio.initial_capital) * 100, 1),
            "trades": len([t for t in portfolio.trades if t["timestamp"][:10] == datetime.now().date().isoformat()]),
            "open": is_market_open()
        }
    except:
        return {"value": 0, "pnl": 0, "pct": 0, "trades": 0, "open": False}

@app.get("/api/trade-logs")
async def get_trade_logs(date: Optional[str] = None):
    """Get trade logs for analysis"""
    try:
        if not date:
            date = datetime.now().strftime("%Y%m%d")
        
        log_file = f'trade_logs/trades_{date}.csv'
        
        if not os.path.exists(log_file):
            return {"error": "No logs for this date", "date": date}
        
        import csv
        trades = []
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
        
        # Calculate summary stats
        total_trades = len(trades)
        buy_trades = len([t for t in trades if t['action'] == 'BUY'])
        sell_trades = len([t for t in trades if t['action'] == 'SELL'])
        
        return {
            "date": date,
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "trades": trades[-20:],  # Last 20 trades
            "log_file": log_file
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/force-exit")
async def force_exit(symbol: str = None):
    """Force exit positions"""
    try:
        if symbol:
            positions_to_exit = [symbol] if symbol in signal_generator.portfolio.positions else []
        else:
            positions_to_exit = list(signal_generator.portfolio.positions.keys())
        
        exits = []
        for sym in positions_to_exit:
            try:
                ticker = yf.Ticker(sym)
                recent = ticker.history(period="1d", interval="1m")
                current_price = recent['Close'].iloc[-1] if not recent.empty else signal_generator.portfolio.positions[sym]['avg_price']
                
                exit_signal = MultiStrategySignal(
                    symbol=sym,
                    primary_action="SELL",
                    strategies_voting={"manual": {"action": "SELL", "confidence": 1.0}},
                    combined_confidence=1.0,
                    price=current_price,
                    zone="manual",
                    zone_multiplier=1.0,
                    technical_indicators={},
                    risk_score=0,
                    expected_return=0,
                    stop_loss=0,
                    take_profit=0,
                    position_size=signal_generator.portfolio.positions[sym]["qty"],
                    timestamp=datetime.now().isoformat(),
                    execution_priority=10,
                    explanations=["MANUAL EXIT"]
                )
                
                result = signal_generator.portfolio.execute_multi_strategy_trade(exit_signal)
                if result["success"]:
                    exits.append(sym)
            except:
                pass
        
        return {
            "success": True,
            "positions_exited": exits,
            "portfolio_value": signal_generator.portfolio.get_total_value()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "portfolio_value": signal_generator.portfolio.get_total_value()
        })
        
        while True:
            await asyncio.sleep(30)
            zone_name, _ = get_current_market_zone()
            await websocket.send_json({
                "type": "heartbeat",
                "zone": zone_name,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)

async def broadcast_to_websockets(message: Dict):
    """Broadcast to all connected clients"""
    disconnected = set()
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except:
            disconnected.add(ws)
    active_websockets.difference_update(disconnected)

# ============================================
# MAIN TRADING LOOP
# ============================================

async def enhanced_trading_loop():
    """Main trading loop with multi-strategy, ML learning, and risk-taking"""
    consecutive_no_trade_cycles = 0
    
    while True:
        try:
            if not is_market_open():
                await asyncio.sleep(60)
                continue
            
            zone_name, zone_config = get_current_market_zone()
            
            # Check exits first
            exits = await exit_manager.check_exits()
            
            # Evaluate exits for ML learning
            if ML_ENABLED and ml_brain and exits:
                for exit in exits:
                    try:
                        # Simple outcome: profit or loss percentage
                        result = exit.get('pnl_percent', 0)
                        # Record with simple market data (we don't need complex features here)
                        ml_brain.evaluate_prediction(
                            {'action': 'BUY' if result > 0 else 'SELL'},
                            result
                        )
                    except:
                        pass
            
            # Generate signals if we have room for positions
            if signal_generator.portfolio.can_open_position():
                signals = await signal_generator.scan_universe()
                
                # Get base threshold from zone config
                threshold = zone_config.get("threshold", MIN_CONFIDENCE)
                
                # RISK-TAKING: Lower threshold gradually if no recent trades
                if consecutive_no_trade_cycles > 3:
                    risk_adjustment = 0.9 ** (consecutive_no_trade_cycles - 3)  # Gradual lowering
                    threshold = max(0.3, threshold * risk_adjustment)  # Floor at 30%
                    logger.info(f"📊 Risk-taking mode: Threshold lowered to {threshold:.2%} after {consecutive_no_trade_cycles} cycles")
                
                # Execute signals above threshold
                executed = False
                for signal in signals[:5]:  # Check top 5 signals
                    if signal.combined_confidence >= threshold:
                        # Add slight randomness for exploration (10% chance to take riskier trades)
                        explore = np.random.random() < 0.1
                        if explore and signal.combined_confidence >= threshold * 0.8:
                            logger.info(f"🎲 Exploration trade: Taking {signal.symbol} at {signal.combined_confidence:.2%}")
                        
                        if signal.combined_confidence >= threshold or explore:
                            result = signal_generator.portfolio.execute_multi_strategy_trade(signal)
                            
                            if result.get("success"):
                                logger.info(f"Trade executed: {signal.primary_action} {signal.symbol} in {zone_name}")
                                
                                # Track for ML learning
                                signal_generator.portfolio.track_trade_for_learning(result['trade'])
                                
                                await broadcast_to_websockets({
                                    "type": "auto_trade",
                                    "signal": signal.to_dict(),
                                    "zone": zone_name,
                                    "ml_enabled": ML_ENABLED
                                })
                                
                                executed = True
                                consecutive_no_trade_cycles = 0  # Reset counter
                                
                                await asyncio.sleep(5)  # Pause between trades
                
                if not executed:
                    consecutive_no_trade_cycles += 1
                else:
                    consecutive_no_trade_cycles = 0
                
                # Update current signals
                global current_signals
                current_signals = signals
            
            # Sleep based on zone activity
            sleep_time = SIGNAL_INTERVAL if zone_config.get("multiplier", 1) < 1.5 else SIGNAL_INTERVAL // 2
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup():
    """Initialize the enhanced trading system"""
    asyncio.create_task(enhanced_trading_loop())
    
    logger.info("="*60)
    logger.info(f"{PLATFORM_NAME} v{PLATFORM_VERSION}")
    logger.info("SIMULATED TRADING - Testing AI Performance")
    logger.info("No real money or trades - Safe testing environment")
    logger.info(f"ML Brain: {'ACTIVE - Simple Learning' if ML_ENABLED else 'DISABLED'}")
    logger.info(f"Auto Trading: {'ENABLED' if AUTO_TRADING_ENABLED else 'DISABLED'}")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE:.1%}")
    logger.info(f"Risk-Taking: Progressive threshold lowering for learning")
    logger.info(f"Stock Universe: {len(STOCK_UNIVERSE)} stocks")
    logger.info("Data Source: Yahoo Finance (real market data)")
    logger.info("="*60)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")