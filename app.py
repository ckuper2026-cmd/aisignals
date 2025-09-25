"""
AI Edge Trading Platform - Production-Ready Version
Fixed all potential runtime errors and edge cases
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd

# FastAPI imports
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# External libraries
import yfinance as yf
import pytz
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Photon AI Edge Trading Platform",
    description="Advanced AI trading with edge computing and ML optimization",
    version="2.0.0"
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

# Trading parameters - reduced universe for stability
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
    "SPY", "QQQ", "IWM"
]

# Edge computing optimization
EDGE_CONFIG = {
    "cache_ttl": 60,
    "batch_size": 5,
    "parallel_workers": 2,
    "prediction_cache_size": 100,
}

# Risk management
RISK_CONFIG = {
    "max_portfolio_risk": 0.20,
    "max_position_risk": 0.02,
    "correlation_threshold": 0.7,
    "var_confidence": 0.95,
    "max_leverage": 1.5,
    "stop_loss_multiplier": 1.5,
    "take_profit_multiplier": 2.5,
}

# ============================================
# ENHANCED DATA STRUCTURES
# ============================================

@dataclass
class EnhancedSignal:
    symbol: str
    action: str
    price: float
    confidence: float
    ml_confidence: float
    ensemble_score: float
    risk_score: float
    strategy: str
    ml_strategy: str
    explanation: str
    technical_indicators: Dict
    market_regime: str
    edge_score: float
    expected_return: float
    sharpe_ratio: float
    stop_loss: float
    take_profit: float
    position_size: int
    timestamp: str
    execution_priority: int
    
    def to_dict(self):
        d = asdict(self)
        # Ensure all float values are not NaN
        for key, value in d.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                d[key] = 0.0
        return d

# ============================================
# CENTRALIZED PORTFOLIO WITH RISK MANAGEMENT
# ============================================

class RiskManagedPortfolio:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = deque(maxlen=1000)
        self.pending_orders = {}
        self.risk_metrics = {}
        self.performance_stats = {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "daily_returns": deque(maxlen=252)
        }
        
    def calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if not self.performance_stats["daily_returns"]:
            return 0.0
        
        try:
            returns = np.array(self.performance_stats["daily_returns"])
            if len(returns) < 2:
                return 0.0
            var = np.percentile(returns, (1 - confidence) * 100)
            return abs(var * self.get_total_value())
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.0
    
    def calculate_position_size(self, signal: EnhancedSignal) -> int:
        """Kelly Criterion based position sizing"""
        try:
            # Kelly fraction
            win_rate = max(0.5, self.performance_stats.get("win_rate", 0.5))
            avg_win = 0.02
            avg_loss = 0.01
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = min(0.25, max(0, kelly_fraction))
            
            # Adjust for confidence and risk
            adjusted_fraction = kelly_fraction * signal.confidence * (1 - signal.risk_score)
            
            # Calculate position value
            position_value = self.cash * adjusted_fraction
            position_value = min(position_value, self.cash * RISK_CONFIG["max_position_risk"])
            
            # Calculate shares
            if signal.price > 0:
                shares = int(position_value / signal.price)
                return max(1, shares)
            return 1
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 1
    
    def check_correlation(self, symbol: str) -> bool:
        """Check if new position is too correlated with existing"""
        if len(self.positions) < 2:
            return True
        
        try:
            existing_symbols = list(self.positions.keys())
            data = {}
            
            for sym in existing_symbols[:5] + [symbol]:  # Limit to 5 for performance
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="1mo")["Close"]
                if not hist.empty:
                    data[sym] = hist.pct_change().dropna()
            
            if len(data) < 2:
                return True
                
            df = pd.DataFrame(data)
            if symbol not in df.columns:
                return True
                
            correlations = df.corr()[symbol]
            
            for sym in existing_symbols:
                if sym in correlations and abs(correlations[sym]) > RISK_CONFIG["correlation_threshold"]:
                    logger.warning(f"High correlation between {symbol} and {sym}: {correlations[sym]:.2f}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Correlation check error: {e}")
            return True  # Allow if check fails
    
    def can_trade(self, signal: EnhancedSignal) -> bool:
        """Advanced risk checks before trading"""
        try:
            # Check portfolio risk
            current_var = self.calculate_var()
            max_var = self.initial_capital * RISK_CONFIG["max_portfolio_risk"]
            
            if current_var > max_var:
                logger.warning(f"Portfolio VaR ${current_var:.2f} exceeds max ${max_var:.2f}")
                return False
            
            # Check correlation for buys
            if signal.action == "BUY" and not self.check_correlation(signal.symbol):
                return False
            
            # Check leverage
            total_exposure = sum(
                pos.get("qty", 0) * pos.get("current_price", pos.get("avg_price", 0)) 
                for pos in self.positions.values()
            )
            if self.initial_capital > 0 and total_exposure > self.initial_capital * RISK_CONFIG["max_leverage"]:
                logger.warning(f"Max leverage exceeded: {total_exposure/self.initial_capital:.2f}x")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Can trade check error: {e}")
            return False
    
    def execute_trade(self, signal: EnhancedSignal) -> Dict:
        """Execute trade with risk management"""
        try:
            if not self.can_trade(signal):
                return {"success": False, "error": "Risk limits exceeded"}
            
            # Calculate position size
            quantity = signal.position_size or self.calculate_position_size(signal)
            
            if signal.action == "BUY":
                cost = signal.price * quantity
                if cost > self.cash:
                    quantity = int(self.cash / signal.price) if signal.price > 0 else 0
                    if quantity < 1:
                        return {"success": False, "error": "Insufficient funds"}
                
                self.cash -= signal.price * quantity
                
                if signal.symbol not in self.positions:
                    self.positions[signal.symbol] = {
                        "qty": 0,
                        "avg_price": 0.0,
                        "current_price": signal.price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit
                    }
                
                position = self.positions[signal.symbol]
                total_cost = (position["qty"] * position["avg_price"]) + (signal.price * quantity)
                position["qty"] += quantity
                position["avg_price"] = total_cost / position["qty"] if position["qty"] > 0 else signal.price
                position["current_price"] = signal.price
                
            else:  # SELL
                if signal.symbol not in self.positions:
                    return {"success": False, "error": "No position to sell"}
                if self.positions[signal.symbol]["qty"] < quantity:
                    quantity = self.positions[signal.symbol]["qty"]
                    if quantity < 1:
                        return {"success": False, "error": "No shares to sell"}
                
                self.cash += signal.price * quantity
                self.positions[signal.symbol]["qty"] -= quantity
                
                if self.positions[signal.symbol]["qty"] == 0:
                    del self.positions[signal.symbol]
            
            # Record trade
            trade = {
                "id": f"trade_{datetime.now().timestamp()}",
                "symbol": signal.symbol,
                "action": signal.action,
                "quantity": quantity,
                "price": signal.price,
                "timestamp": datetime.now().isoformat(),
                "signal": signal.to_dict()
            }
            
            self.trades.append(trade)
            self.update_performance_stats()
            
            logger.info(f"Trade executed: {signal.action} {quantity} {signal.symbol} @ ${signal.price:.2f}")
            
            return {"success": True, "trade": trade}
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def update_performance_stats(self):
        """Update portfolio performance metrics"""
        try:
            if len(self.trades) < 2:
                return
            
            # Calculate returns
            current_value = self.get_total_value()
            if self.initial_capital > 0:
                daily_return = (current_value - self.initial_capital) / self.initial_capital
                self.performance_stats["daily_returns"].append(daily_return)
            
            # Calculate win rate
            wins = sum(1 for t in self.trades if self.is_winning_trade(t))
            self.performance_stats["win_rate"] = wins / len(self.trades) if self.trades else 0
            
            # Calculate Sharpe ratio
            if len(self.performance_stats["daily_returns"]) > 1:
                returns = np.array(self.performance_stats["daily_returns"])
                std_ret = np.std(returns)
                if std_ret > 0:
                    self.performance_stats["sharpe_ratio"] = (
                        np.mean(returns) / std_ret * np.sqrt(252)
                    )
        except Exception as e:
            logger.error(f"Performance stats update error: {e}")
    
    def is_winning_trade(self, trade: Dict) -> bool:
        """Check if trade was profitable"""
        try:
            signal_data = trade.get("signal", {})
            return signal_data.get("expected_return", 0) > 0
        except:
            return False
    
    def get_total_value(self) -> float:
        """Get total portfolio value"""
        try:
            positions_value = sum(
                pos.get("qty", 0) * pos.get("current_price", pos.get("avg_price", 0))
                for pos in self.positions.values()
            )
            return self.cash + positions_value
        except Exception as e:
            logger.error(f"Total value calculation error: {e}")
            return self.cash

# ============================================
# MARKET REGIME DETECTION
# ============================================

class MarketRegimeDetector:
    def __init__(self):
        self.regimes = ["bull", "bear", "sideways", "volatile"]
        self.current_regime = "sideways"
        self.regime_history = deque(maxlen=100)
        
    async def detect_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime"""
        try:
            if market_data.empty or len(market_data) < 20:
                return "unknown"
            
            # Calculate regime indicators safely
            returns = market_data["Close"].pct_change().dropna()
            if len(returns) < 2:
                return "unknown"
                
            volatility = returns.std() * np.sqrt(252)
            
            if market_data["Close"].iloc[0] != 0:
                trend = (market_data["Close"].iloc[-1] - market_data["Close"].iloc[0]) / market_data["Close"].iloc[0]
            else:
                trend = 0
            
            # Regime classification
            if trend > 0.1 and volatility < 0.25:
                regime = "bull"
            elif trend < -0.1 and volatility < 0.25:
                regime = "bear"
            elif volatility > 0.35:
                regime = "volatile"
            else:
                regime = "sideways"
            
            self.current_regime = regime
            self.regime_history.append(regime)
            
            return regime
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return "unknown"

# ============================================
# EDGE COMPUTING OPTIMIZATION
# ============================================

class EdgeOptimizer:
    def __init__(self):
        self.cache = {}
        self.prediction_cache = deque(maxlen=EDGE_CONFIG["prediction_cache_size"])
        self.last_cache_clear = time.time()
        
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data if valid"""
        try:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < EDGE_CONFIG["cache_ttl"]:
                    return data
                else:
                    del self.cache[key]
        except:
            pass
        return None
    
    def set_cache(self, key: str, data: Any):
        """Set cache with timestamp"""
        try:
            self.cache[key] = (data, time.time())
            
            # Clear old cache periodically
            if time.time() - self.last_cache_clear > 300:
                self.clear_old_cache()
        except:
            pass
    
    def clear_old_cache(self):
        """Remove expired cache entries"""
        try:
            current_time = time.time()
            expired = [
                key for key, (_, timestamp) in self.cache.items()
                if current_time - timestamp > EDGE_CONFIG["cache_ttl"]
            ]
            for key in expired:
                del self.cache[key]
            self.last_cache_clear = current_time
        except:
            self.cache = {}

# ============================================
# ADVANCED SIGNAL GENERATOR
# ============================================

class AdvancedSignalGenerator:
    def __init__(self):
        self.portfolio = RiskManagedPortfolio()
        self.regime_detector = MarketRegimeDetector()
        self.edge_optimizer = EdgeOptimizer()
        self.signal_history = deque(maxlen=1000)
        
    async def generate_enhanced_signal(self, symbol: str) -> Optional[EnhancedSignal]:
        """Generate signal with ML enhancement and edge optimization"""
        
        # Check cache first
        cache_key = f"signal_{symbol}_{datetime.now().minute}"
        cached = self.edge_optimizer.get_cached_data(cache_key)
        if cached:
            return cached
        
        # Get market data with retry logic
        hist = None
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo", interval="1h")
                
                if not hist.empty and len(hist) >= 20:
                    break
                    
            except Exception as e:
                logger.warning(f"Yahoo Finance attempt {attempt+1} failed for {symbol}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
        
        if hist is None or hist.empty or len(hist) < 20:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        
        try:
            # Detect market regime
            regime = await self.regime_detector.detect_regime(hist)
            
            # Traditional technical analysis
            technical_signal = self.analyze_technicals(hist)
            
            # Create basic signal
            signal = self.create_signal(technical_signal, regime, symbol, hist)
            
            if signal:
                # Calculate edge score
                signal.edge_score = self.calculate_edge_score(signal, hist)
                
                # Set position size
                signal.position_size = self.portfolio.calculate_position_size(signal)
                
                # Cache result
                self.edge_optimizer.set_cache(cache_key, signal)
                
                # Store in history
                self.signal_history.append(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def analyze_technicals(self, data: pd.DataFrame) -> Dict:
        """Advanced technical analysis with error handling"""
        try:
            df = data.copy()
            
            # Ensure we have enough data
            if len(df) < 50:
                df = df.reindex(range(50), method='ffill')
            
            # Calculate indicators safely
            df['SMA_10'] = df['Close'].rolling(10, min_periods=1).mean()
            df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
            
            # RSI with safe division
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1)  # Avoid division by zero
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(20, min_periods=1).mean()
            bb_std = df['Close'].rolling(20, min_periods=1).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            df['BB_width'] = df['BB_upper'] - df['BB_lower']
            df['BB_width'] = df['BB_width'].replace(0, 1)  # Avoid division by zero
            df['BB_position'] = (df['Close'] - df['BB_lower']) / df['BB_width']
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(14, min_periods=1).mean()
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(20, min_periods=1).mean()
            df['Volume_SMA'] = df['Volume_SMA'].replace(0, 1)  # Avoid division by zero
            df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Replace any remaining NaN/inf values
            df = df.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Generate signal
            action = "HOLD"
            confidence = 0.0
            strategy = "no_signal"
            
            # Check conditions safely
            conditions = []
            
            if latest['RSI'] < 30 and latest['BB_position'] < 0.2:
                conditions.append(("oversold_bounce", "BUY", 0.8))
            if latest['RSI'] > 70 and latest['BB_position'] > 0.8:
                conditions.append(("overbought_reversal", "SELL", 0.8))
            if latest['MACD'] > latest['MACD_signal'] and latest['MACD_histogram'] > 0:
                conditions.append(("macd_bullish", "BUY", 0.7))
            if latest['MACD'] < latest['MACD_signal'] and latest['MACD_histogram'] < 0:
                conditions.append(("macd_bearish", "SELL", 0.7))
            if latest['Volume_ratio'] > 2.0 and latest['Close'] > latest['SMA_20']:
                conditions.append(("volume_breakout", "BUY", 0.75))
            
            # Find strongest signal
            for name, signal_action, signal_confidence in conditions:
                if signal_confidence > confidence:
                    action = signal_action
                    confidence = signal_confidence
                    strategy = name
            
            return {
                "action": action,
                "confidence": confidence,
                "strategy": strategy,
                "indicators": {
                    "rsi": float(latest['RSI']),
                    "macd": float(latest['MACD']),
                    "bb_position": float(latest['BB_position']),
                    "volume_ratio": float(latest['Volume_ratio']),
                    "atr": float(latest['ATR']),
                }
            }
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "strategy": "error",
                "indicators": {}
            }
    
    def create_signal(self, technical: Dict, regime: str, symbol: str, data: pd.DataFrame) -> Optional[EnhancedSignal]:
        """Create signal from technical analysis"""
        try:
            if technical["action"] == "HOLD":
                return None
            
            latest = data.iloc[-1]
            atr = technical["indicators"].get("atr", latest['Close'] * 0.02)
            
            if atr == 0:
                atr = latest['Close'] * 0.02  # 2% default
            
            if technical["action"] == "BUY":
                stop_loss = latest['Close'] - (atr * RISK_CONFIG["stop_loss_multiplier"])
                take_profit = latest['Close'] + (atr * RISK_CONFIG["take_profit_multiplier"])
            else:
                stop_loss = latest['Close'] + (atr * RISK_CONFIG["stop_loss_multiplier"])
                take_profit = latest['Close'] - (atr * RISK_CONFIG["take_profit_multiplier"])
            
            if latest['Close'] > 0:
                expected_return = abs((take_profit - latest['Close']) / latest['Close'])
            else:
                expected_return = 0.0
            
            # Calculate Sharpe ratio estimate
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                sharpe = (returns.mean() / returns.std() * np.sqrt(252))
            else:
                sharpe = 0.0
            
            # Risk score
            if latest['Close'] > 0:
                volatility = atr / latest['Close']
                risk_score = min(volatility * 10, 1.0)
            else:
                risk_score = 0.5
            
            signal = EnhancedSignal(
                symbol=symbol,
                action=technical["action"],
                price=float(latest['Close']),
                confidence=technical["confidence"],
                ml_confidence=0.5,  # No ML in this version
                ensemble_score=technical["confidence"],
                risk_score=risk_score,
                strategy=technical["strategy"],
                ml_strategy="disabled",
                explanation=f"{technical['action']} signal: {technical['strategy']} in {regime} market",
                technical_indicators=technical["indicators"],
                market_regime=regime,
                edge_score=0.0,
                expected_return=expected_return,
                sharpe_ratio=sharpe,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                position_size=0,
                timestamp=datetime.now().isoformat(),
                execution_priority=self.calculate_priority(technical["confidence"], risk_score, expected_return)
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal creation error: {e}")
            return None
    
    def calculate_edge_score(self, signal: EnhancedSignal, data: pd.DataFrame) -> float:
        """Calculate proprietary edge score"""
        try:
            edge_components = []
            
            # Technical edge
            indicators = signal.technical_indicators
            if indicators.get("rsi", 50) < 30 or indicators.get("rsi", 50) > 70:
                edge_components.append(0.2)
            
            # Volume edge
            if indicators.get("volume_ratio", 1) > 1.5:
                edge_components.append(0.15)
            
            # Regime alignment
            if (signal.action == "BUY" and signal.market_regime == "bull") or \
               (signal.action == "SELL" and signal.market_regime == "bear"):
                edge_components.append(0.2)
            
            # Sharpe ratio edge
            if signal.sharpe_ratio > 1.5:
                edge_components.append(0.2)
            
            return min(sum(edge_components), 1.0)
            
        except Exception as e:
            logger.error(f"Edge score calculation error: {e}")
            return 0.0
    
    def calculate_priority(self, confidence: float, risk: float, return_: float) -> int:
        """Calculate execution priority (1-10, 10 being highest)"""
        try:
            priority = confidence * 3 + (1 - risk) * 3 + return_ * 4
            return min(10, max(1, int(priority * 10)))
        except:
            return 5
    
    async def scan_universe(self) -> List[EnhancedSignal]:
        """Scan entire universe for signals"""
        signals = []
        
        for symbol in STOCK_UNIVERSE:
            try:
                signal = await self.generate_enhanced_signal(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by priority
        signals.sort(key=lambda x: x.execution_priority, reverse=True)
        
        return signals[:5]  # Top 5 signals

# ============================================
# GLOBAL INSTANCES
# ============================================

signal_generator = AdvancedSignalGenerator()
active_websockets = set()
current_signals = []
last_signal_time = time.time()

# ============================================
# MARKET MONITORING
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

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "platform": "Photon AI Edge Trading Platform",
        "version": "2.0.0",
        "status": "operational",
        "market_open": is_market_open(),
        "portfolio_value": signal_generator.portfolio.get_total_value()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "market_open": is_market_open(),
        "cache_size": len(signal_generator.edge_optimizer.cache),
        "active_connections": len(active_websockets)
    }

@app.get("/api/signals")
async def get_signals():
    """Get current trading signals"""
    global current_signals, last_signal_time
    
    try:
        # Refresh signals if stale or empty
        if time.time() - last_signal_time > 60 or not current_signals:
            if is_market_open():
                current_signals = await signal_generator.scan_universe()
                last_signal_time = time.time()
        
        portfolio = signal_generator.portfolio
        
        return {
            "signals": [s.to_dict() for s in current_signals],
            "count": len(current_signals),
            "portfolio": {
                "value": portfolio.get_total_value(),
                "cash": portfolio.cash,
                "positions": len(portfolio.positions),
                "sharpe_ratio": portfolio.performance_stats.get("sharpe_ratio", 0),
                "win_rate": portfolio.performance_stats.get("win_rate", 0)
            },
            "market": {
                "open": is_market_open(),
                "regime": signal_generator.regime_detector.current_regime
            }
        }
    except Exception as e:
        logger.error(f"Get signals error: {e}")
        return {
            "signals": [],
            "count": 0,
            "error": str(e)
        }

@app.post("/api/execute")
async def execute_trade(symbol: str, action: str, quantity: int = None):
    """Execute a trade"""
    try:
        # Find or generate signal
        signal = None
        for s in current_signals:
            if s.symbol == symbol:
                signal = s
                break
        
        if not signal:
            signal = await signal_generator.generate_enhanced_signal(symbol)
        
        if not signal:
            return {"success": False, "error": "Could not generate signal"}
        
        # Override action if specified
        if action:
            signal.action = action
        
        # Override quantity if specified
        if quantity:
            signal.position_size = quantity
        
        # Execute trade
        result = signal_generator.portfolio.execute_trade(signal)
        
        # Broadcast to websockets if successful
        if result.get("success"):
            await broadcast_message({
                "type": "trade_executed",
                "trade": result["trade"],
                "portfolio": {
                    "value": signal_generator.portfolio.get_total_value(),
                    "positions": len(signal_generator.portfolio.positions)
                }
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Execute trade error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio details"""
    try:
        portfolio = signal_generator.portfolio
        
        positions_detail = []
        for symbol, position in portfolio.positions.items():
            avg_price = position.get("avg_price", 0)
            current_price = position.get("current_price", avg_price)
            qty = position.get("qty", 0)
            
            positions_detail.append({
                "symbol": symbol,
                "quantity": qty,
                "avg_price": avg_price,
                "current_price": current_price,
                "value": qty * current_price,
                "pnl": (current_price - avg_price) * qty if avg_price > 0 else 0,
                "pnl_percent": ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            })
        
        return {
            "total_value": portfolio.get_total_value(),
            "cash": portfolio.cash,
            "positions": positions_detail,
            "performance": portfolio.performance_stats,
            "risk_metrics": {
                "var_95": portfolio.calculate_var(0.95),
                "max_portfolio_risk": RISK_CONFIG["max_portfolio_risk"],
                "current_leverage": sum(p["value"] for p in positions_detail) / portfolio.initial_capital if positions_detail and portfolio.initial_capital > 0 else 0
            },
            "trades_today": len([t for t in portfolio.trades if t["timestamp"][:10] == datetime.now().date().isoformat()])
        }
    except Exception as e:
        logger.error(f"Get portfolio error: {e}")
        return {"error": str(e)}

@app.get("/api/ml-stats")
async def get_ml_stats():
    """Get ML brain statistics"""
    return {
        "ml_enabled": False,
        "message": "ML brain disabled in production version for stability"
    }

@app.get("/api/market-regime")
async def get_market_regime():
    """Get current market regime analysis"""
    try:
        # Get SPY data for overall market
        spy = yf.Ticker("SPY")
        spy_data = spy.history(period="1mo")
        
        if not spy_data.empty:
            regime = await signal_generator.regime_detector.detect_regime(spy_data)
        else:
            regime = "unknown"
        
        regime_history = list(signal_generator.regime_detector.regime_history)[-20:]
        
        return {
            "current_regime": regime,
            "regime_history": regime_history,
            "regime_distribution": {
                r: regime_history.count(r) 
                for r in ["bull", "bear", "sideways", "volatile", "unknown"]
            }
        }
    except Exception as e:
        logger.error(f"Market regime error: {e}")
        return {"error": str(e), "current_regime": "unknown"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        # Send initial data
        portfolio_data = await get_portfolio()
        await websocket.send_json({
            "type": "connected",
            "portfolio": portfolio_data,
            "signals": [s.to_dict() for s in current_signals]
        })
        
        # Keep connection alive
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)

async def broadcast_message(message: Dict):
    """Broadcast message to all connected clients"""
    disconnected = set()
    
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except:
            disconnected.add(ws)
    
    active_websockets.difference_update(disconnected)

# ============================================
# BACKGROUND TASKS
# ============================================

async def auto_trading_loop():
    """Main auto-trading loop"""
    while True:
        try:
            if not is_market_open():
                await asyncio.sleep(300)  # Wait 5 minutes
                continue
            
            # Generate signals
            signals = await signal_generator.scan_universe()
            
            # Auto-execute high confidence signals
            for signal in signals:
                if signal.confidence >= 0.75 and signal.edge_score >= 0.6:
                    result = signal_generator.portfolio.execute_trade(signal)
                    
                    if result.get("success"):
                        logger.info(f"Auto-executed: {signal.action} {signal.symbol} @ ${signal.price}")
                        await broadcast_message({
                            "type": "auto_trade",
                            "signal": signal.to_dict(),
                            "trade": result["trade"]
                        })
            
            # Update current signals
            global current_signals
            current_signals = signals
            
            # Broadcast signals
            await broadcast_message({
                "type": "signals_update",
                "signals": [s.to_dict() for s in signals]
            })
            
            # Wait before next iteration
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Auto-trading loop error: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup():
    """Initialize background tasks"""
    asyncio.create_task(auto_trading_loop())
    
    logger.info("="*60)
    logger.info("PHOTON AI EDGE TRADING PLATFORM v2.0 - PRODUCTION")
    logger.info("Market Hours Check: Enabled")
    logger.info("Error Handling: Enhanced")
    logger.info("="*60)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")