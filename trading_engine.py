"""
Advanced Trading Engine with Centralized Portfolio Management
Handles all trading logic, risk management, and position tracking
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import deque
import json

logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass 
class Trade:
    id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    timestamp: datetime
    strategy: str
    pnl: float = 0.0

class CentralizedPortfolio:
    """Centralized portfolio for performance testing and risk management"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: Position}
        self.trades = deque(maxlen=1000)
        self.pending_orders = {}
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.daily_returns = deque(maxlen=252)
        
        # Risk limits
        self.max_position_size = 0.1  # 10% per position
        self.max_portfolio_risk = 0.2  # 20% total risk
        self.max_correlation = 0.7
        self.max_daily_loss = 0.05  # 5% daily loss limit
        
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_position_size(self, symbol: str, price: float, confidence: float) -> int:
        """Calculate optimal position size using Kelly Criterion"""
        portfolio_value = self.get_portfolio_value()
        
        # Kelly fraction calculation
        win_rate = max(0.5, self.winning_trades / max(1, self.total_trades))
        avg_win = 0.02  # 2% average win
        avg_loss = 0.01  # 1% average loss
        
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_f = min(0.25, max(0, kelly_f))  # Cap at 25%
        
        # Adjust for confidence
        position_value = portfolio_value * kelly_f * confidence
        
        # Apply risk limits
        max_position = portfolio_value * self.max_position_size
        position_value = min(position_value, max_position)
        
        # Calculate shares
        if price > 0:
            shares = int(position_value / price)
            return max(1, min(shares, int(self.cash / price)))
        return 0
    
    def can_trade(self, symbol: str, side: str, quantity: int, price: float) -> Tuple[bool, str]:
        """Check if trade is allowed under risk management rules"""
        
        # Check daily loss limit
        daily_pnl = self.calculate_daily_pnl()
        if daily_pnl < -self.max_daily_loss * self.initial_capital:
            return False, "Daily loss limit exceeded"
        
        # Check cash for buys
        if side == "BUY":
            required_cash = quantity * price
            if required_cash > self.cash:
                return False, "Insufficient funds"
        
        # Check position for sells
        elif side == "SELL":
            if symbol not in self.positions:
                return False, "No position to sell"
            if self.positions[symbol].quantity < quantity:
                return False, "Insufficient shares"
        
        # Check portfolio concentration
        position_value = quantity * price
        portfolio_value = self.get_portfolio_value()
        if position_value > portfolio_value * self.max_position_size:
            return False, "Position too large"
        
        return True, "OK"
    
    def execute_trade(self, symbol: str, side: str, quantity: int, price: float, 
                     stop_loss: float = None, take_profit: float = None,
                     strategy: str = "manual") -> Dict:
        """Execute a trade with full tracking"""
        
        # Validate trade
        can_trade, reason = self.can_trade(symbol, side, quantity, price)
        if not can_trade:
            return {"success": False, "reason": reason}
        
        trade_id = f"T{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        if side == "BUY":
            # Update cash
            self.cash -= quantity * price
            
            # Create or update position
            if symbol in self.positions:
                # Average up/down
                pos = self.positions[symbol]
                total_quantity = pos.quantity + quantity
                avg_price = ((pos.quantity * pos.entry_price) + (quantity * price)) / total_quantity
                pos.quantity = total_quantity
                pos.entry_price = avg_price
                pos.current_price = price
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    entry_time=datetime.now(),
                    stop_loss=stop_loss or price * 0.98,
                    take_profit=take_profit or price * 1.02
                )
        
        else:  # SELL
            pos = self.positions[symbol]
            
            # Calculate PnL
            pnl = (price - pos.entry_price) * quantity
            self.total_pnl += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update cash
            self.cash += quantity * price
            
            # Update or remove position
            pos.quantity -= quantity
            if pos.quantity == 0:
                del self.positions[symbol]
            else:
                pos.current_price = price
        
        # Record trade
        trade = Trade(
            id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            strategy=strategy,
            pnl=pnl if side == "SELL" else 0
        )
        
        self.trades.append(trade)
        self.total_trades += 1
        
        # Update performance metrics
        self.update_metrics()
        
        return {
            "success": True,
            "trade_id": trade_id,
            "executed": {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price
            }
        }
    
    def update_positions(self, market_prices: Dict[str, float]):
        """Update position prices and calculate unrealized PnL"""
        for symbol, pos in self.positions.items():
            if symbol in market_prices:
                pos.current_price = market_prices[symbol]
                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
    
    def check_stop_losses(self) -> List[Dict]:
        """Check and execute stop losses"""
        stops_hit = []
        
        for symbol, pos in list(self.positions.items()):
            if pos.current_price <= pos.stop_loss:
                # Execute stop loss
                result = self.execute_trade(
                    symbol=symbol,
                    side="SELL",
                    quantity=pos.quantity,
                    price=pos.current_price,
                    strategy="stop_loss"
                )
                
                if result["success"]:
                    stops_hit.append({
                        "symbol": symbol,
                        "price": pos.current_price,
                        "loss": (pos.current_price - pos.entry_price) * pos.quantity
                    })
                    logger.warning(f"Stop loss hit: {symbol} at ${pos.current_price:.2f}")
        
        return stops_hit
    
    def check_take_profits(self) -> List[Dict]:
        """Check and execute take profits"""
        profits_taken = []
        
        for symbol, pos in list(self.positions.items()):
            if pos.current_price >= pos.take_profit:
                # Execute take profit
                result = self.execute_trade(
                    symbol=symbol,
                    side="SELL",
                    quantity=pos.quantity,
                    price=pos.current_price,
                    strategy="take_profit"
                )
                
                if result["success"]:
                    profits_taken.append({
                        "symbol": symbol,
                        "price": pos.current_price,
                        "profit": (pos.current_price - pos.entry_price) * pos.quantity
                    })
                    logger.info(f"Take profit hit: {symbol} at ${pos.current_price:.2f}")
        
        return profits_taken
    
    def calculate_daily_pnl(self) -> float:
        """Calculate today's PnL"""
        today = datetime.now().date()
        daily_pnl = 0.0
        
        for trade in self.trades:
            if trade.timestamp.date() == today:
                daily_pnl += trade.pnl
        
        # Add unrealized PnL
        for pos in self.positions.values():
            daily_pnl += pos.unrealized_pnl
        
        return daily_pnl
    
    def update_metrics(self):
        """Update performance metrics"""
        current_value = self.get_portfolio_value()
        
        # Update peak and drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Calculate daily return
        daily_return = (current_value - self.initial_capital) / self.initial_capital
        self.daily_returns.append(daily_return)
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 1:
            returns_array = np.array(self.daily_returns)
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            if std_return > 0:
                self.sharpe_ratio = (avg_return * 252) / (std_return * np.sqrt(252))
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        total_value = self.get_portfolio_value()
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        return {
            "portfolio_value": total_value,
            "cash": self.cash,
            "positions_count": len(self.positions),
            "total_return": f"{total_return:.2%}",
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "win_rate": f"{win_rate:.2%}",
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "daily_pnl": self.calculate_daily_pnl(),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "pnl_percent": f"{(pos.current_price/pos.entry_price - 1):.2%}"
                }
                for pos in self.positions.values()
            ]
        }

class AdvancedTradingEngine:
    """Main trading engine orchestrating all components"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.portfolio = CentralizedPortfolio()
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_live = api_key and api_secret and api_key != "test"
        
    async def process_signal(self, signal: Dict) -> Dict:
        """Process a trading signal through the portfolio"""
        
        # Extract signal details
        symbol = signal.get("symbol")
        action = signal.get("action")
        price = signal.get("price", 0)
        confidence = signal.get("confidence", 0.5)
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        
        if action == "HOLD":
            return {"success": False, "reason": "No action signal"}
        
        # Calculate position size
        quantity = self.portfolio.get_position_size(symbol, price, confidence)
        
        if quantity == 0:
            return {"success": False, "reason": "Position size too small"}
        
        # Execute trade
        result = self.portfolio.execute_trade(
            symbol=symbol,
            side="BUY" if action == "BUY" else "SELL",
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=signal.get("strategy", "signal")
        )
        
        return result
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        return self.portfolio.get_performance_report()