"""
Backtesting Engine for Personal Trading System
Test strategies on historical data before live deployment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    recovery_factor: float
    calmar_ratio: float

class Backtester:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def backtest_strategy(
        self,
        strategy_func,
        symbols: List[str],
        start_date: str,
        end_date: str,
        **strategy_params
    ) -> BacktestResult:
        """Run backtest on historical data"""
        
        capital = self.initial_capital
        positions = {}
        trades = []
        equity_curve = [capital]
        daily_returns = []
        
        # Fetch historical data for all symbols
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if not df.empty:
                data[symbol] = self._prepare_data(df)
        
        # Simulate trading day by day
        all_dates = sorted(set().union(*[df.index for df in data.values()]))
        
        for date in all_dates:
            # Get signals for this date
            signals = []
            for symbol, df in data.items():
                if date in df.index:
                    signal = strategy_func(df.loc[:date], symbol, **strategy_params)
                    if signal:
                        signals.append(signal)
            
            # Execute trades
            for signal in signals:
                if signal['action'] == 'BUY' and symbol not in positions:
                    # Open position
                    shares = int((capital * signal.get('position_size', 0.02)) / signal['price'])
                    if shares > 0:
                        cost = shares * signal['price']
                        if cost <= capital:
                            positions[signal['symbol']] = {
                                'shares': shares,
                                'entry_price': signal['price'],
                                'entry_date': date,
                                'stop_loss': signal.get('stop_loss'),
                                'take_profit': signal.get('take_profit')
                            }
                            capital -= cost
                            
                elif signal['action'] == 'SELL' and signal['symbol'] in positions:
                    # Close position
                    position = positions[signal['symbol']]
                    exit_price = signal['price']
                    pnl = (exit_price - position['entry_price']) * position['shares']
                    capital += position['shares'] * exit_price
                    
                    trades.append({
                        'symbol': signal['symbol'],
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'return': pnl / (position['entry_price'] * position['shares'])
                    })
                    
                    del positions[signal['symbol']]
            
            # Update positions with current prices
            portfolio_value = capital
            for symbol, position in positions.items():
                if symbol in data and date in data[symbol].index:
                    current_price = data[symbol].loc[date, 'Close']
                    
                    # Check stop loss
                    if position.get('stop_loss') and current_price <= position['stop_loss']:
                        pnl = (current_price - position['entry_price']) * position['shares']
                        capital += position['shares'] * current_price
                        trades.append({
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'shares': position['shares'],
                            'pnl': pnl,
                            'return': pnl / (position['entry_price'] * position['shares']),
                            'exit_reason': 'stop_loss'
                        })
                        positions.pop(symbol, None)
                    
                    # Check take profit
                    elif position.get('take_profit') and current_price >= position['take_profit']:
                        pnl = (current_price - position['entry_price']) * position['shares']
                        capital += position['shares'] * current_price
                        trades.append({
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'shares': position['shares'],
                            'pnl': pnl,
                            'return': pnl / (position['entry_price'] * position['shares']),
                            'exit_reason': 'take_profit'
                        })
                        positions.pop(symbol, None)
                    else:
                        portfolio_value += position['shares'] * current_price
            
            equity_curve.append(portfolio_value)
            
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity_curve, daily_returns)
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df.fillna(method='ffill').fillna(0)
    
    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        daily_returns: List[float]
    ) -> BacktestResult:
        """Calculate backtest performance metrics"""
        
        if not trades:
            return BacktestResult(
                total_return=0, sharpe_ratio=0, max_drawdown=0,
                win_rate=0, total_trades=0, profit_factor=0,
                avg_win=0, avg_loss=0, best_trade=0, worst_trade=0,
                recovery_factor=0, calmar_ratio=0
            )
        
        # Total return
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Win rate
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Average wins and losses
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Best and worst trades
        best_trade = max(trades, key=lambda x: x['return'])['return'] if trades else 0
        worst_trade = min(trades, key=lambda x: x['return'])['return'] if trades else 0
        
        # Sharpe ratio
        if daily_returns:
            sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        years = len(equity_curve) / 252
        annual_return = (equity_curve[-1] / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            recovery_factor=recovery_factor,
            calmar_ratio=calmar_ratio
        )
    
    def optimize_parameters(
        self,
        strategy_func,
        symbols: List[str],
        start_date: str,
        end_date: str,
        param_grid: Dict
    ) -> Dict:
        """Optimize strategy parameters using grid search"""
        
        best_result = None
        best_params = None
        best_sharpe = -float('inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            result = self.backtest_strategy(
                strategy_func,
                symbols,
                start_date,
                end_date,
                **params
            )
            
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_result = result
                best_params = params
        
        return {
            'best_params': best_params,
            'best_result': best_result,
            'all_results': self.results
        }
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations from grid"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def plot_results(self, equity_curve: List[float], trades: List[Dict]):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Equity curve
        axes[0].plot(equity_curve, label='Portfolio Value')
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        peak = equity_curve[0]
        drawdowns = []
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)
        
        axes[1].fill_between(range(len(drawdowns)), drawdowns, alpha=0.3, color='red')
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Trade returns distribution
        returns = [t['return'] * 100 for t in trades]
        axes[2].hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[2].axvline(x=0, color='r', linestyle='--')
        axes[2].set_title('Trade Returns Distribution')
        axes[2].set_xlabel('Return (%)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Example strategy functions for backtesting
def momentum_strategy(df: pd.DataFrame, symbol: str, **params) -> Dict:
    """Simple momentum strategy for backtesting"""
    if len(df) < 50:
        return None
    
    latest = df.iloc[-1]
    lookback = params.get('lookback', 20)
    threshold = params.get('threshold', 0.02)
    
    # Calculate momentum
    momentum = (latest['Close'] - df['Close'].iloc[-lookback]) / df['Close'].iloc[-lookback]
    
    # Generate signal
    if momentum > threshold and latest['Close'] > latest['SMA_50']:
        return {
            'symbol': symbol,
            'action': 'BUY',
            'price': latest['Close'],
            'confidence': min(momentum * 10, 0.9),
            'stop_loss': latest['Close'] * 0.95,
            'take_profit': latest['Close'] * 1.05,
            'position_size': 0.02
        }
    elif momentum < -threshold and latest['Close'] < latest['SMA_50']:
        return {
            'symbol': symbol,
            'action': 'SELL',
            'price': latest['Close'],
            'confidence': min(abs(momentum) * 10, 0.9)
        }
    
    return None

def mean_reversion_strategy(df: pd.DataFrame, symbol: str, **params) -> Dict:
    """Mean reversion strategy for backtesting"""
    if len(df) < 30:
        return None
    
    latest = df.iloc[-1]
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    
    if latest['RSI'] < rsi_oversold and latest['Close'] < latest['BB_Lower']:
        return {
            'symbol': symbol,
            'action': 'BUY',
            'price': latest['Close'],
            'confidence': (rsi_oversold - latest['RSI']) / rsi_oversold,
            'stop_loss': latest['Close'] * 0.97,
            'take_profit': latest['BB_Middle'],
            'position_size': 0.02
        }
    elif latest['RSI'] > rsi_overbought and latest['Close'] > latest['BB_Upper']:
        return {
            'symbol': symbol,
            'action': 'SELL',
            'price': latest['Close'],
            'confidence': (latest['RSI'] - rsi_overbought) / (100 - rsi_overbought)
        }
    
    return None

if __name__ == "__main__":
    # Example backtest
    backtester = Backtester(initial_capital=100000)
    
    # Test momentum strategy
    result = backtester.backtest_strategy(
        momentum_strategy,
        symbols=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        lookback=20,
        threshold=0.02
    )
    
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    
    # Parameter optimization
    param_grid = {
        'lookback': [10, 20, 30],
        'threshold': [0.01, 0.02, 0.03]
    }
    
    optimization_result = backtester.optimize_parameters(
        momentum_strategy,
        symbols=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        param_grid=param_grid
    )
    
    print("\n=== OPTIMIZED PARAMETERS ===")
    print(f"Best Parameters: {optimization_result['best_params']}")
    print(f"Best Sharpe: {optimization_result['best_result'].sharpe_ratio:.2f}")