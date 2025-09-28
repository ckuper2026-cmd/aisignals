"""
Personal Trading System - Optimized for Individual Use
Leverages existing Photon infrastructure with enhanced analytics
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Import ML Engine - COMPLETELY ISOLATED from customer system
from personal_ml_engine import PersonalMLEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PersonalPortfolio:
    cash: float = 100000.0
    positions: Dict = field(default_factory=dict)
    trade_history: List = field(default_factory=list)
    performance: Dict = field(default_factory=dict)
    risk_limits: Dict = field(default_factory=lambda: {
        'max_position_size': 0.15,  # 15% max per position
        'max_sector_exposure': 0.30,  # 30% max per sector
        'max_correlation': 0.7,  # Max correlation between positions
        'max_drawdown': 0.15,  # 15% max drawdown trigger
        'daily_loss_limit': 0.03,  # 3% daily loss limit
        'position_limit': 10  # Max concurrent positions
    })

@dataclass
class TradingSignal:
    symbol: str
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    strategy: str
    risk_reward: float
    expected_return: float
    holding_period: int  # Expected days
    timestamp: datetime = field(default_factory=datetime.now)
    
class PersonalTradingSystem:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.portfolio = PersonalPortfolio()
        self.market_data_cache = {}
        self.correlation_matrix = None
        self.sector_map = self._load_sector_map()
        self.performance_tracker = PerformanceTracker()
        self.risk_manager = RiskManager(self.portfolio)
        self.signal_validator = SignalValidator()
        
        # ISOLATED FROM CUSTOMER SYSTEM - No shared state
        self.system_id = "PERSONAL_SYSTEM"  # Unique identifier
        self.use_customer_signals = False   # Never use customer signals
        
        # ML components with continuous learning
        self.ml_engine = PersonalMLEngine()
        self.is_trained = False
        
        # Strategy weights (dynamically adjusted based on performance)
        self.strategy_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_return': 0})
        
        # Initialize ML training will be done in async context
        self.ml_initialized = False
        
    def _default_config(self) -> Dict:
        return {
            'symbols': [
                # Tech giants
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AMZN',
                # Finance
                'JPM', 'V', 'MA', 'GS', 'BRK-B',
                # Healthcare
                'JNJ', 'UNH', 'PFE', 'ABBV',
                # ETFs for hedging
                'SPY', 'QQQ', 'VXX', 'GLD', 'TLT'
            ],
            'strategies': {
                'momentum': {'enabled': True, 'weight': 0.25},
                'mean_reversion': {'enabled': True, 'weight': 0.20},
                'pairs_trading': {'enabled': True, 'weight': 0.15},
                'volatility_breakout': {'enabled': True, 'weight': 0.20},
                'ml_ensemble': {'enabled': True, 'weight': 0.20}
            },
            'risk_params': {
                'kelly_fraction': 0.25,  # Use 25% of Kelly Criterion
                'var_confidence': 0.95,  # 95% VaR
                'sharpe_target': 1.5,  # Target Sharpe ratio
                'rebalance_threshold': 0.05  # 5% drift triggers rebalance
            },
            'execution': {
                'slippage': 0.001,  # 0.1% slippage assumption
                'commission': 0.0,  # Assuming commission-free broker
                'use_limit_orders': True,
                'partial_fills': True
            }
        }
    
    def _load_sector_map(self) -> Dict:
        """Map symbols to sectors for diversification"""
        return {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'NVDA': 'Technology', 'META': 'Technology', 'AMZN': 'Consumer',
            'JPM': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'GS': 'Finance',
            'BRK-B': 'Finance', 'JNJ': 'Healthcare', 'UNH': 'Healthcare',
            'PFE': 'Healthcare', 'ABBV': 'Healthcare',
            'SPY': 'Index', 'QQQ': 'Index', 'VXX': 'Volatility',
            'GLD': 'Commodity', 'TLT': 'Bonds'
        }
    
    async def fetch_market_data(self, symbol: str, period: str = '3mo') -> pd.DataFrame:
        """Fetch and cache market data"""
        cache_key = f"{symbol}_{period}_{datetime.now().date()}"
        
        if cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval='1d')
            
            # Add technical indicators
            df = self._add_indicators(df)
            
            # Cache the data
            self.market_data_cache[cache_key] = df
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # Price-based
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['ATR'] = self._calculate_atr(df)
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Momentum
        df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
        df['MFI'] = self._calculate_mfi(df)
        
        # Support/Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        return df.fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        positive_flow[typical_price > typical_price.shift(1)] = money_flow[typical_price > typical_price.shift(1)]
        negative_flow[typical_price < typical_price.shift(1)] = money_flow[typical_price < typical_price.shift(1)]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    async def generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals across all strategies"""
        all_signals = []
        
        for symbol in self.config['symbols']:
            df = await self.fetch_market_data(symbol)
            if df.empty:
                continue
            
            # Run each strategy
            for strategy_name, strategy_config in self.config['strategies'].items():
                if not strategy_config['enabled']:
                    continue
                
                signal = await self._run_strategy(strategy_name, symbol, df)
                if signal and signal.confidence > 0.6:
                    # Validate signal
                    if self.signal_validator.validate(signal, self.portfolio, df):
                        all_signals.append(signal)
        
        # Rank and filter signals
        ranked_signals = self._rank_signals(all_signals)
        
        # Apply portfolio constraints
        final_signals = self._apply_portfolio_constraints(ranked_signals)
        
        return final_signals
    
    async def _run_strategy(self, strategy: str, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Run specific strategy and generate signal"""
        
        if strategy == 'momentum':
            return self._momentum_strategy(symbol, df)
        elif strategy == 'mean_reversion':
            return self._mean_reversion_strategy(symbol, df)
        elif strategy == 'pairs_trading':
            return await self._pairs_trading_strategy(symbol, df)
        elif strategy == 'volatility_breakout':
            return self._volatility_breakout_strategy(symbol, df)
        elif strategy == 'ml_ensemble':
            return self._ml_ensemble_strategy(symbol, df)
        
        return None
    
    def _momentum_strategy(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Enhanced momentum strategy"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        
        # Multi-timeframe momentum
        momentum_scores = []
        
        # Price momentum
        if latest['Close'] > latest['SMA_20'] and latest['SMA_20'] > latest['SMA_50']:
            momentum_scores.append(1)
        
        # Volume confirmation
        if latest['Volume_Ratio'] > 1.2:
            momentum_scores.append(1)
        
        # MACD momentum
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
            momentum_scores.append(1)
        
        # Rate of change
        if latest['ROC_10'] > 0.02:
            momentum_scores.append(1)
        
        confidence = sum(momentum_scores) / 4
        
        if confidence >= 0.6:
            atr = latest['ATR']
            entry = latest['Close']
            stop_loss = entry - (atr * 1.5)
            take_profit = entry + (atr * 3)
            
            return TradingSignal(
                symbol=symbol,
                action='BUY',
                confidence=confidence,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(symbol, stop_loss, entry),
                strategy='momentum',
                risk_reward=(take_profit - entry) / (entry - stop_loss),
                expected_return=((take_profit - entry) / entry),
                holding_period=5
            )
        
        return None
    
    def _mean_reversion_strategy(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Mean reversion using multiple indicators"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        
        signals = []
        
        # RSI oversold/overbought
        if latest['RSI'] < 30:
            signals.append(('BUY', 0.8))
        elif latest['RSI'] > 70:
            signals.append(('SELL', 0.8))
        
        # Bollinger Bands
        if latest['BB_Position'] < 0.1:
            signals.append(('BUY', 0.7))
        elif latest['BB_Position'] > 0.9:
            signals.append(('SELL', 0.7))
        
        # Price deviation from mean
        mean_20 = latest['SMA_20']
        deviation = (latest['Close'] - mean_20) / mean_20
        
        if deviation < -0.03:  # 3% below mean
            signals.append(('BUY', 0.6))
        elif deviation > 0.03:  # 3% above mean
            signals.append(('SELL', 0.6))
        
        if not signals:
            return None
        
        # Aggregate signals
        action = max(set([s[0] for s in signals]), key=[s[0] for s in signals].count)
        confidence = np.mean([s[1] for s in signals if s[0] == action])
        
        if confidence >= 0.65:
            atr = latest['ATR']
            entry = latest['Close']
            
            if action == 'BUY':
                stop_loss = entry - (atr * 1.2)
                take_profit = mean_20  # Target mean reversion
            else:
                stop_loss = entry + (atr * 1.2)
                take_profit = mean_20
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(symbol, stop_loss, entry),
                strategy='mean_reversion',
                risk_reward=abs((take_profit - entry) / (entry - stop_loss)),
                expected_return=((take_profit - entry) / entry) if action == 'BUY' else ((entry - take_profit) / entry),
                holding_period=3
            )
        
        return None
    
    async def _pairs_trading_strategy(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Identify pairs trading opportunities"""
        # Find correlated pairs
        pairs = self._find_correlated_pairs(symbol)
        
        if not pairs:
            return None
        
        best_pair = pairs[0]  # Highest correlation pair
        pair_symbol = best_pair['symbol']
        
        # Get pair data
        pair_df = await self.fetch_market_data(pair_symbol)
        if pair_df.empty:
            return None
        
        # Calculate spread
        spread = self._calculate_spread(df, pair_df)
        
        if spread is None:
            return None
        
        # Z-score of spread
        z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
        
        if abs(z_score) > 2:  # Significant deviation
            action = 'SELL' if z_score > 2 else 'BUY'
            confidence = min(abs(z_score) / 3, 0.9)
            
            entry = df['Close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            
            # Tighter stops for pairs trading
            stop_loss = entry - (atr * 0.8) if action == 'BUY' else entry + (atr * 0.8)
            take_profit = entry + (atr * 1.5) if action == 'BUY' else entry - (atr * 1.5)
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(symbol, stop_loss, entry) * 0.5,  # Smaller size for pairs
                strategy=f'pairs_trading ({pair_symbol})',
                risk_reward=abs((take_profit - entry) / (entry - stop_loss)),
                expected_return=abs((take_profit - entry) / entry),
                holding_period=7
            )
        
        return None
    
    def _volatility_breakout_strategy(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Trade volatility breakouts"""
        if len(df) < 30:
            return None
        
        latest = df.iloc[-1]
        prev_day = df.iloc[-2]
        
        # Check for volatility expansion
        current_vol = df['Volatility'].iloc[-5:].mean()
        historic_vol = df['Volatility'].iloc[-30:-5].mean()
        
        vol_expansion = current_vol / historic_vol
        
        # Check for price breakout with volume
        breakout_up = latest['Close'] > latest['Resistance'] and latest['Volume_Ratio'] > 1.5
        breakout_down = latest['Close'] < latest['Support'] and latest['Volume_Ratio'] > 1.5
        
        if vol_expansion > 1.3 and (breakout_up or breakout_down):
            action = 'BUY' if breakout_up else 'SELL'
            confidence = min(vol_expansion / 2, 0.85)
            
            entry = latest['Close']
            atr = latest['ATR']
            
            # Wider stops for volatility trades
            stop_loss = entry - (atr * 2) if action == 'BUY' else entry + (atr * 2)
            take_profit = entry + (atr * 4) if action == 'BUY' else entry - (atr * 4)
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(symbol, stop_loss, entry) * 0.7,  # Reduce size for volatility
                strategy='volatility_breakout',
                risk_reward=2.0,
                expected_return=abs((take_profit - entry) / entry),
                holding_period=10
            )
        
        return None
    
    
    async def _initialize_ml(self):
        """Initialize ML engine with REAL market data training"""
        try:
            logger.info("Initializing Personal ML Engine with REAL market data...")
            
            # Initialize the ML engine
            await self.ml_engine.initialize()
            
            # Enhance with real market data training
            try:
                from real_market_trainer import enhance_ml_training
                logger.info("Loading real historical market data for training...")
                
                if enhance_ml_training(self.ml_engine):
                    logger.info("ML Engine trained on REAL market patterns")
                else:
                    logger.info("Using initial synthetic data, will learn from real trades")
                    
            except Exception as e:
                logger.warning(f"Real market training not available: {e}")
                logger.info("ML will learn from live trading experience")
            
            self.ml_initialized = True
            logger.info("ML Engine ready and learning continuously from real markets")
            
        except Exception as e:
            logger.error(f"ML initialization error: {e}")
            self.ml_initialized = False
    
    def _ml_ensemble_strategy(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """ML-based ensemble prediction - ISOLATED FROM CUSTOMER SYSTEM"""
        if not self.ml_initialized:
            return None
        
        # Get ML prediction
        ml_prediction = self.ml_engine.predict(df)
        
        if not ml_prediction['ml_active'] or ml_prediction['action'] == 'HOLD':
            return None
        
        if ml_prediction['confidence'] > 0.65:
            entry = df['Close'].iloc[-1]
            atr = df['ATR'].iloc[-1]
            
            action = ml_prediction['action']
            
            # Risk management based on ML confidence
            risk_multiplier = 1.0 + (ml_prediction['confidence'] - 0.65) * 2
            
            stop_loss = entry - (atr * 1.5 / risk_multiplier) if action == 'BUY' else entry + (atr * 1.5 / risk_multiplier)
            take_profit = entry + (atr * 2.5 * risk_multiplier) if action == 'BUY' else entry - (atr * 2.5 * risk_multiplier)
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=ml_prediction['confidence'],
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(symbol, stop_loss, entry) * ml_prediction['confidence'],
                strategy=f"ml_ensemble ({ml_prediction.get('dominant_model', 'ensemble')})",
                risk_reward=abs((take_profit - entry) / (entry - stop_loss)),
                expected_return=abs((take_profit - entry) / entry),
                holding_period=5
            )
        
        return None
    
    def _calculate_position_size(self, symbol: str, stop_loss: float, entry: float) -> float:
        """Calculate position size using Kelly Criterion and risk management"""
        # Risk per trade (% of portfolio)
        risk_per_trade = 0.02  # 2% risk per trade
        
        # Calculate position size based on stop loss
        stop_distance = abs(entry - stop_loss) / entry
        
        # Kelly Criterion adjustment
        win_rate = self._get_strategy_win_rate(symbol)
        avg_win = 0.03  # 3% average win
        avg_loss = 0.015  # 1.5% average loss
        
        kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_f = max(0, min(kelly_f, 0.25))  # Cap at 25%
        
        # Final position size
        position_size = min(
            risk_per_trade / stop_distance,
            kelly_f,
            self.portfolio.risk_limits['max_position_size']
        )
        
        return round(position_size, 4)
    
    def _get_strategy_win_rate(self, symbol: str) -> float:
        """Get historical win rate for strategy"""
        # Placeholder - would track actual performance
        return 0.55
    
    def _find_correlated_pairs(self, symbol: str) -> List[Dict]:
        """Find highly correlated symbols for pairs trading"""
        correlations = []
        
        # Would implement correlation calculation
        # For now, return predefined pairs
        pairs_map = {
            'AAPL': ['MSFT', 'GOOGL'],
            'JPM': ['BAC', 'GS'],
            'V': ['MA'],
        }
        
        if symbol in pairs_map:
            return [{'symbol': s, 'correlation': 0.8} for s in pairs_map[symbol]]
        
        return []
    
    def _calculate_spread(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate spread between two price series"""
        try:
            # Align the dataframes
            df1_aligned = df1['Close'].iloc[-30:]
            df2_aligned = df2['Close'].iloc[-30:]
            
            # Normalize prices
            df1_norm = df1_aligned / df1_aligned.iloc[0]
            df2_norm = df2_aligned / df2_aligned.iloc[0]
            
            # Calculate spread
            spread = df1_norm - df2_norm
            
            return spread
        except:
            return None
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for ML models"""
        if len(df) < 50:
            return None
        
        features = []
        latest = df.iloc[-1]
        
        # Technical indicators
        features.extend([
            latest['RSI'] / 100,
            latest['MACD'],
            latest['BB_Position'],
            latest['Volume_Ratio'],
            latest['ROC_10'],
            latest['MFI'] / 100,
            latest['Volatility']
        ])
        
        # Price patterns
        features.extend([
            1 if latest['Close'] > latest['SMA_20'] else 0,
            1 if latest['SMA_20'] > latest['SMA_50'] else 0,
            (latest['Close'] - latest['Support']) / (latest['Resistance'] - latest['Support'])
        ])
        
        return np.array(features).reshape(1, -1)
    
    def _rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Rank signals by expected return and confidence"""
        for signal in signals:
            # Composite score
            signal.score = (
                signal.confidence * 0.3 +
                signal.risk_reward * 0.2 +
                signal.expected_return * 0.3 +
                (1 / signal.holding_period) * 0.2  # Prefer shorter holding periods
            )
        
        return sorted(signals, key=lambda x: x.score, reverse=True)
    
    def _apply_portfolio_constraints(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Apply portfolio-level constraints"""
        final_signals = []
        
        # Check current positions
        current_positions = len(self.portfolio.positions)
        available_slots = self.portfolio.risk_limits['position_limit'] - current_positions
        
        # Sector exposure
        sector_exposure = defaultdict(float)
        for symbol, position in self.portfolio.positions.items():
            sector = self.sector_map.get(symbol, 'Other')
            sector_exposure[sector] += position['value'] / self._get_portfolio_value()
        
        for signal in signals[:available_slots]:
            # Check sector exposure
            sector = self.sector_map.get(signal.symbol, 'Other')
            if sector_exposure[sector] + signal.position_size > self.portfolio.risk_limits['max_sector_exposure']:
                continue
            
            # Check correlation with existing positions
            if not self._check_correlation_constraint(signal.symbol):
                continue
            
            final_signals.append(signal)
            sector_exposure[sector] += signal.position_size
        
        return final_signals
    
    def _check_correlation_constraint(self, symbol: str) -> bool:
        """Check if adding symbol violates correlation constraints"""
        # Simplified - would calculate actual correlations
        return True
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(p['value'] for p in self.portfolio.positions.values())
        return self.portfolio.cash + positions_value
    
    async def execute_signals(self, signals: List[TradingSignal], paper_trade: bool = True):
        """Execute trading signals"""
        results = []
        
        for signal in signals:
            try:
                if paper_trade:
                    result = self._execute_paper_trade(signal)
                else:
                    result = await self._execute_live_trade(signal)
                
                results.append(result)
                
                # Update performance tracking
                self.performance_tracker.record_trade(result)
                
                logger.info(f"Executed: {signal.action} {signal.symbol} @ {signal.entry_price}")
                
            except Exception as e:
                logger.error(f"Execution error for {signal.symbol}: {e}")
        
        return results
    
    def _execute_paper_trade(self, signal: TradingSignal) -> Dict:
        """Execute paper trade"""
        # Calculate shares
        position_value = self.portfolio.cash * signal.position_size
        shares = int(position_value / signal.entry_price)
        
        if shares == 0:
            return {'success': False, 'reason': 'Position too small'}
        
        # Update portfolio
        self.portfolio.cash -= shares * signal.entry_price
        
        self.portfolio.positions[signal.symbol] = {
            'shares': shares,
            'entry_price': signal.entry_price,
            'current_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'value': shares * signal.entry_price,
            'unrealized_pnl': 0,
            'strategy': signal.strategy,
            'entry_time': datetime.now()
        }
        
        trade_record = {
            'success': True,
            'symbol': signal.symbol,
            'action': signal.action,
            'shares': shares,
            'price': signal.entry_price,
            'timestamp': datetime.now(),
            'strategy': signal.strategy
        }
        
        self.portfolio.trade_history.append(trade_record)
        
        return trade_record
    
    async def _execute_live_trade(self, signal: TradingSignal) -> Dict:
        """Execute live trade via broker API"""
        # Would implement actual broker API integration
        pass
    
    async def monitor_positions(self):
        """Monitor and update existing positions"""
        for symbol, position in list(self.portfolio.positions.items()):
            df = await self.fetch_market_data(symbol, period='1d')
            if df.empty:
                continue
            
            current_price = df['Close'].iloc[-1]
            position['current_price'] = current_price
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['shares']
            position['value'] = position['shares'] * current_price
            
            # Check stop loss
            if current_price <= position['stop_loss']:
                await self._close_position(symbol, current_price, 'Stop Loss Hit')
            
            # Check take profit
            elif current_price >= position['take_profit']:
                await self._close_position(symbol, current_price, 'Take Profit Hit')
            
            # Trailing stop adjustment
            elif position['unrealized_pnl'] > position['value'] * 0.05:  # 5% profit
                atr = df['ATR'].iloc[-1]
                new_stop = current_price - (atr * 1.2)
                position['stop_loss'] = max(position['stop_loss'], new_stop)
    
    async def _close_position(self, symbol: str, price: float, reason: str):
        """Close a position"""
        position = self.portfolio.positions[symbol]
        
        # Calculate P&L
        pnl = (price - position['entry_price']) * position['shares']
        
        # Update cash
        self.portfolio.cash += position['shares'] * price
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'action': 'CLOSE',
            'shares': position['shares'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'reason': reason,
            'timestamp': datetime.now(),
            'strategy': position['strategy']
        }
        
        self.portfolio.trade_history.append(trade_record)
        
        # Update strategy performance
        strategy = position['strategy']
        if pnl > 0:
            self.strategy_performance[strategy]['wins'] += 1
        else:
            self.strategy_performance[strategy]['losses'] += 1
        self.strategy_performance[strategy]['total_return'] += pnl / (position['entry_price'] * position['shares'])
        
        # Update ML with outcome if it was an ML trade
        if self.ml_initialized and 'ml_' in strategy:
            outcome = 'profitable' if pnl > 0 else 'loss'
            self.ml_engine.update_with_result(position.get('entry_time'), outcome)
        
        # Remove position
        del self.portfolio.positions[symbol]
        
        logger.info(f"Closed {symbol}: {reason} - P&L: ${pnl:.2f}")
    
    def get_portfolio_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        portfolio_value = self._get_portfolio_value()
        
        # Calculate returns
        trades = self.portfolio.trade_history
        if not trades:
            return {
                'total_value': portfolio_value,
                'cash': self.portfolio.cash,
                'positions': len(self.portfolio.positions),
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
        
        # Win rate
        closed_trades = [t for t in trades if t.get('pnl') is not None]
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        # Returns
        total_return = (portfolio_value - 100000) / 100000
        
        # Sharpe ratio (simplified)
        returns = [t['pnl'] / (t['entry_price'] * t['shares']) for t in closed_trades if 'entry_price' in t]
        if returns:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return {
            'total_value': portfolio_value,
            'cash': self.portfolio.cash,
            'positions': len(self.portfolio.positions),
            'open_positions': list(self.portfolio.positions.keys()),
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'strategy_performance': dict(self.strategy_performance)
        }

class PerformanceTracker:
    """Track and analyze trading performance"""
    
    def __init__(self):
        self.trades = []
        self.daily_returns = []
        self.equity_curve = [100000]  # Starting capital
    
    def record_trade(self, trade: Dict):
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict:
        if not self.trades:
            return {}
        
        # More detailed performance metrics would go here
        return {
            'total_trades': len(self.trades),
            'last_updated': datetime.now().isoformat()
        }

class RiskManager:
    """Manage portfolio risk"""
    
    def __init__(self, portfolio: PersonalPortfolio):
        self.portfolio = portfolio
    
    def check_risk_limits(self) -> Dict:
        """Check if any risk limits are breached"""
        warnings = []
        
        # Implementation would check all risk limits
        
        return {
            'warnings': warnings,
            'risk_score': self.calculate_risk_score()
        }
    
    def calculate_risk_score(self) -> float:
        """Calculate overall portfolio risk score (0-1)"""
        # Simplified risk scoring
        return 0.3

class SignalValidator:
    """Validate signals before execution"""
    
    def validate(self, signal: TradingSignal, portfolio: PersonalPortfolio, df: pd.DataFrame) -> bool:
        """Validate signal meets criteria"""
        
        # Check confidence threshold
        if signal.confidence < 0.6:
            return False
        
        # Check risk-reward ratio
        if signal.risk_reward < 1.5:
            return False
        
        # Check if already in position
        if signal.symbol in portfolio.positions:
            return False
        
        # Check liquidity (volume)
        avg_volume = df['Volume'].iloc[-20:].mean()
        if avg_volume < 1000000:  # Min 1M volume
            return False
        
        return True

async def main():
    """Run personal trading system - COMPLETELY ISOLATED from customer backend"""
    
    # Initialize system
    trader = PersonalTradingSystem()
    
    logger.info("=" * 50)
    logger.info("PERSONAL TRADING SYSTEM INITIALIZED")
    logger.info("System ID: PERSONAL_SYSTEM")
    logger.info("Backend: ISOLATED (No customer signal mixing)")
    logger.info("=" * 50)
    
    # Initialize ML Engine
    await trader._initialize_ml()
    
    # Check if market is closed (weekend/holiday)
    now = datetime.now()
    is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    if is_weekend:
        logger.info("Market closed (weekend) - Running in analysis/backtest mode")
        logger.info("Historical data and ML training still active")
    
    while True:
        try:
            # Generate signals (works with historical data on weekends)
            signals = await trader.generate_signals()
            
            if signals:
                logger.info(f"\nFound {len(signals)} trading signals:")
                for signal in signals:
                    logger.info(f"  {signal.action} {signal.symbol} @ ${signal.entry_price:.2f} "
                              f"(Confidence: {signal.confidence:.1%}, Strategy: {signal.strategy})")
                
                # Execute signals (paper trading)
                results = await trader.execute_signals(signals, paper_trade=True)
                
                for result in results:
                    if result['success']:
                        logger.info(f"  ✓ Executed: {result['symbol']}")
                        
                        # Update ML with trade execution
                        if trader.ml_initialized:
                            # Track for future outcome updates
                            trader.ml_engine.ml.record_prediction({
                                'timestamp': datetime.now(),
                                'symbol': result['symbol'],
                                'action': result['action'],
                                'strategy': result.get('strategy', 'unknown')
                            })
            
            # Monitor existing positions
            await trader.monitor_positions()
            
            # Update ML with closed position outcomes
            for trade in trader.portfolio.trade_history[-10:]:  # Check recent trades
                if trade.get('pnl') is not None and not trade.get('ml_updated'):
                    outcome = 'profitable' if trade['pnl'] > 0 else 'loss'
                    # Update ML if we have the prediction ID
                    trade['ml_updated'] = True
            
            # Display portfolio metrics
            metrics = trader.get_portfolio_metrics()
            logger.info(f"\nPortfolio Status:")
            logger.info(f"  Total Value: ${metrics['total_value']:,.2f}")
            logger.info(f"  Cash: ${metrics['cash']:,.2f}")
            logger.info(f"  Positions: {metrics['positions']}")
            logger.info(f"  Total Return: {metrics['total_return_pct']:.2f}%")
            logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
            
            # Display ML stats if initialized
            if trader.ml_initialized:
                ml_stats = trader.ml_engine.get_stats()
                if ml_stats['is_trained']:
                    logger.info(f"\nML Engine Status:")
                    logger.info(f"  Training Samples: {ml_stats['training_samples']}")
                    logger.info(f"  Total Predictions: {ml_stats['total_predictions']}")
                    if ml_stats.get('best_model'):
                        logger.info(f"  Best Model: {ml_stats['best_model']} ({ml_stats['best_accuracy']:.1%})")
            
            # Wait before next scan
            await asyncio.sleep(300 if not is_weekend else 600)  # 5 min normally, 10 min on weekends
            
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            break
        except Exception as e:
            logger.error(f"System error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())