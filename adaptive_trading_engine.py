"""
Adaptive All-Day Trading Engine
Performs in ALL market conditions, not just chaos periods
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from collections import deque
from typing import Dict, List, Optional, Tuple
import json
import logging

logger = logging.getLogger(__name__)

class AdaptiveTrader:
    """
    Multi-strategy trader that adapts to current market conditions
    Works all day, not just at open/close
    """
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret = os.getenv('ALPACA_SECRET')
        self.data_url = "https://data.alpaca.markets/v2"
        
        # Track performance of each strategy in real-time
        self.strategy_performance = {
            'opening_range': deque(maxlen=20),
            'mean_reversion': deque(maxlen=20),
            'trend_following': deque(maxlen=20),
            'range_trading': deque(maxlen=20),
            'momentum': deque(maxlen=20),
            'vwap_bounce': deque(maxlen=20),
            'news_catalyst': deque(maxlen=20),
            'sector_rotation': deque(maxlen=20),
            'pairs_trading': deque(maxlen=20),
            'lunch_scalp': deque(maxlen=20)
        }
        
        # Market regime detection
        self.market_regime = 'UNKNOWN'
        self.volatility_state = 'NORMAL'
        
    def detect_market_regime(self, spy_data: pd.DataFrame) -> str:
        """
        Identify current market regime to select appropriate strategies
        """
        if len(spy_data) < 50:
            return 'UNKNOWN'
        
        # Calculate various metrics
        returns = spy_data['Close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        trend = (spy_data['Close'].iloc[-1] - spy_data['Close'].iloc[-50]) / spy_data['Close'].iloc[-50]
        
        # Volume profile
        recent_volume = spy_data['Volume'].iloc[-10:].mean()
        avg_volume = spy_data['Volume'].iloc[-50:].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Determine regime
        if abs(trend) < 0.001 and volatility < 0.01:
            return 'RANGE_BOUND'
        elif trend > 0.005 and volatility < 0.015:
            return 'TRENDING_UP'
        elif trend < -0.005 and volatility < 0.015:
            return 'TRENDING_DOWN'
        elif volatility > 0.02:
            return 'HIGH_VOLATILITY'
        elif volume_ratio < 0.7:
            return 'LOW_VOLUME'
        else:
            return 'NORMAL'
    
    async def get_market_internals(self) -> Dict:
        """
        Get market-wide metrics to understand conditions
        """
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret
        }
        
        # Check major indices
        symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
        internals = {}
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                url = f"{self.data_url}/stocks/{symbol}/snapshot"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        internals[symbol] = {
                            'price': data['latestTrade']['p'],
                            'daily_change': data.get('prevDailyBar', {})
                        }
        
        # Calculate breadth
        if 'SPY' in internals and 'IWM' in internals:
            # Small caps vs large caps
            internals['breadth'] = 'POSITIVE' if internals['IWM']['price'] > internals['SPY']['price'] * 0.4 else 'NEGATIVE'
        
        return internals
    
    async def opening_range_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Classic ORB - works 9:30-10:30 AM
        """
        current_time = datetime.now().time()
        if not (time(9, 30) <= current_time <= time(10, 30)):
            return None
        
        if len(data) < 30:
            return None
        
        # Get first 5-minute range
        market_open_data = data.between_time('09:30', '09:35')
        if market_open_data.empty:
            return None
        
        opening_high = market_open_data['High'].max()
        opening_low = market_open_data['Low'].min()
        current_price = data['Close'].iloc[-1]
        
        # Breakout detection
        if current_price > opening_high * 1.001:
            return {
                'action': 'BUY',
                'confidence': 0.75,
                'strategy': 'opening_range',
                'stop': opening_low
            }
        elif current_price < opening_low * 0.999:
            return {
                'action': 'SELL',
                'confidence': 0.75,
                'strategy': 'opening_range',
                'stop': opening_high
            }
        
        return None
    
    async def mean_reversion_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Mean reversion - works best in range-bound markets
        Excellent for mid-day trading
        """
        if len(data) < 20:
            return None
        
        # Calculate Bollinger Bands
        sma20 = data['Close'].rolling(20).mean().iloc[-1]
        std20 = data['Close'].rolling(20).std().iloc[-1]
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        
        current_price = data['Close'].iloc[-1]
        
        # Calculate RSI
        price_diff = data['Close'].diff()
        gain = (price_diff.where(price_diff > 0, 0)).rolling(14).mean()
        loss = (-price_diff.where(price_diff < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # Generate signals
        if current_price < lower_band and rsi < 30:
            confidence = min(0.5 + (30 - rsi) / 100, 0.85)
            return {
                'action': 'BUY',
                'confidence': confidence,
                'strategy': 'mean_reversion',
                'target': sma20,
                'stop': current_price * 0.98
            }
        elif current_price > upper_band and rsi > 70:
            confidence = min(0.5 + (rsi - 70) / 100, 0.85)
            return {
                'action': 'SELL',
                'confidence': confidence,
                'strategy': 'mean_reversion',
                'target': sma20,
                'stop': current_price * 1.02
            }
        
        return None
    
    async def trend_following_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Trend following - works in trending markets
        Great for afternoon trends
        """
        if len(data) < 50:
            return None
        
        # Multiple timeframe analysis
        sma10 = data['Close'].rolling(10).mean().iloc[-1]
        sma20 = data['Close'].rolling(20).mean().iloc[-1]
        sma50 = data['Close'].rolling(50).mean().iloc[-1]
        
        current_price = data['Close'].iloc[-1]
        
        # MACD
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        macd_histogram = macd - signal_line
        
        # Trend strength
        if sma10 > sma20 > sma50 and macd_histogram.iloc[-1] > 0:
            # Strong uptrend
            momentum = (current_price - data['Close'].iloc[-10]) / data['Close'].iloc[-10]
            confidence = min(0.6 + momentum * 10, 0.85)
            
            return {
                'action': 'BUY',
                'confidence': confidence,
                'strategy': 'trend_following',
                'stop': sma20
            }
        elif sma10 < sma20 < sma50 and macd_histogram.iloc[-1] < 0:
            # Strong downtrend
            momentum = (data['Close'].iloc[-10] - current_price) / data['Close'].iloc[-10]
            confidence = min(0.6 + momentum * 10, 0.85)
            
            return {
                'action': 'SELL',
                'confidence': confidence,
                'strategy': 'trend_following',
                'stop': sma20
            }
        
        return None
    
    async def range_trading_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Range trading - perfect for lunch hours (11:30 AM - 2:00 PM)
        When volatility is low and price oscillates
        """
        current_time = datetime.now().time()
        
        if len(data) < 100:
            return None
        
        # Identify range
        recent_data = data.iloc[-100:]
        resistance = recent_data['High'].rolling(20).max().iloc[-1]
        support = recent_data['Low'].rolling(20).min().iloc[-1]
        range_size = (resistance - support) / support
        
        # Only trade if clear range exists (1-3% range)
        if range_size < 0.01 or range_size > 0.03:
            return None
        
        current_price = data['Close'].iloc[-1]
        position_in_range = (current_price - support) / (resistance - support)
        
        # Trade the range
        if position_in_range < 0.2:
            # Near support
            return {
                'action': 'BUY',
                'confidence': 0.70,
                'strategy': 'range_trading',
                'target': resistance * 0.98,
                'stop': support * 0.995
            }
        elif position_in_range > 0.8:
            # Near resistance
            return {
                'action': 'SELL',
                'confidence': 0.70,
                'strategy': 'range_trading',
                'target': support * 1.02,
                'stop': resistance * 1.005
            }
        
        return None
    
    async def vwap_bounce_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        VWAP bounce - institutional levels, works all day
        """
        if len(data) < 30:
            return None
        
        # Calculate VWAP
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        cumulative_tpv = (typical_price * data['Volume']).cumsum()
        cumulative_volume = data['Volume'].cumsum()
        vwap = cumulative_tpv / cumulative_volume
        
        current_price = data['Close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        distance_from_vwap = (current_price - current_vwap) / current_vwap
        
        # Look for VWAP tests with volume
        recent_volume = data['Volume'].iloc[-5:].mean()
        avg_volume = data['Volume'].mean()
        volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Price approaching VWAP with volume
        if abs(distance_from_vwap) < 0.002 and volume_surge > 1.3:
            # Determine direction based on approach
            price_direction = data['Close'].iloc[-1] - data['Close'].iloc[-5]
            
            if price_direction > 0:
                # Bouncing up from VWAP
                return {
                    'action': 'BUY',
                    'confidence': 0.65,
                    'strategy': 'vwap_bounce',
                    'stop': current_vwap * 0.995
                }
            else:
                # Rejecting from VWAP
                return {
                    'action': 'SELL',
                    'confidence': 0.65,
                    'strategy': 'vwap_bounce',
                    'stop': current_vwap * 1.005
                }
        
        return None
    
    async def lunch_scalp_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Lunch hour scalping - 11:30 AM - 1:30 PM
        Low volatility, small moves
        """
        current_time = datetime.now().time()
        if not (time(11, 30) <= current_time <= time(13, 30)):
            return None
        
        if len(data) < 60:
            return None
        
        # Look for micro-breakouts in low volatility
        recent_volatility = data['Close'].pct_change().iloc[-30:].std()
        
        # Only trade in very low volatility
        if recent_volatility > 0.003:
            return None
        
        # 15-minute high/low
        recent_high = data['High'].iloc[-15:].max()
        recent_low = data['Low'].iloc[-15:].min()
        current_price = data['Close'].iloc[-1]
        
        # Micro breakouts
        if current_price > recent_high:
            return {
                'action': 'BUY',
                'confidence': 0.60,
                'strategy': 'lunch_scalp',
                'target': current_price * 1.002,  # 0.2% target
                'stop': recent_low
            }
        elif current_price < recent_low:
            return {
                'action': 'SELL',
                'confidence': 0.60,
                'strategy': 'lunch_scalp',
                'target': current_price * 0.998,  # 0.2% target
                'stop': recent_high
            }
        
        return None
    
    async def momentum_continuation(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Momentum trading - works any time there's movement
        """
        if len(data) < 30:
            return None
        
        # Calculate rate of change
        roc_5 = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
        roc_10 = (data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].iloc[-10]
        
        # Volume confirmation
        recent_volume = data['Volume'].iloc[-5:].mean()
        avg_volume = data['Volume'].mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Strong momentum with volume
        if roc_5 > 0.003 and roc_10 > 0.005 and volume_ratio > 1.5:
            confidence = min(0.6 + roc_5 * 50, 0.85)
            return {
                'action': 'BUY',
                'confidence': confidence,
                'strategy': 'momentum_continuation',
                'stop': data['Low'].iloc[-5:].min()
            }
        elif roc_5 < -0.003 and roc_10 < -0.005 and volume_ratio > 1.5:
            confidence = min(0.6 + abs(roc_5) * 50, 0.85)
            return {
                'action': 'SELL',
                'confidence': confidence,
                'strategy': 'momentum_continuation',
                'stop': data['High'].iloc[-5:].max()
            }
        
        return None
    
    async def sector_rotation_strategy(self, symbols: List[str], market_data: Dict) -> Optional[Dict]:
        """
        Detect sector rotation and trade the leaders/laggards
        """
        # Compare sector ETFs
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples'
        }
        
        sector_strength = {}
        for sector_etf, sector_name in sectors.items():
            if sector_etf in market_data:
                data = market_data[sector_etf]
                if len(data) > 20:
                    # 1-hour momentum
                    momentum = (data['Close'].iloc[-1] - data['Close'].iloc[-60]) / data['Close'].iloc[-60] if len(data) > 60 else 0
                    sector_strength[sector_name] = momentum
        
        if not sector_strength:
            return None
        
        # Find strongest and weakest
        strongest = max(sector_strength, key=sector_strength.get)
        weakest = min(sector_strength, key=sector_strength.get)
        
        # Trade sector rotation
        if sector_strength[strongest] > 0.01 and sector_strength[weakest] < -0.005:
            return {
                'action': 'SECTOR_ROTATION',
                'buy': strongest,
                'sell': weakest,
                'confidence': 0.70,
                'strategy': 'sector_rotation'
            }
        
        return None
    
    async def pairs_trading_strategy(self, pair: Tuple[str, str], data1: pd.DataFrame, data2: pd.DataFrame) -> Optional[Dict]:
        """
        Statistical arbitrage between correlated pairs
        Works all day when pairs diverge
        """
        if len(data1) < 100 or len(data2) < 100:
            return None
        
        # Calculate spread
        ratio = data1['Close'] / data2['Close']
        mean_ratio = ratio.rolling(50).mean().iloc[-1]
        std_ratio = ratio.rolling(50).std().iloc[-1]
        current_ratio = ratio.iloc[-1]
        
        # Z-score
        z_score = (current_ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
        
        # Trade extreme divergences
        if z_score > 2:
            # Pair 1 expensive relative to pair 2
            return {
                'action': 'PAIRS',
                'sell': pair[0],
                'buy': pair[1],
                'confidence': min(0.6 + abs(z_score) * 0.1, 0.85),
                'strategy': 'pairs_trading',
                'z_score': z_score
            }
        elif z_score < -2:
            # Pair 1 cheap relative to pair 2
            return {
                'action': 'PAIRS',
                'buy': pair[0],
                'sell': pair[1],
                'confidence': min(0.6 + abs(z_score) * 0.1, 0.85),
                'strategy': 'pairs_trading',
                'z_score': z_score
            }
        
        return None
    
    async def select_best_strategy(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        """
        Intelligently select the best strategy for current conditions
        """
        # Detect market regime
        market_regime = self.detect_market_regime(data)
        current_time = datetime.now().time()
        
        # Get signals from all applicable strategies
        signals = []
        
        # Time-based strategies
        if time(9, 30) <= current_time <= time(10, 30):
            signal = await self.opening_range_strategy(symbol, data)
            if signal:
                signals.append(signal)
        
        if time(11, 30) <= current_time <= time(13, 30):
            signal = await self.lunch_scalp_strategy(symbol, data)
            if signal:
                signals.append(signal)
        
        # Regime-based strategies
        if market_regime == 'RANGE_BOUND':
            for strategy in [self.mean_reversion_strategy, self.range_trading_strategy]:
                signal = await strategy(symbol, data)
                if signal:
                    signals.append(signal)
        
        elif market_regime in ['TRENDING_UP', 'TRENDING_DOWN']:
            for strategy in [self.trend_following_strategy, self.momentum_continuation]:
                signal = await strategy(symbol, data)
                if signal:
                    signals.append(signal)
        
        # Always-on strategies
        for strategy in [self.vwap_bounce_strategy]:
            signal = await strategy(symbol, data)
            if signal:
                signals.append(signal)
        
        # Select highest confidence signal
        if signals:
            best_signal = max(signals, key=lambda x: x['confidence'])
            
            # Boost confidence if multiple strategies agree
            agreeing = sum(1 for s in signals if s['action'] == best_signal['action'])
            if agreeing > 1:
                best_signal['confidence'] = min(best_signal['confidence'] * 1.1, 0.95)
                best_signal['agreement'] = agreeing
            
            # Add market context
            best_signal['market_regime'] = market_regime
            best_signal['symbol'] = symbol
            best_signal['price'] = data['Close'].iloc[-1]
            best_signal['timestamp'] = datetime.now().isoformat()
            
            return best_signal
        
        return None
    
    async def scan_all_opportunities(self, watchlist: List[str]) -> List[Dict]:
        """
        Scan entire watchlist for opportunities using all strategies
        """
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret
        }
        
        opportunities = []
        
        for symbol in watchlist:
            try:
                # Get data
                params = {
                    'symbols': symbol,
                    'timeframe': '1Min',
                    'start': (datetime.now() - timedelta(hours=8)).isoformat() + 'Z',
                    'limit': 500
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.data_url}/stocks/bars", headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'bars' in data and symbol in data['bars']:
                                df = pd.DataFrame(data['bars'][symbol])
                                df['t'] = pd.to_datetime(df['t'])
                                df.set_index('t', inplace=True)
                                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'TradeCount', 'VWAP']
                                
                                # Get best signal for this stock
                                signal = await self.select_best_strategy(symbol, df)
                                if signal and signal['confidence'] > 0.65:
                                    opportunities.append(signal)
            
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return opportunities[:10]  # Top 10 opportunities