import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yfinance as yf
import json
from dataclasses import dataclass, asdict
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Signal:
    symbol: str
    action: str
    price: float
    confidence: float
    risk_score: float
    strategy: str
    explanation: str
    rsi: float
    volume_ratio: float
    momentum: float
    timestamp: str
    potential_return: float
    stop_loss: float
    take_profit: float

class AdvancedTradingEngine:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://paper-api.alpaca.markets"
        
        # Performance tracking
        self.signal_history = deque(maxlen=1000)
        self.win_rate = 0.0
        self.total_return = 0.0
        
        # Strategy weights (will be optimized over time)
        self.strategy_weights = {
            'trend_following': 0.25,
            'mean_reversion': 0.20,
            'momentum': 0.20,
            'volume_breakout': 0.15,
            'support_resistance': 0.20
        }
        
    async def get_account_info(self) -> Dict:
        """Get Alpaca account information"""
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v2/account", headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logging.error(f"Account API error: {resp.status}")
                        return None
        except Exception as e:
            logging.error(f"Account fetch error: {e}")
            return None
            
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # Price-based indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / df['BB_width']
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Support and Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        df['SR_position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
        
        # Momentum
        df['Momentum_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
        df['Momentum_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
        
        return df.fillna(method='ffill').fillna(0)
        
    def trend_following_strategy(self, df: pd.DataFrame) -> Dict:
        """Trend following using multiple timeframes"""
        latest = df.iloc[-1]
        score = 0
        
        # Trend alignment
        if latest['Close'] > latest['SMA_20']: score += 2
        if latest['SMA_20'] > latest['SMA_50']: score += 2
        if latest['MACD'] > latest['MACD_signal']: score += 1
        if latest['EMA_12'] > latest['EMA_26']: score += 1
        
        # Trend strength
        trend_strength = abs(latest['Close'] - latest['SMA_20']) / latest['SMA_20']
        if trend_strength > 0.02: score += 1
        
        # Volume confirmation
        if latest['Volume_ratio'] > 1.2: score += 1
        
        if score >= 6:
            return {'action': 'BUY', 'confidence': min(score / 8, 0.95), 'score': score}
        elif score <= 2:
            return {'action': 'SELL', 'confidence': min((8 - score) / 8, 0.95), 'score': score}
        else:
            return {'action': 'HOLD', 'confidence': 0.3, 'score': score}
            
    def mean_reversion_strategy(self, df: pd.DataFrame) -> Dict:
        """Mean reversion using Bollinger Bands and RSI"""
        latest = df.iloc[-1]
        
        # Check for oversold/overbought
        if latest['RSI'] < 30 and latest['BB_position'] < 0.2:
            confidence = (30 - latest['RSI']) / 30 + (0.2 - latest['BB_position'])
            return {'action': 'BUY', 'confidence': min(confidence, 0.9)}
        elif latest['RSI'] > 70 and latest['BB_position'] > 0.8:
            confidence = (latest['RSI'] - 70) / 30 + (latest['BB_position'] - 0.8)
            return {'action': 'SELL', 'confidence': min(confidence, 0.9)}
        else:
            return {'action': 'HOLD', 'confidence': 0.3}
            
    def momentum_strategy(self, df: pd.DataFrame) -> Dict:
        """Momentum based on price acceleration"""
        latest = df.iloc[-1]
        
        # Check momentum
        if latest['Momentum_5'] > 0.03 and latest['Momentum_10'] > 0.02:
            if latest['RSI'] < 70:  # Not overbought
                confidence = min(latest['Momentum_5'] * 10, 0.9)
                return {'action': 'BUY', 'confidence': confidence}
        elif latest['Momentum_5'] < -0.03 and latest['Momentum_10'] < -0.02:
            if latest['RSI'] > 30:  # Not oversold
                confidence = min(abs(latest['Momentum_5']) * 10, 0.9)
                return {'action': 'SELL', 'confidence': confidence}
                
        return {'action': 'HOLD', 'confidence': 0.3}
        
    def volume_breakout_strategy(self, df: pd.DataFrame) -> Dict:
        """Volume-based breakout detection"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Volume spike with price movement
        if latest['Volume_ratio'] > 2.0:
            price_change = (latest['Close'] - prev['Close']) / prev['Close']
            if price_change > 0.01:
                return {'action': 'BUY', 'confidence': min(latest['Volume_ratio'] / 3, 0.9)}
            elif price_change < -0.01:
                return {'action': 'SELL', 'confidence': min(latest['Volume_ratio'] / 3, 0.9)}
                
        return {'action': 'HOLD', 'confidence': 0.3}
        
    def support_resistance_strategy(self, df: pd.DataFrame) -> Dict:
        """Support and resistance trading"""
        latest = df.iloc[-1]
        
        # Near support - potential bounce
        if latest['SR_position'] < 0.2 and latest['RSI'] < 40:
            return {'action': 'BUY', 'confidence': 0.7}
        # Near resistance - potential reversal
        elif latest['SR_position'] > 0.8 and latest['RSI'] > 60:
            return {'action': 'SELL', 'confidence': 0.7}
            
        return {'action': 'HOLD', 'confidence': 0.3}
        
    def combine_strategies(self, results: Dict) -> Dict:
        """Combine all strategy signals with weighted voting"""
        buy_score = 0
        sell_score = 0
        
        for strategy, result in results.items():
            weight = self.strategy_weights.get(strategy, 0.2)
            if result['action'] == 'BUY':
                buy_score += result['confidence'] * weight
            elif result['action'] == 'SELL':
                sell_score += result['confidence'] * weight
                
        # Determine final action
        if buy_score > sell_score and buy_score > 0.4:
            return {
                'action': 'BUY',
                'confidence': min(buy_score, 0.95),
                'dominant_strategy': max(results.items(), key=lambda x: x[1]['confidence'])[0]
            }
        elif sell_score > buy_score and sell_score > 0.4:
            return {
                'action': 'SELL',
                'confidence': min(sell_score, 0.95),
                'dominant_strategy': max(results.items(), key=lambda x: x[1]['confidence'])[0]
            }
        else:
            return {
                'action': 'HOLD',
                'confidence': 0.3,
                'dominant_strategy': 'mixed'
            }
            
    def calculate_risk_metrics(self, df: pd.DataFrame, action: str, price: float) -> Dict:
        """Calculate stop loss, take profit, and risk score"""
        atr = df['ATR'].iloc[-1]
        
        if action == 'BUY':
            stop_loss = price - (atr * 1.5)
            take_profit = price + (atr * 2.5)
            potential_return = ((take_profit - price) / price) * 100
        else:  # SELL
            stop_loss = price + (atr * 1.5)
            take_profit = price - (atr * 2.5)
            potential_return = ((price - take_profit) / price) * 100
            
        # Risk score based on volatility and market conditions
        volatility = atr / price
        risk_score = min(volatility * 10, 1.0)
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'potential_return': round(potential_return, 2),
            'risk_score': round(risk_score, 3)
        }
        
    async def analyze_stock(self, symbol: str) -> Optional[Signal]:
        """Complete analysis of a single stock"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2mo", interval="1h")
            
            if len(df) < 50:
                return None
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Run all strategies
            strategies = {
                'trend_following': self.trend_following_strategy(df),
                'mean_reversion': self.mean_reversion_strategy(df),
                'momentum': self.momentum_strategy(df),
                'volume_breakout': self.volume_breakout_strategy(df),
                'support_resistance': self.support_resistance_strategy(df)
            }
            
            # Combine signals
           # Combine signals
            final_signal = self.combine_strategies(strategies)
            
            # ML Enhancement (optional)
            try:
                from ml_brain import ml_brain
        
        # Prepare market data for ML
        ml_market_data = {
            'rsi': df['RSI'].iloc[-1],
            'macd': df['MACD'].iloc[-1],
            'volume_ratio': df['Volume_ratio'].iloc[-1],
            'momentum_5': df['Momentum_5'].iloc[-1],
            'momentum_10': df['Momentum_10'].iloc[-1],
            'sma_20': df['SMA_20'].iloc[-1],
            'sma_50': df['SMA_50'].iloc[-1],
            'atr': df['ATR'].iloc[-1],
            'volatility': df['ATR'].iloc[-1] / df['Close'].iloc[-1]
        }
        
        # Get ML prediction
        ml_prediction = ml_brain.predict(ml_market_data)
        
        # If ML has high confidence, override traditional signal
        if ml_prediction['ml_powered'] and ml_prediction['confidence'] > 0.8:
            final_signal['action'] = ml_prediction['action']
            final_signal['confidence'] = ml_prediction['confidence']
            final_signal['dominant_strategy'] = 'ML_' + final_signal['dominant_strategy']
            
            logging.info(f"ML Override for {symbol}: {ml_prediction['action']} ({ml_prediction['confidence']:.2f})")
    except Exception as e:
        logging.warning(f"ML prediction failed: {e}")
            if final_signal['action'] == 'HOLD':
                return None
                
            # Get current price and calculate risk metrics
            current_price = df['Close'].iloc[-1]
            risk_metrics = self.calculate_risk_metrics(df, final_signal['action'], current_price)
            
            # Create explanation
            explanation = self.generate_explanation(
                final_signal['action'],
                final_signal['dominant_strategy'],
                df.iloc[-1],
                final_signal['confidence']
            )
            
            # Create signal object
            signal = Signal(
                symbol=symbol,
                action=final_signal['action'],
                price=round(current_price, 2),
                confidence=round(final_signal['confidence'], 3),
                risk_score=risk_metrics['risk_score'],
                strategy=final_signal['dominant_strategy'],
                explanation=explanation,
                rsi=round(df['RSI'].iloc[-1], 1),
                volume_ratio=round(df['Volume_ratio'].iloc[-1], 2),
                momentum=round(df['Momentum_5'].iloc[-1] * 100, 2),
                timestamp=datetime.now().isoformat(),
                potential_return=risk_metrics['potential_return'],
                stop_loss=risk_metrics['stop_loss'],
                take_profit=risk_metrics['take_profit']
            )
            
            # Track signal
            self.signal_history.append(signal)
            
            return signal
            
        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {e}")
            return None
            
    def generate_explanation(self, action: str, strategy: str, latest_data: pd.Series, confidence: float) -> str:
        """Generate human-readable explanation for signal"""
        explanations = []
        
        if action == 'BUY':
            if latest_data['RSI'] < 30:
                explanations.append(f"Oversold (RSI: {latest_data['RSI']:.1f})")
            if latest_data['Close'] > latest_data['SMA_20']:
                explanations.append("Price above 20-day average")
            if latest_data['Volume_ratio'] > 1.5:
                explanations.append(f"High volume ({latest_data['Volume_ratio']:.1f}x average)")
            if latest_data['Momentum_5'] > 0.02:
                explanations.append(f"Strong momentum ({latest_data['Momentum_5']*100:.1f}%)")
                
        elif action == 'SELL':
            if latest_data['RSI'] > 70:
                explanations.append(f"Overbought (RSI: {latest_data['RSI']:.1f})")
            if latest_data['Close'] < latest_data['SMA_20']:
                explanations.append("Price below 20-day average")
            if latest_data['BB_position'] > 0.9:
                explanations.append("At upper Bollinger Band")
            if latest_data['Momentum_5'] < -0.02:
                explanations.append(f"Negative momentum ({latest_data['Momentum_5']*100:.1f}%)")
                
        explanation = f"{action} signal from {strategy} strategy ({confidence*100:.0f}% confidence). "
        explanation += "Key factors: " + ", ".join(explanations[:3]) if explanations else "Multiple technical confirmations"
        
        return explanation
        
    async def scan_stocks(self, symbols: List[str]) -> List[Signal]:
        """Scan multiple stocks for signals"""
        signals = []
        
        for symbol in symbols:
            logging.info(f"Analyzing {symbol}...")
            signal = await self.analyze_stock(symbol)
            if signal:
                signals.append(signal)
                
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Calculate win rate if we have history
        if len(self.signal_history) > 20:
            self.calculate_performance()
            
        return signals
        
    def calculate_performance(self):
        """Calculate historical performance metrics"""
        # This would track actual performance in production
        # For now, simulate based on historical signals
        recent_signals = list(self.signal_history)[-100:]
        wins = sum(1 for s in recent_signals if s.confidence > 0.7)
        self.win_rate = wins / len(recent_signals) if recent_signals else 0
        
    async def execute_trade(self, signal: Signal, quantity: int) -> Dict:
        """Execute trade via Alpaca API"""
        headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.api_secret,
            'Content-Type': 'application/json'
        }
        
        order_data = {
            'symbol': signal.symbol,
            'qty': quantity,
            'side': 'buy' if signal.action == 'BUY' else 'sell',
            'type': 'limit',
            'limit_price': signal.price,
            'time_in_force': 'day',
            'order_class': 'bracket',
            'stop_loss': {'stop_price': signal.stop_loss},
            'take_profit': {'limit_price': signal.take_profit}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v2/orders",
                    headers=headers,
                    json=order_data
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logging.info(f"Trade executed: {signal.action} {quantity} {signal.symbol} @ ${signal.price}")
                        return {'success': True, 'order': result}
                    else:
                        error = await resp.text()
                        logging.error(f"Trade failed: {error}")
                        return {'success': False, 'error': error}
        except Exception as e:
            logging.error(f"Execution error: {e}")
            return {'success': False, 'error': str(e)}