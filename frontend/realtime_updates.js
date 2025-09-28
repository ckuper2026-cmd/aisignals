// PHOTON PLATFORM - REAL-TIME UPDATES ENHANCEMENT
// Add this to your index.html or include as a separate JS file

// Enhanced WebSocket connection with real-time updates
class RealTimeTrading {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.wsUrl = apiUrl.replace('https', 'wss') + '/ws';
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 3000;
        this.updateCallbacks = new Map();
        this.lastUpdate = Date.now();
        this.autoRefreshInterval = null;
    }

    // Initialize real-time connection
    init() {
        this.connect();
        this.setupAutoRefresh();
        this.requestNotificationPermission();
    }

    // Establish WebSocket connection
    connect() {
        console.log(`🔌 Connecting to WebSocket: ${this.wsUrl}`);
        
        this.ws = new WebSocket(this.wsUrl);
        
        this.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            this.reconnectAttempts = 0;
            this.updateUI('status', 'connected');
            
            // Request immediate data
            this.requestUpdate();
            
            // Send heartbeat every 30 seconds
            this.startHeartbeat();
        };
        
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
                this.lastUpdate = Date.now();
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            this.updateUI('status', 'error');
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.stopHeartbeat();
            this.handleReconnect();
        };
    }

    // Handle incoming WebSocket messages
    handleMessage(data) {
        console.log(`📨 Message type: ${data.type}`);
        
        switch(data.type) {
            case 'signals_update':
                this.handleSignalsUpdate(data);
                break;
                
            case 'auto_trade_executed':
                this.handleTradeExecuted(data.trade);
                break;
                
            case 'position_update':
                this.handlePositionUpdate(data);
                break;
                
            case 'account_update':
                this.handleAccountUpdate(data);
                break;
                
            case 'market_status':
                this.handleMarketStatus(data);
                break;
                
            case 'ping':
                // Server heartbeat
                break;
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    // Handle signal updates
    handleSignalsUpdate(data) {
        if (data.data && data.data.length > 0) {
            // Update signals display
            this.updateUI('signals', data.data);
            
            // Update metrics
            if (data.auto_trades_today !== undefined) {
                this.updateUI('autoTrades', data.auto_trades_today);
            }
            
            // Flash indicator
            this.flashIndicator('signals');
        }
        
        // Update market status
        if (data.market_open !== undefined) {
            this.updateUI('marketOpen', data.market_open);
        }
    }

    // Handle trade execution
    handleTradeExecuted(trade) {
        console.log('🎯 Trade executed:', trade);
        
        // Add trade notification
        this.addTradeNotification(trade);
        
        // Update trade history
        this.updateUI('tradeHistory', trade);
        
        // Refresh positions and account
        this.fetchPositions();
        this.fetchAccount();
        
        // Show desktop notification
        this.showDesktopNotification({
            title: 'Trade Executed!',
            body: `${trade.order.action} ${trade.order.quantity} ${trade.order.symbol} @ $${trade.order.price}`,
            tag: trade.trade_id
        });
        
        // Play sound
        this.playSound('trade');
        
        // Animate elements
        this.animateElement('autoTradesCount');
        this.animateElement('positionCount');
    }

    // Add trade notification to UI
    addTradeNotification(trade) {
        const container = document.getElementById('tradeNotifications');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = 'trade-notification glass rounded-lg p-3 mb-2 border-l-4 border-green-500 animate-slideIn';
        notification.innerHTML = `
            <div class="flex justify-between items-center">
                <div>
                    <span class="text-green-400 font-semibold">
                        <i class="fas fa-robot animate-pulse"></i> AUTO-TRADE EXECUTED
                    </span>
                    <div class="text-sm mt-1">
                        ${trade.order.action} ${trade.order.quantity} ${trade.order.symbol} @ $${trade.order.price}
                    </div>
                </div>
                <div class="text-right text-sm">
                    <div class="text-zinc-400">Confidence: ${(trade.signal_confidence * 100).toFixed(0)}%</div>
                    <div class="text-zinc-500 text-xs">${new Date().toLocaleTimeString()}</div>
                </div>
            </div>
        `;
        
        container.insertBefore(notification, container.firstChild);
        
        // Remove after 10 seconds
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 500);
        }, 10000);
        
        // Keep only last 5 notifications
        while (container.children.length > 5) {
            container.removeChild(container.lastChild);
        }
    }

    // Update UI elements
    updateUI(type, data) {
        switch(type) {
            case 'status':
                const statusEl = document.getElementById('connectionStatus');
                if (statusEl) {
                    const icons = {
                        connected: '<i class="fas fa-wifi text-green-400"></i> Live',
                        reconnecting: '<i class="fas fa-wifi text-yellow-400"></i> Reconnecting...',
                        error: '<i class="fas fa-wifi text-red-400"></i> Offline'
                    };
                    statusEl.innerHTML = icons[data] || icons.error;
                }
                break;
                
            case 'signals':
                this.updateSignalsDisplay(data);
                break;
                
            case 'autoTrades':
                const elements = ['autoTradeCount', 'autoTradesCount'];
                elements.forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        el.textContent = id === 'autoTradeCount' ? `${data} trades today` : data;
                        this.animateElement(id);
                    }
                });
                break;
                
            case 'marketOpen':
                const marketEl = document.getElementById('marketStatus');
                if (marketEl) {
                    if (data) {
                        marketEl.innerHTML = 'MARKET OPEN';
                        marketEl.className = 'px-3 py-1 bg-green-500 text-black text-xs font-medium rounded-full animate-pulse';
                    } else {
                        marketEl.innerHTML = 'MARKET CLOSED';
                        marketEl.className = 'px-3 py-1 bg-zinc-900 text-xs font-medium rounded-full border border-zinc-800';
                    }
                }
                break;
        }
    }

    // Update signals display with real-time data
    updateSignalsDisplay(signals) {
        const container = document.getElementById('signalsContainer');
        if (!container || !signals) return;
        
        container.innerHTML = signals.map(signal => `
            <div class="glass rounded-lg p-3 border-l-4 ${signal.action === 'BUY' ? 'border-green-500' : 'border-red-500'}
                 ${signal.confidence >= 0.70 ? 'bg-green-900 bg-opacity-10 pulse-glow' : ''}">
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <div class="flex items-center space-x-3 mb-2">
                            <span class="text-xl font-bold">${signal.symbol}</span>
                            <span class="px-2 py-1 ${signal.action === 'BUY' ? 'bg-green-500' : 'bg-red-500'} 
                                   text-black rounded-full font-bold text-xs">
                                ${signal.action}
                            </span>
                            <span class="text-zinc-400">$${signal.price}</span>
                            ${signal.confidence >= 0.70 ? 
                                '<span class="text-green-400 text-xs animate-pulse"><i class="fas fa-robot"></i> AUTO</span>' : ''}
                        </div>
                        <div class="text-xs text-zinc-300 mb-1">${signal.explanation || 'AI detected opportunity'}</div>
                        <div class="grid grid-cols-4 gap-2 text-xs">
                            <div>
                                <span class="text-zinc-500">Conf:</span>
                                <span class="font-bold ${signal.confidence >= 0.7 ? 'text-green-400' : 'text-zinc-400'}">
                                    ${(signal.confidence * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div>
                                <span class="text-zinc-500">Target:</span>
                                <span class="font-bold text-green-400">$${signal.take_profit}</span>
                            </div>
                            <div>
                                <span class="text-zinc-500">Stop:</span>
                                <span class="font-bold text-red-400">$${signal.stop_loss}</span>
                            </div>
                            <div>
                                <span class="text-zinc-500">Return:</span>
                                <span class="font-bold">${signal.potential_return}%</span>
                            </div>
                        </div>
                    </div>
                    <button onclick="manualExecute('${signal.symbol}', '${signal.action}', ${signal.price})" 
                            class="ml-3 px-3 py-1 text-xs ${signal.action === 'BUY' ? 'bg-green-500 hover:bg-green-600' : 'bg-red-500 hover:bg-red-600'} 
                                   text-black rounded font-semibold transition">
                        Manual
                    </button>
                </div>
            </div>
        `).join('');
        
        // Update count
        document.getElementById('activeSignalsCount').textContent = signals.length;
        
        // Flash indicator
        this.flashIndicator('signals');
    }

    // Fetch positions with real-time updates
    async fetchPositions() {
        try {
            const userId = localStorage.getItem('user_id');
            if (!userId) return;
            
            const response = await fetch(`${this.apiUrl}/api/user-positions/${userId}`);
            const data = await response.json();
            
            if (data.positions) {
                this.updatePositionsDisplay(data.positions);
                this.updatePLDisplay(data.total_pl);
            }
        } catch (error) {
            console.error('Error fetching positions:', error);
        }
    }

    // Update positions display
    updatePositionsDisplay(positions) {
        const container = document.getElementById('positionsContainer');
        if (!container) return;
        
        if (!positions || positions.length === 0) {
            container.innerHTML = '<p class="text-zinc-500 text-sm">No open positions</p>';
            return;
        }
        
        container.innerHTML = `
            <div class="space-y-2">
                ${positions.map(pos => `
                    <div class="flex justify-between items-center p-2 bg-zinc-900 rounded ${pos.unrealized_pl >= 0 ? 'border-l-2 border-green-500' : 'border-l-2 border-red-500'}">
                        <div>
                            <span class="font-semibold">${pos.symbol}</span>
                            <span class="text-sm text-zinc-500 ml-2">
                                ${pos.qty} @ $${pos.avg_entry_price}
                            </span>
                        </div>
                        <div class="text-right">
                            <div class="font-semibold ${pos.unrealized_pl >= 0 ? 'text-green-400' : 'text-red-400'}">
                                ${pos.unrealized_pl >= 0 ? '+' : ''}$${Math.abs(pos.unrealized_pl).toFixed(2)}
                            </div>
                            <div class="text-xs text-zinc-500">
                                ${pos.unrealized_plpc >= 0 ? '+' : ''}${pos.unrealized_plpc.toFixed(2)}%
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        // Update count
        document.getElementById('positionCount').textContent = positions.length;
        
        // Animate if changed
        this.animateElement('positionsContainer');
    }

    // Update P&L display
    updatePLDisplay(totalPL) {
        const element = document.getElementById('totalPL');
        if (!element) return;
        
        const formatted = `${totalPL >= 0 ? '+' : ''}$${Math.abs(totalPL).toFixed(2)}`;
        element.textContent = formatted;
        element.className = `text-xl font-semibold ${totalPL >= 0 ? 'text-green-400' : 'text-red-400'}`;
        
        // Animate on change
        this.animateElement('totalPL');
    }

    // Flash indicator for updates
    flashIndicator(type) {
        const indicators = {
            signals: 'signalsContainer',
            trades: 'tradeHistory',
            positions: 'positionsContainer'
        };
        
        const element = document.getElementById(indicators[type]);
        if (element) {
            element.style.borderColor = '#10b981';
            setTimeout(() => {
                element.style.borderColor = '';
            }, 500);
        }
    }

    // Animate element
    animateElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.add('animate-pulse');
            setTimeout(() => {
                element.classList.remove('animate-pulse');
            }, 1000);
        }
    }

    // Play notification sound
    playSound(type) {
        try {
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn');
            audio.volume = 0.3;
            audio.play().catch(() => {});
        } catch (e) {
            // Silent fail
        }
    }

    // Show desktop notification
    showDesktopNotification(options) {
        if ('Notification' in window && Notification.permission === 'granted') {
            const notification = new Notification(options.title, {
                body: options.body,
                icon: '/favicon.ico',
                tag: options.tag || 'trade',
                requireInteraction: false
            });
            
            notification.onclick = () => {
                window.focus();
                notification.close();
            };
            
            setTimeout(() => notification.close(), 5000);
        }
    }

    // Request notification permission
    requestNotificationPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }

    // Send heartbeat to keep connection alive
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'heartbeat' }));
            }
        }, 30000);
    }

    // Stop heartbeat
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    // Request immediate update from server
    requestUpdate() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'request_update' }));
        }
    }

    // Handle reconnection
    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(this.reconnectDelay * this.reconnectAttempts, 30000);
            console.log(`Reconnecting in ${delay/1000}s... (attempt ${this.reconnectAttempts})`);
            this.updateUI('status', 'reconnecting');
            setTimeout(() => this.connect(), delay);
        } else {
            console.error('Max reconnection attempts reached');
            this.updateUI('status', 'error');
        }
    }

    // Setup auto-refresh for positions and account
    setupAutoRefresh() {
        // Fetch positions every 10 seconds
        this.autoRefreshInterval = setInterval(() => {
            if (Date.now() - this.lastUpdate > 60000) {
                // If no update for 1 minute, force refresh
                this.fetchPositions();
                this.fetchAccount();
            }
        }, 10000);
    }

    // Fetch account status
    async fetchAccount() {
        try {
            const userId = localStorage.getItem('user_id');
            if (!userId) return;
            
            const response = await fetch(`${this.apiUrl}/api/alpaca-account/${userId}`);
            const data = await response.json();
            
            if (data.account_number) {
                this.updateAccountDisplay(data);
            }
        } catch (error) {
            console.error('Error fetching account:', error);
        }
    }

    // Update account display
    updateAccountDisplay(account) {
        const fields = {
            'buyingPower': account.buying_power,
            'portfolioValue': account.portfolio_value,
            'positionsValue': account.positions_value || 0
        };
        
        Object.entries(fields).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = `$${this.formatNumber(value)}`;
                this.animateElement(id);
            }
        });
    }

    // Format number with commas
    formatNumber(num) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(num);
    }

    // Cleanup on disconnect
    cleanup() {
        this.stopHeartbeat();
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
        }
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Initialize real-time trading when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const API_URL = 'https://aisignals-production.up.railway.app';  // Update with your Railway URL
    const realTimeTrading = new RealTimeTrading(API_URL);
    realTimeTrading.init();
    
    // Make it globally accessible for debugging
    window.realTimeTrading = realTimeTrading;
    
    console.log('🚀 Real-time trading initialized');
});

// CSS animations to add to your stylesheet
const animationStyles = `
<style>
@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes pulse-glow {
    0%, 100% {
        box-shadow: 0 0 5px rgba(34, 197, 94, 0.5);
    }
    50% {
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.8);
    }
}

.animate-slideIn {
    animation: slideIn 0.5s ease-out;
}

.pulse-glow {
    animation: pulse-glow 2s ease-in-out infinite;
}

.fade-out {
    opacity: 0;
    transition: opacity 0.5s ease-out;
}

/* Real-time update indicators */
.update-flash {
    border-color: #10b981 !important;
    transition: border-color 0.3s ease;
}

/* Trade notification styles */
.trade-notification {
    position: relative;
    overflow: hidden;
}

.trade-notification::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(34, 197, 94, 0.2), transparent);
    animation: sweep 2s ease-out;
}

@keyframes sweep {
    to {
        left: 100%;
    }
}
</style>
`;

// Add styles to document
document.head.insertAdjacentHTML('beforeend', animationStyles);