#!/bin/bash

# Personal Trading System - Quick Start
# =====================================

echo "======================================"
echo "PERSONAL TRADING SYSTEM SETUP"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.8+ required (found $python_version)"
    exit 1
fi

echo "✓ Python $python_version detected"

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements_personal.txt
echo "✓ Dependencies installed"

# Create necessary directories
mkdir -p logs
mkdir -p data
mkdir -p results

echo ""
echo "======================================"
echo "USAGE OPTIONS:"
echo "======================================"
echo ""
echo "1. VERIFY SYSTEM (Run this first!)"
echo "   python3 verify_system.py"
echo ""
echo "2. LIVE TRADING DASHBOARD:"
echo "   python3 dashboard.py live"
echo ""
echo "3. BACKTEST A STRATEGY:"
echo "   python3 dashboard.py backtest --strategy momentum --symbols AAPL,MSFT,GOOGL --start 2023-01-01 --end 2024-01-01"
echo ""
echo "4. ANALYZE PORTFOLIO:"
echo "   python3 dashboard.py analyze"
echo ""
echo "5. DIRECT TRADING (NO UI):"
echo "   python3 personal_trader.py"
echo ""
echo "6. CHECK ML PROGRESS:"
echo "   python3 check_ml.py"
echo ""
echo "======================================"
echo ""

# Ask user what to run
read -p "Select option (1-6) or press Enter to exit: " choice

case $choice in
    1)
        echo "Running system verification..."
        python3 verify_system.py
        ;;
    2)
        echo "Starting live trading dashboard..."
        python3 dashboard.py live
        ;;
    3)
        echo "Running backtest..."
        python3 dashboard.py backtest \
            --strategy momentum \
            --symbols AAPL,MSFT,GOOGL,NVDA,META \
            --start 2023-01-01 \
            --end 2024-01-01 \
            --capital 100000
        ;;
    4)
        echo "Running portfolio analysis..."
        python3 dashboard.py analyze
        ;;
    5)
        echo "Starting direct trading system..."
        python3 personal_trader.py
        ;;
    6)
        echo "Checking ML progress..."
        python3 check_ml.py
        ;;
    *)
        echo "Exiting..."
        ;;
esac