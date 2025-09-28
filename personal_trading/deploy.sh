#!/bin/bash

# Copy personal trading system to your project

echo "======================================"
echo "DEPLOYING PERSONAL TRADING SYSTEM"
echo "======================================"

# Target directory
TARGET_DIR="./personal_trading"

# Create target directory
mkdir -p $TARGET_DIR

# Copy files
echo "Copying files to $TARGET_DIR..."

cp personal_trader.py $TARGET_DIR/
cp personal_ml_engine.py $TARGET_DIR/
cp real_market_trainer.py $TARGET_DIR/
cp backtest_engine.py $TARGET_DIR/
cp dashboard.py $TARGET_DIR/
cp ml_tracker.py $TARGET_DIR/
cp check_ml.py $TARGET_DIR/
cp config.json $TARGET_DIR/
cp requirements_personal.txt $TARGET_DIR/requirements.txt
cp start.sh $TARGET_DIR/
cp verify_system.py $TARGET_DIR/
cp test_system.py $TARGET_DIR/
cp fix_imports.py $TARGET_DIR/
cp README_PERSONAL.md $TARGET_DIR/README.md
cp VERIFICATION_CHECKLIST.md $TARGET_DIR/
cp ML_TRAINING_EXPLAINED.md $TARGET_DIR/
cp ML_TRACKING_GUIDE.md $TARGET_DIR/

echo "✓ Files copied successfully"

# Create .env template
cat > $TARGET_DIR/.env << EOL
# Personal Trading System Configuration

# Trading Mode
PAPER_TRADE=true

# Risk Parameters (override config.json)
MAX_DAILY_LOSS=0.03
MAX_POSITION_SIZE=0.15

# Data Source
DATA_PROVIDER=yfinance

# Notifications (optional)
EMAIL_ALERTS=false
WEBHOOK_URL=

# Alpaca API (if you want live trading)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
EOL

echo "✓ Environment file created"

# Create directories
mkdir -p $TARGET_DIR/logs
mkdir -p $TARGET_DIR/data
mkdir -p $TARGET_DIR/results
mkdir -p $TARGET_DIR/models

echo "✓ Directories created"

echo ""
echo "======================================"
echo "DEPLOYMENT COMPLETE!"
echo "======================================"
echo ""
echo "Your personal trading system is ready at: $TARGET_DIR"
echo ""
echo "Next steps:"
echo "1. cd $TARGET_DIR"
echo "2. pip install -r requirements.txt"
echo "3. ./start.sh"
echo ""
echo "Or run directly:"
echo "- python3 dashboard.py live           # Live dashboard"
echo "- python3 personal_trader.py          # Direct trading"
echo "- python3 dashboard.py backtest ...   # Run backtest"
echo ""