#!/bin/bash
# Complete Setup Script for Standalone Trading System
# Run each section step by step

echo "================================================"
echo "STANDALONE TRADING SYSTEM - COMPLETE SETUP"
echo "================================================"
echo ""
echo "This script shows all commands needed to deploy"
echo "Run each section manually to ensure success"
echo ""
echo "================================================"
echo "STEP 1: CREATE PROJECT FOLDER"
echo "================================================"

cat << 'STEP1'
# Create a fresh folder for deployment
cd ~
mkdir trading-standalone
cd trading-standalone

# Verify you're in the right place
pwd
# Should show: /home/yourname/trading-standalone
STEP1

echo ""
echo "================================================"
echo "STEP 2: COPY REQUIRED FILES"
echo "================================================"

cat << 'STEP2'
# Copy the main trading system (rename to app.py for Railway)
cp /path/to/outputs/standalone_trader.py app.py

# Copy requirements
cp /path/to/outputs/requirements_standalone.txt requirements.txt

# Or create requirements.txt directly:
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
yfinance==0.2.33
pandas==2.1.3
numpy==1.24.3
pytz==2023.3
python-dotenv==1.0.0
httpx==0.25.1
EOF

# Verify files exist
ls -la
# Should show: app.py and requirements.txt
STEP2

echo ""
echo "================================================"
echo "STEP 3: CREATE RAILWAY CONFIG FILES"
echo "================================================"

cat << 'STEP3'
# Create Procfile (tells Railway how to start the app)
cat > Procfile << 'EOF'
web: uvicorn app:app --host 0.0.0.0 --port $PORT
EOF

# Create railway.json (optional but recommended)
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn app:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF

# Verify all files
ls -la
# Should show: app.py, requirements.txt, Procfile, railway.json
STEP3

echo ""
echo "================================================"
echo "STEP 4: TEST LOCALLY (OPTIONAL BUT RECOMMENDED)"
echo "================================================"

cat << 'STEP4'
# Install dependencies locally
pip install -r requirements.txt

# Test the system
python app.py

# In another terminal, test the API
curl http://localhost:8000/
# Should return JSON with system status

# Stop with Ctrl+C when satisfied
STEP4

echo ""
echo "================================================"
echo "STEP 5: INSTALL RAILWAY CLI"
echo "================================================"

cat << 'STEP5'
# Check if Railway CLI is installed
railway --version

# If not installed, install it:
# Option A: Using npm (if you have Node.js)
npm install -g @railway/cli

# Option B: Using brew (on Mac)
brew install railway

# Option C: Using curl (Linux/Mac)
curl -fsSL https://railway.app/install.sh | sh

# Verify installation
railway --version
STEP5

echo ""
echo "================================================"
echo "STEP 6: DEPLOY TO RAILWAY"
echo "================================================"

cat << 'STEP6'
# Login to Railway (opens browser)
railway login

# Initialize new Railway project
railway new

# When prompted, enter project name: trading-standalone
# Or any name you prefer

# Link current directory to Railway project
railway link

# Deploy the application
railway up

# Wait for deployment to complete (2-3 minutes)
# You'll see build logs and "Deployment successful"

# Get your public URL
railway domain

# The output will be something like:
# https://trading-standalone-production.up.railway.app
# SAVE THIS URL!
STEP6

echo ""
echo "================================================"
echo "STEP 7: VERIFY DEPLOYMENT"
echo "================================================"

cat << 'STEP7'
# Replace YOUR_URL with your actual Railway URL
export RAILWAY_URL="https://your-app.up.railway.app"

# Test that it's running
curl $RAILWAY_URL/
# Should return JSON with status: "running"

# Check portfolio
curl $RAILWAY_URL/api/portfolio
# Should show initial $100,000 portfolio

# Check from browser (on phone or computer)
# Go to: https://your-app.up.railway.app/
STEP7

echo ""
echo "================================================"
echo "STEP 8: SETUP MONITORING TOOLS"
echo "================================================"

cat << 'STEP8'
# Copy monitoring scripts to a convenient location
mkdir ~/trading-monitor
cd ~/trading-monitor

# Copy remote monitor
cp /path/to/outputs/remote_monitor.py .

# Update the Railway URL in remote_monitor.py
# Edit line 11:
# API_URL = "https://your-actual-railway-url.up.railway.app"
nano remote_monitor.py
# Or use any text editor

# Test remote monitor
python remote_monitor.py
# Choose option 1 to test connection

# For mobile monitoring:
# Copy mobile_monitor.html somewhere
cp /path/to/outputs/mobile_monitor.html .

# Edit mobile_monitor.html line 315:
# const API_URL = 'https://your-actual-railway-url.up.railway.app';
# Then open in browser on phone
STEP8

echo ""
echo "================================================"
echo "STEP 9: MONITOR YOUR DEPLOYMENT"
echo "================================================"

cat << 'STEP9'
# View Railway logs (see trading activity)
railway logs

# View Railway dashboard
railway open

# Check deployment status
railway status

# If you need to restart
railway restart

# If you need to stop temporarily
railway down

# To resume
railway up
STEP9

echo ""
echo "================================================"
echo "STEP 10: DAILY MONITORING ROUTINE"
echo "================================================"

cat << 'STEP10'
# MORNING (9:30 AM ET)
# From phone browser:
https://your-app.up.railway.app/
# Check: status = "running", market_open = true

# MIDDAY (12:00 PM ET)
# From phone browser:
https://your-app.up.railway.app/api/portfolio
# Check: positions, P&L, cash

# END OF DAY (4:00 PM ET)
# From computer:
cd ~/trading-monitor
python remote_monitor.py
# Choose option 3 for daily summary

# SAVE DAILY RESULTS
curl https://your-app.up.railway.app/api/trades > trades_$(date +%Y%m%d).json
curl https://your-app.up.railway.app/api/performance > performance_$(date +%Y%m%d).json
STEP10

echo ""
echo "================================================"
echo "TROUBLESHOOTING"
echo "================================================"

cat << 'TROUBLE'
# If deployment fails:
1. Check logs: railway logs
2. Verify files: ls -la
3. Check Python version: python --version (needs 3.8+)

# If API doesn't respond:
1. Check deployment status: railway status
2. Restart app: railway restart
3. Check Railway dashboard: railway open

# If no trades happen:
1. Check market hours (9:30 AM - 4:00 PM ET, weekdays)
2. Check logs for errors: railway logs
3. Verify with: curl https://your-app.up.railway.app/

# If you need to update code:
1. Edit app.py locally
2. Run: railway up
3. Changes deploy automatically

# To completely start over:
railway unlink
railway new
railway up
TROUBLE

echo ""
echo "================================================"
echo "QUICK REFERENCE - BOOKMARK THESE URLS"
echo "================================================"
echo ""
echo "After deployment, bookmark on your phone:"
echo ""
echo "1. Status:      https://your-app.up.railway.app/"
echo "2. Portfolio:   https://your-app.up.railway.app/api/portfolio"
echo "3. Performance: https://your-app.up.railway.app/api/performance"
echo "4. Trades:      https://your-app.up.railway.app/api/trades"
echo ""
echo "================================================"
echo "DONE! Your system will trade automatically."
echo "================================================"