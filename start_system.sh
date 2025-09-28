#!/bin/bash
# Start both infrastructure and trading engine

echo "======================================="
echo "STARTING PRODUCTION TRADING SYSTEM v4.0"
echo "======================================="

# Check if files exist
if [ ! -f "platform_infrastructure.py" ]; then
    echo "ERROR: platform_infrastructure.py not found"
    echo "Copy it from the infrastructure folder"
    exit 1
fi

if [ ! -f "app_production.py" ]; then
    echo "ERROR: app_production.py not found"
    exit 1
fi

# Start infrastructure in background
echo ""
echo "Starting Infrastructure Platform on port 8000..."
python platform_infrastructure.py &
INFRA_PID=$!
echo "Infrastructure PID: $INFRA_PID"

# Wait for infrastructure to start
echo "Waiting for infrastructure to initialize..."
sleep 5

# Start trading engine
echo ""
echo "Starting Trading Engine on port 8001..."
python app_production.py &
TRADING_PID=$!
echo "Trading Engine PID: $TRADING_PID"

# Wait for trading engine to start
sleep 3

echo ""
echo "======================================="
echo "BOTH SERVICES RUNNING"
echo "======================================="
echo ""
echo "Infrastructure: http://localhost:8000"
echo "Trading Engine: http://localhost:8001"
echo ""
echo "Test with: python test_integration.py"
echo ""
echo "To stop: kill $INFRA_PID $TRADING_PID"
echo "Or press Ctrl+C"
echo ""

# Keep script running
wait $INFRA_PID $TRADING_PID