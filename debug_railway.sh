#!/bin/bash
# Railway Debugging Script

echo "=== Railway Deployment Debug ==="
echo

echo "1. Checking Railway status..."
railway status

echo
echo "2. Getting deployment URL..."
railway domain

echo
echo "3. Checking recent logs..."
railway logs --lines 50 | grep -E "(ERROR|WARNING|Starting|Listening|404)"

echo
echo "4. Testing endpoints..."
URL=$(railway domain | grep -o "https://[^ ]*" | head -1)
if [ ! -z "$URL" ]; then
    echo "Testing $URL"
    curl -s -o /dev/null -w "Root (/): %{http_code}\n" $URL/
    curl -s -o /dev/null -w "Health (/health): %{http_code}\n" $URL/health
    curl -s -o /dev/null -w "Signals (/api/signals): %{http_code}\n" $URL/api/signals
else
    echo "Could not detect Railway URL"
fi

echo
echo "5. Environment check..."
railway variables

echo
echo "=== Debug Complete ==="
echo
echo "If all endpoints return 404:"
echo "1. Use the fixed railway.json and Procfile"
echo "2. Commit and push changes"
echo "3. Wait for redeploy"
echo "4. Run this script again"