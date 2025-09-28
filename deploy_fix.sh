#!/bin/bash
# Fix Railway 404s - Run these commands

# 1. Copy fixed files to your project
cp railway.json ~/your-trading-project/
cp Procfile ~/your-trading-project/
cp requirements.txt ~/your-trading-project/

# 2. Navigate to project
cd ~/your-trading-project

# 3. Commit and push
git add railway.json Procfile requirements.txt
git commit -m "Fix Railway 404s - remove startup command conflicts"
git push

# 4. Wait 30 seconds for deploy
echo "Waiting for deploy..."
sleep 30

# 5. Test endpoints
railway domain
python test_endpoints.py $(railway domain | grep -o "https://[^ ]*" | head -1)

# 6. Check logs if still having issues
railway logs --lines 100