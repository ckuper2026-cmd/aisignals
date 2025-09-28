#!/usr/bin/env python3
"""
Test that the app can start without errors
"""

import sys
import os

print("=" * 60)
print("STARTUP TEST - Checking if app.py loads correctly")
print("=" * 60)

# Set environment variables for testing
os.environ['MIN_CONFIDENCE_THRESHOLD'] = '0.45'
os.environ['AUTO_TRADING_DEFAULT_ENABLED'] = 'true'
os.environ['PAPER_TRADING_DEFAULT'] = 'true'
os.environ['STOCK_UNIVERSE'] = 'SPY,QQQ,AAPL'

try:
    # Try to import the app
    print("\n1. Testing app import...")
    import app
    print("✓ App imported successfully")
    
    # Check if FastAPI app exists
    print("\n2. Checking FastAPI app...")
    if hasattr(app, 'app'):
        print("✓ FastAPI app object exists")
    else:
        print("✗ FastAPI app object not found")
        sys.exit(1)
    
    # Check ML status
    print("\n3. Checking ML Brain...")
    if hasattr(app, 'ML_ENABLED'):
        if app.ML_ENABLED:
            print("✓ ML Brain is enabled")
        else:
            print("⚠ ML Brain is disabled (will use strategies only)")
    
    # Check signal generator
    print("\n4. Checking signal generator...")
    if hasattr(app, 'signal_generator'):
        print("✓ Signal generator exists")
        portfolio_value = app.signal_generator.portfolio.get_total_value()
        print(f"  Portfolio value: ${portfolio_value:,.0f}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: App can start without errors!")
    print("=" * 60)
    print("\nTo run the app:")
    print("  python app.py")
    print("or")
    print("  uvicorn app:app --host 0.0.0.0 --port 8000")
    
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("\nCheck the error message above and fix any import issues.")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)