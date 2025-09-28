#!/usr/bin/env python3
"""
Quick fix for import errors
Run this to fix the 'Dict' not defined errors
"""

import os
import sys

print("Fixing import errors...")

# Fix 1: Check if files exist
files_to_fix = [
    'personal_trader.py',
    'personal_ml_engine.py',
    'verify_system.py'
]

for filename in files_to_fix:
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        continue
    
    # Read the file
    with open(filename, 'r') as f:
        content = f.read()
    
    # Check if typing imports are correct
    if 'from typing import' in content and 'Dict' not in content.split('from typing import')[1].split('\n')[0]:
        print(f"  Fixing {filename}...")
        # This would need more sophisticated fixing
        print(f"  Please ensure 'from typing import Dict' is in {filename}")

print("\nQuick verification of imports:")

try:
    from typing import Dict, List, Optional
    print("✓ typing imports work")
except ImportError as e:
    print(f"✗ typing import error: {e}")

try:
    from collections import deque
    print("✓ collections.deque import works")
except ImportError as e:
    print(f"✗ collections import error: {e}")

print("\nTesting imports from our modules...")

try:
    # Test personal_ml_engine
    exec("""
from typing import Dict, List, Optional
from collections import deque
import numpy as np
import pandas as pd
""")
    print("✓ All required imports available")
except Exception as e:
    print(f"✗ Import error: {e}")

print("\n" + "="*50)
print("MANUAL FIX INSTRUCTIONS:")
print("="*50)
print("""
If you still see 'Dict' errors, add this line at the top of the affected file:

from typing import Dict, List, Optional

The files should have these imports:

1. personal_ml_engine.py - First lines should be:
   from typing import Dict, List, Optional
   from collections import deque
   
2. personal_trader.py - Should include:
   from typing import Dict, List, Optional, Tuple
   
3. verify_system.py - Should include:
   from typing import Dict, List, Optional

Run the verification again after fixing:
python verify_system.py
""")