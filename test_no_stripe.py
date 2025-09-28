#!/usr/bin/env python3
"""
Platform Infrastructure Test Script - No Stripe Required
Tests core functionality without payment processing
"""

import asyncio
import httpx
import json
from datetime import datetime
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
TEST_EMAIL = f"test_{datetime.now().timestamp()}@example.com"
TEST_PASSWORD = "TestPassword123!"

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_status(message: str, status: str = "info"):
    """Print colored status message"""
    if status == "success":
        print(f"{GREEN}✓ {message}{RESET}")
    elif status == "error":
        print(f"{RED}✗ {message}{RESET}")
    elif status == "warning":
        print(f"{YELLOW}⚠ {message}{RESET}")
    elif status == "info":
        print(f"{BLUE}→ {message}{RESET}")
    else:
        print(f"  {message}")

async def test_health():
    """Test health endpoint"""
    print("\n1. Testing Health Endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                print_status("API is healthy", "success")
                print_status(f"Database: {data['checks'].get('database', 'unknown')}", "info")
                
                # Check if Stripe is configured
                if data.get('stripe_enabled'):
                    print_status(f"Payments: {data['checks'].get('payments', 'unknown')}", "info")
                else:
                    print_status("Payments: Disabled (Stripe not configured)", "warning")
                
                return True
            else:
                print_status(f"Health check failed: {response.status_code}", "error")
                return False
        except Exception as e:
            print_status(f"Cannot connect to API: {e}", "error")
            print_status("Make sure the API is running: python app_production.py", "warning")
            return False

async def test_signup():
    """Test user signup"""
    print("\n2. Testing User Signup...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/api/auth/signup",
                json={
                    "email": TEST_EMAIL,
                    "password": TEST_PASSWORD,
                    "full_name": "Test User"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print_status(f"User created: {data['email']}", "success")
                print_status(f"User ID: {data['user_id']}", "info")
                return data['user_id']
            else:
                print_status(f"Signup failed: {response.text}", "error")
                # Check if it's a Supabase issue
                if "auth.users" in response.text:
                    print_status("Supabase auth may not be configured correctly", "warning")
                return None
        except Exception as e:
            print_status(f"Signup error: {e}", "error")
            return None

async def test_signin():
    """Test user signin"""
    print("\n3. Testing User Signin...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/api/auth/signin",
                json={
                    "email": TEST_EMAIL,
                    "password": TEST_PASSWORD
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print_status("Login successful", "success")
                print_status(f"Access token received: {data['access_token'][:20]}...", "info")
                return data['access_token']
            else:
                print_status(f"Signin failed: {response.text}", "error")
                return None
        except Exception as e:
            print_status(f"Signin error: {e}", "error")
            return None

async def test_profile(token: str):
    """Test profile retrieval"""
    print("\n4. Testing Profile Retrieval...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{API_URL}/api/auth/profile",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print_status(f"Profile retrieved: {data['email']}", "success")
                print_status(f"Subscription: {data['subscription_tier']}", "info")
                print_status(f"Status: {data['subscription_status']}", "info")
                return True
            else:
                print_status(f"Profile retrieval failed: {response.text}", "error")
                return False
        except Exception as e:
            print_status(f"Profile error: {e}", "error")
            return False

async def test_subscription_tiers():
    """Test subscription tiers endpoint"""
    print("\n5. Testing Subscription Tiers...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/api/subscriptions/tiers")
            
            if response.status_code == 200:
                data = response.json()
                print_status("Subscription tiers retrieved", "success")
                for tier, info in data.items():
                    price_str = f"${info['price']}/month" if info['price'] > 0 else "Free"
                    print_status(f"{tier}: {price_str}", "info")
                return True
            else:
                print_status(f"Failed to get tiers: {response.text}", "error")
                return False
        except Exception as e:
            print_status(f"Tiers error: {e}", "error")
            return False

async def test_create_account(token: str):
    """Test trading account creation"""
    print("\n6. Testing Trading Account Creation...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/api/accounts/create",
                headers={"Authorization": f"Bearer {token}"},
                params={"account_type": "paper"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print_status("Paper trading account created", "success")
                print_status(f"Balance: ${data['account']['balance']}", "info")
                print_status(f"Account ID: {data['account']['id']}", "info")
                return data['account']['id']
            else:
                print_status(f"Account creation failed: {response.text}", "error")
                return None
        except Exception as e:
            print_status(f"Account creation error: {e}", "error")
            return None

async def test_get_accounts(token: str):
    """Test getting trading accounts"""
    print("\n7. Testing Account Retrieval...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{API_URL}/api/accounts",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print_status(f"Accounts retrieved: {len(data['accounts'])} account(s)", "success")
                for account in data['accounts']:
                    print_status(f"Type: {account['account_type']}, Balance: ${account['balance']}", "info")
                return True
            else:
                print_status(f"Failed to get accounts: {response.text}", "error")
                return False
        except Exception as e:
            print_status(f"Get accounts error: {e}", "error")
            return False

async def run_all_tests():
    """Run all infrastructure tests"""
    print("=" * 50)
    print("AI TRADING PLATFORM - INFRASTRUCTURE TEST")
    print("(No Stripe Required)")
    print("=" * 50)
    
    # Check environment
    print("\n" + BLUE + "Environment Check:" + RESET)
    
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_URL") != "https://your-project.supabase.co":
        print_status("Supabase configured", "success")
    else:
        print_status("Supabase not configured", "error")
        print_status("Get your keys from: https://app.supabase.com/project/_/settings/api", "info")
        
    if os.getenv("STRIPE_SECRET_KEY") and "sk_test_your" not in os.getenv("STRIPE_SECRET_KEY", ""):
        print_status("Stripe configured", "success")
    else:
        print_status("Stripe not configured (optional)", "warning")
        print_status("App will run without payment processing", "info")
    
    # Run tests
    results = []
    
    # Test 1: Health
    health_ok = await test_health()
    results.append(("Health Check", health_ok))
    
    if not health_ok:
        print_status("\nAPI is not running or not healthy.", "error")
        print_status("Start it with: python app_production.py", "warning")
        return
    
    # Test 2: Signup
    user_id = await test_signup()
    results.append(("User Signup", user_id is not None))
    
    if not user_id:
        print_status("\nSignup failed. Check Supabase configuration.", "error")
        print_status("Make sure you've run the schema SQL in Supabase", "warning")
        return
    
    # Test 3: Signin
    token = await test_signin()
    results.append(("User Signin", token is not None))
    
    if not token:
        print_status("\nSignin failed. Authentication issue.", "error")
        return
    
    # Test 4: Profile
    profile_ok = await test_profile(token)
    results.append(("Profile Retrieval", profile_ok))
    
    # Test 5: Subscription Tiers
    tiers_ok = await test_subscription_tiers()
    results.append(("Subscription Tiers", tiers_ok))
    
    # Test 6: Create Account
    account_id = await test_create_account(token)
    results.append(("Account Creation", account_id is not None))
    
    # Test 7: Get Accounts
    accounts_ok = await test_get_accounts(token)
    results.append(("Account Retrieval", accounts_ok))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for test_name, ok in results:
        status = "success" if ok else "error"
        symbol = "✓" if ok else "✗"
        print_status(f"{test_name}: {symbol}", status)
    
    print("\n" + "=" * 50)
    if passed == total:
        print_status(f"ALL TESTS PASSED ({passed}/{total})", "success")
        print("\n" + GREEN + "✨ Your infrastructure is ready!" + RESET)
        print("\nNext steps:")
        print("  1. Add trading logic to app_production.py")
        print("  2. Build frontend interface")
        print("  3. Configure Stripe when ready for payments")
        print("  4. Deploy to production")
    else:
        print_status(f"SOME TESTS FAILED ({passed}/{total} passed)", "error")
        print("\nTroubleshooting:")
        print("  1. Check Supabase project is active")
        print("  2. Verify database schema was applied")
        print("  3. Check .env configuration")
        print("  4. Look at API logs for detailed errors")

def main():
    """Main entry point"""
    try:
        print(BLUE + "\nStarting infrastructure tests..." + RESET)
        print("This will test core functionality without Stripe.\n")
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()