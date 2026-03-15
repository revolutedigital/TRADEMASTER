#!/usr/bin/env bash
set -euo pipefail

# Emergency Stop Trading - TradeMaster
# Usage: ./emergency-stop-trading.sh <BACKEND_URL> <JWT_TOKEN>

BACKEND_URL="${1:?Usage: $0 <BACKEND_URL> <JWT_TOKEN>}"
JWT_TOKEN="${2:?Usage: $0 <BACKEND_URL> <JWT_TOKEN>}"

echo "=== TradeMaster Emergency Stop ==="
echo "Target: $BACKEND_URL"
echo "Time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

# Step 1: Stop trading engine
echo "[1/4] Stopping trading engine..."
STOP_RESULT=$(curl -s -X POST "$BACKEND_URL/api/v1/trading/engine/stop" \
  -H "Cookie: access_token=$JWT_TOKEN" \
  -H "Content-Type: application/json")
echo "  Result: $STOP_RESULT"

# Step 2: Check open positions
echo "[2/4] Checking open positions..."
POSITIONS=$(curl -s "$BACKEND_URL/api/v1/portfolio/positions" \
  -H "Cookie: access_token=$JWT_TOKEN")
echo "  Open positions: $POSITIONS"

# Step 3: Verify engine stopped
echo "[3/4] Verifying engine status..."
STATUS=$(curl -s "$BACKEND_URL/api/v1/system/status" \
  -H "Cookie: access_token=$JWT_TOKEN")
echo "  Status: $STATUS"

# Step 4: Health check
echo "[4/4] Health check..."
HEALTH=$(curl -s "$BACKEND_URL/api/v1/system/health")
echo "  Health: $HEALTH"

echo ""
echo "=== Emergency Stop Complete ==="
echo "IMPORTANT: Review open positions above and close manually if needed."
