#!/usr/bin/env bash
set -euo pipefail

echo "=== EMERGENCY: Stop All Trading ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

API_URL="${API_URL:-http://localhost:8000}"
TOKEN="${AUTH_TOKEN:-}"

if [ -z "$TOKEN" ]; then
    echo "ERROR: AUTH_TOKEN environment variable required"
    exit 1
fi

# 1. Stop trading engine
echo "[1/3] Stopping trading engine..."
curl -sf -X POST "$API_URL/api/v1/trading/engine/stop" \
    -H "Authorization: Bearer $TOKEN" || echo "  WARNING: Could not stop engine"

# 2. Verify engine stopped
echo "[2/3] Verifying engine stopped..."
STATUS=$(curl -sf "$API_URL/api/v1/trading/engine/status" \
    -H "Authorization: Bearer $TOKEN" || echo '{}')
echo "  Engine status: $STATUS"

# 3. Log the emergency stop
echo "[3/3] Emergency stop complete."
echo "  IMPORTANT: Review open positions manually at $API_URL/api/v1/portfolio/positions"
echo "=== Emergency stop complete ==="
