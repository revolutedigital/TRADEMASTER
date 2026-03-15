#!/usr/bin/env bash
set -euo pipefail

# Flush Redis Cache - TradeMaster
# Usage: ./flush-redis.sh <REDIS_URL>

REDIS_URL="${1:?Usage: $0 <REDIS_URL>}"

echo "=== TradeMaster Redis Flush ==="
echo "Target: $REDIS_URL"
echo "Time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""
echo "WARNING: This will clear all cached data including:"
echo "  - Price cache (price:*)"
echo "  - Rate limiter state"
echo "  - Session data"
echo ""
read -p "Proceed? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "[1/3] Checking connection..."
redis-cli -u "$REDIS_URL" PING

echo "[2/3] Current key count:"
redis-cli -u "$REDIS_URL" DBSIZE

echo "[3/3] Flushing..."
redis-cli -u "$REDIS_URL" FLUSHDB

echo ""
echo "=== Redis Flush Complete ==="
echo "Prices will repopulate when frontend reconnects to Binance WS."
