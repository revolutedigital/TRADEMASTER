#!/usr/bin/env bash
set -euo pipefail

echo "=== TradeMaster Backend Restart Runbook ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Pre-check
echo "[1/4] Pre-check: verifying current state..."
HEALTH=$(curl -sf http://localhost:8000/api/v1/system/health || echo '{"status":"unreachable"}')
echo "  Current health: $HEALTH"

# Graceful shutdown signal
echo "[2/4] Sending graceful shutdown signal..."
# Railway: redeploy the service
if command -v railway &> /dev/null; then
    railway up -s backend --detach
    echo "  Railway redeploy initiated"
else
    echo "  Local mode: restart via docker-compose"
    docker-compose restart backend
fi

# Wait for service to come back
echo "[3/4] Waiting for service to become healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/api/v1/system/health > /dev/null 2>&1; then
        echo "  Service healthy after ${i}s"
        break
    fi
    sleep 1
done

# Post-check
echo "[4/4] Post-check: verifying restored state..."
HEALTH=$(curl -sf http://localhost:8000/api/v1/system/health || echo '{"status":"unreachable"}')
echo "  Final health: $HEALTH"
echo "=== Restart complete ==="
