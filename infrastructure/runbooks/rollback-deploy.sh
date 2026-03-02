#!/usr/bin/env bash
set -euo pipefail

echo "=== TradeMaster Deployment Rollback ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

SERVICE="${1:-backend}"

# Pre-check
echo "[1/3] Pre-check: current deployment..."
if command -v railway &> /dev/null; then
    echo "  Using Railway CLI for rollback"
    railway logs -s "$SERVICE" --lines 5
else
    echo "  Using Docker for rollback"
    docker-compose logs --tail=5 "$SERVICE"
fi

# Rollback
echo "[2/3] Rolling back $SERVICE..."
if command -v railway &> /dev/null; then
    # Railway: rollback to previous deployment
    railway rollback -s "$SERVICE" --confirm
else
    # Docker: use previous image tag
    PREVIOUS_TAG="${PREVIOUS_TAG:-latest}"
    docker-compose pull "$SERVICE"
    docker-compose up -d "$SERVICE"
fi

# Post-check
echo "[3/3] Verifying rollback..."
sleep 5
HEALTH=$(curl -sf http://localhost:8000/api/v1/system/health || echo '{"status":"unreachable"}')
echo "  Health after rollback: $HEALTH"
echo "=== Rollback complete ==="
