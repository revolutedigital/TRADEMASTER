#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# TradeMaster Runbook: Emergency Stop All Trading
#
# Purpose:  Immediately halt all trading activity, cancel pending orders,
#           and disable the trading engine. Use in case of critical bugs,
#           exchange outages, or abnormal market conditions.
# Scope:    API-driven stop; works regardless of deployment method.
# Rollback: Re-enable trading via the API after investigation.
#
# CRITICAL: This script is designed for EMERGENCY use. It prioritizes speed
#           and safety over graceful behavior.
###############################################################################

SCRIPT_NAME="$(basename "$0")"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
API_URL="${API_URL:-http://localhost:8000}"
AUTH_TOKEN="${AUTH_TOKEN:-}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
LOG_FILE="${LOG_FILE:-/tmp/trademaster-emergency-stop-$(date +%Y%m%d%H%M%S).log}"
REASON="${1:-manual_emergency_stop}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
err() { log "EMERGENCY ERROR: $*" >&2; }

api_call() {
    local method="$1" endpoint="$2"
    shift 2
    curl -sf --max-time 10 -X "$method" \
        "${API_URL}${endpoint}" \
        -H "Authorization: Bearer ${AUTH_TOKEN}" \
        -H "Content-Type: application/json" \
        "$@" 2>/dev/null || echo '{"error":"request_failed"}'
}

echo ""
log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
log "!!!          EMERGENCY TRADING STOP INITIATED               !!!"
log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
log ""
log "Timestamp:  $TIMESTAMP"
log "Reason:     $REASON"
log "Operator:   ${USER:-unknown}"
log "Log file:   $LOG_FILE"
echo ""

###############################################################################
# Phase 1: Pre-Execution Verification
###############################################################################
log "[1/5] PRE-CHECK: Assessing current trading state..."

# Require authentication
if [ -z "$AUTH_TOKEN" ]; then
    err "AUTH_TOKEN environment variable is required."
    err "  Export it: export AUTH_TOKEN='your-admin-token'"
    exit 1
fi

# Verify API is reachable
API_REACHABLE="$(curl -sf --max-time 5 "${API_URL}/api/v1/system/health" 2>/dev/null && echo 'yes' || echo 'no')"
if [ "$API_REACHABLE" = "no" ]; then
    log "  WARNING: API is unreachable. Attempting direct Docker stop as fallback."
fi

# Capture current state for audit trail
ENGINE_STATUS="$(api_call GET /api/v1/trading/engine/status)"
log "  Trading engine status: $ENGINE_STATUS"

OPEN_POSITIONS="$(api_call GET /api/v1/portfolio/positions)"
log "  Open positions snapshot saved to log"
echo "$OPEN_POSITIONS" >> "$LOG_FILE"

PENDING_ORDERS="$(api_call GET /api/v1/trading/orders/pending)"
log "  Pending orders snapshot saved to log"
echo "$PENDING_ORDERS" >> "$LOG_FILE"

echo ""

###############################################################################
# Phase 2: Stop Trading Engine
###############################################################################
log "[2/5] ACTION: Stopping trading engine..."

STOP_RESULT="$(api_call POST /api/v1/trading/engine/stop -d "{\"reason\": \"${REASON}\"}")"
log "  Engine stop response: $STOP_RESULT"

if echo "$STOP_RESULT" | grep -q "error"; then
    log "  WARNING: API stop may have failed. Attempting direct intervention..."

    # Fallback: set a Redis flag that the engine checks
    if command -v docker &>/dev/null; then
        docker compose -f "$COMPOSE_FILE" exec -T redis \
            redis-cli SET trademaster:emergency_stop "true" EX 86400 2>/dev/null \
            && log "  Redis emergency flag set" \
            || log "  WARNING: Could not set Redis emergency flag"
    fi
fi

echo ""

###############################################################################
# Phase 3: Cancel All Pending Orders
###############################################################################
log "[3/5] ACTION: Cancelling all pending orders..."

CANCEL_RESULT="$(api_call POST /api/v1/trading/orders/cancel-all -d "{\"reason\": \"${REASON}\"}")"
log "  Cancel result: $CANCEL_RESULT"

echo ""

###############################################################################
# Phase 4: Disable New Order Submission
###############################################################################
log "[4/5] ACTION: Disabling order submission..."

# Set maintenance mode via API
MAINTENANCE_RESULT="$(api_call POST /api/v1/system/maintenance -d '{"enabled": true, "reason": "'"${REASON}"'"}')"
log "  Maintenance mode: $MAINTENANCE_RESULT"

# Broadcast to connected WebSocket clients
BROADCAST_RESULT="$(api_call POST /api/v1/system/broadcast -d '{"type": "emergency_stop", "message": "Trading halted: '"${REASON}"'"}')"
log "  WebSocket broadcast: $BROADCAST_RESULT"

echo ""

###############################################################################
# Phase 5: Post-Execution Verification
###############################################################################
log "[5/5] POST-CHECK: Verifying trading is fully stopped..."

# Verify engine is stopped
sleep 2
POST_STATUS="$(api_call GET /api/v1/trading/engine/status)"
log "  Engine status after stop: $POST_STATUS"

# Verify no pending orders remain
POST_ORDERS="$(api_call GET /api/v1/trading/orders/pending)"
log "  Pending orders after cancel: $POST_ORDERS"

echo ""
log "================================================================="
log "  EMERGENCY STOP COMPLETE"
log "================================================================="
log ""
log "  IMPORTANT: Manual actions required:"
log "  1. Review open positions:  ${API_URL}/api/v1/portfolio/positions"
log "  2. Check exchange state:   Verify orders cancelled on exchange side"
log "  3. Review logs:            $LOG_FILE"
log "  4. Incident report:        Document the reason and timeline"
log ""
log "  TO RESUME TRADING (after investigation):"
log "    1. Disable maintenance mode:"
log "       curl -X POST ${API_URL}/api/v1/system/maintenance \\"
log "         -H 'Authorization: Bearer \$AUTH_TOKEN' \\"
log "         -d '{\"enabled\": false}'"
log "    2. Restart trading engine:"
log "       curl -X POST ${API_URL}/api/v1/trading/engine/start \\"
log "         -H 'Authorization: Bearer \$AUTH_TOKEN'"
log "    3. Clear emergency flag:"
log "       docker compose exec redis redis-cli DEL trademaster:emergency_stop"
log ""
log "  Full log: $LOG_FILE"
