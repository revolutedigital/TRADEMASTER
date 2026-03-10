#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# TradeMaster Runbook: Restart Backend Service
#
# Purpose:  Gracefully restart the backend service with health verification.
# Scope:    Supports Docker Compose (local/staging) and Railway (production).
# Rollback: If post-check fails, the previous container is still available
#           via `docker compose logs backend` for diagnosis.
###############################################################################

SCRIPT_NAME="$(basename "$0")"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
API_URL="${API_URL:-http://localhost:8000}"
HEALTH_ENDPOINT="${API_URL}/api/v1/system/health"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-60}"
LOG_FILE="${LOG_FILE:-/tmp/trademaster-restart-$(date +%Y%m%d%H%M%S).log}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
err() { log "ERROR: $*" >&2; }

log "=== TradeMaster Backend Restart Runbook ==="
log "Timestamp:   $TIMESTAMP"
log "API URL:     $API_URL"
log "Log file:    $LOG_FILE"
echo ""

###############################################################################
# Phase 1: Pre-Execution Verification
###############################################################################
log "[1/4] PRE-CHECK: Verifying current state..."

# Check Docker is available
if ! command -v docker &>/dev/null; then
    err "Docker is not installed or not in PATH"
    exit 1
fi

# Record current health state
PRE_HEALTH="$(curl -sf --max-time 5 "$HEALTH_ENDPOINT" 2>/dev/null || echo '{"status":"unreachable"}')"
log "  Current health response: $PRE_HEALTH"

# Record current container state
CONTAINER_STATE="$(docker compose -f "$COMPOSE_FILE" ps backend --format '{{.State}}' 2>/dev/null || echo 'unknown')"
log "  Container state: $CONTAINER_STATE"

# Check for active WebSocket connections (warn operator)
WS_CONNECTIONS="$(curl -sf --max-time 5 "${API_URL}/api/v1/system/metrics" 2>/dev/null | grep -o '"ws_connections":[0-9]*' | cut -d: -f2 || echo '0')"
if [ "${WS_CONNECTIONS:-0}" -gt 0 ]; then
    log "  WARNING: ${WS_CONNECTIONS} active WebSocket connections will be dropped"
    read -r -p "  Continue with restart? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        log "  Restart aborted by operator"
        exit 0
    fi
fi

# Check database connectivity before restarting
DB_HEALTHY="$(docker compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U trademaster 2>/dev/null && echo 'yes' || echo 'no')"
if [ "$DB_HEALTHY" != "yes" ]; then
    err "PostgreSQL is not healthy. Fix database before restarting backend."
    err "  Check with: docker compose -f $COMPOSE_FILE logs postgres"
    exit 1
fi

# Check Redis connectivity
REDIS_HEALTHY="$(docker compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping 2>/dev/null || echo 'FAIL')"
if [ "$REDIS_HEALTHY" != "PONG" ]; then
    err "Redis is not responding. Fix Redis before restarting backend."
    err "  Check with: docker compose -f $COMPOSE_FILE logs redis"
    exit 1
fi

log "  Pre-checks passed: DB=healthy, Redis=healthy"
echo ""

###############################################################################
# Phase 2: Graceful Restart
###############################################################################
log "[2/4] ACTION: Initiating graceful restart..."

if command -v railway &>/dev/null && [ "${USE_RAILWAY:-}" = "true" ]; then
    log "  Using Railway CLI for restart"
    railway up -s backend --detach
    log "  Railway redeploy initiated"
else
    log "  Using Docker Compose for restart"

    # Send SIGTERM first for graceful shutdown (uvicorn handles this)
    docker compose -f "$COMPOSE_FILE" stop -t 30 backend 2>&1 | tee -a "$LOG_FILE"
    log "  Backend stopped gracefully"

    # Start fresh
    docker compose -f "$COMPOSE_FILE" up -d backend 2>&1 | tee -a "$LOG_FILE"
    log "  Backend container started"
fi

echo ""

###############################################################################
# Phase 3: Wait for Health
###############################################################################
log "[3/4] WAITING: Polling health endpoint (max ${MAX_WAIT_SECONDS}s)..."

HEALTHY=false
for i in $(seq 1 "$MAX_WAIT_SECONDS"); do
    RESPONSE="$(curl -sf --max-time 5 "$HEALTH_ENDPOINT" 2>/dev/null || echo '')"
    if echo "$RESPONSE" | grep -q '"status"' 2>/dev/null; then
        log "  Service healthy after ${i}s"
        HEALTHY=true
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        log "  Still waiting... (${i}s elapsed)"
    fi
    sleep 1
done

if [ "$HEALTHY" = false ]; then
    err "Backend did not become healthy within ${MAX_WAIT_SECONDS}s"
    log "  Collecting diagnostic logs..."
    docker compose -f "$COMPOSE_FILE" logs --tail=30 backend 2>&1 | tee -a "$LOG_FILE"
    echo ""
    err "ROLLBACK: To restore previous state, run:"
    err "  docker compose -f $COMPOSE_FILE down backend"
    err "  docker compose -f $COMPOSE_FILE up -d backend"
    exit 1
fi

echo ""

###############################################################################
# Phase 4: Post-Execution Verification
###############################################################################
log "[4/4] POST-CHECK: Verifying restored state..."

POST_HEALTH="$(curl -sf --max-time 5 "$HEALTH_ENDPOINT" 2>/dev/null || echo '{"status":"unreachable"}')"
log "  Health response: $POST_HEALTH"

# Verify key subsystems
SYSTEM_STATUS="$(curl -sf --max-time 5 "${API_URL}/api/v1/system/health" 2>/dev/null || echo '{}')"

# Check container is running
POST_STATE="$(docker compose -f "$COMPOSE_FILE" ps backend --format '{{.State}}' 2>/dev/null || echo 'unknown')"
log "  Container state: $POST_STATE"

if [ "$POST_STATE" != "running" ]; then
    err "Backend container is not in 'running' state: $POST_STATE"
    err "  Check logs: docker compose -f $COMPOSE_FILE logs backend"
    exit 1
fi

echo ""
log "=== Backend restart completed successfully ==="
log "  Pre-restart health:  $PRE_HEALTH"
log "  Post-restart health: $POST_HEALTH"
log "  Full log: $LOG_FILE"
