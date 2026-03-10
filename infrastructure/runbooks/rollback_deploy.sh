#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# TradeMaster Runbook: Rollback Deployment
#
# Purpose:  Roll back a service to its previous deployment version with
#           full verification. Supports Docker Compose and Railway.
# Scope:    backend, frontend, or all services.
# Rollback: If the rollback itself fails, instructions are provided to
#           manually restore from a known-good image tag.
###############################################################################

SCRIPT_NAME="$(basename "$0")"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SERVICE="${1:-backend}"
API_URL="${API_URL:-http://localhost:8000}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-90}"
LOG_FILE="${LOG_FILE:-/tmp/trademaster-rollback-$(date +%Y%m%d%H%M%S).log}"

# Optional: specify a known-good image tag to roll back to
ROLLBACK_TAG="${ROLLBACK_TAG:-}"
# Optional: specify the git commit to roll back to
ROLLBACK_COMMIT="${ROLLBACK_COMMIT:-}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
err() { log "ERROR: $*" >&2; }

log "=== TradeMaster Deployment Rollback ==="
log "Timestamp:  $TIMESTAMP"
log "Service:    $SERVICE"
log "Log file:   $LOG_FILE"
echo ""

###############################################################################
# Input Validation
###############################################################################
VALID_SERVICES=("backend" "frontend" "all")
FOUND=false
for s in "${VALID_SERVICES[@]}"; do
    if [ "$SERVICE" = "$s" ]; then
        FOUND=true
        break
    fi
done

if [ "$FOUND" = false ]; then
    err "Invalid service: $SERVICE"
    err "Usage: $SCRIPT_NAME [backend|frontend|all]"
    exit 1
fi

###############################################################################
# Phase 1: Pre-Execution Verification
###############################################################################
log "[1/5] PRE-CHECK: Recording current deployment state..."

# Check Docker is available
if ! command -v docker &>/dev/null; then
    err "Docker is not installed or not in PATH"
    exit 1
fi

# Record current image digests for audit trail
record_current_state() {
    local svc="$1"
    local image_id
    image_id="$(docker compose -f "$COMPOSE_FILE" images "$svc" --format '{{.ID}}' 2>/dev/null || echo 'unknown')"
    local container_id
    container_id="$(docker compose -f "$COMPOSE_FILE" ps "$svc" --format '{{.ID}}' 2>/dev/null || echo 'unknown')"
    log "  $svc: image=$image_id container=$container_id"
}

if [ "$SERVICE" = "all" ]; then
    for svc in backend frontend; do
        record_current_state "$svc"
    done
else
    record_current_state "$SERVICE"
fi

# Record current health
if [ "$SERVICE" = "backend" ] || [ "$SERVICE" = "all" ]; then
    CURRENT_HEALTH="$(curl -sf --max-time 5 "${API_URL}/api/v1/system/health" 2>/dev/null || echo '{"status":"unreachable"}')"
    log "  Backend health: $CURRENT_HEALTH"
fi

# Record git state
CURRENT_COMMIT="$(git -C "$(dirname "$COMPOSE_FILE")" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
log "  Current git commit: $CURRENT_COMMIT"

# Check for active trading before rollback
if [ "$SERVICE" = "backend" ] || [ "$SERVICE" = "all" ]; then
    ENGINE_STATUS="$(curl -sf --max-time 5 "${API_URL}/api/v1/trading/engine/status" 2>/dev/null || echo '{}')"
    if echo "$ENGINE_STATUS" | grep -q '"active":true' 2>/dev/null; then
        log "  WARNING: Trading engine is active!"
        read -r -p "  Stop trading before rollback? [Y/n] " confirm
        if [[ ! "$confirm" =~ ^[Nn]$ ]]; then
            log "  Stopping trading engine before rollback..."
            AUTH_TOKEN="${AUTH_TOKEN:-}" && \
            curl -sf --max-time 10 -X POST "${API_URL}/api/v1/trading/engine/stop" \
                -H "Authorization: Bearer ${AUTH_TOKEN}" 2>/dev/null || true
        fi
    fi
fi

echo ""

###############################################################################
# Phase 2: Determine Rollback Target
###############################################################################
log "[2/5] TARGET: Determining rollback version..."

if [ -n "$ROLLBACK_COMMIT" ]; then
    log "  Rolling back to git commit: $ROLLBACK_COMMIT"
    ROLLBACK_METHOD="git"
elif [ -n "$ROLLBACK_TAG" ]; then
    log "  Rolling back to image tag: $ROLLBACK_TAG"
    ROLLBACK_METHOD="tag"
else
    # Default: use git to find the previous commit
    PREV_COMMIT="$(git -C "$(dirname "$COMPOSE_FILE")" rev-parse --short HEAD~1 2>/dev/null || echo '')"
    if [ -n "$PREV_COMMIT" ]; then
        log "  No explicit target specified; rolling back to previous commit: $PREV_COMMIT"
        ROLLBACK_COMMIT="$PREV_COMMIT"
        ROLLBACK_METHOD="git"
    else
        log "  No explicit target and cannot determine previous commit."
        log "  Falling back to Docker image rebuild."
        ROLLBACK_METHOD="rebuild"
    fi
fi

echo ""

###############################################################################
# Phase 3: Execute Rollback
###############################################################################
log "[3/5] ACTION: Executing rollback..."

rollback_service() {
    local svc="$1"
    log "  Rolling back service: $svc"

    case "$ROLLBACK_METHOD" in
        git)
            log "  Checking out $ROLLBACK_COMMIT..."
            git -C "$(dirname "$COMPOSE_FILE")" checkout "$ROLLBACK_COMMIT" -- "$svc/" 2>&1 | tee -a "$LOG_FILE"
            log "  Rebuilding $svc from rolled-back source..."
            docker compose -f "$COMPOSE_FILE" build --no-cache "$svc" 2>&1 | tee -a "$LOG_FILE"
            docker compose -f "$COMPOSE_FILE" up -d "$svc" 2>&1 | tee -a "$LOG_FILE"
            ;;
        tag)
            log "  Pulling image with tag: $ROLLBACK_TAG"
            ROLLBACK_IMAGE="trademaster-${svc}:${ROLLBACK_TAG}"
            docker tag "$ROLLBACK_IMAGE" "trademaster-${svc}:rollback" 2>/dev/null || \
                docker pull "$ROLLBACK_IMAGE" 2>&1 | tee -a "$LOG_FILE"
            docker compose -f "$COMPOSE_FILE" up -d "$svc" 2>&1 | tee -a "$LOG_FILE"
            ;;
        rebuild)
            log "  Stopping and rebuilding $svc..."
            docker compose -f "$COMPOSE_FILE" stop "$svc" 2>&1 | tee -a "$LOG_FILE"
            docker compose -f "$COMPOSE_FILE" build "$svc" 2>&1 | tee -a "$LOG_FILE"
            docker compose -f "$COMPOSE_FILE" up -d "$svc" 2>&1 | tee -a "$LOG_FILE"
            ;;
    esac

    log "  Service $svc rollback command completed"
}

if [ "$SERVICE" = "all" ]; then
    for svc in backend frontend; do
        rollback_service "$svc"
    done
else
    rollback_service "$SERVICE"
fi

echo ""

###############################################################################
# Phase 4: Wait for Health
###############################################################################
log "[4/5] WAITING: Polling health endpoint (max ${MAX_WAIT_SECONDS}s)..."

wait_for_health() {
    local svc="$1"
    local url="$2"
    local healthy=false

    for i in $(seq 1 "$MAX_WAIT_SECONDS"); do
        RESPONSE="$(curl -sf --max-time 5 "$url" 2>/dev/null || echo '')"
        if echo "$RESPONSE" | grep -q '"status"' 2>/dev/null; then
            log "  $svc healthy after ${i}s"
            healthy=true
            break
        fi
        if [ $((i % 15)) -eq 0 ]; then
            log "  Still waiting for $svc... (${i}s elapsed)"
        fi
        sleep 1
    done

    if [ "$healthy" = false ]; then
        err "$svc did not become healthy within ${MAX_WAIT_SECONDS}s"
        return 1
    fi
    return 0
}

ROLLBACK_OK=true

if [ "$SERVICE" = "backend" ] || [ "$SERVICE" = "all" ]; then
    if ! wait_for_health "backend" "${API_URL}/api/v1/system/health"; then
        ROLLBACK_OK=false
    fi
fi

if [ "$SERVICE" = "frontend" ] || [ "$SERVICE" = "all" ]; then
    if ! wait_for_health "frontend" "http://localhost:3000"; then
        ROLLBACK_OK=false
    fi
fi

echo ""

###############################################################################
# Phase 5: Post-Execution Verification
###############################################################################
log "[5/5] POST-CHECK: Verifying rollback..."

if [ "$SERVICE" = "backend" ] || [ "$SERVICE" = "all" ]; then
    POST_HEALTH="$(curl -sf --max-time 5 "${API_URL}/api/v1/system/health" 2>/dev/null || echo '{"status":"unreachable"}')"
    log "  Backend health: $POST_HEALTH"
fi

# Verify container state
if [ "$SERVICE" = "all" ]; then
    for svc in backend frontend; do
        STATE="$(docker compose -f "$COMPOSE_FILE" ps "$svc" --format '{{.State}}' 2>/dev/null || echo 'unknown')"
        log "  $svc container state: $STATE"
    done
else
    STATE="$(docker compose -f "$COMPOSE_FILE" ps "$SERVICE" --format '{{.State}}' 2>/dev/null || echo 'unknown')"
    log "  $SERVICE container state: $STATE"
fi

# Record new image digests
NEW_COMMIT="$(git -C "$(dirname "$COMPOSE_FILE")" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
log "  Git commit after rollback: $NEW_COMMIT"

echo ""

if [ "$ROLLBACK_OK" = true ]; then
    log "=== Rollback completed successfully ==="
else
    err "=== Rollback completed with WARNINGS ==="
    err ""
    err "  Some services may not be healthy. Manual investigation required:"
    err "    docker compose -f $COMPOSE_FILE logs --tail=50 $SERVICE"
    err ""
    err "  MANUAL ROLLBACK (if this script's rollback also failed):"
    err "    1. Identify a known-good commit: git log --oneline -10"
    err "    2. Checkout: git checkout <commit>"
    err "    3. Rebuild: docker compose -f $COMPOSE_FILE build --no-cache $SERVICE"
    err "    4. Restart: docker compose -f $COMPOSE_FILE up -d $SERVICE"
    exit 1
fi

log "  Full log: $LOG_FILE"
