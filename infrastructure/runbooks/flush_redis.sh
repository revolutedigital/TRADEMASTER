#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# TradeMaster Runbook: Flush Redis Cache
#
# Purpose:  Safely flush Redis cache data with pre-flight checks and
#           selective or full flush options.
# Scope:    Docker Compose Redis service.
# Rollback: Redis cache is ephemeral by design. After flush, the application
#           will repopulate caches on demand. No data loss for persistent
#           data (that lives in PostgreSQL).
#
# IMPORTANT: Flushing Redis will:
#   - Drop all cached market data (will be refetched)
#   - Drop all user session caches (users may need to re-authenticate)
#   - Drop rate limiter state (rate limits will reset)
#   - Drop WebSocket channel subscriptions (clients will reconnect)
#   - Clear the emergency stop flag if set
###############################################################################

SCRIPT_NAME="$(basename "$0")"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
LOG_FILE="${LOG_FILE:-/tmp/trademaster-redis-flush-$(date +%Y%m%d%H%M%S).log}"

# Flush mode: "all" flushes everything, "selective" only flushes cache keys
FLUSH_MODE="${FLUSH_MODE:-selective}"

# Key prefixes considered safe to flush (cache-only, not state)
SAFE_PREFIXES=(
    "trademaster:cache:*"
    "trademaster:rate_limit:*"
    "trademaster:market_data:*"
    "trademaster:indicator_cache:*"
)

# Key prefixes that require explicit confirmation
PROTECTED_PREFIXES=(
    "trademaster:session:*"
    "trademaster:emergency_stop"
    "trademaster:engine:*"
)

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
err() { log "ERROR: $*" >&2; }

redis_cmd() {
    docker compose -f "$COMPOSE_FILE" exec -T redis redis-cli "$@" 2>/dev/null
}

log "=== TradeMaster Redis Cache Flush ==="
log "Timestamp:   $TIMESTAMP"
log "Flush mode:  $FLUSH_MODE"
log "Log file:    $LOG_FILE"
echo ""

###############################################################################
# Phase 1: Pre-Execution Verification
###############################################################################
log "[1/4] PRE-CHECK: Verifying Redis state..."

# Check Docker is available
if ! command -v docker &>/dev/null; then
    err "Docker is not installed or not in PATH"
    exit 1
fi

# Check Redis is running and responsive
REDIS_PING="$(redis_cmd ping || echo 'FAIL')"
if [ "$REDIS_PING" != "PONG" ]; then
    err "Redis is not responding (got: $REDIS_PING)"
    err "  Check with: docker compose -f $COMPOSE_FILE logs redis"
    exit 1
fi
log "  Redis ping: OK"

# Record current memory usage
MEMORY_INFO="$(redis_cmd INFO memory | grep -E 'used_memory_human|used_memory_peak_human' || echo 'unavailable')"
log "  Memory info:"
echo "$MEMORY_INFO" | while IFS= read -r line; do
    [ -n "$line" ] && log "    $line"
done

# Count total keys
TOTAL_KEYS="$(redis_cmd DBSIZE | grep -oE '[0-9]+' || echo '0')"
log "  Total keys: $TOTAL_KEYS"

if [ "$TOTAL_KEYS" = "0" ]; then
    log "  Redis is already empty. Nothing to flush."
    exit 0
fi

# Count keys by prefix for visibility
log "  Key distribution:"
for prefix in "${SAFE_PREFIXES[@]}"; do
    COUNT="$(redis_cmd EVAL "return #redis.call('keys', ARGV[1])" 0 "$prefix" 2>/dev/null || echo '0')"
    log "    $prefix: $COUNT keys"
done
for prefix in "${PROTECTED_PREFIXES[@]}"; do
    COUNT="$(redis_cmd EVAL "return #redis.call('keys', ARGV[1])" 0 "$prefix" 2>/dev/null || echo '0')"
    log "    $prefix: $COUNT keys (PROTECTED)"
done

# Check for emergency stop flag
EMERGENCY_FLAG="$(redis_cmd GET trademaster:emergency_stop || echo '')"
if [ "$EMERGENCY_FLAG" = "true" ]; then
    log "  WARNING: Emergency stop flag is SET. Flushing will clear it!"
fi

echo ""

# Confirmation
log "  About to flush Redis ($FLUSH_MODE mode) with $TOTAL_KEYS keys."
read -r -p "  Proceed with flush? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    log "  Flush aborted by operator"
    exit 0
fi

echo ""

###############################################################################
# Phase 2: Execute Flush
###############################################################################
log "[2/4] ACTION: Flushing Redis..."

KEYS_DELETED=0

if [ "$FLUSH_MODE" = "all" ]; then
    log "  Performing FLUSHALL..."

    # Double confirmation for full flush
    read -r -p "  CONFIRM full flush (this clears ALL data including sessions): Type 'FLUSH' to confirm: " double_confirm
    if [ "$double_confirm" != "FLUSH" ]; then
        log "  Full flush aborted (expected 'FLUSH', got '$double_confirm')"
        exit 0
    fi

    redis_cmd FLUSHALL 2>&1 | tee -a "$LOG_FILE"
    KEYS_DELETED="$TOTAL_KEYS"
    log "  FLUSHALL completed"

elif [ "$FLUSH_MODE" = "selective" ]; then
    log "  Performing selective flush (safe prefixes only)..."

    for prefix in "${SAFE_PREFIXES[@]}"; do
        # Use SCAN-based deletion to avoid blocking Redis
        DELETED="$(redis_cmd EVAL "
            local cursor = '0'
            local count = 0
            repeat
                local result = redis.call('SCAN', cursor, 'MATCH', ARGV[1], 'COUNT', 100)
                cursor = result[1]
                local keys = result[2]
                for i, key in ipairs(keys) do
                    redis.call('DEL', key)
                    count = count + 1
                end
            until cursor == '0'
            return count
        " 0 "$prefix" 2>/dev/null || echo '0')"

        log "    Deleted $DELETED keys matching $prefix"
        KEYS_DELETED=$((KEYS_DELETED + DELETED))
    done

    log "  Selective flush completed: $KEYS_DELETED keys deleted"
else
    err "Unknown flush mode: $FLUSH_MODE (expected 'all' or 'selective')"
    exit 1
fi

echo ""

###############################################################################
# Phase 3: Post-Execution Verification
###############################################################################
log "[3/4] POST-CHECK: Verifying Redis state after flush..."

# Verify Redis is still responsive
POST_PING="$(redis_cmd ping || echo 'FAIL')"
if [ "$POST_PING" != "PONG" ]; then
    err "Redis is not responding after flush!"
    err "  Restart Redis: docker compose -f $COMPOSE_FILE restart redis"
    exit 1
fi
log "  Redis ping: OK"

# Check key count
POST_KEYS="$(redis_cmd DBSIZE | grep -oE '[0-9]+' || echo '0')"
log "  Remaining keys: $POST_KEYS (was: $TOTAL_KEYS)"

# Check memory after flush
POST_MEMORY="$(redis_cmd INFO memory | grep 'used_memory_human' | head -1 || echo 'unavailable')"
log "  Memory after flush: $POST_MEMORY"

echo ""

###############################################################################
# Phase 4: Application Health Verification
###############################################################################
log "[4/4] APP-CHECK: Verifying application can reconnect to Redis..."

API_URL="${API_URL:-http://localhost:8000}"
HEALTH="$(curl -sf --max-time 10 "${API_URL}/api/v1/system/health" 2>/dev/null || echo '{"status":"unreachable"}')"
log "  Backend health: $HEALTH"

if echo "$HEALTH" | grep -q "unreachable" 2>/dev/null; then
    log "  WARNING: Backend is not reachable. Cache will repopulate when backend starts."
else
    log "  Backend is healthy. Caches will repopulate on demand."
fi

echo ""
log "=== Redis flush completed ==="
log "  Keys deleted:   $KEYS_DELETED"
log "  Keys remaining: $POST_KEYS"
log "  Flush mode:     $FLUSH_MODE"
log ""
log "  NOTE: The following will be automatically repopulated:"
log "    - Market data caches (on next price update)"
log "    - Rate limiter counters (on next API request)"
log "    - Indicator caches (on next calculation)"
if [ "$FLUSH_MODE" = "all" ]; then
    log "    - User sessions (users will need to re-authenticate)"
fi
log ""
log "  Full log: $LOG_FILE"
