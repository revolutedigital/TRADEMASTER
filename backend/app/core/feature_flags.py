"""Runtime feature flags with Redis persistence for multi-instance consistency."""

import json

import redis.asyncio as aioredis

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_REDIS_KEY = "trademaster:feature_flags"


class FeatureFlags:
    """Feature flag system backed by Redis with in-memory fallback.

    Flags are stored in Redis as a JSON hash so they survive restarts
    and stay consistent across multiple instances.  If Redis is
    unavailable, the in-memory dict is used as a transparent fallback.
    """

    _DEFAULTS: dict[str, bool] = {
        "multi_exchange": False,
        "sentiment_analysis": False,
        "twap_execution": False,
        "portfolio_optimizer": True,
        "monte_carlo_risk": True,
        "kelly_sizing": True,
        "advanced_alerts": True,
        "dca_automation": False,
    }

    def __init__(self) -> None:
        self._flags: dict[str, bool] = dict(self._DEFAULTS)
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Redis lifecycle
    # ------------------------------------------------------------------

    async def connect_redis(self) -> None:
        """Connect to Redis.  Safe to call multiple times."""
        try:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("feature_flags_redis_connected")
        except Exception as e:
            self._redis = None
            logger.warning("feature_flags_redis_unavailable", error=str(e))

    async def _get_redis(self) -> aioredis.Redis | None:
        """Return the active Redis client, reconnecting lazily if needed."""
        if self._redis is not None:
            try:
                await self._redis.ping()
                return self._redis
            except Exception:
                self._redis = None
        # Try one reconnection attempt
        try:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
            await self._redis.ping()
            return self._redis
        except Exception:
            self._redis = None
            return None

    # ------------------------------------------------------------------
    # Sync helpers (Redis operations are async, but callers may be sync)
    # ------------------------------------------------------------------

    async def _load_from_redis(self) -> None:
        """Load flags from Redis into memory.  Missing flags get defaults."""
        r = await self._get_redis()
        if r is None:
            return
        try:
            raw = await r.get(_REDIS_KEY)
            if raw:
                stored = json.loads(raw)
                # Merge: stored values override defaults, new defaults added
                merged = dict(self._DEFAULTS)
                merged.update(stored)
                self._flags = merged
                logger.info("feature_flags_loaded_from_redis", count=len(merged))
            else:
                # First run: push defaults to Redis
                await self._save_to_redis()
                logger.info("feature_flags_defaults_pushed_to_redis")
        except Exception as e:
            logger.warning("feature_flags_redis_load_failed", error=str(e))

    async def _save_to_redis(self) -> None:
        """Persist current flags to Redis (best-effort)."""
        r = await self._get_redis()
        if r is None:
            return
        try:
            await r.set(_REDIS_KEY, json.dumps(self._flags))
        except Exception as e:
            logger.warning("feature_flags_redis_save_failed", error=str(e))

    async def init(self) -> None:
        """Initialize: connect to Redis and load persisted flags.

        Call once at application startup.
        """
        await self.connect_redis()
        await self._load_from_redis()

    # ------------------------------------------------------------------
    # Public API (sync reads from memory, async writes persist to Redis)
    # ------------------------------------------------------------------

    def is_enabled(self, flag: str) -> bool:
        """Check if a feature flag is enabled."""
        return self._flags.get(flag, False)

    async def enable(self, flag: str) -> None:
        """Enable a feature flag."""
        self._flags[flag] = True
        await self._save_to_redis()
        logger.info("feature_flag_enabled", flag=flag)

    async def disable(self, flag: str) -> None:
        """Disable a feature flag."""
        self._flags[flag] = False
        await self._save_to_redis()
        logger.info("feature_flag_disabled", flag=flag)

    def get_all(self) -> dict[str, bool]:
        """Return all flags and their states."""
        return dict(self._flags)

    async def set_flag(self, flag: str, enabled: bool) -> None:
        """Set a flag to a specific state."""
        self._flags[flag] = enabled
        await self._save_to_redis()
        logger.info("feature_flag_set", flag=flag, enabled=enabled)

    def list_flags(self) -> dict[str, bool]:
        """Alias for get_all — backward compat with admin API."""
        return self.get_all()

    async def toggle(self, flag: str) -> bool:
        """Toggle a flag and return the new state."""
        current = self._flags.get(flag, False)
        self._flags[flag] = not current
        await self._save_to_redis()
        logger.info("feature_flag_toggled", flag=flag, enabled=not current)
        return not current


feature_flags = FeatureFlags()
