"""Feature store with Redis caching for ML features."""

import json
from datetime import datetime, timezone

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """Cache computed features in Redis for fast retrieval."""

    def __init__(self):
        self._local_cache: dict[str, dict] = {}

    def _cache_key(self, symbol: str, interval: str) -> str:
        return f"features:{symbol}:{interval}"

    async def get_features(self, symbol: str, interval: str) -> np.ndarray | None:
        """Get cached features for a symbol/interval pair."""
        key = self._cache_key(symbol, interval)

        # Check local cache first
        if key in self._local_cache:
            entry = self._local_cache[key]
            # Check if still fresh (60s TTL)
            age = (datetime.now(timezone.utc) - entry["timestamp"]).total_seconds()
            if age < 60:
                return entry["features"]

        # Try Redis
        try:
            from app.models.base import redis
            if redis:
                cached = await redis.get(key)
                if cached:
                    data = json.loads(cached)
                    features = np.array(data["features"])
                    self._local_cache[key] = {
                        "features": features,
                        "timestamp": datetime.now(timezone.utc),
                    }
                    return features
        except Exception as e:
            logger.debug("feature_store_redis_miss", key=key, error=str(e))

        return None

    async def store_features(self, symbol: str, interval: str, features: np.ndarray):
        """Store computed features in cache."""
        key = self._cache_key(symbol, interval)

        # Local cache
        self._local_cache[key] = {
            "features": features,
            "timestamp": datetime.now(timezone.utc),
        }

        # Redis cache
        try:
            from app.models.base import redis
            if redis:
                data = json.dumps({"features": features.tolist()})
                await redis.setex(key, 60, data)  # 60s TTL
        except Exception as e:
            logger.debug("feature_store_redis_store_failed", key=key, error=str(e))

    def invalidate(self, symbol: str, interval: str):
        """Invalidate cached features."""
        key = self._cache_key(symbol, interval)
        self._local_cache.pop(key, None)


feature_store = FeatureStore()
