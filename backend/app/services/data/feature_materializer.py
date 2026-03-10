"""Feature materialization engine for pre-computing ML features.

Pre-computes ML features in batch and stores them for low-latency online
serving via Redis, with PostgreSQL as the offline feature store. Supports
feature versioning, point-in-time correctness, incremental updates, and
staleness monitoring.
"""

import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

from app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    COMPUTING = "computing"
    FAILED = "failed"


@dataclass
class FeatureVersion:
    """Tracks a specific version of a feature computation."""

    version: int
    created_at: datetime
    schema_hash: str
    description: str
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "schema_hash": self.schema_hash,
            "description": self.description,
            "is_active": self.is_active,
        }


@dataclass
class MaterializedFeature:
    """A single materialized feature value with provenance metadata."""

    name: str
    value: Any
    version: int
    computed_at: datetime
    data_timestamp: datetime
    entity_id: str
    ttl_seconds: int = 300
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        age = (datetime.now(timezone.utc) - self.computed_at).total_seconds()
        return age > self.ttl_seconds

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "version": self.version,
            "computed_at": self.computed_at.isoformat(),
            "data_timestamp": self.data_timestamp.isoformat(),
            "entity_id": self.entity_id,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MaterializedFeature":
        return cls(
            name=data["name"],
            value=data["value"],
            version=data["version"],
            computed_at=datetime.fromisoformat(data["computed_at"]),
            data_timestamp=datetime.fromisoformat(data["data_timestamp"]),
            entity_id=data["entity_id"],
            ttl_seconds=data.get("ttl_seconds", 300),
            metadata=data.get("metadata", {}),
        )


# Type alias for feature computation functions.
# Signature: (entity_id, as_of_timestamp) -> feature_value
FeatureComputeFn = Callable[[str, datetime], Coroutine[Any, Any, Any]]


@dataclass
class _FeatureDefinition:
    """Internal definition of a registered feature."""

    name: str
    compute_fn: FeatureComputeFn
    dependencies: list[str]
    ttl_seconds: int
    version: FeatureVersion


class FeatureMaterializationEngine:
    """Pre-compute ML features in batch and serve with low latency.

    Online serving path: Redis (sub-millisecond reads).
    Offline / audit path: PostgreSQL (versioned, queryable history).

    Features are registered with explicit dependency declarations so the
    engine can perform incremental updates when upstream data changes.
    """

    def __init__(self) -> None:
        self._definitions: dict[str, _FeatureDefinition] = {}
        self._versions: dict[str, list[FeatureVersion]] = defaultdict(list)
        self._local_cache: dict[str, MaterializedFeature] = {}
        self._freshness_thresholds: dict[str, timedelta] = {}
        self._staleness_callbacks: list[Callable] = []
        self._computing: set[str] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_feature(
        self,
        name: str,
        compute_fn: FeatureComputeFn,
        dependencies: list[str] | None = None,
        ttl_seconds: int = 300,
        description: str = "",
    ) -> FeatureVersion:
        """Register a feature with its computation function and dependencies.

        Args:
            name: Unique feature name, e.g. ``"rsi_14_1h"``.
            compute_fn: Async callable ``(entity_id, as_of) -> value``.
            dependencies: Names of upstream features this depends on.
            ttl_seconds: Time-to-live for the materialized value.
            description: Human-readable description of the feature.

        Returns:
            The newly created ``FeatureVersion``.
        """
        deps = dependencies or []
        schema_hash = self._compute_schema_hash(name, deps)

        existing_versions = self._versions[name]
        version_num = len(existing_versions) + 1
        version = FeatureVersion(
            version=version_num,
            created_at=datetime.now(timezone.utc),
            schema_hash=schema_hash,
            description=description,
        )

        # Deactivate previous versions
        for v in existing_versions:
            v.is_active = False
        existing_versions.append(version)

        self._definitions[name] = _FeatureDefinition(
            name=name,
            compute_fn=compute_fn,
            dependencies=deps,
            ttl_seconds=ttl_seconds,
            version=version,
        )

        logger.info(
            "feature_registered",
            feature=name,
            version=version_num,
            dependencies=deps,
        )
        return version

    def set_freshness_threshold(self, feature_name: str, threshold: timedelta) -> None:
        """Set the maximum allowed age before a feature is considered stale."""
        self._freshness_thresholds[feature_name] = threshold

    def on_staleness(self, callback: Callable) -> None:
        """Register a callback invoked when a feature goes stale.

        Callback signature: ``(feature_name: str, age_seconds: float) -> None``.
        """
        self._staleness_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Point-in-time computation
    # ------------------------------------------------------------------

    async def compute_feature(
        self,
        feature_name: str,
        entity_id: str,
        as_of: datetime | None = None,
    ) -> MaterializedFeature:
        """Compute a single feature value with point-in-time correctness.

        ``as_of`` controls which data the computation may see, ensuring no
        future data leaks into historical feature values.

        Args:
            feature_name: Name of the registered feature.
            entity_id: The entity (e.g. symbol) to compute for.
            as_of: Point-in-time cutoff.  Defaults to *now* (UTC).

        Returns:
            The computed ``MaterializedFeature``.

        Raises:
            ValueError: If the feature has not been registered.
        """
        if feature_name not in self._definitions:
            raise ValueError(f"Feature '{feature_name}' is not registered")

        defn = self._definitions[feature_name]
        as_of = as_of or datetime.now(timezone.utc)

        lock_key = f"{feature_name}:{entity_id}:{as_of.isoformat()}"
        if lock_key in self._computing:
            logger.debug("feature_already_computing", feature=feature_name, entity=entity_id)
            # Wait briefly then return from cache if available
            await asyncio.sleep(0.05)
            cached = self._get_from_local_cache(feature_name, entity_id)
            if cached:
                return cached
            raise RuntimeError(f"Circular or concurrent computation for {lock_key}")

        self._computing.add(lock_key)
        try:
            # Compute dependency features first (point-in-time preserved)
            for dep_name in defn.dependencies:
                dep_cached = self._get_from_local_cache(dep_name, entity_id)
                if dep_cached is None or dep_cached.is_expired:
                    await self.compute_feature(dep_name, entity_id, as_of)

            value = await defn.compute_fn(entity_id, as_of)

            feature = MaterializedFeature(
                name=feature_name,
                value=value,
                version=defn.version.version,
                computed_at=datetime.now(timezone.utc),
                data_timestamp=as_of,
                entity_id=entity_id,
                ttl_seconds=defn.ttl_seconds,
                metadata={"schema_hash": defn.version.schema_hash},
            )

            # Store in both tiers
            self._put_local_cache(feature)
            await self._store_to_redis(feature)
            await self._store_to_postgres(feature)

            logger.debug(
                "feature_computed",
                feature=feature_name,
                entity=entity_id,
                version=defn.version.version,
            )
            return feature
        except Exception:
            logger.exception("feature_compute_failed", feature=feature_name, entity=entity_id)
            raise
        finally:
            self._computing.discard(lock_key)

    # ------------------------------------------------------------------
    # Batch / incremental materialisation
    # ------------------------------------------------------------------

    async def materialize_batch(
        self,
        entity_ids: list[str],
        feature_names: list[str] | None = None,
        as_of: datetime | None = None,
    ) -> list[MaterializedFeature]:
        """Compute features for many entities in parallel.

        Args:
            entity_ids: List of entity identifiers.
            feature_names: Subset of features to compute.  ``None`` = all.
            as_of: Point-in-time cutoff.

        Returns:
            List of successfully computed features.
        """
        names = feature_names or list(self._definitions.keys())
        as_of = as_of or datetime.now(timezone.utc)

        tasks = [
            self.compute_feature(name, eid, as_of)
            for eid in entity_ids
            for name in names
        ]

        results: list[MaterializedFeature] = []
        for coro in asyncio.as_completed(tasks):
            try:
                feature = await coro
                results.append(feature)
            except Exception as exc:
                logger.warning("batch_feature_failed", error=str(exc))

        logger.info(
            "batch_materialization_complete",
            total=len(tasks),
            succeeded=len(results),
        )
        return results

    async def incremental_update(self, changed_sources: list[str], entity_ids: list[str]) -> list[MaterializedFeature]:
        """Re-compute only features affected by changed upstream sources.

        Walks the dependency graph to identify which features transitively
        depend on any of the ``changed_sources`` and recomputes them.

        Args:
            changed_sources: Feature or source names whose data has changed.
            entity_ids: Entities that need refreshing.

        Returns:
            List of recomputed features.
        """
        affected = self._get_affected_features(changed_sources)
        if not affected:
            logger.debug("incremental_update_no_affected", sources=changed_sources)
            return []

        logger.info(
            "incremental_update_starting",
            changed_sources=changed_sources,
            affected_features=affected,
            entity_count=len(entity_ids),
        )

        # Invalidate local cache for affected features
        for name in affected:
            for eid in entity_ids:
                cache_key = self._cache_key(name, eid)
                self._local_cache.pop(cache_key, None)

        # Topologically ordered recomputation
        ordered = self._topological_sort(affected)
        return await self.materialize_batch(entity_ids, feature_names=ordered)

    # ------------------------------------------------------------------
    # Serving (online reads)
    # ------------------------------------------------------------------

    async def get_feature(self, feature_name: str, entity_id: str) -> MaterializedFeature | None:
        """Retrieve a materialized feature for online serving.

        Checks local cache first, then Redis, falling back to ``None``
        if nothing is available (caller should trigger computation).
        """
        # L1: local cache
        cached = self._get_from_local_cache(feature_name, entity_id)
        if cached and not cached.is_expired:
            return cached

        # L2: Redis
        loaded = await self._load_from_redis(feature_name, entity_id)
        if loaded:
            self._put_local_cache(loaded)
            return loaded

        return None

    async def get_feature_vector(self, entity_id: str, feature_names: list[str]) -> dict[str, Any]:
        """Return multiple features as a dict for model input."""
        vector: dict[str, Any] = {}
        for name in feature_names:
            feat = await self.get_feature(name, entity_id)
            vector[name] = feat.value if feat else None
        return vector

    # ------------------------------------------------------------------
    # Freshness monitoring
    # ------------------------------------------------------------------

    async def check_freshness(self) -> list[dict]:
        """Check all registered features for staleness.

        Returns a list of alerts for any feature/entity pair that exceeds
        the configured freshness threshold.
        """
        alerts: list[dict] = []
        now = datetime.now(timezone.utc)

        for name, defn in self._definitions.items():
            threshold = self._freshness_thresholds.get(name, timedelta(seconds=defn.ttl_seconds))

            # Check each cached entity
            stale_entities: list[str] = []
            for cache_key, feat in self._local_cache.items():
                if feat.name != name:
                    continue
                age = now - feat.computed_at
                if age > threshold:
                    stale_entities.append(feat.entity_id)

            if stale_entities:
                age_seconds = (now - max(
                    f.computed_at
                    for f in self._local_cache.values()
                    if f.name == name
                )).total_seconds()

                alert = {
                    "feature": name,
                    "status": FeatureStatus.STALE.value,
                    "threshold_seconds": threshold.total_seconds(),
                    "max_age_seconds": age_seconds,
                    "stale_entity_count": len(stale_entities),
                }
                alerts.append(alert)

                for cb in self._staleness_callbacks:
                    try:
                        cb(name, age_seconds)
                    except Exception:
                        logger.exception("staleness_callback_error", feature=name)

                logger.warning(
                    "feature_stale",
                    feature=name,
                    stale_entities=len(stale_entities),
                    max_age_seconds=age_seconds,
                )

        return alerts

    # ------------------------------------------------------------------
    # Versioning helpers
    # ------------------------------------------------------------------

    def get_versions(self, feature_name: str) -> list[FeatureVersion]:
        """Return the full version history for a feature."""
        return list(self._versions.get(feature_name, []))

    def get_active_version(self, feature_name: str) -> FeatureVersion | None:
        """Return the currently active version for a feature."""
        for v in reversed(self._versions.get(feature_name, [])):
            if v.is_active:
                return v
        return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return engine statistics for monitoring dashboards."""
        return {
            "registered_features": len(self._definitions),
            "cached_values": len(self._local_cache),
            "currently_computing": len(self._computing),
            "feature_names": sorted(self._definitions.keys()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(feature_name: str, entity_id: str) -> str:
        return f"{feature_name}:{entity_id}"

    @staticmethod
    def _redis_key(feature_name: str, entity_id: str) -> str:
        return f"mat_feature:{feature_name}:{entity_id}"

    @staticmethod
    def _compute_schema_hash(name: str, dependencies: list[str]) -> str:
        payload = json.dumps({"name": name, "deps": sorted(dependencies)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _get_from_local_cache(self, feature_name: str, entity_id: str) -> MaterializedFeature | None:
        return self._local_cache.get(self._cache_key(feature_name, entity_id))

    def _put_local_cache(self, feature: MaterializedFeature) -> None:
        self._local_cache[self._cache_key(feature.name, feature.entity_id)] = feature

    def _get_affected_features(self, changed_sources: list[str]) -> list[str]:
        """Walk the dependency graph to find all features affected by changes."""
        affected: set[str] = set()
        queue = list(changed_sources)
        while queue:
            current = queue.pop(0)
            for name, defn in self._definitions.items():
                if current in defn.dependencies and name not in affected:
                    affected.add(name)
                    queue.append(name)
            # Also include if directly changed
            if current in self._definitions:
                affected.add(current)
        return list(affected)

    def _topological_sort(self, feature_names: list[str]) -> list[str]:
        """Return features in dependency order (leaves first)."""
        visited: set[str] = set()
        result: list[str] = []

        def _visit(name: str) -> None:
            if name in visited or name not in self._definitions:
                return
            visited.add(name)
            for dep in self._definitions[name].dependencies:
                if dep in feature_names:
                    _visit(dep)
            result.append(name)

        for name in feature_names:
            _visit(name)
        return result

    # ------------------------------------------------------------------
    # Storage backends
    # ------------------------------------------------------------------

    async def _store_to_redis(self, feature: MaterializedFeature) -> None:
        """Persist a materialized feature to Redis for online serving."""
        try:
            from app.models.base import redis

            if redis:
                key = self._redis_key(feature.name, feature.entity_id)
                payload = json.dumps(feature.to_dict())
                await redis.setex(key, feature.ttl_seconds, payload)
        except Exception as exc:
            logger.debug("redis_store_failed", feature=feature.name, error=str(exc))

    async def _load_from_redis(self, feature_name: str, entity_id: str) -> MaterializedFeature | None:
        """Load a materialized feature from Redis."""
        try:
            from app.models.base import redis

            if redis:
                key = self._redis_key(feature_name, entity_id)
                raw = await redis.get(key)
                if raw:
                    return MaterializedFeature.from_dict(json.loads(raw))
        except Exception as exc:
            logger.debug("redis_load_failed", feature=feature_name, error=str(exc))
        return None

    async def _store_to_postgres(self, feature: MaterializedFeature) -> None:
        """Persist a materialized feature to PostgreSQL for offline analysis.

        Uses raw SQL so the engine remains decoupled from ORM model
        definitions.  The target table is ``materialized_features``.
        """
        try:
            from sqlalchemy import text

            from app.models.base import async_session

            async with async_session() as session:
                await session.execute(
                    text("""
                        INSERT INTO materialized_features
                            (name, entity_id, value, version, computed_at,
                             data_timestamp, schema_hash, metadata)
                        VALUES
                            (:name, :entity_id, :value, :version, :computed_at,
                             :data_timestamp, :schema_hash, :metadata)
                        ON CONFLICT (name, entity_id, version) DO UPDATE SET
                            value = EXCLUDED.value,
                            computed_at = EXCLUDED.computed_at,
                            data_timestamp = EXCLUDED.data_timestamp,
                            metadata = EXCLUDED.metadata
                    """),
                    {
                        "name": feature.name,
                        "entity_id": feature.entity_id,
                        "value": json.dumps(feature.value),
                        "version": feature.version,
                        "computed_at": feature.computed_at,
                        "data_timestamp": feature.data_timestamp,
                        "schema_hash": feature.metadata.get("schema_hash", ""),
                        "metadata": json.dumps(feature.metadata),
                    },
                )
                await session.commit()
        except Exception as exc:
            logger.debug("postgres_store_failed", feature=feature.name, error=str(exc))


feature_materializer = FeatureMaterializationEngine()
