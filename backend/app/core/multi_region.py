"""Multi-region deployment support with failover and replication.

Manages cross-region health monitoring and automatic failover.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class RegionStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILOVER = "failover"


@dataclass
class Region:
    name: str
    endpoint: str
    is_primary: bool = False
    status: RegionStatus = RegionStatus.HEALTHY
    latency_ms: float = 0.0
    last_health_check: datetime | None = None
    error_count: int = 0


class MultiRegionManager:
    """Manage multi-region deployment with automatic failover."""

    def __init__(self):
        self._regions: dict[str, Region] = {}
        self._primary: str | None = None

    def add_region(self, name: str, endpoint: str, is_primary: bool = False) -> Region:
        region = Region(name=name, endpoint=endpoint, is_primary=is_primary)
        self._regions[name] = region
        if is_primary:
            self._primary = name
        logger.info("region_added", name=name, endpoint=endpoint, primary=is_primary)
        return region

    def get_primary(self) -> Region | None:
        if self._primary and self._primary in self._regions:
            return self._regions[self._primary]
        return None

    async def health_check(self, region_name: str) -> bool:
        """Check health of a specific region."""
        region = self._regions.get(region_name)
        if not region:
            return False

        region.last_health_check = datetime.now(timezone.utc)
        # In production: HTTP health check to region endpoint
        return region.status == RegionStatus.HEALTHY

    async def failover(self, from_region: str) -> str | None:
        """Failover from one region to the next healthy region."""
        candidates = [
            r for r in self._regions.values()
            if r.name != from_region and r.status == RegionStatus.HEALTHY
        ]
        if not candidates:
            logger.error("no_healthy_regions_for_failover")
            return None

        # Select region with lowest latency
        target = min(candidates, key=lambda r: r.latency_ms)
        target.is_primary = True
        self._primary = target.name

        old_region = self._regions.get(from_region)
        if old_region:
            old_region.is_primary = False
            old_region.status = RegionStatus.FAILOVER

        logger.warning("region_failover", from_region=from_region, to_region=target.name)
        return target.name

    def get_status(self) -> dict:
        return {
            "primary": self._primary,
            "regions": {
                name: {
                    "endpoint": r.endpoint,
                    "status": r.status.value,
                    "is_primary": r.is_primary,
                    "latency_ms": r.latency_ms,
                    "last_check": r.last_health_check.isoformat() if r.last_health_check else None,
                    "error_count": r.error_count,
                }
                for name, r in self._regions.items()
            },
        }

    def initialize_default_regions(self):
        """Initialize default region configuration."""
        self.add_region("us-east-1", "https://us-east.trademaster.app", is_primary=True)
        self.add_region("eu-west-1", "https://eu-west.trademaster.app")
        self.add_region("ap-southeast-1", "https://ap-southeast.trademaster.app")


multi_region = MultiRegionManager()
