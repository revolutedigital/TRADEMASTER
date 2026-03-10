"""Capacity simulation for TradeMaster infrastructure planning.

Models load growth, identifies bottlenecks, projects costs, and generates
scaling recommendations across CPU, memory, DB, and API dimensions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DB_CONNECTIONS = "db_connections"
    API_RATE_LIMIT = "api_rate_limit"
    DISK_IO = "disk_io"
    NETWORK_BANDWIDTH = "network_bandwidth"
    REDIS_MEMORY = "redis_memory"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CapacityProfile:
    """Snapshot of current infrastructure capacity and utilization."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Current resource limits
    cpu_cores: int = 4
    memory_gb: float = 16.0
    db_max_connections: int = 100
    api_rate_limit_per_sec: int = 100
    disk_iops: int = 3000
    network_bandwidth_mbps: float = 1000.0
    redis_max_memory_gb: float = 2.0

    # Current utilization (0.0 - 1.0)
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    db_connection_utilization: float = 0.0
    api_rate_utilization: float = 0.0
    disk_io_utilization: float = 0.0
    network_utilization: float = 0.0
    redis_memory_utilization: float = 0.0

    # Workload characteristics
    active_users: int = 0
    trades_per_second: float = 0.0
    tracked_symbols: int = 0
    data_volume_gb_per_day: float = 0.0

    def utilization_for(self, resource: ResourceType) -> float:
        """Return utilization ratio for a given resource type."""
        mapping = {
            ResourceType.CPU: self.cpu_utilization,
            ResourceType.MEMORY: self.memory_utilization,
            ResourceType.DB_CONNECTIONS: self.db_connection_utilization,
            ResourceType.API_RATE_LIMIT: self.api_rate_utilization,
            ResourceType.DISK_IO: self.disk_io_utilization,
            ResourceType.NETWORK_BANDWIDTH: self.network_utilization,
            ResourceType.REDIS_MEMORY: self.redis_memory_utilization,
        }
        return mapping.get(resource, 0.0)

    def headroom_for(self, resource: ResourceType) -> float:
        """Return remaining capacity as a ratio (1.0 = fully free)."""
        return max(0.0, 1.0 - self.utilization_for(resource))


@dataclass
class GrowthScenario:
    """Defines a growth multiplier scenario for capacity planning."""

    name: str
    user_multiplier: float
    trades_multiplier: float
    data_multiplier: float
    symbol_multiplier: float = 1.0
    description: str = ""

    @classmethod
    def moderate(cls) -> GrowthScenario:
        return cls(
            name="10x",
            user_multiplier=10.0,
            trades_multiplier=10.0,
            data_multiplier=8.0,
            symbol_multiplier=3.0,
            description="Moderate growth: 10x users and trades, 8x data",
        )

    @classmethod
    def aggressive(cls) -> GrowthScenario:
        return cls(
            name="100x",
            user_multiplier=100.0,
            trades_multiplier=100.0,
            data_multiplier=60.0,
            symbol_multiplier=10.0,
            description="Aggressive growth: 100x users and trades, 60x data",
        )

    @classmethod
    def extreme(cls) -> GrowthScenario:
        return cls(
            name="1000x",
            user_multiplier=1000.0,
            trades_multiplier=1000.0,
            data_multiplier=400.0,
            symbol_multiplier=30.0,
            description="Extreme growth: 1000x users and trades, 400x data",
        )


@dataclass
class Bottleneck:
    """An identified resource bottleneck under a growth scenario."""

    resource: ResourceType
    current_utilization: float
    projected_utilization: float
    severity: Severity
    scenario_name: str
    message: str

    @property
    def overcommit_ratio(self) -> float:
        """How many times over capacity the resource is projected to be."""
        if self.projected_utilization <= 1.0:
            return 0.0
        return self.projected_utilization


@dataclass
class ScalingRecommendation:
    """A concrete recommendation for scaling a resource."""

    resource: ResourceType
    current_value: float
    recommended_value: float
    scaling_factor: float
    priority: Severity
    estimated_monthly_cost_usd: float
    action: str
    rationale: str

    @property
    def cost_increase_usd(self) -> float:
        """Estimated additional monthly cost."""
        return self.estimated_monthly_cost_usd


@dataclass
class CostProjection:
    """Monthly cost projection for a growth scenario."""

    scenario_name: str
    current_monthly_cost_usd: float
    projected_monthly_cost_usd: float
    cost_breakdown: dict[ResourceType, float] = field(default_factory=dict)
    recommendations: list[ScalingRecommendation] = field(default_factory=list)

    @property
    def cost_multiplier(self) -> float:
        if self.current_monthly_cost_usd == 0:
            return 0.0
        return self.projected_monthly_cost_usd / self.current_monthly_cost_usd


@dataclass
class WhatIfResult:
    """Result of a what-if capacity analysis."""

    description: str
    additional_symbols: int
    additional_users: int
    resource_impacts: dict[ResourceType, float] = field(default_factory=dict)
    bottlenecks: list[Bottleneck] = field(default_factory=list)
    recommendations: list[ScalingRecommendation] = field(default_factory=list)
    feasible: bool = True
    notes: list[str] = field(default_factory=list)


# -- Cost model constants (monthly USD per unit, approximate cloud pricing) --

_COST_PER_CPU_CORE = 35.0          # per vCPU/month
_COST_PER_GB_MEMORY = 5.0          # per GB RAM/month
_COST_PER_DB_CONN_BLOCK = 25.0     # per 50 connections (larger instance)
_COST_PER_1K_API_RPS = 20.0        # API gateway / load balancer cost per 1k rps
_COST_PER_1K_IOPS = 10.0           # per 1000 provisioned IOPS
_COST_PER_GBPS_NETWORK = 50.0      # per Gbps sustained
_COST_PER_GB_REDIS = 15.0          # per GB Redis memory

# Resource scaling factors per workload dimension
# Each row: how much does 1 unit of workload dimension consume of each resource
_RESOURCE_SCALING_WEIGHTS: dict[ResourceType, dict[str, float]] = {
    ResourceType.CPU: {
        "per_user": 0.005,           # 0.5% CPU per concurrent user
        "per_trade_sec": 0.03,       # 3% CPU per trade/sec
        "per_symbol": 0.002,         # 0.2% CPU per tracked symbol
        "per_gb_data_day": 0.01,     # 1% CPU per GB/day ingested
    },
    ResourceType.MEMORY: {
        "per_user": 0.003,
        "per_trade_sec": 0.01,
        "per_symbol": 0.004,
        "per_gb_data_day": 0.008,
    },
    ResourceType.DB_CONNECTIONS: {
        "per_user": 0.008,
        "per_trade_sec": 0.02,
        "per_symbol": 0.001,
        "per_gb_data_day": 0.005,
    },
    ResourceType.API_RATE_LIMIT: {
        "per_user": 0.01,
        "per_trade_sec": 0.05,
        "per_symbol": 0.003,
        "per_gb_data_day": 0.002,
    },
    ResourceType.DISK_IO: {
        "per_user": 0.001,
        "per_trade_sec": 0.015,
        "per_symbol": 0.001,
        "per_gb_data_day": 0.02,
    },
    ResourceType.NETWORK_BANDWIDTH: {
        "per_user": 0.002,
        "per_trade_sec": 0.005,
        "per_symbol": 0.003,
        "per_gb_data_day": 0.015,
    },
    ResourceType.REDIS_MEMORY: {
        "per_user": 0.005,
        "per_trade_sec": 0.008,
        "per_symbol": 0.006,
        "per_gb_data_day": 0.003,
    },
}


class CapacitySimulator:
    """Simulates infrastructure capacity under growth scenarios.

    Analyzes current resource utilization, projects future needs under
    various growth scenarios, identifies bottlenecks, and generates
    actionable scaling recommendations with cost estimates.
    """

    # Utilization thresholds for severity classification
    THRESHOLD_LOW = 0.60
    THRESHOLD_MEDIUM = 0.75
    THRESHOLD_HIGH = 0.85
    THRESHOLD_CRITICAL = 0.95

    def __init__(
        self,
        profile: CapacityProfile,
        *,
        safety_margin: float = 0.20,
        cost_per_cpu: float = _COST_PER_CPU_CORE,
        cost_per_gb_mem: float = _COST_PER_GB_MEMORY,
    ):
        self.profile = profile
        self.safety_margin = safety_margin
        self._cost_per_cpu = cost_per_cpu
        self._cost_per_gb_mem = cost_per_gb_mem

    # -- Public API --

    def simulate_growth(
        self,
        scenario: GrowthScenario,
    ) -> dict[ResourceType, float]:
        """Project resource utilization under a growth scenario.

        Returns a mapping of ResourceType to projected utilization (can exceed 1.0).
        """
        projected: dict[ResourceType, float] = {}
        for resource in ResourceType:
            projected[resource] = self._project_utilization(resource, scenario)
        return projected

    def find_bottlenecks(
        self,
        scenario: GrowthScenario,
    ) -> list[Bottleneck]:
        """Identify resources that become bottlenecks under the scenario."""
        projected = self.simulate_growth(scenario)
        bottlenecks: list[Bottleneck] = []

        for resource, proj_util in projected.items():
            if proj_util < self.THRESHOLD_LOW:
                continue

            severity = self._classify_severity(proj_util)
            bottlenecks.append(
                Bottleneck(
                    resource=resource,
                    current_utilization=self.profile.utilization_for(resource),
                    projected_utilization=proj_util,
                    severity=severity,
                    scenario_name=scenario.name,
                    message=self._bottleneck_message(resource, proj_util, scenario),
                )
            )

        bottlenecks.sort(
            key=lambda b: b.projected_utilization, reverse=True
        )
        return bottlenecks

    def project_costs(
        self,
        scenarios: list[GrowthScenario] | None = None,
    ) -> list[CostProjection]:
        """Generate cost projections for one or more growth scenarios.

        If no scenarios are provided, uses the three standard scenarios
        (10x, 100x, 1000x).
        """
        if scenarios is None:
            scenarios = [
                GrowthScenario.moderate(),
                GrowthScenario.aggressive(),
                GrowthScenario.extreme(),
            ]

        current_cost = self._estimate_current_cost()
        projections: list[CostProjection] = []

        for scenario in scenarios:
            recs = self.generate_recommendations(scenario)
            cost_breakdown = self._cost_breakdown_for_scenario(scenario)
            projected_cost = sum(cost_breakdown.values())

            projections.append(
                CostProjection(
                    scenario_name=scenario.name,
                    current_monthly_cost_usd=round(current_cost, 2),
                    projected_monthly_cost_usd=round(projected_cost, 2),
                    cost_breakdown=cost_breakdown,
                    recommendations=recs,
                )
            )

        return projections

    def generate_recommendations(
        self,
        scenario: GrowthScenario,
    ) -> list[ScalingRecommendation]:
        """Generate scaling recommendations for a growth scenario."""
        bottlenecks = self.find_bottlenecks(scenario)
        recommendations: list[ScalingRecommendation] = []

        for bn in bottlenecks:
            rec = self._recommendation_for_bottleneck(bn, scenario)
            if rec is not None:
                recommendations.append(rec)

        recommendations.sort(
            key=lambda r: list(Severity).index(r.priority), reverse=True
        )
        return recommendations

    def what_if_add_symbols(
        self,
        additional_symbols: int,
        additional_users: int = 0,
    ) -> WhatIfResult:
        """Analyze infrastructure impact of adding symbols and/or users.

        Example: "If we add 50 symbols, how much more infra do we need?"
        """
        if additional_symbols < 0 or additional_users < 0:
            raise ValueError("Additional counts must be non-negative")

        total_symbols = self.profile.tracked_symbols + additional_symbols
        total_users = self.profile.active_users + additional_users

        # Build an ad-hoc scenario from the deltas
        sym_mult = total_symbols / max(self.profile.tracked_symbols, 1)
        user_mult = total_users / max(self.profile.active_users, 1)

        # Trades and data scale proportionally to symbols * users
        combined_mult = (sym_mult + user_mult) / 2.0
        scenario = GrowthScenario(
            name=f"what_if_+{additional_symbols}sym_+{additional_users}usr",
            user_multiplier=user_mult,
            trades_multiplier=combined_mult,
            data_multiplier=combined_mult * 0.8,
            symbol_multiplier=sym_mult,
            description=(
                f"What-if: adding {additional_symbols} symbols "
                f"and {additional_users} users"
            ),
        )

        projected = self.simulate_growth(scenario)
        bottlenecks = self.find_bottlenecks(scenario)
        recommendations = self.generate_recommendations(scenario)

        feasible = all(proj <= 1.0 for proj in projected.values())
        notes: list[str] = []

        if not feasible:
            over_resources = [
                r.value for r, p in projected.items() if p > 1.0
            ]
            notes.append(
                f"Exceeds capacity on: {', '.join(over_resources)}. "
                "Scaling required before adding this workload."
            )

        if additional_symbols > 100:
            notes.append(
                "Adding >100 symbols may require sharding the market data "
                "stream processor across multiple workers."
            )

        if additional_users > 500:
            notes.append(
                "Adding >500 users may require horizontal scaling of the "
                "backend with a load balancer."
            )

        return WhatIfResult(
            description=scenario.description,
            additional_symbols=additional_symbols,
            additional_users=additional_users,
            resource_impacts=projected,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            feasible=feasible,
            notes=notes,
        )

    def summary(self) -> dict[str, Any]:
        """Return a human-readable summary of current capacity state."""
        profile = self.profile
        resources: dict[str, dict[str, Any]] = {}
        for resource in ResourceType:
            util = profile.utilization_for(resource)
            resources[resource.value] = {
                "utilization_pct": round(util * 100, 1),
                "headroom_pct": round(profile.headroom_for(resource) * 100, 1),
                "status": self._classify_severity(util).value if util >= self.THRESHOLD_LOW else "ok",
            }

        return {
            "timestamp": profile.timestamp.isoformat(),
            "workload": {
                "active_users": profile.active_users,
                "trades_per_second": profile.trades_per_second,
                "tracked_symbols": profile.tracked_symbols,
                "data_volume_gb_per_day": profile.data_volume_gb_per_day,
            },
            "resources": resources,
            "estimated_monthly_cost_usd": round(self._estimate_current_cost(), 2),
        }

    # -- Private helpers --

    def _project_utilization(
        self,
        resource: ResourceType,
        scenario: GrowthScenario,
    ) -> float:
        """Project utilization for a single resource under a scenario."""
        current = self.profile.utilization_for(resource)
        weights = _RESOURCE_SCALING_WEIGHTS.get(resource, {})

        # Current workload contribution
        current_workload = (
            weights.get("per_user", 0) * self.profile.active_users
            + weights.get("per_trade_sec", 0) * self.profile.trades_per_second
            + weights.get("per_symbol", 0) * self.profile.tracked_symbols
            + weights.get("per_gb_data_day", 0) * self.profile.data_volume_gb_per_day
        )

        # Projected workload contribution
        projected_workload = (
            weights.get("per_user", 0) * self.profile.active_users * scenario.user_multiplier
            + weights.get("per_trade_sec", 0) * self.profile.trades_per_second * scenario.trades_multiplier
            + weights.get("per_symbol", 0) * self.profile.tracked_symbols * scenario.symbol_multiplier
            + weights.get("per_gb_data_day", 0) * self.profile.data_volume_gb_per_day * scenario.data_multiplier
        )

        if current_workload > 0:
            # Scale current utilization by the ratio of projected / current workload
            growth_ratio = projected_workload / current_workload
            projected = current * growth_ratio
        else:
            # No existing workload; estimate from scratch
            projected = projected_workload

        # Apply sub-linear scaling for large growth (Amdahl's-like diminishing)
        if projected > 1.0:
            projected = 1.0 + math.log2(projected)

        return round(projected, 4)

    def _classify_severity(self, utilization: float) -> Severity:
        if utilization >= self.THRESHOLD_CRITICAL:
            return Severity.CRITICAL
        if utilization >= self.THRESHOLD_HIGH:
            return Severity.HIGH
        if utilization >= self.THRESHOLD_MEDIUM:
            return Severity.MEDIUM
        return Severity.LOW

    def _bottleneck_message(
        self,
        resource: ResourceType,
        projected: float,
        scenario: GrowthScenario,
    ) -> str:
        pct = round(projected * 100, 1)
        messages = {
            ResourceType.CPU: (
                f"CPU projected at {pct}% under {scenario.name} scenario. "
                "Consider vertical scaling or distributing compute across workers."
            ),
            ResourceType.MEMORY: (
                f"Memory projected at {pct}% under {scenario.name} scenario. "
                "Review in-memory caches and consider offloading to Redis."
            ),
            ResourceType.DB_CONNECTIONS: (
                f"DB connections projected at {pct}% under {scenario.name} scenario. "
                "Implement connection pooling (PgBouncer) or read replicas."
            ),
            ResourceType.API_RATE_LIMIT: (
                f"API rate limit projected at {pct}% under {scenario.name} scenario. "
                "Add request queuing, caching, or increase rate limits."
            ),
            ResourceType.DISK_IO: (
                f"Disk I/O projected at {pct}% under {scenario.name} scenario. "
                "Upgrade to higher IOPS storage or add read replicas."
            ),
            ResourceType.NETWORK_BANDWIDTH: (
                f"Network bandwidth projected at {pct}% under {scenario.name} scenario. "
                "Enable compression, CDN for static assets, or upgrade network tier."
            ),
            ResourceType.REDIS_MEMORY: (
                f"Redis memory projected at {pct}% under {scenario.name} scenario. "
                "Review TTLs, eviction policies, or scale Redis cluster."
            ),
        }
        return messages.get(resource, f"{resource.value} at {pct}%")

    def _recommendation_for_bottleneck(
        self,
        bottleneck: Bottleneck,
        scenario: GrowthScenario,
    ) -> ScalingRecommendation | None:
        """Generate a concrete scaling recommendation for a bottleneck."""
        resource = bottleneck.resource
        proj = bottleneck.projected_utilization

        # Target utilization after scaling: 1.0 - safety_margin
        target = 1.0 - self.safety_margin
        if proj <= target:
            return None

        scaling_factor = math.ceil(proj / target * 10) / 10  # round up to 0.1

        current_val, rec_val, cost, action = self._scaling_details(
            resource, scaling_factor
        )

        return ScalingRecommendation(
            resource=resource,
            current_value=current_val,
            recommended_value=rec_val,
            scaling_factor=scaling_factor,
            priority=bottleneck.severity,
            estimated_monthly_cost_usd=round(cost, 2),
            action=action,
            rationale=bottleneck.message,
        )

    def _scaling_details(
        self,
        resource: ResourceType,
        factor: float,
    ) -> tuple[float, float, float, str]:
        """Return (current_value, recommended_value, monthly_cost, action)."""
        p = self.profile
        details: dict[ResourceType, tuple[float, float, float, str]] = {
            ResourceType.CPU: (
                p.cpu_cores,
                math.ceil(p.cpu_cores * factor),
                math.ceil(p.cpu_cores * factor) * self._cost_per_cpu,
                f"Scale CPU from {p.cpu_cores} to {math.ceil(p.cpu_cores * factor)} cores",
            ),
            ResourceType.MEMORY: (
                p.memory_gb,
                math.ceil(p.memory_gb * factor),
                math.ceil(p.memory_gb * factor) * self._cost_per_gb_mem,
                f"Scale memory from {p.memory_gb}GB to {math.ceil(p.memory_gb * factor)}GB",
            ),
            ResourceType.DB_CONNECTIONS: (
                p.db_max_connections,
                math.ceil(p.db_max_connections * factor / 50) * 50,
                math.ceil(p.db_max_connections * factor / 50) * _COST_PER_DB_CONN_BLOCK,
                (
                    f"Scale DB connections from {p.db_max_connections} to "
                    f"{math.ceil(p.db_max_connections * factor / 50) * 50} "
                    "(deploy PgBouncer if not present)"
                ),
            ),
            ResourceType.API_RATE_LIMIT: (
                p.api_rate_limit_per_sec,
                math.ceil(p.api_rate_limit_per_sec * factor),
                math.ceil(p.api_rate_limit_per_sec * factor / 1000) * _COST_PER_1K_API_RPS,
                (
                    f"Increase API rate limit from {p.api_rate_limit_per_sec} to "
                    f"{math.ceil(p.api_rate_limit_per_sec * factor)} req/s"
                ),
            ),
            ResourceType.DISK_IO: (
                p.disk_iops,
                math.ceil(p.disk_iops * factor / 1000) * 1000,
                math.ceil(p.disk_iops * factor / 1000) * _COST_PER_1K_IOPS,
                (
                    f"Scale disk IOPS from {p.disk_iops} to "
                    f"{math.ceil(p.disk_iops * factor / 1000) * 1000}"
                ),
            ),
            ResourceType.NETWORK_BANDWIDTH: (
                p.network_bandwidth_mbps,
                math.ceil(p.network_bandwidth_mbps * factor),
                math.ceil(p.network_bandwidth_mbps * factor / 1000) * _COST_PER_GBPS_NETWORK,
                (
                    f"Scale network from {p.network_bandwidth_mbps}Mbps to "
                    f"{math.ceil(p.network_bandwidth_mbps * factor)}Mbps"
                ),
            ),
            ResourceType.REDIS_MEMORY: (
                p.redis_max_memory_gb,
                math.ceil(p.redis_max_memory_gb * factor),
                math.ceil(p.redis_max_memory_gb * factor) * _COST_PER_GB_REDIS,
                (
                    f"Scale Redis from {p.redis_max_memory_gb}GB to "
                    f"{math.ceil(p.redis_max_memory_gb * factor)}GB"
                ),
            ),
        }
        return details[resource]

    def _estimate_current_cost(self) -> float:
        """Estimate current monthly infrastructure cost."""
        p = self.profile
        return (
            p.cpu_cores * self._cost_per_cpu
            + p.memory_gb * self._cost_per_gb_mem
            + (p.db_max_connections / 50) * _COST_PER_DB_CONN_BLOCK
            + (p.api_rate_limit_per_sec / 1000) * _COST_PER_1K_API_RPS
            + (p.disk_iops / 1000) * _COST_PER_1K_IOPS
            + (p.network_bandwidth_mbps / 1000) * _COST_PER_GBPS_NETWORK
            + p.redis_max_memory_gb * _COST_PER_GB_REDIS
        )

    def _cost_breakdown_for_scenario(
        self,
        scenario: GrowthScenario,
    ) -> dict[ResourceType, float]:
        """Estimate per-resource cost for a scenario."""
        projected = self.simulate_growth(scenario)
        breakdown: dict[ResourceType, float] = {}

        for resource in ResourceType:
            proj_util = projected[resource]
            # Scale cost proportionally; if over 1.0, we need more infra
            factor = max(1.0, proj_util / (1.0 - self.safety_margin))
            _, _, cost, _ = self._scaling_details(resource, factor)
            breakdown[resource] = round(cost, 2)

        return breakdown
