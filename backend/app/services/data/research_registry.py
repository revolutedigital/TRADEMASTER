"""Fail-closed registry for preregistered research and burned data."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import (
    ResearchDataUse,
    ResearchExperiment,
    ResearchExperimentEvent,
    ResearchHypothesisAttempt,
)
from app.repositories.research_experiment_repo import (
    ResearchExperimentRepository,
    research_experiment_repository,
)
from app.services.data.lineage_tracker import EndToEndLineageTracker, lineage_tracker


SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
GIT_REVISION_PATTERN = re.compile(r"^[a-f0-9]{40}$")
ROLE_ORDER = {
    "DEVELOPMENT": 0,
    "TRAINING": 1,
    "SELECTION": 2,
    "AUDIT": 3,
    "PROSPECTIVE_SHADOW": 4,
}
PROTECTED_ROLES = {"AUDIT", "PROSPECTIVE_SHADOW"}
REQUIRED_ROLES = {"DEVELOPMENT", "TRAINING", "SELECTION", "AUDIT"}
DECISION_STATUSES = {"REJECTED", "INCONCLUSIVE", "APPROVED"}
MIN_PROSPECTIVE_SHADOW_DURATION = timedelta(days=20)
MAX_PROSPECTIVE_SHADOW_DURATION = timedelta(days=30)

V1_PRODUCT = {
    "execution_venue": "binance_usdm_futures",
    "execution_product": "perpetual",
    "symbol": "BTCUSDT",
    "directions": ["LONG", "SHORT"],
    "auxiliary_signal_venue": "binance_spot",
    "horizons_seconds": [5, 15, 30, 120, 300],
    "position_model": "one_net_position",
}
V1_GATE = {
    "min_oos_folds": 3,
    "min_oos_portfolio_trades": 200,
    "min_oos_utc_days": 20,
    "adjusted_one_sided_confidence": 0.95,
    "max_probability_backtest_overfitting": 0.2,
    "book_evidence_min_complete_days": 60,
    "prospective_shadow_min_days": 20,
    "prospective_shadow_max_days": 30,
    "top_p_tails_pct": [1, 2, 5, 10],
}


class ResearchRegistryError(ValueError):
    """Base error for invalid or unsafe experiment lifecycle changes."""


class FrozenExperimentError(ResearchRegistryError):
    """Raised when immutable experiment definition is changed."""


class BurnedDataConflict(ResearchRegistryError):
    """Raised when protected outcomes would be reused as discovery data."""


@dataclass(frozen=True)
class PartitionDefinition:
    role: str
    start_at: datetime
    end_at: datetime
    manifest_sha256: str


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    code_revision: str
    protocol_sha256: str
    product: dict[str, Any]
    cost_profile: dict[str, Any]
    approval_gate: dict[str, Any]
    partitions: tuple[PartitionDefinition, ...]


class ResearchRegistry:
    """Own experiment immutability, hypothesis counting, and data-use safety."""

    def __init__(
        self,
        repository: ResearchExperimentRepository | None = None,
        lineage: EndToEndLineageTracker | None = None,
    ) -> None:
        self._repository = repository or research_experiment_repository
        self._lineage = lineage or lineage_tracker

    async def create_draft(
        self,
        db: AsyncSession,
        definition: ExperimentDefinition,
    ) -> ResearchExperiment:
        """Create metadata only; no data outcome is opened by this operation."""
        self._validate_definition(definition)
        experiment = ResearchExperiment(
            id=str(uuid.uuid4()),
            name=definition.name.strip(),
            status="DRAFT",
            code_revision=definition.code_revision,
            protocol_sha256=definition.protocol_sha256,
            product_json=_canonical_json(definition.product),
            cost_profile_json=_canonical_json(definition.cost_profile),
            approval_gate_json=_canonical_json(definition.approval_gate),
        )
        db.add(experiment)
        await db.flush()
        for partition in definition.partitions:
            db.add(
                ResearchDataUse(
                    experiment_id=experiment.id,
                    role=partition.role,
                    start_at=partition.start_at,
                    end_at=partition.end_at,
                    manifest_sha256=partition.manifest_sha256,
                )
            )
        await self._append_event(db, experiment.id, "DRAFT_CREATED", {})
        await db.flush()
        return experiment

    async def register_hypothesis(
        self,
        db: AsyncSession,
        experiment_id: str,
        *,
        kind: str,
        definition: dict[str, Any],
    ) -> ResearchHypothesisAttempt:
        experiment = await self._require_experiment(db, experiment_id, for_update=True)
        if experiment.status != "DRAFT":
            raise FrozenExperimentError("Only DRAFT experiments accept hypotheses")
        normalized_kind = kind.strip().upper()
        if not normalized_kind or len(normalized_kind) > 40:
            raise ResearchRegistryError("Hypothesis kind must contain 1 to 40 characters")
        definition_json = _canonical_json(definition)
        fingerprint = _sha256(f"{normalized_kind}:{definition_json}")
        hypothesis = ResearchHypothesisAttempt(
            experiment_id=experiment_id,
            kind=normalized_kind,
            fingerprint_sha256=fingerprint,
            definition_json=definition_json,
            status="REGISTERED",
        )
        db.add(hypothesis)
        await self._append_event(
            db,
            experiment_id,
            "HYPOTHESIS_REGISTERED",
            {"kind": normalized_kind, "fingerprint_sha256": fingerprint},
        )
        await db.flush()
        return hypothesis

    async def freeze(
        self,
        db: AsyncSession,
        experiment_id: str,
    ) -> ResearchExperiment:
        """Irreversibly hash the complete experiment before protected outcomes open."""
        experiment = await self._require_experiment(db, experiment_id, for_update=True)
        if experiment.status != "DRAFT":
            raise FrozenExperimentError("Only a DRAFT experiment can be frozen")
        partitions = await self._repository.list_partitions(db, experiment_id, for_update=True)
        hypotheses = await self._repository.list_hypotheses(db, experiment_id)
        if not hypotheses:
            raise ResearchRegistryError("At least one hypothesis must be registered before freeze")
        await self._ensure_no_burned_data_conflicts(db, experiment_id, partitions)

        freeze_payload = {
            "id": experiment.id,
            "name": experiment.name,
            "code_revision": experiment.code_revision,
            "protocol_sha256": experiment.protocol_sha256,
            "product": json.loads(experiment.product_json),
            "cost_profile": json.loads(experiment.cost_profile_json),
            "approval_gate": json.loads(experiment.approval_gate_json),
            "partitions": [
                {
                    "role": partition.role,
                    "start_at": _utc_iso(partition.start_at),
                    "end_at": _utc_iso(partition.end_at),
                    "manifest_sha256": partition.manifest_sha256,
                }
                for partition in partitions
            ],
            "hypotheses": [
                {
                    "kind": hypothesis.kind,
                    "fingerprint_sha256": hypothesis.fingerprint_sha256,
                    "definition": json.loads(hypothesis.definition_json),
                }
                for hypothesis in hypotheses
            ],
        }
        experiment.experiment_sha256 = _sha256(_canonical_json(freeze_payload))
        experiment.status = "FROZEN"
        experiment.frozen_at = datetime.now(UTC)
        await self._append_event(
            db,
            experiment_id,
            "EXPERIMENT_FROZEN",
            {"experiment_sha256": experiment.experiment_sha256},
        )
        await db.flush()
        return experiment

    async def open_partition(
        self,
        db: AsyncSession,
        experiment_id: str,
        *,
        role: str,
    ) -> ResearchDataUse:
        """Record the irreversible first access to one experiment partition."""
        experiment = await self._require_experiment(db, experiment_id, for_update=True)
        if experiment.status != "FROZEN":
            raise ResearchRegistryError("Only FROZEN experiments may open data partitions")
        normalized_role = role.strip().upper()
        partitions = await self._repository.list_partitions(db, experiment_id, for_update=True)
        partition = next((item for item in partitions if item.role == normalized_role), None)
        if partition is None:
            raise ResearchRegistryError(f"Partition role {normalized_role} is not registered")
        if partition.opened_at is not None:
            return partition
        await self._ensure_partition_can_open(db, experiment_id, partition)
        partition.opened_at = datetime.now(UTC)
        await self._append_event(
            db,
            experiment_id,
            "PARTITION_OPENED",
            {
                "role": partition.role,
                "manifest_sha256": partition.manifest_sha256,
            },
        )
        await db.flush()
        return partition

    async def record_decision(
        self,
        db: AsyncSession,
        experiment_id: str,
        *,
        status: str,
        reasons: Iterable[str],
    ) -> ResearchExperiment:
        experiment = await self._require_experiment(db, experiment_id, for_update=True)
        normalized_status = status.strip().upper()
        if experiment.status != "FROZEN":
            raise FrozenExperimentError("Only a FROZEN experiment may receive a decision")
        if normalized_status not in DECISION_STATUSES:
            raise ResearchRegistryError("Invalid research decision status")
        normalized_reasons = [reason.strip() for reason in reasons if reason.strip()]
        if not normalized_reasons:
            raise ResearchRegistryError("A research decision requires at least one reason")
        experiment.status = normalized_status
        experiment.decision_reasons_json = _canonical_json(normalized_reasons)
        experiment.decided_at = datetime.now(UTC)
        await self._append_event(
            db,
            experiment_id,
            "DECISION_RECORDED",
            {"status": normalized_status, "reasons": normalized_reasons},
        )
        await db.flush()
        return experiment

    async def publish_lineage(
        self,
        db: AsyncSession,
        experiment_id: str,
    ) -> None:
        """Publish committed experiment data sources into the existing lineage DAG."""
        experiment = await self._require_experiment(db, experiment_id)
        if experiment.status == "DRAFT" or not experiment.experiment_sha256:
            raise ResearchRegistryError("Only frozen experiments have publishable lineage")
        source_id = f"research-protocol:{experiment.protocol_sha256}"
        self._lineage.register_data_source(
            source_id,
            "Microstructure research protocol",
            protocol_sha256=experiment.protocol_sha256,
        )
        for partition in await self._repository.list_partitions(db, experiment_id):
            self._lineage.register_raw_data(
                data_id=f"research-partition:{partition.manifest_sha256}",
                name=f"{partition.role} partition",
                source_id=source_id,
                experiment_id=experiment.id,
                role=partition.role,
                manifest_sha256=partition.manifest_sha256,
            )

    def _validate_definition(self, definition: ExperimentDefinition) -> None:
        if not 3 <= len(definition.name.strip()) <= 120:
            raise ResearchRegistryError("Experiment name must contain 3 to 120 characters")
        if not GIT_REVISION_PATTERN.fullmatch(definition.code_revision):
            raise ResearchRegistryError("code_revision must be a full lowercase Git SHA")
        _require_sha256(definition.protocol_sha256, "protocol_sha256")
        if definition.product != V1_PRODUCT:
            raise ResearchRegistryError("Product contract does not match microstructure v1")
        for key, expected_value in V1_GATE.items():
            if definition.approval_gate.get(key) != expected_value:
                raise ResearchRegistryError(f"Approval gate field {key} does not match v1")
        if float(definition.approval_gate.get("min_expected_cost_lcb_bps", 0)) <= 0:
            raise ResearchRegistryError("Expected-cost lower bound must be positive")
        if float(definition.approval_gate.get("min_stress_cost_mean_bps", 0)) <= 0:
            raise ResearchRegistryError("Stress-cost mean must be positive")
        self._validate_cost_profile(definition.cost_profile)
        self._validate_partitions(definition.partitions)

    @staticmethod
    def _validate_cost_profile(cost_profile: dict[str, Any]) -> None:
        required = {
            "maker_fee_bps_per_side",
            "taker_fee_bps_per_side",
            "expected_slippage_bps_per_side",
            "expected_latency_ms",
            "stress_roundtrip_bps",
            "default_order_style",
        }
        if set(cost_profile) != required:
            raise ResearchRegistryError("Cost profile fields do not match the v1 contract")
        numeric_fields = required - {"default_order_style"}
        if any(float(cost_profile[field]) < 0 for field in numeric_fields):
            raise ResearchRegistryError("Cost profile values must be non-negative")
        if cost_profile["default_order_style"] != "marketable_taker":
            raise ResearchRegistryError("v1 defaults to marketable taker execution")
        expected_roundtrip = 2 * (
            float(cost_profile["taker_fee_bps_per_side"])
            + float(cost_profile["expected_slippage_bps_per_side"])
        )
        stress_roundtrip = float(cost_profile["stress_roundtrip_bps"])
        if stress_roundtrip < max(20.0, 2 * expected_roundtrip):
            raise ResearchRegistryError(
                "Stress round-trip cost must be at least 20 bps and twice expected friction"
            )

    @staticmethod
    def _validate_partitions(partitions: tuple[PartitionDefinition, ...]) -> None:
        roles = {partition.role for partition in partitions}
        if not REQUIRED_ROLES.issubset(roles) or len(roles) != len(partitions):
            raise ResearchRegistryError(
                "Partitions require unique development/train/select/audit roles"
            )
        manifests = {partition.manifest_sha256 for partition in partitions}
        if len(manifests) != len(partitions):
            raise ResearchRegistryError("One manifest cannot serve multiple temporal roles")
        if any(role not in ROLE_ORDER for role in roles):
            raise ResearchRegistryError("Unknown dataset partition role")
        ordered = sorted(partitions, key=lambda item: ROLE_ORDER[item.role])
        for index, partition in enumerate(ordered):
            _require_sha256(partition.manifest_sha256, "manifest_sha256")
            if partition.start_at.tzinfo is None or partition.end_at.tzinfo is None:
                raise ResearchRegistryError("Partition timestamps must be timezone-aware")
            if partition.end_at <= partition.start_at:
                raise ResearchRegistryError("Partition end must follow its start")
            if partition.role == "PROSPECTIVE_SHADOW":
                duration = partition.end_at - partition.start_at
                if not (
                    MIN_PROSPECTIVE_SHADOW_DURATION
                    <= duration
                    <= MAX_PROSPECTIVE_SHADOW_DURATION
                ):
                    raise ResearchRegistryError(
                        "Prospective shadow partition must be 20 to 30 days"
                    )
            if index and partition.start_at < ordered[index - 1].end_at:
                raise ResearchRegistryError("Temporal partitions may not overlap")

    async def _ensure_no_burned_data_conflicts(
        self,
        db: AsyncSession,
        experiment_id: str,
        partitions: list[ResearchDataUse],
    ) -> None:
        for partition in partitions:
            await self._ensure_partition_can_open(db, experiment_id, partition)

    async def _ensure_partition_can_open(
        self,
        db: AsyncSession,
        experiment_id: str,
        partition: ResearchDataUse,
    ) -> None:
        previous_uses = await self._repository.opened_uses_for_manifest(
            db,
            partition.manifest_sha256,
            excluding_experiment_id=experiment_id,
        )
        for previous_use in previous_uses:
            if previous_use.role in PROTECTED_ROLES or partition.role in PROTECTED_ROLES:
                raise BurnedDataConflict(
                    f"Manifest {partition.manifest_sha256} was already opened as "
                    f"{previous_use.role} and cannot be reused as {partition.role}"
                )

    async def _require_experiment(
        self,
        db: AsyncSession,
        experiment_id: str,
        *,
        for_update: bool = False,
    ) -> ResearchExperiment:
        experiment = await self._repository.get(db, experiment_id, for_update=for_update)
        if experiment is None:
            raise LookupError(f"Research experiment {experiment_id} was not found")
        return experiment

    async def _append_event(
        self,
        db: AsyncSession,
        experiment_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        await self._repository.append_event(
            db,
            ResearchExperimentEvent(
                experiment_id=experiment_id,
                kind=kind,
                payload_json=_canonical_json(payload),
                occurred_at=datetime.now(UTC),
            ),
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: str, field_name: str) -> None:
    if not SHA256_PATTERN.fullmatch(value):
        raise ResearchRegistryError(f"{field_name} must be a lowercase SHA-256")


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


research_registry = ResearchRegistry()
