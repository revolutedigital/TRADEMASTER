"""Strict API contracts for research-only microstructure experiments."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ExperimentStatus = Literal["DRAFT", "FROZEN", "REJECTED", "INCONCLUSIVE", "APPROVED"]
WalAuditStatus = Literal["VALID", "PARTIAL", "MISSING", "INVALID"]
DatasetPartitionRole = Literal[
    "DEVELOPMENT",
    "TRAINING",
    "SELECTION",
    "AUDIT",
    "PROSPECTIVE_SHADOW",
]


class ProductContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_venue: Literal["binance_usdm_futures"]
    execution_product: Literal["perpetual"]
    symbol: Literal["BTCUSDT"]
    directions: tuple[Literal["LONG", "SHORT"], Literal["LONG", "SHORT"]]
    auxiliary_signal_venue: Literal["binance_spot"]
    horizons_seconds: tuple[int, int, int, int, int]
    position_model: Literal["one_net_position"]


class CostProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maker_fee_bps_per_side: float = Field(ge=0)
    taker_fee_bps_per_side: float = Field(ge=0)
    expected_slippage_bps_per_side: float = Field(ge=0)
    expected_latency_ms: int = Field(ge=0)
    stress_roundtrip_bps: float = Field(ge=20)
    default_order_style: Literal["marketable_taker"]


class DatasetPartition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: DatasetPartitionRole
    start_at: datetime
    end_at: datetime
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("start_at", "end_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("partition timestamps must be timezone-aware")
        return value


class ApprovalGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_oos_folds: Literal[3]
    min_oos_portfolio_trades: Literal[200]
    min_oos_utc_days: Literal[20]
    adjusted_one_sided_confidence: Literal[0.95]
    min_expected_cost_lcb_bps: float = Field(gt=0)
    min_stress_cost_mean_bps: float = Field(gt=0)
    max_probability_backtest_overfitting: Literal[0.2]
    book_evidence_min_complete_days: Literal[60]
    prospective_shadow_min_days: Literal[20]
    prospective_shadow_max_days: Literal[30]
    top_p_tails_pct: tuple[int, int, int, int]


class HypothesisDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1, max_length=40)
    definition: dict[str, Any]


class CreateExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=120)
    code_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    protocol_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    product: ProductContract
    cost_profile: CostProfile
    dataset_partitions: tuple[DatasetPartition, ...] = Field(min_length=4, max_length=5)
    approval_gate: ApprovalGate
    hypotheses: tuple[HypothesisDefinition, ...] = Field(min_length=1, max_length=100)


class SafetyBoundary(BaseModel):
    research_only: Literal[True] = True
    order_submission_allowed: Literal[False] = False
    execution_authorization: Literal["none"] = "none"


class BookEvidenceGateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eligible: bool
    required_complete_days: Literal[60]
    audited_days: int = Field(ge=0)
    complete_days: int = Field(ge=0)
    longest_complete_streak_days: int = Field(ge=0)
    streak_start: date | None
    streak_end: date | None
    incomplete_days: list[date]
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reasons: list[str]
    safety: SafetyBoundary


class EvidenceGateStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_available: bool
    audited_start_date: date | None
    audited_end_date: date | None
    audited_days: int = Field(ge=0)
    latest_daily_status: WalAuditStatus | None
    latest_daily_manifest_sha256: str | None = Field(pattern=r"^[a-f0-9]{64}$")
    book_evidence_gate: BookEvidenceGateResponse
    status_reasons: list[str]
    safety: SafetyBoundary
    generated_at: datetime


class TestnetEligibilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    experiment_status: ExperimentStatus
    eligible: bool
    reasons: list[str]
    book_evidence_contiguous_days: int = Field(ge=0)
    prospective_shadow_days: int = Field(ge=0)
    prospective_shadow_outcome_days: int = Field(ge=0)
    prospective_shadow_signal_count: int = Field(ge=0)
    prospective_shadow_outcome_signal_count: int = Field(ge=0)
    prospective_shadow_expected_mean_bps: float | None
    prospective_shadow_stress_mean_bps: float | None
    prospective_shadow_positive: bool
    unresolved_failures: int = Field(ge=0)
    explicit_testnet_release: bool
    release_request_required: bool
    evidence_artifact_available: bool
    order_submission_allowed: Literal[False]
    execution_authorization: Literal["none"]
    safety: SafetyBoundary
    generated_at: datetime


class RecordTestnetReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_phrase: Literal["REQUEST RESEARCH TESTNET RELEASE"]
    reasons: list[str] = Field(min_length=1, max_length=50)

    @field_validator("reasons")
    @classmethod
    def reject_blank_reasons(cls, value: list[str]) -> list[str]:
        normalized = [reason.strip() for reason in value]
        if any(not reason for reason in normalized):
            raise ValueError("release reasons cannot be blank")
        return normalized


class ResearchTestnetReleaseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    experiment_id: str
    release_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    requested_by: str
    reasons: list[str]
    evidence_snapshot: dict[str, Any]
    released_at: datetime
    explicit_testnet_release: Literal[True]
    release_request_required: Literal[False]
    order_submission_allowed: Literal[False]
    execution_authorization: Literal["none"]
    safety: SafetyBoundary


class OpenedPartitionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    experiment_id: str
    role: DatasetPartitionRole
    start_at: datetime
    end_at: datetime
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    opened_at: datetime
    safety: SafetyBoundary


class RecordShadowSignalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_time: datetime
    side: Literal["BUY", "SELL"]
    horizon_seconds: Literal[120, 300]
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_vector: dict[str, float] = Field(min_length=1)

    @field_validator("decision_time")
    @classmethod
    def require_decision_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        return value


class RecordShadowOutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_net_bps: float
    stress_net_bps: float
    label_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class RecordExperimentDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["REJECTED", "INCONCLUSIVE", "APPROVED"]
    reasons: list[str] = Field(min_length=1, max_length=50)

    @field_validator("reasons")
    @classmethod
    def reject_blank_reasons(cls, value: list[str]) -> list[str]:
        normalized = [reason.strip() for reason in value]
        if any(not reason for reason in normalized):
            raise ValueError("decision reasons cannot be blank")
        return normalized


class ShadowSignalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    experiment_id: str
    decision_time: datetime
    recorded_at: datetime
    side: Literal["BUY", "SELL"]
    horizon_seconds: Literal[120, 300]
    probability: float
    threshold: float
    would_enter: bool
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_vector_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    outcome_recorded: bool
    expected_net_bps: float | None
    stress_net_bps: float | None
    label_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    safety: SafetyBoundary


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    status: ExperimentStatus
    code_revision: str
    protocol_sha256: str
    experiment_sha256: str | None
    product: dict[str, Any]
    cost_profile: dict[str, Any]
    dataset_partitions: list[dict[str, Any]]
    approval_gate: dict[str, Any]
    safety: SafetyBoundary = Field(default_factory=SafetyBoundary)
    created_at: datetime
    frozen_at: datetime | None


class ExperimentReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    status: Literal["FROZEN", "REJECTED", "INCONCLUSIVE", "APPROVED"]
    decision_reasons: list[str]
    metrics: dict[str, Any]
    artifact_sha256: str | None = None
    safety: SafetyBoundary = Field(default_factory=SafetyBoundary)
    generated_at: datetime
