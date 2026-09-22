"""Strict API contracts for research-only microstructure experiments."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ExperimentStatus = Literal["DRAFT", "FROZEN", "REJECTED", "INCONCLUSIVE", "APPROVED"]


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

    role: Literal["DEVELOPMENT", "TRAINING", "SELECTION", "AUDIT", "PROSPECTIVE_SHADOW"]
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
