"""Schema contract for the simplified per-asset intelligence workflow."""

from typing import Literal

from pydantic import BaseModel, Field


class AssetStudyCreateRequest(BaseModel):
    symbol: str = Field(pattern=r"^[A-Z0-9]{5,20}$")


class AssetUniverseItem(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: Literal["USDT"]
    quote_volume_24h: float
    price_change_pct_24h: float


class AssetUniverseResponse(BaseModel):
    assets: list[AssetUniverseItem]
    generated_at: str
    minimum_quote_volume_24h: float


class AssetMarketStudy(BaseModel):
    trend: Literal["UPTREND", "DOWNTREND", "RANGE"]
    volatility_pct: float
    liquidity_quote_volume_24h: float
    candles: int


class AssetPatternStudy(BaseModel):
    """Bounded, explainable market-pattern assessment for one closed candle series."""

    regime: Literal["UPTREND", "DOWNTREND", "RANGE", "COMPRESSION", "STRESS"]
    pattern: Literal[
        "TREND_CONTINUATION",
        "COMPRESSION_BREAKOUT",
        "MEAN_REVERSION",
        "OBSERVATION_ONLY",
    ]
    confidence: float = Field(ge=0, le=1)
    relative_volume: float = Field(ge=0)
    taker_buy_imbalance: float | None = Field(default=None, ge=-1, le=1)
    flow_data_available: bool
    explanation: str = Field(min_length=1, max_length=500)


class PredictiveModelStudy(BaseModel):
    model_type: Literal["xgboost"] = "xgboost"
    trained: bool
    validation_accuracy: float | None
    samples: int
    latest_signal: Literal["BUY", "HOLD", "SELL", "UNAVAILABLE"]


class StrategyRecommendation(BaseModel):
    strategy_name: str
    backtest_id: int | None
    deployment_id: int | None
    deployment_status: Literal["APPROVED", "REJECTED", "UNAVAILABLE"]
    reasons: list[str]


class AssetStudyResponse(BaseModel):
    symbol: str
    execution_mode: Literal["PAPER", "TESTNET", "LIVE"]
    market_study: AssetMarketStudy
    pattern_study: AssetPatternStudy
    predictive_model: PredictiveModelStudy
    recommendation: StrategyRecommendation


class AssetStudyJobResponse(BaseModel):
    id: int
    symbol: str
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"]
    message: str | None = None
    study: AssetStudyResponse | None = None
    error_message: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class MarketOpportunityCandidate(BaseModel):
    rank: int = Field(ge=1)
    symbol: str
    screening_score: float
    market_trend: Literal["UPTREND", "DOWNTREND", "RANGE"]
    price_change_pct_24h: float
    quote_volume_24h: float
    status: Literal["SHORTLISTED", "STUDYING", "APPROVED", "REJECTED", "UNAVAILABLE", "FAILED"]
    study: AssetStudyResponse | None = None
    error_message: str | None = None


class MarketOpportunityScanResponse(BaseModel):
    id: int
    status: Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"]
    total_assets: int = Field(ge=0)
    screened_assets: int = Field(ge=0)
    shortlisted_assets: int = Field(ge=0)
    studied_assets: int = Field(ge=0)
    failed_assets: int = Field(ge=0)
    message: str | None = None
    candidates: list[MarketOpportunityCandidate]
    started_at: str | None = None
    completed_at: str | None = None
