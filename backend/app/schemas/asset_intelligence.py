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
    predictive_model: PredictiveModelStudy
    recommendation: StrategyRecommendation
