"""Settings API endpoints for runtime configuration."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import require_auth
from app.config import settings

router = APIRouter()


class TradingConfig(BaseModel):
    trading_mode: str = "testnet"
    symbols: list[str] = ["BTCUSDT", "ETHUSDT"]
    max_risk_per_trade: float = 0.02
    max_total_exposure: float = 0.60


class RiskConfig(BaseModel):
    max_daily_drawdown: float = Field(default=0.03, ge=0.01, le=0.20)
    max_weekly_drawdown: float = Field(default=0.07, ge=0.02, le=0.30)
    max_monthly_drawdown: float = Field(default=0.10, ge=0.03, le=0.50)
    max_total_drawdown: float = Field(default=0.15, ge=0.05, le=0.50)
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    trailing_stop_activation: float = Field(default=0.015, ge=0.005, le=0.10)
    kelly_fraction: float = Field(default=0.15, ge=0.05, le=0.50)
    max_single_asset: float = Field(default=0.30, ge=0.10, le=1.0)


class FullSettings(BaseModel):
    trading: TradingConfig
    risk: RiskConfig
    api_docs_url: str = "/api/docs"


# In-memory runtime overrides (persisted per-process)
_runtime_risk: RiskConfig = RiskConfig()


@router.get("/", response_model=FullSettings)
async def get_settings(_user: dict = Depends(require_auth)):
    """Get current settings."""
    return FullSettings(
        trading=TradingConfig(
            trading_mode="testnet" if settings.binance_testnet else "live",
            symbols=settings.symbols,
        ),
        risk=_runtime_risk,
    )


@router.put("/risk", response_model=RiskConfig)
async def update_risk_settings(
    config: RiskConfig,
    _user: dict = Depends(require_auth),
):
    """Update risk management parameters at runtime."""
    global _runtime_risk
    _runtime_risk = config
    return _runtime_risk
