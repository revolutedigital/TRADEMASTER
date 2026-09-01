"""Settings API endpoints for runtime configuration."""

from fastapi import APIRouter, Depends, HTTPException
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
    max_risk_per_trade: float = Field(default=0.02, gt=0, le=0.10)
    max_total_exposure: float = Field(default=0.60, gt=0, le=1.0)
    max_single_asset: float = Field(default=0.30, gt=0, le=1.0)
    max_daily_drawdown: float = Field(default=0.03, ge=0.01, le=0.20)
    max_weekly_drawdown: float = Field(default=0.07, ge=0.02, le=0.30)
    max_monthly_drawdown: float = Field(default=0.10, ge=0.03, le=0.50)
    max_total_drawdown: float = Field(default=0.15, ge=0.05, le=0.50)
    kelly_fraction: float = Field(default=0.15, ge=0.05, le=0.50)


class FullSettings(BaseModel):
    trading: TradingConfig
    risk: RiskConfig
    api_docs_url: str = "/api/docs"


def _effective_risk_config() -> RiskConfig:
    """Return the exact limits enforced by the deployed engine process."""
    return RiskConfig(
        max_risk_per_trade=settings.trading_max_risk_per_trade,
        max_total_exposure=settings.trading_max_portfolio_exposure,
        max_single_asset=settings.trading_max_single_asset_exposure,
        max_daily_drawdown=settings.trading_max_daily_drawdown,
        max_weekly_drawdown=settings.trading_max_weekly_drawdown,
        max_monthly_drawdown=settings.trading_max_monthly_drawdown,
        max_total_drawdown=settings.trading_max_total_drawdown,
        kelly_fraction=settings.trading_kelly_fraction,
    )


@router.get("/", response_model=FullSettings)
async def get_settings(_user: dict = Depends(require_auth)):
    """Get current settings."""
    return FullSettings(
        trading=TradingConfig(
            trading_mode=settings.execution_mode.value.lower(),
            symbols=settings.symbols_list,
            max_risk_per_trade=settings.trading_max_risk_per_trade,
            max_total_exposure=settings.trading_max_portfolio_exposure,
        ),
        risk=_effective_risk_config(),
    )


@router.put("/risk", response_model=RiskConfig)
async def update_risk_settings(
    config: RiskConfig,
    _user: dict = Depends(require_auth),
):
    """Reject unsafe per-process risk overrides that the engine cannot share durably."""
    del config
    raise HTTPException(
        status_code=409,
        detail=(
            "Risk limits are immutable at runtime. Update the deployed TRADING_* "
            "environment variables and roll out a new release before arming LIVE trading."
        ),
    )
