"""Trading Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class OrderResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    exchange_order_id: str | None
    symbol: str
    side: str
    order_type: str
    status: str
    quantity: float
    price: float | None
    filled_quantity: float
    avg_fill_price: float | None
    commission: float
    created_at: datetime


class PositionResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    symbol: str
    side: str
    entry_price: float
    quantity: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss_price: float | None
    take_profit_price: float | None
    is_open: bool
    opened_at: datetime
    closed_at: datetime | None


class PortfolioSummary(BaseModel):
    total_equity: float
    available_balance: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_exposure: float
    exposure_pct: float = 0.0
    open_positions: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0


class SignalResponse(BaseModel):
    symbol: str
    action: str
    strength: float
    confidence: float
    model: str
    timestamp: str


class BacktestRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", pattern=r"^[A-Z]{3,10}USDT$")
    interval: str = Field(default="1h", pattern=r"^(1m|5m|15m|30m|1h|4h|1d|1w)$")
    initial_capital: float = Field(default=10000.0, ge=100, le=1_000_000)
    signal_threshold: float = Field(default=0.3, ge=0.1, le=0.9)
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    risk_reward_ratio: float = Field(default=2.0, ge=0.5, le=10.0)


class BacktestResponse(BaseModel):
    total_trades: int
    win_rate: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy: float
    equity_curve: list[float]
