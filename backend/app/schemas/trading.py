"""Trading Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel


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
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    initial_capital: float = 10000.0
    signal_threshold: float = 0.3
    atr_stop_multiplier: float = 2.0
    risk_reward_ratio: float = 2.0


class BacktestResponse(BaseModel):
    total_trades: int
    win_rate: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy: float
    equity_curve: list[float]
