"""Trading Pydantic schemas."""

from datetime import datetime
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OrderResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    exchange_order_id: str | None
    symbol: str
    side: str
    order_type: str
    status: str
    execution_mode: str
    quantity: float
    price: float | None
    filled_quantity: float
    avg_fill_price: float | None
    commission: float
    protective_order_list_id: int | None
    protective_quantity: float | None
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
    execution_mode: str
    protective_order_list_id: int | None
    protective_quantity: float | None
    protection_status: str
    opened_at: datetime
    closed_at: datetime | None


class PortfolioSummary(BaseModel):
    execution_mode: Literal["PAPER", "TESTNET", "LIVE"]
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


class SignalVoteResponse(BaseModel):
    """One normalized vote that contributed to a persisted signal."""

    model_config = ConfigDict(extra="forbid")

    model: str
    action: Literal["BUY", "HOLD", "SELL"]
    score: float
    confidence: float


class SignalRegimeResponse(BaseModel):
    """Market regime captured when the signal became actionable."""

    model_config = ConfigDict(extra="forbid")

    market: str
    volatility: str
    confidence: float
    position_size_multiplier: float


class SignalEvidenceResponse(BaseModel):
    """Explainable, non-secret evidence for an actionable strategy candidate."""

    model_config = ConfigDict(extra="forbid")

    signal_source: str
    strategy_deployment_id: int | None = None
    signal_threshold: float
    agreement_ratio: float
    votes: list[SignalVoteResponse]
    regime: SignalRegimeResponse
    price: float
    atr: float
    atr_pct: float


class SignalHistoryItemResponse(BaseModel):
    """One persisted strategy candidate for the trader-facing history."""

    id: int
    symbol: str
    action: Literal["BUY", "HOLD", "SELL"]
    strength: float
    confidence: float
    model_source: str
    timeframe: str
    was_executed: bool
    evidence: SignalEvidenceResponse | None = None
    generated_at: datetime


TechnicalIndicator = Literal[
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger",
    "engulfing",
    "breakout",
]


class TechnicalStrategyConfig(BaseModel):
    """Deterministic, research-only technical strategy configuration."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["technical_ensemble"]
    indicators: list[TechnicalIndicator] = Field(min_length=1, max_length=7)
    indicator_params: dict[str, dict[str, float]] = Field(default_factory=dict)
    min_confirmations: int = Field(default=1, ge=1, le=7)
    execution_profile: Literal["spot_long_only"] = "spot_long_only"

    @model_validator(mode="after")
    def validate_parameters(self) -> "TechnicalStrategyConfig":
        if len(set(self.indicators)) != len(self.indicators):
            raise ValueError("indicators must be unique")
        if self.min_confirmations > len(self.indicators):
            raise ValueError("min_confirmations cannot exceed the selected indicators")

        allowed_parameters = {
            "sma": {"sma_short", "sma_long"},
            "ema": {"ema_short", "ema_long"},
            "rsi": {"rsi_period", "rsi_overbought", "rsi_oversold"},
            "macd": {"macd_fast", "macd_slow", "macd_signal"},
            "bollinger": {"bb_period", "bb_std"},
            "engulfing": set(),
            "breakout": {"breakout_lookback"},
        }
        for indicator, parameters in self.indicator_params.items():
            if indicator not in self.indicators:
                raise ValueError(f"parameters provided for unselected indicator: {indicator}")
            unknown = set(parameters) - allowed_parameters[indicator]
            if unknown:
                raise ValueError(f"unsupported {indicator} parameters: {sorted(unknown)}")
            if any(not isfinite(value) for value in parameters.values()):
                raise ValueError("indicator parameters must be finite numbers")

        sma = self.indicator_params.get("sma", {})
        if not 2 <= sma.get("sma_short", 10) <= 100:
            raise ValueError("sma_short must be between 2 and 100")
        if not 5 <= sma.get("sma_long", 30) <= 200:
            raise ValueError("sma_long must be between 5 and 200")
        if sma and sma.get("sma_short", 10) >= sma.get("sma_long", 30):
            raise ValueError("sma_short must be lower than sma_long")
        ema = self.indicator_params.get("ema", {})
        if not 2 <= ema.get("ema_short", 12) <= 100:
            raise ValueError("ema_short must be between 2 and 100")
        if not 5 <= ema.get("ema_long", 26) <= 200:
            raise ValueError("ema_long must be between 5 and 200")
        if ema and ema.get("ema_short", 12) >= ema.get("ema_long", 26):
            raise ValueError("ema_short must be lower than ema_long")
        rsi = self.indicator_params.get("rsi", {})
        if not 2 <= rsi.get("rsi_period", 14) <= 50:
            raise ValueError("rsi_period must be between 2 and 50")
        if rsi and not 0 < rsi.get("rsi_oversold", 30) < rsi.get("rsi_overbought", 70) < 100:
            raise ValueError("rsi levels must satisfy 0 < oversold < overbought < 100")
        macd = self.indicator_params.get("macd", {})
        if not 2 <= macd.get("macd_fast", 12) <= 50:
            raise ValueError("macd_fast must be between 2 and 50")
        if not 5 <= macd.get("macd_slow", 26) <= 100:
            raise ValueError("macd_slow must be between 5 and 100")
        if not 2 <= macd.get("macd_signal", 9) <= 50:
            raise ValueError("macd_signal must be between 2 and 50")
        if macd and macd.get("macd_fast", 12) >= macd.get("macd_slow", 26):
            raise ValueError("macd_fast must be lower than macd_slow")
        bollinger = self.indicator_params.get("bollinger", {})
        if not 5 <= bollinger.get("bb_period", 20) <= 100:
            raise ValueError("bb_period must be between 5 and 100")
        if not 0.5 <= bollinger.get("bb_std", 2) <= 4:
            raise ValueError("bb_std must be between 0.5 and 4")
        breakout = self.indicator_params.get("breakout", {})
        if not 5 <= breakout.get("breakout_lookback", 20) <= 200:
            raise ValueError("breakout_lookback must be between 5 and 200")
        return self


class StrategySummary(BaseModel):
    """Exact research strategy recorded with a backtest response."""

    name: str
    execution_profile: Literal["spot_long_only", "model_long_short"]
    research_only: bool = True
    indicators: list[str] = Field(default_factory=list)
    min_confirmations: int | None = None


class BacktestRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", pattern=r"^[A-Z]{3,10}USDT$")
    interval: str = Field(default="1h", pattern=r"^(1m|5m|15m|30m|1h|4h|1d|1w)$")
    initial_capital: float = Field(default=10000.0, ge=100, le=1_000_000)
    signal_threshold: float = Field(default=0.3, ge=0.1, le=0.9)
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    risk_reward_ratio: float = Field(default=2.0, ge=0.5, le=10.0)
    strategy: TechnicalStrategyConfig | None = None


class BacktestResponse(BaseModel):
    id: int | None = None
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy: float
    equity_curve: list[float]
    strategy: StrategySummary


class StrategyDeploymentCreateRequest(BaseModel):
    """Promote one persisted technical backtest for runtime validation."""

    source_backtest_id: int = Field(ge=1)
    target_execution_mode: Literal["PAPER", "TESTNET", "LIVE"]


class StrategyDeploymentActivationRequest(BaseModel):
    """Explicit operator confirmation before enabling an approved strategy."""

    confirmation_phrase: str = Field(min_length=1, max_length=100)
    totp_code: str | None = Field(default=None, pattern=r"^\d{6}$")

    @model_validator(mode="after")
    def validate_confirmation_phrase(self) -> "StrategyDeploymentActivationRequest":
        if self.confirmation_phrase != "ACTIVATE STRATEGY":
            raise ValueError("confirmation_phrase must be 'ACTIVATE STRATEGY'")
        return self


class StrategyDeploymentResponse(BaseModel):
    """Persisted validation evidence and runtime state for one strategy."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    source_backtest_id: int
    symbol: str
    interval: str
    target_execution_mode: Literal["PAPER", "TESTNET", "LIVE"]
    status: Literal["APPROVED", "ACTIVE", "REJECTED", "DISABLED"]
    total_test_trades: int
    walk_forward_windows: int
    avg_return_pct: float
    avg_sharpe: float
    avg_max_drawdown_pct: float
    avg_profit_factor: float
    consistency_score: float
    overfitting_score: float
    rejection_reason: str | None
    activated_at: datetime | None
    created_at: datetime
