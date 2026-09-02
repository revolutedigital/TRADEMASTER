import secrets
from enum import StrEnum

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingExecutionMode(StrEnum):
    """Effective destination for an order after resolving the environment flags."""

    PAPER = "PAPER"
    TESTNET = "TESTNET"
    LIVE = "LIVE"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "development"
    app_debug: bool = False
    app_log_level: str = "INFO"

    # Binance
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True
    binance_testnet_api_key: str = ""
    binance_testnet_api_secret: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://trademaster:trademaster@localhost:5432/trademaster"
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT
    jwt_secret_key: str = Field(default="")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # Admin credentials (MUST be set via environment variables in production)
    admin_username: str = Field(default="admin")
    admin_password: str = Field(default="")

    # TOTP two-factor authentication (optional)
    # Set totp_secret via environment to enable 2FA for the admin account.
    totp_enabled: bool = False
    totp_secret: str = ""  # Base32-encoded shared secret

    # Paper trading mode (no real Binance orders)
    paper_mode: bool = True

    # Live trading safety control plane. These values deliberately keep the
    # application locked even when PAPER_MODE=false and BINANCE_TESTNET=false.
    live_trading_enabled: bool = False
    live_trading_arm_code: str = Field(default="", repr=False)
    live_trading_arm_ttl_minutes: int = Field(default=15, ge=1, le=60)
    live_trading_max_notional_per_order: float = Field(default=100.0, gt=0)
    live_trading_max_daily_notional: float = Field(default=300.0, gt=0)
    live_trading_reconciliation_max_age_seconds: int = Field(default=45, ge=5, le=300)
    live_trading_testnet_verification_max_age_days: int = Field(default=30, ge=1, le=90)
    live_trading_allowed_account_assets: str = "USDT,BNB"

    # Frontend URL for CORS (comma-separated for multiple origins)
    frontend_url: str = "https://trademaster.up.railway.app,http://localhost:3000"

    # Trading
    trading_symbols: str = "BTCUSDT,ETHUSDT"
    trading_max_risk_per_trade: float = Field(default=0.02, gt=0, le=0.10)
    trading_max_portfolio_exposure: float = Field(default=0.60, gt=0, le=1.0)
    trading_max_single_asset_exposure: float = Field(default=0.30, gt=0, le=1.0)
    trading_max_daily_drawdown: float = Field(default=0.03, ge=0.01, le=0.20)
    trading_max_weekly_drawdown: float = Field(default=0.07, ge=0.02, le=0.30)
    trading_max_monthly_drawdown: float = Field(default=0.10, ge=0.03, le=0.50)
    trading_max_total_drawdown: float = Field(default=0.15, ge=0.05, le=0.50)
    trading_kelly_fraction: float = Field(default=0.15, gt=0, le=0.50)

    # Testnet starts as a deliberately small canary. These are hard upper
    # bounds in code, not dashboard defaults: the first validated strategies
    # can never fan out across the full catalog or inherit LIVE-size risk.
    testnet_canary_max_active_strategies: int = Field(default=3, ge=1, le=3)
    testnet_canary_max_risk_per_trade: float = Field(default=0.0025, gt=0, le=0.0025)
    testnet_canary_max_portfolio_exposure: float = Field(default=0.20, gt=0, le=0.20)
    testnet_canary_max_single_asset_exposure: float = Field(default=0.05, gt=0, le=0.05)

    # Webhook alerts (Slack/Discord/custom — optional)
    risk_alert_webhook_url: str = ""
    trade_alert_webhook_url: str = ""

    @model_validator(mode="after")
    def _validate_secrets(self) -> "Settings":
        if self.app_env == "production":
            if not self.jwt_secret_key or len(self.jwt_secret_key) < 32:
                raise ValueError("jwt_secret_key must be set (min 32 chars) in production")
            if not self.admin_password or len(self.admin_password) < 8:
                raise ValueError("admin_password must be set (min 8 chars) in production")
        else:
            if not self.jwt_secret_key:
                self.jwt_secret_key = secrets.token_urlsafe(32)
            if not self.admin_password:
                self.admin_password = "admin"

        if self.live_trading_enabled:
            if self.paper_mode or self.binance_testnet:
                raise ValueError(
                    "live_trading_enabled requires PAPER_MODE=false and BINANCE_TESTNET=false"
                )
            if not self.is_production:
                raise ValueError("live_trading_enabled requires APP_ENV=production")
            if not self.totp_enabled or not self.totp_secret:
                raise ValueError("live_trading_enabled requires TOTP_ENABLED with TOTP_SECRET")
            if len(self.live_trading_arm_code) < 20:
                raise ValueError("live_trading_arm_code must be at least 20 characters")
        return self

    @property
    def cors_origins(self) -> list[str]:
        """Parse frontend_url into a list of origins."""
        return [u.strip() for u in self.frontend_url.split(",") if u.strip()]

    @property
    def symbols_list(self) -> list[str]:
        """Parse trading_symbols string into a list."""
        return [s.strip() for s in self.trading_symbols.split(",") if s.strip()]

    @property
    def live_trading_allowed_assets_list(self) -> list[str]:
        """Assets allowed outside tracked positions, such as the quote and fee token."""
        return [
            asset.strip().upper()
            for asset in self.live_trading_allowed_account_assets.split(",")
            if asset.strip()
        ]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def execution_mode(self) -> TradingExecutionMode:
        """Resolve paper, testnet, and real-capital execution unambiguously."""
        if self.paper_mode:
            return TradingExecutionMode.PAPER
        if self.binance_testnet:
            return TradingExecutionMode.TESTNET
        return TradingExecutionMode.LIVE

    @property
    def active_api_key(self) -> str:
        if self.binance_testnet:
            return self.binance_testnet_api_key
        return self.binance_api_key

    @property
    def active_api_secret(self) -> str:
        if self.binance_testnet:
            return self.binance_testnet_api_secret
        return self.binance_api_secret


settings = Settings()
