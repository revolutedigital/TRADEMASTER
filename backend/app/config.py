import secrets

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Frontend URL for CORS (comma-separated for multiple origins)
    frontend_url: str = "https://trademaster.up.railway.app,http://localhost:3000"

    # Trading
    trading_symbols: str = "BTCUSDT,ETHUSDT"
    trading_max_risk_per_trade: float = 0.02
    trading_max_portfolio_exposure: float = 0.60
    trading_max_single_asset_exposure: float = 0.30
    trading_max_daily_drawdown: float = 0.03
    trading_max_weekly_drawdown: float = 0.07
    trading_max_total_drawdown: float = 0.15

    # Webhook alerts (Slack/Discord/custom — optional)
    risk_alert_webhook_url: str = ""
    trade_alert_webhook_url: str = ""

    @model_validator(mode="after")
    def _validate_secrets(self) -> "Settings":
        if self.app_env == "production":
            if not self.jwt_secret_key or len(self.jwt_secret_key) < 32:
                raise ValueError(
                    "jwt_secret_key must be set (min 32 chars) in production"
                )
            if not self.admin_password or len(self.admin_password) < 8:
                raise ValueError(
                    "admin_password must be set (min 8 chars) in production"
                )
        else:
            if not self.jwt_secret_key:
                self.jwt_secret_key = secrets.token_urlsafe(32)
            if not self.admin_password:
                self.admin_password = "admin"
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
    def is_production(self) -> bool:
        return self.app_env == "production"

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
