from pydantic import Field
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
    jwt_secret_key: str = "change-this-to-a-random-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 480  # 8 hours for trading sessions

    # Admin credentials (for dashboard login)
    admin_username: str = "admin"
    admin_password: str = "trademaster2024"  # Override via env var in production

    # Paper trading mode (no real Binance orders)
    paper_mode: bool = True

    # Frontend URL for CORS (comma-separated for multiple origins)
    frontend_url: str = "https://frontend-production-15e9.up.railway.app,http://localhost:3000"

    # Trading
    trading_symbols: str = "BTCUSDT,ETHUSDT"
    trading_max_risk_per_trade: float = 0.02
    trading_max_portfolio_exposure: float = 0.60
    trading_max_single_asset_exposure: float = 0.30
    trading_max_daily_drawdown: float = 0.03
    trading_max_weekly_drawdown: float = 0.07
    trading_max_total_drawdown: float = 0.15

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
