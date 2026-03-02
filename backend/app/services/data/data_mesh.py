"""Data mesh: domain-oriented data ownership and self-serve platform.

Implements data mesh principles:
- Domain ownership (trading, risk, ML, market)
- Data as a product
- Self-serve data infrastructure
- Federated governance
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from app.core.logging import get_logger

logger = get_logger(__name__)


class DataDomain(str, Enum):
    TRADING = "trading"
    RISK = "risk"
    ML = "ml"
    MARKET = "market"
    PORTFOLIO = "portfolio"


@dataclass
class DataProduct:
    id: str
    name: str
    domain: DataDomain
    description: str
    owner: str
    schema_version: str
    tables: list[str]
    quality_sla: dict = field(default_factory=dict)
    access_patterns: list[str] = field(default_factory=list)


class DataMesh:
    """Data mesh implementation with domain ownership."""

    DATA_PRODUCTS = [
        DataProduct(
            id="dp-market-ohlcv",
            name="Market OHLCV Data",
            domain=DataDomain.MARKET,
            description="Historical candlestick data for all tracked symbols",
            owner="market-team",
            schema_version="1.0",
            tables=["ohlcv"],
            quality_sla={"freshness": "< 5 minutes", "completeness": "> 99%", "accuracy": "> 99.9%"},
            access_patterns=["real-time streaming", "batch historical", "time-range query"],
        ),
        DataProduct(
            id="dp-trading-orders",
            name="Trading Orders & Executions",
            domain=DataDomain.TRADING,
            description="Order lifecycle data from placement to fill/cancel",
            owner="trading-team",
            schema_version="1.0",
            tables=["trades", "positions"],
            quality_sla={"freshness": "< 1 second", "completeness": "100%"},
            access_patterns=["event-driven", "historical query"],
        ),
        DataProduct(
            id="dp-risk-metrics",
            name="Risk Metrics & Calculations",
            domain=DataDomain.RISK,
            description="VaR, drawdown, exposure, and circuit breaker status",
            owner="risk-team",
            schema_version="1.0",
            tables=["risk_snapshots"],
            quality_sla={"freshness": "< 30 seconds", "accuracy": "> 99.99%"},
            access_patterns=["real-time dashboard", "alerting"],
        ),
        DataProduct(
            id="dp-ml-signals",
            name="ML Trading Signals",
            domain=DataDomain.ML,
            description="Model predictions, confidence scores, feature importance",
            owner="ml-team",
            schema_version="1.0",
            tables=["signals", "model_metadata"],
            quality_sla={"freshness": "< 1 minute", "model_accuracy": "> 55%"},
            access_patterns=["streaming predictions", "model monitoring"],
        ),
        DataProduct(
            id="dp-portfolio-snapshots",
            name="Portfolio Snapshots",
            domain=DataDomain.PORTFOLIO,
            description="Point-in-time portfolio state with equity and positions",
            owner="portfolio-team",
            schema_version="1.0",
            tables=["portfolio_snapshots"],
            quality_sla={"freshness": "< 1 minute", "completeness": "100%"},
            access_patterns=["time-series query", "comparison"],
        ),
    ]

    def get_data_products(self, domain: DataDomain | None = None) -> list[DataProduct]:
        """List all data products, optionally filtered by domain."""
        products = self.DATA_PRODUCTS
        if domain:
            products = [p for p in products if p.domain == domain]
        return products

    def get_data_product(self, product_id: str) -> DataProduct | None:
        """Get a specific data product by ID."""
        return next((p for p in self.DATA_PRODUCTS if p.id == product_id), None)

    def get_domain_owners(self) -> dict[str, str]:
        """Get domain ownership mapping."""
        owners = {}
        for product in self.DATA_PRODUCTS:
            if product.domain.value not in owners:
                owners[product.domain.value] = product.owner
        return owners

    def get_catalog(self) -> dict:
        """Get the full data catalog."""
        return {
            "total_products": len(self.DATA_PRODUCTS),
            "domains": list(set(p.domain.value for p in self.DATA_PRODUCTS)),
            "products": [
                {
                    "id": p.id,
                    "name": p.name,
                    "domain": p.domain.value,
                    "owner": p.owner,
                    "tables": p.tables,
                    "schema_version": p.schema_version,
                    "quality_sla": p.quality_sla,
                }
                for p in self.DATA_PRODUCTS
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


data_mesh = DataMesh()
