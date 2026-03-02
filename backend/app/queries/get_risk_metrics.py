"""Query: Get current risk metrics."""

from app.core.logging import get_logger
from app.services.risk.manager import risk_manager

logger = get_logger(__name__)


class GetRiskMetricsQuery:
    async def execute(self) -> dict:
        dashboard = risk_manager.get_dashboard()
        return {
            "daily_pnl": float(dashboard.get("daily_pnl", 0)),
            "max_drawdown": float(dashboard.get("max_drawdown", 0)),
            "current_drawdown": float(dashboard.get("current_drawdown", 0)),
            "var_95": float(dashboard.get("var_95", 0)),
            "sharpe_ratio": float(dashboard.get("sharpe_ratio", 0)),
            "circuit_breaker_state": dashboard.get("circuit_breaker_state", "normal"),
            "open_positions": dashboard.get("open_positions", 0),
            "total_exposure": float(dashboard.get("total_exposure", 0)),
        }


get_risk_metrics_query = GetRiskMetricsQuery()
