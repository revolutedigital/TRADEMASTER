"""Automated risk reporting: daily summaries and alerts."""

from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskReport:
    report_type: str  # "daily", "weekly"
    generated_at: str
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    max_drawdown_pct: float
    current_exposure_pct: float
    var_95: float
    open_positions: int
    circuit_breaker_state: str
    alerts: list[str]


class RiskReporter:
    """Generate risk reports and alerts."""

    def __init__(self):
        self._alert_thresholds = {
            "var_warning": 0.05,  # 5% VaR warning
            "exposure_warning": 0.50,  # 50% exposure warning
            "drawdown_warning": 0.05,  # 5% drawdown warning
        }

    async def generate_daily_report(
        self,
        portfolio_value: float,
        daily_pnl: float,
        max_drawdown_pct: float,
        exposure_pct: float,
        var_95: float,
        open_positions: int,
        cb_state: str,
    ) -> RiskReport:
        """Generate a daily risk summary report."""
        alerts = []

        daily_pnl_pct = daily_pnl / portfolio_value if portfolio_value > 0 else 0

        if var_95 > self._alert_thresholds["var_warning"]:
            alerts.append(f"VaR(95%) at {var_95:.2%} exceeds {self._alert_thresholds['var_warning']:.0%} threshold")

        if exposure_pct > self._alert_thresholds["exposure_warning"]:
            alerts.append(f"Portfolio exposure at {exposure_pct:.1%} exceeds {self._alert_thresholds['exposure_warning']:.0%}")

        if max_drawdown_pct > self._alert_thresholds["drawdown_warning"]:
            alerts.append(f"Drawdown at {max_drawdown_pct:.2%} exceeds {self._alert_thresholds['drawdown_warning']:.0%}")

        if cb_state != "NORMAL":
            alerts.append(f"Circuit breaker in {cb_state} state")

        report = RiskReport(
            report_type="daily",
            generated_at=datetime.now(timezone.utc).isoformat(),
            portfolio_value=round(portfolio_value, 2),
            daily_pnl=round(daily_pnl, 2),
            daily_pnl_pct=round(daily_pnl_pct, 4),
            max_drawdown_pct=round(max_drawdown_pct, 4),
            current_exposure_pct=round(exposure_pct, 4),
            var_95=round(var_95, 4),
            open_positions=open_positions,
            circuit_breaker_state=cb_state,
            alerts=alerts,
        )

        logger.info(
            "risk_report_generated",
            type=report.report_type,
            alerts_count=len(alerts),
            portfolio_value=report.portfolio_value,
        )
        return report


risk_reporter = RiskReporter()
