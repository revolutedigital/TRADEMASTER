"""Risk management API endpoints: VaR, correlation, stress testing."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_position_repository, require_auth
from app.repositories.position_repo import PositionRepository
from app.services.risk.var import var_calculator
from app.services.risk.stress_test import stress_test_engine
from app.services.risk.correlation import correlation_tracker
from app.services.risk.drawdown import circuit_breaker
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/metrics")
async def get_risk_metrics(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
    repo: PositionRepository = Depends(get_position_repository),
):
    """Get comprehensive risk metrics: VaR, CVaR, correlation, circuit breaker."""
    # Get portfolio info
    open_positions = await repo.get_open(db)
    total_exposure = sum(float(p.current_price) * float(p.quantity) for p in open_positions)

    # VaR calculation (using unrealized P&L as proxy for returns)
    returns = [float(p.unrealized_pnl) / (float(p.entry_price) * float(p.quantity))
               for p in open_positions
               if float(p.entry_price) * float(p.quantity) > 0]

    var_metrics = var_calculator.calculate_all(
        returns=returns if len(returns) >= 10 else [0.0] * 10,
        portfolio_value=total_exposure or 10000,
    )

    # Correlation matrix
    correlation = correlation_tracker.get_correlation_matrix()

    # Concentration risk
    concentration = correlation_tracker.check_concentration_risk()

    return {
        "var": var_metrics,
        "correlation_matrix": correlation,
        "concentration_risk": concentration,
        "circuit_breaker": circuit_breaker.get_status(),
        "open_positions": len(open_positions),
        "total_exposure": round(total_exposure, 2),
    }


@router.get("/stress-test")
async def run_stress_tests(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(require_auth),
    repo: PositionRepository = Depends(get_position_repository),
):
    """Run all stress test scenarios against current portfolio."""
    open_positions = await repo.get_open(db)

    positions = [
        {
            "symbol": p.symbol,
            "side": p.side,
            "quantity": float(p.quantity),
            "current_price": float(p.current_price),
        }
        for p in open_positions
    ]

    total_value = sum(float(p.current_price) * float(p.quantity) for p in open_positions)

    results = stress_test_engine.run_all_scenarios(positions, total_value or 10000)

    return {
        "portfolio_value": round(total_value, 2),
        "scenarios": [
            {
                "name": r.scenario_name,
                "impact": r.portfolio_impact,
                "value_after": r.portfolio_value_after,
                "impact_pct": r.details.get("impact_pct", 0),
                "description": r.details.get("description", ""),
                "positions_affected": r.positions_affected,
            }
            for r in results
        ],
    }
