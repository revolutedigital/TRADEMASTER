"""Risk management API endpoints: VaR, correlation, stress testing, Monte Carlo, Kelly."""

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_position_repository, require_auth
from app.models.base import async_session_factory
from app.repositories.position_repo import PositionRepository
from app.services.risk.var import var_calculator
from app.services.risk.stress_test import stress_test_engine
from app.services.risk.correlation import correlation_tracker
from app.services.risk.drawdown import circuit_breaker
from app.services.risk.monte_carlo import monte_carlo
from app.services.risk.kelly import kelly_calculator
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


@router.get("/monte-carlo")
async def monte_carlo_simulation(
    horizon_days: int = 30,
    n_simulations: int = 10000,
    _user: dict = Depends(require_auth),
):
    """Run Monte Carlo simulation for portfolio risk projection."""
    async with async_session_factory() as db:
        result = await db.execute(text(
            "SELECT realized_pnl FROM positions WHERE status='CLOSED' ORDER BY closed_at DESC LIMIT 500"
        ))
        pnls = [float(r[0]) for r in result.fetchall() if r[0]]

    equity = 10000.0  # Default
    try:
        from app.services.exchange.binance_client import binance_client
        equity = float(await binance_client.get_balance("USDT"))
    except Exception:
        pass

    if not pnls:
        # Use synthetic returns if no trades yet
        pnls = [equity * r for r in [0.01, -0.005, 0.008, -0.003, 0.012, -0.007, 0.005]]

    returns = [p / equity for p in pnls]
    mc_result = monte_carlo.simulate(equity, returns, n_simulations, horizon_days)
    return {
        "portfolio_value": equity,
        "horizon_days": horizon_days,
        "simulations": n_simulations,
        "median_outcome": round(mc_result.median_outcome, 2),
        "worst_5pct": round(mc_result.worst_5pct, 2),
        "best_5pct": round(mc_result.best_5pct, 2),
        "probability_of_loss": round(mc_result.probability_of_loss, 4),
        "expected_value": round(mc_result.expected_value, 2),
        "var_95": round(mc_result.var_95, 2),
        "cvar_95": round(mc_result.cvar_95, 2),
    }


@router.get("/kelly")
async def kelly_sizing(_user: dict = Depends(require_auth)):
    """Calculate optimal position size using the Kelly Criterion."""
    async with async_session_factory() as db:
        result = await db.execute(text(
            "SELECT realized_pnl FROM positions WHERE status='CLOSED' ORDER BY closed_at DESC LIMIT 200"
        ))
        pnls = [float(r[0]) for r in result.fetchall() if r[0]]

    if not pnls or len(pnls) < 5:
        return {"error": "insufficient_trades", "min_required": 5}

    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls)
    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = float(np.mean(losses)) if losses else 0

    equity = 10000.0
    try:
        from app.services.exchange.binance_client import binance_client
        equity = float(await binance_client.get_balance("USDT"))
    except Exception:
        pass

    kelly_result = kelly_calculator.calculate(win_rate, avg_win, avg_loss, equity)
    return {
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "full_kelly": round(kelly_result.full_kelly, 4),
        "half_kelly": round(kelly_result.half_kelly, 4),
        "recommended_fraction": round(kelly_result.recommended_fraction, 4),
        "optimal_size_usd": round(kelly_result.optimal_size_usd, 2),
        "edge_per_trade": round(kelly_result.edge, 4),
    }
