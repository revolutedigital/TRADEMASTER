"""Command: Run a backtest simulation."""

from dataclasses import dataclass

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunBacktestCommand:
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    parameters: dict | None = None


class RunBacktestHandler:
    async def handle(self, cmd: RunBacktestCommand) -> dict:
        logger.info("cmd_run_backtest", symbol=cmd.symbol, strategy=cmd.strategy)
        from app.services.backtest.engine import backtest_engine
        result = await backtest_engine.run(
            symbol=cmd.symbol,
            strategy=cmd.strategy,
            start_date=cmd.start_date,
            end_date=cmd.end_date,
            initial_capital=cmd.initial_capital,
            parameters=cmd.parameters or {},
        )
        return result


run_backtest_handler = RunBacktestHandler()
