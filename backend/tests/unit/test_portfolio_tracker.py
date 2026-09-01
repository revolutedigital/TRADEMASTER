"""Tests for portfolio tracker service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.portfolio.tracker import PortfolioTracker


class TestPortfolioTracker:
    def test_instantiation(self):
        tracker = PortfolioTracker()
        assert tracker is not None

    @pytest.mark.asyncio
    async def test_open_position_creates_position(self):
        tracker = PortfolioTracker()
        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()

        with patch("app.services.portfolio.tracker.event_bus") as mock_bus:
            mock_bus.publish = AsyncMock()
            position = await tracker.open_position(
                db=mock_db,
                symbol="BTCUSDT",
                side="LONG",
                entry_price=85000.0,
                quantity=0.001,
                stop_loss_price=83000.0,
                take_profit_price=89000.0,
            )

        assert position.symbol == "BTCUSDT"
        assert position.side == "LONG"
        assert float(position.entry_price) == 85000.0
        mock_db.add.assert_called_once()
        mock_db.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_position_long_profit(self):
        tracker = PortfolioTracker()
        mock_db = AsyncMock()
        mock_db.flush = AsyncMock()

        mock_position = MagicMock()
        mock_position.side = "LONG"
        mock_position.entry_price = 85000
        mock_position.quantity = 0.1
        mock_position.realized_pnl = 0
        mock_position.id = 1
        mock_position.symbol = "BTCUSDT"

        with patch("app.services.portfolio.tracker.event_bus") as mock_bus:
            mock_bus.publish = AsyncMock()
            result = await tracker.close_position(mock_db, mock_position, 86000.0)

        # P&L for LONG: (86000 - 85000) * 0.1 = 100
        assert result.realized_pnl == 100.0
        assert result.is_open is False

    @pytest.mark.asyncio
    async def test_close_position_short_profit(self):
        tracker = PortfolioTracker()
        mock_db = AsyncMock()
        mock_db.flush = AsyncMock()

        mock_position = MagicMock()
        mock_position.side = "SHORT"
        mock_position.entry_price = 85000
        mock_position.quantity = 0.1
        mock_position.realized_pnl = 0
        mock_position.id = 1
        mock_position.symbol = "BTCUSDT"

        with patch("app.services.portfolio.tracker.event_bus") as mock_bus:
            mock_bus.publish = AsyncMock()
            result = await tracker.close_position(mock_db, mock_position, 84000.0)

        # P&L for SHORT: (85000 - 84000) * 0.1 = 100
        assert result.realized_pnl == 100.0

    @pytest.mark.asyncio
    async def test_exchange_ledgers_never_receive_a_local_virtual_exit(self):
        tracker = PortfolioTracker()
        position = MagicMock()
        position.execution_mode = "TESTNET"
        position.protection_status = "MISSING"
        position.symbol = "BTCUSDT"
        position.is_open = True
        database = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [position]
        database.execute = AsyncMock(return_value=result)
        database.flush = AsyncMock()

        closed = await tracker.check_stop_losses(
            database,
            {"BTCUSDT": 1},
            execution_mode="TESTNET",
        )

        assert closed == []
        database.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_paper_exit_candidate_does_not_close_the_ledger_before_an_order_fill(self):
        tracker = PortfolioTracker()
        position = MagicMock()
        position.id = 7
        position.symbol = "BTCUSDT"
        position.side = "LONG"
        position.stop_loss_price = 95.0
        position.take_profit_price = None
        position.is_open = True
        database = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [position]
        database.execute = AsyncMock(return_value=result)

        with patch(
            "app.services.portfolio.tracker.stop_loss_calculator.is_stop_hit",
            return_value=True,
        ):
            candidates = await tracker.find_paper_exit_candidates(database, {"BTCUSDT": 94.0})

        assert len(candidates) == 1
        assert candidates[0].reason == "STOP_LOSS"
        assert candidates[0].observed_price == 94.0
        assert position.is_open is True

    @pytest.mark.asyncio
    async def test_price_updates_are_scoped_to_the_requested_execution_ledger(self):
        tracker = PortfolioTracker()
        position = MagicMock()
        position.symbol = "BTCUSDT"
        position.side = "LONG"
        position.entry_price = 100.0
        position.quantity = 1.0
        database = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [position]
        database.execute = AsyncMock(return_value=result)
        database.flush = AsyncMock()

        await tracker.update_prices(
            database,
            {"BTCUSDT": 101.0},
            execution_mode="TESTNET",
        )

        query = str(database.execute.await_args.args[0])
        assert "execution_mode" in query
        assert position.current_price == 101.0
