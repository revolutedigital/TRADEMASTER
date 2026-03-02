"""Tests for portfolio tracker service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

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
        mock_position.id = 1
        mock_position.symbol = "BTCUSDT"

        with patch("app.services.portfolio.tracker.event_bus") as mock_bus:
            mock_bus.publish = AsyncMock()
            result = await tracker.close_position(mock_db, mock_position, 84000.0)

        # P&L for SHORT: (85000 - 84000) * 0.1 = 100
        assert result.realized_pnl == 100.0
