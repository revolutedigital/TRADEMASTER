"""Tests for repository pattern implementations."""

import pytest
from app.repositories.order_repo import OrderRepository
from app.repositories.position_repo import PositionRepository, SnapshotRepository
from app.repositories.market_repo import MarketDataRepository


class TestOrderRepository:
    def test_instantiation(self):
        repo = OrderRepository()
        assert repo is not None

    def test_has_required_methods(self):
        repo = OrderRepository()
        assert hasattr(repo, "create")
        assert hasattr(repo, "get_by_id")
        assert hasattr(repo, "get_open_orders")
        assert hasattr(repo, "get_filled_orders")
        assert hasattr(repo, "get_by_exchange_id")
        assert hasattr(repo, "update")
        assert hasattr(repo, "list_all")


class TestPositionRepository:
    def test_instantiation(self):
        repo = PositionRepository()
        assert repo is not None

    def test_has_required_methods(self):
        repo = PositionRepository()
        assert hasattr(repo, "get_open")
        assert hasattr(repo, "get_closed")
        assert hasattr(repo, "get_by_symbol_and_side")
        assert hasattr(repo, "get_total_exposure")
        assert hasattr(repo, "get_symbol_exposure")


class TestSnapshotRepository:
    def test_instantiation(self):
        repo = SnapshotRepository()
        assert repo is not None

    def test_has_required_methods(self):
        repo = SnapshotRepository()
        assert hasattr(repo, "get_recent")
        assert hasattr(repo, "get_since")


class TestMarketDataRepository:
    def test_instantiation(self):
        repo = MarketDataRepository()
        assert repo is not None

    def test_has_required_methods(self):
        repo = MarketDataRepository()
        assert hasattr(repo, "get_latest_candles")
        assert hasattr(repo, "get_latest_price")
        assert hasattr(repo, "get_candles_since")
