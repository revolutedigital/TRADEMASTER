"""Factory Boy factories for all TradeMaster models.

Provides deterministic, reproducible test data generation.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
import random
import string


class BaseFactory:
    """Base factory with common utilities."""
    
    _counter = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._counter += 1
        return cls._counter

    @classmethod
    def _random_string(cls, length: int = 8) -> str:
        return ''.join(random.choices(string.ascii_lowercase, k=length))


class OHLCVFactory(BaseFactory):
    """Factory for OHLCV candlestick data."""

    @classmethod
    def create(cls, **kwargs):
        base_price = kwargs.get("base_price", 50000.0)
        variation = random.uniform(-0.02, 0.02)
        open_price = base_price * (1 + variation)
        close_price = open_price * (1 + random.uniform(-0.01, 0.01))
        high = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
        low = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
        
        return {
            "symbol": kwargs.get("symbol", "BTCUSDT"),
            "interval": kwargs.get("interval", "1h"),
            "open_time": kwargs.get("open_time", datetime.now(timezone.utc)),
            "open": Decimal(str(round(open_price, 2))),
            "high": Decimal(str(round(high, 2))),
            "low": Decimal(str(round(low, 2))),
            "close": Decimal(str(round(close_price, 2))),
            "volume": Decimal(str(round(random.uniform(100, 5000), 4))),
            "close_time": kwargs.get("close_time", datetime.now(timezone.utc) + timedelta(hours=1)),
            "quote_volume": Decimal(str(round(random.uniform(5000000, 50000000), 2))),
            "trade_count": random.randint(1000, 50000),
        }

    @classmethod
    def create_batch(cls, count: int, **kwargs) -> list[dict]:
        base_time = kwargs.pop("start_time", datetime.now(timezone.utc) - timedelta(hours=count))
        interval_delta = timedelta(hours=1)
        results = []
        for i in range(count):
            open_time = base_time + interval_delta * i
            results.append(cls.create(open_time=open_time, close_time=open_time + interval_delta, **kwargs))
        return results


class TradeFactory(BaseFactory):
    """Factory for trade records."""

    @classmethod
    def create(cls, **kwargs):
        side = kwargs.get("side", random.choice(["BUY", "SELL"]))
        price = kwargs.get("price", round(random.uniform(30000, 70000), 2))
        quantity = kwargs.get("quantity", round(random.uniform(0.001, 1.0), 6))
        
        return {
            "id": cls._next_id(),
            "symbol": kwargs.get("symbol", "BTCUSDT"),
            "side": side,
            "order_type": kwargs.get("order_type", "MARKET"),
            "quantity": Decimal(str(quantity)),
            "price": Decimal(str(price)),
            "commission": Decimal(str(round(price * quantity * 0.001, 4))),
            "status": kwargs.get("status", "FILLED"),
            "exchange_order_id": kwargs.get("exchange_order_id", f"EX{cls._next_id():08d}"),
            "created_at": kwargs.get("created_at", datetime.now(timezone.utc)),
        }

    @classmethod
    def create_batch(cls, count: int, **kwargs) -> list[dict]:
        return [cls.create(**kwargs) for _ in range(count)]


class SignalFactory(BaseFactory):
    """Factory for trading signals."""

    @classmethod
    def create(cls, **kwargs):
        action = kwargs.get("action", random.choice(["BUY", "SELL", "HOLD"]))
        return {
            "id": cls._next_id(),
            "symbol": kwargs.get("symbol", "BTCUSDT"),
            "action": action,
            "confidence": kwargs.get("confidence", round(random.uniform(0.5, 0.95), 4)),
            "model_type": kwargs.get("model_type", random.choice(["lstm", "xgboost", "ensemble"])),
            "features_snapshot": kwargs.get("features_snapshot", "{}"),
            "created_at": kwargs.get("created_at", datetime.now(timezone.utc)),
        }


class PortfolioSnapshotFactory(BaseFactory):
    """Factory for portfolio snapshots."""

    @classmethod
    def create(cls, **kwargs):
        equity = kwargs.get("total_equity", round(random.uniform(8000, 15000), 2))
        unrealized = round(random.uniform(-500, 500), 2)
        return {
            "id": cls._next_id(),
            "total_equity": Decimal(str(equity)),
            "available_balance": Decimal(str(equity - abs(unrealized))),
            "unrealized_pnl": Decimal(str(unrealized)),
            "daily_pnl": Decimal(str(round(random.uniform(-200, 200), 2))),
            "btc_quantity": Decimal(str(round(random.uniform(0, 0.5), 8))),
            "eth_quantity": Decimal(str(round(random.uniform(0, 5), 8))),
            "created_at": kwargs.get("created_at", datetime.now(timezone.utc)),
        }


class UserFactory(BaseFactory):
    """Factory for user records."""

    @classmethod
    def create(cls, **kwargs):
        username = kwargs.get("username", f"user_{cls._random_string(6)}")
        return {
            "id": cls._next_id(),
            "username": username,
            "email": kwargs.get("email", f"{username}@test.com"),
            "password_hash": kwargs.get("password_hash", "$2b$12$FAKEHASHFAKEHASHFAKEHASHFAKEHASHFAKEHASHFAKE"),
            "role": kwargs.get("role", "trader"),
            "is_active": kwargs.get("is_active", True),
            "created_at": kwargs.get("created_at", datetime.now(timezone.utc)),
        }


class AlertFactory(BaseFactory):
    """Factory for price alerts."""

    @classmethod
    def create(cls, **kwargs):
        return {
            "id": cls._next_id(),
            "symbol": kwargs.get("symbol", "BTCUSDT"),
            "condition": kwargs.get("condition", random.choice(["above", "below"])),
            "target_price": Decimal(str(kwargs.get("target_price", round(random.uniform(30000, 70000), 2)))),
            "is_triggered": kwargs.get("is_triggered", False),
            "triggered_at": kwargs.get("triggered_at", None),
            "created_at": kwargs.get("created_at", datetime.now(timezone.utc)),
        }


class JournalEntryFactory(BaseFactory):
    """Factory for journal entries."""

    @classmethod
    def create(cls, **kwargs):
        sentiments = ["bullish", "bearish", "neutral"]
        tags_options = [["swing", "btc"], ["scalp", "eth"], ["breakout"], ["reversal", "btc"]]
        return {
            "id": cls._next_id(),
            "trade_id": kwargs.get("trade_id", None),
            "notes": kwargs.get("notes", f"Trade analysis entry {cls._next_id()}"),
            "tags": kwargs.get("tags", random.choice(tags_options)),
            "sentiment": kwargs.get("sentiment", random.choice(sentiments)),
            "lessons_learned": kwargs.get("lessons_learned", "Always wait for confirmation before entry."),
            "created_at": kwargs.get("created_at", datetime.now(timezone.utc)),
        }
