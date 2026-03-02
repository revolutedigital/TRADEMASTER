"""Execution analytics: measure and optimize trade execution quality."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionRecord:
    order_id: str
    symbol: str
    side: str
    intended_price: Decimal
    fill_price: Decimal
    quantity: Decimal
    slippage_bps: float
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def slippage_cost(self) -> Decimal:
        return abs(self.fill_price - self.intended_price) * self.quantity


class ExecutionAnalytics:
    """Track and analyze trade execution quality."""

    def __init__(self):
        self._records: list[ExecutionRecord] = []

    def record_execution(self, order_id: str, symbol: str, side: str, intended_price: Decimal, fill_price: Decimal, quantity: Decimal, latency_ms: float):
        """Record an execution for analysis."""
        if intended_price > 0:
            slippage_bps = float((fill_price - intended_price) / intended_price * 10000)
            if side == "SELL":
                slippage_bps = -slippage_bps
        else:
            slippage_bps = 0.0

        record = ExecutionRecord(
            order_id=order_id, symbol=symbol, side=side,
            intended_price=intended_price, fill_price=fill_price,
            quantity=quantity, slippage_bps=slippage_bps, latency_ms=latency_ms,
        )
        self._records.append(record)

        if len(self._records) > 10000:
            self._records = self._records[-5000:]

    def get_slippage_analysis(self, symbol: str | None = None, last_n: int = 100) -> dict:
        """Analyze slippage across recent executions."""
        records = self._records[-last_n:]
        if symbol:
            records = [r for r in records if r.symbol == symbol]
        if not records:
            return {"avg_slippage_bps": 0, "total_slippage_cost": "0", "count": 0}

        slippages = [r.slippage_bps for r in records]
        total_cost = sum(r.slippage_cost for r in records)

        return {
            "avg_slippage_bps": round(sum(slippages) / len(slippages), 2),
            "max_slippage_bps": round(max(slippages), 2),
            "min_slippage_bps": round(min(slippages), 2),
            "total_slippage_cost": str(total_cost.quantize(Decimal("0.01"))),
            "count": len(records),
        }

    def get_latency_analysis(self, last_n: int = 100) -> dict:
        """Analyze execution latency."""
        records = self._records[-last_n:]
        if not records:
            return {"avg_latency_ms": 0, "p95_latency_ms": 0, "count": 0}

        latencies = sorted([r.latency_ms for r in records])
        p95_idx = int(len(latencies) * 0.95)

        return {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "p50_latency_ms": round(latencies[len(latencies) // 2], 1),
            "p95_latency_ms": round(latencies[min(p95_idx, len(latencies) - 1)], 1),
            "p99_latency_ms": round(latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)], 1),
            "min_latency_ms": round(latencies[0], 1),
            "max_latency_ms": round(latencies[-1], 1),
            "count": len(records),
        }

    def get_fill_rate(self, last_n: int = 100) -> dict:
        """Analyze fill rates."""
        records = self._records[-last_n:]
        filled = [r for r in records if r.fill_price > 0]
        return {
            "fill_rate": round(len(filled) / len(records) * 100, 1) if records else 0,
            "total_orders": len(records),
            "filled_orders": len(filled),
        }

    def get_best_execution_report(self) -> dict:
        """Generate comprehensive best execution compliance report."""
        return {
            "slippage": self.get_slippage_analysis(),
            "latency": self.get_latency_analysis(),
            "fill_rate": self.get_fill_rate(),
            "total_executions": len(self._records),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


execution_analytics = ExecutionAnalytics()
