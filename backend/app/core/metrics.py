"""Prometheus metrics for TradeMaster monitoring."""

from dataclasses import dataclass, field
from time import time
from typing import Callable

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Counter:
    """Simple counter metric."""

    name: str
    help: str
    _value: float = 0.0
    _labels: dict[str, float] = field(default_factory=dict)

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if labels:
            key = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            self._labels[key] = self._labels.get(key, 0) + value
        else:
            self._value += value

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        if self._labels:
            for key, val in self._labels.items():
                lines.append(f"{self.name}{{{key}}} {val}")
        else:
            lines.append(f"{self.name} {self._value}")
        return "\n".join(lines)


@dataclass
class Gauge:
    """Simple gauge metric."""

    name: str
    help: str
    _value: float = 0.0

    def set(self, value: float) -> None:
        self._value = value

    def inc(self, value: float = 1.0) -> None:
        self._value += value

    def dec(self, value: float = 1.0) -> None:
        self._value -= value

    def to_prometheus(self) -> str:
        return (
            f"# HELP {self.name} {self.help}\n"
            f"# TYPE {self.name} gauge\n"
            f"{self.name} {self._value}"
        )


@dataclass
class Histogram:
    """Simple histogram metric (fixed buckets)."""

    name: str
    help: str
    buckets: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    _counts: list[int] = field(default_factory=list)
    _sum: float = 0.0
    _count: int = 0

    def __post_init__(self):
        self._counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        self._sum += value
        self._count += 1
        for i, b in enumerate(self.buckets):
            if value <= b:
                self._counts[i] += 1
        self._counts[-1] += 1  # +Inf bucket

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        cumulative = 0
        for i, b in enumerate(self.buckets):
            cumulative += self._counts[i]
            lines.append(f'{self.name}_bucket{{le="{b}"}} {cumulative}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


class MetricsRegistry:
    """Central registry for all application metrics."""

    def __init__(self):
        # API metrics
        self.http_requests = Counter(
            "http_requests_total", "Total HTTP requests"
        )
        self.http_request_duration = Histogram(
            "http_request_duration_seconds", "HTTP request duration"
        )

        # Trading metrics
        self.signals_generated = Counter(
            "trading_signals_total", "Total trading signals generated"
        )
        self.orders_executed = Counter(
            "orders_executed_total", "Total orders executed"
        )
        self.orders_failed = Counter(
            "orders_failed_total", "Total orders that failed"
        )

        # Portfolio metrics
        self.total_equity = Gauge(
            "portfolio_total_equity", "Total portfolio equity in USD"
        )
        self.daily_pnl = Gauge(
            "portfolio_daily_pnl", "Daily P&L in USD"
        )
        self.drawdown = Gauge(
            "portfolio_drawdown", "Current drawdown percentage"
        )
        self.circuit_breaker_state = Gauge(
            "circuit_breaker_state", "Circuit breaker state (0=normal, 1=reduced, 2=paused, 3=halted)"
        )

        # ML metrics
        self.ml_inference_duration = Histogram(
            "ml_inference_duration_seconds", "ML inference duration"
        )

        # WebSocket
        self.ws_connections = Gauge(
            "websocket_active_connections", "Active WebSocket connections"
        )

        # Binance
        self.binance_rate_usage = Gauge(
            "binance_rate_limit_usage_ratio", "Binance API rate limit usage ratio"
        )

    def collect(self) -> str:
        """Collect all metrics in Prometheus text format."""
        metrics = [
            self.http_requests,
            self.http_request_duration,
            self.signals_generated,
            self.orders_executed,
            self.orders_failed,
            self.total_equity,
            self.daily_pnl,
            self.drawdown,
            self.circuit_breaker_state,
            self.ml_inference_duration,
            self.ws_connections,
            self.binance_rate_usage,
        ]
        return "\n\n".join(m.to_prometheus() for m in metrics) + "\n"


# Module-level singleton
metrics = MetricsRegistry()
