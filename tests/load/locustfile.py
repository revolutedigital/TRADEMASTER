"""Load testing for TradeMaster API using Locust.

Run with:
    locust -f tests/load/locustfile.py --host http://localhost:8000
    # Then open http://localhost:8089 for the web UI

Headless mode (CI/CD):
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 5 --run-time 5m

Quick smoke test:
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --headless -u 10 -r 2 --run-time 1m --tags read

Environment variables:
    TRADEMASTER_HOST    - API host (default: http://localhost:8000)
    TRADEMASTER_USER    - Login username (default: admin)
    TRADEMASTER_PASS    - Login password (default: trademaster2024)
"""

import json
import logging
import os
import time

from locust import HttpUser, task, between, tag, events

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_HOST = os.getenv("TRADEMASTER_HOST", "http://localhost:8000")
LOGIN_USER = os.getenv("TRADEMASTER_USER", "admin")
LOGIN_PASSWORD = os.getenv("TRADEMASTER_PASS", "trademaster2024")

# SLA thresholds (milliseconds)
SLA_P95_MS = 2000  # 95th percentile must be under 2s
SLA_P99_MS = 5000  # 99th percentile must be under 5s
SLA_FAILURE_RATE = 0.10  # Max 10% failure rate


# ---------------------------------------------------------------------------
# Custom event hooks for metrics collection
# ---------------------------------------------------------------------------

_request_stats: dict[str, list[float]] = {}
_sla_violations: list[str] = []


@events.request.add_listener
def on_request(
    request_type, name, response_time, response_length, response, exception, **kwargs
):
    """Collect per-endpoint response times for custom reporting."""
    key = f"{request_type} {name}"
    if key not in _request_stats:
        _request_stats[key] = []
    _request_stats[key].append(response_time)

    # Log slow requests
    if response_time and response_time > SLA_P95_MS:
        logger.warning(
            "Slow request: %s %s took %.0fms (status=%s)",
            request_type,
            name,
            response_time,
            getattr(response, "status_code", "N/A") if response else "error",
        )

    # Log failures
    if exception:
        logger.error("Request failed: %s %s - %s", request_type, name, exception)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary statistics and check SLA compliance."""
    if not _request_stats:
        return

    logger.info("=" * 72)
    logger.info("Load Test Results Summary")
    logger.info("=" * 72)
    logger.info(
        "%-45s  %6s  %8s  %8s  %8s  %8s",
        "Endpoint", "Count", "Avg(ms)", "P50(ms)", "P95(ms)", "P99(ms)",
    )
    logger.info("-" * 72)

    for endpoint, times in sorted(_request_stats.items()):
        if not times:
            continue
        times_sorted = sorted(times)
        n = len(times_sorted)
        avg = sum(times_sorted) / n
        p50 = times_sorted[int(n * 0.50)]
        p95 = times_sorted[int(min(n * 0.95, n - 1))]
        p99 = times_sorted[int(min(n * 0.99, n - 1))]

        logger.info(
            "%-45s  %6d  %8.1f  %8.1f  %8.1f  %8.1f",
            endpoint, n, avg, p50, p95, p99,
        )

        # Check SLA
        if p95 > SLA_P95_MS:
            _sla_violations.append(f"{endpoint}: p95={p95:.0f}ms > {SLA_P95_MS}ms")
        if p99 > SLA_P99_MS:
            _sla_violations.append(f"{endpoint}: p99={p99:.0f}ms > {SLA_P99_MS}ms")

    logger.info("=" * 72)

    if _sla_violations:
        logger.error("SLA VIOLATIONS:")
        for v in _sla_violations:
            logger.error("  - %s", v)
    else:
        logger.info("All endpoints within SLA thresholds.")

    _request_stats.clear()
    _sla_violations.clear()


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Check failure ratio and set exit code for CI/CD."""
    if environment.stats.total.fail_ratio > SLA_FAILURE_RATE:
        logger.error(
            "FAIL: Error rate %.2f%% exceeds %.0f%% threshold",
            environment.stats.total.fail_ratio * 100,
            SLA_FAILURE_RATE * 100,
        )
        environment.process_exit_code = 1
    elif _sla_violations:
        logger.error("FAIL: SLA violations detected")
        environment.process_exit_code = 1
    else:
        logger.info("PASS: Load test completed within all thresholds")


# ---------------------------------------------------------------------------
# User classes
# ---------------------------------------------------------------------------


class TradeMasterUser(HttpUser):
    """Simulates a typical authenticated TradeMaster dashboard user.

    Task weights reflect realistic usage patterns:
    - Health/dashboard checks are most frequent (weight 5)
    - Market data views are very common (weight 4)
    - Portfolio checks are common (weight 3)
    - Signal checks are moderate (weight 2)
    - Trades and backtests are less frequent (weight 1)
    """

    wait_time = between(1, 3)
    host = DEFAULT_HOST

    def on_start(self):
        """Authenticate and store the JWT token for subsequent requests."""
        self.token = None
        self._login()

    def _login(self):
        """Attempt login and configure auth headers."""
        try:
            # Try username/password login (the TradeMaster default)
            response = self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": LOGIN_USER,
                    "password": LOGIN_PASSWORD,
                },
                name="/api/v1/auth/login",
                catch_response=True,
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token", "")
                self.client.headers.update({
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                })
                response.success()
                logger.info("User logged in successfully")
            else:
                # Proceed without auth -- endpoints may return 401
                self.client.headers.update({"Content-Type": "application/json"})
                response.failure(f"Login failed: {response.status_code}")
                logger.warning(
                    "Login failed with status %d, proceeding without auth",
                    response.status_code,
                )
        except Exception as exc:
            logger.error("Login exception: %s", exc)
            self.client.headers.update({"Content-Type": "application/json"})

    # -- Health / Dashboard (weight 5) --

    @task(5)
    @tag("read", "health", "dashboard")
    def get_health(self):
        """Health check endpoint -- should always be fast."""
        with self.client.get(
            "/api/v1/system/health",
            name="/api/v1/system/health",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                response.failure("Unauthorized - token may be expired")
                self._login()
            else:
                response.failure(f"Status {response.status_code}")

    # -- Market data (weight 4) --

    @task(4)
    @tag("read", "market")
    def get_market_tickers(self):
        """Fetch current market ticker prices."""
        self.client.get(
            "/api/v1/market/tickers",
            name="/api/v1/market/tickers",
        )

    @task(2)
    @tag("read", "market")
    def get_klines(self):
        """Fetch OHLCV candle data for BTC."""
        self.client.get(
            "/api/v1/market/klines/BTCUSDT?interval=1h&limit=100",
            name="/api/v1/market/klines/[symbol]",
        )

    # -- Portfolio (weight 3) --

    @task(3)
    @tag("read", "portfolio")
    def get_portfolio(self):
        """Fetch portfolio summary and positions."""
        self.client.get(
            "/api/v1/portfolio/summary",
            name="/api/v1/portfolio/summary",
        )

    # -- Signals (weight 2) --

    @task(2)
    @tag("read", "signals")
    def get_signals(self):
        """Fetch recent AI prediction signals."""
        self.client.get(
            "/api/v1/signals/history?limit=20",
            name="/api/v1/signals/history",
        )

    # -- Trade history (weight 2) --

    @task(2)
    @tag("read", "trading")
    def get_trade_history(self):
        """Fetch recent trade history."""
        self.client.get(
            "/api/v1/trading/history?limit=20",
            name="/api/v1/trading/history",
        )

    # -- Trading engine status (weight 1) --

    @task(1)
    @tag("read", "trading")
    def get_engine_status(self):
        """Check trading engine status."""
        self.client.get(
            "/api/v1/trading/engine/status",
            name="/api/v1/trading/engine/status",
        )

    # -- Paper trading (weight 1) --

    @task(1)
    @tag("write", "trading")
    def execute_paper_trade(self):
        """Submit a paper trade order."""
        with self.client.post(
            "/api/v1/trading/paper-trade",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": 0.001,
            },
            name="/api/v1/trading/paper-trade",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201):
                response.success()
            elif response.status_code == 401:
                response.failure("Unauthorized")
                self._login()
            else:
                response.failure(f"Status {response.status_code}")

    # -- Backtest (weight 1) --

    @task(1)
    @tag("write", "backtest")
    def run_backtest(self):
        """Trigger a backtest run."""
        with self.client.post(
            "/api/v1/backtest/run",
            json={
                "symbol": "BTCUSDT",
                "strategy": "sma_crossover",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
            },
            name="/api/v1/backtest/run",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201, 202):
                response.success()
            elif response.status_code == 401:
                response.failure("Unauthorized")
                self._login()
            else:
                response.failure(f"Status {response.status_code}")

    # -- Risk metrics (weight 1) --

    @task(1)
    @tag("read", "risk")
    def get_risk_metrics(self):
        """Fetch current risk exposure and VaR metrics."""
        self.client.get(
            "/api/v1/risk/metrics",
            name="/api/v1/risk/metrics",
        )


class TradeMasterHeavyUser(HttpUser):
    """Simulates a power user hitting compute-intensive endpoints.

    Separate user class with lower spawn rate for expensive operations
    like backtests and model training status checks.
    """

    wait_time = between(5, 15)
    host = DEFAULT_HOST
    weight = 1  # 1/3 the spawn rate of TradeMasterUser (default weight=1)

    def on_start(self):
        self.token = None
        self._login()

    def _login(self):
        try:
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": LOGIN_USER, "password": LOGIN_PASSWORD},
                name="/api/v1/auth/login [heavy]",
                catch_response=True,
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token", "")
                self.client.headers.update({
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                })
                response.success()
            else:
                self.client.headers.update({"Content-Type": "application/json"})
                response.failure(f"Login failed: {response.status_code}")
        except Exception as exc:
            logger.error("Heavy user login exception: %s", exc)
            self.client.headers.update({"Content-Type": "application/json"})

    @task(3)
    @tag("write", "backtest", "heavy")
    def run_full_backtest(self):
        """Run a longer backtest period."""
        with self.client.post(
            "/api/v1/backtest/run",
            json={
                "symbol": "BTCUSDT",
                "strategy": "sma_crossover",
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
            },
            name="/api/v1/backtest/run [full]",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201, 202):
                response.success()
            elif response.status_code == 401:
                response.failure("Unauthorized")
                self._login()
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    @tag("read", "market", "heavy")
    def get_large_klines(self):
        """Fetch large OHLCV dataset (500 candles)."""
        self.client.get(
            "/api/v1/market/klines/BTCUSDT?interval=1h&limit=500",
            name="/api/v1/market/klines/[symbol] [500]",
        )

    @task(1)
    @tag("read", "trading", "heavy")
    def get_engine_status(self):
        """Check trading engine status."""
        self.client.get(
            "/api/v1/trading/engine/status",
            name="/api/v1/trading/engine/status [heavy]",
        )
