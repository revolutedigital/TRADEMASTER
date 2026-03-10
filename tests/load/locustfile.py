"""Load testing for TradeMaster API using Locust.

Run with:
    locust -f tests/load/locustfile.py --host http://localhost:8000
    # Then open http://localhost:8089 for the web UI

Headless mode:
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 5 --run-time 5m

Environment variables:
    TRADEMASTER_HOST    - API host (default: http://localhost:8000)
    TRADEMASTER_USER    - Login email (default: admin@trademaster.io)
    TRADEMASTER_PASS    - Login password (default: admin123)
"""

import logging
import os
import time

from locust import HttpUser, task, between, tag, events

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_HOST = os.getenv("TRADEMASTER_HOST", "http://localhost:8000")
LOGIN_EMAIL = os.getenv("TRADEMASTER_USER", "admin@trademaster.io")
LOGIN_PASSWORD = os.getenv("TRADEMASTER_PASS", "admin123")


# ---------------------------------------------------------------------------
# Custom event hooks for metrics collection
# ---------------------------------------------------------------------------

_request_stats: dict[str, list[float]] = {}


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
    if response_time and response_time > 2000:
        logger.warning(
            "Slow request: %s %s took %.0fms (status=%s)",
            request_type,
            name,
            response_time,
            getattr(response, "status_code", "N/A") if response else "error",
        )

    # Log failures
    if exception:
        logger.error(
            "Request failed: %s %s - %s", request_type, name, exception
        )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary statistics when the test ends."""
    if not _request_stats:
        return

    logger.info("=" * 60)
    logger.info("Custom Metrics Summary")
    logger.info("=" * 60)

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
            "%-45s  n=%-6d  avg=%-8.1f  p50=%-8.1f  p95=%-8.1f  p99=%-8.1f",
            endpoint, n, avg, p50, p95, p99,
        )

    _request_stats.clear()


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Check failure ratio and set exit code."""
    if environment.stats.total.fail_ratio > 0.10:
        logger.error(
            "Failure ratio %.2f%% exceeds 10%% threshold",
            environment.stats.total.fail_ratio * 100,
        )
        environment.process_exit_code = 1


# ---------------------------------------------------------------------------
# User classes
# ---------------------------------------------------------------------------


class TradeMasterUser(HttpUser):
    """Simulates a typical authenticated TradeMaster dashboard user.

    Task weights reflect realistic usage patterns:
    - Dashboard views are most frequent (weight 3)
    - Portfolio checks are common (weight 2)
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
            response = self.client.post(
                "/api/v1/auth/login",
                json={
                    "email": LOGIN_EMAIL,
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
                # If login fails, proceed without auth (endpoints may return 401)
                self.client.headers.update({
                    "Content-Type": "application/json",
                })
                response.failure(f"Login failed: {response.status_code}")
                logger.warning(
                    "Login failed with status %d, proceeding without auth",
                    response.status_code,
                )
        except Exception as exc:
            logger.error("Login exception: %s", exc)
            self.client.headers.update({"Content-Type": "application/json"})

    # -- Dashboard (weight 3) --

    @task(3)
    @tag("read", "dashboard")
    def get_dashboard(self):
        """Fetch main dashboard data (health + portfolio summary)."""
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

    # -- Portfolio (weight 2) --

    @task(2)
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

    # -- Market data (weight 2) --

    @task(2)
    @tag("read", "market")
    def get_market_tickers(self):
        """Fetch current market ticker prices."""
        self.client.get(
            "/api/v1/market/tickers",
            name="/api/v1/market/tickers",
        )

    @task(1)
    @tag("read", "market")
    def get_klines(self):
        """Fetch OHLCV candle data for BTC."""
        self.client.get(
            "/api/v1/market/klines/BTCUSDT?interval=1h&limit=100",
            name="/api/v1/market/klines/[symbol]",
        )
