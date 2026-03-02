"""Load testing for TradeMaster API using Locust."""

from locust import HttpUser, task, between, tag


class TradeMasterUser(HttpUser):
    """Simulates a typical TradeMaster dashboard user."""

    wait_time = between(1, 3)
    host = "http://localhost:8000"

    def on_start(self):
        """Login or setup before tasks."""
        self.client.headers.update({"Content-Type": "application/json"})

    @task(5)
    @tag("read", "dashboard")
    def get_health(self):
        self.client.get("/api/v1/system/health")

    @task(3)
    @tag("read", "dashboard")
    def get_portfolio_summary(self):
        self.client.get("/api/v1/portfolio/summary")

    @task(3)
    @tag("read", "market")
    def get_tickers(self):
        self.client.get("/api/v1/market/tickers")

    @task(2)
    @tag("read", "market")
    def get_klines_btc(self):
        self.client.get("/api/v1/market/klines/BTCUSDT?interval=1h&limit=100")

    @task(2)
    @tag("read", "market")
    def get_klines_eth(self):
        self.client.get("/api/v1/market/klines/ETHUSDT?interval=1h&limit=100")

    @task(2)
    @tag("read", "market")
    def get_depth(self):
        self.client.get("/api/v1/market/depth/BTCUSDT?limit=25")

    @task(2)
    @tag("read", "trading")
    def get_trade_history(self):
        self.client.get("/api/v1/trading/history?limit=50")

    @task(1)
    @tag("read", "risk")
    def get_risk_metrics(self):
        self.client.get("/api/v1/risk/metrics")

    @task(1)
    @tag("read", "signals")
    def get_signals(self):
        self.client.get("/api/v1/signals/history?limit=20")

    @task(1)
    @tag("read", "backtest")
    def get_backtest_history(self):
        self.client.get("/api/v1/backtest/history")

    @task(1)
    @tag("write", "trading")
    def execute_paper_trade(self):
        self.client.post("/api/v1/trading/paper-trade", json={
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.001,
        })

    @task(1)
    @tag("write", "backtest")
    def run_backtest(self):
        self.client.post("/api/v1/backtest/run", json={
            "symbol": "BTCUSDT",
            "strategy": "sma_crossover",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
        })


class TradeMasterAdmin(HttpUser):
    """Simulates an admin user checking system health."""

    wait_time = between(5, 10)
    host = "http://localhost:8000"
    weight = 1  # Lower weight = fewer admin users

    @task(3)
    @tag("admin")
    def get_detailed_health(self):
        self.client.get("/api/v1/system/health/detailed")

    @task(2)
    @tag("admin")
    def get_metrics(self):
        self.client.get("/api/v1/system/metrics")

    @task(1)
    @tag("admin")
    def get_engine_status(self):
        self.client.get("/api/v1/trading/engine/status")
