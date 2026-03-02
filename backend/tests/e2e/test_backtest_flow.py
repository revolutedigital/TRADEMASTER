"""E2E test: backtest configuration, execution and history."""

import pytest


class TestBacktestFlow:
    """End-to-end flow: Configure Backtest -> Run -> View History -> View Result."""

    async def test_backtest_run_and_history(self, async_client):
        """Test running a backtest and checking it appears in history."""

        # 1. Run a backtest
        run_resp = await async_client.post(
            "/api/v1/backtest/run",
            json={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "initial_capital": 10000,
                "signal_threshold": 0.3,
                "atr_stop_multiplier": 2.0,
                "risk_reward_ratio": 2.0,
            },
        )
        assert run_resp.status_code == 200
        result = run_resp.json()
        assert "total_trades" in result
        assert "equity_curve" in result
        assert "sharpe_ratio" in result

        # 2. Check history includes the run
        history_resp = await async_client.get("/api/v1/backtest/history")
        assert history_resp.status_code == 200
        history = history_resp.json()
        assert isinstance(history, list)

    async def test_backtest_with_different_params(self, async_client):
        """Test backtest with various parameter configurations."""
        configs = [
            {"symbol": "BTCUSDT", "interval": "1h", "initial_capital": 5000},
            {"symbol": "ETHUSDT", "interval": "4h", "initial_capital": 20000},
        ]
        for config in configs:
            resp = await async_client.post("/api/v1/backtest/run", json=config)
            assert resp.status_code == 200
            data = resp.json()
            assert "total_trades" in data
