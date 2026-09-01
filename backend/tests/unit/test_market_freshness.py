"""Tests for the fail-closed market-data freshness gate."""

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.services.market.freshness import has_recent_closed_candle


def _candles_with_close_time(close_time: datetime) -> pd.DataFrame:
    return pd.DataFrame({"close": [100.0], "close_time": [close_time]})


def test_recent_closed_candle_is_accepted() -> None:
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)

    assert has_recent_closed_candle(
        _candles_with_close_time(now - timedelta(minutes=14)),
        "15m",
        now=now,
    )


def test_stale_candle_is_rejected() -> None:
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)

    assert not has_recent_closed_candle(
        _candles_with_close_time(now - timedelta(hours=4)),
        "1h",
        now=now,
    )


def test_missing_or_future_close_time_is_rejected() -> None:
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)

    assert not has_recent_closed_candle(pd.DataFrame({"close": [100.0]}), "1h", now=now)
    assert not has_recent_closed_candle(
        _candles_with_close_time(now + timedelta(seconds=1)),
        "1h",
        now=now,
    )


def test_gapped_or_duplicated_history_is_rejected_for_a_strategy_window() -> None:
    now = datetime(2026, 9, 1, 12, tzinfo=UTC)
    close_times = [now - timedelta(hours=3), now - timedelta(hours=1), now]
    gapped = pd.DataFrame({"close": [100.0, 101.0, 102.0], "close_time": close_times})
    duplicated = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "close_time": [now - timedelta(hours=2), now - timedelta(hours=2), now],
        }
    )

    assert not has_recent_closed_candle(gapped, "1h", required_candles=3, now=now)
    assert not has_recent_closed_candle(duplicated, "1h", required_candles=3, now=now)
