"""Fail-closed freshness checks for closed OHLCV candles."""

from datetime import UTC, datetime, timedelta

import pandas as pd


MAX_CANDLE_AGE_INTERVALS = 3
INTERVAL_DURATION = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}


def has_recent_closed_candle(
    candles: pd.DataFrame,
    interval: str,
    *,
    required_candles: int = 1,
    now: datetime | None = None,
) -> bool:
    """Return whether the required closed-candle window is fresh and continuous.

    Missing, malformed, future, stale, duplicated, or gapped timestamps are
    unsafe. The caller must reject the operation rather than infer quality from
    a price alone.
    """
    maximum_age = INTERVAL_DURATION.get(interval)
    if (
        candles.empty
        or maximum_age is None
        or required_candles < 1
        or "close_time" not in candles
    ):
        return False

    close_times = pd.to_datetime(candles["close_time"], utc=True, errors="coerce").dropna()
    if len(close_times) < required_candles:
        return False
    recent_close_times = close_times.sort_values().tail(required_candles)
    latest_close = recent_close_times.iloc[-1]
    if pd.isna(latest_close):
        return False

    reference_time = now or datetime.now(UTC)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=UTC)
    candle_age = reference_time - latest_close.to_pydatetime()
    if not timedelta(0) <= candle_age <= maximum_age * MAX_CANDLE_AGE_INTERVALS:
        return False

    if required_candles == 1:
        return True
    expected_delta = pd.Timedelta(maximum_age)
    tolerance = pd.Timedelta(seconds=1)
    actual_deltas = recent_close_times.diff().iloc[1:]
    return bool(((actual_deltas - expected_delta).abs() <= tolerance).all())
