"""Tests for the research-only one-minute FX dataset loader."""

import lzma
import struct
import threading
import time
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

from scripts.research.fx_dataset_m1 import (
    ALL_PAIRS,
    MAJORS,
    build_symbol_frame,
    cache_path,
    day_url,
    decode_minute_candles,
    ensure_cached,
    project_download,
    quality_report_m1,
    trading_days,
)

EURUSD = MAJORS["EURUSD"]
USDJPY = MAJORS["USDJPY"]


def _payload(rows: list[tuple[int, int, int, int, int, float]]) -> bytes:
    """Pack (offset_seconds, open, close, low, high, volume) rows like the provider does."""
    return lzma.compress(b"".join(struct.pack(">IIIIIf", *row) for row in rows))


def _minute(index: int) -> int:
    return index * 60


def test_day_url_uses_the_providers_zero_based_month_and_one_based_day() -> None:
    url = day_url("EURUSD", date(2024, 5, 14), "BID")

    assert url.endswith("/EURUSD/2024/04/14/BID_candles_min_1.bi5")
    assert day_url("EURUSD", date(2024, 1, 1), "ASK").endswith("/2024/00/01/ASK_candles_min_1.bi5")
    with pytest.raises(ValueError):
        day_url("EURUSD", date(2024, 5, 14), "MID")


def test_trading_days_skip_saturday_and_keep_sunday() -> None:
    days = trading_days(date(2024, 5, 10), date(2024, 5, 14))  # Fri, Sat, Sun, Mon, Tue

    assert days == [date(2024, 5, 10), date(2024, 5, 12), date(2024, 5, 13), date(2024, 5, 14)]
    with pytest.raises(ValueError):
        trading_days(date(2024, 5, 14), date(2024, 5, 10))


def test_decode_places_minutes_by_offset_from_midnight_and_drops_placeholders() -> None:
    payload = _payload(
        [
            (_minute(0), 108500, 108500, 108500, 108500, 0.0),  # closed minute
            (_minute(1), 108510, 108520, 108505, 108525, 5.0),
            (_minute(2), 108520, 108515, 108512, 108522, 3.0),
        ]
    )

    frame = decode_minute_candles(payload, day=date(2024, 5, 14), price_scale=100_000)

    assert list(frame.index) == [
        pd.Timestamp("2024-05-14 00:01", tz="UTC"),
        pd.Timestamp("2024-05-14 00:02", tz="UTC"),
    ]
    assert frame.iloc[0]["close"] == pytest.approx(1.0852)


def test_decode_of_an_empty_payload_returns_an_empty_frame() -> None:
    frame = decode_minute_candles(b"", day=date(2024, 5, 14), price_scale=100_000)

    assert frame.empty
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ensure_cached_downloads_only_what_is_missing_and_reports_the_pace(tmp_path: Path) -> None:
    calls: list[str] = []
    ticks = iter([0.0, 30.0])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, content=b"x")

    days = [date(2024, 5, 13), date(2024, 5, 14)]
    cache_path(tmp_path, "EURUSD", days[0], "BID").parent.mkdir(parents=True)
    cache_path(tmp_path, "EURUSD", days[0], "BID").write_bytes(b"cached")

    with _client(handler) as client:
        stats = ensure_cached(
            client, "EURUSD", days, tmp_path, workers=2, clock=lambda: next(ticks)
        )

    assert len(calls) == 3
    assert stats.files_requested == 4
    assert stats.files_downloaded == 3
    assert stats.files_from_cache == 1
    assert stats.files_per_minute == pytest.approx(3 / 0.5)


def test_ensure_cached_never_uses_more_workers_than_allowed(tmp_path: Path) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return httpx.Response(200, content=b"x")

    days = trading_days(date(2024, 5, 6), date(2024, 5, 12))

    with _client(handler) as client:
        ensure_cached(client, "EURUSD", days, tmp_path, workers=2)

    assert peak <= 2


def test_ensure_cached_rejects_zero_workers_and_a_second_pass_hits_the_cache(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x")

    days = [date(2024, 5, 14)]
    with _client(handler) as client:
        with pytest.raises(ValueError):
            ensure_cached(client, "EURUSD", days, tmp_path, workers=0)
        ensure_cached(client, "EURUSD", days, tmp_path)
        again = ensure_cached(client, "EURUSD", days, tmp_path)

    assert again.files_downloaded == 0
    assert again.files_from_cache == 2
    assert again.files_per_minute == 0.0


def test_build_symbol_frame_joins_bid_and_ask_and_measures_the_spread_in_pips(tmp_path: Path) -> None:
    day = date(2024, 5, 14)
    bid = _payload([(_minute(1), 148250, 148300, 148200, 148350, 4.0)])
    ask = _payload([(_minute(1), 148262, 148314, 148212, 148362, 4.0)])
    for side, payload in (("BID", bid), ("ASK", ask)):
        path = cache_path(tmp_path, "USDJPY", day, side)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    frame = build_symbol_frame(tmp_path, USDJPY, [day])

    assert frame["spread_open_pips"].iloc[0] == pytest.approx(1.2)
    assert frame["spread_close_pips"].iloc[0] == pytest.approx(1.4)


def test_build_symbol_frame_refuses_days_that_were_not_downloaded(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_symbol_frame(tmp_path, EURUSD, [date(2024, 5, 14)])
    with pytest.raises(ValueError):
        build_symbol_frame(tmp_path, EURUSD, [])


def _week_of_minutes() -> pd.DataFrame:
    """Sunday 21:00 UTC to Friday 21:00 UTC, every minute present, with a realistic spread."""
    index = pd.date_range("2024-05-12 21:00", "2024-05-17 21:00", freq="min", tz="UTC")
    frame = pd.DataFrame(index=index)
    mid = 1.08 + (pd.Series(range(len(index)), index=index) % 200) * 1e-6
    for side, sign in (("bid", -1), ("ask", 1)):
        frame[f"{side}_open"] = mid + sign * 0.000015
        frame[f"{side}_high"] = mid + 0.00003 + sign * 0.000015
        frame[f"{side}_low"] = mid - 0.00003 + sign * 0.000015
        frame[f"{side}_close"] = mid + sign * 0.000015
    frame["spread_open_pips"] = 0.3
    frame["spread_close_pips"] = 0.3
    return frame


def test_quality_report_accepts_a_clean_session_week_over_many_weeks() -> None:
    # Density is only meaningful over months: a short window ending on a Friday leaves out
    # the trailing weekend and overstates it.
    weeks = [_week_of_minutes().shift(week * 7 * 24 * 60, freq="min") for week in range(26)]
    frame = pd.concat(weeks)

    report = quality_report_m1(frame, EURUSD)

    assert report["problems"] == []
    assert report["density"] == pytest.approx(5 / 7, abs=0.01)
    assert report["saturday_minutes"] == 0
    assert report["spread_median_pips"] == pytest.approx(0.3)


def test_quality_report_flags_saturday_minutes_crossed_quotes_and_wrong_scale() -> None:
    frame = pd.concat([_week_of_minutes().shift(week * 7 * 24 * 60, freq="min") for week in range(4)])
    saturday = pd.Timestamp("2024-05-18 10:00", tz="UTC")
    frame.loc[saturday] = frame.iloc[-1]
    frame = frame.sort_index()
    frame.iloc[10, frame.columns.get_loc("ask_open")] = frame.iloc[10]["bid_open"] - 0.0002

    report = quality_report_m1(frame, EURUSD)
    scaled = frame.copy()
    for column in scaled.columns:
        if column.startswith(("bid_", "ask_")):
            scaled[column] = scaled[column] * 100

    joined = " | ".join(report["problems"])
    assert "Saturday" in joined and "ask below bid" in joined
    assert any("implausible" in problem for problem in quality_report_m1(scaled, EURUSD)["problems"])


def test_quality_report_reports_the_tail_of_the_spread_distribution() -> None:
    frame = pd.concat([_week_of_minutes().shift(week * 7 * 24 * 60, freq="min") for week in range(26)])
    frame.iloc[::100, frame.columns.get_loc("spread_open_pips")] = 6.0

    report = quality_report_m1(frame, EURUSD)

    assert report["share_spread_over_3_pips"] == pytest.approx(0.01, abs=0.002)
    assert report["spread_p99_pips"] >= 0.3


def test_projection_scales_with_pairs_years_and_pace() -> None:
    small = project_download(pairs=1, years=1, files_per_minute=10)
    large = project_download(pairs=10, years=5, files_per_minute=10)

    assert small.files_total == round(365 * 6 / 7 * 2)
    assert large.files_total == pytest.approx(50 * small.files_total, rel=0.001)
    assert large.hours_at_measured_pace == pytest.approx(large.files_total / 10 / 60)
    with pytest.raises(ValueError):
        project_download(pairs=1, years=1, files_per_minute=0)


def test_the_research_universe_is_seven_majors_plus_three_liquid_crosses() -> None:
    assert set(ALL_PAIRS) == set(MAJORS) | {"EURJPY", "GBPJPY", "EURGBP"}
    assert ALL_PAIRS["EURJPY"].pip_size == 0.01 and ALL_PAIRS["EURJPY"].price_scale == 1_000
    assert ALL_PAIRS["EURGBP"].pip_size == 0.0001
