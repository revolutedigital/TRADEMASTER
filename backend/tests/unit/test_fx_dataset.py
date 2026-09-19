"""Tests for the research-only FX bid/ask dataset loader."""

import lzma
import struct
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from scripts.research.fx_dataset import (
    MAJORS,
    FxDataDownloadError,
    FxInstrument,
    build_pair_dataset,
    decode_hourly_candles,
    fetch_month_payload,
    month_range,
    month_url,
    quality_report,
)

EURUSD = MAJORS["EURUSD"]
USDJPY = MAJORS["USDJPY"]


def _payload(rows: list[tuple[int, int, int, int, int, float]]) -> bytes:
    """Pack (offset_seconds, open, close, low, high, volume) rows like the provider does."""
    return lzma.compress(b"".join(struct.pack(">IIIIIf", *row) for row in rows))


def _hour(index: int) -> int:
    return index * 3600


def test_month_url_uses_the_providers_zero_based_month() -> None:
    assert month_url("EURUSD", 2024, 1, "BID").endswith("/EURUSD/2024/00/BID_candles_hour_1.bi5")
    assert month_url("EURUSD", 2024, 12, "ASK").endswith("/EURUSD/2024/11/ASK_candles_hour_1.bi5")


@pytest.mark.parametrize(("month", "side"), [(0, "BID"), (13, "BID"), (1, "MID")])
def test_month_url_rejects_invalid_arguments(month: int, side: str) -> None:
    with pytest.raises(ValueError):
        month_url("EURUSD", 2024, month, side)


def test_decode_drops_flat_zero_volume_placeholders_and_scales_prices() -> None:
    payload = _payload(
        [
            (_hour(0), 110374, 110374, 110374, 110374, 0.0),  # holiday placeholder
            (_hour(1), 110380, 110402, 110371, 110410, 12.5),
            (_hour(2), 110402, 110390, 110385, 110405, 9.0),
            (_hour(3), 110390, 110390, 110390, 110390, 0.0),  # placeholder
        ]
    )

    frame = decode_hourly_candles(payload, year=2024, month=1, price_scale=100_000)

    assert list(frame.index) == [
        pd.Timestamp("2024-01-01 01:00", tz="UTC"),
        pd.Timestamp("2024-01-01 02:00", tz="UTC"),
    ]
    assert frame.iloc[0].to_dict() == pytest.approx(
        {"open": 1.1038, "high": 1.1041, "low": 1.10371, "close": 1.10402, "volume": 12.5}
    )


def test_decode_keeps_a_flat_candle_that_actually_traded() -> None:
    payload = _payload([(_hour(5), 110390, 110390, 110390, 110390, 3.0)])

    frame = decode_hourly_candles(payload, year=2024, month=1, price_scale=100_000)

    assert len(frame) == 1


def test_decode_uses_the_yen_scale_for_jpy_pairs() -> None:
    payload = _payload([(_hour(1), 148250, 148300, 148200, 148350, 4.0)])

    frame = decode_hourly_candles(
        payload, year=2024, month=2, price_scale=USDJPY.price_scale
    )

    assert frame["close"].iloc[0] == pytest.approx(148.3)


def test_decode_of_an_empty_payload_returns_an_empty_frame() -> None:
    frame = decode_hourly_candles(b"", year=2024, month=1, price_scale=100_000)

    assert frame.empty
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_retries_rate_limits_then_caches_the_payload(tmp_path: Path) -> None:
    statuses = iter([429, 503, 200])
    calls: list[str] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        status = next(statuses)
        return httpx.Response(status, content=b"payload" if status == 200 else b"")

    with _client(handler) as client:
        first = fetch_month_payload(
            client, "EURUSD", 2024, 1, "BID", tmp_path, sleep=sleeps.append
        )
        second = fetch_month_payload(
            client, "EURUSD", 2024, 1, "BID", tmp_path, sleep=sleeps.append
        )

    assert first == second == b"payload"
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]


def test_fetch_treats_not_found_as_an_empty_month_and_remembers_it(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    with _client(handler) as client:
        assert fetch_month_payload(client, "EURUSD", 2030, 1, "ASK", tmp_path) == b""
        assert fetch_month_payload(client, "EURUSD", 2030, 1, "ASK", tmp_path) == b""

    assert calls == 1


def test_fetch_gives_up_after_the_retry_budget(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with _client(handler) as client, pytest.raises(FxDataDownloadError, match="after 3 attempts"):
        fetch_month_payload(
            client, "EURUSD", 2024, 1, "BID", tmp_path, retries=3, sleep=lambda _: None
        )


def test_fetch_does_not_retry_permanent_client_errors(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403)

    with _client(handler) as client, pytest.raises(FxDataDownloadError, match="permanently"):
        fetch_month_payload(client, "EURUSD", 2024, 1, "BID", tmp_path, sleep=lambda _: None)

    assert calls == 1


def test_build_pair_dataset_joins_bid_and_ask_and_measures_spread_in_pips(
    tmp_path: Path,
) -> None:
    bid = _payload([(_hour(1), 148250, 148300, 148200, 148350, 4.0)])
    ask = _payload([(_hour(1), 148262, 148314, 148212, 148362, 4.0)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=bid if "BID_" in request.url.path else ask)

    with _client(handler) as client:
        frame = build_pair_dataset(
            client, USDJPY, [(2024, 2)], tmp_path, politeness_seconds=0, sleep=lambda _: None
        )

    assert frame["spread_open_pips"].iloc[0] == pytest.approx(1.2)
    assert frame["spread_close_pips"].iloc[0] == pytest.approx(1.4)


def _week_of_session_candles(sunday_open_hour: int = 22) -> pd.DataFrame:
    """One realistic week: Sunday 22:00 UTC through Friday 21:00 UTC, all hours present."""
    index = pd.date_range(
        pd.Timestamp("2024-01-07", tz="UTC") + pd.Timedelta(hours=sunday_open_hour),
        pd.Timestamp("2024-01-12 21:00", tz="UTC"),
        freq="h",
    )
    rng = np.random.default_rng(7)
    mid = 1.10 + np.cumsum(rng.normal(0, 0.0002, len(index)))
    frame = pd.DataFrame(index=index)
    frame["bid_open"] = mid
    frame["bid_close"] = mid + rng.normal(0, 0.0001, len(index))
    frame["bid_high"] = frame[["bid_open", "bid_close"]].max(axis=1) + 0.0001
    frame["bid_low"] = frame[["bid_open", "bid_close"]].min(axis=1) - 0.0001
    frame["ask_open"] = frame["bid_open"] + 0.00008
    frame["ask_close"] = frame["bid_close"] + 0.00008
    frame["spread_open_pips"] = 0.8
    frame["spread_close_pips"] = 0.8
    return frame


def test_quality_report_accepts_half_a_year_of_clean_sessions() -> None:
    # Density is only meaningful over months: a short window ending on a Friday
    # leaves out the trailing weekend and overstates it.
    frame = pd.concat(
        [_week_of_session_candles().shift(week * 7 * 24, freq="h") for week in range(26)]
    )

    report = quality_report(frame, EURUSD)

    assert report["problems"] == []
    assert report["density"] == pytest.approx(5 / 7, abs=0.01)
    assert report["saturday_candles"] == 0
    assert report["unexplained_gaps"] == 0
    assert report["spread_median_pips"] == pytest.approx(0.8)


def test_quality_report_flags_saturday_candles_and_crossed_quotes() -> None:
    frame = _week_of_session_candles()
    saturday = pd.Timestamp("2024-01-13 10:00", tz="UTC")
    frame.loc[saturday] = frame.iloc[-1]
    frame = frame.sort_index()
    frame.iloc[5, frame.columns.get_loc("ask_open")] = frame.iloc[5]["bid_open"] - 0.0002

    report = quality_report(frame, EURUSD)

    joined = " | ".join(report["problems"])
    assert "Saturday" in joined
    assert "ask below bid" in joined


def test_quality_report_catches_a_wrong_price_scale() -> None:
    frame = _week_of_session_candles()
    for column in ("bid_open", "bid_close", "bid_high", "bid_low", "ask_open", "ask_close"):
        frame[column] = frame[column] * 100  # e.g. a JPY pair decoded with the wrong scale

    report = quality_report(frame, EURUSD)

    assert any("implausible" in problem for problem in report["problems"])


def test_quality_report_reports_gaps_that_are_not_the_weekend() -> None:
    frame = _week_of_session_candles()
    frame = frame.drop(frame.index[40:46])  # six missing hours mid-week

    report = quality_report(frame, EURUSD)

    assert report["unexplained_gaps"] == 1
    assert report["longest_gap_hours"] == 7.0


def test_month_range_is_inclusive_and_crosses_year_boundaries() -> None:
    assert month_range("2023-11", "2024-02") == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
    with pytest.raises(ValueError):
        month_range("2024-02", "2024-01")


def test_every_major_has_a_pip_of_ten_price_units() -> None:
    """The provider quotes a tenth of a pip; a pip is 10 units for every major."""
    for instrument in MAJORS.values():
        assert isinstance(instrument, FxInstrument)
        assert instrument.pip_size * instrument.price_scale == pytest.approx(10)
    assert set(MAJORS) == {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"}


def test_fetch_backoff_doubles_but_never_exceeds_the_cap(tmp_path: Path) -> None:
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with _client(handler) as client, pytest.raises(FxDataDownloadError):
        fetch_month_payload(
            client,
            "EURUSD",
            2024,
            1,
            "BID",
            tmp_path,
            retries=7,
            base_delay_seconds=2.0,
            max_delay_seconds=10.0,
            sleep=sleeps.append,
        )

    assert sleeps == [2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]
