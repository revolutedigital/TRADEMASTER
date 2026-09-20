"""Tests for the HistData tick pipeline: clock rule, tick parsing, M1 bars and validation."""

import io
import sys
import types
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts.research import fx_histdata_ticks as hd
from scripts.research.fx_dataset import ALL_PAIRS

EURUSD = ALL_PAIRS["EURUSD"]


def utc(clock: str) -> pd.Timestamp:
    """Convert one HistData wall-clock string to UTC with the European switch rule."""
    return hd.clock_to_utc(pd.Series(pd.to_datetime([clock])))[0]


def test_the_european_switch_is_the_last_sunday_of_march_and_october() -> None:
    assert hd.last_sunday(2024, 3) == datetime(2024, 3, 31, 1, tzinfo=UTC)
    assert hd.last_sunday(2024, 10) == datetime(2024, 10, 27, 1, tzinfo=UTC)
    assert hd.last_sunday(2025, 3) == datetime(2025, 3, 30, 1, tzinfo=UTC)
    assert hd.last_sunday(2025, 10) == datetime(2025, 10, 26, 1, tzinfo=UTC)
    assert hd.last_sunday(2021, 12) == datetime(2021, 12, 26, 1, tzinfo=UTC)


def test_the_clock_is_four_hours_behind_utc_in_summer_and_five_in_winter() -> None:
    assert utc("2024-07-10 09:00:00") == pd.Timestamp("2024-07-10 13:00", tz="UTC")
    assert utc("2024-01-10 09:00:00") == pd.Timestamp("2024-01-10 14:00", tz="UTC")


def test_in_the_weeks_where_the_calendars_disagree_the_european_rule_is_followed() -> None:
    # The US is already on summer time on 2024-03-25 but the European switch is a week later.
    assert utc("2024-03-25 09:00:00") == pd.Timestamp("2024-03-25 14:00", tz="UTC")
    assert utc("2024-04-01 09:00:00") == pd.Timestamp("2024-04-01 13:00", tz="UTC")
    # The US is still on summer time on 2024-10-30 but Europe went back on the 27th.
    assert utc("2024-10-30 09:00:00") == pd.Timestamp("2024-10-30 14:00", tz="UTC")
    assert utc("2024-10-24 09:00:00") == pd.Timestamp("2024-10-24 13:00", tz="UTC")


def test_an_empty_clock_series_gives_an_empty_index() -> None:
    assert len(hd.clock_to_utc(pd.Series([], dtype="datetime64[ns]"))) == 0


def _raw(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts", "bid", "ask"])


def test_parse_ticks_handles_timestamps_with_and_without_milliseconds() -> None:
    frame = hd.parse_ticks(
        _raw([("20240710 090000123", 1.1, 1.10002), ("20260710 090001", 1.2, 1.20002)])
    )

    assert frame.index[0] == pd.Timestamp("2024-07-10 13:00:00.123", tz="UTC")
    assert frame.index[1] == pd.Timestamp("2026-07-10 13:00:01", tz="UTC")


def test_parse_ticks_keeps_file_order_when_timestamps_tie() -> None:
    frame = hd.parse_ticks(
        _raw(
            [
                ("20260710 090000", 1.1000, 1.1001),
                ("20260710 090000", 1.1005, 1.1006),
                ("20260710 090000", 1.1002, 1.1003),
            ]
        )
    )

    assert frame["bid"].tolist() == [1.1000, 1.1005, 1.1002]


def test_ticks_become_exact_minute_candles_with_the_spread_in_pips() -> None:
    frame = hd.parse_ticks(
        _raw(
            [
                ("20240710 090000000", 1.10000, 1.10010),
                ("20240710 090020000", 1.10030, 1.10038),
                ("20240710 090040000", 1.09990, 1.10000),
                ("20240710 090100000", 1.10010, 1.10020),
            ]
        )
    )

    bars = hd.ticks_to_m1(frame, EURUSD)

    first = bars.iloc[0]
    assert bars.index[0] == pd.Timestamp("2024-07-10 13:00", tz="UTC")
    assert (first["bid_open"], first["bid_high"], first["bid_low"], first["bid_close"]) == (
        1.10000, 1.10030, 1.09990, 1.09990,
    )
    assert (first["ask_open"], first["ask_high"], first["ask_low"], first["ask_close"]) == (
        1.10010, 1.10038, 1.10000, 1.10000,
    )
    assert first["spread_open_pips"] == pytest.approx(1.0)
    assert len(bars) == 2
    with pytest.raises(ValueError):
        hd.ticks_to_m1(frame.iloc[0:0], EURUSD)


def _oracle_and_m1(shift_hours: int = 0, crossed: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    hours = pd.date_range("2024-07-10 12:00", periods=6, freq="h", tz="UTC")
    oracle = pd.DataFrame({"bid_close": [1.10 + i * 0.0001 for i in range(6)]}, index=hours)
    oracle["ask_close"] = oracle["bid_close"] + 0.0001
    minutes = pd.date_range("2024-07-10 11:59", "2024-07-10 18:00", freq="min", tz="UTC")
    m1 = pd.DataFrame(index=minutes)
    hour_index = (minutes.floor("h") - pd.Timedelta(hours=shift_hours))
    m1["bid_close"] = oracle["bid_close"].reindex(hour_index).to_numpy()
    m1["ask_close"] = oracle["ask_close"].reindex(hour_index).to_numpy()
    m1["bid_open"] = m1["bid_close"]
    m1["ask_open"] = m1["ask_close"] - (0.001 if crossed else 0.0)
    return oracle, m1.dropna()


def test_validation_passes_when_the_derived_bars_reproduce_the_oracle() -> None:
    oracle, m1 = _oracle_and_m1()

    check = hd.validate_against_hourly(m1, oracle, EURUSD.pip_size)

    assert check.passes
    assert check.hours_compared >= 4
    assert check.bid_match == check.ask_match == 1.0


def test_validation_fails_when_the_clock_is_an_hour_off() -> None:
    oracle, m1 = _oracle_and_m1(shift_hours=1)

    check = hd.validate_against_hourly(m1, oracle, EURUSD.pip_size)

    assert not check.passes
    assert check.bid_match < 0.5


def test_validation_flags_crossed_quotes_and_empty_input() -> None:
    oracle, m1 = _oracle_and_m1(crossed=True)

    assert hd.validate_against_hourly(m1, oracle, EURUSD.pip_size).crossed_minutes > 0
    assert not hd.validate_against_hourly(m1, oracle, EURUSD.pip_size).passes
    empty = hd.validate_against_hourly(m1.iloc[0:0], oracle, EURUSD.pip_size)
    assert not empty.passes and empty.hours_compared == 0


def test_read_tick_zip_reads_the_csv_inside_the_archive(tmp_path: Path) -> None:
    payload = "20240710 090000123,1.10000,1.10010,0\n20240710 090001456,1.10001,1.10011,0\n"
    archive = tmp_path / "DAT_ASCII_EURUSD_T_202407.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("DAT_ASCII_EURUSD_T_202407.csv", payload)
        handle.writestr("DAT_ASCII_EURUSD_T_202407.txt", "notes")

    frame = hd.read_tick_zip(archive)

    assert len(frame) == 2 and frame["ask"].iloc[1] == 1.10011


def test_month_range_is_inclusive_and_rejects_reversed_input() -> None:
    assert hd.month_range("2023-11", "2024-02") == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
    with pytest.raises(ValueError):
        hd.month_range("2024-02", "2024-01")


def _fake_histdata(monkeypatch, behaviour) -> list[int]:
    calls: list[int] = []
    module = types.ModuleType("histdata")

    def download_hist_data(**kwargs):
        calls.append(1)
        behaviour(kwargs)

    module.download_hist_data = download_hist_data
    api = types.ModuleType("histdata.api")
    api.Platform = types.SimpleNamespace(GENERIC_ASCII="ascii")
    api.TimeFrame = types.SimpleNamespace(TICK_DATA="tick")
    monkeypatch.setitem(sys.modules, "histdata", module)
    monkeypatch.setitem(sys.modules, "histdata.api", api)
    return calls


def test_download_month_reuses_an_existing_archive(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "DAT_ASCII_EURUSD_T_202407.zip").write_bytes(b"x")
    calls = _fake_histdata(monkeypatch, lambda kwargs: None)

    path = hd.download_month("EURUSD", 2024, 7, tmp_path)

    assert path.exists() and calls == []


def test_download_month_retries_then_gives_up(tmp_path: Path, monkeypatch) -> None:
    def fail(kwargs):
        raise ConnectionError("boom")

    calls = _fake_histdata(monkeypatch, fail)

    with pytest.raises(RuntimeError, match="EURUSD 2024-07"):
        hd.download_month("EURUSD", 2024, 7, tmp_path, retries=3, sleep=lambda _: None)

    assert len(calls) == 3


def test_download_month_succeeds_once_the_file_appears(tmp_path: Path, monkeypatch) -> None:
    attempts = {"n": 0}

    def flaky(kwargs):
        attempts["n"] += 1
        if attempts["n"] == 2:
            Path(kwargs["output_directory"], "DAT_ASCII_EURUSD_T_202407.zip").write_bytes(io.BytesIO(b"z").read())

    _fake_histdata(monkeypatch, flaky)

    path = hd.download_month("EURUSD", 2024, 7, tmp_path, retries=4, sleep=lambda _: None)

    assert path.exists() and attempts["n"] == 2
