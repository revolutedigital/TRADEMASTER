"""The banca's live pieces: only closed bars, the backtest's stops, and recent results closed before now."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_banca_demo as banca
from scripts.research import fx_c3_lab as c3


def test_the_forming_bar_is_dropped() -> None:
    bars = [{"timestamp": "2026-09-25T12:00:00.000Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
            {"timestamp": "2026-09-25T13:00:00.000Z", "open": 1.5, "high": 2, "low": 1, "close": 1.8}]
    now = pd.Timestamp("2026-09-25T13:30:00Z").timestamp()

    closed = banca.closed_bars(bars, 3600, now)

    assert closed.shape == (1, 5) and closed[0, 4] == 1.5


def test_stops_follow_the_rule_and_a_rule_without_stop_gets_the_disaster_stop() -> None:
    n = 3
    nan = np.full(n, np.nan)
    unit = np.full(n, 0.01)
    level = c3._plan(n, stop_level=np.array([np.nan, np.nan, 1.09]), target_r=1.5, unit=unit)  # noqa: SLF001
    distance = c3._plan(n, stop_distance=np.full(n, 0.02), unit=unit)  # noqa: SLF001
    none = c3._plan(n, stop_level=nan, unit=unit)  # noqa: SLF001

    assert banca.protective_levels(1, 1.10, level, 2) == (1.09, pytest.approx(1.115))
    assert banca.protective_levels(-1, 1.10, distance, 2) == (pytest.approx(1.12), None)
    assert banca.protective_levels(1, 1.10, none, 2)[0] == pytest.approx(1.05)


def test_recent_results_use_only_trades_closed_before_now() -> None:
    history = pd.DataFrame({"config": ["A", "A", "A"], "symbol": ["X", "X", "Y"], "entry_time": [0.0, 5.0, 6.0],
                            "exit_time": [10.0, 20.0, 15.0], "net": [1.0, 3.0, -1.0]})

    same, every = banca.recent_features(history, "A", "X", now=16.0)

    assert same == pytest.approx(1.0) and every == pytest.approx(0.0)  # at 16: X closed 1.0; all closed 1.0 and -1.0


def test_every_instrument_maps_to_a_fusion_symbol() -> None:
    assert set(banca.FUSION) == set(c3.GROUP_A + c3.GROUP_B)
    assert banca.FUSION["SPXUSD"] == "US500" and banca.FUSION["EURUSD"] == "EURUSD" and banca.FUSION["UDXUSD"] == "USDX"
