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


def test_positions_are_sized_to_the_same_dollars_per_atr_and_skipped_when_the_minimum_is_too_big() -> None:
    # budget US$ 5 per ATR; an index worth US$ 0.74 per ATR per 0.01 lot takes 0.06 lot (US$ 4.44 per ATR)
    assert banca.position_volume(5.0, 74.0, 0.01, 0.01) == pytest.approx(0.06)
    # a pair worth US$ 4.84 per ATR per 1000 units fits once; US$ 9.09 does not fit at all
    assert banca.position_volume(5.0, 0.00484, 1000, 1000) == 1000
    assert banca.position_volume(5.0, 0.00909, 1000, 1000) is None


def test_at_the_target_the_biggest_winners_are_banked_until_the_target_and_losers_never() -> None:
    profits = [("a", 3.0), ("b", 14.0), ("c", -2.0), ("d", 5.0), ("e", 1.0)]

    assert banca.winners_to_bank(profits, 20.0) == ["b", "d", "a"]  # 14 + 5 + 3 = 22 >= 20
    assert banca.winners_to_bank([("x", -1.0)], 20.0) == []


def test_the_trailing_stop_only_tightens_and_stays_behind_the_price() -> None:
    assert banca.tightened_stop(1, best=1.20, atr=0.05, stop=1.10, price=1.19, trail_atr=1.0) == pytest.approx(1.15)
    assert banca.tightened_stop(1, best=1.12, atr=0.05, stop=1.10, price=1.12, trail_atr=1.0) is None  # would loosen
    assert banca.tightened_stop(-1, best=1.00, atr=0.05, stop=1.10, price=1.01, trail_atr=1.0) == pytest.approx(1.05)
    assert banca.tightened_stop(-1, best=1.00, atr=0.05, stop=1.10, price=1.06, trail_atr=1.0) is None  # price above it


def test_gold_silver_and_oil_are_out_of_the_banca() -> None:
    assert banca.EXCLUDED == {"XAUUSD", "XAGUSD", "WTIUSD", "BCOUSD"} and banca.EXCLUDED <= set(banca.FUSION)


def test_the_streak_a_new_trade_sees() -> None:
    assert banca.streak_after([1.0, -1.0, -2.0]) == -2 and banca.streak_after([]) == 0 and banca.streak_after([3.0]) == 1


def _daily(n: int = 260):
    rng = np.random.default_rng(1)
    close = 1.10 + np.cumsum(rng.normal(0, 0.004, n))
    opens = np.concatenate(([close[0]], close[:-1]))
    t = 1_700_000_000 + 86400 * np.arange(n, dtype=float)
    d1 = np.column_stack([t, opens, np.maximum(opens, close) + 0.002, np.minimum(opens, close) - 0.002, close])
    usdx = banca.usdx_from_bars(np.column_stack([t, np.full((n, 3), 100.0) + np.arange(n)[:, None] * 0.01,
                                                 np.full(n, 100.0) + np.arange(n) * 0.01]))
    from scripts.research import fx_c9_lab as c9

    return d1, c9.daily_context(d1[:, 1:], d1[:, 0] + 86400, usdx)


def test_the_exit_row_exists_only_in_profit_and_measures_from_the_entry() -> None:
    d1, daily = _daily()
    last_close = d1[-1, 4]
    entry_time = d1[-5, 0] + 3600  # entered during the fifth bar from the end
    long = {"side": 1, "entry": last_close - 0.01, "unit": 0.005, "entry_time": entry_time, "stop": last_close - 0.03,
            "target": None, "streak": 2.0}
    row = banca.exit_features(long, "TURTLE20", "EURUSD", d1, daily)
    assert row is not None and row["days"] == 5 and row["best"] >= row["profit"] > 0
    assert row["to_stop_atr"] == pytest.approx(0.03 / 0.005) and np.isnan(row["to_target_atr"])
    losing = long | {"entry": last_close + 0.01}
    assert banca.exit_features(losing, "TURTLE20", "EURUSD", d1, daily) is None


class ClosedMarketVenue:
    """Positions 'a' and 'b' are open; 'b' is on a market that refuses orders until `open_all`."""

    def __init__(self) -> None:
        self.open = {"a", "b"}
        self.open_all = False

    async def positions(self):
        return [type("P", (), {"id": i})() for i in self.open]

    async def close(self, position_id: str) -> float:
        if position_id == "b" and not self.open_all:
            raise banca.OrderRejected("Trading is not available: Market is closed")
        self.open.discard(position_id)
        return 0.0


def test_retiring_closes_what_the_market_takes_waits_for_the_rest_and_ends_when_none_is_left() -> None:
    import asyncio

    retiring = banca.Banca.__new__(banca.Banca)
    retiring.venue = ClosedMarketVenue()
    retiring.state = {"positions": {"TSMOM|EURUSD": {"id": "a"}, "TSMOM|GER40": {"id": "b"}}}
    retiring.waiting, logged, exited = set(), [], []
    retiring.log = lambda event, **fields: logged.append(event)

    async def record(key, position, reason):
        exited.append((key, reason))
        retiring.state["positions"].pop(key)

    retiring._record_exit = record
    assert asyncio.run(retiring.retire()) is False
    assert exited == [("TSMOM|EURUSD", "banca desligada")] and logged == ["retire_wait"]
    assert asyncio.run(retiring.retire()) is False and logged == ["retire_wait"]  # the wait is logged once
    retiring.venue.open_all = True
    assert asyncio.run(retiring.retire()) is True
    assert exited[-1] == ("TSMOM|GER40", "banca desligada")
