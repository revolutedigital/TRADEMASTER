"""Round C10: bars aligned to the FX day, causal daily context, and the minute-by-minute day-trade manager."""

import numpy as np
import pandas as pd

from scripts.research import fx_c10_core as core
from scripts.research.fx_c10_families import orb

NAN = np.nan
MORNING = int(pd.Timestamp("2024-07-10 13:00", tz="UTC").timestamp())  # 09:00 New York, eight hours before 17:00


def minutes(prices, start=MORNING):
    """M1 rows (t, o, h, l, c) one minute apart from a list of (open, high, low, close)."""
    return np.array([(start + 60 * i, o, h, l, c) for i, (o, h, l, c) in enumerate(prices)], dtype=float)


def flat(price, count):
    return [(price, price, price, price)] * count


def row(stop_atr=1.0, tp1=0.0, fraction=0.0, runner=0.0, trail=0.0, hold=0.0):
    return np.array([[stop_atr, tp1, fraction, runner, trail, hold]])


def manage(m1, grid, entries=(0,), sides=(1,), stop_refs=(NAN,), atrs=(1.0,), cost=0.0):
    out = core.manage(m1, core.day_end_minutes(m1), np.asarray(entries, dtype=np.int64),
                      np.asarray(sides, dtype=np.int64), np.asarray(stop_refs, dtype=float),
                      np.asarray(atrs, dtype=float), cost, grid, 0)
    return out


def test_the_stop_wins_when_stop_and_target_fit_in_the_same_minute() -> None:
    m1 = minutes([(100, 100.5, 100, 100.2), (100.2, 102.5, 98.5, 100)] + flat(100, 30))
    trade = manage(m1, row(runner=2.0), cost=0.1)[0]
    assert trade[2] == 1
    np.testing.assert_allclose(trade[3:5], [-1.1, -1.2])  # R net of one and of two costs


def test_a_gap_through_the_stop_fills_at_the_open() -> None:
    m1 = minutes([(100, 100.2, 99.8, 100), (98, 98.2, 97.5, 98)] + flat(98, 30))
    np.testing.assert_allclose(manage(m1, row())[0][3], -2.0)


def test_half_at_the_first_target_then_the_rest_leaves_at_break_even() -> None:
    m1 = minutes([(100, 100.2, 99.8, 100), (100, 101.2, 100, 101), (101, 101, 99.9, 100)] + flat(100, 30))
    trade = manage(m1, row(tp1=1.0, fraction=0.5, runner=3.0))[0]
    assert trade[2] == 2
    np.testing.assert_allclose(trade[3], 0.5)


def test_the_runner_banks_the_rest_but_never_in_the_minute_of_the_first_target() -> None:
    m1 = minutes([(100, 100.2, 99.8, 100), (100, 103.5, 100, 103), (103, 103.2, 102.5, 103)] + flat(103, 30))
    trade = manage(m1, row(tp1=1.0, fraction=0.5, runner=3.0))[0]
    assert trade[2] == 2  # minute 1 reaches both levels; the runner only counts from minute 2
    np.testing.assert_allclose(trade[3], 0.5 * 1 + 0.5 * 3)


def test_the_trailing_stop_follows_the_best_price_after_the_first_target() -> None:
    m1 = minutes([(100, 100.2, 99.8, 100), (100, 101.1, 100, 101), (101, 102.5, 101, 102.4),
                  (102.4, 102.4, 101.2, 101.3)] + flat(101.3, 30))
    trade = manage(m1, row(tp1=1.0, fraction=0.5, trail=1.0))[0]
    assert trade[2] == 3
    np.testing.assert_allclose(trade[3], 0.5 * 1 + 0.5 * 1.5)  # the stop of the rest sits at 102.5 - 1


def test_a_short_mirrors_a_long() -> None:
    m1 = minutes([(100, 100.2, 99.8, 100), (100, 100, 98.9, 99), (99, 100.1, 99, 100)] + flat(100, 30))
    np.testing.assert_allclose(manage(m1, row(tp1=1.0, fraction=0.5, runner=3.0), sides=(-1,))[0][3], 0.5)


def test_the_holding_limit_counts_clock_minutes_and_exits_at_the_close() -> None:
    m1 = minutes([(100, 100.2, 99.8, 100.1 + i / 100) for i in range(120)])
    m1[60:, 0] += 30 * 60  # a 30-minute hole in the data: minute 59 is the last one inside the first hour
    trade = manage(m1, row(hold=60.0))[0]
    assert trade[2] == 59
    np.testing.assert_allclose(trade[3], 0.1 + 59 / 100)


def test_every_trade_is_flat_at_the_end_of_the_fx_day_and_late_entries_are_skipped() -> None:
    close = int(pd.Timestamp("2024-07-10 20:40", tz="UTC").timestamp())  # 16:40 New York: 20 minutes before 17:00
    m1 = minutes([(100, 100.2, 99.8, 100 + i / 100) for i in range(40)], start=close)
    trade = manage(m1, row())
    assert len(trade) == 1 and trade[0][2] == 19  # 16:59 is the last minute of the FX day
    assert len(manage(m1, row(), entries=(8,))) == 0  # 16:48: fewer than 15 minutes left


def test_the_structural_stop_comes_from_the_family_and_is_never_closer_than_a_tenth_of_an_atr() -> None:
    m1 = minutes([(100, 100.2, 99.8, 100), (100, 100, 98.9, 99)] + flat(99, 30))
    np.testing.assert_allclose(manage(m1, row(stop_atr=0.0), stop_refs=(99.0,), atrs=(5.0,))[0][3], -1.0)
    dip = minutes([(100, 100.2, 99.8, 100), (100, 100, 99.6, 99.7)] + flat(99.7, 30))
    widened = manage(dip, row(stop_atr=0.0), stop_refs=(99.9,), atrs=(5.0,))[0]
    np.testing.assert_allclose(widened[3], -0.3 / 0.5)  # the stop moves to 99.5 (0.1 of 5); the day ends at 99.7
    assert len(manage(m1, row(stop_atr=0.0))) == 0  # no structural stop, no trade


def test_one_position_at_a_time() -> None:
    m1 = minutes([(100, 100.2, 99.8, 100)] * 5 + [(100, 100, 98.9, 99)] + flat(99, 30))
    out = manage(m1, row(), entries=(0, 3, 6), sides=(1, 1, 1), stop_refs=(NAN,) * 3, atrs=(1.0,) * 3)
    np.testing.assert_array_equal(out[:, 1], [0, 6])


def test_the_management_grid_has_the_registered_144_rows() -> None:
    grid, names = core.management_grid()
    assert grid.shape == (144, 6) and len(set(names)) == 144
    assert {tuple(g) for g in grid[:, [core.STOP_ATR]]} == {(0.0,), (0.25,), (0.5,), (1.0,)}


def test_bars_follow_the_fx_day_that_starts_at_five_pm_new_york() -> None:
    t = np.array([pd.Timestamp(x, tz="UTC").timestamp() for x in
                  ("2024-07-10 20:44", "2024-07-10 20:45", "2024-07-10 20:59", "2024-07-10 21:00")])
    m15, day = core.minute_keys(t, 15), core.minute_keys(t, 1440)
    assert m15[0] != m15[1] == m15[2] != m15[3]
    assert day[0] == day[1] == day[2] != day[3]


def day_of_minutes(start, count, price, step=0.0):
    return [(start + 60 * i, price + step * i, price + step * i + 0.5, price + step * i - 0.5, price + step * i)
            for i in range(count)]


def three_days():
    """Three FX days of minutes from 17:00 New York (21:00 UTC in July), each a flat line at 10, 20 and 30."""
    first = int(pd.Timestamp("2024-07-08 21:00", tz="UTC").timestamp())
    rows = []
    for d, price in enumerate((10.0, 20.0, 30.0)):
        rows += day_of_minutes(first + d * 86400, 1440, price)
    return np.array(rows, dtype=float)


def test_the_daily_context_only_uses_completed_days() -> None:
    m1 = three_days()
    ctx = core.build_context("EURUSD", m1, "H1")
    assert len(ctx.c) == 72
    assert np.isnan(ctx.prev_high[:24]).all()
    np.testing.assert_allclose(ctx.prev_high[24:48], 10.5)
    np.testing.assert_allclose(ctx.prev_close[48:], 20.0)
    np.testing.assert_allclose(ctx.day_open[48:], 30.0)
    changed = m1.copy()
    changed[2 * 1440:, 1:] += 100.0  # rewrite the third day: nothing before it may change
    later = core.build_context("EURUSD", changed, "H1")
    np.testing.assert_allclose(later.prev_high[:48], ctx.prev_high[:48])
    np.testing.assert_allclose(later.c[:48], ctx.c[:48])


def test_orb_trades_the_first_close_beyond_the_opening_range_once_a_day() -> None:
    m1 = three_days()
    m1[:, 2:4] = m1[:, [4, 4]]  # no wicks: the range is the closes
    london = int(pd.Timestamp("2024-07-11 07:00", tz="UTC").timestamp())  # 08:00 London (summer), third FX day
    at = int(np.searchsorted(m1[:, 0], london))
    m1[at:at + 15, 1:] = 30.0 + np.linspace(-0.2, 0.2, 15)[:, None]  # the 15-minute range is 29.8 to 30.2
    m1[at + 20:at + 25, 1:] = 30.5  # the bar of 08:20 closes above it
    m1[at + 40:at + 45, 1:] = 29.0  # a later break below the range is ignored: one signal a day
    ctx = core.build_context("EURUSD", m1, "M5")
    side, stop_ref = orb.signals(ctx, {"range_minutes": 15})
    signal = np.flatnonzero(side)
    assert len(signal) == 1 and side[signal[0]] == 1
    np.testing.assert_allclose(stop_ref[signal[0]], 29.8)
    assert ctx.exchange_minute[signal[0]] == 8 * 60 + 20


def random_walk(days=40, seed=7, start_price=1.1, step=0.0002):
    first = int(pd.Timestamp("2024-03-03 22:00", tz="UTC").timestamp())  # 17:00 New York (winter): an FX day starts
    rng = np.random.default_rng(seed)
    close = start_price + np.cumsum(rng.normal(0, step, days * 1440))
    opens = np.concatenate(([start_price], close[:-1]))
    wick = np.abs(rng.normal(0, step / 2, (2, len(close))))
    return np.column_stack([first + 60 * np.arange(len(close)), opens, np.maximum(opens, close) + wick[0],
                            np.minimum(opens, close) - wick[1], close])


def test_no_family_looks_at_minutes_after_a_bar_closes() -> None:
    from scripts.research.fx_c10_families import FAMILIES

    for symbol, price in (("EURUSD", 1.1), ("SPXUSD", 5000.0)):
        m1 = random_walk(start_price=price, step=price * 0.0002)
        cut = 35 * 1440 + 13 * 60 + 7  # an odd minute in the middle of day 35
        later = m1.copy()
        later[cut:, 1:] = later[cut:, 1:] * 1.03 + np.random.default_rng(1).normal(0, price * 0.001, (len(m1) - cut, 1))
        later[cut:, 2] = later[cut:, 1:5].max(axis=1)
        later[cut:, 3] = later[cut:, 1:5].min(axis=1)
        contexts = {}
        signalled = 0
        for name, module in FAMILIES.items():
            for params in module.PARAMS:
                tf = params["timeframe"]
                if tf not in contexts:
                    contexts[tf] = (core.build_context(symbol, m1, tf), core.build_context(symbol, later, tf))
                before, after = contexts[tf]
                known = before.last_minute < cut
                side_a, stop_a = module.signals(before, params)
                side_b, stop_b = module.signals(after, params)
                np.testing.assert_array_equal(side_a[known], side_b[known], err_msg=f"{symbol} {name} {params}")
                np.testing.assert_array_equal(stop_a[known], stop_b[known], err_msg=f"{symbol} {name} {params}")
                signalled += bool((side_a[known] != 0).any())
        assert signalled >= 25  # the check is not vacuous: most sets trade on the random walk
