"""Round C3: indicators, bars, the minute-by-minute simulator and causality of every strategy."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_c3_lab as c3

NAN = np.nan


def minutes(prices, start=1_700_000_000, spread_high=0.0, spread_low=0.0):
    """M1 rows (t, o, h, l, c) from a list of (open, high, low, close)."""
    rows = [(start + 60 * i, o, h, l, c) for i, (o, h, l, c) in enumerate(prices)]
    return np.array(rows, dtype=float)


def run(m1, last, enter, *, exit_long=None, exit_short=None, stop_level=None, stop_distance=None, target_r=NAN,
        trail_k=NAN, trail_atr=None, max_bars=0, unit=None, bar_high=None, bar_low=None):
    n = len(last)
    zeros = np.zeros(n, dtype=bool)
    nan = np.full(n, NAN)
    return c3.simulate(m1, np.asarray(last, dtype=np.int64),
                       bar_high if bar_high is not None else np.full(n, 0.0),
                       bar_low if bar_low is not None else np.full(n, 0.0),
                       np.asarray(enter, dtype=np.int8), exit_long if exit_long is not None else zeros,
                       exit_short if exit_short is not None else zeros,
                       stop_level if stop_level is not None else nan, stop_distance if stop_distance is not None else nan,
                       target_r, trail_k, trail_atr if trail_atr is not None else nan, max_bars,
                       unit if unit is not None else np.ones(n), float(c3.SEGMENT_GAP_SECONDS))


def test_rsi_and_atr_use_wilder_smoothing() -> None:
    c = np.array([10.0, 11.0, 10.0, 12.0, 11.0])
    # deltas +1 -1 +2 -1; first averages over 2: gain 0.5, loss 0.5 -> 50; then gain (0.5+2)/2=1.25, loss 0.25 -> 83.33
    np.testing.assert_allclose(c3.rsi(c, 2)[2:], [50.0, 100 - 100 / (1 + 1.25 / 0.25), 100 - 100 / (1 + 0.625 / 0.625)])
    h, l = np.array([2.0, 3.0, 4.0]), np.array([1.0, 1.0, 2.0])
    np.testing.assert_allclose(c3.atr(h * 0, h, l, np.array([1.5, 2.0, 3.0]), 2)[1:], [1.5, (1.5 + 2.0) / 2])


def test_channels_exclude_the_current_bar() -> None:
    h = np.array([1.0, 2.0, 3.0, 10.0])
    np.testing.assert_allclose(c3.prior_max(h, 2), [NAN, NAN, 2.0, 3.0])


def test_bars_follow_the_new_york_fx_day_and_four_hour_blocks() -> None:
    # 2024-07-10 20:59 and 21:00 UTC are 16:59 and 17:00 New York (summer): a new FX day and a new H4 block
    t = np.array([pd.Timestamp("2024-07-10 20:59", tz="UTC").timestamp(), pd.Timestamp("2024-07-10 21:00", tz="UTC").timestamp(),
                  pd.Timestamp("2024-07-11 00:59", tz="UTC").timestamp(), pd.Timestamp("2024-07-11 01:00", tz="UTC").timestamp()])
    d1, h4 = c3.bucket_keys(t, "D1"), c3.bucket_keys(t, "H4")
    assert d1[0] != d1[1] and d1[1] == d1[2] == d1[3]
    assert h4[0] != h4[1] and h4[1] == h4[2] != h4[3]
    m1 = np.column_stack([t, [1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5], [0.5, 1.5, 2.5, 3.5], [1.2, 2.2, 3.2, 4.2]])
    last, bars = c3.aggregate(m1, h4)
    np.testing.assert_array_equal(last, [0, 2, 3])
    np.testing.assert_allclose(bars[1], [2, 3.5, 1.5, 3.2])


def test_entry_fills_at_the_next_open_and_a_stop_gap_fills_at_the_open() -> None:
    m1 = minutes([(1.0, 1.0, 1.0, 1.0), (1.1, 1.1, 1.1, 1.1), (1.1, 1.2, 1.1, 1.2), (0.8, 0.8, 0.7, 0.7)])
    trades = run(m1, [0, 1, 2, 3], [1, 0, 0, 0], stop_distance=np.full(4, 0.2))
    side, entry_i, exit_i, entry, exit_, unit, reason = trades[0]
    assert (side, entry_i, entry) == (1, 1, 1.1)
    assert (exit_i, exit_, reason) == (3, 0.8, c3.REASON_STOP)  # the stop at 0.9 was gapped through: filled at 0.8


def test_target_at_a_multiple_of_the_stop_and_stop_first_when_both_fit() -> None:
    m1 = minutes([(1.0,) * 4, (1.0, 1.0, 1.0, 1.0), (1.0, 1.16, 1.0, 1.1), (1.0,) * 4])
    trades = run(m1, [0, 1, 2, 3], [1, 0, 0, 0], stop_level=np.array([0.9, NAN, NAN, NAN]), target_r=1.5)
    assert trades[0][4] == pytest.approx(1.15) and trades[0][6] == c3.REASON_TARGET
    both = minutes([(1.0,) * 4, (1.0,) * 4, (1.0, 1.2, 0.8, 1.0), (1.0,) * 4])
    trades = run(both, [0, 1, 2, 3], [1, 0, 0, 0], stop_level=np.array([0.9, NAN, NAN, NAN]), target_r=1.5)
    assert trades[0][6] == c3.REASON_STOP


def test_an_entry_whose_stop_is_already_behind_the_fill_is_refused() -> None:
    m1 = minutes([(1.0,) * 4, (0.8,) * 4, (0.8,) * 4])
    assert len(run(m1, [0, 1, 2], [1, 0, 0], stop_level=np.array([0.9, NAN, NAN]))) == 0


def test_an_opposite_signal_reverses_at_the_same_open() -> None:
    m1 = minutes([(1.0,) * 4, (1.1,) * 4, (1.2,) * 4, (1.3,) * 4])
    trades = run(m1, [0, 1, 2, 3], [1, -1, 0, 0])
    assert trades[0][0] == 1 and trades[0][2] == 2 and trades[0][6] == c3.REASON_RULE
    assert trades[1][0] == -1 and trades[1][1] == 2 and trades[1][6] == c3.REASON_END


def test_time_exit_and_the_chandelier_stop_only_tightens() -> None:
    m1 = minutes([(1.0,) * 4] + [(1.0 + 0.1 * k,) * 4 for k in range(1, 6)])
    trades = run(m1, list(range(6)), [1, 0, 0, 0, 0, 0], max_bars=3)
    assert trades[0][1] == 1 and trades[0][2] == 4  # 3 bar closes after the entry (bars 1, 2, 3), exit at the next open
    high = np.array([1.0, 1.5, 1.2, 1.2, 1.2, 1.2])
    falling = minutes([(1.0,) * 4, (1.0,) * 4, (1.4,) * 4, (1.29, 1.3, 1.29, 1.3), (1.0,) * 4, (1.0,) * 4])
    trades = run(falling, list(range(6)), [1, 0, 0, 0, 0, 0], trail_k=1.0, trail_atr=np.full(6, 0.2), bar_high=high,
                 bar_low=high * 0)
    # stop 1.5 - 0.2 = 1.3 set at bar 1, not loosened to 1.0 by bar 2, gapped through by the 1.29 open of bar 3
    assert trades[0][6] == c3.REASON_STOP and trades[0][2] == 3 and trades[0][4] == pytest.approx(1.29)


@pytest.mark.parametrize("key", list(c3.STRATEGIES))
def test_no_strategy_decision_changes_when_the_future_changes(key) -> None:
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 1, 700))
    high, low = close + rng.uniform(0, 1, 700), close - rng.uniform(0, 1, 700)
    opens = np.concatenate(([close[0]], close[:-1]))
    build = c3.STRATEGIES[key][1]
    full = build(opens, high, low, close)
    cut = 500
    early = build(opens[:cut], high[:cut], low[:cut], close[:cut])
    for field in ("enter", "exit_long", "exit_short", "stop_level", "stop_distance", "unit"):
        np.testing.assert_array_equal(getattr(full, field)[:cut], getattr(early, field))


def test_the_placebo_keeps_every_minute_range_and_flips_the_moves() -> None:
    m1 = minutes([(1.0, 1.1, 0.9, 1.05), (1.05, 1.2, 1.0, 1.1)])
    flipped = c3.sign_flip(m1, np.array([-1, 1], dtype=np.int8))
    assert flipped[0, 4] == pytest.approx(0.95) and flipped[0, 2] - flipped[0, 3] == pytest.approx(0.2)
    assert flipped[1, 4] - flipped[1, 1] == pytest.approx(0.05)


def test_costs_cover_every_instrument_of_both_groups() -> None:
    for symbol in c3.GROUP_A + c3.GROUP_B:
        assert c3.cost_of(symbol) > 0
    assert c3.cost_of("USDJPY") == pytest.approx(0.009) and c3.cost_of("EURUSD") == pytest.approx(0.00009)


def test_a_histdata_file_is_read_as_utc_from_its_new_york_clock(tmp_path) -> None:
    import zipfile

    path = tmp_path / "DAT_ASCII_EURUSD_M1_2019.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("DAT_ASCII_EURUSD_M1_2019.csv",
                         "20190701 120000;1.1;1.2;1.0;1.15;0\n20190320 120000;1;1;1;1;0\n20160320 120000;1;1;1;1;0\n")
    frame = c3.read_zip(path)

    # July: summer time (UTC-4); 20 March 2019: Europe still in winter time (UTC-5); 20 March 2016: US summer (UTC-4)
    assert frame["t"].tolist() == [pd.Timestamp(s, tz="UTC").timestamp()
                                   for s in ("2019-07-01 16:00", "2019-03-20 17:00", "2016-03-20 16:00")]
    assert frame[["o", "h", "l", "c"]].iloc[0].tolist() == [1.1, 1.2, 1.0, 1.15]


def test_the_manifest_keeps_only_the_pre_registered_months(tmp_path) -> None:
    months = pd.date_range("2011-12-01", "2026-09-30", freq="MS")
    t = np.concatenate([np.arange(m.timestamp(), m.timestamp() + 60 * 20_000, 60) for m in months])
    np.save(tmp_path / "EURUSD.npy", np.column_stack([t, np.ones((t.size, 4))]))

    table = c3.build_manifest(("EURUSD",), tmp_path)

    included = table.loc[table["included"], "month"]
    assert included.min() == "2012-01" and included.max() == "2026-08" and table["symbol_included"].all()
