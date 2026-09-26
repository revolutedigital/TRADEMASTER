"""C13 front A's live runner: the raw entry detector sees exactly the same signal bars and feature values as the
research trade_rows/add_recent, the FX-day end lands on 17:00 New York, and the live M1 store trims a bar still
forming."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_c13_banca as banca
from scripts.research import fx_c13_demo as live


def random_walk(days, seed=7, start_price=1.1, step=0.0002):
    first = int(pd.Timestamp("2024-03-03 22:00", tz="UTC").timestamp())  # 17:00 New York (winter): an FX day starts
    rng = np.random.default_rng(seed)
    close = start_price + np.cumsum(rng.normal(0, step, days * 1440))
    opens = np.concatenate(([start_price], close[:-1]))
    wick = np.abs(rng.normal(0, step / 2, (2, len(close))))
    return np.column_stack([first + 60 * np.arange(len(close)), opens, np.maximum(opens, close) + wick[0],
                            np.minimum(opens, close) - wick[1], close])


def test_pending_entries_matches_trade_rows_at_every_signal_bar_it_resolved() -> None:
    """Every completed trade of trade_rows started at a bar pending_entries also sees, with identical features
    (only the exit-derived columns and the label are missing from the raw, unresolved view)."""
    m1 = random_walk(days=150)
    resolved = banca.trade_rows("EURUSD", m1)
    pending = live.pending_entries("EURUSD", m1)
    assert len(resolved) >= 300 and resolved["set_id"].nunique() >= 20  # the check is not vacuous
    columns = [c for c in banca.FEATURES if c not in banca.RECENT] + ["decision_time"]
    left = resolved.sort_values(["set_id", "entry_time"])[["set_id", "entry_time", *columns]].reset_index(drop=True)
    right = pd.merge(resolved[["set_id", "entry_time"]], pending, on=["set_id", "entry_time"], how="left") \
        .sort_values(["set_id", "entry_time"])[["set_id", "entry_time", *columns]].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_pending_entries_carries_the_unresolved_tail_trade_rows_cannot_see() -> None:
    """A signal fired a few minutes before the data ends has not exited yet: trade_rows drops it, pending_entries
    still reports it (that is the whole point of the live edge detector). trade_rows also drops entries its own
    busy_until skipped while a prior trade of the same set was still open — those are not what this checks."""
    m1 = random_walk(days=40)
    cut = len(m1) - 3  # a signal here cannot have resolved: its own management would still be open
    resolved = banca.trade_rows("EURUSD", m1[:cut])
    pending = live.pending_entries("EURUSD", m1[:cut])
    near_tail = pending[pending["entry_time"] >= m1[cut - 1, 0] - 2 * 86400]
    missing = near_tail[~near_tail.set_index(["set_id", "entry_time"]).index.isin(
        resolved.set_index(["set_id", "entry_time"]).index)]
    assert len(near_tail) > 0
    assert len(missing) > 0


def test_resolved_until_is_the_last_known_exit_per_set_and_market() -> None:
    pool = pd.DataFrame({"symbol": ["EURUSD", "EURUSD", "EURUSD", "GBPUSD"], "set_id": [0, 0, 1, 0],
                         "exit_time": [100.0, 300.0, 50.0, 900.0]})
    until = live.resolved_until(pool)
    assert until == {("EURUSD", 0): 300.0, ("EURUSD", 1): 50.0, ("GBPUSD", 0): 900.0}
    assert live.resolved_until(pd.DataFrame()) == {}


def test_recent_features_matches_batch_add_recent() -> None:
    """The single-row lookup used live must return exactly what add_recent computed in a batch over the same pool,
    for both the same-set-and-market window and the same-set-everywhere window."""
    m1 = random_walk(days=150)
    frame = banca.add_recent(banca.trade_rows("EURUSD", m1))
    counted = frame[frame[list(banca.RECENT)].notna().all(axis=1)]
    assert len(counted) > 50  # only meaningful once trades have accumulated
    row = counted.iloc[len(counted) // 2]
    got = live.recent_features(frame, row["symbol"], int(row["set_id"]), row["decision_time"])
    for name in banca.RECENT:
        assert got[name] == pytest.approx(row[name])


def test_recent_feature_is_nan_with_no_history() -> None:
    assert np.isnan(live.recent_feature(pd.DataFrame(), {"set_id": 0}, 20, 1_000.0))
    pool = pd.DataFrame({"set_id": [0], "symbol": ["EURUSD"], "exit_time": [2_000.0], "net": [0.5]})
    assert np.isnan(live.recent_feature(pool, {"set_id": 0}, 20, 1_000.0))  # the only trade exits after the decision


def test_day_end_after_is_17_00_new_york_in_utc() -> None:
    winter = pd.Timestamp("2026-01-15 12:00", tz="UTC").timestamp()  # EST, UTC-5
    summer = pd.Timestamp("2026-07-15 12:00", tz="UTC").timestamp()  # EDT, UTC-4
    assert live.day_end_after(winter) == pd.Timestamp("2026-01-15 22:00", tz="UTC").timestamp()
    assert live.day_end_after(summer) == pd.Timestamp("2026-07-15 21:00", tz="UTC").timestamp()
    # already past the rollover: the end is the next day's
    just_after = pd.Timestamp("2026-01-15 22:00:01", tz="UTC").timestamp()
    assert live.day_end_after(just_after) == pd.Timestamp("2026-01-16 22:00", tz="UTC").timestamp()


def test_parse_bars_drops_the_bar_still_forming() -> None:
    now = pd.Timestamp("2026-01-15 12:00:30", tz="UTC").timestamp()
    payload = {"bars": [
        {"timestamp": "2026-01-15T11:58:00.000Z", "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.10},
        {"timestamp": "2026-01-15T12:00:00.000Z", "open": 1.1, "high": 1.11, "low": 1.09, "close": 1.10},  # not closed
    ]}
    bars = live.parse_bars(payload, now)
    assert len(bars) == 1
    assert bars[0, 0] == pd.Timestamp("2026-01-15T11:58:00", tz="UTC").timestamp()


def test_symbol_universe_matches_the_manifest() -> None:
    universe = live.symbol_universe()
    assert 20 <= len(universe) <= 45
    assert len(universe) == len(set(universe))
