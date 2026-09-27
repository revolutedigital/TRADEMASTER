"""C13 front A's live runner: the raw entry detector sees exactly the same signal bars and feature values as the
research trade_rows/add_recent, the FX-day end lands on 17:00 New York, and the live M1 store trims a bar still
forming."""

import json

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


def test_an_entry_after_a_data_hole_is_not_valid() -> None:
    """manage() refuses an entry minute more than 5 minutes after the signal bar; pending_entries must mark it so, or
    it would sit as the earliest 'fresh' candidate and hide the valid ones after it."""
    m1 = random_walk(days=40)
    pending = live.pending_entries("EURUSD", m1)
    holed = m1.copy()
    target = pending.iloc[len(pending) // 2]
    k = int(np.searchsorted(m1[:, 0], target["entry_time"]))
    holed[k:, 0] += 600  # the entry minute now comes 11 minutes after the signal bar's last minute
    again = live.pending_entries("EURUSD", holed)
    moved = again[(again["set_id"] == target["set_id"]) & (again["entry_time"] == target["entry_time"] + 600)]
    assert len(moved) == 1 and not bool(moved["valid"].iloc[0])
    assert pending["valid"].mean() > 0.95  # a clean series: only entries across the 17:00 rollover are refused


def test_leaves_room_is_managements_minutes_left_rule() -> None:
    end = pd.Timestamp("2026-01-15 22:00", tz="UTC").timestamp()  # 17:00 New York
    assert live.leaves_room(end - 14 * 60)
    assert not live.leaves_room(end - 13 * 60)


def test_pending_since_keeps_only_the_recent_entries() -> None:
    m1 = random_walk(days=40)
    since = m1[-1, 0] - 2 * 86400
    everything, recent = live.pending_entries("EURUSD", m1), live.pending_entries("EURUSD", m1, since=since)
    assert len(recent) and (recent["entry_time"] >= since).all()
    assert len(recent) == int((everything["entry_time"] >= since).sum())


def test_the_cycle_runs_65_seconds_after_each_five_minute_mark() -> None:
    mark = pd.Timestamp("2026-09-28 14:05", tz="UTC").timestamp()
    assert live.next_cycle(mark) == mark + 65
    assert live.next_cycle(mark + 65) == mark + 365
    assert live.next_cycle(mark - 1) == mark + 65  # 14:04:59 -> 14:06:05 (14:01:05 has passed)
    assert live.next_cycle(mark + 64) == mark + 65


def test_open_risk_adds_the_stops_of_open_positions() -> None:
    assert live.open_risk({}) == 0.0
    assert live.open_risk({"a": {"risk_usd": 9.5}, "b": {"risk_usd": 6.0}, "c": {}}) == pytest.approx(15.5)


class FakeCandles:
    """Serves one bar per minute on weekdays and nothing on Saturday, like the demo's CLI."""

    def __init__(self) -> None:
        self.calls = 0

    def send(self, command: str, timeout: float = 30.0) -> str:
        self.calls += 1
        _, _, _, frm, to = command.split()
        start, end = pd.Timestamp(frm).timestamp(), pd.Timestamp(to).timestamp()
        bars = [{"timestamp": pd.Timestamp(t, unit="s", tz="UTC").isoformat().replace("+00:00", "Z"), "open": 1.0,
                 "high": 1.0, "low": 1.0, "close": 1.0}
                for t in np.arange(start, end + 1, 60) if pd.Timestamp(t, unit="s").weekday() != 5]
        return json.dumps({"bars": bars})


def test_fetch_since_walks_chunks_across_an_empty_weekend() -> None:
    session = FakeCandles()
    last = pd.Timestamp("2026-09-25 20:00", tz="UTC").timestamp()  # Friday
    now = pd.Timestamp("2026-09-27 01:00:30", tz="UTC").timestamp()  # Sunday
    bars = live.fetch_since(session, "EURUSD", last, now)
    friday = pd.Timestamp("2026-09-26", tz="UTC").timestamp()
    assert bars[0, 0] == last + 60 and (np.diff(bars[:, 0]) > 0).all()
    assert (bars[:, 0] < friday).sum() == 239  # 20:01 to 23:59
    assert bars[-1, 0] == pd.Timestamp("2026-09-27 00:59", tz="UTC").timestamp()  # the last closed minute
    assert session.calls <= 4


def test_symbol_universe_matches_the_manifest() -> None:
    universe = live.symbol_universe()
    assert 20 <= len(universe) <= 45
    assert len(universe) == len(set(universe))
