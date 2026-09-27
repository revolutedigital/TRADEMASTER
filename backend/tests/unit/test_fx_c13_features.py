"""The new features of the banca study: the dollar side of a trade and the counts at the same decision time."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_c13_features as feats


def test_the_dollar_side_of_a_trade() -> None:
    symbols = pd.Series(["EURUSD", "USDJPY", "EURJPY", "UDXUSD", "SPXUSD", "GRXEUR"])
    sides = pd.Series([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
    # long EURUSD is short the dollar; long USDJPY long it; a cross has none; the dollar index sold is short it;
    # an index quoted in dollars is not a bet on the dollar
    np.testing.assert_array_equal(feats.usd_side(symbols, sides), [-1.0, 1.0, 0.0, -1.0, 0.0, 0.0])


def test_counts_and_breadth_at_the_same_decision_time() -> None:
    frame = pd.DataFrame({"decision_time": [0, 0, 0, 5], "symbol": ["EURUSD", "GBPUSD", "EURJPY", "EURUSD"],
                          "set_id": [1, 1, 2, 1], "side": [1.0, 1.0, -1.0, -1.0]})
    out = feats.add_features(frame)
    assert out["burst_all"].tolist() == [3, 3, 3, 1] and out["burst_set"].tolist() == [2, 2, 1, 1]
    assert out["usd_breadth"].tolist()[:3] == [-1.0, -1.0, -1.0]  # the instant's dollar, for a cross too
    assert out["usd_aligned"].tolist()[:2] == [1.0, 1.0] and np.isnan(out["usd_aligned"].iloc[2])
    assert out["usd_count"].tolist() == [2, 2, 2, 1]


def random_walk(days, seed=5, step=0.0002):
    first = int(pd.Timestamp("2024-03-03 22:00", tz="UTC").timestamp())  # 17:00 New York: an FX day starts
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(0, step, days * 1440))
    opens = np.concatenate(([1.1], close[:-1]))
    wick = np.abs(rng.normal(0, step / 2, (2, len(close))))
    return np.column_stack([first + 60 * np.arange(len(close)), opens, np.maximum(opens, close) + wick[0],
                            np.minimum(opens, close) - wick[1], close])


def test_the_intraday_features_never_read_past_their_minute() -> None:
    """Computed on the data cut right after the minute, every value is the one computed on the whole series."""
    m1 = random_walk(days=40)
    at = np.array([35 * 1440 + 17, 36 * 1440 + 600, 38 * 1440 + 1300])
    whole = feats.intraday_features(m1, at)
    for k, i in enumerate(at):
        cut = feats.intraday_features(m1[: i + 1], np.array([i]))
        for name, values in whole.items():
            assert np.isclose(cut[name][0], values[k], equal_nan=True), (name, i)
    assert all(np.isfinite(v).all() for v in whole.values())  # 40 days are enough history for every one


def test_sigma_left_shrinks_toward_the_exit() -> None:
    m1 = random_walk(days=40)
    day, slot = feats._slots(m1)
    at = [int(np.flatnonzero((day == 35) & (slot == minute))[0]) for minute in (60, 720, 1380)]  # by the NY clock
    left = feats.intraday_features(m1, np.array(at))["sigma_left"]
    assert left[0] > left[1] > left[2] > 0


def test_vix_features_use_the_last_close_before_the_new_york_date(tmp_path) -> None:
    days = pd.bdate_range("2024-01-01", periods=30)
    closes = np.arange(10.0, 40.0)
    path = tmp_path / "vix.csv"
    pd.DataFrame({"DATE": days.strftime("%m/%d/%Y"), "CLOSE": closes}).to_csv(path, index=False)
    at = pd.Timestamp(f"{days[25].date()} 18:00", tz="America/New_York").timestamp()  # after that day's close
    frame = feats.add_vix(pd.DataFrame({"decision_time": [at, days[0].timestamp()]}), path)
    assert frame["vix_level"].iloc[0] == closes[24]  # the day before, never the same day
    assert frame["vix_ratio"].iloc[0] == pytest.approx(closes[24] / closes[4:24].mean())
    assert frame["vix_change"].iloc[0] == pytest.approx(np.log(closes[24] / closes[19]))
    assert np.isnan(frame["vix_level"].iloc[1])  # nothing known before the first close
