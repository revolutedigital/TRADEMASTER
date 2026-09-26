"""Round C13, front A: the banca's features only read what is known at the signal bar, the recent results only count
trades already closed, the walk-forward never trains on a trade that exits inside the year it decides, and the
pre-registered statistics and choice."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c10_core as core
from scripts.research import fx_c13_banca as banca


def random_walk(days, seed=7, start_price=1.1, step=0.0002):
    first = int(pd.Timestamp("2024-03-03 22:00", tz="UTC").timestamp())  # 17:00 New York (winter): an FX day starts
    rng = np.random.default_rng(seed)
    close = start_price + np.cumsum(rng.normal(0, step, days * 1440))
    opens = np.concatenate(([start_price], close[:-1]))
    wick = np.abs(rng.normal(0, step / 2, (2, len(close))))
    return np.column_stack([first + 60 * np.arange(len(close)), opens, np.maximum(opens, close) + wick[0],
                            np.minimum(opens, close) - wick[1], close])


def test_the_single_management_is_the_registered_c10_row() -> None:
    grid, g = banca.management_row()
    assert core.management_grid()[1][g] == "stop 1 ATR · alvo 3R · até o fim do dia"
    np.testing.assert_array_equal(grid[g], [1.0, 0.0, 0.0, 3.0, 0.0, 0.0])


def test_no_feature_looks_past_the_signal_bar() -> None:
    m1 = random_walk(days=120)
    cut = 110 * 1440 + 9 * 60 + 7  # an odd minute of day 110: ATR(100) of the days is known from day 101 on
    later = m1.copy()
    later[cut:, 1:] = later[cut:, 1:] * 1.03 + np.random.default_rng(1).normal(0, 0.002, (len(m1) - cut, 1))
    later[cut:, 2] = later[cut:, 1:5].max(axis=1)
    later[cut:, 3] = later[cut:, 1:5].min(axis=1)
    before, after = banca.trade_rows("EURUSD", m1), banca.trade_rows("EURUSD", later)
    columns = ["entry_time", "fx_day"] + [f for f in banca.FEATURES if f not in banca.RECENT]
    known = [frame.loc[frame["entry_time"] <= m1[cut, 0], columns].sort_values(["set_id", "entry_time"])
             .reset_index(drop=True) for frame in (before, after)]  # entered at or before the first changed minute
    pd.testing.assert_frame_equal(known[0], known[1])
    assert len(known[0]) >= 300 and known[0]["set_id"].nunique() >= 20  # the check is not vacuous
    assert known[0]["atr_ratio"].notna().sum() >= 20 and known[0]["trend50"].notna().any()
    assert not before.loc[before["entry_time"] > m1[cut, 0] + 86400, "net"].equals(
        after.loc[after["entry_time"] > m1[cut, 0] + 86400, "net"])  # and the change does reach later trades
    assert set(before["side"]) == {-1.0, 1.0}


def test_features_are_the_signal_bar_values_with_the_side_on_the_directional_ones() -> None:
    m1 = random_walk(days=70)
    frame = banca.trade_rows("EURUSD", m1)
    ctx = core.build_context("EURUSD", m1, "M15")
    raw = pd.DataFrame(banca.bar_features(ctx, c3.cost_of("EURUSD")))
    rows = frame[frame["timeframe_minutes"] == 15]
    entry = np.searchsorted(m1[:, 0], rows["entry_time"].to_numpy())
    bar = raw.iloc[np.searchsorted(ctx.last_minute, entry - 1)].reset_index(drop=True)
    side = rows["side"].to_numpy()
    assert set(side) == {-1.0, 1.0}
    for column in raw.columns:
        wanted = bar[column].to_numpy() * (side if column in banca.DIRECTIONAL else 1)
        np.testing.assert_allclose(rows[column].to_numpy(), wanted, err_msg=column)
    features = frame[[f for f in banca.FEATURES if f not in banca.RECENT]].to_numpy(dtype=float)
    assert not np.isinf(features).any()


def test_recent_results_only_count_trades_closed_strictly_before_the_entry() -> None:
    frame = pd.DataFrame([
        ("A", 0, 0, 100, 1.0),
        ("A", 0, 100, 200, 2.0),  # the first exits in the minute this one enters: not known yet
        ("A", 0, 201, 300, 4.0),
        ("B", 0, 150, 160, 10.0),
        ("A", 1, 500, 600, 7.0),  # another set: nothing of set 0 counts
    ], columns=["symbol", "set_id", "decision_time", "exit_time", "net"])
    out = banca.add_recent(frame)
    np.testing.assert_allclose(out["recent_set_market"], [np.nan, np.nan, 1.5, np.nan, np.nan])
    np.testing.assert_allclose(out["recent_set_all"], [np.nan, np.nan, 13 / 3, 1.0, np.nan])


def test_recent_results_are_cut_at_the_signal_bar_close_not_at_a_delayed_entry() -> None:
    # A's M5 signal bar closes at 600; a data hole delays its entry to 720; B's exit minute 600 ends at 660
    frame = pd.DataFrame([("B", 0, 300, 300, 600, -1.0), ("A", 0, 600, 720, 900, 1.0)],
                         columns=["symbol", "set_id", "decision_time", "entry_time", "exit_time", "net"])
    assert np.isnan(banca.add_recent(frame)["recent_set_all"].iloc[1])


def test_recent_results_keep_the_last_20_of_the_market_and_200_of_the_set() -> None:
    i = np.arange(25)
    frame = pd.DataFrame({"symbol": "C", "set_id": 2, "decision_time": 10 * i, "exit_time": 10 * i + 5, "net": i * 1.0})
    out = banca.add_recent(frame)
    assert out["recent_set_market"].iloc[24] == np.mean(np.arange(4, 24))
    assert out["recent_set_all"].iloc[24] == np.mean(np.arange(24))


def _year_frame():
    rows = []
    for year in range(2012, 2027):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp()
        for k in range(4):  # the last one of each year exits in the next year
            entry = start + (k + 1) * 80 * 86400
            rows.append((entry, entry + (120 if k == 3 else 1) * 86400))
    frame = pd.DataFrame(rows, columns=["entry_time", "exit_time"])
    rng = np.random.default_rng(3)
    for feature in banca.FEATURES:
        frame[feature] = rng.normal(size=len(frame))
    frame["net"] = np.arange(len(frame), dtype=float)  # the label names the row: the fake model sees who trained
    return frame


def test_the_walk_forward_trains_only_on_trades_that_exited_before_the_year(monkeypatch) -> None:
    frame = _year_frame()
    seen = []

    class Fake:
        def fit(self, x, y):
            seen.append(y.astype(int))
            return self

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr(banca, "make_model", lambda depth, weight: Fake())
    predictions = banca.walk_forward(frame)
    assert len(seen) == len(banca.YEARS) * len(banca.GRID)
    for year, labels in zip(np.repeat(list(banca.YEARS), len(banca.GRID)), seen, strict=True):
        start = pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp()
        assert (frame["exit_time"].to_numpy()[labels] < start).all()
        assert len(labels) == int((frame["exit_time"] < start).sum())  # and every such trade
    entries = pd.to_datetime(frame["entry_time"].to_numpy()[predictions["row"]], unit="s")
    np.testing.assert_array_equal(entries.year, predictions["year"])
    assert set(predictions["year"]) == set(banca.YEARS)
    assert (frame["entry_time"].to_numpy()[predictions["row"]] < banca.SAMPLE_END).all()
    assert {banca.config_name(d, w) for d, w in banca.GRID} <= set(predictions.columns)


def test_the_split_stops_at_the_end_of_the_sample() -> None:
    frame = pd.DataFrame({"entry_time": [pd.Timestamp(x, tz="UTC").timestamp() for x in
                                         ("2025-12-31 23:59", "2026-01-01", "2026-08-31 23:59", "2026-09-01")]})
    frame["exit_time"] = frame["entry_time"] + 60
    train, test = banca.split(frame, 2026)
    np.testing.assert_array_equal(train, [False, False, False, False])  # the first exits at 2026-01-01 00:00
    np.testing.assert_array_equal(test, [False, True, True, False])


def test_the_real_model_fits_the_walk_forward() -> None:
    frame = _year_frame()
    predictions = banca.walk_forward(frame, grid=((3, 1),), years=(2020,))
    assert len(predictions) == 4 and np.isfinite(predictions["d3_w1"]).all()


def test_score_clusters_by_fx_day_and_counts_markets_with_20_trades() -> None:
    rng = np.random.default_rng(5)
    a = pd.DataFrame({"symbol": "A", "net": rng.normal(0.5, 1, 20), "fx_day": np.arange(20) % 5})
    b = pd.DataFrame({"symbol": "B", "net": rng.normal(-0.5, 0.1, 20), "fx_day": np.arange(20) % 7})
    c = pd.DataFrame({"symbol": "C", "net": np.full(5, 3.0), "fx_day": np.arange(5)})
    frame = pd.concat([a, b, c], ignore_index=True)
    frame["stress"] = frame["net"] - 0.1
    out = banca.score(frame, days=10)
    mean = frame["net"].mean()
    daily = frame.groupby("fx_day")["net"].agg(["sum", "size"])
    t = mean / (np.sqrt(((daily["sum"] - mean * daily["size"]) ** 2).sum()) / len(frame))
    assert out["trades"] == 45 and out["trades_per_day"] == 4.5
    np.testing.assert_allclose([out["mean_net"], out["mean_stress"], out["t"]], [mean, mean - 0.1, t])
    assert out["win_rate"] == pytest.approx((frame["net"] > 0).mean())
    assert out["market_share"] == 0.5 and out["markets"] == 2  # C has only 5 trades
    assert banca.score(frame.iloc[:0])["p_one_sided"] == 1.0
    assert banca.score(frame[frame["symbol"] == "C"])["market_share"] == 0.0


def test_the_choice_is_the_best_mean_with_3000_accepted_and_all_criteria_pass() -> None:
    base = {"max_depth": 3, "min_child_weight": 200, "threshold": 0.0, "mean_stress": 0.05, "p_one_sided": 0.001,
            "market_share": 0.6}
    results = pd.DataFrame([base | {"trades": 2999, "mean_net": 1.0},
                            base | {"trades": 5000, "mean_net": 0.1},
                            base | {"trades": 4000, "mean_net": 0.2, "mean_stress": -0.01, "threshold": 0.05}])
    chosen, passes = banca.choose(results)
    assert chosen["mean_net"] == 0.2 and chosen["threshold"] == 0.05 and not passes
    results.loc[2, "mean_stress"] = 0.01
    assert banca.choose(results)[1]
    results.loc[2, "p_one_sided"] = 0.0063  # above 0.05 / 8
    assert not banca.choose(results)[1]
    assert banca.choose(results.iloc[:1]) == (None, False)


def test_the_report_writes_with_and_without_a_chosen_combination(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(banca, "REPORT", tmp_path / "report.md")
    frame = pd.DataFrame({"set_id": [0, 1], "symbol": ["EURUSD", "SPXUSD"]})
    results = pd.DataFrame([{"max_depth": 3, "min_child_weight": 200, "threshold": 0.0, "trades": 3000,
                             "mean_net": 0.01, "mean_stress": -0.01, "p_one_sided": 0.2, "market_share": 0.5}])
    chosen = results.iloc[0].to_dict()
    by_year = pd.DataFrame({"trades": [10], "mean_net": [0.01]}, index=pd.Index([2016], name="year"))
    banca.write_report(frame, results, chosen, False, by_year, {"trades": 2}, "abc", "sha", {"worst_drawdown_r": 3.0})
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Falhou em" in text and "Em dinheiro" in text and "sha256: sha" in text
    banca.write_report(frame, results, None, False, pd.DataFrame(), {"trades": 2}, "abc", None, {})
    assert "Nenhuma combinação" in (tmp_path / "report.md").read_text(encoding="utf-8")
