"""Causality, streaming parity, fills, gaps, and costs for quote events."""

import numpy as np
import pandas as pd
import pytest

from app.fx.instruments import Instrument
from app.fx.sim.costs import CostScenario
from scripts.research import fx_fast4_events as events

EURUSD = Instrument.from_symbol("EURUSD")
ZERO = CostScenario("zero", spread_multiplier=1.0, slippage_pips=0.0)


def synthetic_ticks(count: int = 1_100, *, start: str = "2021-01-04") -> pd.DataFrame:
    index = pd.date_range(start, periods=count, freq="250ms", tz="UTC")
    steps = np.where(np.arange(count) % 5 < 3, 0.00001, -0.00001)
    mid = 1.10 + np.cumsum(steps)
    spread = 0.00008 + (np.arange(count) % 7) * 0.000001
    return pd.DataFrame({"bid": mid - spread / 2, "ask": mid + spread / 2}, index=index)


def test_future_ticks_cannot_change_a_past_feature() -> None:
    ticks = synthetic_ticks()
    _, first = events.build_feature_frame(ticks, EURUSD)
    decision = first.index[0]
    changed = ticks.copy()
    changed.loc[changed.index > decision, ["bid", "ask"]] += np.linspace(
        0.0, 0.05, (changed.index > decision).sum()
    )[:, None]

    _, second = events.build_feature_frame(changed, EURUSD)
    pd.testing.assert_series_equal(first.loc[decision], second.loc[decision])


def test_batch_features_match_streaming_state() -> None:
    ticks = synthetic_ticks(900)
    _, batch = events.build_feature_frame(ticks, EURUSD)
    state = events.FeatureState(EURUSD)
    streamed: list[tuple[pd.Timestamp, dict[str, float | int]]] = []
    for timestamp, row in ticks.iterrows():
        feature = state.push(timestamp, float(row["bid"]), float(row["ask"]))
        if feature is not None:
            streamed.append((timestamp, feature))

    assert [timestamp for timestamp, _ in streamed] == list(batch.index)
    for timestamp, feature in streamed:
        expected = batch.loc[timestamp].drop(labels="decision_index")
        for name, value in feature.items():
            assert value == pytest.approx(expected[name], abs=1e-10)


def test_cleaning_collapses_exact_duplicates_and_rejects_crossed_quotes() -> None:
    ticks = synthetic_ticks(5)
    duplicate = pd.concat([ticks.iloc[:2], ticks.iloc[[1]], ticks.iloc[2:]]).sort_index(kind="stable")
    assert len(events.clean_ticks(duplicate)) == len(ticks)

    crossed = ticks.copy()
    crossed.iloc[2, crossed.columns.get_loc("ask")] = crossed.iloc[2]["bid"] - 0.00001
    with pytest.raises(ValueError, match="ask cannot be below bid"):
        events.clean_ticks(crossed)


def test_gap_resets_the_event_warmup_and_invalidates_cross_gap_outcome() -> None:
    first = synthetic_ticks(520)
    second = synthetic_ticks(520, start="2021-01-04 00:10:00")
    ticks = pd.concat([first, second])
    cleaned, features = events.build_feature_frame(ticks, EURUSD)

    assert len(features) == 2
    assert features.index[1] >= second.index[events.EVENT_WARMUP - 1]
    outcomes = events.build_outcome_wide(
        cleaned,
        features,
        EURUSD,
        events.EventCosts(0.0, 0.0, ZERO, ZERO),
        horizons_seconds=(600,),
    )
    assert outcomes["h600_long_terminal_r_base"].isna().all()


def test_next_quote_bid_ask_fill_and_terminal_exit() -> None:
    ticks = synthetic_ticks(3_200)
    cleaned, features = events.build_feature_frame(ticks, EURUSD)
    outcomes = events.build_outcome_wide(
        cleaned,
        features.iloc[[0]],
        EURUSD,
        events.EventCosts(0.0, 0.0, ZERO, ZERO),
        horizons_seconds=(30,),
    )
    row = outcomes.iloc[0]
    decision_index = int(features.iloc[0]["decision_index"])
    entry_index = decision_index + 1
    deadline = cleaned.index[decision_index] + pd.Timedelta(seconds=30)
    exit_index = int(cleaned.index.searchsorted(deadline))
    expected_long = (
        cleaned.iloc[exit_index]["bid"] - cleaned.iloc[entry_index]["ask"]
    ) / EURUSD.pip_size
    expected_short = (
        cleaned.iloc[entry_index]["bid"] - cleaned.iloc[exit_index]["ask"]
    ) / EURUSD.pip_size

    assert row["h30_entry_index"] == entry_index
    assert row["h30_exit_index"] == exit_index
    assert row["h30_long_net_pips_base"] == pytest.approx(expected_long)
    assert row["h30_short_net_pips_base"] == pytest.approx(expected_short)
    assert np.isfinite(row["h30_long_mfe_r_base"])
    assert np.isfinite(row["h30_short_mfe_r_base"])


def test_costs_and_stress_are_charged_on_both_fills() -> None:
    ticks = synthetic_ticks(3_200)
    cleaned, features = events.build_feature_frame(ticks, EURUSD)
    base = CostScenario("base", spread_multiplier=1.0, slippage_pips=0.1)
    stress = CostScenario("stress", spread_multiplier=2.0, slippage_pips=0.3)
    outcomes = events.build_outcome_wide(
        cleaned,
        features.iloc[[0]],
        EURUSD,
        events.EventCosts(0.5, 0.5, base, stress),
        horizons_seconds=(30,),
    )
    row = outcomes.iloc[0]

    assert row["h30_long_net_pips_stress"] < row["h30_long_net_pips_base"]
    complete_base_cost = (
        (cleaned.iloc[int(row["h30_entry_index"])]["ask"]
         - cleaned.iloc[int(row["h30_entry_index"])]["bid"]) / EURUSD.pip_size
        + 0.2
        + 0.5
    )
    assert row["h30_risk_pips"] >= 4 * complete_base_cost


def test_directional_orientation_does_not_flip_cost_features() -> None:
    frame = pd.DataFrame({"mid_return_pips_16": [0.4], "spread_pips": [0.8]})
    short = events.directional_features(frame, -1)
    assert short.loc[0, "mid_return_pips_16"] == -0.4
    assert short.loc[0, "spread_pips"] == 0.8
    assert short.loc[0, "side"] == -1
