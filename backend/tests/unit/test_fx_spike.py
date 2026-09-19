"""Tests for the G0 spike simulator, resampler, and statistics."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from app.services.backtest.fx_costs import FxCostModel, NoCommission, SwapSchedule
from scripts.research.fx_spike import (
    STRATEGIES,
    Scenario,
    cluster_bootstrap_mean,
    group_stats,
    mid_ohlc,
    profit_factor,
    resample_session,
    simulate,
    trades_frame,
    wilder_atr,
)

FREE = Scenario("free", FxCostModel(NoCommission()))
ONE_PIP = 0.0001


def _bars(prices: list[tuple[float, float, float, float]], *, spread_pips: float = 0.0,
          start: str = "2024-03-04 22:00") -> pd.DataFrame:
    """Session bars from (open, high, low, close) mids; bid/ask are mid +/- half the spread."""
    index = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    half = spread_pips * ONE_PIP / 2
    frame = pd.DataFrame(index=index)
    frame["bid_open"] = [p[0] - half for p in prices]
    frame["ask_open"] = [p[0] + half for p in prices]
    frame["bid_high"] = [p[1] - half for p in prices]
    frame["ask_high"] = [p[1] + half for p in prices]
    frame["bid_low"] = [p[2] - half for p in prices]
    frame["ask_low"] = [p[2] + half for p in prices]
    frame["bid_close"] = [p[3] - half for p in prices]
    frame["ask_close"] = [p[3] + half for p in prices]
    frame["open_spread_pips"] = spread_pips
    return frame


def _flat(n: int, price: float = 1.1000) -> list[tuple[float, float, float, float]]:
    return [(price, price + 0.0010, price - 0.0010, price)] * n


def _run(prices, signals: dict[int, float], *, atr: float = 0.0100, scenario: Scenario = FREE,
         warmup: int = 0, start: str = "2024-03-04 22:00"):
    bars = _bars(prices, start=start, spread_pips=0.0 if scenario is FREE else 1.0)
    signal_series = pd.Series(0.0, index=bars.index)
    for position, value in signals.items():
        signal_series.iloc[position] = value
    atr_series = pd.Series(atr, index=bars.index)
    return simulate(bars, signal_series, atr_series, symbol="EURUSD", scenario=scenario,
                    warmup=warmup)


def test_a_signal_is_acted_on_at_the_next_bars_open_never_the_same_bar() -> None:
    prices = _flat(3) + [(1.1050, 1.1060, 1.1040, 1.1050)] + _flat(2, 1.1050)

    trades = _run(prices, {2: 1.0})

    assert trades[0].entry_time == pd.Timestamp("2024-03-07 22:00", tz="UTC").to_pydatetime()
    assert trades[0].entry_price == pytest.approx(1.1050)  # bar 3's open, not bar 2's close


def test_the_stop_is_two_atr_from_the_fill_and_the_target_is_two_r() -> None:
    # Long from 1.1000 with ATR 0.0100: stop 1.0800, target 1.1400.
    prices = _flat(2) + [(1.1000, 1.1450, 1.0950, 1.1400)]

    trades = _run(prices, {0: 1.0})

    trade = trades[0]
    assert trade.exit_reason == "take_profit"
    assert trade.gross_pips == pytest.approx(400.0)
    assert trade.r_multiple == pytest.approx(2.0)


def test_a_stop_hit_loses_exactly_one_r_when_there_are_no_costs() -> None:
    prices = _flat(2) + [(1.1000, 1.1010, 1.0790, 1.0800)]

    trade = _run(prices, {0: 1.0})[0]

    assert trade.exit_reason == "stop"
    assert trade.r_multiple == pytest.approx(-1.0)


def test_a_bar_touching_both_stop_and_target_counts_as_a_stop() -> None:
    prices = _flat(2) + [(1.1000, 1.1450, 1.0790, 1.1100)]

    assert _run(prices, {0: 1.0})[0].exit_reason == "stop"


def test_a_short_wins_when_price_falls_to_its_target() -> None:
    prices = _flat(2) + [(1.1000, 1.1050, 1.0590, 1.0600)]

    trade = _run(prices, {0: -1.0})[0]

    assert trade.side == "SHORT"
    assert trade.exit_reason == "take_profit"
    assert trade.r_multiple == pytest.approx(2.0)


def test_a_sunday_gap_through_the_stop_fills_at_the_gapped_open() -> None:
    # Friday bar, then a bar that opens 300 pips below the 1.0800 stop.
    prices = _flat(2) + [(1.1000, 1.1010, 1.0990, 1.1000), (1.0500, 1.0510, 1.0450, 1.0480)]

    trade = _run(prices, {0: 1.0})[0]

    assert trade.exit_reason == "stop_gap"
    assert trade.exit_price == pytest.approx(1.0500)
    assert trade.r_multiple == pytest.approx(-500 / 200)  # a 2.5R loss on a 1R stop


def test_an_opposite_signal_reverses_the_position() -> None:
    prices = _flat(2) + [(1.1000, 1.1020, 1.0990, 1.1010)] + [(1.1010, 1.1020, 1.0990, 1.1000)] * 2

    trades = _run(prices, {0: 1.0, 2: -1.0})

    assert [t.side for t in trades] == ["LONG", "SHORT"]
    assert trades[0].exit_reason == "signal"
    assert trades[0].exit_time == trades[1].entry_time


def test_the_same_direction_signal_does_not_pyramid_or_restart_the_trade() -> None:
    prices = _flat(2) + [(1.1000, 1.1020, 1.0990, 1.1010)] * 4

    trades = _run(prices, {0: 1.0, 2: 1.0})

    assert len(trades) == 1 and trades[0].exit_reason == "end"


def test_no_trade_is_opened_during_the_warmup() -> None:
    prices = _flat(6)

    assert _run(prices, {1: 1.0}, warmup=3) == []


def test_a_market_exit_pays_the_full_spread_and_slippage_on_both_sides() -> None:
    prices = _flat(6)
    costly = Scenario("costly", FxCostModel(NoCommission(), slippage_pips=0.2))

    free_trade = _run(prices, {0: 1.0, 3: -1.0})[0]
    costly_trade = _run(prices, {0: 1.0, 3: -1.0}, scenario=costly)[0]

    assert free_trade.exit_reason == costly_trade.exit_reason == "signal"
    # Buy the ask and sell the bid on a flat market: the 1 pip spread plus 0.2 pip slippage
    # each way.
    assert free_trade.net_pips - costly_trade.net_pips == pytest.approx(1.0 + 0.4, abs=1e-6)


def test_a_target_exit_earns_the_planned_distance_because_costs_move_the_levels_too() -> None:
    """Stop and target are placed from the fill price, so entry costs shift both levels.

    The spread then shows up as a lower hit rate, not a smaller win, which is why the
    edge estimate must come from the whole trade distribution and not from one winner.
    """
    prices = _flat(2) + [(1.1000, 1.1450, 1.0950, 1.1400)]
    costly = Scenario("costly", FxCostModel(NoCommission(), slippage_pips=0.2))

    free_trade = _run(prices, {0: 1.0})[0]
    costly_trade = _run(prices, {0: 1.0}, scenario=costly)[0]

    assert free_trade.net_pips == pytest.approx(costly_trade.net_pips)
    assert costly_trade.entry_price > free_trade.entry_price


def test_a_stop_exit_pays_slippage_beyond_the_stop_level() -> None:
    prices = _flat(2) + [(1.1000, 1.1010, 1.0790, 1.0800)]
    costly = Scenario("costly", FxCostModel(NoCommission(), slippage_pips=0.5))

    trade = _run(prices, {0: 1.0}, scenario=costly)[0]

    assert trade.exit_reason == "stop"
    assert trade.gross_pips == pytest.approx(-200.0 - 0.5)


def test_financing_is_charged_per_rollover_crossed() -> None:
    prices = _flat(6)
    with_swap = Scenario(
        "swap", FxCostModel(NoCommission()), SwapSchedule(-0.5, -0.5)
    )

    # Enter at the Monday-2024-03-04 22:00 UTC bar's next open and hold to the end (6 bars).
    trades = _run(prices, {0: 1.0}, scenario=with_swap)

    trade = trades[0]
    assert trade.swap_pips < 0
    assert trade.swap_pips == pytest.approx(-0.5 * 6, abs=3.5)  # roughly one charge per night
    assert trade.net_pips == pytest.approx(
        trade.gross_pips - trade.commission_pips + trade.swap_pips
    )


def test_mismatched_inputs_are_rejected() -> None:
    bars = _bars(_flat(4))

    with pytest.raises(ValueError, match="aligned"):
        simulate(bars, pd.Series(0.0, index=bars.index[:3]), pd.Series(0.01, index=bars.index),
                 symbol="EURUSD", scenario=FREE)


def _h1_week(start_utc: str, hours: int, *, spread: float = 0.6) -> pd.DataFrame:
    index = pd.date_range(start_utc, periods=hours, freq="h", tz="UTC")
    mid = 1.10 + np.arange(hours) * 0.00001
    frame = pd.DataFrame(index=index)
    for side, sign in (("bid", -1), ("ask", 1)):
        offset = sign * spread * ONE_PIP / 2
        frame[f"{side}_open"] = mid + offset
        frame[f"{side}_high"] = mid + 0.0002 + offset
        frame[f"{side}_low"] = mid - 0.0002 + offset
        frame[f"{side}_close"] = mid + offset
    frame["spread_open_pips"] = spread
    return frame


def test_resampling_anchors_daily_bars_at_the_new_york_close_in_winter_and_summer() -> None:
    winter = resample_session(_h1_week("2024-01-07 22:00", 24 * 3), "1D")
    summer = resample_session(_h1_week("2024-07-07 21:00", 24 * 3), "1D")

    assert winter.index[0] == pd.Timestamp("2024-01-07 22:00", tz="UTC")  # 17:00 EST
    assert summer.index[0] == pd.Timestamp("2024-07-07 21:00", tz="UTC")  # 17:00 EDT
    assert (winter["candles"].iloc[:2] == 24).all()


def test_resampling_takes_the_extremes_and_first_last_prices() -> None:
    h1 = _h1_week("2024-01-07 22:00", 24)

    bars = resample_session(h1, "1D")

    assert len(bars) == 1
    assert bars["bid_open"].iloc[0] == h1["bid_open"].iloc[0]
    assert bars["bid_close"].iloc[0] == h1["bid_close"].iloc[-1]
    assert bars["ask_high"].iloc[0] == h1["ask_high"].max()
    assert bars["bid_low"].iloc[0] == h1["bid_low"].min()


def test_the_entry_spread_ignores_the_wide_rollover_hour() -> None:
    h1 = _h1_week("2024-01-07 22:00", 24)
    h1.iloc[0, h1.columns.get_loc("spread_open_pips")] = 9.0  # the 17:00 New York candle

    bars = resample_session(h1, "1D")

    assert bars["open_spread_pips"].iloc[0] == pytest.approx(0.6)


def test_four_hour_bars_start_on_the_new_york_session_grid() -> None:
    bars = resample_session(_h1_week("2024-01-07 22:00", 24), "4h")

    starts_ny = bars.index.tz_convert("America/New_York").hour.tolist()
    assert starts_ny == [17, 21, 1, 5, 9, 13]


def test_resample_rejects_unknown_rules_and_empty_input() -> None:
    with pytest.raises(ValueError):
        resample_session(_h1_week("2024-01-07 22:00", 24), "1h")
    with pytest.raises(ValueError):
        resample_session(_h1_week("2024-01-07 22:00", 24).iloc[0:0], "1D")


def test_wilder_atr_of_constant_true_range_equals_that_range() -> None:
    bars = _bars([(1.10, 1.11, 1.09, 1.10)] * 40)

    atr = wilder_atr(mid_ohlc(bars))

    assert atr.iloc[-1] == pytest.approx(0.02, rel=1e-6)
    assert atr.iloc[:13].isna().all()


def test_profit_factor_handles_no_losses_and_no_trades() -> None:
    assert profit_factor(np.array([1.0, 2.0])) == float("inf")
    assert profit_factor(np.array([-1.0, -2.0])) == 0.0
    assert profit_factor(np.array([2.0, -1.0])) == pytest.approx(2.0)


def test_bootstrap_separates_a_consistent_edge_from_zero_and_rarely_flags_pure_noise() -> None:
    months = np.repeat([f"m{m:02d}" for m in range(36)], 10)
    rng = np.random.default_rng(1)

    edge_detected = 0
    noise_flagged = 0
    replications = 150
    for replication in range(replications):
        edge = rng.normal(0.4, 1.0, months.size)
        noise = rng.normal(0.0, 1.0, months.size)
        _, edge_low, _, _ = cluster_bootstrap_mean(edge, months, draws=400, seed=replication)
        _, noise_low, _, noise_high = cluster_bootstrap_mean(
            noise, months, draws=400, seed=replication
        )
        edge_detected += edge_low > 0
        noise_flagged += (noise_low > 0) or (noise_high < 0)

    assert edge_detected / replications > 0.95
    # Nominal two-sided false-alarm rate is 5%; percentile bootstraps are a little
    # anti-conservative, so allow up to 12% before calling the interval broken.
    assert noise_flagged / replications < 0.12


def test_the_adjusted_bound_is_never_above_the_nominal_bound() -> None:
    months = np.repeat([f"2024-{m:02d}" for m in range(1, 13)], 8)
    values = np.random.default_rng(3).normal(0.2, 1.0, months.size)

    _, nominal, adjusted, _ = cluster_bootstrap_mean(
        values, months, draws=2000, adjusted_alpha=0.025 / 12
    )

    assert adjusted <= nominal


def test_bootstrap_is_reproducible_and_rejects_empty_input() -> None:
    months = np.array(["2024-01"] * 5 + ["2024-02"] * 5 + ["2024-03"] * 5)
    values = np.linspace(-1, 2, 15)

    assert cluster_bootstrap_mean(values, months, draws=500) == cluster_bootstrap_mean(
        values, months, draws=500
    )
    with pytest.raises(ValueError):
        cluster_bootstrap_mean(np.array([]), np.array([]))


def test_group_stats_reports_power_so_a_null_result_is_not_mistaken_for_no_edge() -> None:
    prices = _flat(2) + [(1.1000, 1.1450, 1.0950, 1.1400)]
    trades = _run(prices, {0: 1.0})
    frame = trades_frame(trades * 30)
    frame["month"] = [f"2024-{(i % 6) + 1:02d}" for i in range(len(frame))]

    stats = group_stats(frame, adjusted_alpha=0.025 / 12)

    assert stats.trades == 30
    assert stats.mean_r == pytest.approx(2.0)
    assert stats.interval is not None
    assert stats.minimum_detectable_mean_r is not None


def test_the_strategy_grid_is_declared_up_front_and_builds_valid_configs() -> None:
    assert len(STRATEGIES) == 6
    for spec in STRATEGIES:
        config = spec.to_config()
        assert "volume_confirmation" not in config.indicators
        assert config.min_confirmations == 1


def test_simulated_times_are_timezone_aware_utc() -> None:
    prices = _flat(2) + [(1.1000, 1.1450, 1.0950, 1.1400)]

    trade = _run(prices, {0: 1.0})[0]

    assert trade.entry_time.tzinfo is not None
    assert trade.exit_time.utcoffset() == datetime(2024, 1, 1, tzinfo=UTC).utcoffset()
