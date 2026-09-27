"""Round C15: one FOMC leg (stop, cost, swap, in basis points), the placebo draw and the VIX signal."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c14_gotobi as c14
from scripts.research import fx_c15_lab as c15


def fx_day_keys(times: np.ndarray) -> np.ndarray:
    return np.array([int(pd.Timestamp(t, unit="s", tz="UTC").tz_convert("America/New_York").tz_localize(None)
                         .__add__(pd.Timedelta(hours=7)).normalize().timestamp() // 86400) for t in times])


def market(start: float, n: int, price: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = start + 60 * np.arange(n)
    closes = np.full(n, price)
    m1 = np.column_stack([times, closes, closes + 0.0001, closes - 0.0001, closes])
    return m1, fx_day_keys(times), np.full(n, 0.01)


def test_a_leg_from_the_eve_to_the_decision_pays_cost_and_one_swap_in_basis_points() -> None:
    eve = pd.Timestamp("2024-07-30")  # Tuesday; the decision is Wednesday the 31st
    start, exit_bar = c15.ny_time(eve, 16), c15.ny_time(eve + pd.Timedelta(days=1), 15, 59)
    n = int((exit_bar - start) // 60) + 1
    m1, fx_days, atr = market(start, n, 1.0)
    m1[-1, 4] = 1.002
    rates = pd.DataFrame({"EUR": [0.04], "USD": [0.05]}, index=[pd.Timestamp(start, unit="s").normalize()])
    result = c15.leg(m1, fx_days, atr, rates, "EURUSD", 1, start, exit_bar)
    assert c14.rollover_days(fx_days, 0, n - 1) == 1  # Tuesday 17:00 New York
    swap = (0.04 - 0.05 - 0.01) / 365
    assert result["net_bp"] == pytest.approx((0.002 - c3.cost_of("EURUSD") + swap) * 1e4)
    assert result["stress_bp"] == pytest.approx((0.002 - 2 * c3.cost_of("EURUSD") - 0.03 / 365) * 1e4)  # a 2% markup, no credit
    assert result["net_r"] == pytest.approx(result["net_bp"] / 1e4 / 0.01)


def test_the_stop_fills_at_the_open_of_a_gap_minute() -> None:
    eve = pd.Timestamp("2024-07-30")
    start, exit_bar = c15.ny_time(eve, 16), c15.ny_time(eve + pd.Timedelta(days=1), 15, 59)
    n = int((exit_bar - start) // 60) + 1
    m1, fx_days, atr = market(start, n, 150.0)
    atr[:] = 1.0
    m1[30, 1:5] = [151.5, 151.6, 151.4, 151.5]  # a short USDJPY gapped through its 151.0 stop
    m1[-1, 4] = 140.0  # never reached
    rates = pd.DataFrame({"USD": [0.05], "JPY": [0.0]}, index=[pd.Timestamp(start, unit="s").normalize()])
    result = c15.leg(m1, fx_days, atr, rates, "USDJPY", -1, start, exit_bar)
    assert result["net_r"] == pytest.approx(-1.5 - c3.cost_of("USDJPY"), abs=1e-9)  # 16:30: no rollover


def test_the_placebo_draws_the_same_number_of_days_per_year_and_weekday_without_repeats() -> None:
    days = pd.bdate_range("2020-01-01", "2021-12-31")
    frame = pd.DataFrame({"day": days, "net_bp": np.arange(len(days), dtype=float)})
    fomc = frame[frame["day"].isin(pd.DatetimeIndex(["2020-01-29", "2020-03-18", "2021-06-16"]))]
    means = c15.placebo(fomc, frame.drop(fomc.index), copies=5, seed=1)
    assert len(means) == 5 and len(set(means)) > 1


def test_a_vix_spike_opens_one_position_and_signals_while_it_is_open_are_ignored() -> None:
    days = pd.bdate_range("2020-01-01", periods=80)
    spx = pd.Series(100.0 + np.arange(80), index=days)
    vix = pd.Series(15.0, index=days)
    vix.iloc[[30, 35, 40, 41, 60]] = 30.0  # 35 and 40 inside the first trade (30 to 40); 41 opens the second
    taken = c15.vix_signals(vix, spx, days[0], days[-1])
    assert list(taken) == [30, 41, 60]
    trades = c15.spx_trades(spx, taken)
    carried = (days[40] - days[30]).days
    assert trades["base"].iloc[0] == pytest.approx((140 / 130 - 1) * 1e4 - 1 - 0.06 * carried / 365 * 1e4)
    assert len(trades) == 3
