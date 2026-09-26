"""Round C14: the gotobi calendar, the entry and exit minutes, the swap days and one leg's result."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_c14_gotobi as c14


def ts(text: str) -> float:
    return pd.Timestamp(text, tz="UTC").timestamp()


def test_gotobi_days_move_to_the_previous_tokyo_business_day() -> None:
    holidays = c14.japanese_holidays()
    days = set(c14.gotobi_days("2024-04-20", "2024-06-30", holidays).strftime("%Y-%m-%d"))
    assert "2024-05-02" in days  # 5 May is a Sunday and a holiday, 3-4 May holidays too: Thursday the 2nd
    assert "2024-06-28" in days and "2024-06-30" not in days  # the 30th is a Sunday
    assert "2024-05-31" not in days  # the 31st never counts
    february = c14.gotobi_days("2024-02-01", "2024-02-29", holidays)
    assert pd.Timestamp("2024-02-29") in february  # a leap February: the 29th stands for the 30th
    new_year = c14.gotobi_days("2024-12-20", "2025-01-10", holidays)
    assert list(new_year.strftime("%Y-%m-%d")) == ["2024-12-20", "2024-12-25", "2024-12-30", "2025-01-10"]  # the 5th
    # (a Sunday) falls back past 1-3 January to the 30th, counted once


def test_the_entry_waits_an_hour_after_a_weekend_reopening_and_the_exit_needs_a_near_bar() -> None:
    friday = ts("2024-07-12 20:00") + 60 * np.arange(59)
    sunday = ts("2024-07-14 21:05") + 60 * np.arange(300)
    times = np.concatenate([friday, sunday]).astype(float)
    assert c14.entry_minute(times, ts("2024-07-12 20:10")) == 10
    e = c14.entry_minute(times, ts("2024-07-14 18:00"))  # Monday 03:00 JST: the market is shut
    assert times[e] == ts("2024-07-14 22:05")
    assert c14.exit_minute(times, ts("2024-07-14 21:30")) is not None
    assert c14.exit_minute(times, ts("2024-07-13 12:00")) is None  # nothing within 5 minutes


def test_wednesday_rollover_counts_three_days_of_swap() -> None:
    tuesday = int(pd.Timestamp("2024-07-16").timestamp() // 86400)
    fx_days = np.array([tuesday, tuesday + 1, tuesday + 2])  # FX days ending Tuesday the 16th, Wednesday, Thursday
    assert c14.rollover_days(fx_days, 0, 1) == 1 and c14.rollover_days(fx_days, 1, 2) == 3


def test_a_leg_pays_cost_earns_swap_and_the_stop_comes_first() -> None:
    start = ts("2024-07-16 18:00")  # Wednesday 03:00 JST = Tuesday 14:00 New York: one Tuesday rollover inside
    n = 420
    times = start + 60 * np.arange(n)
    price = np.full(n, 150.0)
    m1 = np.column_stack([times, price, price + 0.05, price - 0.05, price])
    m1[-1, 4] = 150.5
    fx_days = np.array([int(pd.Timestamp(t, unit="s", tz="UTC").tz_convert("America/New_York").tz_localize(None)
                            .__add__(pd.Timedelta(hours=7)).normalize().timestamp() // 86400) for t in times])
    atr = np.full(n, 1.0)
    day = pd.Timestamp(start, unit="s").normalize()
    rates = pd.DataFrame({"USD": [0.05], "JPY": [0.0]}, index=[day])
    result = c14.leg_result(m1, fx_days, atr, rates, "USDJPY", 1, start, times[-1], 0.009)
    days = c14.rollover_days(fx_days, 0, n - 1)
    assert days == 1  # Tuesday 17:00 New York
    swap = (0.05 - 0.01) * days / 365 * 150.0
    assert result["net"] == pytest.approx(0.5 - 0.009 + swap)
    assert result["stress"] == pytest.approx(0.5 - 0.018)  # the stress takes no swap credit
    m1[100, 3] = 148.9  # a dip through the 1-ATR stop at 149
    assert c14.leg_result(m1, fx_days, atr, rates, "USDJPY", 1, start, times[-1], 0.009)["net"] == pytest.approx(-1.009)


def test_placebo_days_keep_the_monthly_count() -> None:
    holidays = c14.japanese_holidays()
    real = c14.gotobi_days("2023-01-01", "2023-12-31", holidays)
    business = c14.tokyo_business_days("2023-01-01", "2023-12-31", holidays)
    fake = c14.placebo_days(real, business, 7)
    assert (fake.to_period("M").value_counts().sort_index() == real.to_period("M").value_counts().sort_index()).all()
    assert fake.isin(business).all()
