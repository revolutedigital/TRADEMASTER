"""The clock check of the S0 manifest: a week that opens an hour off is caught without any oracle."""

import pandas as pd

from scripts.research import fx_fast2_manifest as manifest


def _week_opening_at(utc_open: str) -> pd.DataFrame:
    index = pd.date_range(utc_open, periods=180, freq="min", tz="UTC")
    return pd.DataFrame({"bid_open": 1.1}, index=index)


def test_a_week_that_opens_at_17_new_york_time_passes_in_summer_and_winter() -> None:
    assert manifest.weekly_open_ok(_week_opening_at("2017-07-09 21:00"))  # Sunday 17:00 EDT
    assert manifest.weekly_open_ok(_week_opening_at("2017-01-08 22:00"))  # Sunday 17:00 EST


def test_a_clock_an_hour_off_is_caught() -> None:
    assert not manifest.weekly_open_ok(_week_opening_at("2017-07-09 22:00"))  # opens at 18:00 New York: read an hour late
    assert not manifest.weekly_open_ok(_week_opening_at("2017-07-09 20:00"))  # opens at 16:00: read an hour early


def test_a_frame_without_a_sunday_says_nothing() -> None:
    frame = pd.DataFrame({"bid_open": 1.1}, index=pd.date_range("2017-07-11 10:00", periods=60, freq="min", tz="UTC"))

    assert manifest.weekly_open_ok(frame)


def test_a_partial_sunday_at_the_utc_month_boundary_is_not_mistaken_for_the_weekly_open() -> None:
    # August 2011 starts at 00:00 UTC on a Monday, which is still Sunday 20:00 in New York. The real
    # 17:00 open belongs to July in UTC and is outside this monthly slice, so the partial Sunday says
    # nothing about the feed's clock.
    frame = pd.DataFrame(
        {"bid_open": 1.1},
        index=pd.date_range("2011-08-01 00:00", periods=60, freq="min", tz="UTC"),
    )

    assert manifest.weekly_open_ok(frame)


def test_a_late_first_quote_on_a_holiday_sunday_is_not_a_clock_error() -> None:
    assert manifest.weekly_open_ok(_week_opening_at("2016-12-25 22:30"))  # 17:30 New York: thin market, same hour
