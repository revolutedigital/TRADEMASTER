"""The Gotobi demo runner's schedule: configuration B's two legs in UTC, and the Monday rule."""

import pandas as pd

from scripts.research import fx_c14_gotobi as c14
from scripts.research import fx_gotobi_demo as runner


def utc(text: str) -> float:
    return pd.Timestamp(text, tz="UTC").timestamp()


def test_a_weekday_gotobi_buys_at_three_tokyo_and_reverses_after_the_fix() -> None:
    before, after = runner.legs_of(pd.Timestamp("2026-09-30"))  # a Wednesday
    assert (before.side, before.open_at, before.close_at) == (1, utc("2026-09-29 18:00"), utc("2026-09-30 00:54:45"))
    assert (after.side, after.open_at, after.close_at) == (-1, utc("2026-09-30 00:56"), utc("2026-09-30 02:59:45"))


def test_a_monday_gotobi_buys_an_hour_after_the_sunday_reopening() -> None:
    before, _ = runner.legs_of(pd.Timestamp("2026-10-05"))  # Monday: 03:00 JST is Sunday 14:00 in New York
    assert before.open_at == utc("2026-10-04 22:00")  # Sunday 18:00 New York (summer time)


def test_the_next_days_come_from_the_registered_calendar() -> None:
    days = runner.upcoming_days(utc("2026-09-26 12:00"), c14.japanese_holidays())
    assert [str(d.date()) for d in days[:3]] == ["2026-09-30", "2026-10-05", "2026-10-09"]  # the 10th is a Saturday
