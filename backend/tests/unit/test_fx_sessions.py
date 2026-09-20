"""Compiled London and New York wall-clock time agrees with zoneinfo over the whole sample."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from app.fx import sessions

ZONES = {sessions.LONDON: ZoneInfo("Europe/London"), sessions.NEW_YORK: ZoneInfo("America/New_York")}
FIRST_YEAR, LAST_YEAR = 2015, 2032


def expected_offset(seconds: int, zone: int) -> int:
    local = datetime.fromtimestamp(seconds, tz=ZONES[zone])
    return int(local.utcoffset().total_seconds())


def random_instants(count: int, seed: int) -> list[int]:
    low = int(datetime(FIRST_YEAR, 1, 1, tzinfo=UTC).timestamp())
    high = int(datetime(LAST_YEAR + 1, 1, 1, tzinfo=UTC).timestamp())
    return np.random.default_rng(seed).integers(low, high, count).tolist()


@pytest.mark.parametrize("zone", [sessions.LONDON, sessions.NEW_YORK])
def test_the_offset_matches_zoneinfo_at_random_instants(zone: int) -> None:
    for instant in random_instants(40_000, seed=zone):
        assert sessions.utc_offset(float(instant), zone) == expected_offset(instant, zone), instant


@pytest.mark.parametrize("zone", [sessions.LONDON, sessions.NEW_YORK])
def test_the_offset_flips_at_exactly_the_second_zoneinfo_flips(zone: int) -> None:
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        moment = datetime(year, 1, 1, tzinfo=UTC)
        flips = []
        while moment.year == year:
            following = moment + timedelta(hours=1)
            if expected_offset(int(moment.timestamp()), zone) != expected_offset(
                int(following.timestamp()), zone
            ):
                flips.append(int(following.timestamp()))
            moment = following
        assert len(flips) == 2, (year, flips)
        for instant in flips:
            before = sessions.utc_offset(float(instant - 1), zone)
            after = sessions.utc_offset(float(instant), zone)
            assert before == expected_offset(instant - 1, zone), (year, instant)
            assert after == expected_offset(instant, zone), (year, instant)
            assert before != after


@pytest.mark.parametrize("zone", [sessions.LONDON, sessions.NEW_YORK])
def test_local_time_of_day_and_day_match_the_zoneinfo_wall_clock(zone: int) -> None:
    for instant in random_instants(20_000, seed=10 + zone):
        local = datetime.fromtimestamp(instant, tz=ZONES[zone])
        seconds = local.hour * 3600 + local.minute * 60 + local.second
        assert sessions.local_time_of_day(float(instant), zone) == seconds
        day_number = sessions.local_day(float(instant), zone)
        assert date.fromordinal(int(day_number) + date(1970, 1, 1).toordinal()) == local.date()


def test_the_fx_day_changes_at_seventeen_hours_new_york_time_in_summer_and_winter() -> None:
    new_york = ZoneInfo("America/New_York")
    for day in (date(2024, 1, 10), date(2024, 7, 10), date(2024, 3, 12), date(2024, 11, 5)):
        before = datetime(day.year, day.month, day.day, 16, 59, 59, tzinfo=new_york)
        after = before + timedelta(seconds=1)
        assert sessions.fx_day(after.timestamp()) == sessions.fx_day(before.timestamp()) + 1
        earlier = before - timedelta(hours=3)
        assert sessions.fx_day(earlier.timestamp()) == sessions.fx_day(before.timestamp())


def test_the_two_cities_disagree_on_the_clocks_in_the_weeks_between_their_switches() -> None:
    # 2024-03-20: the United States is already on summer time, Europe is not yet.
    noon_utc = datetime(2024, 3, 20, 12, tzinfo=UTC).timestamp()
    assert sessions.utc_offset(noon_utc, sessions.LONDON) == 0
    assert sessions.utc_offset(noon_utc, sessions.NEW_YORK) == -4 * 3600
    # 2024-10-30: Europe is back on winter time, the United States is not yet.
    late_october = datetime(2024, 10, 30, 12, tzinfo=UTC).timestamp()
    assert sessions.utc_offset(late_october, sessions.LONDON) == 0
    assert sessions.utc_offset(late_october, sessions.NEW_YORK) == -4 * 3600


def test_the_civil_calendar_helpers_agree_with_python_dates() -> None:
    assert sessions._days_from_civil(1970, 1, 1) == 0
    for ordinal in range(date(2000, 1, 1).toordinal(), date(2040, 12, 31).toordinal(), 17):
        day = date.fromordinal(ordinal)
        days = ordinal - date(1970, 1, 1).toordinal()
        assert sessions._days_from_civil(day.year, day.month, day.day) == days
        assert sessions._year_of_days(days) == day.year


def test_seconds_of_day_parses_a_clock_and_rejects_nonsense() -> None:
    assert sessions.seconds_of_day("00:00") == 0
    assert sessions.seconds_of_day("16:05") == 16 * 3600 + 5 * 60
    assert sessions.seconds_of_day("24:00") == 86_400
    for bad in ("25:00", "10:60", "noon", "10"):
        with pytest.raises(ValueError):
            sessions.seconds_of_day(bad)
