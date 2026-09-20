"""Wall-clock time in London and New York, computed from UTC seconds inside compiled code.

The fast families act at fixed local times (the London open, the 16:00 London fixing, the New York
open, the 17:00 New York rollover), and the two cities change their clocks on different dates:
Europe on the last Sundays of March and October, the United States on the second Sunday of March
and the first Sunday of November. Strategies run inside a compiled loop that cannot call
`zoneinfo`, so the rules are implemented with integer date arithmetic and tested against
`zoneinfo` over the whole sample.
"""

from __future__ import annotations

import numpy as np
from numba import njit

LONDON = 0
NEW_YORK = 1

SECONDS_PER_DAY = 86_400
HOUR = 3_600
ROLLOVER_SECONDS = 17 * HOUR  # the FX day starts at 17:00 New York


def seconds_of_day(clock: str) -> int:
    """`"16:05"` to 57_900, for readable strategy presets."""
    hours, minutes = clock.split(":")
    if not (0 <= int(hours) <= 24 and 0 <= int(minutes) < 60):
        raise ValueError(f"not a wall-clock time: {clock!r}")
    return int(hours) * HOUR + int(minutes) * 60


@njit(cache=True)
def _days_from_civil(year, month, day):
    """Days since 1970-01-01 of a proleptic Gregorian date (Hinnant's algorithm)."""
    y = year - 1 if month <= 2 else year
    era = (y if y >= 0 else y - 399) // 400
    year_of_era = y - era * 400
    shifted_month = (month + 9) % 12  # March is 0
    day_of_year = (153 * shifted_month + 2) // 5 + day - 1
    day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146_097 + day_of_era - 719_468


@njit(cache=True)
def _year_of_days(days):
    """The civil year that contains `days` since 1970-01-01."""
    z = days + 719_468
    era = (z if z >= 0 else z - 146_096) // 146_097
    day_of_era = z - era * 146_097
    year_of_era = (
        day_of_era - day_of_era // 1_460 + day_of_era // 36_524 - day_of_era // 146_096
    ) // 365
    day_of_year = day_of_era - (365 * year_of_era + year_of_era // 4 - year_of_era // 100)
    shifted_month = (5 * day_of_year + 2) // 153
    month = shifted_month + 3 if shifted_month < 10 else shifted_month - 9
    return year_of_era + era * 400 + (1 if month <= 2 else 0)


@njit(cache=True)
def _last_sunday(year, month):
    following = _days_from_civil(year + 1, 1, 1) if month == 12 else _days_from_civil(year, month + 1, 1)
    last_day = following - 1
    weekday_of_last = (last_day + 3) % 7  # Monday is 0; 1970-01-01 was a Thursday
    return last_day - (weekday_of_last + 1) % 7


@njit(cache=True)
def _nth_sunday(year, month, nth):
    first_day = _days_from_civil(year, month, 1)
    weekday_of_first = (first_day + 3) % 7
    return first_day + (6 - weekday_of_first) % 7 + 7 * (nth - 1)


@njit(cache=True)
def utc_offset(seconds, zone):
    """Seconds that local time is ahead of UTC at this instant (London 0 or 3600, New York -18000 or -14400)."""
    instant = np.int64(np.floor(seconds))
    year = _year_of_days(instant // SECONDS_PER_DAY)
    if zone == LONDON:
        start = _last_sunday(year, 3) * SECONDS_PER_DAY + HOUR  # 01:00 UTC
        end = _last_sunday(year, 10) * SECONDS_PER_DAY + HOUR
        return HOUR if start <= instant < end else 0
    start = _nth_sunday(year, 3, 2) * SECONDS_PER_DAY + 7 * HOUR  # 02:00 EST
    end = _nth_sunday(year, 11, 1) * SECONDS_PER_DAY + 6 * HOUR  # 02:00 EDT
    return -4 * HOUR if start <= instant < end else -5 * HOUR


@njit(cache=True)
def local_time_of_day(seconds, zone):
    """Seconds since local midnight, in [0, 86400)."""
    local = np.int64(np.floor(seconds)) + utc_offset(seconds, zone)
    return local % SECONDS_PER_DAY


@njit(cache=True)
def local_day(seconds, zone):
    """A running number for the local calendar day; it changes at local midnight."""
    local = np.int64(np.floor(seconds)) + utc_offset(seconds, zone)
    return local // SECONDS_PER_DAY


@njit(cache=True)
def fx_day(seconds):
    """A running number for the FX day, which runs from 17:00 to 17:00 New York time."""
    local = np.int64(np.floor(seconds)) + utc_offset(seconds, NEW_YORK) - ROLLOVER_SECONDS
    return local // SECONDS_PER_DAY
