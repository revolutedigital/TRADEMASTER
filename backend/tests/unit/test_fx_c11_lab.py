"""Round C11: the calendar of the windows, the month-end sign and the entries at each window's start."""

import numpy as np
import pandas as pd

from scripts.research import fx_c11_lab as c11


def test_turn_of_month_is_the_last_weekday_and_the_first_three() -> None:
    days = c11.turn_of_month_days("2024-05-01", "2024-06-30")
    wanted = {"2024-05-01", "2024-05-02", "2024-05-03", "2024-05-31", "2024-06-03", "2024-06-04", "2024-06-05",
              "2024-06-28"}
    assert days == {pd.Timestamp(d) for d in wanted}
    assert c11.last_weekdays("2024-08-01", "2024-08-31") == {pd.Timestamp("2024-08-30")}


def test_month_end_sign_sells_dollars_when_us_equities_led_the_month() -> None:
    index = pd.to_datetime(["2024-04-30", "2024-05-15", "2024-05-30", "2024-05-31"])
    closes = {"^GSPC": pd.Series([100.0, 105, 110, 50], index=index),
              "^GDAXI": pd.Series([100.0, 101, 102, 500], index=index)}
    assert c11.month_end_sign(closes, pd.Timestamp("2024-05-31")) == -1  # the close of the day itself is not used
    closes["^GDAXI"] = pd.Series([100.0, 101, 120, 1], index=index)
    assert c11.month_end_sign(closes, pd.Timestamp("2024-05-31")) == 1
    assert c11.month_end_sign({"^GDAXI": closes["^GDAXI"]}, pd.Timestamp("2024-05-31")) == 0


def test_entries_are_the_first_minute_at_the_window_start_once_a_day() -> None:
    start = int(pd.Timestamp("2024-07-08 21:00", tz="UTC").timestamp())  # 17:00 New York
    t = start + 60 * np.arange(3 * 1440)
    t = np.delete(t, np.searchsorted(t, start + 65 * 60))  # 18:05 missing: the first evening enters at 18:06
    m1 = np.column_stack([t, np.ones((len(t), 4))])
    window = c11.Window("test", "America/New_York", 18 * 60 + 5, 60, {"SPXUSD": 1})
    entries, multiplier = c11.window_entries(m1, window, None)
    local = pd.to_datetime(m1[entries, 0], unit="s", utc=True).tz_convert("America/New_York")
    assert [x.strftime("%m-%d %H:%M") for x in local] == ["07-08 18:06", "07-09 18:05", "07-10 18:05"]
    tom = c11.Window("test", c11.FX, 65, 0, {"SPXUSD": 1}, days="tom", tolerance=600)
    late = m1[(m1[:, 0] < start + 1440 * 60 + 65 * 60) | (m1[:, 0] >= start + 1440 * 60 + 180 * 60)]  # 18:05-20:00 gone
    entries, _ = c11.window_entries(late, tom, {pd.Timestamp("2024-07-10"): 1})
    assert len(entries) == 1  # the evening of the 9th opens the FX day of the 10th, at its first minute from 18:05
    local = pd.Timestamp(late[entries[0], 0], unit="s", tz="UTC").tz_convert("America/New_York")
    assert (local.day, local.hour, local.minute) == (9, 20, 0)
