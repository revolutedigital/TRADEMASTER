"""Round C12: the overnight return and its summary."""

import pandas as pd

from scripts.research import fx_c12_overnight as c12


def test_overnight_is_the_open_over_the_previous_close_and_drops_data_errors() -> None:
    days = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05", "2000-01-06"])
    frame = pd.DataFrame({"t": (days - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1) + 14 * 3600, "o": [100.0, 101.0, 50.0, 51.0],
                          "c": [100.0, 102.0, 51.0, 50.0]})
    nights = c12.overnight(frame, "SPY")
    assert list(nights["r"].round(6)) == [0.01, 0.0]  # 50/102 - 1 is a split, not a night
    summary = c12.summary(pd.concat([nights, nights.assign(symbol="QQQ")]), 0.0001)
    assert summary["nights"] == 4 and round(summary["mean_bp"], 6) == round((0.005 - 0.0001) * 1e4, 6)
