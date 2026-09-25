"""C5 daily simulator: fills at the next open, reversals, costs and financing per day held."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c5_lab as c5


def frame(opens, closes):
    t = 1_700_000_000 + 86400 * np.arange(len(opens))
    return pd.DataFrame({"t": t, "o": opens, "h": np.maximum(opens, closes), "l": np.minimum(opens, closes), "c": closes})


def plan(enter, exit_long=None, exit_short=None):
    n = len(enter)
    return c3._plan(n, enter=np.array(enter, dtype=np.int8),  # noqa: SLF001
                    exit_long=np.array(exit_long or [False] * n), exit_short=np.array(exit_short or [False] * n),
                    unit=np.ones(n))


def test_entry_and_exit_fill_at_the_next_open_and_a_reversal_is_two_trades() -> None:
    f = frame([100, 101, 102, 103, 104], [100.5, 101.5, 102.5, 103.5, 104.5])
    assert c5.simulate_daily(f, plan([1, 0, -1, 0, 0])) == [(1, 1, 3), (-1, 3, 4)]
    assert c5.simulate_daily(f, plan([1, 0, 0, 0, 0], exit_long=[False, True, False, False, False])) == [(1, 1, 2)]


def test_cost_and_financing_come_off_every_trade() -> None:
    f = frame([100.0, 100.0, 110.0], [100.0, 100.0, 110.0])
    result = c5.trade_results(f, plan([1, 0, 0]), "base")
    held_years = 86400 / c5.YEAR_SECONDS
    assert result["net"].iloc[0] == pytest.approx(10 - 0.0005 * 100 - 0.03 * held_years * 100)
