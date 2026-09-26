"""Round C13: the shared holdout's daily t."""

import numpy as np
import pandas as pd

from scripts.research import fx_c13_lab as lab


def test_daily_t_is_the_mean_over_its_standard_error() -> None:
    series = pd.Series([0.01, -0.005, 0.02, 0.0])
    t, p = lab.daily_t(series)
    assert np.isclose(t, series.mean() / (series.std(ddof=1) / 2)) and 0 < p < 0.5
    assert lab.daily_t(pd.Series([0.0, 0.0]))[1] == 1.0  # no variation: no evidence
