"""Hole detection and cross triangulation used to decide which months enter the lab."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_fast_manifest as mf


def closes(values, start="2024-05-13 00:00"):
    index = pd.date_range(start, periods=len(values), freq="min", tz="UTC")
    series = pd.Series(values, index=index)
    return pd.DataFrame({"bid_close": series - 0.00002, "ask_close": series + 0.00002})


def test_a_month_far_below_the_usual_count_is_a_hole_and_a_short_february_is_not() -> None:
    minutes = pd.Series([31000, 28800, 31500, 21000, 30800, 31200], index=list("abcdef"))

    assert mf.flag_holes(minutes).tolist() == [False, False, False, True, False, False]


def test_a_consistent_cross_has_no_triangulation_error_and_a_wrong_one_is_measured() -> None:
    a = closes(1.10 + np.arange(200) * 1e-5)
    b = closes(150.0 + np.arange(200) * 1e-3)
    exact = closes((a["bid_close"] + a["ask_close"]).to_numpy() / 2 * ((b["bid_close"] + b["ask_close"]).to_numpy() / 2))
    off = closes((a["bid_close"] + a["ask_close"]).to_numpy() / 2 * ((b["bid_close"] + b["ask_close"]).to_numpy() / 2) + 0.05)

    assert mf.triangulation_error_pips(exact, a, b, "multiply", 0.01) == pytest.approx(0.0, abs=1e-6)
    assert mf.triangulation_error_pips(off, a, b, "multiply", 0.01) == pytest.approx(5.0, abs=1e-6)
    ratio = closes((a["bid_close"] + a["ask_close"]).to_numpy() / 2 / ((b["bid_close"] + b["ask_close"]).to_numpy() / 2))
    assert mf.triangulation_error_pips(ratio, a, b, "divide", 0.0001) == pytest.approx(0.0, abs=1e-6)


def test_the_triangulation_of_minutes_without_a_partner_is_not_a_number() -> None:
    a = closes([1.1] * 5)
    b = closes([150.0] * 5, start="2024-06-01 00:00")

    assert np.isnan(mf.triangulation_error_pips(a, a, b, "multiply", 0.01))
