"""Round C13, front C: rates known on each day, excess returns with the swap, the four strategies' weights and the
cost of changing them, on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c13_carry as c13

K = c13.CURRENCIES
RATES = {"USD": 0.02, "EUR": 0.01, "GBP": 0.03, "AUD": 0.05, "NZD": 0.04, "JPY": 0.0, "CHF": -0.005, "CAD": 0.025}
DRIFT = {"EUR": 0.001, "GBP": 0.0005, "AUD": -0.001, "NZD": 0.002, "JPY": -0.0005, "CHF": 0.0008, "CAD": -0.002}
NOISE = {"EUR": 0.002, "GBP": 0.004, "AUD": 0.001, "NZD": 0.003, "JPY": 0.005, "CHF": 0.0025, "CAD": 0.0015}


def frame(days, **columns) -> pd.DataFrame:
    return pd.DataFrame({k: columns.get(k, np.nan) for k in K}, index=pd.DatetimeIndex(days), dtype=float)


def rates_frame(days, **overrides) -> pd.DataFrame:
    return pd.DataFrame({k: overrides.get(k, v) for k, v in RATES.items()}, index=pd.DatetimeIndex(days))


def trending_panel(n: int = 300) -> pd.DataFrame:
    """Each currency's log value moves by its drift a day plus an alternating +-noise, so its 60-day volatility is
    proportional to its noise; the pair's price moves with the pair's sign."""
    days = pd.bdate_range("2023-01-02", periods=n)
    t = np.arange(n)
    value = {k: DRIFT[k] * t + NOISE[k] * (-1.0) ** t for k in K}
    return pd.DataFrame({k: np.exp(c13.SIGNS[k] * value[k]) for k in K}, index=days)


def test_fred_csv_reads_dots_and_empty_cells_as_missing() -> None:
    text = "observation_date,IR3TIB01USM156N\n2020-01-01,1.5\n2020-02-01,.\n2020-03-01,\n2020-04-01,2\n"
    series = c13.parse_fred(text)
    assert list(series.index.astype(str)) == ["2020-01", "2020-02", "2020-03", "2020-04"]
    assert series.iloc[0] == 1.5 and series.iloc[3] == 2.0 and series.iloc[1:3].isna().all()
    with pytest.raises(ValueError):
        c13.parse_fred("<html>\n<body>not found</body>\n")


def test_a_day_uses_the_rate_of_two_months_before_carried_forward() -> None:
    series = pd.Series([1.0, np.nan, 3.0], index=pd.PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M"))
    days = ["2020-03-01", "2020-03-31", "2020-04-15", "2020-05-04", "2021-01-04"]
    # March reads January; April reads February, missing, so January again; May reads March; after the end, March
    assert np.allclose(c13.rate_on(series, days), [0.01, 0.01, 0.01, 0.03, 0.03])
    with pytest.raises(ValueError):
        c13.rate_on(series, ["2020-02-28"])  # would read December 2019, before the first observation


def test_excess_returns_follow_the_currency_and_accrue_calendar_days() -> None:
    days = ["2024-01-04", "2024-01-05", "2024-01-08"]  # Thursday, Friday, Monday
    closes = frame(days, EUR=[1.10, 1.11, 1.11], JPY=[140.0, 141.4, 141.4])
    long, short = c13.excess_returns(closes, rates_frame(days), 0.01)
    assert long.iloc[0].isna().all()  # no previous close
    assert long.at[pd.Timestamp("2024-01-05"), "EUR"] == pytest.approx(np.log(1.11 / 1.10) + (0.01 - 0.02 - 0.01) / 365)
    # USDJPY up 1% is the yen down 1%: the short yen earns it and the dollar's higher rate
    assert long.at[pd.Timestamp("2024-01-05"), "JPY"] == pytest.approx(-np.log(1.01) + (0.0 - 0.02 - 0.01) / 365)
    assert short.at[pd.Timestamp("2024-01-05"), "JPY"] == pytest.approx(np.log(1.01) + (0.02 - 0.0 - 0.01) / 365)
    # Friday to Monday: three days of carry and markup, no price move
    assert long.at[pd.Timestamp("2024-01-08"), "EUR"] == pytest.approx((0.01 - 0.02 - 0.01) * 3 / 365)
    assert short.at[pd.Timestamp("2024-01-08"), "JPY"] == pytest.approx((0.02 - 0.01) * 3 / 365)
    assert long["GBP"].isna().all()


def test_the_return_of_a_gap_starts_at_the_previous_available_close() -> None:
    days = ["2024-01-08", "2024-01-09", "2024-01-10"]
    closes = frame(days, EUR=[1.0, np.nan, 1.02])
    long, _ = c13.excess_returns(closes, rates_frame(days), 0.0)
    assert np.isnan(long.iat[1, 0])
    assert long.iat[2, 0] == pytest.approx(np.log(1.02) + (0.01 - 0.02) * 2 / 365)


def test_weights_earn_from_the_next_day_and_every_change_pays_half_the_round_trip() -> None:
    days = pd.bdate_range("2024-01-08", periods=6)
    closes = frame(days, EUR=[1.0] * 6)
    long = frame(days, EUR=[np.nan, 0.01, 0.02, 0.03, 0.04, 0.05])
    short = frame(days, EUR=[np.nan, -0.1, -0.2, -0.3, -0.4, -0.5])
    # open 0.25 long, flip to 0.25 short, resize to 0.1 short, close
    weights = frame(days, EUR=[0.0, 0.25, -0.25, -0.25, -0.1, 0.0]).fillna(0.0)
    half = c3.cost_of("EURUSD") / 2
    net = c13.net_returns(weights, long, short, closes, 1.0)
    gross = [0.0, 0.0, 0.25 * 0.02, 0.25 * -0.3, 0.25 * -0.4, 0.1 * -0.5]
    turnover = [0.0, 0.25, 0.5, 0.0, 0.15, 0.1]  # the flip pays the closing and the opening half
    assert np.allclose(net.to_numpy(), np.array(gross) - half * np.array(turnover))
    stress = c13.net_returns(weights, long, short, closes, 2.0)
    assert np.allclose(stress.to_numpy(), np.array(gross) - 2 * half * np.array(turnover))


def test_cost_is_a_fraction_of_the_price_at_the_trade() -> None:
    days = pd.bdate_range("2024-01-08", periods=3)
    closes = frame(days, JPY=[150.0, np.nan, 160.0])
    zero = frame(days, JPY=[0.0] * 3)
    weights = frame(days, JPY=[0.5, -0.5, -0.5]).fillna(0.0)  # the flip on a day without a close uses the last one
    net = c13.net_returns(weights, zero, zero, closes, 1.0)
    half = c3.cost_of("USDJPY") / 2
    assert np.allclose(net.to_numpy(), [-0.5 * half / 150, -1.0 * half / 150, 0.0])


def test_decisions_are_the_first_fx_day_of_each_week_and_month() -> None:
    days = pd.DatetimeIndex(["2024-12-27", "2024-12-31", "2025-01-02", "2025-01-06", "2025-01-07", "2025-02-03"])
    # the sample starts on a Friday: that week and month began before it, so the first decision is Tuesday the 31st
    assert list(c13.first_of(days, "W")) == [False, True, False, True, False, True]  # Monday 30 missing: Tuesday
    assert list(c13.first_of(days, "M")) == [False, False, True, False, False, True]
    monday = pd.DatetimeIndex(["2024-12-02", "2024-12-03"])  # the first weekday of its week and month
    assert list(c13.first_of(monday, "W")) == [True, False] and list(c13.first_of(monday, "M")) == [True, False]


def test_carry_is_long_the_two_highest_rates_and_short_the_two_lowest() -> None:
    closes = trending_panel(20)
    rates = rates_frame(closes.index)
    rates.loc["2023-01-11":, "EUR"] = 0.10  # a Wednesday: the change waits for the next Monday
    closes.loc["2023-01-16", "AUD"] = np.nan  # no AUD close at that decision: out, the next rate takes its place
    weights = c13.carry_weights(closes, rates)
    first = dict.fromkeys(K, 0.0) | {"AUD": 0.25, "NZD": 0.25, "JPY": -0.25, "CHF": -0.25}
    assert weights.iloc[0].to_dict() == first
    assert weights.loc["2023-01-13"].to_dict() == first
    assert weights.loc["2023-01-16"].to_dict() == dict.fromkeys(K, 0.0) | {"EUR": 0.25, "NZD": 0.25, "JPY": -0.25,
                                                                             "CHF": -0.25}
    assert weights.loc["2023-01-23"].to_dict() == dict.fromkeys(K, 0.0) | {"EUR": 0.25, "AUD": 0.25, "JPY": -0.25,
                                                                             "CHF": -0.25}
    assert (weights.abs().sum(axis=1) == 1.0).all()


def test_carry_ties_keep_the_fixed_currency_order() -> None:
    closes = trending_panel(3)
    rates = rates_frame(closes.index, EUR=0.05, GBP=0.05, AUD=0.05, JPY=0.0, CHF=0.0)
    weights = c13.carry_weights(closes, rates)
    assert weights.iloc[0].to_dict() == dict.fromkeys(K, 0.0) | {"EUR": 0.25, "GBP": 0.25, "JPY": -0.25, "CHF": -0.25}


def test_carry_trend_keeps_only_the_legs_with_the_63_day_move_in_their_favour() -> None:
    closes = trending_panel(80)
    weights = c13.carry_weights(closes, rates_frame(closes.index), with_trend=True)
    early = closes.index[62]  # fewer than 63 moves measured: every leg vetoed
    assert (weights.loc[:early] == 0).all().all()
    # AUD (long leg) falls, CHF (short leg) rises: vetoed; NZD rises and JPY falls: kept
    assert weights.iloc[-1].to_dict() == dict.fromkeys(K, 0.0) | {"NZD": 0.25, "JPY": -0.25}


def test_tsmom_signs_the_252_day_excess_return_and_scales_by_inverse_volatility() -> None:
    closes = trending_panel(300)
    rates = rates_frame(closes.index)
    long, _ = c13.excess_returns(closes, rates, c13.MARKUP["base"])
    weights = c13.tsmom_weights(closes, long)
    assert (weights.loc[:"2023-12-29"] == 0).all().all()  # the first month start with 252 excess returns is January
    sign = {"EUR": 1, "GBP": 1, "AUD": -1, "NZD": 1, "JPY": -1, "CHF": 1, "CAD": -1}
    raw = pd.Series({k: sign[k] / NOISE[k] for k in K})
    expected = raw / raw.abs().sum()
    assert np.allclose(weights.loc["2024-01-02"].to_numpy(), expected.to_numpy())
    assert np.allclose(weights.iloc[-1].to_numpy(), expected.to_numpy())
    assert weights.iloc[-1].abs().sum() == pytest.approx(1.0)
    closes.loc["2024-02-01", "NZD"] = np.nan  # no close at the decision: out, the others renormalized
    weights = c13.tsmom_weights(closes, c13.excess_returns(closes, rates, 0.01)[0])
    without = raw.drop("NZD")
    assert weights.at[pd.Timestamp("2024-02-01"), "NZD"] == 0.0
    assert np.allclose(weights.loc["2024-02-01", without.index].to_numpy(), (without / without.abs().sum()).to_numpy())


def test_dollar_carry_follows_the_mean_foreign_rate_against_the_dollar() -> None:
    closes = trending_panel(30)
    weights = c13.dollar_carry_weights(closes, rates_frame(closes.index))  # mean foreign 2.14% > 2%
    assert np.allclose(weights.to_numpy(), 1 / 7)
    rates = rates_frame(closes.index, USD=0.03)
    closes.loc["2023-02-01", "CAD"] = np.nan
    weights = c13.dollar_carry_weights(closes, rates)
    assert np.allclose(weights.loc[:"2023-01-31"].to_numpy(), -1 / 7)
    assert weights.at[pd.Timestamp("2023-02-01"), "CAD"] == 0.0  # no close: out, the others keep gross exposure 1
    assert np.allclose(weights.loc["2023-02-01", K[:-1]].to_numpy(), -1 / 6)
    # the direction only averages the currencies it trades: without CAD the mean is 2.08%, with a 2.1% dollar short
    rates = rates_frame(closes.index, USD=0.021, CAD=0.10)
    assert np.allclose(c13.dollar_carry_weights(closes, rates).loc["2023-02-01", K[:-1]].to_numpy(), -1 / 6)


def test_the_stress_keeps_the_decisions_and_only_costs_more() -> None:
    closes = trending_panel(300)
    returns = c13.portfolio(closes, rates_frame(closes.index))
    assert list(returns["base"].columns) == list(c13.STRATEGIES)
    assert (returns["stress"] <= returns["base"] + 1e-15).all().all()
    assert not returns["base"].isna().any().any()


def test_summary_drawdown_years_and_the_pass_rule(tmp_path, monkeypatch) -> None:
    days = pd.bdate_range("2021-12-29", periods=4)  # three days of 2021, one of 2022
    base = pd.DataFrame({name: [0.01, -0.03, 0.01, 0.05] for name in c13.STRATEGIES}, index=days)
    table = c13.summarize(base, base - 0.001)
    row = table.iloc[0]
    assert row["max_drawdown"] == pytest.approx(0.03)  # from the 0.01 peak down to -0.02
    assert row["positive_years"] == 0.5 and row["annual_return"] == pytest.approx(0.01 * c13.days_per_year(days))
    assert c13.days_per_year(pd.bdate_range("2021-01-01", "2022-12-31")) == pytest.approx(261, abs=1)
    days = pd.bdate_range("2021-01-01", periods=600)
    noise = np.random.default_rng(1).normal(0, 0.005, (600, 1))
    base = pd.DataFrame(np.hstack([0.003 + noise, noise, noise * 0, -0.003 + noise]), index=days,
                        columns=list(c13.STRATEGIES))
    table = c13.summarize(base, base - 0.0001)
    assert list(table["passes"]) == [True, False, False, False]
    monkeypatch.setattr(c13, "REPORT", tmp_path / "report.md")
    rates = {"USD": pd.Series([1.0], index=pd.PeriodIndex(["2026-08"], freq="M"))}
    text = c13.write_report(table, base, rates, "abc123")
    assert "US$ 1.000" in text and "3x" in text and "USD 2026-08" in text
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == text


def test_daily_closes_take_the_last_minute_of_each_weekday_fx_day(tmp_path) -> None:
    minutes = ["2024-07-05 20:58", "2024-07-05 20:59",  # Friday 16:59 New York closes the FX day of Friday the 5th
               "2024-07-05 21:01",  # after 17:00 on Friday: the FX day of Saturday, dropped
               "2024-07-07 21:05", "2024-07-08 20:59",  # Sunday evening opens Monday's FX day
               "2024-07-08 21:00"]  # 17:00 New York on Monday: Tuesday's FX day
    t = pd.DatetimeIndex(minutes, tz="UTC").as_unit("s").asi8.astype(float)
    m1 = np.column_stack([t, np.ones((len(t), 3)), np.arange(1.0, len(t) + 1)])
    for pair, _ in c13.PAIRS.values():
        np.save(tmp_path / f"{pair}.npy", m1)
    manifest = pd.DataFrame({"symbol": [p for p, _ in c13.PAIRS.values()], "month": "2024-07", "included": True,
                             "symbol_included": True})
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    closes = c13.daily_closes(tmp_path / "manifest.csv", tmp_path)
    assert list(closes.index.strftime("%Y-%m-%d")) == ["2024-07-05", "2024-07-08", "2024-07-09"]
    assert list(closes.columns) == K
    assert list(closes["EUR"]) == [2.0, 5.0, 6.0]
