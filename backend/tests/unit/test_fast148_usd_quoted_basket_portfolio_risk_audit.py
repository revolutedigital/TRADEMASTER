from __future__ import annotations

import pandas as pd

from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import (
    equity_curve,
    fast148_status,
    max_concurrent_trades,
    summarize_sample,
)


def _trade(
    *,
    pair: str,
    entry: str,
    exit_timestamp: str,
    stress: float,
) -> dict:
    return {
        "pair": pair,
        "entry_policy_id": "d1_donchian_fade_ch100_hold20",
        "risk_policy_id": "stop2_fixed_1r",
        "side": "SHORT",
        "management": "fixed_1r",
        "signal_timestamp": pd.Timestamp(entry) - pd.Timedelta(days=1),
        "entry_timestamp": pd.Timestamp(entry),
        "exit_base_timestamp": pd.Timestamp(exit_timestamp),
        "exit_stress_timestamp": pd.Timestamp(exit_timestamp),
        "net_stress_pips": stress,
        "exit_stress_reason": "target" if stress > 0 else "stop",
    }


def test_equity_curve_reports_drawdown_from_closed_trade_sequence() -> None:
    sample = pd.DataFrame(
        [
            _trade(pair="AUDUSD", entry="2027-01-01T00:00:00Z", exit_timestamp="2027-01-02T00:00:00Z", stress=100.0),
            _trade(pair="EURUSD", entry="2027-01-03T00:00:00Z", exit_timestamp="2027-01-04T00:00:00Z", stress=-40.0),
            _trade(pair="GBPUSD", entry="2027-01-05T00:00:00Z", exit_timestamp="2027-01-06T00:00:00Z", stress=-70.0),
        ]
    )

    curve = equity_curve(sample)

    assert curve["equity_pips"].tolist() == [100.0, 60.0, -10.0]
    assert curve["drawdown_pips"].tolist() == [0.0, -40.0, -110.0]


def test_max_concurrent_trades_counts_overlapping_intervals() -> None:
    sample = pd.DataFrame(
        [
            _trade(pair="AUDUSD", entry="2027-01-01T00:00:00Z", exit_timestamp="2027-01-10T00:00:00Z", stress=10.0),
            _trade(pair="EURUSD", entry="2027-01-02T00:00:00Z", exit_timestamp="2027-01-05T00:00:00Z", stress=10.0),
            _trade(pair="GBPUSD", entry="2027-01-03T00:00:00Z", exit_timestamp="2027-01-04T00:00:00Z", stress=10.0),
        ]
    )

    assert max_concurrent_trades(sample) == 3


def test_fast148_status_requires_controls_for_large_drawdown_and_bad_year_cluster() -> None:
    rows = []
    for index in range(20):
        rows.append(
            _trade(
                pair="AUDUSD",
                entry=f"2027-01-{index + 1:02d}T00:00:00Z",
                exit_timestamp=f"2027-01-{index + 1:02d}T01:00:00Z",
                stress=-100.0,
            )
        )
    for index in range(20):
        rows.append(
            _trade(
                pair="NZDUSD",
                entry=f"2028-01-{index + 1:02d}T00:00:00Z",
                exit_timestamp=f"2028-01-{index + 1:02d}T01:00:00Z",
                stress=-100.0,
            )
        )
    summary = summarize_sample(pd.DataFrame(rows))

    status, reasons = fast148_status(summary, summary)

    assert status == "FAST148_USD_QUOTED_BASKET_REQUIRES_DRAWDOWN_AND_EXPOSURE_CONTROLS"
    assert "pre:historical_drawdown_exceeds_policy_threshold" in reasons
    assert "pre:two_consecutive_negative_years" in reasons
    assert "not_approved_strategy" in reasons
