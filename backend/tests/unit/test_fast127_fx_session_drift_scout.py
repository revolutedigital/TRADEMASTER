from __future__ import annotations

import pandas as pd

from scripts.research.run_fast127_fx_session_drift_scout import (
    Policy,
    fast127_status,
    policy_entry_index,
    policy_grid,
    timebox_trade_rows,
)


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-05T00:00:00Z", periods=181, freq="1min")
    return pd.DataFrame(
        {
            "bid_open": 100.00,
            "bid_high": 100.00,
            "bid_low": 100.00,
            "bid_close": 100.00,
            "ask_open": 100.01,
            "ask_high": 100.01,
            "ask_low": 100.01,
            "ask_close": 100.01,
        },
        index=index,
    )


def test_policy_grid_is_deterministic() -> None:
    policies = policy_grid()

    assert len(policies) == 42
    assert policies[0] == Policy("utc00_wdall_hold240_long", 0, 240, "long", "all")
    assert policies[-1] == Policy("utc20_wdall_hold1440_short", 20, 1440, "short", "all")


def test_policy_entry_index_filters_hour_and_weekday() -> None:
    frame = _frame()

    selected = policy_entry_index(
        frame,
        Policy("utc00_wdmon_hold120_long", 0, 120, "long", "mon"),
    )

    assert [timestamp.isoformat() for timestamp in selected] == ["2026-01-05T00:00:00+00:00"]


def test_timebox_trade_rows_uses_executable_bid_ask() -> None:
    frame = _frame()
    frame.loc[pd.Timestamp("2026-01-05T02:00:00Z"), "bid_open"] = 100.21
    policy = Policy("utc00_wdall_hold120_long", 0, 120, "long", "all")

    rows = timebox_trade_rows(
        frame,
        pair="USDJPY",
        policy=policy,
        commission_pips=0.1,
        slippage_per_side_pips=0.05,
    )

    assert len(rows) == 1
    assert rows.iloc[0]["side"] == "LONG"
    assert round(float(rows.iloc[0]["raw_pips"]), 6) == 20.0
    assert round(float(rows.iloc[0]["net_pips"]), 6) == 19.8


def test_fast127_status_never_approves_strategy() -> None:
    status, reasons = fast127_status(passed_count=1, finalists_count=1)

    assert status == "FAST127_FX_SESSION_DRIFT_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

    status, reasons = fast127_status(passed_count=0, finalists_count=0)

    assert status == "FAST127_FX_SESSION_DRIFT_FAILED_SELECTION"
    assert "no_selection_candidate_positive_after_costs" in reasons
