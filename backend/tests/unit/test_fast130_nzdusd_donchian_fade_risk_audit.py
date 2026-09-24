from __future__ import annotations

import pandas as pd

from scripts.research.run_fast120_fx_currency_strength_scout import Management
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (
    RiskPolicy,
    fast130_status,
    risk_policy_grid,
    simulate_gappy_trade,
)


def test_risk_policy_grid_is_deterministic() -> None:
    policies = risk_policy_grid()

    assert len(policies) == 12
    assert policies[0] == RiskPolicy(
        "stop1_stop_only",
        1.0,
        Management("stop_only", target_r=None, activation_r=None, trail_r=None),
    )
    assert policies[-1].identifier == "stop2_trail_1r_1r"


def test_fast130_status_never_approves_strategy() -> None:
    status, reasons = fast130_status(passed_count=1, finalists_count=1)

    assert status == "FAST130_NZDUSD_DONCHIAN_FADE_RISK_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

    status, reasons = fast130_status(passed_count=0, finalists_count=1)

    assert status == "FAST130_NZDUSD_DONCHIAN_FADE_RISK_FAILED_AUDIT"
    assert "risk_management_variants_failed_2024_2026_audit" in reasons


def test_simulate_gappy_trade_allows_weekend_like_missing_minutes() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-02T21:00:00Z",
            "2026-01-02T21:01:00Z",
            "2026-01-05T00:00:00Z",
            "2026-01-05T21:00:00Z",
        ]
    )
    frame = pd.DataFrame(
        {
            "bid_open": [100.00, 100.00, 100.10, 100.20],
            "bid_high": [100.00, 100.00, 100.10, 100.20],
            "bid_low": [100.00, 100.00, 100.10, 100.20],
            "bid_close": [100.00, 100.00, 100.10, 100.20],
            "ask_open": [100.01, 100.01, 100.11, 100.21],
            "ask_high": [100.01, 100.01, 100.11, 100.21],
            "ask_low": [100.01, 100.01, 100.11, 100.21],
            "ask_close": [100.01, 100.01, 100.11, 100.21],
        },
        index=index,
    )

    result = simulate_gappy_trade(
        frame,
        pair="USDJPY",
        entry_timestamp=pd.Timestamp("2026-01-02T21:00:00Z"),
        side=1,
        risk_pips=50.0,
        hold_minutes=3 * 24 * 60,
        management=Management("stop_only", target_r=None, activation_r=None, trail_r=None),
        commission=0.0,
        slippage_per_side_pips=0.0,
    )

    assert result is not None
    assert result.exit_reason == "timeout"
