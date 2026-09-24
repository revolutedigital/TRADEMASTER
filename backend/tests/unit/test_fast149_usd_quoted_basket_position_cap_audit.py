from __future__ import annotations

import pandas as pd

from scripts.research.run_fast149_usd_quoted_basket_position_cap_audit import (
    PositionOverlay,
    apply_position_overlay,
    fast149_status,
    select_overlay,
)


def _trade(
    *,
    pair: str,
    entry: str,
    exit_timestamp: str,
    stress: float,
) -> dict:
    timestamp = pd.Timestamp(entry)
    exit_time = pd.Timestamp(exit_timestamp)
    return {
        "candidate_id": f"{pair}:d1_donchian_fade_ch100_hold20:stop2_fixed_1r",
        "pair": pair,
        "entry_policy_id": "d1_donchian_fade_ch100_hold20",
        "risk_policy_id": "stop2_fixed_1r",
        "side": "SHORT",
        "management": "fixed_1r",
        "signal_timestamp": timestamp - pd.Timedelta(days=1),
        "entry_timestamp": timestamp,
        "exit_base_timestamp": exit_time,
        "exit_stress_timestamp": exit_time,
        "net_base_pips": stress,
        "net_stress_pips": stress,
        "exit_stress_reason": "target" if stress > 0 else "stop",
    }


def test_apply_position_overlay_enforces_one_open_trade_per_pair() -> None:
    trades = pd.DataFrame(
        [
            _trade(
                pair="AUDUSD",
                entry="2027-01-01T00:00:00Z",
                exit_timestamp="2027-01-10T00:00:00Z",
                stress=10.0,
            ),
            _trade(
                pair="AUDUSD",
                entry="2027-01-02T00:00:00Z",
                exit_timestamp="2027-01-04T00:00:00Z",
                stress=20.0,
            ),
            _trade(
                pair="EURUSD",
                entry="2027-01-02T00:00:00Z",
                exit_timestamp="2027-01-04T00:00:00Z",
                stress=30.0,
            ),
        ]
    )

    accepted, skipped = apply_position_overlay(
        trades,
        PositionOverlay("max_per_pair_1", max_total_open=None, max_open_per_pair=1),
    )

    assert accepted["pair"].tolist() == ["AUDUSD", "EURUSD"]
    assert len(skipped) == 1
    assert skipped.iloc[0]["skip_reason"] == "max_open_per_pair_reached"


def test_select_overlay_uses_pre_passed_candidates_only() -> None:
    selected = select_overlay(
        [
            {
                "overlay_id": "current_only",
                "pre_passed_operability_gate": False,
                "pre": {"max_drawdown_pips": -100.0, "lower_stress_pips": 50.0, "max_concurrent_trades": 1},
            },
            {
                "overlay_id": "pre_passed",
                "pre_passed_operability_gate": True,
                "pre": {"max_drawdown_pips": -500.0, "lower_stress_pips": 10.0, "max_concurrent_trades": 3},
            },
        ]
    )

    assert selected is not None
    assert selected["overlay_id"] == "pre_passed"


def test_fast149_status_keeps_circuit_breaker_requirement_visible() -> None:
    selected = {
        "overlay_id": "max_per_pair_1",
        "current_passed_operability_gate": True,
        "pre": {"two_consecutive_negative_years": True},
    }

    status, reasons = fast149_status(selected)

    assert status == "FAST149_POSITION_CAP_REDUCES_DRAWDOWN_BUT_REQUIRES_CIRCUIT_BREAKER"
    assert "pre_still_has_two_consecutive_negative_years" in reasons
    assert "not_approved_strategy" in reasons
