from __future__ import annotations

import pandas as pd

from scripts.research.run_fast150_usd_quoted_basket_circuit_breaker_audit import (
    CircuitBreakerPolicy,
    apply_drawdown_circuit_breaker,
    fast150_status,
    select_breaker,
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


def test_drawdown_breaker_uses_only_closed_trade_outcomes_before_entry() -> None:
    trades = pd.DataFrame(
        [
            _trade(
                pair="EURUSD",
                entry="2027-01-01T00:00:00Z",
                exit_timestamp="2027-01-10T00:00:00Z",
                stress=-600.0,
            ),
            _trade(
                pair="GBPUSD",
                entry="2027-01-05T00:00:00Z",
                exit_timestamp="2027-01-06T00:00:00Z",
                stress=100.0,
            ),
            _trade(
                pair="AUDUSD",
                entry="2027-01-11T00:00:00Z",
                exit_timestamp="2027-01-12T00:00:00Z",
                stress=50.0,
            ),
            _trade(
                pair="NZDUSD",
                entry="2027-02-20T00:00:00Z",
                exit_timestamp="2027-02-21T00:00:00Z",
                stress=50.0,
            ),
        ]
    )

    accepted, skipped, triggers = apply_drawdown_circuit_breaker(
        trades,
        CircuitBreakerPolicy(
            identifier="dd500_cooldown30d",
            closed_equity_drawdown_threshold_pips=500.0,
            cooldown_days=30,
        ),
    )

    assert accepted["pair"].tolist() == ["EURUSD", "GBPUSD", "NZDUSD"]
    assert skipped["pair"].tolist() == ["AUDUSD"]
    assert skipped.iloc[0]["skip_reason"] == "circuit_breaker_pause_active"
    assert len(triggers) >= 1
    assert triggers.iloc[0]["trigger_timestamp"] == pd.Timestamp("2027-01-10T00:00:00Z")


def test_select_breaker_uses_pre_passed_candidates_only() -> None:
    selected = select_breaker(
        [
            {
                "breaker_id": "current_only",
                "pre_passed_breaker_gate": False,
                "pre": {"max_drawdown_pips": -100.0, "lower_stress_pips": 50.0},
                "pre_skipped_count": 1,
                "cooldown_days": 60,
            },
            {
                "breaker_id": "pre_passed",
                "pre_passed_breaker_gate": True,
                "pre": {"max_drawdown_pips": -900.0, "lower_stress_pips": 20.0},
                "pre_skipped_count": 3,
                "cooldown_days": 180,
            },
        ]
    )

    assert selected is not None
    assert selected["breaker_id"] == "pre_passed"


def test_fast150_status_keeps_prospective_observation_requirement() -> None:
    selected = {
        "breaker_id": "dd800_cooldown180d",
        "current_passed_breaker_gate": True,
    }

    status, reasons = fast150_status(selected)

    assert status == "FAST150_CIRCUIT_BREAKER_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "pre_and_current_breaker_gates_passed" in reasons
    assert "not_approved_strategy" in reasons
