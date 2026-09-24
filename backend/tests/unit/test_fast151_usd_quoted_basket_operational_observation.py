from __future__ import annotations

import pandas as pd

from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (
    PREREGISTRATION_TIMESTAMP,
    apply_operational_controls,
    fast151_status,
    select_base_candidates,
)


def _trade(
    *,
    pair: str,
    entry: str,
    exit_timestamp: str,
    stress: float,
    side: str = "SHORT",
) -> dict:
    timestamp = pd.Timestamp(entry)
    exit_time = pd.Timestamp(exit_timestamp)
    return {
        "candidate_id": f"{pair}:d1_donchian_fade_ch100_hold20:stop2_fixed_1r",
        "pair": pair,
        "entry_policy_id": "d1_donchian_fade_ch100_hold20",
        "risk_policy_id": "stop2_fixed_1r",
        "channel_days": 100,
        "hold_days": 20,
        "stop_atr_multiple": 2.0,
        "management": "fixed_1r",
        "signal_timestamp": timestamp - pd.Timedelta(days=1),
        "entry_timestamp": timestamp,
        "side": side,
        "daily_atr_pips": 100.0,
        "risk_pips": 200.0,
        "breakout_distance_pips": 20.0,
        "raw_base_pips": stress,
        "net_base_pips": stress,
        "exit_base_timestamp": exit_time,
        "exit_base_reason": "target" if stress > 0 else "stop",
        "raw_stress_pips": stress,
        "net_stress_pips": stress,
        "exit_stress_timestamp": exit_time,
        "exit_stress_reason": "target" if stress > 0 else "stop",
    }


def test_select_base_candidates_requires_post_preregistration_short_target_pair() -> None:
    rows = pd.DataFrame(
        [
            _trade(
                pair="AUDUSD",
                entry="2026-09-24T06:18:19-03:00",
                exit_timestamp="2026-09-25T00:00:00Z",
                stress=10.0,
            ),
            _trade(
                pair="AUDUSD",
                entry="2026-09-24T06:19:19-03:00",
                exit_timestamp="2026-09-25T00:00:00Z",
                stress=10.0,
            ),
            _trade(
                pair="EURUSD",
                entry="2026-09-24T06:20:19-03:00",
                exit_timestamp="2026-09-25T00:00:00Z",
                stress=10.0,
                side="LONG",
            ),
            _trade(
                pair="USDJPY",
                entry="2026-09-24T06:21:19-03:00",
                exit_timestamp="2026-09-25T00:00:00Z",
                stress=10.0,
            ),
        ]
    )

    selected = select_base_candidates(rows, prospective_start=PREREGISTRATION_TIMESTAMP)

    assert selected["pair"].tolist() == ["AUDUSD"]
    assert selected.iloc[0]["fast151_policy_id"].endswith(":max_per_pair_1:dd800_cooldown180d")


def test_apply_operational_controls_starts_from_post_anchor_rows_only() -> None:
    rows = select_base_candidates(
        pd.DataFrame(
            [
                _trade(
                    pair="AUDUSD",
                    entry="2026-09-24T06:19:19-03:00",
                    exit_timestamp="2026-10-20T00:00:00Z",
                    stress=-900.0,
                ),
                _trade(
                    pair="AUDUSD",
                    entry="2026-09-24T06:20:19-03:00",
                    exit_timestamp="2026-10-21T00:00:00Z",
                    stress=50.0,
                ),
                _trade(
                    pair="EURUSD",
                    entry="2026-10-21T00:01:00Z",
                    exit_timestamp="2026-10-22T00:00:00Z",
                    stress=50.0,
                ),
            ]
        ),
        prospective_start=PREREGISTRATION_TIMESTAMP,
    )

    observations, position_skipped, breaker_skipped, breaker_triggers = apply_operational_controls(rows)

    assert observations["pair"].tolist() == ["AUDUSD"]
    assert position_skipped["pair"].tolist() == ["AUDUSD"]
    assert breaker_skipped["pair"].tolist() == ["EURUSD"]
    assert len(breaker_triggers) == 1


def test_fast151_status_for_no_observations() -> None:
    status, reasons = fast151_status(
        summary={"eligible_observation_count": 0},
        gates={"early_fail": False, "shadow_candidate": False},
    )

    assert status == "FAST151_OBSERVATION_NO_ELIGIBLE_OPERATIONAL_TRADES_YET"
    assert "no_post_preregistration_operational_trades" in reasons
    assert "not_approved_strategy" in reasons
