from __future__ import annotations

import pandas as pd

from scripts.research.run_fast147_usd_quoted_major_basket_observation import (
    ENTRY_POLICY_ID,
    PREREGISTRATION_TIMESTAMP,
    RISK_POLICY_ID,
    SIDE_FILTER,
    TARGET_PAIRS,
    evaluate_fast147_gates,
    fast147_status,
    select_eligible_observations,
    summarize_observations,
    summarize_pairs,
)


def _row(
    *,
    pair: str,
    side: str,
    entry_timestamp: str,
    stress: float = 30.0,
    base: float | None = None,
) -> dict:
    timestamp = pd.Timestamp(entry_timestamp)
    return {
        "candidate_id": f"{pair}:{ENTRY_POLICY_ID}:{RISK_POLICY_ID}",
        "pair": pair,
        "entry_policy_id": ENTRY_POLICY_ID,
        "risk_policy_id": RISK_POLICY_ID,
        "channel_days": 100,
        "hold_days": 20,
        "stop_atr_multiple": 2.0,
        "management": "fixed_1r",
        "signal_timestamp": timestamp - pd.Timedelta(days=1),
        "entry_timestamp": timestamp,
        "side": side,
        "daily_atr_pips": 50.0,
        "risk_pips": 100.0,
        "breakout_distance_pips": 12.0,
        "raw_base_pips": stress if base is None else base,
        "net_base_pips": stress if base is None else base,
        "exit_base_timestamp": timestamp + pd.Timedelta(days=1),
        "exit_base_reason": "target",
        "raw_stress_pips": stress,
        "net_stress_pips": stress,
        "exit_stress_timestamp": timestamp + pd.Timedelta(days=1),
        "exit_stress_reason": "target",
    }


def test_select_eligible_observations_filters_timestamp_pair_and_side() -> None:
    rows = pd.DataFrame(
        [
            _row(pair="AUDUSD", side=SIDE_FILTER, entry_timestamp="2026-09-24T05:49:20-03:00"),
            _row(pair="AUDUSD", side=SIDE_FILTER, entry_timestamp="2026-09-25T00:00:00Z"),
            _row(pair="USDCAD", side=SIDE_FILTER, entry_timestamp="2026-09-25T00:00:00Z"),
            _row(pair="NZDUSD", side="LONG", entry_timestamp="2026-09-25T00:00:00Z"),
        ]
    )

    selected = select_eligible_observations(rows, prospective_start=PREREGISTRATION_TIMESTAMP)

    assert len(selected) == 1
    assert selected.iloc[0]["pair"] == "AUDUSD"
    assert selected.iloc[0]["side"] == SIDE_FILTER
    assert selected.iloc[0]["fast147_policy_id"].startswith("basket:usd_quoted_majors")


def test_fast147_gates_allow_shadow_only_after_full_prospective_requirements() -> None:
    rows = []
    for index in range(48):
        pair = TARGET_PAIRS[index % len(TARGET_PAIRS)]
        year = 2027 if index < 24 else 2028
        day = (index % 24) + 1
        rows.append(
            _row(
                pair=pair,
                side=SIDE_FILTER,
                entry_timestamp=f"{year}-01-{day:02d}T00:00:00Z",
                stress=35.0,
            )
        )
    observations = select_eligible_observations(
        pd.DataFrame(rows),
        prospective_start=PREREGISTRATION_TIMESTAMP,
    )
    summary = summarize_observations(observations, alpha=0.05)
    pair_summary = summarize_pairs(observations)
    gates = evaluate_fast147_gates(
        summary=summary,
        observations=observations,
        pair_summary=pair_summary,
    )
    status, reasons = fast147_status(summary=summary, gates=gates)

    assert gates["shadow_candidate"] is True
    assert status == "FAST147_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW"
    assert "no_order_execution" in reasons


def test_fast147_status_for_no_observations() -> None:
    observations = select_eligible_observations(
        pd.DataFrame(),
        prospective_start=PREREGISTRATION_TIMESTAMP,
    )
    summary = summarize_observations(observations, alpha=0.05)
    gates = evaluate_fast147_gates(
        summary=summary,
        observations=observations,
        pair_summary=summarize_pairs(observations),
    )
    status, reasons = fast147_status(summary=summary, gates=gates)

    assert status == "FAST147_OBSERVATION_NO_ELIGIBLE_BASKET_TRADES_YET"
    assert "not_approved_strategy" in reasons
