from __future__ import annotations

import pandas as pd

from scripts.research.run_fast137_fx_daily_donchian_fade_temporal_reverse import (
    audit_current_sample,
    fast137_status,
    select_pre_finalists,
    summarize_pre_selection,
)


def _trade(
    *,
    candidate_id: str,
    entry_timestamp: str,
    net_base_pips: float = 6.0,
    net_stress_pips: float = 5.0,
) -> dict:
    timestamp = pd.Timestamp(entry_timestamp)
    return {
        "sample": "synthetic",
        "candidate_id": candidate_id,
        "pair": candidate_id.split(":")[0],
        "entry_policy_id": "d1_donchian_fade_ch100_hold20",
        "risk_policy_id": "stop2_fixed_1r",
        "channel_days": 100,
        "hold_days": 20,
        "stop_atr_multiple": 2.0,
        "management": "fixed_1r",
        "signal_timestamp": timestamp - pd.Timedelta(days=1),
        "entry_timestamp": timestamp,
        "side": "LONG",
        "breakout_direction": "DOWN",
        "daily_atr_pips": 50.0,
        "risk_pips": 100.0,
        "breakout_distance_pips": 8.0,
        "raw_base_pips": net_base_pips + 1.0,
        "net_base_pips": net_base_pips,
        "exit_base_timestamp": timestamp + pd.Timedelta(days=2),
        "exit_base_reason": "target",
        "raw_stress_pips": net_stress_pips + 1.0,
        "net_stress_pips": net_stress_pips,
        "exit_stress_timestamp": timestamp + pd.Timedelta(days=2),
        "exit_stress_reason": "target",
    }


def test_pre_selection_filters_positive_pre2019_candidates() -> None:
    candidate = "NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r"
    loser = "EURUSD:d1_donchian_fade_ch20_hold5:stop1_stop_only"
    rows = [
        _trade(candidate_id=candidate, entry_timestamp=f"{year}-01-{day:02d}T00:00:00Z")
        for year in range(2011, 2016)
        for day in range(1, 7)
    ]
    rows.extend(
        _trade(
            candidate_id=loser,
            entry_timestamp=f"{year}-02-{day:02d}T00:00:00Z",
            net_base_pips=-1.0,
            net_stress_pips=-2.0,
        )
        for year in range(2011, 2016)
        for day in range(1, 7)
    )

    selection = summarize_pre_selection(pd.DataFrame(rows))
    finalists = select_pre_finalists(selection)

    assert len(selection) == 2
    assert len(finalists) == 1
    assert finalists.iloc[0]["candidate_id"] == candidate


def test_current_audit_uses_bonferroni_gate() -> None:
    candidate = "NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r"
    pre_rows = [
        _trade(candidate_id=candidate, entry_timestamp=f"{year}-01-{day:02d}T00:00:00Z")
        for year in range(2011, 2016)
        for day in range(1, 7)
    ]
    finalists = select_pre_finalists(summarize_pre_selection(pd.DataFrame(pre_rows)))
    current_rows = [
        _trade(candidate_id=candidate, entry_timestamp=f"{year}-03-{day:02d}T00:00:00Z", net_base_pips=10.0, net_stress_pips=9.0)
        for year in range(2019, 2022)
        for day in range(1, 6)
    ]

    audit = audit_current_sample(pd.DataFrame(current_rows), finalists)

    assert audit.iloc[0]["bonferroni_alpha"] == 0.05
    assert bool(audit.iloc[0]["pilot_gate"]) is True
    assert audit.iloc[0]["audit_lower_stress_pips"] > 0


def test_fast137_status_never_approves_strategy() -> None:
    status, reasons = fast137_status(passed_count=1, finalists_count=1)

    assert status == "FAST137_FX_DAILY_DONCHIAN_FADE_TEMPORAL_REVERSE_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

    status, reasons = fast137_status(passed_count=0, finalists_count=1)

    assert status == "FAST137_FX_DAILY_DONCHIAN_FADE_TEMPORAL_REVERSE_FAILED_CURRENT_AUDIT"
    assert "pre2019_positive_candidates_failed_2019_2026_audit" in reasons
