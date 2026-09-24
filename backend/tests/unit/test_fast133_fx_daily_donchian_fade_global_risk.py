from __future__ import annotations

import pandas as pd

from scripts.research.run_fast129_fx_daily_donchian_fade_scout import policy_grid as entry_policy_grid
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import risk_policy_grid
from scripts.research.run_fast133_fx_daily_donchian_fade_global_risk import (
    fast133_status,
    select_finalists,
    summarize_candidates,
)


def _row(
    *,
    candidate_id: str,
    entry_timestamp: str,
    net_base_pips: float = 4.0,
    net_stress_pips: float = 3.0,
) -> dict:
    timestamp = pd.Timestamp(entry_timestamp)
    return {
        "candidate_id": candidate_id,
        "pair": "NZDUSD",
        "entry_policy_id": "d1_donchian_fade_ch100_hold20",
        "risk_policy_id": "stop2_fixed_2r",
        "channel_days": 100,
        "hold_days": 20,
        "stop_atr_multiple": 2.0,
        "management": "fixed_2r",
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


def test_global_candidate_grid_size() -> None:
    assert len(entry_policy_grid()) == 9
    assert len(risk_policy_grid()) == 12
    assert len(entry_policy_grid()) * len(risk_policy_grid()) * 10 == 1080


def test_summarize_candidates_and_select_finalists() -> None:
    rows = [
        _row(candidate_id="NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_2r", entry_timestamp=f"2023-01-{day:02d}T00:00:00Z")
        for day in range(1, 31)
    ]
    rows.extend(
        [
                _row(
                    candidate_id="EURUSD:d1_donchian_fade_ch20_hold5:stop1_fixed_1r",
                entry_timestamp=f"2023-03-{day:02d}T00:00:00Z",
                net_base_pips=-1.0,
                net_stress_pips=-2.0,
            )
            for day in range(1, 31)
        ]
    )

    summary = summarize_candidates(pd.DataFrame(rows))
    finalists = select_finalists(summary)

    assert len(summary) == 2
    assert len(finalists) == 1
    assert finalists.iloc[0]["candidate_id"] == "NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_2r"


def test_fast133_status_never_approves_strategy() -> None:
    status, reasons = fast133_status(passed_count=1, finalists_count=1)

    assert status == "FAST133_FX_DAILY_DONCHIAN_FADE_GLOBAL_RISK_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

    status, reasons = fast133_status(passed_count=0, finalists_count=1)

    assert status == "FAST133_FX_DAILY_DONCHIAN_FADE_GLOBAL_RISK_FAILED_AUDIT"
    assert "selection_positive_candidates_failed_2024_2026_audit" in reasons
