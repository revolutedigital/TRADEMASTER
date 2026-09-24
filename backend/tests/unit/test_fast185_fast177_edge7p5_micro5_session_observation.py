from __future__ import annotations

from scripts.research.run_fast185_fast177_edge7p5_micro5_session_observation import (
    PREREGISTRATION_TIMESTAMP,
    fast185_status,
    filter_post_preregistration_rows,
    with_fast185_eligibility,
)


def test_filter_post_preregistration_rows_is_strict() -> None:
    rows = [
        {"decision_time": PREREGISTRATION_TIMESTAMP.isoformat()},
        {"decision_time": "2026-09-24T14:55:00+00:00"},
    ]

    assert [row["decision_time"] for row in filter_post_preregistration_rows(rows)] == [
        "2026-09-24T14:55:00+00:00"
    ]


def test_with_fast185_eligibility_requires_session_edge_regime_and_side() -> None:
    base = {
        "decision_time": "2026-09-24T05:10:00+00:00",
        "edge": 7.5,
        "regime_gate_value": 100_000_000.0,
        "side": "LONG",
    }

    eligible = with_fast185_eligibility(base)
    outside_session = with_fast185_eligibility({**base, "decision_time": "2026-09-24T17:10:00+00:00"})
    low_edge = with_fast185_eligibility({**base, "edge": 7.49})
    low_regime = with_fast185_eligibility({**base, "regime_gate_value": 99_999_999.0})
    no_side = with_fast185_eligibility({**base, "side": "FLAT"})

    assert eligible["fast185_eligible"] is True
    assert outside_session["fast185_eligible"] is False
    assert low_edge["fast185_eligible"] is False
    assert low_regime["fast185_eligible"] is False
    assert no_side["fast185_eligible"] is False


def test_fast185_waits_for_future_rows() -> None:
    status, reasons = fast185_status(
        prospective_row_count=0,
        observation_count=0,
        gates={"early_fail": False, "shadow_candidate": False, "shadow_reasons": []},
    )

    assert status == "FAST185_WAITING_FOR_POST_PREREGISTRATION_DATA"
    assert "no_post_preregistration_decision_rows" in reasons
