from __future__ import annotations

from scripts.research.run_fast183_fast177_edge7p5_micro8_management_observation import (
    PREREGISTRATION_TIMESTAMP,
    fast183_status,
    filter_post_preregistration_rows,
)


def test_filter_post_preregistration_rows_is_strict() -> None:
    rows = [
        {"decision_time": PREREGISTRATION_TIMESTAMP.isoformat()},
        {"decision_time": "2026-09-24T14:35:00+00:00"},
    ]

    assert [row["decision_time"] for row in filter_post_preregistration_rows(rows)] == [
        "2026-09-24T14:35:00+00:00"
    ]


def test_fast183_waits_for_future_rows() -> None:
    status, reasons = fast183_status(
        prospective_row_count=0,
        observation_count=0,
        gates={"early_fail": False, "shadow_candidate": False, "shadow_reasons": []},
    )

    assert status == "FAST183_WAITING_FOR_POST_PREREGISTRATION_DATA"
    assert "no_post_preregistration_decision_rows" in reasons
