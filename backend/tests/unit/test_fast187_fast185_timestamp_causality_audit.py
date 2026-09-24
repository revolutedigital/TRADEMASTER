from __future__ import annotations

from scripts.research.run_fast187_fast185_timestamp_causality_audit import fast187_status


def test_fast187_rejects_candidate_when_close_delay_fails() -> None:
    status, reasons = fast187_status(
        [
            {
                "entry_delay_seconds": 0,
                "stress_total_bps": 100.0,
                "positive_day_count": 9,
            },
            {
                "entry_delay_seconds": 60,
                "stress_total_bps": -1.0,
                "positive_day_count": 3,
            },
        ]
    )

    assert status == "FAST187_REJECTS_FAST185_UNDER_CAUSAL_CLOSE_DELAY"
    assert "feature_timestamp_or_entry_alignment_must_be_fixed_before_strategy_approval" in reasons
