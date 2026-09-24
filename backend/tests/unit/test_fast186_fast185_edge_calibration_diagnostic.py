from __future__ import annotations

from scripts.research.run_fast184_robust_session_filter_historical_audit import POST_TRAINING_DATES
from scripts.research.run_fast186_fast185_edge_calibration_diagnostic import (
    fast186_status,
    live_row_passes_threshold,
)


def test_live_row_passes_threshold_requires_all_fast185_gates() -> None:
    row = {
        "edge": 2.5,
        "fast185_regime_gate_passed": True,
        "fast185_session_gate_passed": True,
        "fast185_side_gate_passed": True,
    }

    assert live_row_passes_threshold(row, 2.5) is True
    assert live_row_passes_threshold({**row, "edge": 2.49}, 2.5) is False
    assert live_row_passes_threshold({**row, "fast185_session_gate_passed": False}, 2.5) is False


def test_fast186_rejects_simple_edge_lowering_without_lower_robust_survivor() -> None:
    status, reasons = fast186_status(
        historical_summaries=[
            {
                "edge_threshold_bps": 2.5,
                "positive_day_count": 6,
                "worst_day_stress_total_bps": -100.0,
            },
            {
                "edge_threshold_bps": 7.5,
                "positive_day_count": len(POST_TRAINING_DATES),
                "worst_day_stress_total_bps": 1.7,
            },
        ],
        live_threshold_summaries=[
            {"edge_threshold_bps": 2.5, "row_count": 2},
            {"edge_threshold_bps": 7.5, "row_count": 0},
        ],
    )

    assert status == "FAST186_REJECTS_SIMPLE_EDGE_LOWERING"
    assert "keep_fast185_gate_until_new_filter_is_found" in reasons
