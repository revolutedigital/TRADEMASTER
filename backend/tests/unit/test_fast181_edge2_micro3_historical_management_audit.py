from __future__ import annotations

import pandas as pd

from scripts.research.run_fast174_crypto_cross_asset_leadlag_smoke import EXTENSION_BLOCK, VALIDATION_BLOCK
from scripts.research.run_fast181_edge2_micro3_historical_management_audit import (
    fast181_status,
    summarize_rows,
)


def test_summarize_rows_uses_stress_and_terminal_totals() -> None:
    rows = pd.DataFrame(
        [
            {"stress_net_bps": 2.0, "terminal_net_bps": -3.0, "exit_reason": "BREAKEVEN_STOP"},
            {"stress_net_bps": -1.0, "terminal_net_bps": 4.0, "exit_reason": "INITIAL_STOP"},
        ]
    )

    summary = summarize_rows(rows, block="fast-test")

    assert summary["block"] == "fast-test"
    assert summary["trade_count"] == 2
    assert summary["stress_total_bps"] == 1.0
    assert summary["terminal_total_bps"] == 1.0
    assert summary["stress_win_rate"] == 0.5
    assert summary["exit_reason_counts"] == {"BREAKEVEN_STOP": 1, "INITIAL_STOP": 1}


def test_fast181_status_requires_validation_and_extension_positive() -> None:
    status, reasons = fast181_status(
        [
            {
                "edge_threshold_bps": 10.0,
                "blocks": [
                    {"block": VALIDATION_BLOCK, "stress_total_bps": 2.0},
                    {"block": EXTENSION_BLOCK, "stress_total_bps": -1.0},
                ],
            }
        ]
    )

    assert status == "FAST181_EDGE2_MICRO3_FAILED_EXTENSION"
    assert "extension_not_positive" in reasons


def test_fast181_status_marks_strict_historical_candidate() -> None:
    status, reasons = fast181_status(
        [
            {
                "edge_threshold_bps": 12.0,
                "blocks": [
                    {"block": VALIDATION_BLOCK, "stress_total_bps": 2.0},
                    {"block": EXTENSION_BLOCK, "stress_total_bps": 1.0},
                ],
            }
        ]
    )

    assert status == "FAST181_EDGE2_MICRO3_HISTORICAL_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "fresh_future_preregistration_required" in reasons
