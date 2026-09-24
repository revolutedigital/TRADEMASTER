from __future__ import annotations

import pandas as pd

from scripts.research.run_fast182_edge_management_grid_historical_audit import (
    fast182_status,
    summarize_candidates,
)


def test_fast182_status_reports_strict_survivors() -> None:
    status, reasons = fast182_status(
        [
            {
                "strict_passed": True,
                "edge_threshold_bps": 7.5,
                "management_policy": "micro_8_3",
                "stress_total_bps": 10.0,
                "stress_mean_bps": 1.0,
                "trade_count": 5,
            }
        ]
    )

    assert status == "FAST182_EDGE_MANAGEMENT_GRID_STRICT_HISTORICAL_SURVIVORS"
    assert "fresh_future_preregistration_required" in reasons


def test_summarize_candidates_sorts_strict_survivors_first() -> None:
    rows = pd.DataFrame(
        [
            {"edge_threshold_bps": 7.5, "management_policy": "a", "block": "fast53", "stress_net_bps": 2.0, "terminal_net_bps": 0.0, "exit_reason": "X"},
            {"edge_threshold_bps": 7.5, "management_policy": "a", "block": "fast56", "stress_net_bps": 1.0, "terminal_net_bps": 0.0, "exit_reason": "X"},
            {"edge_threshold_bps": 2.0, "management_policy": "b", "block": "fast53", "stress_net_bps": 9.0, "terminal_net_bps": 0.0, "exit_reason": "X"},
            {"edge_threshold_bps": 2.0, "management_policy": "b", "block": "fast56", "stress_net_bps": -1.0, "terminal_net_bps": 0.0, "exit_reason": "X"},
        ]
    ).to_dict("records")

    summaries = summarize_candidates(rows)

    assert summaries[0]["management_policy"] == "a"
    assert summaries[0]["strict_passed"] is True
