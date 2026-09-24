import pandas as pd

from scripts.research.run_fast106_long_horizon_lowcost_scout import (
    exact_target_net_bps,
    fast106_status,
    select_top_one_position,
)


def test_exact_target_net_uses_target_specific_stop_order() -> None:
    rows = pd.DataFrame(
        {
            "first_touch": ["STOP", "BREAKEVEN", "TIMEOUT"],
            "paid_expected_before_stop": [False, True, False],
            "terminal_bps": [4.0, 5.0, 3.0],
        }
    )

    net = exact_target_net_bps(
        rows,
        target_bps=12.0,
        explicit_cost_bps=2.0,
        initial_stop_bps=20.0,
    )

    assert net.tolist() == [-22.0, 10.0, 1.0]


def test_select_top_one_position_skips_overlapping_horizon() -> None:
    rows = pd.DataFrame(
        {
            "decision_time_ms": [0, 10_000, 70_000],
            "score": [3.0, 5.0, 4.0],
        }
    )

    selected = select_top_one_position(
        rows,
        score_threshold_value=0.0,
        horizon_seconds=60,
    )

    assert selected["decision_time_ms"].tolist() == [0, 70_000]


def test_fast106_status_rejects_screen_survivor_that_fails_extension() -> None:
    status, reasons = fast106_status(
        screen_survivors=[{"candidate": "x"}],
        validations=[
            {
                "historical": {
                    "negative_sufficient_blocks": [],
                    "net_total_bps": 10.0,
                },
                "extension": {"trade_count": 3, "net_total_bps": -1.0},
            }
        ],
    )

    assert status == "FAST106_LONG_HORIZON_FAILED_WALKFORWARD_OR_EXTENSION"
    assert "no_screen_survivor_passed_walkforward_and_extension" in reasons
