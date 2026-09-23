import pandas as pd

from scripts.research.run_fast75_lowcost_candidate_universe_scout import (
    approx_path_net_bps,
    fast75_status,
    select_top_one_position,
)


def test_approx_path_net_bps_charges_stop_target_and_terminal() -> None:
    values = approx_path_net_bps(
        first_touch=pd.Series(["STOP", "BREAKEVEN", "TIMEOUT"]),
        mfe_bps=pd.Series([12.0, 9.0, 3.0]),
        terminal_bps=pd.Series([5.0, 1.0, 2.5]),
        target_bps=8.0,
        explicit_cost_bps=1.0,
        initial_stop_bps=20.0,
    )

    assert values.tolist() == [-21.0, 7.0, 1.5]


def test_select_top_one_position_keeps_highest_score_without_overlap() -> None:
    frame = pd.DataFrame(
        {
            "decision_time_ms": [1_000, 1_000, 2_000, 400_000],
            "score": [0.8, 0.9, 0.95, 0.7],
            "net_bps": [1.0, 2.0, 3.0, 4.0],
        }
    )

    selected = select_top_one_position(
        frame,
        score_threshold_value=0.7,
        horizon_seconds=300,
    )

    assert selected["decision_time_ms"].tolist() == [1_000, 400_000]
    assert selected["score"].tolist() == [0.9, 0.7]


def test_fast75_status_fails_validation_survivor_with_negative_blocks() -> None:
    status, reasons = fast75_status(
        screen_survivors=[{"horizon_seconds": 300}],
        validations=[
            {
                "historical": {
                    "negative_sufficient_blocks": ["block-a"],
                    "net_total_bps": 1.0,
                },
                "extension": {
                    "trade_count": 10,
                    "net_total_bps": 5.0,
                },
            }
        ],
    )

    assert status == "FAST75_LOWCOST_UNIVERSE_FAILED_VALIDATION"
    assert "no_screen_survivor_passed_denser_walkforward_and_extension" in reasons


def test_fast75_status_fails_empty_screen() -> None:
    status, reasons = fast75_status(screen_survivors=[], validations=[])

    assert status == "FAST75_LOWCOST_UNIVERSE_FAILED_SCREEN"
    assert "no_screen_survivor" in reasons
