import pandas as pd

from scripts.research.run_fast79_rare_pair_setup_audit import (
    Condition,
    apply_conditions,
    evaluate_candidate,
    fast79_status,
)


def test_apply_conditions_filters_high_and_low_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [3.0, 2.0, 1.0],
        }
    )

    filtered = apply_conditions(
        frame,
        (
            Condition("feature_a", "high", 2.0, 0.50),
            Condition("feature_b", "low", 2.0, 0.50),
        ),
    )

    assert filtered.index.tolist() == [1, 2]


def test_evaluate_candidate_counts_only_sufficient_negative_blocks() -> None:
    frame = pd.DataFrame(
        {
            "block": ["a", "a", "b", "b", "b"],
            "decision_time_ms": [1, 3, 1, 3, 5],
            "entry_time_ms": [2, 4, 2, 4, 6],
            "exit_time_ms": [2, 4, 2, 4, 6],
            "probability": [0.9, 0.8, 0.9, 0.8, 0.7],
            "expected_net_bps": [1.0, 1.0, -1.0, -1.0, -1.0],
            "stress_net_bps": [1.0, 1.0, -1.0, -1.0, -1.0],
            "feature_a": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    result = evaluate_candidate(
        frame,
        conditions=(Condition("feature_a", "high", 0.0, 0.50),),
        minimum_block_trades=3,
    )

    assert result["sufficient_block_count"] == 1
    assert result["negative_sufficient_block_count"] == 1


def test_fast79_status_keeps_extension_survivor_as_preregistration_only() -> None:
    status, reasons = fast79_status(
        historical_survivor_count=11,
        extension_survivor_count=1,
    )

    assert status == "FAST79_RARE_PAIR_SETUP_WEAK_SURVIVORS_REQUIRE_PREREGISTRATION"
    assert "staged_search_multiple_testing_not_strategy" in reasons
