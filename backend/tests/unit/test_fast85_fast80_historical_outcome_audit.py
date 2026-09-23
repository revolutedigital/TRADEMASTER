from scripts.research.run_fast85_fast80_historical_outcome_audit import (
    fast85_status,
    group_by_label,
    is_fast80_top_p_rule_match,
    normalize_candidate,
)


def test_fast85_filters_only_fast80_rule_matches_that_would_enter() -> None:
    assert is_fast80_top_p_rule_match(
        {"would_enter": True, "fast80_rule_match_diagnostic": True}
    )
    assert not is_fast80_top_p_rule_match(
        {"would_enter": False, "fast80_rule_match_diagnostic": True}
    )
    assert not is_fast80_top_p_rule_match(
        {"would_enter": True, "fast80_rule_match_diagnostic": False}
    )


def test_fast85_candidate_normalization_keeps_research_safety_flags() -> None:
    normalized = normalize_candidate(
        {
            "label": "fast",
            "would_enter": True,
            "fast80_rule_match_diagnostic": True,
        }
    )

    assert normalized["fast85_historical_diagnostic_only"] is True
    assert normalized["approved_strategy"] is False
    assert normalized["order_submission_allowed"] is False
    assert normalized["execution_authorization"] == "none"


def test_fast85_groups_candidates_by_label() -> None:
    grouped = group_by_label(
        [
            {"label": "a", "decision_time_ms": 1},
            {"label": "b", "decision_time_ms": 2},
            {"label": "a", "decision_time_ms": 3},
        ]
    )

    assert [row["decision_time_ms"] for row in grouped["a"]] == [1, 3]
    assert [row["decision_time_ms"] for row in grouped["b"]] == [2]


def test_fast85_status_fails_when_stress_total_is_negative() -> None:
    status, reasons = fast85_status(
        aggregate={
            "candidate_count": 2,
            "complete_outcome_count": 2,
        },
        independent_summary={"stress_total_log_return_bps": -1.0},
        one_position_summary={"stress_total_log_return_bps": 2.0},
    )

    assert status == "FAST85_FAST80_HISTORICAL_OUTCOME_FAILED"
    assert "stress_total_non_positive_after_trailing_replay" in reasons


def test_fast85_status_positive_is_still_diagnostic_only() -> None:
    status, reasons = fast85_status(
        aggregate={
            "candidate_count": 2,
            "complete_outcome_count": 2,
        },
        independent_summary={"stress_total_log_return_bps": 1.0},
        one_position_summary={"stress_total_log_return_bps": 2.0},
    )

    assert status == "FAST85_FAST80_HISTORICAL_OUTCOME_POSITIVE_DIAGNOSTIC_ONLY"
    assert "prospective_confirmation_still_required" in reasons
