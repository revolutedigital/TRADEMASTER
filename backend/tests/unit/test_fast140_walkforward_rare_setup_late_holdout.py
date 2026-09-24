from __future__ import annotations

from scripts.research.run_fast140_walkforward_rare_setup_late_holdout import (
    Condition,
    Rule,
    apply_one_position_by_label,
    fast140_status,
    is_late_survivor,
    rule_matches,
)


def _row(
    *,
    label: str = "prospective-wal-fast82-30m",
    decision_time_ms: int = 1_000,
    exit_time_ms: int = 2_000,
    feature: float = 2.0,
    stress_net_bps: float = 3.0,
) -> dict:
    return {
        "label": label,
        "decision_time_ms": decision_time_ms,
        "exit_time_ms": exit_time_ms,
        "probability": 0.8,
        "threshold": 0.9,
        "causal_feature_values": {"alpha_feature": feature},
        "expected_net_bps": stress_net_bps + 1.0,
        "stress_net_bps": stress_net_bps,
    }


def test_rule_matches_nested_causal_feature_thresholds() -> None:
    rule = Rule((Condition("alpha_feature", "high", 1.5, 0.8),))

    assert rule_matches(rule, _row(feature=2.0))
    assert not rule_matches(rule, _row(feature=1.0))


def test_apply_one_position_by_label_keeps_independent_labels() -> None:
    rows = [
        _row(label="a", decision_time_ms=1_000, exit_time_ms=3_000),
        _row(label="a", decision_time_ms=2_000, exit_time_ms=4_000),
        _row(label="b", decision_time_ms=2_000, exit_time_ms=4_000),
    ]

    selected = apply_one_position_by_label(rows)

    assert [(row["label"], row["decision_time_ms"]) for row in selected] == [("a", 1_000), ("b", 2_000)]


def test_late_survivor_requires_trade_label_and_stress_gates() -> None:
    evaluation = {
        "late_summary": {
            "trade_count": 5,
            "label_count": 2,
            "stress_total_bps": 10.0,
            "negative_label_count": 0,
        }
    }

    assert is_late_survivor(evaluation, min_late_trades=5, min_late_labels=2)

    evaluation["late_summary"]["negative_label_count"] = 1
    assert not is_late_survivor(evaluation, min_late_trades=5, min_late_labels=2)


def test_fast140_status_never_approves_candidate() -> None:
    status, reasons = fast140_status(
        finalist_count=1,
        late_survivor_count=1,
        finalists=[
            {
                "late_summary": {
                    "trade_count": 5,
                    "label_count": 2,
                    "stress_total_bps": 10.0,
                    "negative_label_count": 0,
                }
            }
        ],
        min_late_trades=5,
        min_late_labels=2,
    )

    assert status == "FAST140_RARE_SETUP_LATE_HOLDOUT_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons
