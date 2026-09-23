from scripts.research.run_fast87_directional_filter_scout import (
    apply_conflict_policy,
    fast87_status,
    feature,
    group_by_decision_timestamp,
    normalize_candidate,
)


def test_feature_reads_top_level_then_causal_feature_values() -> None:
    assert feature({"x": 1.5, "causal_feature_values": {"x": 2.0}}, "x") == 1.5
    assert feature({"causal_feature_values": {"x": 2.0}}, "x") == 2.0


def test_apply_conflict_policy_veto_drops_buy_sell_timestamp() -> None:
    rows = [
        row("a", 1, "BUY", 0.9),
        row("a", 1, "SELL", 0.8),
        row("a", 2, "BUY", 0.7),
    ]

    selected = apply_conflict_policy(rows, "veto")

    assert [(item["decision_time_ms"], item["side"]) for item in selected] == [(2, "BUY")]


def test_apply_conflict_policy_top_probability_keeps_highest_probability() -> None:
    rows = [
        row("a", 1, "BUY", 0.9),
        row("a", 1, "SELL", 0.95),
        row("a", 2, "BUY", 0.7),
    ]

    selected = apply_conflict_policy(rows, "top_probability")

    assert [(item["decision_time_ms"], item["side"]) for item in selected] == [
        (1, "SELL"),
        (2, "BUY"),
    ]


def test_group_by_decision_timestamp_uses_label_and_time() -> None:
    grouped = group_by_decision_timestamp(
        [
            row("a", 1, "BUY", 0.9),
            row("b", 1, "BUY", 0.8),
        ]
    )

    assert sorted(grouped) == [("a", 1), ("b", 1)]


def test_fast87_status_requires_positive_stress_minimum_sample_and_no_negative_label() -> None:
    status, reasons = fast87_status(
        [
            variant("weak_positive", trade_count=4, stress_total_bps=10.0, negative_labels=0),
            variant("negative_label", trade_count=5, stress_total_bps=10.0, negative_labels=1),
        ]
    )

    assert status == "FAST87_NO_ROBUST_DIRECTIONAL_FILTER_FOUND"
    assert "no_variant_has_positive_stress_with_minimum_sample_and_no_negative_label" in reasons


def test_fast87_status_marks_candidate_as_diagnostic_only() -> None:
    status, reasons = fast87_status(
        [variant("candidate", trade_count=5, stress_total_bps=1.0, negative_labels=0)]
    )

    assert status == "FAST87_DIRECTIONAL_FILTER_CANDIDATE_DIAGNOSTIC_ONLY"
    assert "requires_new_preregistration" in reasons


def test_normalize_candidate_keeps_safety_flags() -> None:
    normalized = normalize_candidate({"label": "a"})

    assert normalized["fast87_directional_scout_only"] is True
    assert normalized["approved_strategy"] is False
    assert normalized["order_submission_allowed"] is False
    assert normalized["execution_authorization"] == "none"


def row(label: str, decision_time_ms: int, side: str, probability: float) -> dict[str, object]:
    return {
        "label": label,
        "decision_time_ms": decision_time_ms,
        "side": side,
        "probability": probability,
    }


def variant(
    name: str,
    *,
    trade_count: int,
    stress_total_bps: float,
    negative_labels: int,
) -> dict[str, object]:
    return {
        "name": name,
        "one_position_summary": {
            "trade_count": trade_count,
            "stress_total_bps": stress_total_bps,
            "negative_label_count": negative_labels,
        },
    }
