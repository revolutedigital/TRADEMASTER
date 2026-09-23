from scripts.research.run_fast90_near_threshold_scout import (
    fast90_status,
    parse_probability_margins,
    passes_predicate,
    passes_probability_margin,
    probability_margin_name,
    variant_specs,
)


def test_parse_probability_margins_sorts_and_deduplicates_values() -> None:
    assert parse_probability_margins("0.05,0.02,0.05") == (0.02, 0.05)


def test_probability_margin_name_is_stable_for_report_keys() -> None:
    assert probability_margin_name(0.05) == "050"


def test_passes_probability_margin_uses_row_threshold() -> None:
    assert passes_probability_margin(row(probability=0.86, threshold=0.90), 0.05)
    assert not passes_probability_margin(row(probability=0.84, threshold=0.90), 0.05)


def test_predicates_use_side_aligned_features() -> None:
    assert passes_predicate(row(directed_flow_imbalance_5s=0.1), "flow5_positive")
    assert not passes_predicate(row(directed_flow_imbalance_5s=-0.1), "flow5_positive")
    assert passes_predicate(
        row(directed_return_5s_bps=0.1, directed_flow_imbalance_5s=0.1),
        "return5_and_flow5_positive",
    )
    assert not passes_predicate(
        row(directed_return_5s_bps=-0.1, directed_flow_imbalance_5s=0.1),
        "return5_and_flow5_positive",
    )


def test_variant_specs_include_directional_near_threshold_cases() -> None:
    names = {spec.name for spec in variant_specs((0.05,))}

    assert "near050_all_top_probability" in names
    assert "near050_flow5_positive_top_probability" in names
    assert "near050_vol15_flow5_positive_veto" in names


def test_fast90_status_rejects_positive_but_tiny_samples() -> None:
    status, reasons = fast90_status(
        candidate_universe_count=3,
        complete_outcome_count=3,
        variants=[variant(trades=3, labels=2, stress_total=10.0, negative_labels=0)],
    )

    assert status == "FAST90_NO_NEAR_THRESHOLD_EDGE_FOUND"
    assert "no_variant_positive_with_minimum_sample_and_no_negative_label" in reasons


def test_fast90_status_marks_diagnostic_candidate_only() -> None:
    status, reasons = fast90_status(
        candidate_universe_count=12,
        complete_outcome_count=12,
        variants=[variant(trades=10, labels=2, stress_total=1.0, negative_labels=0)],
    )

    assert status == "FAST90_NEAR_THRESHOLD_CANDIDATE_DIAGNOSTIC_ONLY"
    assert "requires_new_preregistration_before_any_reuse" in reasons


def row(
    *,
    probability: float = 0.90,
    threshold: float = 0.91,
    directed_flow_imbalance_5s: float = 0.0,
    directed_return_5s_bps: float = 0.0,
    realized_vol_30s_bps: float = 20.0,
) -> dict[str, object]:
    return {
        "probability": probability,
        "threshold": threshold,
        "causal_feature_values": {
            "directed_flow_imbalance_5s": directed_flow_imbalance_5s,
            "directed_return_5s_bps": directed_return_5s_bps,
            "realized_vol_30s_bps": realized_vol_30s_bps,
        },
    }


def variant(
    *,
    trades: int,
    labels: int,
    stress_total: float,
    negative_labels: int,
) -> dict[str, object]:
    return {
        "one_position_summary": {
            "trade_count": trades,
            "label_count": labels,
            "stress_total_bps": stress_total,
            "stress_mean_bps": stress_total / trades,
            "negative_label_count": negative_labels,
        }
    }
