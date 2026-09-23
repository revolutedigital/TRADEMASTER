from scripts.research.run_fast91_lolo_gate_audit import (
    BaseSpec,
    GateSpec,
    aggregate_holdout_results,
    apply_gate,
    fast91_status,
    holdout_status,
    nearest_rank_quantile,
    safe_feature,
    train_candidate_score,
)


def test_safe_feature_reads_threshold_distance() -> None:
    assert abs(safe_feature({"threshold": 0.9, "probability": 0.84}, "threshold_distance") - 0.06) < 1e-12


def test_apply_gate_filters_gte_and_lte() -> None:
    rows = [row(1.0), row(2.0), row(3.0)]

    assert [item["probability"] for item in apply_gate(rows, GateSpec("probability", "gte", 0.5, 2.0))] == [
        2.0,
        3.0,
    ]
    assert [item["probability"] for item in apply_gate(rows, GateSpec("probability", "lte", 0.5, 2.0))] == [
        1.0,
        2.0,
    ]


def test_nearest_rank_quantile_uses_sorted_position() -> None:
    assert nearest_rank_quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0


def test_train_candidate_score_penalizes_negative_labels() -> None:
    clean = summary(stress=10.0, negative_labels=0)
    dirty = summary(stress=10.0, negative_labels=1)

    assert train_candidate_score(clean) > train_candidate_score(dirty)


def test_holdout_status_requires_trades_and_positive_stress() -> None:
    assert holdout_status(summary(trades=0, stress=10.0), minimum_holdout_trades=1) == (
        "FAST91_HOLDOUT_INSUFFICIENT_TRADES"
    )
    assert holdout_status(summary(trades=1, stress=-1.0), minimum_holdout_trades=1) == (
        "FAST91_HOLDOUT_STRESS_NON_POSITIVE"
    )
    assert holdout_status(summary(trades=1, stress=1.0), minimum_holdout_trades=1) == (
        "FAST91_HOLDOUT_POSITIVE"
    )


def test_fast91_status_fails_when_not_all_holdouts_are_positive() -> None:
    status, reasons = fast91_status(
        {
            "holdout_count": 2,
            "positive_holdout_count": 1,
            "negative_holdout_count": 1,
            "insufficient_holdout_count": 0,
        }
    )

    assert status == "FAST91_LOLO_GATE_FAILED"
    assert "positive_holdouts:1/2" in reasons


def test_aggregate_holdout_results_counts_statuses() -> None:
    aggregate = aggregate_holdout_results(
        [
            holdout("a", "FAST91_HOLDOUT_POSITIVE", 2.0, 1),
            holdout("b", "FAST91_HOLDOUT_STRESS_NON_POSITIVE", -1.0, 1),
        ]
    )

    assert aggregate["positive_holdout_count"] == 1
    assert aggregate["negative_holdout_count"] == 1
    assert aggregate["holdout_stress_total_bps"] == 1.0


def row(probability: float) -> dict[str, object]:
    return {"probability": probability, "threshold": 0.9}


def summary(
    *,
    trades: int = 10,
    stress: float = 10.0,
    negative_labels: int = 0,
) -> dict[str, object]:
    return {
        "trade_count": trades,
        "label_count": 3,
        "stress_total_bps": stress,
        "negative_label_count": negative_labels,
    }


def holdout(label: str, status: str, stress: float, trades: int) -> dict[str, object]:
    return {
        "holdout_label": label,
        "status": status,
        "holdout_summary": {
            "stress_total_bps": stress,
            "trade_count": trades,
        },
    }


def test_base_spec_name_is_stable() -> None:
    assert BaseSpec(0.05, "all", "veto").name == "near050_all_veto"
