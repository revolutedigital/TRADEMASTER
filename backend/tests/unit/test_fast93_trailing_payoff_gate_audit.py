from scripts.research.run_fast93_trailing_payoff_gate_audit import (
    GateSpec,
    apply_gate,
    fast93_status,
    holdout_status,
    safe_feature,
    train_candidate_score,
    training_summary_survives,
)


def test_safe_feature_reads_threshold_distance() -> None:
    row = {"threshold": 0.90, "probability": 0.84}

    assert abs(safe_feature(row, "threshold_distance") - 0.06) < 1e-12


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


def test_training_summary_requires_positive_stress_and_no_negative_labels_by_default() -> None:
    assert training_summary_survives(
        summary(stress=1.0, negative_labels=0),
        minimum_train_trades=10,
        minimum_train_labels=3,
        maximum_train_negative_labels=0,
    )
    assert not training_summary_survives(
        summary(stress=-1.0, negative_labels=0),
        minimum_train_trades=10,
        minimum_train_labels=3,
        maximum_train_negative_labels=0,
    )
    assert not training_summary_survives(
        summary(stress=1.0, negative_labels=1),
        minimum_train_trades=10,
        minimum_train_labels=3,
        maximum_train_negative_labels=0,
    )


def test_train_candidate_score_penalizes_negative_labels() -> None:
    clean = summary(stress=10.0, negative_labels=0)
    dirty = summary(stress=10.0, negative_labels=1)

    assert train_candidate_score(clean) > train_candidate_score(dirty)


def test_holdout_status_requires_trade_and_positive_stress() -> None:
    assert holdout_status(summary(trades=0, stress=1.0), minimum_holdout_trades=1) == (
        "FAST93_HOLDOUT_INSUFFICIENT_TRADES"
    )
    assert holdout_status(summary(trades=1, stress=-1.0), minimum_holdout_trades=1) == (
        "FAST93_HOLDOUT_STRESS_NON_POSITIVE"
    )
    assert holdout_status(summary(trades=1, stress=1.0), minimum_holdout_trades=1) == (
        "FAST93_HOLDOUT_POSITIVE"
    )


def test_fast93_status_reports_no_training_survivors() -> None:
    status, reasons = fast93_status(
        {
            "holdout_count": 2,
            "positive_holdout_count": 0,
            "negative_holdout_count": 0,
            "insufficient_holdout_count": 0,
            "no_strict_training_survivor_count": 2,
            "maximum_train_negative_labels": 0,
        }
    )

    assert status == "FAST93_NO_STRICT_TRAINING_SURVIVORS"
    assert "no_gate_positive_in_training_under_negative_label_limit:0" in reasons


def test_fast93_status_marks_diagnostic_survivor_only_when_all_holdouts_positive() -> None:
    status, reasons = fast93_status(
        {
            "holdout_count": 2,
            "positive_holdout_count": 2,
            "negative_holdout_count": 0,
            "insufficient_holdout_count": 0,
            "no_strict_training_survivor_count": 0,
        }
    )

    assert status == "FAST93_TRAILING_PAYOFF_GATE_SURVIVOR_DIAGNOSTIC_ONLY"
    assert "requires_new_preregistration" in reasons


def row(probability: float) -> dict[str, object]:
    return {"probability": probability, "threshold": 0.9}


def summary(
    *,
    trades: int = 10,
    labels: int = 3,
    stress: float = 10.0,
    negative_labels: int = 0,
) -> dict[str, object]:
    return {
        "trade_count": trades,
        "label_count": labels,
        "stress_total_bps": stress,
        "stress_mean_bps": stress / trades if trades else 0.0,
        "negative_label_count": negative_labels,
    }
