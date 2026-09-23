from scripts.research.run_fast94_tail_risk_utility_model_audit import (
    apply_one_position,
    fast94_status,
    feature_value,
    holdout_status,
    score_cutoff,
    target_value,
    training_summary_survives,
)


def test_target_value_penalizes_tail_risk_and_initial_stop() -> None:
    base = settlement(stress=-1.0, maximum_adverse=-10.0, exit_reason="INITIAL_STOP")

    assert target_value(base, "stress") == -1.0
    assert target_value(base, "tail05") == -8.5
    assert target_value(base, "tail10") == -14.0
    assert target_value(base, "positive_safe") == 0.0


def test_positive_safe_requires_positive_stress_and_no_large_adverse_tail() -> None:
    assert target_value(settlement(stress=1.0, maximum_adverse=-7.9), "positive_safe") == 1.0
    assert target_value(settlement(stress=1.0, maximum_adverse=-8.1), "positive_safe") == 0.0


def test_feature_value_reads_threshold_distance_and_causal_features() -> None:
    row = {
        "probability": 0.84,
        "threshold": 0.90,
        "causal_feature_values": {"trade_count_30s": 123},
    }

    assert abs(feature_value(row, "threshold_distance") - 0.06) < 1e-12
    assert feature_value(row, "trade_count_30s") == 123.0


def test_score_cutoff_uses_training_distribution() -> None:
    assert score_cutoff([1.0, 2.0, 3.0, 4.0], 0.25) == 3.25


def test_apply_one_position_skips_overlapping_rows() -> None:
    rows = [
        decision(1000, 2000, 0.9),
        decision(1500, 2500, 0.95),
        decision(2501, 3000, 0.8),
    ]

    selected = apply_one_position(rows)

    assert [row["decision_time_ms"] for row in selected] == [1000, 2501]


def test_training_summary_survival_gate() -> None:
    assert training_summary_survives(summary(stress=1.0, negative_labels=2))
    assert not training_summary_survives(summary(stress=-1.0, negative_labels=0))
    assert not training_summary_survives(summary(stress=1.0, negative_labels=3))


def test_holdout_status_requires_trade_and_positive_stress() -> None:
    assert holdout_status(summary(trades=0, stress=1.0)) == "FAST94_HOLDOUT_EMPTY"
    assert holdout_status(summary(trades=1, stress=-1.0)) == "FAST94_HOLDOUT_STRESS_NON_POSITIVE"
    assert holdout_status(summary(trades=1, stress=1.0)) == "FAST94_HOLDOUT_POSITIVE"


def test_fast94_status_rejects_partial_holdout_survivor() -> None:
    status, reasons = fast94_status(
        [
                {
                    "model_name": "candidate",
                    "target_name": "stress",
                    "model_kind": "regressor",
                    "top_fraction": 0.1,
                    "aggregate": {
                        "holdout_count": 2,
                        "train_survivor_count": 1,
                        "holdout_trade_count": 2,
                        "positive_holdout_count": 1,
                        "negative_holdout_count": 1,
                        "empty_holdout_count": 0,
                        "holdout_stress_total_bps": -1.0,
                        "worst_holdout_stress_total_bps": -2.0,
                    },
                }
            ]
        )

    assert status == "FAST94_UTILITY_MODEL_FAILED_LOLO"
    assert "best_positive_holdouts:1/2" in reasons


def settlement(
    *,
    stress: float,
    maximum_adverse: float,
    exit_reason: str = "TIMEOUT",
) -> dict[str, object]:
    return {
        "stress_net_bps": stress,
        "maximum_adverse_bps": maximum_adverse,
        "exit_reason": exit_reason,
    }


def decision(decision_time_ms: int, exit_time_ms: int, score: float) -> dict[str, object]:
    return {
        "decision_time_ms": decision_time_ms,
        "exit_time_ms": exit_time_ms,
        "_fast94_score": score,
    }


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
        "negative_label_count": negative_labels,
    }
