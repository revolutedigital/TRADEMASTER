import json

import pandas as pd

from scripts.research.run_fast74_non_stop_model_audit import (
    NON_STOP_SCORE_COLUMN,
    apply_score_volume_density_gate,
    evaluate_sep23_scoreability,
    fast74_status,
    score_threshold_from_training_scores,
)


def test_score_threshold_uses_upper_tail_of_training_scores() -> None:
    threshold = score_threshold_from_training_scores(
        pd.Series([0.10, 0.20, 0.30, 0.40, 0.50]),
        score_tail=0.20,
    )

    assert threshold == 0.42000000000000004


def test_apply_score_volume_density_gate_filters_before_density() -> None:
    frame = pd.DataFrame(
        {
            "decision_time_ms": [1_000, 2_000, 3_000, 310_000],
            "probability": [0.80, 0.90, 0.70, 0.60],
            "quote_volume_30s": [200.0, 200.0, 50.0, 200.0],
            NON_STOP_SCORE_COLUMN: [0.95, 0.96, 0.99, 0.97],
        }
    )

    filtered = apply_score_volume_density_gate(
        frame,
        quote_volume_threshold=100.0,
        score_threshold=0.90,
        density_lookback_seconds=300,
        density_limit=1,
    )

    assert filtered["decision_time_ms"].tolist() == [1_000, 2_000, 310_000]
    assert filtered["recent_filtered_signal_count"].tolist() == [0, 1, 0]


def test_fast74_status_fails_extension_before_promotion() -> None:
    status, reasons = fast74_status(
        honest_historical_aggregate={"negative_sufficient_blocks": []},
        extension_aggregate={
            "negative_sufficient_blocks": ["extension"],
            "stress_total_log_return_bps": 10.0,
        },
        sep23={"available": False},
    )

    assert status == "FAST74_NON_STOP_MODEL_FAILED_EXTENSION"
    assert "extension_negative_blocks:1" in reasons


def test_sep23_scoreability_reports_missing_full_feature_vector(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    settlement_path = tmp_path / "settlements.jsonl"
    raw_path.write_text(
        json.dumps({"directional_gate_values": {"quote_volume_30s": 100.0}}) + "\n",
        encoding="utf-8",
    )
    settlement_path.write_text(json.dumps({"settlement_complete": True}) + "\n", encoding="utf-8")

    scoreability = evaluate_sep23_scoreability(
        raw_decisions_path=raw_path,
        settlements_path=settlement_path,
        required_feature_columns=["quote_volume_30s", "flow_imbalance_30s"],
    )

    assert scoreability["available"] is True
    assert scoreability["scoreable"] is False
    assert scoreability["missing_feature_examples"] == ["flow_imbalance_30s"]
    assert scoreability["reasons"] == ["sep23_shadow_decisions_missing_full_causal_feature_vector"]
