import pandas as pd

from scripts.research.run_fast65_causal_feature_gate_walkforward import (
    FeatureGate,
    apply_gate,
    causal_feature_columns,
    fast65_status,
    one_position_summary,
    select_training_gate,
)


def test_causal_feature_columns_excludes_outcome_fields() -> None:
    frame = pd.DataFrame(
        {
            "decision_time_ms": [1, 2, 3, 4, 5],
            "stress_net_bps": [1, 2, 3, 4, 5],
            "mfe_bps": [1, 2, 3, 4, 5],
            "flow_imbalance_5s": [0.1, 0.2, 0.3, 0.4, 0.5],
            "return_5s_bps": [1, 2, 3, 4, 5],
        }
    )

    assert causal_feature_columns(frame) == []

    larger = pd.concat([frame] * 3, ignore_index=True)
    larger["flow_imbalance_5s"] = range(len(larger))

    assert "flow_imbalance_5s" in causal_feature_columns(larger)
    assert "stress_net_bps" not in causal_feature_columns(larger)
    assert "mfe_bps" not in causal_feature_columns(larger)


def test_apply_gate_filters_low_feature_values() -> None:
    frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})

    filtered = apply_gate(
        frame,
        FeatureGate(name="feature<=2", feature="feature", direction="low", threshold=2.0),
    )

    assert filtered["feature"].tolist() == [1.0, 2.0]


def test_select_training_gate_finds_positive_feature_pocket() -> None:
    rows = []
    for block in ("a", "b"):
        for index in range(8):
            rows.append(
                _trade(
                    block=block,
                    decision_time_ms=index * 1_000,
                    feature=0.1 + index * 0.01,
                    stress=2.0,
                )
            )
            rows.append(
                _trade(
                    block=block,
                    decision_time_ms=(index + 20) * 1_000,
                    feature=3.0 + index * 0.01,
                    stress=-4.0,
                )
            )
    frame = pd.DataFrame(rows)

    gate, evaluations = select_training_gate(
        frame,
        feature_columns=["feature"],
        minimum_training_total_trades=5,
        minimum_training_block_trades=2,
    )

    assert evaluations
    assert gate.feature == "feature"
    assert gate.direction == "low"


def test_one_position_summary_skips_overlapping_trades() -> None:
    frame = pd.DataFrame(
        [
            _trade(block="a", decision_time_ms=0, exit_time_ms=10_000, feature=1.0, stress=2.0),
            _trade(block="a", decision_time_ms=1_000, exit_time_ms=2_000, feature=1.0, stress=99.0),
            _trade(block="a", decision_time_ms=11_000, exit_time_ms=12_000, feature=1.0, stress=3.0),
        ]
    )

    summary = one_position_summary(frame)

    assert summary["trade_count"] == 2
    assert summary["stress_total_log_return_bps"] == 5.0


def test_fast65_status_fails_when_not_all_holdouts_pass() -> None:
    status, reasons = fast65_status(
        aggregate={
            "holdout_count": 2,
            "passed_stress_count": 1,
            "stress_total_log_return_bps": 10.0,
        },
        minimum_holdout_trades=3,
    )

    assert status == "FAST65_CAUSAL_FEATURE_GATE_FAILED_WALKFORWARD"
    assert "passed_stress_holdouts:1<2" in reasons


def _trade(
    *,
    block: str,
    decision_time_ms: int,
    feature: float,
    stress: float,
    exit_time_ms: int | None = None,
) -> dict[str, object]:
    return {
        "block": block,
        "decision_time_ms": decision_time_ms,
        "entry_time_ms": decision_time_ms,
        "exit_time_ms": exit_time_ms if exit_time_ms is not None else decision_time_ms + 100,
        "side": "BUY",
        "horizon_seconds": 300,
        "probability": 0.95,
        "entry_price": 100.0,
        "exit_price": 100.0,
        "exit_reason": "TIMEOUT",
        "gross_bps": stress + 2.0,
        "expected_net_bps": stress + 1.0,
        "stress_net_bps": stress,
        "maximum_favorable_bps": max(stress, 0.0),
        "maximum_adverse_bps": min(stress, 0.0),
        "partial_take_fraction": 0.0,
        "feature": feature,
    }
