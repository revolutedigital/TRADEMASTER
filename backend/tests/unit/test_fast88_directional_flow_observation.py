import json
from pathlib import Path

from scripts.research.run_fast88_directional_flow_observation import (
    HORIZON_SECONDS,
    MODEL_SHA256,
    PREREGISTRATION_TIMESTAMP,
    REALIZED_VOL_30S_THRESHOLD,
    audit_required_features,
    fast88_status,
    feature_value_from_record,
    normalize_record,
    passes_fast88_rule,
    resolve_probability_conflicts,
    run_fast88_directional_flow_observation,
)


def test_feature_value_reads_causal_feature_values() -> None:
    record = {
        "causal_feature_values": {
            "realized_vol_30s_bps": 16.0,
            "directed_flow_imbalance_5s": 0.25,
        }
    }

    assert feature_value_from_record(record, "realized_vol_30s_bps") == 16.0
    assert feature_value_from_record(record, "directed_flow_imbalance_5s") == 0.25


def test_normalize_record_promotes_required_features_from_nested_values() -> None:
    normalized = normalize_record(
        {
            "decision_time": "2026-09-23T20:19:35Z",
            "causal_feature_values": {
                "realized_vol_30s_bps": 17.0,
                "directed_flow_imbalance_5s": 0.2,
            },
        }
    )

    assert normalized["decision_time_ms"] == preregistration_ms() + 60_000
    assert normalized["realized_vol_30s_bps"] == 17.0
    assert normalized["directed_flow_imbalance_5s"] == 0.2


def test_passes_fast88_rule_requires_top_p_volatility_and_positive_directional_flow() -> None:
    assert passes_fast88_rule(base_scored_record())
    assert not passes_fast88_rule(base_scored_record(would_enter=False))
    assert not passes_fast88_rule(base_scored_record(realized_vol_30s_bps=14.99))
    assert not passes_fast88_rule(base_scored_record(directed_flow_imbalance_5s=0.0))


def test_resolve_probability_conflicts_keeps_highest_probability_per_label_time() -> None:
    rows = [
        base_scored_record(side="BUY", probability=0.91),
        base_scored_record(side="SELL", probability=0.95),
        base_scored_record(
            decision_time_ms=preregistration_ms() + 120_000,
            side="BUY",
            probability=0.90,
        ),
    ]

    selected = resolve_probability_conflicts(rows)

    assert [(row["decision_time_ms"], row["side"]) for row in selected] == [
        (preregistration_ms() + 60_000, "SELL"),
        (preregistration_ms() + 120_000, "BUY"),
    ]


def test_fast88_runner_fails_closed_when_directional_feature_is_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "missing_directional.jsonl"
    record = base_scored_record()
    del record["directed_flow_imbalance_5s"]
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = run_fast88_directional_flow_observation(
        input_path=input_path,
        output_root=tmp_path / "out",
    )

    assert report["status"] == "FAST88_OBSERVATION_NOT_SCOREABLE_MISSING_FEATURES"
    assert "directed_flow_imbalance_5s" in report["feature_audit"]["missing_required_features"]
    assert not report["gates"]["testnet_allowed"]
    assert not report["gates"]["live_allowed"]


def test_fast88_runner_ignores_records_before_preregistration(tmp_path: Path) -> None:
    input_path = tmp_path / "pre_prereg.jsonl"
    input_path.write_text(
        json.dumps(base_scored_record(decision_time_ms=preregistration_ms() - 1_000)) + "\n",
        encoding="utf-8",
    )

    report = run_fast88_directional_flow_observation(
        input_path=input_path,
        output_root=tmp_path / "out",
    )

    assert report["status"] == "FAST88_OBSERVATION_NO_PROSPECTIVE_RECORDS_YET"
    assert report["post_preregistration_record_count"] == 0
    assert report["eligible_observation_count"] == 0


def test_audit_required_features_marks_complete_records_as_scoreable() -> None:
    audit = audit_required_features([base_scored_record()])

    assert audit["scoreable"]
    assert audit["missing_required_features"] == []


def test_fast88_status_marks_shadow_candidate_as_review_only() -> None:
    status, reasons = fast88_status(
        input_record_count=120,
        feature_audit={"missing_required_features": []},
        post_preregistration_record_count=120,
        candidate_universe_count=110,
        rule_match_count=105,
        complete_outcome_count=100,
        summary={},
        gates={
            "early_fail_triggered": False,
            "block_fail_triggered": False,
            "minimum_shadow_candidate_gate_passed": True,
        },
    )

    assert status == "FAST88_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW"
    assert "testnet_still_requires_new_preregistration" in reasons


def preregistration_ms() -> int:
    return int(PREREGISTRATION_TIMESTAMP.timestamp() * 1000)


def base_scored_record(
    *,
    decision_time_ms: int | None = None,
    label: str = "fast89",
    side: str = "BUY",
    probability: float = 0.95,
    would_enter: bool = True,
    realized_vol_30s_bps: float = REALIZED_VOL_30S_THRESHOLD + 1.0,
    directed_flow_imbalance_5s: float = 0.25,
) -> dict[str, object]:
    if decision_time_ms is None:
        decision_time_ms = preregistration_ms() + 60_000
    return {
        "decision_time_ms": decision_time_ms,
        "label": label,
        "side": side,
        "model_sha256": MODEL_SHA256,
        "horizon_seconds": HORIZON_SECONDS,
        "probability": probability,
        "threshold": 0.90,
        "would_enter": would_enter,
        "realized_vol_30s_bps": realized_vol_30s_bps,
        "directed_flow_imbalance_5s": directed_flow_imbalance_5s,
    }
