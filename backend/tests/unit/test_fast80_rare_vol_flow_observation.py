import json
from pathlib import Path

from scripts.research.run_fast80_rare_vol_flow_observation import (
    FLOW_IMBALANCE_5S_THRESHOLD,
    HORIZON_SECONDS,
    MODEL_SHA256,
    PREREGISTRATION_TIMESTAMP,
    REALIZED_VOL_30S_THRESHOLD,
    audit_required_features,
    run_fast80_rare_vol_flow_observation,
)


def test_fast80_runner_fails_closed_when_flow_feature_is_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "missing_flow.jsonl"
    input_path.write_text(json.dumps(base_record(include_flow=False)) + "\n", encoding="utf-8")

    report = run_fast80_rare_vol_flow_observation(
        input_path=input_path,
        output_root=tmp_path / "out",
    )

    assert report["status"] == "FAST80_OBSERVATION_NOT_SCOREABLE_MISSING_FEATURES"
    assert "flow_imbalance_5s" in report["feature_audit"]["missing_required_features"]
    assert not report["gates"]["testnet_allowed"]


def test_fast80_runner_ignores_records_before_preregistration(tmp_path: Path) -> None:
    record = base_record(decision_time_ms=int(PREREGISTRATION_TIMESTAMP.timestamp() * 1000) - 1_000)
    input_path = tmp_path / "pre_prereg.jsonl"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = run_fast80_rare_vol_flow_observation(
        input_path=input_path,
        output_root=tmp_path / "out",
    )

    assert report["status"] == "FAST80_OBSERVATION_NO_PROSPECTIVE_RECORDS_YET"
    assert report["eligible_observation_count"] == 0


def test_fast80_runner_marks_shadow_candidate_but_keeps_testnet_blocked(tmp_path: Path) -> None:
    start_ms = int(PREREGISTRATION_TIMESTAMP.timestamp() * 1000) + 60_000
    records = []
    for index in range(100):
        day = index // 10
        decision_time_ms = start_ms + day * 86_400_000 + (index % 10) * 600_000
        records.append(
            base_record(
                decision_time_ms=decision_time_ms,
                expected_net_bps=4.0,
                stress_net_bps=3.0,
                label=f"prospective-day-{day:02d}",
            )
        )
    input_path = tmp_path / "candidate.jsonl"
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    report = run_fast80_rare_vol_flow_observation(
        input_path=input_path,
        output_root=tmp_path / "out",
    )

    assert report["status"] == "FAST80_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW"
    assert report["eligible_observation_count"] == 100
    assert not report["gates"]["testnet_allowed"]
    assert not report["gates"]["live_allowed"]


def test_audit_required_features_reports_complete_records_as_scoreable() -> None:
    audit = audit_required_features([base_record()])

    assert audit["scoreable"]
    assert audit["missing_required_features"] == []


def base_record(
    *,
    decision_time_ms: int | None = None,
    expected_net_bps: float = 4.0,
    stress_net_bps: float = 3.0,
    label: str = "prospective-day-00",
    include_flow: bool = True,
) -> dict[str, object]:
    if decision_time_ms is None:
        decision_time_ms = int(PREREGISTRATION_TIMESTAMP.timestamp() * 1000) + 60_000
    record: dict[str, object] = {
        "decision_time_ms": decision_time_ms,
        "entry_time_ms": decision_time_ms + 100,
        "exit_time_ms": decision_time_ms + 300_000,
        "label": label,
        "side": "BUY",
        "model_sha256": MODEL_SHA256,
        "horizon_seconds": HORIZON_SECONDS,
        "probability": 0.95,
        "realized_vol_30s_bps": REALIZED_VOL_30S_THRESHOLD + 1.0,
        "expected_net_bps": expected_net_bps,
        "stress_net_bps": stress_net_bps,
        "exit_reason": "TIMEOUT",
    }
    if include_flow:
        record["flow_imbalance_5s"] = FLOW_IMBALANCE_5S_THRESHOLD - 0.01
    return record
