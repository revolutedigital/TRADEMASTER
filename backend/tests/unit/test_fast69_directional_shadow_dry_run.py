import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.research.run_fast69_directional_shadow_dry_run import (
    DirectionalGate,
    build_fast69_policy_artifact,
    directional_gate_passed,
    fast69_status,
    shadow_decision_to_fast69_row,
)


def test_directional_gate_requires_all_thresholds() -> None:
    gate = DirectionalGate(
        quote_volume_floor=100.0,
        realized_vol_floor=10.0,
        quote_volume_cap=None,
        realized_vol_cap=50.0,
        directional_feature="directed_flow_imbalance_30s",
        directional_floor=-0.25,
    )

    assert directional_gate_passed(
        {
            "quote_volume_30s": 100.0,
            "realized_vol_30s_bps": 20.0,
            "directed_flow_imbalance_30s": -0.20,
        },
        gate,
    )
    assert not directional_gate_passed(
        {
            "quote_volume_30s": 100.0,
            "realized_vol_30s_bps": 60.0,
            "directed_flow_imbalance_30s": -0.20,
        },
        gate,
    )
    assert not directional_gate_passed(
        {
            "quote_volume_30s": 100.0,
            "realized_vol_30s_bps": 20.0,
            "directed_flow_imbalance_30s": -0.30,
        },
        gate,
    )


def test_shadow_decision_row_clears_would_enter_when_gate_fails() -> None:
    gate = DirectionalGate(
        quote_volume_floor=100.0,
        realized_vol_floor=10.0,
        quote_volume_cap=None,
        realized_vol_cap=None,
        directional_feature="directed_flow_imbalance_30s",
        directional_floor=0.0,
    )

    row = shadow_decision_to_fast69_row(
        _Decision(
            feature_vector={
                "quote_volume_30s": 200.0,
                "realized_vol_30s_bps": 20.0,
                "directed_flow_imbalance_30s": -0.1,
            }
        ),
        label="wal",
        gate=gate,
    )

    assert row["top_p_would_enter"] is True
    assert row["directional_gate_passed"] is False
    assert row["would_enter"] is False


def test_build_fast69_artifact_preserves_research_only_boundary(tmp_path: Path) -> None:
    top_p_path = tmp_path / "top-p.json"
    fast68_path = tmp_path / "fast68.json"
    top_p_path.write_text("top-p", encoding="utf-8")
    fast68_path.write_text("fast68", encoding="utf-8")

    artifact = build_fast69_policy_artifact(
        top_p_report=_top_p_report(),
        fast68_report=_fast68_report(),
        top_p_report_path=top_p_path,
        fast68_report_path=fast68_path,
    )

    assert artifact["research_only"] is True
    assert artifact["order_submission_allowed"] is False
    assert artifact["execution_authorization"] == "none"
    assert artifact["approved_strategy"] is False
    assert artifact["directional_gate"]["thresholds"]["directional_feature"] == (
        "directed_flow_imbalance_30s"
    )


def test_build_fast69_artifact_rejects_unsafe_top_p_report(tmp_path: Path) -> None:
    top_p_path = tmp_path / "top-p.json"
    fast68_path = tmp_path / "fast68.json"
    top_p_path.write_text("top-p", encoding="utf-8")
    fast68_path.write_text("fast68", encoding="utf-8")
    unsafe = _top_p_report()
    unsafe["order_submission_allowed"] = True

    with pytest.raises(ValueError, match="must not allow orders"):
        build_fast69_policy_artifact(
            top_p_report=unsafe,
            fast68_report=_fast68_report(),
            top_p_report_path=top_p_path,
            fast68_report_path=fast68_path,
        )


def test_fast69_status_is_research_only_signal_state() -> None:
    status, reasons = fast69_status(
        {
            "wal_root_count": 1,
            "run_count": 1,
            "raw_top_p_count": 10,
            "gated_selected_count": 3,
            "runs_with_gated_signals": 1,
        }
    )

    assert status == "FAST69_DIRECTIONAL_SHADOW_SIGNALS_RESEARCH_ONLY"
    assert "not_approved_strategy" in reasons


@dataclass(frozen=True)
class _Decision:
    feature_vector: dict[str, float]
    decision_time: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    side: str = "BUY"
    horizon_seconds: int = 300
    probability: float = 0.95
    threshold: float = 0.9
    would_enter: bool = True
    model_sha256: str = "model"
    feature_vector_sha256: str = "features"


def _top_p_report() -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "model_sha256": "model",
        "frozen_artifact": {
            "research_only": True,
            "order_submission_allowed": False,
            "execution_authorization": "none",
            "model_sha256": "model",
            "probability_threshold": 0.9,
            "tail_fraction": 0.01,
            "horizon_seconds": 300,
        },
    }


def _fast68_report() -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "status": "FAST68_DIRECTIONAL_GATE_DIAGNOSTIC_SURVIVORS_FOUND_REQUIRES_SHADOW",
        "top_candidate": {
            "spec": {
                "directional_feature": "directed_flow_imbalance_30s",
                "directional_quantile": 0.33,
                "name": "spec",
            },
            "frozen_thresholds": {
                "directional_feature": "directed_flow_imbalance_30s",
                "directional_floor": -0.4,
                "quote_volume_cap": None,
                "quote_volume_floor": 100.0,
                "realized_vol_cap": 300.0,
                "realized_vol_floor": 10.0,
                "spec_name": "spec",
            },
            "historical_aggregate": {"stress_total_log_return_bps": 1.0},
            "extension_aggregate": {"stress_total_log_return_bps": 1.0},
        },
    }
