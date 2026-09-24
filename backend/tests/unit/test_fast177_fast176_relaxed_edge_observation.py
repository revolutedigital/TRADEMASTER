from __future__ import annotations

import pandas as pd

from scripts.research.run_fast177_fast176_relaxed_edge_observation import (
    EDGE_THRESHOLD_BPS,
    REGIME_GATE_FEATURE,
    REGIME_GATE_THRESHOLD,
    fast177_status,
    prospective_diagnostic_record,
    score_and_filter_rows,
    score_rows,
    summarize_prospective_diagnostics,
)


class _DummyModel:
    def __init__(self, predictions: list[float]) -> None:
        self.predictions = predictions

    def predict(self, _features: pd.DataFrame) -> list[float]:
        return self.predictions


def test_score_and_filter_rows_applies_edge_and_regime_gate() -> None:
    rows = pd.DataFrame(
        {
            "actual_bps": [10.0, 10.0, -10.0, 10.0],
            REGIME_GATE_FEATURE: [
                REGIME_GATE_THRESHOLD + 1,
                REGIME_GATE_THRESHOLD - 1,
                REGIME_GATE_THRESHOLD + 1,
                REGIME_GATE_THRESHOLD + 1,
            ],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )
    model = _DummyModel([EDGE_THRESHOLD_BPS + 1, EDGE_THRESHOLD_BPS + 1, -EDGE_THRESHOLD_BPS - 1, 0.0])

    filtered = score_and_filter_rows(model, rows, ["feature"])

    assert filtered["side_label"].tolist() == ["LONG", "SHORT"]
    assert filtered["net_bps"].tolist() == [8.0, 8.0]


def test_score_rows_keeps_rejected_rows_for_diagnostics() -> None:
    rows = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(["2026-09-24T11:55:00Z", "2026-09-24T12:00:00Z"]),
            "actual_bps": [5.0, -5.0],
            REGIME_GATE_FEATURE: [REGIME_GATE_THRESHOLD + 1, REGIME_GATE_THRESHOLD - 1],
            "feature": [1.0, 2.0],
        }
    )
    model = _DummyModel([EDGE_THRESHOLD_BPS + 1, EDGE_THRESHOLD_BPS + 2])

    scored = score_rows(model, rows, ["feature"])
    diagnostics = summarize_prospective_diagnostics(scored)

    assert scored["eligible"].tolist() == [True, False]
    assert diagnostics["row_count"] == 2
    assert diagnostics["edge_gate_pass_count"] == 2
    assert diagnostics["regime_gate_pass_count"] == 1
    assert diagnostics["eligible_count"] == 1
    assert diagnostics["latest_eligible"] is False


def test_prospective_diagnostic_record_explains_gate_failures() -> None:
    rows = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(["2026-09-24T12:05:00Z"]),
            "date": ["2026-09-24"],
            "actual_bps": [3.0],
            REGIME_GATE_FEATURE: [REGIME_GATE_THRESHOLD - 1],
            "feature": [1.0],
        }
    )
    scored = score_rows(_DummyModel([EDGE_THRESHOLD_BPS - 1]), rows, ["feature"])

    record = prospective_diagnostic_record(scored.iloc[0])

    assert record["decision_time"] == "2026-09-24T12:05:00+00:00"
    assert record["edge_gate_passed"] is False
    assert record["regime_gate_passed"] is False
    assert record["eligible"] is False
    assert record["edge_threshold"] == EDGE_THRESHOLD_BPS
    assert record["regime_gate_threshold"] == REGIME_GATE_THRESHOLD


def test_fast177_status_waits_when_no_post_preregistration_rows() -> None:
    status, reasons = fast177_status(
        prospective_rows=pd.DataFrame(),
        observations=pd.DataFrame(),
        gates={"early_fail": False, "shadow_candidate": False, "shadow_reasons": []},
    )

    assert status == "FAST177_WAITING_FOR_POST_PREREGISTRATION_DATA"
    assert "no_post_preregistration_decision_rows" in reasons


def test_fast177_status_reports_fresh_data_without_eligible_decisions() -> None:
    status, reasons = fast177_status(
        prospective_rows=pd.DataFrame({"decision_time": [1]}),
        observations=pd.DataFrame(),
        gates={"early_fail": False, "shadow_candidate": False, "shadow_reasons": []},
    )

    assert status == "FAST177_DATA_FRESH_NO_ELIGIBLE_DECISIONS_YET"
    assert "edge_or_regime_gate_not_met" in reasons
