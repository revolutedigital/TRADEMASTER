"""CLI report readers for the statistical gate fail closed."""

import json

from scripts.research.evaluate_statistical_gate import (
    _read_prospective_shadow_positive,
    _read_top_p_monotonicity,
    _report_sha256,
    _with_artifact_sha256,
)


def test_top_p_report_requires_research_safety_flags(tmp_path) -> None:
    report_path = tmp_path / "top-p.json"
    report_path.write_text(
        json.dumps(
            {
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "none",
                "results": [{"summary": {"top_p_monotonic": True}}],
            }
        ),
        encoding="utf-8",
    )

    monotonic, reasons = _read_top_p_monotonicity(report_path)

    assert monotonic is True
    assert reasons == []


def test_top_p_report_fails_closed_when_execution_authorization_is_not_none(tmp_path) -> None:
    report_path = tmp_path / "top-p.json"
    report_path.write_text(
        json.dumps(
            {
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "testnet",
                "results": [{"summary": {"top_p_monotonic": True}}],
            }
        ),
        encoding="utf-8",
    )

    monotonic, reasons = _read_top_p_monotonicity(report_path)

    assert monotonic is False
    assert reasons == ["top_p_report_has_execution_authorization"]


def test_prospective_shadow_report_must_be_committed_and_positive(tmp_path) -> None:
    report_path = tmp_path / "shadow.json"
    report_path.write_text(json.dumps(_shadow_report()), encoding="utf-8")

    positive, reasons = _read_prospective_shadow_positive(report_path)

    assert positive is True
    assert reasons == []


def test_prospective_shadow_report_fails_closed_on_dry_run_or_negative_mean(tmp_path) -> None:
    report_path = tmp_path / "shadow.json"
    report = _shadow_report()
    report["committed"] = False
    report["dry_run"] = True
    report["stress_mean_bps"] = -0.1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    positive, reasons = _read_prospective_shadow_positive(report_path)

    assert positive is False
    assert "prospective_shadow_report_not_committed" in reasons
    assert "prospective_shadow_report_is_dry_run" in reasons
    assert "prospective_shadow_stress_mean_not_positive" in reasons


def test_prospective_shadow_report_fails_closed_when_outcomes_are_incomplete(tmp_path) -> None:
    report_path = tmp_path / "shadow.json"
    report = _shadow_report()
    report["outcome_count"] = 1
    report["decision_day_count"] = 1
    report["outcome_day_count"] = 1
    report["complete"] = False
    report["outcomes"] = report["outcomes"][:1]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    positive, reasons = _read_prospective_shadow_positive(report_path)

    assert positive is False
    assert "prospective_shadow_report_incomplete" in reasons
    assert "prospective_shadow_outcome_count_mismatch" in reasons
    assert "prospective_shadow_outcome_list_incomplete" in reasons
    assert "prospective_shadow_has_fewer_than_20_days" in reasons


def test_prospective_shadow_report_fails_closed_on_malformed_outcome(tmp_path) -> None:
    report_path = tmp_path / "shadow.json"
    report = _shadow_report()
    report["outcomes"][1] = {
        "signal_id": 1,
        "expected_net_bps": "nan",
        "stress_net_bps": 0.4,
        "label_sha256": "not-a-hash",
        "order_id": "execution-field-must-not-exist",
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    positive, reasons = _read_prospective_shadow_positive(report_path)

    assert positive is False
    assert "prospective_shadow_outcome_1_signal_id_duplicate" in reasons
    assert "prospective_shadow_outcome_1_label_sha256_invalid" in reasons
    assert "prospective_shadow_outcome_1_expected_net_bps_invalid" in reasons
    assert "prospective_shadow_outcome_1_contains_execution_field" in reasons
    assert "prospective_shadow_outcome_dates_mismatch" in reasons


def test_statistical_gate_artifact_hash_excludes_its_own_field() -> None:
    payload = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "attempted_hypotheses": 1,
        "top_p_monotonic": True,
        "top_p_monotonic_reasons": [],
        "prospective_shadow_positive": True,
        "prospective_shadow_reasons": [],
        "decision_counts": {"APPROVED": 1},
        "results": [{"decision": "APPROVED"}],
    }

    report = _with_artifact_sha256(payload)
    recomputed = _report_sha256(
        {key: value for key, value in report.items() if key != "artifact_sha256"}
    )

    assert len(report["artifact_sha256"]) == 64
    assert report["artifact_sha256"] == recomputed


def _shadow_report() -> dict[str, object]:
    outcomes = [
        {
            "signal_id": day + 1,
            "decision_time": f"2026-01-{day + 1:02d}T12:00:00+00:00",
            "would_enter": True,
            "expected_net_bps": 1.2,
            "stress_net_bps": 0.4,
            "label_sha256": "a" * 64,
            "policy_name": "wide",
        }
        for day in range(20)
    ]
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "committed": True,
        "dry_run": False,
        "signal_count": len(outcomes),
        "outcome_count": len(outcomes),
        "decision_day_count": 20,
        "outcome_day_count": 20,
        "complete": True,
        "expected_mean_bps": 1.2,
        "stress_mean_bps": 0.4,
        "outcomes": outcomes,
    }
