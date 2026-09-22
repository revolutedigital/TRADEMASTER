"""CLI report readers for the statistical gate fail closed."""

import json

from scripts.research.evaluate_statistical_gate import _read_prospective_shadow_positive


def test_prospective_shadow_report_must_be_committed_and_positive(tmp_path) -> None:
    report_path = tmp_path / "shadow.json"
    report_path.write_text(
        json.dumps(
            {
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "none",
                "committed": True,
                "signal_count": 20,
                "expected_mean_bps": 1.2,
                "stress_mean_bps": 0.4,
            }
        ),
        encoding="utf-8",
    )

    positive, reasons = _read_prospective_shadow_positive(report_path)

    assert positive is True
    assert reasons == []


def test_prospective_shadow_report_fails_closed_on_dry_run_or_negative_mean(tmp_path) -> None:
    report_path = tmp_path / "shadow.json"
    report_path.write_text(
        json.dumps(
            {
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "none",
                "committed": False,
                "signal_count": 20,
                "expected_mean_bps": 1.2,
                "stress_mean_bps": -0.1,
            }
        ),
        encoding="utf-8",
    )

    positive, reasons = _read_prospective_shadow_positive(report_path)

    assert positive is False
    assert "prospective_shadow_report_not_committed" in reasons
    assert "prospective_shadow_stress_mean_not_positive" in reasons

