from __future__ import annotations

from scripts.research.run_fast174_crypto_cross_asset_leadlag_smoke import fast174_status


def test_fast174_status_requires_strict_survivor_for_candidate() -> None:
    status, reasons = fast174_status(
        screen_survivor_count=3,
        validation_survivor_count=2,
        strict_survivor_count=1,
    )

    assert status == "FAST174_CROSS_ASSET_LEADLAG_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "strict_survivors:1" in reasons
    assert "fresh_preregistered_observation_required" in reasons


def test_fast174_status_reports_extension_failure_after_validation_survivor() -> None:
    status, reasons = fast174_status(
        screen_survivor_count=4,
        validation_survivor_count=1,
        strict_survivor_count=0,
    )

    assert status == "FAST174_CROSS_ASSET_LEADLAG_FAILED_EXTENSION"
    assert "validation_survivors:1" in reasons
    assert "validation_survivors_failed_extension" in reasons


def test_fast174_status_reports_validation_failure_after_screen_survivor() -> None:
    status, reasons = fast174_status(
        screen_survivor_count=5,
        validation_survivor_count=0,
        strict_survivor_count=0,
    )

    assert status == "FAST174_CROSS_ASSET_LEADLAG_FAILED_VALIDATION"
    assert "screen_survivors:5" in reasons
    assert "screen_survivors_failed_validation" in reasons
