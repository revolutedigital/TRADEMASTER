"""Fast119 focused temporal-audit tests."""

from scripts.research.run_fast119_usdjpy_weekly_gap_temporal_audit import (
    FROZEN_POLICY,
    temporal_status,
)


def test_frozen_policy_matches_the_fast118_usdjpy_clue() -> None:
    assert FROZEN_POLICY.identifier == "continuation_gap10_d15_h12_fixed_full_gap"
    assert FROZEN_POLICY.direction == "continuation"
    assert FROZEN_POLICY.minimum_gap_pips == 10.0
    assert FROZEN_POLICY.entry_delay_minutes == 15
    assert FROZEN_POLICY.max_hold_minutes == 720
    assert FROZEN_POLICY.management.name == "fixed_full_gap"


def test_temporal_status_rejects_pre_2019_instability() -> None:
    status, reasons = temporal_status(
        [
            {"slice": "pre_2011_2018", "mean_stress_pips": -0.1},
            {"slice": "current_2019_2026", "mean_stress_pips": 1.0},
        ]
    )

    assert status == "FAST119_USDJPY_WEEKLY_GAP_TEMPORAL_INSTABILITY_REJECTED"
    assert "not_approved_strategy" in reasons
    assert "pre_2011_2018_mean_stress_not_positive" in reasons
