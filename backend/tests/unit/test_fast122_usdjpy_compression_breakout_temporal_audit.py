"""Fast122 focused temporal-audit tests."""

from scripts.research.run_fast122_usdjpy_compression_breakout_temporal_audit import (
    FROZEN_POLICY,
    temporal_status,
)


def test_frozen_policy_matches_the_fast121_usdjpy_clue() -> None:
    assert FROZEN_POLICY.identifier == "tf1h_ch20_cq0.4_atr1.5_hold720_fixed_3r"
    assert FROZEN_POLICY.timeframe == "1h"
    assert FROZEN_POLICY.channel_bars == 20
    assert FROZEN_POLICY.compression_quantile == 0.40
    assert FROZEN_POLICY.stop_atr_multiple == 1.5
    assert FROZEN_POLICY.max_hold_minutes == 720
    assert FROZEN_POLICY.management.name == "fixed_3r"


def test_temporal_status_rejects_weak_pre_sample_stability() -> None:
    status, reasons = temporal_status(
        [
            {
                "slice": "pre_2011_2018",
                "mean_stress_pips": 0.1,
                "lower_stress_pips": -1.0,
                "positive_stress_year_share": 0.3,
            },
            {
                "slice": "current_2019_2026",
                "mean_stress_pips": 2.0,
                "lower_stress_pips": 0.4,
                "positive_stress_year_share": 0.75,
            },
        ]
    )

    assert status == "FAST122_USDJPY_COMPRESSION_BREAKOUT_WEAK_TEMPORAL_STABILITY_REJECTED"
    assert "not_approved_strategy" in reasons
    assert "regime_instability" in reasons
