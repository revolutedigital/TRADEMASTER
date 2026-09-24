"""Fast123 USDJPY compression-regime filter tests."""

import pandas as pd

from scripts.research.run_fast123_usdjpy_compression_regime_filter_scout import (
    FilterSpec,
    add_numeric_buckets,
    apply_filter,
    build_filter_specs,
    fast123_status,
    filter_passes,
    session_bucket,
)


def test_session_bucket_maps_entry_hours() -> None:
    assert session_bucket(1) == "asia"
    assert session_bucket(8) == "london"
    assert session_bucket(14) == "new_york"
    assert session_bucket(21) == "late_us"


def test_numeric_buckets_use_train_thresholds_only() -> None:
    frame = pd.DataFrame(
        {
            "atr_pips": [1.0, 2.0, 100.0],
            "risk_pips": [1.0, 2.0, 100.0],
            "compression_ratio": [0.1, 0.5, 2.0],
            "breakout_over_atr": [0.1, 0.5, 2.0],
        }
    )

    bucketed = add_numeric_buckets(frame, frame.iloc[:2])

    assert bucketed.loc[2, "atr_pips_bucket"] == "high"
    assert bucketed.loc[0, "compression_ratio_bucket"] == "low"


def test_build_filter_specs_includes_entry_known_single_and_pair_filters() -> None:
    frame = pd.DataFrame(
        {
            "side": ["LONG", "SHORT"],
            "session_bucket": ["london", "asia"],
            "month_bucket": ["q1", "q2"],
            "entry_weekday": [0, 1],
            "atr_pips_bucket": ["low", "high"],
            "compression_ratio_bucket": ["low", "high"],
            "breakout_over_atr_bucket": ["low", "high"],
        }
    )

    ids = {spec.filter_id for spec in build_filter_specs(frame)}

    assert "all" in ids
    assert "side=LONG" in ids
    assert "session_bucket=london&side=LONG" in ids


def test_apply_filter_matches_all_conditions() -> None:
    frame = pd.DataFrame({"side": ["LONG", "SHORT"], "session_bucket": ["london", "london"]})
    sample = apply_filter(frame, FilterSpec("x", (("side", "LONG"), ("session_bucket", "london"))))

    assert len(sample) == 1
    assert sample.iloc[0]["side"] == "LONG"


def test_filter_passes_requires_validation_lower_bounds_for_strict() -> None:
    row = {
        "train": {
            "trade_count": 200,
            "mean_stress_pips": 1.0,
            "lower_stress_pips": 0.2,
            "positive_year_share": 1.0,
        },
        "pre_validation": {
            "trade_count": 100,
            "mean_stress_pips": 1.0,
            "lower_stress_pips": -0.1,
            "positive_year_share": 1.0,
        },
        "current": {
            "trade_count": 250,
            "mean_stress_pips": 1.0,
            "lower_stress_pips": 0.2,
            "positive_year_share": 1.0,
        },
    }

    weak, strict, failed = filter_passes(row)

    assert weak is True
    assert strict is False
    assert "pre_validation_lower_stress_not_positive" in failed


def test_fast123_status_never_approves_strategy() -> None:
    status, reasons = fast123_status(strict_count=1, weak_count=1)

    assert status == "FAST123_USDJPY_COMPRESSION_REGIME_STRICT_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons
