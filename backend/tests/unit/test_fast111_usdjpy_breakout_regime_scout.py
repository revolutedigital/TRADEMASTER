from __future__ import annotations

import pandas as pd

from scripts.research.run_fast111_usdjpy_breakout_regime_scout import (
    FUSION_ZERO,
    STRESS,
    build_filter_specs,
    fast111_status,
    stage_gate,
    stop_bucket,
    summarize_filter,
)
from scripts.research.run_fast111_usdjpy_breakout_regime_scout import FilterSpec


def test_stop_bucket_uses_discovery_thresholds() -> None:
    assert stop_bucket(10.0, low=20.0, high=40.0) == "narrow"
    assert stop_bucket(30.0, low=20.0, high=40.0) == "middle"
    assert stop_bucket(50.0, low=20.0, high=40.0) == "wide"


def test_build_filter_specs_includes_explainable_one_and_two_dimensional_filters() -> None:
    discovery = pd.DataFrame(
        {
            "side_label": ["LONG", "SHORT"],
            "london_weekday": [0, 1],
            "london_entry_hour": [8, 9],
            "stop_bucket": ["narrow", "wide"],
        }
    )

    ids = {spec.filter_id for spec in build_filter_specs(discovery)}

    assert "all" in ids
    assert "side_label=LONG" in ids
    assert "london_weekday=0&side_label=LONG" in ids
    assert "london_entry_hour=8&stop_bucket=narrow" in ids


def test_stage_gate_distinguishes_weak_and_strict_stability() -> None:
    summary = {
        "base": {
            "trade_count": 30,
            "mean_r": 0.10,
            "t": 1.0,
            "year_count": 5,
            "positive_year_count": 5,
            "min_year_mean_r": 0.01,
        },
        "stress": {
            "trade_count": 30,
            "mean_r": 0.05,
            "t": 0.7,
            "year_count": 5,
            "positive_year_count": 3,
            "min_year_mean_r": -0.02,
        },
    }

    gate = stage_gate(summary, min_trades=20, strict_year_share=0.8)

    assert gate["weak_pass"] is True
    assert gate["strict_pass"] is False
    assert "stress_year_share_too_low" in gate["failed"]


def test_summarize_filter_applies_conditions_to_base_and_stress_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": FUSION_ZERO.name,
                "side_label": "LONG",
                "r_multiple": 1.0,
                "fx_day": 1,
                "year": 2024,
            },
            {
                "scenario": STRESS.name,
                "side_label": "LONG",
                "r_multiple": 0.5,
                "fx_day": 1,
                "year": 2024,
            },
            {
                "scenario": STRESS.name,
                "side_label": "SHORT",
                "r_multiple": -1.0,
                "fx_day": 2,
                "year": 2024,
            },
        ]
    )

    summary = summarize_filter(frame, FilterSpec("side_label=LONG", (("side_label", "LONG"),)))

    assert summary["base"]["trade_count"] == 1
    assert summary["base"]["mean_r"] == 1.0
    assert summary["stress"]["trade_count"] == 1
    assert summary["stress"]["mean_r"] == 0.5


def test_fast111_status_never_approves_strategy() -> None:
    status, reasons = fast111_status(
        strict_survivor_count=0,
        weak_survivor_count=0,
        selected_count=0,
    )

    assert status == "FAST111_USDJPY_BREAKOUT_REGIME_FAILED_DISCOVERY_SELECTION"
    assert "not_approved_strategy" in reasons

    status, reasons = fast111_status(
        strict_survivor_count=0,
        weak_survivor_count=2,
        selected_count=3,
    )

    assert status == "FAST111_USDJPY_BREAKOUT_REGIME_WEAK_ONLY_FAILED_STRICT_STABILITY"
    assert "historical_filter_positive_but_failed_strict_oos_stability" in reasons
