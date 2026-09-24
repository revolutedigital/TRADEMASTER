from __future__ import annotations

import json

from scripts.research.run_fast110_fx_fast2_pair_validation import (
    fast110_status,
    load_candidate_identities,
    stage_gate,
    validate_candidate,
)


def test_load_candidate_identities_deduplicates_by_config_pair(tmp_path) -> None:
    path = tmp_path / "weak.json"
    path.write_text(
        json.dumps(
            [
                {
                    "config": "F1a.config",
                    "pair": "GBPUSD",
                    "family": "F1a",
                    "base_mean_r": 0.1,
                    "stress_mean_r": 0.05,
                    "stress_t": 0.7,
                    "positive_stress_year_count": 3,
                    "year_count": 5,
                },
                {
                    "config": "F1a.config",
                    "pair": "GBPUSD",
                    "family": "F1a",
                    "base_mean_r": 0.2,
                    "stress_mean_r": 0.1,
                },
            ]
        ),
        encoding="utf-8",
    )

    candidates = load_candidate_identities(path)

    assert candidates == [
        {
            "config": "F1a.config",
            "pair": "GBPUSD",
            "family": "F1a",
            "fast109_base_mean_r": 0.1,
            "fast109_stress_mean_r": 0.05,
            "fast109_stress_t": 0.7,
            "fast109_positive_stress_year_count": 3,
            "fast109_year_count": 5,
        }
    ]


def test_stage_gate_separates_weak_from_strict_year_stability() -> None:
    row = {
        "stage": "replication",
        "base_trade_count": 120,
        "stress_trade_count": 120,
        "base_mean_r": 0.05,
        "stress_mean_r": 0.02,
        "base_t": 1.1,
        "stress_t": 0.5,
        "year_count": 5,
        "positive_base_year_count": 5,
        "positive_stress_year_count": 4,
        "min_year_stress_mean_r": -0.01,
    }

    gate = stage_gate(row, min_trades=100)

    assert gate["weak_pass"] is True
    assert gate["strict_pass"] is False
    assert "stress_not_positive_every_year" in gate["failed"]


def test_validate_candidate_requires_replication_and_confirmation() -> None:
    candidate = {"config": "F1a.config", "pair": "GBPUSD", "family": "F1a"}
    passing_stage = {
        "stage": "replication",
        "base_trade_count": 120,
        "stress_trade_count": 120,
        "base_mean_r": 0.05,
        "stress_mean_r": 0.02,
        "base_t": 1.1,
        "stress_t": 0.5,
        "year_count": 3,
        "positive_base_year_count": 3,
        "positive_stress_year_count": 3,
        "min_year_stress_mean_r": 0.01,
    }
    failing_stage = {**passing_stage, "stage": "confirmation", "stress_mean_r": -0.01}

    result = validate_candidate(
        candidate,
        passing_stage,
        failing_stage,
        replication_min_trades=100,
        confirmation_min_trades=30,
    )

    assert result["weak_validated"] is False
    assert result["strict_validated"] is False
    assert "stress_mean_not_positive" in result["confirmation"]["failed"]


def test_fast110_status_never_approves_strategy() -> None:
    status, reasons = fast110_status(strict_survivor_count=0, weak_survivor_count=0)

    assert status == "FAST110_FX_FAST2_PAIR_VALIDATION_FAILED_OUT_OF_SAMPLE"
    assert "not_approved_strategy" in reasons

    status, reasons = fast110_status(strict_survivor_count=0, weak_survivor_count=2)

    assert status == "FAST110_FX_FAST2_PAIR_VALIDATION_WEAK_ONLY_FAILED_STABILITY"
    assert "out_of_sample_positive_but_failed_strict_year_stability" in reasons
