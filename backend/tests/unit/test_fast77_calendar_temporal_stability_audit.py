import pandas as pd

from scripts.research.run_fast77_calendar_temporal_stability_audit import (
    audit_scope,
    fast77_status,
    temporal_flags,
)


def test_temporal_flags_reject_aggregate_positive_with_negative_recent_year() -> None:
    candidate = {
        "selection": {"mean_base_pips": 1.0, "mean_stress_pips": 1.0},
        "audit": {
            "mean_base_pips": 1.0,
            "mean_stress_pips": 1.0,
            "lower_base_pips_uncorrected": -1.0,
            "lower_stress_pips_uncorrected": -1.0,
            "lower_base_pips_bonferroni": -1.0,
            "lower_stress_pips_bonferroni": -1.0,
        },
        "selection_years": [
            {"year": 2021, "mean_stress_pips": 1.0},
            {"year": 2022, "mean_stress_pips": 1.0},
        ],
        "audit_years": [
            {"year": 2024, "trades": 4, "mean_base_pips": 2.0, "mean_stress_pips": 2.0},
            {"year": 2025, "trades": 4, "mean_base_pips": 2.0, "mean_stress_pips": 2.0},
            {"year": 2026, "trades": 4, "mean_base_pips": -1.0, "mean_stress_pips": -1.0},
        ],
    }

    flags = temporal_flags(candidate, minimum_audit_year_trades=4)

    assert not flags["temporal_weak_survivor"]
    assert not flags["recent_audit_year_positive_stress"]


def test_audit_scope_detects_three_year_weak_survivor_without_confirming_strategy() -> None:
    trades = pd.DataFrame(
        {
            "event_timestamp": pd.to_datetime(
                [
                    "2023-01-01T00:00:00Z",
                    "2023-02-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ],
                utc=True,
            ),
            "pair": ["USDJPY"] * 5,
            "event_name": ["FOMC rate decision"] * 5,
            "policy_id": ["drift"] * 5,
            "net_base_pips": [1.0, 1.0, 10.0, 1.0, 1.0],
            "net_stress_pips": [1.0, 1.0, 10.0, 1.0, 1.0],
        }
    )

    result = audit_scope(
        trades,
        candidates=[
            {
                "pair": "USDJPY",
                "event_name": "FOMC rate decision",
                "policy_id": "drift",
            }
        ],
        scope_name="pair_event_policy",
        group_columns=("pair", "event_name", "policy_id"),
        alpha=0.05,
        minimum_audit_year_trades=1,
    )

    assert result["summary"]["temporal_weak_survivor_count"] == 1
    assert result["summary"]["confirmed_temporal_candidate_count"] == 0
    assert result["temporal_weak_survivors"][0]["pair"] == "USDJPY"


def test_fast77_status_keeps_temporal_weak_survivor_in_preregistration_only() -> None:
    status, reasons = fast77_status(
        pair_policy={
            "summary": {
                "temporal_weak_survivor_count": 0,
                "confirmed_temporal_candidate_count": 0,
            }
        },
        pair_event_policy={
            "summary": {
                "temporal_weak_survivor_count": 1,
                "confirmed_temporal_candidate_count": 0,
            }
        },
    )

    assert status == "FAST77_CALENDAR_TEMPORAL_STABILITY_WEAK_SURVIVOR_REQUIRES_PREREGISTRATION"
    assert "preregistered_observation_only" in reasons
