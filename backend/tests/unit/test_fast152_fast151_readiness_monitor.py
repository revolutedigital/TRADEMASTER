from __future__ import annotations

import pandas as pd

from scripts.research.run_fast152_fast151_readiness_monitor import fast152_status


def test_fast152_status_waits_for_post_preregistration_m1_data() -> None:
    status, reasons = fast152_status(
        raw_m1_min_pair_end=pd.Timestamp("2026-09-18T20:59:00Z"),
        base_candidate_count=0,
        eligible_observation_count=0,
        observation_status="FAST151_OBSERVATION_NO_ELIGIBLE_OPERATIONAL_TRADES_YET",
    )

    assert status == "FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA"
    assert "raw_m1_min_pair_end_before_preregistration_timestamp" in reasons
    assert "not_approved_strategy" in reasons


def test_fast152_status_distinguishes_fresh_data_without_base_candidate() -> None:
    status, reasons = fast152_status(
        raw_m1_min_pair_end=pd.Timestamp("2026-10-01T00:00:00Z"),
        base_candidate_count=0,
        eligible_observation_count=0,
        observation_status="FAST151_OBSERVATION_NO_ELIGIBLE_OPERATIONAL_TRADES_YET",
    )

    assert status == "FAST152_DATA_FRESH_NO_BASE_CANDIDATE_YET"
    assert "no_base_policy_candidate_after_preregistration" in reasons


def test_fast152_status_distinguishes_filtered_candidates_without_accepted_trade() -> None:
    status, reasons = fast152_status(
        raw_m1_min_pair_end=pd.Timestamp("2026-10-01T00:00:00Z"),
        base_candidate_count=2,
        eligible_observation_count=0,
        observation_status="FAST151_OBSERVATION_NO_ELIGIBLE_OPERATIONAL_TRADES_YET",
    )

    assert status == "FAST152_DATA_FRESH_BUT_NO_ACCEPTED_OPERATIONAL_TRADE_YET"
    assert "base_candidates_exist_after_preregistration" in reasons


def test_fast152_status_flags_accepted_observations_for_review() -> None:
    status, reasons = fast152_status(
        raw_m1_min_pair_end=pd.Timestamp("2026-10-30T00:00:00Z"),
        base_candidate_count=3,
        eligible_observation_count=1,
        observation_status="FAST151_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
    )

    assert status == "FAST152_FAST151_OBSERVATION_READY_FOR_REVIEW"
    assert "accepted_operational_observation_exists" in reasons
    assert "review_fast151_report_before_any_promotion" in reasons
