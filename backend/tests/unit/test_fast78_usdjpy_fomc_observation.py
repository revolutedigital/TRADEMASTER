import pandas as pd

from scripts.research.run_fast78_usdjpy_fomc_observation import (
    OBSERVATION_EVENT,
    OBSERVATION_PAIR,
    OBSERVATION_POLICY_ID,
    PREREGISTRATION_TIMESTAMP,
    evaluate_gates,
    fast78_status,
    select_eligible_observations,
    summarize_observations,
)


def test_select_eligible_observations_filters_to_frozen_candidate_after_preregistration() -> None:
    trades = make_observations(
        timestamps=[
            "2026-09-23T18:26:11Z",
            "2026-11-04T19:00:00Z",
            "2026-12-16T19:00:00Z",
        ],
        stress=[10.0, 11.0, 12.0],
    )
    trades.loc[2, "event_name"] = "CPI US"

    observations = select_eligible_observations(
        trades,
        prospective_start=PREREGISTRATION_TIMESTAMP,
    )

    assert len(observations) == 1
    assert observations.iloc[0]["event_timestamp"] == pd.Timestamp("2026-11-04T19:00:00Z")


def test_evaluate_gates_fails_after_first_eight_events_lose_too_much() -> None:
    observations = make_observations(
        timestamps=[f"2027-01-{day:02d}T19:00:00Z" for day in range(1, 9)],
        stress=[-3.0] * 8,
    )

    summary = summarize_observations(observations, alpha=0.05)
    gates = evaluate_gates(summary)
    status, reasons = fast78_status(summary=summary, gates=gates)

    assert gates["early_fail_triggered"]
    assert status == "FAST78_OBSERVATION_FAILED_PREREGISTERED_GATE"
    assert "preregistered_failure_gate_triggered" in reasons


def test_evaluate_gates_marks_shadow_candidate_but_not_testnet() -> None:
    observations = make_observations(
        timestamps=[f"2027-01-{day:02d}T19:00:00Z" for day in range(1, 21)],
        stress=[3.0] * 20,
    )

    summary = summarize_observations(observations, alpha=0.05)
    gates = evaluate_gates(summary)
    status, reasons = fast78_status(summary=summary, gates=gates)

    assert gates["minimum_shadow_candidate_gate_passed"]
    assert not gates["testnet_allowed"]
    assert status == "FAST78_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW"
    assert "testnet_still_requires_new_preregistration" in reasons


def test_fast78_status_waits_when_no_events_exist() -> None:
    observations = make_observations(timestamps=[], stress=[])

    summary = summarize_observations(observations, alpha=0.05)
    gates = evaluate_gates(summary)
    status, reasons = fast78_status(summary=summary, gates=gates)

    assert status == "FAST78_OBSERVATION_NO_ELIGIBLE_EVENTS_YET"
    assert "waiting_for_post_preregistration_fomc" in reasons


def make_observations(*, timestamps: list[str], stress: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "event_timestamp": pd.to_datetime(timestamps, utc=True),
            "pair": [OBSERVATION_PAIR] * len(timestamps),
            "event_name": [OBSERVATION_EVENT] * len(timestamps),
            "policy_id": [OBSERVATION_POLICY_ID] * len(timestamps),
            "side": ["long"] * len(timestamps),
            "net_base_pips": stress,
            "net_stress_pips": stress,
            "exit_base_timestamp": pd.to_datetime(timestamps, utc=True),
            "exit_base_reason": ["timeout"] * len(timestamps),
            "exit_stress_timestamp": pd.to_datetime(timestamps, utc=True),
            "exit_stress_reason": ["timeout"] * len(timestamps),
        }
    )
    return frame
