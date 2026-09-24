from __future__ import annotations

import pandas as pd

from scripts.research.run_fast112_usdjpy_breakout_observation import (
    BASE_SCENARIO,
    OBSERVATION_CONFIG,
    OBSERVATION_PAIR,
    PREREGISTRATION_TIMESTAMP,
    STRESS_SCENARIO,
    STOP_PIPS_WIDE_THRESHOLD,
    evaluate_gates,
    fast112_status,
    select_eligible_observations,
    summarize_observations,
)


def _row(
    *,
    entry_time: str,
    scenario: str = STRESS_SCENARIO,
    config: str = OBSERVATION_CONFIG,
    pair: str = OBSERVATION_PAIR,
    hour: int = 8,
    stop_pips: float = STOP_PIPS_WIDE_THRESHOLD + 1.0,
    r_multiple: float = 0.10,
    year: int = 2027,
) -> dict:
    return {
        "stage": "future",
        "config": config,
        "pair": pair,
        "scenario": scenario,
        "entry_time": entry_time,
        "london_entry_hour": hour,
        "stop_pips": stop_pips,
        "r_multiple": r_multiple,
        "side_label": "LONG",
        "year": year,
    }


def test_select_eligible_observations_requires_post_preregistration_exact_rule() -> None:
    rows = pd.DataFrame(
        [
            _row(entry_time="2026-09-24T03:00:00Z"),
            _row(entry_time="2026-09-24T02:00:00Z"),
            _row(entry_time="2026-09-24T03:00:00Z", hour=9),
            _row(entry_time="2026-09-24T03:00:00Z", stop_pips=STOP_PIPS_WIDE_THRESHOLD),
            _row(entry_time="2026-09-24T03:00:00Z", pair="EURUSD"),
        ]
    )
    rows["entry_time"] = pd.to_datetime(rows["entry_time"], utc=True)

    selected = select_eligible_observations(rows, prospective_start=PREREGISTRATION_TIMESTAMP)

    assert len(selected) == 1
    assert selected.iloc[0]["pair"] == OBSERVATION_PAIR
    assert selected.iloc[0]["london_entry_hour"] == 8
    assert selected.iloc[0]["stop_pips"] > STOP_PIPS_WIDE_THRESHOLD


def test_summarize_observations_splits_base_and_stress_scenarios() -> None:
    rows = pd.DataFrame(
        [
            _row(entry_time="2026-09-24T03:00:00Z", scenario=STRESS_SCENARIO, r_multiple=0.2),
            _row(entry_time="2026-09-25T03:00:00Z", scenario=STRESS_SCENARIO, r_multiple=0.4),
            _row(entry_time="2026-09-24T03:00:00Z", scenario=BASE_SCENARIO, r_multiple=0.3),
            _row(entry_time="2026-09-25T03:00:00Z", scenario=BASE_SCENARIO, r_multiple=0.5),
        ]
    )
    rows["entry_time"] = pd.to_datetime(rows["entry_time"], utc=True)

    summary = summarize_observations(rows, alpha=0.05)

    assert summary["eligible_stress_count"] == 2
    assert summary["eligible_base_count"] == 2
    assert summary["stress"]["mean_r"] == 0.30000000000000004
    assert summary["base"]["mean_r"] == 0.4


def test_evaluate_gates_detects_shadow_candidate_and_early_fail() -> None:
    candidate_rows = []
    for index in range(50):
        candidate_rows.append(
            _row(
                entry_time=f"2027-01-{(index % 28) + 1:02d}T03:00:00Z",
                scenario=STRESS_SCENARIO,
                r_multiple=0.20,
                year=2027 + index // 25,
            )
        )
        candidate_rows.append(
            _row(
                entry_time=f"2027-01-{(index % 28) + 1:02d}T03:00:00Z",
                scenario=BASE_SCENARIO,
                r_multiple=0.10,
                year=2027 + index // 25,
            )
        )
    candidate = summarize_observations(pd.DataFrame(candidate_rows), alpha=0.05)

    assert evaluate_gates(candidate)["shadow_candidate"] is True

    fail_rows = pd.DataFrame(
        [
            _row(
                entry_time=f"2027-01-{index + 1:02d}T03:00:00Z",
                scenario=STRESS_SCENARIO,
                r_multiple=-0.30,
            )
            for index in range(12)
        ]
    )
    failed = summarize_observations(fail_rows, alpha=0.05)

    gates = evaluate_gates(failed)

    assert gates["early_fail"] is True
    assert "early_fail_stress_total_r" in gates["early_fail_reasons"]


def test_fast112_status_never_approves_strategy() -> None:
    empty = summarize_observations(pd.DataFrame(columns=["scenario", "r_multiple", "year"]), alpha=0.05)
    status, reasons = fast112_status(summary=empty, gates=evaluate_gates(empty))

    assert status == "FAST112_OBSERVATION_NO_ELIGIBLE_TRADES_YET"
    assert "not_approved_strategy" in reasons

    candidate_rows = pd.DataFrame(
        [
            _row(entry_time=f"2027-01-{index + 1:02d}T03:00:00Z", scenario=STRESS_SCENARIO, r_multiple=0.01)
            for index in range(3)
        ]
    )
    summary = summarize_observations(candidate_rows, alpha=0.05)
    status, reasons = fast112_status(summary=summary, gates=evaluate_gates(summary))

    assert status == "FAST112_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    assert "insufficient_stress_trades" in reasons
