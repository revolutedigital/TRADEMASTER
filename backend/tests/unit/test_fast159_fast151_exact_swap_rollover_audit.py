from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.research.run_fast159_fast151_exact_swap_rollover_audit import (
    add_rollover_days,
    apply_broker_swap_schedule,
    apply_uniform_adverse_swap_cost,
    estimate_uniform_break_even_pips_per_rollover_day,
    fast159_status,
    load_broker_swap_schedule,
)


def test_add_rollover_days_uses_new_york_rollover_and_wednesday_triple() -> None:
    trades = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                [
                    "2024-01-09T14:00:00Z",
                    "2024-01-10T14:00:00Z",
                    "2024-01-11T14:00:00Z",
                    "2024-01-09T14:00:00Z",
                ],
                utc=True,
            ),
            "exit_stress_timestamp": pd.to_datetime(
                [
                    "2024-01-10T23:00:00Z",
                    "2024-01-11T23:00:00Z",
                    "2024-01-11T20:00:00Z",
                    "2024-01-16T23:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    enriched = add_rollover_days(trades)

    assert enriched["swap_rollover_days"].tolist() == [4, 4, 0, 8]


def test_apply_uniform_adverse_swap_cost_subtracts_rollover_days() -> None:
    trades = pd.DataFrame({"net_stress_pips": [10.0, -5.0], "swap_rollover_days": [2, 4]})

    adjusted = apply_uniform_adverse_swap_cost(trades, adverse_short_pips_per_rollover_day=0.75)

    assert adjusted["swap_cost_pips"].tolist() == [1.5, 3.0]
    assert adjusted["swap_pips"].tolist() == [-1.5, -3.0]
    assert adjusted["net_stress_pips"].tolist() == [8.5, -8.0]


def test_apply_broker_swap_schedule_uses_signed_side_specific_swap() -> None:
    trades = pd.DataFrame(
        {
            "pair": ["EURUSD", "EURUSD"],
            "side": ["LONG", "SHORT"],
            "net_stress_pips": [10.0, 10.0],
            "swap_rollover_days": [3, 3],
        }
    )
    schedule = pd.DataFrame(
        {
            "pair": ["EURUSD"],
            "long_pips_per_rollover_day": [-0.6],
            "short_pips_per_rollover_day": [0.2],
        }
    )

    adjusted = apply_broker_swap_schedule(trades, schedule)

    assert adjusted["swap_pips"].tolist() == [-1.7999999999999998, 0.6000000000000001]
    assert adjusted["net_stress_pips"].tolist() == [8.2, 10.6]


def test_load_broker_swap_schedule_validates_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "swap.csv"
    path.write_text("pair,short_pips_per_rollover_day\nEURUSD,-0.4\n")

    with pytest.raises(ValueError, match="missing columns"):
        load_broker_swap_schedule(path)


def test_estimate_uniform_break_even_pips_per_rollover_day_uses_exact_rollover_days() -> None:
    trades = pd.DataFrame(
        {
            "pair": ["EURUSD", "GBPUSD", "AUDUSD"],
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
                utc=True,
            ),
            "exit_stress_timestamp": pd.to_datetime(
                ["2026-01-03T00:00:00Z", "2026-01-04T00:00:00Z", "2026-01-05T00:00:00Z"],
                utc=True,
            ),
            "net_stress_pips": [10.0, 10.0, 10.0],
            "exit_stress_reason": ["hold", "hold", "hold"],
        }
    )
    enriched = trades.assign(swap_rollover_days=2)

    assert estimate_uniform_break_even_pips_per_rollover_day(enriched) == pytest.approx(5.0)


def test_fast159_status_requires_broker_table_when_missing() -> None:
    def evaluation(identifier: str, current_lcb_positive: bool) -> dict:
        return {"scenario_id": identifier, "current_lcb_positive": current_lcb_positive}

    status, reasons = fast159_status(
        uniform_evaluations=[evaluation("adverse_short_swap_1_00_pip_per_rollover_day", False)],
        break_even={"current_lcb_zero_pips_per_rollover_day": 0.95},
        broker_schedule_supplied=False,
        broker_evaluation=None,
    )

    assert status == "FAST159_FAST151_EXACT_ROLLOVER_CONFIRMS_TIGHT_SWAP_MARGIN_REQUIRES_BROKER_TABLE"
    assert "current_lcb_fails_at_1_00_pip_per_rollover_day" in reasons
    assert "broker_swap_schedule_missing" in reasons
    assert "not_approved_strategy" in reasons
