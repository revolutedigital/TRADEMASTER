from __future__ import annotations

import pandas as pd
import pytest

from scripts.research.run_fast158_fast151_carry_cost_audit import (
    add_carry_days,
    apply_carry_cost,
    estimate_lcb_break_even_pips_per_day,
    fast158_status,
)


def test_add_carry_days_ceil_elapsed_days_with_minimum_one_day() -> None:
    trades = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
                utc=True,
            ),
            "exit_stress_timestamp": pd.to_datetime(
                ["2026-01-01T12:00:00Z", "2026-01-03T00:00:00Z", "2026-01-03T01:00:00Z"],
                utc=True,
            ),
        }
    )

    enriched = add_carry_days(trades)

    assert enriched["carry_elapsed_days"].tolist() == [0.5, 2.0, pytest.approx(2.0416666666666665)]
    assert enriched["carry_billed_days"].tolist() == [1, 2, 3]


def test_apply_carry_cost_subtracts_billed_days_times_daily_carry() -> None:
    trades = pd.DataFrame({"net_stress_pips": [10.0, -5.0], "carry_billed_days": [2, 4]})

    adjusted = apply_carry_cost(trades, adverse_pips_per_billed_day=0.75)

    assert adjusted["carry_cost_pips"].tolist() == [1.5, 3.0]
    assert adjusted["net_stress_pips"].tolist() == [8.5, -8.0]


def test_estimate_lcb_break_even_pips_per_day_uses_billed_exposure() -> None:
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

    enriched = add_carry_days(trades)

    assert estimate_lcb_break_even_pips_per_day(enriched) == pytest.approx(5.0)


def test_fast158_status_flags_tight_current_carry_margin() -> None:
    def evaluation(identifier: str, current_lcb_positive: bool) -> dict:
        return {"scenario_id": identifier, "current_lcb_positive": current_lcb_positive}

    status, reasons = fast158_status(
        evaluations=[
            evaluation("carry_0_75_pip_per_day", True),
            evaluation("carry_1_00_pip_per_day", False),
        ],
        break_even={"current_lcb_zero_pips_per_day": 0.9},
    )

    assert status == "FAST158_FAST151_CARRY_MARGIN_POSITIVE_BUT_TIGHT_REQUIRES_SWAP_MEASUREMENT"
    assert "current_lcb_fails_at_1_00_pip_per_day" in reasons
    assert "current_lcb_break_even_below_1_pip_per_day" in reasons
    assert "not_approved_strategy" in reasons
