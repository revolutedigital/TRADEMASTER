from __future__ import annotations

import pandas as pd

from scripts.research.run_fast154_fast151_cost_sensitivity_audit import (
    CostScenario,
    apply_cost_scenario,
    cost_drag_diagnostics,
    fast154_status,
)


def test_apply_cost_scenario_subtracts_fixed_extra_pips() -> None:
    trades = pd.DataFrame({"net_stress_pips": [10.0, -5.0]})

    adjusted = apply_cost_scenario(trades, CostScenario("stress_plus_2pips", extra_fixed_pips=2.0))

    assert adjusted["net_stress_pips"].tolist() == [8.0, -7.0]


def test_apply_cost_scenario_multiplies_base_to_stress_drag() -> None:
    trades = pd.DataFrame({"net_base_pips": [10.0, -5.0], "net_stress_pips": [8.0, -8.0]})

    adjusted = apply_cost_scenario(
        trades,
        CostScenario("double_base_to_stress_drag", base_to_stress_drag_multiplier=2.0),
    )

    assert adjusted["net_stress_pips"].tolist() == [6.0, -11.0]


def test_cost_drag_diagnostics_reports_expected_means() -> None:
    trades = pd.DataFrame(
        {
            "raw_base_pips": [10.0, 20.0],
            "net_base_pips": [9.0, 18.0],
            "raw_stress_pips": [8.0, 15.0],
            "net_stress_pips": [6.0, 12.0],
        }
    )

    diagnostics = cost_drag_diagnostics(trades)

    assert diagnostics["mean_base_execution_cost_pips"] == 1.5
    assert diagnostics["mean_stress_execution_cost_pips"] == 2.5
    assert diagnostics["mean_total_base_to_stress_delta_pips"] == 4.5


def test_fast154_status_flags_when_harsher_cost_breaks_lcb() -> None:
    def evaluation(identifier: str, passed: bool) -> dict:
        return {"scenario_id": identifier, "both_lcb_positive": passed}

    status, reasons = fast154_status(
        [
            evaluation("stress_plus_5pips", True),
            evaluation("stress_plus_10pips", False),
            evaluation("double_base_to_stress_drag", True),
        ]
    )

    assert status == "FAST154_FAST151_COST_MARGIN_POSITIVE_BUT_NOT_UNLIMITED_REQUIRES_PROSPECTIVE_OBSERVATION"
    assert "fails_extra_10_pips_lcb_gate" in reasons
    assert "not_approved_strategy" in reasons
