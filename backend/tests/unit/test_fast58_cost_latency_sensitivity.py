import pandas as pd

from scripts.research.run_fast58_cost_latency_sensitivity import (
    build_sensitivity_scenario,
    fast58_status,
    parse_float_grid,
    parse_int_grid,
    summarize_sensitivity_scenarios,
)


def test_parse_grids_deduplicate_and_keep_order() -> None:
    assert parse_int_grid("100,250,100", name="latency") == (100, 250)
    assert parse_float_grid("2,3.5,2", name="cost") == (2.0, 3.5)


def test_sensitivity_summary_reports_break_even_by_latency() -> None:
    scenarios = [
        build_sensitivity_scenario(
            latency_ms=100,
            expected_cost_bps=1.0,
            stress_cost_bps=2.0,
            gated_trades=_trades((5.0, 4.0), stress_cost_bps=2.0),
            ungated_trades=_trades((1.0, -1.0), stress_cost_bps=2.0),
        ),
        build_sensitivity_scenario(
            latency_ms=100,
            expected_cost_bps=1.0,
            stress_cost_bps=5.0,
            gated_trades=_trades((5.0, 4.0), stress_cost_bps=5.0),
            ungated_trades=_trades((1.0, -1.0), stress_cost_bps=5.0),
        ),
        build_sensitivity_scenario(
            latency_ms=500,
            expected_cost_bps=1.0,
            stress_cost_bps=2.0,
            gated_trades=_trades((2.5, 2.0), stress_cost_bps=2.0),
            ungated_trades=_trades((1.0, -1.0), stress_cost_bps=2.0),
        ),
    ]

    summary = summarize_sensitivity_scenarios(scenarios)

    assert summary["scenario_count"] == 3
    assert summary["positive_stress_scenario_count"] == 2
    assert summary["all_scenarios_stress_positive"] is False
    by_latency = summary["by_latency"]
    assert by_latency[0]["latency_ms"] == 100
    assert by_latency[0]["break_even_roundtrip_cost_bps"] == 4.5
    assert by_latency[0]["max_tested_positive_stress_cost_bps"] == 2.0
    assert by_latency[1]["latency_ms"] == 500


def test_fast58_status_distinguishes_fragile_from_all_positive() -> None:
    all_positive = {
        "scenario_count": 2,
        "positive_stress_scenario_count": 2,
        "gated_beats_ungated_scenario_count": 2,
        "all_scenarios_stress_positive": True,
        "all_scenarios_beat_ungated": True,
    }
    fragile = {
        **all_positive,
        "positive_stress_scenario_count": 1,
        "all_scenarios_stress_positive": False,
    }

    assert fast58_status(all_positive)[0] == "FAST58_SENSITIVITY_ALL_POSITIVE_RESEARCH_ONLY"
    assert fast58_status(fragile)[0] == "FAST58_SENSITIVITY_FRAGILE_RESEARCH_ONLY"


def _trades(gross_values: tuple[float, ...], *, stress_cost_bps: float = 2.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time_ms": [1789953790154 + index * 1000 for index in range(len(gross_values))],
            "gross_bps": list(gross_values),
            "expected_net_bps": [gross - 1.0 for gross in gross_values],
            "stress_net_bps": [gross - stress_cost_bps for gross in gross_values],
        }
    )
