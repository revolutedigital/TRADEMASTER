from __future__ import annotations

import numpy as np

from scripts.research.run_fast156_fast151_bootstrap_risk_audit import (
    fast156_status,
    max_drawdown,
    simulation_distribution_summary,
)


def test_max_drawdown_uses_running_equity_peak() -> None:
    values = np.array([100.0, -40.0, -90.0, 30.0])

    assert max_drawdown(values) == -130.0


def test_simulation_distribution_summary_reports_tail_probabilities() -> None:
    summary = simulation_distribution_summary(
        total_pips=np.array([10.0, -5.0, 20.0, 0.0]),
        mean_pips=np.array([1.0, -0.5, 2.0, 0.0]),
        drawdowns=np.array([-100.0, -1000.0, -1500.0, -2000.0]),
        extra={"bootstrap_mode": "test"},
    )

    assert summary["probability_total_pips_lte_0"] == 0.5
    assert summary["probability_max_drawdown_lte_1000_pips"] == 0.75
    assert summary["probability_max_drawdown_lte_1500_pips"] == 0.5


def test_fast156_status_flags_risk_tails() -> None:
    status, reasons = fast156_status(
        pre_year_bootstrap={
            "probability_total_pips_lte_0": 0.02,
            "probability_max_drawdown_lte_1000_pips": 0.20,
        },
        current_trade_bootstrap={"probability_max_drawdown_lte_1000_pips": 0.12},
        current_year_bootstrap={"probability_at_least_3_losing_years": 0.31},
    )

    assert status == "FAST156_FAST151_EDGE_POSITIVE_BUT_DRAWDOWN_TAIL_REQUIRES_SIZING_AND_PROSPECTIVE_OBSERVATION"
    assert "pre_year_block_drawdown_tail_above_10pct" in reasons
    assert "not_approved_strategy" in reasons
