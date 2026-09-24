from __future__ import annotations

import pandas as pd

from scripts.research.run_fast157_fast151_sizing_envelope_audit import (
    add_r_units,
    build_sizing_envelope,
    fast157_status,
    risk_candidate_drawdown_rows,
)


def test_add_r_units_converts_pips_to_r() -> None:
    trades = pd.DataFrame({"net_stress_pips": [50.0, -100.0], "risk_pips": [100.0, 200.0]})

    converted = add_r_units(trades)

    assert converted["net_stress_r"].tolist() == [0.5, -0.5]


def test_build_sizing_envelope_uses_absolute_tail_drawdown() -> None:
    envelope = build_sizing_envelope(tail_drawdown_r=-10.0)

    by_budget = {row["drawdown_budget_fraction"]: row for row in envelope}
    assert by_budget[0.10]["max_risk_per_trade_fraction"] == 0.01
    assert by_budget[0.05]["max_risk_per_trade_percent"] == 0.5


def test_risk_candidate_drawdown_rows_projects_account_drawdown() -> None:
    rows = risk_candidate_drawdown_rows(tail_drawdown_r=-8.0)
    by_risk = {row["risk_per_trade_fraction"]: row for row in rows}

    assert by_risk[0.005]["tail_drawdown_fraction"] == 0.04
    assert by_risk[0.005]["tail_drawdown_percent"] == 4.0


def test_fast157_status_flags_conservative_sizing_need() -> None:
    status, reasons = fast157_status(
        sizing_envelope=[
            {
                "drawdown_budget_fraction": 0.10,
                "max_risk_per_trade_fraction": 0.0075,
                "max_risk_per_trade_percent": 0.75,
            }
        ],
        tail_drawdown_r=-12.0,
    )

    assert status == "FAST157_FAST151_SIZING_ENVELOPE_REQUIRES_CONSERVATIVE_RISK_AND_PROSPECTIVE_OBSERVATION"
    assert "ten_percent_drawdown_budget_requires_less_than_1pct_risk_per_trade" in reasons
    assert "bootstrap_tail_drawdown_exceeds_10r" in reasons
