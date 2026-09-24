from __future__ import annotations

from scripts.research.run_fast163_fast151_decision_gate import evaluate_decision_gate


def _report(status: str = "OK") -> dict:
    return {"status": status}


def test_decision_gate_fails_closed_when_required_report_is_missing() -> None:
    gate = evaluate_decision_gate(
        {
            "fast152_readiness": None,
            "fast156_bootstrap_risk": _report(),
            "fast157_sizing": _report(),
            "fast159_exact_swap": _report(),
            "fast161_subbasket": _report(),
            "fast162_management": _report(),
        }
    )

    assert gate["status"] == "FAST163_MISSING_REQUIRED_REPORTS"
    assert gate["approved_strategy"] is False
    assert gate["shadow_allowed"] is False
    assert gate["hard_blockers"][0]["id"] == "missing_required_reports"


def test_decision_gate_blocks_candidate_without_forward_data_and_broker_swap() -> None:
    gate = evaluate_decision_gate(
        {
            "fast152_readiness": {
                "status": "FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA",
                "has_post_preregistration_m1_data": False,
                "eligible_observation_count": 0,
                "base_candidate_count": 0,
                "fast151_observation_status": "FAST151_OBSERVATION_NO_ELIGIBLE_OPERATIONAL_TRADES_YET",
                "raw_m1_min_pair_end": "2026-09-18T20:59:00+00:00",
                "next_action": "refresh_or_import_m1_after_fast151_preregistration_then_rerun_fast151_and_fast152",
            },
            "fast156_bootstrap_risk": {
                "status": "FAST156_FAST151_EDGE_POSITIVE_BUT_DRAWDOWN_TAIL_REQUIRES_SIZING_AND_PROSPECTIVE_OBSERVATION",
                "summary": {
                    "pre": {
                        "actual": {"trade_count": 51, "mean_stress_pips": 67.0},
                        "year_block_bootstrap": {
                            "probability_max_drawdown_lte_1000_pips": 0.2,
                            "probability_max_drawdown_lte_1500_pips": 0.1,
                        },
                    },
                    "current": {
                        "actual": {"trade_count": 86, "mean_stress_pips": 33.0},
                        "trade_bootstrap": {
                            "probability_total_pips_lte_0": 0.01,
                            "probability_max_drawdown_lte_1000_pips": 0.12,
                        },
                        "year_block_bootstrap": {"probability_at_least_3_losing_years": 0.31},
                    },
                },
            },
            "fast157_sizing": {
                "status": "FAST157_FAST151_SIZING_ENVELOPE_REQUIRES_CONSERVATIVE_RISK_AND_PROSPECTIVE_OBSERVATION",
                "summary": {
                    "worst_tail_drawdown_r_p05": -12.69,
                    "sizing_envelope": [
                        {"drawdown_budget_fraction": 0.1, "max_risk_per_trade_percent": 0.78}
                    ],
                    "risk_candidate_drawdowns": [
                        {
                            "risk_per_trade_fraction": 0.0025,
                            "tail_drawdown_percent": 3.17,
                        },
                        {
                            "risk_per_trade_fraction": 0.005,
                            "tail_drawdown_percent": 6.34,
                        },
                        {
                            "risk_per_trade_fraction": 0.01,
                            "tail_drawdown_percent": 12.69,
                        },
                    ],
                },
            },
            "fast159_exact_swap": {
                "status": "FAST159_FAST151_EXACT_ROLLOVER_CONFIRMS_TIGHT_SWAP_MARGIN_REQUIRES_BROKER_TABLE",
                "broker_swap_path": None,
                "summary": {
                    "break_even": {
                        "current_lcb_zero_pips_per_rollover_day": 0.95,
                        "pre_lcb_zero_pips_per_rollover_day": 3.1,
                    },
                    "rollover_day_diagnostics": {
                        "current": {
                            "mean_rollover_days": 9.8,
                            "total_rollover_days": 845,
                        }
                    },
                },
            },
            "fast161_subbasket": {
                "status": "FAST161_STRICT_GATE_KEEP_CURRENT_BASKET_NO_SUBBASKET_REPLACEMENT",
                "summary": {
                    "strict_selected_basket": {
                        "basket_id": "AUDUSD+EURUSD+GBPUSD+NZDUSD",
                        "pre": {"summary": {"lower_stress_pips": 32.0}},
                        "current": {
                            "summary": {
                                "lower_stress_pips": 9.6,
                                "trade_count": 86,
                            }
                        },
                    }
                },
            },
            "fast162_management": {
                "status": "FAST162_FIXED_1R_REMAINS_PREFERRED_OVER_TRAILING",
                "summary": {
                    "strict_selected_variant": {
                        "risk_policy_id": "stop2_fixed_1r",
                        "current": {
                            "summary": {"lower_stress_pips": 9.6},
                            "uniform_swap_break_even_lcb_zero_pips_per_rollover_day": 0.95,
                        },
                    },
                    "best_trailing_variant": {
                        "risk_policy_id": "stop1.5_trail_1r_1r",
                        "current": {"summary": {"lower_stress_pips": 4.0}},
                    },
                },
            },
        }
    )

    assert gate["status"] == "FAST163_FAST151_CANDIDATE_BLOCKED_BY_FORWARD_DATA_AND_BROKER_SWAP"
    assert gate["approved_strategy"] is False
    assert gate["shadow_allowed"] is False
    blocker_ids = {blocker["id"] for blocker in gate["hard_blockers"]}
    assert "post_preregistration_m1_data_missing" in blocker_ids
    assert "broker_swap_table_missing" in blocker_ids
    assert "explicit_execution_authorization_missing" in blocker_ids
