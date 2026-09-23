from scripts.research.run_fast63_closest_edge_feasibility_audit import (
    closest_edge_feasibility,
)


def test_closest_edge_feasibility_rejects_thin_worst_window() -> None:
    report = {
        "status": "PAYOFF_MODEL_FAILED_REPLICATION",
        "best_candidate": {
            "model": "ridge_payoff",
            "target_bps": 1.0,
            "trailing_distance_bps": 2.0,
            "top_fraction": 0.02,
            "selected_signal_count": 20,
            "holdout_count": 2,
            "aggregate_by_mode": {
                "one_position": {
                    "executable_count": 10,
                    "gross_mean_bps": 0.5,
                    "net_mean_by_cost_bps": {"0": 0.5, "2": -1.5},
                    "net_total_by_cost_bps": {"0": 5.0, "2": -15.0},
                    "positive_window_count_by_cost_bps": {"0": 2, "2": 1},
                }
            },
            "holdouts": [
                _holdout("fast1", break_even=0.01),
                _holdout("fast2", break_even=1.2),
            ],
        },
    }

    audit = closest_edge_feasibility(
        report,
        stress_cost_bps=2.0,
        minimum_window_cost_buffer_bps=0.25,
    )

    assert audit["status"] == "FAST63_CLOSEST_EDGE_TOO_THIN_AND_REGIME_UNSTABLE"
    assert audit["worst_window"]["label"] == "fast1"
    assert "stress_total_non_positive_at_2bps" in audit["status_reasons"]
    assert any(
        reason.startswith("worst_window_break_even_cost_below_buffer")
        for reason in audit["status_reasons"]
    )


def _holdout(label: str, *, break_even: float) -> dict[str, object]:
    return {
        "label": label,
        "one_position_summary": {
            "executable_count": 5,
            "break_even_explicit_cost_bps": break_even,
            "cost_sensitivity": [
                {"explicit_cost_bps": 0.0, "net_executable_mean_bps": break_even},
                {"explicit_cost_bps": 1.0, "net_executable_mean_bps": break_even - 1.0},
                {"explicit_cost_bps": 2.0, "net_executable_mean_bps": break_even - 2.0},
            ],
        },
    }
