import pandas as pd

from scripts.research.run_fast104_fixed_outlier_veto_audit import (
    FixedGateComponent,
    FixedGatePolicy,
    aggregate_holdouts,
    apply_materialized_policy,
    materialize_policy,
)


def test_materialize_policy_uses_training_quantile() -> None:
    rows = pd.DataFrame({"stress_feature": [1.0, 2.0, 3.0, 4.0]})
    policy = FixedGatePolicy(
        "stress_low",
        (FixedGateComponent("stress_feature", "lte", 0.75),),
    )

    materialized = materialize_policy(policy, rows)

    assert len(materialized) == 1
    assert materialized[0].threshold == 3.25


def test_apply_materialized_policy_combines_components() -> None:
    rows = pd.DataFrame(
        {
            "spread": [0.0, 1.0, 2.0, 3.0],
            "return": [-2.0, -0.5, 0.5, 1.0],
        }
    )
    training = pd.DataFrame(
        {
            "spread": [0.0, 1.0, 2.0, 3.0],
            "return": [-2.0, -0.5, 0.5, 1.0],
        }
    )
    policy = FixedGatePolicy(
        "combined",
        (
            FixedGateComponent("spread", "lte", 0.75),
            FixedGateComponent("return", "gte", 0.25),
        ),
    )

    selected = apply_materialized_policy(rows, materialize_policy(policy, training))

    assert selected.to_dict("records") == [
        {"spread": 1.0, "return": -0.5},
        {"spread": 2.0, "return": 0.5},
    ]


def test_aggregate_holdouts_counts_positive_windows_and_minimum_executions() -> None:
    aggregate = aggregate_holdouts(
        [
            holdout("a", gross=6.0, trades=3),
            holdout("b", gross=1.0, trades=2),
        ],
        cost_levels_bps=(0.0, 1.0, 2.0),
        profit_cost_bps=1.0,
        stress_cost_bps=2.0,
        minimum_executable_per_window=3,
    )

    assert aggregate["net_total_by_cost_bps"]["1"] == 2.0
    assert aggregate["positive_window_count_by_cost_bps"]["1"] == 1
    assert aggregate["positive_window_count_by_cost_bps"]["2"] == 0
    assert aggregate["minimum_execution_window_count"] == 1


def holdout(label: str, *, gross: float, trades: int) -> dict[str, object]:
    return {
        "holdout_label": label,
        "selected_signal_count": trades,
        "one_position_summary": {
            "executable_count": trades,
            "gross_executable_total_bps": gross,
            "cost_sensitivity": [
                {
                    "explicit_cost_bps": 0.0,
                    "net_executable_total_bps": gross,
                },
                {
                    "explicit_cost_bps": 1.0,
                    "net_executable_total_bps": gross - trades,
                },
                {
                    "explicit_cost_bps": 2.0,
                    "net_executable_total_bps": gross - 2 * trades,
                },
            ],
        },
    }
