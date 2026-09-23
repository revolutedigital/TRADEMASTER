import pandas as pd

from scripts.research.run_fast96_fast95_regime_gate_audit import (
    GateSpec,
    apply_gate_frame,
    apply_one_position_by_label,
    fast96_status,
    holdout_status,
    summarize_labelled_one_position_rows,
    train_candidate_score,
    training_summary_survives,
)


def test_apply_gate_frame_filters_lte_and_gte() -> None:
    rows = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "gross_executable_bps": [1.0, 1.0, 1.0],
        }
    )

    assert apply_gate_frame(rows, GateSpec("feature", "lte", 0.5, 2.0, 0.5))[
        "feature"
    ].tolist() == [1.0, 2.0]
    assert apply_gate_frame(rows, GateSpec("feature", "gte", 0.5, 2.0, 0.5))[
        "feature"
    ].tolist() == [2.0, 3.0]


def test_apply_one_position_by_label_keeps_non_overlapping_per_window() -> None:
    rows = pd.DataFrame(
        [
            row("a", 1000, 2000, 0.90),
            row("a", 1500, 2100, 0.99),
            row("a", 2101, 2500, 0.70),
            row("b", 1000, 2000, 0.10),
        ]
    )

    selected = apply_one_position_by_label(rows)

    assert selected[["window_label", "decision_time_ms"]].to_dict("records") == [
        {"window_label": "a", "decision_time_ms": 1000},
        {"window_label": "a", "decision_time_ms": 2101},
        {"window_label": "b", "decision_time_ms": 1000},
    ]


def test_training_summary_requires_profit_positive_enough_labels_and_no_negatives() -> None:
    good = summarize_labelled_one_position_rows(
        pd.DataFrame(
            [
                row("a", 1000, 2000, 1.0, gross=3.0),
                row("b", 3000, 4000, 1.0, gross=3.0),
            ]
        ),
        expected_labels=["a", "b"],
        selected_signal_count=2,
        profit_cost_bps=1.0,
        stress_cost_bps=2.0,
    )
    bad = summarize_labelled_one_position_rows(
        pd.DataFrame(
            [
                row("a", 1000, 2000, 1.0, gross=3.0),
                row("b", 3000, 4000, 1.0, gross=-3.0),
            ]
        ),
        expected_labels=["a", "b"],
        selected_signal_count=2,
        profit_cost_bps=1.0,
        stress_cost_bps=2.0,
    )

    assert training_summary_survives(
        good,
        minimum_train_trades=2,
        minimum_train_labels=2,
        maximum_train_negative_labels=0,
    )
    assert not training_summary_survives(
        bad,
        minimum_train_trades=2,
        minimum_train_labels=2,
        maximum_train_negative_labels=0,
    )


def test_train_candidate_score_prefers_better_worst_label() -> None:
    weak = summary(worst=0.1, total=20.0)
    strong = summary(worst=3.0, total=18.0)

    assert train_candidate_score(strong) > train_candidate_score(weak)


def test_holdout_status_separates_profit_negative_and_stress_fragile() -> None:
    assert (
        holdout_status(
            executable_summary(trades=0, gross_total=10.0),
            profit_cost_bps=1.0,
            stress_cost_bps=2.0,
            minimum_holdout_trades=1,
        )
        == "FAST96_HOLDOUT_INSUFFICIENT_TRADES"
    )
    assert (
        holdout_status(
            executable_summary(trades=2, gross_total=1.0),
            profit_cost_bps=1.0,
            stress_cost_bps=2.0,
            minimum_holdout_trades=1,
        )
        == "FAST96_HOLDOUT_PROFIT_COST_NON_POSITIVE"
    )
    assert (
        holdout_status(
            executable_summary(trades=2, gross_total=3.0),
            profit_cost_bps=1.0,
            stress_cost_bps=2.0,
            minimum_holdout_trades=1,
        )
        == "FAST96_HOLDOUT_PROFIT_POSITIVE_STRESS_FRAGILE"
    )
    assert (
        holdout_status(
            executable_summary(trades=2, gross_total=5.0),
            profit_cost_bps=1.0,
            stress_cost_bps=2.0,
            minimum_holdout_trades=1,
        )
        == "FAST96_HOLDOUT_STRESS_POSITIVE"
    )


def test_fast96_status_rejects_partial_replication() -> None:
    status, reasons = fast96_status(
        {
            "holdout_count": 2,
            "stress_positive_holdout_count": 1,
            "profit_positive_holdout_count": 1,
            "profit_negative_holdout_count": 1,
            "insufficient_holdout_count": 0,
            "no_strict_training_survivor_count": 0,
        }
    )

    assert status == "FAST96_FAST95_REGIME_GATE_FAILED_REPLICATION"
    assert "profit_positive_holdouts:1/2" in reasons


def row(
    label: str,
    decision_time_ms: int,
    exit_time_ms: int,
    score: float,
    *,
    gross: float = 1.0,
) -> dict[str, object]:
    return {
        "window_label": label,
        "decision_time_ms": decision_time_ms,
        "exit_time_ms": exit_time_ms,
        "model_score": score,
        "gross_executable_bps": gross,
    }


def summary(*, worst: float, total: float) -> dict[str, object]:
    return {
        "trade_count": 10,
        "label_count": 3,
        "empty_label_count": 0,
        "profit_net_total_bps": total,
        "profit_net_mean_bps": total / 10,
        "stress_net_total_bps": total - 5,
        "positive_label_count": 3,
        "negative_label_count": 0,
        "worst_label_profit_net_total_bps": worst,
    }


def executable_summary(*, trades: int, gross_total: float) -> dict[str, object]:
    return {
        "executable_count": trades,
        "cost_sensitivity": [
            {
                "explicit_cost_bps": 1.0,
                "net_executable_total_bps": gross_total - trades,
            },
            {
                "explicit_cost_bps": 2.0,
                "net_executable_total_bps": gross_total - 2 * trades,
            },
        ],
    }
