from scripts.research.run_fast86_directional_conflict_audit import (
    decision_timestamp_key,
    fast86_status,
    group_by_decision_timestamp,
    summarize_rows,
    top_probability_per_timestamp,
)


def test_groups_rows_by_label_and_decision_time() -> None:
    rows = [
        row("a", 1, "BUY", 0.9, 1.0),
        row("a", 1, "SELL", 0.8, -1.0),
        row("a", 2, "BUY", 0.7, 2.0),
    ]

    grouped = group_by_decision_timestamp(rows)

    assert decision_timestamp_key(rows[0]) == ("a", 1)
    assert len(grouped[("a", 1)]) == 2
    assert len(grouped[("a", 2)]) == 1


def test_top_probability_per_timestamp_keeps_highest_probability_row() -> None:
    rows = [
        row("a", 1, "BUY", 0.9, 1.0),
        row("a", 1, "SELL", 0.95, -1.0),
        row("a", 2, "BUY", 0.7, 2.0),
    ]

    selected = top_probability_per_timestamp(rows)

    assert [(item["decision_time_ms"], item["side"]) for item in selected] == [
        (1, "SELL"),
        (2, "BUY"),
    ]


def test_summarize_rows_reports_stress_totals_and_exit_counts() -> None:
    summary = summarize_rows(
        [
            row("a", 1, "BUY", 0.9, 1.0, exit_reason="BREAKEVEN_STOP"),
            row("a", 2, "BUY", 0.7, -2.0, exit_reason="INITIAL_STOP"),
        ]
    )

    assert summary["trade_count"] == 2
    assert summary["stress_total_bps"] == -1.0
    assert summary["stress_mean_bps"] == -0.5
    assert summary["breakeven_stop_count"] == 1
    assert summary["initial_stop_count"] == 1


def test_fast86_status_fails_when_no_variant_has_positive_stress() -> None:
    status, reasons = fast86_status(
        {
            "all": {"trade_count": 2, "stress_total_bps": -1.0},
            "empty": {"trade_count": 0, "stress_total_bps": 5.0},
        }
    )

    assert status == "FAST86_DIRECTIONAL_CONFLICT_FILTERS_FAILED"
    assert "no_variant_has_positive_stress_total" in reasons


def test_fast86_positive_variant_is_diagnostic_only() -> None:
    status, reasons = fast86_status(
        {
            "all": {"trade_count": 2, "stress_total_bps": -1.0},
            "buy_only": {"trade_count": 1, "stress_total_bps": 3.0},
        }
    )

    assert status == "FAST86_DIRECTIONAL_CONFLICT_FILTER_HAS_POSITIVE_DIAGNOSTIC_ONLY"
    assert "requires_preregistration_before_reuse" in reasons


def row(
    label: str,
    decision_time_ms: int,
    side: str,
    probability: float,
    stress_net_bps: float,
    *,
    exit_reason: str = "TIMEOUT",
) -> dict[str, object]:
    return {
        "label": label,
        "decision_time_ms": decision_time_ms,
        "side": side,
        "probability": probability,
        "expected_net_bps": stress_net_bps + 1.0,
        "stress_net_bps": stress_net_bps,
        "exit_reason": exit_reason,
    }
