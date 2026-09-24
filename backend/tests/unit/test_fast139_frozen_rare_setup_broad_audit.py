from __future__ import annotations

from scripts.research.run_fast139_frozen_rare_setup_broad_audit import (
    FROZEN_RULES,
    apply_one_position_by_label,
    fast139_status,
    rule_matches,
)


def _row(
    *,
    label: str = "prospective-wal-fast82-30m",
    decision_time_ms: int = 1_000,
    exit_time_ms: int = 2_000,
    probability: float = 0.91,
    threshold: float = 0.90,
    stress_net_bps: float = 2.0,
) -> dict:
    return {
        "label": label,
        "decision_time_ms": decision_time_ms,
        "exit_time_ms": exit_time_ms,
        "probability": probability,
        "threshold": threshold,
        "side": "BUY",
        "realized_vol_30s_bps": 50.0,
        "flow_imbalance_5s": -0.3,
        "quote_volume_30s": 90_000_000.0,
        "directed_flow_imbalance_30s": -0.2,
        "expected_net_bps": stress_net_bps + 1.0,
        "stress_net_bps": stress_net_bps,
    }


def test_fast80_top_p_rule_requires_probability_threshold() -> None:
    top_p_rule = next(rule for rule in FROZEN_RULES if rule.name == "fast80_rare_vol_flow_top_p_only")
    near_rule = next(rule for rule in FROZEN_RULES if rule.name == "fast80_rare_vol_flow_near200")

    assert rule_matches(near_rule, _row(probability=0.80, threshold=0.90))
    assert not rule_matches(top_p_rule, _row(probability=0.80, threshold=0.90))


def test_apply_one_position_is_label_scoped() -> None:
    rows = [
        _row(label="a", decision_time_ms=1_000, exit_time_ms=3_000),
        _row(label="a", decision_time_ms=2_000, exit_time_ms=4_000),
        _row(label="b", decision_time_ms=2_000, exit_time_ms=4_000),
    ]

    selected = apply_one_position_by_label(rows)

    assert [(row["label"], row["decision_time_ms"]) for row in selected] == [("a", 1_000), ("b", 2_000)]


def test_fast139_status_reports_no_late_coverage() -> None:
    status, reasons = fast139_status(
        [
            {
                "late_one_position_summary": {
                    "trade_count": 0,
                    "stress_total_bps": 0.0,
                    "negative_label_count": 0,
                }
            }
        ],
        minimum_late_trades=5,
    )

    assert status == "FAST139_FROZEN_RARE_SETUP_NO_LATE_AUDIT_COVERAGE"
    assert "not_approved_strategy" in reasons


def test_fast139_status_never_approves_survivor() -> None:
    status, reasons = fast139_status(
        [
            {
                "late_one_position_summary": {
                    "trade_count": 5,
                    "stress_total_bps": 10.0,
                    "negative_label_count": 0,
                }
            }
        ],
        minimum_late_trades=5,
    )

    assert status == "FAST139_FROZEN_RARE_SETUP_DIAGNOSTIC_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "no_order_execution" in reasons
