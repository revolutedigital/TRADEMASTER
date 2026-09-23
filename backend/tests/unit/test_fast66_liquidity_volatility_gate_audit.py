import pandas as pd

from scripts.research.run_fast66_liquidity_volatility_gate_audit import (
    apply_gate,
    fast66_status,
    make_gate,
    run_fixed_gate_walkforward,
)


def test_make_gate_uses_training_quantiles_only() -> None:
    frame = pd.DataFrame(
        {
            "quote_volume_30s": [10.0, 20.0, 30.0, 40.0],
            "realized_vol_30s_bps": [1.0, 2.0, 3.0, 4.0],
        }
    )

    gate = make_gate(frame, quote_volume_quantile=0.5, realized_vol_quantile=0.5)

    assert gate.quote_volume_threshold == 25.0
    assert gate.realized_vol_threshold == 2.5


def test_apply_gate_requires_liquidity_and_volatility() -> None:
    frame = pd.DataFrame(
        {
            "quote_volume_30s": [100.0, 10.0, 100.0],
            "realized_vol_30s_bps": [50.0, 50.0, 5.0],
        }
    )
    gate = make_gate(
        pd.DataFrame(
            {
                "quote_volume_30s": [50.0, 100.0],
                "realized_vol_30s_bps": [10.0, 50.0],
            }
        ),
        quote_volume_quantile=0.0,
        realized_vol_quantile=0.0,
    )

    filtered = apply_gate(frame, gate)

    assert filtered.index.tolist() == [0]


def test_run_fixed_gate_walkforward_reports_holdout_failures() -> None:
    frame = pd.DataFrame(
        [
            _trade("a", 0, 100.0, 50.0, 2.0),
            _trade("a", 10_000, 100.0, 50.0, 2.0),
            _trade("b", 0, 100.0, 50.0, -3.0),
            _trade("b", 10_000, 100.0, 50.0, -3.0),
        ]
    )

    holdouts = run_fixed_gate_walkforward(
        frame,
        quote_volume_quantile=0.0,
        realized_vol_quantile=0.0,
        minimum_holdout_trades=1,
    )

    assert len(holdouts) == 2
    assert [row["passed_stress"] for row in holdouts] == [True, False]


def test_fast66_status_marks_promising_but_failed_when_not_all_holdouts_pass() -> None:
    status, reasons = fast66_status(
        fixed_aggregate={
            "holdout_count": 3,
            "passed_stress_count": 2,
            "stress_total_log_return_bps": 10.0,
        },
        selected_aggregate={
            "holdout_count": 3,
            "passed_stress_count": 1,
            "stress_total_log_return_bps": 20.0,
        },
        minimum_holdout_trades=3,
    )

    assert status == "FAST66_LIQUIDITY_VOLATILITY_GATE_PROMISING_BUT_FAILED_ROBUSTNESS"
    assert "best_passed_stress_holdouts:2<3" in reasons


def _trade(
    block: str,
    decision_time_ms: int,
    quote_volume: float,
    realized_vol: float,
    stress: float,
) -> dict[str, object]:
    return {
        "block": block,
        "decision_time_ms": decision_time_ms,
        "entry_time_ms": decision_time_ms,
        "exit_time_ms": decision_time_ms + 1_000,
        "probability": 0.9,
        "quote_volume_30s": quote_volume,
        "realized_vol_30s_bps": realized_vol,
        "expected_net_bps": stress + 1.0,
        "stress_net_bps": stress,
    }
