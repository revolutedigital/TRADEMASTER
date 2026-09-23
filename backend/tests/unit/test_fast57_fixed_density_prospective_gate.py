from scripts.research.run_fast57_fixed_density_prospective_gate import (
    evaluate_fast57_status,
)


def _summary(
    *,
    trades: int,
    stress_total: float,
    stress_mean: float,
    drawdown: float = 10.0,
) -> dict[str, float | int]:
    return {
        "trade_count": trades,
        "stress_total_log_return_bps": stress_total,
        "stress_mean_bps": stress_mean,
        "maximum_drawdown_bps": drawdown,
    }


def test_fast57_waits_for_minimum_sample() -> None:
    status, reasons, criteria = evaluate_fast57_status(
        gated_summary=_summary(trades=28, stress_total=68, stress_mean=2.4),
        baseline_summary=_summary(trades=121, stress_total=-12, stress_mean=-0.1),
        complete_utc_days=2,
        gated_daily_summary=[
            {"stress_total_log_return_bps": 30},
            {"stress_total_log_return_bps": 38},
        ],
        gated_block_summary=[{"stress_total_log_return_bps": 68}],
        minimum_days=10,
        minimum_trades=100,
        maximum_days=30,
    )

    assert status == "FAST57_INSUFFICIENT_PROSPECTIVE_SAMPLE"
    assert "complete_days_below_minimum:2<10" in reasons
    assert "trade_count_below_minimum:28<100" in reasons
    assert criteria["minimum_sample_met"] is False


def test_fast57_passes_only_when_all_confirmation_criteria_pass() -> None:
    status, reasons, criteria = evaluate_fast57_status(
        gated_summary=_summary(trades=120, stress_total=240, stress_mean=2, drawdown=80),
        baseline_summary=_summary(trades=300, stress_total=-10, stress_mean=-0.03),
        complete_utc_days=10,
        gated_daily_summary=[
            {"stress_total_log_return_bps": 20},
            {"stress_total_log_return_bps": 10},
            {"stress_total_log_return_bps": 30},
            {"stress_total_log_return_bps": 15},
            {"stress_total_log_return_bps": 12},
            {"stress_total_log_return_bps": 18},
            {"stress_total_log_return_bps": -5},
            {"stress_total_log_return_bps": -3},
            {"stress_total_log_return_bps": 40},
            {"stress_total_log_return_bps": 103},
        ],
        gated_block_summary=[
            {"stress_total_log_return_bps": 120},
            {"stress_total_log_return_bps": -20},
            {"stress_total_log_return_bps": 140},
        ],
        minimum_days=10,
        minimum_trades=100,
        maximum_days=30,
    )

    assert status == "FAST57_PASSED_RESEARCH_ONLY_NEXT_GATE_REQUIRED"
    assert "not_approved_strategy" in reasons
    assert criteria["gated_beats_ungated"] is True


def test_fast57_fails_when_gated_does_not_beat_ungated() -> None:
    status, reasons, criteria = evaluate_fast57_status(
        gated_summary=_summary(trades=120, stress_total=100, stress_mean=0.8, drawdown=50),
        baseline_summary=_summary(trades=300, stress_total=120, stress_mean=0.4),
        complete_utc_days=10,
        gated_daily_summary=[
            {"stress_total_log_return_bps": value}
            for value in (10, 12, 8, 15, 9, 11, -1, -2, 20, 18)
        ],
        gated_block_summary=[{"stress_total_log_return_bps": 100}],
        minimum_days=10,
        minimum_trades=100,
        maximum_days=30,
    )

    assert status == "FAST57_FAILED_CONFIRMATION"
    assert "failed:gated_beats_ungated" in reasons
    assert criteria["gated_beats_ungated"] is False
