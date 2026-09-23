import pandas as pd

from scripts.research.run_fast64_session_side_filter_walkforward import (
    FilterSpec,
    apply_filter,
    fast64_status,
    select_training_filter,
)


def test_apply_filter_uses_only_side_hour_and_probability() -> None:
    frame = pd.DataFrame(
        [
            {"side": "BUY", "utc_hour": 9, "probability": 0.95},
            {"side": "BUY", "utc_hour": 14, "probability": 0.96},
            {"side": "SELL", "utc_hour": 9, "probability": 0.99},
        ]
    )

    filtered = apply_filter(
        frame,
        FilterSpec(
            name="side=BUY&utc_06_11&probability",
            side="BUY",
            hour_start=6,
            hour_end=11,
            probability_threshold=0.90,
        ),
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["side"] == "BUY"
    assert filtered.iloc[0]["utc_hour"] == 9


def test_select_training_filter_prefers_positive_stress_blocks() -> None:
    rows = []
    for block in ("a", "b"):
        for index in range(4):
            rows.append(_trade(block=block, side="SELL", hour=8, stress=2.0, index=index))
            rows.append(_trade(block=block, side="BUY", hour=18, stress=-4.0, index=index + 10))
    frame = pd.DataFrame(rows)

    selected, evaluations = select_training_filter(frame, minimum_training_block_trades=2)

    assert evaluations
    assert selected.name in {"side=SELL", "utc_06_11", "side=SELL&utc_06_11"}


def test_fast64_status_fails_when_not_all_holdouts_pass() -> None:
    status, reasons = fast64_status(
        aggregate={
            "holdout_count": 3,
            "passed_stress_count": 2,
            "stress_total_log_return_bps": 10.0,
        },
        minimum_holdout_trades=3,
    )

    assert status == "FAST64_SESSION_SIDE_FILTER_FAILED_WALKFORWARD"
    assert "passed_stress_holdouts:2<3" in reasons


def _trade(*, block: str, side: str, hour: int, stress: float, index: int) -> dict[str, object]:
    return {
        "block": block,
        "decision_time_ms": index * 1_000,
        "entry_time_ms": index * 1_000,
        "exit_time_ms": index * 1_000 + 100,
        "side": side,
        "horizon_seconds": 300,
        "probability": 0.95,
        "entry_price": 100.0,
        "exit_price": 100.0,
        "exit_reason": "TIMEOUT",
        "gross_bps": stress + 2.0,
        "expected_net_bps": stress + 1.0,
        "stress_net_bps": stress,
        "maximum_favorable_bps": max(stress, 0.0),
        "maximum_adverse_bps": min(stress, 0.0),
        "partial_take_fraction": 0.0,
        "utc_hour": hour,
    }
