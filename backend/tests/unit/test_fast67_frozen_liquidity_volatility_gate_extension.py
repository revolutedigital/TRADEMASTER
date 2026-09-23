import pandas as pd

from scripts.research.run_fast67_frozen_liquidity_volatility_gate_extension import (
    aggregate_block_summaries,
    apply_frozen_gate,
    fast67_status,
    freeze_gate_from_historical,
    summarize_blocks,
)


def test_freeze_gate_uses_historical_numeric_thresholds() -> None:
    historical = pd.DataFrame(
        {
            "quote_volume_30s": [10.0, 20.0, 30.0, 40.0],
            "realized_vol_30s_bps": [1.0, 2.0, 3.0, 4.0],
        }
    )

    gate = freeze_gate_from_historical(
        historical,
        quote_volume_quantile=0.5,
        realized_vol_quantile=0.5,
        calibration_scope="unit",
    )

    assert gate.quote_volume_threshold == 25.0
    assert gate.realized_vol_threshold == 2.5
    assert gate.calibration_scope == "unit"


def test_apply_frozen_gate_requires_both_conditions() -> None:
    frame = pd.DataFrame(
        {
            "quote_volume_30s": [100.0, 100.0, 10.0],
            "realized_vol_30s_bps": [50.0, 5.0, 50.0],
        }
    )
    gate = freeze_gate_from_historical(
        pd.DataFrame(
            {
                "quote_volume_30s": [100.0],
                "realized_vol_30s_bps": [50.0],
            }
        ),
        quote_volume_quantile=0.0,
        realized_vol_quantile=0.0,
        calibration_scope="unit",
    )

    filtered = apply_frozen_gate(frame, gate)

    assert filtered.index.tolist() == [0]


def test_summarize_blocks_keeps_empty_blocks_visible() -> None:
    blocks = summarize_blocks(
        pd.DataFrame([_trade("a", 1.0)]),
        all_block_names=("a", "b"),
        minimum_block_trades=1,
    )

    assert [row["block"] for row in blocks] == ["a", "b"]
    assert [row["trade_count"] for row in blocks] == [1, 0]


def test_fast67_status_fails_on_extension_negative_sufficient_block() -> None:
    historical_blocks = summarize_blocks(
        pd.DataFrame([_trade("historical", 2.0)]),
        all_block_names=("historical",),
        minimum_block_trades=1,
    )
    extension_blocks = summarize_blocks(
        pd.DataFrame([_trade("extension", -3.0)]),
        all_block_names=("extension",),
        minimum_block_trades=1,
    )

    status, reasons = fast67_status(
        historical_aggregate=aggregate_block_summaries(
            historical_blocks,
            minimum_block_trades=1,
        ),
        extension_aggregate=aggregate_block_summaries(
            extension_blocks,
            minimum_block_trades=1,
        ),
        minimum_block_trades=1,
    )

    assert status == "FAST67_FROZEN_GATE_FAILED_EXTENSION"
    assert "extension_negative_sufficient_blocks:1" in reasons


def _trade(block: str, stress_net_bps: float) -> dict[str, object]:
    return {
        "block": block,
        "decision_time_ms": 1_000,
        "entry_time_ms": 1_000,
        "exit_time_ms": 2_000,
        "probability": 0.9,
        "expected_net_bps": stress_net_bps + 1.0,
        "stress_net_bps": stress_net_bps,
    }
