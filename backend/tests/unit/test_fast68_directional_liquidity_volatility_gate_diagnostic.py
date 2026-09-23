import pandas as pd

from scripts.research.run_fast68_directional_liquidity_volatility_gate_diagnostic import (
    DirectionalGateSpec,
    apply_directional_gate,
    fast68_status,
    generate_specs,
    is_survivor,
    thresholds_from_training,
)


def test_thresholds_from_training_include_directional_floor() -> None:
    frame = pd.DataFrame(
        {
            "quote_volume_30s": [10.0, 20.0, 30.0],
            "realized_vol_30s_bps": [1.0, 2.0, 3.0],
            "directed_return_5s_bps": [-1.0, 0.0, 1.0],
        }
    )
    spec = DirectionalGateSpec(
        quote_volume_floor_quantile=0.5,
        realized_vol_floor_quantile=0.5,
        quote_volume_cap_quantile=None,
        realized_vol_cap_quantile=None,
        directional_feature="directed_return_5s_bps",
        directional_quantile=0.5,
    )

    thresholds = thresholds_from_training(frame, spec)

    assert thresholds.quote_volume_floor == 20.0
    assert thresholds.realized_vol_floor == 2.0
    assert thresholds.directional_floor == 0.0


def test_apply_directional_gate_requires_microstructure_and_direction() -> None:
    frame = pd.DataFrame(
        {
            "quote_volume_30s": [100.0, 100.0, 100.0],
            "realized_vol_30s_bps": [50.0, 50.0, 5.0],
            "directed_return_5s_bps": [1.0, -1.0, 1.0],
        }
    )
    spec = DirectionalGateSpec(
        quote_volume_floor_quantile=0.0,
        realized_vol_floor_quantile=0.0,
        quote_volume_cap_quantile=None,
        realized_vol_cap_quantile=None,
        directional_feature="directed_return_5s_bps",
        directional_quantile=0.5,
    )
    thresholds = thresholds_from_training(
        pd.DataFrame(
            {
                "quote_volume_30s": [100.0],
                "realized_vol_30s_bps": [50.0],
                "directed_return_5s_bps": [0.0],
            }
        ),
        spec,
    )

    filtered = apply_directional_gate(frame, thresholds)

    assert filtered.index.tolist() == [0]


def test_generate_specs_skips_missing_directional_columns() -> None:
    frame = pd.DataFrame(
        {
            "quote_volume_30s": [1.0, 2.0],
            "realized_vol_30s_bps": [1.0, 2.0],
        }
    )

    specs = generate_specs(frame)

    assert specs
    assert {spec.directional_feature for spec in specs} == {None}


def test_is_survivor_requires_no_negative_sufficient_blocks() -> None:
    candidate = {
        "historical_aggregate": {
            "minimum_trade_block_count": 5,
            "negative_sufficient_blocks": [],
            "stress_total_log_return_bps": 100.0,
        },
        "extension_aggregate": {
            "minimum_trade_block_count": 1,
            "negative_sufficient_blocks": [],
            "trade_count": 3,
            "stress_total_log_return_bps": 1.0,
        },
    }

    assert is_survivor(candidate, minimum_block_trades=3, minimum_survivor_blocks=5)

    candidate["extension_aggregate"]["negative_sufficient_blocks"] = ["extension"]

    assert not is_survivor(candidate, minimum_block_trades=3, minimum_survivor_blocks=5)


def test_fast68_status_keeps_survivors_research_only() -> None:
    status, reasons = fast68_status(
        survivor_count=2,
        extension_available=True,
        minimum_survivor_blocks=5,
    )

    assert status == "FAST68_DIRECTIONAL_GATE_DIAGNOSTIC_SURVIVORS_FOUND_REQUIRES_SHADOW"
    assert "requires_clean_future_prospective_shadow_before_testnet" in reasons
