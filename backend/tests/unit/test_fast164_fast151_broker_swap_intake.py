from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.research.run_fast164_fast151_broker_swap_intake import (
    TARGET_PAIRS,
    build_raw_template,
    fast164_status,
    normalize_raw_swap_table,
    raw_to_normalized,
)


def test_raw_to_normalized_converts_points_to_pips_with_row_level_precision() -> None:
    raw = pd.DataFrame(
        {
            "pair": ["EURUSD", "GBPUSD"],
            "long_swap_value": [-6.0, -5.0],
            "short_swap_value": [2.0, 1.5],
            "value_unit": ["points_per_rollover_day", "points_per_rollover_day"],
            "points_per_pip": [10.0, pd.NA],
        }
    )

    normalized = raw_to_normalized(raw, default_points_per_pip=5.0)

    assert normalized.to_dict("records") == [
        {
            "pair": "EURUSD",
            "long_pips_per_rollover_day": -0.6,
            "short_pips_per_rollover_day": 0.2,
        },
        {
            "pair": "GBPUSD",
            "long_pips_per_rollover_day": -1.0,
            "short_pips_per_rollover_day": 0.3,
        },
    ]


def test_normalize_raw_swap_table_accepts_already_normalized_schema(tmp_path: Path) -> None:
    path = tmp_path / "normalized.csv"
    pd.DataFrame(
        {
            "pair": list(reversed(TARGET_PAIRS)),
            "long_pips_per_rollover_day": [-0.1, -0.2, -0.3, -0.4],
            "short_pips_per_rollover_day": [0.1, 0.2, 0.3, 0.4],
            "ignored_extra_column": ["x", "x", "x", "x"],
        }
    ).to_csv(path, index=False)

    normalized = normalize_raw_swap_table(path, points_per_pip=10.0)

    assert normalized["pair"].tolist() == sorted(TARGET_PAIRS)
    assert normalized.columns.tolist() == [
        "pair",
        "long_pips_per_rollover_day",
        "short_pips_per_rollover_day",
    ]


def test_normalize_raw_swap_table_requires_all_target_pairs(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    pd.DataFrame(
        {
            "pair": ["AUDUSD", "EURUSD", "GBPUSD"],
            "long_swap_value": [-1.0, -1.0, -1.0],
            "short_swap_value": [0.1, 0.1, 0.1],
            "value_unit": ["pips_per_rollover_day"] * 3,
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing target pairs"):
        normalize_raw_swap_table(path, points_per_pip=10.0)


def test_raw_to_normalized_rejects_unsupported_unit() -> None:
    raw = pd.DataFrame(
        {
            "pair": ["EURUSD"],
            "long_swap_value": [-6.0],
            "short_swap_value": [2.0],
            "value_unit": ["money_per_lot"],
        }
    )

    with pytest.raises(ValueError, match="unsupported swap unit"):
        raw_to_normalized(raw, default_points_per_pip=10.0)


def test_build_raw_template_includes_sign_convention_and_pair_limits() -> None:
    template = build_raw_template(
        [
            {
                "pair": "EURUSD",
                "current_pair_only_break_even_pips_per_rollover_day": 3.4728,
            }
        ]
    )

    row = template.iloc[0].to_dict()
    assert row["pair"] == "EURUSD"
    assert row["value_unit"] == "pips_per_rollover_day"
    assert row["points_per_pip"] == 10.0
    assert "positive=credit" in row["notes"]
    assert row["current_pair_only_break_even_pips_per_rollover_day"] == 3.4728


def test_fast164_status_keeps_gate_pending_without_raw_broker_table() -> None:
    status, reasons = fast164_status(raw_swap_supplied=False, broker_gate_report=None)

    assert status == "FAST164_BROKER_SWAP_INPUT_PENDING"
    assert "raw_broker_swap_table_not_supplied" in reasons
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons
