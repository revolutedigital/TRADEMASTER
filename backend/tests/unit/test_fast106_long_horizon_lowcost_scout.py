from datetime import date

import numpy as np
import pandas as pd

from scripts.research import run_fast106_long_horizon_lowcost_scout as fast106


def test_exact_target_net_uses_target_specific_stop_order() -> None:
    rows = pd.DataFrame(
        {
            "first_touch": ["STOP", "BREAKEVEN", "TIMEOUT"],
            "paid_expected_before_stop": [False, True, False],
            "terminal_bps": [4.0, 5.0, 3.0],
        }
    )

    net = fast106.exact_target_net_bps(
        rows,
        target_bps=12.0,
        explicit_cost_bps=2.0,
        initial_stop_bps=20.0,
    )

    assert net.tolist() == [-22.0, 10.0, 1.0]


def test_select_top_one_position_skips_overlapping_horizon() -> None:
    rows = pd.DataFrame(
        {
            "decision_time_ms": [0, 10_000, 70_000],
            "score": [3.0, 5.0, 4.0],
        }
    )

    selected = fast106.select_top_one_position(
        rows,
        score_threshold_value=0.0,
        horizon_seconds=60,
    )

    assert selected["decision_time_ms"].tolist() == [0, 70_000]


def test_fast106_status_rejects_screen_survivor_that_fails_extension() -> None:
    status, reasons = fast106.fast106_status(
        screen_survivors=[{"candidate": "x"}],
        validations=[
            {
                "historical": {
                    "negative_sufficient_blocks": [],
                    "net_total_bps": 10.0,
                },
                "extension": {"trade_count": 3, "net_total_bps": -1.0},
            }
        ],
    )

    assert status == "FAST106_LONG_HORIZON_FAILED_WALKFORWARD_OR_EXTENSION"
    assert "no_screen_survivor_passed_walkforward_and_extension" in reasons


def test_build_block_rows_reuses_loaded_day_across_targets(tmp_path, monkeypatch) -> None:
    root = tmp_path / "research-lowcost-fast99-holdout"
    (root / "date=2026-01-01").mkdir(parents=True)
    load_calls = []

    def fake_load_date_trades_and_decisions(**kwargs):
        load_calls.append(kwargs)
        return pd.DataFrame({"event_time_ms": [0, 1]}), np.array([1000], dtype=np.int64)

    def fake_build_rows_from_trades(**kwargs):
        return pd.DataFrame(
            {
                "decision_time_ms": kwargs["decisions"],
                "net_bps": [float(kwargs["target_bps"])],
            }
        )

    monkeypatch.setattr(fast106, "load_date_trades_and_decisions", fake_load_date_trades_and_decisions)
    monkeypatch.setattr(fast106, "build_rows_from_trades", fake_build_rows_from_trades)

    rows_by_target = fast106.build_block_rows_by_target(
        root=root,
        trade_root=tmp_path / "aggTrades",
        horizon_seconds=900,
        target_bps_values=(8.0, 12.0),
        decision_stride_seconds=300,
        explicit_cost_bps=2.0,
        initial_stop_bps=20.0,
        max_days_per_root=None,
    )

    assert len(load_calls) == 1
    assert load_calls[0]["utc_date"] == date(2026, 1, 1)
    assert rows_by_target[8.0]["net_bps"].tolist() == [8.0]
    assert rows_by_target[12.0]["net_bps"].tolist() == [12.0]
