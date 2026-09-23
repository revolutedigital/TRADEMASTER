import pandas as pd

from scripts.research.run_fast62_payoff_filter_survivorship_audit import (
    discover_historical_holdout_trade_paths,
    fast60_filter_candidates,
    fast62_status,
    historical_side_survivorship,
)


def test_fast60_filter_candidates_identifies_positive_side_filter() -> None:
    rows = []
    for index in range(6):
        rows.append(
            _trade(
                decision_time_ms=index * 1_000,
                side="SELL",
                stress_net_bps=2.0,
                expected_net_bps=3.0,
            )
        )
    for index in range(6, 12):
        rows.append(
            _trade(
                decision_time_ms=index * 1_000,
                side="BUY",
                stress_net_bps=-5.0,
                expected_net_bps=-4.0,
            )
        )

    candidates = fast60_filter_candidates(rows, minimum_filter_trades=5)

    assert candidates[0]["name"] == "side=SELL"
    assert candidates[0]["diagnostic_positive"] is True
    assert candidates[0]["one_position_summary"]["stress_total_log_return_bps"] == 12.0


def test_historical_side_survivorship_aggregates_blocks(tmp_path) -> None:
    path_a = tmp_path / "fast43" / "frozen_holdout_trades.parquet"
    path_b = tmp_path / "fast44" / "frozen_holdout_trades.parquet"
    path_a.parent.mkdir()
    path_b.parent.mkdir()
    pd.DataFrame(
        [
            {"side": "BUY", "expected_net_bps": 1.0, "stress_net_bps": -1.0},
            {"side": "SELL", "expected_net_bps": 3.0, "stress_net_bps": 2.0},
        ]
    ).to_parquet(path_a)
    pd.DataFrame(
        [
            {"side": "BUY", "expected_net_bps": 2.0, "stress_net_bps": -2.0},
            {"side": "SELL", "expected_net_bps": -5.0, "stress_net_bps": -6.0},
        ]
    ).to_parquet(path_b)

    audit = historical_side_survivorship((path_a, path_b))

    assert audit["block_count"] == 2
    assert audit["aggregate_by_side"]["BUY"]["stress_total_log_return_bps"] == -3.0
    assert audit["aggregate_by_side"]["SELL"]["positive_block_count"] == 1


def test_discover_historical_holdout_trade_paths_filters_fast_range(tmp_path) -> None:
    reports = tmp_path / "reports"
    inside = (
        reports
        / "latency-adjusted-frozen-top-p-fast43-lowcost-micro-holdout-jul07-jul13-100ms"
        / "frozen_holdout_trades.parquet"
    )
    outside = (
        reports
        / "latency-adjusted-frozen-top-p-fast54-lowcost-micro-holdout-jul07-jul13-100ms"
        / "frozen_holdout_trades.parquet"
    )
    inside.parent.mkdir(parents=True)
    outside.parent.mkdir(parents=True)
    inside.write_text("", encoding="utf-8")
    outside.write_text("", encoding="utf-8")

    paths = discover_historical_holdout_trade_paths(tmp_path, fast_start=43, fast_end=53)

    assert paths == (inside,)


def test_fast62_status_blocks_positive_one_day_filter_when_historical_side_fails() -> None:
    status, reasons = fast62_status(
        filter_candidates=[
            {
                "name": "side=SELL",
                "diagnostic_positive": True,
                "one_position_summary": {
                    "trade_count": 5,
                    "stress_mean_bps": 1.0,
                    "stress_total_log_return_bps": 5.0,
                },
            }
        ],
        historical_side_audit={
            "aggregate_by_side": {
                "BUY": {"stress_total_log_return_bps": -3.0},
                "SELL": {"stress_total_log_return_bps": -7.0},
            }
        },
        complete_count=10,
        historical_path_count=2,
    )

    assert status == "FAST62_FAST60_FILTER_POSITIVE_BUT_HISTORICAL_SIDE_SURVIVORSHIP_FAILED"
    assert "historical_side_failures:BUY,SELL" in reasons


def _trade(
    *,
    decision_time_ms: int,
    side: str,
    stress_net_bps: float,
    expected_net_bps: float,
) -> dict[str, object]:
    return {
        "label": "test",
        "decision_time_ms": decision_time_ms,
        "entry_time_ms": decision_time_ms,
        "exit_time_ms": decision_time_ms + 100,
        "side": side,
        "horizon_seconds": 300,
        "probability": 0.95,
        "threshold": 0.9,
        "entry_price": 100.0,
        "exit_price": 100.0,
        "exit_reason": "TIMEOUT",
        "gross_bps": stress_net_bps + 2.0,
        "expected_net_bps": expected_net_bps,
        "stress_net_bps": stress_net_bps,
        "maximum_favorable_bps": max(stress_net_bps, 0.0),
        "maximum_adverse_bps": min(stress_net_bps, 0.0),
        "partial_take_fraction": 0.0,
        "recent_signal_count": 0,
    }
