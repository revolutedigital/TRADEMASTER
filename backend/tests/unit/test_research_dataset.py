"""Research rows bind causal features to both directional path labels."""

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.research.research_dataset import (
    ResearchDatasetConfig,
    build_research_rows,
    load_book_interval,
    load_trade_interval,
)


def test_build_research_rows_produces_each_side_and_horizon() -> None:
    times = np.arange(0, 11_000, 100, dtype=np.int64)
    prices = 100 + np.sin(np.arange(len(times)) / 5)
    trades = pd.DataFrame(
        {
            "event_time_ms": times,
            "sequence_id": np.arange(len(times)),
            "price": prices,
            "quantity": np.ones(len(times)),
            "is_buyer_maker": np.arange(len(times)) % 2 == 0,
        }
    )
    config = ResearchDatasetConfig(
        decision_stride_seconds=1,
        horizons_seconds=(2, 5),
        expected_cost_bps=10,
        stress_cost_bps=20,
        initial_stop_bps=30,
        feature_windows_seconds=(1, 5),
    )

    rows = build_research_rows(trades, np.array([5000]), config)

    assert len(rows) == 4
    assert set(rows["side"]) == {"BUY", "SELL"}
    assert set(rows["horizon_seconds"]) == {2, 5}
    assert set(rows["target"]).issubset({0, 1})
    buy = rows[rows["side"] == "BUY"].iloc[0]
    assert buy["directed_flow_imbalance_1s"] == buy["flow_imbalance_1s"]
    sell = rows[rows["side"] == "SELL"].iloc[0]
    assert sell["directed_flow_imbalance_1s"] == -sell["flow_imbalance_1s"]


def test_build_research_rows_can_require_fresh_book_features() -> None:
    trades = _trade_frame()
    book_events = pd.DataFrame(
        {
            "event_time_ms": [4_900],
            "sequence_id": [1],
            "bid_price": [99.0],
            "bid_quantity": [3.0],
            "ask_price": [101.0],
            "ask_quantity": [1.0],
        }
    )
    config = ResearchDatasetConfig(
        horizons_seconds=(2,),
        feature_windows_seconds=(1,),
        require_book_features=True,
        max_book_staleness_ms=250,
    )

    rows = build_research_rows(trades, np.array([5_000]), config, book_events=book_events)

    buy = rows[rows["side"] == "BUY"].iloc[0]
    sell = rows[rows["side"] == "SELL"].iloc[0]
    assert buy["book_available"] == 1
    assert buy["book_update_age_ms"] == 100
    assert buy["directed_depth_imbalance"] == pytest.approx(buy["depth_imbalance"])
    assert sell["directed_depth_imbalance"] == pytest.approx(-sell["depth_imbalance"])
    assert buy["directed_book_pressure_imbalance_1s"] == pytest.approx(
        buy["book_pressure_imbalance_1s"]
    )
    assert sell["directed_book_pressure_imbalance_1s"] == pytest.approx(
        -sell["book_pressure_imbalance_1s"]
    )


def test_required_book_features_fail_closed_when_stale() -> None:
    config = ResearchDatasetConfig(
        horizons_seconds=(2,),
        feature_windows_seconds=(1,),
        require_book_features=True,
        max_book_staleness_ms=100,
    )
    with pytest.raises(ValueError, match="incomplete"):
        build_research_rows(
            _trade_frame(),
            np.array([5_000]),
            config,
            book_events=pd.DataFrame(
                {
                    "event_time_ms": [4_000],
                    "sequence_id": [1],
                    "bid_price": [99.0],
                    "bid_quantity": [3.0],
                    "ask_price": [101.0],
                    "ask_quantity": [1.0],
                }
            ),
        )


def test_load_book_interval_reads_prospective_wal_jsonl(tmp_path: Path) -> None:
    event_time = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)
    partition = tmp_path / "date=2026-01-01"
    partition.mkdir(parents=True)
    with gzip.open(partition / "events.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event_time": event_time.isoformat(),
                    "sequence_id": 42,
                    "bid_price": 99.0,
                    "bid_quantity": 2.0,
                    "ask_price": 101.0,
                    "ask_quantity": 3.0,
                }
            )
            + "\n"
        )

    frame = load_book_interval(
        tmp_path,
        datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert frame["event_time_ms"].tolist() == [int(event_time.timestamp() * 1000)]
    assert frame["bid_quantity"].tolist() == [2.0]


def test_load_trade_interval_converts_event_time_to_epoch_ms(tmp_path: Path) -> None:
    partition = tmp_path / "date=1970-01-01"
    partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_time": [
                pd.Timestamp("1970-01-01T00:00:01Z"),
                pd.Timestamp("1970-01-01T00:00:02Z"),
            ],
            "sequence_id": [1, 2],
            "price": [100.0, 101.0],
            "quantity": [1.0, 1.0],
            "is_buyer_maker": [False, True],
        }
    ).to_parquet(partition / "events.parquet", index=False)

    frame = load_trade_interval(
        tmp_path,
        datetime(1970, 1, 1, 0, 0, tzinfo=UTC),
        datetime(1970, 1, 1, 0, 0, 2, tzinfo=UTC),
    )

    assert frame["event_time_ms"].tolist() == [1_000, 2_000]


def _trade_frame() -> pd.DataFrame:
    times = np.arange(0, 11_000, 100, dtype=np.int64)
    prices = 100 + np.sin(np.arange(len(times)) / 5)
    return pd.DataFrame(
        {
            "event_time_ms": times,
            "sequence_id": np.arange(len(times)),
            "price": prices,
            "quantity": np.ones(len(times)),
            "is_buyer_maker": np.arange(len(times)) % 2 == 0,
        }
    )
