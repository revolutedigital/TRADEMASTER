"""Research rows bind causal features to both directional path labels."""

import numpy as np
import pandas as pd

from app.services.research.research_dataset import (
    ResearchDatasetConfig,
    build_research_rows,
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
