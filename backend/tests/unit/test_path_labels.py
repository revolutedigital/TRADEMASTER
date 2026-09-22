"""Path labels use true event order, including equal timestamps."""

import numpy as np
import pytest

from app.services.backtest.event_replay import OrderSide
from app.services.research.path_labels import (
    EventPathLabeler,
    FirstTouch,
    PathLabelConfig,
    PriceRangeIndex,
)


def test_range_index_returns_extrema_and_first_crossing() -> None:
    index = PriceRangeIndex(np.array([100, 101, 99, 103, 98], dtype=float))
    assert index.extrema(1, 4) == (99, 103)
    assert index.first_at_least(1, 5, 102) == 3
    assert index.first_at_most(1, 5, 98.5) == 4
    assert index.first_at_least(0, 3, 200) is None


def test_long_label_records_stop_before_recovery_at_same_timestamp() -> None:
    labeler = EventPathLabeler(
        event_times_ms=np.array([0, 1000, 1000, 2000, 5000]),
        sequence_ids=np.array([10, 11, 12, 13, 14]),
        prices=np.array([100, 99, 101, 102, 100]),
        config=PathLabelConfig(
            horizons_seconds=(5,),
            expected_cost_bps=50,
            stress_cost_bps=100,
            initial_stop_bps=50,
        ),
    )

    label = labeler.label(0, OrderSide.BUY, 5)

    assert label.first_touch == FirstTouch.STOP
    assert label.first_touch_sequence_id == 11
    assert label.expected_touch_sequence_id == 12
    assert label.stress_touch_sequence_id == 13
    assert label.paid_expected_before_stop is False
    assert label.event_count == 4


def test_short_label_finds_favorable_path_before_stop() -> None:
    labeler = EventPathLabeler(
        event_times_ms=np.array([0, 1000, 2000, 3000]),
        sequence_ids=np.array([1, 2, 3, 4]),
        prices=np.array([100, 99, 98, 101]),
        config=PathLabelConfig(
            horizons_seconds=(3,),
            expected_cost_bps=50,
            stress_cost_bps=150,
            initial_stop_bps=50,
        ),
    )

    label = labeler.label(0, OrderSide.SELL, 3)

    assert label.first_touch == FirstTouch.BREAKEVEN
    assert label.expected_touch_sequence_id == 2
    assert label.stress_touch_sequence_id == 3
    assert label.stop_touch_sequence_id == 4
    assert label.paid_stress_before_stop is True
    assert label.mfe_bps > 0
    assert label.mae_bps < 0


def test_incomplete_horizon_is_rejected() -> None:
    labeler = EventPathLabeler(
        np.array([0, 1000]),
        np.array([1, 2]),
        np.array([100, 101]),
        PathLabelConfig(horizons_seconds=(5,)),
    )
    with pytest.raises(ValueError, match="extends"):
        labeler.label(0, OrderSide.BUY, 5)
