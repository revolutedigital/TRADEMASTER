"""Pre-activation time-stop behavior for round 8."""

import pandas as pd
import pytest

from app.fx.instruments import Instrument
from app.fx.sim.costs import CostScenario
from scripts.research.fx_fast4_events import EventCosts
from scripts.research.fx_fast5_events import EXIT_INITIAL_STOP, EXIT_PRE_ACTIVATION_TIMEOUT
from scripts.research.fx_fast8_policy import _add_time_stop_outcome

ZERO = CostScenario(
    name="zero",
    spread_multiplier=1.0,
    slippage_pips=0.0,
    slippage_range_fraction=0.0,
    commission_base_per_lot_per_side=0.0,
)
COSTS = EventCosts(0.0, 0.0, base=ZERO, stress=ZERO)


def test_time_stop_exits_a_nonresponsive_trade_before_full_stop() -> None:
    index = pd.DatetimeIndex(
        [
            "2021-04-01T10:00:00Z",
            "2021-04-01T10:00:01Z",
            "2021-04-01T10:01:01Z",
            "2021-04-01T10:01:41Z",
        ]
    )
    ticks = pd.DataFrame(
        {
            "bid": [1.0, 1.0, 0.99995, 0.9999],
            "ask": [1.0, 1.0, 0.99995, 0.9999],
        },
        index=index,
    )
    candidates = pd.DataFrame(
        {
            "decision_index": [0],
            "side": [1],
            "risk_pips": [1.0],
            "mid_range_pips_256": [1.0],
        },
        index=pd.DatetimeIndex([index[0]]),
    )
    early = _add_time_stop_outcome(
        candidates, ticks, Instrument.from_symbol("EURUSD"), COSTS, 60
    )
    late = _add_time_stop_outcome(
        candidates, ticks, Instrument.from_symbol("EURUSD"), COSTS, 120
    )
    assert early.iloc[0]["exit_reason_base"] == EXIT_PRE_ACTIVATION_TIMEOUT
    assert early.iloc[0]["base_r"] == pytest.approx(-0.5)
    assert late.iloc[0]["exit_reason_base"] == EXIT_INITIAL_STOP
    assert late.iloc[0]["base_r"] == pytest.approx(-1.0)
