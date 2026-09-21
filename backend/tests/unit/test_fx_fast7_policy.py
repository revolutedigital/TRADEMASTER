"""Early-protection manager behavior for round 7."""

import pandas as pd

from app.fx.instruments import Instrument
from app.fx.sim.costs import CostScenario
from scripts.research.fx_fast4_events import EventCosts
from scripts.research.fx_fast7_policy import _add_management_outcome

ZERO = CostScenario(
    name="zero",
    spread_multiplier=1.0,
    slippage_pips=0.0,
    slippage_range_fraction=0.0,
    commission_base_per_lot_per_side=0.0,
)
COSTS = EventCosts(0.0, 0.0, base=ZERO, stress=ZERO)


def test_earlier_activation_protects_a_small_favorable_move() -> None:
    index = pd.date_range("2021-04-01T10:00:00Z", periods=6, freq="s")
    ticks = pd.DataFrame(
        {
            "bid": [1.0, 1.0, 1.00002, 1.00001, 1.0, 0.9999],
            "ask": [1.0, 1.0, 1.00002, 1.00001, 1.0, 0.9999],
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
    early = _add_management_outcome(candidates, ticks, Instrument.from_symbol("EURUSD"), COSTS, 0.1)
    late = _add_management_outcome(candidates, ticks, Instrument.from_symbol("EURUSD"), COSTS, 0.3)
    assert early.iloc[0]["activated_base"] == 1.0
    assert early.iloc[0]["base_r"] >= 0.0
    assert late.iloc[0]["activated_base"] == 0.0
    assert late.iloc[0]["base_r"] < -0.999
