"""Explicit basis-point cost model shared by microstructure research stages."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RoundTripCostModel:
    """Complete expected friction for one entry and one exit.

    Per-side fields are doubled. ``spread_bps`` is the complete quoted spread
    paid by a marketable round trip, while funding is already a round-trip value
    attributable to the holding interval.
    """

    fee_bps_per_side: float
    slippage_bps_per_side: float
    spread_bps: float = 0.0
    funding_bps: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.fee_bps_per_side,
            self.slippage_bps_per_side,
            self.spread_bps,
            self.funding_bps,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Cost components must be finite")
        if any(value < 0 for value in values):
            raise ValueError("Cost components must be non-negative")

    @property
    def round_trip_bps(self) -> float:
        return (
            2 * (self.fee_bps_per_side + self.slippage_bps_per_side)
            + self.spread_bps
            + self.funding_bps
        )

    def net_bps(self, gross_bps: float) -> float:
        if not math.isfinite(gross_bps):
            raise ValueError("Gross result must be finite")
        return gross_bps - self.round_trip_bps


def stressed_cost_model(
    expected: RoundTripCostModel,
    *,
    minimum_round_trip_bps: float = 20.0,
    multiplier: float = 2.0,
) -> RoundTripCostModel:
    """Create a conservative all-slippage scenario with an exact total cost."""
    if not math.isfinite(minimum_round_trip_bps) or minimum_round_trip_bps < 0:
        raise ValueError("minimum_round_trip_bps must be finite and non-negative")
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError("multiplier must be finite and at least one")
    total = max(minimum_round_trip_bps, expected.round_trip_bps * multiplier)
    return RoundTripCostModel(
        fee_bps_per_side=0.0,
        slippage_bps_per_side=total / 2,
    )
