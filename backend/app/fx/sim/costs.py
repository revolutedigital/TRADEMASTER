"""Costs that surround a simulated trade: stress spreads, slippage, commission, and swap.

The simulator already pays the real bid/ask of every bar, which carries the time-of-day and
rollover widening in the data itself. What it does not know is the rest of what a broker
charges, and this module supplies it in the units a trader reasons about (pips):

* a stress multiplier on the spread, for the conditions the sample under-represents;
* slippage, fixed or scaled by recent volatility using only bars that had already closed;
* commission, converted from money per lot into pips for the pair and its price;
* swap, charged at the 17:00 New York rollover (three times on Wednesday, never on the
  weekend), computed for millions of trades at once and aware of daylight saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.fx.instruments import STANDARD_LOT_UNITS, ConversionRates, Instrument, pip_value
from app.fx.strategy import (
    ASK_CLOSE,
    ASK_HIGH,
    ASK_LOW,
    ASK_OPEN,
    BAR_TIME,
    BID_CLOSE,
    BID_HIGH,
    BID_LOW,
    BID_OPEN,
)

NEW_YORK = "America/New_York"
ROLLOVER_HOUR = 17
WEDNESDAY = 2


def widen_spread(matrix: np.ndarray, multiplier: float) -> np.ndarray:
    """Return a copy of the bars with the bid/ask spread scaled symmetrically around the mid."""
    if multiplier < 1.0:
        raise ValueError("a multiplier below 1 would model a tighter spread than the market quoted")
    widened = matrix.copy()
    for bid_column, ask_column in (
        (BID_OPEN, ASK_OPEN), (BID_HIGH, ASK_HIGH), (BID_LOW, ASK_LOW), (BID_CLOSE, ASK_CLOSE),
    ):
        mid = 0.5 * (matrix[:, bid_column] + matrix[:, ask_column])
        half = 0.5 * (matrix[:, ask_column] - matrix[:, bid_column]) * multiplier
        widened[:, bid_column] = mid - half
        widened[:, ask_column] = mid + half
    return widened


def fixed_slippage(count: int, pips: float, instrument: Instrument) -> np.ndarray:
    if pips < 0:
        raise ValueError("slippage cannot be negative")
    return np.full(count, pips * instrument.pip_size, dtype=np.float64)


def volatility_slippage(
    matrix: np.ndarray,
    instrument: Instrument,
    *,
    base_pips: float,
    range_fraction: float,
    window: int = 20,
) -> np.ndarray:
    """Slippage that grows with the average bar range of the previous `window` bars.

    The value for bar t uses bars up to t-1 only, so it can be known before the fill it prices.
    """
    if base_pips < 0 or range_fraction < 0 or window < 1:
        raise ValueError("base_pips and range_fraction cannot be negative and window must be >= 1")
    mid_high = 0.5 * (matrix[:, BID_HIGH] + matrix[:, ASK_HIGH])
    mid_low = 0.5 * (matrix[:, BID_LOW] + matrix[:, ASK_LOW])
    range_pips = pd.Series((mid_high - mid_low) / instrument.pip_size)
    recent = range_pips.rolling(window, min_periods=1).mean().shift(1).fillna(0.0).to_numpy()
    return (base_pips + range_fraction * recent) * instrument.pip_size


def commission_round_trip_pips(
    instrument: Instrument,
    price: float,
    rates: ConversionRates,
    *,
    usd_per_lot_per_side: float = 0.0,
    base_currency_per_lot_per_side: float = 0.0,
    notional_basis_points: float = 0.0,
    minimum_usd_per_order: float = 0.0,
    units: float = STANDARD_LOT_UNITS,
    account_currency: str = "USD",
) -> float:
    """Round-trip commission, expressed in pips of this pair at the given size and price."""
    lots = units / STANDARD_LOT_UNITS
    # Some brokers (Fusion on cTrader) charge per 100,000 of notional in the pair's BASE currency.
    per_lot = lots * (usd_per_lot_per_side + base_currency_per_lot_per_side * rates.usd_per(instrument.base))
    notional_usd = units * price * rates.usd_per(instrument.quote) / rates.usd_per("USD")
    percentage = notional_usd * notional_basis_points / 10_000
    per_side_usd = max(per_lot, percentage, minimum_usd_per_order)
    per_side_account = per_side_usd / rates.usd_per(account_currency)
    return 2 * per_side_account / pip_value(instrument, units, rates, account_currency)


class RolloverCalendar:
    """Counts the daily financing charges a position pays between two instants.

    A charge happens when a 17:00 New York rollover falls after the entry and at or before the
    exit. Weekday rollovers count once, the triple weekday (Wednesday) counts three times, and
    Saturday and Sunday count zero because the Wednesday charge already settled them.
    """

    def __init__(self, first_year: int = 2015, last_year: int = 2032, triple_weekday: int = WEDNESDAY) -> None:
        if not 0 <= triple_weekday <= 4:
            raise ValueError("triple_weekday must be a weekday between 0 and 4")
        days = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31", freq="D")
        local = (days + pd.Timedelta(hours=ROLLOVER_HOUR)).tz_localize(
            NEW_YORK, ambiguous="NaT", nonexistent="shift_forward"
        )
        valid = ~pd.isna(local)
        local = local[valid]
        weekdays = days[valid].dayofweek.to_numpy()
        weights = np.where(weekdays < 5, 1, 0)
        weights[weekdays == triple_weekday] = 3
        self._instants = (
            (local.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
        ).to_numpy(dtype=np.float64)
        self._cumulative = np.concatenate(([0], np.cumsum(weights))).astype(np.int64)

    def charged_days(self, entry_seconds: np.ndarray, exit_seconds: np.ndarray) -> np.ndarray:
        """Financing days paid by each trade; times are seconds since the UTC epoch."""
        entry = np.asarray(entry_seconds, dtype=np.float64)
        exit_ = np.asarray(exit_seconds, dtype=np.float64)
        if np.any(exit_ < entry):
            raise ValueError("a trade cannot exit before it enters")
        if entry.size and (entry.min() < self._instants[0] or exit_.max() > self._instants[-1]):
            raise ValueError("trade times fall outside the calendar; widen first_year/last_year")
        after_entry = np.searchsorted(self._instants, entry, side="right")
        up_to_exit = np.searchsorted(self._instants, exit_, side="right")
        return self._cumulative[up_to_exit] - self._cumulative[after_entry]


@dataclass(frozen=True)
class CostScenario:
    """A complete set of cost assumptions for one simulation run."""

    name: str
    spread_multiplier: float = 1.0
    slippage_pips: float = 0.1
    slippage_range_fraction: float = 0.0
    commission_usd_per_lot_per_side: float = 0.0
    commission_base_per_lot_per_side: float = 0.0
    swap_long_pips_per_day: float = 0.0
    swap_short_pips_per_day: float = 0.0
    description: str = field(default="", compare=False)


FUSION_ZERO = CostScenario(
    "fusion_zero",
    slippage_pips=0.1,
    slippage_range_fraction=0.05,
    commission_base_per_lot_per_side=2.25,
    description="raw spread from the data + $2.25 per lot per side + slippage that grows with volatility",
)
STRESS = CostScenario(
    "stress",
    spread_multiplier=2.0,
    slippage_pips=0.3,
    slippage_range_fraction=0.10,
    commission_base_per_lot_per_side=2.25,
    description="spread doubled and slippage tripled, for conditions the sample under-represents",
)
ADVERSE_SWAP = CostScenario(
    "adverse_swap",
    slippage_pips=0.1,
    slippage_range_fraction=0.05,
    commission_base_per_lot_per_side=2.25,
    swap_long_pips_per_day=-0.3,
    swap_short_pips_per_day=-0.3,
    description="base costs plus a 0.3 pip per day financing debit on both sides",
)
SCENARIOS = {scenario.name: scenario for scenario in (FUSION_ZERO, STRESS, ADVERSE_SWAP)}


def prepare_run(
    matrix: np.ndarray, instrument: Instrument, scenario: CostScenario
) -> tuple[np.ndarray, np.ndarray]:
    """Bars and per-bar slippage to hand to the simulator for a scenario."""
    bars = widen_spread(matrix, scenario.spread_multiplier)
    slippage = volatility_slippage(
        bars,
        instrument,
        base_pips=scenario.slippage_pips,
        range_fraction=scenario.slippage_range_fraction,
    )
    return bars, slippage


INTRABAR_EXIT_SECONDS = 1.0  # after the bar's opening instant, so a rollover at the open counts


def finalize_trades(
    result: tuple[np.ndarray, ...],
    bars: np.ndarray,
    instrument: Instrument,
    scenario: CostScenario,
    *,
    commission_pips: float,
    calendar: RolloverCalendar,
    signal_exit_reason: int = 1,
) -> pd.DataFrame:
    """Turn simulator output into trades measured in pips and in multiples of the stop distance."""
    entry_index, exit_index, side, entry_price, exit_price, stop_distance, reason = result
    pip = instrument.pip_size
    entry_seconds = bars[entry_index, BAR_TIME] if len(entry_index) else np.empty(0)
    exit_open = bars[exit_index, BAR_TIME] if len(exit_index) else np.empty(0)
    exit_seconds = exit_open + np.where(reason == signal_exit_reason, 0.0, INTRABAR_EXIT_SECONDS)

    gross = (exit_price - entry_price) * side / pip
    days = calendar.charged_days(entry_seconds, exit_seconds)
    swap = days * np.where(
        side > 0, scenario.swap_long_pips_per_day, scenario.swap_short_pips_per_day
    )
    net = gross - commission_pips + swap
    stop_pips = stop_distance / pip
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(entry_seconds, unit="s", utc=True),
            "exit_time": pd.to_datetime(exit_seconds, unit="s", utc=True),
            "side": side.astype(int),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_pips": stop_pips,
            "gross_pips": gross,
            "commission_pips": commission_pips,
            "swap_pips": swap,
            "net_pips": net,
            "r_multiple": net / stop_pips,
            "reason": reason.astype(int),
            "minutes_held": (exit_index - entry_index).astype(int),
        }
    )
