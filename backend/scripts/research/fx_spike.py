"""G0 spike: do simple technical strategies survive realistic FX costs?

Research-only. This decides whether rebuilding the platform for forex is worth it, so
it is built to avoid finding an edge that is not there:

* Strategies are pre-declared with the indicator library's default parameters. Nothing
  is fitted, so the whole sample is out of sample and there is no parameter overfitting.
* Every trade pays the real bid/ask crossing, slippage, commission, and swap, and a
  stop crossed by a gap fills at the gapped price (see fx_costs and fx_gap).
* The number of configurations tried is fixed in advance (strategies x timeframes) and
  the confidence interval is corrected for it.
* Trades are bootstrapped by calendar month, because trades on different pairs the
  same month are not independent (they all move with the dollar).
* Entry spreads are measured one hour after the 17:00 New York rollover, when the
  spread is not artificially wide, because a daily bar opens exactly at the rollover.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from app.schemas.trading import TechnicalStrategyConfig
from app.services.backtest.fx_costs import (
    NEW_YORK,
    STANDARD_LOT_UNITS,
    FxCostModel,
    NoCommission,
    NotionalCommission,
    PerLotCommission,
    Side,
    SwapSchedule,
    pip_size,
    pip_value_usd,
    swap_pnl_usd,
)
from app.services.backtest.fx_gap import resolve_protective_fill, stop_fill, take_profit_fill
from app.services.backtest.technical_strategy import build_technical_strategy_signals

ROLLOVER_HOUR = 17
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
RISK_REWARD_RATIO = 2.0
WARMUP_BARS = 120  # the slowest indicator (SMA 100, Donchian 55, MACD) needs ~110 bars
BOOTSTRAP_DRAWS = 20_000  # the corrected bound sits near the 0.2% quantile
BOOTSTRAP_SEED = 20260919
NOMINAL_ALPHA = 0.025  # one-sided, equivalent to a 95% two-sided interval
MIN_TRADES_FOR_TEST = 30  # fewer trades make the t statistic meaningless
G0_SIGNIFICANCE = 0.05  # adjusted p-value against the shuffled-data maximum
G0_MIN_PAIRS_POSITIVE = 4  # of 7: an edge must be broad, not one lucky pair


@dataclass(frozen=True)
class StrategySpec:
    """A pre-declared strategy. Parameters are the indicator library defaults."""

    name: str
    indicators: tuple[str, ...]
    indicator_params: dict[str, dict[str, float]] = field(default_factory=dict)
    min_confirmations: int = 1

    def to_config(self) -> TechnicalStrategyConfig:
        return TechnicalStrategyConfig(
            kind="technical_ensemble",
            indicators=list(self.indicators),
            indicator_params=self.indicator_params,
            min_confirmations=self.min_confirmations,
        )


STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec("ema_macd", ("ema", "macd")),
    StrategySpec("sma_rsi", ("sma", "rsi")),
    StrategySpec("sma_20_100", ("sma",), {"sma": {"sma_short": 20, "sma_long": 100}}),
    StrategySpec("donchian_20", ("breakout",), {"breakout": {"breakout_lookback": 20}}),
    StrategySpec("donchian_55", ("breakout",), {"breakout": {"breakout_lookback": 55}}),
    StrategySpec("bollinger_reversion", ("bollinger",)),
)
TIMEFRAMES: tuple[str, ...] = ("4h", "1D")
CONFIGURATION_COUNT = len(STRATEGIES) * len(TIMEFRAMES)


@dataclass(frozen=True)
class ExperimentDesign:
    """Which configurations are tested; the multiple-testing burden is their count."""

    name: str
    title: str
    strategies: tuple[StrategySpec, ...]
    timeframes: tuple[str, ...]

    @property
    def configuration_count(self) -> int:
        return len(self.strategies) * len(self.timeframes)


DISCOVERY = ExperimentDesign(
    "discovery",
    "G0: estratégias técnicas simples sobrevivem ao custo real de forex?",
    STRATEGIES,
    TIMEFRAMES,
)
_BY_NAME = {spec.name: spec for spec in STRATEGIES}
# Pre-registered before the confirmation data was downloaded or looked at
# (docs/forex/g0-confirmation-preregistration.md): the two daily configurations that led
# the discovery sample, tested unchanged on history the discovery never saw.
CONFIRMATION = ExperimentDesign(
    "confirmation",
    "G0, confirmação em amostra nunca vista: sma_rsi e bollinger_reversion no diário",
    (_BY_NAME["sma_rsi"], _BY_NAME["bollinger_reversion"]),
    ("1D",),
)
DESIGNS = {design.name: design for design in (DISCOVERY, CONFIRMATION)}


@dataclass(frozen=True)
class Scenario:
    """A cost assumption set. Spreads always come from the downloaded quotes."""

    name: str
    cost_model: FxCostModel
    swap: SwapSchedule | None = None
    description: str = ""
    spread_column: str = "open_spread_pips"


SCENARIOS: dict[str, Scenario] = {
    "base": Scenario(
        "base",
        FxCostModel(NotionalCommission(basis_points=0.2, minimum_usd=2.0), slippage_pips=0.2),
        description="interbank spread + IBKR-style commission (0.2 bp, $2 minimum) + 0.2 pip slippage",
    ),
    "ecn_raw": Scenario(
        "ecn_raw",
        FxCostModel(PerLotCommission(2.25), slippage_pips=0.1),
        description="interbank spread + $2.25 per lot per side + 0.1 pip slippage",
    ),
    "liquid_entry": Scenario(
        "liquid_entry",
        FxCostModel(NotionalCommission(basis_points=0.2, minimum_usd=2.0), slippage_pips=0.2),
        description="base costs, but the spread is the bar's median outside the rollover hour, "
        "as if orders were placed in liquid hours instead of the first hour of the session",
        spread_column="median_spread_pips",
    ),
    "stress": Scenario(
        "stress",
        FxCostModel(NoCommission(), slippage_pips=0.5, spread_multiplier=2.0),
        description="spread doubled + 0.5 pip slippage (a retail standard account)",
    ),
    "adverse_swap": Scenario(
        "adverse_swap",
        FxCostModel(NotionalCommission(basis_points=0.2, minimum_usd=2.0), slippage_pips=0.2),
        swap=SwapSchedule(long_pips_per_day=-0.3, short_pips_per_day=-0.3),
        description="base costs + a 0.3 pip/day financing debit on both sides",
    ),
}


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


def resample_session(h1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate hourly bid/ask candles into FX-session bars anchored at 17:00 New York.

    Returns one row per bar, indexed by the bar's start in UTC, with mid OHLC, bid and
    ask extremes, and `open_spread_pips` measured outside the rollover hour.
    """
    if rule not in {"4h", "1D"}:
        raise ValueError("rule must be '4h' or '1D'")
    if h1.empty:
        raise ValueError("no candles to resample")

    local = h1.tz_convert(NEW_YORK)
    wall_clock = local.index.tz_localize(None)
    shifted = wall_clock - pd.Timedelta(hours=ROLLOVER_HOUR)
    bar_key = shifted.floor("1D" if rule == "1D" else "4h")

    frame = h1.copy()
    frame["bar_key"] = bar_key
    at_rollover = wall_clock.hour == ROLLOVER_HOUR
    frame["spread_outside_rollover"] = frame["spread_open_pips"].where(~at_rollover)

    grouped = frame.groupby("bar_key", sort=True)
    bars = pd.DataFrame(
        {
            "bid_open": grouped["bid_open"].first(),
            "bid_high": grouped["bid_high"].max(),
            "bid_low": grouped["bid_low"].min(),
            "bid_close": grouped["bid_close"].last(),
            "ask_open": grouped["ask_open"].first(),
            "ask_high": grouped["ask_high"].max(),
            "ask_low": grouped["ask_low"].min(),
            "ask_close": grouped["ask_close"].last(),
            "spread_outside_rollover": grouped["spread_outside_rollover"].first(),
            "median_spread_pips": grouped["spread_outside_rollover"].median(),
            "spread_open_pips": grouped["spread_open_pips"].first(),
            "candles": grouped["bid_open"].size(),
        }
    )
    bars["open_spread_pips"] = bars["spread_outside_rollover"].fillna(bars["spread_open_pips"])
    bars["median_spread_pips"] = bars["median_spread_pips"].fillna(bars["open_spread_pips"])
    bars = bars.drop(columns=["spread_outside_rollover", "spread_open_pips"])

    starts = pd.DatetimeIndex(bars.index) + pd.Timedelta(hours=ROLLOVER_HOUR)
    bars.index = starts.tz_localize(
        NEW_YORK, ambiguous=True, nonexistent="shift_forward"
    ).tz_convert(UTC)
    bars.index.name = "bar_start"
    return bars


def mid_ohlc(bars: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": (bars["bid_open"] + bars["ask_open"]) / 2,
            "high": (bars["bid_high"] + bars["ask_high"]) / 2,
            "low": (bars["bid_low"] + bars["ask_low"]) / 2,
            "close": (bars["bid_close"] + bars["ask_close"]) / 2,
            "volume": 0.0,
        },
        index=bars.index,
    )


def wilder_atr(mid: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    previous_close = mid["close"].shift(1)
    true_range = pd.concat(
        [
            mid["high"] - mid["low"],
            (mid["high"] - previous_close).abs(),
            (mid["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def build_signals(mid: pd.DataFrame, spec: StrategySpec) -> pd.Series:
    signals, _definition = build_technical_strategy_signals(mid, spec.to_config())
    return signals


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: Side
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    stop_distance_pips: float
    gross_pips: float
    execution_cost_pips: float
    commission_pips: float
    swap_pips: float
    net_pips: float
    r_multiple: float
    exit_reason: str
    bars_held: int


@dataclass
class _OpenPosition:
    side: Side
    entry_index: int
    entry_time: datetime
    entry_price: float
    entry_cost_pips: float
    stop_price: float
    take_profit_price: float
    stop_distance_price: float


def simulate(
    bars: pd.DataFrame,
    signals: pd.Series,
    atr: pd.Series,
    *,
    symbol: str,
    scenario: Scenario,
    units: float = STANDARD_LOT_UNITS,
    warmup: int = WARMUP_BARS,
) -> list[Trade]:
    """Stop-and-reverse long/short simulation with a fixed ATR stop and 2R target.

    A signal is read at a bar's close and acted on at the next bar's open, so there is
    no look-ahead. A position closes on its stop, its target, or an opposite signal.
    """
    if not (len(bars) == len(signals) == len(atr)):
        raise ValueError("bars, signals and atr must be aligned")
    if scenario.spread_column not in bars:
        raise ValueError(f"bars have no {scenario.spread_column!r} column")

    model = scenario.cost_model
    one_pip = pip_size(symbol)
    times = list(bars.index)
    trades: list[Trade] = []
    position: _OpenPosition | None = None
    pending = 0.0

    def quoted_half_spread_pips(index: int) -> float:
        """Half the spread the market shows; the cost model applies any stress multiplier."""
        return float(bars[scenario.spread_column].iloc[index]) / 2

    def open_quote(index: int) -> tuple[float, float]:
        mid = (float(bars["bid_open"].iloc[index]) + float(bars["ask_open"].iloc[index])) / 2
        half = quoted_half_spread_pips(index) * one_pip
        return mid - half, mid + half

    def crossing_cost_pips(index: int, *, with_slippage: bool) -> float:
        """Pips lost to the spread and slippage on one fill, versus trading at the mid."""
        return quoted_half_spread_pips(index) * model.spread_multiplier + (
            model.slippage_pips if with_slippage else 0.0
        )

    def close_position(
        index: int, price: float, when: datetime, reason: str, exit_cost_pips: float
    ) -> None:
        nonlocal position
        assert position is not None
        direction = 1.0 if position.side == "LONG" else -1.0
        gross_pips = direction * (price - position.entry_price) / one_pip
        pip_usd = pip_value_usd(symbol, price, units)
        commission_usd = model.commission_usd(
            symbol=symbol, price=position.entry_price, units=units
        ) + model.commission_usd(symbol=symbol, price=price, units=units)
        swap_usd = (
            swap_pnl_usd(
                scenario.swap,
                symbol=symbol,
                side=position.side,
                units=units,
                price=price,
                entry_time=position.entry_time,
                exit_time=when,
            )
            if scenario.swap
            else 0.0
        )
        commission_pips = commission_usd / pip_usd
        swap_pips = swap_usd / pip_usd
        net_pips = gross_pips - commission_pips + swap_pips
        stop_pips = position.stop_distance_price / one_pip
        trades.append(
            Trade(
                symbol=symbol,
                side=position.side,
                entry_time=position.entry_time,
                exit_time=when,
                entry_price=position.entry_price,
                exit_price=price,
                stop_distance_pips=stop_pips,
                gross_pips=gross_pips,
                execution_cost_pips=position.entry_cost_pips + exit_cost_pips,
                commission_pips=commission_pips,
                swap_pips=swap_pips,
                net_pips=net_pips,
                r_multiple=net_pips / stop_pips,
                exit_reason=reason,
                bars_held=index - position.entry_index,
            )
        )
        position = None

    for index in range(len(bars)):
        bar_start = times[index].to_pydatetime()

        # 1. Act on the signal read at the previous close, at this bar's open.
        if pending != 0.0 and index > 0:
            wanted: Side = "LONG" if pending > 0 else "SHORT"
            bid, ask = open_quote(index)
            if position is not None and position.side != wanted:
                exit_price = model.fill_price(
                    symbol=symbol, side=position.side, action="EXIT", bid=bid, ask=ask
                )
                close_position(
                    index,
                    exit_price,
                    bar_start,
                    "signal",
                    crossing_cost_pips(index, with_slippage=True),
                )
            if position is None:
                prior_atr = float(atr.iloc[index - 1])
                if np.isfinite(prior_atr) and prior_atr > 0:
                    entry = model.fill_price(
                        symbol=symbol, side=wanted, action="ENTRY", bid=bid, ask=ask
                    )
                    distance = ATR_STOP_MULTIPLIER * prior_atr
                    sign = 1.0 if wanted == "LONG" else -1.0
                    position = _OpenPosition(
                        side=wanted,
                        entry_index=index,
                        entry_time=bar_start,
                        entry_price=entry,
                        entry_cost_pips=crossing_cost_pips(index, with_slippage=True),
                        stop_price=entry - sign * distance,
                        take_profit_price=entry + sign * distance * RISK_REWARD_RATIO,
                        stop_distance_price=distance,
                    )
        pending = 0.0

        # 2. Test the protective orders inside this bar.
        if position is not None:
            bid_open, ask_open = open_quote(index)
            stop = stop_fill(
                side=position.side,
                stop_price=position.stop_price,
                bar_open_bid=bid_open,
                bar_open_ask=ask_open,
                bar_low_bid=float(bars["bid_low"].iloc[index]),
                bar_high_ask=float(bars["ask_high"].iloc[index]),
            )
            target = take_profit_fill(
                side=position.side,
                limit_price=position.take_profit_price,
                bar_high_bid=float(bars["bid_high"].iloc[index]),
                bar_low_ask=float(bars["ask_low"].iloc[index]),
            )
            outcome = resolve_protective_fill(stop, target)
            if outcome.triggered and outcome.price is not None:
                stopped = outcome is stop
                # A stress multiplier widens the spread the trader crosses, and a stop
                # also slips; a resting limit target does not slip.
                widening = quoted_half_spread_pips(index) * (model.spread_multiplier - 1) * one_pip
                slippage = model.slippage_pips * one_pip if stopped else 0.0
                adverse = widening + slippage
                filled = outcome.price - adverse if position.side == "LONG" else (
                    outcome.price + adverse
                )
                reason = "stop_gap" if stop.triggered and stop.gapped else (
                    "stop" if stopped else "take_profit"
                )
                close_position(
                    index,
                    filled,
                    bar_start + timedelta(seconds=1),
                    reason,
                    crossing_cost_pips(index, with_slippage=stopped),
                )

        # 3. Read the signal at this bar's close for the next open.
        if index >= warmup:
            pending = float(signals.iloc[index])

    if position is not None:
        last = len(bars) - 1
        bid = float(bars["bid_close"].iloc[last])
        ask = float(bars["ask_close"].iloc[last])
        exit_price = model.fill_price(
            symbol=symbol, side=position.side, action="EXIT", bid=bid, ask=ask
        )
        close_position(
            last,
            exit_price,
            times[last].to_pydatetime() + timedelta(hours=1),
            "end",
            crossing_cost_pips(last, with_slippage=True),
        )
    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    mean: float
    lower_nominal: float
    lower_adjusted: float
    upper_nominal: float


@dataclass(frozen=True)
class GroupStats:
    trades: int
    mean_r: float
    mean_net_pips: float
    profit_factor: float
    win_rate: float
    mean_cost_pips: float
    mean_bars_held: float
    interval: Interval | None
    minimum_detectable_mean_r: float | None
    t_stat: float | None = None


def trades_frame(trades: Sequence[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol", "side", "entry_time", "exit_time", "net_pips", "r_multiple",
                "gross_pips", "execution_cost_pips", "commission_pips", "swap_pips",
                "bars_held", "month",
            ]
        )
    frame = pd.DataFrame([trade.__dict__ for trade in trades])
    frame["month"] = pd.to_datetime(frame["exit_time"], utc=True).dt.strftime("%Y-%m")
    return frame


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    adjusted_alpha: float = NOMINAL_ALPHA,
) -> tuple[float, float, float, float]:
    """Mean with a cluster bootstrap: (mean, lower_nominal, lower_adjusted, upper_nominal)."""
    if values.size == 0:
        raise ValueError("no values to bootstrap")
    labels, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=labels.size)
    counts = np.bincount(inverse, minlength=labels.size).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, labels.size, size=(draws, labels.size))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, NOMINAL_ALPHA)),
        float(np.quantile(means, adjusted_alpha)),
        float(np.quantile(means, 1 - NOMINAL_ALPHA)),
    )


def trades_needed_to_confirm(
    stats: GroupStats, configuration_count: int = CONFIGURATION_COUNT
) -> float | None:
    """Sample size at which the observed mean would clear the multiple-testing bar with 80% power."""
    if stats.t_stat is None or stats.t_stat <= 0 or stats.mean_r <= 0 or stats.trades < 2:
        return None
    sd = stats.mean_r * np.sqrt(stats.trades) / stats.t_stat
    z = NormalDist().inv_cdf(1 - NOMINAL_ALPHA / configuration_count) + NormalDist().inv_cdf(0.8)
    return float((z * sd / stats.mean_r) ** 2)


def profit_factor(values: np.ndarray) -> float:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def group_stats(
    frame: pd.DataFrame, *, adjusted_alpha: float, with_interval: bool = True
) -> GroupStats:
    if frame.empty:
        return GroupStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None)
    r = frame["r_multiple"].to_numpy(dtype=float)
    interval = None
    detectable = None
    if with_interval and frame["month"].nunique() >= 3:
        mean, low_n, low_a, high_n = cluster_bootstrap_mean(
            r, frame["month"].to_numpy(), adjusted_alpha=adjusted_alpha
        )
        interval = Interval(mean, low_n, low_a, high_n)
        z = NormalDist().inv_cdf(1 - adjusted_alpha) + NormalDist().inv_cdf(0.8)
        detectable = float(z * r.std(ddof=1) / np.sqrt(r.size)) if r.size > 1 else None
    t_stat = None
    if r.size > 1 and r.std(ddof=1) > 0:
        t_stat = float(r.mean() / (r.std(ddof=1) / np.sqrt(r.size)))
    return GroupStats(
        t_stat=t_stat,
        trades=int(r.size),
        mean_r=float(r.mean()),
        mean_net_pips=float(frame["net_pips"].mean()),
        profit_factor=profit_factor(r),
        win_rate=float((r > 0).mean()),
        mean_cost_pips=float(
            (frame["execution_cost_pips"] + frame["commission_pips"]).mean()
        ),
        mean_bars_held=float(frame["bars_held"].mean()),
        interval=interval,
        minimum_detectable_mean_r=detectable,
    )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigResult:
    strategy: str
    timeframe: str
    scenario: str
    pooled: GroupStats
    per_pair: dict[str, GroupStats]
    by_year: dict[str, GroupStats]
    pairs_positive: int
    pairs_significant: int

    @property
    def passes_written_g0(self) -> bool:
        """The plan's rule: positive after costs, CI excluding zero, in at least 2 pairs."""
        return self.pairs_significant >= 2

    @property
    def passes_multiple_testing(self) -> bool:
        interval = self.pooled.interval
        return interval is not None and interval.lower_adjusted > 0

    @property
    def passes_g0(self) -> bool:
        return self.passes_written_g0 and self.passes_multiple_testing

    @property
    def testable_t(self) -> float | None:
        """The pooled t statistic, only when there are enough trades to trust it."""
        if self.pooled.trades < MIN_TRADES_FOR_TEST:
            return None
        return self.pooled.t_stat


def adjusted_p_value(real_t: float, null_max_t: Sequence[float]) -> float:
    """Probability that the best of all configurations on shuffled data reaches `real_t`.

    Comparing against the maximum over configurations, per shuffle, accounts for trying
    many strategies at once and for their dependence, without distributional assumptions.
    """
    if not null_max_t:
        raise ValueError("null distribution is empty")
    exceed = sum(1 for value in null_max_t if value >= real_t)
    return (1 + exceed) / (1 + len(null_max_t))


@dataclass(frozen=True)
class G0Row:
    result: ConfigResult
    adjusted_p: float | None
    passes: bool


def evaluate_g0(
    results: Sequence[ConfigResult], null_max_t: Sequence[float]
) -> list[G0Row]:
    """Apply the calibrated G0 rule to the base-scenario results."""
    rows = []
    for result in results:
        t_value = result.testable_t
        p_value = adjusted_p_value(t_value, null_max_t) if t_value is not None else None
        passes = (
            p_value is not None
            and p_value <= G0_SIGNIFICANCE
            and result.pooled.mean_r > 0
            and result.pairs_positive >= G0_MIN_PAIRS_POSITIVE
        )
        rows.append(G0Row(result, p_value, passes))
    return rows


def load_pair_bars(data_dir: Path, symbol: str, rule: str) -> pd.DataFrame:
    h1 = pd.read_parquet(data_dir / f"{symbol}_H1.parquet")
    return resample_session(h1, rule)


def run_config(
    bars_by_pair: dict[str, pd.DataFrame],
    spec: StrategySpec,
    timeframe: str,
    scenario: Scenario,
    *,
    adjusted_alpha: float,
) -> ConfigResult:
    all_trades: list[Trade] = []
    for symbol, bars in bars_by_pair.items():
        mid = mid_ohlc(bars)
        signals = build_signals(mid, spec)
        all_trades += simulate(
            bars, signals, wilder_atr(mid), symbol=symbol, scenario=scenario
        )
    frame = trades_frame(all_trades)

    per_pair = {
        symbol: group_stats(frame[frame["symbol"] == symbol], adjusted_alpha=adjusted_alpha)
        for symbol in bars_by_pair
    }
    frame["year"] = pd.to_datetime(frame["exit_time"], utc=True).dt.year.astype(str)
    by_year = {
        year: group_stats(group, adjusted_alpha=adjusted_alpha, with_interval=False)
        for year, group in frame.groupby("year")
    }
    return ConfigResult(
        strategy=spec.name,
        timeframe=timeframe,
        scenario=scenario.name,
        pooled=group_stats(frame, adjusted_alpha=adjusted_alpha),
        per_pair=per_pair,
        by_year=by_year,
        pairs_positive=sum(1 for s in per_pair.values() if s.trades and s.mean_r > 0),
        pairs_significant=sum(
            1 for s in per_pair.values() if s.interval and s.interval.lower_nominal > 0
        ),
    )


def run_experiment(
    data_dir: Path,
    symbols: Sequence[str],
    *,
    scenario_names: Sequence[str],
    design: ExperimentDesign = DISCOVERY,
) -> list[ConfigResult]:
    adjusted_alpha = NOMINAL_ALPHA / design.configuration_count
    results: list[ConfigResult] = []
    for timeframe in design.timeframes:
        bars_by_pair = {symbol: load_pair_bars(data_dir, symbol, timeframe) for symbol in symbols}
        for spec in design.strategies:
            for name in scenario_names:
                results.append(
                    run_config(
                        bars_by_pair,
                        spec,
                        timeframe,
                        SCENARIOS[name],
                        adjusted_alpha=adjusted_alpha,
                    )
                )
    return results


# ---------------------------------------------------------------------------
# Placebo: does the whole procedure approve data that has no edge?
# ---------------------------------------------------------------------------

BID_COLUMNS = ("bid_open", "bid_high", "bid_low", "bid_close")
ASK_COLUMNS = ("ask_open", "ask_high", "ask_low", "ask_close")


def shuffle_bars(
    bars_by_pair: dict[str, pd.DataFrame], rng: np.random.Generator
) -> dict[str, pd.DataFrame]:
    """Shuffle whole bars, with the gap that precedes each, using one permutation for all pairs.

    Every bar keeps its own shape (range, direction, spread) and its own opening gap, so
    volatility, spreads and gap sizes are unchanged. What is destroyed is the order, so
    trend and mean reversion between bars disappear. Using the same permutation for every
    pair keeps the cross-pair correlation of the moves, which matters for the bootstrap.
    """
    common = sorted(set.intersection(*(set(bars.index) for bars in bars_by_pair.values())))
    if len(common) < 3:
        raise ValueError("pairs share too few bars to shuffle")
    index = pd.DatetimeIndex(common)
    permutation = rng.permutation(len(index))

    shuffled: dict[str, pd.DataFrame] = {}
    for symbol, bars in bars_by_pair.items():
        aligned = bars.loc[index]
        bid_open = aligned["bid_open"].to_numpy(dtype=float)
        ask_open = aligned["ask_open"].to_numpy(dtype=float)
        relative_close = aligned["bid_close"].to_numpy(dtype=float) / bid_open
        previous_close = np.concatenate(([bid_open[0]], aligned["bid_close"].to_numpy()[:-1]))
        gap = bid_open / previous_close

        chosen_gap = gap[permutation].copy()
        chosen_close = relative_close[permutation]
        factors = np.ones(len(index))
        factors[1:] = chosen_close[:-1] * chosen_gap[1:]
        new_bid_open = bid_open[0] * np.cumprod(factors)
        new_ask_open = new_bid_open * (ask_open[permutation] / bid_open[permutation])

        frame = aligned.iloc[permutation].copy()
        for column in BID_COLUMNS:
            frame[column] = aligned[column].to_numpy()[permutation] / bid_open[permutation] * new_bid_open
        for column in ASK_COLUMNS:
            frame[column] = aligned[column].to_numpy()[permutation] / ask_open[permutation] * new_ask_open
        frame.index = index
        shuffled[symbol] = frame
    return shuffled


@dataclass(frozen=True)
class PlaceboSummary:
    replications: int
    any_written: int
    any_multiple_testing: int
    any_g0: int
    per_config_g0: dict[str, int]
    null_max_t: tuple[float, ...] = ()

    @property
    def false_pass_rate(self) -> float:
        return self.any_g0 / self.replications if self.replications else float("nan")


def run_placebo(
    data_dir: Path,
    symbols: Sequence[str],
    *,
    replications: int,
    seed: int = BOOTSTRAP_SEED,
    scenario_name: str = "base",
    first_replication: int = 0,
    design: ExperimentDesign = DISCOVERY,
) -> PlaceboSummary:
    """Run the full experiment on shuffled data and count how often it would approve."""
    adjusted_alpha = NOMINAL_ALPHA / design.configuration_count
    real = {
        timeframe: {symbol: load_pair_bars(data_dir, symbol, timeframe) for symbol in symbols}
        for timeframe in design.timeframes
    }
    any_written = any_multiple = any_g0 = 0
    per_config: dict[str, int] = {}
    null_max_t: list[float] = []
    for replication in range(first_replication, first_replication + replications):
        rng = np.random.default_rng(seed + replication)
        written = multiple = full = False
        best_t = float("-inf")
        for timeframe in design.timeframes:
            shuffled = shuffle_bars(real[timeframe], rng)
            for spec in design.strategies:
                result = run_config(
                    shuffled, spec, timeframe, SCENARIOS[scenario_name], adjusted_alpha=adjusted_alpha
                )
                if result.testable_t is not None:
                    best_t = max(best_t, result.testable_t)
                written = written or result.passes_written_g0
                multiple = multiple or result.passes_multiple_testing
                if result.passes_g0:
                    full = True
                    key = f"{spec.name}/{timeframe}"
                    per_config[key] = per_config.get(key, 0) + 1
        any_written += written
        any_multiple += multiple
        any_g0 += full
        null_max_t.append(best_t)
    return PlaceboSummary(
        replications, any_written, any_multiple, any_g0, per_config, tuple(null_max_t)
    )


def save_null(summary: PlaceboSummary, path: Path, *, seed: int) -> None:
    payload = {
        "scenario": "base",
        "replications": summary.replications,
        "seed": seed,
        "old_rule_false_pass": {
            "plan_rule_2_pairs": summary.any_written,
            "multiple_testing_bound": summary.any_multiple_testing,
            "both": summary.any_g0,
        },
        "null_max_t": list(summary.null_max_t),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_nulls(paths: Sequence[Path]) -> dict[str, object]:
    """Combine null distributions computed in separate processes (disjoint seeds)."""
    if not paths:
        raise ValueError("nothing to merge")
    merged: list[float] = []
    old_rule = {"plan_rule_2_pairs": 0, "multiple_testing_bound": 0, "both": 0}
    replications = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        merged += [float(v) for v in payload["null_max_t"]]
        replications += int(payload["replications"])
        for key in old_rule:
            old_rule[key] += int(payload["old_rule_false_pass"][key])
    return {
        "scenario": "base",
        "replications": replications,
        "seed": BOOTSTRAP_SEED,
        "old_rule_false_pass": old_rule,
        "null_max_t": merged,
    }


def load_null_details(path: Path) -> dict[str, object]:
    """The raw null file, for reports that quote how the older rules behaved on shuffles."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_null(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = [float(v) for v in payload["null_max_t"] if v is not None and np.isfinite(v)]
    if len(values) < 50:
        raise ValueError("the null distribution needs at least 50 shuffles to be usable")
    return values


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(
    results: Sequence[ConfigResult],
    *,
    symbols: Sequence[str],
    data_range: str,
    generated: str,
    null_max_t: Sequence[float] | None = None,
    null_details: dict[str, object] | None = None,
    design: ExperimentDesign = DISCOVERY,
) -> str:
    base = [r for r in results if r.scenario == "base"]
    calibrated = null_max_t is not None
    rows = evaluate_g0(base, null_max_t) if calibrated else []
    winners = [row for row in rows if row.passes]
    threshold = float(np.quantile(null_max_t, 1 - G0_SIGNIFICANCE)) if calibrated else None
    lines = [
        f"# {design.title}",
        "",
        f"Gerado em {generated}. Pares: {', '.join(symbols)}. Dados: {data_range}.",
        f"Configurações testadas: {design.configuration_count} (estratégias x timeframes), parâmetros de fábrica, "
        "sem otimização: toda a amostra é fora da amostra.",
        "Custo base = " + SCENARIOS["base"].description + ". Carry/swap real não modelado no caso base.",
        "",
        "## Critério de aprovação (calibrado por placebo)",
        "",
        "O critério usado compara a MELHOR configuração real com a melhor configuração em dados "
        "embaralhados (sem vantagem por construção): só passa quem tiver "
        f"valor-p ajustado <= {G0_SIGNIFICANCE:.2f}, média R positiva, ao menos {G0_MIN_PAIRS_POSITIVE} "
        f"de {len(symbols)} pares positivos e {MIN_TRADES_FOR_TEST}+ trades.",
        "",
        "## Veredito",
        "",
    ]
    if not calibrated:
        lines.append("**SEM CALIBRAÇÃO: sem a distribuição do placebo não há veredito.** "
                     "Rode com --placebo N --null-out arquivo e depois com --null-file arquivo.")
    elif winners:
        lines.append(f"**G0 PASSOU** em {len(winners)} configuração(ões): "
                     + ", ".join(f"{w.result.strategy}/{w.result.timeframe} (p={w.adjusted_p:.3f})"
                                 for w in winners) + ".")
    else:
        lines.append("**G0 NÃO PASSOU.** Nenhuma configuração se separa do que a melhor de "
                     f"{design.configuration_count} configurações faria em dados sem vantagem "
                     f"(limite t de 95% no placebo: {threshold:.2f}, {len(null_max_t)} embaralhamentos).")
        best = max((row for row in rows if row.result.testable_t is not None),
                   key=lambda row: row.result.testable_t, default=None)
        if best is not None:
            lines.append("")
            lines.append(f"A mais próxima foi {best.result.strategy}/{best.result.timeframe}: "
                         f"t = {best.result.testable_t:.2f}, valor-p ajustado = {best.adjusted_p:.3f}, "
                         f"média R = {best.result.pooled.mean_r:.3f}, "
                         f"{best.result.pairs_positive}/{len(symbols)} pares positivos.")

    if calibrated and null_details:
        rule = null_details["old_rule_false_pass"]
        count = int(null_details["replications"])
        lines += [
            "",
            f"Calibração ({count} embaralhamentos): a regra escrita no plano (IC 95% nominal excluindo zero "
            f"em 2+ pares) aprovou dado sem vantagem em {rule['plan_rule_2_pairs']} ({rule['plan_rule_2_pairs'] / count:.0%}) "
            f"das rodadas; o limite por bootstrap com Bonferroni aprovou "
            f"{rule['multiple_testing_bound']} ({rule['multiple_testing_bound'] / count:.1%}), conservador demais. "
            f"Por isso o critério é a distribuição da melhor configuração: mediana t = {np.median(null_max_t):.2f}, "
            f"p90 = {np.quantile(null_max_t, 0.90):.2f}, p95 = {threshold:.2f}, p99 = {np.quantile(null_max_t, 0.99):.2f}.",
        ]

    lines += ["", "## Configurações, caso base", "",
              "| Estratégia | TF | Trades | Média R | t | Valor-p ajustado | PF | Custo por trade (pips: spread + slippage + comissão) | Pares > 0 | Efeito mínimo detectável (R) | Passa |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    by_key = {(row.result.strategy, row.result.timeframe): row for row in rows}
    for r in sorted(base, key=lambda item: item.pooled.mean_r, reverse=True):
        pooled = r.pooled
        row = by_key.get((r.strategy, r.timeframe))
        p_text = _fmt(row.adjusted_p) if row else "n/a"
        verdict = ("sim" if row.passes else "não") if row else "n/a"
        lines.append(
            f"| {r.strategy} | {r.timeframe} | {pooled.trades} | {_fmt(pooled.mean_r)} | {_fmt(pooled.t_stat, 2)} | "
            f"{p_text} | {_fmt(pooled.profit_factor, 2)} | {_fmt(pooled.mean_cost_pips, 2)} | "
            f"{r.pairs_positive}/{len(r.per_pair)} | {_fmt(pooled.minimum_detectable_mean_r)} | {verdict} |"
        )

    top = max(base, key=lambda item: item.pooled.mean_r)
    lines += ["", f"## Maior média R: {top.strategy} / {top.timeframe}", "",
              "| Par | Trades | Média R | Média pips líquidos | PF | Win rate |", "|---|---|---|---|---|---|"]
    for symbol, stats in top.per_pair.items():
        lines.append(f"| {symbol} | {stats.trades} | {_fmt(stats.mean_r)} | {_fmt(stats.mean_net_pips, 1)} | "
                     f"{_fmt(stats.profit_factor, 2)} | {_fmt(stats.win_rate, 2)} |")
    lines += ["", "Por ano civil (estabilidade):", "", "| Ano | Trades | Média R |", "|---|---|---|"]
    for year, stats in sorted(top.by_year.items()):
        lines.append(f"| {year} | {stats.trades} | {_fmt(stats.mean_r)} |")
    needed = trades_needed_to_confirm(top.pooled, design.configuration_count)
    if needed is not None and top.pooled.trades:
        years_of_data = max(len(top.by_year), 1)
        per_year = top.pooled.trades / years_of_data
        lines += ["", f"Para confirmar um efeito de {_fmt(top.pooled.mean_r)} R com 80% de poder e "
                  f"significância ajustada seriam necessários cerca de {needed:,.0f} trades, "
                  f"contra {top.pooled.trades} nesta amostra (≈ {needed / per_year:.0f} anos nesta cadência "
                  "com estes 7 pares)."]

    lines += ["", "## Sensibilidade a custo, horário de entrada e carry (média R agregada)", "",
              "| Estratégia | TF | " + " | ".join(SCENARIOS) + " |",
              "|---|---|" + "|".join("---" for _ in SCENARIOS) + "|"]
    keyed = {(r.strategy, r.timeframe, r.scenario): r for r in results}
    for spec in design.strategies:
        for timeframe in design.timeframes:
            cells = []
            for name in SCENARIOS:
                item = keyed.get((spec.name, timeframe, name))
                cells.append(_fmt(item.pooled.mean_r) if item else "n/a")
            lines.append(f"| {spec.name} | {timeframe} | " + " | ".join(cells) + " |")

    lines += ["", "## Como ler", "",
              "- Média R é o resultado médio por trade em múltiplos do risco (distância do stop, 2 ATR).",
              "- t é a média dividida pelo erro padrão. O valor-p ajustado é a chance de a MELHOR das "
              f"{design.configuration_count} configurações, em dados embaralhados, atingir esse t.",
              "- Efeito mínimo detectável é o menor R médio que esta amostra separaria de zero (80% de poder). "
              "Se for maior que qualquer efeito plausível, o resultado é INCONCLUSIVO por falta de amostra, "
              "não prova de ausência de vantagem.",
              "- O caso base não inclui carry: em pares como USDJPY o diferencial de juros pode somar ou "
              "subtrair pips por dia. Ver adverse_swap. Custo pesa pouco em estratégias lentas (poucos pips "
              "contra uma distância de stop de 50 a 150 pips): o que decide é a vantagem, não o custo.",
              "- Spreads vêm do feed interbancário da Dukascopy. liquid_entry mede o efeito de executar em "
              "horário líquido; stress dobra o spread (conta de varejo padrão)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fx"))
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--design",
        choices=sorted(DESIGNS),
        default="discovery",
        help="discovery tests every configuration; confirmation tests the two pre-registered ones",
    )
    parser.add_argument(
        "--placebo",
        type=int,
        default=0,
        help="instead of the real run, shuffle the bars N times and save the null distribution",
    )
    parser.add_argument("--placebo-start", type=int, default=0, help="first shuffle index (for parallel chunks)")
    parser.add_argument("--merge-null", type=Path, nargs="+", default=None, help="merge null files and exit")
    parser.add_argument("--null-out", type=Path, default=None, help="where --placebo saves the null")
    parser.add_argument("--null-file", type=Path, default=None, help="null distribution for the real run")
    args = parser.parse_args(argv)

    symbols = args.symbols or sorted(
        path.name.removesuffix("_H1.parquet") for path in args.data_dir.glob("*_H1.parquet")
    )
    if not symbols:
        sys.stderr.write("no parquet datasets found; run fx_dataset first\n")
        return 2

    if args.merge_null:
        merged = merge_nulls(args.merge_null)
        target = args.null_out or Path("null.json")
        target.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        sys.stdout.write(f"merged {merged['replications']} shuffles into {target}\n")
        return 0

    design = DESIGNS[args.design]
    if args.placebo:
        summary = run_placebo(
            args.data_dir,
            symbols,
            replications=args.placebo,
            first_replication=args.placebo_start,
            design=design,
        )
        if args.null_out:
            save_null(summary, args.null_out, seed=BOOTSTRAP_SEED)
        sys.stdout.write(
            f"placebo: {summary.replications} replications on shuffled bars\n"
            f"  any configuration passes the plan's rule (2+ pairs): {summary.any_written}\n"
            f"  any passes the multiple-testing bound: {summary.any_multiple_testing}\n"
            f"  any passes full G0: {summary.any_g0} "
            f"({summary.false_pass_rate:.1%}; a sound gate should stay near or below 5%)\n"
            f"  by configuration: {summary.per_config_g0}\n"
            f"  null distribution of the best t: median {np.median(summary.null_max_t):.2f}, "
            f"95th percentile {np.quantile(summary.null_max_t, 0.95):.2f}\n"
        )
        return 0

    results = run_experiment(args.data_dir, symbols, scenario_names=list(SCENARIOS), design=design)
    first = pd.read_parquet(args.data_dir / f"{symbols[0]}_H1.parquet")
    null_max_t = load_null(args.null_file) if args.null_file else None
    null_details = load_null_details(args.null_file) if args.null_file else None
    report = render_report(
        results,
        null_max_t=null_max_t,
        null_details=null_details,
        design=design,
        symbols=symbols,
        data_range=f"{first.index[0]:%Y-%m-%d} a {first.index[-1]:%Y-%m-%d}",
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
