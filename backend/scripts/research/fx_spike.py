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
    return GroupStats(
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
    data_dir: Path, symbols: Sequence[str], *, scenario_names: Sequence[str]
) -> list[ConfigResult]:
    adjusted_alpha = NOMINAL_ALPHA / CONFIGURATION_COUNT
    results: list[ConfigResult] = []
    for timeframe in TIMEFRAMES:
        bars_by_pair = {symbol: load_pair_bars(data_dir, symbol, timeframe) for symbol in symbols}
        for spec in STRATEGIES:
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
# Report
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(
    results: Sequence[ConfigResult], *, symbols: Sequence[str], data_range: str, generated: str
) -> str:
    base = [r for r in results if r.scenario == "base"]
    winners = [r for r in base if r.passes_g0]
    written_only = [r for r in base if r.passes_written_g0 and not r.passes_multiple_testing]
    lines = [
        "# G0: estratégias técnicas simples sobrevivem ao custo real de forex?",
        "",
        f"Gerado em {generated}. Pares: {', '.join(symbols)}. Dados: {data_range}.",
        f"Configurações testadas: {CONFIGURATION_COUNT} (estratégias x timeframes). "
        f"Correção de múltiplos testes: limite inferior no quantil {NOMINAL_ALPHA / CONFIGURATION_COUNT:.4f}.",
        "Parâmetros de fábrica, sem otimização: toda a amostra é fora da amostra. "
        "Custo base = " + SCENARIOS["base"].description + ". Carry/swap real não modelado no caso base.",
        "",
        "## Veredito",
        "",
    ]
    if winners:
        lines.append(f"**G0 PASSOU** em {len(winners)} configuração(ões): "
                     + ", ".join(f"{w.strategy}/{w.timeframe}" for w in winners) + ".")
    else:
        lines.append("**G0 NÃO PASSOU.** Nenhuma configuração tem expectativa positiva depois do custo "
                     "com IC excluindo zero em pelo menos 2 pares E limite inferior corrigido acima de zero.")
        if written_only:
            lines.append("")
            lines.append("Passaram só na regra do plano (2+ pares) e não na correção de múltiplos testes: "
                         + ", ".join(f"{w.strategy}/{w.timeframe}" for w in written_only) + ".")
    lines += ["", "## Configurações, caso base", "",
              "| Estratégia | TF | Trades | Média R | PF | Custo por trade (pips: spread + slippage + comissão) | IC95 nominal | Limite inf. corrigido | Pares > 0 | Pares signif. | Efeito mínimo detectável (R) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(base, key=lambda item: item.pooled.mean_r, reverse=True):
        p = r.pooled
        interval = p.interval
        nominal = f"[{_fmt(interval.lower_nominal)}, {_fmt(interval.upper_nominal)}]" if interval else "n/a"
        lines.append(
            f"| {r.strategy} | {r.timeframe} | {p.trades} | {_fmt(p.mean_r)} | {_fmt(p.profit_factor, 2)} | "
            f"{_fmt(p.mean_cost_pips, 2)} | {nominal} | {_fmt(interval.lower_adjusted if interval else None)} | "
            f"{r.pairs_positive}/{len(r.per_pair)} | {r.pairs_significant} | {_fmt(p.minimum_detectable_mean_r)} |"
        )

    best = max(base, key=lambda item: item.pooled.mean_r)
    lines += ["", f"## Melhor configuração por média R: {best.strategy} / {best.timeframe}", "",
              "| Par | Trades | Média R | Média pips líquidos | PF | Win rate |", "|---|---|---|---|---|---|"]
    for symbol, s in best.per_pair.items():
        lines.append(f"| {symbol} | {s.trades} | {_fmt(s.mean_r)} | {_fmt(s.mean_net_pips, 1)} | "
                     f"{_fmt(s.profit_factor, 2)} | {_fmt(s.win_rate, 2)} |")
    lines += ["", "Por ano civil (estabilidade):", "", "| Ano | Trades | Média R |", "|---|---|---|"]
    for year, s in sorted(best.by_year.items()):
        lines.append(f"| {year} | {s.trades} | {_fmt(s.mean_r)} |")

    lines += ["", "## Sensibilidade a custo e a carry (média R agregada)", "",
              "| Estratégia | TF | " + " | ".join(SCENARIOS) + " |",
              "|---|---|" + "|".join("---" for _ in SCENARIOS) + "|"]
    keyed = {(r.strategy, r.timeframe, r.scenario): r for r in results}
    for spec in STRATEGIES:
        for timeframe in TIMEFRAMES:
            cells = []
            for name in SCENARIOS:
                item = keyed.get((spec.name, timeframe, name))
                cells.append(_fmt(item.pooled.mean_r) if item else "n/a")
            lines.append(f"| {spec.name} | {timeframe} | " + " | ".join(cells) + " |")

    lines += ["", "## Como ler", "",
              "- Média R é o resultado médio por trade em múltiplos do risco (distância do stop). "
              "Positivo depois do custo é o mínimo; o IC diz se dá para separar de zero.",
              "- Efeito mínimo detectável é o menor R médio que esta amostra conseguiria separar de zero "
              "(80% de poder). Se for maior que qualquer efeito plausível, o resultado é INCONCLUSIVO por "
              "falta de amostra, não prova de ausência de vantagem.",
              "- O caso base não inclui carry: em pares como USDJPY o diferencial de juros pode somar ou "
              "subtrair pips por dia e pode mudar o sinal de uma estratégia de tendência. Ver a coluna adverse_swap.",
              "- Spreads vêm do feed interbancário da Dukascopy; uma conta de varejo padrão paga mais (coluna stress)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fx"))
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    symbols = args.symbols or sorted(
        path.name.removesuffix("_H1.parquet") for path in args.data_dir.glob("*_H1.parquet")
    )
    if not symbols:
        sys.stderr.write("no parquet datasets found; run fx_dataset first\n")
        return 2

    results = run_experiment(args.data_dir, symbols, scenario_names=list(SCENARIOS))
    first = pd.read_parquet(args.data_dir / f"{symbols[0]}_H1.parquet")
    report = render_report(
        results,
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
