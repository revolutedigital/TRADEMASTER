"""Causal feature and outcome factory for round-3 short-trade research.

This module only turns bid/ask bars into research examples. It does not fit a model, choose a
threshold, touch the database, or send an order. A decision belongs to the close of an M5 bar and
its hypothetical fill belongs to the following M5 open.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from app.fx import strategy as fx
from app.fx.bars import aggregate_minutes
from app.fx.instruments import Instrument
from app.fx.sim.costs import CostScenario, FUSION_ZERO, STRESS, prepare_run

M5_SECONDS = 300
M15_SECONDS = 900
H1_SECONDS = 3_600
LOOKBACK_BARS = 72
ATR_BARS = 24
HORIZONS_MINUTES = (15, 60, 180, 360)
BARRIER_TARGETS = (0.5, 1.0, 2.0)
SIDES = (fx.LONG, fx.SHORT)
EPSILON = 1e-12

DIRECTIONAL_FEATURES = frozenset(
    {
        *(f"return_{window}" for window in (1, 3, 6, 12, 24, 72)),
        *(f"close_minus_ema_{window}_atr" for window in (6, 12, 24, 72)),
        "candle_body_atr",
        "close_location",
        "m1_return_atr",
        "m1_body_imbalance",
        "m15_trend_atr",
        "h1_trend_atr",
        "usd_factor_return_3",
        "common_factor_return_3",
    }
)


@dataclass(frozen=True)
class EventCosts:
    """Scenario inputs whose monetary commission has already been converted to pips."""

    base_commission_pips: float
    stress_commission_pips: float
    base: CostScenario = FUSION_ZERO
    stress: CostScenario = STRESS

    def __post_init__(self) -> None:
        if self.base_commission_pips < 0 or self.stress_commission_pips < 0:
            raise ValueError("commission cannot be negative")


def _validate_bars(matrix: np.ndarray) -> np.ndarray:
    bars = np.ascontiguousarray(matrix, dtype=np.float64)
    if bars.ndim != 2 or bars.shape[1] != fx.BAR_WIDTH:
        raise ValueError(f"bars must have shape (n, {fx.BAR_WIDTH})")
    if len(bars) and np.any(np.diff(bars[:, fx.BAR_TIME]) <= 0):
        raise ValueError("bar times must be strictly increasing")
    for bid, ask in (
        (fx.BID_OPEN, fx.ASK_OPEN),
        (fx.BID_HIGH, fx.ASK_HIGH),
        (fx.BID_LOW, fx.ASK_LOW),
        (fx.BID_CLOSE, fx.ASK_CLOSE),
    ):
        if np.any(bars[:, ask] < bars[:, bid]):
            raise ValueError("ask cannot be below bid")
    return bars


def _mid_ohlc(bars: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(  # type: ignore[return-value]
        0.5 * (bars[:, bid] + bars[:, ask])
        for bid, ask in (
            (fx.BID_OPEN, fx.ASK_OPEN),
            (fx.BID_HIGH, fx.ASK_HIGH),
            (fx.BID_LOW, fx.ASK_LOW),
            (fx.BID_CLOSE, fx.ASK_CLOSE),
        )
    )


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> pd.Series:
    previous = np.concatenate(([np.nan], close[:-1]))
    return pd.Series(np.maximum.reduce((high - low, np.abs(high - previous), np.abs(low - previous))))


def _completed_context(
    decision_close: np.ndarray,
    context_bars: np.ndarray,
    seconds: int,
    *,
    trend_bars: int,
    volatility_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the latest fully closed coarse-bar trend and volatility onto M5 decisions."""
    _, high, low, close = _mid_ohlc(context_bars)
    true_range = _true_range(high, low, close)
    atr = true_range.rolling(volatility_bars, min_periods=volatility_bars).mean().to_numpy()
    trend = (pd.Series(close) - pd.Series(close).shift(trend_bars)).to_numpy() / np.maximum(atr, EPSILON)
    log_return = pd.Series(np.log(close)).diff()
    volatility = log_return.rolling(volatility_bars, min_periods=volatility_bars).std(ddof=0).to_numpy()
    available_at = context_bars[:, fx.BAR_TIME] + seconds
    indices = np.searchsorted(available_at, decision_close, side="right") - 1
    valid = indices >= 0
    mapped_trend = np.full(len(decision_close), np.nan)
    mapped_volatility = np.full(len(decision_close), np.nan)
    mapped_trend[valid] = trend[indices[valid]]
    mapped_volatility[valid] = volatility[indices[valid]]
    return mapped_trend, mapped_volatility


def _m1_pulse(minutes: np.ndarray, m5: np.ndarray, atr: np.ndarray, pip_size: float) -> dict[str, np.ndarray]:
    bucket = np.floor_divide(minutes[:, fx.BAR_TIME], M5_SECONDS)
    m5_bucket = np.floor_divide(m5[:, fx.BAR_TIME], M5_SECONDS)
    first = np.searchsorted(bucket, m5_bucket, side="left")
    last = np.searchsorted(bucket, m5_bucket, side="right")
    result = {name: np.full(len(m5), np.nan) for name in (
        "m1_return_atr", "m1_range_atr", "m1_body_imbalance", "m1_spread_pips"
    )}
    mid_open = 0.5 * (minutes[:, fx.BID_OPEN] + minutes[:, fx.ASK_OPEN])
    mid_high = 0.5 * (minutes[:, fx.BID_HIGH] + minutes[:, fx.ASK_HIGH])
    mid_low = 0.5 * (minutes[:, fx.BID_LOW] + minutes[:, fx.ASK_LOW])
    mid_close = 0.5 * (minutes[:, fx.BID_CLOSE] + minutes[:, fx.ASK_CLOSE])
    for index, (start, stop) in enumerate(zip(first, last, strict=True)):
        if stop <= start or minutes[stop - 1, fx.BAR_TIME] - minutes[start, fx.BAR_TIME] > 4 * 60:
            continue
        scale = max(atr[index], EPSILON)
        bodies = mid_close[start:stop] - mid_open[start:stop]
        result["m1_return_atr"][index] = (mid_close[stop - 1] - mid_open[start]) / scale
        result["m1_range_atr"][index] = (mid_high[start:stop].max() - mid_low[start:stop].min()) / scale
        result["m1_body_imbalance"][index] = bodies.sum() / max(np.abs(bodies).sum(), EPSILON)
        result["m1_spread_pips"][index] = (
            minutes[stop - 1, fx.ASK_CLOSE] - minutes[stop - 1, fx.BID_CLOSE]
        ) / pip_size
    return result


def build_feature_frame(minutes: np.ndarray, instrument: Instrument) -> tuple[np.ndarray, pd.DataFrame]:
    """Return M5 bars and causal features indexed by their decision-close time."""
    source = _validate_bars(minutes)
    if instrument.pip_size <= 0:
        raise ValueError("pip size must be positive")
    m5 = aggregate_minutes(source, M5_SECONDS)
    m15 = aggregate_minutes(source, M15_SECONDS)
    h1 = aggregate_minutes(source, H1_SECONDS)
    open_, high, low, close = _mid_ohlc(m5)
    close_series = pd.Series(close)
    log_close = np.log(close_series)
    true_range = _true_range(high, low, close)
    atr = true_range.rolling(ATR_BARS, min_periods=ATR_BARS).mean()
    safe_atr = atr.clip(lower=EPSILON)
    feature: dict[str, np.ndarray | pd.Series] = {}
    for window in (1, 3, 6, 12, 24, 72):
        feature[f"return_{window}"] = log_close.diff(window)
    for window in (6, 12, 24, 72):
        ema = close_series.ewm(span=window, adjust=False, min_periods=window).mean()
        feature[f"close_minus_ema_{window}_atr"] = (close_series - ema) / safe_atr
    feature["atr_price"] = atr
    feature["true_range_atr"] = true_range / safe_atr
    feature["candle_body_atr"] = (close - open_) / safe_atr
    feature["upper_wick_atr"] = (high - np.maximum(open_, close)) / safe_atr
    feature["lower_wick_atr"] = (np.minimum(open_, close) - low) / safe_atr
    feature["close_location"] = (close - low) / np.maximum(high - low, EPSILON) - 0.5
    one_bar_return = log_close.diff()
    for window in (6, 24, 72):
        volatility = one_bar_return.rolling(window, min_periods=window).std(ddof=0)
        feature[f"realized_volatility_{window}"] = volatility
        feature[f"compression_{window}"] = (
            pd.Series(high).rolling(window, min_periods=window).max()
            - pd.Series(low).rolling(window, min_periods=window).min()
        ) / safe_atr
        rolling_high = pd.Series(high).rolling(window, min_periods=window).max()
        rolling_low = pd.Series(low).rolling(window, min_periods=window).min()
        feature[f"distance_high_{window}_atr"] = (rolling_high - close_series) / safe_atr
        feature[f"distance_low_{window}_atr"] = (close_series - rolling_low) / safe_atr
    feature["volatility_ratio_6_24"] = (
        pd.Series(feature["realized_volatility_6"]) / pd.Series(feature["realized_volatility_24"])
    )
    feature["volatility_ratio_24_72"] = (
        pd.Series(feature["realized_volatility_24"]) / pd.Series(feature["realized_volatility_72"])
    )
    spread = pd.Series((m5[:, fx.ASK_CLOSE] - m5[:, fx.BID_CLOSE]) / instrument.pip_size)
    feature["spread_pips"] = spread
    feature["spread_change_24"] = spread - spread.shift(24)
    feature["spread_percentile_72"] = spread.rolling(72, min_periods=72).rank(pct=True)
    feature.update(_m1_pulse(source, m5, atr.to_numpy(), instrument.pip_size))

    decision_close = m5[:, fx.BAR_TIME] + M5_SECONDS
    feature["m15_trend_atr"], feature["m15_volatility"] = _completed_context(
        decision_close, m15, M15_SECONDS, trend_bars=4, volatility_bars=8
    )
    feature["h1_trend_atr"], feature["h1_volatility"] = _completed_context(
        decision_close, h1, H1_SECONDS, trend_bars=6, volatility_bars=12
    )
    timestamps = pd.to_datetime(decision_close, unit="s", utc=True)
    minute_of_week = timestamps.dayofweek.to_numpy() * 1_440 + timestamps.hour.to_numpy() * 60 + timestamps.minute.to_numpy()
    angle = 2 * np.pi * minute_of_week / (7 * 1_440)
    feature["week_sin"] = np.sin(angle)
    feature["week_cos"] = np.cos(angle)
    london = timestamps.tz_convert("Europe/London")
    new_york = timestamps.tz_convert("America/New_York")
    feature["london_session"] = ((london.hour >= 8) & (london.hour < 17)).astype(np.int8)
    feature["new_york_session"] = ((new_york.hour >= 8) & (new_york.hour < 17)).astype(np.int8)

    # Series carry a RangeIndex. Convert positionally before assigning the DatetimeIndex, otherwise
    # pandas aligns unlike labels and silently turns every Series-backed feature into NaN.
    positional_feature = {name: np.asarray(values) for name, values in feature.items()}
    frame = pd.DataFrame(
        positional_feature, index=pd.DatetimeIndex(timestamps, name="decision_time")
    )
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    contiguous = pd.Series(m5[:, fx.BAR_TIME]).diff().eq(M5_SECONDS)
    frame["history_contiguous"] = (
        contiguous.rolling(LOOKBACK_BARS, min_periods=LOOKBACK_BARS).sum().to_numpy() == LOOKBACK_BARS
    )
    return m5, frame


def directional_features(features: pd.DataFrame, side: int) -> pd.DataFrame:
    """Orient signed features so positive always means movement in the proposed side."""
    if side not in SIDES:
        raise ValueError("side must be LONG (1) or SHORT (-1)")
    oriented = features.copy()
    for name in DIRECTIONAL_FEATURES & set(oriented.columns):
        oriented[name] = oriented[name] * side
    oriented["side"] = side
    return oriented


def _forward_is_contiguous(times: np.ndarray, start: int, stop: int) -> bool:
    return stop < len(times) and bool(np.all(np.diff(times[start:stop + 1]) == M5_SECONDS))


def _barrier_result(
    bars: np.ndarray,
    slippage: np.ndarray,
    entry_index: int,
    exit_index: int,
    side: int,
    entry_price: float,
    risk_price: float,
    target_r: float,
) -> int:
    stop_price = entry_price - side * risk_price
    target_price = entry_price + side * target_r * risk_price
    for index in range(entry_index, exit_index + 1):
        if side == fx.LONG:
            stop_hit = bars[index, fx.BID_LOW] - slippage[index] <= stop_price
            target_hit = bars[index, fx.BID_HIGH] - slippage[index] >= target_price
        else:
            stop_hit = bars[index, fx.ASK_HIGH] + slippage[index] >= stop_price
            target_hit = bars[index, fx.ASK_LOW] + slippage[index] <= target_price
        if stop_hit:
            return 0
        if target_hit:
            return 1
    return 0


def build_outcome_frame(
    m5: np.ndarray,
    features: pd.DataFrame,
    instrument: Instrument,
    costs: EventCosts,
    *,
    horizons_minutes: Iterable[int] = HORIZONS_MINUTES,
) -> pd.DataFrame:
    """Build executable next-open outcomes for both sides and every declared horizon."""
    bars = _validate_bars(m5)
    if len(features) != len(bars):
        raise ValueError("features and M5 bars must have equal length")
    horizons = tuple(int(value) for value in horizons_minutes)
    if not horizons or any(value <= 0 or value % 5 for value in horizons):
        raise ValueError("horizons must be positive whole multiples of five minutes")
    base_bars, base_slippage = prepare_run(bars, instrument, costs.base)
    stress_bars, stress_slippage = prepare_run(bars, instrument, costs.stress)
    atr = features["atr_price"].to_numpy(dtype=np.float64)
    history_contiguous = features["history_contiguous"].fillna(False).to_numpy(dtype=bool)
    rows: list[dict[str, object]] = []
    for decision_index in range(len(bars)):
        if not history_contiguous[decision_index] or not np.isfinite(atr[decision_index]):
            continue
        entry_index = decision_index + 1
        if entry_index >= len(bars) or bars[entry_index, fx.BAR_TIME] - bars[decision_index, fx.BAR_TIME] != M5_SECONDS:
            continue
        for horizon_minutes in horizons:
            horizon_bars = horizon_minutes // 5
            exit_index = decision_index + horizon_bars
            if not _forward_is_contiguous(bars[:, fx.BAR_TIME], decision_index, exit_index):
                continue
            for side in SIDES:
                row = _outcome_row(
                    base_bars,
                    base_slippage,
                    stress_bars,
                    stress_slippage,
                    decision_index,
                    entry_index,
                    exit_index,
                    side,
                    atr[decision_index],
                    instrument,
                    costs,
                    horizon_minutes,
                    features.index[decision_index],
                )
                rows.append(row)
    return pd.DataFrame(rows)


@njit(cache=True)
def _compiled_outcome(  # noqa: PLR0913
    base_bars,
    base_slippage,
    stress_bars,
    stress_slippage,
    atr,
    history_contiguous,
    horizon_bars,
    side,
    pip,
    base_commission_pips,
    stress_commission_pips,
):
    """Vector-width outcome kernel; rows stay aligned to the feature matrix."""
    count = base_bars.shape[0]
    output = np.full((count, 14), np.nan)
    for decision_index in range(count - horizon_bars):
        if not history_contiguous[decision_index] or not np.isfinite(atr[decision_index]):
            continue
        entry_index = decision_index + 1
        exit_index = decision_index + horizon_bars
        continuous = True
        for index in range(decision_index, exit_index):
            if base_bars[index + 1, fx.BAR_TIME] - base_bars[index, fx.BAR_TIME] != M5_SECONDS:
                continuous = False
                break
        if not continuous:
            continue
        if side == fx.LONG:
            base_entry = base_bars[entry_index, fx.ASK_OPEN] + base_slippage[entry_index]
            stress_entry = stress_bars[entry_index, fx.ASK_OPEN] + stress_slippage[entry_index]
            base_exit = base_bars[exit_index, fx.BID_CLOSE] - base_slippage[exit_index]
            stress_exit = stress_bars[exit_index, fx.BID_CLOSE] - stress_slippage[exit_index]
        else:
            base_entry = base_bars[entry_index, fx.BID_OPEN] - base_slippage[entry_index]
            stress_entry = stress_bars[entry_index, fx.BID_OPEN] - stress_slippage[entry_index]
            base_exit = base_bars[exit_index, fx.ASK_CLOSE] + base_slippage[exit_index]
            stress_exit = stress_bars[exit_index, fx.ASK_CLOSE] + stress_slippage[exit_index]
        observed_spread_pips = (
            base_bars[entry_index, fx.ASK_OPEN] - base_bars[entry_index, fx.BID_OPEN]
        ) / pip
        floor_pips = 4.0 * (
            observed_spread_pips + 2.0 * base_slippage[entry_index] / pip + base_commission_pips
        )
        risk_price = max(atr[decision_index], floor_pips * pip)
        terminal_base = ((base_exit - base_entry) * side / pip - base_commission_pips) * pip / risk_price
        terminal_stress = (
            ((stress_exit - stress_entry) * side / pip - stress_commission_pips) * pip / risk_price
        )
        maximum_favorable = -np.inf
        maximum_adverse = -np.inf
        target_hit = np.zeros(3, dtype=np.float64)
        base_status = np.zeros(3, dtype=np.int8)  # 0 timeout, 1 target, -1 stop
        stress_status = np.zeros(3, dtype=np.int8)
        base_stop_slip = np.zeros(3, dtype=np.float64)
        stress_stop_slip = np.zeros(3, dtype=np.float64)
        base_stop_price = base_entry - side * risk_price
        stress_stop_price = stress_entry - side * risk_price
        for index in range(entry_index, exit_index + 1):
            if side == fx.LONG:
                executable_high = base_bars[index, fx.BID_HIGH] - base_slippage[index]
                executable_low = base_bars[index, fx.BID_LOW] - base_slippage[index]
                stress_high = stress_bars[index, fx.BID_HIGH] - stress_slippage[index]
                stress_low = stress_bars[index, fx.BID_LOW] - stress_slippage[index]
                favorable = executable_high - base_entry
                adverse = base_entry - executable_low
                base_stopped = executable_low <= base_stop_price
                stress_stopped = stress_low <= stress_stop_price
            else:
                executable_high = base_bars[index, fx.ASK_HIGH] + base_slippage[index]
                executable_low = base_bars[index, fx.ASK_LOW] + base_slippage[index]
                stress_high = stress_bars[index, fx.ASK_HIGH] + stress_slippage[index]
                stress_low = stress_bars[index, fx.ASK_LOW] + stress_slippage[index]
                favorable = base_entry - executable_low
                adverse = executable_high - base_entry
                base_stopped = executable_high >= base_stop_price
                stress_stopped = stress_high >= stress_stop_price
            maximum_favorable = max(maximum_favorable, favorable)
            maximum_adverse = max(maximum_adverse, adverse)
            for target_index in range(3):
                if base_status[target_index] == 0:
                    base_target = base_entry + side * BARRIER_TARGETS[target_index] * risk_price
                    base_reached = (
                        executable_high >= base_target
                        if side == fx.LONG
                        else executable_low <= base_target
                    )
                    if base_stopped:
                        base_status[target_index] = -1
                        base_stop_slip[target_index] = base_slippage[index]
                    elif base_reached:
                        base_status[target_index] = 1
                        target_hit[target_index] = 1.0
                if stress_status[target_index] == 0:
                    stress_target = stress_entry + side * BARRIER_TARGETS[target_index] * risk_price
                    stress_reached = (
                        stress_high >= stress_target
                        if side == fx.LONG
                        else stress_low <= stress_target
                    )
                    if stress_stopped:
                        stress_status[target_index] = -1
                        stress_stop_slip[target_index] = stress_slippage[index]
                    elif stress_reached:
                        stress_status[target_index] = 1
        output[decision_index, 0] = risk_price
        output[decision_index, 1] = terminal_base
        output[decision_index, 2] = terminal_stress
        output[decision_index, 3] = (maximum_favorable - base_commission_pips * pip) / risk_price
        output[decision_index, 4] = (maximum_adverse + base_commission_pips * pip) / risk_price
        output[decision_index, 5:8] = target_hit
        base_commission_r = base_commission_pips * pip / risk_price
        stress_commission_r = stress_commission_pips * pip / risk_price
        for target_index in range(3):
            target_r = BARRIER_TARGETS[target_index]
            if base_status[target_index] == 1:
                base_barrier_r = target_r - base_commission_r
            elif base_status[target_index] == -1:
                base_barrier_r = (
                    -1.0 - base_commission_r - base_stop_slip[target_index] / risk_price
                )
            else:
                base_barrier_r = terminal_base
            if stress_status[target_index] == 1:
                stress_barrier_r = target_r - stress_commission_r
            elif stress_status[target_index] == -1:
                stress_barrier_r = (
                    -1.0
                    - stress_commission_r
                    - stress_stop_slip[target_index] / risk_price
                )
            else:
                stress_barrier_r = terminal_stress
            output[decision_index, 8 + 2 * target_index] = base_barrier_r
            output[decision_index, 9 + 2 * target_index] = stress_barrier_r
    return output


def build_outcome_wide(
    m5: np.ndarray,
    features: pd.DataFrame,
    instrument: Instrument,
    costs: EventCosts,
    *,
    horizons_minutes: Iterable[int] = HORIZONS_MINUTES,
) -> pd.DataFrame:
    """Build the same outcomes as `build_outcome_frame` without duplicating feature rows.

    The compiled, wide representation is used for full-history materialization. Each column name
    encodes horizon and side; invalid rows remain NaN so alignment with causal features is exact.
    """
    bars = _validate_bars(m5)
    if len(features) != len(bars):
        raise ValueError("features and M5 bars must have equal length")
    horizons = tuple(int(value) for value in horizons_minutes)
    if not horizons or any(value <= 0 or value % 5 for value in horizons):
        raise ValueError("horizons must be positive whole multiples of five minutes")
    base_bars, base_slippage = prepare_run(bars, instrument, costs.base)
    stress_bars, stress_slippage = prepare_run(bars, instrument, costs.stress)
    atr = features["atr_price"].to_numpy(dtype=np.float64)
    contiguous = features["history_contiguous"].fillna(False).to_numpy(dtype=bool)
    names = (
        "risk_price",
        "terminal_r_base",
        "terminal_r_stress",
        "mfe_r_base",
        "mae_r_base",
        "target_0_5r_before_stop",
        "target_1_0r_before_stop",
        "target_2_0r_before_stop",
        "barrier_0_5r_base",
        "barrier_0_5r_stress",
        "barrier_1_0r_base",
        "barrier_1_0r_stress",
        "barrier_2_0r_base",
        "barrier_2_0r_stress",
    )
    columns: dict[str, np.ndarray] = {}
    for horizon_minutes in horizons:
        for side, side_name in ((fx.LONG, "long"), (fx.SHORT, "short")):
            values = _compiled_outcome(
                base_bars,
                base_slippage,
                stress_bars,
                stress_slippage,
                atr,
                contiguous,
                horizon_minutes // 5,
                side,
                instrument.pip_size,
                costs.base_commission_pips,
                costs.stress_commission_pips,
            )
            for position, name in enumerate(names):
                columns[f"h{horizon_minutes}_{side_name}_{name}"] = values[:, position]
    return pd.DataFrame(columns, index=features.index)


def _outcome_row(  # noqa: PLR0913
    base_bars: np.ndarray,
    base_slippage: np.ndarray,
    stress_bars: np.ndarray,
    stress_slippage: np.ndarray,
    decision_index: int,
    entry_index: int,
    exit_index: int,
    side: int,
    atr_price: float,
    instrument: Instrument,
    costs: EventCosts,
    horizon_minutes: int,
    decision_time: pd.Timestamp,
) -> dict[str, object]:
    pip = instrument.pip_size
    if side == fx.LONG:
        base_entry = base_bars[entry_index, fx.ASK_OPEN] + base_slippage[entry_index]
        stress_entry = stress_bars[entry_index, fx.ASK_OPEN] + stress_slippage[entry_index]
        base_exit = base_bars[exit_index, fx.BID_CLOSE] - base_slippage[exit_index]
        stress_exit = stress_bars[exit_index, fx.BID_CLOSE] - stress_slippage[exit_index]
        favorable = base_bars[entry_index:exit_index + 1, fx.BID_HIGH] - base_slippage[entry_index:exit_index + 1]
        adverse = base_bars[entry_index:exit_index + 1, fx.BID_LOW] - base_slippage[entry_index:exit_index + 1]
        mfe_price = favorable.max() - base_entry
        mae_price = base_entry - adverse.min()
    else:
        base_entry = base_bars[entry_index, fx.BID_OPEN] - base_slippage[entry_index]
        stress_entry = stress_bars[entry_index, fx.BID_OPEN] - stress_slippage[entry_index]
        base_exit = base_bars[exit_index, fx.ASK_CLOSE] + base_slippage[exit_index]
        stress_exit = stress_bars[exit_index, fx.ASK_CLOSE] + stress_slippage[exit_index]
        favorable = base_bars[entry_index:exit_index + 1, fx.ASK_LOW] + base_slippage[entry_index:exit_index + 1]
        adverse = base_bars[entry_index:exit_index + 1, fx.ASK_HIGH] + base_slippage[entry_index:exit_index + 1]
        mfe_price = base_entry - favorable.min()
        mae_price = adverse.max() - base_entry
    observed_spread_pips = (
        base_bars[entry_index, fx.ASK_OPEN] - base_bars[entry_index, fx.BID_OPEN]
    ) / pip
    cost_floor_pips = 4.0 * (
        observed_spread_pips
        + 2.0 * base_slippage[entry_index] / pip
        + costs.base_commission_pips
    )
    risk_price = max(atr_price, cost_floor_pips * pip)
    base_terminal = ((base_exit - base_entry) * side / pip - costs.base_commission_pips) * pip / risk_price
    stress_terminal = ((stress_exit - stress_entry) * side / pip - costs.stress_commission_pips) * pip / risk_price
    row: dict[str, object] = {
        "decision_time": decision_time,
        "decision_index": decision_index,
        "entry_index": entry_index,
        "exit_index": exit_index,
        "side": side,
        "horizon_minutes": horizon_minutes,
        "risk_price": risk_price,
        "terminal_r_base": base_terminal,
        "terminal_r_stress": stress_terminal,
        "mfe_r_base": (mfe_price - costs.base_commission_pips * pip) / risk_price,
        "mae_r_base": (mae_price + costs.base_commission_pips * pip) / risk_price,
    }
    for target_r in BARRIER_TARGETS:
        label = str(target_r).replace(".", "_")
        row[f"target_{label}r_before_stop"] = _barrier_result(
            base_bars,
            base_slippage,
            entry_index,
            exit_index,
            side,
            base_entry,
            risk_price,
            target_r,
        )
    return row


def add_cross_pair_context(feature_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add a causal equal-weight USD/common-market momentum factor to aligned pair frames."""
    if not feature_frames:
        return {}
    signed_returns: list[pd.Series] = []
    usd_sign = {
        "EURUSD": -1,
        "GBPUSD": -1,
        "AUDUSD": -1,
        "NZDUSD": -1,
        "USDCAD": 1,
        "USDCHF": 1,
        "USDJPY": 1,
    }
    for pair, frame in feature_frames.items():
        if pair in usd_sign:
            signed_returns.append(frame["return_3"].rename(pair) * usd_sign[pair])
    usd_factor = pd.concat(signed_returns, axis=1).mean(axis=1, skipna=True) if signed_returns else pd.Series(dtype=float)
    common_factor = pd.concat(
        [frame["return_3"].rename(pair) for pair, frame in feature_frames.items()], axis=1
    ).mean(axis=1, skipna=True)
    result: dict[str, pd.DataFrame] = {}
    for pair, frame in feature_frames.items():
        enriched = frame.copy()
        enriched["usd_factor_return_3"] = usd_factor.reindex(frame.index)
        enriched["common_factor_return_3"] = common_factor.reindex(frame.index)
        result[pair] = enriched
    return result
