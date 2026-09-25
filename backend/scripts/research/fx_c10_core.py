"""Core of round C10: day-trade signals on 5- to 60-minute bars, managed minute by minute with the grid of stops,
partial profits, runner targets and trailing stops, in the 37 banca markets.

A *family* turns the bars of one timeframe into signals: `side[i]` in {-1, 0, +1} decided at the close of bar i with
only bars up to i (and daily bars already closed), and `stop_ref[i]`, a structural stop level or NaN. Entries execute
at the open of the next minute, as in every earlier round. Every trade is flat by the end of its FX day (17:00 New
York): this is day trading.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

from scripts.research import fx_c3_lab as c3

TIMEFRAME_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60}
MIN_MINUTES_LEFT = 15  # no entry in the last 15 minutes of an FX day
MIN_STOP_ATR = 0.1  # a structural stop closer than 0.1 daily ATR is widened to it
EXCHANGE = {  # main exchange clock of each index: its cash session (open, close in local minutes) defines "the session"
    "SPXUSD": ("America/New_York", 9 * 60 + 30, 16 * 60), "NSXUSD": ("America/New_York", 9 * 60 + 30, 16 * 60),
    "UDXUSD": ("America/New_York", 8 * 60 + 20, 16 * 60), "GRXEUR": ("Europe/Berlin", 9 * 60, 17 * 60 + 30),
    "FRXEUR": ("Europe/Paris", 9 * 60, 17 * 60 + 30), "UKXGBP": ("Europe/London", 8 * 60, 16 * 60 + 30),
    "JPXJPY": ("Asia/Tokyo", 9 * 60, 15 * 60), "HKXHKD": ("Asia/Hong_Kong", 9 * 60 + 30, 16 * 60),
    "AUXAUD": ("Australia/Sydney", 10 * 60, 16 * 60),
}
FX_EXCHANGE = ("Europe/London", 8 * 60, 16 * 60 + 30)  # for currency pairs the session is London's


def minute_keys(seconds: np.ndarray, minutes: int) -> np.ndarray:
    """Bar id of every minute for a timeframe of `minutes`, aligned to the FX day that starts at 17:00 New York."""
    local = pd.to_datetime(seconds, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    shifted = (local + pd.Timedelta(hours=7)).to_numpy().astype("datetime64[m]").astype(np.int64)
    return shifted // minutes


def local_minutes(seconds: np.ndarray, zone: str) -> tuple[np.ndarray, np.ndarray]:
    """Minute of the day and calendar day number in a timezone."""
    local = pd.to_datetime(seconds, unit="s", utc=True).tz_convert(zone).tz_localize(None)
    minute = local.to_numpy().astype("datetime64[m]").astype(np.int64)
    day = minute // 1440
    return minute - day * 1440, day


@dataclass
class Context:
    """Everything a family may look at for one market and one timeframe (index = bar of that timeframe)."""

    symbol: str
    timeframe: str
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    open_time: np.ndarray  # UTC seconds of the bar's first minute
    last_minute: np.ndarray  # index in the minute array of the bar's last minute
    fx_day: np.ndarray  # FX day id (17:00 New York to 17:00 New York)
    ny_minute: np.ndarray  # New York minute of the day of the bar's open
    exchange_minute: np.ndarray  # minute of the day of the bar's open on the market's main exchange clock
    exchange_day: np.ndarray
    exchange_open: int  # the market's session open and close (minutes after local midnight)
    exchange_close: int
    day_open: np.ndarray  # open of the current FX day (known from its first minute)
    prev_high: np.ndarray  # previous completed FX day's high, low, close, open and range
    prev_low: np.ndarray
    prev_close: np.ndarray
    prev_open: np.ndarray
    atr_d1: np.ndarray  # ATR(14) of completed FX days
    sma50_d1: np.ndarray  # 50-day average of completed FX days' closes
    d1: np.ndarray  # every FX day's (open, high, low, close); a bar may only read rows before its bar_day
    bar_day: np.ndarray  # row of d1 of the FX day each bar belongs to
    extra: dict = field(default_factory=dict)


def build_context(symbol: str, m1: np.ndarray, timeframe: str) -> Context:
    keys = minute_keys(m1[:, 0], TIMEFRAME_MINUTES[timeframe])
    last, bars = c3.aggregate(m1, keys)
    first = np.concatenate(([0], last[:-1] + 1))
    open_time = m1[first, 0]
    day_keys = minute_keys(m1[:, 0], 1440)
    d_last, d1 = c3.aggregate(m1, day_keys)
    d_first = np.concatenate(([0], d_last[:-1] + 1))
    atr14 = c3.atr(d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3], 14)
    sma50 = c3.sma(d1[:, 3], 50)
    # the FX day each bar belongs to, and the last FX day already complete when the bar closes
    bar_day = np.searchsorted(d_last, last, side="left")
    prev = bar_day - 1
    ok = prev >= 0

    def previous(values):
        out = np.full(len(last), np.nan)
        out[ok] = values[prev[ok]]
        return out

    ny_minute, _ = local_minutes(open_time, "America/New_York")
    zone, exchange_open, exchange_close = EXCHANGE.get(symbol, FX_EXCHANGE)
    exchange_minute, exchange_day = local_minutes(open_time, zone)
    return Context(
        symbol=symbol, timeframe=timeframe, o=bars[:, 0], h=bars[:, 1], l=bars[:, 2], c=bars[:, 3], open_time=open_time,
        last_minute=last, fx_day=day_keys[last], ny_minute=ny_minute, exchange_minute=exchange_minute,
        exchange_day=exchange_day, exchange_open=exchange_open, exchange_close=exchange_close,
        day_open=m1[d_first[bar_day], 1], prev_high=previous(d1[:, 1]), prev_low=previous(d1[:, 2]),
        prev_close=previous(d1[:, 3]), prev_open=previous(d1[:, 0]), atr_d1=previous(atr14), sma50_d1=previous(sma50),
        d1=d1, bar_day=bar_day,
    )


def in_session(ctx: Context, start: int, end: int) -> np.ndarray:
    """Bars that open at or after `start` and before `end` (minutes of the exchange's local day)."""
    return (ctx.exchange_minute >= start) & (ctx.exchange_minute < end)


def window_extremes(ctx: Context, window: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running high and low, per `day`, of the bars inside `window` up to and including each bar (NaN before)."""
    frame = pd.DataFrame({"day": day, "h": np.where(window, ctx.h, np.nan), "l": np.where(window, ctx.l, np.nan)})
    grouped = frame.groupby("day", sort=False)
    # cummax leaves NaN on the bars outside the window: carry the running extreme forward within the day
    running = pd.DataFrame({"day": day, "h": grouped["h"].cummax(), "l": grouped["l"].cummin()}).groupby("day", sort=False)
    return running["h"].ffill().to_numpy(), running["l"].ffill().to_numpy()


@njit(cache=True)
def first_of_day(day, wanted):
    """Only the first nonzero signal of each `day` survives."""
    side = np.zeros(wanted.shape[0], dtype=np.int8)
    done_day = -(2**62)
    for i in range(wanted.shape[0]):
        if wanted[i] != 0 and day[i] != done_day:
            side[i] = wanted[i]
            done_day = day[i]
    return side


def first_break(day: np.ndarray, active: np.ndarray, close: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """+1 at the first active bar of each `day` that closes above `up`, -1 below `down`; one signal a day."""
    with np.errstate(invalid="ignore"):
        wanted = np.where(active & (close > up), 1, np.where(active & (close < down), -1, 0)).astype(np.int8)
    return first_of_day(np.asarray(day, dtype=np.int64), wanted)


def stops_by_side(side: np.ndarray, long_stop: np.ndarray, short_stop: np.ndarray) -> np.ndarray:
    return np.where(side == 1, long_stop, np.where(side == -1, short_stop, np.nan))


def session_price(ctx: Context, minute: int) -> np.ndarray:
    """Close of the last bar that opened before `minute` of the exchange's local day, on that day's bars from `minute`
    on (NaN before: the price is not known yet)."""
    before = ctx.exchange_minute < minute
    frame = pd.DataFrame({"day": ctx.exchange_day, "c": np.where(before, ctx.c, np.nan)})
    carried = frame.groupby("day", sort=False)["c"].ffill().to_numpy()
    return np.where(before, np.nan, carried)


def previous_session_close(ctx: Context) -> np.ndarray:
    """Close of the previous exchange day's session (its last bar that opened before the session close)."""
    inside = ctx.exchange_minute < ctx.exchange_close
    frame = pd.DataFrame({"day": ctx.exchange_day, "c": np.where(inside, ctx.c, np.nan)})
    last = frame.groupby("day", sort=True)["c"].last().dropna()  # days with no bar before the close (a Sunday
    # evening in London) are not sessions: Monday's previous session is Friday's
    return pd.Series(ctx.exchange_day).map(last.shift(1)).to_numpy(dtype=float)


def day_end_minutes(m1: np.ndarray) -> np.ndarray:
    """For every minute, the index of the last minute of its FX day."""
    day = minute_keys(m1[:, 0], 1440)
    change = np.flatnonzero(np.diff(day)) + 1
    ends = np.concatenate((change - 1, [len(day) - 1]))
    return ends[np.searchsorted(change, np.arange(len(day)), side="right")]


# --- management -----------------------------------------------------------------------------------------------------

# Management columns: stop in daily ATRs (0 = the family's structural stop), first target in R (0 = none), fraction
# banked at the first target, runner target in R (0 = none), trailing distance in R (0 = none), maximum minutes
# (0 = until the end of the FX day). After the first target the stop of the rest moves to break-even.
STOP_ATR, TP1_R, TP1_FRACTION, RUNNER_R, TRAIL_R, MAX_MINUTES = range(6)


@njit(cache=True)
def manage(m1, day_end, entries, sides, stop_refs, atrs, cost, grid, g):
    """Every trade of management row `g`: one position at a time, flat at the end of the FX day.
    Returns rows (grid index, entry minute, exit minute, R net, R net with double cost, % of price net)."""
    out = np.empty((entries.shape[0], 6))
    times = m1[:, 0].copy()
    count = 0
    busy_until = -1
    for k in range(entries.shape[0]):
        e = entries[k]
        if e <= busy_until or e >= m1.shape[0]:
            continue
        end = day_end[e]
        if m1[end, 0] - m1[e, 0] < (MIN_MINUTES_LEFT - 1) * 60:
            continue
        if grid[g, MAX_MINUTES] > 0:  # clock minutes, not data minutes: the last minute starting before the limit
            end = min(end, np.searchsorted(times, m1[e, 0] + grid[g, MAX_MINUTES] * 60) - 1)
        side = sides[k]
        entry = m1[e, 1]
        if grid[g, STOP_ATR] > 0:
            distance = grid[g, STOP_ATR] * atrs[k]
        elif not np.isnan(stop_refs[k]):
            distance = max(side * (entry - stop_refs[k]), MIN_STOP_ATR * atrs[k])
        else:
            continue
        if not distance > 0:
            continue
        stop = entry - side * distance
        tp1 = entry + side * grid[g, TP1_R] * distance if grid[g, TP1_R] > 0 else np.nan
        runner = entry + side * grid[g, RUNNER_R] * distance if grid[g, RUNNER_R] > 0 else np.nan
        trail = grid[g, TRAIL_R] * distance
        remaining, gross, best = 1.0, 0.0, entry
        first_done = np.isnan(tp1)
        exit_minute = end
        for i in range(e, end + 1):
            o, h, l = m1[i, 1], m1[i, 2], m1[i, 3]
            banked_now = False
            # the stop is checked first: when stop and target fit in the same minute, the stop wins
            if (side == 1 and l <= stop) or (side == -1 and h >= stop):
                fill = min(stop, o) if side == 1 else max(stop, o)
                gross += remaining * side * (fill - entry)
                remaining = 0.0
                exit_minute = i
                break
            if not first_done and ((side == 1 and h >= tp1) or (side == -1 and l <= tp1)):
                fraction = grid[g, TP1_FRACTION]
                gross += fraction * side * (tp1 - entry)
                remaining -= fraction
                first_done = True
                banked_now = True
                stop = entry  # break-even for the rest
                best = h if side == 1 else l
                if remaining <= 1e-12:
                    exit_minute = i
                    break
            # the order inside a minute is unknown: the runner never fills in the minute that banked the first part
            if not banked_now and not np.isnan(runner) and ((side == 1 and h >= runner) or (side == -1 and l <= runner)):
                gross += remaining * side * (runner - entry)
                remaining = 0.0
                exit_minute = i
                break
            if trail > 0 and first_done:
                if side == 1:
                    best = max(best, h)
                    stop = max(stop, best - trail)
                else:
                    best = min(best, l)
                    stop = min(stop, best + trail)
        if remaining > 0:
            gross += remaining * side * (m1[exit_minute, 4] - entry)
        out[count, 0] = g
        out[count, 1] = e
        out[count, 2] = exit_minute
        out[count, 3] = (gross - cost) / distance
        out[count, 4] = (gross - 2 * cost) / distance
        out[count, 5] = (gross - cost) / entry * 100
        count += 1
        busy_until = exit_minute
    return out[:count]


def management_grid() -> tuple[np.ndarray, list[str]]:
    """The fixed management grid: 4 stops (the family's own, 0.25, 0.5, 1 daily ATR) x 12 exit shapes x 3 maximum
    holding times = 144 rows."""
    shapes = []
    for target in (0.5, 1.0, 2.0, 3.0):
        shapes.append((f"alvo {target:g}R", 0.0, 0.0, target, 0.0))
    for tp1 in (0.5, 1.0):
        for runner in (3.0, 6.0):
            shapes.append((f"metade em {tp1:g}R, resto em {runner:g}R", tp1, 0.5, runner, 0.0))
        for trail in (0.5, 1.0):
            shapes.append((f"metade em {tp1:g}R, resto com trailing {trail:g}R", tp1, 0.5, 0.0, trail))
    rows, names = [], []
    for stop in (0.0, 0.25, 0.5, 1.0):
        for name, tp1, fraction, runner, trail in shapes:
            for hold in (60.0, 240.0, 0.0):
                rows.append((stop, tp1, fraction, runner, trail, hold))
                stop_name = "stop estrutural" if stop == 0 else f"stop {stop:g} ATR"
                names.append(f"{stop_name} · {name} · " + ("até o fim do dia" if hold == 0 else f"máx {hold:g} min"))
    return np.array(rows, dtype=float), names


Family = Callable[[Context, dict], tuple[np.ndarray, np.ndarray]]


def entry_arrays(ctx: Context, side: np.ndarray, stop_ref: np.ndarray, m1_length: int):
    """Signals at bar closes -> entries at the next minute (index), their side, structural stop and daily ATR."""
    signal = np.flatnonzero((side != 0) & ~np.isnan(ctx.atr_d1))
    entries = ctx.last_minute[signal] + 1
    keep = entries < m1_length
    signal = signal[keep]
    return (ctx.last_minute[signal] + 1).astype(np.int64), side[signal].astype(np.int64), \
        np.asarray(stop_ref, dtype=float)[signal], ctx.atr_d1[signal]
