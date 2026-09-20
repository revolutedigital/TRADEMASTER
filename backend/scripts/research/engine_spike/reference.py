"""Independent pure-Python reference for the compiled simulator core.

Written from the rules rather than from the numba code, and using pandas for the indicators, so
that agreement between the two is evidence of correctness and not of a copied mistake. It is slow
by design and only used in tests and to size the speed-up.
"""

from __future__ import annotations

import pandas as pd

LONG, SHORT = 1, -1
EXIT_SIGNAL, EXIT_STOP, EXIT_STOP_GAP, EXIT_TARGET, EXIT_END = 1, 2, 3, 4, 5


def simulate_reference(
    bars: pd.DataFrame,
    *,
    fast_span: int,
    slow_span: int,
    atr_period: int,
    stop_atr: float,
    reward_risk: float,
    slippage: float,
    warmup: int,
) -> list[dict[str, float]]:
    """Trades of an EMA-crossover strategy with an ATR stop, from bid/ask OHLC bars."""
    mid_close = (bars["bid_close"] + bars["ask_close"]) / 2
    mid_high = (bars["bid_high"] + bars["ask_high"]) / 2
    mid_low = (bars["bid_low"] + bars["ask_low"]) / 2
    fast = mid_close.ewm(span=fast_span, adjust=False).mean()
    slow = mid_close.ewm(span=slow_span, adjust=False).mean()
    previous = mid_close.shift(1)
    true_range = pd.concat(
        [mid_high - mid_low, (mid_high - previous).abs(), (mid_low - previous).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / atr_period, adjust=False).mean()

    signal = pd.Series(0, index=bars.index)
    crossed_up = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
    crossed_down = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
    signal[crossed_up] = LONG
    signal[crossed_down] = SHORT
    signal.iloc[:warmup] = 0

    values = {
        name: bars[name].to_numpy(dtype=float)
        for name in (
            "bid_open", "bid_high", "bid_low", "bid_close",
            "ask_open", "ask_high", "ask_low", "ask_close",
        )
    }
    trades: list[dict[str, float]] = []
    open_trade: dict[str, float] | None = None
    pending = 0

    def close(index: int, price: float, reason: int) -> None:
        nonlocal open_trade
        assert open_trade is not None
        trades.append({**open_trade, "exit_index": index, "exit_price": price, "reason": reason})
        open_trade = None

    for t in range(len(bars)):
        if pending != 0 and t > 0:
            if open_trade is not None and open_trade["side"] != pending:
                price = (
                    values["bid_open"][t] - slippage
                    if open_trade["side"] == LONG
                    else values["ask_open"][t] + slippage
                )
                close(t, price, EXIT_SIGNAL)
            if open_trade is None and atr.iloc[t - 1] > 0:
                distance = stop_atr * atr.iloc[t - 1]
                if pending == LONG:
                    entry = values["ask_open"][t] + slippage
                    stop, target = entry - distance, entry + reward_risk * distance
                else:
                    entry = values["bid_open"][t] - slippage
                    stop, target = entry + distance, entry - reward_risk * distance
                open_trade = {
                    "side": pending, "entry_index": t, "entry_price": entry,
                    "stop": stop, "target": target, "distance": distance,
                }
        pending = 0

        if open_trade is not None:
            if open_trade["side"] == LONG:
                if values["bid_open"][t] <= open_trade["stop"]:
                    close(t, values["bid_open"][t] - slippage, EXIT_STOP_GAP)
                elif values["bid_low"][t] <= open_trade["stop"]:
                    close(t, open_trade["stop"] - slippage, EXIT_STOP)
                elif values["bid_high"][t] >= open_trade["target"]:
                    close(t, open_trade["target"], EXIT_TARGET)
            else:
                if values["ask_open"][t] >= open_trade["stop"]:
                    close(t, values["ask_open"][t] + slippage, EXIT_STOP_GAP)
                elif values["ask_high"][t] >= open_trade["stop"]:
                    close(t, open_trade["stop"] + slippage, EXIT_STOP)
                elif values["ask_low"][t] <= open_trade["target"]:
                    close(t, open_trade["target"], EXIT_TARGET)

        if t >= warmup and signal.iloc[t] != 0:
            pending = int(signal.iloc[t])

    if open_trade is not None:
        last = len(bars) - 1
        price = (
            values["bid_close"][last] - slippage
            if open_trade["side"] == LONG
            else values["ask_close"][last] + slippage
        )
        close(last, price, EXIT_END)
    return trades
