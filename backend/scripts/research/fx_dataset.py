"""Research-only loader for hourly FX candles that carry both bid and ask.

Source: the public Dukascopy datafeed (monthly hourly candle files, LZMA-compressed
big-endian records). The data is for personal research on cost-aware backtests; check
the provider's terms before any other use. This module never touches the trading
engine, the database, or an exchange account.

Two provider quirks matter for backtests and are handled here:

* Every hour of the month is present. Weekend and holiday hours are filled with
  flat candles (open == high == low == close, volume 0). Keeping them would make a
  backtest trade a closed market and would hide the real 5/7 session density.
* The month in the URL is zero-based.
"""

from __future__ import annotations

import argparse
import lzma
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

DATAFEED_URL = "https://datafeed.dukascopy.com/datafeed"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
CANDLE_DTYPE = np.dtype(
    [
        ("offset_seconds", ">u4"),
        ("open", ">u4"),
        ("close", ">u4"),
        ("low", ">u4"),
        ("high", ">u4"),
        ("volume", ">f4"),
    ]
)
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class FxInstrument:
    """Static facts needed to decode prices and express costs in pips."""

    symbol: str
    price_scale: int
    pip_size: float
    plausible_price: tuple[float, float]


MAJORS: dict[str, FxInstrument] = {
    "EURUSD": FxInstrument("EURUSD", 100_000, 0.0001, (0.5, 2.5)),
    "GBPUSD": FxInstrument("GBPUSD", 100_000, 0.0001, (0.5, 2.5)),
    "AUDUSD": FxInstrument("AUDUSD", 100_000, 0.0001, (0.3, 1.5)),
    "NZDUSD": FxInstrument("NZDUSD", 100_000, 0.0001, (0.3, 1.5)),
    "USDCAD": FxInstrument("USDCAD", 100_000, 0.0001, (0.8, 2.0)),
    "USDCHF": FxInstrument("USDCHF", 100_000, 0.0001, (0.5, 1.5)),
    "USDJPY": FxInstrument("USDJPY", 1_000, 0.01, (60.0, 220.0)),
}


class FxDataDownloadError(RuntimeError):
    """Raised when the provider keeps failing after every retry."""


def month_url(symbol: str, year: int, month: int, side: str) -> str:
    """Build the monthly hourly-candle URL; the provider's month index starts at 0."""
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    if side not in {"BID", "ASK"}:
        raise ValueError("side must be BID or ASK")
    return f"{DATAFEED_URL}/{symbol}/{year}/{month - 1:02d}/{side}_candles_hour_1.bi5"


def _empty_candles() -> pd.DataFrame:
    frame = pd.DataFrame(columns=[*PRICE_COLUMNS, "volume"], dtype="float64")
    frame.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return frame


def decode_hourly_candles(
    payload: bytes, *, year: int, month: int, price_scale: int
) -> pd.DataFrame:
    """Decode one monthly file into candles with real trading activity only."""
    if not payload:
        return _empty_candles()
    raw = np.frombuffer(lzma.decompress(payload), dtype=CANDLE_DTYPE)
    if raw.size == 0:
        return _empty_candles()

    month_start = pd.Timestamp(year=year, month=month, day=1, tz=UTC)
    index = month_start + pd.to_timedelta(raw["offset_seconds"].astype("int64"), unit="s")
    frame = pd.DataFrame(
        {
            name: raw[name].astype("float64") / price_scale
            for name in ("open", "high", "low", "close")
        }
        | {"volume": raw["volume"].astype("float64")},
        index=pd.DatetimeIndex(index, name="timestamp"),
    )
    is_placeholder = (
        (frame["open"] == frame["high"])
        & (frame["open"] == frame["low"])
        & (frame["open"] == frame["close"])
        & (frame["volume"] == 0)
    )
    return frame.loc[~is_placeholder]


def fetch_month_payload(
    client: httpx.Client,
    symbol: str,
    year: int,
    month: int,
    side: str,
    cache_dir: Path,
    *,
    retries: int = 10,
    base_delay_seconds: float = 2.0,
    max_delay_seconds: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Return the raw file for one month, caching it (an empty file means no data)."""
    cache_path = cache_dir / symbol / f"{year}-{month:02d}-{side}.bi5"
    if cache_path.exists():
        return cache_path.read_bytes()

    url = month_url(symbol, year, month, side)
    last_problem = "no attempt made"
    for attempt in range(retries):
        try:
            response = client.get(url)
        except httpx.TransportError as error:
            last_problem = f"transport error: {error!r}"
        else:
            if response.status_code == 200:
                payload = response.content
                break
            if response.status_code == 404:
                payload = b""
                break
            last_problem = f"HTTP {response.status_code}"
            if response.status_code not in RETRYABLE_STATUS:
                raise FxDataDownloadError(f"{url} failed permanently: {last_problem}")
        sleep(min(base_delay_seconds * 2**attempt, max_delay_seconds))
    else:
        raise FxDataDownloadError(f"{url} failed after {retries} attempts: {last_problem}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    return payload


def build_pair_dataset(
    client: httpx.Client,
    instrument: FxInstrument,
    months: list[tuple[int, int]],
    cache_dir: Path,
    *,
    politeness_seconds: float = 1.5,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download bid and ask for every month and join them on the candle timestamp."""
    sides: dict[str, list[pd.DataFrame]] = {"BID": [], "ASK": []}
    for year, month in months:
        for side in sides:
            cached = (cache_dir / instrument.symbol / f"{year}-{month:02d}-{side}.bi5").exists()
            payload = fetch_month_payload(
                client, instrument.symbol, year, month, side, cache_dir, sleep=sleep
            )
            if not cached:
                sleep(politeness_seconds)
            sides[side].append(
                decode_hourly_candles(
                    payload, year=year, month=month, price_scale=instrument.price_scale
                )
            )

    bid = pd.concat(sides["BID"]).add_prefix("bid_").drop(columns="bid_volume")
    ask = pd.concat(sides["ASK"]).add_prefix("ask_").drop(columns="ask_volume")
    joined = bid.join(ask, how="inner").sort_index()
    joined["spread_open_pips"] = (joined["ask_open"] - joined["bid_open"]) / instrument.pip_size
    joined["spread_close_pips"] = (joined["ask_close"] - joined["bid_close"]) / instrument.pip_size
    return joined


def _is_weekend_gap(previous: pd.Timestamp, current: pd.Timestamp) -> bool:
    """A gap that starts on Friday and ends on Sunday or Monday is the normal weekend."""
    return previous.dayofweek == 4 and current.dayofweek in (6, 0)


def quality_report(frame: pd.DataFrame, instrument: FxInstrument) -> dict[str, object]:
    """Summarize the properties a cost-aware backtest depends on.

    Session density is only meaningful over months of data: a short window that ends
    on a Friday leaves out the trailing weekend and overstates it.
    """
    if frame.empty:
        return {"symbol": instrument.symbol, "candles": 0, "problems": ["no candles"]}

    hours_in_range = int((frame.index[-1] - frame.index[0]) / pd.Timedelta(hours=1)) + 1
    density = len(frame) / hours_in_range
    gaps = frame.index.to_series().diff().dropna()
    unexplained_gaps = [
        (previous, current)
        for previous, current in zip(frame.index[:-1], frame.index[1:], strict=True)
        if current - previous > pd.Timedelta(hours=1) and not _is_weekend_gap(previous, current)
    ]
    saturday_candles = int((frame.index.dayofweek == 5).sum())
    crossed_quotes = int((frame["ask_open"] < frame["bid_open"]).sum())
    bad_ohlc = int(
        (
            (frame["bid_low"] > frame[["bid_open", "bid_close"]].min(axis=1))
            | (frame["bid_high"] < frame[["bid_open", "bid_close"]].max(axis=1))
        ).sum()
    )
    median_price = float(frame["bid_close"].median())
    low, high = instrument.plausible_price

    problems: list[str] = []
    if not 0.66 <= density <= 0.74:
        problems.append(f"session density {density:.3f} is outside 5/7 ± 0.05")
    if saturday_candles:
        problems.append(f"{saturday_candles} candles on Saturday")
    if crossed_quotes:
        problems.append(f"{crossed_quotes} candles with ask below bid")
    if bad_ohlc:
        problems.append(f"{bad_ohlc} candles with inconsistent OHLC")
    if not low <= median_price <= high:
        problems.append(f"median price {median_price:.4f} is implausible for {instrument.symbol}")

    return {
        "symbol": instrument.symbol,
        "candles": len(frame),
        "first": frame.index[0].isoformat(),
        "last": frame.index[-1].isoformat(),
        "density": round(density, 3),
        "saturday_candles": saturday_candles,
        "unexplained_gaps": len(unexplained_gaps),
        "longest_gap_hours": round(gaps.max() / pd.Timedelta(hours=1), 1),
        "spread_median_pips": round(float(frame["spread_open_pips"].median()), 3),
        "spread_p95_pips": round(float(frame["spread_open_pips"].quantile(0.95)), 3),
        "median_price": round(median_price, 5),
        "problems": problems,
    }


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) between two YYYY-MM strings."""
    first = datetime.strptime(start, "%Y-%m").replace(tzinfo=UTC)
    last = datetime.strptime(end, "%Y-%m").replace(tzinfo=UTC)
    if last < first:
        raise ValueError("end must not be before start")
    months: list[tuple[int, int]] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--symbols", nargs="+", default=list(MAJORS), choices=list(MAJORS))
    parser.add_argument("--start", default="2021-09", help="first month, YYYY-MM")
    parser.add_argument("--end", default="2026-08", help="last month, YYYY-MM")
    parser.add_argument("--out", type=Path, default=Path("data/raw/fx"))
    parser.add_argument("--politeness-seconds", type=float, default=1.5)
    args = parser.parse_args(argv)

    months = month_range(args.start, args.end)
    cache_dir = args.out / "cache"
    failed = False
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        for symbol in args.symbols:
            instrument = MAJORS[symbol]
            frame = build_pair_dataset(
                client,
                instrument,
                months,
                cache_dir,
                politeness_seconds=args.politeness_seconds,
            )
            args.out.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(args.out / f"{symbol}_H1.parquet")
            report = quality_report(frame, instrument)
            _emit(f"{symbol}: {report}")
            failed = failed or bool(report["problems"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
