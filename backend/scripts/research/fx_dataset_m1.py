"""Research-only loader for one-minute FX candles that carry both bid and ask.

Source: the public Dukascopy datafeed, one LZMA-compressed file per day and side
(`.../SYMBOL/YYYY/MM/DD/BID_candles_min_1.bi5`). As with the hourly files, the month in the
URL is zero-based, every minute of the day is present, and closed minutes are flat
zero-volume placeholders that must be dropped. The data is for personal research; check the
provider's terms before any other use. This module never touches the trading engine, the
database, or an exchange account.

The provider answers slowly (10 to 20 seconds per file) and returns 503 when it is loaded, so
downloads are cached on disk, retried with backoff, and limited to a few concurrent
connections. The command line prints the measured pace and projects how long a full download takes.
"""

from __future__ import annotations

import argparse
import lzma
import sys
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from scripts.research.fx_dataset import (
    CANDLE_DTYPE,
    ALL_PAIRS,
    DATAFEED_URL,
    MAJORS,
    USER_AGENT,
    FxInstrument,
    fetch_cached,
)

SATURDAY = 5
DEFAULT_WORKERS = 4  # the provider returned 503 storms above this


def day_url(symbol: str, day: date, side: str) -> str:
    """Build the daily M1 URL; the provider's month index starts at 0, the day at 1."""
    if side not in {"BID", "ASK"}:
        raise ValueError("side must be BID or ASK")
    return f"{DATAFEED_URL}/{symbol}/{day.year}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def trading_days(start: date, end: date) -> list[date]:
    """Every calendar day in [start, end] except Saturday, when the FX market is closed all day (UTC)."""
    if end < start:
        raise ValueError("end must not be before start")
    days = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return [day for day in days if day.weekday() != SATURDAY]


def cache_path(cache_dir: Path, symbol: str, day: date, side: str) -> Path:
    return cache_dir / symbol / f"{day:%Y-%m-%d}-{side}.bi5"


def decode_minute_candles(payload: bytes, *, day: date, price_scale: int) -> pd.DataFrame:
    """Decode one daily file into minutes with real trading activity only."""
    columns = ["open", "high", "low", "close", "volume"]
    if not payload:
        empty = pd.DataFrame(columns=columns, dtype="float64")
        empty.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return empty

    raw = np.frombuffer(lzma.decompress(payload), dtype=CANDLE_DTYPE)
    midnight = pd.Timestamp(year=day.year, month=day.month, day=day.day, tz=UTC)
    index = midnight + pd.to_timedelta(raw["offset_seconds"].astype("int64"), unit="s")
    frame = pd.DataFrame(
        {name: raw[name].astype("float64") / price_scale for name in ("open", "high", "low", "close")}
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


@dataclass(frozen=True)
class DownloadStats:
    files_requested: int
    files_downloaded: int
    files_from_cache: int
    seconds: float

    @property
    def files_per_minute(self) -> float:
        if self.seconds <= 0 or self.files_downloaded == 0:
            return 0.0
        return self.files_downloaded / (self.seconds / 60)


def ensure_cached(
    client: httpx.Client,
    symbol: str,
    days: Iterable[date],
    cache_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> DownloadStats:
    """Download every missing daily file with a small pool of workers."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    wanted = [(day, side) for day in days for side in ("BID", "ASK")]
    missing = [(day, side) for day, side in wanted if not cache_path(cache_dir, symbol, day, side).exists()]

    def fetch(item: tuple[date, str]) -> None:
        day, side = item
        fetch_cached(
            client,
            day_url(symbol, day, side),
            cache_path(cache_dir, symbol, day, side),
            sleep=sleep,
        )

    started = clock()
    if missing:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fetch, missing))
    return DownloadStats(
        files_requested=len(wanted),
        files_downloaded=len(missing),
        files_from_cache=len(wanted) - len(missing),
        seconds=clock() - started,
    )


def build_symbol_frame(
    cache_dir: Path, instrument: FxInstrument, days: Iterable[date]
) -> pd.DataFrame:
    """Join cached bid and ask minutes into one frame with the spread in pips."""
    bids: list[pd.DataFrame] = []
    asks: list[pd.DataFrame] = []
    for day in days:
        for side, target in (("BID", bids), ("ASK", asks)):
            path = cache_path(cache_dir, instrument.symbol, day, side)
            if not path.exists():
                raise FileNotFoundError(f"{path} was not downloaded")
            target.append(
                decode_minute_candles(path.read_bytes(), day=day, price_scale=instrument.price_scale)
            )
    if not bids:
        raise ValueError("no days requested")

    bid = pd.concat(bids).add_prefix("bid_").drop(columns="bid_volume")
    ask = pd.concat(asks).add_prefix("ask_").drop(columns="ask_volume")
    joined = bid.join(ask, how="inner").sort_index()
    joined["spread_open_pips"] = (joined["ask_open"] - joined["bid_open"]) / instrument.pip_size
    joined["spread_close_pips"] = (joined["ask_close"] - joined["bid_close"]) / instrument.pip_size
    return joined


def quality_report_m1(frame: pd.DataFrame, instrument: FxInstrument) -> dict[str, object]:
    """Properties that a minute-level backtest depends on."""
    if frame.empty:
        return {"symbol": instrument.symbol, "minutes": 0, "problems": ["no minutes"]}

    minutes_in_range = int((frame.index[-1] - frame.index[0]) / pd.Timedelta(minutes=1)) + 1
    density = len(frame) / minutes_in_range
    gaps = frame.index.to_series().diff().dropna()
    long_pauses = gaps[(gaps > pd.Timedelta(minutes=30)) & (gaps < pd.Timedelta(hours=40))]
    saturday_minutes = int((frame.index.dayofweek == SATURDAY).sum())
    crossed = int((frame["ask_open"] < frame["bid_open"]).sum())
    bad_ohlc = int(
        (
            (frame["bid_low"] > frame[["bid_open", "bid_close"]].min(axis=1))
            | (frame["bid_high"] < frame[["bid_open", "bid_close"]].max(axis=1))
        ).sum()
    )
    median_price = float(frame["bid_close"].median())
    low, high = instrument.plausible_price
    spread = frame["spread_open_pips"]

    problems: list[str] = []
    if not 0.66 <= density <= 0.74:
        problems.append(f"session density {density:.3f} is outside 5/7 ± 0.05")
    if saturday_minutes:
        problems.append(f"{saturday_minutes} minutes on Saturday")
    if crossed:
        problems.append(f"{crossed} minutes with ask below bid")
    if bad_ohlc:
        problems.append(f"{bad_ohlc} minutes with inconsistent OHLC")
    if not low <= median_price <= high:
        problems.append(f"median price {median_price:.4f} is implausible for {instrument.symbol}")

    return {
        "symbol": instrument.symbol,
        "minutes": len(frame),
        "first": frame.index[0].isoformat(),
        "last": frame.index[-1].isoformat(),
        "density": round(density, 3),
        "saturday_minutes": saturday_minutes,
        "pauses_over_30min_inside_week": len(long_pauses),
        "spread_median_pips": round(float(spread.median()), 3),
        "spread_p95_pips": round(float(spread.quantile(0.95)), 3),
        "spread_p99_pips": round(float(spread.quantile(0.99)), 3),
        "share_spread_over_3_pips": round(float((spread > 3).mean()), 4),
        "median_price": round(median_price, 5),
        "problems": problems,
    }


@dataclass(frozen=True)
class Projection:
    files_total: int
    hours_at_measured_pace: float
    files_per_minute: float


def project_download(
    *, pairs: int, years: float, files_per_minute: float
) -> Projection:
    """Files and hours needed for `pairs` symbols and `years` of daily bid and ask files."""
    if files_per_minute <= 0:
        raise ValueError("files_per_minute must be positive")
    days_per_year = 365 * 6 / 7  # every day except Saturday
    files_total = round(pairs * years * days_per_year * 2)
    return Projection(files_total, files_total / files_per_minute / 60, files_per_minute)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _emit(line: str = "") -> None:
    sys.stdout.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--symbols", nargs="+", default=["EURUSD"], choices=list(ALL_PAIRS))
    parser.add_argument("--start", type=parse_date, required=True, help="first day, YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, required=True, help="last day, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=Path("data/raw/fx_m1"))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--project-pairs", type=int, default=10)
    parser.add_argument("--project-years", type=float, default=5.0)
    args = parser.parse_args(argv)

    days = trading_days(args.start, args.end)
    cache_dir = args.out / "cache"
    failed = False
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0) as client:
        for symbol in args.symbols:
            instrument = ALL_PAIRS[symbol]
            stats = ensure_cached(client, symbol, days, cache_dir, workers=args.workers)
            frame = build_symbol_frame(cache_dir, instrument, days)
            args.out.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(args.out / f"{symbol}_M1_{args.start:%Y%m%d}_{args.end:%Y%m%d}.parquet")
            report = quality_report_m1(frame, instrument)
            failed = failed or bool(report["problems"])
            _emit(f"{symbol}: {report}")
            if stats.files_downloaded:
                projection = project_download(
                    pairs=args.project_pairs,
                    years=args.project_years,
                    files_per_minute=stats.files_per_minute,
                )
                _emit(
                    f"  pace: {stats.files_downloaded} files in {stats.seconds:.0f}s = "
                    f"{stats.files_per_minute:.1f} files/min with {args.workers} workers"
                )
                _emit(
                    f"  projection: {args.project_pairs} pairs x {args.project_years:g} years = "
                    f"{projection.files_total:,} files = {projection.hours_at_measured_pace:.0f} hours"
                )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
