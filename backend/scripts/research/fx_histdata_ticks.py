"""Research-only pipeline: HistData bid/ask ticks to validated M1 bid/ask bars.

HistData publishes free monthly tick files (`timestamp,bid,ask,volume`). Two facts make the raw
files easy to misuse, and both are handled here:

* The clock is New York time (UTC-5 in winter, UTC-4 in summer), but the daylight-saving switch
  follows the European dates (last Sunday of March and of October), not the American ones. Reading
  it with the American rule shifts everything by an hour during the weeks in which the two
  calendars disagree. The rule is verified per file against an independent oracle (the Dukascopy
  hourly bars already in the project) instead of being trusted.
* Some months have holes and, since 2026-06-28, the timestamps carry no milliseconds. Holes are
  reported so they can be filled from another source, and ordering falls back to a stable sort.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from scripts.research.fx_dataset import ALL_PAIRS, FxInstrument

SUMMER_OFFSET = pd.Timedelta(hours=4)
WINTER_OFFSET = pd.Timedelta(hours=5)
MATCH_TOLERANCE_PIPS = 0.05
MIN_HOURLY_MATCH = 0.99


def last_sunday(year: int, month: int) -> datetime:
    """The last Sunday of a month at 01:00 UTC, when the European switch happens."""
    first_of_next = datetime(year + (month == 12), month % 12 + 1, 1, 1, tzinfo=UTC)
    last_day = first_of_next - timedelta(days=1)
    return last_day - timedelta(days=(last_day.weekday() + 1) % 7)


def european_summer_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC instants between which the HistData clock is on its summer offset."""
    return pd.Timestamp(last_sunday(year, 3)), pd.Timestamp(last_sunday(year, 10))


def clock_to_utc(clock: pd.Series) -> pd.DatetimeIndex:
    """Convert HistData wall-clock timestamps to UTC using the European switch dates."""
    if clock.empty:
        return pd.DatetimeIndex([], tz="UTC")
    summer_guess = (clock + SUMMER_OFFSET).dt.tz_localize("UTC")
    in_summer = pd.Series(False, index=clock.index)
    for year in sorted(set(clock.dt.year) | set((clock + SUMMER_OFFSET).dt.year)):
        start, end = european_summer_bounds(year)
        in_summer |= (summer_guess >= start) & (summer_guess < end)
    winter = (clock + WINTER_OFFSET).dt.tz_localize("UTC")
    return pd.DatetimeIndex(summer_guess.where(in_summer, winter))


def parse_ticks(lines: pd.DataFrame) -> pd.DataFrame:
    """Turn raw `ts,bid,ask,volume` rows into a UTC-indexed frame ordered by time."""
    stamp = lines["ts"].astype(str)
    stamp = stamp.where(stamp.str.len() != 15, stamp + "000")  # no milliseconds since 2026-06-28
    clock = pd.to_datetime(stamp, format="%Y%m%d %H%M%S%f")
    frame = pd.DataFrame(
        {"bid": lines["bid"].to_numpy(dtype=float), "ask": lines["ask"].to_numpy(dtype=float)},
        index=clock_to_utc(clock.reset_index(drop=True)),
    )
    return frame.sort_index(kind="stable")


def read_tick_zip(path: Path) -> pd.DataFrame:
    """Read one monthly HistData tick archive."""
    with zipfile.ZipFile(path) as archive:
        name = next(entry for entry in archive.namelist() if entry.endswith(".csv"))
        with archive.open(name) as handle:
            raw = pd.read_csv(
                handle, header=None, names=["ts", "bid", "ask", "volume"], dtype={"ts": str}
            )
    return parse_ticks(raw)


def ticks_to_m1(ticks: pd.DataFrame, instrument: FxInstrument) -> pd.DataFrame:
    """Aggregate ticks into one-minute bid and ask candles with the spread in pips."""
    if ticks.empty:
        raise ValueError("no ticks")
    bars = pd.DataFrame(index=ticks["bid"].resample("1min").first().dropna().index)
    for side in ("bid", "ask"):
        grouped = ticks[side].resample("1min")
        bars[f"{side}_open"] = grouped.first()
        bars[f"{side}_high"] = grouped.max()
        bars[f"{side}_low"] = grouped.min()
        bars[f"{side}_close"] = grouped.last()
    bars = bars.dropna()
    bars["spread_open_pips"] = (bars["ask_open"] - bars["bid_open"]) / instrument.pip_size
    bars["spread_close_pips"] = (bars["ask_close"] - bars["bid_close"]) / instrument.pip_size
    bars.index.name = "timestamp"
    return bars


@dataclass(frozen=True)
class HourlyValidation:
    """How well derived M1 bars reproduce an independent hourly oracle."""

    hours_compared: int
    bid_match: float
    ask_match: float
    oracle_hours_without_data: float
    crossed_minutes: int

    @property
    def passes(self) -> bool:
        return (
            self.hours_compared > 0
            and min(self.bid_match, self.ask_match) >= MIN_HOURLY_MATCH
            and self.crossed_minutes == 0
        )


def validate_against_hourly(
    m1: pd.DataFrame, oracle: pd.DataFrame, pip_size: float
) -> HourlyValidation:
    """Compare the last M1 close of each hour with the oracle's hourly bid and ask close."""
    if m1.empty:
        return HourlyValidation(0, 0.0, 0.0, 1.0, 0)
    window = oracle.loc[m1.index[0].ceil("1h") : m1.index[-1].floor("1h") - pd.Timedelta(hours=1)]
    hourly = pd.DataFrame(
        {
            "bid": m1["bid_close"].resample("1h").last(),
            "ask": m1["ask_close"].resample("1h").last(),
        }
    ).dropna()
    joined = window.join(hourly, how="inner", rsuffix="_hd")
    if joined.empty:
        return HourlyValidation(0, 0.0, 0.0, 1.0, int((m1["ask_open"] < m1["bid_open"]).sum()))

    def matches(oracle_column: str, histdata_column: str) -> float:
        error = ((joined[histdata_column] - joined[oracle_column]).abs() / pip_size).to_numpy()
        return float((error <= MATCH_TOLERANCE_PIPS).mean())

    return HourlyValidation(
        hours_compared=len(joined),
        bid_match=matches("bid_close", "bid"),
        ask_match=matches("ask_close", "ask"),
        oracle_hours_without_data=float(1 - len(joined) / max(len(window), 1)),
        crossed_minutes=int((m1["ask_open"] < m1["bid_open"]).sum()),
    )


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    first = datetime.strptime(start, "%Y-%m")
    last = datetime.strptime(end, "%Y-%m")
    if last < first:
        raise ValueError("end must not be before start")
    months, year, month = [], first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def download_month(
    pair: str,
    year: int,
    month: int,
    directory: Path,
    *,
    retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download one month of ticks (once) and return the archive path."""
    from histdata import download_hist_data  # optional dependency, imported late
    from histdata.api import Platform, TimeFrame

    target = directory / f"DAT_ASCII_{pair}_T_{year}{month:02d}.zip"
    if target.exists():
        return target
    directory.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            download_hist_data(
                year=str(year),
                month=str(month),
                pair=pair.lower(),
                platform=Platform.GENERIC_ASCII,
                time_frame=TimeFrame.TICK_DATA,
                output_directory=str(directory),
                verbose=False,
            )
            if target.exists():
                return target
        except Exception as error:  # noqa: BLE001 - the package raises assorted errors
            last_error = error
        sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {pair} {year}-{month:02d}: {last_error!r}")


def _emit(line: str = "") -> None:
    sys.stdout.write(line + "\n")


def build_pair(
    pair: str,
    months: list[tuple[int, int]],
    workdir: Path,
    oracle: pd.DataFrame | None,
    *,
    workers: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    instrument = ALL_PAIRS[pair]
    zip_dir = workdir / "zips"

    def process(item: tuple[int, int]) -> tuple[pd.DataFrame, dict[str, object]]:
        year, month = item
        archive = download_month(pair, year, month, zip_dir)
        m1 = ticks_to_m1(read_tick_zip(archive), instrument)
        row: dict[str, object] = {"pair": pair, "month": f"{year}-{month:02d}", "minutes": len(m1)}
        if oracle is not None:
            check = validate_against_hourly(m1, oracle, instrument.pip_size)
            row |= {
                "hours_compared": check.hours_compared,
                "bid_match": round(check.bid_match, 4),
                "ask_match": round(check.ask_match, 4),
                "oracle_hours_without_data": round(check.oracle_hours_without_data, 4),
                "crossed_minutes": check.crossed_minutes,
                "passes": check.passes,
            }
        return m1, row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process, months))
    frames = [frame for frame, _ in results]
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined, [row for _, row in results]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", nargs="+", default=["EURUSD"], choices=list(ALL_PAIRS))
    parser.add_argument("--start", default="2021-09")
    parser.add_argument("--end", default="2026-08")
    parser.add_argument("--workdir", type=Path, default=Path("data/raw/histdata"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/fx_m1_hd"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--oracle-dir", type=Path, nargs="*", default=[Path("data/raw/fx"), Path("data/raw/fx_confirm")]
    )
    args = parser.parse_args(argv)

    months = month_range(args.start, args.end)
    args.out.mkdir(parents=True, exist_ok=True)
    failed = False
    for pair in args.pairs:
        oracle_parts = [
            pd.read_parquet(directory / f"{pair}_H1.parquet")[["bid_close", "ask_close"]]
            for directory in args.oracle_dir
            if (directory / f"{pair}_H1.parquet").exists()
        ]
        oracle = pd.concat(oracle_parts).sort_index() if oracle_parts else None
        started = time.monotonic()
        m1, rows = build_pair(pair, months, args.workdir, oracle, workers=args.workers)
        m1.to_parquet(args.out / f"{pair}_M1_{args.start.replace('-', '')}_{args.end.replace('-', '')}.parquet")
        pd.DataFrame(rows).to_csv(args.out / f"{pair}_validation.csv", index=False)
        bad = [row for row in rows if "passes" in row and not row["passes"]]
        failed = failed or bool(bad)
        _emit(
            f"{pair}: {len(m1):,} minutes, {len(months)} months in {time.monotonic() - started:.0f}s, "
            f"{len(bad)} months below the {MIN_HOURLY_MATCH:.0%} match bar"
        )
        for row in bad[:12]:
            _emit(f"  {row}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
