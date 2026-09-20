"""Compare HistData one-minute bars with the Dukascopy hourly bid closes we already hold.

Research-only evidence for the data-source decision (docs/forex/data-m1-spike.md). HistData
publishes free M1 bars, one file per pair and year, in seconds instead of the hours the
Dukascopy datafeed needs. This script downloads them, tests two readings of the timestamp
(New York local time with daylight saving versus fixed EST) and reports how far the hourly
close differs from the Dukascopy bid.

Needs the optional `histdata` package (`pip install histdata`). Bid only: the source has no ask.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

NEW_YORK = "America/New_York"


def load_year(pair: str, year: int, workdir: Path) -> pd.DataFrame:
    """Download (once) and read one year of HistData M1 bars for a pair."""
    from histdata import download_hist_data  # imported late: optional dependency
    from histdata.api import Platform, TimeFrame

    name = f"DAT_ASCII_{pair.upper()}_M1_{year}"
    csv_path = workdir / f"{name}.csv"
    if not csv_path.exists():
        download_hist_data(
            year=str(year),
            month=None,
            pair=pair.lower(),
            platform=Platform.GENERIC_ASCII,
            time_frame=TimeFrame.ONE_MINUTE,
            output_directory=str(workdir),
            verbose=False,
        )
        with zipfile.ZipFile(workdir / f"{name}.zip") as archive:
            archive.extractall(workdir)
    frame = pd.read_csv(
        csv_path, sep=";", header=None, names=["ts", "open", "high", "low", "close", "volume"],
        dtype={"ts": str},
    )
    frame["ts"] = pd.to_datetime(frame["ts"], format="%Y%m%d %H%M%S")
    return frame


def to_utc(frame: pd.DataFrame, reading: str) -> pd.DataFrame:
    """Index the bars in UTC under one reading of the provider's clock."""
    if reading == "new_york_dst":
        utc = frame["ts"].dt.tz_localize(NEW_YORK, ambiguous="NaT", nonexistent="NaT")
        utc = utc.dt.tz_convert("UTC")
    elif reading == "fixed_est":
        utc = (frame["ts"] + pd.Timedelta(hours=5)).dt.tz_localize("UTC")
    else:
        raise ValueError("reading must be new_york_dst or fixed_est")
    return frame.assign(utc=utc).dropna(subset=["utc"]).set_index("utc")


def compare_with_dukascopy(minutes: pd.DataFrame, hourly: pd.DataFrame, pip: float) -> dict[str, float]:
    """Difference in pips between the last M1 close of each hour and the hourly bid close."""
    hd_hourly = minutes["close"].resample("1h").last().dropna().rename("histdata")
    joined = hourly[["bid_close"]].join(hd_hourly, how="inner")
    error = ((joined["histdata"] - joined["bid_close"]).abs() / pip).astype(float)
    return {
        "hours": float(len(joined)),
        "median_pips": float(error.median()),
        "share_over_0.05_pip": float((error > 0.05).mean()),
        "share_over_0.5_pip": float((error > 0.5).mean()),
        "share_over_2_pips": float((error > 2).mean()),
        "share_over_10_pips": float((error > 10).mean()),
        "max_pips": float(error.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--first-year", type=int, default=2016)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--workdir", type=Path, default=Path("data/raw/histdata"))
    parser.add_argument("--hourly", type=Path, nargs="+", required=True, help="Dukascopy H1 parquet files")
    args = parser.parse_args(argv)

    args.workdir.mkdir(parents=True, exist_ok=True)
    frames = [load_year(args.pair, year, args.workdir) for year in range(args.first_year, args.last_year + 1)]
    bars = pd.concat(frames)
    hourly = pd.concat([pd.read_parquet(path) for path in args.hourly]).sort_index()
    pip = 0.01 if args.pair.upper().endswith("JPY") else 0.0001

    sys.stdout.write(f"{args.pair}: {len(bars):,} minutes, {args.first_year}-{args.last_year}\n")
    for reading in ("new_york_dst", "fixed_est"):
        result = compare_with_dukascopy(to_utc(bars, reading), hourly, pip)
        sys.stdout.write(f"  {reading}: {result}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
