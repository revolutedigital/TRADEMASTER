"""Which pair-months of the M1 dataset are clean enough for the fast-strategy lab.

The pre-registration excludes a pair-month if the source has a hole in it (far fewer minutes than a
normal month) or if it fails validation: the majors against the Dukascopy hourly bars (at least 99%
of the hours identical and no crossed quote), the crosses by triangulation against the majors
(median error of the minute close of at most one pip). The decision uses only data quality and is
written to `docs/forex/fast-data-manifest.csv` before any strategy runs on the data.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_histdata_ticks import validate_against_hourly

HOLE_FRACTION = 0.85  # a month with fewer minutes than this share of the pair's median is a hole
MAX_TRIANGULATION_PIPS = 1.0
CROSSES = {"EURJPY": ("EURUSD", "USDJPY", "multiply"), "GBPJPY": ("GBPUSD", "USDJPY", "multiply"),
           "EURGBP": ("EURUSD", "GBPUSD", "divide")}
DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "docs" / "forex" / "fast-data-manifest.csv"


def load_m1(pair: str, directory: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(path) for path in sorted(directory.glob(f"{pair}_M1_*.parquet"))]
    if not parts:
        raise FileNotFoundError(f"no M1 file for {pair} in {directory}")
    frame = pd.concat(parts).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def month_key(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.tz_localize(None).strftime("%Y-%m"), index=index)


def flag_holes(minutes: pd.Series) -> pd.Series:
    """True for the months whose minute count falls well below the pair's usual month."""
    return minutes < HOLE_FRACTION * minutes.median()


def triangulation_error_pips(cross: pd.DataFrame, first: pd.DataFrame, second: pd.DataFrame,
                             operation: str, pip: float) -> float:
    """Median absolute gap, in pips, between a cross and the two majors it is made of."""
    columns = pd.concat(
        {name: 0.5 * (frame["bid_close"] + frame["ask_close"])
         for name, frame in (("cross", cross), ("a", first), ("b", second))},
        axis=1,
    ).dropna()
    composed = columns["a"] * columns["b"] if operation == "multiply" else columns["a"] / columns["b"]
    return float(((columns["cross"] - composed).abs() / pip).median()) if len(columns) else np.nan


def build_manifest(data_dir: Path, oracle_dirs: list[Path]) -> pd.DataFrame:
    m1 = {pair: load_m1(pair, data_dir) for pair in ALL_PAIRS}
    rows = []
    for pair, frame in m1.items():
        instrument = ALL_PAIRS[pair]
        oracle = None
        if pair not in CROSSES:
            oracle = pd.concat(
                [pd.read_parquet(d / f"{pair}_H1.parquet")[["bid_close", "ask_close"]] for d in oracle_dirs]
            ).sort_index()
        months = month_key(frame.index)
        minutes = frame.groupby(months).size()
        holes = flag_holes(minutes)
        for month, month_frame in frame.groupby(months):
            row = {"pair": pair, "month": month, "minutes": len(month_frame), "hole": bool(holes[month]),
                   "crossed_minutes": int((month_frame["ask_open"] < month_frame["bid_open"]).sum())}
            if oracle is not None:
                check = validate_against_hourly(month_frame, oracle, instrument.pip_size)
                row |= {"bid_match": round(check.bid_match, 4), "ask_match": round(check.ask_match, 4),
                        "quality": check.passes}
            else:
                first, second, operation = CROSSES[pair]
                gap = triangulation_error_pips(
                    month_frame, m1[first].loc[month_frame.index[0]:month_frame.index[-1]],
                    m1[second].loc[month_frame.index[0]:month_frame.index[-1]], operation, instrument.pip_size,
                )
                row |= {"triangulation_pips": round(gap, 3),
                        "quality": bool(gap <= MAX_TRIANGULATION_PIPS and row["crossed_minutes"] == 0)}
            row["included"] = bool(row["quality"] and not row["hole"])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    data = Path("data/raw/fx_m1_hd")
    manifest = build_manifest(data, [Path("data/raw/fx"), Path("data/raw/fx_confirm")])
    manifest.to_csv(DEFAULT_MANIFEST, index=False)
    excluded = manifest[~manifest["included"]]
    sys.stdout.write(f"{len(manifest)} pair-months, {len(excluded)} excluded\n")
    sys.stdout.write(excluded.groupby("month")["pair"].apply(", ".join).to_string() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
