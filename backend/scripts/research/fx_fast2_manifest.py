"""Which pair-months of the pre-2019 dataset (S0 of round 2) are clean enough for the lab.

The criteria are fixed in docs/forex/fast2-preregistration.md ("Qualidade do dado de S0") and use only data
quality, never a strategy result. The decision is written to `docs/forex/fast2-data-manifest.csv`, and
committed before any strategy runs on S0.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast_manifest import (
    CROSSES,
    MAX_TRIANGULATION_PIPS,
    flag_holes,
    load_m1,
    month_key,
    triangulation_error_pips,
)
from scripts.research.fx_histdata_ticks import LEGACY_MIN_HOURLY_MATCH, LEGACY_TOLERANCE_PIPS, validate_against_hourly

DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "docs" / "forex" / "fast2-data-manifest.csv"
MAX_MEDIAN_SPREAD_PIPS = 3.0
ORACLE_START = "2016-09"  # the Dukascopy hourly bars the project holds start here
OPEN_EARLIEST, OPEN_LATEST = pd.Timedelta(hours=17), pd.Timedelta(hours=17, minutes=15)


def weekly_open_ok(frame: pd.DataFrame) -> bool:
    """True if in every Sunday of the frame the first bar, in New York time, opens between 17:00 and 17:15.

    A clock that is an hour off (the wrong daylight-saving calendar, a fixed offset) opens the week at 16:00
    or 18:00, so this needs no oracle. A frame without a Sunday says nothing and passes.
    """
    local = frame.index.tz_convert("America/New_York")
    sunday = local[(local.weekday == 6) & (local.hour >= 12)]
    if sunday.empty:
        return True
    first = pd.Series(sunday, index=sunday).groupby(sunday.date).min()
    offsets = first.map(lambda stamp: pd.Timedelta(hours=stamp.hour, minutes=stamp.minute, seconds=stamp.second))
    return bool(((offsets >= OPEN_EARLIEST) & (offsets <= OPEN_LATEST)).all())


def month_row(pair: str, month: str, month_frame: pd.DataFrame, hole: bool, oracle: pd.DataFrame | None) -> dict:
    instrument = ALL_PAIRS[pair]
    row = {"pair": pair, "month": month, "minutes": len(month_frame), "hole": hole,
           "crossed_minutes": int((month_frame["ask_open"] < month_frame["bid_open"]).sum()),
           "median_spread_pips": round(float(month_frame["spread_open_pips"].median()), 3),
           "clock_ok": weekly_open_ok(month_frame)}
    if oracle is not None and month >= ORACLE_START:
        check = validate_against_hourly(month_frame, oracle, instrument.pip_size, LEGACY_TOLERANCE_PIPS,
                                        LEGACY_MIN_HOURLY_MATCH)
        row |= {"bid_match": round(check.bid_match, 4), "ask_match": round(check.ask_match, 4), "oracle_ok": check.passes}
    else:
        row["oracle_ok"] = True  # no oracle for this month: the other criteria decide
    return row


def build_manifest(data_dir: Path, oracle_dirs: list[Path]) -> pd.DataFrame:
    m1 = {pair: load_m1(pair, data_dir) for pair in ALL_PAIRS}
    rows = []
    for pair, frame in m1.items():
        oracle = None
        if pair not in CROSSES:
            oracle = pd.concat(
                [pd.read_parquet(d / f"{pair}_H1.parquet")[["bid_close", "ask_close"]] for d in oracle_dirs]
            ).sort_index()
        months = month_key(frame.index)
        minutes = frame.groupby(months).size()
        holes = flag_holes(minutes)
        for month, month_frame in frame.groupby(months):
            row = month_row(pair, month, month_frame, bool(holes[month]), oracle)
            if pair in CROSSES:
                first, second, operation = CROSSES[pair]
                span = slice(month_frame.index[0], month_frame.index[-1])
                gap = triangulation_error_pips(month_frame, m1[first].loc[span], m1[second].loc[span], operation,
                                               ALL_PAIRS[pair].pip_size)
                row |= {"triangulation_pips": round(gap, 3)}
                row["oracle_ok"] = bool(gap <= MAX_TRIANGULATION_PIPS)
            row["included"] = bool(
                row["oracle_ok"] and not row["hole"] and row["crossed_minutes"] == 0 and row["clock_ok"]
                and row["median_spread_pips"] <= MAX_MEDIAN_SPREAD_PIPS
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    manifest = build_manifest(Path("data/raw/fx_m1_hd_pre"), [Path("data/raw/fx"), Path("data/raw/fx_confirm")])
    manifest.to_csv(DEFAULT_MANIFEST, index=False)
    excluded = manifest[~manifest["included"]]
    sys.stdout.write(f"{len(manifest)} pair-months, {len(excluded)} excluded\n")
    reasons = pd.DataFrame({
        "hole": manifest["hole"], "crossed": manifest["crossed_minutes"] > 0, "clock": ~manifest["clock_ok"],
        "spread": manifest["median_spread_pips"] > MAX_MEDIAN_SPREAD_PIPS, "oracle/triangulation": ~manifest["oracle_ok"],
    })
    sys.stdout.write(reasons.sum().to_string() + "\n")
    sys.stdout.write(excluded.groupby("pair").size().to_string() + "\n" if len(excluded) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
