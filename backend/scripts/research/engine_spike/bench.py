"""Speed benchmark for the compiled simulator core on years of M1 bars.

Uses HistData bid closes (fast to obtain) with a constant synthetic ask, because this measures
speed and not strategy quality. Prints bars per second for one configuration on one core, for a
parallel parameter sweep on all cores, and for the pure-Python reference.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.engine_spike import numba_core as core
from scripts.research.engine_spike.reference import simulate_reference

PIP = 0.0001


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")


def load_histdata(directory: Path, pair: str, years: range) -> pd.DataFrame:
    frames = []
    for year in years:
        path = directory / f"DAT_ASCII_{pair}_M1_{year}.csv"
        frame = pd.read_csv(path, sep=";", header=None, names=["ts", "open", "high", "low", "close", "v"], dtype={"ts": str})
        frames.append(frame)
    bars = pd.concat(frames, ignore_index=True)
    half = 0.1 * PIP
    out = pd.DataFrame()
    for side, sign in (("bid", -0.0), ("ask", 2 * half)):
        for name in ("open", "high", "low", "close"):
            out[f"{side}_{name}"] = bars[name].to_numpy() + sign
    return out


def arrays(frame: pd.DataFrame) -> list[np.ndarray]:
    names = ("bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close")
    return [np.ascontiguousarray(frame[name].to_numpy(dtype=np.float64)) for name in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv-dir", type=Path, default=Path("/tmp/claude-1000/hd"))
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--first-year", type=int, default=2016)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--configs", type=int, default=480)
    parser.add_argument("--reference-bars", type=int, default=60_000)
    args = parser.parse_args(argv)

    frame = load_histdata(args.csv_dir, args.pair, range(args.first_year, args.last_year + 1))
    data = arrays(frame)
    n = len(frame)
    cores = os.cpu_count() or 1
    _emit(f"{args.pair}: {n:,} M1 bars ({args.first_year}-{args.last_year}), {cores} cores")

    started = time.perf_counter()
    core.simulate(*[a[:5000] for a in data], 8, 21, 14, 1.5, 2.0, 0.2 * PIP, 0.45 * PIP, 60)
    _emit(f"first call (compile or load from cache): {time.perf_counter() - started:.2f}s")

    started = time.perf_counter()
    result = core.simulate(*data, 8, 21, 14, 1.5, 2.0, 0.2 * PIP, 0.45 * PIP, 60)
    elapsed = time.perf_counter() - started
    _emit(f"one configuration, one core: {elapsed:.3f}s = {n / elapsed / 1e6:.1f} million bars/s ({len(result[2]):,} trades)")

    rng = np.random.default_rng(7)
    fast = rng.integers(3, 30, args.configs).astype(np.int64)
    slow = (fast + rng.integers(5, 80, args.configs)).astype(np.int64)
    core.sweep(*[a[:5000] for a in data], fast[:2], slow[:2], 14, 1.5, 2.0, 0.2 * PIP, 0.45 * PIP, PIP, 60)
    started = time.perf_counter()
    core.sweep(*data, fast, slow, 14, 1.5, 2.0, 0.2 * PIP, 0.45 * PIP, PIP, 60)
    elapsed = time.perf_counter() - started
    bars_total = n * args.configs
    _emit(
        f"sweep of {args.configs} configurations on {cores} cores: {elapsed:.1f}s = "
        f"{bars_total / elapsed / 1e6:.0f} million bars/s ({args.configs / elapsed:.1f} configurations per second)"
    )

    small = frame.iloc[: args.reference_bars].reset_index(drop=True)
    started = time.perf_counter()
    simulate_reference(small, fast_span=8, slow_span=21, atr_period=14, stop_atr=1.5,
                       reward_risk=2.0, slippage=0.2 * PIP, warmup=60)
    elapsed = time.perf_counter() - started
    _emit(f"pure-Python reference: {args.reference_bars / elapsed / 1e3:.0f} thousand bars/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
