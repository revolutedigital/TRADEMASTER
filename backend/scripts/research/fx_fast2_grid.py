"""The configuration grid of round 2 of the fast-strategy lab (docs/forex/fast2-preregistration.md).

Round 1 tested ten configurations with fixed parameters. Round 2 calibrates every family over a grid of
its parameters, and the statistics pay for the number of combinations (max-t over the whole grid). The
grid is generated here, written to `docs/forex/fast2-grid.json`, and the registry stores the SHA-256 of
that file: a combination that is not in it cannot be run, and the file cannot change without an amendment.

    python -m scripts.research.fx_fast2_grid --write docs/forex/fast2-grid.json

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.fx.sessions import LONDON, NEW_YORK
from app.fx.strategies import fixing_flow, pairs_spread, session_breakout, spike_fade
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast_lab import SYNTHETIC, USD_PAIRS, Configuration

DEFAULT_GRID = Path(__file__).resolve().parents[3] / "docs" / "forex" / "fast2-grid.json"

F1_REWARD_RISK = (1.0, 1.5, 2.5)
F1_WIDTH = ((0.3, 1.2), (0.3, 0.8), (0.6, 2.0), (0.05, 50.0))  # (min, max) range width in daily ATRs; the last is no filter
F1_SIGNAL_END = ("09:30", "11:00")
F1_ZONES = {  # zone: (label, range start, range end, exits, minimum range bars)
    "london": ("a", "00:00", "08:00", ("12:30", "16:30"), 24),
    "new_york": ("b", "03:00", "08:00", ("12:00", "16:00"), 15),
}
F2_PRE = {"entry": ("14:30", "15:00", "15:30"), "exit": ("15:55", "16:00"), "stop_atr": (2.0, 3.0, 5.0)}
F2_POST = {"entry": ("16:05", "16:15"), "exit": ("16:45", "17:00", "17:30"), "stop_atr": (2.0, 3.0, 5.0)}
F3_THRESHOLD, F3_STOP, F3_RETRACE, F3_HOLD = (3.0, 4.0, 6.0, 8.0), (0.5, 1.0), (0.5, 0.75), (12, 24)
F5_ENTRY_Z, F5_STOP_Z, F5_HOLD = (1.5, 2.0, 2.5, 3.0), (3.5, 4.5), (60, 120, 240)
F5_SERIES = {"a": "EURGBP", "b": SYNTHETIC}


def grid_points() -> list[dict]:
    """Every combination, as JSON-safe dicts: key, family, timeframe, pairs and the parameter arguments."""
    points: list[dict] = []
    every = list(ALL_PAIRS)
    for zone, (label, start, end, exits, min_bars) in F1_ZONES.items():
        for reward, (low, high), signal_end, exit_time in itertools.product(F1_REWARD_RISK, F1_WIDTH, F1_SIGNAL_END, exits):
            width = "none" if high >= 50 else f"{low}-{high}"
            points.append({
                "key": f"F1{label}.rr{reward}.w{width}.se{signal_end}.x{exit_time}", "family": f"F1{label}",
                "seconds": 900, "pairs": every,
                "args": {"zone": zone, "range_start": start, "range_end": end, "signal_start": end,
                         "signal_end": signal_end, "exit_time": exit_time, "min_range_bars": min_bars,
                         "min_atr_multiple": low, "max_atr_multiple": high, "reward_risk": reward},
            })
    for label, spec, buy_dollar in (("a", F2_PRE, True), ("b", F2_POST, False)):
        for entry, exit_time, stop_atr in itertools.product(spec["entry"], spec["exit"], spec["stop_atr"]):
            points.append({
                "key": f"F2{label}.in{entry}.out{exit_time}.atr{stop_atr}", "family": f"F2{label}", "seconds": 300,
                "pairs": list(USD_PAIRS),
                "args": {"buy_dollar": buy_dollar, "entry_time": entry, "exit_time": exit_time, "stop_atr": stop_atr},
            })
    for threshold, stop, retrace, hold in itertools.product(F3_THRESHOLD, F3_STOP, F3_RETRACE, F3_HOLD):
        points.append({
            "key": f"F3.th{threshold}.stop{stop}.ret{retrace}.hold{hold}", "family": "F3", "seconds": 300, "pairs": every,
            "args": {"threshold": threshold, "stop_range": stop, "retrace": retrace, "max_bars": hold},
        })
    for label, pair in F5_SERIES.items():
        for entry_z, stop_z, hold in itertools.product(F5_ENTRY_Z, F5_STOP_Z, F5_HOLD):
            points.append({
                "key": f"F5{label}.z{entry_z}.stop{stop_z}.hold{hold}", "family": f"F5{label}", "seconds": 3600,
                "pairs": [pair], "args": {"entry_z": entry_z, "stop_z": stop_z, "max_bars": hold},
            })
    return points


def write_grid(path: Path = DEFAULT_GRID) -> None:
    path.write_text(json.dumps(grid_points(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


def load_grid(path: Path = DEFAULT_GRID) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _params_builder(point: dict) -> Callable[[str], np.ndarray]:
    args, family = point["args"], point["family"]
    if family.startswith("F1"):
        zone = LONDON if args["zone"] == "london" else NEW_YORK
        array = session_breakout.session_breakout_params(
            zone=zone, range_start=args["range_start"], range_end=args["range_end"],
            signal_start=args["signal_start"], signal_end=args["signal_end"], exit_time=args["exit_time"],
            min_range_bars=args["min_range_bars"], min_atr_multiple=args["min_atr_multiple"],
            max_atr_multiple=args["max_atr_multiple"], reward_risk=args["reward_risk"],
        )
        return lambda pair: array
    if family.startswith("F2"):
        return lambda pair: fixing_flow.fixing_flow_params(
            pair_side=fixing_flow.pair_side_for_dollar(pair, buy_dollar=args["buy_dollar"]),
            entry_time=args["entry_time"], exit_time=args["exit_time"], stop_atr=args["stop_atr"],
        )
    if family == "F3":
        array = spike_fade.spike_fade_params(threshold=args["threshold"], stop_range=args["stop_range"],
                                             retrace=args["retrace"], max_bars=args["max_bars"])
        return lambda pair: array
    array = pairs_spread.pairs_spread_params(entry_z=args["entry_z"], stop_z=args["stop_z"], max_bars=args["max_bars"])
    return lambda pair: array


def build_configurations(points: list[dict]) -> dict[str, Configuration]:
    """The lab's `Configuration` for every grid point."""
    machinery = {
        "F1": (session_breakout.session_breakout_step, session_breakout.session_breakout_init,
               session_breakout.SESSION_BREAKOUT_STATE_SIZE),
        "F2": (fixing_flow.fixing_flow_step, fixing_flow.fixing_flow_init, fixing_flow.FIXING_FLOW_STATE_SIZE),
        "F3": (spike_fade.spike_fade_step, spike_fade.spike_fade_init, spike_fade.SPIKE_FADE_STATE_SIZE),
        "F5": (pairs_spread.pairs_spread_step, pairs_spread.pairs_spread_init, pairs_spread.PAIRS_SPREAD_STATE_SIZE),
    }
    configs = {}
    for point in points:
        step, init, size = machinery[point["family"][:2]]
        configs[point["key"]] = Configuration(point["key"], point["seconds"], tuple(point["pairs"]), step, init, size,
                                              _params_builder(point))
    return configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", type=Path, help="write the grid JSON to this path")
    arguments = parser.parse_args(argv)
    if arguments.write:
        write_grid(arguments.write)
    points = grid_points()
    by_family: dict[str, int] = {}
    for point in points:
        by_family[point["family"]] = by_family.get(point["family"], 0) + 1
    sys.stdout.write(f"{len(points)} combinations: {by_family}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
