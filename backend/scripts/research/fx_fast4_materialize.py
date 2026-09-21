"""Materialize round-4 quote features and executable outcomes, one pair at a time."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.fx import strategy as fx
from app.fx.instruments import ConversionRates, Instrument
from app.fx.sim.costs import FUSION_ZERO, STRESS, commission_round_trip_pips
from scripts.research.fx_dataset import ALL_PAIRS, MAJORS
from scripts.research.fx_fast3_dataset import load_pair_sample
from scripts.research.fx_fast4_dataset import tick_archives
from scripts.research.fx_fast4_events import EventCosts, build_feature_frame, build_outcome_wide
from scripts.research.fx_histdata_ticks import read_tick_zip

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median_rates() -> ConversionRates:
    """Reuse the already guarded development bars for commission currency conversion."""
    mids: dict[str, float] = {}
    for pair in MAJORS:
        matrix = load_pair_sample(pair, "development")
        mids[pair] = float(np.median(0.5 * (matrix[:, fx.BID_CLOSE] + matrix[:, fx.ASK_CLOSE])))
    return ConversionRates(mids)


def _commission_pips(pair: str, price: float, rates: ConversionRates, scenario=FUSION_ZERO) -> float:
    return commission_round_trip_pips(
        Instrument.from_symbol(pair),
        price,
        rates,
        base_currency_per_lot_per_side=scenario.commission_base_per_lot_per_side,
    )


def _compact(frame: pd.DataFrame) -> pd.DataFrame:
    compact = frame.copy()
    for column in compact.select_dtypes(include=["float64"]).columns:
        if column.endswith(("entry_index", "exit_index")):
            continue
        compact[column] = compact[column].astype("float32")
    return compact


def _month_from_archive(path: Path) -> str:
    compact = path.stem.rsplit("_", 1)[-1]
    if len(compact) != 6 or not compact.isdigit():
        raise ValueError(f"cannot parse month from {path.name}")
    return f"{compact[:4]}-{compact[4:]}"


def _select_archives(paths: tuple[Path, ...], first_month: str | None, end_month: str | None) -> tuple[Path, ...]:
    selected = tuple(
        path
        for path in paths
        if (first_month is None or _month_from_archive(path) >= first_month)
        and (end_month is None or _month_from_archive(path) < end_month)
    )
    if not selected:
        raise ValueError("the requested month slice contains no declared archive")
    return selected


def _load_archives(paths: tuple[Path, ...]) -> pd.DataFrame:
    parts = [read_tick_zip(path, rule="europe") for path in paths]
    ticks = pd.concat(parts).sort_index(kind="stable")
    del parts
    gc.collect()
    return ticks


def materialize_pair(
    pair: str,
    sample: str,
    output: Path,
    rates: ConversionRates,
    *,
    unlock_protected: bool = False,
    first_month: str | None = None,
    end_month: str | None = None,
) -> dict[str, object]:
    """Materialize one pair and return its auditable manifest entry."""
    declared = tick_archives(pair, sample, unlock_protected=unlock_protected)
    selected = _select_archives(declared, first_month, end_month)
    ticks = _load_archives(selected)
    instrument = Instrument.from_symbol(pair)
    cleaned, features = build_feature_frame(ticks, instrument)
    median_price = float(np.median(0.5 * (cleaned["bid"] + cleaned["ask"])))
    costs = EventCosts(
        base_commission_pips=_commission_pips(pair, median_price, rates, FUSION_ZERO),
        stress_commission_pips=_commission_pips(pair, median_price, rates, STRESS),
    )
    outcomes = build_outcome_wide(cleaned, features, instrument, costs)
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / f"{pair}-features.parquet"
    outcome_path = output / f"{pair}-outcomes.parquet"
    _compact(features).to_parquet(feature_path, compression="zstd", index=True)
    _compact(outcomes).to_parquet(outcome_path, compression="zstd", index=True)
    valid = outcomes.filter(like="terminal_r_base").notna().sum().to_dict()
    report: dict[str, object] = {
        "pair": pair,
        "archives": len(selected),
        "first_month": _month_from_archive(selected[0]),
        "end_month": _month_from_archive(selected[-1]),
        "raw_ticks": len(ticks),
        "clean_ticks": len(cleaned),
        "events": len(features),
        "valid_outcomes": {name: int(count) for name, count in valid.items()},
        "first_decision": features.index[0].isoformat() if len(features) else None,
        "last_decision": features.index[-1].isoformat() if len(features) else None,
        "base_commission_pips": costs.base_commission_pips,
        "stress_commission_pips": costs.stress_commission_pips,
        "features_sha256": _sha256(feature_path),
        "outcomes_sha256": _sha256(outcome_path),
    }
    del ticks, cleaned, features, outcomes
    gc.collect()
    return report


def materialize(
    sample: str,
    output: Path,
    *,
    pairs: tuple[str, ...] = ALL_PAIRS,
    unlock_protected: bool = False,
    first_month: str | None = None,
    end_month: str | None = None,
) -> dict[str, object]:
    if sample != "development" and (first_month is not None or end_month is not None):
        raise ValueError("protected samples cannot be sliced")
    unknown = set(pairs) - set(ALL_PAIRS)
    if unknown:
        raise ValueError(f"pairs outside the declared universe: {sorted(unknown)}")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report.get("sample") != sample or not isinstance(report.get("pairs"), dict):
            raise ValueError(f"existing manifest {manifest_path} belongs to another run")
    else:
        report = {"sample": sample, "pairs": {}}
    rates = _median_rates()
    for pair in pairs:
        pair_report = materialize_pair(
            pair,
            sample,
            output,
            rates,
            unlock_protected=unlock_protected,
            first_month=first_month,
            end_month=end_month,
        )
        report["pairs"][pair] = pair_report  # type: ignore[index]
        sys.stdout.write(
            f"{pair}: {pair_report['clean_ticks']:,} quotes -> {pair_report['events']:,} events\n"
        )
        sys.stdout.flush()
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("development", "holdout", "confirmation"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", default=",".join(ALL_PAIRS))
    parser.add_argument("--first-month")
    parser.add_argument("--end-month")
    parser.add_argument("--unlock-protected", action="store_true")
    arguments = parser.parse_args(argv)
    pairs = tuple(value.strip().upper() for value in arguments.pairs.split(",") if value.strip())
    output = arguments.output or DEFAULT_OUTPUT_ROOT / arguments.sample
    materialize(
        arguments.sample,
        output,
        pairs=pairs,
        unlock_protected=arguments.unlock_protected,
        first_month=arguments.first_month,
        end_month=arguments.end_month,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
