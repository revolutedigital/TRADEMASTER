"""Materialize round-5 first-touch labels for 2019-2021 without opening 2022."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from app.fx.instruments import ConversionRates, Instrument
from app.fx.sim.costs import FUSION_ZERO, STRESS
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_dataset import tick_archives
from scripts.research.fx_fast4_events import EventCosts, build_feature_frame
from scripts.research.fx_fast4_materialize import (
    DEFAULT_OUTPUT_ROOT as FAST4_OUTPUT_ROOT,
    _commission_pips,
    _load_archives,
    _median_rates,
    _month_from_archive,
    _sha256,
)
from scripts.research.fx_fast5_events import build_entry_labels

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast5" / "entry_labels"
DEFAULT_FEATURE_PANEL = FAST4_OUTPUT_ROOT / "development"
ALLOWED_YEARS = (2019, 2020, 2021)


def expected_feature_index(panel: Path, pair: str, year: int) -> pd.DatetimeIndex:
    whole = panel / f"{pair}-features.parquet"
    partition = panel / f"{pair}-{year}-features.parquet"
    if partition.exists():
        frame = pd.read_parquet(partition, columns=["decision_index"])
    elif whole.exists():
        frame = pd.read_parquet(whole, columns=["decision_index"])
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    else:
        raise FileNotFoundError(f"round-4 features not found for {pair} {year}")
    return pd.DatetimeIndex(frame.index)


def context_archives(pair: str, year: int) -> tuple[Path, ...]:
    declared = tick_archives(pair, "development")
    months = [_month_from_archive(path) for path in declared]
    positions = [position for position, month in enumerate(months) if month.startswith(str(year))]
    if not positions:
        raise ValueError(f"year {year} is absent for {pair}")
    start = max(0, positions[0] - 1)
    stop = min(len(declared), positions[-1] + 2)
    return declared[start:stop]


def _compact(labels: pd.DataFrame) -> pd.DataFrame:
    compact = labels.copy()
    for column in compact.select_dtypes(include=["float64"]).columns:
        if column != "entry_index":
            compact[column] = compact[column].astype("float32")
    return compact


def materialize_partition(
    pair: str,
    year: int,
    output: Path,
    feature_panel: Path,
    rates: ConversionRates,
) -> dict[str, object]:
    if year not in ALLOWED_YEARS:
        raise ValueError(f"round-5 development labels are restricted to {ALLOWED_YEARS}")
    paths = context_archives(pair, year)
    ticks = _load_archives(paths)
    instrument = Instrument.from_symbol(pair)
    cleaned, features = build_feature_frame(ticks, instrument)
    median_price = float((cleaned["bid"] + cleaned["ask"]).median() * 0.5)
    costs = EventCosts(
        base_commission_pips=_commission_pips(pair, median_price, rates, FUSION_ZERO),
        stress_commission_pips=_commission_pips(pair, median_price, rates, STRESS),
    )
    labels = build_entry_labels(cleaned, features, instrument, costs)
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    labels = labels.loc[(labels.index >= start) & (labels.index < end)]
    expected = expected_feature_index(feature_panel, pair, year)
    if not labels.index.equals(expected):
        missing = len(expected.difference(labels.index))
        extra = len(labels.index.difference(expected))
        raise ValueError(
            f"round-5 labels misalign with round-4 features: missing={missing}, extra={extra}"
        )

    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{pair}-{year}-entry-labels.parquet"
    _compact(labels).to_parquet(path, compression="zstd", index=True)
    valid = labels["long_hit_base"].notna()
    report = {
        "pair": pair,
        "year": year,
        "context_first_month": _month_from_archive(paths[0]),
        "context_end_month": _month_from_archive(paths[-1]),
        "events": len(labels),
        "valid_events": int(valid.sum()),
        "long_hit_rate_base": float(labels.loc[valid, "long_hit_base"].mean()),
        "long_hit_rate_stress": float(labels.loc[valid, "long_hit_stress"].mean()),
        "short_hit_rate_base": float(labels.loc[valid, "short_hit_base"].mean()),
        "short_hit_rate_stress": float(labels.loc[valid, "short_hit_stress"].mean()),
        "base_commission_pips": costs.base_commission_pips,
        "stress_commission_pips": costs.stress_commission_pips,
        "sha256": _sha256(path),
    }
    del ticks, cleaned, features, labels
    gc.collect()
    return report


def materialize(
    pairs: tuple[str, ...],
    years: tuple[int, ...],
    output: Path = DEFAULT_OUTPUT,
    feature_panel: Path = DEFAULT_FEATURE_PANEL,
) -> dict[str, object]:
    if set(pairs) - set(ALL_PAIRS):
        raise ValueError("unknown pair")
    if set(years) - set(ALLOWED_YEARS):
        raise ValueError("2022 and protected years are closed")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"kind": "round5_first_touch_labels", "partitions": {}}
    partitions = manifest["partitions"]
    if not isinstance(partitions, dict):
        raise ValueError("invalid round-5 manifest")
    rates: ConversionRates | None = None
    for pair in pairs:
        for year in years:
            key = f"{pair}-{year}"
            existing = partitions.get(key)
            partition_path = output / f"{key}-entry-labels.parquet"
            if (
                isinstance(existing, dict)
                and partition_path.exists()
                and existing.get("sha256") == _sha256(partition_path)
            ):
                sys.stdout.write(f"{key}: verified existing partition\n")
                sys.stdout.flush()
                continue
            if rates is None:
                rates = _median_rates()
            report = materialize_partition(pair, year, output, feature_panel, rates)
            partitions[key] = report
            output.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            sys.stdout.write(
                f"{key}: {report['valid_events']:,}/{report['events']:,} valid, "
                f"long={report['long_hit_rate_base']:.3f}, short={report['short_hit_rate_base']:.3f}\n"
            )
            sys.stdout.flush()
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(partitions, sort_keys=True).encode()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=",".join(ALL_PAIRS))
    parser.add_argument("--years", default=",".join(str(year) for year in ALLOWED_YEARS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-panel", type=Path, default=DEFAULT_FEATURE_PANEL)
    arguments = parser.parse_args(argv)
    pairs = tuple(value.strip().upper() for value in arguments.pairs.split(",") if value.strip())
    years = tuple(int(value) for value in arguments.years.split(",") if value.strip())
    materialize(pairs, years, arguments.output, arguments.feature_panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
