"""Materialize exact managed-payoff labels for round 9 without opening Q2 or holdouts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.fx.instruments import ConversionRates, Instrument
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast4_events import build_feature_frame
from scripts.research.fx_fast4_materialize import _load_archives, _median_rates, _sha256
from scripts.research.fx_fast5_events import simulate_trailing_batch
from scripts.research.fx_fast5_materialize import (
    DEFAULT_FEATURE_PANEL,
    DEFAULT_OUTPUT as ENTRY_LABEL_ROOT,
    context_archives,
    expected_feature_index,
)
from scripts.research.fx_fast5_policy import _costs

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "lab_fast9" / "managed_payoffs"
ALLOWED_YEARS = (2019, 2020, 2021)
TRAINING_STRIDE = 4
Q1_END = pd.Timestamp("2021-04-01", tz="UTC")
ACTIVATION_R = 0.1
ACTIVATION_SECONDS = 600
TRAIL_DISTANCE_R = 0.5
MAX_HOLD_SECONDS = 6 * 60 * 60


def sampled_positions(
    index: pd.DatetimeIndex, pair: str, year: int
) -> np.ndarray:
    """Return the preregistered deterministic subset for a partition."""
    if pair not in ALL_PAIRS:
        raise ValueError(f"unknown pair: {pair}")
    if year not in ALLOWED_YEARS:
        raise ValueError(f"round-9 labels are restricted to {ALLOWED_YEARS}")
    eligible = np.flatnonzero(index < Q1_END) if year == 2021 else np.arange(len(index))
    if year == 2021:
        return eligible
    offset = list(ALL_PAIRS).index(pair) % TRAINING_STRIDE
    return eligible[offset::TRAINING_STRIDE]


def _candidate_frame(
    features: pd.DataFrame, entry_labels: pd.DataFrame, side: int
) -> pd.DataFrame:
    candidates = features[["mid_range_pips_256"]].join(
        entry_labels[["entry_index", "risk_pips"]], how="inner", validate="one_to_one"
    )
    candidates["decision_index"] = candidates["entry_index"].astype(np.int64) - 1
    candidates["side"] = side
    return candidates.drop(columns="entry_index")


def _compact(frame: pd.DataFrame) -> pd.DataFrame:
    compact = frame.copy()
    protected = {name for name in compact.columns if name.endswith("_index")}
    for column in compact.select_dtypes(include=["float64"]).columns:
        if column not in protected:
            compact[column] = compact[column].astype("float32")
    return compact


def materialize_partition(
    pair: str,
    year: int,
    output: Path,
    feature_panel: Path,
    entry_label_root: Path,
    rates: ConversionRates,
) -> dict[str, object]:
    expected = expected_feature_index(feature_panel, pair, year)
    positions = sampled_positions(expected, pair, year)
    selected_index = expected[positions]
    feature_path = feature_panel / f"{pair}-{year}-features.parquet"
    if feature_path.exists():
        stored_features = pd.read_parquet(feature_path).loc[selected_index]
    else:
        whole = pd.read_parquet(feature_panel / f"{pair}-features.parquet")
        stored_features = whole.loc[selected_index]
    entry_labels = pd.read_parquet(
        entry_label_root / f"{pair}-{year}-entry-labels.parquet"
    ).loc[selected_index]
    finite = np.isfinite(
        stored_features["mid_range_pips_256"].to_numpy(dtype=np.float64)
    )
    finite &= entry_labels[["entry_index", "risk_pips"]].notna().all(axis=1).to_numpy()
    stored_features = stored_features.loc[finite]
    entry_labels = entry_labels.loc[finite]
    valid_index = stored_features.index

    ticks = _load_archives(context_archives(pair, year))
    instrument = Instrument.from_symbol(pair)
    cleaned, rebuilt_features = build_feature_frame(ticks, instrument)
    rebuilt_year = rebuilt_features.loc[expected]
    if not rebuilt_year.index.equals(expected):
        raise ValueError(f"rebuilt features do not align for {pair} {year}")
    if not np.array_equal(
        entry_labels["entry_index"].to_numpy(dtype=np.int64) - 1,
        rebuilt_year.loc[valid_index, "decision_index"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(f"entry labels do not align with yearly ticks for {pair} {year}")

    costs = _costs(pair, cleaned, rates)
    output_frame = stored_features[["mid_range_pips_256"]].copy()
    output_frame["decision_index"] = entry_labels["entry_index"].astype(np.int64) - 1
    output_frame["risk_pips"] = entry_labels["risk_pips"]
    for side, side_name in ((1, "long"), (-1, "short")):
        candidates = _candidate_frame(stored_features, entry_labels, side)
        for scenario in ("base", "stress"):
            outcomes = simulate_trailing_batch(
                cleaned,
                candidates,
                instrument,
                costs,
                scenario=scenario,
                activation_r=ACTIVATION_R,
                activation_seconds=ACTIVATION_SECONDS,
                trail_distance_r=TRAIL_DISTANCE_R,
                max_hold_seconds=MAX_HOLD_SECONDS,
            )
            for name in outcomes.columns:
                output_frame[f"{side_name}_{name}_{scenario}"] = outcomes[name].to_numpy()

    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{pair}-{year}-managed-payoffs.parquet"
    _compact(output_frame).to_parquet(path, compression="zstd", index=True)
    valid = output_frame["long_result_r_base"].notna()
    report = {
        "pair": pair,
        "year": year,
        "source_events": len(expected),
        "sampled_events": len(selected_index),
        "eligible_events": len(output_frame),
        "valid_events": int(valid.sum()),
        "sampling": "all_q1" if year == 2021 else f"stride_{TRAINING_STRIDE}",
        "long_mean_base_r": float(output_frame.loc[valid, "long_result_r_base"].mean()),
        "short_mean_base_r": float(output_frame.loc[valid, "short_result_r_base"].mean()),
        "sha256": _sha256(path),
    }
    del ticks, cleaned, rebuilt_features, rebuilt_year, stored_features, entry_labels, output_frame
    gc.collect()
    return report


def materialize(
    pairs: tuple[str, ...],
    years: tuple[int, ...],
    output: Path = DEFAULT_OUTPUT,
    feature_panel: Path = DEFAULT_FEATURE_PANEL,
    entry_label_root: Path = ENTRY_LABEL_ROOT,
) -> dict[str, object]:
    if set(pairs) - set(ALL_PAIRS):
        raise ValueError("unknown pair")
    if set(years) - set(ALLOWED_YEARS):
        raise ValueError("Q2, H2, 2022, and protected years are closed")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "kind": "round9_exact_managed_payoffs",
            "protected_samples_opened": False,
            "q2_opened": False,
            "manager": {
                "activation_r": ACTIVATION_R,
                "activation_seconds": ACTIVATION_SECONDS,
                "trail_distance_r": TRAIL_DISTANCE_R,
                "max_hold_seconds": MAX_HOLD_SECONDS,
            },
            "partitions": {},
        }
    partitions = manifest["partitions"]
    if not isinstance(partitions, dict):
        raise ValueError("invalid round-9 manifest")
    rates: ConversionRates | None = None
    for pair in pairs:
        for year in years:
            key = f"{pair}-{year}"
            path = output / f"{key}-managed-payoffs.parquet"
            existing = partitions.get(key)
            if isinstance(existing, dict) and path.exists() and existing.get("sha256") == _sha256(path):
                sys.stdout.write(f"{key}: verified existing partition\n")
                sys.stdout.flush()
                continue
            if rates is None:
                rates = _median_rates()
            report = materialize_partition(
                pair, year, output, feature_panel, entry_label_root, rates
            )
            partitions[key] = report
            output.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            sys.stdout.write(
                f"{key}: {report['valid_events']:,}/{report['eligible_events']:,} valid\n"
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
    parser.add_argument("--entry-label-root", type=Path, default=ENTRY_LABEL_ROOT)
    arguments = parser.parse_args(argv)
    pairs = tuple(value.strip().upper() for value in arguments.pairs.split(",") if value.strip())
    years = tuple(int(value) for value in arguments.years.split(",") if value.strip())
    materialize(
        pairs,
        years,
        arguments.output,
        arguments.feature_panel,
        arguments.entry_label_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
