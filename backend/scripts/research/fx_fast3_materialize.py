"""Materialize causal round-3 features and executable outcomes, one pair per partition."""

from __future__ import annotations

import argparse
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
from scripts.research.fx_fast3_events import EventCosts, add_cross_pair_context, build_feature_frame, build_outcome_wide

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median_rates(sample: str, unlock_protected: bool) -> ConversionRates:
    mids: dict[str, float] = {}
    for pair in MAJORS:
        matrix = load_pair_sample(pair, sample, unlock_protected=unlock_protected)
        mids[pair] = float(
            np.median(0.5 * (matrix[:, fx.BID_CLOSE] + matrix[:, fx.ASK_CLOSE]))
        )
    return ConversionRates(mids)


def _commission_pips(
    pair: str, price: float, rates: ConversionRates, base_currency_per_lot_per_side: float
) -> float:
    return commission_round_trip_pips(
        Instrument.from_symbol(pair),
        price,
        rates,
        base_currency_per_lot_per_side=base_currency_per_lot_per_side,
    )


def _compact(frame: pd.DataFrame) -> pd.DataFrame:
    compact = frame.copy()
    for column in compact.select_dtypes(include=["float64"]).columns:
        compact[column] = compact[column].astype("float32")
    return compact


def materialize(sample: str, output: Path, *, unlock_protected: bool = False) -> dict[str, object]:
    """Write pair partitions and return their auditable manifest."""
    output.mkdir(parents=True, exist_ok=True)
    rates = _median_rates(sample, unlock_protected)
    feature_paths: dict[str, Path] = {}
    outcome_paths: dict[str, Path] = {}
    report: dict[str, object] = {"sample": sample, "pairs": {}}
    for pair in ALL_PAIRS:
        instrument = Instrument.from_symbol(pair)
        minutes = load_pair_sample(pair, sample, unlock_protected=unlock_protected)
        if not len(minutes):
            raise ValueError(f"{pair} has no rows in sample {sample}")
        median_price = float(
            np.median(0.5 * (minutes[:, fx.BID_CLOSE] + minutes[:, fx.ASK_CLOSE]))
        )
        costs = EventCosts(
            base_commission_pips=_commission_pips(
                pair, median_price, rates, FUSION_ZERO.commission_base_per_lot_per_side
            ),
            stress_commission_pips=_commission_pips(
                pair, median_price, rates, STRESS.commission_base_per_lot_per_side
            ),
        )
        m5, features = build_feature_frame(minutes, instrument)
        outcomes = build_outcome_wide(m5, features, instrument, costs)
        feature_path = output / f"{pair}-features.parquet"
        outcome_path = output / f"{pair}-outcomes.parquet"
        _compact(features).to_parquet(feature_path, compression="zstd", index=True)
        _compact(outcomes).to_parquet(outcome_path, compression="zstd", index=True)
        feature_paths[pair], outcome_paths[pair] = feature_path, outcome_path
        report["pairs"][pair] = {  # type: ignore[index]
            "minutes": len(minutes),
            "m5_rows": len(m5),
            "first_decision": features.index[0].isoformat(),
            "last_decision": features.index[-1].isoformat(),
            "base_commission_pips": costs.base_commission_pips,
            "stress_commission_pips": costs.stress_commission_pips,
        }
        sys.stdout.write(f"{pair}: {len(minutes):,} M1 -> {len(m5):,} M5\n")
        sys.stdout.flush()

    frames = {
        pair: pd.read_parquet(path, columns=["return_3"])
        for pair, path in feature_paths.items()
    }
    enriched = add_cross_pair_context(frames)
    for pair, path in feature_paths.items():
        features = pd.read_parquet(path)
        features["usd_factor_return_3"] = enriched[pair]["usd_factor_return_3"]
        features["common_factor_return_3"] = enriched[pair]["common_factor_return_3"]
        _compact(features).to_parquet(path, compression="zstd", index=True)

    for pair in ALL_PAIRS:
        pair_report = report["pairs"][pair]  # type: ignore[index]
        pair_report["features_sha256"] = _sha256(feature_paths[pair])
        pair_report["outcomes_sha256"] = _sha256(outcome_paths[pair])
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", choices=("development", "holdout", "confirmation"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unlock-protected", action="store_true")
    arguments = parser.parse_args(argv)
    output = arguments.output or DEFAULT_OUTPUT_ROOT / arguments.sample
    materialize(arguments.sample, output, unlock_protected=arguments.unlock_protected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
