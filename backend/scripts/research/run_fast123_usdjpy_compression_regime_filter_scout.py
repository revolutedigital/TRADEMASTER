"""Fast123: causal regime-filter scout for the Fast122 USDJPY compression clue.

Fast122 showed the frozen USDJPY H1 compression-breakout clue is positive in
2019-2026 but weak/unstable before 2019. This script tests whether simple,
entry-known filters selected only on the old pre-2016 slice can explain that
instability and validate out of sample. It is research-only and never sends
orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json
from scripts.research.run_fast121_fx_compression_breakout_scout import lower_confidence_bound

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast122"
    / "usdjpy-compression-breakout-temporal-audit"
    / "frozen-policy-trades.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast123"
    / "usdjpy-compression-regime-filter-scout"
)

TRAIN_END = pd.Timestamp("2016-01-01T00:00:00Z")
PRE_VALIDATION_END = pd.Timestamp("2018-12-01T00:00:00Z")
CURRENT_START = pd.Timestamp("2019-01-01T00:00:00Z")
MIN_TRAIN_TRADES = 120
MIN_PRE_VALIDATION_TRADES = 80
MIN_CURRENT_TRADES = 200


@dataclass(frozen=True)
class FilterSpec:
    """One entry-known filter candidate."""

    filter_id: str
    conditions: tuple[tuple[str, str], ...]


def load_trade_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "entry_timestamp",
        "side",
        "atr_pips",
        "risk_pips",
        "breakout_distance_pips",
        "compression_value_pips",
        "compression_threshold_pips",
        "net_base_pips",
        "net_stress_pips",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"source trade rows missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    frame["entry_year"] = frame["entry_timestamp"].dt.year.astype(int)
    frame["entry_month"] = frame["entry_timestamp"].dt.month.astype(int)
    frame["entry_weekday"] = frame["entry_timestamp"].dt.weekday.astype(int)
    frame["entry_hour"] = frame["entry_timestamp"].dt.hour.astype(int)
    frame["session_bucket"] = frame["entry_hour"].map(session_bucket)
    frame["month_bucket"] = frame["entry_month"].map(month_bucket)
    frame["compression_ratio"] = frame["compression_value_pips"] / frame["compression_threshold_pips"]
    frame["breakout_over_atr"] = frame["breakout_distance_pips"] / frame["atr_pips"]
    frame["stress_r"] = frame["net_stress_pips"] / frame["risk_pips"]
    return frame


def session_bucket(hour: int) -> str:
    if 0 <= hour <= 6:
        return "asia"
    if 7 <= hour <= 12:
        return "london"
    if 13 <= hour <= 18:
        return "new_york"
    return "late_us"


def month_bucket(month: int) -> str:
    if month <= 3:
        return "q1"
    if month <= 6:
        return "q2"
    if month <= 9:
        return "q3"
    return "q4"


def add_numeric_buckets(frame: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Add causal quantile buckets using thresholds learned on the train slice only."""
    result = frame.copy()
    for column in ("atr_pips", "risk_pips", "compression_ratio", "breakout_over_atr"):
        values = train[column].replace([np.inf, -np.inf], np.nan).dropna()
        low = float(values.quantile(1 / 3))
        high = float(values.quantile(2 / 3))
        result[f"{column}_bucket"] = pd.cut(
            result[column],
            bins=[-np.inf, low, high, np.inf],
            labels=("low", "mid", "high"),
            include_lowest=True,
        ).astype(str)
    return result


def build_filter_specs(frame: pd.DataFrame) -> tuple[FilterSpec, ...]:
    """Build one- and two-dimensional entry-known filters."""
    dimensions = (
        "side",
        "session_bucket",
        "month_bucket",
        "entry_weekday",
        "atr_pips_bucket",
        "compression_ratio_bucket",
        "breakout_over_atr_bucket",
    )
    specs: list[FilterSpec] = [FilterSpec("all", ())]

    def make_filter(conditions: dict[str, str]) -> FilterSpec:
        items = tuple(sorted((key, str(value)) for key, value in conditions.items()))
        return FilterSpec("&".join(f"{key}={value}" for key, value in items), items)

    for dimension in dimensions:
        for value in sorted(frame[dimension].dropna().astype(str).unique()):
            specs.append(make_filter({dimension: value}))
    focused_pairs = (
        ("side", "session_bucket"),
        ("side", "atr_pips_bucket"),
        ("side", "compression_ratio_bucket"),
        ("session_bucket", "atr_pips_bucket"),
        ("session_bucket", "breakout_over_atr_bucket"),
        ("month_bucket", "atr_pips_bucket"),
    )
    for left, right in focused_pairs:
        for left_value in sorted(frame[left].dropna().astype(str).unique()):
            for right_value in sorted(frame[right].dropna().astype(str).unique()):
                specs.append(make_filter({left: left_value, right: right_value}))
    unique = {spec.filter_id: spec for spec in specs}
    return tuple(unique[key] for key in sorted(unique))


def apply_filter(frame: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, value in spec.conditions:
        mask &= frame[column].astype(str).eq(value)
    return frame.loc[mask]


def summarize_sample(sample: pd.DataFrame) -> dict[str, Any]:
    if sample.empty:
        return {
            "trade_count": 0,
            "mean_base_pips": None,
            "mean_stress_pips": None,
            "lower_stress_pips": None,
            "positive_year_share": None,
            "positive_years": 0,
            "years": 0,
            "mean_stress_r": None,
        }
    by_year = sample.groupby("entry_year", sort=True)["net_stress_pips"].mean()
    return {
        "trade_count": int(len(sample)),
        "mean_base_pips": float(sample["net_base_pips"].mean()),
        "mean_stress_pips": float(sample["net_stress_pips"].mean()),
        "lower_stress_pips": lower_confidence_bound(sample["net_stress_pips"], 0.05),
        "positive_year_share": float((by_year > 0).mean()) if len(by_year) else None,
        "positive_years": int((by_year > 0).sum()),
        "years": int(len(by_year)),
        "mean_stress_r": float(sample["stress_r"].mean()),
    }


def evaluate_filter(frame: pd.DataFrame, spec: FilterSpec) -> dict[str, Any]:
    train = apply_filter(frame.loc[frame["entry_timestamp"] < TRAIN_END], spec)
    pre_validation = apply_filter(
        frame.loc[
            (frame["entry_timestamp"] >= TRAIN_END)
            & (frame["entry_timestamp"] < PRE_VALIDATION_END)
        ],
        spec,
    )
    current = apply_filter(frame.loc[frame["entry_timestamp"] >= CURRENT_START], spec)
    return {
        "filter_id": spec.filter_id,
        "conditions": list(spec.conditions),
        "train": summarize_sample(train),
        "pre_validation": summarize_sample(pre_validation),
        "current": summarize_sample(current),
    }


def filter_passes(row: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    failed: list[str] = []
    train = row["train"]
    pre_validation = row["pre_validation"]
    current = row["current"]
    if train["trade_count"] < MIN_TRAIN_TRADES:
        failed.append("train_trades_below_min")
    if pre_validation["trade_count"] < MIN_PRE_VALIDATION_TRADES:
        failed.append("pre_validation_trades_below_min")
    if current["trade_count"] < MIN_CURRENT_TRADES:
        failed.append("current_trades_below_min")
    for label, summary in (
        ("train", train),
        ("pre_validation", pre_validation),
        ("current", current),
    ):
        if summary["mean_stress_pips"] is None or summary["mean_stress_pips"] <= 0:
            failed.append(f"{label}_mean_stress_not_positive")
    weak = not failed
    strict_failed = list(failed)
    if current["lower_stress_pips"] is None or current["lower_stress_pips"] <= 0:
        strict_failed.append("current_lower_stress_not_positive")
    if pre_validation["lower_stress_pips"] is None or pre_validation["lower_stress_pips"] <= 0:
        strict_failed.append("pre_validation_lower_stress_not_positive")
    if (
        current["positive_year_share"] is None
        or current["positive_year_share"] < 2 / 3
        or pre_validation["positive_year_share"] is None
        or pre_validation["positive_year_share"] < 2 / 3
    ):
        strict_failed.append("year_stability_below_two_thirds")
    return weak, weak and not strict_failed, strict_failed


def fast123_status(strict_count: int, weak_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if strict_count:
        return (
            "FAST123_USDJPY_COMPRESSION_REGIME_STRICT_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [f"strict_filters:{strict_count}", "fresh_preregistered_prospective_observation_required"],
        )
    if weak_count:
        return (
            "FAST123_USDJPY_COMPRESSION_REGIME_WEAK_ONLY_REJECTED",
            reasons + [f"weak_filters:{weak_count}", "strict_stability_failed"],
        )
    return (
        "FAST123_USDJPY_COMPRESSION_REGIME_FILTER_FAILED",
        reasons + ["no_entry_known_filter_passed_train_and_validation"],
    )


def run(
    *,
    source: Path = DEFAULT_SOURCE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    trades = load_trade_rows(source)
    train = trades.loc[trades["entry_timestamp"] < TRAIN_END]
    trades = add_numeric_buckets(trades, train)
    specs = build_filter_specs(trades)
    evaluations = []
    weak_rows = []
    strict_rows = []
    for spec in specs:
        row = evaluate_filter(trades, spec)
        weak, strict, failed = filter_passes(row)
        row["weak_pass"] = weak
        row["strict_pass"] = strict
        row["failed"] = failed
        evaluations.append(row)
        if weak:
            weak_rows.append(row)
        if strict:
            strict_rows.append(row)
    evaluations = sorted(
        evaluations,
        key=lambda row: (
            row["strict_pass"],
            row["weak_pass"],
            row["current"]["mean_stress_pips"] or float("-inf"),
            row["pre_validation"]["mean_stress_pips"] or float("-inf"),
            row["train"]["mean_stress_pips"] or float("-inf"),
        ),
        reverse=True,
    )
    status, status_reasons = fast123_status(len(strict_rows), len(weak_rows))
    evaluations_path = output_root / "filter-evaluations.json"
    weak_path = output_root / "weak-filters.json"
    strict_path = output_root / "strict-filters.json"
    write_json(evaluations_path, evaluations)
    write_json(weak_path, weak_rows)
    write_json(strict_path, strict_rows)
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast123_is_regime_filter_scout_not_strategy_approval",
            "filters_are_entry_known_but_selected_on_historical_slices",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast123_usdjpy_compression_regime_filter_scout",
        "status": status,
        "status_reasons": status_reasons,
        "source_path": str(source.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(source),
        "filter_count": len(specs),
        "weak_filter_count": len(weak_rows),
        "strict_filter_count": len(strict_rows),
        "top_filters": evaluations[:10],
        "artifacts": {
            "filter_evaluations": {
                "path": str(evaluations_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(evaluations_path),
            },
            "weak_filters": {
                "path": str(weak_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(weak_path),
            },
            "strict_filters": {
                "path": str(strict_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(strict_path),
            },
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path.relative_to(REPO_ROOT))})
    console = {
        "status": status,
        "status_reasons": status_reasons,
        "approved_strategy": report["approved_strategy"],
        "orders_sent": report["orders_sent"],
        "filter_count": len(specs),
        "weak_filter_count": len(weak_rows),
        "strict_filter_count": len(strict_rows),
        "report_path": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": sha256_file(report_path),
    }
    sys.stdout.write(json.dumps(console, indent=2, sort_keys=True) + "\n")
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    run(source=arguments.source, output_root=arguments.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
