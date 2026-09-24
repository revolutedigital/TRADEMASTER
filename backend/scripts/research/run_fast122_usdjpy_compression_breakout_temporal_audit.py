"""Fast122: temporal audit for the Fast121 USDJPY H1 compression-breakout clue.

Fast121 found a USDJPY H1 compression-breakout line that was positive in both
selection and 2024-2026 audit, but failed the strict lower-bound gate. This
script freezes that rule and checks whether it also existed in the local
pre-2019 M1 history. It is research-only and never sends orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.fx.instruments import ConversionRates
from app.fx.sim.costs import FUSION_ZERO, STRESS
from scripts.research.run_fast121_fx_compression_breakout_scout import (
    AUDIT_END,
    DATA_START,
    Management,
    PairData,
    Policy,
    commission_pips,
    evaluate_pair_policy,
    lower_confidence_bound,
    mid_ohlc,
    positive_year_share,
    sha256_file,
    widen_quotes,
)
from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast122" / "usdjpy-compression-breakout-temporal-audit"
)
PAIR = "USDJPY"
PRE_START = pd.Timestamp("2011-01-01T00:00:00Z")
PRE_END = pd.Timestamp("2018-12-01T00:00:00Z")
FROZEN_POLICY = Policy(
    identifier="tf1h_ch20_cq0.4_atr1.5_hold720_fixed_3r",
    timeframe="1h",
    channel_bars=20,
    compression_quantile=0.40,
    stop_atr_multiple=1.5,
    max_hold_minutes=720,
    management=Management("fixed_3r", target_r=3.0, activation_r=None, trail_r=None),
)


def default_pre_paths() -> tuple[Path, ...]:
    return (REPO_ROOT / "backend" / "data" / "raw" / "fx_m1_hd_pre" / "USDJPY_M1_201101_201811.parquet",)


def default_current_paths() -> tuple[Path, ...]:
    current = REPO_ROOT / "backend" / "data" / "raw" / "fx_m1_hd"
    return (
        current / "USDJPY_M1_201901_202108.parquet",
        current / "USDJPY_M1_202109_202608.parquet",
    )


def load_bars(paths: tuple[Path, ...], *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing bar files: {[str(path) for path in missing]}")
    frame = pd.concat(pd.read_parquet(path, columns=columns) for path in paths).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.index.tz is None:
        raise ValueError("USDJPY bars must be timezone-aware")
    raw_bars = len(frame)
    frame = frame.loc[(frame.index >= start) & (frame.index < end)].copy()
    crossed = (
        frame[["ask_open", "ask_high", "ask_low", "ask_close"]].to_numpy()
        < frame[["bid_open", "bid_high", "bid_low", "bid_close"]].to_numpy()
    ).any(axis=1)
    if crossed.any():
        frame = frame.loc[~crossed].copy()
    frame.attrs["raw_bars"] = raw_bars
    frame.attrs["dropped_crossed_quote_bars"] = int(crossed.sum())
    return frame


def evaluate_slice(
    label: str,
    paths: tuple[Path, ...],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = load_bars(paths, start=start, end=end)
    median_price = float(np.median(mid_ohlc(frame)["close"].to_numpy()))
    rates = ConversionRates({"USDJPY": median_price})
    pair_data = PairData(
        bars=frame,
        stress_bars=widen_quotes(frame, STRESS.spread_multiplier),
        median_price=median_price,
        base_commission_pips=commission_pips(PAIR, median_price, rates, FUSION_ZERO),
        stress_commission_pips=commission_pips(PAIR, median_price, rates, STRESS),
        quality={},
    )
    rows = evaluate_pair_policy(PAIR, pair_data, FROZEN_POLICY)
    for row in rows:
        row["slice"] = label
    trades = pd.DataFrame(rows)
    summary = slice_summary(label, frame, trades)
    return rows, summary


def slice_summary(label: str, frame: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    base = {
        "slice": label,
        "bars": int(len(frame)),
        "raw_bars": int(frame.attrs.get("raw_bars", len(frame))),
        "dropped_crossed_quote_bars": int(frame.attrs.get("dropped_crossed_quote_bars", 0)),
        "first_bar": frame.index.min().isoformat() if len(frame) else None,
        "last_bar": frame.index.max().isoformat() if len(frame) else None,
        "trades": int(len(trades)),
    }
    if trades.empty:
        return base
    share, positive_years, years = positive_year_share(trades, "net_stress_pips")
    by_year = (
        trades.assign(year=trades["entry_timestamp"].dt.year)
        .groupby("year", sort=True)["net_stress_pips"]
        .mean()
        .to_dict()
    )
    return base | {
        "mean_base_pips": float(trades["net_base_pips"].mean()),
        "mean_stress_pips": float(trades["net_stress_pips"].mean()),
        "median_stress_pips": float(trades["net_stress_pips"].median()),
        "total_stress_pips": float(trades["net_stress_pips"].sum()),
        "lower_stress_pips": lower_confidence_bound(trades["net_stress_pips"], 0.05),
        "positive_stress_year_share": share,
        "positive_stress_years": positive_years,
        "years": years,
        "mean_risk_pips": float(trades["risk_pips"].mean()),
        "stress_mean_by_year": {str(year): float(value) for year, value in by_year.items()},
    }


def temporal_status(summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    by_label = {str(row["slice"]): row for row in summaries}
    pre = by_label.get("pre_2011_2018", {})
    current = by_label.get("current_2019_2026", {})
    pre_mean = float(pre.get("mean_stress_pips", float("nan")))
    pre_lower = float(pre.get("lower_stress_pips", float("nan")))
    pre_year_share = float(pre.get("positive_stress_year_share", float("nan")))
    current_mean = float(current.get("mean_stress_pips", float("nan")))
    current_lower = float(current.get("lower_stress_pips", float("nan")))
    current_year_share = float(current.get("positive_stress_year_share", float("nan")))
    if current_mean <= 0:
        return (
            "FAST122_USDJPY_COMPRESSION_BREAKOUT_CURRENT_SAMPLE_REJECTED",
            reasons + ["current_2019_2026_mean_stress_not_positive"],
        )
    if current_lower <= 0:
        return (
            "FAST122_USDJPY_COMPRESSION_BREAKOUT_CURRENT_LOWER_BOUND_REJECTED",
            reasons + ["current_2019_2026_lower_stress_not_positive"],
        )
    if pre_mean <= 0:
        return (
            "FAST122_USDJPY_COMPRESSION_BREAKOUT_PRE_SAMPLE_REJECTED",
            reasons + ["pre_2011_2018_mean_stress_not_positive"],
        )
    if pre_lower <= 0 or pre_year_share < 2 / 3 or current_year_share < 2 / 3:
        return (
            "FAST122_USDJPY_COMPRESSION_BREAKOUT_WEAK_TEMPORAL_STABILITY_REJECTED",
            reasons + ["pre_2011_2018_lower_bound_or_year_stability_failed", "regime_instability"],
        )
    return (
        "FAST122_USDJPY_COMPRESSION_BREAKOUT_HISTORICAL_CLUE_REQUIRES_PROSPECTIVE_PREREGISTRATION",
        reasons + ["historical_slices_positive_but_no_prospective_evidence"],
    )


def run(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    pre_paths: tuple[Path, ...] | None = None,
    current_paths: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    pre = pre_paths if pre_paths is not None else default_pre_paths()
    current = current_paths if current_paths is not None else default_current_paths()
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for label, paths, start, end in (
        ("pre_2011_2018", pre, PRE_START, PRE_END),
        ("current_2019_2026", current, DATA_START, AUDIT_END),
    ):
        slice_rows, summary = evaluate_slice(label, paths, start=start, end=end)
        rows.extend(slice_rows)
        summaries.append(summary)
        sys.stdout.write(
            f"{label}: {summary['trades']} trades, stress_mean={summary.get('mean_stress_pips')}, "
            f"stress_lower={summary.get('lower_stress_pips')}\n"
        )
        sys.stdout.flush()

    trades = pd.DataFrame(rows).sort_values(["slice", "entry_timestamp"])
    trades_path = output_root / "frozen-policy-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)
    summary_path = output_root / "slice-summary.json"
    write_json(summary_path, summaries)
    status, status_reasons = temporal_status(summaries)
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast122_is_temporal_audit_not_strategy_approval",
            "frozen_rule_was_identified_by_fast121_historical_scout",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast122_usdjpy_compression_breakout_temporal_audit",
        "status": status,
        "status_reasons": status_reasons,
        "pair": PAIR,
        "frozen_policy": {
            **asdict(FROZEN_POLICY),
            "management": asdict(FROZEN_POLICY.management),
        },
        "pre_paths": [str(path.relative_to(REPO_ROOT)) for path in pre],
        "current_paths": [str(path.relative_to(REPO_ROOT)) for path in current],
        "slices": summaries,
        "costs": {
            "base": asdict(FUSION_ZERO),
            "stress": asdict(STRESS),
            "slippage_model_used": "fixed per side from scenario; stress spread multiplier applied",
        },
        "artifacts": {
            "trades": {"path": str(trades_path.relative_to(REPO_ROOT)), "sha256": sha256_file(trades_path)},
            "slice_summary": {"path": str(summary_path.relative_to(REPO_ROOT)), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path.relative_to(REPO_ROOT))})
    console = {
        "status": status,
        "status_reasons": status_reasons,
        "approved_strategy": report["approved_strategy"],
        "orders_sent": report["orders_sent"],
        "report_path": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": sha256_file(report_path),
    }
    sys.stdout.write(json.dumps(console, indent=2, sort_keys=True) + "\n")
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    run(output_root=arguments.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
