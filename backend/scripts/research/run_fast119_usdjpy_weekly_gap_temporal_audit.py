"""Fast119: temporal audit for the Fast118 USDJPY weekly-gap clue.

Fast118 found a small-sample USDJPY continuation clue after large weekly gaps.
This script freezes that rule and checks whether it existed in the local
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

from app.fx.instruments import ConversionRates, Instrument
from app.fx.sim.costs import FUSION_ZERO, STRESS, CostScenario, commission_round_trip_pips
from scripts.research.run_fast118_fx_weekly_gap_scout import (
    AUDIT_END,
    DATA_START,
    Management,
    Policy,
    find_weekly_gaps,
    lower_confidence_bound,
    positive_year_share,
    side_for_gap,
    simulate_scenario,
    widen_quotes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast119" / "usdjpy-weekly-gap-temporal-audit"
)
PRE_START = pd.Timestamp("2011-01-01T00:00:00Z")
PRE_END = pd.Timestamp("2018-12-01T00:00:00Z")
PAIR = "USDJPY"
FROZEN_POLICY = Policy(
    identifier="continuation_gap10_d15_h12_fixed_full_gap",
    direction="continuation",
    minimum_gap_pips=10.0,
    entry_delay_minutes=15,
    max_hold_minutes=720,
    management=Management(
        "fixed_full_gap",
        stop_gap_multiple=1.0,
        target_gap_fraction=1.0,
        activation_gap_fraction=None,
        trail_gap_fraction=None,
    ),
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
    """Load USDJPY executable M1 bars for an explicit temporal slice."""
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


def commission_pips(price: float, rates: ConversionRates, scenario: CostScenario) -> float:
    return commission_round_trip_pips(
        Instrument.from_symbol(PAIR),
        price,
        rates,
        base_currency_per_lot_per_side=scenario.commission_base_per_lot_per_side,
    )


def evaluate_slice(
    label: str,
    paths: tuple[Path, ...],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the frozen rule on one temporal slice."""
    frame = load_bars(paths, start=start, end=end)
    stress_frame = widen_quotes(frame, STRESS.spread_multiplier)
    median_price = float(np.median(0.5 * (frame["bid_close"] + frame["ask_close"])))
    rates = ConversionRates({"USDJPY": median_price})
    base_commission = commission_pips(median_price, rates, FUSION_ZERO)
    stress_commission = commission_pips(median_price, rates, STRESS)
    gaps = find_weekly_gaps(frame, PAIR)
    rows: list[dict[str, Any]] = []
    for gap in gaps:
        side = side_for_gap(gap, FROZEN_POLICY)
        base = simulate_scenario(
            frame,
            gap,
            FROZEN_POLICY,
            side=side,
            commission=base_commission,
            slippage_per_side_pips=FUSION_ZERO.slippage_pips,
        )
        stress = simulate_scenario(
            stress_frame,
            gap,
            FROZEN_POLICY,
            side=side,
            commission=stress_commission,
            slippage_per_side_pips=STRESS.slippage_pips,
        )
        if base is None or stress is None:
            continue
        rows.append(
            {
                "slice": label,
                "pair": PAIR,
                "policy_id": FROZEN_POLICY.identifier,
                "gap_reopen_timestamp": gap.reopen_timestamp,
                "previous_timestamp": gap.previous_timestamp,
                "side": "LONG" if side == 1 else "SHORT",
                "gap_pips": gap.gap_pips,
                "absolute_gap_pips": gap.absolute_gap_pips,
                "raw_base_pips": base.raw_pips,
                "net_base_pips": base.net_pips,
                "exit_base_timestamp": base.exit_timestamp,
                "exit_base_reason": base.exit_reason,
                "raw_stress_pips": stress.raw_pips,
                "net_stress_pips": stress.net_pips,
                "exit_stress_timestamp": stress.exit_timestamp,
                "exit_stress_reason": stress.exit_reason,
                "base_commission_pips": base_commission,
                "stress_commission_pips": stress_commission,
            }
        )
    trades = pd.DataFrame(rows)
    summary = slice_summary(label, frame, gaps, trades)
    return rows, summary


def slice_summary(
    label: str,
    frame: pd.DataFrame,
    gaps: tuple[Any, ...],
    trades: pd.DataFrame,
) -> dict[str, Any]:
    if trades.empty:
        return {
            "slice": label,
            "bars": int(len(frame)),
            "raw_bars": int(frame.attrs.get("raw_bars", len(frame))),
            "dropped_crossed_quote_bars": int(frame.attrs.get("dropped_crossed_quote_bars", 0)),
            "weekly_gaps": int(len(gaps)),
            "trades": 0,
        }
    share, positive_years, years = positive_year_share(trades, "net_stress_pips")
    by_year = (
        trades.assign(year=trades["gap_reopen_timestamp"].dt.year)
        .groupby("year", sort=True)["net_stress_pips"]
        .mean()
        .to_dict()
    )
    return {
        "slice": label,
        "bars": int(len(frame)),
        "raw_bars": int(frame.attrs.get("raw_bars", len(frame))),
        "dropped_crossed_quote_bars": int(frame.attrs.get("dropped_crossed_quote_bars", 0)),
        "first_bar": frame.index.min().isoformat(),
        "last_bar": frame.index.max().isoformat(),
        "weekly_gaps": int(len(gaps)),
        "trades": int(len(trades)),
        "mean_base_pips": float(trades["net_base_pips"].mean()),
        "mean_stress_pips": float(trades["net_stress_pips"].mean()),
        "median_stress_pips": float(trades["net_stress_pips"].median()),
        "total_stress_pips": float(trades["net_stress_pips"].sum()),
        "lower_stress_pips": lower_confidence_bound(trades["net_stress_pips"], 0.05),
        "positive_stress_year_share": share,
        "positive_stress_years": positive_years,
        "years": years,
        "stress_mean_by_year": {str(year): float(value) for year, value in by_year.items()},
    }


def temporal_status(summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    by_label = {str(row["slice"]): row for row in summaries}
    pre = by_label.get("pre_2011_2018", {})
    current = by_label.get("current_2019_2026", {})
    if float(pre.get("mean_stress_pips", float("nan"))) <= 0:
        return (
            "FAST119_USDJPY_WEEKLY_GAP_TEMPORAL_INSTABILITY_REJECTED",
            reasons + ["pre_2011_2018_mean_stress_not_positive", "regime_instability"],
        )
    if float(current.get("mean_stress_pips", float("nan"))) <= 0:
        return (
            "FAST119_USDJPY_WEEKLY_GAP_CURRENT_SAMPLE_REJECTED",
            reasons + ["current_2019_2026_mean_stress_not_positive"],
        )
    return (
        "FAST119_USDJPY_WEEKLY_GAP_HISTORICAL_CLUE_REQUIRES_PROSPECTIVE_PREREGISTRATION",
        reasons + ["historical_slices_positive_but_no_prospective_evidence"],
    )


def run(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    pre_paths: tuple[Path, ...] | None = None,
    current_paths: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    """Run the focused temporal audit and write artifacts."""
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
            f"{label}: {summary['weekly_gaps']} weekly gaps -> {summary['trades']} trades, "
            f"stress_mean={summary.get('mean_stress_pips')}\n"
        )
        sys.stdout.flush()

    trades = pd.DataFrame(rows).sort_values(["slice", "gap_reopen_timestamp"])
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
            "fast119_is_temporal_audit_not_strategy_approval",
            "frozen_rule_was_identified_by_fast118_historical_scout",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast119_usdjpy_weekly_gap_temporal_audit",
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
            "trades": {
                "path": str(trades_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(trades_path),
            },
            "slice_summary": {
                "path": str(summary_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(summary_path),
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
        "report_path": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": sha256_file(report_path),
    }
    sys.stdout.write(json.dumps(console, indent=2, sort_keys=True) + "\n")
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(plain_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    run(output_root=arguments.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
