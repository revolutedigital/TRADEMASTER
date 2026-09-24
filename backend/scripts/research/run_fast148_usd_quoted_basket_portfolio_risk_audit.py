"""Fast148: portfolio risk audit for the Fast147 USD-quoted basket candidate.

Fast146 found a stronger basket candidate than NZDUSD alone. Fast147 froze it
for prospective observation. Fast148 asks whether the historical basket is
operable from a portfolio-risk perspective: drawdown, concurrent positions,
bad-year clustering, pair concentration and exit reasons.

This is historical research only. It writes offline artifacts and never touches
the trading engine, credentials, ledger, Demo, Testnet or LIVE accounts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    lower_confidence_bound,
    positive_year_share,
)

DEFAULT_PRE_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast137"
    / "fx-daily-donchian-fade-temporal-reverse"
    / "temporal-reverse-trades.parquet"
)
DEFAULT_CURRENT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast133"
    / "fx-daily-donchian-fade-global-risk"
    / "daily-donchian-fade-global-risk-trades.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast148"
    / "usd-quoted-basket-portfolio-risk-audit"
)

PRE_SAMPLE_LABEL = "pre_2011_2018_selection"
CURRENT_SAMPLE_LABEL = "current_2019_2026_full_grid"
GROUP_ID = "usd_quoted_majors"
TARGET_PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "NZDUSD")
ENTRY_POLICY_ID = "d1_donchian_fade_ch100_hold20"
RISK_POLICY_ID = "stop2_fixed_1r"
SIDE_FILTER = "SHORT"
MANAGEMENT = "fixed_1r"
TARGET_POLICY_ID = f"basket:{GROUP_ID}:{ENTRY_POLICY_ID}:{RISK_POLICY_ID}:short_only"
MAX_ACCEPTABLE_HISTORICAL_DRAWDOWN_PIPS = 1500.0
MAX_ACCEPTABLE_CONCURRENT_TRADES = 8
SAFETY_FLAGS = {
    "research_only": True,
    "order_submission_allowed": False,
    "execution_authorization": "none",
    "orders_sent": False,
    "committed": False,
    "dry_run": True,
    "approved_strategy": False,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-input-path", type=Path, default=DEFAULT_PRE_INPUT_PATH)
    parser.add_argument("--current-input-path", type=Path, default=DEFAULT_CURRENT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(
        pre_input_path=arguments.pre_input_path,
        current_input_path=arguments.current_input_path,
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def filtered_sample(
    frame: pd.DataFrame,
    *,
    sample_column: str | None,
    sample_value: str | None,
) -> pd.DataFrame:
    sample = frame.copy()
    if sample_column and sample_value:
        sample = sample.loc[sample[sample_column].eq(sample_value)].copy()
    for column in ("signal_timestamp", "entry_timestamp", "exit_base_timestamp", "exit_stress_timestamp"):
        sample[column] = pd.to_datetime(sample[column], utc=True)
    return sample.loc[
        sample["pair"].isin(TARGET_PAIRS)
        & sample["entry_policy_id"].eq(ENTRY_POLICY_ID)
        & sample["risk_policy_id"].eq(RISK_POLICY_ID)
        & sample["side"].eq(SIDE_FILTER)
        & sample["management"].eq(MANAGEMENT)
    ].sort_values(["entry_timestamp", "pair"], ignore_index=True)


def equity_curve(sample: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame(
            columns=["exit_stress_timestamp", "pair", "net_stress_pips", "equity_pips", "drawdown_pips"]
        )
    curve = sample.sort_values(["exit_stress_timestamp", "entry_timestamp", "pair"]).copy()
    curve["equity_pips"] = curve["net_stress_pips"].astype("float64").cumsum()
    curve["drawdown_pips"] = curve["equity_pips"] - curve["equity_pips"].cummax()
    return curve[
        ["exit_stress_timestamp", "pair", "net_stress_pips", "equity_pips", "drawdown_pips"]
    ].reset_index(drop=True)


def max_concurrent_trades(sample: pd.DataFrame) -> int:
    events: list[tuple[pd.Timestamp, int]] = []
    for row in sample.itertuples(index=False):
        events.append((pd.Timestamp(row.entry_timestamp), 1))
        events.append((pd.Timestamp(row.exit_stress_timestamp), -1))
    concurrent = 0
    maximum = 0
    for _timestamp, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        concurrent += delta
        maximum = max(maximum, concurrent)
    return maximum


def longest_losing_streak(sample: pd.DataFrame) -> int:
    streak = 0
    maximum = 0
    for value in sample.sort_values(["exit_stress_timestamp", "pair"])["net_stress_pips"]:
        if float(value) < 0:
            streak += 1
            maximum = max(maximum, streak)
        else:
            streak = 0
    return maximum


def yearly_summary(sample: pd.DataFrame) -> list[dict[str, Any]]:
    yearly = (
        sample.assign(year=sample["entry_timestamp"].dt.year)
        .groupby("year", sort=True)["net_stress_pips"]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    return [
        {
            "year": int(row["year"]),
            "trade_count": int(row["size"]),
            "total_stress_pips": float(row["sum"]),
            "mean_stress_pips": float(row["mean"]),
        }
        for row in yearly.to_dict("records")
    ]


def has_two_consecutive_negative_years(yearly: list[dict[str, Any]]) -> bool:
    last_negative = False
    for row in yearly:
        negative = row["mean_stress_pips"] < 0
        if negative and last_negative:
            return True
        last_negative = negative
    return False


def grouped_summary(sample: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    grouped = (
        sample.groupby(group_column, sort=True)["net_stress_pips"]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    return [
        {
            str(group_column): str(row[group_column]),
            "trade_count": int(row["size"]),
            "total_stress_pips": float(row["sum"]),
            "mean_stress_pips": float(row["mean"]),
        }
        for row in grouped.to_dict("records")
    ]


def summarize_sample(sample: pd.DataFrame) -> dict[str, Any]:
    if sample.empty:
        return {
            "trade_count": 0,
            "total_stress_pips": 0.0,
            "mean_stress_pips": None,
            "lower_stress_pips": None,
            "max_drawdown_pips": None,
            "max_concurrent_trades": 0,
            "longest_losing_streak": 0,
            "yearly": [],
            "pair_summary": [],
            "exit_reason_summary": [],
        }
    values = sample["net_stress_pips"].astype("float64")
    curve = equity_curve(sample)
    yearly = yearly_summary(sample)
    positive_share, positive_years, year_count = positive_year_share(sample, "net_stress_pips")
    return {
        "trade_count": int(len(sample)),
        "total_stress_pips": float(values.sum()),
        "mean_stress_pips": float(values.mean()),
        "median_stress_pips": float(values.median()),
        "lower_stress_pips": lower_confidence_bound(values, 0.05),
        "worst_trade_pips": float(values.min()),
        "best_trade_pips": float(values.max()),
        "positive_year_share": positive_share,
        "positive_year_count": int(positive_years),
        "year_count": int(year_count),
        "two_consecutive_negative_years": has_two_consecutive_negative_years(yearly),
        "max_drawdown_pips": float(curve["drawdown_pips"].min()),
        "max_concurrent_trades": max_concurrent_trades(sample),
        "longest_losing_streak": longest_losing_streak(sample),
        "yearly": yearly,
        "pair_summary": grouped_summary(sample, "pair"),
        "exit_reason_summary": grouped_summary(sample, "exit_stress_reason"),
    }


def risk_flags(summary: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    max_drawdown = summary["max_drawdown_pips"]
    if max_drawdown is not None and abs(max_drawdown) > MAX_ACCEPTABLE_HISTORICAL_DRAWDOWN_PIPS:
        flags.append("historical_drawdown_exceeds_policy_threshold")
    if summary["max_concurrent_trades"] > MAX_ACCEPTABLE_CONCURRENT_TRADES:
        flags.append("concurrent_trade_count_exceeds_policy_threshold")
    if summary.get("two_consecutive_negative_years"):
        flags.append("two_consecutive_negative_years")
    return flags


def fast148_status(pre_summary: dict[str, Any], current_summary: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "portfolio_risk_audit"]
    pre_flags = risk_flags(pre_summary)
    current_flags = risk_flags(current_summary)
    if pre_flags or current_flags:
        return (
            "FAST148_USD_QUOTED_BASKET_REQUIRES_DRAWDOWN_AND_EXPOSURE_CONTROLS",
            reasons
            + [f"pre:{flag}" for flag in pre_flags]
            + [f"current:{flag}" for flag in current_flags]
            + ["fresh_preregistered_prospective_observation_remains_required"],
        )
    return (
        "FAST148_USD_QUOTED_BASKET_RISK_AUDIT_CLEAN_REQUIRES_PREREGISTRATION",
        reasons + ["fresh_preregistered_prospective_observation_required_before_any_execution"],
    )


def run(
    *,
    pre_input_path: Path = DEFAULT_PRE_INPUT_PATH,
    current_input_path: Path = DEFAULT_CURRENT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    pre_source = pd.read_parquet(pre_input_path)
    current_source = pd.read_parquet(current_input_path)
    pre_sample = filtered_sample(
        pre_source,
        sample_column="sample",
        sample_value=PRE_SAMPLE_LABEL,
    )
    current_sample = filtered_sample(
        current_source,
        sample_column=None,
        sample_value=None,
    )
    if pre_sample.empty:
        raise ValueError("Fast148 pre sample is empty after filtering")
    if current_sample.empty:
        raise ValueError("Fast148 current sample is empty after filtering")

    pre_summary = summarize_sample(pre_sample)
    current_summary = summarize_sample(current_sample)
    status, status_reasons = fast148_status(pre_summary, current_summary)

    output_root.mkdir(parents=True, exist_ok=True)
    pre_trades_path = output_root / "pre-basket-trades.csv"
    current_trades_path = output_root / "current-basket-trades.csv"
    pre_curve_path = output_root / "pre-equity-curve.csv"
    current_curve_path = output_root / "current-equity-curve.csv"
    summary_path = output_root / "portfolio-risk-summary.json"
    pre_sample.to_csv(pre_trades_path, index=False)
    current_sample.to_csv(current_trades_path, index=False)
    equity_curve(pre_sample).to_csv(pre_curve_path, index=False)
    equity_curve(current_sample).to_csv(current_curve_path, index=False)

    summary = {
        "pre": pre_summary,
        "current": current_summary,
        "risk_thresholds": {
            "max_acceptable_historical_drawdown_pips": MAX_ACCEPTABLE_HISTORICAL_DRAWDOWN_PIPS,
            "max_acceptable_concurrent_trades": MAX_ACCEPTABLE_CONCURRENT_TRADES,
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast148_is_historical_portfolio_risk_audit_not_strategy_approval",
            "portfolio_controls_are_required_before_shadow_or_testnet",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast148_usd_quoted_basket_portfolio_risk_audit",
        "status": status,
        "status_reasons": status_reasons,
        "target_policy_id": TARGET_POLICY_ID,
        "pre_input_path": str(pre_input_path),
        "pre_input_sha256": sha256_file(pre_input_path),
        "current_input_path": str(current_input_path),
        "current_input_sha256": sha256_file(current_input_path),
        "pre_sample_label": PRE_SAMPLE_LABEL,
        "current_sample_label": CURRENT_SAMPLE_LABEL,
        "policy": {
            "group_id": GROUP_ID,
            "target_pairs": TARGET_PAIRS,
            "entry_policy_id": ENTRY_POLICY_ID,
            "risk_policy_id": RISK_POLICY_ID,
            "side_filter": SIDE_FILTER,
            "management": MANAGEMENT,
        },
        "summary": summary,
        "artifacts": {
            "pre_trades": {"path": str(pre_trades_path), "sha256": sha256_file(pre_trades_path)},
            "current_trades": {"path": str(current_trades_path), "sha256": sha256_file(current_trades_path)},
            "pre_equity_curve": {"path": str(pre_curve_path), "sha256": sha256_file(pre_curve_path)},
            "current_equity_curve": {"path": str(current_curve_path), "sha256": sha256_file(current_curve_path)},
            "portfolio_risk_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "target_policy_id": report["target_policy_id"],
        "pre": {
            key: report["summary"]["pre"][key]
            for key in (
                "trade_count",
                "total_stress_pips",
                "mean_stress_pips",
                "lower_stress_pips",
                "max_drawdown_pips",
                "max_concurrent_trades",
                "two_consecutive_negative_years",
            )
        },
        "current": {
            key: report["summary"]["current"][key]
            for key in (
                "trade_count",
                "total_stress_pips",
                "mean_stress_pips",
                "lower_stress_pips",
                "max_drawdown_pips",
                "max_concurrent_trades",
                "two_consecutive_negative_years",
            )
        },
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
