"""Fast142: side-filter audit for the NZDUSD D1 Donchian-fade clue.

Fast135 showed mixed temporal stability for the frozen Fast134 candidate. This
runner asks a narrower, chronological question: if we use the pre-2019 sample
only to choose a simple causal side filter (LONG / SHORT / ALL), does that
filter survive the current 2019-2026 sample?

This is still historical research only. It writes offline artifacts and never
touches the trading engine, credentials, ledger, Demo, Testnet or LIVE accounts.
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

from scripts.research.run_fast104_fixed_outlier_veto_audit import sha256_file  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    lower_confidence_bound,
    positive_year_share,
)
from scripts.research.run_fast134_nzdusd_donchian_fade_fixed1r_observation import (  # noqa: E402
    FROZEN_POLICY_ID,
)

DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast135"
    / "nzdusd-donchian-fade-fixed1r-temporal-audit"
    / "temporal-audit-trades.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast142"
    / "nzdusd-donchian-fade-side-filter"
)
SELECTION_SLICE = "pre_2011_2018"
AUDIT_SLICE = "current_2019_2026"
MIN_SELECTION_TRADES = 12
MIN_SELECTION_YEARS = 3
MIN_AUDIT_TRADES = 12
MIN_AUDIT_YEARS = 3
MIN_POSITIVE_YEAR_SHARE = 2 / 3
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
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(input_path=arguments.input_path, output_root=arguments.output_root)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def side_filter_candidates() -> tuple[dict[str, str | None], ...]:
    return (
        {"filter_id": "all_sides", "side": None},
        {"filter_id": "long_only", "side": "LONG"},
        {"filter_id": "short_only", "side": "SHORT"},
    )


def apply_filter(trades: pd.DataFrame, side: str | None) -> pd.DataFrame:
    if side is None:
        return trades.copy()
    return trades.loc[trades["side"].eq(side)].copy()


def summarize_sample(sample: pd.DataFrame) -> dict[str, Any]:
    if sample.empty:
        return {
            "trade_count": 0,
            "mean_stress_pips": None,
            "total_stress_pips": 0.0,
            "lower_stress_pips": None,
            "positive_year_share": None,
            "positive_year_count": 0,
            "year_count": 0,
            "yearly": [],
            "top_positive_contribution_share": None,
            "worst_trade_pips": None,
            "best_trade_pips": None,
        }
    values = sample["net_stress_pips"].astype("float64")
    positive_year_share, positive_year_count, year_count = positive_year_share_for(sample)
    total = float(values.sum())
    positive_values = values.loc[values > 0].sort_values(ascending=False)
    top_positive = float(positive_values.head(3).sum()) if total > 0 else None
    return {
        "trade_count": int(len(sample)),
        "mean_stress_pips": float(values.mean()),
        "total_stress_pips": total,
        "lower_stress_pips": lower_confidence_bound(values, 0.05),
        "positive_year_share": positive_year_share,
        "positive_year_count": positive_year_count,
        "year_count": year_count,
        "yearly": yearly_summary(sample),
        "top_positive_contribution_share": (top_positive / total if top_positive is not None else None),
        "worst_trade_pips": float(values.min()),
        "best_trade_pips": float(values.max()),
    }


def positive_year_share_for(sample: pd.DataFrame) -> tuple[float | None, int, int]:
    if sample.empty:
        return None, 0, 0
    share, positive, years = positive_year_share(sample, "net_stress_pips")
    return share, int(positive), int(years)


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


def passes_selection_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["trade_count"] >= MIN_SELECTION_TRADES
        and summary["year_count"] >= MIN_SELECTION_YEARS
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["lower_stress_pips"] is not None
        and summary["lower_stress_pips"] > 0
        and summary["positive_year_share"] is not None
        and summary["positive_year_share"] >= MIN_POSITIVE_YEAR_SHARE
    )


def passes_audit_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["trade_count"] >= MIN_AUDIT_TRADES
        and summary["year_count"] >= MIN_AUDIT_YEARS
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["lower_stress_pips"] is not None
        and summary["lower_stress_pips"] > 0
        and summary["positive_year_share"] is not None
        and summary["positive_year_share"] >= MIN_POSITIVE_YEAR_SHARE
    )


def evaluate_filters(trades: pd.DataFrame) -> list[dict[str, Any]]:
    selection = trades.loc[trades["slice"].eq(SELECTION_SLICE)].copy()
    audit = trades.loc[trades["slice"].eq(AUDIT_SLICE)].copy()
    evaluations: list[dict[str, Any]] = []
    for candidate in side_filter_candidates():
        selection_summary = summarize_sample(apply_filter(selection, candidate["side"]))
        audit_summary = summarize_sample(apply_filter(audit, candidate["side"]))
        selection_passed = passes_selection_gate(selection_summary)
        audit_passed = selection_passed and passes_audit_gate(audit_summary)
        evaluations.append(
            {
                "filter_id": candidate["filter_id"],
                "side": candidate["side"],
                "selection": selection_summary,
                "audit": audit_summary,
                "selection_passed": selection_passed,
                "audit_passed": audit_passed,
            }
        )
    return evaluations


def select_best_passed(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [evaluation for evaluation in evaluations if evaluation["audit_passed"]]
    if not passed:
        return None
    return sorted(
        passed,
        key=lambda item: (
            item["audit"]["lower_stress_pips"],
            item["audit"]["mean_stress_pips"],
            item["audit"]["trade_count"],
        ),
        reverse=True,
    )[0]


def fast142_status(best_passed: dict[str, Any] | None) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "chronological_side_filter_audit"]
    if best_passed is None:
        return (
            "FAST142_NZDUSD_SIDE_FILTER_FAILED_AUDIT",
            reasons + ["no_side_filter_passed_pre2019_selection_and_2019_2026_audit"],
        )
    if best_passed["filter_id"] == "short_only":
        return (
            "FAST142_NZDUSD_SHORT_ONLY_SIDE_FILTER_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                "short_only_selected_on_pre2019_and_passed_2019_2026_audit",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    return (
        "FAST142_NZDUSD_SIDE_FILTER_CANDIDATE_REQUIRES_PREREGISTRATION",
        reasons
        + [
            f"selected_filter:{best_passed['filter_id']}",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
        ],
    )


def run(*, input_path: Path = DEFAULT_INPUT_PATH, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    trades = pd.read_parquet(input_path)
    if trades.empty:
        raise ValueError("Fast142 input trades are empty")
    for column in ("entry_timestamp", "exit_stress_timestamp"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    evaluations = evaluate_filters(trades)
    best_passed = select_best_passed(evaluations)
    status, status_reasons = fast142_status(best_passed)

    output_root.mkdir(parents=True, exist_ok=True)
    evaluations_path = output_root / "side-filter-evaluations.json"
    passed_path = output_root / "passed-side-filters.json"
    write_json(evaluations_path, evaluations)
    write_json(passed_path, [evaluation for evaluation in evaluations if evaluation["audit_passed"]])

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast142_is_historical_side_filter_audit_not_strategy_approval",
            "entry_policy_was_discovered_historically_before_this_side_filter_audit",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast142_nzdusd_donchian_fade_side_filter",
        "status": status,
        "status_reasons": status_reasons,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "selection_slice": SELECTION_SLICE,
        "audit_slice": AUDIT_SLICE,
        "minimums": {
            "min_selection_trades": MIN_SELECTION_TRADES,
            "min_selection_years": MIN_SELECTION_YEARS,
            "min_audit_trades": MIN_AUDIT_TRADES,
            "min_audit_years": MIN_AUDIT_YEARS,
            "min_positive_year_share": MIN_POSITIVE_YEAR_SHARE,
        },
        "evaluations": evaluations,
        "best_passed_filter": best_passed,
        "artifacts": {
            "evaluations": {"path": str(evaluations_path), "sha256": sha256_file(evaluations_path)},
            "passed_filters": {"path": str(passed_path), "sha256": sha256_file(passed_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    best = report["best_passed_filter"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "frozen_policy_id": report["frozen_policy_id"],
        "best_passed_filter": {
            "filter_id": best["filter_id"],
            "selection": best["selection"],
            "audit": best["audit"],
        }
        if best
        else None,
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
