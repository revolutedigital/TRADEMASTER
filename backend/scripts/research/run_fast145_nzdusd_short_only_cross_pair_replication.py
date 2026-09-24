"""Fast145: cross-pair replication audit for the NZDUSD short-only fade.

Fast144 showed the edge is narrow inside the NZDUSD parameter neighborhood.
Fast145 asks the next falsification question: does the exact frozen setup
replicate across other FX pairs, or is it specific to NZDUSD?

Inputs are existing materialized artifacts:

* Fast137 pre-2019 temporal-reverse trades for the full pair grid.
* Fast133 2019-2026 global-risk trades for the full current pair grid.

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
from scripts.research.run_fast121_fx_compression_breakout_scout import PAIRS, sha256_file  # noqa: E402
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
    / "lab_fast145"
    / "nzdusd-short-only-cross-pair-replication"
)

PRE_SAMPLE_LABEL = "pre_2011_2018_selection"
CURRENT_SAMPLE_LABEL = "current_2019_2026_full_grid"
ANCHOR_PAIR = "NZDUSD"
ENTRY_POLICY_ID = "d1_donchian_fade_ch100_hold20"
RISK_POLICY_ID = "stop2_fixed_1r"
SIDE = "SHORT"
MANAGEMENT = "fixed_1r"
FROZEN_POLICY_ID = f"{ANCHOR_PAIR}:{ENTRY_POLICY_ID}:{RISK_POLICY_ID}:short_only"
MIN_SAMPLE_TRADES = 12
MIN_SAMPLE_YEARS = 3
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


def pair_candidate_id(pair: str) -> str:
    return f"{pair}:{ENTRY_POLICY_ID}:{RISK_POLICY_ID}"


def filtered_sample(
    frame: pd.DataFrame,
    *,
    sample_column: str | None,
    sample_value: str | None,
) -> pd.DataFrame:
    sample = frame.copy()
    if sample_column and sample_value:
        sample = sample.loc[sample[sample_column].eq(sample_value)].copy()
    sample["entry_timestamp"] = pd.to_datetime(sample["entry_timestamp"], utc=True)
    return sample.loc[
        sample["entry_policy_id"].eq(ENTRY_POLICY_ID)
        & sample["risk_policy_id"].eq(RISK_POLICY_ID)
        & sample["side"].eq(SIDE)
        & sample["management"].eq(MANAGEMENT)
        & sample["pair"].isin(PAIRS)
    ].sort_values(["pair", "entry_timestamp"], ignore_index=True)


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


def summarize_sample(sample: pd.DataFrame) -> dict[str, Any]:
    if sample.empty:
        return {
            "trade_count": 0,
            "mean_base_pips": None,
            "mean_stress_pips": None,
            "total_stress_pips": 0.0,
            "lower_stress_pips": None,
            "positive_year_share": None,
            "positive_year_count": 0,
            "year_count": 0,
            "worst_trade_pips": None,
            "best_trade_pips": None,
            "yearly": [],
        }
    values = sample["net_stress_pips"].astype("float64")
    positive_share, positive_years, years = positive_year_share(sample, "net_stress_pips")
    return {
        "trade_count": int(len(sample)),
        "mean_base_pips": float(sample["net_base_pips"].mean()),
        "mean_stress_pips": float(values.mean()),
        "total_stress_pips": float(values.sum()),
        "lower_stress_pips": lower_confidence_bound(values, 0.05),
        "positive_year_share": positive_share,
        "positive_year_count": int(positive_years),
        "year_count": int(years),
        "worst_trade_pips": float(values.min()),
        "best_trade_pips": float(values.max()),
        "yearly": yearly_summary(sample),
    }


def sample_passes_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["trade_count"] >= MIN_SAMPLE_TRADES
        and summary["year_count"] >= MIN_SAMPLE_YEARS
        and summary["mean_base_pips"] is not None
        and summary["mean_base_pips"] > 0
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["lower_stress_pips"] is not None
        and summary["lower_stress_pips"] > 0
        and summary["positive_year_share"] is not None
        and summary["positive_year_share"] >= MIN_POSITIVE_YEAR_SHARE
    )


def evaluate_pairs(pre_sample: pd.DataFrame, current_sample: pd.DataFrame) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for pair in PAIRS:
        pre_summary = summarize_sample(pre_sample.loc[pre_sample["pair"].eq(pair)])
        current_summary = summarize_sample(current_sample.loc[current_sample["pair"].eq(pair)])
        pre_passed = sample_passes_gate(pre_summary)
        current_passed = sample_passes_gate(current_summary)
        evaluations.append(
            {
                "pair": pair,
                "candidate_id": pair_candidate_id(pair),
                "entry_policy_id": ENTRY_POLICY_ID,
                "risk_policy_id": RISK_POLICY_ID,
                "side": SIDE,
                "management": MANAGEMENT,
                "is_anchor_pair": pair == ANCHOR_PAIR,
                "pre": pre_summary,
                "current": current_summary,
                "pre_passed": pre_passed,
                "current_passed": current_passed,
                "both_passed": pre_passed and current_passed,
            }
        )
    return evaluations


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row = {key: value for key, value in evaluation.items() if key not in {"pre", "current"}}
        for prefix in ("pre", "current"):
            summary = evaluation[prefix]
            for key, value in summary.items():
                if key == "yearly":
                    continue
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def summarize_replication(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    anchor = next(item for item in evaluations if item["is_anchor_pair"])
    passed = [item for item in evaluations if item["both_passed"]]
    current_only = [item for item in evaluations if item["current_passed"] and not item["pre_passed"]]
    pre_only = [item for item in evaluations if item["pre_passed"] and not item["current_passed"]]
    return {
        "pair_count": len(evaluations),
        "pre_pass_count": int(sum(item["pre_passed"] for item in evaluations)),
        "current_pass_count": int(sum(item["current_passed"] for item in evaluations)),
        "both_pass_count": len(passed),
        "passed_pairs": [item["pair"] for item in passed],
        "current_only_pairs": [item["pair"] for item in current_only],
        "pre_only_pairs": [item["pair"] for item in pre_only],
        "anchor_pair": anchor,
        "non_anchor_pass_count": int(sum(item["both_passed"] and not item["is_anchor_pair"] for item in evaluations)),
    }


def fast145_status(summary: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "cross_pair_replication_audit",
        "uses_full_current_grid_from_fast133_not_fast137_current_finalist_subset",
    ]
    anchor = summary["anchor_pair"]
    if not anchor["both_passed"]:
        return (
            "FAST145_NZDUSD_SHORT_ONLY_ANCHOR_FAILED_CROSS_PAIR_AUDIT",
            reasons + ["anchor_pair_failed_pre_or_current_gate"],
        )
    if summary["non_anchor_pass_count"] > 0:
        return (
            "FAST145_NZDUSD_SHORT_ONLY_CROSS_PAIR_REPLICATED_REQUIRES_OBSERVATION",
            reasons
            + [
                f"passed_pairs:{','.join(summary['passed_pairs'])}",
                "fresh_preregistered_prospective_observation_remains_required",
            ],
        )
    return (
        "FAST145_NZDUSD_SHORT_ONLY_CROSS_PAIR_NOT_REPLICATED_REQUIRES_OBSERVATION",
        reasons
        + [
            "anchor_pair_passed_pre_and_current",
            "no_non_anchor_pair_passed_pre_and_current",
            f"current_only_pairs:{','.join(summary['current_only_pairs']) or 'none'}",
            "fresh_preregistered_prospective_observation_remains_required",
        ],
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
        raise ValueError("Fast145 pre sample is empty after filtering")
    if current_sample.empty:
        raise ValueError("Fast145 current sample is empty after filtering")

    evaluations = evaluate_pairs(pre_sample, current_sample)
    summary = summarize_replication(evaluations)
    status, status_reasons = fast145_status(summary)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "cross-pair-metrics.csv"
    passed_path = output_root / "passed-pairs.csv"
    summary_path = output_root / "cross-pair-summary.json"
    evaluations_path = output_root / "cross-pair-evaluations.json"

    metrics = pd.DataFrame(flat_metric_rows(evaluations)).sort_values(
        [
            "both_passed",
            "current_lower_stress_pips",
            "current_mean_stress_pips",
            "pre_lower_stress_pips",
        ],
        ascending=[False, False, False, False],
    )
    metrics.to_csv(metrics_path, index=False)
    metrics.loc[metrics["both_passed"]].to_csv(passed_path, index=False)
    write_json(summary_path, summary)
    write_json(evaluations_path, evaluations)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast145_is_historical_cross_pair_audit_not_strategy_approval",
            "fast143_prospective_observation_is_still_empty",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast145_nzdusd_short_only_cross_pair_replication",
        "status": status,
        "status_reasons": status_reasons,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "pre_input_path": str(pre_input_path),
        "pre_input_sha256": sha256_file(pre_input_path),
        "current_input_path": str(current_input_path),
        "current_input_sha256": sha256_file(current_input_path),
        "pre_sample_label": PRE_SAMPLE_LABEL,
        "current_sample_label": CURRENT_SAMPLE_LABEL,
        "policy": {
            "anchor_pair": ANCHOR_PAIR,
            "entry_policy_id": ENTRY_POLICY_ID,
            "risk_policy_id": RISK_POLICY_ID,
            "side": SIDE,
            "management": MANAGEMENT,
        },
        "minimums": {
            "min_sample_trades": MIN_SAMPLE_TRADES,
            "min_sample_years": MIN_SAMPLE_YEARS,
            "min_positive_year_share": MIN_POSITIVE_YEAR_SHARE,
        },
        "source_rows": {
            "pre_filtered_rows": int(len(pre_sample)),
            "current_filtered_rows": int(len(current_sample)),
        },
        "summary": summary,
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "passed_pairs": {"path": str(passed_path), "sha256": sha256_file(passed_path)},
            "cross_pair_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "evaluations": {"path": str(evaluations_path), "sha256": sha256_file(evaluations_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    anchor = summary["anchor_pair"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "frozen_policy_id": report["frozen_policy_id"],
        "anchor_pair_both_passed": anchor["both_passed"],
        "anchor_pre": anchor["pre"],
        "anchor_current": anchor["current"],
        "pair_count": summary["pair_count"],
        "pre_pass_count": summary["pre_pass_count"],
        "current_pass_count": summary["current_pass_count"],
        "both_pass_count": summary["both_pass_count"],
        "passed_pairs": summary["passed_pairs"],
        "current_only_pairs": summary["current_only_pairs"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
