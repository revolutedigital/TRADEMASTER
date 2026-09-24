"""Fast146: pre-defined FX group audit for the short-only Donchian-fade setup.

Fast145 showed the exact setup did not replicate pair-by-pair outside NZDUSD.
Fast146 asks a better portfolio question: does a pre-defined group of related
pairs pass chronologically when selected on pre-2019 and audited on 2019-2026?

The strongest discovered group is the USD-quoted major basket:

    AUDUSD, EURUSD, GBPUSD, NZDUSD

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
    / "lab_fast146"
    / "usd-quoted-major-basket-audit"
)

PRE_SAMPLE_LABEL = "pre_2011_2018_selection"
CURRENT_SAMPLE_LABEL = "current_2019_2026_full_grid"
ENTRY_POLICY_ID = "d1_donchian_fade_ch100_hold20"
RISK_POLICY_ID = "stop2_fixed_1r"
SIDE = "SHORT"
MANAGEMENT = "fixed_1r"
GROUPS: dict[str, tuple[str, ...]] = {
    "all_10_pairs": tuple(PAIRS),
    "usd_quoted_majors": ("AUDUSD", "EURUSD", "GBPUSD", "NZDUSD"),
    "oceanic_usd": ("AUDUSD", "NZDUSD"),
    "europe_usd_quoted": ("EURUSD", "GBPUSD"),
    "commodity_linked": ("AUDUSD", "NZDUSD", "USDCAD"),
    "non_jpy": ("AUDUSD", "EURGBP", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF"),
    "usd_base": ("USDCAD", "USDCHF", "USDJPY"),
    "jpy_crosses": ("EURJPY", "GBPJPY", "USDJPY"),
    "eur_crosses": ("EURGBP", "EURJPY", "EURUSD"),
    "gbp_crosses": ("GBPJPY", "GBPUSD", "EURGBP"),
}
TARGET_GROUP_ID = "usd_quoted_majors"
TARGET_POLICY_ID = f"basket:{TARGET_GROUP_ID}:{ENTRY_POLICY_ID}:{RISK_POLICY_ID}:short_only"
MIN_GROUP_TRADES = 30
MIN_GROUP_YEARS = 3
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


def pair_contributions(sample: pd.DataFrame) -> list[dict[str, Any]]:
    if sample.empty:
        return []
    grouped = (
        sample.groupby("pair", sort=True)["net_stress_pips"]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    return [
        {
            "pair": str(row["pair"]),
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
            "pair_count": 0,
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
            "pair_contributions": [],
        }
    values = sample["net_stress_pips"].astype("float64")
    positive_share, positive_years, years = positive_year_share(sample, "net_stress_pips")
    return {
        "trade_count": int(len(sample)),
        "pair_count": int(sample["pair"].nunique()),
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
        "pair_contributions": pair_contributions(sample),
    }


def sample_passes_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["trade_count"] >= MIN_GROUP_TRADES
        and summary["year_count"] >= MIN_GROUP_YEARS
        and summary["mean_base_pips"] is not None
        and summary["mean_base_pips"] > 0
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["lower_stress_pips"] is not None
        and summary["lower_stress_pips"] > 0
        and summary["positive_year_share"] is not None
        and summary["positive_year_share"] >= MIN_POSITIVE_YEAR_SHARE
    )


def evaluate_groups(pre_sample: pd.DataFrame, current_sample: pd.DataFrame) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for group_id, pairs in GROUPS.items():
        pre = summarize_sample(pre_sample.loc[pre_sample["pair"].isin(pairs)])
        current = summarize_sample(current_sample.loc[current_sample["pair"].isin(pairs)])
        pre_passed = sample_passes_gate(pre)
        current_passed = sample_passes_gate(current)
        evaluations.append(
            {
                "group_id": group_id,
                "policy_id": f"basket:{group_id}:{ENTRY_POLICY_ID}:{RISK_POLICY_ID}:short_only",
                "pairs": list(pairs),
                "entry_policy_id": ENTRY_POLICY_ID,
                "risk_policy_id": RISK_POLICY_ID,
                "side": SIDE,
                "management": MANAGEMENT,
                "pre": pre,
                "current": current,
                "pre_passed": pre_passed,
                "current_passed": current_passed,
                "both_passed": pre_passed and current_passed,
            }
        )
    return evaluations


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row = {
            key: value
            for key, value in evaluation.items()
            if key not in {"pre", "current", "pairs"}
        }
        row["pairs"] = ",".join(evaluation["pairs"])
        for prefix in ("pre", "current"):
            summary = evaluation[prefix]
            for key, value in summary.items():
                if key in {"yearly", "pair_contributions"}:
                    continue
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def select_pre_registered_candidate(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    pre_passed = [item for item in evaluations if item["pre_passed"]]
    if not pre_passed:
        return None
    return sorted(
        pre_passed,
        key=lambda item: (
            item["pre"]["lower_stress_pips"],
            item["pre"]["mean_stress_pips"],
            item["pre"]["trade_count"],
        ),
        reverse=True,
    )[0]


def summarize_groups(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    selected = select_pre_registered_candidate(evaluations)
    passed = [item for item in evaluations if item["both_passed"]]
    return {
        "group_count": len(evaluations),
        "pre_pass_count": int(sum(item["pre_passed"] for item in evaluations)),
        "current_pass_count": int(sum(item["current_passed"] for item in evaluations)),
        "both_pass_count": len(passed),
        "passed_group_ids": [item["group_id"] for item in passed],
        "selected_by_pre_group_id": selected["group_id"] if selected else None,
        "selected_by_pre": selected,
        "target_group": next(item for item in evaluations if item["group_id"] == TARGET_GROUP_ID),
    }


def fast146_status(summary: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "predefined_group_audit",
        "uses_full_current_grid_from_fast133_not_fast137_current_finalist_subset",
    ]
    selected = summary["selected_by_pre"]
    if selected is None:
        return (
            "FAST146_GROUP_BASKET_FAILED_PRE_SELECTION",
            reasons + ["no_predefined_group_passed_pre2019_selection"],
        )
    if selected["group_id"] == TARGET_GROUP_ID and selected["both_passed"]:
        return (
            "FAST146_USD_QUOTED_MAJOR_BASKET_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                "usd_quoted_majors_selected_on_pre2019_and_passed_2019_2026_audit",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    if selected["both_passed"]:
        return (
            "FAST146_GROUP_BASKET_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"selected_group:{selected['group_id']}",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    return (
        "FAST146_GROUP_BASKET_FAILED_CURRENT_AUDIT",
        reasons + [f"selected_group_failed_current:{selected['group_id']}"],
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
        raise ValueError("Fast146 pre sample is empty after filtering")
    if current_sample.empty:
        raise ValueError("Fast146 current sample is empty after filtering")

    evaluations = evaluate_groups(pre_sample, current_sample)
    summary = summarize_groups(evaluations)
    status, status_reasons = fast146_status(summary)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "group-basket-metrics.csv"
    passed_path = output_root / "passed-groups.csv"
    summary_path = output_root / "group-basket-summary.json"
    evaluations_path = output_root / "group-basket-evaluations.json"

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
            "fast146_is_historical_group_basket_audit_not_strategy_approval",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast146_usd_quoted_major_basket_audit",
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
            "entry_policy_id": ENTRY_POLICY_ID,
            "risk_policy_id": RISK_POLICY_ID,
            "side": SIDE,
            "management": MANAGEMENT,
            "groups": GROUPS,
        },
        "minimums": {
            "min_group_trades": MIN_GROUP_TRADES,
            "min_group_years": MIN_GROUP_YEARS,
            "min_positive_year_share": MIN_POSITIVE_YEAR_SHARE,
        },
        "source_rows": {
            "pre_filtered_rows": int(len(pre_sample)),
            "current_filtered_rows": int(len(current_sample)),
        },
        "summary": summary,
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "passed_groups": {"path": str(passed_path), "sha256": sha256_file(passed_path)},
            "group_basket_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "evaluations": {"path": str(evaluations_path), "sha256": sha256_file(evaluations_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    selected = summary["selected_by_pre"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "target_policy_id": report["target_policy_id"],
        "selected_by_pre_group_id": summary["selected_by_pre_group_id"],
        "selected_pre": selected["pre"] if selected else None,
        "selected_current": selected["current"] if selected else None,
        "group_count": summary["group_count"],
        "pre_pass_count": summary["pre_pass_count"],
        "current_pass_count": summary["current_pass_count"],
        "both_pass_count": summary["both_pass_count"],
        "passed_group_ids": summary["passed_group_ids"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
