"""Fast153: robustness/concentration audit for the Fast151 operating policy.

Fast151 is still not approved and still lacks prospective observations. This
audit asks a different historical question: is the selected operating policy
mostly one pair or one year in disguise?

It uses the Fast150 selected policy trades because those are the frozen
Fast151 historical operating-policy rows:

    max_per_pair_1 + dd800_cooldown180d

Nothing here sends orders, touches ledger state, or authorizes shadow/Testnet.
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
from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import (  # noqa: E402
    TARGET_POLICY_ID,
    summarize_sample,
)
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)

DEFAULT_PRE_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast150"
    / "usd-quoted-basket-circuit-breaker-audit"
    / "selected-breaker-pre-trades.csv"
)
DEFAULT_CURRENT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast150"
    / "usd-quoted-basket-circuit-breaker-audit"
    / "selected-breaker-current-trades.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast153" / "fast151-robustness-audit"
TIMESTAMP_COLUMNS = ("signal_timestamp", "entry_timestamp", "exit_base_timestamp", "exit_stress_timestamp")


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


def run(
    *,
    pre_input_path: Path = DEFAULT_PRE_INPUT_PATH,
    current_input_path: Path = DEFAULT_CURRENT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    pre = load_trades(pre_input_path)
    current = load_trades(current_input_path)
    pre_summary = summarize_sample(pre)
    current_summary = summarize_sample(current)
    pre_pair = contribution_rows(pre, "pair")
    current_pair = contribution_rows(current, "pair")
    pre_year = contribution_rows(with_year(pre), "year")
    current_year = contribution_rows(with_year(current), "year")
    pre_leave_pair = leave_one_rows(pre, "pair")
    current_leave_pair = leave_one_rows(current, "pair")
    pre_leave_year = leave_one_rows(with_year(pre), "year")
    current_leave_year = leave_one_rows(with_year(current), "year")

    status, status_reasons = fast153_status(
        pre_pair=pre_pair,
        current_pair=current_pair,
        pre_leave_pair=pre_leave_pair,
        current_leave_pair=current_leave_pair,
        pre_leave_year=pre_leave_year,
        current_leave_year=current_leave_year,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    pair_path = output_root / "pair-contribution.csv"
    year_path = output_root / "year-contribution.csv"
    leave_pair_path = output_root / "leave-one-pair.csv"
    leave_year_path = output_root / "leave-one-year.csv"
    summary_path = output_root / "robustness-summary.json"
    pd.DataFrame([*tag_rows(pre_pair, "pre"), *tag_rows(current_pair, "current")]).to_csv(pair_path, index=False)
    pd.DataFrame([*tag_rows(pre_year, "pre"), *tag_rows(current_year, "current")]).to_csv(year_path, index=False)
    pd.DataFrame([*tag_rows(pre_leave_pair, "pre"), *tag_rows(current_leave_pair, "current")]).to_csv(
        leave_pair_path,
        index=False,
    )
    pd.DataFrame([*tag_rows(pre_leave_year, "pre"), *tag_rows(current_leave_year, "current")]).to_csv(
        leave_year_path,
        index=False,
    )
    summary = {
        "pre": pre_summary,
        "current": current_summary,
        "pre_pair_contribution": pre_pair,
        "current_pair_contribution": current_pair,
        "pre_year_contribution": pre_year,
        "current_year_contribution": current_year,
        "pre_leave_one_pair": pre_leave_pair,
        "current_leave_one_pair": current_leave_pair,
        "pre_leave_one_year": pre_leave_year,
        "current_leave_one_year": current_leave_year,
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast153_is_historical_robustness_audit_not_strategy_approval",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast153_fast151_robustness_audit",
        "status": status,
        "status_reasons": status_reasons,
        "target_policy_id": TARGET_POLICY_ID,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "pre_input_path": str(pre_input_path),
        "pre_input_sha256": sha256_file(pre_input_path),
        "current_input_path": str(current_input_path),
        "current_input_sha256": sha256_file(current_input_path),
        "summary": summary,
        "artifacts": {
            "pair_contribution": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
            "year_contribution": {"path": str(year_path), "sha256": sha256_file(year_path)},
            "leave_one_pair": {"path": str(leave_pair_path), "sha256": sha256_file(leave_pair_path)},
            "leave_one_year": {"path": str(leave_year_path), "sha256": sha256_file(leave_year_path)},
            "robustness_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values(["entry_timestamp", "pair", "candidate_id"], ignore_index=True)


def with_year(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(year=frame["entry_timestamp"].dt.year)


def contribution_rows(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    total = float(frame["net_stress_pips"].sum())
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(group_column, sort=True):
        group_total = float(group["net_stress_pips"].sum())
        rows.append(
            {
                group_column: value.item() if hasattr(value, "item") else value,
                "trade_count": int(len(group)),
                "total_stress_pips": group_total,
                "mean_stress_pips": float(group["net_stress_pips"].mean()),
                "share_of_total_stress": None if total == 0 else float(group_total / total),
                "positive_total_stress": bool(group_total > 0),
            }
        )
    return rows


def leave_one_rows(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in sorted(frame[group_column].unique()):
        sample = frame.loc[~frame[group_column].eq(value)].copy()
        summary = summarize_sample(sample)
        rows.append(
            {
                f"dropped_{group_column}": value.item() if hasattr(value, "item") else value,
                "trade_count": summary["trade_count"],
                "total_stress_pips": summary["total_stress_pips"],
                "mean_stress_pips": summary["mean_stress_pips"],
                "lower_stress_pips": summary["lower_stress_pips"],
                "max_drawdown_pips": summary["max_drawdown_pips"],
                "max_concurrent_trades": summary["max_concurrent_trades"],
                "two_consecutive_negative_years": summary["two_consecutive_negative_years"],
                "passes_positive_lcb": bool(
                    summary["lower_stress_pips"] is not None and summary["lower_stress_pips"] > 0
                ),
            }
        )
    return rows


def tag_rows(rows: list[dict[str, Any]], sample: str) -> list[dict[str, Any]]:
    return [{"sample": sample, **row} for row in rows]


def fast153_status(
    *,
    pre_pair: list[dict[str, Any]],
    current_pair: list[dict[str, Any]],
    pre_leave_pair: list[dict[str, Any]],
    current_leave_pair: list[dict[str, Any]],
    pre_leave_year: list[dict[str, Any]],
    current_leave_year: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "robustness_audit"]
    flags: list[str] = []
    if not all(row["positive_total_stress"] for row in pre_pair):
        flags.append("pre_pair_contribution_not_all_positive")
    if not all(row["positive_total_stress"] for row in current_pair):
        flags.append("current_pair_contribution_not_all_positive")
    if not all(row["passes_positive_lcb"] for row in pre_leave_pair):
        flags.append("pre_leave_one_pair_lcb_not_all_positive")
    if not all(row["passes_positive_lcb"] for row in current_leave_pair):
        flags.append("current_leave_one_pair_lcb_not_all_positive")
    if not all(row["passes_positive_lcb"] for row in pre_leave_year):
        flags.append("pre_leave_one_year_lcb_not_all_positive")
    if not all(row["passes_positive_lcb"] for row in current_leave_year):
        flags.append("current_leave_one_year_lcb_not_all_positive")
    if any(row["two_consecutive_negative_years"] for row in current_leave_pair):
        flags.append("current_leave_one_pair_can_reintroduce_bad_year_cluster")
    if any(row["two_consecutive_negative_years"] for row in current_leave_year):
        flags.append("current_leave_one_year_can_reintroduce_bad_year_cluster")
    if flags:
        return (
            "FAST153_FAST151_ROBUST_BUT_CONCENTRATION_RISKS_REQUIRE_PROSPECTIVE_OBSERVATION",
            reasons + flags + ["fresh_preregistered_prospective_observation_still_required"],
        )
    return (
        "FAST153_FAST151_ROBUSTNESS_AUDIT_CLEAN_REQUIRES_PROSPECTIVE_OBSERVATION",
        reasons + ["fresh_preregistered_prospective_observation_required_before_any_execution"],
    )


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "operational_policy_id": report["operational_policy_id"],
        "pre": {
            key: report["summary"]["pre"][key]
            for key in (
                "trade_count",
                "total_stress_pips",
                "mean_stress_pips",
                "lower_stress_pips",
                "max_drawdown_pips",
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
                "two_consecutive_negative_years",
            )
        },
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
