"""Fast108 pair-specific audit for Fast6 FX Q2 policy trades.

Fast6 rejected the global FX Q2 trailing policy family. Fast108 asks whether
that rejection hid a pair-specific survivor. It only reads already-materialized
Fast6 policy trade parquet files and never sends orders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_POLICY_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast6" / "policies"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast108" / "fx-fast6-pair-audit"
DEFAULT_MIN_TRADES = 10
DEFAULT_MIN_MONTHS = 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-root", type=Path, default=DEFAULT_POLICY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--min-months", type=int, default=DEFAULT_MIN_MONTHS)
    arguments = parser.parse_args()
    report = run_fast108_fx_fast6_pair_audit(
        policy_root=arguments.policy_root,
        output_root=arguments.output_root,
        min_trades=arguments.min_trades,
        min_months=arguments.min_months,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast108_fx_fast6_pair_audit(
    *,
    policy_root: Path = DEFAULT_POLICY_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    min_trades: int = DEFAULT_MIN_TRADES,
    min_months: int = DEFAULT_MIN_MONTHS,
) -> dict[str, Any]:
    validate_parameters(min_trades=min_trades, min_months=min_months)
    output_root.mkdir(parents=True, exist_ok=True)
    policy_paths = tuple(sorted(policy_root.glob("*-trades.parquet")))
    if not policy_paths:
        raise FileNotFoundError(f"no Fast6 policy trade parquets found in {policy_root}")

    evaluations = []
    for path in policy_paths:
        trades = load_policy_trades(path)
        evaluations.extend(evaluate_policy_pairs(policy_name=policy_name_from_path(path), trades=trades))

    candidate_rows = [
        row
        for row in evaluations
        if row["trade_count"] >= min_trades
        and row["month_count"] >= min_months
        and row["base_mean_r"] > 0
        and row["stress_mean_r"] > 0
        and row["positive_stress_month_count"] == row["month_count"]
    ]
    weak_rows = [
        row
        for row in evaluations
        if row["trade_count"] >= min_trades
        and row["month_count"] >= min_months
        and row["base_mean_r"] > 0
        and row["stress_mean_r"] > 0
    ]
    top_rows = sorted(
        [row for row in evaluations if row["trade_count"] >= min_trades],
        key=lambda row: (row["stress_mean_r"], row["base_mean_r"], row["trade_count"]),
        reverse=True,
    )[:25]

    evaluations_path = output_root / "pair_policy_evaluations.json"
    evaluations_path.write_text(
        json.dumps(evaluations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidates_path = output_root / "strict_pair_policy_candidates.json"
    candidates_path.write_text(
        json.dumps(candidate_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    top_path = output_root / "top_pair_policy_diagnostics.json"
    top_path.write_text(
        json.dumps(top_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status, status_reasons = fast108_status(
        candidate_count=len(candidate_rows),
        weak_count=len(weak_rows),
    )
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast108_is_pair_specific_diagnostic_for_rejected_fast6_q2_family",
            "reads_historical_policy_trade_parquets_only",
            "requires_fresh_preregistration_before_any_prospective_use",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast108_fx_fast6_pair_specific_audit",
        "status": status,
        "status_reasons": status_reasons,
        "policy_root": str(policy_root),
        "policy_count": len(policy_paths),
        "pair_policy_evaluation_count": len(evaluations),
        "min_trades": min_trades,
        "min_months": min_months,
        "weak_candidate_count": len(weak_rows),
        "strict_candidate_count": len(candidate_rows),
        "top_pair_policy_diagnostics": top_rows,
        "evaluations_path": str(evaluations_path),
        "evaluations_sha256": sha256_file(evaluations_path),
        "strict_candidates_path": str(candidates_path),
        "strict_candidates_sha256": sha256_file(candidates_path),
        "top_diagnostics_path": str(top_path),
        "top_diagnostics_sha256": sha256_file(top_path),
    }
    return write_report(output_root, report)


def validate_parameters(*, min_trades: int, min_months: int) -> None:
    if min_trades <= 0:
        raise ValueError("min_trades must be positive")
    if min_months <= 0:
        raise ValueError("min_months must be positive")


def policy_name_from_path(path: Path) -> str:
    return path.name.removesuffix("-trades.parquet")


def load_policy_trades(path: Path) -> pd.DataFrame:
    trades = pd.read_parquet(path)
    if trades.empty:
        return trades
    frame = trades.reset_index()
    required = {"decision_time", "pair", "base_r", "stress_r"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    frame["month"] = pd.to_datetime(frame["decision_time"], utc=True).dt.strftime("%Y-%m")
    return frame


def evaluate_policy_pairs(*, policy_name: str, trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    rows = []
    for pair, pair_trades in trades.groupby("pair", sort=True):
        month_stats = (
            pair_trades.groupby("month", sort=True)
            .agg(
                trade_count=("stress_r", "size"),
                base_mean_r=("base_r", "mean"),
                stress_mean_r=("stress_r", "mean"),
                base_total_r=("base_r", "sum"),
                stress_total_r=("stress_r", "sum"),
            )
            .reset_index()
        )
        rows.append(
            {
                "policy": policy_name,
                "pair": str(pair),
                "trade_count": int(len(pair_trades)),
                "month_count": int(len(month_stats)),
                "base_mean_r": float(pair_trades["base_r"].mean()),
                "stress_mean_r": float(pair_trades["stress_r"].mean()),
                "base_total_r": float(pair_trades["base_r"].sum()),
                "stress_total_r": float(pair_trades["stress_r"].sum()),
                "positive_base_month_count": int((month_stats["base_mean_r"] > 0).sum()),
                "positive_stress_month_count": int((month_stats["stress_mean_r"] > 0).sum()),
                "min_month_stress_mean_r": float(month_stats["stress_mean_r"].min()),
                "min_month_stress_total_r": float(month_stats["stress_total_r"].min()),
            }
        )
    return rows


def fast108_status(*, candidate_count: int, weak_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if candidate_count:
        return "FAST108_PAIR_SPECIFIC_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION", reasons + [
            f"strict_pair_policy_candidates:{candidate_count}",
            "requires_fresh_preregistration_before_prospective_use",
        ]
    if weak_count:
        return "FAST108_PAIR_SPECIFIC_WEAK_ONLY_FAILED_STABILITY", reasons + [
            f"weak_pair_policy_candidates:{weak_count}",
            "no_candidate_positive_in_all_months",
        ]
    return "FAST108_PAIR_SPECIFIC_FAILED_SCREEN", reasons + [
        "no_pair_policy_positive_base_and_stress",
    ]


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "policy_count": report["policy_count"],
        "pair_policy_evaluation_count": report["pair_policy_evaluation_count"],
        "weak_candidate_count": report["weak_candidate_count"],
        "strict_candidate_count": report["strict_candidate_count"],
        "top_pair_policy_diagnostics": report["top_pair_policy_diagnostics"][:10],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
