"""Fast57 prospective gate evaluator.

Fast57 is pre-registered in docs/forex/fast57-fixed-density-prospective-preregistration.md.
This runner evaluates the frozen candidate on new complete daily partitions and
fails closed until the minimum sample is reached.

It is research-only, never sends orders, and never approves a strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.trailing_portfolio import summarize_portfolio  # noqa: E402
from scripts.research.run_latency_adjusted_fixed_density_gate_holdout import (  # noqa: E402
    run_fixed_density_gate_holdout,
)
from scripts.research.run_latency_adjusted_density_gate_walkforward import (  # noqa: E402
    _parse_block_specs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-report", type=Path, required=True)
    parser.add_argument("--blocks", nargs="+", required=True)
    parser.add_argument(
        "--trade-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "normalized"
        / "aggTrades",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "fast57-fixed-density-prospective-gate",
    )
    parser.add_argument("--lookback-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--policy-set", choices=("v1", "low_cost"), default="low_cost")
    parser.add_argument("--policy-name", default="micro_8_3")
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--horizon-seconds", type=int, default=300)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--minimum-days", type=int, default=10)
    parser.add_argument("--minimum-trades", type=int, default=100)
    parser.add_argument("--maximum-days", type=int, default=30)
    arguments = parser.parse_args()

    report = run_fast57(
        artifact_report=arguments.artifact_report,
        block_specs=_parse_block_specs(arguments.blocks),
        trade_root=arguments.trade_root,
        output_root=arguments.output_root,
        lookback_seconds=arguments.lookback_seconds,
        limit=arguments.limit,
        policy_set=arguments.policy_set,
        policy_name=arguments.policy_name,
        tail_fraction=arguments.tail_fraction,
        horizon_seconds=arguments.horizon_seconds,
        latency_ms=arguments.latency_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        minimum_days=arguments.minimum_days,
        minimum_trades=arguments.minimum_trades,
        maximum_days=arguments.maximum_days,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast57(
    *,
    artifact_report: Path,
    block_specs: tuple[Any, ...],
    trade_root: Path,
    output_root: Path,
    lookback_seconds: int,
    limit: int,
    policy_set: str,
    policy_name: str,
    tail_fraction: float,
    horizon_seconds: int,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
    minimum_days: int,
    minimum_trades: int,
    maximum_days: int,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    gated_report = run_fixed_density_gate_holdout(
        artifact_report=artifact_report,
        block_specs=block_specs,
        trade_root=trade_root,
        output_root=output_root / "gated",
        lookback_seconds=lookback_seconds,
        limit=limit,
        policy_set=policy_set,
        policy_name=policy_name,
        tail_fraction=tail_fraction,
        horizon_seconds=horizon_seconds,
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    baseline_report = run_fixed_density_gate_holdout(
        artifact_report=artifact_report,
        block_specs=block_specs,
        trade_root=trade_root,
        output_root=output_root / "ungated_baseline",
        lookback_seconds=lookback_seconds,
        limit=999_999_999,
        policy_set=policy_set,
        policy_name=policy_name,
        tail_fraction=tail_fraction,
        horizon_seconds=horizon_seconds,
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    gated_trades = _read_trades(Path(str(gated_report["trades_path"])))
    baseline_trades = _read_trades(Path(str(baseline_report["trades_path"])))
    complete_days = _complete_utc_days(block_specs)
    gated_daily = _daily_summary(gated_trades)
    gated_block = _block_summary(gated_trades)
    status, status_reasons, criteria = evaluate_fast57_status(
        gated_summary=gated_report["summary"],
        baseline_summary=baseline_report["summary"],
        complete_utc_days=len(complete_days),
        gated_daily_summary=gated_daily,
        gated_block_summary=gated_block,
        minimum_days=minimum_days,
        minimum_trades=minimum_trades,
        maximum_days=maximum_days,
    )
    report_path = output_root / "report.json"
    report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "approval_reasons": [
            "fast57_research_only",
            "offline_aggtrades_replay_not_exchange_execution",
            "book_or_shadow_confirmation_required_even_if_passed",
        ],
        "experiment_stage": "fast57_fixed_density_prospective_gate",
        "status": status,
        "status_reasons": status_reasons,
        "criteria": criteria,
        "complete_utc_days": complete_days,
        "minimum_days": minimum_days,
        "minimum_trades": minimum_trades,
        "maximum_days": maximum_days,
        "candidate": {
            "model_sha256": gated_report["model_sha256"],
            "threshold": gated_report["threshold"],
            "tail_fraction": tail_fraction,
            "horizon_seconds": horizon_seconds,
            "policy_set": policy_set,
            "policy_name": policy_name,
            "density_gate": {
                "lookback_seconds": lookback_seconds,
                "limit": limit,
            },
            "latency_ms": latency_ms,
            "expected_cost_bps": expected_cost_bps,
            "stress_cost_bps": stress_cost_bps,
        },
        "gated_summary": gated_report["summary"],
        "baseline_summary": baseline_report["summary"],
        "gated_report_path": gated_report["report_path"],
        "baseline_report_path": baseline_report["report_path"],
        "gated_daily_summary": gated_daily,
        "gated_block_summary": gated_block,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def evaluate_fast57_status(
    *,
    gated_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    complete_utc_days: int,
    gated_daily_summary: list[dict[str, Any]],
    gated_block_summary: list[dict[str, Any]],
    minimum_days: int,
    minimum_trades: int,
    maximum_days: int,
) -> tuple[str, list[str], dict[str, object]]:
    trade_count = int(gated_summary["trade_count"])
    stress_total = float(gated_summary["stress_total_log_return_bps"])
    stress_mean = float(gated_summary["stress_mean_bps"])
    baseline_stress_total = float(baseline_summary["stress_total_log_return_bps"])
    max_drawdown = float(gated_summary["maximum_drawdown_bps"])
    positive_days = sum(float(day["stress_total_log_return_bps"]) > 0 for day in gated_daily_summary)
    day_count = len(gated_daily_summary)
    positive_day_rate = positive_days / day_count if day_count else 0.0
    positive_blocks = [block for block in gated_block_summary if block["stress_total_log_return_bps"] > 0]
    average_positive_block_gain = (
        sum(float(block["stress_total_log_return_bps"]) for block in positive_blocks)
        / len(positive_blocks)
        if positive_blocks
        else 0.0
    )
    worst_block_loss = min(
        (float(block["stress_total_log_return_bps"]) for block in gated_block_summary),
        default=0.0,
    )
    criteria = {
        "minimum_sample_met": complete_utc_days >= minimum_days and trade_count >= minimum_trades,
        "maximum_days_reached": complete_utc_days >= maximum_days,
        "stress_total_positive": stress_total > 0,
        "stress_mean_positive": stress_mean > 0,
        "positive_day_rate": positive_day_rate,
        "positive_day_rate_at_least_60pct": positive_day_rate >= 0.60,
        "drawdown_less_than_stress_total": stress_total > 0 and max_drawdown < stress_total,
        "gated_beats_ungated": stress_total > baseline_stress_total,
        "worst_block_loss": worst_block_loss,
        "average_positive_block_gain": average_positive_block_gain,
        "no_weekly_loss_gt_2x_average_weekly_gain": (
            average_positive_block_gain > 0
            and abs(min(worst_block_loss, 0.0)) <= 2 * average_positive_block_gain
        ),
    }
    if not criteria["minimum_sample_met"]:
        reasons = []
        if complete_utc_days < minimum_days:
            reasons.append(f"complete_days_below_minimum:{complete_utc_days}<{minimum_days}")
        if trade_count < minimum_trades:
            reasons.append(f"trade_count_below_minimum:{trade_count}<{minimum_trades}")
        if criteria["maximum_days_reached"]:
            return (
                "FAST57_FAILED_MINIMUM_TRADES_AFTER_MAXIMUM_DAYS",
                reasons + ["do_not_promote_to_strategy"],
                criteria,
            )
        return "FAST57_INSUFFICIENT_PROSPECTIVE_SAMPLE", reasons, criteria
    failed = [name for name, passed in criteria.items() if name not in {"positive_day_rate", "worst_block_loss", "average_positive_block_gain", "maximum_days_reached"} and passed is False]
    if failed:
        return (
            "FAST57_FAILED_CONFIRMATION",
            [f"failed:{name}" for name in failed] + ["do_not_promote_to_strategy"],
            criteria,
        )
    return (
        "FAST57_PASSED_RESEARCH_ONLY_NEXT_GATE_REQUIRED",
        ["not_approved_strategy", "book_or_shadow_confirmation_required"],
        criteria,
    )


def _read_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _complete_utc_days(block_specs: tuple[Any, ...]) -> list[str]:
    dates: set[str] = set()
    for block in block_specs:
        for path in block.dataset_root.glob("date=*/research_rows.parquet"):
            dates.add(path.parent.name.removeprefix("date="))
    return sorted(dates)


def _daily_summary(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["entry_time_ms"], unit="ms", utc=True).dt.date.astype(str)
    rows = []
    for date, group in frame.groupby("date", sort=True):
        summary = summarize_portfolio(group.drop(columns=["block", "date"], errors="ignore"))
        rows.append({"date": str(date), **summary})
    return rows


def _block_summary(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty or "block" not in trades.columns:
        return []
    rows = []
    for block, group in trades.groupby("block", sort=True):
        summary = summarize_portfolio(group.drop(columns=["block"], errors="ignore"))
        rows.append({"block": str(block), **summary})
    return rows


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "criteria": report["criteria"],
        "complete_utc_days": report["complete_utc_days"],
        "gated_summary": report["gated_summary"],
        "baseline_summary": report["baseline_summary"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
