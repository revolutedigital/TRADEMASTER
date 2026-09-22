"""Evaluate every development portfolio against the pre-registered gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.research.statistical_gate import (
    evaluate_statistical_gate,
    probability_of_backtest_overfitting,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--portfolio-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "trailing-pilot-stress-target",
    )
    parser.add_argument("--attempted-hypotheses", type=int, default=44)
    parser.add_argument(
        "--top-p-report",
        type=Path,
        help="JSON report emitted by run_top_p_pilot.py; required to prove top-p monotonicity.",
    )
    parser.add_argument(
        "--prospective-shadow-report",
        type=Path,
        help=(
            "Committed JSON report emitted by settle_shadow_outcomes.py; required to prove "
            "prospective shadow positivity."
        ),
    )
    arguments = parser.parse_args()
    paths = sorted(arguments.portfolio_root.glob("policy=*.parquet"))
    if not paths:
        parser.error("no portfolio trade files found")
    top_p_monotonic, top_p_reasons = _read_top_p_monotonicity(arguments.top_p_report)
    prospective_positive, prospective_reasons = _read_prospective_shadow_positive(
        arguments.prospective_shadow_report
    )
    daily_series = {}
    trades_by_strategy = {}
    for path in paths:
        trades = pd.read_parquet(path)
        strategy = path.stem
        trades_by_strategy[strategy] = trades
        dates = pd.to_datetime(trades["entry_time_ms"], unit="ms", utc=True).dt.date
        daily_series[strategy] = (
            trades.assign(utc_date=dates).groupby("utc_date")["expected_net_bps"].sum()
        )
    daily_matrix = pd.DataFrame(daily_series).fillna(0)
    pbo = probability_of_backtest_overfitting(daily_matrix)
    results = []
    for strategy, trades in trades_by_strategy.items():
        gate = evaluate_statistical_gate(
            trades,
            attempted_hypotheses=arguments.attempted_hypotheses,
            temporal_fold_count=3,
            top_p_monotonic=top_p_monotonic,
            prospective_positive=prospective_positive,
            pbo=pbo,
        )
        results.append({"strategy": strategy, **gate.to_dict()})
    report = _with_artifact_sha256(
        {
            "research_only": True,
            "order_submission_allowed": False,
            "execution_authorization": "none",
            "attempted_hypotheses": arguments.attempted_hypotheses,
            "top_p_monotonic": top_p_monotonic,
            "top_p_monotonic_reasons": top_p_reasons,
            "prospective_shadow_positive": prospective_positive,
            "prospective_shadow_reasons": prospective_reasons,
            "decision_counts": pd.Series(
                [row["decision"] for row in results]
            ).value_counts().to_dict(),
            "results": results,
        }
    )
    output = arguments.portfolio_root / "statistical-gate.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["decision_counts"], indent=2))  # noqa: T201
    return 0


def _read_top_p_monotonicity(path: Path | None) -> tuple[bool, list[str]]:
    if path is None:
        return False, ["top_p_report_missing"]
    if not path.exists():
        return False, [f"top_p_report_missing:{path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("research_only") is not True:
        return False, ["top_p_report_not_research_only"]
    if payload.get("order_submission_allowed") is not False:
        return False, ["top_p_report_allows_order_submission"]
    if payload.get("execution_authorization") != "none":
        return False, ["top_p_report_has_execution_authorization"]
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return False, ["top_p_report_has_no_results"]
    reasons: list[str] = []
    for index, result in enumerate(results):
        summary = result.get("summary") if isinstance(result, dict) else None
        if not isinstance(summary, dict):
            reasons.append(f"result_{index}_summary_missing")
            continue
        if summary.get("top_p_monotonic") is not True:
            monotonicity = summary.get("top_p_monotonicity")
            if isinstance(monotonicity, dict) and isinstance(monotonicity.get("reasons"), list):
                reasons.extend(str(reason) for reason in monotonicity["reasons"])
            else:
                reasons.append(f"result_{index}_top_p_monotonicity_missing")
    return not reasons, reasons


def _read_prospective_shadow_positive(path: Path | None) -> tuple[bool, list[str]]:
    if path is None:
        return False, ["prospective_shadow_report_missing"]
    if not path.exists():
        return False, [f"prospective_shadow_report_missing:{path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    reasons: list[str] = []
    if payload.get("research_only") is not True:
        reasons.append("prospective_shadow_report_not_research_only")
    if payload.get("order_submission_allowed") is not False:
        reasons.append("prospective_shadow_report_allows_order_submission")
    if payload.get("execution_authorization") != "none":
        reasons.append("prospective_shadow_report_has_execution_authorization")
    if payload.get("committed") is not True:
        reasons.append("prospective_shadow_report_not_committed")
    if payload.get("dry_run") is not False:
        reasons.append("prospective_shadow_report_is_dry_run")
    signal_count = _positive_int_or_zero(payload.get("signal_count"))
    outcome_count = _positive_int_or_zero(payload.get("outcome_count"))
    decision_day_count = _positive_int_or_zero(payload.get("decision_day_count"))
    outcome_day_count = _positive_int_or_zero(payload.get("outcome_day_count"))
    outcomes = payload.get("outcomes")
    if payload.get("complete") is not True:
        reasons.append("prospective_shadow_report_incomplete")
    if signal_count <= 0:
        reasons.append("prospective_shadow_report_has_no_signals")
    if outcome_count != signal_count:
        reasons.append("prospective_shadow_outcome_count_mismatch")
    if decision_day_count < 20:
        reasons.append("prospective_shadow_has_fewer_than_20_days")
    if decision_day_count > 30:
        reasons.append("prospective_shadow_exceeds_30_days")
    if outcome_day_count != decision_day_count:
        reasons.append("prospective_shadow_outcome_day_count_mismatch")
    if not isinstance(outcomes, list):
        reasons.append("prospective_shadow_outcomes_missing")
    elif len(outcomes) != signal_count:
        reasons.append("prospective_shadow_outcome_list_incomplete")
    else:
        outcome_reasons, outcome_dates = _shadow_outcome_reasons(outcomes)
        reasons.extend(outcome_reasons)
        if len(outcome_dates) != decision_day_count:
            reasons.append("prospective_shadow_outcome_dates_mismatch")
    expected_mean = _finite_float(payload.get("expected_mean_bps"))
    stress_mean = _finite_float(payload.get("stress_mean_bps"))
    if expected_mean is None or expected_mean <= 0:
        reasons.append("prospective_shadow_expected_mean_not_positive")
    if stress_mean is None or stress_mean <= 0:
        reasons.append("prospective_shadow_stress_mean_not_positive")
    return not reasons, reasons


def _shadow_outcome_reasons(outcomes: list[Any]) -> tuple[list[str], set[str]]:
    reasons: list[str] = []
    outcome_dates: set[str] = set()
    seen_signal_ids: set[int] = set()
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, dict):
            reasons.append(f"prospective_shadow_outcome_{index}_malformed")
            continue
        signal_id = outcome.get("signal_id")
        if type(signal_id) is not int or signal_id <= 0:
            reasons.append(f"prospective_shadow_outcome_{index}_signal_id_invalid")
        elif signal_id in seen_signal_ids:
            reasons.append(f"prospective_shadow_outcome_{index}_signal_id_duplicate")
        else:
            seen_signal_ids.add(signal_id)
        label_sha256 = outcome.get("label_sha256")
        if not isinstance(label_sha256, str) or not _is_sha256(label_sha256):
            reasons.append(f"prospective_shadow_outcome_{index}_label_sha256_invalid")
        if _finite_float(outcome.get("expected_net_bps")) is None:
            reasons.append(f"prospective_shadow_outcome_{index}_expected_net_bps_invalid")
        if _finite_float(outcome.get("stress_net_bps")) is None:
            reasons.append(f"prospective_shadow_outcome_{index}_stress_net_bps_invalid")
        if "order_id" in outcome or "execution_id" in outcome:
            reasons.append(f"prospective_shadow_outcome_{index}_contains_execution_field")
        decision_date = _outcome_decision_date(outcome.get("decision_time"))
        if decision_date is None:
            reasons.append(f"prospective_shadow_outcome_{index}_decision_time_invalid")
        else:
            outcome_dates.add(decision_date)
    return reasons, outcome_dates


def _positive_int_or_zero(value: object) -> int:
    if type(value) is int and value > 0:
        return value
    return 0


def _finite_float(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _outcome_decision_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).date().isoformat()


def _with_artifact_sha256(report: dict[str, object]) -> dict[str, object]:
    payload = {key: value for key, value in report.items() if key != "artifact_sha256"}
    return {
        "artifact_sha256": _report_sha256(payload),
        **payload,
    }


def _report_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
