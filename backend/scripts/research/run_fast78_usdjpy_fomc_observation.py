"""Fast78 preregistered USDJPY/FOMC observation runner.

Fast78 is not a trading strategy. It is a fail-closed, research-only evaluator
for the single weak survivor frozen in
docs/forex/fast78-usdjpy-fomc-observation-preregistration.md.

The runner filters for exactly one hypothesis, only after the preregistration
commit timestamp, writes local observation artifacts, and never sends orders or
commits ledger state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_TRADES_PATH = REPO_ROOT / "backend" / "data" / "lab_fast13" / "calendar-m1-trades.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast13" / "fast78-usdjpy-fomc-observation"
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast78-usdjpy-fomc-observation-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast78-registry.jsonl"
PREREGISTRATION_COMMIT = "97d7b900a9fbe016dc37a120374f54d8c07c915b"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-23T18:26:11Z")
PREREGISTRATION_SHA256 = "844d2c4e9d7b2e2e6edb25a801773b0ba1343c369d3e7eea5357032d87e3e99c"
SOURCE_FAST77_REPORT_SHA256 = "8d1d8a157d495bd8016860b18cba185e007a526d5eb98b955b29fb0604a86aee"
OBSERVATION_PAIR = "USDJPY"
OBSERVATION_EVENT = "FOMC rate decision"
OBSERVATION_POLICY_ID = "drift_thr2_d5_h120_trail_1r_0_75r"
DEFAULT_ALPHA = 0.05
EARLY_FAIL_EVENT_COUNT = 8
EARLY_FAIL_STRESS_TOTAL_PIPS = -20.0
SHADOW_CANDIDATE_MINIMUM_EVENTS = 20
SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS = 2.0
SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_RATE = 0.60


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args()
    report = run_fast78_usdjpy_fomc_observation(
        trades_path=arguments.trades_path,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast78_usdjpy_fomc_observation(
    *,
    trades_path: Path = DEFAULT_TRADES_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    validate_parameters(alpha=alpha)
    preregistration = validate_preregistration(
        preregistration_path=preregistration_path,
        registry_path=registry_path,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    trades = load_trades(trades_path)
    observations = select_eligible_observations(trades, prospective_start=PREREGISTRATION_TIMESTAMP)
    observation_records = [observation_record(row) for _, row in observations.iterrows()]

    observations_path = output_root / "observations.jsonl"
    write_jsonl(observations_path, observation_records)

    summary = summarize_observations(observations, alpha=alpha)
    gates = evaluate_gates(summary)
    status, status_reasons = fast78_status(summary=summary, gates=gates)

    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast78_is_preregistered_observation_only",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_preregistration_and_explicit_authorization",
        ],
        "experiment_stage": "fast78_usdjpy_fomc_observation",
        "status": status,
        "status_reasons": status_reasons,
        "observation_pair": OBSERVATION_PAIR,
        "observation_event": OBSERVATION_EVENT,
        "observation_policy_id": OBSERVATION_POLICY_ID,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": _sha256_file(registry_path),
        "source_fast77_report_sha256": SOURCE_FAST77_REPORT_SHA256,
        "trades_path": str(trades_path),
        "trades_sha256": _sha256_file(trades_path),
        "trade_count": int(len(trades)),
        "eligible_observation_count": len(observation_records),
        "observations_path": str(observations_path),
        "observations_sha256": _sha256_file(observations_path),
        "summary": summary,
        "gates": gates,
    }
    return _write_report(output_root, report)


def validate_parameters(*, alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def validate_preregistration(*, preregistration_path: Path, registry_path: Path) -> dict[str, str]:
    if not preregistration_path.exists():
        raise FileNotFoundError(preregistration_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    preregistration_sha256 = _sha256_file(preregistration_path)
    if preregistration_sha256 != PREREGISTRATION_SHA256:
        raise ValueError("Fast78 preregistration hash changed")
    registry_events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in registry_events
        if event.get("event") == "usdjpy_fomc_observation_preregistration_created"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast78 registry must contain exactly one preregistration event")
    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_fast77_report_sha256": SOURCE_FAST77_REPORT_SHA256,
        "observation_pair": OBSERVATION_PAIR,
        "observation_event": OBSERVATION_EVENT,
        "observation_policy_id": OBSERVATION_POLICY_ID,
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast78 registry mismatch: {mismatches}")
    return {"preregistration_sha256": preregistration_sha256}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    trades = pd.read_parquet(path)
    required = {
        "event_timestamp",
        "pair",
        "event_name",
        "policy_id",
        "side",
        "net_base_pips",
        "net_stress_pips",
        "exit_base_timestamp",
        "exit_base_reason",
        "exit_stress_timestamp",
        "exit_stress_reason",
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing columns: {sorted(missing)}")
    trades = trades.copy()
    trades["event_timestamp"] = pd.to_datetime(trades["event_timestamp"], utc=True)
    trades["exit_base_timestamp"] = pd.to_datetime(trades["exit_base_timestamp"], utc=True)
    trades["exit_stress_timestamp"] = pd.to_datetime(trades["exit_stress_timestamp"], utc=True)
    return trades


def select_eligible_observations(
    trades: pd.DataFrame,
    *,
    prospective_start: pd.Timestamp,
) -> pd.DataFrame:
    mask = (
        (trades["event_timestamp"] > prospective_start)
        & trades["pair"].eq(OBSERVATION_PAIR)
        & trades["event_name"].eq(OBSERVATION_EVENT)
        & trades["policy_id"].eq(OBSERVATION_POLICY_ID)
    )
    observations = trades.loc[mask].sort_values("event_timestamp").copy()
    duplicated = observations["event_timestamp"].duplicated(keep=False)
    if duplicated.any():
        duplicate_timestamps = observations.loc[duplicated, "event_timestamp"].astype(str).tolist()
        raise ValueError(f"Fast78 expected one observation per event timestamp: {duplicate_timestamps}")
    return observations


def observation_record(row: pd.Series) -> dict[str, Any]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "pair": str(row["pair"]),
        "event_name": str(row["event_name"]),
        "policy_id": str(row["policy_id"]),
        "event_timestamp": row["event_timestamp"].isoformat(),
        "side": str(row["side"]),
        "net_base_pips": float(row["net_base_pips"]),
        "net_stress_pips": float(row["net_stress_pips"]),
        "exit_base_timestamp": row["exit_base_timestamp"].isoformat(),
        "exit_base_reason": str(row["exit_base_reason"]),
        "exit_stress_timestamp": row["exit_stress_timestamp"].isoformat(),
        "exit_stress_reason": str(row["exit_stress_reason"]),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")


def summarize_observations(observations: pd.DataFrame, *, alpha: float) -> dict[str, Any]:
    yearly = yearly_summaries(observations, alpha=alpha)
    positive_years = sum(year["mean_stress_pips"] > 0 for year in yearly)
    year_count = len(yearly)
    return {
        "events": int(observations["event_timestamp"].nunique()) if not observations.empty else 0,
        "trades": int(len(observations)),
        "mean_base_pips": safe_mean(observations["net_base_pips"]),
        "mean_stress_pips": safe_mean(observations["net_stress_pips"]),
        "total_stress_pips": safe_sum(observations["net_stress_pips"]),
        "median_stress_pips": safe_median(observations["net_stress_pips"]),
        "lower_stress_pips_uncorrected": lower_confidence_bound(
            observations["net_stress_pips"],
            alpha=alpha,
        ),
        "first_8_total_stress_pips": first_n_total(
            observations["net_stress_pips"],
            EARLY_FAIL_EVENT_COUNT,
        ),
        "yearly": yearly,
        "positive_stress_year_count": positive_years,
        "year_count": year_count,
        "positive_stress_year_rate": positive_years / year_count if year_count else 0.0,
        "has_two_consecutive_negative_years": has_two_consecutive_negative_years(yearly),
    }


def yearly_summaries(observations: pd.DataFrame, *, alpha: float) -> list[dict[str, Any]]:
    if observations.empty:
        return []
    yearly = observations.assign(year=observations["event_timestamp"].dt.year)
    summaries = []
    for year, group in yearly.groupby("year", sort=True):
        summaries.append(
            {
                "year": int(year),
                "events": int(group["event_timestamp"].nunique()),
                "trades": int(len(group)),
                "mean_base_pips": safe_mean(group["net_base_pips"]),
                "mean_stress_pips": safe_mean(group["net_stress_pips"]),
                "total_stress_pips": safe_sum(group["net_stress_pips"]),
                "lower_stress_pips_uncorrected": lower_confidence_bound(
                    group["net_stress_pips"],
                    alpha=alpha,
                ),
            }
        )
    return summaries


def evaluate_gates(summary: dict[str, Any]) -> dict[str, Any]:
    early_fail_total = summary["first_8_total_stress_pips"]
    early_fail = (
        summary["events"] >= EARLY_FAIL_EVENT_COUNT
        and early_fail_total <= EARLY_FAIL_STRESS_TOTAL_PIPS
    )
    consecutive_year_fail = summary["has_two_consecutive_negative_years"]
    minimum_shadow_candidate = (
        summary["events"] >= SHADOW_CANDIDATE_MINIMUM_EVENTS
        and summary["mean_stress_pips"] > SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS
        and summary["mean_base_pips"] > 0
        and summary["lower_stress_pips_uncorrected"] > 0
        and summary["positive_stress_year_rate"] >= SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_RATE
        and not early_fail
        and not consecutive_year_fail
    )
    return {
        "early_fail_event_count": EARLY_FAIL_EVENT_COUNT,
        "early_fail_stress_total_pips": EARLY_FAIL_STRESS_TOTAL_PIPS,
        "early_fail_triggered": early_fail,
        "consecutive_negative_year_fail_triggered": consecutive_year_fail,
        "minimum_shadow_candidate_event_count": SHADOW_CANDIDATE_MINIMUM_EVENTS,
        "minimum_shadow_candidate_stress_mean_pips": SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS,
        "minimum_shadow_candidate_positive_year_rate": (
            SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_RATE
        ),
        "minimum_shadow_candidate_gate_passed": minimum_shadow_candidate,
        "testnet_allowed": False,
        "live_allowed": False,
    }


def fast78_status(
    *,
    summary: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if summary["events"] == 0:
        return "FAST78_OBSERVATION_NO_ELIGIBLE_EVENTS_YET", reasons + [
            "waiting_for_post_preregistration_fomc",
        ]
    if gates["early_fail_triggered"] or gates["consecutive_negative_year_fail_triggered"]:
        return "FAST78_OBSERVATION_FAILED_PREREGISTERED_GATE", reasons + [
            "preregistered_failure_gate_triggered",
        ]
    if gates["minimum_shadow_candidate_gate_passed"]:
        return "FAST78_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW", reasons + [
            "minimum_shadow_candidate_gate_passed",
            "testnet_still_requires_new_preregistration",
        ]
    return "FAST78_OBSERVATION_CONTINUES_COLLECTING", reasons + [
        "minimum_shadow_candidate_gate_not_passed",
    ]


def first_n_total(series: pd.Series, n: int) -> float | None:
    if len(series) < n:
        return None
    return float(series.iloc[:n].astype(float).sum())


def has_two_consecutive_negative_years(yearly: list[dict[str, Any]]) -> bool:
    for left, right in zip(yearly, yearly[1:], strict=False):
        if (
            right["year"] == left["year"] + 1
            and left["mean_stress_pips"] < 0
            and right["mean_stress_pips"] < 0
        ):
            return True
    return False


def safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.astype(float).mean())


def safe_sum(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.astype(float).sum())


def safe_median(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.astype(float).median())


def lower_confidence_bound(values: list[float] | pd.Series | np.ndarray, *, alpha: float | None) -> float | None:
    series = pd.Series(values, dtype="float64").dropna()
    sample_size = len(series)
    if sample_size < 2:
        return None
    standard_error = float(series.std(ddof=1) / np.sqrt(sample_size))
    return float(series.mean() - t.ppf(1.0 - alpha, sample_size - 1) * standard_error)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "eligible_observation_count": report["eligible_observation_count"],
        "summary": report["summary"],
        "gates": report["gates"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
