"""Fast112 preregistered USDJPY London-breakout observation runner.

Fast112 freezes the best Fast111 weak line and observes it prospectively only:
USDJPY F1a reward/risk 2.5, London entry hour 08, and stop_pips above the S0
wide threshold. This runner validates the preregistration artifact, filters only
post-preregistration rows, writes local observation evidence, and never sends
orders or commits ledger state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import t

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_TRADE_ROWS_PATH = (
    REPO_ROOT / "backend" / "data" / "lab_fast111" / "usdjpy-breakout-regime-scout" / "trade_rows.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast112" / "usdjpy-breakout-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast112-usdjpy-breakout-prospective-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast112-registry.jsonl"

PREREGISTRATION_COMMIT = "e9c77709d66136a82ec875acdb63872c558ea102"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-24T02:59:14Z")
PREREGISTRATION_SHA256 = "c9bcf46b615ac2572efd806c44a342b9976fc60613706a97f2d238a2fd5fd0f4"
SOURCE_FAST111_REPORT_SHA256 = "29967566629b6764a42be9289de02bdf835ba8df055fcc138034b6cedefe0f4b"
OBSERVATION_PAIR = "USDJPY"
OBSERVATION_CONFIG = "F1a.rr2.5.w0.6-2.0.se11:00.x16:30"
OBSERVATION_FILTER = "london_entry_hour=8&&stop_pips>94.88491666666664"
STOP_PIPS_WIDE_THRESHOLD = 94.88491666666664
STRESS_SCENARIO = "stress"
BASE_SCENARIO = "fusion_zero"
DEFAULT_ALPHA = 0.05
EARLY_FAIL_TRADE_COUNT = 12
EARLY_FAIL_STRESS_TOTAL_R = -3.0
SHADOW_CANDIDATE_MINIMUM_TRADES = 50
SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_R = 0.05
SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_RATE = 0.60


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-rows-path", type=Path, default=DEFAULT_TRADE_ROWS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args()
    report = run_fast112_usdjpy_breakout_observation(
        trade_rows_path=arguments.trade_rows_path,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast112_usdjpy_breakout_observation(
    *,
    trade_rows_path: Path = DEFAULT_TRADE_ROWS_PATH,
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
    trade_rows = load_trade_rows(trade_rows_path)
    observations = select_eligible_observations(
        trade_rows,
        prospective_start=PREREGISTRATION_TIMESTAMP,
    )
    observation_records = [observation_record(row) for _, row in observations.iterrows()]

    observations_path = output_root / "observations.jsonl"
    write_jsonl(observations_path, observation_records)

    summary = summarize_observations(observations, alpha=alpha)
    gates = evaluate_gates(summary)
    status, status_reasons = fast112_status(summary=summary, gates=gates)
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast112_is_preregistered_observation_only",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast112_usdjpy_breakout_observation",
        "status": status,
        "status_reasons": status_reasons,
        "observation_pair": OBSERVATION_PAIR,
        "observation_config": OBSERVATION_CONFIG,
        "observation_filter": OBSERVATION_FILTER,
        "stop_pips_wide_threshold": STOP_PIPS_WIDE_THRESHOLD,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "source_fast111_report_sha256": SOURCE_FAST111_REPORT_SHA256,
        "trade_rows_path": str(trade_rows_path),
        "trade_rows_sha256": sha256_file(trade_rows_path),
        "trade_row_count": int(len(trade_rows)),
        "eligible_observation_count": len(observation_records),
        "observations_path": str(observations_path),
        "observations_sha256": sha256_file(observations_path),
        "summary": summary,
        "gates": gates,
    }
    return write_report(output_root, report)


def validate_parameters(*, alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def validate_preregistration(*, preregistration_path: Path, registry_path: Path) -> dict[str, str]:
    if not preregistration_path.exists():
        raise FileNotFoundError(preregistration_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    preregistration_sha256 = sha256_file(preregistration_path)
    if preregistration_sha256 != PREREGISTRATION_SHA256:
        raise ValueError("Fast112 preregistration hash changed")
    registry_events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in registry_events
        if event.get("event") == "fast112_usdjpy_breakout_prospective_preregistered"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast112 registry must contain exactly one preregistration event")
    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_fast111_report_sha256": SOURCE_FAST111_REPORT_SHA256,
        "pair": OBSERVATION_PAIR,
        "config": OBSERVATION_CONFIG,
        "filter": OBSERVATION_FILTER,
        "stop_pips_wide_threshold": STOP_PIPS_WIDE_THRESHOLD,
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast112 registry mismatch: {mismatches}")
    return {"preregistration_sha256": preregistration_sha256}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_trade_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_parquet(path)
    required = {
        "stage",
        "config",
        "pair",
        "scenario",
        "entry_time",
        "london_entry_hour",
        "stop_pips",
        "r_multiple",
        "side_label",
        "year",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"trade rows missing columns: {sorted(missing)}")
    rows = rows.copy()
    rows["entry_time"] = pd.to_datetime(rows["entry_time"], utc=True)
    return rows


def select_eligible_observations(
    rows: pd.DataFrame,
    *,
    prospective_start: pd.Timestamp,
) -> pd.DataFrame:
    mask = (
        (rows["entry_time"] > prospective_start)
        & rows["pair"].eq(OBSERVATION_PAIR)
        & rows["config"].eq(OBSERVATION_CONFIG)
        & rows["london_entry_hour"].eq(8)
        & (rows["stop_pips"] > STOP_PIPS_WIDE_THRESHOLD)
    )
    observations = rows.loc[mask].sort_values(["entry_time", "scenario"]).copy()
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
        "config": str(row["config"]),
        "scenario": str(row["scenario"]),
        "entry_time": row["entry_time"].isoformat(),
        "side_label": str(row["side_label"]),
        "london_entry_hour": int(row["london_entry_hour"]),
        "stop_pips": float(row["stop_pips"]),
        "r_multiple": float(row["r_multiple"]),
        "year": int(row["year"]),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")


def summarize_observations(observations: pd.DataFrame, *, alpha: float) -> dict[str, Any]:
    base = scenario_summary(observations, scenario=BASE_SCENARIO, alpha=alpha)
    stress = scenario_summary(observations, scenario=STRESS_SCENARIO, alpha=alpha)
    return {
        "base": base,
        "stress": stress,
        "eligible_base_count": base["trade_count"],
        "eligible_stress_count": stress["trade_count"],
    }


def scenario_summary(observations: pd.DataFrame, *, scenario: str, alpha: float) -> dict[str, Any]:
    rows = observations.loc[observations["scenario"].eq(scenario)]
    yearly = yearly_summaries(rows)
    values = rows["r_multiple"].to_numpy(dtype=float)
    lower = lower_confidence_bound(values, alpha=alpha)
    positive_years = sum(year["mean_r"] > 0 for year in yearly)
    year_count = len(yearly)
    return {
        "scenario": scenario,
        "trade_count": int(len(rows)),
        "total_r": float(values.sum()) if len(values) else 0.0,
        "mean_r": float(values.mean()) if len(values) else None,
        "lower_confidence_bound_r": lower,
        "year_count": year_count,
        "positive_year_count": positive_years,
        "positive_year_rate": positive_years / year_count if year_count else None,
        "two_consecutive_negative_years": has_two_consecutive_negative_years(yearly),
        "yearly": yearly,
    }


def yearly_summaries(rows: pd.DataFrame) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    grouped = rows.groupby("year", sort=True)["r_multiple"].agg(["size", "sum", "mean"]).reset_index()
    return [
        {
            "year": int(row["year"]),
            "trade_count": int(row["size"]),
            "total_r": float(row["sum"]),
            "mean_r": float(row["mean"]),
        }
        for row in grouped.to_dict("records")
    ]


def lower_confidence_bound(values, *, alpha: float) -> float | None:
    count = len(values)
    if count < 2:
        return None
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return mean - float(t.ppf(1 - alpha, count - 1)) * std / (count**0.5)


def has_two_consecutive_negative_years(yearly: list[dict[str, Any]]) -> bool:
    previous_negative = False
    for year in yearly:
        current_negative = year["mean_r"] < 0
        if previous_negative and current_negative:
            return True
        previous_negative = current_negative
    return False


def evaluate_gates(summary: dict[str, Any]) -> dict[str, Any]:
    stress = summary["stress"]
    base = summary["base"]
    early_fail_reasons = []
    if stress["trade_count"] >= EARLY_FAIL_TRADE_COUNT and stress["total_r"] <= EARLY_FAIL_STRESS_TOTAL_R:
        early_fail_reasons.append("early_fail_stress_total_r")
    if stress["two_consecutive_negative_years"]:
        early_fail_reasons.append("two_consecutive_negative_stress_years")

    shadow_reasons = []
    if stress["trade_count"] < SHADOW_CANDIDATE_MINIMUM_TRADES:
        shadow_reasons.append("insufficient_stress_trades")
    if stress["mean_r"] is None or stress["mean_r"] <= SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_R:
        shadow_reasons.append("stress_mean_below_threshold")
    if base["mean_r"] is None or base["mean_r"] <= 0:
        shadow_reasons.append("base_mean_not_positive")
    if stress["lower_confidence_bound_r"] is None or stress["lower_confidence_bound_r"] <= 0:
        shadow_reasons.append("stress_lower_confidence_bound_not_positive")
    if (
        stress["positive_year_rate"] is None
        or stress["positive_year_rate"] < SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_RATE
    ):
        shadow_reasons.append("positive_year_rate_below_threshold")

    return {
        "early_fail": not not early_fail_reasons,
        "early_fail_reasons": early_fail_reasons,
        "shadow_candidate": not shadow_reasons and not early_fail_reasons,
        "shadow_candidate_reasons": shadow_reasons,
    }


def fast112_status(*, summary: dict[str, Any], gates: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if summary["eligible_stress_count"] == 0:
        return "FAST112_OBSERVATION_NO_ELIGIBLE_TRADES_YET", reasons + ["no_post_preregistration_trades"]
    if gates["early_fail"]:
        return "FAST112_OBSERVATION_FAILED_EARLY", reasons + gates["early_fail_reasons"]
    if gates["shadow_candidate"]:
        return (
            "FAST112_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW",
            reasons + ["shadow_candidate_gate_passed_but_not_testnet_authorized"],
        )
    return (
        "FAST112_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
        reasons + gates["shadow_candidate_reasons"],
    )


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def sha256_file(path: Path) -> str:
    import hashlib

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
        "eligible_observation_count": report["eligible_observation_count"],
        "summary": report["summary"],
        "gates": report["gates"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
