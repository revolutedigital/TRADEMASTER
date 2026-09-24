"""Fast131 preregistered NZDUSD daily Donchian-fade observation runner.

Fast131 freezes the best Fast130 candidate and observes it prospectively only.
It reconstructs the frozen NZDUSD D1 Donchian fade with the fixed 2 ATR / 2R
risk policy, filters only post-preregistration entries, writes offline evidence,
and never sends orders or commits ledger state.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import (  # noqa: E402
    Management,
    plain_json,
    write_json,
)
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    DEFAULT_M1_DIR,
    sha256_file,
)
from scripts.research.run_fast124_usdjpy_compression_observation import (  # noqa: E402
    absolute_path,
    load_jsonl,
    source_m1_files,
    source_validation_summary,
    write_jsonl,
)
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    FROZEN_ENTRY_POLICY,
    PAIR,
    RiskPolicy,
    evaluate_risk_policy,
    frozen_entry_signals,
    load_pair_data,
    lower_confidence_bound,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast131" / "nzdusd-donchian-fade-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast131-nzdusd-donchian-fade-prospective-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast131-registry.jsonl"

PREREGISTRATION_COMMIT = "84de45b417c03ce267b537fe79453670825f0d69"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-24T02:50:09-03:00")
PREREGISTRATION_SHA256 = "6c6293797a319e40acc41ea5e47f5b9beb0a8138a36bdc4e77c0581550f9478a"
SOURCE_FAST129_REPORT_SHA256 = "d6ae2769b745be9d72ead8bd0f24aa6ddfde19092e2a31b919931a1a5c94051f"
SOURCE_FAST130_REPORT_SHA256 = "b6c473330d22aaf8f1f2a01671ade72bb09a3664cb817169acd9fa7a919007e3"

FROZEN_RISK_POLICY = RiskPolicy(
    identifier="stop2_fixed_2r",
    stop_atr_multiple=2.0,
    management=Management("fixed_2r", target_r=2.0, activation_r=None, trail_r=None),
)
FROZEN_POLICY_ID = f"{PAIR}:{FROZEN_ENTRY_POLICY.identifier}:{FROZEN_RISK_POLICY.identifier}"
DEFAULT_ALPHA = 0.05
EARLY_FAIL_MINIMUM_TRADES = 8
EARLY_FAIL_MAXIMUM_STRESS_TOTAL_PIPS = -100.0
SHADOW_CANDIDATE_MINIMUM_TRADES = 20
SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS = 20.0
SHADOW_CANDIDATE_MINIMUM_YEARS = 2
SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_SHARE = 2 / 3
TIMESTAMP_COLUMNS = (
    "signal_timestamp",
    "entry_timestamp",
    "exit_base_timestamp",
    "exit_stress_timestamp",
)
TRADE_COLUMNS = (
    "candidate_id",
    "pair",
    "entry_policy_id",
    "risk_policy_id",
    "channel_days",
    "hold_days",
    "stop_atr_multiple",
    "management",
    "signal_timestamp",
    "entry_timestamp",
    "side",
    "daily_atr_pips",
    "risk_pips",
    "breakout_distance_pips",
    "raw_base_pips",
    "net_base_pips",
    "exit_base_timestamp",
    "exit_base_reason",
    "raw_stress_pips",
    "net_stress_pips",
    "exit_stress_timestamp",
    "exit_stress_reason",
)
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
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args(argv)
    report = run_fast131_nzdusd_donchian_fade_observation(
        m1_dir=arguments.m1_dir,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast131_nzdusd_donchian_fade_observation(
    *,
    m1_dir: Path = DEFAULT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    validate_parameters(alpha=alpha)
    m1_dir = absolute_path(m1_dir)
    output_root = absolute_path(output_root)
    preregistration = validate_preregistration(
        preregistration_path=absolute_path(preregistration_path),
        registry_path=absolute_path(registry_path),
    )
    output_root.mkdir(parents=True, exist_ok=True)

    source_files = source_m1_files(m1_dir, PAIR)
    source_validation = source_validation_summary(m1_dir, PAIR)
    pair_data = load_pair_data(m1_dir)
    signals = frozen_entry_signals(pair_data)
    signals_path = output_root / "entry_signals.parquet"
    signals.to_parquet(signals_path, compression="zstd", index=False)

    materialized_rows = materialize_frozen_policy_trades(pair_data=pair_data, signals=signals)
    materialized_path = output_root / "materialized_trade_rows.parquet"
    materialized_rows.to_parquet(materialized_path, compression="zstd", index=False)

    observations = select_eligible_observations(
        materialized_rows,
        prospective_start=PREREGISTRATION_TIMESTAMP,
    )
    observation_records = [observation_record(row) for _, row in observations.iterrows()]
    observations_path = output_root / "observations.jsonl"
    write_jsonl(observations_path, observation_records)

    summary = summarize_observations(observations, alpha=alpha)
    gates = evaluate_gates(summary)
    status, status_reasons = fast131_status(summary=summary, gates=gates)
    report = build_report(
        preregistration=preregistration,
        m1_dir=m1_dir,
        source_files=source_files,
        source_validation=source_validation,
        pair_quality=pair_data.quality,
        signals=signals,
        signals_path=signals_path,
        materialized_rows=materialized_rows,
        materialized_path=materialized_path,
        observations_path=observations_path,
        observation_records=observation_records,
        summary=summary,
        gates=gates,
        status=status,
        status_reasons=status_reasons,
        preregistration_path=absolute_path(preregistration_path),
        registry_path=absolute_path(registry_path),
    )
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
        raise ValueError("Fast131 preregistration hash changed")

    events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in events
        if event.get("event") == "fast131_nzdusd_donchian_fade_prospective_preregistered"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast131 registry must contain exactly one preregistration event")

    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "fast129_report_sha256": SOURCE_FAST129_REPORT_SHA256,
        "fast130_report_sha256": SOURCE_FAST130_REPORT_SHA256,
        "pair": PAIR,
        "entry_policy_id": FROZEN_ENTRY_POLICY.identifier,
        "risk_policy_id": FROZEN_RISK_POLICY.identifier,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "timeframe": "D1",
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast131 registry mismatch: {mismatches}")

    return {"preregistration_sha256": preregistration_sha256}


def materialize_frozen_policy_trades(*, pair_data: Any, signals: pd.DataFrame) -> pd.DataFrame:
    rows = evaluate_risk_policy(pair_data, signals, FROZEN_RISK_POLICY)
    if not rows:
        return empty_trade_rows()
    frame = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    for column in TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values(["entry_timestamp", "candidate_id"], ignore_index=True)


def empty_trade_rows() -> pd.DataFrame:
    frame = pd.DataFrame(columns=TRADE_COLUMNS)
    for column in TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def select_eligible_observations(
    rows: pd.DataFrame,
    *,
    prospective_start: pd.Timestamp,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    frame = rows.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    mask = (
        (frame["entry_timestamp"] > prospective_start)
        & frame["candidate_id"].eq(FROZEN_POLICY_ID)
        & frame["pair"].eq(PAIR)
        & frame["entry_policy_id"].eq(FROZEN_ENTRY_POLICY.identifier)
        & frame["risk_policy_id"].eq(FROZEN_RISK_POLICY.identifier)
        & frame["channel_days"].eq(FROZEN_ENTRY_POLICY.channel_days)
        & frame["hold_days"].eq(FROZEN_ENTRY_POLICY.hold_days)
        & frame["stop_atr_multiple"].eq(FROZEN_RISK_POLICY.stop_atr_multiple)
        & frame["management"].eq(FROZEN_RISK_POLICY.management.name)
    )
    return frame.loc[mask].sort_values(["entry_timestamp", "candidate_id"]).copy()


def observation_record(row: pd.Series) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "pair": str(row["pair"]),
        "candidate_id": str(row["candidate_id"]),
        "entry_policy_id": str(row["entry_policy_id"]),
        "risk_policy_id": str(row["risk_policy_id"]),
        "channel_days": int(row["channel_days"]),
        "hold_days": int(row["hold_days"]),
        "stop_atr_multiple": float(row["stop_atr_multiple"]),
        "management": str(row["management"]),
        "signal_timestamp": pd.Timestamp(row["signal_timestamp"]).isoformat(),
        "entry_timestamp": pd.Timestamp(row["entry_timestamp"]).isoformat(),
        "side": str(row["side"]),
        "daily_atr_pips": float(row["daily_atr_pips"]),
        "risk_pips": float(row["risk_pips"]),
        "breakout_distance_pips": float(row["breakout_distance_pips"]),
        "raw_base_pips": float(row["raw_base_pips"]),
        "net_base_pips": float(row["net_base_pips"]),
        "exit_base_timestamp": pd.Timestamp(row["exit_base_timestamp"]).isoformat(),
        "exit_base_reason": str(row["exit_base_reason"]),
        "raw_stress_pips": float(row["raw_stress_pips"]),
        "net_stress_pips": float(row["net_stress_pips"]),
        "exit_stress_timestamp": pd.Timestamp(row["exit_stress_timestamp"]).isoformat(),
        "exit_stress_reason": str(row["exit_stress_reason"]),
    }


def summarize_observations(observations: pd.DataFrame, *, alpha: float) -> dict[str, Any]:
    base = metric_summary(observations, column="net_base_pips", alpha=alpha)
    stress = metric_summary(observations, column="net_stress_pips", alpha=alpha)
    return {
        "base": base,
        "stress": stress,
        "eligible_observation_count": int(len(observations)),
    }


def metric_summary(observations: pd.DataFrame, *, column: str, alpha: float) -> dict[str, Any]:
    if observations.empty:
        return {
            "column": column,
            "trade_count": 0,
            "total_pips": 0.0,
            "mean_pips": None,
            "median_pips": None,
            "lower_confidence_bound_pips": None,
            "year_count": 0,
            "positive_year_count": 0,
            "positive_year_share": None,
            "two_consecutive_negative_years": False,
            "yearly": [],
        }
    frame = observations.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    values = frame[column].astype("float64")
    yearly = yearly_summaries(frame, column=column)
    positive_years = sum(year["mean_pips"] > 0 for year in yearly)
    return {
        "column": column,
        "trade_count": int(len(frame)),
        "total_pips": float(values.sum()),
        "mean_pips": float(values.mean()),
        "median_pips": float(values.median()),
        "lower_confidence_bound_pips": lower_confidence_bound(values, alpha),
        "year_count": int(len(yearly)),
        "positive_year_count": int(positive_years),
        "positive_year_share": positive_years / len(yearly) if yearly else None,
        "two_consecutive_negative_years": has_two_consecutive_negative_years(yearly),
        "yearly": yearly,
    }


def yearly_summaries(observations: pd.DataFrame, *, column: str) -> list[dict[str, Any]]:
    if observations.empty:
        return []
    grouped = (
        observations.assign(year=observations["entry_timestamp"].dt.year)
        .groupby("year", sort=True)[column]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    return [
        {
            "year": int(row["year"]),
            "trade_count": int(row["size"]),
            "total_pips": float(row["sum"]),
            "mean_pips": float(row["mean"]),
        }
        for row in grouped.to_dict("records")
    ]


def has_two_consecutive_negative_years(yearly: list[dict[str, Any]]) -> bool:
    previous_negative = False
    for row in yearly:
        current_negative = row["mean_pips"] < 0
        if previous_negative and current_negative:
            return True
        previous_negative = current_negative
    return False


def evaluate_gates(summary: dict[str, Any]) -> dict[str, Any]:
    stress = summary["stress"]
    base = summary["base"]

    early_fail_reasons = []
    if (
        stress["trade_count"] >= EARLY_FAIL_MINIMUM_TRADES
        and stress["total_pips"] <= EARLY_FAIL_MAXIMUM_STRESS_TOTAL_PIPS
    ):
        early_fail_reasons.append("early_fail_stress_total_pips")
    if stress["two_consecutive_negative_years"]:
        early_fail_reasons.append("two_consecutive_negative_stress_years")

    shadow_reasons = []
    if stress["trade_count"] < SHADOW_CANDIDATE_MINIMUM_TRADES:
        shadow_reasons.append("insufficient_stress_trades")
    if stress["mean_pips"] is None or stress["mean_pips"] <= SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS:
        shadow_reasons.append("stress_mean_below_threshold")
    if stress["lower_confidence_bound_pips"] is None or stress["lower_confidence_bound_pips"] <= 0:
        shadow_reasons.append("stress_lower_confidence_bound_not_positive")
    if base["mean_pips"] is None or base["mean_pips"] <= 0:
        shadow_reasons.append("base_mean_not_positive")
    if stress["year_count"] < SHADOW_CANDIDATE_MINIMUM_YEARS:
        shadow_reasons.append("insufficient_prospective_years")
    if (
        stress["positive_year_share"] is None
        or stress["positive_year_share"] < SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_SHARE
    ):
        shadow_reasons.append("positive_year_share_below_threshold")

    return {
        "early_fail": bool(early_fail_reasons),
        "early_fail_reasons": early_fail_reasons,
        "shadow_candidate": not shadow_reasons and not early_fail_reasons,
        "shadow_candidate_reasons": shadow_reasons,
    }


def fast131_status(*, summary: dict[str, Any], gates: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if summary["eligible_observation_count"] == 0:
        return "FAST131_OBSERVATION_NO_ELIGIBLE_TRADES_YET", reasons + [
            "no_post_preregistration_trades"
        ]
    if gates["early_fail"]:
        return "FAST131_OBSERVATION_FAILED_EARLY", reasons + gates["early_fail_reasons"]
    if gates["shadow_candidate"]:
        return (
            "FAST131_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW",
            reasons + ["shadow_candidate_gate_passed_but_not_testnet_authorized"],
        )
    return (
        "FAST131_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
        reasons + gates["shadow_candidate_reasons"],
    )


def build_report(
    *,
    preregistration: dict[str, str],
    m1_dir: Path,
    source_files: list[dict[str, Any]],
    source_validation: dict[str, Any],
    pair_quality: dict[str, Any],
    signals: pd.DataFrame,
    signals_path: Path,
    materialized_rows: pd.DataFrame,
    materialized_path: Path,
    observations_path: Path,
    observation_records: list[dict[str, Any]],
    summary: dict[str, Any],
    gates: dict[str, Any],
    status: str,
    status_reasons: list[str],
    preregistration_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast131_is_preregistered_observation_only",
            "frozen_rule_was_identified_by_historical_research",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast131_nzdusd_donchian_fade_observation",
        "materialization_source": "raw_m1",
        "status": status,
        "status_reasons": status_reasons,
        "observation_pair": PAIR,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "frozen_entry_policy": asdict(FROZEN_ENTRY_POLICY),
        "frozen_risk_policy": {
            **asdict(FROZEN_RISK_POLICY),
            "management": asdict(FROZEN_RISK_POLICY.management),
        },
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "source_fast129_report_sha256": SOURCE_FAST129_REPORT_SHA256,
        "source_fast130_report_sha256": SOURCE_FAST130_REPORT_SHA256,
        "m1_dir": str(m1_dir),
        "raw_m1_files": source_files,
        "raw_m1_validation": source_validation,
        "pair_quality": pair_quality,
        "raw_m1_row_count": int(pair_quality["usable_bars"]),
        "raw_m1_start": pair_quality["first_bar"],
        "raw_m1_end": pair_quality["last_bar"],
        "dropped_crossed_quote_bars": int(pair_quality["dropped_crossed_quote_bars"]),
        "signal_count": int(len(signals)),
        "signals_path": str(signals_path),
        "signals_sha256": sha256_file(signals_path),
        "materialized_trade_rows_path": str(materialized_path),
        "materialized_trade_rows_sha256": sha256_file(materialized_path),
        "materialized_trade_row_count": int(len(materialized_rows)),
        "eligible_observation_count": len(observation_records),
        "observations_path": str(observations_path),
        "observations_sha256": sha256_file(observations_path),
        "summary": summary,
        "gates": gates,
    }


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    write_json(report_path, payload)
    return plain_json({**payload, "report_sha256": sha256_file(report_path)})


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "raw_m1_start": report["raw_m1_start"],
        "raw_m1_end": report["raw_m1_end"],
        "raw_m1_row_count": report["raw_m1_row_count"],
        "signal_count": report["signal_count"],
        "materialized_trade_row_count": report["materialized_trade_row_count"],
        "eligible_observation_count": report["eligible_observation_count"],
        "summary": report["summary"],
        "gates": report["gates"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
