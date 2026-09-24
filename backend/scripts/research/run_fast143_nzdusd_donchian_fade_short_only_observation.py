"""Fast143 preregistered NZDUSD daily Donchian-fade short-only observation.

Fast143 freezes the Fast142 short-only side filter on top of the Fast134
NZDUSD D1 Donchian fade fixed-1R candidate. It observes prospectively only,
filters only post-preregistration SHORT entries, writes offline evidence, and
never sends orders or commits ledger state.
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

from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import DEFAULT_M1_DIR, sha256_file  # noqa: E402
from scripts.research.run_fast124_usdjpy_compression_observation import (  # noqa: E402
    absolute_path,
    load_jsonl,
    source_m1_files,
    source_validation_summary,
    write_jsonl,
)
from scripts.research.run_fast131_nzdusd_donchian_fade_observation import (  # noqa: E402
    DEFAULT_ALPHA,
    SAFETY_FLAGS,
    TIMESTAMP_COLUMNS,
    TRADE_COLUMNS,
    empty_trade_rows,
    summarize_observations,
    validate_parameters,
)
from scripts.research.run_fast134_nzdusd_donchian_fade_fixed1r_observation import (  # noqa: E402
    FROZEN_ENTRY_POLICY,
    FROZEN_POLICY_ID as BASE_FROZEN_POLICY_ID,
    FROZEN_RISK_POLICY,
    PAIR,
    load_prospective_pair_data,
    materialize_frozen_policy_trades,
)
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import frozen_entry_signals  # noqa: E402

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast143" / "nzdusd-donchian-fade-short-only-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast143-nzdusd-donchian-fade-short-only-prospective-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast143-registry.jsonl"

PREREGISTRATION_COMMIT = "a97e289beebf12281e31e5ec18d2218040e73524"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-24T05:25:14-03:00")
PREREGISTRATION_SHA256 = "4191ff3601d085737f0e04f55b77348c754b0866a1d64ec5a411e928f7a0c253"
SOURCE_FAST142_REPORT_SHA256 = "b0f0cb45b717ec22d4caee6ed570d7cf0e110c222002800ad8a50abeb5d714f6"
SIDE_FILTER = "SHORT"
FROZEN_POLICY_ID = f"{BASE_FROZEN_POLICY_ID}:short_only"

EARLY_FAIL_MINIMUM_TRADES = 8
EARLY_FAIL_MAXIMUM_STRESS_TOTAL_PIPS = -100.0
SHADOW_CANDIDATE_MINIMUM_TRADES = 12
SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS = 30.0
SHADOW_CANDIDATE_MINIMUM_YEARS = 2
SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_SHARE = 2 / 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args(argv)
    report = run_fast143_nzdusd_donchian_fade_short_only_observation(
        m1_dir=arguments.m1_dir,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast143_nzdusd_donchian_fade_short_only_observation(
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
    pair_data = load_prospective_pair_data(m1_dir)
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
    gates = evaluate_fast143_gates(summary=summary, observations=observations)
    status, status_reasons = fast143_status(summary=summary, gates=gates)
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


def validate_preregistration(*, preregistration_path: Path, registry_path: Path) -> dict[str, str]:
    if not preregistration_path.exists():
        raise FileNotFoundError(preregistration_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    preregistration_sha256 = sha256_file(preregistration_path)
    if preregistration_sha256 != PREREGISTRATION_SHA256:
        raise ValueError("Fast143 preregistration hash changed")

    events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in events
        if event.get("event") == "fast143_nzdusd_donchian_fade_short_only_prospective_preregistered"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast143 registry must contain exactly one preregistration event")

    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "fast142_report_sha256": SOURCE_FAST142_REPORT_SHA256,
        "pair": PAIR,
        "entry_policy_id": FROZEN_ENTRY_POLICY.identifier,
        "risk_policy_id": FROZEN_RISK_POLICY.identifier,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "side_filter": SIDE_FILTER,
        "timeframe": "D1",
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast143 registry mismatch: {mismatches}")

    return {"preregistration_sha256": preregistration_sha256}


def select_eligible_observations(
    rows: pd.DataFrame,
    *,
    prospective_start: pd.Timestamp,
) -> pd.DataFrame:
    if rows.empty:
        frame = empty_trade_rows()
        frame["fast143_policy_id"] = pd.Series(dtype="object")
        frame["side_filter"] = pd.Series(dtype="object")
        return frame
    frame = rows.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    mask = (
        (frame["entry_timestamp"] > prospective_start)
        & frame["candidate_id"].eq(BASE_FROZEN_POLICY_ID)
        & frame["pair"].eq(PAIR)
        & frame["entry_policy_id"].eq(FROZEN_ENTRY_POLICY.identifier)
        & frame["risk_policy_id"].eq(FROZEN_RISK_POLICY.identifier)
        & frame["channel_days"].eq(FROZEN_ENTRY_POLICY.channel_days)
        & frame["hold_days"].eq(FROZEN_ENTRY_POLICY.hold_days)
        & frame["stop_atr_multiple"].eq(FROZEN_RISK_POLICY.stop_atr_multiple)
        & frame["management"].eq(FROZEN_RISK_POLICY.management.name)
        & frame["side"].eq(SIDE_FILTER)
    )
    selected = frame.loc[mask].sort_values(["entry_timestamp", "candidate_id"]).copy()
    selected["fast143_policy_id"] = FROZEN_POLICY_ID
    selected["side_filter"] = SIDE_FILTER
    return selected


def observation_record(row: pd.Series) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "pair": str(row["pair"]),
        "candidate_id": str(row["candidate_id"]),
        "fast143_policy_id": str(row["fast143_policy_id"]),
        "side_filter": str(row["side_filter"]),
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


def evaluate_fast143_gates(*, summary: dict[str, Any], observations: pd.DataFrame) -> dict[str, Any]:
    stress = summary["stress"]
    base = summary["base"]
    non_short_observation_count = int((~observations["side"].eq(SIDE_FILTER)).sum()) if not observations.empty else 0

    early_fail_reasons = []
    if (
        stress["trade_count"] >= EARLY_FAIL_MINIMUM_TRADES
        and stress["total_pips"] <= EARLY_FAIL_MAXIMUM_STRESS_TOTAL_PIPS
    ):
        early_fail_reasons.append("early_fail_stress_total_pips")
    if stress["two_consecutive_negative_years"]:
        early_fail_reasons.append("two_consecutive_negative_stress_years")
    if non_short_observation_count:
        early_fail_reasons.append("non_short_observation_detected")

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
    if non_short_observation_count:
        shadow_reasons.append("non_short_observation_detected")

    return {
        "early_fail": bool(early_fail_reasons),
        "early_fail_reasons": early_fail_reasons,
        "shadow_candidate": not shadow_reasons and not early_fail_reasons,
        "shadow_candidate_reasons": shadow_reasons,
        "non_short_observation_count": non_short_observation_count,
        "required_side_filter": SIDE_FILTER,
    }


def fast143_status(*, summary: dict[str, Any], gates: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "short_only_observation"]
    if summary["eligible_observation_count"] == 0:
        return "FAST143_OBSERVATION_NO_ELIGIBLE_SHORT_TRADES_YET", reasons + [
            "no_post_preregistration_short_trades"
        ]
    if gates["early_fail"]:
        return "FAST143_OBSERVATION_FAILED_EARLY", reasons + gates["early_fail_reasons"]
    if gates["shadow_candidate"]:
        return (
            "FAST143_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW",
            reasons + ["shadow_candidate_gate_passed_but_not_testnet_authorized"],
        )
    return (
        "FAST143_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
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
            "fast143_is_preregistered_short_only_observation_only",
            "base_rule_and_side_filter_were_identified_by_historical_research",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast143_nzdusd_donchian_fade_short_only_observation",
        "materialization_source": "raw_m1",
        "status": status,
        "status_reasons": status_reasons,
        "observation_pair": PAIR,
        "base_frozen_policy_id": BASE_FROZEN_POLICY_ID,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "side_filter": SIDE_FILTER,
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
        "source_fast142_report_sha256": SOURCE_FAST142_REPORT_SHA256,
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
