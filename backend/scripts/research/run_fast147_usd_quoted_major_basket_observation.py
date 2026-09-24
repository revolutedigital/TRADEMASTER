"""Fast147 preregistered USD-quoted major basket observation runner.

Fast147 freezes the Fast146 basket:

    AUDUSD, EURUSD, GBPUSD, NZDUSD
    d1_donchian_fade_ch100_hold20:stop2_fixed_1r:SHORT

It observes prospectively only, filters only post-preregistration SHORT entries,
writes offline evidence, and never sends orders or commits ledger state.
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

from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    DEFAULT_M1_DIR,
    load_market,
    sha256_file,
)
from scripts.research.run_fast124_usdjpy_compression_observation import (  # noqa: E402
    absolute_path,
    load_jsonl,
    source_m1_files,
    source_validation_summary,
    write_jsonl,
)
from scripts.research.run_fast129_fx_daily_donchian_fade_scout import (  # noqa: E402
    policy_grid as entry_policy_grid,
)
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    risk_policy_grid,
)
from scripts.research.run_fast131_nzdusd_donchian_fade_observation import (  # noqa: E402
    DEFAULT_ALPHA,
    SAFETY_FLAGS,
    TIMESTAMP_COLUMNS,
    TRADE_COLUMNS,
    summarize_observations,
    validate_parameters,
)
from scripts.research.run_fast133_fx_daily_donchian_fade_global_risk import (  # noqa: E402
    enrich_signals_with_daily_atr,
    evaluate_pair_policy,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast147" / "usd-quoted-major-basket-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast147-usd-quoted-major-basket-prospective-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast147-registry.jsonl"

PREREGISTRATION_COMMIT = "4187adf0892ffdbd698a42513b7b0c95c4883dce"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-24T05:49:20-03:00")
PREREGISTRATION_SHA256 = "0af7aa3b1fe11f3b21cecb480886fbc062de8e05bd3f325704b41b3b67e60646"
SOURCE_FAST146_REPORT_SHA256 = "450971147b555a83151ec99bf9e5d50f26d99c68868a3257ff79170468cad6a2"
GROUP_ID = "usd_quoted_majors"
TARGET_PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "NZDUSD")
ENTRY_POLICY_ID = "d1_donchian_fade_ch100_hold20"
RISK_POLICY_ID = "stop2_fixed_1r"
SIDE_FILTER = "SHORT"
TARGET_POLICY_ID = f"basket:{GROUP_ID}:{ENTRY_POLICY_ID}:{RISK_POLICY_ID}:short_only"

EARLY_FAIL_MINIMUM_TRADES = 20
EARLY_FAIL_MAXIMUM_STRESS_TOTAL_PIPS = -250.0
SHADOW_CANDIDATE_MINIMUM_TRADES = 40
SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS = 25.0
SHADOW_CANDIDATE_MINIMUM_YEARS = 2
SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_SHARE = 2 / 3
SHADOW_CANDIDATE_MINIMUM_NONNEGATIVE_PAIRS = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args(argv)
    report = run_fast147_usd_quoted_major_basket_observation(
        m1_dir=arguments.m1_dir,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast147_usd_quoted_major_basket_observation(
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

    source_files = {pair: source_m1_files(m1_dir, pair) for pair in TARGET_PAIRS}
    source_validation = {pair: source_validation_summary(m1_dir, pair) for pair in TARGET_PAIRS}
    market = load_market(m1_dir)
    pair_quality = {pair: market[pair].quality for pair in TARGET_PAIRS}
    entry_policy = frozen_entry_policy()
    risk_policy = frozen_risk_policy()

    signal_frames: list[pd.DataFrame] = []
    trade_rows: list[dict[str, Any]] = []
    for pair in TARGET_PAIRS:
        signals = enrich_signals_with_daily_atr(pair, market[pair], entry_policy)
        if not signals.empty:
            signals = signals.copy()
            signals["pair"] = pair
        signal_frames.append(signals)
        trade_rows.extend(evaluate_pair_policy(pair, market[pair], entry_policy, risk_policy, signals))

    signals = pd.concat(signal_frames, ignore_index=True) if signal_frames else pd.DataFrame()
    signals_path = output_root / "entry_signals.parquet"
    signals.to_parquet(signals_path, compression="zstd", index=False)

    materialized_rows = materialize_trade_rows(trade_rows)
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
    pair_summary = summarize_pairs(observations)
    gates = evaluate_fast147_gates(summary=summary, observations=observations, pair_summary=pair_summary)
    status, status_reasons = fast147_status(summary=summary, gates=gates)
    report = build_report(
        preregistration=preregistration,
        m1_dir=m1_dir,
        source_files=source_files,
        source_validation=source_validation,
        pair_quality=pair_quality,
        signals=signals,
        signals_path=signals_path,
        materialized_rows=materialized_rows,
        materialized_path=materialized_path,
        observations_path=observations_path,
        observation_records=observation_records,
        summary=summary,
        pair_summary=pair_summary,
        gates=gates,
        status=status,
        status_reasons=status_reasons,
        preregistration_path=absolute_path(preregistration_path),
        registry_path=absolute_path(registry_path),
    )
    return write_report(output_root, report)


def frozen_entry_policy() -> Any:
    return next(policy for policy in entry_policy_grid() if policy.identifier == ENTRY_POLICY_ID)


def frozen_risk_policy() -> Any:
    return next(policy for policy in risk_policy_grid() if policy.identifier == RISK_POLICY_ID)


def validate_preregistration(*, preregistration_path: Path, registry_path: Path) -> dict[str, str]:
    if not preregistration_path.exists():
        raise FileNotFoundError(preregistration_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    preregistration_sha256 = sha256_file(preregistration_path)
    if preregistration_sha256 != PREREGISTRATION_SHA256:
        raise ValueError("Fast147 preregistration hash changed")

    events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in events
        if event.get("event") == "fast147_usd_quoted_major_basket_prospective_preregistered"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast147 registry must contain exactly one preregistration event")

    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "fast146_report_sha256": SOURCE_FAST146_REPORT_SHA256,
        "group_id": GROUP_ID,
        "pairs": list(TARGET_PAIRS),
        "entry_policy_id": ENTRY_POLICY_ID,
        "risk_policy_id": RISK_POLICY_ID,
        "target_policy_id": TARGET_POLICY_ID,
        "side_filter": SIDE_FILTER,
        "timeframe": "D1",
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast147 registry mismatch: {mismatches}")

    return {"preregistration_sha256": preregistration_sha256}


def materialize_trade_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
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
        frame = empty_trade_rows()
        frame["fast147_policy_id"] = pd.Series(dtype="object")
        frame["group_id"] = pd.Series(dtype="object")
        return frame
    frame = rows.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    mask = (
        (frame["entry_timestamp"] > prospective_start)
        & frame["pair"].isin(TARGET_PAIRS)
        & frame["entry_policy_id"].eq(ENTRY_POLICY_ID)
        & frame["risk_policy_id"].eq(RISK_POLICY_ID)
        & frame["channel_days"].eq(frozen_entry_policy().channel_days)
        & frame["hold_days"].eq(frozen_entry_policy().hold_days)
        & frame["stop_atr_multiple"].eq(frozen_risk_policy().stop_atr_multiple)
        & frame["management"].eq(frozen_risk_policy().management.name)
        & frame["side"].eq(SIDE_FILTER)
    )
    selected = frame.loc[mask].sort_values(["entry_timestamp", "pair", "candidate_id"]).copy()
    selected["fast147_policy_id"] = TARGET_POLICY_ID
    selected["group_id"] = GROUP_ID
    return selected


def observation_record(row: pd.Series) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "pair": str(row["pair"]),
        "candidate_id": str(row["candidate_id"]),
        "fast147_policy_id": str(row["fast147_policy_id"]),
        "group_id": str(row["group_id"]),
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


def summarize_pairs(observations: pd.DataFrame) -> list[dict[str, Any]]:
    if observations.empty:
        return []
    rows: list[dict[str, Any]] = []
    for pair, sample in observations.groupby("pair", sort=True):
        stress = sample["net_stress_pips"].astype("float64")
        rows.append(
            {
                "pair": str(pair),
                "trade_count": int(len(sample)),
                "total_stress_pips": float(stress.sum()),
                "mean_stress_pips": float(stress.mean()),
                "nonnegative_mean_stress": bool(stress.mean() >= 0),
            }
        )
    return rows


def evaluate_fast147_gates(
    *,
    summary: dict[str, Any],
    observations: pd.DataFrame,
    pair_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    stress = summary["stress"]
    base = summary["base"]
    non_short_observation_count = int((~observations["side"].eq(SIDE_FILTER)).sum()) if not observations.empty else 0
    invalid_pair_count = int((~observations["pair"].isin(TARGET_PAIRS)).sum()) if not observations.empty else 0
    nonnegative_pair_count = int(sum(item["nonnegative_mean_stress"] for item in pair_summary))

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
    if invalid_pair_count:
        early_fail_reasons.append("invalid_pair_observation_detected")

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
    if nonnegative_pair_count < SHADOW_CANDIDATE_MINIMUM_NONNEGATIVE_PAIRS:
        shadow_reasons.append("insufficient_nonnegative_pairs")
    if non_short_observation_count:
        shadow_reasons.append("non_short_observation_detected")
    if invalid_pair_count:
        shadow_reasons.append("invalid_pair_observation_detected")

    return {
        "early_fail": bool(early_fail_reasons),
        "early_fail_reasons": early_fail_reasons,
        "shadow_candidate": not shadow_reasons and not early_fail_reasons,
        "shadow_candidate_reasons": shadow_reasons,
        "non_short_observation_count": non_short_observation_count,
        "invalid_pair_count": invalid_pair_count,
        "nonnegative_pair_count": nonnegative_pair_count,
        "required_side_filter": SIDE_FILTER,
        "target_pairs": list(TARGET_PAIRS),
    }


def fast147_status(*, summary: dict[str, Any], gates: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "basket_observation"]
    if summary["eligible_observation_count"] == 0:
        return "FAST147_OBSERVATION_NO_ELIGIBLE_BASKET_TRADES_YET", reasons + [
            "no_post_preregistration_basket_trades"
        ]
    if gates["early_fail"]:
        return "FAST147_OBSERVATION_FAILED_EARLY", reasons + gates["early_fail_reasons"]
    if gates["shadow_candidate"]:
        return (
            "FAST147_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW",
            reasons + ["shadow_candidate_gate_passed_but_not_testnet_authorized"],
        )
    return (
        "FAST147_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
        reasons + gates["shadow_candidate_reasons"],
    )


def raw_m1_bounds(pair_quality: dict[str, dict[str, Any]]) -> dict[str, Any]:
    starts = [quality["first_bar"] for quality in pair_quality.values() if quality["first_bar"]]
    ends = [quality["last_bar"] for quality in pair_quality.values() if quality["last_bar"]]
    return {
        "raw_m1_row_count": int(sum(quality["usable_bars"] for quality in pair_quality.values())),
        "raw_m1_start": min(starts) if starts else None,
        "raw_m1_end": max(ends) if ends else None,
        "dropped_crossed_quote_bars": int(
            sum(quality["dropped_crossed_quote_bars"] for quality in pair_quality.values())
        ),
    }


def build_report(
    *,
    preregistration: dict[str, str],
    m1_dir: Path,
    source_files: dict[str, list[dict[str, Any]]],
    source_validation: dict[str, dict[str, Any]],
    pair_quality: dict[str, dict[str, Any]],
    signals: pd.DataFrame,
    signals_path: Path,
    materialized_rows: pd.DataFrame,
    materialized_path: Path,
    observations_path: Path,
    observation_records: list[dict[str, Any]],
    summary: dict[str, Any],
    pair_summary: list[dict[str, Any]],
    gates: dict[str, Any],
    status: str,
    status_reasons: list[str],
    preregistration_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    bounds = raw_m1_bounds(pair_quality)
    return {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast147_is_preregistered_basket_observation_only",
            "basket_rule_was_identified_by_historical_research",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast147_usd_quoted_major_basket_observation",
        "materialization_source": "raw_m1",
        "status": status,
        "status_reasons": status_reasons,
        "group_id": GROUP_ID,
        "target_pairs": list(TARGET_PAIRS),
        "target_policy_id": TARGET_POLICY_ID,
        "side_filter": SIDE_FILTER,
        "entry_policy_id": ENTRY_POLICY_ID,
        "risk_policy_id": RISK_POLICY_ID,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "source_fast146_report_sha256": SOURCE_FAST146_REPORT_SHA256,
        "m1_dir": str(m1_dir),
        "raw_m1_files": source_files,
        "raw_m1_validation": source_validation,
        "pair_quality": pair_quality,
        **bounds,
        "signal_count": int(len(signals)),
        "signal_counts_by_pair": signals.groupby("pair").size().to_dict() if "pair" in signals.columns else {},
        "signals_path": str(signals_path),
        "signals_sha256": sha256_file(signals_path),
        "materialized_trade_rows_path": str(materialized_path),
        "materialized_trade_rows_sha256": sha256_file(materialized_path),
        "materialized_trade_row_count": int(len(materialized_rows)),
        "materialized_trade_counts_by_pair": materialized_rows.groupby("pair").size().to_dict()
        if not materialized_rows.empty
        else {},
        "eligible_observation_count": len(observation_records),
        "observations_path": str(observations_path),
        "observations_sha256": sha256_file(observations_path),
        "summary": summary,
        "pair_summary": pair_summary,
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
        "signal_counts_by_pair": report["signal_counts_by_pair"],
        "materialized_trade_row_count": report["materialized_trade_row_count"],
        "materialized_trade_counts_by_pair": report["materialized_trade_counts_by_pair"],
        "eligible_observation_count": report["eligible_observation_count"],
        "summary": report["summary"],
        "pair_summary": report["pair_summary"],
        "gates": report["gates"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
