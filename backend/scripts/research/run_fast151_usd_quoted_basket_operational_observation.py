"""Fast151 preregistered USD-quoted basket operational observation runner.

Fast151 freezes the first complete operating policy from this research line:

    AUDUSD, EURUSD, GBPUSD, NZDUSD
    D1 Donchian fade ch100 hold20
    stop2_fixed_1r
    SHORT only
    max_per_pair_1
    dd800_cooldown180d

It observes prospectively only, initializes portfolio-control state at the
Fast151 preregistration timestamp, writes offline evidence, and never sends
orders or commits ledger state.
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
    sha256_file,
)
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
    TRADE_COLUMNS,
    summarize_observations,
    validate_parameters,
)
from scripts.research.run_fast133_fx_daily_donchian_fade_global_risk import (  # noqa: E402
    enrich_signals_with_daily_atr,
    evaluate_pair_policy,
)
from scripts.research.run_fast147_usd_quoted_major_basket_observation import (  # noqa: E402
    empty_trade_rows,
    frozen_entry_policy,
    frozen_risk_policy,
    load_prospective_market,
    materialize_trade_rows,
    raw_m1_bounds,
)
from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import (  # noqa: E402
    ENTRY_POLICY_ID,
    GROUP_ID,
    MANAGEMENT,
    RISK_POLICY_ID,
    SIDE_FILTER,
    TARGET_PAIRS,
    TARGET_POLICY_ID,
    summarize_sample,
)
from scripts.research.run_fast149_usd_quoted_basket_position_cap_audit import (  # noqa: E402
    PositionOverlay,
    apply_position_overlay,
)
from scripts.research.run_fast150_usd_quoted_basket_circuit_breaker_audit import (  # noqa: E402
    CircuitBreakerPolicy,
    apply_drawdown_circuit_breaker,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast151"
    / "usd-quoted-basket-operational-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast151-usd-quoted-basket-operational-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast151-registry.jsonl"

PREREGISTRATION_COMMIT = "2f8eeba307dd93030da2751a894a29790de9be68"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-24T06:18:19-03:00")
PREREGISTRATION_SHA256 = "7a6ded6a5846d36faa609d2e2cdfd10d856a35f8b08bc8521c181443e3981856"
SOURCE_FAST150_REPORT_SHA256 = "d3d3f98c60ffc74335ecaa8db8eeb79603dc9dc68574e14a998d362755f0019e"
POSITION_OVERLAY_ID = "max_per_pair_1"
CIRCUIT_BREAKER_ID = "dd800_cooldown180d"
OPERATIONAL_POLICY_ID = (
    f"{TARGET_POLICY_ID}:{POSITION_OVERLAY_ID}:{CIRCUIT_BREAKER_ID}"
)
SELECTED_POSITION_OVERLAY = PositionOverlay(
    POSITION_OVERLAY_ID,
    max_total_open=None,
    max_open_per_pair=1,
)
SELECTED_CIRCUIT_BREAKER = CircuitBreakerPolicy(
    CIRCUIT_BREAKER_ID,
    closed_equity_drawdown_threshold_pips=800.0,
    cooldown_days=180,
)

EARLY_FAIL_MINIMUM_TRADES = 20
EARLY_FAIL_MAXIMUM_STRESS_TOTAL_PIPS = -250.0
EARLY_FAIL_MAXIMUM_DRAWDOWN_PIPS = -1000.0
SHADOW_CANDIDATE_MINIMUM_TRADES = 40
SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS = 25.0
SHADOW_CANDIDATE_MINIMUM_YEARS = 2
SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_SHARE = 2 / 3
SHADOW_CANDIDATE_MINIMUM_NONNEGATIVE_PAIRS = 3
SHADOW_CANDIDATE_MAXIMUM_DRAWDOWN_PIPS = -1000.0
SHADOW_CANDIDATE_MAXIMUM_CONCURRENT_TRADES = 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args(argv)
    report = run_fast151_usd_quoted_basket_operational_observation(
        m1_dir=arguments.m1_dir,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast151_usd_quoted_basket_operational_observation(
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
    market = load_prospective_market(m1_dir)
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

    base_candidates = select_base_candidates(
        materialized_rows,
        prospective_start=PREREGISTRATION_TIMESTAMP,
    )
    base_candidates_path = output_root / "base-prospective-candidates.parquet"
    base_candidates.to_parquet(base_candidates_path, compression="zstd", index=False)

    observations, position_skipped, breaker_skipped, breaker_triggers = apply_operational_controls(base_candidates)
    observations_path = output_root / "observations.jsonl"
    observation_records = [observation_record(row) for _, row in observations.iterrows()]
    write_jsonl(observations_path, observation_records)

    observation_rows_path = output_root / "observation-trade-rows.parquet"
    position_skipped_path = output_root / "position-skipped.csv"
    breaker_skipped_path = output_root / "breaker-skipped.csv"
    breaker_triggers_path = output_root / "breaker-triggers.csv"
    observations.to_parquet(observation_rows_path, compression="zstd", index=False)
    position_skipped.to_csv(position_skipped_path, index=False)
    breaker_skipped.to_csv(breaker_skipped_path, index=False)
    breaker_triggers.to_csv(breaker_triggers_path, index=False)

    summary = summarize_observations(observations, alpha=alpha)
    pair_summary = summarize_pairs(observations)
    portfolio_summary = summarize_sample(observations)
    gates = evaluate_fast151_gates(
        summary=summary,
        observations=observations,
        pair_summary=pair_summary,
        portfolio_summary=portfolio_summary,
    )
    status, status_reasons = fast151_status(summary=summary, gates=gates)
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
        base_candidates=base_candidates,
        base_candidates_path=base_candidates_path,
        observations_path=observations_path,
        observation_rows_path=observation_rows_path,
        observation_records=observation_records,
        position_skipped=position_skipped,
        position_skipped_path=position_skipped_path,
        breaker_skipped=breaker_skipped,
        breaker_skipped_path=breaker_skipped_path,
        breaker_triggers=breaker_triggers,
        breaker_triggers_path=breaker_triggers_path,
        summary=summary,
        pair_summary=pair_summary,
        portfolio_summary=portfolio_summary,
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
        raise ValueError("Fast151 preregistration hash changed")

    events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in events
        if event.get("event") == "fast151_usd_quoted_basket_operational_preregistered"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast151 registry must contain exactly one preregistration event")

    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "fast150_report_sha256": SOURCE_FAST150_REPORT_SHA256,
        "group_id": GROUP_ID,
        "pairs": list(TARGET_PAIRS),
        "entry_policy_id": ENTRY_POLICY_ID,
        "risk_policy_id": RISK_POLICY_ID,
        "target_policy_id": TARGET_POLICY_ID,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "side_filter": SIDE_FILTER,
        "position_overlay_id": POSITION_OVERLAY_ID,
        "circuit_breaker_id": CIRCUIT_BREAKER_ID,
        "timeframe": "D1",
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast151 registry mismatch: {mismatches}")

    return {"preregistration_sha256": preregistration_sha256}


def select_base_candidates(rows: pd.DataFrame, *, prospective_start: pd.Timestamp) -> pd.DataFrame:
    if rows.empty:
        return empty_operational_rows()
    frame = rows.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    entry_policy = frozen_entry_policy()
    risk_policy = frozen_risk_policy()
    mask = (
        (frame["entry_timestamp"] > prospective_start)
        & frame["pair"].isin(TARGET_PAIRS)
        & frame["entry_policy_id"].eq(ENTRY_POLICY_ID)
        & frame["risk_policy_id"].eq(RISK_POLICY_ID)
        & frame["channel_days"].eq(entry_policy.channel_days)
        & frame["hold_days"].eq(entry_policy.hold_days)
        & frame["stop_atr_multiple"].eq(risk_policy.stop_atr_multiple)
        & frame["management"].eq(risk_policy.management.name)
        & frame["side"].eq(SIDE_FILTER)
    )
    selected = frame.loc[mask].sort_values(["entry_timestamp", "pair", "candidate_id"]).copy()
    return add_operational_columns(selected)


def empty_operational_rows() -> pd.DataFrame:
    frame = empty_trade_rows()
    return add_operational_columns(frame)


def add_operational_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["fast151_policy_id"] = OPERATIONAL_POLICY_ID
    enriched["operational_policy_id"] = OPERATIONAL_POLICY_ID
    enriched["target_policy_id"] = TARGET_POLICY_ID
    enriched["group_id"] = GROUP_ID
    enriched["position_overlay_id"] = POSITION_OVERLAY_ID
    enriched["circuit_breaker_id"] = CIRCUIT_BREAKER_ID
    return enriched


def apply_operational_controls(
    base_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if base_candidates.empty:
        return (
            empty_operational_rows(),
            empty_position_skipped_frame(),
            empty_breaker_skipped_frame(),
            empty_breaker_trigger_frame(),
        )
    position_accepted, position_skipped = apply_position_overlay(base_candidates, SELECTED_POSITION_OVERLAY)
    breaker_accepted, breaker_skipped, breaker_triggers = apply_drawdown_circuit_breaker(
        position_accepted,
        SELECTED_CIRCUIT_BREAKER,
    )
    return (
        add_operational_columns(breaker_accepted),
        position_skipped,
        breaker_skipped,
        breaker_triggers,
    )


def empty_position_skipped_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "overlay_id",
            "skip_reason",
            "pair",
            "entry_timestamp",
            "exit_stress_timestamp",
            "net_stress_pips",
        ]
    )


def empty_breaker_skipped_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "breaker_id",
            "skip_reason",
            "pair",
            "entry_timestamp",
            "exit_stress_timestamp",
            "net_stress_pips",
            "pause_until",
        ]
    )


def empty_breaker_trigger_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "breaker_id",
            "trigger_timestamp",
            "closed_trade_pair",
            "closed_trade_stress_pips",
            "closed_equity_pips",
            "closed_peak_pips",
            "closed_drawdown_pips",
            "pause_until",
        ]
    )


def observation_record(row: pd.Series) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "pair": str(row["pair"]),
        "candidate_id": str(row["candidate_id"]),
        "fast151_policy_id": str(row["fast151_policy_id"]),
        "operational_policy_id": str(row["operational_policy_id"]),
        "target_policy_id": str(row["target_policy_id"]),
        "group_id": str(row["group_id"]),
        "position_overlay_id": str(row["position_overlay_id"]),
        "circuit_breaker_id": str(row["circuit_breaker_id"]),
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


def evaluate_fast151_gates(
    *,
    summary: dict[str, Any],
    observations: pd.DataFrame,
    pair_summary: list[dict[str, Any]],
    portfolio_summary: dict[str, Any],
) -> dict[str, Any]:
    stress = summary["stress"]
    base = summary["base"]
    non_short_observation_count = int((~observations["side"].eq(SIDE_FILTER)).sum()) if not observations.empty else 0
    invalid_pair_count = int((~observations["pair"].isin(TARGET_PAIRS)).sum()) if not observations.empty else 0
    nonnegative_pair_count = int(sum(item["nonnegative_mean_stress"] for item in pair_summary))
    max_drawdown_pips = portfolio_summary["max_drawdown_pips"]
    max_concurrent_trades = portfolio_summary["max_concurrent_trades"]

    early_fail_reasons = []
    if (
        stress["trade_count"] >= EARLY_FAIL_MINIMUM_TRADES
        and stress["total_pips"] <= EARLY_FAIL_MAXIMUM_STRESS_TOTAL_PIPS
    ):
        early_fail_reasons.append("early_fail_stress_total_pips")
    if (
        stress["trade_count"] >= EARLY_FAIL_MINIMUM_TRADES
        and max_drawdown_pips is not None
        and max_drawdown_pips < EARLY_FAIL_MAXIMUM_DRAWDOWN_PIPS
    ):
        early_fail_reasons.append("early_fail_closed_drawdown_pips")
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
    if max_drawdown_pips is not None and max_drawdown_pips < SHADOW_CANDIDATE_MAXIMUM_DRAWDOWN_PIPS:
        shadow_reasons.append("closed_drawdown_worse_than_threshold")
    if max_concurrent_trades > SHADOW_CANDIDATE_MAXIMUM_CONCURRENT_TRADES:
        shadow_reasons.append("concurrent_trade_count_above_threshold")
    if stress["two_consecutive_negative_years"]:
        shadow_reasons.append("two_consecutive_negative_stress_years")
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
        "max_drawdown_pips": max_drawdown_pips,
        "max_concurrent_trades": max_concurrent_trades,
        "required_side_filter": SIDE_FILTER,
        "target_pairs": list(TARGET_PAIRS),
        "position_overlay_id": POSITION_OVERLAY_ID,
        "circuit_breaker_id": CIRCUIT_BREAKER_ID,
    }


def fast151_status(*, summary: dict[str, Any], gates: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "operational_observation"]
    if summary["eligible_observation_count"] == 0:
        return "FAST151_OBSERVATION_NO_ELIGIBLE_OPERATIONAL_TRADES_YET", reasons + [
            "no_post_preregistration_operational_trades"
        ]
    if gates["early_fail"]:
        return "FAST151_OBSERVATION_FAILED_EARLY", reasons + gates["early_fail_reasons"]
    if gates["shadow_candidate"]:
        return (
            "FAST151_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW",
            reasons + ["shadow_candidate_gate_passed_but_not_testnet_authorized"],
        )
    return (
        "FAST151_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
        reasons + gates["shadow_candidate_reasons"],
    )


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
    base_candidates: pd.DataFrame,
    base_candidates_path: Path,
    observations_path: Path,
    observation_rows_path: Path,
    observation_records: list[dict[str, Any]],
    position_skipped: pd.DataFrame,
    position_skipped_path: Path,
    breaker_skipped: pd.DataFrame,
    breaker_skipped_path: Path,
    breaker_triggers: pd.DataFrame,
    breaker_triggers_path: Path,
    summary: dict[str, Any],
    pair_summary: list[dict[str, Any]],
    portfolio_summary: dict[str, Any],
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
            "fast151_is_preregistered_operational_observation_only",
            "operating_policy_was_identified_by_historical_research",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast151_usd_quoted_basket_operational_observation",
        "materialization_source": "raw_m1",
        "status": status,
        "status_reasons": status_reasons,
        "group_id": GROUP_ID,
        "target_pairs": list(TARGET_PAIRS),
        "target_policy_id": TARGET_POLICY_ID,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "side_filter": SIDE_FILTER,
        "entry_policy_id": ENTRY_POLICY_ID,
        "risk_policy_id": RISK_POLICY_ID,
        "position_overlay_id": POSITION_OVERLAY_ID,
        "circuit_breaker_id": CIRCUIT_BREAKER_ID,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "source_fast150_report_sha256": SOURCE_FAST150_REPORT_SHA256,
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
        "base_candidate_count": int(len(base_candidates)),
        "base_candidates_path": str(base_candidates_path),
        "base_candidates_sha256": sha256_file(base_candidates_path),
        "eligible_observation_count": len(observation_records),
        "observations_path": str(observations_path),
        "observations_sha256": sha256_file(observations_path),
        "observation_rows_path": str(observation_rows_path),
        "observation_rows_sha256": sha256_file(observation_rows_path),
        "position_skipped_count": int(len(position_skipped)),
        "position_skipped_path": str(position_skipped_path),
        "position_skipped_sha256": sha256_file(position_skipped_path),
        "breaker_skipped_count": int(len(breaker_skipped)),
        "breaker_skipped_path": str(breaker_skipped_path),
        "breaker_skipped_sha256": sha256_file(breaker_skipped_path),
        "breaker_trigger_count": int(len(breaker_triggers)),
        "breaker_triggers_path": str(breaker_triggers_path),
        "breaker_triggers_sha256": sha256_file(breaker_triggers_path),
        "summary": summary,
        "pair_summary": pair_summary,
        "portfolio_summary": portfolio_summary,
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
        "operational_policy_id": report["operational_policy_id"],
        "raw_m1_start": report["raw_m1_start"],
        "raw_m1_end": report["raw_m1_end"],
        "raw_m1_min_pair_end": report["raw_m1_min_pair_end"],
        "raw_m1_end_by_pair": report["raw_m1_end_by_pair"],
        "raw_m1_row_count": report["raw_m1_row_count"],
        "signal_count": report["signal_count"],
        "signal_counts_by_pair": report["signal_counts_by_pair"],
        "materialized_trade_row_count": report["materialized_trade_row_count"],
        "base_candidate_count": report["base_candidate_count"],
        "eligible_observation_count": report["eligible_observation_count"],
        "position_skipped_count": report["position_skipped_count"],
        "breaker_skipped_count": report["breaker_skipped_count"],
        "breaker_trigger_count": report["breaker_trigger_count"],
        "summary": report["summary"],
        "pair_summary": report["pair_summary"],
        "portfolio_summary": report["portfolio_summary"],
        "gates": report["gates"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
