"""Fast134 preregistered NZDUSD daily Donchian-fade fixed-1R observation.

Fast134 freezes the best Fast133 candidate and observes it prospectively only.
It reconstructs the frozen NZDUSD D1 Donchian fade with the fixed 2 ATR / 1R
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

from app.fx.instruments import ConversionRates  # noqa: E402
from app.fx.sim.costs import FUSION_ZERO, STRESS  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import (  # noqa: E402
    Management,
    plain_json,
    write_json,
)
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    DEFAULT_M1_DIR,
    MAJOR_RATE_PAIRS,
    PairData,
    commission_pips,
    mid_ohlc,
    sha256_file,
    widen_quotes,
)
from scripts.research.run_fast124_usdjpy_compression_observation import (  # noqa: E402
    absolute_path,
    load_observation_bars,
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
)
from scripts.research.run_fast131_nzdusd_donchian_fade_observation import (  # noqa: E402
    DEFAULT_ALPHA,
    SAFETY_FLAGS,
    TIMESTAMP_COLUMNS,
    TRADE_COLUMNS,
    empty_trade_rows,
    evaluate_gates,
    summarize_observations,
    validate_parameters,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast134" / "nzdusd-donchian-fade-fixed1r-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast134-nzdusd-donchian-fade-fixed1r-prospective-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast134-registry.jsonl"

PREREGISTRATION_COMMIT = "cc7cf43de3e2e2a02203e5b20f2ebc786665ccfb"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-24T04:04:40-03:00")
PREREGISTRATION_SHA256 = "30cba913140050354521dc0651f7b35247b3fe6f6fff6f9d5b4b38ca45c13ac0"
SOURCE_FAST133_REPORT_SHA256 = "9d482ba974a6061f26a8b1a1e6f5dc702d93094f7424496af88871e7f18ca6fd"

FROZEN_RISK_POLICY = RiskPolicy(
    identifier="stop2_fixed_1r",
    stop_atr_multiple=2.0,
    management=Management("fixed_1r", target_r=1.0, activation_r=None, trail_r=None),
)
FROZEN_POLICY_ID = f"{PAIR}:{FROZEN_ENTRY_POLICY.identifier}:{FROZEN_RISK_POLICY.identifier}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args(argv)
    report = run_fast134_nzdusd_donchian_fade_fixed1r_observation(
        m1_dir=arguments.m1_dir,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast134_nzdusd_donchian_fade_fixed1r_observation(
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
    gates = evaluate_gates(summary)
    status, status_reasons = fast134_status(summary=summary, gates=gates)
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
        raise ValueError("Fast134 preregistration hash changed")

    events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in events
        if event.get("event") == "fast134_nzdusd_donchian_fade_fixed1r_prospective_preregistered"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast134 registry must contain exactly one preregistration event")

    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "fast133_report_sha256": SOURCE_FAST133_REPORT_SHA256,
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
        raise ValueError(f"Fast134 registry mismatch: {mismatches}")

    return {"preregistration_sha256": preregistration_sha256}


def load_prospective_pair_data(m1_dir: Path) -> PairData:
    """Load raw M1 bars without the historical audit end cut used by Fast130.

    Fast130's `load_pair_data()` deliberately truncates the historical audit at
    2026-09-01. Fast134 is a prospective observer, so it must ingest later M1
    files as they arrive while keeping the frozen signal/risk rule unchanged.
    """
    frames = {rate_pair: load_observation_bars(rate_pair, m1_dir) for rate_pair in MAJOR_RATE_PAIRS}
    rates = ConversionRates(
        {
            rate_pair: float(mid_ohlc(frame)["close"].median())
            for rate_pair, frame in frames.items()
        }
    )
    frame = frames[PAIR]
    median_price = float(mid_ohlc(frame)["close"].median())
    return PairData(
        bars=frame,
        stress_bars=widen_quotes(frame, STRESS.spread_multiplier),
        median_price=median_price,
        base_commission_pips=commission_pips(PAIR, median_price, rates, FUSION_ZERO),
        stress_commission_pips=commission_pips(PAIR, median_price, rates, STRESS),
        quality={
            "pair": PAIR,
            "raw_bars": int(frame.attrs.get("raw_bars", len(frame))),
            "usable_bars": int(len(frame)),
            "dropped_crossed_quote_bars": int(frame.attrs.get("dropped_crossed_quote_bars", 0)),
            "first_bar": frame.index.min().isoformat() if len(frame) else None,
            "last_bar": frame.index.max().isoformat() if len(frame) else None,
        },
    )


def materialize_frozen_policy_trades(*, pair_data: Any, signals: pd.DataFrame) -> pd.DataFrame:
    rows = evaluate_risk_policy(pair_data, signals, FROZEN_RISK_POLICY)
    if not rows:
        return empty_trade_rows()
    frame = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    for column in TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values(["entry_timestamp", "candidate_id"], ignore_index=True)


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


def fast134_status(*, summary: dict[str, Any], gates: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if summary["eligible_observation_count"] == 0:
        return "FAST134_OBSERVATION_NO_ELIGIBLE_TRADES_YET", reasons + [
            "no_post_preregistration_trades"
        ]
    if gates["early_fail"]:
        return "FAST134_OBSERVATION_FAILED_EARLY", reasons + gates["early_fail_reasons"]
    if gates["shadow_candidate"]:
        return (
            "FAST134_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW",
            reasons + ["shadow_candidate_gate_passed_but_not_testnet_authorized"],
        )
    return (
        "FAST134_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
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
            "fast134_is_preregistered_observation_only",
            "frozen_rule_was_identified_by_historical_research",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast134_nzdusd_donchian_fade_fixed1r_observation",
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
        "source_fast133_report_sha256": SOURCE_FAST133_REPORT_SHA256,
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
