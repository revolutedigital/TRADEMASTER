"""Fast124 preregistered USDJPY compression-breakout observation runner.

Fast124 freezes the best Fast121/Fast122 clue and observes it prospectively
only. It reconstructs the USDJPY H1 compression-breakout policy from local M1
bid/ask bars, filters only post-preregistration entries, writes offline
evidence, and never sends orders or commits ledger state.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.fx.instruments import ConversionRates  # noqa: E402
from app.fx.sim.costs import FUSION_ZERO, STRESS  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import (  # noqa: E402
    plain_json,
    write_json,
)
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    Management,
    PairData,
    Policy,
    commission_pips,
    evaluate_pair_policy,
    lower_confidence_bound,
    mid_ohlc,
    sha256_file,
    widen_quotes,
)

DEFAULT_RAW_M1_DIR = REPO_ROOT / "backend" / "data" / "raw" / "fx_m1_hd"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast124" / "usdjpy-compression-breakout-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast124-usdjpy-compression-breakout-prospective-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast124-registry.jsonl"

PREREGISTRATION_COMMIT = "0a0fb2183ecd3de19b10d2832230194133afe79d"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-24T01:50:12-03:00")
PREREGISTRATION_SHA256 = "1de2b2f892a284cf6025fd994011f5ef32cf27f7060dfaceb5b0d5277452688e"
SOURCE_FAST121_REPORT_SHA256 = "f16ce76377e58966720ff4cc2e79c93cf0c9b41fcaa83b7d7e92e0b328a5f1f9"
SOURCE_FAST122_REPORT_SHA256 = "df85e715f44d87305d3e3137f2be67c3d256b520cf701bd0c6fa37a6c2facbb4"
SOURCE_FAST123_REPORT_SHA256 = "b1a82fae2259df69c16f17ba1e9046da50e0d92d8eb401a876073ffd49b89fad"

OBSERVATION_PAIR = "USDJPY"
FROZEN_POLICY = Policy(
    identifier="tf1h_ch20_cq0.4_atr1.5_hold720_fixed_3r",
    timeframe="1h",
    channel_bars=20,
    compression_quantile=0.40,
    stop_atr_multiple=1.5,
    max_hold_minutes=720,
    management=Management("fixed_3r", target_r=3.0, activation_r=None, trail_r=None),
)
DEFAULT_ALPHA = 0.05
EARLY_FAIL_MINIMUM_TRADES = 50
EARLY_FAIL_MAXIMUM_STRESS_MEAN_PIPS = -2.0
SHADOW_CANDIDATE_MINIMUM_TRADES = 100
SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS = 1.0
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
    "policy_id",
    "pair",
    "timeframe",
    "channel_bars",
    "compression_quantile",
    "stop_atr_multiple",
    "max_hold_minutes",
    "management",
    "signal_timestamp",
    "entry_timestamp",
    "side",
    "breakout_distance_pips",
    "atr_pips",
    "risk_pips",
    "compression_value_pips",
    "compression_threshold_pips",
    "raw_base_pips",
    "net_base_pips",
    "exit_base_timestamp",
    "exit_base_reason",
    "raw_stress_pips",
    "net_stress_pips",
    "exit_stress_timestamp",
    "exit_stress_reason",
    "base_commission_pips",
    "stress_commission_pips",
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
    parser.add_argument("--raw-m1-dir", type=Path, default=DEFAULT_RAW_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--start", help="Optional UTC inclusive start timestamp for diagnostics")
    parser.add_argument("--end", help="Optional UTC inclusive end timestamp for diagnostics")
    arguments = parser.parse_args(argv)
    report = run_fast124_usdjpy_compression_observation(
        raw_m1_dir=arguments.raw_m1_dir,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
        start=arguments.start,
        end=arguments.end,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast124_usdjpy_compression_observation(
    *,
    raw_m1_dir: Path = DEFAULT_RAW_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    alpha: float = DEFAULT_ALPHA,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    validate_parameters(alpha=alpha)
    raw_m1_dir = absolute_path(raw_m1_dir)
    output_root = absolute_path(output_root)
    preregistration = validate_preregistration(
        preregistration_path=absolute_path(preregistration_path),
        registry_path=absolute_path(registry_path),
    )
    output_root.mkdir(parents=True, exist_ok=True)

    source_files = source_m1_files(raw_m1_dir, OBSERVATION_PAIR)
    source_validation = source_validation_summary(raw_m1_dir, OBSERVATION_PAIR)
    raw_m1 = load_observation_bars(OBSERVATION_PAIR, raw_m1_dir, start=start, end=end)
    materialized_rows = materialize_frozen_policy_trades(raw_m1)
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
    status, status_reasons = fast124_status(summary=summary, gates=gates)
    report = build_report(
        preregistration=preregistration,
        raw_m1_dir=raw_m1_dir,
        source_files=source_files,
        source_validation=source_validation,
        raw_m1=raw_m1,
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
        start=start,
        end=end,
    )
    return write_report(output_root, report)


def absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


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
        raise ValueError("Fast124 preregistration hash changed")
    events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in events
        if event.get("event") == "fast124_usdjpy_compression_breakout_prospective_preregistered"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast124 registry must contain exactly one preregistration event")
    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_fast121_report_sha256": SOURCE_FAST121_REPORT_SHA256,
        "source_fast122_report_sha256": SOURCE_FAST122_REPORT_SHA256,
        "source_fast123_report_sha256": SOURCE_FAST123_REPORT_SHA256,
        "pair": OBSERVATION_PAIR,
        "frozen_policy_id": FROZEN_POLICY.identifier,
        "timeframe": FROZEN_POLICY.timeframe,
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast124 registry mismatch: {mismatches}")
    return {"preregistration_sha256": preregistration_sha256}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_m1_files(raw_m1_dir: Path, pair: str) -> list[dict[str, Any]]:
    files = sorted(raw_m1_dir.glob(f"{pair}_M1_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no M1 file for {pair} in {raw_m1_dir}")
    return [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]


def source_validation_summary(raw_m1_dir: Path, pair: str) -> dict[str, Any]:
    validation_path = raw_m1_dir / f"{pair}_validation.csv"
    if not validation_path.exists():
        return {
            "available": False,
            "path": str(validation_path),
            "reported_month_count": 0,
            "reported_months": [],
            "failed_reported_months": [],
            "zero_oracle_hour_months": [],
            "all_reported_months_passed": False,
            "note": "no_validation_csv_found_for_pair",
        }

    rows = pd.read_csv(validation_path)
    if rows.empty:
        return {
            "available": True,
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
            "reported_month_count": 0,
            "reported_months": [],
            "failed_reported_months": [],
            "zero_oracle_hour_months": [],
            "all_reported_months_passed": False,
            "note": "validation_csv_empty",
        }
    required = {"month"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"validation csv missing columns: {sorted(missing)}")

    rows = rows.copy()
    reported_months = sorted(str(value) for value in rows["month"].dropna().unique())
    if "passes" not in rows.columns:
        zero_oracle: list[str] = []
        if "hours_compared" in rows.columns:
            zero_oracle = rows.loc[
                rows["hours_compared"].fillna(0).astype(int).eq(0),
                "month",
            ].astype(str).tolist()
        return {
            "available": True,
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
            "reported_month_count": int(len(reported_months)),
            "reported_months": reported_months,
            "failed_reported_months": [],
            "zero_oracle_hour_months": zero_oracle,
            "all_reported_months_passed": False,
            "note": "validation_csv_without_passes_column_treated_as_unverified",
        }

    rows["passes"] = rows["passes"].map(parse_validation_bool)
    failed = rows.loc[~rows["passes"], "month"].astype(str).tolist()
    zero_oracle: list[str] = []
    if "hours_compared" in rows.columns:
        zero_oracle = rows.loc[
            rows["hours_compared"].fillna(0).astype(int).eq(0),
            "month",
        ].astype(str).tolist()
    return {
        "available": True,
        "path": str(validation_path),
        "sha256": sha256_file(validation_path),
        "reported_month_count": int(len(reported_months)),
        "reported_months": reported_months,
        "failed_reported_months": failed,
        "zero_oracle_hour_months": zero_oracle,
        "all_reported_months_passed": not failed,
        "note": "validation_csv_scope_is_the_last_fx_histdata_ticks_run",
    }


def parse_validation_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return bool(value)


def load_observation_bars(
    pair: str,
    raw_m1_dir: Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    paths = sorted(raw_m1_dir.glob(f"{pair}_M1_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no M1 file for {pair} in {raw_m1_dir}")
    columns = [
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]
    frame = pd.concat(pd.read_parquet(path, columns=columns) for path in paths).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.index.tz is None:
        raise ValueError(f"{pair} bars must be timezone-aware")
    start_timestamp = parse_optional_timestamp(start, name="start")
    end_timestamp = parse_optional_timestamp(end, name="end")
    raw_bars = len(frame)
    if start_timestamp is not None:
        frame = frame.loc[frame.index >= start_timestamp]
    if end_timestamp is not None:
        frame = frame.loc[frame.index <= end_timestamp]
    crossed = (
        frame[["ask_open", "ask_high", "ask_low", "ask_close"]].to_numpy()
        < frame[["bid_open", "bid_high", "bid_low", "bid_close"]].to_numpy()
    ).any(axis=1)
    if crossed.any():
        frame = frame.loc[~crossed].copy()
    if frame.empty:
        raise ValueError("M1 frame is empty after applying optional start/end filters")
    frame.attrs["raw_bars"] = raw_bars
    frame.attrs["dropped_crossed_quote_bars"] = int(crossed.sum())
    return frame


def parse_optional_timestamp(value: str | None, *, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include a timezone, preferably Z/UTC")
    return timestamp.tz_convert("UTC")


def materialize_frozen_policy_trades(raw_m1: pd.DataFrame) -> pd.DataFrame:
    median_price = float(np.median(mid_ohlc(raw_m1)["close"].to_numpy()))
    rates = ConversionRates({OBSERVATION_PAIR: median_price})
    pair_data = PairData(
        bars=raw_m1,
        stress_bars=widen_quotes(raw_m1, STRESS.spread_multiplier),
        median_price=median_price,
        base_commission_pips=commission_pips(OBSERVATION_PAIR, median_price, rates, FUSION_ZERO),
        stress_commission_pips=commission_pips(OBSERVATION_PAIR, median_price, rates, STRESS),
        quality={},
    )
    rows = evaluate_pair_policy(OBSERVATION_PAIR, pair_data, FROZEN_POLICY)
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
        & frame["pair"].eq(OBSERVATION_PAIR)
        & frame["policy_id"].eq(FROZEN_POLICY.identifier)
        & frame["timeframe"].eq(FROZEN_POLICY.timeframe)
        & frame["channel_bars"].eq(FROZEN_POLICY.channel_bars)
        & frame["compression_quantile"].eq(FROZEN_POLICY.compression_quantile)
        & frame["stop_atr_multiple"].eq(FROZEN_POLICY.stop_atr_multiple)
        & frame["max_hold_minutes"].eq(FROZEN_POLICY.max_hold_minutes)
        & frame["management"].eq(FROZEN_POLICY.management.name)
    )
    return frame.loc[mask].sort_values(["entry_timestamp", "candidate_id"]).copy()


def observation_record(row: pd.Series) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "pair": str(row["pair"]),
        "policy_id": str(row["policy_id"]),
        "candidate_id": str(row["candidate_id"]),
        "timeframe": str(row["timeframe"]),
        "signal_timestamp": pd.Timestamp(row["signal_timestamp"]).isoformat(),
        "entry_timestamp": pd.Timestamp(row["entry_timestamp"]).isoformat(),
        "side": str(row["side"]),
        "breakout_distance_pips": float(row["breakout_distance_pips"]),
        "atr_pips": float(row["atr_pips"]),
        "risk_pips": float(row["risk_pips"]),
        "compression_value_pips": float(row["compression_value_pips"]),
        "compression_threshold_pips": float(row["compression_threshold_pips"]),
        "raw_base_pips": float(row["raw_base_pips"]),
        "net_base_pips": float(row["net_base_pips"]),
        "exit_base_timestamp": pd.Timestamp(row["exit_base_timestamp"]).isoformat(),
        "exit_base_reason": str(row["exit_base_reason"]),
        "raw_stress_pips": float(row["raw_stress_pips"]),
        "net_stress_pips": float(row["net_stress_pips"]),
        "exit_stress_timestamp": pd.Timestamp(row["exit_stress_timestamp"]).isoformat(),
        "exit_stress_reason": str(row["exit_stress_reason"]),
        "base_commission_pips": float(row["base_commission_pips"]),
        "stress_commission_pips": float(row["stress_commission_pips"]),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(plain_json(record), sort_keys=True) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")


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
        and stress["mean_pips"] is not None
        and stress["mean_pips"] <= EARLY_FAIL_MAXIMUM_STRESS_MEAN_PIPS
    ):
        early_fail_reasons.append("early_fail_stress_mean_pips")
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


def fast124_status(*, summary: dict[str, Any], gates: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if summary["eligible_observation_count"] == 0:
        return "FAST124_OBSERVATION_NO_ELIGIBLE_TRADES_YET", reasons + [
            "no_post_preregistration_trades"
        ]
    if gates["early_fail"]:
        return "FAST124_OBSERVATION_FAILED_EARLY", reasons + gates["early_fail_reasons"]
    if gates["shadow_candidate"]:
        return (
            "FAST124_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW",
            reasons + ["shadow_candidate_gate_passed_but_not_testnet_authorized"],
        )
    return (
        "FAST124_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
        reasons + gates["shadow_candidate_reasons"],
    )


def build_report(
    *,
    preregistration: dict[str, str],
    raw_m1_dir: Path,
    source_files: list[dict[str, Any]],
    source_validation: dict[str, Any],
    raw_m1: pd.DataFrame,
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
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast124_is_preregistered_observation_only",
            "frozen_rule_was_identified_by_historical_research",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast124_usdjpy_compression_breakout_observation",
        "materialization_source": "raw_m1",
        "status": status,
        "status_reasons": status_reasons,
        "observation_pair": OBSERVATION_PAIR,
        "frozen_policy": {
            **asdict(FROZEN_POLICY),
            "management": asdict(FROZEN_POLICY.management),
        },
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "source_fast121_report_sha256": SOURCE_FAST121_REPORT_SHA256,
        "source_fast122_report_sha256": SOURCE_FAST122_REPORT_SHA256,
        "source_fast123_report_sha256": SOURCE_FAST123_REPORT_SHA256,
        "raw_m1_dir": str(raw_m1_dir),
        "raw_m1_files": source_files,
        "raw_m1_validation": source_validation,
        "raw_m1_row_count": int(len(raw_m1)),
        "raw_m1_start": raw_m1.index[0].isoformat(),
        "raw_m1_end": raw_m1.index[-1].isoformat(),
        "dropped_crossed_quote_bars": int(raw_m1.attrs.get("dropped_crossed_quote_bars", 0)),
        "optional_start_filter": start,
        "optional_end_filter": end,
        "materialized_trade_rows_path": str(materialized_path),
        "materialized_trade_rows_sha256": sha256_file(materialized_path),
        "materialized_trade_row_count": int(len(materialized_rows)),
        "eligible_observation_count": len(observation_records),
        "observations_path": str(observations_path),
        "observations_sha256": sha256_file(observations_path),
        "summary": summary,
        "gates": gates,
        "costs": {
            "base": asdict(FUSION_ZERO),
            "stress": asdict(STRESS),
            "slippage_model_used": "fixed per side from scenario; stress spread multiplier applied",
        },
    }


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    write_json(report_path, payload)
    return plain_json(payload)


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
        "materialized_trade_row_count": report["materialized_trade_row_count"],
        "eligible_observation_count": report["eligible_observation_count"],
        "summary": report["summary"],
        "gates": report["gates"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
