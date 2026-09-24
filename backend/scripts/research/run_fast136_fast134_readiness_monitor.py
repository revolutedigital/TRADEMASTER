"""Fast136: fail-closed readiness monitor for the Fast134 NZDUSD clue.

Fast134 froze one research candidate:
``NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r``.

This monitor answers the operational research question without sending orders:
do we have post-preregistration M1 data, did a post-preregistration signal
appear, is there a completed Fast134 observation, and how far is the latest D1
bar from the frozen Donchian fade trigger?
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

from app.fx.instruments import Instrument  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    DEFAULT_M1_DIR,
    sha256_file,
)
from scripts.research.run_fast124_usdjpy_compression_observation import (  # noqa: E402
    absolute_path,
    source_m1_files,
    source_validation_summary,
)
from scripts.research.run_fast128_fx_daily_donchian_scout import daily_mid_bars  # noqa: E402
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    PAIR,
    daily_atr_pips,
    frozen_entry_signals,
    load_pair_data,
)
from scripts.research.run_fast134_nzdusd_donchian_fade_fixed1r_observation import (  # noqa: E402
    FROZEN_ENTRY_POLICY,
    FROZEN_POLICY_ID,
    FROZEN_RISK_POLICY,
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_SHA256,
    PREREGISTRATION_TIMESTAMP,
    SAFETY_FLAGS,
    SOURCE_FAST133_REPORT_SHA256,
    materialize_frozen_policy_trades,
    select_eligible_observations,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast136" / "fast134-readiness-monitor"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run_fast136_fast134_readiness_monitor(
        m1_dir=arguments.m1_dir,
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast136_fast134_readiness_monitor(
    *,
    m1_dir: Path = DEFAULT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    m1_dir = absolute_path(m1_dir)
    output_root = absolute_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_files = source_m1_files(m1_dir, PAIR)
    source_validation = source_validation_summary(m1_dir, PAIR)
    pair_data = load_pair_data(m1_dir)

    signals = frozen_entry_signals(pair_data)
    post_preregistration_signals = select_post_preregistration_signals(signals)
    materialized_rows = materialize_frozen_policy_trades(pair_data=pair_data, signals=signals)
    eligible_observations = select_eligible_observations(
        materialized_rows,
        prospective_start=PREREGISTRATION_TIMESTAMP,
    )
    readiness_snapshot = donchian_readiness_snapshot(
        pair_data.bars,
        pair=PAIR,
        channel_days=FROZEN_ENTRY_POLICY.channel_days,
    )

    raw_m1_end = pd.Timestamp(pair_data.quality["last_bar"]) if pair_data.quality["last_bar"] else None
    status, status_reasons = fast136_status(
        raw_m1_end=raw_m1_end,
        post_preregistration_signal_count=len(post_preregistration_signals),
        completed_eligible_observation_count=len(eligible_observations),
    )

    signals_path = output_root / "post-preregistration-signals.json"
    write_json(signals_path, signal_records(post_preregistration_signals))

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast136_is_monitoring_readiness_only",
            "fast134_candidate_still_requires_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast136_fast134_readiness_monitor",
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
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "source_fast133_report_sha256": SOURCE_FAST133_REPORT_SHA256,
        "m1_dir": str(m1_dir),
        "raw_m1_files": source_files,
        "raw_m1_validation": source_validation,
        "pair_quality": pair_data.quality,
        "raw_m1_start": pair_data.quality["first_bar"],
        "raw_m1_end": pair_data.quality["last_bar"],
        "raw_m1_row_count": int(pair_data.quality["usable_bars"]),
        "has_post_preregistration_m1_data": bool(
            raw_m1_end is not None and raw_m1_end > PREREGISTRATION_TIMESTAMP
        ),
        "signal_count": int(len(signals)),
        "post_preregistration_signal_count": int(len(post_preregistration_signals)),
        "post_preregistration_signals_path": str(signals_path),
        "post_preregistration_signals_sha256": sha256_file(signals_path),
        "materialized_trade_row_count": int(len(materialized_rows)),
        "completed_eligible_observation_count": int(len(eligible_observations)),
        "latest_d1_donchian_snapshot": readiness_snapshot,
        "next_action": next_action_for_status(status),
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return plain_json({**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)})


def select_post_preregistration_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    frame = signals.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    frame["signal_timestamp"] = pd.to_datetime(frame["signal_timestamp"], utc=True)
    return frame.loc[frame["entry_timestamp"] > PREREGISTRATION_TIMESTAMP].sort_values(
        ["entry_timestamp", "signal_timestamp"],
        ignore_index=True,
    )


def donchian_readiness_snapshot(
    frame: pd.DataFrame,
    *,
    pair: str,
    channel_days: int,
) -> dict[str, Any]:
    daily_bars = daily_mid_bars(frame)
    atr = daily_atr_pips(frame, pair)
    return donchian_readiness_snapshot_from_daily_bars(
        daily_bars,
        pair=pair,
        channel_days=channel_days,
        atr_pips=atr,
    )


def donchian_readiness_snapshot_from_daily_bars(
    daily_bars: pd.DataFrame,
    *,
    pair: str,
    channel_days: int,
    atr_pips: pd.Series | None = None,
) -> dict[str, Any]:
    if len(daily_bars) < channel_days + 1:
        return {
            "available": False,
            "reason": "insufficient_daily_bars_for_channel",
            "daily_bar_count": int(len(daily_bars)),
            "required_daily_bars": int(channel_days + 1),
            "channel_days": int(channel_days),
        }
    channel_high = daily_bars["high"].rolling(channel_days, min_periods=channel_days).max().shift(1)
    channel_low = daily_bars["low"].rolling(channel_days, min_periods=channel_days).min().shift(1)
    valid_index = daily_bars.index[channel_high.notna() & channel_low.notna()]
    if valid_index.empty:
        return {
            "available": False,
            "reason": "no_valid_shifted_channel",
            "daily_bar_count": int(len(daily_bars)),
            "channel_days": int(channel_days),
        }

    latest_timestamp = pd.Timestamp(valid_index.max())
    latest_bar = daily_bars.loc[latest_timestamp]
    upper = float(channel_high.loc[latest_timestamp])
    lower = float(channel_low.loc[latest_timestamp])
    close = float(latest_bar["close"])
    pip_size = Instrument.from_symbol(pair).pip_size
    pips_to_upper_breakout = (upper - close) / pip_size
    pips_to_lower_breakout = (close - lower) / pip_size
    breakout_up = close > upper
    breakout_down = close < lower
    if breakout_up:
        current_breakout_direction = "UP"
        current_fade_side = "SHORT"
        nearest_trigger = "active_upper_breakout"
        nearest_trigger_distance_pips = 0.0
    elif breakout_down:
        current_breakout_direction = "DOWN"
        current_fade_side = "LONG"
        nearest_trigger = "active_lower_breakout"
        nearest_trigger_distance_pips = 0.0
    elif pips_to_upper_breakout <= pips_to_lower_breakout:
        current_breakout_direction = None
        current_fade_side = None
        nearest_trigger = "upper_breakout_then_short_fade"
        nearest_trigger_distance_pips = float(pips_to_upper_breakout)
    else:
        current_breakout_direction = None
        current_fade_side = None
        nearest_trigger = "lower_breakout_then_long_fade"
        nearest_trigger_distance_pips = float(pips_to_lower_breakout)

    atr_value = None if atr_pips is None else atr_pips.get(latest_timestamp)
    latest_atr_pips = None if pd.isna(atr_value) else float(atr_value)
    return {
        "available": True,
        "pair": pair,
        "channel_days": int(channel_days),
        "hold_days": int(FROZEN_ENTRY_POLICY.hold_days),
        "latest_signal_bar_timestamp": latest_timestamp.isoformat(),
        "entry_timestamp_if_signal": (latest_timestamp + pd.Timedelta(minutes=1)).isoformat(),
        "latest_open": float(latest_bar["open"]),
        "latest_high": float(latest_bar["high"]),
        "latest_low": float(latest_bar["low"]),
        "latest_close": close,
        "shifted_channel_high": upper,
        "shifted_channel_low": lower,
        "pips_to_upper_breakout": float(pips_to_upper_breakout),
        "pips_to_lower_breakout": float(pips_to_lower_breakout),
        "nearest_trigger": nearest_trigger,
        "nearest_trigger_distance_pips": nearest_trigger_distance_pips,
        "is_breakout_signal_now": bool(breakout_up or breakout_down),
        "current_breakout_direction": current_breakout_direction,
        "current_fade_side": current_fade_side,
        "daily_atr_pips": latest_atr_pips,
        "risk_pips_if_signal": None
        if latest_atr_pips is None
        else float(FROZEN_RISK_POLICY.stop_atr_multiple * latest_atr_pips),
        "post_preregistration_signal_bar": bool(latest_timestamp > PREREGISTRATION_TIMESTAMP),
    }


def fast136_status(
    *,
    raw_m1_end: pd.Timestamp | None,
    post_preregistration_signal_count: int,
    completed_eligible_observation_count: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if raw_m1_end is None:
        return "FAST136_WAITING_FOR_M1_DATA", reasons + ["no_m1_data_loaded"]
    if raw_m1_end <= PREREGISTRATION_TIMESTAMP:
        return (
            "FAST136_WAITING_FOR_POST_PREREGISTRATION_M1_DATA",
            reasons + ["raw_m1_end_before_preregistration_timestamp"],
        )
    if completed_eligible_observation_count > 0:
        return (
            "FAST136_ELIGIBLE_OBSERVATION_READY_FOR_FAST134_REVIEW",
            reasons + ["completed_post_preregistration_observation_exists", "rerun_fast134_required"],
        )
    if post_preregistration_signal_count > 0:
        return (
            "FAST136_POST_PREREGISTRATION_SIGNAL_WAITING_FOR_FULL_OUTCOME",
            reasons + ["post_preregistration_signal_exists", "completed_outcome_not_materialized_yet"],
        )
    return (
        "FAST136_DATA_FRESH_NO_POST_PREREGISTRATION_SIGNAL_YET",
        reasons + ["post_preregistration_data_loaded", "no_frozen_signal_after_preregistration"],
    )


def signal_records(signals: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in signals.to_dict("records"):
        record: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
            elif isinstance(value, np.generic):
                record[key] = value.item()
            else:
                record[key] = value
        records.append(record)
    return records


def next_action_for_status(status: str) -> str:
    if status == "FAST136_WAITING_FOR_POST_PREREGISTRATION_M1_DATA":
        return "import_or_capture_nzdusd_m1_after_fast134_preregistration_then_rerun_fast134_and_fast136"
    if status == "FAST136_POST_PREREGISTRATION_SIGNAL_WAITING_FOR_FULL_OUTCOME":
        return "wait_until_the_20_day_hold_or_stop_target_outcome_is_fully_materialized_then_rerun_fast134"
    if status == "FAST136_ELIGIBLE_OBSERVATION_READY_FOR_FAST134_REVIEW":
        return "rerun_fast134_and_review_shadow_offline_gate_still_without_order_execution"
    if status == "FAST136_DATA_FRESH_NO_POST_PREREGISTRATION_SIGNAL_YET":
        return "keep_monitoring_new_m1_data_for_the_frozen_fast134_rule"
    return "fix_data_availability_before_any_strategy_review"


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    latest_snapshot = report["latest_d1_donchian_snapshot"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "raw_m1_end": report["raw_m1_end"],
        "has_post_preregistration_m1_data": report["has_post_preregistration_m1_data"],
        "post_preregistration_signal_count": report["post_preregistration_signal_count"],
        "completed_eligible_observation_count": report["completed_eligible_observation_count"],
        "latest_d1_donchian_snapshot": {
            "available": latest_snapshot["available"],
            "latest_signal_bar_timestamp": latest_snapshot.get("latest_signal_bar_timestamp"),
            "latest_close": latest_snapshot.get("latest_close"),
            "nearest_trigger": latest_snapshot.get("nearest_trigger"),
            "nearest_trigger_distance_pips": latest_snapshot.get("nearest_trigger_distance_pips"),
            "is_breakout_signal_now": latest_snapshot.get("is_breakout_signal_now"),
            "current_fade_side": latest_snapshot.get("current_fade_side"),
            "risk_pips_if_signal": latest_snapshot.get("risk_pips_if_signal"),
        },
        "next_action": report["next_action"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
