"""Fast152: fail-closed readiness monitor for Fast151.

Fast151 froze the operating policy:

    basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d

This monitor does not discover, tune, download, trade, or approve anything. It
runs the canonical Fast151 observation runner against the current local M1
data, then emits a small readiness report: do we have post-preregistration M1
data, do we have base candidates, do we have accepted operational observations,
and what should happen next?
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
from scripts.research.run_fast124_usdjpy_compression_observation import absolute_path  # noqa: E402
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as DEFAULT_FAST151_OUTPUT_ROOT,
    OPERATIONAL_POLICY_ID,
    PREREGISTRATION_SHA256,
    PREREGISTRATION_TIMESTAMP,
    SAFETY_FLAGS,
    run_fast151_usd_quoted_basket_operational_observation,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast152" / "fast151-readiness-monitor"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fast151-output-root", type=Path, default=DEFAULT_FAST151_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run_fast152_fast151_readiness_monitor(
        m1_dir=arguments.m1_dir,
        output_root=arguments.output_root,
        fast151_output_root=arguments.fast151_output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast152_fast151_readiness_monitor(
    *,
    m1_dir: Path = DEFAULT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fast151_output_root: Path = DEFAULT_FAST151_OUTPUT_ROOT,
) -> dict[str, Any]:
    m1_dir = absolute_path(m1_dir)
    output_root = absolute_path(output_root)
    fast151_output_root = absolute_path(fast151_output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    observation_report = run_fast151_usd_quoted_basket_operational_observation(
        m1_dir=m1_dir,
        output_root=fast151_output_root,
    )
    raw_m1_min_pair_end = parse_optional_timestamp(observation_report.get("raw_m1_min_pair_end"))
    status, status_reasons = fast152_status(
        raw_m1_min_pair_end=raw_m1_min_pair_end,
        base_candidate_count=int(observation_report["base_candidate_count"]),
        eligible_observation_count=int(observation_report["eligible_observation_count"]),
        observation_status=str(observation_report["status"]),
    )

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast152_is_monitoring_readiness_only",
            "fast151_candidate_still_requires_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast152_fast151_readiness_monitor",
        "materialization_source": "raw_m1",
        "status": status,
        "status_reasons": status_reasons,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "m1_dir": str(m1_dir),
        "has_post_preregistration_m1_data": bool(
            raw_m1_min_pair_end is not None and raw_m1_min_pair_end > PREREGISTRATION_TIMESTAMP
        ),
        "raw_m1_start": observation_report["raw_m1_start"],
        "raw_m1_end": observation_report["raw_m1_end"],
        "raw_m1_min_pair_end": observation_report["raw_m1_min_pair_end"],
        "raw_m1_end_by_pair": observation_report["raw_m1_end_by_pair"],
        "raw_m1_row_count": observation_report["raw_m1_row_count"],
        "signal_count": observation_report["signal_count"],
        "signal_counts_by_pair": observation_report["signal_counts_by_pair"],
        "materialized_trade_row_count": observation_report["materialized_trade_row_count"],
        "base_candidate_count": observation_report["base_candidate_count"],
        "eligible_observation_count": observation_report["eligible_observation_count"],
        "position_skipped_count": observation_report["position_skipped_count"],
        "breaker_skipped_count": observation_report["breaker_skipped_count"],
        "breaker_trigger_count": observation_report["breaker_trigger_count"],
        "fast151_observation_status": observation_report["status"],
        "fast151_observation_status_reasons": observation_report["status_reasons"],
        "fast151_observation_report_path": observation_report["report_path"],
        "fast151_observation_report_sha256": observation_report["report_sha256"],
        "fast151_summary": observation_report["summary"],
        "fast151_gates": observation_report["gates"],
        "next_action": next_action_for_status(status),
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return plain_json({**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)})


def parse_optional_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def fast152_status(
    *,
    raw_m1_min_pair_end: pd.Timestamp | None,
    base_candidate_count: int,
    eligible_observation_count: int,
    observation_status: str,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "fast151_readiness_monitor"]
    if raw_m1_min_pair_end is None:
        return "FAST152_WAITING_FOR_M1_DATA", reasons + ["no_m1_data_loaded"]
    if raw_m1_min_pair_end <= PREREGISTRATION_TIMESTAMP:
        return (
            "FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA",
            reasons + ["raw_m1_min_pair_end_before_preregistration_timestamp"],
        )
    if eligible_observation_count > 0:
        return (
            "FAST152_FAST151_OBSERVATION_READY_FOR_REVIEW",
            reasons
            + [
                "accepted_operational_observation_exists",
                f"fast151_status:{observation_status}",
                "review_fast151_report_before_any_promotion",
            ],
        )
    if base_candidate_count > 0:
        return (
            "FAST152_DATA_FRESH_BUT_NO_ACCEPTED_OPERATIONAL_TRADE_YET",
            reasons
            + [
                "base_candidates_exist_after_preregistration",
                "all_candidates_filtered_or_waiting_for_operational_outcome",
                f"fast151_status:{observation_status}",
            ],
        )
    return (
        "FAST152_DATA_FRESH_NO_BASE_CANDIDATE_YET",
        reasons + ["post_preregistration_data_loaded", "no_base_policy_candidate_after_preregistration"],
    )


def next_action_for_status(status: str) -> str:
    if status == "FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA":
        return "refresh_or_import_m1_after_fast151_preregistration_then_rerun_fast151_and_fast152"
    if status == "FAST152_DATA_FRESH_NO_BASE_CANDIDATE_YET":
        return "keep_monitoring_new_m1_data_for_the_frozen_fast151_rule"
    if status == "FAST152_DATA_FRESH_BUT_NO_ACCEPTED_OPERATIONAL_TRADE_YET":
        return "inspect_fast151_candidate_skips_and_wait_for_completed_operational_observations"
    if status == "FAST152_FAST151_OBSERVATION_READY_FOR_REVIEW":
        return "review_fast151_gates_and_write_a_new_research_report_before_any_shadow_or_testnet_step"
    return "fix_data_availability_before_any_strategy_review"


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
        "raw_m1_end": report["raw_m1_end"],
        "raw_m1_min_pair_end": report["raw_m1_min_pair_end"],
        "raw_m1_end_by_pair": report["raw_m1_end_by_pair"],
        "has_post_preregistration_m1_data": report["has_post_preregistration_m1_data"],
        "base_candidate_count": report["base_candidate_count"],
        "eligible_observation_count": report["eligible_observation_count"],
        "fast151_observation_status": report["fast151_observation_status"],
        "next_action": report["next_action"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
