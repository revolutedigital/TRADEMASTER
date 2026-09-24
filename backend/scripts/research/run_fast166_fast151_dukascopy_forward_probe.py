"""Fast166: Dukascopy forward-source probe for the Fast151 blocker.

Fast165 proved HistData still stops before the Fast151 preregistration
timestamp. Fast166 checks the next bounded option already documented in the
project: Dukascopy M1 as a small independent forward-data probe.

The probe downloads only the preregistration date and a control date into an
isolated cache. It does not overwrite canonical M1, does not run discovery, and
does not touch broker/execution/ledger state.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.fx_dataset import ALL_PAIRS, USER_AGENT  # noqa: E402
from scripts.research.fx_dataset_m1 import (  # noqa: E402
    build_symbol_frame,
    cache_path,
    ensure_cached,
    quality_report_m1,
)
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast124_usdjpy_compression_observation import absolute_path  # noqa: E402
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    PREREGISTRATION_SHA256,
    PREREGISTRATION_TIMESTAMP,
    SAFETY_FLAGS,
    TARGET_PAIRS,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast166" / "fast151-dukascopy-forward-probe"
)
DEFAULT_PREREGISTRATION_DATE = PREREGISTRATION_TIMESTAMP.date()
DEFAULT_CONTROL_DATE = DEFAULT_PREREGISTRATION_DATE - timedelta(days=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-date", type=parse_date, default=DEFAULT_PREREGISTRATION_DATE)
    parser.add_argument("--control-date", type=parse_date, default=DEFAULT_CONTROL_DATE)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)
    report = run(
        output_root=arguments.output_root,
        preregistration_date=arguments.preregistration_date,
        control_date=arguments.control_date,
        workers=arguments.workers,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def run(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_date: date = DEFAULT_PREREGISTRATION_DATE,
    control_date: date = DEFAULT_CONTROL_DATE,
    workers: int = 4,
) -> dict[str, Any]:
    output_root = absolute_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = output_root / "cache"
    frame_dir = output_root / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0) as client:
        for probe_date in (control_date, preregistration_date):
            for pair in TARGET_PAIRS:
                rows.append(
                    probe_pair_date(
                        client=client,
                        pair=pair,
                        probe_date=probe_date,
                        cache_dir=cache_dir,
                        frame_dir=frame_dir,
                        workers=workers,
                    )
                )

    status, status_reasons = fast166_status(
        rows=rows,
        preregistration_date=preregistration_date,
        control_date=control_date,
    )
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_date": preregistration_date.isoformat(),
        "control_date": control_date.isoformat(),
        "target_pairs": list(TARGET_PAIRS),
        "rows": rows,
        "date_summary": summarize_dates(rows),
        "next_action": next_action_for_status(status),
    }
    summary_path = output_root / "dukascopy-forward-probe-summary.json"
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast166_is_bounded_data_source_probe_not_strategy_approval",
            "canonical_m1_not_overwritten",
            "dukascopy_used_only_as_small_independent_forward_probe",
            "no_broker_contact",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast166_fast151_dukascopy_forward_probe",
        "status": status,
        "status_reasons": status_reasons,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "summary": summary,
        "artifacts": {
            "dukascopy_forward_probe_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            }
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def probe_pair_date(
    *,
    client: httpx.Client,
    pair: str,
    probe_date: date,
    cache_dir: Path,
    frame_dir: Path,
    workers: int,
) -> dict[str, Any]:
    instrument = ALL_PAIRS[pair]
    stats = ensure_cached(client, pair, [probe_date], cache_dir, workers=workers)
    frame = build_symbol_frame(cache_dir, instrument, [probe_date])
    frame_path = frame_dir / f"{pair}_M1_{probe_date:%Y%m%d}_{probe_date:%Y%m%d}.parquet"
    frame.to_parquet(frame_path, compression="zstd")
    quality = quality_report_m1(frame, instrument)
    timestamp_index = pd.to_datetime(frame.index, utc=True)
    post_preregistration_minutes = (
        0
        if frame.empty
        else int((timestamp_index > PREREGISTRATION_TIMESTAMP).sum())
    )
    return {
        "pair": pair,
        "date": probe_date.isoformat(),
        "row_count": int(len(frame)),
        "first": None if frame.empty else timestamp_index.min().isoformat(),
        "last": None if frame.empty else timestamp_index.max().isoformat(),
        "post_preregistration_minutes": post_preregistration_minutes,
        "has_post_preregistration_minutes": post_preregistration_minutes > 0,
        "frame_path": str(frame_path),
        "frame_sha256": sha256_file(frame_path),
        "cache_bytes": {
            side: cache_path(cache_dir, pair, probe_date, side).stat().st_size
            for side in ("BID", "ASK")
        },
        "files_requested": stats.files_requested,
        "files_downloaded": stats.files_downloaded,
        "files_from_cache": stats.files_from_cache,
        "download_seconds": stats.seconds,
        "quality": quality,
    }


def summarize_dates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for probe_date in sorted({row["date"] for row in rows}):
        date_rows = [row for row in rows if row["date"] == probe_date]
        summary[probe_date] = {
            "pairs": [row["pair"] for row in date_rows],
            "total_rows": int(sum(row["row_count"] for row in date_rows)),
            "pairs_with_rows": [row["pair"] for row in date_rows if row["row_count"] > 0],
            "pairs_with_post_preregistration_minutes": [
                row["pair"] for row in date_rows if row["has_post_preregistration_minutes"]
            ],
            "min_last": min((row["last"] for row in date_rows if row["last"] is not None), default=None),
            "max_last": max((row["last"] for row in date_rows if row["last"] is not None), default=None),
            "cache_bytes_total": int(
                sum(sum(row["cache_bytes"].values()) for row in date_rows)
            ),
        }
    return summary


def fast166_status(
    *,
    rows: list[dict[str, Any]],
    preregistration_date: date,
    control_date: date,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "dukascopy_forward_probe"]
    prereg_rows = [row for row in rows if row["date"] == preregistration_date.isoformat()]
    control_rows = [row for row in rows if row["date"] == control_date.isoformat()]
    prereg_post_pairs = {row["pair"] for row in prereg_rows if row["has_post_preregistration_minutes"]}
    control_pairs_with_rows = {row["pair"] for row in control_rows if row["row_count"] > 0}
    prereg_pairs_with_rows = {row["pair"] for row in prereg_rows if row["row_count"] > 0}
    target_pairs = set(TARGET_PAIRS)

    if prereg_post_pairs == target_pairs:
        return (
            "FAST166_DUKASCOPY_HAS_POST_PREREGISTRATION_M1_IMPORT_REQUIRED",
            reasons + ["all_target_pairs_have_post_preregistration_m1", "canonical_overlay_import_required"],
        )
    if control_pairs_with_rows and not prereg_pairs_with_rows:
        return (
            "FAST166_DUKASCOPY_SOURCE_ALIVE_BUT_PREREGISTRATION_DATE_UNAVAILABLE",
            reasons + ["control_date_has_m1_data", "preregistration_date_has_no_m1_data"],
        )
    if prereg_pairs_with_rows:
        return (
            "FAST166_DUKASCOPY_PREREGISTRATION_DATE_INCOMPLETE_FOR_FAST151",
            reasons
            + [
                "preregistration_date_has_some_m1_data",
                "not_all_target_pairs_have_post_preregistration_m1",
            ],
        )
    return (
        "FAST166_DUKASCOPY_NO_RECENT_M1_DATA",
        reasons + ["control_date_missing_m1_data", "preregistration_date_missing_m1_data"],
    )


def next_action_for_status(status: str) -> str:
    if status == "FAST166_DUKASCOPY_HAS_POST_PREREGISTRATION_M1_IMPORT_REQUIRED":
        return "build_isolated_overlay_m1_dir_then_rerun_fast151_fast152_without_overwriting_canonical_histdata"
    if status == "FAST166_DUKASCOPY_SOURCE_ALIVE_BUT_PREREGISTRATION_DATE_UNAVAILABLE":
        return "wait_for_dukascopy_or_histdata_to_publish_preregistration_date_or_try_a_broker_exported_m1_file"
    if status == "FAST166_DUKASCOPY_PREREGISTRATION_DATE_INCOMPLETE_FOR_FAST151":
        return "wait_until_all_target_pairs_have_post_preregistration_m1_or_import_another_independent_source"
    return "fix_independent_m1_source_before_forward_readiness_review"


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "control_date": summary["control_date"],
        "preregistration_date": summary["preregistration_date"],
        "date_summary": summary["date_summary"],
        "next_action": summary["next_action"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
