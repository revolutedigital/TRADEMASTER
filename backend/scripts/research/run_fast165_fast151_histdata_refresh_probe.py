"""Fast165: isolated HistData refresh probe for the Fast151 forward blocker.

Fast152 still waits for M1 bars after the Fast151 preregistration timestamp.
Fast165 checks whether that is a local-cache problem or a source-availability
problem by downloading the current HistData month into an isolated area,
normalizing it to M1, and comparing bounds with the canonical local M1 files.

No canonical market data is overwritten. No broker is contacted. No orders, no
ledger, no Demo/Testnet/LIVE.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.fx_histdata_ticks import build_pair, month_range  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    DEFAULT_M1_DIR,
    sha256_file,
)
from scripts.research.run_fast124_usdjpy_compression_observation import absolute_path  # noqa: E402
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    PREREGISTRATION_SHA256,
    PREREGISTRATION_TIMESTAMP,
    SAFETY_FLAGS,
    TARGET_PAIRS,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast165" / "fast151-histdata-refresh-probe"
)
DEFAULT_MONTH = PREREGISTRATION_TIMESTAMP.strftime("%Y-%m")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fresh-m1-dir", type=Path, default=None)
    parser.add_argument("--month", default=DEFAULT_MONTH)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)
    report = run(
        m1_dir=arguments.m1_dir,
        output_root=arguments.output_root,
        fresh_m1_dir=arguments.fresh_m1_dir,
        month=arguments.month,
        workers=arguments.workers,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
    *,
    m1_dir: Path = DEFAULT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fresh_m1_dir: Path | None = None,
    month: str = DEFAULT_MONTH,
    workers: int = 4,
) -> dict[str, Any]:
    m1_dir = absolute_path(m1_dir)
    output_root = absolute_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    local_bounds = pair_bounds_from_directory(m1_dir, month=month)
    fresh_source, fresh_dir, fresh_generation_rows = prepare_fresh_month(
        output_root=output_root,
        fresh_m1_dir=fresh_m1_dir,
        month=month,
        workers=workers,
    )
    fresh_bounds = pair_bounds_from_directory(fresh_dir, month=month)
    status, status_reasons = fast165_status(local_bounds=local_bounds, fresh_bounds=fresh_bounds)

    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "month": month,
        "fresh_source": fresh_source,
        "canonical_m1_dir": str(m1_dir),
        "fresh_m1_dir": str(fresh_dir),
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "local": bounds_summary(local_bounds),
        "fresh": bounds_summary(fresh_bounds),
        "fresh_generation_rows": fresh_generation_rows,
        "next_action": next_action_for_status(status),
    }
    summary_path = output_root / "histdata-refresh-probe-summary.json"
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast165_is_source_freshness_probe_not_strategy_approval",
            "canonical_m1_not_overwritten",
            "no_broker_contact",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast165_fast151_histdata_refresh_probe",
        "status": status,
        "status_reasons": status_reasons,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "summary": summary,
        "artifacts": {
            "histdata_refresh_probe_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            }
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def prepare_fresh_month(
    *,
    output_root: Path,
    fresh_m1_dir: Path | None,
    month: str,
    workers: int,
) -> tuple[str, Path, list[dict[str, Any]]]:
    if fresh_m1_dir is not None:
        return "supplied_fresh_m1_dir", absolute_path(fresh_m1_dir), []

    generated_dir = output_root / "fresh_m1"
    generated_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fast165-histdata-download-") as temporary_directory:
        download_workdir = Path(temporary_directory)
        for pair in TARGET_PAIRS:
            frame, pair_rows = build_pair(
                pair,
                month_range(month, month),
                download_workdir,
                oracle=None,
                workers=workers,
            )
            frame.to_parquet(
                generated_dir / f"{pair}_M1_{month_key(month)}_{month_key(month)}.parquet",
                compression="zstd",
            )
            rows.extend(pair_rows)
    return "histdata_downloaded_by_fast165", generated_dir, rows


def month_key(month: str) -> str:
    timestamp = pd.Timestamp(f"{month}-01")
    return timestamp.strftime("%Y%m")


def pair_bounds_from_directory(directory: Path, *, month: str) -> dict[str, dict[str, Any]]:
    bounds: dict[str, dict[str, Any]] = {}
    for pair in TARGET_PAIRS:
        path = directory / f"{pair}_M1_{month_key(month)}_{month_key(month)}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing M1 month file for {pair}: {path}")
        bounds[pair] = parquet_bounds(path)
    return bounds


def parquet_bounds(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    timestamps = frame["timestamp"] if "timestamp" in frame.columns else frame.index
    timestamps = pd.to_datetime(timestamps, utc=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": int(len(frame)),
        "start": None if frame.empty else timestamps.min().isoformat(),
        "end": None if frame.empty else timestamps.max().isoformat(),
    }


def bounds_summary(bounds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    end_values = [
        pd.Timestamp(row["end"])
        for row in bounds.values()
        if row.get("end") is not None
    ]
    start_values = [
        pd.Timestamp(row["start"])
        for row in bounds.values()
        if row.get("start") is not None
    ]
    return {
        "pair_bounds": bounds,
        "min_pair_start": None if not start_values else min(start_values).isoformat(),
        "max_pair_start": None if not start_values else max(start_values).isoformat(),
        "min_pair_end": None if not end_values else min(end_values).isoformat(),
        "max_pair_end": None if not end_values else max(end_values).isoformat(),
        "has_post_preregistration_m1_data": bool(end_values and min(end_values) > PREREGISTRATION_TIMESTAMP),
    }


def fast165_status(
    *,
    local_bounds: dict[str, dict[str, Any]],
    fresh_bounds: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "histdata_refresh_probe"]
    local_min_end = min_pair_end(local_bounds)
    fresh_min_end = min_pair_end(fresh_bounds)
    if fresh_min_end is None:
        return "FAST165_HISTDATA_REFRESH_NO_FRESH_M1_DATA", reasons + ["fresh_m1_data_missing"]
    if fresh_min_end <= PREREGISTRATION_TIMESTAMP:
        return (
            "FAST165_HISTDATA_REFRESH_CONFIRMS_SOURCE_NOT_POST_PREREGISTRATION",
            reasons + ["fresh_histdata_min_pair_end_before_preregistration_timestamp"],
        )
    if local_min_end is None or local_min_end < fresh_min_end:
        return (
            "FAST165_HISTDATA_REFRESH_SOURCE_NEWER_THAN_LOCAL_IMPORT_REQUIRED",
            reasons + ["fresh_histdata_has_post_preregistration_data", "canonical_m1_import_required"],
        )
    return (
        "FAST165_CANONICAL_M1_HAS_POST_PREREGISTRATION_DATA_RERUN_FAST151_FAST152",
        reasons + ["canonical_m1_has_post_preregistration_data"],
    )


def min_pair_end(bounds: dict[str, dict[str, Any]]) -> pd.Timestamp | None:
    values = [pd.Timestamp(row["end"]) for row in bounds.values() if row.get("end") is not None]
    return None if not values else min(values)


def next_action_for_status(status: str) -> str:
    if status == "FAST165_HISTDATA_REFRESH_CONFIRMS_SOURCE_NOT_POST_PREREGISTRATION":
        return "wait_for_histdata_to_publish_m1_after_fast151_preregistration_or_import_an_independent_m1_source"
    if status == "FAST165_HISTDATA_REFRESH_SOURCE_NEWER_THAN_LOCAL_IMPORT_REQUIRED":
        return "import_fresh_m1_into_canonical_dataset_then_rerun_fast151_and_fast152"
    if status == "FAST165_CANONICAL_M1_HAS_POST_PREREGISTRATION_DATA_RERUN_FAST151_FAST152":
        return "rerun_fast151_and_fast152_against_canonical_m1"
    return "fix_fresh_histdata_download_before_forward_readiness_review"


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "month": report["summary"]["month"],
        "fresh_source": report["summary"]["fresh_source"],
        "local_min_pair_end": report["summary"]["local"]["min_pair_end"],
        "fresh_min_pair_end": report["summary"]["fresh"]["min_pair_end"],
        "fresh_has_post_preregistration_m1_data": report["summary"]["fresh"][
            "has_post_preregistration_m1_data"
        ],
        "next_action": report["summary"]["next_action"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
