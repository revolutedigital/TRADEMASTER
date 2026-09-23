"""Materialize complete Fast57 days and re-run the frozen prospective gate.

This script is intentionally boring: it downloads only complete public daily
archives, builds the same low-cost research partitions used by Fast57, and then
calls the pre-registered Fast57 evaluator without changing model, threshold,
top-p, gate, trailing policy, latency, or costs.

It is research-only, never sends orders, and never approves a strategy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.research_dataset import (  # noqa: E402
    ResearchDatasetConfig,
    build_research_partition,
)
from scripts.research.build_microstructure_dataset import (  # noqa: E402
    DEFAULT_ROOT as DEFAULT_DATA_ROOT,
)
from scripts.research.build_microstructure_dataset import (
    build as build_microstructure_dataset,
)
from scripts.research.run_fast57_fixed_density_prospective_gate import (  # noqa: E402
    run_fast57,
)
from scripts.research.run_latency_adjusted_density_gate_walkforward import (  # noqa: E402
    _parse_block_specs,
)


DEFAULT_ARTIFACT_REPORT = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "latency-adjusted-frozen-top-p-fast56-new-holdout-sep21-sep22-100ms"
    / "report.json"
)
DEFAULT_SEED_RESEARCH_ROOT = DEFAULT_DATA_ROOT / "research-lowcost-fast56-holdout-new"
DEFAULT_PROSPECTIVE_RESEARCH_ROOT = DEFAULT_DATA_ROOT / "research-lowcost-fast57-prospective"
DEFAULT_TRADE_ROOT = DEFAULT_DATA_ROOT / "normalized" / "aggTrades"
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT / "reports" / "fast57-fixed-density-prospective-update-current"
)
DEFAULT_START_DATE = date(2026, 9, 23)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--market",
        choices=("usdm_perpetual",),
        default="usdm_perpetual",
        help="Fast57 is frozen on USD-M perpetual aggTrades.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=DEFAULT_START_DATE,
        help="First complete UTC day to materialize beyond the Fast56 seed block.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Last complete UTC day to materialize. Defaults to yesterday UTC.",
    )
    parser.add_argument(
        "--as-of-date",
        type=date.fromisoformat,
        help="UTC date used to compute yesterday in tests/replays. Defaults to now UTC.",
    )
    parser.add_argument("--artifact-report", type=Path, default=DEFAULT_ARTIFACT_REPORT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--trade-root", type=Path, default=DEFAULT_TRADE_ROOT)
    parser.add_argument("--seed-research-root", type=Path, default=DEFAULT_SEED_RESEARCH_ROOT)
    parser.add_argument(
        "--prospective-research-root",
        type=Path,
        default=DEFAULT_PROSPECTIVE_RESEARCH_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--download-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download missing complete daily aggTrades archives. Use --no-download-missing to audit only.",
    )
    parser.add_argument(
        "--force-rebuild-research",
        action="store_true",
        help="Rebuild existing Fast57 prospective research partitions.",
    )
    arguments = parser.parse_args()

    report = asyncio.run(run_fast57_prospective_update(arguments))
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


async def run_fast57_prospective_update(arguments: argparse.Namespace) -> dict[str, Any]:
    as_of_date = arguments.as_of_date or datetime.now(tz=UTC).date()
    end_date = arguments.end_date or latest_complete_utc_archive_date(as_of_date)
    if end_date < arguments.start_date:
        update_dates: list[date] = []
    else:
        update_dates = list(date_range(arguments.start_date, end_date))

    before_download_missing = [
        utc_date
        for utc_date in update_dates
        if not normalized_trade_partition_exists(arguments.trade_root, utc_date)
    ]
    download_summary: dict[str, Any] | None = None
    if before_download_missing and arguments.download_missing:
        download_summary = await build_microstructure_dataset(
            argparse.Namespace(
                symbol=arguments.symbol,
                market=arguments.market,
                start_date=min(before_download_missing),
                end_date=max(before_download_missing),
                kinds=["aggTrades"],
                root=arguments.data_root,
            )
        )

    after_download_missing = [
        utc_date
        for utc_date in update_dates
        if not normalized_trade_partition_exists(arguments.trade_root, utc_date)
    ]
    materialized_research: list[dict[str, Any]] = []
    skipped_existing_research: list[str] = []
    missing_research_inputs: list[str] = []
    research_config = fast57_research_dataset_config()
    for utc_date in update_dates:
        if utc_date in after_download_missing:
            missing_research_inputs.append(utc_date.isoformat())
            continue
        if (
            research_partition_exists(arguments.prospective_research_root, utc_date)
            and not arguments.force_rebuild_research
        ):
            skipped_existing_research.append(utc_date.isoformat())
            continue
        result = build_research_partition(
            source_root=arguments.trade_root,
            output_root=arguments.prospective_research_root,
            utc_date=utc_date,
            config=research_config,
        )
        materialized_research.append(result.__dict__)

    block_arguments = fast57_block_arguments(
        seed_research_root=arguments.seed_research_root,
        prospective_research_root=arguments.prospective_research_root,
    )
    if block_arguments:
        fast57_report = run_fast57(
            artifact_report=arguments.artifact_report,
            block_specs=_parse_block_specs(block_arguments),
            trade_root=arguments.trade_root,
            output_root=arguments.output_root / "fast57",
            lookback_seconds=300,
            limit=0,
            policy_set="low_cost",
            policy_name="micro_8_3",
            tail_fraction=0.01,
            horizon_seconds=300,
            latency_ms=100,
            expected_cost_bps=1.0,
            stress_cost_bps=2.0,
            minimum_days=10,
            minimum_trades=100,
            maximum_days=30,
        )
    else:
        fast57_report = {
            "status": "FAST57_NO_RESEARCH_PARTITIONS",
            "status_reasons": ["seed_and_prospective_research_partitions_missing"],
            "report_path": None,
        }

    arguments.output_root.mkdir(parents=True, exist_ok=True)
    update_report_path = arguments.output_root / "update-report.json"
    report: dict[str, Any] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "approval_reasons": [
            "fast57_update_research_only",
            "offline_public_daily_aggtrades_replay_not_exchange_execution",
            "runner_keeps_approved_strategy_false",
        ],
        "experiment_stage": "fast57_prospective_update",
        "as_of_date": as_of_date.isoformat(),
        "start_date": arguments.start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "new_complete_dates": [utc_date.isoformat() for utc_date in update_dates],
        "normalized_missing_before_download": [
            utc_date.isoformat() for utc_date in before_download_missing
        ],
        "normalized_missing_after_download": [
            utc_date.isoformat() for utc_date in after_download_missing
        ],
        "download_attempted": bool(before_download_missing and arguments.download_missing),
        "download_summary": download_summary,
        "research_config": {
            "decision_stride_seconds": research_config.decision_stride_seconds,
            "horizons_seconds": list(research_config.horizons_seconds),
            "expected_cost_bps": research_config.expected_cost_bps,
            "stress_cost_bps": research_config.stress_cost_bps,
            "initial_stop_bps": research_config.initial_stop_bps,
            "config_sha256": research_config.sha256,
        },
        "materialized_research": materialized_research,
        "skipped_existing_research": skipped_existing_research,
        "missing_research_inputs": missing_research_inputs,
        "block_arguments": block_arguments,
        "fast57_status": fast57_report["status"],
        "fast57_status_reasons": fast57_report["status_reasons"],
        "fast57_report_path": fast57_report["report_path"],
        "fast57_complete_utc_days": fast57_report.get("complete_utc_days"),
        "fast57_gated_summary": fast57_report.get("gated_summary"),
        "fast57_baseline_summary": fast57_report.get("baseline_summary"),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(update_report_path),
    }
    update_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def latest_complete_utc_archive_date(as_of_date: date) -> date:
    return as_of_date - timedelta(days=1)


def date_range(start_date: date, end_date: date) -> tuple[date, ...]:
    if end_date < start_date:
        return ()
    day_count = (end_date - start_date).days
    return tuple(start_date + timedelta(days=offset) for offset in range(day_count + 1))


def normalized_trade_partition_exists(trade_root: Path, utc_date: date) -> bool:
    return (trade_root / f"date={utc_date.isoformat()}" / "events.parquet").exists()


def research_partition_exists(research_root: Path, utc_date: date) -> bool:
    return (research_root / f"date={utc_date.isoformat()}" / "research_rows.parquet").exists()


def has_research_partitions(research_root: Path) -> bool:
    return any(research_root.glob("date=*/research_rows.parquet"))


def fast57_block_arguments(
    *,
    seed_research_root: Path,
    prospective_research_root: Path,
) -> list[str]:
    blocks: list[str] = []
    if has_research_partitions(seed_research_root):
        blocks.append(f"seed={seed_research_root}")
    if has_research_partitions(prospective_research_root):
        blocks.append(f"prospective={prospective_research_root}")
    return blocks


def fast57_research_dataset_config() -> ResearchDatasetConfig:
    return ResearchDatasetConfig(
        decision_stride_seconds=5,
        horizons_seconds=(30, 120, 300),
        expected_cost_bps=1.0,
        stress_cost_bps=2.0,
        initial_stop_bps=20.0,
    )


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "approved_strategy": report["approved_strategy"],
        "new_complete_dates": report["new_complete_dates"],
        "materialized_research": report["materialized_research"],
        "fast57_status": report["fast57_status"],
        "fast57_status_reasons": report["fast57_status_reasons"],
        "fast57_complete_utc_days": report["fast57_complete_utc_days"],
        "fast57_gated_summary": report["fast57_gated_summary"],
        "fast57_baseline_summary": report["fast57_baseline_summary"],
        "report_path": report["report_path"],
        "fast57_report_path": report["fast57_report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
