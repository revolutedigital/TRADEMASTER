"""Fast64 nested session/side filter walk-forward for frozen top-p trades.

The test is intentionally simple and causal: select a no-trade filter using only
UTC session, side, and frozen model probability on training blocks, then apply
that frozen filter to the held-out block. It audits whether the failed Fast43-
Fast53 top-p stream hides a robust time-of-day/side pocket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.trailing_portfolio import summarize_portfolio  # noqa: E402
from scripts.research.run_fast61_fast60_shadow_settlement import (  # noqa: E402
    _apply_one_position_constraint,
    _sha256_file,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast64-session-side-filter-walkforward"
FAST_NUMBER_PATTERN = re.compile(r"fast(?P<number>\d+)")
SESSION_WINDOWS = (
    ("utc_00_05", 0, 5),
    ("utc_06_11", 6, 11),
    ("utc_12_17", 12, 17),
    ("utc_18_23", 18, 23),
)


@dataclass(frozen=True)
class FilterSpec:
    name: str
    side: str | None = None
    hour_start: int | None = None
    hour_end: int | None = None
    probability_threshold: float | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--historical-fast-start", type=int, default=43)
    parser.add_argument("--historical-fast-end", type=int, default=53)
    parser.add_argument("--minimum-training-block-trades", type=int, default=3)
    parser.add_argument("--minimum-holdout-trades", type=int, default=3)
    arguments = parser.parse_args()

    report = run_fast64_session_side_filter_walkforward(
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        historical_fast_start=arguments.historical_fast_start,
        historical_fast_end=arguments.historical_fast_end,
        minimum_training_block_trades=arguments.minimum_training_block_trades,
        minimum_holdout_trades=arguments.minimum_holdout_trades,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast64_session_side_filter_walkforward(
    *,
    data_root: Path,
    output_root: Path,
    historical_fast_start: int = 43,
    historical_fast_end: int = 53,
    minimum_training_block_trades: int = 3,
    minimum_holdout_trades: int = 3,
) -> dict[str, Any]:
    _validate_parameters(
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        minimum_training_block_trades=minimum_training_block_trades,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    trade_paths = discover_historical_holdout_trade_paths(
        data_root,
        fast_start=historical_fast_start,
        fast_end=historical_fast_end,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if not trade_paths:
        report = _base_report(
            data_root=data_root,
            historical_fast_start=historical_fast_start,
            historical_fast_end=historical_fast_end,
            minimum_training_block_trades=minimum_training_block_trades,
            minimum_holdout_trades=minimum_holdout_trades,
        ) | {
            "status": "FAST64_MISSING_HISTORICAL_TRADES",
            "status_reasons": ["historical_holdout_trade_paths_not_found"],
            "historical_holdout_path_count": 0,
            "holdouts": [],
        }
        return _write_report(output_root, report)

    trades = load_historical_holdout_trades(trade_paths)
    baseline_rows = []
    for block, block_frame in trades.groupby("block", sort=True):
        baseline_rows.append(
            {
                "block": str(block),
                "summary": _summary(_apply_one_position_constraint(_records(block_frame))),
            }
        )
    holdouts = []
    for holdout_block in sorted(trades["block"].unique()):
        training = trades[trades["block"] != holdout_block].copy()
        holdout = trades[trades["block"] == holdout_block].copy()
        selected_filter, training_evaluations = select_training_filter(
            training,
            minimum_training_block_trades=minimum_training_block_trades,
        )
        holdout_rows = apply_filter(holdout, selected_filter)
        holdout_one_position = _apply_one_position_constraint(_records(holdout_rows))
        holdout_summary = _summary(holdout_one_position)
        holdouts.append(
            {
                "holdout_block": str(holdout_block),
                "selected_filter": asdict(selected_filter),
                "training_candidate_count": len(training_evaluations),
                "training_best": training_evaluations[0] if training_evaluations else None,
                "holdout_selected_signal_count": int(len(holdout_rows)),
                "holdout_summary": holdout_summary,
                "passed_stress": (
                    int(holdout_summary["trade_count"]) >= minimum_holdout_trades
                    and float(holdout_summary["stress_total_log_return_bps"]) > 0
                    and float(holdout_summary["stress_mean_bps"]) > 0
                ),
            }
        )

    aggregate = aggregate_holdouts(holdouts)
    status, status_reasons = fast64_status(
        aggregate=aggregate,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    holdouts_path = output_root / "holdouts.json"
    holdouts_path.write_text(json.dumps(holdouts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    baseline_path = output_root / "baseline_by_block.json"
    baseline_path.write_text(
        json.dumps(baseline_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = _base_report(
        data_root=data_root,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        minimum_training_block_trades=minimum_training_block_trades,
        minimum_holdout_trades=minimum_holdout_trades,
    ) | {
        "status": status,
        "status_reasons": status_reasons,
        "historical_holdout_path_count": len(trade_paths),
        "historical_holdout_paths": [str(path) for path in trade_paths],
        "trade_count": int(len(trades)),
        "block_count": int(trades["block"].nunique()),
        "baseline_by_block_path": str(baseline_path),
        "baseline_by_block_sha256": _sha256_file(baseline_path),
        "holdouts_path": str(holdouts_path),
        "holdouts_sha256": _sha256_file(holdouts_path),
        "aggregate": aggregate,
        "holdouts": holdouts,
    }
    return _write_report(output_root, report)


def discover_historical_holdout_trade_paths(
    data_root: Path,
    *,
    fast_start: int,
    fast_end: int,
) -> tuple[Path, ...]:
    paths = []
    for path in sorted(
        (data_root / "reports").glob(
            "latency-adjusted-frozen-top-p-fast*-lowcost-micro-holdout-*-100ms/"
            "frozen_holdout_trades.parquet"
        )
    ):
        match = FAST_NUMBER_PATTERN.search(str(path))
        if match is None:
            continue
        fast_number = int(match.group("number"))
        if fast_start <= fast_number <= fast_end:
            paths.append(path)
    return tuple(paths)


def load_historical_holdout_trades(paths: tuple[Path, ...]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["block"] = path.parent.name
        frame["utc_hour"] = pd.to_datetime(
            frame["decision_time_ms"], unit="ms", utc=True
        ).dt.hour
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["block", "decision_time_ms"],
        kind="stable",
    )


def select_training_filter(
    training: pd.DataFrame,
    *,
    minimum_training_block_trades: int,
) -> tuple[FilterSpec, list[dict[str, Any]]]:
    evaluations = [
        evaluate_filter(
            training,
            spec,
            minimum_block_trades=minimum_training_block_trades,
        )
        for spec in filter_candidates(training)
    ]
    evaluations = [item for item in evaluations if int(item["total_trade_count"]) > 0]
    evaluations.sort(
        key=lambda item: (
            int(item["positive_block_count"]),
            int(item["minimum_trade_block_count"]),
            float(item["min_block_stress_total_log_return_bps"]),
            float(item["stress_total_log_return_bps"]),
        ),
        reverse=True,
    )
    if not evaluations:
        return FilterSpec(name="all"), []
    return filter_from_dict(evaluations[0]["filter"]), evaluations


def filter_candidates(training: pd.DataFrame) -> list[FilterSpec]:
    specs = [FilterSpec(name="all")]
    for side in ("BUY", "SELL"):
        specs.append(FilterSpec(name=f"side={side}", side=side))
    for session_name, hour_start, hour_end in SESSION_WINDOWS:
        specs.append(FilterSpec(name=session_name, hour_start=hour_start, hour_end=hour_end))
        for side in ("BUY", "SELL"):
            specs.append(
                FilterSpec(
                    name=f"side={side}&{session_name}",
                    side=side,
                    hour_start=hour_start,
                    hour_end=hour_end,
                )
            )
    for quantile in (0.50, 0.70, 0.90, 0.95):
        threshold = float(training["probability"].quantile(quantile))
        specs.append(
            FilterSpec(
                name=f"probability>=q{quantile:.2f}:{threshold:.6f}",
                probability_threshold=threshold,
            )
        )
        for side in ("BUY", "SELL"):
            specs.append(
                FilterSpec(
                    name=f"side={side}&probability>=q{quantile:.2f}:{threshold:.6f}",
                    side=side,
                    probability_threshold=threshold,
                )
            )
    return specs


def evaluate_filter(
    frame: pd.DataFrame,
    spec: FilterSpec,
    *,
    minimum_block_trades: int,
) -> dict[str, Any]:
    block_summaries = []
    total_rows = []
    for block, block_frame in frame.groupby("block", sort=True):
        filtered = apply_filter(block_frame, spec)
        one_position = _apply_one_position_constraint(_records(filtered))
        summary = _summary(one_position)
        block_summaries.append({"block": str(block), **summary})
        total_rows.extend(one_position)
    total_summary = _summary(total_rows)
    return {
        "filter": asdict(spec),
        "block_count": len(block_summaries),
        "positive_block_count": sum(
            float(row["stress_total_log_return_bps"]) > 0
            and float(row["stress_mean_bps"]) > 0
            for row in block_summaries
        ),
        "minimum_trade_block_count": sum(
            int(row["trade_count"]) >= minimum_block_trades for row in block_summaries
        ),
        "min_block_stress_total_log_return_bps": min(
            (float(row["stress_total_log_return_bps"]) for row in block_summaries),
            default=0.0,
        ),
        "total_trade_count": int(total_summary["trade_count"]),
        "stress_total_log_return_bps": float(total_summary["stress_total_log_return_bps"]),
        "stress_mean_bps": float(total_summary["stress_mean_bps"]),
        "expected_win_rate": float(total_summary["expected_win_rate"]),
        "block_summaries": block_summaries,
    }


def apply_filter(frame: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    mask = pd.Series([True] * len(frame), index=frame.index)
    if spec.side is not None:
        mask &= frame["side"].eq(spec.side)
    if spec.hour_start is not None and spec.hour_end is not None:
        mask &= frame["utc_hour"].between(spec.hour_start, spec.hour_end)
    if spec.probability_threshold is not None:
        mask &= frame["probability"].ge(spec.probability_threshold)
    return frame[mask].copy()


def aggregate_holdouts(holdouts: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [holdout["holdout_summary"] for holdout in holdouts]
    return {
        "holdout_count": len(holdouts),
        "passed_stress_count": sum(bool(holdout["passed_stress"]) for holdout in holdouts),
        "minimum_trade_holdout_count": sum(int(row["trade_count"]) >= 3 for row in summaries),
        "stress_total_log_return_bps": float(
            sum(float(row["stress_total_log_return_bps"]) for row in summaries)
        ),
        "expected_total_log_return_bps": float(
            sum(float(row["expected_total_log_return_bps"]) for row in summaries)
        ),
        "trade_count": int(sum(int(row["trade_count"]) for row in summaries)),
        "min_holdout_stress_total_log_return_bps": min(
            (float(row["stress_total_log_return_bps"]) for row in summaries),
            default=0.0,
        ),
        "selected_filters": [
            holdout["selected_filter"]["name"] for holdout in holdouts
        ],
    }


def fast64_status(
    *,
    aggregate: dict[str, Any],
    minimum_holdout_trades: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "diagnostic_only", "no_order_execution"]
    holdout_count = int(aggregate["holdout_count"])
    if holdout_count <= 0:
        return "FAST64_NO_HOLDOUTS", reasons + ["no_holdout_blocks_evaluated"]
    if int(aggregate["passed_stress_count"]) < holdout_count:
        return "FAST64_SESSION_SIDE_FILTER_FAILED_WALKFORWARD", reasons + [
            f"passed_stress_holdouts:{aggregate['passed_stress_count']}<{holdout_count}",
            f"minimum_holdout_trades:{minimum_holdout_trades}",
        ]
    if float(aggregate["stress_total_log_return_bps"]) <= 0:
        return "FAST64_SESSION_SIDE_FILTER_FAILED_WALKFORWARD", reasons + [
            "aggregate_stress_total_non_positive"
        ]
    return "FAST64_SESSION_SIDE_FILTER_CANDIDATE_RESEARCH_ONLY", reasons + [
        "requires_prospective_preregistration_before_shadow_or_testnet"
    ]


def filter_from_dict(payload: dict[str, Any]) -> FilterSpec:
    return FilterSpec(
        name=str(payload["name"]),
        side=payload.get("side"),
        hour_start=payload.get("hour_start"),
        hour_end=payload.get("hour_end"),
        probability_threshold=payload.get("probability_threshold"),
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return summarize_portfolio(pd.DataFrame())
    return summarize_portfolio(pd.DataFrame(rows))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.to_dict("records")


def _base_report(
    *,
    data_root: Path,
    historical_fast_start: int,
    historical_fast_end: int,
    minimum_training_block_trades: int,
    minimum_holdout_trades: int,
) -> dict[str, Any]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast64_is_diagnostic_walkforward_only",
            "filters_are_session_side_probability_only",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast64_session_side_filter_walkforward",
        "data_root": str(data_root),
        "historical_fast_start": historical_fast_start,
        "historical_fast_end": historical_fast_end,
        "minimum_training_block_trades": minimum_training_block_trades,
        "minimum_holdout_trades": minimum_holdout_trades,
    }


def _validate_parameters(
    *,
    historical_fast_start: int,
    historical_fast_end: int,
    minimum_training_block_trades: int,
    minimum_holdout_trades: int,
) -> None:
    if historical_fast_start > historical_fast_end:
        raise ValueError("historical_fast_start cannot exceed historical_fast_end")
    if minimum_training_block_trades < 1:
        raise ValueError("minimum_training_block_trades must be positive")
    if minimum_holdout_trades < 1:
        raise ValueError("minimum_holdout_trades must be positive")


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "block_count": report.get("block_count"),
        "trade_count": report.get("trade_count"),
        "aggregate": report.get("aggregate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
