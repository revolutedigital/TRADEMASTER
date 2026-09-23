"""Materialize a short prospective WAL with book features and report readiness.

This is a research-only scout for captures that are shorter than a full UTC day.
It exists to make the next viable lane concrete: trade + fresh book/depth,
optionally mark and spot-perp, with exact causal path labels. It does not train
or approve a strategy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.research_dataset import (  # noqa: E402
    ResearchDatasetConfig,
    build_research_rows,
    load_book_interval,
    load_mark_interval,
    load_trade_interval,
)


@dataclass(frozen=True)
class StreamSummary:
    name: str
    present: bool
    row_count: int
    first_event_time_ms: int | None
    last_event_time_ms: int | None
    first_event_time: str | None
    last_event_time: str | None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wal-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "prospective-wal-smoke-v9",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-wal-scout",
    )
    parser.add_argument("--horizons-seconds", default="30")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--minimum-decisions-for-modeling", type=int, default=200)
    arguments = parser.parse_args()
    horizons = _parse_horizons(arguments.horizons_seconds)
    report = run_book_wal_scout(
        wal_root=arguments.wal_root,
        output_root=arguments.output_root,
        horizons_seconds=horizons,
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        minimum_decisions_for_modeling=arguments.minimum_decisions_for_modeling,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_wal_scout(
    *,
    wal_root: Path,
    output_root: Path,
    horizons_seconds: tuple[int, ...] = (30,),
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    minimum_decisions_for_modeling: int = 200,
) -> dict[str, object]:
    if decision_stride_seconds <= 0:
        raise ValueError("decision_stride_seconds must be positive")
    if max_book_staleness_ms < 0:
        raise ValueError("max_book_staleness_ms cannot be negative")
    if minimum_decisions_for_modeling < 1:
        raise ValueError("minimum_decisions_for_modeling must be positive")
    config = ResearchDatasetConfig(
        decision_stride_seconds=decision_stride_seconds,
        horizons_seconds=horizons_seconds,
        expected_cost_bps=12.0,
        stress_cost_bps=24.0,
        initial_stop_bps=20.0,
        feature_windows_seconds=(1, 5, 30),
        require_book_features=True,
        max_book_staleness_ms=max_book_staleness_ms,
    )
    streams = {
        "trade": summarize_stream(wal_root / "trade"),
        "depth": summarize_stream(wal_root / "depth"),
        "mark_price": summarize_stream(wal_root / "mark_price"),
        "spot_trade": summarize_stream(wal_root / "spot_trade"),
    }
    base_report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "experiment_stage": "book_wal_scout",
        "wal_root": str(wal_root),
        "config": asdict(config),
        "minimum_decisions_for_modeling": minimum_decisions_for_modeling,
        "streams": {name: asdict(summary) for name, summary in streams.items()},
    }
    missing_required = [
        name for name in ("trade", "depth") if not streams[name].present
    ]
    if missing_required:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "MISSING_REQUIRED_STREAMS",
                "status_reasons": [f"{name}_missing" for name in missing_required],
                "modeling_ready": False,
            },
        )
    decisions = decision_times_for_wal(
        trade=streams["trade"],
        book=streams["depth"],
        config=config,
    )
    if len(decisions) == 0:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "INSUFFICIENT_INTERVAL",
                "status_reasons": ["no_complete_decisions_after_history_book_and_label_bounds"],
                "decision_count": 0,
                "row_count": 0,
                "modeling_ready": False,
            },
        )

    interval_start = _dt(min(_not_none(streams["trade"].first_event_time_ms), decisions[0]))
    interval_end = _dt(_not_none(streams["trade"].last_event_time_ms))
    trades = load_trade_interval(
        wal_root / "trade",
        interval_start,
        interval_end,
        expected_product="usdm_perpetual",
    )
    book_events = load_book_interval(wal_root / "depth", interval_start, interval_end)
    mark_events = (
        load_mark_interval(wal_root / "mark_price", interval_start, interval_end)
        if streams["mark_price"].present
        else None
    )
    spot_events = (
        load_trade_interval(
            wal_root / "spot_trade",
            interval_start,
            interval_end,
            expected_product="spot",
        )
        if streams["spot_trade"].present
        else None
    )
    rows = build_research_rows(
        trades,
        decisions,
        config,
        book_events=book_events,
        mark_events=mark_events,
        spot_trade_events=spot_events,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows_path = output_root / "research_rows.parquet"
    rows.to_parquet(rows_path, index=False, compression="zstd")
    decision_count = int(rows["decision_time_ms"].nunique())
    summary = summarize_rows(rows)
    modeling_ready = decision_count >= minimum_decisions_for_modeling
    report = {
        **base_report,
        "status": "READY_FOR_SCOUT_MODELING" if modeling_ready else "CAPTURE_TOO_SHORT",
        "status_reasons": []
        if modeling_ready
        else [
            f"decision_count_below_minimum:{decision_count}<{minimum_decisions_for_modeling}"
        ],
        "modeling_ready": modeling_ready,
        "decision_count": decision_count,
        "row_count": int(len(rows)),
        "rows_path": str(rows_path),
        "interval_start": _iso(int(rows["decision_time_ms"].min())),
        "interval_end": _iso(int(rows["decision_time_ms"].max())),
        "summary": summary,
    }
    return _write_report(output_root, report)


def summarize_stream(stream_root: Path) -> StreamSummary:
    first_ms: int | None = None
    last_ms: int | None = None
    row_count = 0
    for path in sorted(stream_root.glob("date=*/events.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                event_time_ms = _event_time_ms(payload)
                row_count += 1
                first_ms = event_time_ms if first_ms is None else min(first_ms, event_time_ms)
                last_ms = event_time_ms if last_ms is None else max(last_ms, event_time_ms)
    return StreamSummary(
        name=stream_root.name,
        present=row_count > 0,
        row_count=row_count,
        first_event_time_ms=first_ms,
        last_event_time_ms=last_ms,
        first_event_time=_iso(first_ms) if first_ms is not None else None,
        last_event_time=_iso(last_ms) if last_ms is not None else None,
    )


def decision_times_for_wal(
    *,
    trade: StreamSummary,
    book: StreamSummary,
    config: ResearchDatasetConfig,
) -> np.ndarray:
    trade_first = _not_none(trade.first_event_time_ms)
    trade_last = _not_none(trade.last_event_time_ms)
    book_first = _not_none(book.first_event_time_ms)
    book_last = _not_none(book.last_event_time_ms)
    history_ms = max(config.feature_windows_seconds) * 1_000
    horizon_ms = max(config.horizons_seconds) * 1_000
    stride_ms = config.decision_stride_seconds * 1_000
    first_decision = max(trade_first + history_ms, book_first)
    last_decision = min(trade_last - horizon_ms, book_last + config.max_book_staleness_ms)
    first_decision = int(math.ceil(first_decision / stride_ms) * stride_ms)
    last_decision = int(math.floor(last_decision / stride_ms) * stride_ms)
    if last_decision < first_decision:
        return np.array([], dtype=np.int64)
    return np.arange(first_decision, last_decision + stride_ms, stride_ms, dtype=np.int64)


def summarize_rows(rows: pd.DataFrame) -> dict[str, object]:
    grouped = (
        rows.groupby(["horizon_seconds", "side"], dropna=False)
        .agg(
            rows=("decision_time_ms", "count"),
            expected_target_rate=("paid_expected_before_stop", "mean"),
            stress_target_rate=("paid_stress_before_stop", "mean"),
            terminal_mean_bps=("terminal_bps", "mean"),
            mfe_mean_bps=("mfe_bps", "mean"),
            mae_mean_bps=("mae_bps", "mean"),
        )
        .reset_index()
    )
    book_age = rows["book_update_age_ms"].to_numpy(dtype=np.float64)
    return {
        "book_available_min": float(rows["book_available"].min()),
        "book_update_age_ms_max": float(np.max(book_age)),
        "book_update_age_ms_mean": float(np.mean(book_age)),
        "has_mark_features": "mark_index_basis_bps" in rows.columns,
        "has_spot_perp_features": "spot_perp_basis_bps" in rows.columns,
        "targets_by_horizon_side": grouped.to_dict(orient="records"),
    }


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _event_time_ms(payload: dict[str, Any]) -> int:
    value = payload.get("event_time")
    if value is None:
        value = payload.get("time")
    if isinstance(value, int | float):
        return int(value)
    if not isinstance(value, str):
        raise ValueError("WAL event is missing event_time")
    return int(pd.Timestamp(value).timestamp() * 1000)


def _parse_horizons(value: str) -> tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("--horizons-seconds must contain positive integers")
    return horizons


def _not_none(value: int | None) -> int:
    if value is None:
        raise ValueError("required stream has no event time bounds")
    return value


def _dt(event_time_ms: int) -> datetime:
    return datetime.fromtimestamp(event_time_ms / 1000, tz=UTC)


def _iso(event_time_ms: int | None) -> str | None:
    if event_time_ms is None:
        return None
    return _dt(event_time_ms).isoformat().replace("+00:00", "Z")


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "status_reasons": report.get("status_reasons", []),
        "decision_count": report.get("decision_count", 0),
        "row_count": report.get("row_count", 0),
        "modeling_ready": report.get("modeling_ready", False),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
