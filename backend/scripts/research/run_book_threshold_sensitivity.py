"""Measure reachable short-horizon targets on a prospective book WAL.

This is a research-only diagnostic. It asks whether target sizes are physically
reachable before the stop on the captured path. It does not fit a model, choose
entries, or approve a strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

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
from scripts.research.run_book_wal_scout import (  # noqa: E402
    StreamSummary,
    decision_times_for_wal,
    summarize_stream,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wal-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "prospective-wal-fast16-18m",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-threshold-sensitivity",
    )
    parser.add_argument("--horizons-seconds", default="30")
    parser.add_argument("--thresholds-bps", default="0.5,1,2,3,5,8,12,24")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--minimum-decisions-for-modeling", type=int, default=200)
    arguments = parser.parse_args()
    report = run_book_threshold_sensitivity(
        wal_root=arguments.wal_root,
        output_root=arguments.output_root,
        horizons_seconds=_parse_int_list(arguments.horizons_seconds),
        thresholds_bps=_parse_float_list(arguments.thresholds_bps),
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        minimum_decisions_for_modeling=arguments.minimum_decisions_for_modeling,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_threshold_sensitivity(
    *,
    wal_root: Path,
    output_root: Path,
    horizons_seconds: tuple[int, ...] = (30,),
    thresholds_bps: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 24.0),
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    initial_stop_bps: float = 20.0,
    minimum_decisions_for_modeling: int = 200,
) -> dict[str, object]:
    if not thresholds_bps or any(threshold <= 0 for threshold in thresholds_bps):
        raise ValueError("thresholds_bps must contain positive values")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    base_config = ResearchDatasetConfig(
        decision_stride_seconds=decision_stride_seconds,
        horizons_seconds=horizons_seconds,
        expected_cost_bps=thresholds_bps[0],
        stress_cost_bps=thresholds_bps[0],
        initial_stop_bps=initial_stop_bps,
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
        "experiment_stage": "book_threshold_sensitivity",
        "wal_root": str(wal_root),
        "config": {
            **asdict(base_config),
            "thresholds_bps": list(thresholds_bps),
        },
        "minimum_decisions_for_modeling": minimum_decisions_for_modeling,
        "streams": {name: asdict(summary) for name, summary in streams.items()},
        "interpretation": (
            "gross_path_payoff_mean_bps is a reachability diagnostic, not fee-net edge"
        ),
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
        config=base_config,
    )
    if len(decisions) == 0:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "INSUFFICIENT_INTERVAL",
                "status_reasons": ["no_complete_decisions_after_history_book_and_label_bounds"],
                "decision_count": 0,
                "modeling_ready": False,
                "thresholds": [],
            },
        )

    interval_start = _dt(
        min(_not_none(streams["trade"].first_event_time_ms), int(decisions[0]))
    )
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

    threshold_reports: list[dict[str, object]] = []
    best_unconditional: dict[str, object] | None = None
    best_with_target_touch: dict[str, object] | None = None
    best_touch_rate: dict[str, object] | None = None
    for threshold_bps in thresholds_bps:
        config = ResearchDatasetConfig(
            decision_stride_seconds=decision_stride_seconds,
            horizons_seconds=horizons_seconds,
            expected_cost_bps=threshold_bps,
            stress_cost_bps=threshold_bps,
            initial_stop_bps=initial_stop_bps,
            feature_windows_seconds=(1, 5, 30),
            require_book_features=True,
            max_book_staleness_ms=max_book_staleness_ms,
        )
        rows = build_research_rows(
            trades,
            decisions,
            config,
            book_events=book_events,
            mark_events=mark_events,
            spot_trade_events=spot_events,
        )
        summary = _summarize_threshold_rows(rows, threshold_bps, initial_stop_bps)
        threshold_reports.append(summary)
        for item in summary["by_side"]:
            candidate = {
                "threshold_bps": threshold_bps,
                **item,
            }
            if (
                best_touch_rate is None
                or candidate["target_before_stop_rate"]
                > best_touch_rate["target_before_stop_rate"]
            ):
                best_touch_rate = candidate
            if best_unconditional is None or candidate[
                "gross_path_payoff_mean_bps"
            ] > best_unconditional[
                "gross_path_payoff_mean_bps"
            ]:
                best_unconditional = candidate
            if candidate["target_before_stop_rate"] > 0 and (
                best_with_target_touch is None
                or candidate["gross_path_payoff_mean_bps"]
                > best_with_target_touch["gross_path_payoff_mean_bps"]
            ):
                best_with_target_touch = candidate

    decision_count = int(len(decisions))
    modeling_ready = decision_count >= minimum_decisions_for_modeling
    return _write_report(
        output_root,
        {
            **base_report,
            "status": "READY_FOR_SENSITIVITY_REVIEW"
            if modeling_ready
            else "CAPTURE_TOO_SHORT",
            "status_reasons": []
            if modeling_ready
            else [
                f"decision_count_below_minimum:{decision_count}<"
                f"{minimum_decisions_for_modeling}"
            ],
            "modeling_ready": modeling_ready,
            "decision_count": decision_count,
            "thresholds": threshold_reports,
            "best_touch_rate": best_touch_rate,
            "best_with_target_touch_gross_path": best_with_target_touch,
            "best_unconditional_gross_path": best_unconditional,
            "interval_start": _iso(int(decisions.min())),
            "interval_end": _iso(int(decisions.max())),
        },
    )


def _summarize_threshold_rows(
    rows: pd.DataFrame,
    threshold_bps: float,
    initial_stop_bps: float,
) -> dict[str, object]:
    enriched = rows.copy()
    enriched["gross_path_payoff_bps"] = np.where(
        enriched["paid_expected_before_stop"],
        threshold_bps,
        np.where(
            enriched["first_touch"] == "STOP",
            -initial_stop_bps,
            enriched["terminal_bps"],
        ),
    )
    by_side = []
    for side, group in enriched.groupby("side", sort=True):
        by_side.append(
            {
                "side": str(side),
                "rows": int(len(group)),
                "target_before_stop_rate": float(group["paid_expected_before_stop"].mean()),
                "stop_before_target_rate": float((group["first_touch"] == "STOP").mean()),
                "timeout_rate": float((group["first_touch"] == "TIMEOUT").mean()),
                "terminal_mean_bps": float(group["terminal_bps"].mean()),
                "mfe_mean_bps": float(group["mfe_bps"].mean()),
                "mae_mean_bps": float(group["mae_bps"].mean()),
                "gross_path_payoff_mean_bps": float(
                    group["gross_path_payoff_bps"].mean()
                ),
            }
        )
    return {
        "threshold_bps": threshold_bps,
        "row_count": int(len(enriched)),
        "decision_count": int(enriched["decision_time_ms"].nunique()),
        "by_side": by_side,
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


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("list must contain positive integers")
    return values


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("list must contain positive floats")
    return values


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
        "modeling_ready": report.get("modeling_ready", False),
        "best_touch_rate": report.get("best_touch_rate"),
        "best_with_target_touch_gross_path": report.get(
            "best_with_target_touch_gross_path"
        ),
        "best_unconditional_gross_path": report.get("best_unconditional_gross_path"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
