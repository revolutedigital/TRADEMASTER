"""Freeze a book-rule candidate on development WAL and test it on confirmation WAL."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
from scripts.research.run_book_rule_scout import (  # noqa: E402
    _add_gross_path_payoff,
)
from scripts.research.run_book_wal_scout import (  # noqa: E402
    StreamSummary,
    decision_times_for_wal,
    summarize_stream,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-wal-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "prospective-wal-fast16-18m",
    )
    parser.add_argument(
        "--confirmation-wal-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-rule-confirmation",
    )
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    parser.add_argument("--condition-fraction", type=float, default=0.20)
    parser.add_argument(
        "--conditions-json",
        default='[{"feature":"directed_spot_perp_basis_bps","direction":"high"}]',
    )
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--minimum-selected-count", type=int, default=20)
    arguments = parser.parse_args()
    report = run_book_rule_confirmation(
        development_wal_root=arguments.development_wal_root,
        confirmation_wal_root=arguments.confirmation_wal_root,
        output_root=arguments.output_root,
        horizon_seconds=arguments.horizon_seconds,
        threshold_bps=arguments.threshold_bps,
        condition_fraction=arguments.condition_fraction,
        conditions=_parse_conditions(arguments.conditions_json),
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        minimum_selected_count=arguments.minimum_selected_count,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_rule_confirmation(
    *,
    development_wal_root: Path,
    confirmation_wal_root: Path,
    output_root: Path,
    horizon_seconds: int = 120,
    threshold_bps: float = 5.0,
    condition_fraction: float = 0.20,
    conditions: tuple[dict[str, str], ...] = (
        {"feature": "directed_spot_perp_basis_bps", "direction": "high"},
    ),
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    initial_stop_bps: float = 20.0,
    minimum_selected_count: int = 20,
) -> dict[str, object]:
    _validate_parameters(
        horizon_seconds=horizon_seconds,
        threshold_bps=threshold_bps,
        condition_fraction=condition_fraction,
        conditions=conditions,
        initial_stop_bps=initial_stop_bps,
        minimum_selected_count=minimum_selected_count,
    )
    config = ResearchDatasetConfig(
        decision_stride_seconds=decision_stride_seconds,
        horizons_seconds=(horizon_seconds,),
        expected_cost_bps=threshold_bps,
        stress_cost_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
        feature_windows_seconds=(1, 5, 30),
        require_book_features=True,
        max_book_staleness_ms=max_book_staleness_ms,
    )
    development = _load_rows(
        wal_root=development_wal_root,
        config=config,
        threshold_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
    )
    base_report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "experiment_stage": "book_rule_confirmation",
        "approved_strategy": False,
        "approval_reasons": [
            "diagnostic_only",
            "single_development_window",
            "single_confirmation_window",
            "no_statistical_gate",
            "no_transaction_cost_confirmation",
        ],
        "development_wal_root": str(development_wal_root),
        "confirmation_wal_root": str(confirmation_wal_root),
        "config": asdict(config),
        "candidate": {
            "horizon_seconds": horizon_seconds,
            "threshold_bps": threshold_bps,
            "condition_fraction": condition_fraction,
            "conditions": list(conditions),
        },
        "development_streams": {
            name: asdict(summary) for name, summary in development["streams"].items()
        },
        "minimum_selected_count": minimum_selected_count,
    }
    if development["status"] != "READY":
        return _write_report(
            output_root,
            {
                **base_report,
                "status": f"DEVELOPMENT_{development['status']}",
                "status_reasons": development["status_reasons"],
                "modeling_ready": False,
            },
        )
    frozen_thresholds = fit_rule_thresholds(
        development["rows"],
        conditions=conditions,
        condition_fraction=condition_fraction,
    )
    confirmation = _load_rows(
        wal_root=confirmation_wal_root,
        config=config,
        threshold_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
    )
    report = {
        **base_report,
        "frozen_thresholds": frozen_thresholds,
        "confirmation_streams": {
            name: asdict(summary) for name, summary in confirmation["streams"].items()
        },
    }
    if confirmation["status"] != "READY":
        return _write_report(
            output_root,
            {
                **report,
                "status": f"CONFIRMATION_{confirmation['status']}",
                "status_reasons": confirmation["status_reasons"],
                "modeling_ready": False,
            },
        )
    selected = apply_frozen_rule(confirmation["rows"], frozen_thresholds)
    confirmation_metrics = summarize_selection(confirmation["rows"], selected)
    selected_count = int(confirmation_metrics["selected_count"])
    status = (
        "CONFIRMATION_READY_FOR_REPLICATION"
        if selected_count >= minimum_selected_count
        else "CONFIRMATION_INSUFFICIENT_SELECTIONS"
    )
    status_reasons = (
        []
        if selected_count >= minimum_selected_count
        else [f"selected_count_below_minimum:{selected_count}<{minimum_selected_count}"]
    )
    return _write_report(
        output_root,
        {
            **report,
            "status": status,
            "status_reasons": status_reasons,
            "modeling_ready": selected_count >= minimum_selected_count,
            "development_metrics": summarize_selection(
                development["rows"],
                apply_frozen_rule(development["rows"], frozen_thresholds),
            ),
            "confirmation_metrics": confirmation_metrics,
        },
    )


def fit_rule_thresholds(
    rows: pd.DataFrame,
    *,
    conditions: tuple[dict[str, str], ...],
    condition_fraction: float,
) -> list[dict[str, object]]:
    thresholds: list[dict[str, object]] = []
    for condition in conditions:
        feature = condition["feature"]
        direction = condition["direction"]
        quantile = 1 - condition_fraction if direction == "high" else condition_fraction
        thresholds.append(
            {
                "feature": feature,
                "direction": direction,
                "quantile": quantile,
                "threshold": float(rows[feature].quantile(quantile)),
            }
        )
    return thresholds


def apply_frozen_rule(
    rows: pd.DataFrame,
    frozen_thresholds: list[dict[str, object]],
) -> pd.DataFrame:
    mask = np.ones(len(rows), dtype=bool)
    for item in frozen_thresholds:
        feature = str(item["feature"])
        direction = str(item["direction"])
        threshold = float(item["threshold"])
        values = rows[feature].to_numpy(dtype=np.float64)
        if direction == "high":
            mask &= values >= threshold
        elif direction == "low":
            mask &= values <= threshold
        else:
            raise ValueError(f"unknown rule direction: {direction}")
    return rows.loc[mask]


def summarize_selection(rows: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    base_payoff = float(rows["gross_path_payoff_bps"].mean()) if len(rows) else 0.0
    selected_payoff = (
        float(selected["gross_path_payoff_bps"].mean()) if len(selected) else 0.0
    )
    base_target_rate = float(rows["target"].mean()) if len(rows) else 0.0
    selected_target_rate = float(selected["target"].mean()) if len(selected) else 0.0
    side_counts = (
        selected["side"].value_counts().sort_index().astype(int).to_dict()
        if len(selected)
        else {}
    )
    return {
        "row_count": int(len(rows)),
        "decision_count": int(rows["decision_time_ms"].nunique()) if len(rows) else 0,
        "selected_count": int(len(selected)),
        "side_counts": side_counts,
        "base_target_rate": base_target_rate,
        "selected_target_rate": selected_target_rate,
        "target_rate_lift_pp": float(selected_target_rate - base_target_rate),
        "base_gross_path_payoff_mean_bps": base_payoff,
        "selected_gross_path_payoff_mean_bps": selected_payoff,
        "gross_path_payoff_lift_bps": float(selected_payoff - base_payoff),
    }


def _load_rows(
    *,
    wal_root: Path,
    config: ResearchDatasetConfig,
    threshold_bps: float,
    initial_stop_bps: float,
) -> dict[str, Any]:
    streams = {
        "trade": summarize_stream(wal_root / "trade"),
        "depth": summarize_stream(wal_root / "depth"),
        "mark_price": summarize_stream(wal_root / "mark_price"),
        "spot_trade": summarize_stream(wal_root / "spot_trade"),
    }
    missing_required = [
        name for name in ("trade", "depth") if not streams[name].present
    ]
    if missing_required:
        return {
            "status": "MISSING_REQUIRED_STREAMS",
            "status_reasons": [f"{name}_missing" for name in missing_required],
            "streams": streams,
        }
    decisions = decision_times_for_wal(
        trade=streams["trade"],
        book=streams["depth"],
        config=config,
    )
    if len(decisions) == 0:
        return {
            "status": "INSUFFICIENT_INTERVAL",
            "status_reasons": ["no_complete_decisions_after_history_book_and_label_bounds"],
            "streams": streams,
        }
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
    rows = build_research_rows(
        trades,
        decisions,
        config,
        book_events=book_events,
        mark_events=mark_events,
        spot_trade_events=spot_events,
    )
    rows = _add_gross_path_payoff(rows, threshold_bps, initial_stop_bps)
    return {
        "status": "READY",
        "status_reasons": [],
        "streams": streams,
        "rows": rows,
    }


def _parse_conditions(value: str) -> tuple[dict[str, str], ...]:
    payload = json.loads(value)
    if not isinstance(payload, list) or not payload:
        raise ValueError("conditions must be a non-empty JSON list")
    conditions: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each condition must be an object")
        feature = item.get("feature")
        direction = item.get("direction")
        if not isinstance(feature, str) or not isinstance(direction, str):
            raise ValueError("condition feature and direction must be strings")
        conditions.append({"feature": feature, "direction": direction})
    return tuple(conditions)


def _validate_parameters(
    *,
    horizon_seconds: int,
    threshold_bps: float,
    condition_fraction: float,
    conditions: tuple[dict[str, str], ...],
    initial_stop_bps: float,
    minimum_selected_count: int,
) -> None:
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if threshold_bps <= 0:
        raise ValueError("threshold_bps must be positive")
    if not 0 < condition_fraction < 0.5:
        raise ValueError("condition_fraction must be between 0 and 0.5")
    if not conditions:
        raise ValueError("conditions cannot be empty")
    for condition in conditions:
        if condition["direction"] not in {"high", "low"}:
            raise ValueError("condition direction must be high or low")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if minimum_selected_count < 1:
        raise ValueError("minimum_selected_count must be positive")


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _not_none(value: int | None) -> int:
    if value is None:
        raise ValueError("required stream has no event time bounds")
    return value


def _dt(event_time_ms: int) -> datetime:
    return datetime.fromtimestamp(event_time_ms / 1000, tz=UTC)


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "approved_strategy": report["approved_strategy"],
        "frozen_thresholds": report.get("frozen_thresholds"),
        "confirmation_metrics": report.get("confirmation_metrics"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
