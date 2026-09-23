"""Leak-safe multi-window model scout with executable trailing-stop replay.

This diagnostic answers a narrower question than "can we trade now?": can
causal book/flow features rank entries that still make money after a realistic
top-of-book entry, latency, spread, explicit costs, and a simple trailing-stop
exit model across independent prospective windows?

It is intentionally research-only. It never submits orders and never approves a
strategy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.event_replay import OrderSide  # noqa: E402
from app.services.research.research_dataset import (  # noqa: E402
    ResearchDatasetConfig,
    build_research_rows,
    load_book_interval,
    load_mark_interval,
    load_trade_interval,
)
from scripts.research.run_book_executable_replay import (  # noqa: E402
    _book_arrays,
    summarize_executable_trades,
)
from scripts.research.run_book_feature_model_scout import (  # noqa: E402
    LABEL_AND_FUTURE_COLUMNS,
    select_causal_feature_columns,
)
from scripts.research.run_book_wal_scout import (  # noqa: E402
    decision_times_for_wal,
    summarize_stream,
)


@dataclass(frozen=True)
class WalRootRef:
    label: str
    path: Path


@dataclass(frozen=True)
class LoadedWindow:
    label: str
    rows: pd.DataFrame
    book_events: pd.DataFrame
    streams: dict[str, object]
    decision_count: int
    row_count: int
    interval_start: str
    interval_end: str
    book_fresh_filter: dict[str, object]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wal-roots",
        required=True,
        help="Comma-separated LABEL=path list. Each label is a prospective window.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-multi-window-model-scout",
    )
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--target-bps-values", default="5")
    parser.add_argument("--top-fractions", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--trailing-distance-bps-values", default="3,5,8")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--minimum-windows", type=int, default=4)
    parser.add_argument("--minimum-executable-per-window", type=int, default=3)
    arguments = parser.parse_args()

    report = run_book_multi_window_model_scout(
        wal_root_refs=_parse_wal_root_refs(arguments.wal_roots),
        output_root=arguments.output_root,
        horizon_seconds=arguments.horizon_seconds,
        target_bps_values=_parse_float_list(arguments.target_bps_values),
        top_fractions=_parse_float_list(arguments.top_fractions),
        trailing_distance_bps_values=_parse_float_list(
            arguments.trailing_distance_bps_values
        ),
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        latency_ms=arguments.latency_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
        profit_cost_bps=arguments.profit_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        minimum_windows=arguments.minimum_windows,
        minimum_executable_per_window=arguments.minimum_executable_per_window,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_multi_window_model_scout(
    *,
    wal_root_refs: tuple[WalRootRef, ...],
    output_root: Path,
    horizon_seconds: int = 120,
    target_bps_values: tuple[float, ...] = (5.0,),
    top_fractions: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20),
    trailing_distance_bps_values: tuple[float, ...] = (3.0, 5.0, 8.0),
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    latency_ms: int = 100,
    initial_stop_bps: float = 20.0,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    profit_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    minimum_windows: int = 4,
    minimum_executable_per_window: int = 3,
) -> dict[str, object]:
    _validate_parameters(
        wal_root_refs=wal_root_refs,
        horizon_seconds=horizon_seconds,
        target_bps_values=target_bps_values,
        top_fractions=top_fractions,
        trailing_distance_bps_values=trailing_distance_bps_values,
        decision_stride_seconds=decision_stride_seconds,
        max_book_staleness_ms=max_book_staleness_ms,
        latency_ms=latency_ms,
        initial_stop_bps=initial_stop_bps,
        cost_levels_bps=cost_levels_bps,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_windows=minimum_windows,
        minimum_executable_per_window=minimum_executable_per_window,
    )
    base_report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "approval_reasons": [
            "diagnostic_only",
            "multi_window_model_scout_not_strategy_approval",
            "top_of_book_trailing_replay_not_exchange_execution",
            "no_queue_partial_fill_or_gap_slippage_model",
            "future_shadow_replay_required",
        ],
        "experiment_stage": "book_multi_window_model_scout",
        "wal_roots": [
            {"label": ref.label, "path": str(ref.path)} for ref in wal_root_refs
        ],
        "config": {
            "horizon_seconds": horizon_seconds,
            "target_bps_values": list(target_bps_values),
            "top_fractions": list(top_fractions),
            "trailing_distance_bps_values": list(trailing_distance_bps_values),
            "breakeven_trigger_equals_target_bps": True,
            "trailing_activation_equals_target_bps": True,
            "decision_stride_seconds": decision_stride_seconds,
            "max_book_staleness_ms": max_book_staleness_ms,
            "latency_ms": latency_ms,
            "initial_stop_bps": initial_stop_bps,
            "cost_levels_bps": list(cost_levels_bps),
            "profit_cost_bps": profit_cost_bps,
            "stress_cost_bps": stress_cost_bps,
            "minimum_windows": minimum_windows,
            "minimum_executable_per_window": minimum_executable_per_window,
        },
        "leakage_guard": {
            "validation": "leave_one_window_out",
            "excluded_columns": sorted(LABEL_AND_FUTURE_COLUMNS),
            "selection_tie_breaker": "model_score_then_chronological",
        },
        "exit_model": {
            "name": "breakeven_then_trailing_stop",
            "description": (
                "initial stop until target_bps MFE; then stop moves to breakeven "
                "and trails best executable exit quote by trailing_distance_bps"
            ),
            "fixed_profit_target": False,
        },
    }
    if len(wal_root_refs) < minimum_windows:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "INSUFFICIENT_WINDOWS",
                "status_reasons": [
                    f"window_count_below_minimum:{len(wal_root_refs)}<{minimum_windows}"
                ],
                "modeling_ready": False,
                "candidates": [],
            },
        )

    all_candidates: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    window_reports_by_target: dict[str, list[dict[str, object]]] = {}
    for target_bps in target_bps_values:
        config = ResearchDatasetConfig(
            decision_stride_seconds=decision_stride_seconds,
            horizons_seconds=(horizon_seconds,),
            expected_cost_bps=target_bps,
            stress_cost_bps=target_bps,
            initial_stop_bps=initial_stop_bps,
            feature_windows_seconds=(1, 5, 30),
            require_book_features=True,
            max_book_staleness_ms=max_book_staleness_ms,
        )
        loaded_windows: list[LoadedWindow] = []
        for ref in wal_root_refs:
            loaded = _load_window(ref, config=config)
            if isinstance(loaded, dict):
                skipped.append(
                    {
                        "target_bps": target_bps,
                        "window": ref.label,
                        "stage": "load_window",
                        **loaded,
                    }
                )
                continue
            loaded_windows.append(loaded)
        window_reports_by_target[_float_key(target_bps)] = [
            _window_report(window) for window in loaded_windows
        ]
        if len(loaded_windows) < minimum_windows:
            skipped.append(
                {
                    "target_bps": target_bps,
                    "stage": "load_target",
                    "reason": "loaded_window_count_below_minimum",
                    "loaded_window_count": len(loaded_windows),
                    "minimum_windows": minimum_windows,
                }
            )
            continue
        threshold_candidates, threshold_skipped = _run_leave_one_window_models(
            loaded_windows=loaded_windows,
            target_bps=target_bps,
            top_fractions=top_fractions,
            trailing_distance_bps_values=trailing_distance_bps_values,
            horizon_seconds=horizon_seconds,
            initial_stop_bps=initial_stop_bps,
            latency_ms=latency_ms,
            max_book_staleness_ms=max_book_staleness_ms,
            cost_levels_bps=cost_levels_bps,
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
            minimum_executable_per_window=minimum_executable_per_window,
        )
        all_candidates.extend(threshold_candidates)
        skipped.extend(threshold_skipped)

    best_candidate = _select_best_candidate(
        all_candidates,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    status, status_reasons = _status(
        candidates=all_candidates,
        best_candidate=best_candidate,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    return _write_report(
        output_root,
        {
            **base_report,
            "status": status,
            "status_reasons": status_reasons,
            "modeling_ready": bool(all_candidates),
            "window_reports_by_target_bps": window_reports_by_target,
            "candidate_count": len(all_candidates),
            "candidates": sorted(
                all_candidates,
                key=lambda item: (
                    item["target_bps"],
                    item["trailing_distance_bps"],
                    item["model"],
                    item["top_fraction"],
                ),
            ),
            "best_candidate": best_candidate,
            "skipped": skipped,
        },
    )


def _run_leave_one_window_models(
    *,
    loaded_windows: list[LoadedWindow],
    target_bps: float,
    top_fractions: tuple[float, ...],
    trailing_distance_bps_values: tuple[float, ...],
    horizon_seconds: int,
    initial_stop_bps: float,
    latency_ms: int,
    max_book_staleness_ms: int,
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_per_window: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    combined_rows = pd.concat([window.rows for window in loaded_windows], ignore_index=True)
    feature_columns = select_causal_feature_columns(combined_rows)
    if not feature_columns:
        return [], [
            {
                "target_bps": target_bps,
                "stage": "feature_selection",
                "reason": "no_complete_causal_numeric_features",
            }
        ]
    books_by_window = {
        window.label: _book_arrays(window.book_events) for window in loaded_windows
    }
    candidate_holdouts: dict[tuple[float, str, float], list[dict[str, object]]] = defaultdict(list)
    skipped: list[dict[str, object]] = []
    model_factories = {
        "logistic_balanced": _make_logistic_model,
        "hist_gradient": _make_hist_gradient_model,
    }
    labels = [window.label for window in loaded_windows]
    for holdout_label in labels:
        train = combined_rows[combined_rows["window_label"] != holdout_label]
        test = combined_rows[combined_rows["window_label"] == holdout_label]
        if train["target"].nunique() < 2:
            skipped.append(
                {
                    "target_bps": target_bps,
                    "holdout": holdout_label,
                    "stage": "fit",
                    "reason": "train_target_not_binary",
                    "train_target_rate": float(train["target"].mean()) if len(train) else None,
                }
            )
            continue
        for model_name, model_factory in model_factories.items():
            model = model_factory()
            model.fit(train[feature_columns], train["target"].to_numpy())
            probabilities = model.predict_proba(test[feature_columns])[:, 1]
            for top_fraction in top_fractions:
                selected = _select_top_fraction(test, probabilities, top_fraction)
                for trailing_distance_bps in trailing_distance_bps_values:
                    replay_rows = _replay_selected_rows_with_trailing(
                        selected,
                        books_by_window=books_by_window,
                        horizon_seconds=horizon_seconds,
                        breakeven_trigger_bps=target_bps,
                        trailing_activation_bps=target_bps,
                        trailing_distance_bps=trailing_distance_bps,
                        initial_stop_bps=initial_stop_bps,
                        latency_ms=latency_ms,
                        max_book_staleness_ms=max_book_staleness_ms,
                    )
                    trades = pd.DataFrame([row for row in replay_rows if row["would_enter"]])
                    one_position = _apply_one_position_constraint_by_score(trades)
                    independent_summary = summarize_executable_trades(
                        trades,
                        selected_signal_count=len(selected),
                        cost_levels_bps=cost_levels_bps,
                    )
                    one_position_summary = summarize_executable_trades(
                        one_position,
                        selected_signal_count=len(selected),
                        cost_levels_bps=cost_levels_bps,
                    )
                    key = (trailing_distance_bps, model_name, top_fraction)
                    candidate_holdouts[key].append(
                        {
                            "label": holdout_label,
                            "test_count": int(len(test)),
                            "test_target_rate": float(test["target"].mean()),
                            "selected_signal_count": int(len(selected)),
                            "selected_target_rate": float(selected["target"].mean())
                            if len(selected)
                            else 0.0,
                            "independent_summary": independent_summary,
                            "one_position_summary": one_position_summary,
                        }
                    )
    candidates = []
    for (trailing_distance_bps, model_name, top_fraction), holdouts in candidate_holdouts.items():
        if len(holdouts) != len(labels):
            skipped.append(
                {
                    "target_bps": target_bps,
                    "trailing_distance_bps": trailing_distance_bps,
                    "model": model_name,
                    "top_fraction": top_fraction,
                    "stage": "candidate_finalize",
                    "reason": "missing_holdout_fold",
                    "holdout_count": len(holdouts),
                    "expected_holdout_count": len(labels),
                }
            )
            continue
        candidates.append(
            _finalize_candidate(
                target_bps=target_bps,
                trailing_distance_bps=trailing_distance_bps,
                model_name=model_name,
                top_fraction=top_fraction,
                feature_count=len(feature_columns),
                holdouts=holdouts,
                cost_levels_bps=cost_levels_bps,
                profit_cost_bps=profit_cost_bps,
                stress_cost_bps=stress_cost_bps,
                minimum_executable_per_window=minimum_executable_per_window,
            )
        )
    return candidates, skipped


def _load_window(
    ref: WalRootRef,
    *,
    config: ResearchDatasetConfig,
) -> LoadedWindow | dict[str, object]:
    streams = {
        "trade": summarize_stream(ref.path / "trade"),
        "depth": summarize_stream(ref.path / "depth"),
        "mark_price": summarize_stream(ref.path / "mark_price"),
        "spot_trade": summarize_stream(ref.path / "spot_trade"),
    }
    missing_required = [
        name for name in ("trade", "depth") if not streams[name].present
    ]
    if missing_required:
        return {
            "reason": "missing_required_streams",
            "status_reasons": [f"{name}_missing" for name in missing_required],
            "streams": {name: asdict(summary) for name, summary in streams.items()},
        }
    decisions = decision_times_for_wal(
        trade=streams["trade"],
        book=streams["depth"],
        config=config,
    )
    if len(decisions) == 0:
        return {
            "reason": "insufficient_interval",
            "status_reasons": ["no_complete_decisions_after_history_book_and_label_bounds"],
            "streams": {name: asdict(summary) for name, summary in streams.items()},
        }
    interval_start = _dt(min(_not_none(streams["trade"].first_event_time_ms), int(decisions[0])))
    interval_end = _dt(_not_none(streams["trade"].last_event_time_ms))
    trades = load_trade_interval(
        ref.path / "trade",
        interval_start,
        interval_end,
        expected_product="usdm_perpetual",
    )
    book_events = load_book_interval(ref.path / "depth", interval_start, interval_end)
    mark_events = (
        load_mark_interval(ref.path / "mark_price", interval_start, interval_end)
        if streams["mark_price"].present
        else None
    )
    spot_events = (
        load_trade_interval(
            ref.path / "spot_trade",
            interval_start,
            interval_end,
            expected_product="spot",
        )
        if streams["spot_trade"].present
        else None
    )
    rows, book_fresh_filter = _build_research_rows_with_book_fresh_filter(
        trades,
        decisions,
        config,
        book_events=book_events,
        mark_events=mark_events,
        spot_trade_events=spot_events,
    )
    rows = rows.copy()
    if rows.empty:
        return {
            "reason": "insufficient_fresh_book_decisions",
            "status_reasons": ["no_decisions_after_book_fresh_filter"],
            "streams": {name: asdict(summary) for name, summary in streams.items()},
            "book_fresh_filter": book_fresh_filter,
        }
    rows["window_label"] = ref.label
    return LoadedWindow(
        label=ref.label,
        rows=rows,
        book_events=book_events,
        streams={name: asdict(summary) for name, summary in streams.items()},
        decision_count=int(rows["decision_time_ms"].nunique()),
        row_count=int(len(rows)),
        interval_start=_iso(int(rows["decision_time_ms"].min())),
        interval_end=_iso(int(rows["decision_time_ms"].max())),
        book_fresh_filter=book_fresh_filter,
    )


def _build_research_rows_with_book_fresh_filter(
    trades: pd.DataFrame,
    decisions: np.ndarray,
    config: ResearchDatasetConfig,
    *,
    book_events: pd.DataFrame,
    mark_events: pd.DataFrame | None,
    spot_trade_events: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    original_decision_count = int(len(decisions))
    decisions = _filter_decisions_with_complete_trade_paths(trades, decisions, config)
    path_filtered_decision_count = int(len(decisions))
    try:
        rows = build_research_rows(
            trades,
            decisions,
            config,
            book_events=book_events,
            mark_events=mark_events,
            spot_trade_events=spot_trade_events,
        )
        return rows, {
            "applied": original_decision_count != path_filtered_decision_count,
            "reason": None,
            "path_filter_applied": original_decision_count != path_filtered_decision_count,
            "book_staleness_filter_applied": False,
            "original_decision_count": original_decision_count,
            "path_filtered_decision_count": path_filtered_decision_count,
            "retained_decision_count": int(rows["decision_time_ms"].nunique()),
            "dropped_decision_count": int(original_decision_count - path_filtered_decision_count),
            "max_book_staleness_ms": config.max_book_staleness_ms,
        }
    except ValueError as error:
        if not str(error).startswith("book features are incomplete:"):
            raise
        relaxed_config = replace(config, require_book_features=False)
        rows = build_research_rows(
            trades,
            decisions,
            relaxed_config,
            book_events=book_events,
            mark_events=mark_events,
            spot_trade_events=spot_trade_events,
        )
        fresh_rows = rows[
            (rows["book_available"] == 1)
            & (rows["book_update_age_ms"] <= config.max_book_staleness_ms)
        ].copy()
        retained_decision_count = int(fresh_rows["decision_time_ms"].nunique())
        return fresh_rows.reset_index(drop=True), {
            "applied": True,
            "reason": str(error),
            "path_filter_applied": original_decision_count != path_filtered_decision_count,
            "book_staleness_filter_applied": True,
            "original_decision_count": original_decision_count,
            "path_filtered_decision_count": path_filtered_decision_count,
            "retained_decision_count": retained_decision_count,
            "dropped_decision_count": int(original_decision_count - retained_decision_count),
            "max_book_staleness_ms": config.max_book_staleness_ms,
        }


def _filter_decisions_with_complete_trade_paths(
    trades: pd.DataFrame,
    decisions: np.ndarray,
    config: ResearchDatasetConfig,
) -> np.ndarray:
    if len(decisions) == 0:
        return decisions
    times = trades.sort_values(["event_time_ms", "sequence_id"], kind="stable")[
        "event_time_ms"
    ].to_numpy(dtype=np.int64)
    kept = []
    for decision in np.asarray(decisions, dtype=np.int64):
        entry_index = int(np.searchsorted(times, int(decision), side="right") - 1)
        if entry_index < 0:
            continue
        complete = True
        for horizon_seconds in config.horizons_seconds:
            horizon_end = int(decision) + int(horizon_seconds) * 1000
            if times[-1] < horizon_end:
                complete = False
                break
            path_right = int(np.searchsorted(times, horizon_end, side="right"))
            if entry_index + 1 >= path_right:
                complete = False
                break
        if complete:
            kept.append(int(decision))
    return np.asarray(kept, dtype=np.int64)


def _select_top_fraction(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    top_fraction: float,
) -> pd.DataFrame:
    selected_count = max(1, int(np.ceil(len(test) * top_fraction)))
    ordered = test.copy()
    ordered["model_score"] = probabilities
    return ordered.sort_values(
        ["model_score", "decision_time_ms"],
        ascending=[False, True],
        kind="stable",
    ).head(selected_count)


def _replay_selected_rows_with_trailing(
    selected_rows: pd.DataFrame,
    *,
    books_by_window: dict[str, dict[str, np.ndarray]],
    horizon_seconds: int,
    breakeven_trigger_bps: float,
    trailing_activation_bps: float,
    trailing_distance_bps: float,
    initial_stop_bps: float,
    latency_ms: int,
    max_book_staleness_ms: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    ordered = selected_rows.sort_values(
        ["decision_time_ms", "model_score"],
        ascending=[True, False],
        kind="stable",
    )
    for row in ordered.itertuples(index=False):
        window_label = str(row.window_label)
        result = simulate_executable_trailing_trade(
            book=books_by_window[window_label],
            decision_time_ms=int(row.decision_time_ms),
            side=OrderSide(str(row.side)),
            horizon_seconds=horizon_seconds,
            breakeven_trigger_bps=breakeven_trigger_bps,
            trailing_activation_bps=trailing_activation_bps,
            trailing_distance_bps=trailing_distance_bps,
            initial_stop_bps=initial_stop_bps,
            latency_ms=latency_ms,
            max_book_staleness_ms=max_book_staleness_ms,
        )
        result["window_label"] = window_label
        result["model_score"] = float(row.model_score)
        results.append(result)
    return results


def simulate_executable_trailing_trade(
    *,
    book: dict[str, np.ndarray],
    decision_time_ms: int,
    side: OrderSide,
    horizon_seconds: int,
    breakeven_trigger_bps: float,
    trailing_activation_bps: float,
    trailing_distance_bps: float,
    initial_stop_bps: float,
    latency_ms: int,
    max_book_staleness_ms: int,
) -> dict[str, object]:
    times = book["event_time_ms"]
    execution_time_ms = decision_time_ms + latency_ms
    entry_index = int(np.searchsorted(times, execution_time_ms, side="right") - 1)
    deadline_ms = decision_time_ms + horizon_seconds * 1000
    if entry_index < 0:
        return _skipped_trade(decision_time_ms, side, "no_entry_book")
    entry_book_age_ms = execution_time_ms - int(times[entry_index])
    if entry_book_age_ms > max_book_staleness_ms:
        return _skipped_trade(decision_time_ms, side, "stale_entry_book")
    exit_right = int(np.searchsorted(times, deadline_ms, side="right"))
    if entry_index + 1 >= exit_right:
        return _skipped_trade(decision_time_ms, side, "no_exit_path")

    if side == OrderSide.BUY:
        entry_price = float(book["ask_price"][entry_index])
        path_prices = book["bid_price"][entry_index + 1 : exit_right]
        initial_stop_price = entry_price * math.exp(-initial_stop_bps / 10_000)
    else:
        entry_price = float(book["bid_price"][entry_index])
        path_prices = book["ask_price"][entry_index + 1 : exit_right]
        initial_stop_price = entry_price * math.exp(initial_stop_bps / 10_000)
    path_times = times[entry_index + 1 : exit_right]

    stop_price = float(initial_stop_price)
    stop_reason = "INITIAL_STOP"
    exit_index = exit_right - 1
    exit_price = float(path_prices[-1])
    exit_reason = "TIMEOUT"
    for offset, price in enumerate(path_prices):
        price = float(price)
        if _stop_was_hit(side, price, stop_price):
            exit_index = entry_index + 1 + offset
            exit_price = stop_price
            exit_reason = stop_reason
            break
        favorable_bps = _gross_bps(side, entry_price, price)
        if favorable_bps >= breakeven_trigger_bps:
            if side == OrderSide.BUY and stop_price < entry_price:
                stop_price = entry_price
                stop_reason = "BREAKEVEN_STOP"
            elif side == OrderSide.SELL and stop_price > entry_price:
                stop_price = entry_price
                stop_reason = "BREAKEVEN_STOP"
        if favorable_bps >= trailing_activation_bps:
            if side == OrderSide.BUY:
                trailing_stop_price = price * math.exp(-trailing_distance_bps / 10_000)
                if trailing_stop_price > stop_price:
                    stop_price = trailing_stop_price
                    stop_reason = "TRAILING_STOP"
            else:
                trailing_stop_price = price * math.exp(trailing_distance_bps / 10_000)
                if trailing_stop_price < stop_price:
                    stop_price = trailing_stop_price
                    stop_reason = "TRAILING_STOP"

    exit_book_age_ms = deadline_ms - int(path_times[-1]) if exit_reason == "TIMEOUT" else 0
    if exit_reason == "TIMEOUT" and exit_book_age_ms > max_book_staleness_ms:
        return _skipped_trade(decision_time_ms, side, "stale_timeout_book")
    path_gross = np.array([_gross_bps(side, entry_price, price) for price in path_prices])
    return {
        "would_enter": True,
        "skip_reason": None,
        "decision_time_ms": decision_time_ms,
        "entry_time_ms": execution_time_ms,
        "entry_book_time_ms": int(times[entry_index]),
        "entry_book_age_ms": int(entry_book_age_ms),
        "exit_time_ms": int(times[exit_index]),
        "side": side.value,
        "horizon_seconds": horizon_seconds,
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_executable_bps": _gross_bps(side, entry_price, float(exit_price)),
        "maximum_favorable_executable_bps": float(path_gross.max()),
        "maximum_adverse_executable_bps": float(path_gross.min()),
    }


def _apply_one_position_constraint_by_score(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    ordered = trades.sort_values(
        ["decision_time_ms", "model_score"],
        ascending=[True, False],
        kind="stable",
    )
    kept = []
    position_exit_ms = -1
    for row in ordered.itertuples(index=False):
        decision_time_ms = int(row.decision_time_ms)
        if decision_time_ms <= position_exit_ms:
            continue
        kept.append(row._asdict())
        position_exit_ms = int(row.exit_time_ms)
    return pd.DataFrame(kept)


def _finalize_candidate(
    *,
    target_bps: float,
    trailing_distance_bps: float,
    model_name: str,
    top_fraction: float,
    feature_count: int,
    holdouts: list[dict[str, object]],
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_per_window: int,
) -> dict[str, object]:
    aggregate_by_mode = {
        mode: _aggregate_holdouts(
            holdouts,
            mode=mode,
            cost_levels_bps=cost_levels_bps,
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
            minimum_executable_per_window=minimum_executable_per_window,
        )
        for mode in ("independent", "one_position")
    }
    return {
        "target_bps": float(target_bps),
        "breakeven_trigger_bps": float(target_bps),
        "trailing_activation_bps": float(target_bps),
        "trailing_distance_bps": float(trailing_distance_bps),
        "model": model_name,
        "top_fraction": float(top_fraction),
        "feature_count": int(feature_count),
        "holdout_count": len(holdouts),
        "selected_signal_count": int(
            sum(int(holdout["selected_signal_count"]) for holdout in holdouts)
        ),
        "mean_selected_target_rate": float(
            np.mean([float(holdout["selected_target_rate"]) for holdout in holdouts])
        ),
        "holdouts": holdouts,
        "aggregate_by_mode": aggregate_by_mode,
        "replicated_at_profit_cost": all(
            aggregate_by_mode[mode]["all_windows_positive_at_profit_cost"]
            and aggregate_by_mode[mode]["all_windows_meet_minimum_executions"]
            for mode in ("independent", "one_position")
        ),
        "replicated_at_stress_cost": all(
            aggregate_by_mode[mode]["all_windows_positive_at_stress_cost"]
            and aggregate_by_mode[mode]["all_windows_meet_minimum_executions"]
            for mode in ("independent", "one_position")
        ),
    }


def _aggregate_holdouts(
    holdouts: list[dict[str, object]],
    *,
    mode: str,
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_per_window: int,
) -> dict[str, object]:
    summary_key = f"{mode}_summary"
    executable_count = sum(
        int(holdout[summary_key]["executable_count"]) for holdout in holdouts
    )
    gross_total = sum(
        float(holdout[summary_key]["gross_executable_total_bps"]) for holdout in holdouts
    )
    selected_signal_count = sum(
        int(holdout[summary_key]["selected_signal_count"]) for holdout in holdouts
    )
    net_mean_by_cost: dict[str, float] = {}
    net_total_by_cost: dict[str, float] = {}
    positive_window_count_by_cost: dict[str, int] = {}
    min_window_net_by_cost: dict[str, float] = {}
    for cost_bps in cost_levels_bps:
        key = _float_key(cost_bps)
        window_net_values = []
        for holdout in holdouts:
            row = _cost_row(holdout[summary_key]["cost_sensitivity"], cost_bps)
            window_net_values.append(float(row["net_executable_mean_bps"]))
        net_total = gross_total - executable_count * cost_bps
        net_total_by_cost[key] = float(net_total)
        net_mean_by_cost[key] = float(net_total / executable_count) if executable_count else 0.0
        positive_window_count_by_cost[key] = sum(value > 0 for value in window_net_values)
        min_window_net_by_cost[key] = float(min(window_net_values)) if window_net_values else 0.0
    exit_reasons: dict[str, int] = {}
    minimum_execution_window_count = 0
    for holdout in holdouts:
        summary = holdout[summary_key]
        if int(summary["executable_count"]) >= minimum_executable_per_window:
            minimum_execution_window_count += 1
        for reason, count in summary["exit_reasons"].items():
            exit_reasons[str(reason)] = exit_reasons.get(str(reason), 0) + int(count)
    return {
        "selected_signal_count": int(selected_signal_count),
        "executable_count": int(executable_count),
        "gross_mean_bps": float(gross_total / executable_count) if executable_count else 0.0,
        "gross_total_bps": float(gross_total),
        "net_mean_by_cost_bps": net_mean_by_cost,
        "net_total_by_cost_bps": net_total_by_cost,
        "positive_window_count_by_cost_bps": positive_window_count_by_cost,
        "min_window_net_by_cost_bps": min_window_net_by_cost,
        "exit_reasons": exit_reasons,
        "minimum_execution_window_count": minimum_execution_window_count,
        "all_windows_meet_minimum_executions": minimum_execution_window_count
        == len(holdouts),
        "all_windows_positive_at_profit_cost": positive_window_count_by_cost[
            _float_key(profit_cost_bps)
        ]
        == len(holdouts),
        "all_windows_positive_at_stress_cost": positive_window_count_by_cost[
            _float_key(stress_cost_bps)
        ]
        == len(holdouts),
    }


def _select_best_candidate(
    candidates: list[dict[str, object]],
    *,
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> dict[str, object] | None:
    if not candidates:
        return None
    stress_replicated = [
        candidate for candidate in candidates if candidate["replicated_at_stress_cost"]
    ]
    profit_replicated = [
        candidate for candidate in candidates if candidate["replicated_at_profit_cost"]
    ]
    pool = stress_replicated or profit_replicated or candidates

    def score(candidate: dict[str, object]) -> tuple[bool, bool, float, float]:
        one_position = candidate["aggregate_by_mode"]["one_position"]
        independent = candidate["aggregate_by_mode"]["independent"]
        return (
            bool(candidate["replicated_at_stress_cost"]),
            bool(candidate["replicated_at_profit_cost"]),
            float(one_position["net_mean_by_cost_bps"][_float_key(profit_cost_bps)]),
            float(independent["net_mean_by_cost_bps"][_float_key(profit_cost_bps)]),
        )

    return max(pool, key=score)


def _status(
    *,
    candidates: list[dict[str, object]],
    best_candidate: dict[str, object] | None,
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> tuple[str, list[str]]:
    if not candidates:
        return "NO_VALID_MODEL_FOLDS", ["no_candidate_with_valid_training_folds"]
    if best_candidate is None:
        return "NO_VALID_MODEL_FOLDS", ["no_best_candidate"]
    if bool(best_candidate["replicated_at_stress_cost"]):
        return (
            "MULTI_WINDOW_MODEL_REPLICATED_RESEARCH_ONLY",
            [
                "not_approved_strategy",
                "future_shadow_replay_required",
                "testnet_release_still_requires_separate_authorization",
            ],
        )
    if bool(best_candidate["replicated_at_profit_cost"]):
        return (
            "MULTI_WINDOW_MODEL_PROFIT_COST_REPLICATED_STRESS_FRAGILE",
            [
                f"replicated_at_{profit_cost_bps:g}bps_cost",
                f"failed_or_fragile_at_{stress_cost_bps:g}bps_cost",
                "future_shadow_replay_required",
            ],
        )
    return (
        "MULTI_WINDOW_MODEL_FAILED_REPLICATION",
        [
            f"no_candidate_replicated_all_windows_at_{profit_cost_bps:g}bps_cost",
            "do_not_promote_to_strategy",
        ],
    )


def _window_report(window: LoadedWindow) -> dict[str, object]:
    return {
        "label": window.label,
        "decision_count": window.decision_count,
        "row_count": window.row_count,
        "interval_start": window.interval_start,
        "interval_end": window.interval_end,
        "target_rate": float(window.rows["target"].mean()),
        "book_fresh_filter": window.book_fresh_filter,
        "streams": window.streams,
    }


def _make_logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, class_weight="balanced"),
            ),
        ]
    )


def _make_hist_gradient_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_leaf_nodes=7,
        learning_rate=0.05,
        max_iter=100,
        min_samples_leaf=10,
        random_state=1,
    )


def _stop_was_hit(side: OrderSide, price: float, stop_price: float) -> bool:
    if side == OrderSide.BUY:
        return price <= stop_price
    return price >= stop_price


def _gross_bps(side: OrderSide, entry_price: float, exit_price: float) -> float:
    if side == OrderSide.BUY:
        return math.log(exit_price / entry_price) * 10_000
    return math.log(entry_price / exit_price) * 10_000


def _skipped_trade(
    decision_time_ms: int,
    side: OrderSide,
    reason: str,
) -> dict[str, object]:
    return {
        "would_enter": False,
        "skip_reason": reason,
        "decision_time_ms": decision_time_ms,
        "side": side.value,
    }


def _cost_row(rows: Iterable[dict[str, Any]], cost_bps: float) -> dict[str, Any]:
    for row in rows:
        if float(row["explicit_cost_bps"]) == cost_bps:
            return row
    raise ValueError(f"cost level missing from summary: {cost_bps:g} bps")


def _validate_parameters(
    *,
    wal_root_refs: tuple[WalRootRef, ...],
    horizon_seconds: int,
    target_bps_values: tuple[float, ...],
    top_fractions: tuple[float, ...],
    trailing_distance_bps_values: tuple[float, ...],
    decision_stride_seconds: int,
    max_book_staleness_ms: int,
    latency_ms: int,
    initial_stop_bps: float,
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_windows: int,
    minimum_executable_per_window: int,
) -> None:
    if len({ref.label for ref in wal_root_refs}) != len(wal_root_refs):
        raise ValueError("wal root labels must be unique")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if decision_stride_seconds <= 0:
        raise ValueError("decision_stride_seconds must be positive")
    if max_book_staleness_ms < 0:
        raise ValueError("max_book_staleness_ms cannot be negative")
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if minimum_windows < 2:
        raise ValueError("minimum_windows must be at least 2")
    if minimum_executable_per_window < 1:
        raise ValueError("minimum_executable_per_window must be positive")
    if not target_bps_values or any(value <= 0 for value in target_bps_values):
        raise ValueError("target_bps_values must contain positive values")
    if not trailing_distance_bps_values or any(
        value <= 0 for value in trailing_distance_bps_values
    ):
        raise ValueError("trailing_distance_bps_values must contain positive values")
    if not top_fractions or any(not 0 < value < 1 for value in top_fractions):
        raise ValueError("top_fractions must contain values between 0 and 1")
    if not cost_levels_bps or any(value < 0 for value in cost_levels_bps):
        raise ValueError("cost_levels_bps must contain non-negative values")
    if profit_cost_bps not in cost_levels_bps:
        raise ValueError("profit_cost_bps must be present in cost_levels_bps")
    if stress_cost_bps not in cost_levels_bps:
        raise ValueError("stress_cost_bps must be present in cost_levels_bps")


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_wal_root_refs(value: str) -> tuple[WalRootRef, ...]:
    refs = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("wal roots must use LABEL=path")
        label, raw_path = part.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("wal root label cannot be empty")
        path = Path(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"wal root not found: {path}")
        refs.append(WalRootRef(label=label, path=path))
    if not refs:
        raise ValueError("at least one wal root is required")
    return tuple(refs)


def _parse_float_list(value: str, *, allow_zero: bool = False) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("list must contain values")
    if allow_zero:
        if any(item < 0 for item in values):
            raise ValueError("list values cannot be negative")
    elif any(item <= 0 for item in values):
        raise ValueError("list values must be positive")
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


def _float_key(value: float) -> str:
    return f"{value:g}"


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "status_reasons": report.get("status_reasons", []),
        "approved_strategy": report["approved_strategy"],
        "candidate_count": report.get("candidate_count", 0),
        "best_candidate": report.get("best_candidate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
