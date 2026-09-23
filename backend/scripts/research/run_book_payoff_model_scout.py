"""Multi-window executable-payoff model scout.

The previous probability scouts ranked entries by "touched target before stop".
Fast36 showed why that is not enough: a high target-rate can still lose after
spread, stop path, overlap, and explicit cost. This diagnostic trains models to
rank entries by executable trailing-stop payoff instead.

It is research-only, never submits orders, and never approves a strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.research_dataset import ResearchDatasetConfig  # noqa: E402
from scripts.research.run_book_executable_replay import (  # noqa: E402
    _book_arrays,
    summarize_executable_trades,
)
from scripts.research.run_book_feature_model_scout import LABEL_AND_FUTURE_COLUMNS  # noqa: E402
from scripts.research.run_book_frozen_model_holdout import _prepare_feature_frames  # noqa: E402
from scripts.research.run_book_multi_window_model_scout import (  # noqa: E402
    WalRootRef,
    _apply_one_position_constraint_by_score,
    _float_key,
    _load_window,
    _parse_float_list,
    _parse_wal_root_refs,
    _replay_selected_rows_with_trailing,
)


PAYOFF_REPLAY_AND_FUTURE_COLUMNS = frozenset(
    {
        "would_enter",
        "skip_reason",
        "entry_time_ms",
        "entry_book_time_ms",
        "entry_book_age_ms",
        "exit_time_ms",
        "exit_price",
        "exit_reason",
        "gross_executable_bps",
        "maximum_favorable_executable_bps",
        "maximum_adverse_executable_bps",
        "model_score",
        "model_score_replay",
    }
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wal-roots",
        required=True,
        help="Comma-separated LABEL=path list. Each label is one prospective window.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-payoff-model-scout",
    )
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--target-bps-values", default="3,5,8")
    parser.add_argument("--trailing-distance-bps-values", default="3,5,8")
    parser.add_argument("--top-fractions", default="0.02,0.05,0.10,0.20")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--minimum-windows", type=int, default=4)
    parser.add_argument("--minimum-executable-per-window", type=int, default=5)
    arguments = parser.parse_args()

    report = run_book_payoff_model_scout(
        wal_root_refs=_parse_wal_root_refs(arguments.wal_roots),
        output_root=arguments.output_root,
        horizon_seconds=arguments.horizon_seconds,
        target_bps_values=_parse_float_list(arguments.target_bps_values),
        trailing_distance_bps_values=_parse_float_list(
            arguments.trailing_distance_bps_values
        ),
        top_fractions=_parse_float_list(arguments.top_fractions),
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


def run_book_payoff_model_scout(
    *,
    wal_root_refs: tuple[WalRootRef, ...],
    output_root: Path,
    horizon_seconds: int = 120,
    target_bps_values: tuple[float, ...] = (3.0, 5.0, 8.0),
    trailing_distance_bps_values: tuple[float, ...] = (3.0, 5.0, 8.0),
    top_fractions: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20),
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    latency_ms: int = 100,
    initial_stop_bps: float = 20.0,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    profit_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    minimum_windows: int = 4,
    minimum_executable_per_window: int = 5,
) -> dict[str, object]:
    _validate_parameters(
        wal_root_refs=wal_root_refs,
        horizon_seconds=horizon_seconds,
        target_bps_values=target_bps_values,
        trailing_distance_bps_values=trailing_distance_bps_values,
        top_fractions=top_fractions,
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
            "payoff_model_scout_not_strategy_approval",
            "top_of_book_trailing_replay_not_exchange_execution",
            "no_queue_partial_fill_or_gap_slippage_model",
            "future_shadow_replay_required",
        ],
        "experiment_stage": "book_payoff_model_scout",
        "wal_roots": [
            {"label": ref.label, "path": str(ref.path)} for ref in wal_root_refs
        ],
        "config": {
            "horizon_seconds": horizon_seconds,
            "target_bps_values": list(target_bps_values),
            "trailing_distance_bps_values": list(trailing_distance_bps_values),
            "top_fractions": list(top_fractions),
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
        "label_definition": {
            "training_target": "gross_executable_bps_from_trailing_replay",
            "selection_score": "predicted_executable_payoff_bps",
            "why": "aligns model target with realized trade PnL instead of target-touch probability",
        },
        "leakage_guard": {
            "validation": "leave_one_window_out",
            "training_only_feature_selection": True,
            "excluded_columns": sorted(LABEL_AND_FUTURE_COLUMNS),
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
        loaded_windows = []
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
        for trailing_distance_bps in trailing_distance_bps_values:
            labelled_windows = _label_windows_with_executable_payoff(
                loaded_windows,
                target_bps=target_bps,
                trailing_distance_bps=trailing_distance_bps,
                horizon_seconds=horizon_seconds,
                initial_stop_bps=initial_stop_bps,
                latency_ms=latency_ms,
                max_book_staleness_ms=max_book_staleness_ms,
            )
            candidates, candidate_skips = _run_payoff_models(
                labelled_windows=labelled_windows,
                target_bps=target_bps,
                trailing_distance_bps=trailing_distance_bps,
                top_fractions=top_fractions,
                cost_levels_bps=cost_levels_bps,
                profit_cost_bps=profit_cost_bps,
                stress_cost_bps=stress_cost_bps,
                minimum_executable_per_window=minimum_executable_per_window,
            )
            all_candidates.extend(candidates)
            skipped.extend(candidate_skips)

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


def _label_windows_with_executable_payoff(
    loaded_windows: list[Any],
    *,
    target_bps: float,
    trailing_distance_bps: float,
    horizon_seconds: int,
    initial_stop_bps: float,
    latency_ms: int,
    max_book_staleness_ms: int,
) -> dict[str, pd.DataFrame]:
    books_by_window = {
        window.label: _book_arrays(window.book_events) for window in loaded_windows
    }
    labelled: dict[str, pd.DataFrame] = {}
    for window in loaded_windows:
        rows = window.rows.copy()
        rows["model_score"] = 0.0
        replay_rows = _replay_selected_rows_with_trailing(
            rows,
            books_by_window=books_by_window,
            horizon_seconds=horizon_seconds,
            breakeven_trigger_bps=target_bps,
            trailing_activation_bps=target_bps,
            trailing_distance_bps=trailing_distance_bps,
            initial_stop_bps=initial_stop_bps,
            latency_ms=latency_ms,
            max_book_staleness_ms=max_book_staleness_ms,
        )
        replay = pd.DataFrame(replay_rows)
        if replay.empty:
            labelled[window.label] = rows.iloc[0:0].copy()
            continue
        merged = rows.merge(
            replay,
            on=["window_label", "decision_time_ms", "side", "horizon_seconds"],
            how="left",
            suffixes=("", "_replay"),
            validate="one_to_one",
        )
        merged = merged[merged["would_enter"] == True].copy()  # noqa: E712
        labelled[window.label] = merged.reset_index(drop=True)
    return labelled


def _run_payoff_models(
    *,
    labelled_windows: dict[str, pd.DataFrame],
    target_bps: float,
    trailing_distance_bps: float,
    top_fractions: tuple[float, ...],
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_per_window: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    labels = sorted(labelled_windows)
    combined_rows = pd.concat(labelled_windows.values(), ignore_index=True)
    feature_columns = select_payoff_feature_columns(combined_rows)
    skipped: list[dict[str, object]] = []
    if not feature_columns:
        return [], [
            {
                "target_bps": target_bps,
                "trailing_distance_bps": trailing_distance_bps,
                "stage": "feature_selection",
                "reason": "no_training_causal_numeric_features",
            }
        ]
    candidate_holdouts: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    model_factories = {
        "hist_gradient_payoff": _make_hist_gradient_regressor,
        "ridge_payoff": _make_ridge_regressor,
    }
    for holdout_label in labels:
        train = combined_rows[combined_rows["window_label"] != holdout_label]
        test = combined_rows[combined_rows["window_label"] == holdout_label]
        if train.empty or test.empty:
            skipped.append(
                {
                    "target_bps": target_bps,
                    "trailing_distance_bps": trailing_distance_bps,
                    "holdout": holdout_label,
                    "stage": "fit",
                    "reason": "empty_train_or_test_rows",
                    "train_count": int(len(train)),
                    "test_count": int(len(test)),
                }
            )
            continue
        prepared_train, prepared_test, _medians = _prepare_feature_frames(
            train,
            test,
            feature_columns=feature_columns,
        )
        y_train = train["gross_executable_bps"].to_numpy(dtype=np.float64)
        if np.nanstd(y_train) == 0:
            skipped.append(
                {
                    "target_bps": target_bps,
                    "trailing_distance_bps": trailing_distance_bps,
                    "holdout": holdout_label,
                    "stage": "fit",
                    "reason": "training_payoff_has_no_variance",
                }
            )
            continue
        for model_name, model_factory in model_factories.items():
            model = model_factory()
            model.fit(prepared_train, y_train)
            scores = model.predict(prepared_test)
            for top_fraction in top_fractions:
                selected = _select_top_fraction(test, scores, top_fraction)
                independent_summary = summarize_executable_trades(
                    selected,
                    selected_signal_count=len(selected),
                    cost_levels_bps=cost_levels_bps,
                )
                one_position = _apply_one_position_constraint_by_score(selected)
                one_position_summary = summarize_executable_trades(
                    one_position,
                    selected_signal_count=len(selected),
                    cost_levels_bps=cost_levels_bps,
                )
                candidate_holdouts[(model_name, top_fraction)].append(
                    {
                        "label": holdout_label,
                        "test_count": int(len(test)),
                        "test_gross_mean_bps": float(test["gross_executable_bps"].mean()),
                        "selected_signal_count": int(len(selected)),
                        "selected_predicted_payoff_mean_bps": float(
                            selected["model_score"].mean()
                        ),
                        "selected_gross_mean_bps": float(
                            selected["gross_executable_bps"].mean()
                        )
                        if len(selected)
                        else 0.0,
                        "independent_summary": independent_summary,
                        "one_position_summary": one_position_summary,
                    }
                )
    candidates = []
    for (model_name, top_fraction), holdouts in candidate_holdouts.items():
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


def _select_top_fraction(
    rows: pd.DataFrame,
    scores: np.ndarray,
    top_fraction: float,
) -> pd.DataFrame:
    selected_count = max(1, int(np.ceil(len(rows) * top_fraction)))
    scored = rows.copy()
    scored["model_score"] = scores
    return scored.sort_values(
        ["model_score", "decision_time_ms"],
        ascending=[False, True],
        kind="stable",
    ).head(selected_count)


def select_payoff_feature_columns(rows: pd.DataFrame) -> list[str]:
    """Return numeric causal features, excluding replay labels/outcomes."""
    blocked = LABEL_AND_FUTURE_COLUMNS | PAYOFF_REPLAY_AND_FUTURE_COLUMNS
    columns: list[str] = []
    for column in rows.columns:
        if column in blocked or column == "window_label" or column.endswith("_replay"):
            continue
        if not pd.api.types.is_numeric_dtype(rows[column]):
            continue
        values = rows[column].to_numpy(dtype=np.float64)
        if np.isfinite(values).any():
            columns.append(column)
    return columns


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
        "trailing_distance_bps": float(trailing_distance_bps),
        "model": model_name,
        "top_fraction": float(top_fraction),
        "feature_count": int(feature_count),
        "holdout_count": len(holdouts),
        "selected_signal_count": int(
            sum(int(holdout["selected_signal_count"]) for holdout in holdouts)
        ),
        "mean_selected_gross_bps": float(
            np.mean([float(holdout["selected_gross_mean_bps"]) for holdout in holdouts])
        ),
        "holdouts": holdouts,
        "aggregate_by_mode": aggregate_by_mode,
        "one_position_replicated_at_profit_cost": (
            aggregate_by_mode["one_position"]["all_windows_positive_at_profit_cost"]
            and aggregate_by_mode["one_position"]["all_windows_meet_minimum_executions"]
        ),
        "one_position_replicated_at_stress_cost": (
            aggregate_by_mode["one_position"]["all_windows_positive_at_stress_cost"]
            and aggregate_by_mode["one_position"]["all_windows_meet_minimum_executions"]
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
    stress = [candidate for candidate in candidates if candidate["one_position_replicated_at_stress_cost"]]
    profit = [candidate for candidate in candidates if candidate["one_position_replicated_at_profit_cost"]]
    pool = stress or profit or candidates

    def score(candidate: dict[str, object]) -> tuple[bool, bool, float, float]:
        one_position = candidate["aggregate_by_mode"]["one_position"]
        independent = candidate["aggregate_by_mode"]["independent"]
        return (
            bool(candidate["one_position_replicated_at_stress_cost"]),
            bool(candidate["one_position_replicated_at_profit_cost"]),
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
        return "NO_VALID_PAYOFF_MODEL_FOLDS", ["no_candidate_with_valid_training_folds"]
    if best_candidate is None:
        return "NO_VALID_PAYOFF_MODEL_FOLDS", ["no_best_candidate"]
    if bool(best_candidate["one_position_replicated_at_stress_cost"]):
        return (
            "PAYOFF_MODEL_REPLICATED_RESEARCH_ONLY",
            [
                "not_approved_strategy",
                "future_shadow_replay_required",
                "testnet_release_still_requires_separate_authorization",
            ],
        )
    if bool(best_candidate["one_position_replicated_at_profit_cost"]):
        return (
            "PAYOFF_MODEL_PROFIT_COST_REPLICATED_STRESS_FRAGILE",
            [
                f"one_position_replicated_at_{profit_cost_bps:g}bps_cost",
                f"failed_or_fragile_at_{stress_cost_bps:g}bps_cost",
                "future_shadow_replay_required",
            ],
        )
    return (
        "PAYOFF_MODEL_FAILED_REPLICATION",
        [
            f"no_one_position_candidate_replicated_all_windows_at_{profit_cost_bps:g}bps_cost",
            "do_not_promote_to_strategy",
        ],
    )


def _make_hist_gradient_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_leaf_nodes=7,
        learning_rate=0.05,
        max_iter=100,
        min_samples_leaf=10,
        random_state=1,
    )


def _make_ridge_regressor() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def _window_report(window: Any) -> dict[str, object]:
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
    trailing_distance_bps_values: tuple[float, ...],
    top_fractions: tuple[float, ...],
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
