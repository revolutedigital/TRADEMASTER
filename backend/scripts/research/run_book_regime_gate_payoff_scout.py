"""Nested regime/no-trade gate scout for executable payoff models.

Fast37 showed that a payoff model can look positive in aggregate while failing
specific regimes. This diagnostic adds an abstention gate before trade
selection. For each outer holdout window, the gate is selected only from
out-of-fold predictions on the remaining windows, then frozen and evaluated on
the holdout.

It is research-only, never submits orders, and never approves a strategy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.research_dataset import ResearchDatasetConfig  # noqa: E402
from scripts.research.run_book_executable_replay import (  # noqa: E402
    _book_arrays,
    summarize_executable_trades,
)
from scripts.research.run_book_frozen_model_holdout import _prepare_feature_frames  # noqa: E402
from scripts.research.run_book_multi_window_model_scout import (  # noqa: E402
    WalRootRef,
    _apply_one_position_constraint_by_score,
    _float_key,
    _load_window,
    _parse_wal_root_refs,
    _replay_selected_rows_with_trailing,
)
from scripts.research.run_book_payoff_model_scout import (  # noqa: E402
    _make_hist_gradient_regressor,
    _make_ridge_regressor,
    select_payoff_feature_columns,
)


@dataclass(frozen=True)
class RegimeGateCandidate:
    feature: str | None
    direction: str
    threshold: float | None
    keep_fraction: float
    selected_training_count: int
    one_position_training_net_at_profit_cost_bps: float
    one_position_training_min_window_net_at_profit_cost_bps: float
    one_position_training_positive_windows_at_profit_cost: int
    one_position_training_minimum_execution_window_count: int


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
        / "book-regime-gate-payoff-scout",
    )
    parser.add_argument(
        "--model",
        choices=("hist_gradient_payoff", "ridge_payoff"),
        default="hist_gradient_payoff",
    )
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--target-bps", type=float, default=8.0)
    parser.add_argument("--trailing-distance-bps", type=float, default=8.0)
    parser.add_argument("--top-fraction", type=float, default=0.02)
    parser.add_argument("--gate-keep-fractions", default="0.50,0.70,0.85,1.00")
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

    report = run_book_regime_gate_payoff_scout(
        wal_root_refs=_parse_wal_root_refs(arguments.wal_roots),
        output_root=arguments.output_root,
        model_name=arguments.model,
        horizon_seconds=arguments.horizon_seconds,
        target_bps=arguments.target_bps,
        trailing_distance_bps=arguments.trailing_distance_bps,
        top_fraction=arguments.top_fraction,
        gate_keep_fractions=_parse_float_list(arguments.gate_keep_fractions),
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


def run_book_regime_gate_payoff_scout(
    *,
    wal_root_refs: tuple[WalRootRef, ...],
    output_root: Path,
    model_name: str = "hist_gradient_payoff",
    horizon_seconds: int = 120,
    target_bps: float = 8.0,
    trailing_distance_bps: float = 8.0,
    top_fraction: float = 0.02,
    gate_keep_fractions: tuple[float, ...] = (0.50, 0.70, 0.85, 1.00),
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
        model_name=model_name,
        horizon_seconds=horizon_seconds,
        target_bps=target_bps,
        trailing_distance_bps=trailing_distance_bps,
        top_fraction=top_fraction,
        gate_keep_fractions=gate_keep_fractions,
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
            "nested_regime_gate_scout_not_strategy_approval",
            "top_of_book_trailing_replay_not_exchange_execution",
            "no_queue_partial_fill_or_gap_slippage_model",
            "future_shadow_replay_required",
        ],
        "experiment_stage": "book_regime_gate_payoff_scout",
        "wal_roots": [
            {"label": ref.label, "path": str(ref.path)} for ref in wal_root_refs
        ],
        "config": {
            "model": model_name,
            "horizon_seconds": horizon_seconds,
            "target_bps": target_bps,
            "trailing_distance_bps": trailing_distance_bps,
            "top_fraction": top_fraction,
            "gate_keep_fractions": list(gate_keep_fractions),
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
            "outer_validation": "leave_one_window_out",
            "gate_selection": "training_windows_only_with_inner_oof_predictions",
            "holdout_never_used_for_gate_or_model_fit": True,
            "gate_type": "single_causal_feature_threshold_or_no_gate",
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
                "holdouts": [],
            },
        )

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
    skipped = []
    for ref in wal_root_refs:
        loaded = _load_window(ref, config=config)
        if isinstance(loaded, dict):
            skipped.append({"window": ref.label, **loaded})
            continue
        loaded_windows.append(loaded)
    if len(loaded_windows) < minimum_windows:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "LOADED_WINDOW_COUNT_BELOW_MINIMUM",
                "status_reasons": [
                    f"loaded_window_count_below_minimum:{len(loaded_windows)}<{minimum_windows}"
                ],
                "modeling_ready": False,
                "holdouts": [],
                "skipped": skipped,
            },
        )

    labelled_windows = _label_windows_with_executable_payoff(
        loaded_windows,
        target_bps=target_bps,
        trailing_distance_bps=trailing_distance_bps,
        horizon_seconds=horizon_seconds,
        initial_stop_bps=initial_stop_bps,
        latency_ms=latency_ms,
        max_book_staleness_ms=max_book_staleness_ms,
    )
    holdout_results = []
    for holdout_label in sorted(labelled_windows):
        result = _run_outer_holdout(
            labelled_windows=labelled_windows,
            holdout_label=holdout_label,
            model_name=model_name,
            top_fraction=top_fraction,
            gate_keep_fractions=gate_keep_fractions,
            cost_levels_bps=cost_levels_bps,
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
            minimum_executable_per_window=minimum_executable_per_window,
        )
        holdout_results.append(result)

    aggregate_by_mode = {
        mode: _aggregate_holdouts(
            holdout_results,
            mode=mode,
            cost_levels_bps=cost_levels_bps,
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
            minimum_executable_per_window=minimum_executable_per_window,
        )
        for mode in ("independent", "one_position")
    }
    status, status_reasons = _status(
        holdout_results=holdout_results,
        aggregate_by_mode=aggregate_by_mode,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    return _write_report(
        output_root,
        {
            **base_report,
            "status": status,
            "status_reasons": status_reasons,
            "modeling_ready": True,
            "window_reports": [_window_report(window) for window in loaded_windows],
            "holdouts": holdout_results,
            "aggregate_by_mode": aggregate_by_mode,
            "skipped": skipped,
        },
    )


def _run_outer_holdout(
    *,
    labelled_windows: dict[str, pd.DataFrame],
    holdout_label: str,
    model_name: str,
    top_fraction: float,
    gate_keep_fractions: tuple[float, ...],
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_per_window: int,
) -> dict[str, object]:
    training_labels = [label for label in sorted(labelled_windows) if label != holdout_label]
    training_frame = pd.concat(
        [labelled_windows[label] for label in training_labels],
        ignore_index=True,
    )
    holdout_frame = labelled_windows[holdout_label]
    feature_columns = select_payoff_feature_columns(training_frame)
    gate_feature_columns = _gate_feature_columns(feature_columns)
    inner_scored_training = _inner_oof_predictions(
        labelled_windows=labelled_windows,
        training_labels=training_labels,
        model_name=model_name,
        feature_columns=feature_columns,
    )
    gate = _select_regime_gate(
        scored_training=inner_scored_training,
        gate_feature_columns=gate_feature_columns,
        top_fraction=top_fraction,
        gate_keep_fractions=gate_keep_fractions,
        cost_levels_bps=cost_levels_bps,
        profit_cost_bps=profit_cost_bps,
        minimum_executable_per_window=minimum_executable_per_window,
    )
    final_model = _make_model(model_name)
    prepared_training, prepared_holdout, _medians = _prepare_feature_frames(
        training_frame,
        holdout_frame,
        feature_columns=feature_columns,
    )
    final_model.fit(
        prepared_training,
        training_frame["gross_executable_bps"].to_numpy(dtype=np.float64),
    )
    scored_holdout = holdout_frame.copy()
    scored_holdout["model_score"] = final_model.predict(prepared_holdout)
    selected_holdout = _select_gated_top_fraction(
        scored_holdout,
        gate=gate,
        top_fraction=top_fraction,
    )
    independent_summary = summarize_executable_trades(
        selected_holdout,
        selected_signal_count=len(selected_holdout),
        cost_levels_bps=cost_levels_bps,
    )
    one_position = _apply_one_position_constraint_by_score(selected_holdout)
    one_position_summary = summarize_executable_trades(
        one_position,
        selected_signal_count=len(selected_holdout),
        cost_levels_bps=cost_levels_bps,
    )
    return {
        "label": holdout_label,
        "training_labels": training_labels,
        "feature_count": len(feature_columns),
        "gate_feature_count": len(gate_feature_columns),
        "selected_gate": asdict(gate),
        "test_count": int(len(holdout_frame)),
        "test_gross_mean_bps": float(holdout_frame["gross_executable_bps"].mean())
        if len(holdout_frame)
        else 0.0,
        "selected_signal_count": int(len(selected_holdout)),
        "selected_gross_mean_bps": float(selected_holdout["gross_executable_bps"].mean())
        if len(selected_holdout)
        else 0.0,
        "selected_predicted_payoff_mean_bps": float(selected_holdout["model_score"].mean())
        if len(selected_holdout)
        else 0.0,
        "independent_summary": independent_summary,
        "one_position_summary": one_position_summary,
        "passed_one_position_profit_cost": _positive_at_cost(
            one_position_summary,
            profit_cost_bps,
        )
        and int(one_position_summary["executable_count"]) >= minimum_executable_per_window,
        "passed_one_position_stress_cost": _positive_at_cost(
            one_position_summary,
            stress_cost_bps,
        )
        and int(one_position_summary["executable_count"]) >= minimum_executable_per_window,
    }


def _inner_oof_predictions(
    *,
    labelled_windows: dict[str, pd.DataFrame],
    training_labels: list[str],
    model_name: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    frames = []
    for inner_holdout in training_labels:
        inner_train_labels = [label for label in training_labels if label != inner_holdout]
        inner_train = pd.concat(
            [labelled_windows[label] for label in inner_train_labels],
            ignore_index=True,
        )
        inner_test = labelled_windows[inner_holdout].copy()
        prepared_train, prepared_test, _medians = _prepare_feature_frames(
            inner_train,
            inner_test,
            feature_columns=feature_columns,
        )
        model = _make_model(model_name)
        model.fit(
            prepared_train,
            inner_train["gross_executable_bps"].to_numpy(dtype=np.float64),
        )
        inner_test["model_score"] = model.predict(prepared_test)
        frames.append(inner_test)
    return pd.concat(frames, ignore_index=True)


def _select_regime_gate(
    *,
    scored_training: pd.DataFrame,
    gate_feature_columns: list[str],
    top_fraction: float,
    gate_keep_fractions: tuple[float, ...],
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    minimum_executable_per_window: int,
) -> RegimeGateCandidate:
    candidates = [
        _evaluate_gate(
            scored_training,
            gate=RegimeGateCandidate(
                feature=None,
                direction="none",
                threshold=None,
                keep_fraction=1.0,
                selected_training_count=0,
                one_position_training_net_at_profit_cost_bps=0.0,
                one_position_training_min_window_net_at_profit_cost_bps=-math.inf,
                one_position_training_positive_windows_at_profit_cost=0,
                one_position_training_minimum_execution_window_count=0,
            ),
            top_fraction=top_fraction,
            cost_levels_bps=cost_levels_bps,
            profit_cost_bps=profit_cost_bps,
            minimum_executable_per_window=minimum_executable_per_window,
        )
    ]
    for feature in gate_feature_columns:
        values = scored_training[feature].replace([np.inf, -np.inf], np.nan).dropna()
        if values.nunique() < 3:
            continue
        for keep_fraction in gate_keep_fractions:
            if keep_fraction >= 1.0:
                continue
            low_threshold = float(values.quantile(keep_fraction))
            high_threshold = float(values.quantile(1.0 - keep_fraction))
            candidates.append(
                _evaluate_gate(
                    scored_training,
                    gate=RegimeGateCandidate(
                        feature=feature,
                        direction="low",
                        threshold=low_threshold,
                        keep_fraction=keep_fraction,
                        selected_training_count=0,
                        one_position_training_net_at_profit_cost_bps=0.0,
                        one_position_training_min_window_net_at_profit_cost_bps=-math.inf,
                        one_position_training_positive_windows_at_profit_cost=0,
                        one_position_training_minimum_execution_window_count=0,
                    ),
                    top_fraction=top_fraction,
                    cost_levels_bps=cost_levels_bps,
                    profit_cost_bps=profit_cost_bps,
                    minimum_executable_per_window=minimum_executable_per_window,
                )
            )
            candidates.append(
                _evaluate_gate(
                    scored_training,
                    gate=RegimeGateCandidate(
                        feature=feature,
                        direction="high",
                        threshold=high_threshold,
                        keep_fraction=keep_fraction,
                        selected_training_count=0,
                        one_position_training_net_at_profit_cost_bps=0.0,
                        one_position_training_min_window_net_at_profit_cost_bps=-math.inf,
                        one_position_training_positive_windows_at_profit_cost=0,
                        one_position_training_minimum_execution_window_count=0,
                    ),
                    top_fraction=top_fraction,
                    cost_levels_bps=cost_levels_bps,
                    profit_cost_bps=profit_cost_bps,
                    minimum_executable_per_window=minimum_executable_per_window,
                )
            )

    def score(candidate: RegimeGateCandidate) -> tuple[bool, int, int, float, float]:
        training_window_count = scored_training["window_label"].nunique()
        all_positive = (
            candidate.one_position_training_positive_windows_at_profit_cost
            == training_window_count
        )
        return (
            all_positive,
            candidate.one_position_training_minimum_execution_window_count,
            candidate.one_position_training_positive_windows_at_profit_cost,
            candidate.one_position_training_min_window_net_at_profit_cost_bps,
            candidate.one_position_training_net_at_profit_cost_bps,
        )

    return max(candidates, key=score)


def _evaluate_gate(
    scored_training: pd.DataFrame,
    *,
    gate: RegimeGateCandidate,
    top_fraction: float,
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    minimum_executable_per_window: int,
) -> RegimeGateCandidate:
    selected = _select_gated_top_fraction(scored_training, gate=gate, top_fraction=top_fraction)
    if selected.empty:
        return gate
    net_values = []
    positive_count = 0
    min_exec_count = 0
    total_gross = 0.0
    total_exec = 0
    for _label, window in selected.groupby("window_label"):
        one_position = _apply_one_position_constraint_by_score(window)
        summary = summarize_executable_trades(
            one_position,
            selected_signal_count=len(window),
            cost_levels_bps=cost_levels_bps,
        )
        row = _cost_row(summary["cost_sensitivity"], profit_cost_bps)
        net_value = float(row["net_executable_mean_bps"])
        net_values.append(net_value)
        if net_value > 0:
            positive_count += 1
        if int(summary["executable_count"]) >= minimum_executable_per_window:
            min_exec_count += 1
        total_gross += float(summary["gross_executable_total_bps"])
        total_exec += int(summary["executable_count"])
    mean_net = (total_gross / total_exec - profit_cost_bps) if total_exec else 0.0
    return RegimeGateCandidate(
        feature=gate.feature,
        direction=gate.direction,
        threshold=gate.threshold,
        keep_fraction=gate.keep_fraction,
        selected_training_count=int(len(selected)),
        one_position_training_net_at_profit_cost_bps=float(mean_net),
        one_position_training_min_window_net_at_profit_cost_bps=float(min(net_values))
        if net_values
        else -math.inf,
        one_position_training_positive_windows_at_profit_cost=positive_count,
        one_position_training_minimum_execution_window_count=min_exec_count,
    )


def _select_gated_top_fraction(
    rows: pd.DataFrame,
    *,
    gate: RegimeGateCandidate,
    top_fraction: float,
) -> pd.DataFrame:
    allowed = rows
    if gate.feature is not None and gate.threshold is not None:
        values = rows[gate.feature]
        if gate.direction == "low":
            allowed = rows[values <= gate.threshold]
        elif gate.direction == "high":
            allowed = rows[values >= gate.threshold]
        else:
            raise ValueError(f"unsupported gate direction: {gate.direction}")
    selected_count = max(1, int(np.ceil(len(rows) * top_fraction)))
    return allowed.sort_values(
        ["model_score", "decision_time_ms"],
        ascending=[False, True],
        kind="stable",
    ).head(selected_count)


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


def _status(
    *,
    holdout_results: list[dict[str, object]],
    aggregate_by_mode: dict[str, dict[str, object]],
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> tuple[str, list[str]]:
    if not holdout_results:
        return "NO_VALID_REGIME_GATE_FOLDS", ["no_outer_holdout_results"]
    one_position = aggregate_by_mode["one_position"]
    if (
        one_position["all_windows_positive_at_stress_cost"]
        and one_position["all_windows_meet_minimum_executions"]
    ):
        return (
            "REGIME_GATE_PAYOFF_REPLICATED_RESEARCH_ONLY",
            [
                "not_approved_strategy",
                "future_shadow_replay_required",
                "testnet_release_still_requires_separate_authorization",
            ],
        )
    if (
        one_position["all_windows_positive_at_profit_cost"]
        and one_position["all_windows_meet_minimum_executions"]
    ):
        return (
            "REGIME_GATE_PAYOFF_PROFIT_COST_REPLICATED_STRESS_FRAGILE",
            [
                f"one_position_replicated_at_{profit_cost_bps:g}bps_cost",
                f"failed_or_fragile_at_{stress_cost_bps:g}bps_cost",
                "future_shadow_replay_required",
            ],
        )
    return (
        "REGIME_GATE_PAYOFF_FAILED_REPLICATION",
        [
            f"no_one_position_gate_replicated_all_windows_at_{profit_cost_bps:g}bps_cost",
            "do_not_promote_to_strategy",
        ],
    )


def _make_model(model_name: str) -> Any:
    if model_name == "hist_gradient_payoff":
        return _make_hist_gradient_regressor()
    if model_name == "ridge_payoff":
        return _make_ridge_regressor()
    raise ValueError(f"unsupported model: {model_name}")


def _gate_feature_columns(feature_columns: list[str]) -> list[str]:
    blocked_parts = ("side_sign", "side")
    return [
        column
        for column in feature_columns
        if not any(part in column for part in blocked_parts)
    ]


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


def _positive_at_cost(summary: dict[str, object], cost_bps: float) -> bool:
    return bool(_cost_row(summary["cost_sensitivity"], cost_bps)["net_positive"])


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


def _validate_parameters(
    *,
    wal_root_refs: tuple[WalRootRef, ...],
    model_name: str,
    horizon_seconds: int,
    target_bps: float,
    trailing_distance_bps: float,
    top_fraction: float,
    gate_keep_fractions: tuple[float, ...],
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
    if model_name not in {"hist_gradient_payoff", "ridge_payoff"}:
        raise ValueError("unsupported model")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if target_bps <= 0:
        raise ValueError("target_bps must be positive")
    if trailing_distance_bps <= 0:
        raise ValueError("trailing_distance_bps must be positive")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1")
    if not gate_keep_fractions or any(not 0 < value <= 1 for value in gate_keep_fractions):
        raise ValueError("gate_keep_fractions must contain values in (0, 1]")
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
        "aggregate_by_mode": report.get("aggregate_by_mode"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
