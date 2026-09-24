"""Fast96 veto-gate audit for the Fast95 near-edge payoff candidate.

Fast95 found a near-edge candidate, but not an approved strategy: the one-position
trailing replay was positive in 9/10 windows at 1 bps cost and still fragile to
cost/regime shifts. Fast96 asks the only useful follow-up: can a causal no-trade
gate, selected only on other windows, veto the bad regimes without leaking the
held-out window?

This is research-only. It never sends orders and never approves Testnet or LIVE.
"""

from __future__ import annotations

import argparse
import hashlib
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

from app.services.research.research_dataset import ResearchDatasetConfig  # noqa: E402
from scripts.research.run_book_executable_replay import (  # noqa: E402
    summarize_executable_trades,
)
from scripts.research.run_book_frozen_model_holdout import _prepare_feature_frames  # noqa: E402
from scripts.research.run_book_multi_window_model_scout import (  # noqa: E402
    WalRootRef,
    _apply_one_position_constraint_by_score,
    _load_window,
    _parse_float_list,
    _parse_wal_root_refs,
)
from scripts.research.run_book_payoff_model_scout import (  # noqa: E402
    _label_windows_with_executable_payoff,
    _make_hist_gradient_regressor,
    _make_ridge_regressor,
    _select_top_fraction,
    select_payoff_feature_columns,
)
from scripts.research.run_fast61_fast60_shadow_settlement import _write_jsonl  # noqa: E402


DEFAULT_WAL_ROOTS = ",".join(
    [
        "fast16=backend/data/microstructure_v1/prospective-wal-fast16-18m",
        "fast20=backend/data/microstructure_v1/prospective-wal-fast20-30m",
        "fast21=backend/data/microstructure_v1/prospective-wal-fast21-30m",
        "fast22=backend/data/microstructure_v1/prospective-wal-fast22-30m",
        "fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m",
        "fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m",
        "fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m",
        "fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m",
        "fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m",
        "fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m",
    ]
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast96-fast95-regime-veto-gate-audit"
)
DEFAULT_GATE_KEEP_FRACTIONS = (0.50, 0.70, 0.85, 1.00)
DEFAULT_MINIMUM_TRAIN_TRADES = 20
DEFAULT_MINIMUM_TRAIN_LABELS = 7
DEFAULT_MINIMUM_HOLDOUT_TRADES = 1
DEFAULT_MAXIMUM_TRAIN_NEGATIVE_LABELS = 0


@dataclass(frozen=True)
class GateSpec:
    feature_name: str | None
    direction: str
    quantile: float | None
    threshold: float | None
    keep_fraction: float

    @property
    def name(self) -> str:
        if self.feature_name is None:
            return "none"
        return (
            f"{self.feature_name}_{self.direction}_q{self.quantile:g}"
            f"_{self.threshold:.10g}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wal-roots",
        default=DEFAULT_WAL_ROOTS,
        help="Comma-separated LABEL=path list. Each label is one prospective window.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model",
        choices=("hist_gradient_payoff", "ridge_payoff"),
        default="hist_gradient_payoff",
    )
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--target-bps", type=float, default=3.0)
    parser.add_argument("--trailing-distance-bps", type=float, default=8.0)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--gate-keep-fractions", default="0.50,0.70,0.85,1.00")
    parser.add_argument(
        "--gate-feature-keywords",
        default="",
        help=(
            "Optional comma-separated substrings. When set, only gate features "
            "whose column names contain at least one keyword are evaluated."
        ),
    )
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--minimum-windows", type=int, default=8)
    parser.add_argument("--minimum-train-trades", type=int, default=DEFAULT_MINIMUM_TRAIN_TRADES)
    parser.add_argument("--minimum-train-labels", type=int, default=DEFAULT_MINIMUM_TRAIN_LABELS)
    parser.add_argument(
        "--minimum-holdout-trades",
        type=int,
        default=DEFAULT_MINIMUM_HOLDOUT_TRADES,
    )
    parser.add_argument(
        "--maximum-train-negative-labels",
        type=int,
        default=DEFAULT_MAXIMUM_TRAIN_NEGATIVE_LABELS,
    )
    arguments = parser.parse_args()

    report = run_fast96_fast95_regime_gate_audit(
        wal_root_refs=_parse_wal_root_refs(arguments.wal_roots),
        output_root=arguments.output_root,
        model_name=arguments.model,
        horizon_seconds=arguments.horizon_seconds,
        target_bps=arguments.target_bps,
        trailing_distance_bps=arguments.trailing_distance_bps,
        top_fraction=arguments.top_fraction,
        gate_keep_fractions=_parse_float_list(arguments.gate_keep_fractions),
        gate_feature_keywords=_parse_string_list(arguments.gate_feature_keywords),
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        latency_ms=arguments.latency_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
        profit_cost_bps=arguments.profit_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        minimum_windows=arguments.minimum_windows,
        minimum_train_trades=arguments.minimum_train_trades,
        minimum_train_labels=arguments.minimum_train_labels,
        minimum_holdout_trades=arguments.minimum_holdout_trades,
        maximum_train_negative_labels=arguments.maximum_train_negative_labels,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast96_fast95_regime_gate_audit(
    *,
    wal_root_refs: tuple[WalRootRef, ...],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model_name: str = "hist_gradient_payoff",
    horizon_seconds: int = 120,
    target_bps: float = 3.0,
    trailing_distance_bps: float = 8.0,
    top_fraction: float = 0.10,
    gate_keep_fractions: tuple[float, ...] = DEFAULT_GATE_KEEP_FRACTIONS,
    gate_feature_keywords: tuple[str, ...] = (),
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    latency_ms: int = 100,
    initial_stop_bps: float = 20.0,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    profit_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    minimum_windows: int = 8,
    minimum_train_trades: int = DEFAULT_MINIMUM_TRAIN_TRADES,
    minimum_train_labels: int = DEFAULT_MINIMUM_TRAIN_LABELS,
    minimum_holdout_trades: int = DEFAULT_MINIMUM_HOLDOUT_TRADES,
    maximum_train_negative_labels: int = DEFAULT_MAXIMUM_TRAIN_NEGATIVE_LABELS,
) -> dict[str, Any]:
    validate_parameters(
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
        minimum_train_trades=minimum_train_trades,
        minimum_train_labels=minimum_train_labels,
        minimum_holdout_trades=minimum_holdout_trades,
        maximum_train_negative_labels=maximum_train_negative_labels,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    base_report: dict[str, Any] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast96_is_regime_veto_gate_audit_for_fast95_candidate",
            "gate_selected_on_training_windows_only",
            "entry_model_and_gate_use_causal_pre_trade_features_only",
            "top_of_book_trailing_replay_not_exchange_execution",
            "future_shadow_replay_required_before_any_release",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast96_fast95_regime_veto_gate_audit",
        "leakage_guard": {
            "outer_validation": "leave_one_window_out",
            "training_entry_selection": "inner_leave_one_training_window_out",
            "holdout_never_used_for_gate_or_model_fit": True,
            "gate_semantics": "causal_veto_after_fast95_top_fraction_selection",
        },
        "wal_roots": [
            {"label": ref.label, "path": str(ref.path)} for ref in wal_root_refs
        ],
        "config": {
            "model": model_name,
            "horizon_seconds": horizon_seconds,
            "target_bps": target_bps,
            "breakeven_trigger_bps": target_bps,
            "trailing_activation_bps": target_bps,
            "trailing_distance_bps": trailing_distance_bps,
            "top_fraction": top_fraction,
            "gate_keep_fractions": list(gate_keep_fractions),
            "gate_feature_keywords": list(gate_feature_keywords),
            "decision_stride_seconds": decision_stride_seconds,
            "max_book_staleness_ms": max_book_staleness_ms,
            "latency_ms": latency_ms,
            "initial_stop_bps": initial_stop_bps,
            "cost_levels_bps": list(cost_levels_bps),
            "profit_cost_bps": profit_cost_bps,
            "stress_cost_bps": stress_cost_bps,
            "minimum_windows": minimum_windows,
            "minimum_train_trades": minimum_train_trades,
            "minimum_train_labels": minimum_train_labels,
            "minimum_holdout_trades": minimum_holdout_trades,
            "maximum_train_negative_labels": maximum_train_negative_labels,
        },
    }
    if len(wal_root_refs) < minimum_windows:
        return write_report(
            output_root,
            {
                **base_report,
                "status": "FAST96_INSUFFICIENT_WINDOWS",
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
        return write_report(
            output_root,
            {
                **base_report,
                "status": "FAST96_LOADED_WINDOW_COUNT_BELOW_MINIMUM",
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
    holdout_results = [
        evaluate_holdout_label(
            labelled_windows=labelled_windows,
            holdout_label=holdout_label,
            model_name=model_name,
            top_fraction=top_fraction,
            gate_keep_fractions=gate_keep_fractions,
            gate_feature_keywords=gate_feature_keywords,
            cost_levels_bps=cost_levels_bps,
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
            minimum_train_trades=minimum_train_trades,
            minimum_train_labels=minimum_train_labels,
            minimum_holdout_trades=minimum_holdout_trades,
            maximum_train_negative_labels=maximum_train_negative_labels,
        )
        for holdout_label in sorted(labelled_windows)
    ]
    aggregate = aggregate_holdout_results(
        holdout_results,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    aggregate["maximum_train_negative_labels"] = maximum_train_negative_labels
    aggregate["minimum_holdout_trades"] = minimum_holdout_trades
    status, status_reasons = fast96_status(aggregate)

    holdout_path = output_root / "holdout_results.jsonl"
    _write_jsonl(holdout_path, holdout_results)
    return write_report(
        output_root,
        {
            **base_report,
            "status": status,
            "status_reasons": status_reasons,
            "modeling_ready": True,
            "window_reports": [window_report(window) for window in loaded_windows],
            "labelled_row_count": int(sum(len(frame) for frame in labelled_windows.values())),
            "gate_candidate_semantics": {
                "none": "no veto after model top_fraction selection",
                "lte": "keep selected entries whose causal feature is <= training threshold",
                "gte": "keep selected entries whose causal feature is >= training threshold",
            },
            "holdout_results_path": str(holdout_path),
            "holdout_results_sha256": sha256_file(holdout_path),
            "holdouts": holdout_results,
            "aggregate": aggregate,
            "skipped": skipped,
        },
    )


def validate_parameters(
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
    minimum_train_trades: int,
    minimum_train_labels: int,
    minimum_holdout_trades: int,
    maximum_train_negative_labels: int,
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
    if minimum_train_trades < 1:
        raise ValueError("minimum_train_trades must be positive")
    if minimum_train_labels < 1:
        raise ValueError("minimum_train_labels must be positive")
    if minimum_holdout_trades < 1:
        raise ValueError("minimum_holdout_trades must be positive")
    if maximum_train_negative_labels < 0:
        raise ValueError("maximum_train_negative_labels cannot be negative")
    if not cost_levels_bps or any(value < 0 for value in cost_levels_bps):
        raise ValueError("cost_levels_bps must contain non-negative values")
    if profit_cost_bps not in cost_levels_bps:
        raise ValueError("profit_cost_bps must be present in cost_levels_bps")
    if stress_cost_bps not in cost_levels_bps:
        raise ValueError("stress_cost_bps must be present in cost_levels_bps")


def evaluate_holdout_label(
    *,
    labelled_windows: dict[str, pd.DataFrame],
    holdout_label: str,
    model_name: str,
    top_fraction: float,
    gate_keep_fractions: tuple[float, ...],
    gate_feature_keywords: tuple[str, ...],
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_train_trades: int,
    minimum_train_labels: int,
    minimum_holdout_trades: int,
    maximum_train_negative_labels: int,
) -> dict[str, Any]:
    training_labels = [label for label in sorted(labelled_windows) if label != holdout_label]
    training_frame = pd.concat(
        [labelled_windows[label] for label in training_labels],
        ignore_index=True,
    )
    holdout_frame = labelled_windows[holdout_label].copy()
    feature_columns = select_payoff_feature_columns(training_frame)
    if not feature_columns:
        return empty_holdout_result(
            holdout_label=holdout_label,
            training_labels=training_labels,
            status="FAST96_NO_CAUSAL_PAYOFF_FEATURES",
            reason="no_training_causal_numeric_features",
            minimum_holdout_trades=minimum_holdout_trades,
        )

    selected_training_rows = inner_oof_top_fraction_rows(
        labelled_windows=labelled_windows,
        training_labels=training_labels,
        model_name=model_name,
        feature_columns=feature_columns,
        top_fraction=top_fraction,
    )
    gate_feature_columns = gate_feature_columns_from(feature_columns)
    gate_feature_columns = filter_gate_feature_columns(
        gate_feature_columns,
        keywords=gate_feature_keywords,
    )
    candidates, evaluated_gate_count = fit_training_gate_candidates(
        selected_training_rows,
        expected_labels=training_labels,
        gate_feature_columns=gate_feature_columns,
        gate_keep_fractions=gate_keep_fractions,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_train_trades=minimum_train_trades,
        minimum_train_labels=minimum_train_labels,
        maximum_train_negative_labels=maximum_train_negative_labels,
    )
    if not candidates:
        return empty_holdout_result(
            holdout_label=holdout_label,
            training_labels=training_labels,
            status="FAST96_NO_STRICT_TRAINING_SURVIVOR",
            reason="no_gate_positive_in_training_under_negative_label_limit",
            minimum_holdout_trades=minimum_holdout_trades,
            extra={
                "feature_count": len(feature_columns),
                "gate_feature_count": len(gate_feature_columns),
                "training_selected_signal_count": int(len(selected_training_rows)),
                "evaluated_gate_count": evaluated_gate_count,
            },
        )

    selected_candidate = max(candidates, key=lambda candidate: float(candidate["train_score"]))
    gate = gate_from_dict(selected_candidate["gate"])
    scored_holdout = score_holdout_rows(
        training_frame=training_frame,
        holdout_frame=holdout_frame,
        model_name=model_name,
        feature_columns=feature_columns,
    )
    pre_gate_holdout = _select_top_fraction(
        scored_holdout,
        scored_holdout["model_score"].to_numpy(dtype=np.float64),
        top_fraction,
    )
    gated_holdout = apply_gate_frame(pre_gate_holdout, gate)
    independent_summary = summarize_executable_trades(
        gated_holdout,
        selected_signal_count=len(gated_holdout),
        cost_levels_bps=cost_levels_bps,
    )
    one_position = _apply_one_position_constraint_by_score(gated_holdout)
    one_position_summary = summarize_executable_trades(
        one_position,
        selected_signal_count=len(gated_holdout),
        cost_levels_bps=cost_levels_bps,
    )
    status = holdout_status(
        one_position_summary,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    return {
        "holdout_label": holdout_label,
        "training_labels": training_labels,
        "status": status,
        "selected": selected_candidate,
        "candidate_count": len(candidates),
        "evaluated_gate_count": evaluated_gate_count,
        "feature_count": len(feature_columns),
        "gate_feature_count": len(gate_feature_columns),
        "training_selected_signal_count": int(len(selected_training_rows)),
        "test_count": int(len(holdout_frame)),
        "pre_gate_selected_signal_count": int(len(pre_gate_holdout)),
        "selected_signal_count": int(len(gated_holdout)),
        "selected_gross_mean_bps": float(gated_holdout["gross_executable_bps"].mean())
        if len(gated_holdout)
        else 0.0,
        "selected_predicted_payoff_mean_bps": float(gated_holdout["model_score"].mean())
        if len(gated_holdout)
        else 0.0,
        "independent_summary": independent_summary,
        "one_position_summary": one_position_summary,
        "minimum_holdout_trades": minimum_holdout_trades,
    }


def inner_oof_top_fraction_rows(
    *,
    labelled_windows: dict[str, pd.DataFrame],
    training_labels: list[str],
    model_name: str,
    feature_columns: list[str],
    top_fraction: float,
) -> pd.DataFrame:
    frames = []
    for inner_holdout in training_labels:
        inner_train_labels = [label for label in training_labels if label != inner_holdout]
        if not inner_train_labels:
            continue
        inner_train = pd.concat(
            [labelled_windows[label] for label in inner_train_labels],
            ignore_index=True,
        )
        inner_test = labelled_windows[inner_holdout].copy()
        if inner_train.empty or inner_test.empty:
            continue
        y_train = inner_train["gross_executable_bps"].to_numpy(dtype=np.float64)
        if np.nanstd(y_train) == 0:
            continue
        prepared_train, prepared_test, _medians = _prepare_feature_frames(
            inner_train,
            inner_test,
            feature_columns=feature_columns,
        )
        model = make_model(model_name)
        model.fit(prepared_train, y_train)
        scores = model.predict(prepared_test)
        frames.append(_select_top_fraction(inner_test, scores, top_fraction))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def score_holdout_rows(
    *,
    training_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    model_name: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    prepared_training, prepared_holdout, _medians = _prepare_feature_frames(
        training_frame,
        holdout_frame,
        feature_columns=feature_columns,
    )
    model = make_model(model_name)
    model.fit(
        prepared_training,
        training_frame["gross_executable_bps"].to_numpy(dtype=np.float64),
    )
    scored = holdout_frame.copy()
    scored["model_score"] = model.predict(prepared_holdout)
    return scored


def fit_training_gate_candidates(
    rows: pd.DataFrame,
    *,
    expected_labels: list[str],
    gate_feature_columns: list[str],
    gate_keep_fractions: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_train_trades: int,
    minimum_train_labels: int,
    maximum_train_negative_labels: int,
) -> tuple[list[dict[str, Any]], int]:
    candidates = []
    sorted_rows = rows.sort_values(
        ["window_label", "decision_time_ms", "model_score"],
        ascending=[True, True, False],
        kind="stable",
    )
    gate_specs = fit_gate_specs(sorted_rows, gate_feature_columns, gate_keep_fractions)
    for gate in gate_specs:
        gated = apply_gate_frame(sorted_rows, gate)
        one_position = apply_one_position_by_label(gated, presorted=True)
        summary = summarize_labelled_one_position_rows(
            one_position,
            expected_labels=expected_labels,
            selected_signal_count=len(gated),
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
        )
        if not training_summary_survives(
            summary,
            minimum_train_trades=minimum_train_trades,
            minimum_train_labels=minimum_train_labels,
            maximum_train_negative_labels=maximum_train_negative_labels,
        ):
            continue
        candidates.append(
            {
                "name": gate.name,
                "gate": gate_to_dict(gate),
                "train_summary": summary,
                "train_score": train_candidate_score(summary),
            }
        )
    return sorted(
        candidates,
        key=lambda candidate: float(candidate["train_score"]),
        reverse=True,
    ), len(gate_specs)


def fit_gate_specs(
    rows: pd.DataFrame,
    gate_feature_columns: list[str],
    gate_keep_fractions: tuple[float, ...],
) -> list[GateSpec]:
    gates = [
        GateSpec(
            feature_name=None,
            direction="none",
            quantile=None,
            threshold=None,
            keep_fraction=1.0,
        )
    ]
    if rows.empty:
        return gates
    for feature_name in gate_feature_columns:
        if feature_name not in rows.columns:
            continue
        values = rows[feature_name].replace([np.inf, -np.inf], np.nan).dropna()
        if values.nunique() < 3:
            continue
        for keep_fraction in sorted(set(gate_keep_fractions)):
            if keep_fraction >= 1.0:
                continue
            lte_threshold = float(values.quantile(keep_fraction))
            gte_quantile = 1.0 - keep_fraction
            gte_threshold = float(values.quantile(gte_quantile))
            gates.append(
                GateSpec(
                    feature_name=feature_name,
                    direction="lte",
                    quantile=keep_fraction,
                    threshold=lte_threshold,
                    keep_fraction=keep_fraction,
                )
            )
            gates.append(
                GateSpec(
                    feature_name=feature_name,
                    direction="gte",
                    quantile=gte_quantile,
                    threshold=gte_threshold,
                    keep_fraction=keep_fraction,
                )
            )
    return gates


def apply_gate_frame(rows: pd.DataFrame, gate: GateSpec) -> pd.DataFrame:
    if gate.feature_name is None:
        return rows.copy()
    if gate.feature_name not in rows.columns:
        return rows.iloc[0:0].copy()
    values = rows[gate.feature_name].replace([np.inf, -np.inf], np.nan)
    if gate.direction == "lte":
        mask = values.notna() & (values <= float(gate.threshold))
    elif gate.direction == "gte":
        mask = values.notna() & (values >= float(gate.threshold))
    else:
        raise ValueError(f"unsupported gate direction: {gate.direction}")
    return rows.loc[mask].copy()


def apply_one_position_by_label(rows: pd.DataFrame, *, presorted: bool = False) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    if presorted:
        return apply_presorted_one_position_by_label(rows)
    frames = [
        _apply_one_position_constraint_by_score(window)
        for _label, window in rows.groupby("window_label", sort=True)
    ]
    if not frames:
        return rows.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def apply_presorted_one_position_by_label(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    kept_indices = []
    current_label = None
    position_exit_ms = -1
    for index, row in zip(rows.index, rows.itertuples(index=False), strict=True):
        window_label = row.window_label
        if window_label != current_label:
            current_label = window_label
            position_exit_ms = -1
        decision_time_ms = int(row.decision_time_ms)
        if decision_time_ms <= position_exit_ms:
            continue
        kept_indices.append(index)
        position_exit_ms = int(row.exit_time_ms)
    return rows.loc[kept_indices].reset_index(drop=True)


def apply_presorted_one_position_by_score(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    kept_indices = []
    position_exit_ms = -1
    for index, row in zip(rows.index, rows.itertuples(index=False), strict=True):
        decision_time_ms = int(row.decision_time_ms)
        if decision_time_ms <= position_exit_ms:
            continue
        kept_indices.append(index)
        position_exit_ms = int(row.exit_time_ms)
    return rows.loc[kept_indices].reset_index(drop=True)


def summarize_labelled_one_position_rows(
    rows: pd.DataFrame,
    *,
    expected_labels: list[str],
    selected_signal_count: int,
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> dict[str, Any]:
    rows_by_label = {
        str(label): frame.copy() for label, frame in rows.groupby("window_label", sort=True)
    }
    label_summaries = {}
    trade_count = 0
    gross_total = 0.0
    positive_profit_labels = 0
    negative_profit_labels = 0
    positive_stress_labels = 0
    empty_labels = 0
    traded_label_count = 0
    profit_values = []
    stress_values = []
    for label in expected_labels:
        frame = rows_by_label.get(label, rows.iloc[0:0])
        label_trade_count = int(len(frame))
        label_gross_total = (
            float(frame["gross_executable_bps"].sum()) if label_trade_count else 0.0
        )
        profit_net = label_gross_total - label_trade_count * profit_cost_bps
        stress_net = label_gross_total - label_trade_count * stress_cost_bps
        if label_trade_count:
            traded_label_count += 1
            profit_values.append(profit_net)
            stress_values.append(stress_net)
            if profit_net > 0:
                positive_profit_labels += 1
            elif profit_net < 0:
                negative_profit_labels += 1
            if stress_net > 0:
                positive_stress_labels += 1
        else:
            empty_labels += 1
        trade_count += label_trade_count
        gross_total += label_gross_total
        label_summaries[label] = {
            "trade_count": label_trade_count,
            "gross_total_bps": float(label_gross_total),
            "profit_net_total_bps": float(profit_net),
            "stress_net_total_bps": float(stress_net),
        }
    profit_total = gross_total - trade_count * profit_cost_bps
    stress_total = gross_total - trade_count * stress_cost_bps
    return {
        "selected_signal_count": int(selected_signal_count),
        "trade_count": int(trade_count),
        "input_label_count": len(expected_labels),
        "label_count": int(traded_label_count),
        "empty_label_count": int(empty_labels),
        "gross_total_bps": float(gross_total),
        "gross_mean_bps": float(gross_total / trade_count) if trade_count else 0.0,
        "profit_cost_bps": float(profit_cost_bps),
        "profit_net_total_bps": float(profit_total),
        "profit_net_mean_bps": float(profit_total / trade_count) if trade_count else 0.0,
        "stress_cost_bps": float(stress_cost_bps),
        "stress_net_total_bps": float(stress_total),
        "stress_net_mean_bps": float(stress_total / trade_count) if trade_count else 0.0,
        "positive_label_count": int(positive_profit_labels),
        "negative_label_count": int(negative_profit_labels),
        "stress_positive_label_count": int(positive_stress_labels),
        "worst_label_profit_net_total_bps": float(min(profit_values))
        if profit_values
        else None,
        "worst_label_stress_net_total_bps": float(min(stress_values))
        if stress_values
        else None,
        "label_summaries": label_summaries,
    }


def training_summary_survives(
    summary: dict[str, Any],
    *,
    minimum_train_trades: int,
    minimum_train_labels: int,
    maximum_train_negative_labels: int,
) -> bool:
    return (
        int(summary["trade_count"]) >= minimum_train_trades
        and int(summary["label_count"]) >= minimum_train_labels
        and float(summary["profit_net_total_bps"]) > 0
        and float(summary["profit_net_mean_bps"]) > 0
        and int(summary["negative_label_count"]) <= maximum_train_negative_labels
    )


def train_candidate_score(summary: dict[str, Any]) -> float:
    worst_profit = summary["worst_label_profit_net_total_bps"]
    worst_component = -100.0 if worst_profit is None else float(worst_profit)
    return (
        5.0 * worst_component
        + float(summary["profit_net_total_bps"])
        + 0.25 * float(summary["stress_net_total_bps"])
        + 5.0 * int(summary["positive_label_count"])
        + 0.02 * int(summary["trade_count"])
        - 50.0 * int(summary["negative_label_count"])
        - 2.0 * int(summary["empty_label_count"])
    )


def holdout_status(
    summary: dict[str, Any],
    *,
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_holdout_trades: int,
) -> str:
    if int(summary["executable_count"]) < minimum_holdout_trades:
        return "FAST96_HOLDOUT_INSUFFICIENT_TRADES"
    if net_total_at_cost(summary, profit_cost_bps) <= 0:
        return "FAST96_HOLDOUT_PROFIT_COST_NON_POSITIVE"
    if net_total_at_cost(summary, stress_cost_bps) <= 0:
        return "FAST96_HOLDOUT_PROFIT_POSITIVE_STRESS_FRAGILE"
    return "FAST96_HOLDOUT_STRESS_POSITIVE"


def aggregate_holdout_results(
    holdout_results: list[dict[str, Any]],
    *,
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> dict[str, Any]:
    stress_positive = [
        result
        for result in holdout_results
        if result["status"] == "FAST96_HOLDOUT_STRESS_POSITIVE"
    ]
    stress_fragile = [
        result
        for result in holdout_results
        if result["status"] == "FAST96_HOLDOUT_PROFIT_POSITIVE_STRESS_FRAGILE"
    ]
    profit_negative = [
        result
        for result in holdout_results
        if result["status"] == "FAST96_HOLDOUT_PROFIT_COST_NON_POSITIVE"
    ]
    insufficient = [
        result
        for result in holdout_results
        if result["status"] == "FAST96_HOLDOUT_INSUFFICIENT_TRADES"
    ]
    no_survivor = [
        result
        for result in holdout_results
        if result["status"] == "FAST96_NO_STRICT_TRAINING_SURVIVOR"
    ]
    evaluated = [
        result
        for result in holdout_results
        if result["status"]
        not in {
            "FAST96_NO_STRICT_TRAINING_SURVIVOR",
            "FAST96_NO_CAUSAL_PAYOFF_FEATURES",
        }
    ]
    profit_net_values = [
        net_total_at_cost(result["one_position_summary"], profit_cost_bps)
        for result in evaluated
    ]
    stress_net_values = [
        net_total_at_cost(result["one_position_summary"], stress_cost_bps)
        for result in evaluated
    ]
    trade_counts = [
        int(result["one_position_summary"]["executable_count"]) for result in evaluated
    ]
    return {
        "holdout_count": len(holdout_results),
        "training_survivor_count": len(evaluated),
        "stress_positive_holdout_count": len(stress_positive),
        "profit_positive_holdout_count": len(stress_positive) + len(stress_fragile),
        "stress_fragile_holdout_count": len(stress_fragile),
        "profit_negative_holdout_count": len(profit_negative),
        "insufficient_holdout_count": len(insufficient),
        "no_strict_training_survivor_count": len(no_survivor),
        "holdout_trade_count": sum(trade_counts),
        "holdout_profit_net_total_bps": float(sum(profit_net_values)),
        "holdout_stress_net_total_bps": float(sum(stress_net_values)),
        "worst_holdout_profit_net_total_bps": float(min(profit_net_values))
        if profit_net_values
        else None,
        "worst_holdout_stress_net_total_bps": float(min(stress_net_values))
        if stress_net_values
        else None,
        "stress_positive_holdout_labels": [
            result["holdout_label"] for result in stress_positive
        ],
        "stress_fragile_holdout_labels": [result["holdout_label"] for result in stress_fragile],
        "profit_negative_holdout_labels": [
            result["holdout_label"] for result in profit_negative
        ],
        "insufficient_holdout_labels": [result["holdout_label"] for result in insufficient],
        "no_survivor_holdout_labels": [result["holdout_label"] for result in no_survivor],
    }


def fast96_status(aggregate: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "leave_one_window_out_fast95_regime_veto_gate_audit",
    ]
    holdout_count = int(aggregate["holdout_count"])
    if holdout_count == 0:
        return "FAST96_NO_HOLDOUT_WINDOWS", reasons + ["no_holdout_results"]
    if int(aggregate["no_strict_training_survivor_count"]) == holdout_count:
        return "FAST96_NO_STRICT_TRAINING_SURVIVORS", reasons + [
            (
                "no_gate_positive_in_training_under_negative_label_limit:"
                f"{aggregate.get('maximum_train_negative_labels', 0)}"
            )
        ]
    if int(aggregate["stress_positive_holdout_count"]) == holdout_count:
        return "FAST96_FAST95_REGIME_GATE_SURVIVOR_DIAGNOSTIC_ONLY", reasons + [
            "all_holdouts_positive_at_stress_cost",
            "requires_new_preregistration_and_shadow_replay",
        ]
    if (
        int(aggregate["profit_positive_holdout_count"]) == holdout_count
        and int(aggregate["insufficient_holdout_count"]) == 0
        and int(aggregate["no_strict_training_survivor_count"]) == 0
    ):
        return (
            "FAST96_FAST95_REGIME_GATE_PROFIT_COST_REPLICATED_STRESS_FRAGILE_DIAGNOSTIC_ONLY",
            reasons
            + [
                "all_holdouts_positive_at_profit_cost",
                "at_least_one_holdout_fragile_at_stress_cost",
                "requires_new_preregistration_and_shadow_replay",
            ],
        )
    return "FAST96_FAST95_REGIME_GATE_FAILED_REPLICATION", reasons + [
        f"profit_positive_holdouts:{aggregate['profit_positive_holdout_count']}/{holdout_count}",
        f"stress_positive_holdouts:{aggregate['stress_positive_holdout_count']}/{holdout_count}",
        f"profit_negative_holdouts:{aggregate['profit_negative_holdout_count']}",
        f"insufficient_holdouts:{aggregate['insufficient_holdout_count']}",
        f"no_strict_training_survivors:{aggregate['no_strict_training_survivor_count']}",
        "do_not_promote_to_strategy",
    ]


def net_total_at_cost(summary: dict[str, Any], cost_bps: float) -> float:
    for row in summary["cost_sensitivity"]:
        if float(row["explicit_cost_bps"]) == float(cost_bps):
            return float(row["net_executable_total_bps"])
    raise ValueError(f"cost level missing from summary: {cost_bps:g} bps")


def gate_feature_columns_from(feature_columns: list[str]) -> list[str]:
    blocked_parts = ("side_sign", "side")
    return [
        column
        for column in feature_columns
        if not any(part in column for part in blocked_parts)
    ]


def filter_gate_feature_columns(
    columns: list[str],
    *,
    keywords: tuple[str, ...],
) -> list[str]:
    if not keywords:
        return columns
    normalized_keywords = tuple(keyword.lower() for keyword in keywords)
    return [
        column
        for column in columns
        if any(keyword in column.lower() for keyword in normalized_keywords)
    ]


def _parse_string_list(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def make_model(model_name: str) -> Any:
    if model_name == "hist_gradient_payoff":
        return _make_hist_gradient_regressor()
    if model_name == "ridge_payoff":
        return _make_ridge_regressor()
    raise ValueError(f"unsupported model: {model_name}")


def empty_holdout_result(
    *,
    holdout_label: str,
    training_labels: list[str],
    status: str,
    reason: str,
    minimum_holdout_trades: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "holdout_label": holdout_label,
        "training_labels": training_labels,
        "status": status,
        "status_reason": reason,
        "selected": None,
        "candidate_count": 0,
        "holdout_summary": empty_summary(),
        "one_position_summary": empty_executable_summary(),
        "independent_summary": empty_executable_summary(),
        "minimum_holdout_trades": minimum_holdout_trades,
    }
    if extra:
        payload.update(extra)
    return payload


def empty_summary() -> dict[str, Any]:
    return {
        "selected_signal_count": 0,
        "trade_count": 0,
        "input_label_count": 0,
        "label_count": 0,
        "empty_label_count": 0,
        "gross_total_bps": 0.0,
        "gross_mean_bps": 0.0,
        "profit_cost_bps": 0.0,
        "profit_net_total_bps": 0.0,
        "profit_net_mean_bps": 0.0,
        "stress_cost_bps": 0.0,
        "stress_net_total_bps": 0.0,
        "stress_net_mean_bps": 0.0,
        "positive_label_count": 0,
        "negative_label_count": 0,
        "stress_positive_label_count": 0,
        "worst_label_profit_net_total_bps": None,
        "worst_label_stress_net_total_bps": None,
        "label_summaries": {},
    }


def empty_executable_summary() -> dict[str, Any]:
    return {
        "selected_signal_count": 0,
        "executable_count": 0,
        "skipped_count": 0,
        "gross_executable_mean_bps": 0.0,
        "gross_executable_total_bps": 0.0,
        "break_even_explicit_cost_bps": 0.0,
        "gross_win_rate": 0.0,
        "maximum_drawdown_bps": 0.0,
        "exit_reasons": {},
        "cost_sensitivity": [
            {
                "explicit_cost_bps": 0.0,
                "net_executable_mean_bps": 0.0,
                "net_executable_total_bps": 0.0,
                "net_positive": False,
            }
        ],
    }


def gate_to_dict(gate: GateSpec) -> dict[str, Any]:
    return asdict(gate)


def gate_from_dict(payload: dict[str, Any]) -> GateSpec:
    return GateSpec(
        feature_name=payload["feature_name"],
        direction=str(payload["direction"]),
        quantile=None if payload["quantile"] is None else float(payload["quantile"]),
        threshold=None if payload["threshold"] is None else float(payload["threshold"]),
        keep_fraction=float(payload["keep_fraction"]),
    )


def window_report(window: Any) -> dict[str, Any]:
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


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    payload = {
        **report,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "aggregate": report.get("aggregate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
