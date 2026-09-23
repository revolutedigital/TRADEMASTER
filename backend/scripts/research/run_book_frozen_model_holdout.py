"""Evaluate one frozen multi-window model candidate on a prospective holdout.

The scout in ``run_book_multi_window_model_scout.py`` searches a small grid.
This script is deliberately narrower: it takes one fixed candidate, trains the
specified model on prior windows only, scores one holdout window, and replays a
research-only executable trailing-stop exit. It never submits orders and never
approves a strategy.
"""

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

from app.services.research.research_dataset import ResearchDatasetConfig  # noqa: E402
from scripts.research.run_book_executable_replay import (  # noqa: E402
    _book_arrays,
    summarize_executable_trades,
)
from scripts.research.run_book_feature_model_scout import (  # noqa: E402
    LABEL_AND_FUTURE_COLUMNS,
)
from scripts.research.run_book_multi_window_model_scout import (  # noqa: E402
    WalRootRef,
    _apply_one_position_constraint_by_score,
    _float_key,
    _load_window,
    _make_hist_gradient_model,
    _make_logistic_model,
    _parse_float_list,
    _parse_wal_root_refs,
    _replay_selected_rows_with_trailing,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-wal-roots",
        required=True,
        help="Comma-separated LABEL=path list used only for model fitting.",
    )
    parser.add_argument(
        "--holdout-wal-root",
        required=True,
        help="Single holdout in LABEL=path format.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-frozen-model-holdout",
    )
    parser.add_argument(
        "--model",
        choices=("hist_gradient", "logistic_balanced"),
        default="hist_gradient",
    )
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--target-bps", type=float, default=8.0)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--trailing-distance-bps", type=float, default=3.0)
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--minimum-executable-count", type=int, default=5)
    arguments = parser.parse_args()

    report = run_book_frozen_model_holdout(
        training_wal_root_refs=_parse_wal_root_refs(arguments.training_wal_roots),
        holdout_wal_root_ref=_parse_single_wal_root_ref(arguments.holdout_wal_root),
        output_root=arguments.output_root,
        model_name=arguments.model,
        horizon_seconds=arguments.horizon_seconds,
        target_bps=arguments.target_bps,
        top_fraction=arguments.top_fraction,
        trailing_distance_bps=arguments.trailing_distance_bps,
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        latency_ms=arguments.latency_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
        profit_cost_bps=arguments.profit_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        minimum_executable_count=arguments.minimum_executable_count,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_frozen_model_holdout(
    *,
    training_wal_root_refs: tuple[WalRootRef, ...],
    holdout_wal_root_ref: WalRootRef,
    output_root: Path,
    model_name: str = "hist_gradient",
    horizon_seconds: int = 120,
    target_bps: float = 8.0,
    top_fraction: float = 0.05,
    trailing_distance_bps: float = 3.0,
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    latency_ms: int = 100,
    initial_stop_bps: float = 20.0,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    profit_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    minimum_executable_count: int = 5,
) -> dict[str, object]:
    _validate_parameters(
        training_wal_root_refs=training_wal_root_refs,
        holdout_wal_root_ref=holdout_wal_root_ref,
        model_name=model_name,
        horizon_seconds=horizon_seconds,
        target_bps=target_bps,
        top_fraction=top_fraction,
        trailing_distance_bps=trailing_distance_bps,
        decision_stride_seconds=decision_stride_seconds,
        max_book_staleness_ms=max_book_staleness_ms,
        latency_ms=latency_ms,
        initial_stop_bps=initial_stop_bps,
        cost_levels_bps=cost_levels_bps,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_executable_count=minimum_executable_count,
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
    base_report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "approval_reasons": [
            "single_frozen_holdout_is_not_strategy_approval",
            "top_of_book_trailing_replay_not_exchange_execution",
            "no_queue_partial_fill_or_gap_slippage_model",
            "additional_prospective_windows_required",
        ],
        "experiment_stage": "book_frozen_model_holdout",
        "training_wal_roots": [
            {"label": ref.label, "path": str(ref.path)} for ref in training_wal_root_refs
        ],
        "holdout_wal_root": {
            "label": holdout_wal_root_ref.label,
            "path": str(holdout_wal_root_ref.path),
        },
        "config": {
            **asdict(config),
            "model": model_name,
            "top_fraction": top_fraction,
            "trailing_distance_bps": trailing_distance_bps,
            "latency_ms": latency_ms,
            "cost_levels_bps": list(cost_levels_bps),
            "profit_cost_bps": profit_cost_bps,
            "stress_cost_bps": stress_cost_bps,
            "minimum_executable_count": minimum_executable_count,
            "breakeven_trigger_equals_target_bps": True,
            "trailing_activation_equals_target_bps": True,
        },
        "leakage_guard": {
            "training_only_feature_selection": True,
            "holdout_rows_never_used_for_fit": True,
            "excluded_columns": sorted(LABEL_AND_FUTURE_COLUMNS),
            "imputation": "training_median_for_non_finite_feature_values",
        },
    }

    loaded_training = []
    skipped = []
    for ref in training_wal_root_refs:
        loaded = _load_window(ref, config=config)
        if isinstance(loaded, dict):
            skipped.append({"window": ref.label, **loaded})
            continue
        loaded_training.append(loaded)
    if len(loaded_training) != len(training_wal_root_refs):
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "TRAINING_WINDOW_LOAD_FAILED",
                "status_reasons": ["at_least_one_training_window_failed_to_load"],
                "modeling_ready": False,
                "skipped": skipped,
            },
        )
    loaded_holdout = _load_window(holdout_wal_root_ref, config=config)
    if isinstance(loaded_holdout, dict):
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "HOLDOUT_WINDOW_LOAD_FAILED",
                "status_reasons": ["holdout_window_failed_to_load"],
                "modeling_ready": False,
                "skipped": [{"window": holdout_wal_root_ref.label, **loaded_holdout}],
            },
        )

    training_rows = pd.concat(
        [window.rows for window in loaded_training],
        ignore_index=True,
    )
    holdout_rows = loaded_holdout.rows
    feature_columns = select_training_feature_columns(training_rows)
    if not feature_columns:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "NO_TRAINING_FEATURES",
                "status_reasons": ["no_complete_training_causal_numeric_features"],
                "modeling_ready": False,
                "training_windows": [_window_report(window) for window in loaded_training],
                "holdout_window": _window_report(loaded_holdout),
            },
        )
    if training_rows["target"].nunique() < 2:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "TRAINING_TARGET_NOT_BINARY",
                "status_reasons": ["training_target_not_binary"],
                "modeling_ready": False,
                "training_target_rate": float(training_rows["target"].mean()),
                "training_windows": [_window_report(window) for window in loaded_training],
                "holdout_window": _window_report(loaded_holdout),
            },
        )

    prepared_training, prepared_holdout, medians = _prepare_feature_frames(
        training_rows,
        holdout_rows,
        feature_columns=feature_columns,
    )
    model = _make_model(model_name)
    model.fit(prepared_training, training_rows["target"].to_numpy())
    probabilities = model.predict_proba(prepared_holdout)[:, 1]
    selected = _select_top_fraction(
        holdout_rows,
        probabilities,
        top_fraction=top_fraction,
    )
    replay_rows = _replay_selected_rows_with_trailing(
        selected,
        books_by_window={loaded_holdout.label: _book_arrays(loaded_holdout.book_events)},
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
    status, status_reasons = _status(
        independent_summary=independent_summary,
        one_position_summary=one_position_summary,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_executable_count=minimum_executable_count,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    selected_path = output_root / "selected_holdout_rows.parquet"
    trades_path = output_root / "holdout_trades.parquet"
    one_position_path = output_root / "holdout_one_position_trades.parquet"
    selected.to_parquet(selected_path, index=False, compression="zstd")
    trades.to_parquet(trades_path, index=False, compression="zstd")
    one_position.to_parquet(one_position_path, index=False, compression="zstd")
    return _write_report(
        output_root,
        {
            **base_report,
            "status": status,
            "status_reasons": status_reasons,
            "modeling_ready": True,
            "training_windows": [_window_report(window) for window in loaded_training],
            "holdout_window": _window_report(loaded_holdout),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "feature_medians": {column: float(medians[column]) for column in feature_columns},
            "training_row_count": int(len(training_rows)),
            "holdout_row_count": int(len(holdout_rows)),
            "holdout_target_rate": float(holdout_rows["target"].mean()),
            "selected_signal_count": int(len(selected)),
            "selected_target_rate": float(selected["target"].mean()) if len(selected) else 0.0,
            "independent_summary": independent_summary,
            "one_position_summary": one_position_summary,
            "selected_rows_path": str(selected_path),
            "trades_path": str(trades_path),
            "one_position_trades_path": str(one_position_path),
            "skipped": skipped,
        },
    )


def select_training_feature_columns(rows: pd.DataFrame) -> list[str]:
    """Return numeric causal features selected on training data only."""
    columns: list[str] = []
    for column in rows.columns:
        if column in LABEL_AND_FUTURE_COLUMNS or column == "window_label":
            continue
        if not pd.api.types.is_numeric_dtype(rows[column]):
            continue
        values = rows[column].to_numpy(dtype=np.float64)
        if np.isfinite(values).any():
            columns.append(column)
    return columns


def _prepare_feature_frames(
    training_rows: pd.DataFrame,
    holdout_rows: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    training = training_rows[feature_columns].replace([np.inf, -np.inf], np.nan)
    medians = training.median()
    medians = medians.fillna(0.0)
    prepared_training = training.fillna(medians)
    holdout = holdout_rows[feature_columns].replace([np.inf, -np.inf], np.nan)
    prepared_holdout = holdout.fillna(medians)
    return prepared_training, prepared_holdout, medians


def _select_top_fraction(
    rows: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    top_fraction: float,
) -> pd.DataFrame:
    selected_count = max(1, int(np.ceil(len(rows) * top_fraction)))
    scored = rows.copy()
    scored["model_score"] = probabilities
    return scored.sort_values(
        ["model_score", "decision_time_ms"],
        ascending=[False, True],
        kind="stable",
    ).head(selected_count)


def _make_model(model_name: str) -> Any:
    if model_name == "hist_gradient":
        return _make_hist_gradient_model()
    if model_name == "logistic_balanced":
        return _make_logistic_model()
    raise ValueError(f"unsupported model: {model_name}")


def _status(
    *,
    independent_summary: dict[str, object],
    one_position_summary: dict[str, object],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_count: int,
) -> tuple[str, list[str]]:
    reasons = []
    if int(one_position_summary["executable_count"]) < minimum_executable_count:
        reasons.append(
            "one_position_executable_count_below_minimum:"
            f"{one_position_summary['executable_count']}<{minimum_executable_count}"
        )
    if int(independent_summary["executable_count"]) < minimum_executable_count:
        reasons.append(
            "independent_executable_count_below_minimum:"
            f"{independent_summary['executable_count']}<{minimum_executable_count}"
        )
    if not _positive_at_cost(one_position_summary, profit_cost_bps):
        reasons.append(f"one_position_net_at_{profit_cost_bps:g}bps_non_positive")
    if not _positive_at_cost(independent_summary, profit_cost_bps):
        reasons.append(f"independent_net_at_{profit_cost_bps:g}bps_non_positive")
    if reasons:
        return "FROZEN_MODEL_HOLDOUT_FAILED", reasons
    if not _positive_at_cost(one_position_summary, stress_cost_bps) or not _positive_at_cost(
        independent_summary,
        stress_cost_bps,
    ):
        return (
            "FROZEN_MODEL_HOLDOUT_PROFIT_COST_PASSED_STRESS_FRAGILE_RESEARCH_ONLY",
            [
                f"passed_{profit_cost_bps:g}bps_profit_cost",
                f"failed_or_fragile_at_{stress_cost_bps:g}bps_stress_cost",
                "not_approved_strategy",
            ],
        )
    return (
        "FROZEN_MODEL_HOLDOUT_PASSED_RESEARCH_ONLY",
        [
            "single_holdout_passed",
            "not_approved_strategy",
            "additional_prospective_windows_required",
        ],
    )


def _positive_at_cost(summary: dict[str, object], cost_bps: float) -> bool:
    for item in summary["cost_sensitivity"]:
        if float(item["explicit_cost_bps"]) == cost_bps:
            return bool(item["net_positive"])
    raise ValueError(f"cost level missing from summary: {cost_bps:g}")


def _window_report(window: Any) -> dict[str, object]:
    return {
        "label": window.label,
        "decision_count": window.decision_count,
        "row_count": window.row_count,
        "interval_start": window.interval_start,
        "interval_end": window.interval_end,
        "target_rate": float(window.rows["target"].mean()),
        "streams": window.streams,
    }


def _validate_parameters(
    *,
    training_wal_root_refs: tuple[WalRootRef, ...],
    holdout_wal_root_ref: WalRootRef,
    model_name: str,
    horizon_seconds: int,
    target_bps: float,
    top_fraction: float,
    trailing_distance_bps: float,
    decision_stride_seconds: int,
    max_book_staleness_ms: int,
    latency_ms: int,
    initial_stop_bps: float,
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_count: int,
) -> None:
    labels = [ref.label for ref in training_wal_root_refs]
    if not training_wal_root_refs:
        raise ValueError("at least one training WAL root is required")
    if len(set(labels)) != len(labels):
        raise ValueError("training WAL labels must be unique")
    if holdout_wal_root_ref.label in set(labels):
        raise ValueError("holdout label must not duplicate a training label")
    if model_name not in {"hist_gradient", "logistic_balanced"}:
        raise ValueError("unsupported model")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if target_bps <= 0:
        raise ValueError("target_bps must be positive")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1")
    if trailing_distance_bps <= 0:
        raise ValueError("trailing_distance_bps must be positive")
    if decision_stride_seconds <= 0:
        raise ValueError("decision_stride_seconds must be positive")
    if max_book_staleness_ms < 0:
        raise ValueError("max_book_staleness_ms cannot be negative")
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if minimum_executable_count < 1:
        raise ValueError("minimum_executable_count must be positive")
    if not cost_levels_bps or any(value < 0 for value in cost_levels_bps):
        raise ValueError("cost_levels_bps must contain non-negative values")
    if profit_cost_bps not in cost_levels_bps:
        raise ValueError("profit_cost_bps must be present in cost_levels_bps")
    if stress_cost_bps not in cost_levels_bps:
        raise ValueError("stress_cost_bps must be present in cost_levels_bps")


def _parse_single_wal_root_ref(value: str) -> WalRootRef:
    refs = _parse_wal_root_refs(value)
    if len(refs) != 1:
        raise ValueError("holdout WAL root must contain exactly one LABEL=path ref")
    return refs[0]


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
        "selected_signal_count": report.get("selected_signal_count", 0),
        "independent_summary": report.get("independent_summary"),
        "one_position_summary": report.get("one_position_summary"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
