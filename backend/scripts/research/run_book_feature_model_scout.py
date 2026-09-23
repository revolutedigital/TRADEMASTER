"""Leak-safe feature model scout on prospective book WAL captures.

The script tests whether causal book/flow features can rank short-horizon
targets better than the chronological baseline. It is intentionally scoped as a
diagnostic: no order submission, no strategy approval, and no Testnet/LIVE
authorization.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
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

from app.services.research.research_dataset import (  # noqa: E402
    ResearchDatasetConfig,
    build_research_rows,
    load_book_interval,
    load_mark_interval,
    load_trade_interval,
)
from scripts.research.run_book_wal_scout import (  # noqa: E402
    decision_times_for_wal,
    summarize_stream,
)

LABEL_AND_FUTURE_COLUMNS = frozenset(
    {
        "decision_time_ms",
        "entry_sequence_id",
        "entry_price",
        "side",
        "horizon_seconds",
        "event_count",
        "mfe_bps",
        "mae_bps",
        "terminal_bps",
        "first_touch",
        "first_touch_time_ms",
        "first_touch_sequence_id",
        "expected_touch_time_ms",
        "expected_touch_sequence_id",
        "stress_touch_time_ms",
        "stress_touch_sequence_id",
        "stop_touch_time_ms",
        "stop_touch_sequence_id",
        "paid_expected_before_stop",
        "paid_stress_before_stop",
        "target",
        "gross_path_payoff_bps",
    }
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
        / "book-feature-model-scout",
    )
    parser.add_argument("--horizons-seconds", default="30")
    parser.add_argument("--thresholds-bps", default="0.5,1,2,3")
    parser.add_argument("--top-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--temporal-blocks", type=int, default=4)
    parser.add_argument("--minimum-train-blocks", type=int, default=2)
    parser.add_argument("--minimum-selected-count", type=int, default=10)
    arguments = parser.parse_args()
    report = run_book_feature_model_scout(
        wal_root=arguments.wal_root,
        output_root=arguments.output_root,
        horizons_seconds=_parse_int_list(arguments.horizons_seconds),
        thresholds_bps=_parse_float_list(arguments.thresholds_bps),
        top_fractions=_parse_float_list(arguments.top_fractions),
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        temporal_blocks=arguments.temporal_blocks,
        minimum_train_blocks=arguments.minimum_train_blocks,
        minimum_selected_count=arguments.minimum_selected_count,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_feature_model_scout(
    *,
    wal_root: Path,
    output_root: Path,
    horizons_seconds: tuple[int, ...] = (30,),
    thresholds_bps: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    initial_stop_bps: float = 20.0,
    temporal_blocks: int = 4,
    minimum_train_blocks: int = 2,
    minimum_selected_count: int = 10,
) -> dict[str, object]:
    _validate_scout_parameters(
        thresholds_bps=thresholds_bps,
        top_fractions=top_fractions,
        initial_stop_bps=initial_stop_bps,
        temporal_blocks=temporal_blocks,
        minimum_train_blocks=minimum_train_blocks,
        minimum_selected_count=minimum_selected_count,
    )
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
        "experiment_stage": "book_feature_model_scout",
        "wal_root": str(wal_root),
        "config": {
            **asdict(base_config),
            "thresholds_bps": list(thresholds_bps),
            "top_fractions": list(top_fractions),
            "temporal_blocks": temporal_blocks,
            "minimum_train_blocks": minimum_train_blocks,
            "minimum_selected_count": minimum_selected_count,
        },
        "leakage_guard": {
            "excluded_columns": sorted(LABEL_AND_FUTURE_COLUMNS),
            "future_columns_excluded": ["mfe_bps", "mae_bps", "terminal_bps"],
        },
        "streams": {name: asdict(summary) for name, summary in streams.items()},
        "approved_strategy": False,
        "approval_reasons": [
            "diagnostic_only",
            "short_single_capture",
            "no_statistical_gate",
            "no_transaction_cost_confirmation",
        ],
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
                "candidates": [],
            },
        )

    decisions = decision_times_for_wal(
        trade=streams["trade"],
        book=streams["depth"],
        config=base_config,
    )
    if len(decisions) < temporal_blocks:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "INSUFFICIENT_INTERVAL",
                "status_reasons": [
                    f"decision_count_below_temporal_blocks:{len(decisions)}<{temporal_blocks}"
                ],
                "decision_count": int(len(decisions)),
                "modeling_ready": False,
                "candidates": [],
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
    all_candidates: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
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
        rows = _add_gross_path_payoff(rows, threshold_bps, initial_stop_bps)
        feature_columns = select_causal_feature_columns(rows)
        if not feature_columns:
            skipped.append(
                {
                    "threshold_bps": threshold_bps,
                    "reason": "no_complete_causal_numeric_features",
                }
            )
            continue
        threshold_candidates, threshold_skipped = _run_threshold_models(
            rows=rows,
            threshold_bps=threshold_bps,
            feature_columns=feature_columns,
            top_fractions=top_fractions,
            temporal_blocks=temporal_blocks,
            minimum_train_blocks=minimum_train_blocks,
        )
        all_candidates.extend(threshold_candidates)
        skipped.extend(threshold_skipped)

    eligible = [
        candidate
        for candidate in all_candidates
        if candidate["selected_count"] >= minimum_selected_count
    ]
    best_candidate = (
        max(eligible, key=lambda item: item["selected_gross_path_payoff_mean_bps"])
        if eligible
        else None
    )
    return _write_report(
        output_root,
        {
            **base_report,
            "status": "READY_FOR_MODEL_SCOUT_REVIEW"
            if all_candidates
            else "NO_VALID_MODEL_FOLDS",
            "status_reasons": [] if all_candidates else ["no_candidate_with_binary_folds"],
            "modeling_ready": bool(all_candidates),
            "decision_count": int(len(decisions)),
            "row_count_per_threshold": int(len(decisions) * 2 * len(horizons_seconds)),
            "feature_count": int(all_candidates[0]["feature_count"])
            if all_candidates
            else 0,
            "candidates": sorted(
                all_candidates,
                key=lambda item: (
                    item["threshold_bps"],
                    item["model"],
                    item["top_fraction"],
                ),
            ),
            "skipped": skipped,
            "best_candidate": best_candidate,
            "interval_start": _iso(int(decisions.min())),
            "interval_end": _iso(int(decisions.max())),
        },
    )


def select_causal_feature_columns(rows: pd.DataFrame) -> list[str]:
    """Return finite numeric columns that are known at decision time."""
    columns: list[str] = []
    for column in rows.columns:
        if column in LABEL_AND_FUTURE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(rows[column]):
            continue
        values = rows[column].to_numpy(dtype=np.float64)
        if np.isfinite(values).all():
            columns.append(column)
    return columns


def _run_threshold_models(
    *,
    rows: pd.DataFrame,
    threshold_bps: float,
    feature_columns: list[str],
    top_fractions: tuple[float, ...],
    temporal_blocks: int,
    minimum_train_blocks: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    unique_decisions = np.array(sorted(rows["decision_time_ms"].unique()), dtype=np.int64)
    decision_blocks = np.array_split(unique_decisions, temporal_blocks)
    candidate_accumulator: dict[tuple[str, float], dict[str, Any]] = {}
    skipped: list[dict[str, object]] = []
    model_factories = {
        "logistic_balanced": _make_logistic_model,
        "hist_gradient": _make_hist_gradient_model,
    }
    for test_block_index in range(minimum_train_blocks, temporal_blocks):
        train_times = np.concatenate(decision_blocks[:test_block_index])
        test_times = decision_blocks[test_block_index]
        train = rows[rows["decision_time_ms"].isin(train_times)]
        test = rows[rows["decision_time_ms"].isin(test_times)]
        if train["target"].nunique() < 2 or test["target"].nunique() < 2:
            skipped.append(
                {
                    "threshold_bps": threshold_bps,
                    "fold": test_block_index,
                    "reason": "train_or_test_target_not_binary",
                    "train_target_rate": float(train["target"].mean())
                    if len(train)
                    else None,
                    "test_target_rate": float(test["target"].mean())
                    if len(test)
                    else None,
                }
            )
            continue
        for model_name, model_factory in model_factories.items():
            model = model_factory()
            model.fit(train[feature_columns], train["target"].to_numpy())
            probabilities = model.predict_proba(test[feature_columns])[:, 1]
            for top_fraction in top_fractions:
                selected = _select_top_fraction(test, probabilities, top_fraction)
                key = (model_name, top_fraction)
                aggregate = candidate_accumulator.setdefault(
                    key,
                    {
                        "threshold_bps": threshold_bps,
                        "model": model_name,
                        "top_fraction": top_fraction,
                        "fold_count": 0,
                        "test_count": 0,
                        "test_target_hits": 0,
                        "test_gross_path_payoff_sum_bps": 0.0,
                        "selected_count": 0,
                        "selected_target_hits": 0,
                        "selected_gross_path_payoff_sum_bps": 0.0,
                        "feature_count": len(feature_columns),
                    },
                )
                _accumulate_candidate(aggregate, test, selected)
    return [
        _finalize_candidate(candidate)
        for candidate in candidate_accumulator.values()
        if candidate["selected_count"] > 0 and candidate["test_count"] > 0
    ], skipped


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


def _select_top_fraction(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    top_fraction: float,
) -> pd.DataFrame:
    selected_count = max(1, int(np.ceil(len(test) * top_fraction)))
    selected_positions = np.argsort(probabilities)[-selected_count:]
    return test.iloc[selected_positions]


def _accumulate_candidate(
    aggregate: dict[str, Any],
    test: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    aggregate["fold_count"] += 1
    aggregate["test_count"] += int(len(test))
    aggregate["test_target_hits"] += int(test["target"].sum())
    aggregate["test_gross_path_payoff_sum_bps"] += float(
        test["gross_path_payoff_bps"].sum()
    )
    aggregate["selected_count"] += int(len(selected))
    aggregate["selected_target_hits"] += int(selected["target"].sum())
    aggregate["selected_gross_path_payoff_sum_bps"] += float(
        selected["gross_path_payoff_bps"].sum()
    )


def _finalize_candidate(candidate: dict[str, Any]) -> dict[str, object]:
    test_count = int(candidate["test_count"])
    selected_count = int(candidate["selected_count"])
    base_target_rate = candidate["test_target_hits"] / test_count
    selected_target_rate = candidate["selected_target_hits"] / selected_count
    base_payoff = candidate["test_gross_path_payoff_sum_bps"] / test_count
    selected_payoff = candidate["selected_gross_path_payoff_sum_bps"] / selected_count
    return {
        "threshold_bps": float(candidate["threshold_bps"]),
        "model": str(candidate["model"]),
        "top_fraction": float(candidate["top_fraction"]),
        "fold_count": int(candidate["fold_count"]),
        "test_count": test_count,
        "selected_count": selected_count,
        "base_target_rate": float(base_target_rate),
        "selected_target_rate": float(selected_target_rate),
        "target_rate_lift_pp": float(selected_target_rate - base_target_rate),
        "base_gross_path_payoff_mean_bps": float(base_payoff),
        "selected_gross_path_payoff_mean_bps": float(selected_payoff),
        "gross_path_payoff_lift_bps": float(selected_payoff - base_payoff),
        "feature_count": int(candidate["feature_count"]),
    }


def _add_gross_path_payoff(
    rows: pd.DataFrame,
    threshold_bps: float,
    initial_stop_bps: float,
) -> pd.DataFrame:
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
    return enriched


def _validate_scout_parameters(
    *,
    thresholds_bps: tuple[float, ...],
    top_fractions: tuple[float, ...],
    initial_stop_bps: float,
    temporal_blocks: int,
    minimum_train_blocks: int,
    minimum_selected_count: int,
) -> None:
    if not thresholds_bps or any(threshold <= 0 for threshold in thresholds_bps):
        raise ValueError("thresholds_bps must contain positive values")
    if not top_fractions or any(not 0 < fraction < 1 for fraction in top_fractions):
        raise ValueError("top_fractions must contain values between 0 and 1")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if temporal_blocks < 3:
        raise ValueError("temporal_blocks must be at least 3")
    if minimum_train_blocks < 1 or minimum_train_blocks >= temporal_blocks:
        raise ValueError("minimum_train_blocks must be between 1 and temporal_blocks - 1")
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
        "approved_strategy": report["approved_strategy"],
        "decision_count": report.get("decision_count", 0),
        "feature_count": report.get("feature_count", 0),
        "best_candidate": report.get("best_candidate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
