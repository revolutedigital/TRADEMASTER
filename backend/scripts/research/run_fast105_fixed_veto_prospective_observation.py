"""Fast105 preregistered prospective observation for the fixed outlier veto.

Fast105 freezes the strongest Fast104 diagnostic into a prospective-only
observer:

historical 13-window training -> hist_gradient_payoff top 5% ->
spot_return_1s_bps >= 0 -> one-position replay.

It is research-only. It never sends orders, writes ledger state, or approves
Testnet/LIVE trading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.research_dataset import ResearchDatasetConfig  # noqa: E402
from scripts.research.run_book_executable_replay import summarize_executable_trades  # noqa: E402
from scripts.research.run_book_frozen_model_holdout import _prepare_feature_frames  # noqa: E402
from scripts.research.run_book_multi_window_model_scout import (  # noqa: E402
    WalRootRef,
    _apply_one_position_constraint_by_score,
    _load_window,
    _parse_wal_root_refs,
)
from scripts.research.run_book_payoff_model_scout import (  # noqa: E402
    _label_windows_with_executable_payoff,
    _select_top_fraction,
    select_payoff_feature_columns,
)
from scripts.research.run_fast61_fast60_shadow_settlement import _write_jsonl  # noqa: E402
from scripts.research.run_fast104_fixed_outlier_veto_audit import (  # noqa: E402
    DEFAULT_COST_LEVELS_BPS,
    DEFAULT_WAL_ROOTS,
    float_key,
    make_model,
    net_total_at_cost,
    window_report,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast105-fixed-veto-prospective-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast105-fixed-veto-prospective-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast105-registry.jsonl"
PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-23T23:30:31Z")
PREREGISTRATION_SHA256 = "609b9d070309f0da8dc7389bffa285d2df430aeae8028bb734c9ec5c32669aa9"
SOURCE_FAST104_REPORT_SHA256 = (
    "17a21f43e4c63fb27460c063cfa49c18ef88ada3aaeefe58e462711c4af90d35"
)
FIXED_VETO_FEATURE = "spot_return_1s_bps"
FIXED_VETO_MIN = 0.0
MINIMUM_PROSPECTIVE_TRADES_FOR_REVIEW = 30
MINIMUM_PROSPECTIVE_WINDOWS_FOR_REVIEW = 3
EARLY_FAIL_MINIMUM_TRADES = 10
EARLY_FAIL_STRESS_TOTAL_BPS = -20.0
WINDOW_FAIL_MINIMUM_TRADES = 3
WINDOW_FAIL_STRESS_TOTAL_BPS = -15.0
MINIMUM_STRESS_MEAN_BPS_FOR_REVIEW = 1.5
MINIMUM_PROFIT_MEAN_BPS_FOR_REVIEW = 2.5
MINIMUM_STRESS_POSITIVE_WINDOW_FRACTION = 0.75


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-wal-roots", default=DEFAULT_WAL_ROOTS)
    parser.add_argument(
        "--validation-wal-roots",
        default="",
        help=(
            "Comma-separated LABEL=path list for WALs collected after the "
            "Fast105 preregistration. Empty means wait."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--model",
        choices=("hist_gradient_payoff", "ridge_payoff"),
        default="hist_gradient_payoff",
    )
    parser.add_argument("--horizon-seconds", type=int, default=300)
    parser.add_argument("--target-bps", type=float, default=10.0)
    parser.add_argument("--trailing-distance-bps", type=float, default=15.0)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--minimum-training-windows", type=int, default=13)
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    arguments = parser.parse_args()
    validation_refs = (
        _parse_wal_root_refs(arguments.validation_wal_roots)
        if arguments.validation_wal_roots.strip()
        else ()
    )
    report = run_fast105_fixed_veto_prospective_observation(
        training_wal_root_refs=_parse_wal_root_refs(arguments.training_wal_roots),
        validation_wal_root_refs=validation_refs,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        model_name=arguments.model,
        horizon_seconds=arguments.horizon_seconds,
        target_bps=arguments.target_bps,
        trailing_distance_bps=arguments.trailing_distance_bps,
        top_fraction=arguments.top_fraction,
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        latency_ms=arguments.latency_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        minimum_training_windows=arguments.minimum_training_windows,
        profit_cost_bps=arguments.profit_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast105_fixed_veto_prospective_observation(
    *,
    training_wal_root_refs: tuple[WalRootRef, ...],
    validation_wal_root_refs: tuple[WalRootRef, ...] = (),
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    model_name: str = "hist_gradient_payoff",
    horizon_seconds: int = 300,
    target_bps: float = 10.0,
    trailing_distance_bps: float = 15.0,
    top_fraction: float = 0.05,
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    latency_ms: int = 100,
    initial_stop_bps: float = 20.0,
    cost_levels_bps: tuple[float, ...] = DEFAULT_COST_LEVELS_BPS,
    minimum_training_windows: int = 13,
    profit_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
) -> dict[str, Any]:
    validate_parameters(
        training_wal_root_refs=training_wal_root_refs,
        validation_wal_root_refs=validation_wal_root_refs,
        model_name=model_name,
        horizon_seconds=horizon_seconds,
        target_bps=target_bps,
        trailing_distance_bps=trailing_distance_bps,
        top_fraction=top_fraction,
        minimum_training_windows=minimum_training_windows,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        cost_levels_bps=cost_levels_bps,
    )
    preregistration = validate_preregistration(
        preregistration_path=preregistration_path,
        registry_path=registry_path,
    )
    base_report = base_report_payload(
        training_wal_root_refs=training_wal_root_refs,
        validation_wal_root_refs=validation_wal_root_refs,
        preregistration_path=preregistration_path,
        registry_path=registry_path,
        preregistration=preregistration,
        model_name=model_name,
        horizon_seconds=horizon_seconds,
        target_bps=target_bps,
        trailing_distance_bps=trailing_distance_bps,
        top_fraction=top_fraction,
        decision_stride_seconds=decision_stride_seconds,
        max_book_staleness_ms=max_book_staleness_ms,
        latency_ms=latency_ms,
        initial_stop_bps=initial_stop_bps,
        minimum_training_windows=minimum_training_windows,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if not validation_wal_root_refs:
        return write_fast105_report(
            output_root,
            {
                **base_report,
                "status": "FAST105_WAITING_FOR_PROSPECTIVE_WAL_ROOTS",
                "status_reasons": [
                    "not_approved_strategy",
                    "research_only",
                    "validation_wal_roots_required",
                    "waiting_for_post_preregistration_wal",
                ],
                "modeling_ready": False,
                "training_window_reports": [],
                "validation_window_reports": [],
                "prospective_results": [],
                "aggregate": empty_aggregate(cost_levels_bps),
                "gates": empty_gates(),
            },
        )

    overlapping_labels = sorted(
        {ref.label for ref in training_wal_root_refs}
        & {ref.label for ref in validation_wal_root_refs}
    )
    if overlapping_labels:
        return write_fast105_report(
            output_root,
            {
                **base_report,
                "status": "FAST105_VALIDATION_LABEL_OVERLAPS_TRAINING",
                "status_reasons": [
                    "not_approved_strategy",
                    "research_only",
                    "validation_labels_must_be_new",
                    f"overlapping_labels:{','.join(overlapping_labels)}",
                ],
                "modeling_ready": False,
                "training_window_reports": [],
                "validation_window_reports": [],
                "prospective_results": [],
                "aggregate": empty_aggregate(cost_levels_bps),
                "gates": empty_gates(),
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
    training_windows, training_skipped = load_windows(training_wal_root_refs, config=config)
    validation_windows, validation_skipped = load_windows(validation_wal_root_refs, config=config)
    if len(training_windows) < minimum_training_windows:
        return write_fast105_report(
            output_root,
            {
                **base_report,
                "status": "FAST105_INSUFFICIENT_TRAINING_WINDOWS",
                "status_reasons": [
                    "not_approved_strategy",
                    "research_only",
                    (
                        "training_window_count_below_minimum:"
                        f"{len(training_windows)}<{minimum_training_windows}"
                    ),
                ],
                "modeling_ready": False,
                "training_window_reports": [window_report(window) for window in training_windows],
                "validation_window_reports": [window_report(window) for window in validation_windows],
                "training_skipped": training_skipped,
                "validation_skipped": validation_skipped,
                "prospective_results": [],
                "aggregate": empty_aggregate(cost_levels_bps),
                "gates": empty_gates(),
            },
        )
    if not validation_windows:
        return write_fast105_report(
            output_root,
            {
                **base_report,
                "status": "FAST105_NO_LOADABLE_VALIDATION_WINDOWS",
                "status_reasons": [
                    "not_approved_strategy",
                    "research_only",
                    "no_loadable_validation_windows",
                ],
                "modeling_ready": False,
                "training_window_reports": [window_report(window) for window in training_windows],
                "validation_window_reports": [],
                "training_skipped": training_skipped,
                "validation_skipped": validation_skipped,
                "prospective_results": [],
                "aggregate": empty_aggregate(cost_levels_bps),
                "gates": empty_gates(),
            },
        )

    labelled_training = _label_windows_with_executable_payoff(
        training_windows,
        target_bps=target_bps,
        trailing_distance_bps=trailing_distance_bps,
        horizon_seconds=horizon_seconds,
        initial_stop_bps=initial_stop_bps,
        latency_ms=latency_ms,
        max_book_staleness_ms=max_book_staleness_ms,
    )
    labelled_validation = _label_windows_with_executable_payoff(
        validation_windows,
        target_bps=target_bps,
        trailing_distance_bps=trailing_distance_bps,
        horizon_seconds=horizon_seconds,
        initial_stop_bps=initial_stop_bps,
        latency_ms=latency_ms,
        max_book_staleness_ms=max_book_staleness_ms,
    )
    post_preregistration_frames, validation_time_audit = post_preregistration_frames_by_label(
        labelled_validation
    )
    if not post_preregistration_frames:
        empty_paths = write_result_artifacts(output_root, [], [], [])
        return write_fast105_report(
            output_root,
            {
                **base_report,
                "status": "FAST105_WAITING_FOR_POST_PREREGISTRATION_DATA",
                "status_reasons": [
                    "not_approved_strategy",
                    "research_only",
                    "no_decisions_after_preregistration_timestamp",
                ],
                "modeling_ready": True,
                "training_window_reports": [window_report(window) for window in training_windows],
                "validation_window_reports": [window_report(window) for window in validation_windows],
                "validation_time_audit": validation_time_audit,
                "training_skipped": training_skipped,
                "validation_skipped": validation_skipped,
                "prospective_results": [],
                "aggregate": empty_aggregate(cost_levels_bps),
                "gates": empty_gates(),
                **empty_paths,
            },
        )

    training_frame = pd.concat(labelled_training.values(), ignore_index=True)
    validation_frame = pd.concat(post_preregistration_frames.values(), ignore_index=True)
    feature_audit = audit_required_features(validation_frame)
    if feature_audit["missing_required_features"]:
        empty_paths = write_result_artifacts(output_root, [], [], [])
        return write_fast105_report(
            output_root,
            {
                **base_report,
                "status": "FAST105_MISSING_REQUIRED_FEATURES",
                "status_reasons": [
                    "not_approved_strategy",
                    "research_only",
                    "required_feature_missing_fail_closed",
                    *[
                        f"missing_feature:{feature}"
                        for feature in feature_audit["missing_required_features"]
                    ],
                ],
                "modeling_ready": False,
                "training_window_reports": [window_report(window) for window in training_windows],
                "validation_window_reports": [window_report(window) for window in validation_windows],
                "validation_time_audit": validation_time_audit,
                "feature_audit": feature_audit,
                "training_skipped": training_skipped,
                "validation_skipped": validation_skipped,
                "prospective_results": [],
                "aggregate": empty_aggregate(cost_levels_bps),
                "gates": empty_gates(),
                **empty_paths,
            },
        )

    prospective_results, scored_rows, selected_rows, one_position_rows = evaluate_prospective(
        training_frame=training_frame,
        validation_frame=validation_frame,
        model_name=model_name,
        top_fraction=top_fraction,
        cost_levels_bps=cost_levels_bps,
    )
    aggregate = aggregate_prospective(
        prospective_results,
        cost_levels_bps=cost_levels_bps,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    gates = evaluate_gates(
        aggregate=aggregate,
        prospective_results=prospective_results,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    status, status_reasons = fast105_status(gates)
    artifact_paths = write_result_artifacts(
        output_root,
        jsonable_records(scored_rows),
        jsonable_records(selected_rows),
        jsonable_records(one_position_rows),
    )
    return write_fast105_report(
        output_root,
        {
            **base_report,
            "status": status,
            "status_reasons": status_reasons,
            "modeling_ready": True,
            "training_window_reports": [window_report(window) for window in training_windows],
            "validation_window_reports": [window_report(window) for window in validation_windows],
            "validation_time_audit": validation_time_audit,
            "feature_audit": feature_audit,
            "training_skipped": training_skipped,
            "validation_skipped": validation_skipped,
            "training_row_count": int(len(training_frame)),
            "post_preregistration_row_count": int(len(validation_frame)),
            "prospective_results": prospective_results,
            "aggregate": aggregate,
            "gates": gates,
            **artifact_paths,
        },
    )


def validate_parameters(
    *,
    training_wal_root_refs: tuple[WalRootRef, ...],
    validation_wal_root_refs: tuple[WalRootRef, ...],
    model_name: str,
    horizon_seconds: int,
    target_bps: float,
    trailing_distance_bps: float,
    top_fraction: float,
    minimum_training_windows: int,
    profit_cost_bps: float,
    stress_cost_bps: float,
    cost_levels_bps: tuple[float, ...],
) -> None:
    if len({ref.label for ref in training_wal_root_refs}) != len(training_wal_root_refs):
        raise ValueError("training wal root labels must be unique")
    if len({ref.label for ref in validation_wal_root_refs}) != len(validation_wal_root_refs):
        raise ValueError("validation wal root labels must be unique")
    if model_name not in {"hist_gradient_payoff", "ridge_payoff"}:
        raise ValueError("unsupported model")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if target_bps <= 0 or trailing_distance_bps <= 0:
        raise ValueError("target and trailing distance must be positive")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1")
    if minimum_training_windows < 2:
        raise ValueError("minimum_training_windows must be at least 2")
    if profit_cost_bps not in cost_levels_bps or stress_cost_bps not in cost_levels_bps:
        raise ValueError("profit and stress costs must be included in cost_levels_bps")


def validate_preregistration(*, preregistration_path: Path, registry_path: Path) -> dict[str, str]:
    if not preregistration_path.exists():
        raise FileNotFoundError(preregistration_path)
    if not registry_path.exists():
        raise FileNotFoundError(registry_path)
    preregistration_sha256 = sha256_file(preregistration_path)
    if preregistration_sha256 != PREREGISTRATION_SHA256:
        raise ValueError("Fast105 preregistration hash changed")
    registry_events = load_jsonl(registry_path)
    matching_events = [
        event
        for event in registry_events
        if event.get("event") == "fast105_fixed_veto_prospective_preregistered"
    ]
    if len(matching_events) != 1:
        raise ValueError("Fast105 registry must contain exactly one preregistration event")
    event = matching_events[0]
    expected = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "approved_strategy": False,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "fast104_source_report_sha256": SOURCE_FAST104_REPORT_SHA256,
        "fixed_veto": "spot_return_1s_bps>=0.0",
        "model": "hist_gradient_payoff",
        "target_bps": 10.0,
        "trailing_distance_bps": 15.0,
        "top_fraction": 0.05,
        "preregistration_timestamp": "2026-09-23T23:30:31Z",
    }
    mismatches = {
        key: {"expected": value, "actual": event.get(key)}
        for key, value in expected.items()
        if event.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fast105 registry mismatch: {mismatches}")
    return {"preregistration_sha256": preregistration_sha256}


def load_windows(
    wal_root_refs: tuple[WalRootRef, ...],
    *,
    config: ResearchDatasetConfig,
) -> tuple[list[Any], list[dict[str, Any]]]:
    loaded_windows = []
    skipped = []
    for ref in wal_root_refs:
        loaded = _load_window(ref, config=config)
        if isinstance(loaded, dict):
            skipped.append({"window": ref.label, **loaded})
        else:
            loaded_windows.append(loaded)
    return loaded_windows, skipped


def post_preregistration_frames_by_label(
    labelled_validation: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    cutoff_ms = preregistration_timestamp_ms()
    frames = {}
    audit = []
    for label, frame in labelled_validation.items():
        post_frame = frame.loc[frame["decision_time_ms"].astype("int64") > cutoff_ms].copy()
        pre_count = int(len(frame) - len(post_frame))
        audit.append(
            {
                "label": label,
                "total_row_count": int(len(frame)),
                "pre_or_at_preregistration_row_count": pre_count,
                "post_preregistration_row_count": int(len(post_frame)),
                "first_decision_time_ms": int(frame["decision_time_ms"].min()) if len(frame) else None,
                "last_decision_time_ms": int(frame["decision_time_ms"].max()) if len(frame) else None,
                "eligible_after_ms": cutoff_ms,
            }
        )
        if not post_frame.empty:
            frames[label] = post_frame
    return frames, audit


def audit_required_features(rows: pd.DataFrame) -> dict[str, Any]:
    missing_required_features = []
    missing_counts = {}
    for feature_name in (FIXED_VETO_FEATURE,):
        if feature_name not in rows.columns:
            missing_required_features.append(feature_name)
            missing_counts[feature_name] = int(len(rows))
        else:
            missing_counts[feature_name] = int(rows[feature_name].isna().sum())
            if missing_counts[feature_name] > 0:
                missing_required_features.append(feature_name)
    return {
        "record_count": int(len(rows)),
        "missing_counts": missing_counts,
        "missing_required_features": missing_required_features,
        "scoreable": not missing_required_features,
    }


def evaluate_prospective(
    *,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    model_name: str,
    top_fraction: float,
    cost_levels_bps: tuple[float, ...],
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = select_payoff_feature_columns(training_frame)
    prepared_training, prepared_validation, _medians = _prepare_feature_frames(
        training_frame,
        validation_frame,
        feature_columns=feature_columns,
    )
    model = make_model(model_name)
    model.fit(
        prepared_training,
        training_frame["gross_executable_bps"].to_numpy(dtype=np.float64),
    )
    scored_validation = validation_frame.copy()
    scored_validation["model_score"] = model.predict(prepared_validation)
    selected_frames = []
    one_position_frames = []
    prospective_results = []
    for label, label_rows in scored_validation.groupby("window_label", sort=True):
        selected = _select_top_fraction(
            label_rows,
            label_rows["model_score"].to_numpy(dtype=np.float64),
            top_fraction,
        )
        gated = apply_fixed_veto(selected)
        one_position = _apply_one_position_constraint_by_score(gated)
        selected_frames.append(gated)
        one_position_frames.append(one_position)
        prospective_results.append(
            {
                "label": str(label),
                "post_preregistration_row_count": int(len(label_rows)),
                "pre_gate_selected_signal_count": int(len(selected)),
                "selected_signal_count": int(len(gated)),
                "fixed_veto": {
                    "feature_name": FIXED_VETO_FEATURE,
                    "direction": "gte",
                    "threshold": FIXED_VETO_MIN,
                },
                "independent_summary": summarize_executable_trades(
                    gated,
                    selected_signal_count=len(gated),
                    cost_levels_bps=cost_levels_bps,
                ),
                "one_position_summary": summarize_executable_trades(
                    one_position,
                    selected_signal_count=len(gated),
                    cost_levels_bps=cost_levels_bps,
                ),
            }
        )
    selected_rows = concat_or_empty(selected_frames)
    one_position_rows = concat_or_empty(one_position_frames)
    return prospective_results, scored_validation, selected_rows, one_position_rows


def apply_fixed_veto(rows: pd.DataFrame) -> pd.DataFrame:
    if FIXED_VETO_FEATURE not in rows.columns:
        return rows.iloc[0:0].copy()
    values = rows[FIXED_VETO_FEATURE].replace([np.inf, -np.inf], np.nan)
    return rows.loc[values.notna() & (values >= FIXED_VETO_MIN)].copy()


def aggregate_prospective(
    prospective_results: list[dict[str, Any]],
    *,
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> dict[str, Any]:
    executable_count = sum(
        int(result["one_position_summary"]["executable_count"])
        for result in prospective_results
    )
    selected_signal_count = sum(int(result["selected_signal_count"]) for result in prospective_results)
    gross_total = sum(
        float(result["one_position_summary"]["gross_executable_total_bps"])
        for result in prospective_results
    )
    net_total_by_cost = {}
    net_mean_by_cost = {}
    positive_window_count_by_cost = {}
    min_window_net_by_cost = {}
    for cost_bps in cost_levels_bps:
        key = float_key(cost_bps)
        window_net_values = [
            net_total_at_cost(result["one_position_summary"], cost_bps)
            for result in prospective_results
        ]
        net_total = gross_total - executable_count * cost_bps
        net_total_by_cost[key] = float(net_total)
        net_mean_by_cost[key] = float(net_total / executable_count) if executable_count else 0.0
        positive_window_count_by_cost[key] = int(sum(value > 0 for value in window_net_values))
        min_window_net_by_cost[key] = float(min(window_net_values)) if window_net_values else 0.0
    eligible_window_count = sum(
        int(result["one_position_summary"]["executable_count"]) > 0
        for result in prospective_results
    )
    sufficient_window_count = sum(
        int(result["one_position_summary"]["executable_count"]) >= WINDOW_FAIL_MINIMUM_TRADES
        for result in prospective_results
    )
    return {
        "selected_signal_count": int(selected_signal_count),
        "executable_count": int(executable_count),
        "eligible_window_count": int(eligible_window_count),
        "sufficient_window_count": int(sufficient_window_count),
        "gross_total_bps": float(gross_total),
        "gross_mean_bps": float(gross_total / executable_count) if executable_count else 0.0,
        "net_total_by_cost_bps": net_total_by_cost,
        "net_mean_by_cost_bps": net_mean_by_cost,
        "positive_window_count_by_cost_bps": positive_window_count_by_cost,
        "min_window_net_by_cost_bps": min_window_net_by_cost,
        "profit_cost_bps": float(profit_cost_bps),
        "stress_cost_bps": float(stress_cost_bps),
    }


def evaluate_gates(
    *,
    aggregate: dict[str, Any],
    prospective_results: list[dict[str, Any]],
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> dict[str, Any]:
    profit_key = float_key(profit_cost_bps)
    stress_key = float_key(stress_cost_bps)
    stress_total = float(aggregate["net_total_by_cost_bps"][stress_key])
    profit_mean = float(aggregate["net_mean_by_cost_bps"][profit_key])
    stress_mean = float(aggregate["net_mean_by_cost_bps"][stress_key])
    executable_count = int(aggregate["executable_count"])
    eligible_window_count = int(aggregate["eligible_window_count"])
    positive_windows = int(aggregate["positive_window_count_by_cost_bps"][stress_key])
    positive_window_fraction = (
        float(positive_windows / eligible_window_count)
        if eligible_window_count
        else 0.0
    )
    window_failures = [
        {
            "label": result["label"],
            "executable_count": int(result["one_position_summary"]["executable_count"]),
            "stress_total_bps": net_total_at_cost(
                result["one_position_summary"],
                stress_cost_bps,
            ),
        }
        for result in prospective_results
        if int(result["one_position_summary"]["executable_count"]) >= WINDOW_FAIL_MINIMUM_TRADES
        and net_total_at_cost(result["one_position_summary"], stress_cost_bps)
        <= WINDOW_FAIL_STRESS_TOTAL_BPS
    ]
    early_total_fail = (
        executable_count >= EARLY_FAIL_MINIMUM_TRADES
        and stress_total <= EARLY_FAIL_STRESS_TOTAL_BPS
    )
    early_window_fail = bool(window_failures)
    review_gate = {
        "minimum_trades": executable_count >= MINIMUM_PROSPECTIVE_TRADES_FOR_REVIEW,
        "minimum_windows": eligible_window_count >= MINIMUM_PROSPECTIVE_WINDOWS_FOR_REVIEW,
        "stress_mean": stress_mean > MINIMUM_STRESS_MEAN_BPS_FOR_REVIEW,
        "profit_mean": profit_mean > MINIMUM_PROFIT_MEAN_BPS_FOR_REVIEW,
        "stress_positive_window_fraction": (
            positive_window_fraction >= MINIMUM_STRESS_POSITIVE_WINDOW_FRACTION
        ),
        "no_early_total_fail": not early_total_fail,
        "no_early_window_fail": not early_window_fail,
    }
    return {
        "early_total_fail": bool(early_total_fail),
        "early_window_fail": bool(early_window_fail),
        "window_failures": window_failures,
        "candidate_for_shadow_offline_review": all(review_gate.values()),
        "review_gate": review_gate,
        "metrics": {
            "executable_count": executable_count,
            "eligible_window_count": eligible_window_count,
            "stress_total_bps": stress_total,
            "profit_mean_bps": profit_mean,
            "stress_mean_bps": stress_mean,
            "stress_positive_window_fraction": positive_window_fraction,
        },
        "thresholds": {
            "minimum_trades": MINIMUM_PROSPECTIVE_TRADES_FOR_REVIEW,
            "minimum_windows": MINIMUM_PROSPECTIVE_WINDOWS_FOR_REVIEW,
            "minimum_stress_mean_bps": MINIMUM_STRESS_MEAN_BPS_FOR_REVIEW,
            "minimum_profit_mean_bps": MINIMUM_PROFIT_MEAN_BPS_FOR_REVIEW,
            "minimum_stress_positive_window_fraction": MINIMUM_STRESS_POSITIVE_WINDOW_FRACTION,
            "early_fail_minimum_trades": EARLY_FAIL_MINIMUM_TRADES,
            "early_fail_stress_total_bps": EARLY_FAIL_STRESS_TOTAL_BPS,
            "window_fail_minimum_trades": WINDOW_FAIL_MINIMUM_TRADES,
            "window_fail_stress_total_bps": WINDOW_FAIL_STRESS_TOTAL_BPS,
        },
    }


def fast105_status(gates: dict[str, Any]) -> tuple[str, list[str]]:
    base_reasons = [
        "not_approved_strategy",
        "research_only",
        "prospective_observation_only",
        "testnet_requires_new_preregistration_and_explicit_authorization",
    ]
    if gates["early_total_fail"] or gates["early_window_fail"]:
        return "FAST105_PROSPECTIVE_EARLY_FAIL", base_reasons + [
            "early_fail_gate_triggered",
            "do_not_promote_to_strategy",
        ]
    if gates["candidate_for_shadow_offline_review"]:
        return "FAST105_SHADOW_OFFLINE_REVIEW_CANDIDATE_NOT_TESTNET", base_reasons + [
            "minimum_prospective_gates_passed",
            "shadow_offline_review_required",
            "do_not_promote_to_testnet",
        ]
    missing = [
        key
        for key, passed in gates["review_gate"].items()
        if not passed and key.startswith("minimum_")
    ]
    if missing:
        return "FAST105_INSUFFICIENT_PROSPECTIVE_EVIDENCE", base_reasons + [
            *[f"gate_not_met:{key}" for key in missing],
            "keep_collecting_or_stop_if_timeout",
        ]
    return "FAST105_PROSPECTIVE_FAILED_GATES", base_reasons + [
        *[
            f"gate_not_met:{key}"
            for key, passed in gates["review_gate"].items()
            if not passed
        ],
        "do_not_promote_to_strategy",
    ]


def base_report_payload(
    *,
    training_wal_root_refs: tuple[WalRootRef, ...],
    validation_wal_root_refs: tuple[WalRootRef, ...],
    preregistration_path: Path,
    registry_path: Path,
    preregistration: dict[str, str],
    model_name: str,
    horizon_seconds: int,
    target_bps: float,
    trailing_distance_bps: float,
    top_fraction: float,
    decision_stride_seconds: int,
    max_book_staleness_ms: int,
    latency_ms: int,
    initial_stop_bps: float,
    minimum_training_windows: int,
    profit_cost_bps: float,
    stress_cost_bps: float,
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
            "fast105_is_preregistered_prospective_observation_only",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_preregistration_and_explicit_authorization",
        ],
        "experiment_stage": "fast105_fixed_veto_prospective_observation",
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "source_fast104_report_sha256": SOURCE_FAST104_REPORT_SHA256,
        "training_wal_roots": [
            {"label": ref.label, "path": str(ref.path)} for ref in training_wal_root_refs
        ],
        "validation_wal_roots": [
            {"label": ref.label, "path": str(ref.path)} for ref in validation_wal_root_refs
        ],
        "config": {
            "model": model_name,
            "horizon_seconds": horizon_seconds,
            "target_bps": target_bps,
            "trailing_distance_bps": trailing_distance_bps,
            "top_fraction": top_fraction,
            "decision_stride_seconds": decision_stride_seconds,
            "max_book_staleness_ms": max_book_staleness_ms,
            "latency_ms": latency_ms,
            "initial_stop_bps": initial_stop_bps,
            "minimum_training_windows": minimum_training_windows,
            "profit_cost_bps": profit_cost_bps,
            "stress_cost_bps": stress_cost_bps,
            "fixed_veto": {
                "feature_name": FIXED_VETO_FEATURE,
                "direction": "gte",
                "threshold": FIXED_VETO_MIN,
            },
        },
    }


def write_result_artifacts(
    output_root: Path,
    scored_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    one_position_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scored_path = output_root / "scored_decisions.jsonl"
    selected_path = output_root / "selected_after_fixed_veto.jsonl"
    one_position_path = output_root / "one_position_settlements.jsonl"
    _write_jsonl(scored_path, scored_rows)
    _write_jsonl(selected_path, selected_rows)
    _write_jsonl(one_position_path, one_position_rows)
    return {
        "scored_decisions_path": str(scored_path),
        "scored_decisions_sha256": sha256_file(scored_path),
        "selected_after_fixed_veto_path": str(selected_path),
        "selected_after_fixed_veto_sha256": sha256_file(selected_path),
        "one_position_settlements_path": str(one_position_path),
        "one_position_settlements_sha256": sha256_file(one_position_path),
    }


def jsonable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def empty_aggregate(cost_levels_bps: tuple[float, ...]) -> dict[str, Any]:
    zero_by_cost = {float_key(cost_bps): 0.0 for cost_bps in cost_levels_bps}
    zero_counts = {float_key(cost_bps): 0 for cost_bps in cost_levels_bps}
    return {
        "selected_signal_count": 0,
        "executable_count": 0,
        "eligible_window_count": 0,
        "sufficient_window_count": 0,
        "gross_total_bps": 0.0,
        "gross_mean_bps": 0.0,
        "net_total_by_cost_bps": zero_by_cost,
        "net_mean_by_cost_bps": zero_by_cost,
        "positive_window_count_by_cost_bps": zero_counts,
        "min_window_net_by_cost_bps": zero_by_cost,
    }


def empty_gates() -> dict[str, Any]:
    return {
        "early_total_fail": False,
        "early_window_fail": False,
        "window_failures": [],
        "candidate_for_shadow_offline_review": False,
        "review_gate": {},
        "metrics": {},
        "thresholds": {},
    }


def write_fast105_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    payload = {
        **report,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preregistration_timestamp_ms() -> int:
    return int(PREREGISTRATION_TIMESTAMP.timestamp() * 1_000)


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "aggregate": report.get("aggregate"),
        "gates": report.get("gates"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
