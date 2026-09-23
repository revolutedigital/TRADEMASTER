"""Fast104 fixed outlier-veto audit for the large-move candidate.

Fast101 showed the large-move candidate is positive in most windows, but fast24
and fast36 can dominate the PnL. Fast102/Fast103 tried adaptive single-feature
gates and failed. Fast104 tests a narrower diagnostic question: do fixed causal
"outlier stress" vetoes, with thresholds computed from training windows only,
remove enough tail risk to justify a fresh pre-registration?

This is research-only discovery. It never approves execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
        "fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m",
        "fast83=backend/data/microstructure_v1/prospective-wal-fast83-30m",
        "fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m",
    ]
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast104-fixed-outlier-veto-audit"
)
DEFAULT_COST_LEVELS_BPS = (0.0, 0.25, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class FixedGateComponent:
    feature_name: str
    direction: str
    quantile: float

    @property
    def name(self) -> str:
        return f"{self.feature_name}_{self.direction}_q{self.quantile:g}"


@dataclass(frozen=True)
class FixedGatePolicy:
    name: str
    components: tuple[FixedGateComponent, ...]


@dataclass(frozen=True)
class MaterializedGateComponent:
    feature_name: str
    direction: str
    quantile: float
    threshold: float


DEFAULT_POLICIES = (
    FixedGatePolicy("no_gate", ()),
    FixedGatePolicy(
        "spread_widening_5s_low80",
        (FixedGateComponent("book_spread_widening_bps_5s", "lte", 0.80),),
    ),
    FixedGatePolicy(
        "realized_vol_30s_low80",
        (FixedGateComponent("realized_vol_30s_bps", "lte", 0.80),),
    ),
    FixedGatePolicy(
        "quote_volume_1s_low80",
        (FixedGateComponent("quote_volume_1s", "lte", 0.80),),
    ),
    FixedGatePolicy(
        "spot_quote_volume_1s_low80",
        (FixedGateComponent("spot_quote_volume_1s", "lte", 0.80),),
    ),
    FixedGatePolicy(
        "directed_spot_return_5s_not_crash30",
        (FixedGateComponent("directed_spot_return_5s_bps", "gte", 0.30),),
    ),
    FixedGatePolicy(
        "spot_return_1s_not_crash30",
        (FixedGateComponent("spot_return_1s_bps", "gte", 0.30),),
    ),
    FixedGatePolicy(
        "spread_low_and_spot_return_not_crash",
        (
            FixedGateComponent("book_spread_widening_bps_5s", "lte", 0.80),
            FixedGateComponent("directed_spot_return_5s_bps", "gte", 0.30),
        ),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wal-roots", default=DEFAULT_WAL_ROOTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    parser.add_argument("--minimum-windows", type=int, default=11)
    parser.add_argument("--minimum-executable-per-window", type=int, default=3)
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    arguments = parser.parse_args()

    report = run_fast104_fixed_outlier_veto_audit(
        wal_root_refs=_parse_wal_root_refs(arguments.wal_roots),
        output_root=arguments.output_root,
        model_name=arguments.model,
        horizon_seconds=arguments.horizon_seconds,
        target_bps=arguments.target_bps,
        trailing_distance_bps=arguments.trailing_distance_bps,
        top_fraction=arguments.top_fraction,
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        latency_ms=arguments.latency_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        minimum_windows=arguments.minimum_windows,
        minimum_executable_per_window=arguments.minimum_executable_per_window,
        profit_cost_bps=arguments.profit_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast104_fixed_outlier_veto_audit(
    *,
    wal_root_refs: tuple[WalRootRef, ...],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
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
    minimum_windows: int = 11,
    minimum_executable_per_window: int = 3,
    profit_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    policies: tuple[FixedGatePolicy, ...] = DEFAULT_POLICIES,
) -> dict[str, Any]:
    validate_parameters(
        wal_root_refs=wal_root_refs,
        model_name=model_name,
        horizon_seconds=horizon_seconds,
        target_bps=target_bps,
        trailing_distance_bps=trailing_distance_bps,
        top_fraction=top_fraction,
        minimum_windows=minimum_windows,
        minimum_executable_per_window=minimum_executable_per_window,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        cost_levels_bps=cost_levels_bps,
    )
    base_report: dict[str, Any] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast104_is_fixed_outlier_veto_discovery_diagnostic",
            "gate_family_chosen_after_fast101_outlier_profile",
            "thresholds_computed_from_training_windows_only",
            "positive_result_requires_fresh_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast104_fixed_outlier_veto_audit",
        "leakage_guard": {
            "outer_validation": "leave_one_window_out",
            "threshold_fit": "training_selected_rows_only",
            "holdout_never_used_for_threshold_or_model_fit": True,
            "gate_family_is_discovery_not_approval": True,
        },
        "wal_roots": [
            {"label": ref.label, "path": str(ref.path)} for ref in wal_root_refs
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
            "minimum_windows": minimum_windows,
            "minimum_executable_per_window": minimum_executable_per_window,
            "profit_cost_bps": profit_cost_bps,
            "stress_cost_bps": stress_cost_bps,
            "policies": [policy_to_dict(policy) for policy in policies],
        },
    }
    if len(wal_root_refs) < minimum_windows:
        return write_report(
            output_root,
            {
                **base_report,
                "status": "FAST104_INSUFFICIENT_WINDOWS",
                "status_reasons": [
                    f"window_count_below_minimum:{len(wal_root_refs)}<{minimum_windows}"
                ],
                "modeling_ready": False,
                "policy_results": [],
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
        else:
            loaded_windows.append(loaded)
    if len(loaded_windows) < minimum_windows:
        return write_report(
            output_root,
            {
                **base_report,
                "status": "FAST104_LOADED_WINDOW_COUNT_BELOW_MINIMUM",
                "status_reasons": [
                    f"loaded_window_count_below_minimum:{len(loaded_windows)}<{minimum_windows}"
                ],
                "modeling_ready": False,
                "policy_results": [],
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
    policy_holdouts: dict[str, list[dict[str, Any]]] = {policy.name: [] for policy in policies}
    for holdout_label in sorted(labelled_windows):
        holdout_results = evaluate_holdout(
            labelled_windows=labelled_windows,
            holdout_label=holdout_label,
            model_name=model_name,
            top_fraction=top_fraction,
            policies=policies,
            cost_levels_bps=cost_levels_bps,
        )
        for policy_name, result in holdout_results.items():
            policy_holdouts[policy_name].append(result)

    policy_results = [
        finalize_policy_result(
            policy=policy,
            holdouts=policy_holdouts[policy.name],
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
            minimum_executable_per_window=minimum_executable_per_window,
        )
        for policy in policies
    ]
    best_policy = max(
        policy_results,
        key=lambda result: (
            result["aggregate"]["positive_window_count_by_cost_bps"][
                float_key(profit_cost_bps)
            ],
            result["aggregate"]["minimum_execution_window_count"],
            result["aggregate"]["net_total_by_cost_bps"][float_key(profit_cost_bps)],
        ),
    )
    status, status_reasons = fast104_status(
        best_policy=best_policy,
        holdout_count=len(labelled_windows),
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    holdout_path = output_root / "policy_holdouts.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        holdout_path,
        [
            {"policy": policy_name, **holdout}
            for policy_name, holdouts in policy_holdouts.items()
            for holdout in holdouts
        ],
    )
    return write_report(
        output_root,
        {
            **base_report,
            "status": status,
            "status_reasons": status_reasons,
            "modeling_ready": True,
            "window_reports": [window_report(window) for window in loaded_windows],
            "policy_results": policy_results,
            "best_policy": best_policy,
            "policy_holdouts_path": str(holdout_path),
            "policy_holdouts_sha256": sha256_file(holdout_path),
            "skipped": skipped,
        },
    )


def evaluate_holdout(
    *,
    labelled_windows: dict[str, pd.DataFrame],
    holdout_label: str,
    model_name: str,
    top_fraction: float,
    policies: tuple[FixedGatePolicy, ...],
    cost_levels_bps: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    combined = pd.concat(labelled_windows.values(), ignore_index=True)
    training_frame = combined[combined["window_label"] != holdout_label].copy()
    holdout_frame = combined[combined["window_label"] == holdout_label].copy()
    feature_columns = select_payoff_feature_columns(training_frame)
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
    scored_training = training_frame.copy()
    scored_training["model_score"] = model.predict(prepared_training)
    selected_training = select_top_fraction_by_label(scored_training, top_fraction)
    scored_holdout = holdout_frame.copy()
    scored_holdout["model_score"] = model.predict(prepared_holdout)
    selected_holdout = _select_top_fraction(
        scored_holdout,
        scored_holdout["model_score"].to_numpy(dtype=np.float64),
        top_fraction,
    )
    results = {}
    for policy in policies:
        materialized = materialize_policy(policy, selected_training)
        gated = apply_materialized_policy(selected_holdout, materialized)
        one_position = _apply_one_position_constraint_by_score(gated)
        results[policy.name] = {
            "holdout_label": holdout_label,
            "training_selected_signal_count": int(len(selected_training)),
            "pre_gate_selected_signal_count": int(len(selected_holdout)),
            "selected_signal_count": int(len(gated)),
            "materialized_policy": [asdict(component) for component in materialized],
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
    return results


def select_top_fraction_by_label(rows: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    frames = []
    for _label, group in rows.groupby("window_label", sort=True):
        selected_count = max(1, int(np.ceil(len(group) * top_fraction)))
        frames.append(
            group.sort_values(
                ["model_score", "decision_time_ms"],
                ascending=[False, True],
                kind="stable",
            ).head(selected_count)
        )
    if not frames:
        return rows.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def materialize_policy(
    policy: FixedGatePolicy,
    selected_training: pd.DataFrame,
) -> tuple[MaterializedGateComponent, ...]:
    components = []
    for component in policy.components:
        if component.feature_name not in selected_training.columns:
            threshold = float("nan")
        else:
            values = (
                selected_training[component.feature_name]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            threshold = float(values.quantile(component.quantile)) if len(values) else float("nan")
        components.append(
            MaterializedGateComponent(
                feature_name=component.feature_name,
                direction=component.direction,
                quantile=component.quantile,
                threshold=threshold,
            )
        )
    return tuple(components)


def apply_materialized_policy(
    rows: pd.DataFrame,
    components: tuple[MaterializedGateComponent, ...],
) -> pd.DataFrame:
    if not components:
        return rows.copy()
    mask = pd.Series(True, index=rows.index)
    for component in components:
        if component.feature_name not in rows.columns or not np.isfinite(component.threshold):
            mask &= False
            continue
        values = rows[component.feature_name].replace([np.inf, -np.inf], np.nan)
        if component.direction == "lte":
            mask &= values.notna() & (values <= component.threshold)
        elif component.direction == "gte":
            mask &= values.notna() & (values >= component.threshold)
        else:
            raise ValueError(f"unsupported direction: {component.direction}")
    return rows.loc[mask].copy()


def finalize_policy_result(
    *,
    policy: FixedGatePolicy,
    holdouts: list[dict[str, Any]],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_per_window: int,
) -> dict[str, Any]:
    aggregate = aggregate_holdouts(
        holdouts,
        cost_levels_bps=DEFAULT_COST_LEVELS_BPS,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_executable_per_window=minimum_executable_per_window,
    )
    return {
        "policy": policy_to_dict(policy),
        "aggregate": aggregate,
        "holdouts": holdouts,
    }


def aggregate_holdouts(
    holdouts: list[dict[str, Any]],
    *,
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_per_window: int,
) -> dict[str, Any]:
    executable_count = sum(
        int(holdout["one_position_summary"]["executable_count"]) for holdout in holdouts
    )
    gross_total = sum(
        float(holdout["one_position_summary"]["gross_executable_total_bps"])
        for holdout in holdouts
    )
    selected_signal_count = sum(int(holdout["selected_signal_count"]) for holdout in holdouts)
    net_total_by_cost = {}
    net_mean_by_cost = {}
    positive_window_count_by_cost = {}
    min_window_net_by_cost = {}
    for cost_bps in cost_levels_bps:
        key = float_key(cost_bps)
        window_net_values = [
            net_total_at_cost(holdout["one_position_summary"], cost_bps)
            for holdout in holdouts
        ]
        net_total = gross_total - executable_count * cost_bps
        net_total_by_cost[key] = float(net_total)
        net_mean_by_cost[key] = float(net_total / executable_count) if executable_count else 0.0
        positive_window_count_by_cost[key] = sum(value > 0 for value in window_net_values)
        min_window_net_by_cost[key] = float(min(window_net_values)) if window_net_values else 0.0
    minimum_execution_window_count = sum(
        int(holdout["one_position_summary"]["executable_count"])
        >= minimum_executable_per_window
        for holdout in holdouts
    )
    return {
        "selected_signal_count": int(selected_signal_count),
        "executable_count": int(executable_count),
        "gross_total_bps": float(gross_total),
        "gross_mean_bps": float(gross_total / executable_count) if executable_count else 0.0,
        "net_total_by_cost_bps": net_total_by_cost,
        "net_mean_by_cost_bps": net_mean_by_cost,
        "positive_window_count_by_cost_bps": positive_window_count_by_cost,
        "min_window_net_by_cost_bps": min_window_net_by_cost,
        "minimum_execution_window_count": int(minimum_execution_window_count),
        "all_windows_positive_at_profit_cost": positive_window_count_by_cost[
            float_key(profit_cost_bps)
        ]
        == len(holdouts),
        "all_windows_positive_at_stress_cost": positive_window_count_by_cost[
            float_key(stress_cost_bps)
        ]
        == len(holdouts),
        "all_windows_meet_minimum_executions": minimum_execution_window_count
        == len(holdouts),
    }


def fast104_status(
    *,
    best_policy: dict[str, Any],
    holdout_count: int,
    profit_cost_bps: float,
    stress_cost_bps: float,
) -> tuple[str, list[str]]:
    aggregate = best_policy["aggregate"]
    reasons = [
        "not_approved_strategy",
        "research_only",
        "fixed_outlier_veto_discovery_diagnostic",
        "positive_result_requires_fresh_preregistration",
    ]
    if aggregate["all_windows_positive_at_stress_cost"]:
        return "FAST104_FIXED_OUTLIER_VETO_STRESS_SURVIVOR_DIAGNOSTIC_ONLY", reasons + [
            f"best_policy:{best_policy['policy']['name']}",
            f"all_holdouts_positive_at_{stress_cost_bps:g}bps",
        ]
    if aggregate["all_windows_positive_at_profit_cost"]:
        return "FAST104_FIXED_OUTLIER_VETO_PROFIT_SURVIVOR_STRESS_FRAGILE", reasons + [
            f"best_policy:{best_policy['policy']['name']}",
            f"all_holdouts_positive_at_{profit_cost_bps:g}bps",
            f"not_all_holdouts_positive_at_{stress_cost_bps:g}bps",
        ]
    return "FAST104_FIXED_OUTLIER_VETO_FAILED_REPLICATION", reasons + [
        f"best_policy:{best_policy['policy']['name']}",
        (
            "profit_positive_holdouts:"
            f"{aggregate['positive_window_count_by_cost_bps'][float_key(profit_cost_bps)]}"
            f"/{holdout_count}"
        ),
        (
            "stress_positive_holdouts:"
            f"{aggregate['positive_window_count_by_cost_bps'][float_key(stress_cost_bps)]}"
            f"/{holdout_count}"
        ),
        "do_not_promote_to_strategy",
    ]


def validate_parameters(
    *,
    wal_root_refs: tuple[WalRootRef, ...],
    model_name: str,
    horizon_seconds: int,
    target_bps: float,
    trailing_distance_bps: float,
    top_fraction: float,
    minimum_windows: int,
    minimum_executable_per_window: int,
    profit_cost_bps: float,
    stress_cost_bps: float,
    cost_levels_bps: tuple[float, ...],
) -> None:
    if len({ref.label for ref in wal_root_refs}) != len(wal_root_refs):
        raise ValueError("wal root labels must be unique")
    if model_name not in {"hist_gradient_payoff", "ridge_payoff"}:
        raise ValueError("unsupported model")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if target_bps <= 0 or trailing_distance_bps <= 0:
        raise ValueError("target and trailing distance must be positive")
    if not 0 < top_fraction < 1:
        raise ValueError("top_fraction must be between 0 and 1")
    if minimum_windows < 2:
        raise ValueError("minimum_windows must be at least 2")
    if minimum_executable_per_window < 1:
        raise ValueError("minimum_executable_per_window must be positive")
    if profit_cost_bps not in cost_levels_bps or stress_cost_bps not in cost_levels_bps:
        raise ValueError("profit and stress costs must be included in cost_levels_bps")


def make_model(model_name: str) -> Any:
    if model_name == "hist_gradient_payoff":
        return _make_hist_gradient_regressor()
    if model_name == "ridge_payoff":
        return _make_ridge_regressor()
    raise ValueError(f"unsupported model: {model_name}")


def net_total_at_cost(summary: dict[str, Any], cost_bps: float) -> float:
    for row in summary["cost_sensitivity"]:
        if float(row["explicit_cost_bps"]) == float(cost_bps):
            return float(row["net_executable_total_bps"])
    raise ValueError(f"cost level missing from summary: {cost_bps:g}")


def policy_to_dict(policy: FixedGatePolicy) -> dict[str, Any]:
    return {
        "name": policy.name,
        "components": [asdict(component) for component in policy.components],
    }


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


def float_key(value: float) -> str:
    return f"{value:g}"


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


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "best_policy": {
            "policy": report.get("best_policy", {}).get("policy"),
            "aggregate": report.get("best_policy", {}).get("aggregate"),
        },
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
