"""Fast138: forward audit for the Fast104 large-move fixed-veto family.

Fast104 found a promising fixed outlier veto for the large-move microstructure
candidate, but it was still diagnostic. Fast138 adds chronological pressure:
select the fixed-veto policy only on older windows (fast16..fast36), then audit
the selected policy on later windows (fast82, fast83, fast89). This remains
historical research only and never touches trading engine, credentials, ledger,
Demo, Testnet or LIVE accounts.
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
    DEFAULT_POLICIES,
    DEFAULT_WAL_ROOTS,
    FixedGatePolicy,
    aggregate_holdouts,
    apply_materialized_policy,
    evaluate_holdout,
    float_key,
    make_model,
    materialize_policy,
    net_total_at_cost,
    policy_to_dict,
    select_top_fraction_by_label,
    sha256_file,
    validate_parameters,
    window_report,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast138-large-move-fixed-veto-forward-audit"
)
TRAIN_LABELS = ("fast16", "fast20", "fast21", "fast22", "fast24", "fast28", "fast29", "fast32", "fast35", "fast36")
AUDIT_LABELS = ("fast82", "fast83", "fast89")
SAFETY_FLAGS = {
    "research_only": True,
    "order_submission_allowed": False,
    "execution_authorization": "none",
    "orders_sent": False,
    "committed": False,
    "dry_run": True,
    "approved_strategy": False,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wal-roots", default=DEFAULT_WAL_ROOTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", choices=("hist_gradient_payoff", "ridge_payoff"), default="hist_gradient_payoff")
    parser.add_argument("--horizon-seconds", type=int, default=300)
    parser.add_argument("--target-bps", type=float, default=10.0)
    parser.add_argument("--trailing-distance-bps", type=float, default=15.0)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--minimum-train-positive-windows", type=int, default=8)
    parser.add_argument("--minimum-executable-per-window", type=int, default=3)
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    arguments = parser.parse_args(argv)
    report = run(
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
        minimum_train_positive_windows=arguments.minimum_train_positive_windows,
        minimum_executable_per_window=arguments.minimum_executable_per_window,
        profit_cost_bps=arguments.profit_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
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
    minimum_train_positive_windows: int = 8,
    minimum_executable_per_window: int = 3,
    profit_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    policies: tuple[FixedGatePolicy, ...] = DEFAULT_POLICIES,
    train_labels: tuple[str, ...] = TRAIN_LABELS,
    audit_labels: tuple[str, ...] = AUDIT_LABELS,
) -> dict[str, Any]:
    validate_parameters(
        wal_root_refs=wal_root_refs,
        model_name=model_name,
        horizon_seconds=horizon_seconds,
        target_bps=target_bps,
        trailing_distance_bps=trailing_distance_bps,
        top_fraction=top_fraction,
        minimum_windows=len(train_labels),
        minimum_executable_per_window=minimum_executable_per_window,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        cost_levels_bps=cost_levels_bps,
    )
    base_report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast138_is_forward_chronological_audit_not_strategy_approval",
            "fixed_veto_family_was_inspired_by_prior_diagnostics",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast138_large_move_fixed_veto_forward_audit",
        "leakage_guard": {
            "policy_selection_windows": list(train_labels),
            "audit_windows": list(audit_labels),
            "policy_choice_uses_audit_windows": False,
            "final_model_fit_windows": list(train_labels),
            "audit_windows_never_used_for_model_or_threshold_fit": True,
        },
        "wal_roots": [{"label": ref.label, "path": str(ref.path)} for ref in wal_root_refs],
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
            "minimum_train_positive_windows": minimum_train_positive_windows,
            "minimum_executable_per_window": minimum_executable_per_window,
            "profit_cost_bps": profit_cost_bps,
            "stress_cost_bps": stress_cost_bps,
            "policies": [policy_to_dict(policy) for policy in policies],
        },
    }
    refs_by_label = {ref.label: ref for ref in wal_root_refs}
    missing_labels = [label for label in (*train_labels, *audit_labels) if label not in refs_by_label]
    if missing_labels:
        return write_report(
            output_root,
            {
                **base_report,
                "status": "FAST138_MISSING_REQUIRED_WINDOWS",
                "status_reasons": [f"missing_windows:{','.join(missing_labels)}"],
                "modeling_ready": False,
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
    for label in (*train_labels, *audit_labels):
        loaded = _load_window(refs_by_label[label], config=config)
        if isinstance(loaded, dict):
            skipped.append({"window": label, **loaded})
        else:
            loaded_windows.append(loaded)
    loaded_labels = {window.label for window in loaded_windows}
    missing_loaded = [label for label in (*train_labels, *audit_labels) if label not in loaded_labels]
    if missing_loaded:
        return write_report(
            output_root,
            {
                **base_report,
                "status": "FAST138_LOADED_WINDOW_COUNT_BELOW_MINIMUM",
                "status_reasons": [f"loaded_windows_missing:{','.join(missing_loaded)}"],
                "modeling_ready": False,
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
    train_labelled_windows = {label: labelled_windows[label] for label in train_labels}
    train_policy_results = select_policy_on_train_windows(
        labelled_windows=train_labelled_windows,
        model_name=model_name,
        top_fraction=top_fraction,
        policies=policies,
        cost_levels_bps=cost_levels_bps,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_executable_per_window=minimum_executable_per_window,
    )
    best_train_policy = choose_best_train_policy(train_policy_results, stress_cost_bps, profit_cost_bps)
    audit_holdouts = audit_policy_on_later_windows(
        labelled_windows=labelled_windows,
        train_labels=train_labels,
        audit_labels=audit_labels,
        model_name=model_name,
        top_fraction=top_fraction,
        policy=policy_from_results(best_train_policy, policies),
        cost_levels_bps=cost_levels_bps,
    )
    audit_aggregate = aggregate_holdouts(
        audit_holdouts,
        cost_levels_bps=cost_levels_bps,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        minimum_executable_per_window=minimum_executable_per_window,
    )
    status, status_reasons = fast138_status(
        best_train_policy=best_train_policy,
        audit_aggregate=audit_aggregate,
        audit_holdout_count=len(audit_holdouts),
        minimum_train_positive_windows=minimum_train_positive_windows,
        stress_cost_bps=stress_cost_bps,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    train_holdouts_path = output_root / "train-policy-holdouts.jsonl"
    audit_holdouts_path = output_root / "audit-holdouts.jsonl"
    _write_jsonl(
        train_holdouts_path,
        [
            {"policy": result["policy"]["name"], **holdout}
            for result in train_policy_results
            for holdout in result["holdouts"]
        ],
    )
    _write_jsonl(audit_holdouts_path, audit_holdouts)
    return write_report(
        output_root,
        {
            **base_report,
            "status": status,
            "status_reasons": status_reasons,
            "modeling_ready": True,
            "window_reports": [window_report(window) for window in loaded_windows],
            "train_policy_results": train_policy_results,
            "best_train_policy": best_train_policy,
            "audit_holdouts": audit_holdouts,
            "audit_aggregate": audit_aggregate,
            "train_policy_holdouts_path": str(train_holdouts_path),
            "train_policy_holdouts_sha256": sha256_file(train_holdouts_path),
            "audit_holdouts_path": str(audit_holdouts_path),
            "audit_holdouts_sha256": sha256_file(audit_holdouts_path),
            "skipped": skipped,
        },
    )


def select_policy_on_train_windows(
    *,
    labelled_windows: dict[str, pd.DataFrame],
    model_name: str,
    top_fraction: float,
    policies: tuple[FixedGatePolicy, ...],
    cost_levels_bps: tuple[float, ...],
    profit_cost_bps: float,
    stress_cost_bps: float,
    minimum_executable_per_window: int,
) -> list[dict[str, Any]]:
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
    return [
        {
            "policy": policy_to_dict(policy),
            "aggregate": aggregate_holdouts(
                policy_holdouts[policy.name],
                cost_levels_bps=cost_levels_bps,
                profit_cost_bps=profit_cost_bps,
                stress_cost_bps=stress_cost_bps,
                minimum_executable_per_window=minimum_executable_per_window,
            ),
            "holdouts": policy_holdouts[policy.name],
        }
        for policy in policies
    ]


def choose_best_train_policy(
    train_policy_results: list[dict[str, Any]],
    stress_cost_bps: float,
    profit_cost_bps: float,
) -> dict[str, Any]:
    stress_key = float_key(stress_cost_bps)
    profit_key = float_key(profit_cost_bps)
    return max(
        train_policy_results,
        key=lambda result: (
            result["aggregate"]["positive_window_count_by_cost_bps"][stress_key],
            result["aggregate"]["minimum_execution_window_count"],
            result["aggregate"]["net_total_by_cost_bps"][stress_key],
            result["aggregate"]["positive_window_count_by_cost_bps"][profit_key],
            result["aggregate"]["net_total_by_cost_bps"][profit_key],
        ),
    )


def policy_from_results(result: dict[str, Any], policies: tuple[FixedGatePolicy, ...]) -> FixedGatePolicy:
    name = result["policy"]["name"]
    for policy in policies:
        if policy.name == name:
            return policy
    raise ValueError(f"policy not found: {name}")


def audit_policy_on_later_windows(
    *,
    labelled_windows: dict[str, pd.DataFrame],
    train_labels: tuple[str, ...],
    audit_labels: tuple[str, ...],
    model_name: str,
    top_fraction: float,
    policy: FixedGatePolicy,
    cost_levels_bps: tuple[float, ...],
) -> list[dict[str, Any]]:
    training_frame = pd.concat([labelled_windows[label] for label in train_labels], ignore_index=True)
    audit_frame = pd.concat([labelled_windows[label] for label in audit_labels], ignore_index=True)
    feature_columns = select_payoff_feature_columns(training_frame)
    prepared_training, prepared_audit, _medians = _prepare_feature_frames(
        training_frame,
        audit_frame,
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
    materialized = materialize_policy(policy, selected_training)
    scored_audit = audit_frame.copy()
    scored_audit["model_score"] = model.predict(prepared_audit)
    selected_audit = _select_top_fraction(
        scored_audit,
        scored_audit["model_score"].to_numpy(dtype=np.float64),
        top_fraction,
    )
    holdouts: list[dict[str, Any]] = []
    for label in audit_labels:
        selected_label = selected_audit.loc[selected_audit["window_label"] == label].copy()
        gated = apply_materialized_policy(selected_label, materialized)
        one_position = _apply_one_position_constraint_by_score(gated)
        holdouts.append(
            {
                "holdout_label": label,
                "training_selected_signal_count": int(len(selected_training)),
                "pre_gate_selected_signal_count": int(len(selected_label)),
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
        )
    return holdouts


def fast138_status(
    *,
    best_train_policy: dict[str, Any],
    audit_aggregate: dict[str, Any],
    audit_holdout_count: int,
    minimum_train_positive_windows: int,
    stress_cost_bps: float,
) -> tuple[str, list[str]]:
    stress_key = float_key(stress_cost_bps)
    train_stress_positive = best_train_policy["aggregate"]["positive_window_count_by_cost_bps"][stress_key]
    audit_stress_positive = audit_aggregate["positive_window_count_by_cost_bps"][stress_key]
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "forward_chronological_audit"]
    if train_stress_positive < minimum_train_positive_windows:
        return (
            "FAST138_LARGE_MOVE_FIXED_VETO_FAILED_TRAIN_SELECTION",
            reasons
            + [
                f"best_policy:{best_train_policy['policy']['name']}",
                f"train_stress_positive_windows:{train_stress_positive}<{minimum_train_positive_windows}",
            ],
        )
    if (
        audit_aggregate["all_windows_positive_at_stress_cost"]
        and audit_aggregate["all_windows_meet_minimum_executions"]
        and audit_aggregate["net_total_by_cost_bps"][stress_key] > 0
    ):
        return (
            "FAST138_LARGE_MOVE_FIXED_VETO_FORWARD_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"best_policy:{best_train_policy['policy']['name']}",
                f"audit_stress_positive_windows:{audit_stress_positive}/{audit_holdout_count}",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    return (
        "FAST138_LARGE_MOVE_FIXED_VETO_FAILED_LATE_AUDIT",
        reasons
        + [
            f"best_policy:{best_train_policy['policy']['name']}",
            f"audit_stress_positive_windows:{audit_stress_positive}/{audit_holdout_count}",
            "do_not_promote_to_strategy",
        ],
    )


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    payload = {
        **report,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**payload, "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "best_train_policy": {
            "policy": report.get("best_train_policy", {}).get("policy"),
            "aggregate": report.get("best_train_policy", {}).get("aggregate"),
        },
        "audit_aggregate": report.get("audit_aggregate"),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
