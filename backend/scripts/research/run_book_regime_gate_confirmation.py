"""Freeze one regime gate and test it on a separate book WAL.

Unlike the regime-gate scout, this script does not search across gate features.
It fits the requested gate threshold on gate-development rows only, then reports
the frozen base+gate rule on a separate confirmation capture.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.research_dataset import ResearchDatasetConfig  # noqa: E402
from scripts.research.run_book_feature_model_scout import (  # noqa: E402
    LABEL_AND_FUTURE_COLUMNS,
)
from scripts.research.run_book_regime_gate_scout import (  # noqa: E402
    apply_gate_threshold,
    fit_single_gate_threshold,
)
from scripts.research.run_book_rule_confirmation import (  # noqa: E402
    _load_rows,
    _parse_conditions,
    _parse_float_list,
    apply_frozen_rule,
    fit_rule_thresholds,
    summarize_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-development-wal-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "prospective-wal-fast16-18m",
    )
    parser.add_argument("--gate-development-wal-roots", required=True)
    parser.add_argument("--confirmation-wal-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-regime-gate-confirmation",
    )
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    parser.add_argument("--base-condition-fraction", type=float, default=0.20)
    parser.add_argument(
        "--base-conditions-json",
        default='[{"feature":"directed_spot_perp_basis_bps","direction":"high"}]',
    )
    parser.add_argument("--gate-feature", required=True)
    parser.add_argument("--gate-direction", choices=("high", "low"), required=True)
    parser.add_argument("--gate-fraction", type=float, required=True)
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--minimum-confirmation-selected-count", type=int, default=10)
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    arguments = parser.parse_args()
    report = run_book_regime_gate_confirmation(
        base_development_wal_root=arguments.base_development_wal_root,
        gate_development_wal_roots=_parse_paths(arguments.gate_development_wal_roots),
        confirmation_wal_root=arguments.confirmation_wal_root,
        output_root=arguments.output_root,
        horizon_seconds=arguments.horizon_seconds,
        threshold_bps=arguments.threshold_bps,
        base_condition_fraction=arguments.base_condition_fraction,
        base_conditions=_parse_conditions(arguments.base_conditions_json),
        gate_feature=arguments.gate_feature,
        gate_direction=arguments.gate_direction,
        gate_fraction=arguments.gate_fraction,
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        minimum_confirmation_selected_count=arguments.minimum_confirmation_selected_count,
        profit_cost_bps=arguments.profit_cost_bps,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_regime_gate_confirmation(
    *,
    base_development_wal_root: Path,
    gate_development_wal_roots: tuple[Path, ...],
    confirmation_wal_root: Path,
    output_root: Path,
    horizon_seconds: int = 120,
    threshold_bps: float = 5.0,
    base_condition_fraction: float = 0.20,
    base_conditions: tuple[dict[str, str], ...] = (
        {"feature": "directed_spot_perp_basis_bps", "direction": "high"},
    ),
    gate_feature: str,
    gate_direction: str,
    gate_fraction: float,
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    initial_stop_bps: float = 20.0,
    minimum_confirmation_selected_count: int = 10,
    profit_cost_bps: float = 1.0,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> dict[str, object]:
    _validate_parameters(
        gate_development_wal_roots=gate_development_wal_roots,
        horizon_seconds=horizon_seconds,
        threshold_bps=threshold_bps,
        base_condition_fraction=base_condition_fraction,
        base_conditions=base_conditions,
        gate_feature=gate_feature,
        gate_direction=gate_direction,
        gate_fraction=gate_fraction,
        initial_stop_bps=initial_stop_bps,
        minimum_confirmation_selected_count=minimum_confirmation_selected_count,
        profit_cost_bps=profit_cost_bps,
        cost_levels_bps=cost_levels_bps,
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
    base_report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "experiment_stage": "book_regime_gate_confirmation",
        "approved_strategy": False,
        "approval_reasons": [
            "diagnostic_only",
            "fixed_gate_confirmation_not_execution",
            "replication_or_shadow_replay_required",
            "no_statistical_gate",
            "no_order_execution",
        ],
        "base_development_wal_root": str(base_development_wal_root),
        "gate_development_wal_roots": [str(path) for path in gate_development_wal_roots],
        "confirmation_wal_root": str(confirmation_wal_root),
        "config": asdict(config),
        "base_rule": {
            "condition_fraction": base_condition_fraction,
            "conditions": list(base_conditions),
        },
        "gate_rule": {
            "feature": gate_feature,
            "direction": gate_direction,
            "fraction": gate_fraction,
        },
        "minimum_confirmation_selected_count": minimum_confirmation_selected_count,
        "profit_cost_bps": profit_cost_bps,
        "cost_levels_bps": list(cost_levels_bps),
        "leakage_guard": {
            "excluded_columns": sorted(LABEL_AND_FUTURE_COLUMNS),
            "base_threshold_fit_on_base_development_only": True,
            "gate_threshold_fit_on_gate_development_only": True,
            "confirmation_used_only_after_freezing_base_and_gate": True,
        },
    }
    base_development = _load_rows(
        wal_root=base_development_wal_root,
        config=config,
        threshold_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
    )
    if base_development["status"] != "READY":
        return _write_report(
            output_root,
            {
                **base_report,
                "status": f"BASE_DEVELOPMENT_{base_development['status']}",
                "status_reasons": base_development["status_reasons"],
                "modeling_ready": False,
            },
        )
    base_thresholds = fit_rule_thresholds(
        base_development["rows"],
        conditions=base_conditions,
        condition_fraction=base_condition_fraction,
    )

    gate_frames = []
    gate_streams: dict[str, object] = {}
    for wal_root in gate_development_wal_roots:
        loaded = _load_rows(
            wal_root=wal_root,
            config=config,
            threshold_bps=threshold_bps,
            initial_stop_bps=initial_stop_bps,
        )
        key = str(wal_root)
        gate_streams[key] = {
            "status": loaded["status"],
            "status_reasons": loaded["status_reasons"],
        }
        if loaded["status"] != "READY":
            return _write_report(
                output_root,
                {
                    **base_report,
                    "base_thresholds": base_thresholds,
                    "gate_development_status": gate_streams,
                    "status": f"GATE_DEVELOPMENT_{loaded['status']}",
                    "status_reasons": loaded["status_reasons"],
                    "modeling_ready": False,
                },
            )
        gate_frames.append(loaded["rows"])
    gate_development_rows = _concat_rows(gate_frames)
    gate_development_base = apply_frozen_rule(gate_development_rows, base_thresholds)
    if gate_feature not in gate_development_base.columns:
        return _write_report(
            output_root,
            {
                **base_report,
                "base_thresholds": base_thresholds,
                "gate_development_status": gate_streams,
                "status": "GATE_FEATURE_MISSING",
                "status_reasons": [f"gate_feature_missing:{gate_feature}"],
                "modeling_ready": False,
            },
        )
    frozen_gate = fit_single_gate_threshold(
        gate_development_base,
        feature=gate_feature,
        direction=gate_direction,
        fraction=gate_fraction,
    )
    report = {
        **base_report,
        "base_thresholds": base_thresholds,
        "gate_development_status": gate_streams,
        "frozen_gate": frozen_gate,
        "base_development_metrics": summarize_selection(
            base_development["rows"],
            apply_frozen_rule(base_development["rows"], base_thresholds),
            cost_levels_bps=cost_levels_bps,
        ),
        "gate_development_base_rule_metrics": summarize_selection(
            gate_development_rows,
            gate_development_base,
            cost_levels_bps=cost_levels_bps,
        ),
        "gate_development_gated_metrics": summarize_selection(
            gate_development_rows,
            apply_gate_threshold(gate_development_base, frozen_gate),
            cost_levels_bps=cost_levels_bps,
        ),
    }
    confirmation = _load_rows(
        wal_root=confirmation_wal_root,
        config=config,
        threshold_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
    )
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

    confirmation_base = apply_frozen_rule(confirmation["rows"], base_thresholds)
    confirmation_selected = apply_gate_threshold(confirmation_base, frozen_gate)
    confirmation_metrics = summarize_selection(
        confirmation["rows"],
        confirmation_selected,
        cost_levels_bps=cost_levels_bps,
    )
    confirmation_count = int(confirmation_metrics["selected_count"])
    confirmation_net = _net_at_cost(confirmation_metrics, profit_cost_bps)
    status_reasons = []
    if confirmation_count < minimum_confirmation_selected_count:
        status_reasons.append(
            "confirmation_selected_count_below_minimum:"
            f"{confirmation_count}<{minimum_confirmation_selected_count}"
        )
    if confirmation_net <= 0:
        status_reasons.append(
            f"confirmation_net_at_{profit_cost_bps:g}bps_cost_non_positive:"
            f"{confirmation_net}"
        )
    if not status_reasons:
        status_reasons.append(
            "replication_or_shadow_replay_required_before_strategy_approval"
        )
    modeling_ready = (
        confirmation_count >= minimum_confirmation_selected_count
        and confirmation_net > 0
    )
    return _write_report(
        output_root,
        {
            **report,
            "status": (
                "FIXED_GATE_CONFIRMATION_READY_FOR_REPLICATION"
                if modeling_ready
                else "FIXED_GATE_CONFIRMATION_FAILED"
            ),
            "status_reasons": status_reasons,
            "modeling_ready": modeling_ready,
            "confirmation_base_rule_metrics": summarize_selection(
                confirmation["rows"],
                confirmation_base,
                cost_levels_bps=cost_levels_bps,
            ),
            "confirmation_gated_metrics": confirmation_metrics,
        },
    )


def _validate_parameters(
    *,
    gate_development_wal_roots: tuple[Path, ...],
    horizon_seconds: int,
    threshold_bps: float,
    base_condition_fraction: float,
    base_conditions: tuple[dict[str, str], ...],
    gate_feature: str,
    gate_direction: str,
    gate_fraction: float,
    initial_stop_bps: float,
    minimum_confirmation_selected_count: int,
    profit_cost_bps: float,
    cost_levels_bps: tuple[float, ...],
) -> None:
    if not gate_development_wal_roots:
        raise ValueError("gate_development_wal_roots cannot be empty")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if threshold_bps <= 0:
        raise ValueError("threshold_bps must be positive")
    if not 0 < base_condition_fraction < 0.5:
        raise ValueError("base_condition_fraction must be between 0 and 0.5")
    if not base_conditions:
        raise ValueError("base_conditions cannot be empty")
    for condition in base_conditions:
        if condition["direction"] not in {"high", "low"}:
            raise ValueError("base condition direction must be high or low")
    if not gate_feature:
        raise ValueError("gate_feature cannot be empty")
    if gate_direction not in {"high", "low"}:
        raise ValueError("gate_direction must be high or low")
    if not 0 < gate_fraction <= 0.5:
        raise ValueError("gate_fraction must be between 0 and 0.5")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if minimum_confirmation_selected_count < 1:
        raise ValueError("minimum_confirmation_selected_count must be positive")
    if profit_cost_bps < 0:
        raise ValueError("profit_cost_bps must be non-negative")
    if not cost_levels_bps or any(cost_bps < 0 for cost_bps in cost_levels_bps):
        raise ValueError("cost_levels_bps must contain non-negative values")


def _concat_rows(frames: list[object]) -> object:
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def _net_at_cost(metrics: dict[str, object], cost_bps: float) -> float:
    for item in metrics["cost_sensitivity"]:
        if float(item["total_cost_bps"]) == cost_bps:
            return float(item["selected_net_mean_bps"])
    return float(metrics["selected_gross_path_payoff_mean_bps"]) - cost_bps


def _parse_paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(part.strip()) for part in value.split(",") if part.strip())
    if not paths:
        raise ValueError("path list cannot be empty")
    return paths


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
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
        "approved_strategy": report["approved_strategy"],
        "frozen_gate": report.get("frozen_gate"),
        "confirmation_gated_metrics": report.get("confirmation_gated_metrics"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
