"""Train a causal regime gate for a frozen book rule and test on holdout WAL.

The script keeps the base rule frozen from an earlier development capture, then
searches a single additional regime condition on later development captures.
The selected gate is ranked only by gate-development performance and then
measured on a separate confirmation capture.
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
from scripts.research.run_book_feature_model_scout import (  # noqa: E402
    LABEL_AND_FUTURE_COLUMNS,
    select_causal_feature_columns,
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
    parser.add_argument(
        "--gate-development-wal-roots",
        default=(
            "backend/data/microstructure_v1/prospective-wal-fast20-30m,"
            "backend/data/microstructure_v1/prospective-wal-fast21-30m"
        ),
    )
    parser.add_argument("--confirmation-wal-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-regime-gate-scout",
    )
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    parser.add_argument("--base-condition-fraction", type=float, default=0.20)
    parser.add_argument(
        "--base-conditions-json",
        default='[{"feature":"directed_spot_perp_basis_bps","direction":"high"}]',
    )
    parser.add_argument("--gate-fractions", default="0.10,0.20,0.33,0.50")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--minimum-gate-development-selected-count", type=int, default=20)
    parser.add_argument("--minimum-confirmation-selected-count", type=int, default=10)
    parser.add_argument("--max-report-candidates", type=int, default=80)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    arguments = parser.parse_args()
    report = run_book_regime_gate_scout(
        base_development_wal_root=arguments.base_development_wal_root,
        gate_development_wal_roots=_parse_paths(arguments.gate_development_wal_roots),
        confirmation_wal_root=arguments.confirmation_wal_root,
        output_root=arguments.output_root,
        horizon_seconds=arguments.horizon_seconds,
        threshold_bps=arguments.threshold_bps,
        base_condition_fraction=arguments.base_condition_fraction,
        base_conditions=_parse_conditions(arguments.base_conditions_json),
        gate_fractions=_parse_float_list(arguments.gate_fractions),
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        minimum_gate_development_selected_count=(
            arguments.minimum_gate_development_selected_count
        ),
        minimum_confirmation_selected_count=arguments.minimum_confirmation_selected_count,
        max_report_candidates=arguments.max_report_candidates,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_regime_gate_scout(
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
    gate_fractions: tuple[float, ...] = (0.10, 0.20, 0.33, 0.50),
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    initial_stop_bps: float = 20.0,
    minimum_gate_development_selected_count: int = 20,
    minimum_confirmation_selected_count: int = 10,
    max_report_candidates: int = 80,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> dict[str, object]:
    _validate_parameters(
        gate_development_wal_roots=gate_development_wal_roots,
        horizon_seconds=horizon_seconds,
        threshold_bps=threshold_bps,
        base_condition_fraction=base_condition_fraction,
        base_conditions=base_conditions,
        gate_fractions=gate_fractions,
        initial_stop_bps=initial_stop_bps,
        minimum_gate_development_selected_count=minimum_gate_development_selected_count,
        minimum_confirmation_selected_count=minimum_confirmation_selected_count,
        max_report_candidates=max_report_candidates,
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
    base_development = _load_rows(
        wal_root=base_development_wal_root,
        config=config,
        threshold_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
    )
    base_report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "experiment_stage": "book_regime_gate_scout",
        "approved_strategy": False,
        "approval_reasons": [
            "diagnostic_only",
            "gate_development_window_count_limited",
            "single_confirmation_window",
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
        "gate_fractions": list(gate_fractions),
        "minimum_gate_development_selected_count": (
            minimum_gate_development_selected_count
        ),
        "minimum_confirmation_selected_count": minimum_confirmation_selected_count,
        "cost_levels_bps": list(cost_levels_bps),
        "leakage_guard": {
            "excluded_columns": sorted(LABEL_AND_FUTURE_COLUMNS),
            "base_threshold_fit_on_base_development_only": True,
            "gate_threshold_fit_on_gate_development_only": True,
            "selected_gate_ranked_without_confirmation": True,
            "confirmation_minimum_count_checked_after_selection": True,
        },
    }
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
    gate_frames: list[pd.DataFrame] = []
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
        frame = loaded["rows"].copy()
        frame["source_wal_root"] = key
        gate_frames.append(frame)
    gate_development_rows = pd.concat(gate_frames, ignore_index=True)
    confirmation = _load_rows(
        wal_root=confirmation_wal_root,
        config=config,
        threshold_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
    )
    report = {
        **base_report,
        "base_thresholds": base_thresholds,
        "gate_development_status": gate_streams,
        "base_development_metrics": summarize_selection(
            base_development["rows"],
            apply_frozen_rule(base_development["rows"], base_thresholds),
            cost_levels_bps=cost_levels_bps,
        ),
        "gate_development_base_rule_metrics": summarize_selection(
            gate_development_rows,
            apply_frozen_rule(gate_development_rows, base_thresholds),
            cost_levels_bps=cost_levels_bps,
        ),
    }
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

    candidates = evaluate_gate_candidates(
        gate_development_rows=gate_development_rows,
        confirmation_rows=confirmation["rows"],
        base_thresholds=base_thresholds,
        gate_fractions=gate_fractions,
        minimum_gate_development_selected_count=(
            minimum_gate_development_selected_count
        ),
        cost_levels_bps=cost_levels_bps,
    )
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["gate_development_metrics"]["selected_gross_path_payoff_mean_bps"],
            item["gate_development_metrics"]["selected_count"],
        ),
        reverse=True,
    )
    selected_gate = ranked[0] if ranked else None
    selected_confirmation_count = (
        int(selected_gate["confirmation_metrics"]["selected_count"])
        if selected_gate
        else 0
    )
    has_confirmation_count = (
        selected_confirmation_count >= minimum_confirmation_selected_count
    )
    status_reasons = []
    if not selected_gate:
        status_reasons.append("no_gate_met_gate_development_minimum_count")
    elif not has_confirmation_count:
        status_reasons.append(
            "confirmation_selected_count_below_minimum:"
            f"{selected_confirmation_count}<{minimum_confirmation_selected_count}"
        )
    confirmation_base_selected = apply_frozen_rule(confirmation["rows"], base_thresholds)
    return _write_report(
        output_root,
        {
            **report,
            "status": (
                "READY_FOR_REGIME_GATE_REVIEW"
                if selected_gate and has_confirmation_count
                else "CONFIRMATION_INSUFFICIENT_SELECTIONS"
                if selected_gate
                else "NO_GATE_CANDIDATE"
            ),
            "status_reasons": status_reasons,
            "modeling_ready": bool(selected_gate and has_confirmation_count),
            "feature_count": len(select_gate_feature_columns(gate_development_rows)),
            "candidates_evaluated": len(candidates),
            "selected_gate": selected_gate,
            "top_candidates_by_gate_development": ranked[:max_report_candidates],
            "confirmation_base_rule_metrics": summarize_selection(
                confirmation["rows"],
                confirmation_base_selected,
                cost_levels_bps=cost_levels_bps,
            ),
        },
    )


def evaluate_gate_candidates(
    *,
    gate_development_rows: pd.DataFrame,
    confirmation_rows: pd.DataFrame,
    base_thresholds: list[dict[str, object]],
    gate_fractions: tuple[float, ...],
    minimum_gate_development_selected_count: int,
    cost_levels_bps: tuple[float, ...],
) -> list[dict[str, object]]:
    base_feature_names = {str(item["feature"]) for item in base_thresholds}
    gate_development_base = apply_frozen_rule(gate_development_rows, base_thresholds)
    confirmation_base = apply_frozen_rule(confirmation_rows, base_thresholds)
    candidates: list[dict[str, object]] = []
    for feature in select_gate_feature_columns(gate_development_rows):
        if feature in base_feature_names:
            continue
        for direction in ("high", "low"):
            for fraction in gate_fractions:
                frozen_gate = fit_single_gate_threshold(
                    gate_development_base,
                    feature=feature,
                    direction=direction,
                    fraction=fraction,
                )
                gate_development_selected = apply_gate_threshold(
                    gate_development_base,
                    frozen_gate,
                )
                if len(gate_development_selected) < minimum_gate_development_selected_count:
                    continue
                confirmation_selected = apply_gate_threshold(
                    confirmation_base,
                    frozen_gate,
                )
                candidates.append(
                    {
                        "gate": frozen_gate,
                        "gate_development_metrics": summarize_selection(
                            gate_development_rows,
                            gate_development_selected,
                            cost_levels_bps=cost_levels_bps,
                        ),
                        "confirmation_metrics": summarize_selection(
                            confirmation_rows,
                            confirmation_selected,
                            cost_levels_bps=cost_levels_bps,
                        ),
                    }
                )
    return candidates


def select_gate_feature_columns(rows: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in select_causal_feature_columns(rows):
        if column in {"side_sign"}:
            continue
        if rows[column].nunique(dropna=False) <= 1:
            continue
        columns.append(column)
    return columns


def fit_single_gate_threshold(
    rows: pd.DataFrame,
    *,
    feature: str,
    direction: str,
    fraction: float,
) -> dict[str, object]:
    if direction not in {"high", "low"}:
        raise ValueError("direction must be high or low")
    quantile = 1 - fraction if direction == "high" else fraction
    return {
        "feature": feature,
        "direction": direction,
        "fraction": float(fraction),
        "quantile": float(quantile),
        "threshold": float(rows[feature].quantile(quantile)),
    }


def apply_gate_threshold(rows: pd.DataFrame, gate: dict[str, object]) -> pd.DataFrame:
    feature = str(gate["feature"])
    direction = str(gate["direction"])
    threshold = float(gate["threshold"])
    values = rows[feature].to_numpy(dtype=np.float64)
    if direction == "high":
        mask = values >= threshold
    elif direction == "low":
        mask = values <= threshold
    else:
        raise ValueError("direction must be high or low")
    return rows.loc[mask]


def _validate_parameters(
    *,
    gate_development_wal_roots: tuple[Path, ...],
    horizon_seconds: int,
    threshold_bps: float,
    base_condition_fraction: float,
    base_conditions: tuple[dict[str, str], ...],
    gate_fractions: tuple[float, ...],
    initial_stop_bps: float,
    minimum_gate_development_selected_count: int,
    minimum_confirmation_selected_count: int,
    max_report_candidates: int,
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
    if not gate_fractions or any(not 0 < fraction <= 0.5 for fraction in gate_fractions):
        raise ValueError("gate_fractions must contain values between 0 and 0.5")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if minimum_gate_development_selected_count < 1:
        raise ValueError("minimum_gate_development_selected_count must be positive")
    if minimum_confirmation_selected_count < 1:
        raise ValueError("minimum_confirmation_selected_count must be positive")
    if max_report_candidates < 1:
        raise ValueError("max_report_candidates must be positive")
    if not cost_levels_bps or any(cost_bps < 0 for cost_bps in cost_levels_bps):
        raise ValueError("cost_levels_bps must contain non-negative values")


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
        "selected_gate": report.get("selected_gate"),
        "confirmation_base_rule_metrics": report.get("confirmation_base_rule_metrics"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
