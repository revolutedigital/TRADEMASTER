"""Scout a second causal filter for the fixed book-gate shadow signals.

The fixed spot-age gate is not robust enough as a standalone strategy. This
diagnostic looks for a simple additional threshold over decision-time features
that improves the already-emitted shadow signals. It only consumes dry-run
settlement reports and never approves a strategy.
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

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_book_executable_replay import (  # noqa: E402
    _apply_one_position_constraint,
    summarize_executable_trades,
)
from scripts.research.run_book_rule_confirmation import _parse_float_list  # noqa: E402


DEFAULT_FEATURES = (
    "book_update_age_ms",
    "spread_bps",
    "spot_update_age_ms",
    "directed_spot_perp_basis_bps",
)
DEFAULT_QUANTILES = (0.1, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9)


@dataclass(frozen=True)
class SettlementReportRef:
    label: str
    path: Path


@dataclass(frozen=True)
class Candidate:
    feature: str
    direction: str
    threshold: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settlement-reports",
        required=True,
        help="Comma-separated LABEL=report.json list.",
    )
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--quantiles", default=",".join(str(value) for value in DEFAULT_QUANTILES))
    parser.add_argument("--minimum-one-position-count-per-window", type=int, default=2)
    parser.add_argument("--profit-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--fixed-candidate",
        help="Optional frozen candidate as FEATURE:DIRECTION:THRESHOLD.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-fixed-gate-second-filter-scout",
    )
    arguments = parser.parse_args()
    report = run_book_fixed_gate_second_filter_scout(
        settlement_report_refs=_parse_report_refs(arguments.settlement_reports),
        output_root=arguments.output_root,
        features=_parse_string_list(arguments.features),
        quantiles=_parse_quantiles(arguments.quantiles),
        minimum_one_position_count_per_window=arguments.minimum_one_position_count_per_window,
        profit_cost_bps=arguments.profit_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
        top_n=arguments.top_n,
        fixed_candidate=_parse_fixed_candidate(arguments.fixed_candidate)
        if arguments.fixed_candidate
        else None,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_fixed_gate_second_filter_scout(
    *,
    settlement_report_refs: tuple[SettlementReportRef, ...],
    output_root: Path,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    minimum_one_position_count_per_window: int = 2,
    profit_cost_bps: float = 1.0,
    stress_cost_bps: float = 2.0,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    top_n: int = 20,
    fixed_candidate: Candidate | None = None,
) -> dict[str, object]:
    _validate_parameters(
        settlement_report_refs=settlement_report_refs,
        features=features,
        quantiles=quantiles,
        minimum_one_position_count_per_window=minimum_one_position_count_per_window,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        cost_levels_bps=cost_levels_bps,
        top_n=top_n,
    )
    loaded = [_load_window(ref) for ref in settlement_report_refs]
    rows = pd.concat([item["rows"] for item in loaded], ignore_index=True)
    candidate_grid = _candidate_grid(rows, features=features, quantiles=quantiles)
    scored_candidates = [
        scored
        for candidate in candidate_grid
        if (
            scored := _score_candidate(
                rows,
                candidate=candidate,
                required_windows=tuple(item["label"] for item in loaded),
                minimum_one_position_count_per_window=minimum_one_position_count_per_window,
                profit_cost_bps=profit_cost_bps,
                stress_cost_bps=stress_cost_bps,
                cost_levels_bps=cost_levels_bps,
            )
        )
        is not None
    ]
    scored_candidates.sort(key=_candidate_sort_key, reverse=True)
    top_candidates = scored_candidates[:top_n]
    walkforward = _walkforward_selection(
        rows,
        candidate_grid=candidate_grid,
        window_labels=tuple(item["label"] for item in loaded),
        minimum_one_position_count_per_window=minimum_one_position_count_per_window,
        profit_cost_bps=profit_cost_bps,
        stress_cost_bps=stress_cost_bps,
        cost_levels_bps=cost_levels_bps,
    )
    fixed_candidate_result = (
        _score_candidate(
            rows,
            candidate=fixed_candidate,
            required_windows=tuple(item["label"] for item in loaded),
            minimum_one_position_count_per_window=1,
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
            cost_levels_bps=cost_levels_bps,
        )
        if fixed_candidate is not None
        else None
    )
    status, reasons = _status(top_candidates=top_candidates, walkforward=walkforward)
    return _write_report(
        output_root,
        {
            "research_only": True,
            "order_submission_allowed": False,
            "execution_authorization": "none",
            "committed": False,
            "dry_run": True,
            "approved_strategy": False,
            "experiment_stage": "book_fixed_gate_second_filter_scout",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "features": list(features),
            "quantiles": list(quantiles),
            "minimum_one_position_count_per_window": minimum_one_position_count_per_window,
            "profit_cost_bps": profit_cost_bps,
            "stress_cost_bps": stress_cost_bps,
            "cost_levels_bps": list(cost_levels_bps),
            "source_reports": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"rows"}
                }
                for item in loaded
            ],
            "candidate_count": len(scored_candidates),
            "top_candidates": top_candidates,
            "fixed_candidate_result": fixed_candidate_result,
            "walkforward": walkforward,
            "status": status,
            "status_reasons": reasons,
            "approval_reasons": [
                "diagnostic_only",
                "candidate_selected_after_reviewing_shadow_windows",
                "future_holdout_required_before_testnet",
                "no_order_execution",
            ],
        },
    )


def _load_window(ref: SettlementReportRef) -> dict[str, object]:
    report = json.loads(ref.path.read_text(encoding="utf-8"))
    _validate_settlement_report(report, ref)
    decisions_path = _resolve_artifact_path(report["shadow_decisions_path"], ref.path)
    settlements_path = _resolve_artifact_path(report["settlements_path"], ref.path)
    decisions = _load_decisions(decisions_path, label=ref.label)
    settlements = pd.read_parquet(settlements_path)
    rows = settlements.merge(
        decisions,
        on=("decision_time_ms", "feature_vector_sha256"),
        how="left",
        validate="one_to_one",
    )
    if rows["window_label"].isna().any():
        raise ValueError(f"{ref.label} has settlements without joined decision features")
    rows["window_label"] = ref.label
    return {
        "label": ref.label,
        "settlement_report_path": str(ref.path),
        "settlement_report_sha256": _sha256_file(ref.path),
        "shadow_decisions_path": str(decisions_path),
        "shadow_decisions_sha256": _sha256_file(decisions_path),
        "settlements_path": str(settlements_path),
        "settlements_sha256": _sha256_file(settlements_path),
        "latency_ms": int(report["execution_replay"]["latency_ms"]),
        "source_status": report["status"],
        "source_status_reasons": report.get("status_reasons", []),
        "rows": rows,
    }


def _load_decisions(path: Path, *, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            feature_vector = payload["feature_vector"]
            rows.append(
                {
                    "window_label": label,
                    "decision_time_ms": int(payload["decision_time_ms"]),
                    "feature_vector_sha256": payload["feature_vector_sha256"],
                    **{key: float(value) for key, value in feature_vector.items()},
                }
            )
    if not rows:
        raise ValueError(f"shadow decision file has no rows: {path}")
    return pd.DataFrame(rows)


def _candidate_grid(
    rows: pd.DataFrame,
    *,
    features: tuple[str, ...],
    quantiles: tuple[float, ...],
) -> tuple[Candidate, ...]:
    candidates: list[Candidate] = []
    for feature in features:
        if feature not in rows.columns:
            raise ValueError(f"feature missing from joined rows: {feature}")
        thresholds = sorted({float(rows[feature].quantile(quantile)) for quantile in quantiles})
        for threshold in thresholds:
            candidates.append(Candidate(feature=feature, direction="low", threshold=threshold))
            candidates.append(Candidate(feature=feature, direction="high", threshold=threshold))
    return tuple(candidates)


def _score_candidate(
    rows: pd.DataFrame,
    *,
    candidate: Candidate,
    required_windows: tuple[str, ...],
    minimum_one_position_count_per_window: int,
    profit_cost_bps: float,
    stress_cost_bps: float,
    cost_levels_bps: tuple[float, ...],
) -> dict[str, object] | None:
    selected = _apply_candidate(rows, candidate)
    if selected.empty:
        return None
    windows: dict[str, object] = {}
    one_position_profit_total = 0.0
    one_position_count = 0
    independent_profit_total = 0.0
    independent_count = 0
    for window_label in required_windows:
        window_rows = selected[selected["window_label"] == window_label]
        if window_rows.empty:
            return None
        independent = summarize_executable_trades(
            window_rows,
            selected_signal_count=len(window_rows),
            cost_levels_bps=cost_levels_bps,
        )
        one_position_rows = _apply_one_position_constraint(window_rows)
        if len(one_position_rows) < minimum_one_position_count_per_window:
            return None
        one_position = summarize_executable_trades(
            one_position_rows,
            selected_signal_count=len(window_rows),
            cost_levels_bps=cost_levels_bps,
        )
        independent_profit = _net_at_cost(independent, profit_cost_bps)
        one_position_profit = _net_at_cost(one_position, profit_cost_bps)
        one_position_stress = _net_at_cost(one_position, stress_cost_bps)
        windows[window_label] = {
            "selected_signal_count": int(len(window_rows)),
            "independent_executable_count": int(independent["executable_count"]),
            "independent_net_at_profit_cost_bps": independent_profit,
            "one_position_executable_count": int(one_position["executable_count"]),
            "one_position_net_at_profit_cost_bps": one_position_profit,
            "one_position_net_at_stress_cost_bps": one_position_stress,
            "one_position_exit_reasons": one_position["exit_reasons"],
        }
        independent_profit_total += independent_profit * int(independent["executable_count"])
        independent_count += int(independent["executable_count"])
        one_position_profit_total += one_position_profit * int(one_position["executable_count"])
        one_position_count += int(one_position["executable_count"])
    one_position_profit_by_window = [
        float(window["one_position_net_at_profit_cost_bps"]) for window in windows.values()
    ]
    one_position_stress_by_window = [
        float(window["one_position_net_at_stress_cost_bps"]) for window in windows.values()
    ]
    return {
        "candidate": asdict(candidate),
        "window_count": len(windows),
        "selected_signal_count": int(len(selected)),
        "independent_executable_count": independent_count,
        "independent_weighted_net_at_profit_cost_bps": independent_profit_total
        / independent_count,
        "one_position_executable_count": one_position_count,
        "one_position_weighted_net_at_profit_cost_bps": one_position_profit_total
        / one_position_count,
        "one_position_min_window_net_at_profit_cost_bps": min(one_position_profit_by_window),
        "one_position_positive_window_count_at_profit_cost": sum(
            value > 0 for value in one_position_profit_by_window
        ),
        "one_position_positive_window_count_at_stress_cost": sum(
            value > 0 for value in one_position_stress_by_window
        ),
        "windows": windows,
    }


def _walkforward_selection(
    rows: pd.DataFrame,
    *,
    candidate_grid: tuple[Candidate, ...],
    window_labels: tuple[str, ...],
    minimum_one_position_count_per_window: int,
    profit_cost_bps: float,
    stress_cost_bps: float,
    cost_levels_bps: tuple[float, ...],
) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    for holdout in window_labels:
        development_labels = tuple(label for label in window_labels if label != holdout)
        development_rows = rows[rows["window_label"].isin(development_labels)]
        development_scores = [
            scored
            for candidate in candidate_grid
            if (
                scored := _score_candidate(
                    development_rows,
                    candidate=candidate,
                    required_windows=development_labels,
                    minimum_one_position_count_per_window=minimum_one_position_count_per_window,
                    profit_cost_bps=profit_cost_bps,
                    stress_cost_bps=stress_cost_bps,
                    cost_levels_bps=cost_levels_bps,
                )
            )
            is not None
        ]
        development_scores.sort(key=_candidate_sort_key, reverse=True)
        if not development_scores:
            folds.append(
                {
                    "holdout": holdout,
                    "development_windows": list(development_labels),
                    "selected_candidate": None,
                    "holdout_result": None,
                    "status": "NO_DEVELOPMENT_CANDIDATE",
                }
            )
            continue
        selected = development_scores[0]
        candidate = Candidate(**selected["candidate"])
        holdout_score = _score_candidate(
            rows[rows["window_label"] == holdout],
            candidate=candidate,
            required_windows=(holdout,),
            minimum_one_position_count_per_window=1,
            profit_cost_bps=profit_cost_bps,
            stress_cost_bps=stress_cost_bps,
            cost_levels_bps=cost_levels_bps,
        )
        if holdout_score is None:
            status = "HOLDOUT_HAS_NO_EXECUTABLE_SELECTION"
        elif (
            holdout_score["one_position_min_window_net_at_profit_cost_bps"] <= 0
            or holdout_score["one_position_executable_count"]
            < minimum_one_position_count_per_window
        ):
            status = "HOLDOUT_FAILED_PROFIT_COST"
        else:
            status = "HOLDOUT_PASSED_PROFIT_COST"
        folds.append(
            {
                "holdout": holdout,
                "development_windows": list(development_labels),
                "selected_candidate": selected,
                "holdout_result": holdout_score,
                "status": status,
            }
        )
    return folds


def _apply_candidate(rows: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    if candidate.direction == "low":
        return rows[rows[candidate.feature] <= candidate.threshold].copy()
    if candidate.direction == "high":
        return rows[rows[candidate.feature] >= candidate.threshold].copy()
    raise ValueError("candidate direction must be low or high")


def _candidate_sort_key(score: dict[str, object]) -> tuple[object, ...]:
    return (
        score["one_position_positive_window_count_at_profit_cost"],
        score["one_position_min_window_net_at_profit_cost_bps"],
        score["one_position_weighted_net_at_profit_cost_bps"],
        score["one_position_executable_count"],
        score["independent_weighted_net_at_profit_cost_bps"],
    )


def _status(
    *,
    top_candidates: list[dict[str, object]],
    walkforward: list[dict[str, object]],
) -> tuple[str, list[str]]:
    if not top_candidates:
        return "SECOND_FILTER_SCOUT_NO_CANDIDATE", ["no_candidate_met_minimum_counts"]
    passed_folds = sum(fold["status"] == "HOLDOUT_PASSED_PROFIT_COST" for fold in walkforward)
    if passed_folds != len(walkforward):
        return (
            "SECOND_FILTER_SCOUT_IN_SAMPLE_CANDIDATE_WALKFORWARD_UNSTABLE",
            [
                f"walkforward_passed_folds:{passed_folds}<{len(walkforward)}",
                "future_holdout_required_before_strategy_approval",
            ],
        )
    return (
        "SECOND_FILTER_SCOUT_READY_FOR_FUTURE_HOLDOUT",
        ["future_holdout_required_before_strategy_approval"],
    )


def _validate_parameters(
    *,
    settlement_report_refs: tuple[SettlementReportRef, ...],
    features: tuple[str, ...],
    quantiles: tuple[float, ...],
    minimum_one_position_count_per_window: int,
    profit_cost_bps: float,
    stress_cost_bps: float,
    cost_levels_bps: tuple[float, ...],
    top_n: int,
) -> None:
    if not settlement_report_refs:
        raise ValueError("at least one settlement report is required")
    if not features:
        raise ValueError("at least one feature is required")
    if not quantiles or any(quantile <= 0 or quantile >= 1 for quantile in quantiles):
        raise ValueError("quantiles must be between 0 and 1")
    if minimum_one_position_count_per_window < 1:
        raise ValueError("minimum_one_position_count_per_window must be positive")
    if profit_cost_bps not in cost_levels_bps:
        raise ValueError("profit_cost_bps must be present in cost_levels_bps")
    if stress_cost_bps not in cost_levels_bps:
        raise ValueError("stress_cost_bps must be present in cost_levels_bps")
    if top_n < 1:
        raise ValueError("top_n must be positive")


def _validate_settlement_report(
    report: dict[str, Any],
    ref: SettlementReportRef,
) -> None:
    if report.get("research_only") is not True:
        raise ValueError(f"{ref.label} report must be research_only")
    if report.get("order_submission_allowed") is not False:
        raise ValueError(f"{ref.label} report must not allow orders")
    if report.get("execution_authorization") != "none":
        raise ValueError(f"{ref.label} report must have execution_authorization=none")
    if report.get("committed") is not False or report.get("dry_run") is not True:
        raise ValueError(f"{ref.label} report must be dry-run only")
    if report.get("approved_strategy") is not False:
        raise ValueError(f"{ref.label} report must not approve a strategy")
    if "shadow_decisions_path" not in report or "settlements_path" not in report:
        raise ValueError(f"{ref.label} report is missing shadow decision or settlement path")


def _net_at_cost(summary: dict[str, object], cost_bps: float) -> float:
    for item in summary["cost_sensitivity"]:
        if float(item["explicit_cost_bps"]) == cost_bps:
            return float(item["net_executable_mean_bps"])
    raise ValueError(f"cost level missing: {cost_bps:g} bps")


def _parse_report_refs(value: str) -> tuple[SettlementReportRef, ...]:
    refs = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("settlement reports must use LABEL=path")
        label, raw_path = part.split("=", 1)
        path = Path(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"settlement report not found: {path}")
        refs.append(SettlementReportRef(label=label.strip(), path=path))
    return tuple(refs)


def _parse_string_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("expected at least one item")
    return items


def _parse_quantiles(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _parse_string_list(value))


def _parse_fixed_candidate(value: str) -> Candidate:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("fixed candidate must use FEATURE:DIRECTION:THRESHOLD")
    feature, direction, raw_threshold = (part.strip() for part in parts)
    if direction not in {"low", "high"}:
        raise ValueError("fixed candidate direction must be low or high")
    return Candidate(feature=feature, direction=direction, threshold=float(raw_threshold))


def _resolve_artifact_path(raw_value: object, report_path: Path) -> Path:
    raw_path = Path(str(raw_value))
    candidates = (
        raw_path,
        BACKEND_ROOT / raw_path,
        REPO_ROOT / raw_path,
        report_path.parent / raw_path.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"artifact path not found: {raw_path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    top_candidate = report["top_candidates"][0] if report["top_candidates"] else None
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "candidate_count": report["candidate_count"],
        "top_candidate": top_candidate,
        "fixed_candidate_result": report["fixed_candidate_result"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
