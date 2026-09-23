"""Fast91 leave-one-label-out gate audit for near-threshold candidates.

Fast90 showed threshold relaxation does not create a robust edge. Fast91 tests
whether a simple causal gate can be selected on every other WAL label and still
work on the held-out label.

This is still diagnostic research. A survivor would only become a hypothesis
for a new preregistration; it would not approve Testnet or LIVE execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast61_fast60_shadow_settlement import (  # noqa: E402
    _apply_one_position_constraint,
    _write_jsonl,
)
from scripts.research.run_fast87_directional_filter_scout import (  # noqa: E402
    apply_conflict_policy,
    feature,
    summarize_variant_rows,
)
from scripts.research.run_fast90_near_threshold_scout import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as FAST90_OUTPUT_ROOT,
    passes_predicate,
    passes_probability_margin,
)


DEFAULT_INPUT_PATH = FAST90_OUTPUT_ROOT.with_name(
    "fast90-near-threshold-scout-all-local"
) / "independent_settlements.jsonl"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast91-lolo-gate-audit"
)
DEFAULT_PROBABILITY_MARGINS = (0.02, 0.05)
BASE_PREDICATES = (
    "all",
    "flow5_positive",
    "return5_and_flow5_positive",
    "vol15_flow5_positive",
)
CONFLICT_POLICIES = ("top_probability", "veto")
GATE_FEATURES = (
    "probability",
    "threshold_distance",
    "realized_vol_30s_bps",
    "realized_vol_5s_bps",
    "quote_volume_30s",
    "trade_count_30s",
    "mean_interarrival_ms_30s",
    "directed_flow_imbalance_1s",
    "directed_flow_imbalance_5s",
    "directed_flow_imbalance_30s",
    "directed_return_1s_bps",
    "directed_return_5s_bps",
    "directed_return_30s_bps",
)
GATE_QUANTILES = (0.2, 0.33, 0.5, 0.67, 0.8)
MINIMUM_TRAIN_TRADES = 10
MINIMUM_HOLDOUT_TRADES = 1


@dataclass(frozen=True)
class GateSpec:
    feature_name: str | None
    direction: str
    quantile: float | None
    threshold: float | None

    @property
    def name(self) -> str:
        if self.feature_name is None:
            return "none"
        return f"{self.feature_name}_{self.direction}_q{self.quantile:g}_{self.threshold:.10g}"


@dataclass(frozen=True)
class BaseSpec:
    probability_margin: float
    predicate_name: str
    conflict_policy: str

    @property
    def name(self) -> str:
        margin_name = f"{int(round(self.probability_margin * 1000)):03d}"
        return f"near{margin_name}_{self.predicate_name}_{self.conflict_policy}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--probability-margins",
        default=",".join(f"{margin:g}" for margin in DEFAULT_PROBABILITY_MARGINS),
    )
    parser.add_argument("--minimum-train-trades", type=int, default=MINIMUM_TRAIN_TRADES)
    parser.add_argument("--minimum-holdout-trades", type=int, default=MINIMUM_HOLDOUT_TRADES)
    arguments = parser.parse_args()
    report = run_fast91_lolo_gate_audit(
        input_path=arguments.input_path,
        output_root=arguments.output_root,
        probability_margins=parse_probability_margins(arguments.probability_margins),
        minimum_train_trades=arguments.minimum_train_trades,
        minimum_holdout_trades=arguments.minimum_holdout_trades,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast91_lolo_gate_audit(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    probability_margins: tuple[float, ...] = DEFAULT_PROBABILITY_MARGINS,
    minimum_train_trades: int = MINIMUM_TRAIN_TRADES,
    minimum_holdout_trades: int = MINIMUM_HOLDOUT_TRADES,
) -> dict[str, Any]:
    validate_parameters(
        probability_margins=probability_margins,
        minimum_train_trades=minimum_train_trades,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    rows = load_jsonl(input_path)
    output_root.mkdir(parents=True, exist_ok=True)
    base_specs = tuple(
        BaseSpec(margin, predicate_name, conflict_policy)
        for margin in probability_margins
        for predicate_name in BASE_PREDICATES
        for conflict_policy in CONFLICT_POLICIES
    )
    holdout_results = []
    for holdout_label in sorted({str(row["label"]) for row in rows}):
        train_rows = [row for row in rows if str(row["label"]) != holdout_label]
        holdout_rows = [row for row in rows if str(row["label"]) == holdout_label]
        result = evaluate_holdout_label(
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            holdout_label=holdout_label,
            base_specs=base_specs,
            minimum_train_trades=minimum_train_trades,
            minimum_holdout_trades=minimum_holdout_trades,
        )
        holdout_results.append(result)
    aggregate = aggregate_holdout_results(holdout_results)
    status, status_reasons = fast91_status(aggregate)
    holdout_path = output_root / "holdout_results.jsonl"
    _write_jsonl(holdout_path, holdout_results)
    report_path = output_root / "report.json"
    report: dict[str, Any] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast91_is_leave_one_label_out_diagnostic",
            "gate_selected_on_training_labels_only",
            "positive_result_would_require_new_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast91_lolo_gate_audit",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "row_count": len(rows),
        "label_count": len(holdout_results),
        "probability_margins": list(probability_margins),
        "minimum_train_trades": minimum_train_trades,
        "minimum_holdout_trades": minimum_holdout_trades,
        "base_spec_count": len(base_specs),
        "gate_feature_count": len(GATE_FEATURES),
        "gate_quantiles": list(GATE_QUANTILES),
        "holdout_results_path": str(holdout_path),
        "holdout_results_sha256": _sha256_file(holdout_path),
        "aggregate": aggregate,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_parameters(
    *,
    probability_margins: tuple[float, ...],
    minimum_train_trades: int,
    minimum_holdout_trades: int,
) -> None:
    if not probability_margins:
        raise ValueError("probability_margins cannot be empty")
    if any(margin < 0 or margin > 0.20 for margin in probability_margins):
        raise ValueError("probability_margins must be between 0 and 0.20")
    if minimum_train_trades < 1:
        raise ValueError("minimum_train_trades must be positive")
    if minimum_holdout_trades < 1:
        raise ValueError("minimum_holdout_trades must be positive")


def parse_probability_margins(raw: str) -> tuple[float, ...]:
    return tuple(sorted({float(part.strip()) for part in raw.split(",") if part.strip()}))


def evaluate_holdout_label(
    *,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    holdout_label: str,
    base_specs: tuple[BaseSpec, ...],
    minimum_train_trades: int,
    minimum_holdout_trades: int,
) -> dict[str, Any]:
    candidates = []
    for base_spec in base_specs:
        train_base = apply_base(train_rows, base_spec)
        if len(train_base) < minimum_train_trades:
            continue
        for gate in fit_gate_candidates(train_base):
            train_selected = apply_gate(train_base, gate)
            train_one_position = _apply_one_position_constraint(train_selected)
            if len(train_one_position) < minimum_train_trades:
                continue
            train_summary = summarize_variant_rows(train_one_position)
            candidates.append(
                {
                    "base": base_spec.name,
                    "gate": gate.name,
                    "gate_spec": gate_to_dict(gate),
                    "train_summary": train_summary,
                    "train_score": train_candidate_score(train_summary),
                }
            )
    if not candidates:
        return {
            "holdout_label": holdout_label,
            "status": "FAST91_NO_TRAINING_GATE_CANDIDATE",
            "selected": None,
            "holdout_summary": empty_summary(),
            "minimum_holdout_trades": minimum_holdout_trades,
        }
    selected = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate["train_score"]),
            float(candidate["train_summary"]["stress_total_bps"]),
            int(candidate["train_summary"]["trade_count"]),
        ),
        reverse=True,
    )[0]
    selected_base = base_spec_from_name(str(selected["base"]), base_specs)
    holdout_base = apply_base(holdout_rows, selected_base)
    holdout_selected = apply_gate(holdout_base, gate_from_dict(selected["gate_spec"]))
    holdout_one_position = _apply_one_position_constraint(holdout_selected)
    holdout_summary = summarize_variant_rows(holdout_one_position)
    status = holdout_status(
        holdout_summary,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    return {
        "holdout_label": holdout_label,
        "status": status,
        "selected": selected,
        "holdout_summary": holdout_summary,
        "minimum_holdout_trades": minimum_holdout_trades,
        "candidate_count": len(candidates),
    }


def apply_base(rows: list[dict[str, Any]], spec: BaseSpec) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if passes_probability_margin(row, spec.probability_margin)
        and passes_predicate(row, spec.predicate_name)
    ]
    return apply_conflict_policy(selected, spec.conflict_policy)


def fit_gate_candidates(rows: list[dict[str, Any]]) -> list[GateSpec]:
    gates = [GateSpec(feature_name=None, direction="none", quantile=None, threshold=None)]
    for feature_name in GATE_FEATURES:
        values = sorted(
            {
                feature_value
                for row in rows
                if (feature_value := safe_feature(row, feature_name)) is not None
                and math.isfinite(feature_value)
            }
        )
        if len(values) < 2:
            continue
        for quantile in GATE_QUANTILES:
            threshold = nearest_rank_quantile(values, quantile)
            gates.append(
                GateSpec(
                    feature_name=feature_name,
                    direction="gte",
                    quantile=quantile,
                    threshold=threshold,
                )
            )
            gates.append(
                GateSpec(
                    feature_name=feature_name,
                    direction="lte",
                    quantile=quantile,
                    threshold=threshold,
                )
            )
    return gates


def apply_gate(rows: list[dict[str, Any]], gate: GateSpec) -> list[dict[str, Any]]:
    if gate.feature_name is None:
        return rows
    selected = []
    for row in rows:
        value = safe_feature(row, gate.feature_name)
        if value is None:
            continue
        if gate.direction == "gte" and value >= float(gate.threshold):
            selected.append(row)
        elif gate.direction == "lte" and value <= float(gate.threshold):
            selected.append(row)
    return selected


def safe_feature(row: dict[str, Any], feature_name: str) -> float | None:
    if feature_name == "threshold_distance":
        return float(row["threshold"]) - float(row["probability"])
    try:
        return feature(row, feature_name)
    except KeyError:
        return None


def nearest_rank_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    index = max(0, math.ceil(quantile * len(values)) - 1)
    return float(values[index])


def train_candidate_score(summary: dict[str, Any]) -> float:
    stress_total = float(summary["stress_total_bps"])
    negative_labels = int(summary["negative_label_count"])
    label_count = max(1, int(summary["label_count"]))
    trade_count = max(1, int(summary["trade_count"]))
    return stress_total - 25.0 * negative_labels - 2.0 * (label_count < 3) + 0.01 * trade_count


def holdout_status(
    summary: dict[str, Any],
    *,
    minimum_holdout_trades: int,
) -> str:
    if int(summary["trade_count"]) < minimum_holdout_trades:
        return "FAST91_HOLDOUT_INSUFFICIENT_TRADES"
    if float(summary["stress_total_bps"]) <= 0:
        return "FAST91_HOLDOUT_STRESS_NON_POSITIVE"
    return "FAST91_HOLDOUT_POSITIVE"


def aggregate_holdout_results(holdout_results: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [
        result
        for result in holdout_results
        if result["status"] == "FAST91_HOLDOUT_POSITIVE"
    ]
    negative = [
        result
        for result in holdout_results
        if result["status"] == "FAST91_HOLDOUT_STRESS_NON_POSITIVE"
    ]
    insufficient = [
        result
        for result in holdout_results
        if result["status"] == "FAST91_HOLDOUT_INSUFFICIENT_TRADES"
    ]
    no_candidate = [
        result
        for result in holdout_results
        if result["status"] == "FAST91_NO_TRAINING_GATE_CANDIDATE"
    ]
    stress_values = [
        float(result["holdout_summary"]["stress_total_bps"])
        for result in holdout_results
    ]
    trade_counts = [
        int(result["holdout_summary"]["trade_count"])
        for result in holdout_results
    ]
    return {
        "holdout_count": len(holdout_results),
        "positive_holdout_count": len(positive),
        "negative_holdout_count": len(negative),
        "insufficient_holdout_count": len(insufficient),
        "no_training_candidate_count": len(no_candidate),
        "holdout_trade_count": sum(trade_counts),
        "holdout_stress_total_bps": float(sum(stress_values)),
        "holdout_stress_mean_per_label_bps": (
            float(sum(stress_values) / len(stress_values)) if stress_values else 0.0
        ),
        "worst_holdout_stress_total_bps": min(stress_values) if stress_values else None,
        "best_holdout_stress_total_bps": max(stress_values) if stress_values else None,
        "positive_holdout_labels": [result["holdout_label"] for result in positive],
        "negative_holdout_labels": [result["holdout_label"] for result in negative],
        "insufficient_holdout_labels": [result["holdout_label"] for result in insufficient],
    }


def fast91_status(aggregate: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "leave_one_label_out_diagnostic",
    ]
    if int(aggregate["holdout_count"]) == 0:
        return "FAST91_NO_HOLDOUT_LABELS", reasons
    if int(aggregate["positive_holdout_count"]) == int(aggregate["holdout_count"]):
        return "FAST91_LOLO_GATE_SURVIVOR_DIAGNOSTIC_ONLY", reasons + [
            "all_holdouts_positive",
            "requires_new_preregistration",
        ]
    return "FAST91_LOLO_GATE_FAILED", reasons + [
        f"positive_holdouts:{aggregate['positive_holdout_count']}/{aggregate['holdout_count']}",
        f"negative_holdouts:{aggregate['negative_holdout_count']}",
        f"insufficient_holdouts:{aggregate['insufficient_holdout_count']}",
    ]


def empty_summary() -> dict[str, Any]:
    return {
        "trade_count": 0,
        "label_count": 0,
        "stress_total_bps": 0.0,
        "stress_mean_bps": 0.0,
        "expected_total_bps": 0.0,
        "expected_mean_bps": 0.0,
        "expected_win_rate": 0.0,
        "maximum_drawdown_bps": 0.0,
        "positive_label_count": 0,
        "negative_label_count": 0,
        "worst_label_stress_total_bps": None,
        "label_summaries": {},
    }


def gate_to_dict(gate: GateSpec) -> dict[str, Any]:
    return {
        "feature_name": gate.feature_name,
        "direction": gate.direction,
        "quantile": gate.quantile,
        "threshold": gate.threshold,
    }


def gate_from_dict(payload: dict[str, Any]) -> GateSpec:
    return GateSpec(
        feature_name=payload["feature_name"],
        direction=str(payload["direction"]),
        quantile=None if payload["quantile"] is None else float(payload["quantile"]),
        threshold=None if payload["threshold"] is None else float(payload["threshold"]),
    )


def base_spec_from_name(name: str, base_specs: tuple[BaseSpec, ...]) -> BaseSpec:
    for spec in base_specs:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown base spec: {name}")


def parse_jsonl_line(line: str) -> dict[str, Any]:
    return json.loads(line)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [parse_jsonl_line(line) for line in handle if line.strip()]


def _sha256_file(path: Path) -> str:
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
        "row_count": report["row_count"],
        "label_count": report["label_count"],
        "aggregate": report["aggregate"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
