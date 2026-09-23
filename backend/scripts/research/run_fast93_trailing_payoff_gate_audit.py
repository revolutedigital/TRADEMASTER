"""Fast93 causal gate audit for trailing payoff.

Fast92 showed that opening the entry universe does not create a robust edge.
Fast93 asks the next, stricter question: can a causal pre-trade gate be selected
on other WAL labels because it already pays the trailing policy, then survive on
the held-out label?

This is diagnostic research only. It never approves Testnet or LIVE execution.
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


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast92-broad-universe-scout-all-local"
    / "independent_settlements.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast93-trailing-payoff-gate-audit"
)
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
GATE_QUANTILES = (0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9)
CONFLICT_POLICIES = ("top_probability", "veto")
MINIMUM_TRAIN_TRADES = 10
MINIMUM_TRAIN_LABELS = 3
MINIMUM_HOLDOUT_TRADES = 1
MAXIMUM_TRAIN_NEGATIVE_LABELS = 0


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
class CandidateSpec:
    conflict_policy: str
    gate: GateSpec

    @property
    def name(self) -> str:
        return f"{self.conflict_policy}__{self.gate.name}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--minimum-train-trades", type=int, default=MINIMUM_TRAIN_TRADES)
    parser.add_argument("--minimum-train-labels", type=int, default=MINIMUM_TRAIN_LABELS)
    parser.add_argument("--minimum-holdout-trades", type=int, default=MINIMUM_HOLDOUT_TRADES)
    parser.add_argument(
        "--maximum-train-negative-labels",
        type=int,
        default=MAXIMUM_TRAIN_NEGATIVE_LABELS,
    )
    arguments = parser.parse_args()

    report = run_fast93_trailing_payoff_gate_audit(
        input_path=arguments.input_path,
        output_root=arguments.output_root,
        minimum_train_trades=arguments.minimum_train_trades,
        minimum_train_labels=arguments.minimum_train_labels,
        minimum_holdout_trades=arguments.minimum_holdout_trades,
        maximum_train_negative_labels=arguments.maximum_train_negative_labels,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast93_trailing_payoff_gate_audit(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    minimum_train_trades: int = MINIMUM_TRAIN_TRADES,
    minimum_train_labels: int = MINIMUM_TRAIN_LABELS,
    minimum_holdout_trades: int = MINIMUM_HOLDOUT_TRADES,
    maximum_train_negative_labels: int = MAXIMUM_TRAIN_NEGATIVE_LABELS,
) -> dict[str, Any]:
    validate_parameters(
        minimum_train_trades=minimum_train_trades,
        minimum_train_labels=minimum_train_labels,
        minimum_holdout_trades=minimum_holdout_trades,
        maximum_train_negative_labels=maximum_train_negative_labels,
    )
    rows = load_jsonl(input_path)
    output_root.mkdir(parents=True, exist_ok=True)

    holdout_results = [
        evaluate_holdout_label(
            train_rows=[row for row in rows if str(row["label"]) != holdout_label],
            holdout_rows=[row for row in rows if str(row["label"]) == holdout_label],
            holdout_label=holdout_label,
            minimum_train_trades=minimum_train_trades,
            minimum_train_labels=minimum_train_labels,
            minimum_holdout_trades=minimum_holdout_trades,
            maximum_train_negative_labels=maximum_train_negative_labels,
        )
        for holdout_label in sorted({str(row["label"]) for row in rows})
    ]
    aggregate = aggregate_holdout_results(holdout_results)
    aggregate["maximum_train_negative_labels"] = maximum_train_negative_labels
    status, status_reasons = fast93_status(aggregate)

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
            "fast93_is_leave_one_label_out_trailing_payoff_gate_audit",
            "gate_selected_on_training_labels_only",
            "uses_only_causal_pre_trade_features",
            "positive_result_would_require_new_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast93_trailing_payoff_gate_audit",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "row_count": len(rows),
        "label_count": len(holdout_results),
        "conflict_policies": list(CONFLICT_POLICIES),
        "gate_features": list(GATE_FEATURES),
        "gate_quantiles": list(GATE_QUANTILES),
        "minimum_train_trades": minimum_train_trades,
        "minimum_train_labels": minimum_train_labels,
        "minimum_holdout_trades": minimum_holdout_trades,
        "maximum_train_negative_labels": maximum_train_negative_labels,
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
    minimum_train_trades: int,
    minimum_train_labels: int,
    minimum_holdout_trades: int,
    maximum_train_negative_labels: int,
) -> None:
    if minimum_train_trades < 1:
        raise ValueError("minimum_train_trades must be positive")
    if minimum_train_labels < 1:
        raise ValueError("minimum_train_labels must be positive")
    if minimum_holdout_trades < 1:
        raise ValueError("minimum_holdout_trades must be positive")
    if maximum_train_negative_labels < 0:
        raise ValueError("maximum_train_negative_labels cannot be negative")


def evaluate_holdout_label(
    *,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    holdout_label: str,
    minimum_train_trades: int,
    minimum_train_labels: int,
    minimum_holdout_trades: int,
    maximum_train_negative_labels: int,
) -> dict[str, Any]:
    candidates = fit_training_candidates(
        train_rows,
        minimum_train_trades=minimum_train_trades,
        minimum_train_labels=minimum_train_labels,
        maximum_train_negative_labels=maximum_train_negative_labels,
    )
    if not candidates:
        return {
            "holdout_label": holdout_label,
            "status": "FAST93_NO_STRICT_TRAINING_SURVIVOR",
            "selected": None,
            "holdout_summary": empty_summary(),
            "candidate_count": 0,
            "minimum_holdout_trades": minimum_holdout_trades,
        }

    selected = max(candidates, key=lambda candidate: float(candidate["train_score"]))
    spec = candidate_spec_from_dict(selected["spec"])
    holdout_selected = _apply_one_position_constraint(
        apply_gate(apply_conflict_policy(holdout_rows, spec.conflict_policy), spec.gate)
    )
    holdout_summary = summarize_variant_rows(holdout_selected)
    status = holdout_status(
        holdout_summary,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    return {
        "holdout_label": holdout_label,
        "status": status,
        "selected": selected,
        "holdout_summary": holdout_summary,
        "candidate_count": len(candidates),
        "minimum_holdout_trades": minimum_holdout_trades,
    }


def fit_training_candidates(
    rows: list[dict[str, Any]],
    *,
    minimum_train_trades: int,
    minimum_train_labels: int,
    maximum_train_negative_labels: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for conflict_policy in CONFLICT_POLICIES:
        base_rows = apply_conflict_policy(rows, conflict_policy)
        for gate in fit_gate_candidates(base_rows):
            selected_rows = _apply_one_position_constraint(apply_gate(base_rows, gate))
            summary = summarize_variant_rows(selected_rows)
            if not training_summary_survives(
                summary,
                minimum_train_trades=minimum_train_trades,
                minimum_train_labels=minimum_train_labels,
                maximum_train_negative_labels=maximum_train_negative_labels,
            ):
                continue
            spec = CandidateSpec(conflict_policy=conflict_policy, gate=gate)
            candidates.append(
                {
                    "name": spec.name,
                    "spec": candidate_spec_to_dict(spec),
                    "train_summary": summary,
                    "train_score": train_candidate_score(summary),
                }
            )
    return sorted(candidates, key=lambda candidate: float(candidate["train_score"]), reverse=True)


def fit_gate_candidates(rows: list[dict[str, Any]]) -> list[GateSpec]:
    gates = [GateSpec(feature_name=None, direction="none", quantile=None, threshold=None)]
    for feature_name in GATE_FEATURES:
        values = sorted(
            {
                value
                for row in rows
                if (value := safe_feature(row, feature_name)) is not None and math.isfinite(value)
            }
        )
        if len(values) < 2:
            continue
        for quantile in GATE_QUANTILES:
            threshold = nearest_rank_quantile(values, quantile)
            gates.append(GateSpec(feature_name, "gte", quantile, threshold))
            gates.append(GateSpec(feature_name, "lte", quantile, threshold))
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
        and float(summary["stress_total_bps"]) > 0
        and float(summary["stress_mean_bps"]) > 0
        and int(summary["negative_label_count"]) <= maximum_train_negative_labels
    )


def train_candidate_score(summary: dict[str, Any]) -> float:
    return (
        float(summary["stress_total_bps"])
        - 50.0 * int(summary["negative_label_count"])
        + 0.02 * int(summary["trade_count"])
    )


def holdout_status(
    summary: dict[str, Any],
    *,
    minimum_holdout_trades: int,
) -> str:
    if int(summary["trade_count"]) < minimum_holdout_trades:
        return "FAST93_HOLDOUT_INSUFFICIENT_TRADES"
    if float(summary["stress_total_bps"]) <= 0:
        return "FAST93_HOLDOUT_STRESS_NON_POSITIVE"
    return "FAST93_HOLDOUT_POSITIVE"


def aggregate_holdout_results(holdout_results: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [result for result in holdout_results if result["status"] == "FAST93_HOLDOUT_POSITIVE"]
    negative = [
        result for result in holdout_results if result["status"] == "FAST93_HOLDOUT_STRESS_NON_POSITIVE"
    ]
    insufficient = [
        result for result in holdout_results if result["status"] == "FAST93_HOLDOUT_INSUFFICIENT_TRADES"
    ]
    no_survivor = [
        result for result in holdout_results if result["status"] == "FAST93_NO_STRICT_TRAINING_SURVIVOR"
    ]
    stress_values = [float(result["holdout_summary"]["stress_total_bps"]) for result in holdout_results]
    trade_counts = [int(result["holdout_summary"]["trade_count"]) for result in holdout_results]
    return {
        "holdout_count": len(holdout_results),
        "positive_holdout_count": len(positive),
        "negative_holdout_count": len(negative),
        "insufficient_holdout_count": len(insufficient),
        "no_strict_training_survivor_count": len(no_survivor),
        "holdout_trade_count": sum(trade_counts),
        "holdout_stress_total_bps": float(sum(stress_values)),
        "worst_holdout_stress_total_bps": min(stress_values) if stress_values else None,
        "best_holdout_stress_total_bps": max(stress_values) if stress_values else None,
        "positive_holdout_labels": [result["holdout_label"] for result in positive],
        "negative_holdout_labels": [result["holdout_label"] for result in negative],
        "insufficient_holdout_labels": [result["holdout_label"] for result in insufficient],
        "no_survivor_holdout_labels": [result["holdout_label"] for result in no_survivor],
    }


def fast93_status(aggregate: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "leave_one_label_out_trailing_payoff_gate_audit",
    ]
    holdout_count = int(aggregate["holdout_count"])
    if holdout_count == 0:
        return "FAST93_NO_HOLDOUT_LABELS", reasons
    if int(aggregate["no_strict_training_survivor_count"]) == holdout_count:
        return "FAST93_NO_STRICT_TRAINING_SURVIVORS", reasons + [
            (
                "no_gate_positive_in_training_under_negative_label_limit:"
                f"{aggregate.get('maximum_train_negative_labels', 0)}"
            )
        ]
    if int(aggregate["positive_holdout_count"]) == holdout_count:
        return "FAST93_TRAILING_PAYOFF_GATE_SURVIVOR_DIAGNOSTIC_ONLY", reasons + [
            "all_holdouts_positive",
            "requires_new_preregistration",
        ]
    return "FAST93_TRAILING_PAYOFF_GATE_FAILED", reasons + [
        f"positive_holdouts:{aggregate['positive_holdout_count']}/{holdout_count}",
        f"negative_holdouts:{aggregate['negative_holdout_count']}",
        f"insufficient_holdouts:{aggregate['insufficient_holdout_count']}",
        f"no_strict_training_survivors:{aggregate['no_strict_training_survivor_count']}",
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


def candidate_spec_to_dict(spec: CandidateSpec) -> dict[str, Any]:
    return {
        "conflict_policy": spec.conflict_policy,
        "gate": gate_to_dict(spec.gate),
    }


def candidate_spec_from_dict(payload: dict[str, Any]) -> CandidateSpec:
    return CandidateSpec(
        conflict_policy=str(payload["conflict_policy"]),
        gate=gate_from_dict(payload["gate"]),
    )


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
