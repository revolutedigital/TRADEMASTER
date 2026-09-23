"""Fast94 tail-risk utility model audit.

Fast93 showed that simple causal gates do not find stable entries that pay the
Fast57/Fast60 trailing policy. Fast94 tests the next natural hypothesis: train a
model directly on trailing payoff utility, with tail-risk penalties, and select
signals by a frozen score cutoff learned from training labels only.

This is research-only. Positive diagnostics would require a new preregistration;
this script never approves Testnet or LIVE execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


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
    / "fast94-tail-risk-utility-model-audit"
)
FEATURE_NAMES = (
    "probability",
    "threshold_distance",
    "realized_vol_30s_bps",
    "realized_vol_5s_bps",
    "quote_volume_30s",
    "quote_volume_5s",
    "trade_count_30s",
    "trade_count_5s",
    "mean_interarrival_ms_30s",
    "mean_interarrival_ms_5s",
    "directed_flow_imbalance_1s",
    "directed_flow_imbalance_5s",
    "directed_flow_imbalance_30s",
    "directed_return_1s_bps",
    "directed_return_5s_bps",
    "directed_return_30s_bps",
    "hour_sin",
    "hour_cos",
    "side_sign",
)
TOP_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)
MINIMUM_TRAIN_TRADES = 10
MINIMUM_TRAIN_LABELS = 3
MAXIMUM_TRAIN_NEGATIVE_LABELS = 2

ModelKind = Literal["regressor", "classifier"]
TargetName = Literal["stress", "tail05", "tail10", "positive_safe"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_name: TargetName
    estimator: BaseEstimator
    kind: ModelKind


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    report = run_fast94_tail_risk_utility_model_audit(
        input_path=arguments.input_path,
        output_root=arguments.output_root,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast94_tail_risk_utility_model_audit(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_results = evaluate_candidates(rows)
    candidate_results = sorted(
        candidate_results,
        key=lambda candidate: (
            int(candidate["aggregate"]["positive_holdout_count"]),
            float(candidate["aggregate"]["holdout_stress_total_bps"]),
            int(candidate["aggregate"]["holdout_trade_count"]),
        ),
        reverse=True,
    )
    status, status_reasons = fast94_status(candidate_results)

    candidate_results_path = output_root / "candidate_results.json"
    candidate_results_path.write_text(
        json.dumps(candidate_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            "fast94_is_leave_one_label_out_tail_risk_utility_model_audit",
            "score_cutoff_is_learned_on_training_labels_only",
            "uses_only_causal_pre_trade_features",
            "positive_result_would_require_new_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast94_tail_risk_utility_model_audit",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "row_count": len(rows),
        "label_count": len({str(row["label"]) for row in rows}),
        "feature_names": list(FEATURE_NAMES),
        "top_fractions": list(TOP_FRACTIONS),
        "model_specs": [
            {"name": spec.name, "target_name": spec.target_name, "kind": spec.kind}
            for spec in model_specs()
        ],
        "candidate_count": len(candidate_results),
        "candidate_results_path": str(candidate_results_path),
        "candidate_results_sha256": _sha256_file(candidate_results_path),
        "top_candidates": [
            summarize_candidate_for_report(candidate)
            for candidate in candidate_results[:10]
        ],
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def evaluate_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = sorted({str(row["label"]) for row in rows})
    results: list[dict[str, Any]] = []
    for spec in model_specs():
        for top_fraction in TOP_FRACTIONS:
            holdout_results = [
                evaluate_holdout(
                    rows=rows,
                    labels=labels,
                    holdout_label=holdout_label,
                    spec=spec,
                    top_fraction=top_fraction,
                )
                for holdout_label in labels
            ]
            aggregate = aggregate_holdouts(holdout_results)
            results.append(
                {
                    "model_name": spec.name,
                    "target_name": spec.target_name,
                    "model_kind": spec.kind,
                    "top_fraction": top_fraction,
                    "aggregate": aggregate,
                    "holdout_results": holdout_results,
                }
            )
    return results


def evaluate_holdout(
    *,
    rows: list[dict[str, Any]],
    labels: list[str],
    holdout_label: str,
    spec: ModelSpec,
    top_fraction: float,
) -> dict[str, Any]:
    train_rows = [row for row in rows if str(row["label"]) != holdout_label]
    holdout_rows = [row for row in rows if str(row["label"]) == holdout_label]
    train_targets = np.array([target_value(row, spec.target_name) for row in train_rows])
    if spec.kind == "classifier" and len(set(float(value) for value in train_targets)) < 2:
        return empty_holdout_result(
            holdout_label,
            status="FAST94_TRAINING_TARGET_SINGLE_CLASS",
            labels=labels,
        )

    estimator = clone(spec.estimator)
    estimator.fit(feature_matrix(train_rows), train_targets)
    train_scores = score_rows(estimator, feature_matrix(train_rows), spec.kind)
    holdout_scores = score_rows(estimator, feature_matrix(holdout_rows), spec.kind)
    cutoff = score_cutoff(train_scores, top_fraction)
    train_selected = attach_scores_above_cutoff(train_rows, train_scores, cutoff)
    holdout_selected = attach_scores_above_cutoff(holdout_rows, holdout_scores, cutoff)
    train_one_position = apply_one_position(train_selected)
    holdout_one_position = apply_one_position(holdout_selected)
    train_summary = summarize_rows(train_one_position)
    holdout_summary = summarize_rows(holdout_one_position)
    return {
        "holdout_label": holdout_label,
        "status": holdout_status(holdout_summary),
        "score_cutoff": cutoff,
        "train_summary": train_summary,
        "train_survived": training_summary_survives(train_summary),
        "holdout_summary": holdout_summary,
    }


def model_specs() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            name="ridge_stress",
            target_name="stress",
            estimator=make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            kind="regressor",
        ),
        ModelSpec(
            name="ridge_tail05",
            target_name="tail05",
            estimator=make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            kind="regressor",
        ),
        ModelSpec(
            name="ridge_tail10",
            target_name="tail10",
            estimator=make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            kind="regressor",
        ),
        ModelSpec(
            name="hgb_stress",
            target_name="stress",
            estimator=HistGradientBoostingRegressor(
                max_iter=80,
                learning_rate=0.05,
                l2_regularization=1.0,
                min_samples_leaf=20,
                random_state=7,
            ),
            kind="regressor",
        ),
        ModelSpec(
            name="hgb_tail05",
            target_name="tail05",
            estimator=HistGradientBoostingRegressor(
                max_iter=80,
                learning_rate=0.05,
                l2_regularization=1.0,
                min_samples_leaf=20,
                random_state=7,
            ),
            kind="regressor",
        ),
        ModelSpec(
            name="hgb_tail10",
            target_name="tail10",
            estimator=HistGradientBoostingRegressor(
                max_iter=80,
                learning_rate=0.05,
                l2_regularization=1.0,
                min_samples_leaf=20,
                random_state=7,
            ),
            kind="regressor",
        ),
        ModelSpec(
            name="logit_positive_safe",
            target_name="positive_safe",
            estimator=make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=500, class_weight="balanced"),
            ),
            kind="classifier",
        ),
        ModelSpec(
            name="hgb_positive_safe",
            target_name="positive_safe",
            estimator=HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.05,
                l2_regularization=1.0,
                min_samples_leaf=20,
                random_state=7,
            ),
            kind="classifier",
        ),
    )


def target_value(row: dict[str, Any], target_name: TargetName) -> float:
    stress = float(row["stress_net_bps"])
    maximum_adverse = float(row["maximum_adverse_bps"])
    initial_stop_penalty = 1.0 if str(row["exit_reason"]) == "INITIAL_STOP" else 0.0
    if target_name == "stress":
        return stress
    if target_name == "tail05":
        return stress - 0.5 * max(0.0, -maximum_adverse - 5.0) - 5.0 * initial_stop_penalty
    if target_name == "tail10":
        return stress - 1.0 * max(0.0, -maximum_adverse - 5.0) - 8.0 * initial_stop_penalty
    if target_name == "positive_safe":
        return float(stress > 0 and maximum_adverse > -8.0 and not initial_stop_penalty)
    raise ValueError(f"unknown target: {target_name}")


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[feature_value(row, feature_name) for feature_name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )


def feature_value(row: dict[str, Any], feature_name: str) -> float:
    if feature_name == "threshold_distance":
        return float(row["threshold"]) - float(row["probability"])
    if feature_name in row and isinstance(row[feature_name], int | float):
        return float(row[feature_name])
    causal_features = row.get("causal_feature_values") or {}
    raw_value = causal_features.get(feature_name, 0.0)
    value = float(raw_value or 0.0)
    return value if math.isfinite(value) else 0.0


def score_rows(estimator: BaseEstimator, features: np.ndarray, kind: ModelKind) -> np.ndarray:
    if not len(features):
        return np.array([], dtype=float)
    if kind == "classifier":
        return estimator.predict_proba(features)[:, 1]
    return estimator.predict(features)


def score_cutoff(scores: np.ndarray, top_fraction: float) -> float:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    return float(np.quantile(scores, 1.0 - top_fraction))


def attach_scores_above_cutoff(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    cutoff: float,
) -> list[dict[str, Any]]:
    selected = []
    for row, score in zip(rows, scores, strict=True):
        if float(score) >= cutoff:
            selected.append({**row, "_fast94_score": float(score)})
    return selected


def apply_one_position(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    position_exit_ms = -1
    for row in sorted(
        rows,
        key=lambda item: (int(item["decision_time_ms"]), -float(item["_fast94_score"])),
    ):
        if int(row["decision_time_ms"]) <= position_exit_ms:
            continue
        selected.append(row)
        position_exit_ms = int(row["exit_time_ms"])
    return selected


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return empty_summary()
    label_stress: defaultdict[str, float] = defaultdict(float)
    label_expected: defaultdict[str, float] = defaultdict(float)
    stress_total = 0.0
    expected_total = 0.0
    winners = 0
    drawdown = 0.0
    equity = 0.0
    peak = 0.0
    for row in rows:
        stress = float(row["stress_net_bps"])
        expected = float(row["expected_net_bps"])
        label = str(row["label"])
        stress_total += stress
        expected_total += expected
        label_stress[label] += stress
        label_expected[label] += expected
        winners += stress > 0
        equity += stress
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "trade_count": len(rows),
        "label_count": len(label_stress),
        "stress_total_bps": stress_total,
        "stress_mean_bps": stress_total / len(rows),
        "expected_total_bps": expected_total,
        "expected_mean_bps": expected_total / len(rows),
        "expected_win_rate": winners / len(rows),
        "maximum_drawdown_bps": drawdown,
        "positive_label_count": sum(value > 0 for value in label_stress.values()),
        "negative_label_count": sum(value < 0 for value in label_stress.values()),
        "worst_label_stress_total_bps": min(label_stress.values()),
        "label_summaries": {
            label: {
                "trade_count": sum(1 for row in rows if str(row["label"]) == label),
                "stress_total_bps": stress_value,
                "expected_total_bps": label_expected[label],
            }
            for label, stress_value in sorted(label_stress.items())
        },
    }


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


def training_summary_survives(summary: dict[str, Any]) -> bool:
    return (
        int(summary["trade_count"]) >= MINIMUM_TRAIN_TRADES
        and int(summary["label_count"]) >= MINIMUM_TRAIN_LABELS
        and float(summary["stress_total_bps"]) > 0
        and int(summary["negative_label_count"]) <= MAXIMUM_TRAIN_NEGATIVE_LABELS
    )


def holdout_status(summary: dict[str, Any]) -> str:
    if int(summary["trade_count"]) == 0:
        return "FAST94_HOLDOUT_EMPTY"
    if float(summary["stress_total_bps"]) <= 0:
        return "FAST94_HOLDOUT_STRESS_NON_POSITIVE"
    return "FAST94_HOLDOUT_POSITIVE"


def empty_holdout_result(
    holdout_label: str,
    *,
    status: str,
    labels: list[str],
) -> dict[str, Any]:
    _ = labels
    return {
        "holdout_label": holdout_label,
        "status": status,
        "score_cutoff": None,
        "train_summary": empty_summary(),
        "train_survived": False,
        "holdout_summary": empty_summary(),
    }


def aggregate_holdouts(holdout_results: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [result for result in holdout_results if result["status"] == "FAST94_HOLDOUT_POSITIVE"]
    negative = [
        result for result in holdout_results if result["status"] == "FAST94_HOLDOUT_STRESS_NON_POSITIVE"
    ]
    empty = [result for result in holdout_results if result["status"] == "FAST94_HOLDOUT_EMPTY"]
    single_class = [
        result for result in holdout_results if result["status"] == "FAST94_TRAINING_TARGET_SINGLE_CLASS"
    ]
    summaries = [result["holdout_summary"] for result in holdout_results]
    train_survivors = [result for result in holdout_results if bool(result["train_survived"])]
    return {
        "holdout_count": len(holdout_results),
        "positive_holdout_count": len(positive),
        "negative_holdout_count": len(negative),
        "empty_holdout_count": len(empty),
        "single_class_holdout_count": len(single_class),
        "train_survivor_count": len(train_survivors),
        "holdout_trade_count": sum(int(summary["trade_count"]) for summary in summaries),
        "holdout_stress_total_bps": float(sum(float(summary["stress_total_bps"]) for summary in summaries)),
        "worst_holdout_stress_total_bps": min(
            (float(summary["stress_total_bps"]) for summary in summaries),
            default=0.0,
        ),
        "positive_holdout_labels": [result["holdout_label"] for result in positive],
        "negative_holdout_labels": [result["holdout_label"] for result in negative],
        "empty_holdout_labels": [result["holdout_label"] for result in empty],
    }


def fast94_status(candidate_results: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "leave_one_label_out_tail_risk_utility_model_audit",
    ]
    if not candidate_results:
        return "FAST94_NO_CANDIDATES", reasons
    survivors = [
        candidate
        for candidate in candidate_results
        if int(candidate["aggregate"]["positive_holdout_count"])
        == int(candidate["aggregate"]["holdout_count"])
        and int(candidate["aggregate"]["negative_holdout_count"]) == 0
        and int(candidate["aggregate"]["empty_holdout_count"]) == 0
    ]
    if survivors:
        return "FAST94_UTILITY_MODEL_SURVIVOR_DIAGNOSTIC_ONLY", reasons + [
            f"survivor_count:{len(survivors)}",
            "requires_new_preregistration",
        ]
    best = summarize_candidate_for_report(candidate_results[0])
    return "FAST94_UTILITY_MODEL_FAILED_LOLO", reasons + [
        f"best_model:{best['model_name']}",
        f"best_top_fraction:{best['top_fraction']}",
        f"best_positive_holdouts:{best['positive_holdout_count']}/{best['holdout_count']}",
        f"best_stress_total_bps:{best['holdout_stress_total_bps']}",
    ]


def summarize_candidate_for_report(candidate: dict[str, Any]) -> dict[str, Any]:
    aggregate = candidate["aggregate"]
    return {
        "model_name": candidate["model_name"],
        "target_name": candidate["target_name"],
        "model_kind": candidate["model_kind"],
        "top_fraction": candidate["top_fraction"],
        "holdout_count": aggregate["holdout_count"],
        "train_survivor_count": aggregate["train_survivor_count"],
        "holdout_trade_count": aggregate["holdout_trade_count"],
        "positive_holdout_count": aggregate["positive_holdout_count"],
        "negative_holdout_count": aggregate["negative_holdout_count"],
        "empty_holdout_count": aggregate["empty_holdout_count"],
        "holdout_stress_total_bps": aggregate["holdout_stress_total_bps"],
        "worst_holdout_stress_total_bps": aggregate["worst_holdout_stress_total_bps"],
    }


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
        "candidate_count": report["candidate_count"],
        "top_candidates": report["top_candidates"][:5],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
