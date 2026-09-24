"""Fast140: walk-forward rare-setup search with a late holdout.

Fast139 killed previously frozen rare rules. Fast140 restarts the rare-setup
search correctly: select one/two-condition causal rules only on old broad
near-threshold windows (fast16..fast36), then audit the selected finalists on
late windows (fast82/fast83). This is still diagnostic research only: the broad
near-threshold universe is post-outcome research and no orders are sent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast61_fast60_shadow_settlement import (  # noqa: E402
    _apply_one_position_constraint,
    _write_jsonl,
)
from scripts.research.run_fast87_directional_filter_scout import (  # noqa: E402
    group_by_label,
    summarize_variant_rows,
)
from scripts.research.run_fast104_fixed_outlier_veto_audit import sha256_file  # noqa: E402

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
    / "fast140-walkforward-rare-setup-late-holdout"
)
TRAIN_FAST_NUMBERS = (16, 20, 21, 22, 24, 28, 29, 32, 35, 36)
LATE_FAST_NUMBERS = (82, 83)
DEFAULT_QUANTILES = (0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 0.90)
DEFAULT_ATOM_SEED_COUNT = 120
DEFAULT_FINALIST_COUNT = 100
DEFAULT_MIN_TRAIN_TRADES = 10
DEFAULT_MIN_TRAIN_LABELS = 3
DEFAULT_MIN_LATE_TRADES = 5
DEFAULT_MIN_LATE_LABELS = 2
FORBIDDEN_FEATURES = {
    "decision_time_ms",
    "entry_time_ms",
    "exit_time_ms",
    "entry_price",
    "exit_price",
    "expected_net_bps",
    "stress_net_bps",
    "gross_bps",
    "maximum_adverse_bps",
    "maximum_favorable_bps",
    "settlement_complete",
    "threshold",
    "horizon_seconds",
}
SAFETY_FLAGS = {
    "research_only": True,
    "order_submission_allowed": False,
    "execution_authorization": "none",
    "orders_sent": False,
    "committed": False,
    "dry_run": True,
    "approved_strategy": False,
}


@dataclass(frozen=True)
class Condition:
    feature: str
    direction: str
    threshold: float
    quantile: float

    @property
    def name(self) -> str:
        operator = ">=" if self.direction == "high" else "<="
        return f"{self.feature}{operator}q{self.quantile:g}:{self.threshold:.10g}"


@dataclass(frozen=True)
class Rule:
    conditions: tuple[Condition, ...]

    @property
    def name(self) -> str:
        return " AND ".join(condition.name for condition in self.conditions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--atom-seed-count", type=int, default=DEFAULT_ATOM_SEED_COUNT)
    parser.add_argument("--finalist-count", type=int, default=DEFAULT_FINALIST_COUNT)
    parser.add_argument("--min-train-trades", type=int, default=DEFAULT_MIN_TRAIN_TRADES)
    parser.add_argument("--min-train-labels", type=int, default=DEFAULT_MIN_TRAIN_LABELS)
    parser.add_argument("--min-late-trades", type=int, default=DEFAULT_MIN_LATE_TRADES)
    parser.add_argument("--min-late-labels", type=int, default=DEFAULT_MIN_LATE_LABELS)
    arguments = parser.parse_args(argv)
    report = run(
        input_path=arguments.input_path,
        output_root=arguments.output_root,
        atom_seed_count=arguments.atom_seed_count,
        finalist_count=arguments.finalist_count,
        min_train_trades=arguments.min_train_trades,
        min_train_labels=arguments.min_train_labels,
        min_late_trades=arguments.min_late_trades,
        min_late_labels=arguments.min_late_labels,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    atom_seed_count: int = DEFAULT_ATOM_SEED_COUNT,
    finalist_count: int = DEFAULT_FINALIST_COUNT,
    min_train_trades: int = DEFAULT_MIN_TRAIN_TRADES,
    min_train_labels: int = DEFAULT_MIN_TRAIN_LABELS,
    min_late_trades: int = DEFAULT_MIN_LATE_TRADES,
    min_late_labels: int = DEFAULT_MIN_LATE_LABELS,
) -> dict[str, Any]:
    validate_parameters(
        atom_seed_count=atom_seed_count,
        finalist_count=finalist_count,
        min_train_trades=min_train_trades,
        min_train_labels=min_train_labels,
        min_late_trades=min_late_trades,
        min_late_labels=min_late_labels,
    )
    rows = load_jsonl(input_path)
    train_rows = [row for row in rows if fast_number(str(row.get("label", ""))) in TRAIN_FAST_NUMBERS]
    late_rows = [row for row in rows if fast_number(str(row.get("label", ""))) in LATE_FAST_NUMBERS]
    feature_names = discover_feature_names(train_rows)
    atom_rules = [Rule((condition,)) for condition in condition_candidates(train_rows, feature_names)]
    atom_evaluations = [evaluate_rule(rule, train_rows, late_rows) for rule in atom_rules]
    eligible_atoms = eligible_train_rules(
        atom_evaluations,
        min_train_trades=min_train_trades,
        min_train_labels=min_train_labels,
    )
    eligible_atoms.sort(key=train_sort_key, reverse=True)
    seed_rules = [rule_from_dict(evaluation["rule"]) for evaluation in eligible_atoms[:atom_seed_count]]

    pair_rules = [Rule(tuple(sorted((left.conditions[0], right.conditions[0]), key=lambda item: item.name))) for left, right in combinations(seed_rules, 2)]
    pair_rules = dedupe_rules(pair_rules)
    pair_evaluations = [evaluate_rule(rule, train_rows, late_rows) for rule in pair_rules]
    all_evaluations = atom_evaluations + pair_evaluations
    eligible_rules = eligible_train_rules(
        all_evaluations,
        min_train_trades=min_train_trades,
        min_train_labels=min_train_labels,
    )
    eligible_rules.sort(key=train_sort_key, reverse=True)
    finalists = eligible_rules[:finalist_count]
    late_survivors = [
        evaluation
        for evaluation in finalists
        if is_late_survivor(
            evaluation,
            min_late_trades=min_late_trades,
            min_late_labels=min_late_labels,
        )
    ]
    status, status_reasons = fast140_status(
        finalist_count=len(finalists),
        late_survivor_count=len(late_survivors),
        finalists=finalists,
        min_late_trades=min_late_trades,
        min_late_labels=min_late_labels,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    atom_path = output_root / "atom-evaluations.json"
    pair_path = output_root / "pair-evaluations.json"
    finalists_path = output_root / "selected-finalists.json"
    late_survivors_path = output_root / "late-survivors.json"
    finalist_rows_path = output_root / "finalist-one-position-rows.jsonl"
    atom_path.write_text(json.dumps(strip_row_payloads(eligible_atoms[:250]), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pair_path.write_text(json.dumps(strip_row_payloads(pair_evaluations[:500]), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finalists_path.write_text(json.dumps(strip_row_payloads(finalists), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    late_survivors_path.write_text(json.dumps(strip_row_payloads(late_survivors), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(
        finalist_rows_path,
        [
            {"rule": evaluation["rule"]["name"], "sample": sample, **row}
            for evaluation in finalists
            for sample, rows_for_sample in (
                ("train", evaluation["train_one_position_rows"]),
                ("late", evaluation["late_one_position_rows"]),
            )
            for row in rows_for_sample
        ],
    )

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast140_selects_rare_setups_on_train_windows_and_audits_late_holdout",
            "broad_near_threshold_universe_is_post_outcome_diagnostic",
            "positive_late_result_would_still_require_fresh_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast140_walkforward_rare_setup_late_holdout",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_row_count": len(rows),
        "train_fast_numbers": list(TRAIN_FAST_NUMBERS),
        "late_fast_numbers": list(LATE_FAST_NUMBERS),
        "train_row_count": len(train_rows),
        "late_row_count": len(late_rows),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "atom_candidate_count": len(atom_rules),
        "eligible_atom_count": len(eligible_atoms),
        "seed_atom_count": len(seed_rules),
        "pair_candidate_count": len(pair_rules),
        "eligible_rule_count": len(eligible_rules),
        "finalist_count": len(finalists),
        "late_survivor_count": len(late_survivors),
        "minimums": {
            "min_train_trades": min_train_trades,
            "min_train_labels": min_train_labels,
            "min_late_trades": min_late_trades,
            "min_late_labels": min_late_labels,
        },
        "top_finalists": strip_row_payloads(finalists[:15]),
        "late_survivors": strip_row_payloads(late_survivors),
        "artifacts": {
            "atom_evaluations": {"path": str(atom_path), "sha256": sha256_file(atom_path)},
            "pair_evaluations": {"path": str(pair_path), "sha256": sha256_file(pair_path)},
            "finalists": {"path": str(finalists_path), "sha256": sha256_file(finalists_path)},
            "late_survivors": {"path": str(late_survivors_path), "sha256": sha256_file(late_survivors_path)},
            "finalist_rows": {"path": str(finalist_rows_path), "sha256": sha256_file(finalist_rows_path)},
        },
    }
    return write_report(output_root, report)


def validate_parameters(
    *,
    atom_seed_count: int,
    finalist_count: int,
    min_train_trades: int,
    min_train_labels: int,
    min_late_trades: int,
    min_late_labels: int,
) -> None:
    if atom_seed_count < 2:
        raise ValueError("atom_seed_count must be at least 2")
    if finalist_count < 1:
        raise ValueError("finalist_count must be positive")
    if min(min_train_trades, min_train_labels, min_late_trades, min_late_labels) < 1:
        raise ValueError("minimums must be positive")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    frame = pd.DataFrame(flatten_features(row) for row in rows)
    features = []
    for column in frame.columns:
        if column in FORBIDDEN_FEATURES:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < 30 or values.nunique(dropna=True) < 5:
            continue
        features.append(column)
    return sorted(features)


def flatten_features(row: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, bool):
            flattened[key] = int(value)
        elif isinstance(value, int | float):
            flattened[key] = value
    causal = row.get("causal_feature_values", {})
    if isinstance(causal, dict):
        for key, value in causal.items():
            if isinstance(value, int | float):
                flattened.setdefault(key, value)
    flattened["probability_margin"] = float(row["probability"]) - float(row["threshold"])
    return flattened


def condition_candidates(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> list[Condition]:
    frame = pd.DataFrame(flatten_features(row) for row in rows)
    candidates: list[Condition] = []
    for feature in feature_names:
        values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.nunique() < 5:
            continue
        for quantile in quantiles:
            threshold = float(values.quantile(quantile))
            candidates.append(Condition(feature=feature, direction="low", threshold=threshold, quantile=quantile))
            candidates.append(Condition(feature=feature, direction="high", threshold=threshold, quantile=quantile))
    return candidates


def evaluate_rule(rule: Rule, train_rows: list[dict[str, Any]], late_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_selected = [row for row in train_rows if rule_matches(rule, row)]
    late_selected = [row for row in late_rows if rule_matches(rule, row)]
    train_one_position = apply_one_position_by_label(train_selected)
    late_one_position = apply_one_position_by_label(late_selected)
    return {
        "rule": rule_to_dict(rule),
        "train_raw_selected_count": len(train_selected),
        "late_raw_selected_count": len(late_selected),
        "train_one_position_rows": train_one_position,
        "late_one_position_rows": late_one_position,
        "train_summary": summarize_variant_rows(train_one_position),
        "late_summary": summarize_variant_rows(late_one_position),
    }


def rule_matches(rule: Rule, row: dict[str, Any]) -> bool:
    features = flatten_features(row)
    for condition in rule.conditions:
        value = features.get(condition.feature)
        if value is None or not np.isfinite(float(value)):
            return False
        if condition.direction == "low" and float(value) > condition.threshold:
            return False
        if condition.direction == "high" and float(value) < condition.threshold:
            return False
    return True


def apply_one_position_by_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for _label, label_rows in group_by_label(rows).items():
        selected.extend(_apply_one_position_constraint(label_rows))
    return sorted(selected, key=lambda row: (str(row["label"]), int(row["decision_time_ms"])))


def eligible_train_rules(
    evaluations: list[dict[str, Any]],
    *,
    min_train_trades: int,
    min_train_labels: int,
) -> list[dict[str, Any]]:
    return [
        evaluation
        for evaluation in evaluations
        if evaluation["train_summary"]["trade_count"] >= min_train_trades
        and evaluation["train_summary"]["label_count"] >= min_train_labels
        and evaluation["train_summary"]["stress_total_bps"] > 0
    ]


def is_late_survivor(
    evaluation: dict[str, Any],
    *,
    min_late_trades: int,
    min_late_labels: int,
) -> bool:
    late = evaluation["late_summary"]
    return (
        late["trade_count"] >= min_late_trades
        and late["label_count"] >= min_late_labels
        and late["stress_total_bps"] > 0
        and late["negative_label_count"] == 0
    )


def train_sort_key(evaluation: dict[str, Any]) -> tuple[int, float, float, int, int]:
    summary = evaluation["train_summary"]
    return (
        -int(summary["negative_label_count"]),
        float(summary["stress_total_bps"]),
        float(summary["stress_mean_bps"]),
        int(summary["positive_label_count"]),
        int(summary["trade_count"]),
    )


def fast140_status(
    *,
    finalist_count: int,
    late_survivor_count: int,
    finalists: list[dict[str, Any]],
    min_late_trades: int,
    min_late_labels: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "walkforward_rare_setup_late_holdout"]
    if finalist_count == 0:
        return "FAST140_RARE_SETUP_FAILED_TRAIN_SELECTION", reasons + ["no_train_positive_finalists"]
    late_covered = [
        evaluation
        for evaluation in finalists
        if evaluation["late_summary"]["trade_count"] >= min_late_trades
        and evaluation["late_summary"]["label_count"] >= min_late_labels
    ]
    if not late_covered:
        return (
            "FAST140_RARE_SETUP_NO_LATE_AUDIT_COVERAGE",
            reasons + [f"no_finalist_reached_late_minimum:{min_late_trades}_trades_{min_late_labels}_labels"],
        )
    if late_survivor_count:
        return (
            "FAST140_RARE_SETUP_LATE_HOLDOUT_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"late_survivor_count:{late_survivor_count}",
                "broad_near_threshold_universe_is_post_outcome_diagnostic",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    return "FAST140_RARE_SETUP_FAILED_LATE_HOLDOUT", reasons + ["late_covered_finalists_failed_stress_or_label_gate"]


def dedupe_rules(rules: list[Rule]) -> list[Rule]:
    seen = set()
    deduped = []
    for rule in rules:
        if rule.name in seen:
            continue
        seen.add(rule.name)
        deduped.append(rule)
    return deduped


def fast_number(label: str) -> int | None:
    match = re.search(r"fast(?P<number>\d+)", label)
    return int(match.group("number")) if match else None


def rule_to_dict(rule: Rule) -> dict[str, Any]:
    return {
        "name": rule.name,
        "conditions": [asdict(condition) | {"name": condition.name} for condition in rule.conditions],
    }


def rule_from_dict(payload: dict[str, Any]) -> Rule:
    return Rule(tuple(condition_from_dict(item) for item in payload["conditions"]))


def condition_from_dict(payload: dict[str, Any]) -> Condition:
    return Condition(
        feature=str(payload["feature"]),
        direction=str(payload["direction"]),
        threshold=float(payload["threshold"]),
        quantile=float(payload["quantile"]),
    )


def strip_row_payloads(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in evaluation.items() if key not in {"train_one_position_rows", "late_one_position_rows"}}
        for evaluation in evaluations
    ]


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
        "feature_count": report["feature_count"],
        "atom_candidate_count": report["atom_candidate_count"],
        "pair_candidate_count": report["pair_candidate_count"],
        "eligible_rule_count": report["eligible_rule_count"],
        "finalist_count": report["finalist_count"],
        "late_survivor_count": report["late_survivor_count"],
        "top_finalists": report["top_finalists"][:5],
        "late_survivors": report["late_survivors"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
