"""Fast139: frozen rare-setup audit on the broad near-threshold universe.

Fast79/Fast80 and Fast68 produced rare microstructure setup hypotheses. Fast139
does not search for new thresholds. It applies those frozen rules to the settled
Fast92 all-local broad universe and checks whether they fire, survive costs, and
cover late windows. This is diagnostic research only and never touches trading
engine, credentials, ledger, Demo, Testnet or LIVE accounts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
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
    / "fast139-frozen-rare-setup-broad-audit"
)
FAST80_REALIZED_VOL_30S_MIN = 45.38807193200911
FAST80_FLOW_IMBALANCE_5S_MAX = -0.2193072079643347
FAST68_QUOTE_VOLUME_30S_MIN = 79775575.91570444
FAST68_REALIZED_VOL_30S_MAX = 351.4988811749231
FAST68_DIRECTED_FLOW_IMBALANCE_30S_MIN = -0.4391006273798949
LATE_FAST_NUMBER_START = 82
MINIMUM_LATE_TRADES = 5
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
class FrozenRule:
    name: str
    source: str
    probability_mode: str
    description: str


FROZEN_RULES = (
    FrozenRule(
        name="fast80_rare_vol_flow_near200",
        source="fast80",
        probability_mode="near200",
        description=(
            "Fast80 rare setup on broad near-threshold rows: "
            "realized_vol_30s_bps >= 45.38807193200911 and "
            "flow_imbalance_5s <= -0.2193072079643347."
        ),
    ),
    FrozenRule(
        name="fast80_rare_vol_flow_top_p_only",
        source="fast80",
        probability_mode="top_p_only",
        description=(
            "Fast80 rare setup restricted to original top-p rows: "
            "probability >= threshold plus rare vol/flow rule."
        ),
    ),
    FrozenRule(
        name="fast68_directional_gate_near200",
        source="fast68",
        probability_mode="near200",
        description=(
            "Fast68 best directional liquidity/volatility gate on broad near-threshold rows."
        ),
    ),
    FrozenRule(
        name="fast68_directional_gate_top_p_only",
        source="fast68",
        probability_mode="top_p_only",
        description=(
            "Fast68 best directional liquidity/volatility gate restricted to original top-p rows."
        ),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--minimum-late-trades", type=int, default=MINIMUM_LATE_TRADES)
    arguments = parser.parse_args(argv)
    report = run(
        input_path=arguments.input_path,
        output_root=arguments.output_root,
        minimum_late_trades=arguments.minimum_late_trades,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    minimum_late_trades: int = MINIMUM_LATE_TRADES,
) -> dict[str, Any]:
    if minimum_late_trades < 1:
        raise ValueError("minimum_late_trades must be positive")
    rows = load_jsonl(input_path)
    output_root.mkdir(parents=True, exist_ok=True)
    evaluations = [evaluate_rule(rule, rows) for rule in FROZEN_RULES]
    status, status_reasons = fast139_status(evaluations, minimum_late_trades=minimum_late_trades)

    variant_rows = [
        {"variant": evaluation["rule"]["name"], **row}
        for evaluation in evaluations
        for row in evaluation["one_position_rows"]
    ]
    variant_rows_path = output_root / "variant-one-position-rows.jsonl"
    _write_jsonl(variant_rows_path, variant_rows)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast139_applies_frozen_rare_setup_rules_without_new_threshold_search",
            "broad_near_threshold_universe_is_diagnostic_not_deployable",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast139_frozen_rare_setup_broad_audit",
        "status": status,
        "status_reasons": status_reasons,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_row_count": len(rows),
        "late_fast_number_start": LATE_FAST_NUMBER_START,
        "minimum_late_trades": minimum_late_trades,
        "frozen_rules": [asdict(rule) for rule in FROZEN_RULES],
        "rule_evaluations": [strip_rows(evaluation) for evaluation in evaluations],
        "variant_rows_path": str(variant_rows_path),
        "variant_rows_sha256": sha256_file(variant_rows_path),
    }
    return write_report(output_root, report)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_rule(rule: FrozenRule, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if rule_matches(rule, row)]
    one_position_rows = apply_one_position_by_label(selected)
    late_rows = [row for row in one_position_rows if is_late_label(str(row.get("label", "")))]
    return {
        "rule": asdict(rule),
        "raw_selected_count": len(selected),
        "one_position_rows": one_position_rows,
        "one_position_summary": summarize_variant_rows(one_position_rows),
        "late_one_position_rows": late_rows,
        "late_one_position_summary": summarize_variant_rows(late_rows),
    }


def apply_one_position_by_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for _label, label_rows in group_by_label(rows).items():
        selected.extend(_apply_one_position_constraint(label_rows))
    return sorted(selected, key=lambda row: (str(row["label"]), int(row["decision_time_ms"])))


def rule_matches(rule: FrozenRule, row: dict[str, Any]) -> bool:
    if rule.probability_mode == "top_p_only" and float(row["probability"]) < float(row["threshold"]):
        return False
    if rule.source == "fast80":
        return fast80_rule_matches(row)
    if rule.source == "fast68":
        return fast68_rule_matches(row)
    raise ValueError(f"unknown frozen rule source: {rule.source}")


def fast80_rule_matches(row: dict[str, Any]) -> bool:
    realized_vol = feature_value(row, "realized_vol_30s_bps")
    flow = feature_value(row, "flow_imbalance_5s")
    return (
        realized_vol is not None
        and flow is not None
        and realized_vol >= FAST80_REALIZED_VOL_30S_MIN
        and flow <= FAST80_FLOW_IMBALANCE_5S_MAX
    )


def fast68_rule_matches(row: dict[str, Any]) -> bool:
    quote_volume = feature_value(row, "quote_volume_30s")
    realized_vol = feature_value(row, "realized_vol_30s_bps")
    directed_flow = feature_value(row, "directed_flow_imbalance_30s")
    return (
        quote_volume is not None
        and realized_vol is not None
        and directed_flow is not None
        and quote_volume >= FAST68_QUOTE_VOLUME_30S_MIN
        and FAST80_REALIZED_VOL_30S_MIN <= realized_vol <= FAST68_REALIZED_VOL_30S_MAX
        and directed_flow >= FAST68_DIRECTED_FLOW_IMBALANCE_30S_MIN
    )


def feature_value(row: dict[str, Any], name: str) -> float | None:
    if row.get(name) is not None:
        return float(row[name])
    causal = row.get("causal_feature_values", {})
    if isinstance(causal, dict) and causal.get(name) is not None:
        return float(causal[name])
    return None


def is_late_label(label: str) -> bool:
    match = re.search(r"fast(?P<number>\d+)", label)
    return bool(match and int(match.group("number")) >= LATE_FAST_NUMBER_START)


def fast139_status(
    evaluations: list[dict[str, Any]],
    *,
    minimum_late_trades: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "frozen_rare_setup_broad_audit"]
    late_candidates = [
        evaluation
        for evaluation in evaluations
        if evaluation["late_one_position_summary"]["trade_count"] >= minimum_late_trades
    ]
    if not late_candidates:
        return (
            "FAST139_FROZEN_RARE_SETUP_NO_LATE_AUDIT_COVERAGE",
            reasons + [f"no_rule_reached_minimum_late_trades:{minimum_late_trades}"],
        )
    survivors = [
        evaluation
        for evaluation in late_candidates
        if evaluation["late_one_position_summary"]["stress_total_bps"] > 0
        and evaluation["late_one_position_summary"]["negative_label_count"] == 0
    ]
    if survivors:
        return (
            "FAST139_FROZEN_RARE_SETUP_DIAGNOSTIC_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"survivor_count:{len(survivors)}",
                "broad_near_threshold_universe_is_post_outcome_diagnostic",
                "fresh_preregistered_prospective_observation_required",
            ],
        )
    return (
        "FAST139_FROZEN_RARE_SETUP_FAILED_LATE_AUDIT",
        reasons + ["late_rules_had_negative_stress_or_negative_labels"],
    )


def strip_rows(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"one_position_rows", "late_one_position_rows"}
    }


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
        "rule_evaluations": [
            {
                "rule": evaluation["rule"]["name"],
                "raw_selected_count": evaluation["raw_selected_count"],
                "one_position_trade_count": evaluation["one_position_summary"]["trade_count"],
                "one_position_stress_total_bps": evaluation["one_position_summary"]["stress_total_bps"],
                "late_trade_count": evaluation["late_one_position_summary"]["trade_count"],
                "late_stress_total_bps": evaluation["late_one_position_summary"]["stress_total_bps"],
            }
            for evaluation in report["rule_evaluations"]
        ],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
