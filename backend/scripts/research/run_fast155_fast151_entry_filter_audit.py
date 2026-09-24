"""Fast155: entry-filter audit for the Fast151 operating policy.

The user hypothesis is that a probability/top-p style entry filter might
improve the frozen candidate. This audit tests a small, pre-declared family of
causal entry-quality filters using only fields known at entry time:

* breakout distance in pips;
* breakout distance as R;
* daily ATR/risk regime.

Each filter is calibrated on the pre-2019 sample only, applied before the
Fast151 position cap and circuit breaker, then audited on 2019-2026. If no
filter improves the pre-selected operating-policy gate without failing current,
Fast151 remains the frozen candidate.

Historical research only. No orders, no ledger, no Demo/Testnet/LIVE.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import (  # noqa: E402
    DEFAULT_CURRENT_INPUT_PATH,
    DEFAULT_PRE_INPUT_PATH,
    PRE_SAMPLE_LABEL,
    TARGET_POLICY_ID,
    filtered_sample,
    summarize_sample,
)
from scripts.research.run_fast149_usd_quoted_basket_position_cap_audit import (  # noqa: E402
    PositionOverlay,
    apply_position_overlay,
)
from scripts.research.run_fast150_usd_quoted_basket_circuit_breaker_audit import (  # noqa: E402
    CircuitBreakerPolicy,
    apply_drawdown_circuit_breaker,
)
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast155" / "fast151-entry-filter-audit"
SELECTED_POSITION_OVERLAY = PositionOverlay("max_per_pair_1", max_total_open=None, max_open_per_pair=1)
SELECTED_CIRCUIT_BREAKER = CircuitBreakerPolicy(
    "dd800_cooldown180d",
    closed_equity_drawdown_threshold_pips=800.0,
    cooldown_days=180,
)
MIN_PRE_TRADES = 50
MIN_CURRENT_TRADES = 50
MAX_DRAWDOWN_PIPS = 1000.0
MAX_CONCURRENT_TRADES = 4


@dataclass(frozen=True)
class EntryFilter:
    identifier: str
    feature: str | None = None
    operator: str = "all"
    threshold: float | None = None
    lower_threshold: float | None = None
    upper_threshold: float | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-input-path", type=Path, default=DEFAULT_PRE_INPUT_PATH)
    parser.add_argument("--current-input-path", type=Path, default=DEFAULT_CURRENT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(
        pre_input_path=arguments.pre_input_path,
        current_input_path=arguments.current_input_path,
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
    *,
    pre_input_path: Path = DEFAULT_PRE_INPUT_PATH,
    current_input_path: Path = DEFAULT_CURRENT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    pre_source = pd.read_parquet(pre_input_path)
    current_source = pd.read_parquet(current_input_path)
    pre_base = add_entry_features(filtered_sample(pre_source, sample_column="sample", sample_value=PRE_SAMPLE_LABEL))
    current_base = add_entry_features(filtered_sample(current_source, sample_column=None, sample_value=None))
    filters = entry_filter_grid(pre_base)
    evaluations = [evaluate_filter(pre_base, current_base, entry_filter) for entry_filter in filters]
    selected = select_filter(evaluations)
    status, status_reasons = fast155_status(selected)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "entry-filter-metrics.csv"
    summary_path = output_root / "entry-filter-summary.json"
    pd.DataFrame(flat_metric_rows(evaluations)).to_csv(metrics_path, index=False)
    summary = {
        "selected_filter": selected,
        "evaluations": evaluations,
        "thresholds": {
            "min_pre_trades": MIN_PRE_TRADES,
            "min_current_trades": MIN_CURRENT_TRADES,
            "max_drawdown_pips": MAX_DRAWDOWN_PIPS,
            "max_concurrent_trades": MAX_CONCURRENT_TRADES,
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast155_is_historical_entry_filter_audit_not_strategy_approval",
            "filters_selected_on_pre_sample_only",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast155_fast151_entry_filter_audit",
        "status": status,
        "status_reasons": status_reasons,
        "target_policy_id": TARGET_POLICY_ID,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "pre_input_path": str(pre_input_path),
        "pre_input_sha256": sha256_file(pre_input_path),
        "current_input_path": str(current_input_path),
        "current_input_sha256": sha256_file(current_input_path),
        "summary": summary,
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "entry_filter_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def add_entry_features(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    frame["breakout_r"] = frame["breakout_distance_pips"].astype("float64") / frame["risk_pips"].astype("float64")
    frame["atr_pips"] = frame["daily_atr_pips"].astype("float64")
    frame["risk_pips_float"] = frame["risk_pips"].astype("float64")
    return frame


def entry_filter_grid(pre_base: pd.DataFrame) -> tuple[EntryFilter, ...]:
    quantiles = {
        "breakout_distance_pips": pre_base["breakout_distance_pips"].quantile([0.50, 0.75, 0.90]).to_dict(),
        "breakout_r": pre_base["breakout_r"].quantile([0.50, 0.75, 0.90]).to_dict(),
        "atr_pips": pre_base["atr_pips"].quantile([0.25, 0.50, 0.75, 0.90]).to_dict(),
        "risk_pips_float": pre_base["risk_pips_float"].quantile([0.75]).to_dict(),
    }
    return (
        EntryFilter("no_extra_filter"),
        EntryFilter(
            "breakout_distance_lte_pre_p50",
            feature="breakout_distance_pips",
            operator="lte",
            threshold=float(quantiles["breakout_distance_pips"][0.50]),
        ),
        EntryFilter(
            "breakout_distance_lte_pre_p75",
            feature="breakout_distance_pips",
            operator="lte",
            threshold=float(quantiles["breakout_distance_pips"][0.75]),
        ),
        EntryFilter(
            "breakout_distance_lte_pre_p90",
            feature="breakout_distance_pips",
            operator="lte",
            threshold=float(quantiles["breakout_distance_pips"][0.90]),
        ),
        EntryFilter(
            "breakout_r_lte_pre_p50",
            feature="breakout_r",
            operator="lte",
            threshold=float(quantiles["breakout_r"][0.50]),
        ),
        EntryFilter(
            "breakout_r_lte_pre_p75",
            feature="breakout_r",
            operator="lte",
            threshold=float(quantiles["breakout_r"][0.75]),
        ),
        EntryFilter(
            "breakout_r_lte_pre_p90",
            feature="breakout_r",
            operator="lte",
            threshold=float(quantiles["breakout_r"][0.90]),
        ),
        EntryFilter(
            "atr_gte_pre_p25",
            feature="atr_pips",
            operator="gte",
            threshold=float(quantiles["atr_pips"][0.25]),
        ),
        EntryFilter(
            "atr_gte_pre_p50",
            feature="atr_pips",
            operator="gte",
            threshold=float(quantiles["atr_pips"][0.50]),
        ),
        EntryFilter(
            "atr_lte_pre_p75",
            feature="atr_pips",
            operator="lte",
            threshold=float(quantiles["atr_pips"][0.75]),
        ),
        EntryFilter(
            "atr_lte_pre_p90",
            feature="atr_pips",
            operator="lte",
            threshold=float(quantiles["atr_pips"][0.90]),
        ),
        EntryFilter(
            "atr_between_pre_p25_p75",
            feature="atr_pips",
            operator="between",
            lower_threshold=float(quantiles["atr_pips"][0.25]),
            upper_threshold=float(quantiles["atr_pips"][0.75]),
        ),
        EntryFilter(
            "risk_lte_pre_p75",
            feature="risk_pips_float",
            operator="lte",
            threshold=float(quantiles["risk_pips_float"][0.75]),
        ),
    )


def filter_mask(trades: pd.DataFrame, entry_filter: EntryFilter) -> pd.Series:
    if entry_filter.operator == "all":
        return pd.Series(True, index=trades.index)
    if entry_filter.feature is None:
        raise ValueError("feature is required for non-all filters")
    values = trades[entry_filter.feature]
    if entry_filter.operator == "lte":
        if entry_filter.threshold is None:
            raise ValueError("threshold is required for lte filter")
        return values <= entry_filter.threshold
    if entry_filter.operator == "gte":
        if entry_filter.threshold is None:
            raise ValueError("threshold is required for gte filter")
        return values >= entry_filter.threshold
    if entry_filter.operator == "between":
        if entry_filter.lower_threshold is None or entry_filter.upper_threshold is None:
            raise ValueError("both thresholds are required for between filter")
        return (values >= entry_filter.lower_threshold) & (values <= entry_filter.upper_threshold)
    raise ValueError(f"unknown filter operator: {entry_filter.operator}")


def apply_fast151_controls(trades: pd.DataFrame) -> pd.DataFrame:
    position_accepted, _position_skipped = apply_position_overlay(trades, SELECTED_POSITION_OVERLAY)
    breaker_accepted, _breaker_skipped, _breaker_triggers = apply_drawdown_circuit_breaker(
        position_accepted,
        SELECTED_CIRCUIT_BREAKER,
    )
    return breaker_accepted


def evaluate_filter(
    pre_base: pd.DataFrame,
    current_base: pd.DataFrame,
    entry_filter: EntryFilter,
) -> dict[str, Any]:
    pre_filtered = pre_base.loc[filter_mask(pre_base, entry_filter)].copy()
    current_filtered = current_base.loc[filter_mask(current_base, entry_filter)].copy()
    pre_operational = apply_fast151_controls(pre_filtered)
    current_operational = apply_fast151_controls(current_filtered)
    pre_summary = summarize_sample(pre_operational)
    current_summary = summarize_sample(current_operational)
    pre_passed = passes_gate(pre_summary, min_trades=MIN_PRE_TRADES)
    current_passed = passes_gate(current_summary, min_trades=MIN_CURRENT_TRADES)
    return {
        "filter_id": entry_filter.identifier,
        "feature": entry_filter.feature,
        "operator": entry_filter.operator,
        "threshold": entry_filter.threshold,
        "lower_threshold": entry_filter.lower_threshold,
        "upper_threshold": entry_filter.upper_threshold,
        "pre_base_candidate_count": int(len(pre_filtered)),
        "current_base_candidate_count": int(len(current_filtered)),
        "pre": pre_summary,
        "current": current_summary,
        "pre_passed_gate": pre_passed,
        "current_passed_gate": current_passed,
        "both_passed_gate": pre_passed and current_passed,
    }


def passes_gate(summary: dict[str, Any], *, min_trades: int) -> bool:
    return bool(
        summary["trade_count"] >= min_trades
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["lower_stress_pips"] is not None
        and summary["lower_stress_pips"] > 0
        and summary["max_drawdown_pips"] is not None
        and abs(summary["max_drawdown_pips"]) <= MAX_DRAWDOWN_PIPS
        and summary["max_concurrent_trades"] <= MAX_CONCURRENT_TRADES
        and not summary["two_consecutive_negative_years"]
    )


def select_filter(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline = next(evaluation for evaluation in evaluations if evaluation["filter_id"] == "no_extra_filter")
    passed_pre = [evaluation for evaluation in evaluations if evaluation["pre_passed_gate"]]
    candidates = [
        evaluation
        for evaluation in passed_pre
        if evaluation["filter_id"] != "no_extra_filter"
        and evaluation["pre"]["lower_stress_pips"] > baseline["pre"]["lower_stress_pips"]
    ]
    if not candidates:
        return baseline
    return sorted(
        candidates,
        key=lambda item: (
            -item["pre"]["lower_stress_pips"],
            -item["pre"]["trade_count"],
            item["filter_id"],
        ),
    )[0]


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row = {
            key: value
            for key, value in evaluation.items()
            if key not in {"pre", "current"}
        }
        for prefix in ("pre", "current"):
            summary = evaluation[prefix]
            for key, value in summary.items():
                if key in {"yearly", "pair_summary", "exit_reason_summary"}:
                    continue
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def fast155_status(selected: dict[str, Any] | None) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "entry_filter_audit"]
    if selected is None:
        return "FAST155_ENTRY_FILTER_AUDIT_FAILED_SELECTION", reasons + ["no_selected_filter"]
    if selected["filter_id"] == "no_extra_filter":
        return (
            "FAST155_NO_ENTRY_FILTER_IMPROVES_FAST151_PRE_GATE",
            reasons
            + [
                "baseline_fast151_operating_policy_remains_selected",
                "simple_top_p_proxy_filters_rejected",
                "fresh_preregistered_prospective_observation_still_required",
            ],
        )
    if selected["current_passed_gate"]:
        return (
            "FAST155_ENTRY_FILTER_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"selected_filter:{selected['filter_id']}",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    return (
        "FAST155_ENTRY_FILTER_FAILED_CURRENT_AUDIT",
        reasons + [f"selected_filter_failed_current:{selected['filter_id']}"],
    )


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    selected = report["summary"]["selected_filter"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "operational_policy_id": report["operational_policy_id"],
        "selected_filter_id": selected["filter_id"] if selected else None,
        "selected_pre": {
            key: selected["pre"][key]
            for key in (
                "trade_count",
                "total_stress_pips",
                "mean_stress_pips",
                "lower_stress_pips",
                "max_drawdown_pips",
                "two_consecutive_negative_years",
            )
        }
        if selected
        else None,
        "selected_current": {
            key: selected["current"][key]
            for key in (
                "trade_count",
                "total_stress_pips",
                "mean_stress_pips",
                "lower_stress_pips",
                "max_drawdown_pips",
                "two_consecutive_negative_years",
            )
        }
        if selected
        else None,
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
