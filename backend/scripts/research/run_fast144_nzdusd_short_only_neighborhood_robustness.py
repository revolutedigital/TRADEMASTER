"""Fast144: robustness audit for the NZDUSD short-only Donchian-fade clue.

Fast142 found the cleanest historical candidate so far:

    NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only

Fast144 asks whether this is only a single fitted point or whether nearby
variants carry the same signal. It reuses already materialized historical
artifacts:

* Fast137 pre-2019 temporal-reverse trades for the full candidate grid.
* Fast133 2019-2026 global-risk trades for the full current candidate grid.

This is historical research only. It writes offline artifacts and never touches
the trading engine, credentials, ledger, Demo, Testnet or LIVE accounts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    lower_confidence_bound,
    positive_year_share,
)

DEFAULT_PRE_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast137"
    / "fx-daily-donchian-fade-temporal-reverse"
    / "temporal-reverse-trades.parquet"
)
DEFAULT_CURRENT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast133"
    / "fx-daily-donchian-fade-global-risk"
    / "daily-donchian-fade-global-risk-trades.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast144"
    / "nzdusd-short-only-neighborhood-robustness"
)

PRE_SAMPLE_LABEL = "pre_2011_2018_selection"
CURRENT_SAMPLE_LABEL = "current_2019_2026_full_grid"
PAIR = "NZDUSD"
SIDE = "SHORT"
MANAGEMENT = "fixed_1r"
CHANNEL_DAYS = (20, 55, 100)
HOLD_DAYS = (5, 10, 20)
STOP_ATR_MULTIPLES = (1.0, 1.5, 2.0)
FROZEN_CHANNEL_DAYS = 100
FROZEN_HOLD_DAYS = 20
FROZEN_STOP_ATR_MULTIPLE = 2.0
FROZEN_ENTRY_POLICY_ID = "d1_donchian_fade_ch100_hold20"
FROZEN_RISK_POLICY_ID = "stop2_fixed_1r"
FROZEN_CANDIDATE_ID = f"{PAIR}:{FROZEN_ENTRY_POLICY_ID}:{FROZEN_RISK_POLICY_ID}"
FROZEN_POLICY_ID = f"{FROZEN_CANDIDATE_ID}:short_only"
MIN_SAMPLE_TRADES = 12
MIN_SAMPLE_YEARS = 3
MIN_POSITIVE_YEAR_SHARE = 2 / 3
LOCAL_CHANNEL_DAYS = (55, 100)
LOCAL_HOLD_DAYS = (10, 20)
LOCAL_STOP_ATR_MULTIPLES = (1.5, 2.0)
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


def entry_policy_id(channel_days: int, hold_days: int) -> str:
    return f"d1_donchian_fade_ch{channel_days}_hold{hold_days}"


def risk_policy_id(stop_atr_multiple: float) -> str:
    return f"stop{stop_atr_multiple:g}_fixed_1r"


def candidate_id(channel_days: int, hold_days: int, stop_atr_multiple: float) -> str:
    return f"{PAIR}:{entry_policy_id(channel_days, hold_days)}:{risk_policy_id(stop_atr_multiple)}"


def candidate_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    channel_index = {value: index for index, value in enumerate(CHANNEL_DAYS)}
    hold_index = {value: index for index, value in enumerate(HOLD_DAYS)}
    stop_index = {value: index for index, value in enumerate(STOP_ATR_MULTIPLES)}
    for channel_days in CHANNEL_DAYS:
        for hold_days in HOLD_DAYS:
            for stop_atr_multiple in STOP_ATR_MULTIPLES:
                identifier = candidate_id(channel_days, hold_days, stop_atr_multiple)
                grid_distance = (
                    abs(channel_index[channel_days] - channel_index[FROZEN_CHANNEL_DAYS])
                    + abs(hold_index[hold_days] - hold_index[FROZEN_HOLD_DAYS])
                    + abs(stop_index[stop_atr_multiple] - stop_index[FROZEN_STOP_ATR_MULTIPLE])
                )
                rows.append(
                    {
                        "candidate_id": identifier,
                        "pair": PAIR,
                        "entry_policy_id": entry_policy_id(channel_days, hold_days),
                        "risk_policy_id": risk_policy_id(stop_atr_multiple),
                        "channel_days": channel_days,
                        "hold_days": hold_days,
                        "stop_atr_multiple": stop_atr_multiple,
                        "management": MANAGEMENT,
                        "side": SIDE,
                        "grid_distance_from_frozen": grid_distance,
                        "is_exact_frozen_candidate": identifier == FROZEN_CANDIDATE_ID,
                        "is_local_neighborhood": (
                            channel_days in LOCAL_CHANNEL_DAYS
                            and hold_days in LOCAL_HOLD_DAYS
                            and stop_atr_multiple in LOCAL_STOP_ATR_MULTIPLES
                        ),
                        "is_one_step_neighbor": grid_distance <= 1,
                    }
                )
    return pd.DataFrame(rows)


def filtered_sample(
    frame: pd.DataFrame,
    *,
    sample_column: str | None,
    sample_value: str | None,
) -> pd.DataFrame:
    sample = frame.copy()
    if sample_column and sample_value:
        sample = sample.loc[sample[sample_column].eq(sample_value)].copy()
    sample["entry_timestamp"] = pd.to_datetime(sample["entry_timestamp"], utc=True)
    sample = sample.loc[
        sample["pair"].eq(PAIR)
        & sample["side"].eq(SIDE)
        & sample["management"].eq(MANAGEMENT)
        & sample["channel_days"].isin(CHANNEL_DAYS)
        & sample["hold_days"].isin(HOLD_DAYS)
        & sample["stop_atr_multiple"].isin(STOP_ATR_MULTIPLES)
    ].copy()
    return sample.sort_values(["candidate_id", "entry_timestamp"], ignore_index=True)


def top_positive_contribution_share(values: pd.Series) -> float | None:
    total = float(values.sum())
    if total <= 0:
        return None
    positive_values = values.loc[values > 0].sort_values(ascending=False)
    return float(positive_values.head(3).sum() / total)


def yearly_summary(sample: pd.DataFrame) -> list[dict[str, Any]]:
    yearly = (
        sample.assign(year=sample["entry_timestamp"].dt.year)
        .groupby("year", sort=True)["net_stress_pips"]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    return [
        {
            "year": int(row["year"]),
            "trade_count": int(row["size"]),
            "total_stress_pips": float(row["sum"]),
            "mean_stress_pips": float(row["mean"]),
        }
        for row in yearly.to_dict("records")
    ]


def summarize_sample(sample: pd.DataFrame) -> dict[str, Any]:
    if sample.empty:
        return {
            "trade_count": 0,
            "mean_base_pips": None,
            "mean_stress_pips": None,
            "total_stress_pips": 0.0,
            "lower_stress_pips": None,
            "positive_year_share": None,
            "positive_year_count": 0,
            "year_count": 0,
            "worst_trade_pips": None,
            "best_trade_pips": None,
            "top_positive_contribution_share": None,
            "yearly": [],
        }

    values = sample["net_stress_pips"].astype("float64")
    positive_share, positive_years, years = positive_year_share(sample, "net_stress_pips")
    return {
        "trade_count": int(len(sample)),
        "mean_base_pips": float(sample["net_base_pips"].mean()),
        "mean_stress_pips": float(values.mean()),
        "total_stress_pips": float(values.sum()),
        "lower_stress_pips": lower_confidence_bound(values, 0.05),
        "positive_year_share": positive_share,
        "positive_year_count": int(positive_years),
        "year_count": int(years),
        "worst_trade_pips": float(values.min()),
        "best_trade_pips": float(values.max()),
        "top_positive_contribution_share": top_positive_contribution_share(values),
        "yearly": yearly_summary(sample),
    }


def sample_passes_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["trade_count"] >= MIN_SAMPLE_TRADES
        and summary["year_count"] >= MIN_SAMPLE_YEARS
        and summary["mean_base_pips"] is not None
        and summary["mean_base_pips"] > 0
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["lower_stress_pips"] is not None
        and summary["lower_stress_pips"] > 0
        and summary["positive_year_share"] is not None
        and summary["positive_year_share"] >= MIN_POSITIVE_YEAR_SHARE
    )


def evaluate_candidates(pre_sample: pd.DataFrame, current_sample: pd.DataFrame) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for candidate in candidate_grid().to_dict("records"):
        pre_summary = summarize_sample(pre_sample.loc[pre_sample["candidate_id"].eq(candidate["candidate_id"])])
        current_summary = summarize_sample(
            current_sample.loc[current_sample["candidate_id"].eq(candidate["candidate_id"])]
        )
        pre_passed = sample_passes_gate(pre_summary)
        current_passed = sample_passes_gate(current_summary)
        evaluations.append(
            {
                **candidate,
                "pre": pre_summary,
                "current": current_summary,
                "pre_passed": pre_passed,
                "current_passed": current_passed,
                "both_passed": pre_passed and current_passed,
            }
        )
    return evaluations


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
                if key == "yearly":
                    continue
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def summarize_neighborhood(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    broad = evaluations
    local = [item for item in evaluations if item["is_local_neighborhood"]]
    one_step = [item for item in evaluations if item["is_one_step_neighbor"]]
    passed = [item for item in evaluations if item["both_passed"]]
    exact = next(item for item in evaluations if item["is_exact_frozen_candidate"])

    def count(items: list[dict[str, Any]], key: str) -> int:
        return int(sum(bool(item[key]) for item in items))

    passed_channel_hold = sorted(
        {
            f"ch{item['channel_days']}_hold{item['hold_days']}"
            for item in passed
        }
    )
    return {
        "grid_candidate_count": len(broad),
        "pre_pass_count": count(broad, "pre_passed"),
        "current_pass_count": count(broad, "current_passed"),
        "both_pass_count": len(passed),
        "both_pass_rate": len(passed) / len(broad) if broad else None,
        "local_candidate_count": len(local),
        "local_both_pass_count": count(local, "both_passed"),
        "local_both_pass_rate": count(local, "both_passed") / len(local) if local else None,
        "one_step_candidate_count": len(one_step),
        "one_step_both_pass_count": count(one_step, "both_passed"),
        "one_step_both_pass_rate": count(one_step, "both_passed") / len(one_step) if one_step else None,
        "passed_channel_hold_shapes": passed_channel_hold,
        "passed_candidate_ids": [item["candidate_id"] for item in passed],
        "exact_candidate": exact,
    }


def fast144_status(summary: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        "neighborhood_robustness_audit",
        "uses_full_current_grid_from_fast133_not_fast137_current_finalist_subset",
    ]
    exact = summary["exact_candidate"]
    if not exact["both_passed"]:
        return (
            "FAST144_NZDUSD_SHORT_ONLY_NEIGHBORHOOD_FAILED_EXACT_CANDIDATE",
            reasons + ["frozen_candidate_failed_pre_or_current_neighborhood_gate"],
        )
    if summary["local_both_pass_count"] >= 2:
        return (
            "FAST144_NZDUSD_SHORT_ONLY_EDGE_REPLICATED_BUT_NARROW_REQUIRES_OBSERVATION",
            reasons
            + [
                "frozen_candidate_passed_pre_and_current",
                f"local_passed:{summary['local_both_pass_count']}/{summary['local_candidate_count']}",
                f"broad_passed:{summary['both_pass_count']}/{summary['grid_candidate_count']}",
                "robustness_is_concentrated_in_channel100_hold20_risk_variants",
                "fresh_preregistered_prospective_observation_remains_required",
            ],
        )
    return (
        "FAST144_NZDUSD_SHORT_ONLY_EXACT_CANDIDATE_FRAGILE_REQUIRES_OBSERVATION",
        reasons
        + [
            "frozen_candidate_passed_pre_and_current",
            "too_few_neighbors_passed_independent_gates",
            "fresh_preregistered_prospective_observation_remains_required",
        ],
    )


def run(
    *,
    pre_input_path: Path = DEFAULT_PRE_INPUT_PATH,
    current_input_path: Path = DEFAULT_CURRENT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    pre_source = pd.read_parquet(pre_input_path)
    current_source = pd.read_parquet(current_input_path)
    pre_sample = filtered_sample(
        pre_source,
        sample_column="sample",
        sample_value=PRE_SAMPLE_LABEL,
    )
    current_sample = filtered_sample(
        current_source,
        sample_column=None,
        sample_value=None,
    )
    if pre_sample.empty:
        raise ValueError("Fast144 pre sample is empty after filtering")
    if current_sample.empty:
        raise ValueError("Fast144 current sample is empty after filtering")

    evaluations = evaluate_candidates(pre_sample, current_sample)
    summary = summarize_neighborhood(evaluations)
    status, status_reasons = fast144_status(summary)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "candidate-neighborhood-metrics.csv"
    passed_path = output_root / "passed-candidates.csv"
    summary_path = output_root / "neighborhood-summary.json"
    evaluations_path = output_root / "candidate-neighborhood-evaluations.json"

    metrics = pd.DataFrame(flat_metric_rows(evaluations)).sort_values(
        [
            "both_passed",
            "current_lower_stress_pips",
            "current_mean_stress_pips",
            "pre_lower_stress_pips",
        ],
        ascending=[False, False, False, False],
    )
    metrics.to_csv(metrics_path, index=False)
    metrics.loc[metrics["both_passed"]].to_csv(passed_path, index=False)
    write_json(summary_path, summary)
    write_json(evaluations_path, evaluations)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast144_is_historical_neighborhood_audit_not_strategy_approval",
            "fast143_prospective_observation_is_still_empty",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast144_nzdusd_short_only_neighborhood_robustness",
        "status": status,
        "status_reasons": status_reasons,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "pre_input_path": str(pre_input_path),
        "pre_input_sha256": sha256_file(pre_input_path),
        "current_input_path": str(current_input_path),
        "current_input_sha256": sha256_file(current_input_path),
        "pre_sample_label": PRE_SAMPLE_LABEL,
        "current_sample_label": CURRENT_SAMPLE_LABEL,
        "grid": {
            "pair": PAIR,
            "side": SIDE,
            "management": MANAGEMENT,
            "channel_days": CHANNEL_DAYS,
            "hold_days": HOLD_DAYS,
            "stop_atr_multiples": STOP_ATR_MULTIPLES,
        },
        "minimums": {
            "min_sample_trades": MIN_SAMPLE_TRADES,
            "min_sample_years": MIN_SAMPLE_YEARS,
            "min_positive_year_share": MIN_POSITIVE_YEAR_SHARE,
        },
        "source_rows": {
            "pre_filtered_rows": int(len(pre_sample)),
            "current_filtered_rows": int(len(current_sample)),
        },
        "summary": summary,
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "passed_candidates": {"path": str(passed_path), "sha256": sha256_file(passed_path)},
            "neighborhood_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "evaluations": {"path": str(evaluations_path), "sha256": sha256_file(evaluations_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    exact = summary["exact_candidate"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "frozen_policy_id": report["frozen_policy_id"],
        "exact_candidate_both_passed": exact["both_passed"],
        "exact_pre": exact["pre"],
        "exact_current": exact["current"],
        "broad_pass_count": summary["both_pass_count"],
        "grid_candidate_count": summary["grid_candidate_count"],
        "local_pass_count": summary["local_both_pass_count"],
        "local_candidate_count": summary["local_candidate_count"],
        "passed_candidate_ids": summary["passed_candidate_ids"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
