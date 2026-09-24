"""Fast159: exact rollover/swap gate for the Fast151 operating policy.

Fast158 used a conservative billed-calendar-day approximation to estimate carry
sensitivity. Fast159 tightens that audit to the broker convention already used
by the FX simulator: financing is charged at 17:00 New York, Wednesday counts
triple, and weekends are not charged directly.

The default run does not pretend to know the live broker's swap table. It
evaluates uniform adverse short-swap debits and writes a broker-swap template.
If a broker table is supplied, the same runner can score the frozen Fast151
trades with pair-specific long/short swap pips per rollover day.

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

from app.fx.sim.costs import RolloverCalendar  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import (  # noqa: E402
    TARGET_POLICY_ID,
    summarize_sample,
)
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)
from scripts.research.run_fast156_fast151_bootstrap_risk_audit import (  # noqa: E402
    DEFAULT_CURRENT_INPUT_PATH,
    DEFAULT_PRE_INPUT_PATH,
    load_trades,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast159" / "fast151-exact-swap-rollover-audit"
)
REQUIRED_SWAP_COLUMNS = ("pair", "long_pips_per_rollover_day", "short_pips_per_rollover_day")


@dataclass(frozen=True)
class UniformSwapScenario:
    identifier: str
    adverse_short_pips_per_rollover_day: float


def uniform_swap_scenarios() -> tuple[UniformSwapScenario, ...]:
    return (
        UniformSwapScenario("no_extra_swap", 0.0),
        UniformSwapScenario("adverse_short_swap_0_10_pip_per_rollover_day", 0.10),
        UniformSwapScenario("adverse_short_swap_0_25_pip_per_rollover_day", 0.25),
        UniformSwapScenario("adverse_short_swap_0_50_pip_per_rollover_day", 0.50),
        UniformSwapScenario("adverse_short_swap_0_75_pip_per_rollover_day", 0.75),
        UniformSwapScenario("adverse_short_swap_1_00_pip_per_rollover_day", 1.00),
        UniformSwapScenario("adverse_short_swap_1_50_pip_per_rollover_day", 1.50),
        UniformSwapScenario("adverse_short_swap_2_00_pip_per_rollover_day", 2.00),
        UniformSwapScenario("adverse_short_swap_3_00_pip_per_rollover_day", 3.00),
        UniformSwapScenario("adverse_short_swap_5_00_pip_per_rollover_day", 5.00),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-input-path", type=Path, default=DEFAULT_PRE_INPUT_PATH)
    parser.add_argument("--current-input-path", type=Path, default=DEFAULT_CURRENT_INPUT_PATH)
    parser.add_argument("--broker-swap-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(
        pre_input_path=arguments.pre_input_path,
        current_input_path=arguments.current_input_path,
        broker_swap_path=arguments.broker_swap_path,
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
    *,
    pre_input_path: Path = DEFAULT_PRE_INPUT_PATH,
    current_input_path: Path = DEFAULT_CURRENT_INPUT_PATH,
    broker_swap_path: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    pre = add_rollover_days(load_trades(pre_input_path))
    current = add_rollover_days(load_trades(current_input_path))
    uniform_evaluations = [
        evaluate_uniform_swap_scenario(pre, current, scenario) for scenario in uniform_swap_scenarios()
    ]
    baseline = next(
        evaluation for evaluation in uniform_evaluations if evaluation["scenario_id"] == "no_extra_swap"
    )
    break_even = {
        "pre_lcb_zero_pips_per_rollover_day": estimate_uniform_break_even_pips_per_rollover_day(pre),
        "current_lcb_zero_pips_per_rollover_day": estimate_uniform_break_even_pips_per_rollover_day(current),
    }
    broker_schedule = load_broker_swap_schedule(broker_swap_path) if broker_swap_path else None
    broker_evaluation = (
        evaluate_broker_swap_schedule(pre, current, broker_schedule) if broker_schedule is not None else None
    )
    status, status_reasons = fast159_status(
        uniform_evaluations=uniform_evaluations,
        break_even=break_even,
        broker_schedule_supplied=broker_schedule is not None,
        broker_evaluation=broker_evaluation,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "exact-swap-rollover-metrics.csv"
    summary_path = output_root / "exact-swap-rollover-summary.json"
    template_path = output_root / "broker-swap-template.csv"
    pd.DataFrame(flat_metric_rows(uniform_evaluations, broker_evaluation)).to_csv(metrics_path, index=False)
    broker_swap_template(pre, current).to_csv(template_path, index=False)
    summary = {
        "baseline": baseline,
        "uniform_evaluations": uniform_evaluations,
        "break_even": break_even,
        "rollover_day_diagnostics": {
            "pre": rollover_day_diagnostics(pre),
            "current": rollover_day_diagnostics(current),
        },
        "broker_swap_schedule": None if broker_schedule is None else broker_schedule.to_dict("records"),
        "broker_evaluation": broker_evaluation,
        "swap_model": {
            "rollover_time": "17:00 America/New_York",
            "triple_weekday": "Wednesday",
            "weekend_rollovers": "not_charged_directly",
            "unit": "pips_per_rollover_day",
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast159_is_historical_exact_swap_rollover_audit_not_strategy_approval",
            "broker_specific_swap_table_required_before_shadow_or_testnet",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast159_fast151_exact_swap_rollover_audit",
        "status": status,
        "status_reasons": status_reasons,
        "target_policy_id": TARGET_POLICY_ID,
        "operational_policy_id": OPERATIONAL_POLICY_ID,
        "pre_input_path": str(pre_input_path),
        "pre_input_sha256": sha256_file(pre_input_path),
        "current_input_path": str(current_input_path),
        "current_input_sha256": sha256_file(current_input_path),
        "broker_swap_path": None if broker_swap_path is None else str(broker_swap_path),
        "broker_swap_sha256": None if broker_swap_path is None else sha256_file(broker_swap_path),
        "summary": summary,
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "exact_swap_rollover_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "broker_swap_template": {"path": str(template_path), "sha256": sha256_file(template_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def add_rollover_days(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    first_year = int(frame["entry_timestamp"].dt.year.min()) - 1
    last_year = int(frame["exit_stress_timestamp"].dt.year.max()) + 1
    calendar = RolloverCalendar(first_year=first_year, last_year=last_year)
    frame["swap_rollover_days"] = calendar.charged_days(
        timestamp_seconds(frame["entry_timestamp"]),
        timestamp_seconds(frame["exit_stress_timestamp"]),
    ).astype("int64")
    return frame


def timestamp_seconds(values: pd.Series) -> pd.Series:
    return values.map(lambda value: pd.Timestamp(value).timestamp())


def apply_uniform_adverse_swap_cost(
    trades: pd.DataFrame,
    adverse_short_pips_per_rollover_day: float,
) -> pd.DataFrame:
    adjusted = trades.copy()
    adjusted["swap_cost_pips"] = (
        adjusted["swap_rollover_days"].astype("float64") * adverse_short_pips_per_rollover_day
    )
    adjusted["swap_pips"] = -adjusted["swap_cost_pips"]
    adjusted["net_stress_pips"] = adjusted["net_stress_pips"].astype("float64") + adjusted["swap_pips"]
    return adjusted


def load_broker_swap_schedule(path: Path) -> pd.DataFrame:
    schedule = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_SWAP_COLUMNS if column not in schedule.columns]
    if missing_columns:
        raise ValueError(f"broker swap table missing columns: {missing_columns}")
    schedule = schedule[list(REQUIRED_SWAP_COLUMNS)].copy()
    schedule["pair"] = schedule["pair"].astype("string").str.upper()
    if schedule["pair"].duplicated().any():
        duplicates = sorted(schedule.loc[schedule["pair"].duplicated(), "pair"].unique().tolist())
        raise ValueError(f"broker swap table has duplicate pairs: {duplicates}")
    for column in ("long_pips_per_rollover_day", "short_pips_per_rollover_day"):
        schedule[column] = pd.to_numeric(schedule[column], errors="raise").astype("float64")
    return schedule


def apply_broker_swap_schedule(trades: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    adjusted = trades.merge(schedule, how="left", on="pair", validate="many_to_one")
    missing = sorted(adjusted.loc[adjusted["short_pips_per_rollover_day"].isna(), "pair"].unique().tolist())
    if missing:
        raise ValueError(f"broker swap table missing pairs: {missing}")
    adjusted["broker_swap_pips_per_rollover_day"] = adjusted.apply(
        lambda row: row["long_pips_per_rollover_day"]
        if row["side"] == "LONG"
        else row["short_pips_per_rollover_day"],
        axis=1,
    )
    adjusted["swap_pips"] = (
        adjusted["swap_rollover_days"].astype("float64") * adjusted["broker_swap_pips_per_rollover_day"]
    )
    adjusted["net_stress_pips"] = adjusted["net_stress_pips"].astype("float64") + adjusted["swap_pips"]
    return adjusted


def evaluate_uniform_swap_scenario(
    pre: pd.DataFrame,
    current: pd.DataFrame,
    scenario: UniformSwapScenario,
) -> dict[str, Any]:
    pre_adjusted = apply_uniform_adverse_swap_cost(pre, scenario.adverse_short_pips_per_rollover_day)
    current_adjusted = apply_uniform_adverse_swap_cost(current, scenario.adverse_short_pips_per_rollover_day)
    pre_summary = summarize_sample(pre_adjusted)
    current_summary = summarize_sample(current_adjusted)
    return {
        "scenario_id": scenario.identifier,
        "adverse_short_pips_per_rollover_day": scenario.adverse_short_pips_per_rollover_day,
        "pre_total_rollover_days": int(pre["swap_rollover_days"].sum()),
        "current_total_rollover_days": int(current["swap_rollover_days"].sum()),
        "pre": pre_summary,
        "current": current_summary,
        "pre_lcb_positive": bool(pre_summary["lower_stress_pips"] is not None and pre_summary["lower_stress_pips"] > 0),
        "current_lcb_positive": bool(
            current_summary["lower_stress_pips"] is not None and current_summary["lower_stress_pips"] > 0
        ),
    }


def evaluate_broker_swap_schedule(
    pre: pd.DataFrame,
    current: pd.DataFrame,
    schedule: pd.DataFrame,
) -> dict[str, Any]:
    pre_adjusted = apply_broker_swap_schedule(pre, schedule)
    current_adjusted = apply_broker_swap_schedule(current, schedule)
    pre_summary = summarize_sample(pre_adjusted)
    current_summary = summarize_sample(current_adjusted)
    return {
        "scenario_id": "broker_swap_schedule",
        "pre": pre_summary,
        "current": current_summary,
        "pair_swap_days": pair_rollover_diagnostics(current_adjusted),
        "pre_lcb_positive": bool(pre_summary["lower_stress_pips"] is not None and pre_summary["lower_stress_pips"] > 0),
        "current_lcb_positive": bool(
            current_summary["lower_stress_pips"] is not None and current_summary["lower_stress_pips"] > 0
        ),
    }


def estimate_uniform_break_even_pips_per_rollover_day(trades: pd.DataFrame) -> float:
    low = 0.0
    high = 10.0
    while summarize_sample(apply_uniform_adverse_swap_cost(trades, high))["lower_stress_pips"] > 0:
        high *= 2
    for _ in range(40):
        mid = (low + high) / 2
        if summarize_sample(apply_uniform_adverse_swap_cost(trades, mid))["lower_stress_pips"] > 0:
            low = mid
        else:
            high = mid
    return high


def rollover_day_diagnostics(trades: pd.DataFrame) -> dict[str, Any]:
    return {
        "trade_count": int(len(trades)),
        "total_rollover_days": int(trades["swap_rollover_days"].sum()),
        "mean_rollover_days": float(trades["swap_rollover_days"].mean()),
        "median_rollover_days": float(trades["swap_rollover_days"].median()),
        "max_rollover_days": int(trades["swap_rollover_days"].max()),
        "zero_rollover_trade_count": int(trades["swap_rollover_days"].eq(0).sum()),
        "pair_rollover_diagnostics": pair_rollover_diagnostics(trades),
    }


def pair_rollover_diagnostics(trades: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        trades.groupby("pair", sort=True)["swap_rollover_days"]
        .agg(["size", "sum", "mean", "median", "max"])
        .reset_index()
    )
    return [
        {
            "pair": str(row["pair"]),
            "trade_count": int(row["size"]),
            "total_rollover_days": int(row["sum"]),
            "mean_rollover_days": float(row["mean"]),
            "median_rollover_days": float(row["median"]),
            "max_rollover_days": int(row["max"]),
        }
        for row in grouped.to_dict("records")
    ]


def broker_swap_template(pre: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    pairs = sorted(set(pre["pair"].unique()).union(set(current["pair"].unique())))
    return pd.DataFrame(
        {
            "pair": pairs,
            "long_pips_per_rollover_day": ["" for _ in pairs],
            "short_pips_per_rollover_day": ["" for _ in pairs],
        }
    )


def flat_metric_rows(
    uniform_evaluations: list[dict[str, Any]],
    broker_evaluation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in uniform_evaluations + ([] if broker_evaluation is None else [broker_evaluation]):
        row = {
            key: value
            for key, value in evaluation.items()
            if key not in {"pre", "current", "pair_swap_days"}
        }
        for prefix in ("pre", "current"):
            summary = evaluation[prefix]
            for key, value in summary.items():
                if key in {"yearly", "pair_summary", "exit_reason_summary"}:
                    continue
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def fast159_status(
    *,
    uniform_evaluations: list[dict[str, Any]],
    break_even: dict[str, float],
    broker_schedule_supplied: bool,
    broker_evaluation: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "exact_swap_rollover_audit"]
    scenario_by_id = {evaluation["scenario_id"]: evaluation for evaluation in uniform_evaluations}
    flags: list[str] = []
    if not scenario_by_id["adverse_short_swap_1_00_pip_per_rollover_day"]["current_lcb_positive"]:
        flags.append("current_lcb_fails_at_1_00_pip_per_rollover_day")
    if break_even["current_lcb_zero_pips_per_rollover_day"] < 1.0:
        flags.append("current_lcb_break_even_below_1_pip_per_rollover_day")
    if not broker_schedule_supplied:
        return (
            "FAST159_FAST151_EXACT_ROLLOVER_CONFIRMS_TIGHT_SWAP_MARGIN_REQUIRES_BROKER_TABLE",
            reasons + flags + ["broker_swap_schedule_missing", "fresh_preregistered_prospective_observation_still_required"],
        )
    if broker_evaluation is None:
        raise ValueError("broker_evaluation is required when broker_schedule_supplied is true")
    if not broker_evaluation["pre_lcb_positive"]:
        flags.append("broker_swap_pre_lcb_non_positive")
    if not broker_evaluation["current_lcb_positive"]:
        flags.append("broker_swap_current_lcb_non_positive")
    if flags:
        return (
            "FAST159_FAST151_BROKER_SWAP_TABLE_FAILS_GATE",
            reasons + flags + ["fresh_preregistered_prospective_observation_still_required"],
        )
    return (
        "FAST159_FAST151_BROKER_SWAP_TABLE_PASSES_REQUIRES_PROSPECTIVE_OBSERVATION",
        reasons + ["broker_swap_lcb_gate_passed", "fresh_preregistered_prospective_observation_required_before_any_execution"],
    )


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    baseline = report["summary"]["baseline"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "operational_policy_id": report["operational_policy_id"],
        "baseline_pre": {
            key: baseline["pre"][key]
            for key in ("trade_count", "mean_stress_pips", "lower_stress_pips", "max_drawdown_pips")
        },
        "baseline_current": {
            key: baseline["current"][key]
            for key in ("trade_count", "mean_stress_pips", "lower_stress_pips", "max_drawdown_pips")
        },
        "break_even": report["summary"]["break_even"],
        "rollover_day_diagnostics": report["summary"]["rollover_day_diagnostics"],
        "broker_swap_path": report["broker_swap_path"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
