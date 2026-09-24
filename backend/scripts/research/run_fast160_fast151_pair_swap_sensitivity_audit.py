"""Fast160: pair-specific swap sensitivity for the Fast151 operating policy.

Fast159 showed that a uniform adverse short swap near 1 pip per rollover day
can erase the current-sample LCB. Fast160 asks the next operational question:
which pair is most sensitive if only that pair has adverse short swap while the
other pairs are neutral?

This does not replace the broker-table gate. It creates pair-level tolerances so
we can decide whether a real broker's swap table is acceptable, whether a single
pair should be excluded, or whether the candidate should stay research-only.

Historical research only. No orders, no ledger, no Demo/Testnet/LIVE.
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
from scripts.research.run_fast159_fast151_exact_swap_rollover_audit import (  # noqa: E402
    add_rollover_days,
    estimate_uniform_break_even_pips_per_rollover_day,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast160" / "fast151-pair-swap-sensitivity-audit"
)
STRESS_POINTS = (0.5, 1.0, 2.0, 3.0, 5.0)


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
    pre = add_rollover_days(load_trades(pre_input_path))
    current = add_rollover_days(load_trades(current_input_path))
    pairs = sorted(set(pre["pair"].unique()).union(set(current["pair"].unique())))
    pair_evaluations = [evaluate_pair_sensitivity(pre, current, pair) for pair in pairs]
    uniform_break_even = {
        "pre_lcb_zero_pips_per_rollover_day": estimate_uniform_break_even_pips_per_rollover_day(pre),
        "current_lcb_zero_pips_per_rollover_day": estimate_uniform_break_even_pips_per_rollover_day(current),
    }
    status, status_reasons = fast160_status(
        pair_evaluations=pair_evaluations,
        uniform_break_even=uniform_break_even,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "pair-swap-sensitivity-metrics.csv"
    summary_path = output_root / "pair-swap-sensitivity-summary.json"
    pd.DataFrame(flat_metric_rows(pair_evaluations)).to_csv(metrics_path, index=False)
    summary = {
        "pair_evaluations": pair_evaluations,
        "uniform_break_even": uniform_break_even,
        "stress_points_pips_per_rollover_day": list(STRESS_POINTS),
        "interpretation": {
            "pair_break_even": "adverse short swap applied to only that pair; other pairs neutral",
            "uniform_break_even": "same adverse short swap applied to every pair simultaneously",
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast160_is_historical_pair_swap_sensitivity_audit_not_strategy_approval",
            "pair_specific_tolerances_do_not_replace_broker_swap_table_gate",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast160_fast151_pair_swap_sensitivity_audit",
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
            "pair_swap_sensitivity_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def apply_pair_adverse_swap_cost(
    trades: pd.DataFrame,
    *,
    pair: str,
    adverse_short_pips_per_rollover_day: float,
) -> pd.DataFrame:
    adjusted = trades.copy()
    adjusted["swap_pips"] = 0.0
    target = adjusted["pair"].eq(pair)
    adjusted.loc[target, "swap_pips"] = -(
        adjusted.loc[target, "swap_rollover_days"].astype("float64")
        * adverse_short_pips_per_rollover_day
    )
    adjusted["net_stress_pips"] = adjusted["net_stress_pips"].astype("float64") + adjusted["swap_pips"]
    return adjusted


def estimate_pair_break_even_pips_per_rollover_day(trades: pd.DataFrame, pair: str) -> float:
    if not trades["pair"].eq(pair).any():
        raise ValueError(f"pair {pair!r} not present in trades")
    low = 0.0
    high = 10.0
    while summarize_sample(
        apply_pair_adverse_swap_cost(trades, pair=pair, adverse_short_pips_per_rollover_day=high)
    )["lower_stress_pips"] > 0:
        high *= 2
    for _ in range(40):
        mid = (low + high) / 2
        if summarize_sample(
            apply_pair_adverse_swap_cost(trades, pair=pair, adverse_short_pips_per_rollover_day=mid)
        )["lower_stress_pips"] > 0:
            low = mid
        else:
            high = mid
    return high


def evaluate_pair_sensitivity(pre: pd.DataFrame, current: pd.DataFrame, pair: str) -> dict[str, Any]:
    return {
        "pair": pair,
        "pre": evaluate_pair_sample(pre, pair),
        "current": evaluate_pair_sample(current, pair),
    }


def evaluate_pair_sample(trades: pd.DataFrame, pair: str) -> dict[str, Any]:
    pair_trades = trades.loc[trades["pair"].eq(pair)]
    if pair_trades.empty:
        return {
            "pair_trade_count": 0,
            "pair_total_rollover_days": 0,
            "pair_mean_rollover_days": None,
            "pair_baseline_total_stress_pips": 0.0,
            "pair_baseline_mean_stress_pips": None,
            "pair_only_break_even_lcb_zero_pips_per_rollover_day": None,
            "stress_evaluations": [],
        }
    stress_evaluations = [
        evaluate_pair_stress_point(trades, pair, stress_point) for stress_point in STRESS_POINTS
    ]
    return {
        "pair_trade_count": int(len(pair_trades)),
        "pair_total_rollover_days": int(pair_trades["swap_rollover_days"].sum()),
        "pair_mean_rollover_days": float(pair_trades["swap_rollover_days"].mean()),
        "pair_baseline_total_stress_pips": float(pair_trades["net_stress_pips"].sum()),
        "pair_baseline_mean_stress_pips": float(pair_trades["net_stress_pips"].mean()),
        "pair_only_break_even_lcb_zero_pips_per_rollover_day": estimate_pair_break_even_pips_per_rollover_day(
            trades,
            pair,
        ),
        "stress_evaluations": stress_evaluations,
    }


def evaluate_pair_stress_point(
    trades: pd.DataFrame,
    pair: str,
    adverse_short_pips_per_rollover_day: float,
) -> dict[str, Any]:
    adjusted = apply_pair_adverse_swap_cost(
        trades,
        pair=pair,
        adverse_short_pips_per_rollover_day=adverse_short_pips_per_rollover_day,
    )
    summary = summarize_sample(adjusted)
    return {
        "adverse_short_pips_per_rollover_day": adverse_short_pips_per_rollover_day,
        "portfolio_mean_stress_pips": summary["mean_stress_pips"],
        "portfolio_lower_stress_pips": summary["lower_stress_pips"],
        "portfolio_max_drawdown_pips": summary["max_drawdown_pips"],
        "portfolio_two_consecutive_negative_years": summary["two_consecutive_negative_years"],
        "portfolio_lcb_positive": bool(summary["lower_stress_pips"] is not None and summary["lower_stress_pips"] > 0),
    }


def flat_metric_rows(pair_evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in pair_evaluations:
        row: dict[str, Any] = {"pair": evaluation["pair"]}
        for sample_name in ("pre", "current"):
            sample = evaluation[sample_name]
            for key, value in sample.items():
                if key == "stress_evaluations":
                    continue
                row[f"{sample_name}_{key}"] = value
            for stress in sample["stress_evaluations"]:
                prefix = f"{sample_name}_stress_{stress['adverse_short_pips_per_rollover_day']:.2f}"
                row[f"{prefix}_lcb"] = stress["portfolio_lower_stress_pips"]
                row[f"{prefix}_mean"] = stress["portfolio_mean_stress_pips"]
                row[f"{prefix}_dd"] = stress["portfolio_max_drawdown_pips"]
                row[f"{prefix}_bad_year_cluster"] = stress["portfolio_two_consecutive_negative_years"]
        rows.append(row)
    return rows


def fast160_status(
    *,
    pair_evaluations: list[dict[str, Any]],
    uniform_break_even: dict[str, float],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "pair_swap_sensitivity_audit"]
    current_break_evens = [
        evaluation["current"]["pair_only_break_even_lcb_zero_pips_per_rollover_day"]
        for evaluation in pair_evaluations
        if evaluation["current"]["pair_only_break_even_lcb_zero_pips_per_rollover_day"] is not None
    ]
    flags: list[str] = []
    if uniform_break_even["current_lcb_zero_pips_per_rollover_day"] < 1.0:
        flags.append("uniform_current_swap_break_even_below_1_pip_per_rollover_day")
    if current_break_evens and min(current_break_evens) < 5.0:
        flags.append("single_pair_swap_break_even_below_5_pips_per_rollover_day")
    return (
        "FAST160_FAST151_PAIR_SWAP_LIMITS_IDENTIFIED_BROKER_TABLE_AND_FORWARD_REQUIRED",
        reasons + flags + ["broker_swap_schedule_required", "fresh_preregistered_prospective_observation_still_required"],
    )


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    pair_rows = [
        {
            "pair": evaluation["pair"],
            "pre_pair_break_even": evaluation["pre"]["pair_only_break_even_lcb_zero_pips_per_rollover_day"],
            "current_pair_break_even": evaluation["current"]["pair_only_break_even_lcb_zero_pips_per_rollover_day"],
            "current_total_rollover_days": evaluation["current"]["pair_total_rollover_days"],
            "current_pair_baseline_mean_stress_pips": evaluation["current"]["pair_baseline_mean_stress_pips"],
        }
        for evaluation in report["summary"]["pair_evaluations"]
    ]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "operational_policy_id": report["operational_policy_id"],
        "uniform_break_even": report["summary"]["uniform_break_even"],
        "pair_break_evens": pair_rows,
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
