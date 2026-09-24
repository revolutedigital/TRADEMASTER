"""Fast157: sizing-envelope audit for the Fast151 operating policy.

Fast156 quantified drawdown tails in pips. Fast157 converts the same frozen
Fast151 historical trades into R units (`net_stress_pips / risk_pips`) and
derives a conservative risk-per-trade envelope.

This is still research-only. It does not place orders, touch ledger state,
approve Testnet, or approve LIVE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)
from scripts.research.run_fast156_fast151_bootstrap_risk_audit import (  # noqa: E402
    DEFAULT_CURRENT_INPUT_PATH,
    DEFAULT_PRE_INPUT_PATH,
    DEFAULT_SIMULATION_COUNT,
    TARGET_POLICY_ID,
    YEAR_BLOCK_BOOTSTRAP_SEED,
    TRADE_BOOTSTRAP_SEED,
    load_trades,
    max_drawdown,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast157" / "fast151-sizing-envelope-audit"
RISK_PER_TRADE_CANDIDATES = (0.001, 0.0025, 0.005, 0.0075, 0.01)
DRAWDOWN_BUDGETS = (0.05, 0.10, 0.20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-input-path", type=Path, default=DEFAULT_PRE_INPUT_PATH)
    parser.add_argument("--current-input-path", type=Path, default=DEFAULT_CURRENT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--simulation-count", type=int, default=DEFAULT_SIMULATION_COUNT)
    arguments = parser.parse_args(argv)
    report = run(
        pre_input_path=arguments.pre_input_path,
        current_input_path=arguments.current_input_path,
        output_root=arguments.output_root,
        simulation_count=arguments.simulation_count,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run(
    *,
    pre_input_path: Path = DEFAULT_PRE_INPUT_PATH,
    current_input_path: Path = DEFAULT_CURRENT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    simulation_count: int = DEFAULT_SIMULATION_COUNT,
) -> dict[str, Any]:
    if simulation_count <= 0:
        raise ValueError("simulation_count must be positive")
    pre = add_r_units(load_trades(pre_input_path))
    current = add_r_units(load_trades(current_input_path))
    pre_actual = actual_r_summary(pre)
    current_actual = actual_r_summary(current)
    pre_trade_bootstrap = bootstrap_r_summary(
        pre,
        simulation_count=simulation_count,
        seed=TRADE_BOOTSTRAP_SEED,
        mode="trade",
    )
    current_trade_bootstrap = bootstrap_r_summary(
        current,
        simulation_count=simulation_count,
        seed=TRADE_BOOTSTRAP_SEED + 1,
        mode="trade",
    )
    pre_year_bootstrap = bootstrap_r_summary(
        pre,
        simulation_count=simulation_count,
        seed=YEAR_BLOCK_BOOTSTRAP_SEED,
        mode="year_block",
    )
    current_year_bootstrap = bootstrap_r_summary(
        current,
        simulation_count=simulation_count,
        seed=YEAR_BLOCK_BOOTSTRAP_SEED + 1,
        mode="year_block",
    )
    tail_drawdown_r = min(
        pre_trade_bootstrap["max_drawdown_r_p05"],
        current_trade_bootstrap["max_drawdown_r_p05"],
        pre_year_bootstrap["max_drawdown_r_p05"],
        current_year_bootstrap["max_drawdown_r_p05"],
    )
    sizing_envelope = build_sizing_envelope(tail_drawdown_r=tail_drawdown_r)
    risk_candidate_rows = risk_candidate_drawdown_rows(tail_drawdown_r=tail_drawdown_r)
    status, status_reasons = fast157_status(sizing_envelope=sizing_envelope, tail_drawdown_r=tail_drawdown_r)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "sizing-envelope-metrics.csv"
    summary_path = output_root / "sizing-envelope-summary.json"
    summary = {
        "simulation_count": simulation_count,
        "seeds": {
            "pre_trade_bootstrap": TRADE_BOOTSTRAP_SEED,
            "current_trade_bootstrap": TRADE_BOOTSTRAP_SEED + 1,
            "pre_year_block_bootstrap": YEAR_BLOCK_BOOTSTRAP_SEED,
            "current_year_block_bootstrap": YEAR_BLOCK_BOOTSTRAP_SEED + 1,
        },
        "pre": {
            "actual": pre_actual,
            "trade_bootstrap": pre_trade_bootstrap,
            "year_block_bootstrap": pre_year_bootstrap,
        },
        "current": {
            "actual": current_actual,
            "trade_bootstrap": current_trade_bootstrap,
            "year_block_bootstrap": current_year_bootstrap,
        },
        "worst_tail_drawdown_r_p05": tail_drawdown_r,
        "sizing_envelope": sizing_envelope,
        "risk_candidate_drawdowns": risk_candidate_rows,
    }
    write_json(summary_path, summary)
    pd.DataFrame(flat_metric_rows(summary)).to_csv(metrics_path, index=False)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast157_is_historical_sizing_envelope_audit_not_strategy_approval",
            "risk_per_trade_envelope_depends_on_historical_bootstrap_tail",
            "fast151_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast157_fast151_sizing_envelope_audit",
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
            "sizing_envelope_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def add_r_units(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    frame["net_stress_r"] = frame["net_stress_pips"].astype("float64") / frame["risk_pips"].astype("float64")
    return frame


def ordered_r(trades: pd.DataFrame) -> np.ndarray:
    return (
        trades.sort_values(["exit_stress_timestamp", "entry_timestamp", "pair"])["net_stress_r"]
        .astype("float64")
        .to_numpy()
    )


def actual_r_summary(trades: pd.DataFrame) -> dict[str, Any]:
    values = ordered_r(trades)
    return {
        "trade_count": int(len(values)),
        "total_r": float(values.sum()),
        "mean_r": float(values.mean()),
        "median_r": float(np.median(values)),
        "worst_trade_r": float(values.min()),
        "best_trade_r": float(values.max()),
        "max_drawdown_r": max_drawdown(values),
    }


def bootstrap_r_summary(
    trades: pd.DataFrame,
    *,
    simulation_count: int,
    seed: int,
    mode: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    total_r = np.empty(simulation_count)
    mean_r = np.empty(simulation_count)
    drawdown_r = np.empty(simulation_count)
    if mode == "trade":
        values = ordered_r(trades)
        for index in range(simulation_count):
            path = rng.choice(values, size=len(values), replace=True)
            total_r[index] = path.sum()
            mean_r[index] = path.mean()
            drawdown_r[index] = max_drawdown(path)
    elif mode == "year_block":
        frame = trades.copy()
        frame["year"] = frame["entry_timestamp"].dt.year
        years = sorted(frame["year"].unique())
        groups = {
            year: ordered_r(frame.loc[frame["year"].eq(year)].copy())
            for year in years
        }
        for index in range(simulation_count):
            sampled_years = rng.choice(years, size=len(years), replace=True)
            path = np.concatenate([groups[year] for year in sampled_years])
            total_r[index] = path.sum()
            mean_r[index] = path.mean()
            drawdown_r[index] = max_drawdown(path)
    else:
        raise ValueError(f"unknown bootstrap mode: {mode}")
    return {
        "bootstrap_mode": mode,
        "total_r_p05": quantile(total_r, 0.05),
        "total_r_p50": quantile(total_r, 0.50),
        "total_r_p95": quantile(total_r, 0.95),
        "mean_r_p05": quantile(mean_r, 0.05),
        "mean_r_p50": quantile(mean_r, 0.50),
        "mean_r_p95": quantile(mean_r, 0.95),
        "max_drawdown_r_p05": quantile(drawdown_r, 0.05),
        "max_drawdown_r_p50": quantile(drawdown_r, 0.50),
        "max_drawdown_r_p95": quantile(drawdown_r, 0.95),
        "probability_total_r_lte_0": probability(total_r <= 0),
        "probability_drawdown_lte_8r": probability(drawdown_r <= -8.0),
        "probability_drawdown_lte_10r": probability(drawdown_r <= -10.0),
    }


def build_sizing_envelope(*, tail_drawdown_r: float) -> list[dict[str, float]]:
    absolute_tail_drawdown_r = abs(tail_drawdown_r)
    return [
        {
            "drawdown_budget_fraction": budget,
            "max_risk_per_trade_fraction": budget / absolute_tail_drawdown_r,
            "max_risk_per_trade_percent": 100 * budget / absolute_tail_drawdown_r,
        }
        for budget in DRAWDOWN_BUDGETS
    ]


def risk_candidate_drawdown_rows(*, tail_drawdown_r: float) -> list[dict[str, float]]:
    return [
        {
            "risk_per_trade_fraction": risk,
            "risk_per_trade_percent": 100 * risk,
            "tail_drawdown_fraction": abs(tail_drawdown_r) * risk,
            "tail_drawdown_percent": 100 * abs(tail_drawdown_r) * risk,
        }
        for risk in RISK_PER_TRADE_CANDIDATES
    ]


def quantile(values: np.ndarray, value: float) -> float:
    return float(np.quantile(values, value))


def probability(mask: np.ndarray) -> float:
    return float(mask.mean())


def flat_metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in ("pre", "current"):
        rows.append({"sample": sample, "mode": "actual", **summary[sample]["actual"]})
        for mode in ("trade_bootstrap", "year_block_bootstrap"):
            rows.append({"sample": sample, "mode": mode, **summary[sample][mode]})
    for row in summary["sizing_envelope"]:
        rows.append({"sample": "combined", "mode": "sizing_envelope", **row})
    for row in summary["risk_candidate_drawdowns"]:
        rows.append({"sample": "combined", "mode": "risk_candidate_drawdown", **row})
    return rows


def fast157_status(
    *,
    sizing_envelope: list[dict[str, float]],
    tail_drawdown_r: float,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "sizing_envelope_audit"]
    max_risk_for_10pct = next(
        row["max_risk_per_trade_fraction"]
        for row in sizing_envelope
        if row["drawdown_budget_fraction"] == 0.10
    )
    flags: list[str] = []
    if max_risk_for_10pct < 0.01:
        flags.append("ten_percent_drawdown_budget_requires_less_than_1pct_risk_per_trade")
    if abs(tail_drawdown_r) > 10:
        flags.append("bootstrap_tail_drawdown_exceeds_10r")
    return (
        "FAST157_FAST151_SIZING_ENVELOPE_REQUIRES_CONSERVATIVE_RISK_AND_PROSPECTIVE_OBSERVATION",
        reasons + flags + ["fresh_preregistered_prospective_observation_still_required"],
    )


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "operational_policy_id": report["operational_policy_id"],
        "worst_tail_drawdown_r_p05": summary["worst_tail_drawdown_r_p05"],
        "sizing_envelope": summary["sizing_envelope"],
        "risk_candidate_drawdowns": summary["risk_candidate_drawdowns"],
        "pre_actual": summary["pre"]["actual"],
        "current_actual": summary["current"]["actual"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
