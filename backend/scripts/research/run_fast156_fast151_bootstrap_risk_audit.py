"""Fast156: bootstrap risk audit for the Fast151 operating policy.

Fast151 has a positive historical edge but is not approved and still lacks
prospective observations. Fast156 asks a risk question instead of another
entry-search question:

    If the observed trade/year distribution repeats imperfectly, how often do
    we see bad drawdown tails or a non-positive total?

Two bootstrap views are used:

* trade bootstrap: resample individual trades with replacement;
* year-block bootstrap: resample calendar-year blocks with replacement,
  preserving within-year sequence and regime clustering.

Historical research only. No orders, no ledger, no Demo/Testnet/LIVE.
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
from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import (  # noqa: E402
    TARGET_POLICY_ID,
    summarize_sample,
)
from scripts.research.run_fast151_usd_quoted_basket_operational_observation import (  # noqa: E402
    OPERATIONAL_POLICY_ID,
    SAFETY_FLAGS,
)

DEFAULT_PRE_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast150"
    / "usd-quoted-basket-circuit-breaker-audit"
    / "selected-breaker-pre-trades.csv"
)
DEFAULT_CURRENT_INPUT_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast150"
    / "usd-quoted-basket-circuit-breaker-audit"
    / "selected-breaker-current-trades.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast156" / "fast151-bootstrap-risk-audit"
TIMESTAMP_COLUMNS = ("signal_timestamp", "entry_timestamp", "exit_base_timestamp", "exit_stress_timestamp")
DEFAULT_SIMULATION_COUNT = 20_000
TRADE_BOOTSTRAP_SEED = 20260924
YEAR_BLOCK_BOOTSTRAP_SEED = 20260925
DRAWDOWN_THRESHOLDS = (-500.0, -1000.0, -1500.0)


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
    pre = load_trades(pre_input_path)
    current = load_trades(current_input_path)
    pre_actual = actual_risk_summary(pre)
    current_actual = actual_risk_summary(current)
    pre_trade_bootstrap = trade_bootstrap_summary(
        pre,
        simulation_count=simulation_count,
        seed=TRADE_BOOTSTRAP_SEED,
    )
    current_trade_bootstrap = trade_bootstrap_summary(
        current,
        simulation_count=simulation_count,
        seed=TRADE_BOOTSTRAP_SEED + 1,
    )
    pre_year_bootstrap = year_block_bootstrap_summary(
        pre,
        simulation_count=simulation_count,
        seed=YEAR_BLOCK_BOOTSTRAP_SEED,
    )
    current_year_bootstrap = year_block_bootstrap_summary(
        current,
        simulation_count=simulation_count,
        seed=YEAR_BLOCK_BOOTSTRAP_SEED + 1,
    )
    status, status_reasons = fast156_status(
        pre_year_bootstrap=pre_year_bootstrap,
        current_trade_bootstrap=current_trade_bootstrap,
        current_year_bootstrap=current_year_bootstrap,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "bootstrap-risk-metrics.csv"
    summary_path = output_root / "bootstrap-risk-summary.json"
    summary = {
        "simulation_count": simulation_count,
        "seeds": {
            "pre_trade_bootstrap": TRADE_BOOTSTRAP_SEED,
            "current_trade_bootstrap": TRADE_BOOTSTRAP_SEED + 1,
            "pre_year_block_bootstrap": YEAR_BLOCK_BOOTSTRAP_SEED,
            "current_year_block_bootstrap": YEAR_BLOCK_BOOTSTRAP_SEED + 1,
        },
        "drawdown_thresholds": list(DRAWDOWN_THRESHOLDS),
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
    }
    write_json(summary_path, summary)
    pd.DataFrame(flat_metric_rows(summary)).to_csv(metrics_path, index=False)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast156_is_historical_bootstrap_risk_audit_not_strategy_approval",
            "fast151_still_requires_fresh_prospective_observation",
            "bootstrap_risk_tail_requires_position_sizing_before_shadow_or_testnet",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast156_fast151_bootstrap_risk_audit",
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
            "bootstrap_risk_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame.sort_values(["entry_timestamp", "pair", "candidate_id"], ignore_index=True)


def ordered_returns(trades: pd.DataFrame) -> np.ndarray:
    return (
        trades.sort_values(["exit_stress_timestamp", "entry_timestamp", "pair"])["net_stress_pips"]
        .astype("float64")
        .to_numpy()
    )


def max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    equity = np.cumsum(values)
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def actual_risk_summary(trades: pd.DataFrame) -> dict[str, Any]:
    values = ordered_returns(trades)
    summary = summarize_sample(trades)
    return {
        "trade_count": summary["trade_count"],
        "year_count": summary["year_count"],
        "total_stress_pips": summary["total_stress_pips"],
        "mean_stress_pips": summary["mean_stress_pips"],
        "lower_stress_pips": summary["lower_stress_pips"],
        "max_drawdown_pips": max_drawdown(values),
        "two_consecutive_negative_years": summary["two_consecutive_negative_years"],
    }


def trade_bootstrap_summary(
    trades: pd.DataFrame,
    *,
    simulation_count: int,
    seed: int,
) -> dict[str, Any]:
    values = ordered_returns(trades)
    rng = np.random.default_rng(seed)
    total_pips = np.empty(simulation_count)
    mean_pips = np.empty(simulation_count)
    drawdowns = np.empty(simulation_count)
    for index in range(simulation_count):
        path = rng.choice(values, size=len(values), replace=True)
        total_pips[index] = path.sum()
        mean_pips[index] = path.mean()
        drawdowns[index] = max_drawdown(path)
    return simulation_distribution_summary(
        total_pips=total_pips,
        mean_pips=mean_pips,
        drawdowns=drawdowns,
        extra={"bootstrap_mode": "trade", "resampled_trade_count": int(len(values))},
    )


def year_block_bootstrap_summary(
    trades: pd.DataFrame,
    *,
    simulation_count: int,
    seed: int,
) -> dict[str, Any]:
    frame = trades.copy()
    frame["year"] = frame["entry_timestamp"].dt.year
    years = sorted(frame["year"].unique())
    groups = {
        year: ordered_returns(frame.loc[frame["year"].eq(year)].copy())
        for year in years
    }
    rng = np.random.default_rng(seed)
    total_pips = np.empty(simulation_count)
    mean_pips = np.empty(simulation_count)
    drawdowns = np.empty(simulation_count)
    losing_year_counts = np.empty(simulation_count)
    for index in range(simulation_count):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        chunks = [groups[year] for year in sampled_years]
        path = np.concatenate(chunks)
        yearly_totals = np.array([groups[year].sum() for year in sampled_years], dtype="float64")
        total_pips[index] = path.sum()
        mean_pips[index] = path.mean()
        drawdowns[index] = max_drawdown(path)
        losing_year_counts[index] = np.sum(yearly_totals < 0)
    return simulation_distribution_summary(
        total_pips=total_pips,
        mean_pips=mean_pips,
        drawdowns=drawdowns,
        extra={
            "bootstrap_mode": "year_block",
            "resampled_year_count": int(len(years)),
            "losing_year_count_p50": quantile(losing_year_counts, 0.50),
            "losing_year_count_p95": quantile(losing_year_counts, 0.95),
            "probability_at_least_3_losing_years": probability(losing_year_counts >= 3),
        },
    )


def simulation_distribution_summary(
    *,
    total_pips: np.ndarray,
    mean_pips: np.ndarray,
    drawdowns: np.ndarray,
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        **extra,
        "total_pips_p05": quantile(total_pips, 0.05),
        "total_pips_p50": quantile(total_pips, 0.50),
        "total_pips_p95": quantile(total_pips, 0.95),
        "mean_pips_p05": quantile(mean_pips, 0.05),
        "mean_pips_p50": quantile(mean_pips, 0.50),
        "mean_pips_p95": quantile(mean_pips, 0.95),
        "max_drawdown_pips_p05": quantile(drawdowns, 0.05),
        "max_drawdown_pips_p50": quantile(drawdowns, 0.50),
        "max_drawdown_pips_p95": quantile(drawdowns, 0.95),
        "probability_total_pips_lte_0": probability(total_pips <= 0),
    }
    for threshold in DRAWDOWN_THRESHOLDS:
        payload[f"probability_max_drawdown_lte_{abs(int(threshold))}_pips"] = probability(drawdowns <= threshold)
    return payload


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
    return rows


def fast156_status(
    *,
    pre_year_bootstrap: dict[str, Any],
    current_trade_bootstrap: dict[str, Any],
    current_year_bootstrap: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "bootstrap_risk_audit"]
    flags: list[str] = []
    if pre_year_bootstrap["probability_total_pips_lte_0"] > 0.01:
        flags.append("pre_year_block_probability_total_nonpositive_above_1pct")
    if pre_year_bootstrap["probability_max_drawdown_lte_1000_pips"] > 0.10:
        flags.append("pre_year_block_drawdown_tail_above_10pct")
    if current_trade_bootstrap["probability_max_drawdown_lte_1000_pips"] > 0.10:
        flags.append("current_trade_bootstrap_drawdown_tail_above_10pct")
    if current_year_bootstrap["probability_at_least_3_losing_years"] > 0.20:
        flags.append("current_year_block_losing_year_tail_above_20pct")
    if flags:
        return (
            "FAST156_FAST151_EDGE_POSITIVE_BUT_DRAWDOWN_TAIL_REQUIRES_SIZING_AND_PROSPECTIVE_OBSERVATION",
            reasons + flags + ["fresh_preregistered_prospective_observation_still_required"],
        )
    return (
        "FAST156_FAST151_BOOTSTRAP_RISK_ACCEPTABLE_REQUIRES_PROSPECTIVE_OBSERVATION",
        reasons + ["fresh_preregistered_prospective_observation_required_before_any_execution"],
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
        "simulation_count": summary["simulation_count"],
        "pre_year_block": {
            key: summary["pre"]["year_block_bootstrap"][key]
            for key in (
                "total_pips_p05",
                "probability_total_pips_lte_0",
                "max_drawdown_pips_p05",
                "probability_max_drawdown_lte_1000_pips",
                "probability_max_drawdown_lte_1500_pips",
            )
        },
        "current_trade_bootstrap": {
            key: summary["current"]["trade_bootstrap"][key]
            for key in (
                "total_pips_p05",
                "probability_total_pips_lte_0",
                "max_drawdown_pips_p05",
                "probability_max_drawdown_lte_1000_pips",
                "probability_max_drawdown_lte_1500_pips",
            )
        },
        "current_year_block": {
            key: summary["current"]["year_block_bootstrap"][key]
            for key in (
                "total_pips_p05",
                "probability_total_pips_lte_0",
                "max_drawdown_pips_p05",
                "probability_max_drawdown_lte_1000_pips",
                "probability_at_least_3_losing_years",
            )
        },
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
