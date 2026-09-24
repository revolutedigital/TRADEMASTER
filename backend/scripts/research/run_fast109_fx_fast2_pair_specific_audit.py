"""Fast109 pair-specific diagnostic audit for the rejected Fast2 FX grid.

Fast2 evaluated the pre-registered FX strategy grid in aggregate and rejected
every configuration. Fast109 asks a narrower diagnostic question: did the
aggregate rejection hide a pair-specific survivor? This script reruns the same
Fast2 simulation machinery, aggregates each config/pair separately, and writes
research-only evidence. It never sends orders and never approves a strategy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.fx import strategy as fx  # noqa: E402
from app.fx.bars import aggregate_minutes  # noqa: E402
from app.fx.instruments import Instrument  # noqa: E402
from app.fx.sim import core  # noqa: E402
from app.fx.sim.costs import FUSION_ZERO, STRESS, finalize_trades, prepare_run  # noqa: E402
from scripts.research import fx_fast2_lab as fast2  # noqa: E402
from scripts.research import fx_fast_lab as fast_lab  # noqa: E402
from scripts.research.fx_fast_stats import MDE_STANDARD_ERRORS  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast109" / "fx-fast2-pair-specific-audit"
)
DEFAULT_MIN_TRADES = 100
DEFAULT_MIN_MONTHS = 12
DEFAULT_MIN_YEARS = 3


@dataclass
class ScenarioAccumulator:
    trade_count: int = 0
    total_r: float = 0.0
    period_sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    period_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    year_sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    year_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    day_sums: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    day_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, *, r_values: np.ndarray, entry_seconds: np.ndarray) -> None:
        if len(r_values) != len(entry_seconds):
            raise ValueError("r_values and entry_seconds must have the same length")
        if len(r_values) == 0:
            return
        self.trade_count += int(len(r_values))
        self.total_r += float(np.sum(r_values))

        timestamps = pd.to_datetime(entry_seconds, unit="s", utc=True)
        periods = timestamps.strftime("%Y-%m")
        years = timestamps.strftime("%Y")
        days = fast_lab.fx_day_array(entry_seconds)

        for period, value in zip(periods, r_values, strict=True):
            name = str(period)
            self.period_sums[name] += float(value)
            self.period_counts[name] += 1
        for year, value in zip(years, r_values, strict=True):
            name = str(year)
            self.year_sums[name] += float(value)
            self.year_counts[name] += 1
        for day, value in zip(days, r_values, strict=True):
            day_key = int(day)
            self.day_sums[day_key] += float(value)
            self.day_counts[day_key] += 1

    @property
    def mean_r(self) -> float | None:
        if self.trade_count == 0:
            return None
        return self.total_r / self.trade_count


@dataclass
class PairConfigAccumulator:
    config: str
    pair: str
    family: str
    base: ScenarioAccumulator = field(default_factory=ScenarioAccumulator)
    stress: ScenarioAccumulator = field(default_factory=ScenarioAccumulator)
    min_lot_stop_fit_count: int = 0
    min_lot_stop_total_count: int = 0

    def add_stop_fit(self, *, stop_pips: np.ndarray, limit_pips: float) -> None:
        if len(stop_pips) == 0:
            return
        self.min_lot_stop_fit_count += int((stop_pips <= limit_pips).sum())
        self.min_lot_stop_total_count += int(len(stop_pips))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(fast2.SAMPLES), default="discovery")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--min-months", type=int, default=DEFAULT_MIN_MONTHS)
    parser.add_argument("--min-years", type=int, default=DEFAULT_MIN_YEARS)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    arguments = parser.parse_args()

    report = run_fast109_fx_fast2_pair_specific_audit(
        stage=arguments.stage,
        output_root=arguments.output_root,
        min_trades=arguments.min_trades,
        min_months=arguments.min_months,
        min_years=arguments.min_years,
        max_configs=arguments.max_configs,
        progress=arguments.progress,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast109_fx_fast2_pair_specific_audit(
    *,
    stage: str = "discovery",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    min_trades: int = DEFAULT_MIN_TRADES,
    min_months: int = DEFAULT_MIN_MONTHS,
    min_years: int = DEFAULT_MIN_YEARS,
    max_configs: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    validate_parameters(
        stage=stage,
        min_trades=min_trades,
        min_months=min_months,
        min_years=min_years,
        max_configs=max_configs,
    )
    output_root = output_root if output_root.is_absolute() else (Path.cwd() / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    configs = fast2.configurations()
    if max_configs is not None:
        configs = dict(list(configs.items())[:max_configs])

    with working_directory(BACKEND_ROOT):
        matrices, windows, _universe, rates = fast2.setup(stage)
        accumulators = simulate_pair_config_accumulators(
            matrices=matrices,
            windows=windows,
            rates=rates,
            configs=configs,
            progress=progress,
        )

    evaluations = sorted(
        (
            summarize_pair_config_accumulator(accumulator)
            for accumulator in accumulators.values()
            if accumulator.base.trade_count or accumulator.stress.trade_count
        ),
        key=lambda row: (row["config"], row["pair"]),
    )
    strict_candidates = strict_pair_config_candidates(
        evaluations,
        min_trades=min_trades,
        min_months=min_months,
        min_years=min_years,
    )
    weak_candidates = weak_pair_config_candidates(
        evaluations,
        min_trades=min_trades,
        min_months=min_months,
        min_years=min_years,
    )
    top_diagnostics = top_pair_config_diagnostics(evaluations, min_trades=min(25, min_trades))

    evaluations_path = output_root / "pair_config_evaluations.json"
    strict_path = output_root / "strict_pair_config_candidates.json"
    weak_path = output_root / "weak_pair_config_candidates.json"
    top_path = output_root / "top_pair_config_diagnostics.json"
    write_json(evaluations_path, evaluations)
    write_json(strict_path, strict_candidates)
    write_json(weak_path, weak_candidates)
    write_json(top_path, top_diagnostics)

    status, status_reasons = fast109_status(
        strict_candidate_count=len(strict_candidates),
        weak_candidate_count=len(weak_candidates),
    )
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast109_is_pair_specific_diagnostic_for_rejected_fast2_grid",
            "post_hoc_pair_specific_diagnostic_requires_fresh_preregistration",
            "same_fast2_simulation_machinery_no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast109_fx_fast2_pair_specific_audit",
        "status": status,
        "status_reasons": status_reasons,
        "stage": stage,
        "config_count": len(configs),
        "pair_config_evaluation_count": len(evaluations),
        "min_trades": min_trades,
        "min_months": min_months,
        "min_years": min_years,
        "max_configs": max_configs,
        "weak_candidate_count": len(weak_candidates),
        "strict_candidate_count": len(strict_candidates),
        "top_pair_config_diagnostics": top_diagnostics,
        "evaluations_path": str(evaluations_path),
        "evaluations_sha256": sha256_file(evaluations_path),
        "strict_candidates_path": str(strict_path),
        "strict_candidates_sha256": sha256_file(strict_path),
        "weak_candidates_path": str(weak_path),
        "weak_candidates_sha256": sha256_file(weak_path),
        "top_diagnostics_path": str(top_path),
        "top_diagnostics_sha256": sha256_file(top_path),
    }
    return write_report(output_root, report)


def validate_parameters(
    *,
    stage: str,
    min_trades: int,
    min_months: int,
    min_years: int,
    max_configs: int | None,
) -> None:
    if stage not in fast2.SAMPLES:
        raise ValueError(f"unknown stage: {stage}")
    if min_trades <= 0:
        raise ValueError("min_trades must be positive")
    if min_months <= 0:
        raise ValueError("min_months must be positive")
    if min_years <= 0:
        raise ValueError("min_years must be positive")
    if max_configs is not None and max_configs <= 0:
        raise ValueError("max_configs must be positive when provided")


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def simulate_pair_config_accumulators(
    *,
    matrices: dict[str, np.ndarray],
    windows: dict[str, list[tuple[float, float]]],
    rates,
    configs: dict[str, fast_lab.Configuration],
    progress: bool = False,
) -> dict[tuple[str, str], PairConfigAccumulator]:
    accumulators: dict[tuple[str, str], PairConfigAccumulator] = {}
    for pair_index, (pair, pair_windows) in enumerate(windows.items(), start=1):
        instrument = Instrument.from_symbol(pair)
        if progress:
            print(  # noqa: T201
                f"pair {pair_index}/{len(windows)} {pair}: "
                f"{len(pair_windows)} windows, {len(configs)} configs"
            )
        for window in pair_windows:
            minutes = fast_lab.pair_minutes(pair, matrices, window, coins=None)
            if len(minutes) < 2:
                continue
            price = float(np.median(0.5 * (minutes[:, fx.BID_CLOSE] + minutes[:, fx.ASK_CLOSE])))
            aggregated: dict[int, np.ndarray] = {}
            prepared: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
            for key, config in configs.items():
                if pair not in config.pairs:
                    continue
                accumulator = accumulators.setdefault(
                    (key, pair),
                    PairConfigAccumulator(config=key, pair=pair, family=family_from_config(key)),
                )
                params = config.params(pair)
                for scenario in (FUSION_ZERO, STRESS):
                    if (config.seconds, scenario.name) not in prepared:
                        if config.seconds not in aggregated:
                            aggregated[config.seconds] = aggregate_minutes(minutes, config.seconds)
                        prepared[(config.seconds, scenario.name)] = prepare_run(
                            aggregated[config.seconds],
                            instrument,
                            scenario,
                        )
                    scenario_bars, slippage = prepared[(config.seconds, scenario.name)]
                    if pair == fast_lab.SYNTHETIC:
                        slippage = 2.0 * slippage
                    result = core.run_simulation(
                        config.step,
                        config.init,
                        params,
                        config.state_size,
                        scenario_bars,
                        slippage,
                        anchor_signal=True,
                        max_entry_gap=float(config.seconds),
                    )
                    trades = finalize_trades(
                        result,
                        scenario_bars,
                        instrument,
                        scenario,
                        commission_pips=fast_lab.commission_pips(
                            pair,
                            price,
                            rates,
                            scenario.commission_base_per_lot_per_side,
                        ),
                        calendar=fast_lab.CALENDAR,
                    )
                    entry_seconds = scenario_bars[result[0], fx.BAR_TIME]
                    r_values = trades["r_multiple"].to_numpy(dtype=float)
                    if scenario is STRESS:
                        accumulator.stress.add(r_values=r_values, entry_seconds=entry_seconds)
                    else:
                        accumulator.base.add(r_values=r_values, entry_seconds=entry_seconds)
                        quote_group = "JPY" if instrument.quote == "JPY" else "other"
                        accumulator.add_stop_fit(
                            stop_pips=trades["stop_pips"].to_numpy(dtype=float),
                            limit_pips=fast_lab.MIN_LOT_STOP_PIPS[quote_group],
                        )
    return accumulators


def family_from_config(config_key: str) -> str:
    if config_key.startswith(("F1a", "F1b")):
        return config_key[:3]
    if config_key.startswith(("F2a", "F2b")):
        return config_key[:3]
    if config_key.startswith("F3"):
        return "F3"
    if config_key.startswith(("F5a", "F5b")):
        return config_key[:3]
    return config_key.split(".", 1)[0]


def summarize_pair_config_accumulator(accumulator: PairConfigAccumulator) -> dict[str, Any]:
    base_months = period_summary(accumulator.base.period_sums, accumulator.base.period_counts)
    stress_months = period_summary(accumulator.stress.period_sums, accumulator.stress.period_counts)
    base_years = period_summary(accumulator.base.year_sums, accumulator.base.year_counts)
    stress_years = period_summary(accumulator.stress.year_sums, accumulator.stress.year_counts)
    base_t = clustered_t_from_days(accumulator.base.day_sums, accumulator.base.day_counts)
    stress_t = clustered_t_from_days(accumulator.stress.day_sums, accumulator.stress.day_counts)

    return plain_json(
        {
            "config": accumulator.config,
            "pair": accumulator.pair,
            "family": accumulator.family,
            "base_trade_count": accumulator.base.trade_count,
            "stress_trade_count": accumulator.stress.trade_count,
            "base_total_r": accumulator.base.total_r,
            "stress_total_r": accumulator.stress.total_r,
            "base_mean_r": accumulator.base.mean_r,
            "stress_mean_r": accumulator.stress.mean_r,
            "base_t": base_t["t"],
            "stress_t": stress_t["t"],
            "base_mde_r": base_t["mde_r"],
            "stress_mde_r": stress_t["mde_r"],
            "month_count": len(stress_months),
            "year_count": len(stress_years),
            "positive_base_month_count": count_positive_periods(base_months),
            "positive_stress_month_count": count_positive_periods(stress_months),
            "positive_base_year_count": count_positive_periods(base_years),
            "positive_stress_year_count": count_positive_periods(stress_years),
            "min_month_stress_mean_r": min_period_mean(stress_months),
            "min_year_stress_mean_r": min_period_mean(stress_years),
            "stop_fits_min_lot": stop_fit_share(accumulator),
            "base_months": base_months,
            "stress_months": stress_months,
            "base_years": base_years,
            "stress_years": stress_years,
        }
    )


def period_summary(
    period_sums: dict[str, float],
    period_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for period in sorted(period_counts):
        count = int(period_counts[period])
        total = float(period_sums[period])
        rows.append({"period": period, "trade_count": count, "total_r": total, "mean_r": total / count})
    return rows


def count_positive_periods(rows: list[dict[str, Any]]) -> int:
    return sum(row["mean_r"] > 0 for row in rows)


def min_period_mean(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return min(row["mean_r"] for row in rows)


def clustered_t_from_days(day_sums: dict[int, float], day_counts: dict[int, int]) -> dict[str, float | None]:
    total_count = sum(day_counts.values())
    if total_count == 0:
        return {"t": None, "error": None, "mde_r": None}
    mean = sum(day_sums.values()) / total_count
    residual_square_sum = sum(
        (day_sums[day] - mean * day_counts[day]) ** 2
        for day in day_counts
    )
    error = math.sqrt(residual_square_sum) / total_count
    if error == 0:
        t_value = None
    else:
        t_value = mean / error
    return {"t": t_value, "error": error, "mde_r": MDE_STANDARD_ERRORS * error}


def stop_fit_share(accumulator: PairConfigAccumulator) -> float | None:
    if accumulator.min_lot_stop_total_count == 0:
        return None
    return accumulator.min_lot_stop_fit_count / accumulator.min_lot_stop_total_count


def strict_pair_config_candidates(
    evaluations: list[dict[str, Any]],
    *,
    min_trades: int,
    min_months: int,
    min_years: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in evaluations
        if passes_common_candidate_filter(row, min_trades=min_trades, min_months=min_months, min_years=min_years)
        and row["positive_stress_year_count"] == row["year_count"]
        and row["positive_base_year_count"] == row["year_count"]
    ]


def weak_pair_config_candidates(
    evaluations: list[dict[str, Any]],
    *,
    min_trades: int,
    min_months: int,
    min_years: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in evaluations
        if passes_common_candidate_filter(row, min_trades=min_trades, min_months=min_months, min_years=min_years)
    ]


def passes_common_candidate_filter(
    row: dict[str, Any],
    *,
    min_trades: int,
    min_months: int,
    min_years: int,
) -> bool:
    base_mean = row["base_mean_r"]
    stress_mean = row["stress_mean_r"]
    return (
        row["base_trade_count"] >= min_trades
        and row["stress_trade_count"] >= min_trades
        and row["month_count"] >= min_months
        and row["year_count"] >= min_years
        and base_mean is not None
        and stress_mean is not None
        and base_mean > 0
        and stress_mean > 0
    )


def top_pair_config_diagnostics(
    evaluations: list[dict[str, Any]],
    *,
    min_trades: int,
    limit: int = 25,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in evaluations
        if row["base_trade_count"] >= min_trades and row["stress_trade_count"] >= min_trades
    ]
    return sorted(
        eligible,
        key=lambda row: (
            sortable_number(row["stress_mean_r"]),
            sortable_number(row["base_mean_r"]),
            sortable_number(row["stress_t"]),
            row["stress_trade_count"],
        ),
        reverse=True,
    )[:limit]


def sortable_number(value: float | None) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def fast109_status(
    *,
    strict_candidate_count: int,
    weak_candidate_count: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if strict_candidate_count:
        return (
            "FAST109_FX_FAST2_PAIR_SPECIFIC_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"strict_pair_config_candidates:{strict_candidate_count}",
                "post_hoc_pair_specific_diagnostic_requires_fresh_preregistration",
            ],
        )
    if weak_candidate_count:
        return (
            "FAST109_FX_FAST2_PAIR_SPECIFIC_WEAK_ONLY_FAILED_STABILITY",
            reasons
            + [
                f"weak_pair_config_candidates:{weak_candidate_count}",
                "positive_base_and_stress_but_not_stable_across_all_years",
            ],
        )
    return (
        "FAST109_FX_FAST2_PAIR_SPECIFIC_FAILED_SCREEN",
        reasons + ["no_pair_config_positive_base_and_stress_after_minimum_sample"],
    )


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(plain_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(plain_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain_json(item) for item in value]
    if hasattr(value, "item"):
        return plain_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "stage": report["stage"],
        "config_count": report["config_count"],
        "pair_config_evaluation_count": report["pair_config_evaluation_count"],
        "weak_candidate_count": report["weak_candidate_count"],
        "strict_candidate_count": report["strict_candidate_count"],
        "top_pair_config_diagnostics": [
            compact_pair_config_diagnostic(row)
            for row in report["top_pair_config_diagnostics"][:10]
        ],
        "report_path": report["report_path"],
    }


def compact_pair_config_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": row["config"],
        "pair": row["pair"],
        "base_trade_count": row["base_trade_count"],
        "stress_trade_count": row["stress_trade_count"],
        "base_mean_r": row["base_mean_r"],
        "stress_mean_r": row["stress_mean_r"],
        "base_t": row["base_t"],
        "stress_t": row["stress_t"],
        "positive_base_year_count": row["positive_base_year_count"],
        "positive_stress_year_count": row["positive_stress_year_count"],
        "year_count": row["year_count"],
        "min_year_stress_mean_r": row["min_year_stress_mean_r"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
