"""Fast111 regime scout for the Fast110 USDJPY London-breakout weak survivors.

Fast110 left a narrow weak line: USDJPY F1a London breakout with the 0.6-2.0 ATR
range filter and exit at 16:30. Fast111 asks whether simple entry-known regime
filters can turn that weak line into a more stable research candidate. The
filter family is intentionally small and explainable: side, London weekday,
London entry hour and stop-size bucket fixed from the discovery sample.

This is research-only. It never sends orders and never approves a strategy for
Demo, Testnet or LIVE execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
from scripts.research.run_fast109_fx_fast2_pair_specific_audit import (  # noqa: E402
    clustered_t_from_days,
    plain_json,
    sha256_file,
    working_directory,
)


DEFAULT_SOURCE_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast110"
    / "fx-fast2-pair-validation"
    / "weak_validation_survivors.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast111" / "usdjpy-breakout-regime-scout"
)
DEFAULT_DISCOVERY_MIN_TRADES = 20
DEFAULT_REPLICATION_MIN_TRADES = 20
DEFAULT_CONFIRMATION_MIN_TRADES = 10
PAIR = "USDJPY"
LONDON = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class FilterSpec:
    filter_id: str
    conditions: tuple[tuple[str, Any], ...]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--discovery-min-trades", type=int, default=DEFAULT_DISCOVERY_MIN_TRADES)
    parser.add_argument("--replication-min-trades", type=int, default=DEFAULT_REPLICATION_MIN_TRADES)
    parser.add_argument("--confirmation-min-trades", type=int, default=DEFAULT_CONFIRMATION_MIN_TRADES)
    parser.add_argument("--progress", action="store_true")
    arguments = parser.parse_args()
    report = run_fast111_usdjpy_breakout_regime_scout(
        source_path=arguments.source_path,
        output_root=arguments.output_root,
        discovery_min_trades=arguments.discovery_min_trades,
        replication_min_trades=arguments.replication_min_trades,
        confirmation_min_trades=arguments.confirmation_min_trades,
        progress=arguments.progress,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast111_usdjpy_breakout_regime_scout(
    *,
    source_path: Path = DEFAULT_SOURCE_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    discovery_min_trades: int = DEFAULT_DISCOVERY_MIN_TRADES,
    replication_min_trades: int = DEFAULT_REPLICATION_MIN_TRADES,
    confirmation_min_trades: int = DEFAULT_CONFIRMATION_MIN_TRADES,
    progress: bool = False,
) -> dict[str, Any]:
    validate_parameters(
        discovery_min_trades=discovery_min_trades,
        replication_min_trades=replication_min_trades,
        confirmation_min_trades=confirmation_min_trades,
    )
    output_root = output_root if output_root.is_absolute() else (Path.cwd() / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_path = source_path if source_path.is_absolute() else (Path.cwd() / source_path).resolve()

    configs = load_usdjpy_config_keys(source_path)
    trades = build_trade_table(config_keys=configs, progress=progress)
    trades = assign_stop_buckets_from_discovery(trades)
    filters = build_filter_specs(trades[trades["stage"] == "discovery"])
    evaluations = evaluate_filters(
        trades,
        filters=filters,
        discovery_min_trades=discovery_min_trades,
        replication_min_trades=replication_min_trades,
        confirmation_min_trades=confirmation_min_trades,
    )
    selected = [row for row in evaluations if row["selected_on_discovery"]]
    weak_survivors = [row for row in selected if row["weak_validated"]]
    strict_survivors = [row for row in selected if row["strict_validated"]]

    trade_path = output_root / "trade_rows.parquet"
    filters_path = output_root / "filter_evaluations.json"
    selected_path = output_root / "selected_discovery_filters.json"
    weak_path = output_root / "weak_validation_survivors.json"
    strict_path = output_root / "strict_validation_survivors.json"
    trades.to_parquet(trade_path, index=False)
    write_json(filters_path, evaluations)
    write_json(selected_path, selected)
    write_json(weak_path, weak_survivors)
    write_json(strict_path, strict_survivors)

    status, status_reasons = fast111_status(
        strict_survivor_count=len(strict_survivors),
        weak_survivor_count=len(weak_survivors),
        selected_count=len(selected),
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
            "fast111_is_post_hoc_regime_scout_for_fast110_weak_line",
            "filters_are_entry_known_but_selected_on_historical_discovery",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast111_usdjpy_breakout_regime_scout",
        "status": status,
        "status_reasons": status_reasons,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "pair": PAIR,
        "config_count": len(configs),
        "configs": configs,
        "filter_count": len(filters),
        "evaluation_count": len(evaluations),
        "selected_discovery_filter_count": len(selected),
        "weak_survivor_count": len(weak_survivors),
        "strict_survivor_count": len(strict_survivors),
        "discovery_min_trades": discovery_min_trades,
        "replication_min_trades": replication_min_trades,
        "confirmation_min_trades": confirmation_min_trades,
        "trade_rows_path": str(trade_path),
        "trade_rows_sha256": sha256_file(trade_path),
        "filter_evaluations_path": str(filters_path),
        "filter_evaluations_sha256": sha256_file(filters_path),
        "selected_filters_path": str(selected_path),
        "selected_filters_sha256": sha256_file(selected_path),
        "weak_survivors_path": str(weak_path),
        "weak_survivors_sha256": sha256_file(weak_path),
        "strict_survivors_path": str(strict_path),
        "strict_survivors_sha256": sha256_file(strict_path),
        "top_selected_filters": top_filters(selected),
        "top_weak_survivors": top_filters(weak_survivors),
    }
    return write_report(output_root, report)


def validate_parameters(
    *,
    discovery_min_trades: int,
    replication_min_trades: int,
    confirmation_min_trades: int,
) -> None:
    if discovery_min_trades <= 0:
        raise ValueError("discovery_min_trades must be positive")
    if replication_min_trades <= 0:
        raise ValueError("replication_min_trades must be positive")
    if confirmation_min_trades <= 0:
        raise ValueError("confirmation_min_trades must be positive")


def load_usdjpy_config_keys(source_path: Path) -> list[str]:
    if not source_path.exists():
        raise FileNotFoundError(f"source not found: {source_path}")
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    configs = sorted({str(row["config"]) for row in rows if row.get("pair") == PAIR})
    if not configs:
        raise ValueError(f"no {PAIR} configs found in {source_path}")
    return configs


def build_trade_table(*, config_keys: list[str], progress: bool) -> pd.DataFrame:
    frames = []
    for stage in ("discovery", "replication", "confirmation"):
        frames.append(build_stage_trade_table(stage=stage, config_keys=config_keys, progress=progress))
    return pd.concat(frames, ignore_index=True)


def build_stage_trade_table(*, stage: str, config_keys: list[str], progress: bool) -> pd.DataFrame:
    configs_by_key = fast2.configurations()
    configs = {key: configs_by_key[key] for key in config_keys}
    rows = []
    with working_directory(BACKEND_ROOT):
        matrices, windows, _universe, rates = fast2.setup(stage)
        pair_windows = windows[PAIR]
        if progress:
            print(f"{stage}: {PAIR} {len(pair_windows)} windows, {len(configs)} configs")  # noqa: T201
        instrument = Instrument.from_symbol(PAIR)
        for window in pair_windows:
            minutes = fast_lab.pair_minutes(PAIR, matrices, window, coins=None)
            if len(minutes) < 2:
                continue
            price = float(np.median(0.5 * (minutes[:, fx.BID_CLOSE] + minutes[:, fx.ASK_CLOSE])))
            aggregated: dict[int, np.ndarray] = {}
            prepared: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
            for key, config in configs.items():
                params = config.params(PAIR)
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
                            PAIR,
                            price,
                            rates,
                            scenario.commission_base_per_lot_per_side,
                        ),
                        calendar=fast_lab.CALENDAR,
                    )
                    rows.extend(trade_rows(stage=stage, config=key, scenario=scenario.name, trades=trades))
    if not rows:
        raise RuntimeError(f"no trades generated for {stage}")
    return pd.DataFrame(rows)


def trade_rows(*, stage: str, config: str, scenario: str, trades: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for trade in trades.to_dict("records"):
        entry = pd.Timestamp(trade["entry_time"])
        london_entry = entry.tz_convert(LONDON)
        rows.append(
            {
                "stage": stage,
                "config": config,
                "pair": PAIR,
                "scenario": scenario,
                "entry_time": entry.isoformat(),
                "entry_epoch": entry.timestamp(),
                "fx_day": int(fast_lab.fx_day_array(np.array([entry.timestamp()], dtype=float))[0]),
                "london_weekday": int(london_entry.weekday()),
                "london_weekday_name": london_entry.day_name(),
                "london_entry_hour": int(london_entry.hour),
                "london_month": int(london_entry.month),
                "year": int(london_entry.year),
                "side": int(trade["side"]),
                "side_label": "LONG" if int(trade["side"]) > 0 else "SHORT",
                "stop_pips": float(trade["stop_pips"]),
                "r_multiple": float(trade["r_multiple"]),
            }
        )
    return rows


def assign_stop_buckets_from_discovery(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    frame["stop_bucket"] = "unknown"
    for config, config_rows in frame[frame["stage"] == "discovery"].groupby("config"):
        stress_stops = config_rows.loc[config_rows["scenario"] == STRESS.name, "stop_pips"]
        low = float(stress_stops.quantile(1 / 3))
        high = float(stress_stops.quantile(2 / 3))
        mask = frame["config"] == config
        frame.loc[mask, "stop_bucket"] = frame.loc[mask, "stop_pips"].map(
            lambda value, low=low, high=high: stop_bucket(value, low=low, high=high)
        )
    return frame


def stop_bucket(value: float, *, low: float, high: float) -> str:
    if value <= low:
        return "narrow"
    if value <= high:
        return "middle"
    return "wide"


def build_filter_specs(discovery: pd.DataFrame) -> list[FilterSpec]:
    specs = [FilterSpec("all", ())]
    dimensions = {
        "side_label": sorted(discovery["side_label"].unique().tolist()),
        "london_weekday": sorted(int(value) for value in discovery["london_weekday"].unique()),
        "london_entry_hour": sorted(int(value) for value in discovery["london_entry_hour"].unique()),
        "stop_bucket": ["narrow", "middle", "wide"],
    }
    for dimension, values in dimensions.items():
        for value in values:
            specs.append(make_filter({dimension: value}))
    pairings = (
        ("side_label", "london_weekday"),
        ("side_label", "london_entry_hour"),
        ("side_label", "stop_bucket"),
        ("london_weekday", "stop_bucket"),
        ("london_entry_hour", "stop_bucket"),
    )
    for left, right in pairings:
        for left_value in dimensions[left]:
            for right_value in dimensions[right]:
                specs.append(make_filter({left: left_value, right: right_value}))
    unique = {spec.filter_id: spec for spec in specs}
    return [unique[key] for key in sorted(unique)]


def make_filter(conditions: dict[str, Any]) -> FilterSpec:
    ordered = tuple(sorted(conditions.items()))
    label = "&".join(f"{key}={value}" for key, value in ordered)
    return FilterSpec(label, ordered)


def evaluate_filters(
    trades: pd.DataFrame,
    *,
    filters: list[FilterSpec],
    discovery_min_trades: int,
    replication_min_trades: int,
    confirmation_min_trades: int,
) -> list[dict[str, Any]]:
    rows = []
    for config in sorted(trades["config"].unique()):
        config_trades = trades[trades["config"] == config]
        for spec in filters:
            stage_summaries = {
                stage: summarize_filter(config_trades[config_trades["stage"] == stage], spec)
                for stage in ("discovery", "replication", "confirmation")
            }
            gates = {
                "discovery": stage_gate(
                    stage_summaries["discovery"],
                    min_trades=discovery_min_trades,
                    strict_year_share=0.8,
                ),
                "replication": stage_gate(
                    stage_summaries["replication"],
                    min_trades=replication_min_trades,
                    strict_year_share=1.0,
                ),
                "confirmation": stage_gate(
                    stage_summaries["confirmation"],
                    min_trades=confirmation_min_trades,
                    strict_year_share=1.0,
                ),
            }
            selected = gates["discovery"]["selection_pass"]
            weak_validated = (
                selected
                and gates["replication"]["weak_pass"]
                and gates["confirmation"]["weak_pass"]
            )
            strict_validated = (
                selected
                and gates["replication"]["strict_pass"]
                and gates["confirmation"]["strict_pass"]
            )
            rows.append(
                plain_json(
                    {
                        "config": config,
                        "pair": PAIR,
                        "filter_id": spec.filter_id,
                        "conditions": dict(spec.conditions),
                        "discovery": gates["discovery"],
                        "replication": gates["replication"],
                        "confirmation": gates["confirmation"],
                        "selected_on_discovery": selected,
                        "weak_validated": weak_validated,
                        "strict_validated": strict_validated,
                    }
                )
            )
    return rows


def summarize_filter(stage_trades: pd.DataFrame, spec: FilterSpec) -> dict[str, Any]:
    filtered = apply_filter(stage_trades, spec)
    summaries = {}
    for scenario in (FUSION_ZERO.name, STRESS.name):
        scenario_rows = filtered[filtered["scenario"] == scenario]
        summaries[scenario] = scenario_summary(scenario_rows)
    return {
        "filter_id": spec.filter_id,
        "base": summaries[FUSION_ZERO.name],
        "stress": summaries[STRESS.name],
    }


def apply_filter(frame: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    selected = frame
    for column, value in spec.conditions:
        selected = selected[selected[column] == value]
    return selected


def scenario_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return empty_scenario_summary()
    day_sums = frame.groupby("fx_day")["r_multiple"].sum().to_dict()
    day_counts = frame.groupby("fx_day")["r_multiple"].size().to_dict()
    t_stats = clustered_t_from_days(day_sums, day_counts)
    year_stats = (
        frame.groupby("year", sort=True)["r_multiple"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .rename(columns={"size": "trade_count", "sum": "total_r", "mean": "mean_r"})
    )
    years = [
        {
            "year": int(row["year"]),
            "trade_count": int(row["trade_count"]),
            "total_r": float(row["total_r"]),
            "mean_r": float(row["mean_r"]),
        }
        for row in year_stats.to_dict("records")
    ]
    return {
        "trade_count": int(len(frame)),
        "total_r": float(frame["r_multiple"].sum()),
        "mean_r": float(frame["r_multiple"].mean()),
        "t": t_stats["t"],
        "mde_r": t_stats["mde_r"],
        "year_count": len(years),
        "positive_year_count": sum(year["mean_r"] > 0 for year in years),
        "min_year_mean_r": min((year["mean_r"] for year in years), default=None),
        "years": years,
    }


def empty_scenario_summary() -> dict[str, Any]:
    return {
        "trade_count": 0,
        "total_r": 0.0,
        "mean_r": None,
        "t": None,
        "mde_r": None,
        "year_count": 0,
        "positive_year_count": 0,
        "min_year_mean_r": None,
        "years": [],
    }


def stage_gate(summary: dict[str, Any], *, min_trades: int, strict_year_share: float) -> dict[str, Any]:
    base = summary["base"]
    stress = summary["stress"]
    failed = []
    if base["trade_count"] < min_trades:
        failed.append(f"base_trades_below_{min_trades}")
    if stress["trade_count"] < min_trades:
        failed.append(f"stress_trades_below_{min_trades}")
    if base["mean_r"] is None or base["mean_r"] <= 0:
        failed.append("base_mean_not_positive")
    if stress["mean_r"] is None or stress["mean_r"] <= 0:
        failed.append("stress_mean_not_positive")
    weak_pass = not failed

    strict_failed = list(failed)
    required_positive_years = math.ceil(strict_year_share * stress["year_count"])
    if stress["year_count"] <= 0:
        strict_failed.append("no_stress_years")
    elif stress["positive_year_count"] < required_positive_years:
        strict_failed.append("stress_year_share_too_low")
    if base["year_count"] <= 0:
        strict_failed.append("no_base_years")
    elif base["positive_year_count"] < math.ceil(strict_year_share * base["year_count"]):
        strict_failed.append("base_year_share_too_low")

    return {
        "selection_pass": weak_pass and not strict_failed,
        "weak_pass": weak_pass,
        "strict_pass": weak_pass and not strict_failed,
        "failed": strict_failed,
        "summary": compact_stage_summary(summary),
    }


def compact_stage_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_trade_count": summary["base"]["trade_count"],
        "base_mean_r": summary["base"]["mean_r"],
        "base_t": summary["base"]["t"],
        "base_year_count": summary["base"]["year_count"],
        "base_positive_year_count": summary["base"]["positive_year_count"],
        "stress_trade_count": summary["stress"]["trade_count"],
        "stress_mean_r": summary["stress"]["mean_r"],
        "stress_t": summary["stress"]["t"],
        "stress_year_count": summary["stress"]["year_count"],
        "stress_positive_year_count": summary["stress"]["positive_year_count"],
        "stress_min_year_mean_r": summary["stress"]["min_year_mean_r"],
    }


def top_filters(rows: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row["strict_validated"],
            row["weak_validated"],
            score_value(row, "confirmation"),
            score_value(row, "replication"),
            score_value(row, "discovery"),
        ),
        reverse=True,
    )[:limit]


def score_value(row: dict[str, Any], stage: str) -> float:
    value = row[stage]["summary"]["stress_mean_r"]
    if value is None:
        return float("-inf")
    return float(value)


def fast111_status(
    *,
    strict_survivor_count: int,
    weak_survivor_count: int,
    selected_count: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if strict_survivor_count:
        return (
            "FAST111_USDJPY_BREAKOUT_REGIME_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"strict_regime_survivors:{strict_survivor_count}",
                "fresh_preregistered_prospective_observation_required",
            ],
        )
    if weak_survivor_count:
        return (
            "FAST111_USDJPY_BREAKOUT_REGIME_WEAK_ONLY_FAILED_STRICT_STABILITY",
            reasons
            + [
                f"weak_regime_survivors:{weak_survivor_count}",
                "historical_filter_positive_but_failed_strict_oos_stability",
            ],
        )
    if selected_count:
        return (
            "FAST111_USDJPY_BREAKOUT_REGIME_FAILED_OUT_OF_SAMPLE",
            reasons + ["discovery_filters_failed_replication_or_confirmation"],
        )
    return (
        "FAST111_USDJPY_BREAKOUT_REGIME_FAILED_DISCOVERY_SELECTION",
        reasons + ["no_entry_known_regime_filter_passed_discovery_selection"],
    )


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(plain_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(plain_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_string(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "filter_count": report["filter_count"],
        "selected_discovery_filter_count": report["selected_discovery_filter_count"],
        "weak_survivor_count": report["weak_survivor_count"],
        "strict_survivor_count": report["strict_survivor_count"],
        "top_selected_filters": report["top_selected_filters"],
        "top_weak_survivors": report["top_weak_survivors"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
