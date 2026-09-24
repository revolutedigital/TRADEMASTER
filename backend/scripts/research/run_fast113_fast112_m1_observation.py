"""Fast113 M1 materializer for the preregistered Fast112 USDJPY observation.

Fast112 froze a single research-only USDJPY breakout candidate, but its first
runner consumed the historical Fast111 trade table. Fast113 rebuilds the same
candidate directly from local M1 bid/ask data, applies the exact Fast112
prospective filter, and keeps the result strictly offline. It never sends
orders, commits ledger state, or approves a strategy for Demo, Testnet, or LIVE.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from app.fx.instruments import ConversionRates, Instrument  # noqa: E402
from app.fx.sim import core  # noqa: E402
from app.fx.sim.costs import FUSION_ZERO, STRESS, finalize_trades, prepare_run  # noqa: E402
from scripts.research import fx_fast2_lab as fast2  # noqa: E402
from scripts.research import fx_fast_lab as fast_lab  # noqa: E402
from scripts.research.fx_fast_manifest import load_m1  # noqa: E402
from scripts.research.run_fast112_usdjpy_breakout_observation import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_PREREGISTRATION_PATH,
    DEFAULT_REGISTRY_PATH,
    OBSERVATION_CONFIG,
    OBSERVATION_FILTER,
    OBSERVATION_PAIR,
    PREREGISTRATION_COMMIT,
    PREREGISTRATION_TIMESTAMP,
    STOP_PIPS_WIDE_THRESHOLD,
    console_summary as fast112_console_summary,
    evaluate_gates,
    fast112_status,
    observation_record,
    select_eligible_observations,
    sha256_file,
    summarize_observations,
    validate_parameters,
    validate_preregistration,
    write_jsonl,
    write_report,
)

DEFAULT_RAW_M1_DIR = REPO_ROOT / "backend" / "data" / "raw" / "fx_m1_hd"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast113" / "fast112-m1-observation"
)
LONDON = ZoneInfo("Europe/London")
TRADE_ROW_COLUMNS = [
    "stage",
    "config",
    "pair",
    "scenario",
    "entry_time",
    "entry_epoch",
    "fx_day",
    "london_weekday",
    "london_weekday_name",
    "london_entry_hour",
    "london_month",
    "year",
    "side",
    "side_label",
    "stop_pips",
    "r_multiple",
]
SAFETY_FLAGS = {
    "research_only": True,
    "order_submission_allowed": False,
    "execution_authorization": "none",
    "orders_sent": False,
    "committed": False,
    "dry_run": True,
    "approved_strategy": False,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-m1-dir", type=Path, default=DEFAULT_RAW_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--start", help="Optional UTC inclusive start timestamp for local diagnostics")
    parser.add_argument("--end", help="Optional UTC inclusive end timestamp for local diagnostics")
    parser.add_argument("--progress", action="store_true")
    arguments = parser.parse_args()
    report = run_fast113_fast112_m1_observation(
        raw_m1_dir=arguments.raw_m1_dir,
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
        start=arguments.start,
        end=arguments.end,
        progress=arguments.progress,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast113_fast112_m1_observation(
    *,
    raw_m1_dir: Path = DEFAULT_RAW_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    alpha: float = DEFAULT_ALPHA,
    start: str | None = None,
    end: str | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    validate_parameters(alpha=alpha)
    preregistration = validate_preregistration(
        preregistration_path=preregistration_path,
        registry_path=registry_path,
    )
    raw_m1_dir = absolute_path(raw_m1_dir)
    output_root = absolute_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_files = source_m1_files(raw_m1_dir, OBSERVATION_PAIR)
    raw_m1 = load_observation_m1(OBSERVATION_PAIR, raw_m1_dir, start=start, end=end)
    materialized_rows = materialize_fast112_trade_rows(raw_m1, progress=progress)
    materialized_path = output_root / "materialized_trade_rows.parquet"
    materialized_rows.to_parquet(materialized_path, index=False)

    observations = select_eligible_observations(
        materialized_rows,
        prospective_start=PREREGISTRATION_TIMESTAMP,
    )
    observation_records = [observation_record(row) for _, row in observations.iterrows()]
    observations_path = output_root / "observations.jsonl"
    write_jsonl(observations_path, observation_records)

    summary = summarize_observations(observations, alpha=alpha)
    gates = evaluate_gates(summary)
    status, status_reasons = fast112_status(summary=summary, gates=gates)
    report = build_report(
        preregistration=preregistration,
        raw_m1_dir=raw_m1_dir,
        source_files=source_files,
        raw_m1=raw_m1,
        materialized_rows=materialized_rows,
        materialized_path=materialized_path,
        observations_path=observations_path,
        observation_records=observation_records,
        summary=summary,
        gates=gates,
        status=status,
        status_reasons=status_reasons,
        preregistration_path=preregistration_path,
        registry_path=registry_path,
        start=start,
        end=end,
    )
    return write_report(output_root, report)


def absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def source_m1_files(raw_m1_dir: Path, pair: str) -> list[dict[str, Any]]:
    files = sorted(raw_m1_dir.glob(f"{pair}_M1_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no M1 file for {pair} in {raw_m1_dir}")
    return [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]


def load_observation_m1(
    pair: str,
    raw_m1_dir: Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    frame = load_m1(pair, raw_m1_dir)
    if frame.index.tz is None:
        raise ValueError("M1 frame must have a timezone-aware UTC index")
    frame = frame.tz_convert("UTC") if hasattr(frame, "tz_convert") else frame
    start_timestamp = parse_optional_timestamp(start, name="start")
    end_timestamp = parse_optional_timestamp(end, name="end")
    if start_timestamp is not None:
        frame = frame.loc[frame.index >= start_timestamp]
    if end_timestamp is not None:
        frame = frame.loc[frame.index <= end_timestamp]
    if frame.empty:
        raise ValueError("M1 frame is empty after applying optional start/end filters")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    return frame


def parse_optional_timestamp(value: str | None, *, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include a timezone, preferably Z/UTC")
    return timestamp.tz_convert("UTC")


def materialize_fast112_trade_rows(raw_m1: pd.DataFrame, *, progress: bool = False) -> pd.DataFrame:
    config = fast2.configurations()[OBSERVATION_CONFIG]
    instrument = Instrument.from_symbol(OBSERVATION_PAIR)
    minute_matrix = fx.bars_to_matrix(raw_m1)
    if len(minute_matrix) < 2:
        return empty_trade_rows()

    price = median_mid_price(minute_matrix)
    rates = ConversionRates({OBSERVATION_PAIR: price})
    bars = aggregate_minutes(minute_matrix, config.seconds)
    params = config.params(OBSERVATION_PAIR)
    rows: list[dict[str, Any]] = []
    for scenario in (FUSION_ZERO, STRESS):
        if progress:
            print(f"simulating {OBSERVATION_PAIR} {OBSERVATION_CONFIG} {scenario.name}")  # noqa: T201
        scenario_bars, slippage = prepare_run(bars, instrument, scenario)
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
                OBSERVATION_PAIR,
                price,
                rates,
                scenario.commission_base_per_lot_per_side,
            ),
            calendar=fast_lab.CALENDAR,
        )
        rows.extend(
            trade_rows_from_simulated_trades(
                config_key=OBSERVATION_CONFIG,
                pair=OBSERVATION_PAIR,
                scenario=scenario.name,
                trades=trades,
            )
        )
    if not rows:
        return empty_trade_rows()
    frame = pd.DataFrame(rows, columns=TRADE_ROW_COLUMNS)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    return frame.sort_values(
        ["entry_time", "scenario"],
        ignore_index=True,
    )


def median_mid_price(matrix: np.ndarray) -> float:
    return float(np.median(0.5 * (matrix[:, fx.BID_CLOSE] + matrix[:, fx.ASK_CLOSE])))


def trade_rows_from_simulated_trades(
    *,
    config_key: str,
    pair: str,
    scenario: str,
    trades: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for trade in trades.to_dict("records"):
        entry = pd.Timestamp(trade["entry_time"])
        london_entry = entry.tz_convert(LONDON)
        rows.append(
            {
                "stage": "m1_materialized",
                "config": config_key,
                "pair": pair,
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


def empty_trade_rows() -> pd.DataFrame:
    frame = pd.DataFrame(columns=TRADE_ROW_COLUMNS)
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    return frame


def build_report(
    *,
    preregistration: dict[str, str],
    raw_m1_dir: Path,
    source_files: list[dict[str, Any]],
    raw_m1: pd.DataFrame,
    materialized_rows: pd.DataFrame,
    materialized_path: Path,
    observations_path: Path,
    observation_records: list[dict[str, Any]],
    summary: dict[str, Any],
    gates: dict[str, Any],
    status: str,
    status_reasons: list[str],
    preregistration_path: Path,
    registry_path: Path,
    start: str | None,
    end: str | None,
) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast113_materializes_fast112_observation_only",
            "fast112_rule_remains_preregistered_and_unapproved",
            "no_order_execution",
            "no_ledger_commit",
            "testnet_requires_new_report_and_explicit_authorization",
        ],
        "experiment_stage": "fast113_fast112_m1_observation",
        "materialization_source": "raw_m1",
        "status": status,
        "status_reasons": status_reasons,
        "observation_pair": OBSERVATION_PAIR,
        "observation_config": OBSERVATION_CONFIG,
        "observation_filter": OBSERVATION_FILTER,
        "stop_pips_wide_threshold": STOP_PIPS_WIDE_THRESHOLD,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "raw_m1_dir": str(raw_m1_dir),
        "raw_m1_files": source_files,
        "raw_m1_row_count": int(len(raw_m1)),
        "raw_m1_start": raw_m1.index[0].isoformat(),
        "raw_m1_end": raw_m1.index[-1].isoformat(),
        "optional_start_filter": start,
        "optional_end_filter": end,
        "materialized_trade_rows_path": str(materialized_path),
        "materialized_trade_rows_sha256": sha256_file(materialized_path),
        "materialized_trade_row_count": int(len(materialized_rows)),
        "eligible_observation_count": len(observation_records),
        "observations_path": str(observations_path),
        "observations_sha256": sha256_file(observations_path),
        "summary": summary,
        "gates": gates,
    }


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = fast112_console_summary(report)
    summary.update(
        {
            "raw_m1_start": report["raw_m1_start"],
            "raw_m1_end": report["raw_m1_end"],
            "raw_m1_row_count": report["raw_m1_row_count"],
            "materialized_trade_row_count": report["materialized_trade_row_count"],
        }
    )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
