"""Fast106 long-horizon low-cost payoff scout.

Fast105 closed the short prospective veto line. Fast106 tests a different
product idea: fewer decisions, larger targets, and longer horizons using the
existing normalized BTCUSDT aggTrades archive. This is a scout only; the path
payoff label is exact for each tested target because rows are rebuilt with the
target as the path-label expected threshold.

No orders, no ledger writes, no Testnet/LIVE approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.event_replay import OrderSide  # noqa: E402
from app.services.research.research_dataset import (  # noqa: E402
    ResearchDatasetConfig,
    build_research_rows,
    load_trade_interval,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_TRADE_ROOT = DEFAULT_DATA_ROOT / "normalized" / "aggTrades"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast106-long-horizon-lowcost-scout"
DEFAULT_HORIZONS_SECONDS = (900, 1800)
DEFAULT_TARGET_BPS_VALUES = (8.0, 12.0, 16.0, 24.0)
DEFAULT_TOP_FRACTIONS = (0.005, 0.01, 0.02, 0.05)
DEFAULT_DECISION_STRIDE_SECONDS = 300
DEFAULT_EXPLICIT_COST_BPS = 2.0
DEFAULT_INITIAL_STOP_BPS = 20.0
FEATURE_WINDOWS_SECONDS = (1, 5, 30, 60)
FEATURE_COLUMNS = (
    "trade_count_1s",
    "quote_volume_1s",
    "flow_imbalance_1s",
    "return_1s_bps",
    "realized_vol_1s_bps",
    "mean_interarrival_ms_1s",
    "trade_count_5s",
    "quote_volume_5s",
    "flow_imbalance_5s",
    "return_5s_bps",
    "realized_vol_5s_bps",
    "mean_interarrival_ms_5s",
    "trade_count_30s",
    "quote_volume_30s",
    "flow_imbalance_30s",
    "return_30s_bps",
    "realized_vol_30s_bps",
    "mean_interarrival_ms_30s",
    "trade_count_60s",
    "quote_volume_60s",
    "flow_imbalance_60s",
    "return_60s_bps",
    "realized_vol_60s_bps",
    "mean_interarrival_ms_60s",
    "hour_sin",
    "hour_cos",
    "side_sign",
    "directed_flow_imbalance_1s",
    "directed_return_1s_bps",
    "directed_flow_imbalance_5s",
    "directed_return_5s_bps",
    "directed_flow_imbalance_30s",
    "directed_return_30s_bps",
    "directed_flow_imbalance_60s",
    "directed_return_60s_bps",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--trade-root", type=Path, default=DEFAULT_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--horizons-seconds", type=parse_int_tuple, default=DEFAULT_HORIZONS_SECONDS)
    parser.add_argument("--target-bps-values", type=parse_float_tuple, default=DEFAULT_TARGET_BPS_VALUES)
    parser.add_argument("--top-fractions", type=parse_float_tuple, default=DEFAULT_TOP_FRACTIONS)
    parser.add_argument("--decision-stride-seconds", type=int, default=DEFAULT_DECISION_STRIDE_SECONDS)
    parser.add_argument("--explicit-cost-bps", type=float, default=DEFAULT_EXPLICIT_COST_BPS)
    parser.add_argument("--initial-stop-bps", type=float, default=DEFAULT_INITIAL_STOP_BPS)
    parser.add_argument(
        "--historical-root-limit",
        type=int,
        default=None,
        help="Use only the N most recent historical low-cost blocks for a bounded scout.",
    )
    parser.add_argument(
        "--max-days-per-root",
        type=int,
        default=None,
        help="Use only the N most recent dates inside each historical/extension block.",
    )
    parser.add_argument("--progress", action="store_true")
    arguments = parser.parse_args()
    report = run_fast106_long_horizon_lowcost_scout(
        data_root=arguments.data_root,
        trade_root=arguments.trade_root,
        output_root=arguments.output_root,
        horizons_seconds=arguments.horizons_seconds,
        target_bps_values=arguments.target_bps_values,
        top_fractions=arguments.top_fractions,
        decision_stride_seconds=arguments.decision_stride_seconds,
        explicit_cost_bps=arguments.explicit_cost_bps,
        initial_stop_bps=arguments.initial_stop_bps,
        historical_root_limit=arguments.historical_root_limit,
        max_days_per_root=arguments.max_days_per_root,
        progress=arguments.progress,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast106_long_horizon_lowcost_scout(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    trade_root: Path = DEFAULT_TRADE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    horizons_seconds: tuple[int, ...] = DEFAULT_HORIZONS_SECONDS,
    target_bps_values: tuple[float, ...] = DEFAULT_TARGET_BPS_VALUES,
    top_fractions: tuple[float, ...] = DEFAULT_TOP_FRACTIONS,
    decision_stride_seconds: int = DEFAULT_DECISION_STRIDE_SECONDS,
    explicit_cost_bps: float = DEFAULT_EXPLICIT_COST_BPS,
    initial_stop_bps: float = DEFAULT_INITIAL_STOP_BPS,
    historical_root_limit: int | None = None,
    max_days_per_root: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    validate_parameters(
        horizons_seconds=horizons_seconds,
        target_bps_values=target_bps_values,
        top_fractions=top_fractions,
        decision_stride_seconds=decision_stride_seconds,
        explicit_cost_bps=explicit_cost_bps,
        initial_stop_bps=initial_stop_bps,
        historical_root_limit=historical_root_limit,
        max_days_per_root=max_days_per_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    historical_roots, extension_root = discover_lowcost_roots(data_root)
    if historical_root_limit is not None:
        historical_roots = historical_roots[-historical_root_limit:]
    rows_by_spec = build_rows_by_spec(
        historical_roots=historical_roots,
        extension_root=extension_root,
        trade_root=trade_root,
        horizons_seconds=horizons_seconds,
        target_bps_values=target_bps_values,
        decision_stride_seconds=decision_stride_seconds,
        explicit_cost_bps=explicit_cost_bps,
        initial_stop_bps=initial_stop_bps,
        max_days_per_root=max_days_per_root,
        progress=progress,
    )
    screen_rows = run_screen(
        rows_by_spec=rows_by_spec,
        historical_roots=historical_roots,
        extension_root=extension_root,
        top_fractions=top_fractions,
    )
    screen_survivors = [
        row
        for row in screen_rows
        if row["extension"]["trade_count"] >= 3
        and row["extension"]["net_total_bps"] > 0
        and row["historical"]["net_total_bps"] > 0
    ]
    validations = [
        validate_candidate(
            candidate,
            rows_by_spec=rows_by_spec,
            historical_roots=historical_roots,
            extension_root=extension_root,
        )
        for candidate in screen_survivors
    ]
    status, status_reasons = fast106_status(
        screen_survivors=screen_survivors,
        validations=validations,
    )
    screen_path = output_root / "screen_results.json"
    screen_path.write_text(json.dumps(screen_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validations_path = output_root / "validations.json"
    validations_path.write_text(json.dumps(validations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row_manifest_path = output_root / "row_manifest.json"
    row_manifest_path.write_text(
        json.dumps(row_manifest(rows_by_spec), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
            "fast106_is_long_horizon_lowcost_scout",
            "normalized_aggtrades_only_no_book_or_fill_model",
            "extension_and_walkforward_must_pass_before_any_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast106_long_horizon_lowcost_scout",
        "status": status,
        "status_reasons": status_reasons,
        "data_root": str(data_root),
        "trade_root": str(trade_root),
        "historical_roots": [str(root) for root in historical_roots],
        "extension_root": str(extension_root),
        "horizons_seconds": list(horizons_seconds),
        "target_bps_values": list(target_bps_values),
        "top_fractions": list(top_fractions),
        "decision_stride_seconds": decision_stride_seconds,
        "explicit_cost_bps": explicit_cost_bps,
        "initial_stop_bps": initial_stop_bps,
        "historical_root_limit": historical_root_limit,
        "max_days_per_root": max_days_per_root,
        "progress_enabled": progress,
        "feature_windows_seconds": list(FEATURE_WINDOWS_SECONDS),
        "feature_count": len(FEATURE_COLUMNS),
        "screen_result_count": len(screen_rows),
        "screen_survivor_count": len(screen_survivors),
        "screen_survivors": screen_survivors,
        "validation_count": len(validations),
        "validations": validations,
        "screen_results_path": str(screen_path),
        "screen_results_sha256": sha256_file(screen_path),
        "validations_path": str(validations_path),
        "validations_sha256": sha256_file(validations_path),
        "row_manifest_path": str(row_manifest_path),
        "row_manifest_sha256": sha256_file(row_manifest_path),
    }
    return write_report(output_root, report)


def validate_parameters(
    *,
    horizons_seconds: tuple[int, ...],
    target_bps_values: tuple[float, ...],
    top_fractions: tuple[float, ...],
    decision_stride_seconds: int,
    explicit_cost_bps: float,
    initial_stop_bps: float,
    historical_root_limit: int | None,
    max_days_per_root: int | None,
) -> None:
    if not horizons_seconds or any(horizon <= 0 for horizon in horizons_seconds):
        raise ValueError("horizons must be positive")
    if not target_bps_values or any(target <= 0 for target in target_bps_values):
        raise ValueError("targets must be positive")
    if not top_fractions or any(not 0 < top_fraction < 1 for top_fraction in top_fractions):
        raise ValueError("top_fractions must be between zero and one")
    if decision_stride_seconds <= 0:
        raise ValueError("decision_stride_seconds must be positive")
    if explicit_cost_bps < 0:
        raise ValueError("explicit_cost_bps cannot be negative")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if historical_root_limit is not None and historical_root_limit <= 0:
        raise ValueError("historical_root_limit must be positive")
    if max_days_per_root is not None and max_days_per_root <= 0:
        raise ValueError("max_days_per_root must be positive")


def parse_int_tuple(value: str) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parts


def parse_float_tuple(value: str) -> tuple[float, ...]:
    parts = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one float")
    return parts


def discover_lowcost_roots(data_root: Path) -> tuple[tuple[Path, ...], Path]:
    roots = sorted(data_root.glob("research-lowcost-fast*"))
    historical = tuple(root for root in roots if "fast41" not in root.name and "fast56" not in root.name)
    extension = data_root / "research-lowcost-fast56-holdout-new"
    if not historical:
        raise FileNotFoundError("no historical research-lowcost roots found")
    if not extension.exists():
        raise FileNotFoundError(f"extension root missing: {extension}")
    return historical, extension


def build_rows_by_spec(
    *,
    historical_roots: tuple[Path, ...],
    extension_root: Path,
    trade_root: Path,
    horizons_seconds: tuple[int, ...],
    target_bps_values: tuple[float, ...],
    decision_stride_seconds: int,
    explicit_cost_bps: float,
    initial_stop_bps: float,
    max_days_per_root: int | None,
    progress: bool,
) -> dict[tuple[int, float, str], pd.DataFrame]:
    roots = (*historical_roots, extension_root)
    rows_by_spec = {}
    for horizon_seconds in horizons_seconds:
        for target_bps in target_bps_values:
            for root in roots:
                if progress:
                    print(  # noqa: T201
                        "building rows "
                        f"horizon={horizon_seconds}s target={target_bps}bps block={root.name}",
                        file=sys.stderr,
                        flush=True,
                    )
                rows = build_block_rows(
                    root=root,
                    trade_root=trade_root,
                    horizon_seconds=horizon_seconds,
                    target_bps=target_bps,
                    decision_stride_seconds=decision_stride_seconds,
                    explicit_cost_bps=explicit_cost_bps,
                    initial_stop_bps=initial_stop_bps,
                    max_days_per_root=max_days_per_root,
                )
                rows_by_spec[(horizon_seconds, target_bps, root.name)] = rows
    return rows_by_spec


def build_block_rows(
    *,
    root: Path,
    trade_root: Path,
    horizon_seconds: int,
    target_bps: float,
    decision_stride_seconds: int,
    explicit_cost_bps: float,
    initial_stop_bps: float,
    max_days_per_root: int | None,
) -> pd.DataFrame:
    parts = []
    config = ResearchDatasetConfig(
        decision_stride_seconds=decision_stride_seconds,
        horizons_seconds=(horizon_seconds,),
        expected_cost_bps=target_bps,
        stress_cost_bps=target_bps,
        initial_stop_bps=initial_stop_bps,
        feature_windows_seconds=FEATURE_WINDOWS_SECONDS,
    )
    date_paths = sorted(root.glob("date=*"))
    if max_days_per_root is not None:
        date_paths = date_paths[-max_days_per_root:]
    for date_path in date_paths:
        utc_date = date.fromisoformat(date_path.name.removeprefix("date="))
        rows = build_date_rows(
            trade_root=trade_root,
            utc_date=utc_date,
            config=config,
            target_bps=target_bps,
            explicit_cost_bps=explicit_cost_bps,
            initial_stop_bps=initial_stop_bps,
        )
        if not rows.empty:
            parts.append(rows)
    if not parts:
        return pd.DataFrame()
    block = pd.concat(parts, ignore_index=True)
    block["block"] = root.name
    return block


def build_date_rows(
    *,
    trade_root: Path,
    utc_date: date,
    config: ResearchDatasetConfig,
    target_bps: float,
    explicit_cost_bps: float,
    initial_stop_bps: float,
) -> pd.DataFrame:
    day_start = datetime.combine(utc_date, datetime.min.time(), tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    history_start = day_start - timedelta(seconds=max(config.feature_windows_seconds))
    label_end = day_end + timedelta(seconds=max(config.horizons_seconds))
    try:
        trades = load_trade_interval(
            trade_root,
            history_start,
            label_end,
            expected_product="usdm_perpetual",
        )
    except (FileNotFoundError, ValueError):
        return pd.DataFrame()
    stride_ms = config.decision_stride_seconds * 1000
    decisions = np.arange(
        int(day_start.timestamp() * 1000),
        int(day_end.timestamp() * 1000),
        stride_ms,
        dtype=np.int64,
    )
    event_times = trades["event_time_ms"].to_numpy(dtype=np.int64)
    complete = (
        decisions - max(config.feature_windows_seconds) * 1000 >= event_times[0]
    ) & (decisions + max(config.horizons_seconds) * 1000 <= event_times[-1])
    decisions = decisions[complete]
    if not len(decisions):
        return pd.DataFrame()
    rows = build_research_rows(trades, decisions, config)
    rows["net_bps"] = exact_target_net_bps(
        rows,
        target_bps=target_bps,
        explicit_cost_bps=explicit_cost_bps,
        initial_stop_bps=initial_stop_bps,
    )
    return rows


def exact_target_net_bps(
    rows: pd.DataFrame,
    *,
    target_bps: float,
    explicit_cost_bps: float,
    initial_stop_bps: float,
) -> np.ndarray:
    gross = np.where(
        rows["first_touch"].eq("STOP"),
        -initial_stop_bps,
        np.where(rows["paid_expected_before_stop"], target_bps, rows["terminal_bps"]),
    )
    return gross - explicit_cost_bps


def run_screen(
    *,
    rows_by_spec: dict[tuple[int, float, str], pd.DataFrame],
    historical_roots: tuple[Path, ...],
    extension_root: Path,
    top_fractions: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows = []
    specs = sorted({(horizon, target) for horizon, target, _block in rows_by_spec})
    for horizon_seconds, target_bps in specs:
        historical = pd.concat(
            [
                rows_by_spec[(horizon_seconds, target_bps, root.name)]
                for root in historical_roots
            ],
            ignore_index=True,
        )
        extension = rows_by_spec[(horizon_seconds, target_bps, extension_root.name)]
        model = train_payoff_model(historical)
        historical_scored = score_rows(model, historical)
        extension_scored = score_rows(model, extension)
        for top_fraction in top_fractions:
            threshold = score_threshold(historical_scored, top_fraction=top_fraction)
            historical_selected = select_top_one_position(
                historical_scored,
                score_threshold_value=threshold,
                horizon_seconds=horizon_seconds,
            )
            extension_selected = select_top_one_position(
                extension_scored,
                score_threshold_value=threshold,
                horizon_seconds=horizon_seconds,
            )
            rows.append(
                {
                    "horizon_seconds": horizon_seconds,
                    "target_bps": target_bps,
                    "top_fraction": top_fraction,
                    "score_threshold": threshold,
                    "historical": summarize_net(historical_selected),
                    "extension": summarize_net(extension_selected),
                }
            )
    return rows


def validate_candidate(
    candidate: dict[str, Any],
    *,
    rows_by_spec: dict[tuple[int, float, str], pd.DataFrame],
    historical_roots: tuple[Path, ...],
    extension_root: Path,
) -> dict[str, Any]:
    horizon_seconds = int(candidate["horizon_seconds"])
    target_bps = float(candidate["target_bps"])
    top_fraction = float(candidate["top_fraction"])
    by_block = {
        root.name: rows_by_spec[(horizon_seconds, target_bps, root.name)]
        for root in historical_roots
    }
    historical = pd.concat(by_block.values(), ignore_index=True)
    blocks = []
    selected_parts = []
    for block_name, holdout in by_block.items():
        training = historical[historical["block"].ne(block_name)]
        model = train_payoff_model(training)
        training_scored = score_rows(model, training)
        threshold = score_threshold(training_scored, top_fraction=top_fraction)
        holdout_scored = score_rows(model, holdout)
        selected = select_top_one_position(
            holdout_scored,
            score_threshold_value=threshold,
            horizon_seconds=horizon_seconds,
        )
        selected_parts.append(selected)
        blocks.append({"block": block_name, **summarize_net(selected)})
    all_selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    sufficient = [block for block in blocks if block["trade_count"] >= 3]
    historical_summary = summarize_net(all_selected) | {
        "sufficient_block_count": len(sufficient),
        "positive_sufficient_block_count": sum(block["net_total_bps"] > 0 for block in sufficient),
        "negative_sufficient_blocks": [
            block["block"] for block in sufficient if block["net_total_bps"] <= 0
        ],
        "min_sufficient_block_net_total_bps": min(
            (block["net_total_bps"] for block in sufficient),
            default=0.0,
        ),
    }
    extension = rows_by_spec[(horizon_seconds, target_bps, extension_root.name)]
    model = train_payoff_model(historical)
    historical_scored = score_rows(model, historical)
    threshold = score_threshold(historical_scored, top_fraction=top_fraction)
    extension_selected = select_top_one_position(
        score_rows(model, extension),
        score_threshold_value=threshold,
        horizon_seconds=horizon_seconds,
    )
    return {
        "horizon_seconds": horizon_seconds,
        "target_bps": target_bps,
        "top_fraction": top_fraction,
        "historical": historical_summary,
        "blocks": blocks,
        "extension": summarize_net(extension_selected),
    }


def train_payoff_model(frame: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    model.fit(frame[list(FEATURE_COLUMNS)], frame["net_bps"])
    return model


def score_rows(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = model.predict(scored[list(FEATURE_COLUMNS)]) if not scored.empty else []
    return scored


def score_threshold(frame: pd.DataFrame, *, top_fraction: float) -> float:
    return float(pd.to_numeric(frame["score"], errors="coerce").quantile(1.0 - top_fraction))


def select_top_one_position(
    frame: pd.DataFrame,
    *,
    score_threshold_value: float,
    horizon_seconds: int,
) -> pd.DataFrame:
    filtered = frame[frame["score"] >= score_threshold_value].copy()
    if filtered.empty:
        return filtered
    ordered = filtered.sort_values(
        ["decision_time_ms", "score"],
        ascending=[True, False],
        kind="stable",
    )
    keep = []
    position_exit_ms = -1
    for index, row in ordered.iterrows():
        decision_time_ms = int(row["decision_time_ms"])
        if decision_time_ms <= position_exit_ms:
            continue
        keep.append(index)
        position_exit_ms = decision_time_ms + horizon_seconds * 1000
    return ordered.loc[keep]


def summarize_net(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trade_count": 0,
            "distinct_days": 0,
            "net_total_bps": 0.0,
            "net_mean_bps": 0.0,
            "win_rate": 0.0,
        }
    dates = pd.to_datetime(frame["decision_time_ms"], unit="ms", utc=True).dt.date
    return {
        "trade_count": int(len(frame)),
        "distinct_days": int(dates.nunique()),
        "net_total_bps": float(frame["net_bps"].sum()),
        "net_mean_bps": float(frame["net_bps"].mean()),
        "win_rate": float((frame["net_bps"] > 0).mean()),
    }


def row_manifest(
    rows_by_spec: dict[tuple[int, float, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    return [
        {
            "horizon_seconds": horizon_seconds,
            "target_bps": target_bps,
            "block": block,
            "row_count": int(len(rows)),
            "day_count": int(
                pd.to_datetime(rows["decision_time_ms"], unit="ms", utc=True).dt.date.nunique()
            )
            if not rows.empty
            else 0,
        }
        for (horizon_seconds, target_bps, block), rows in sorted(rows_by_spec.items())
    ]


def fast106_status(
    *,
    screen_survivors: list[dict[str, Any]],
    validations: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if not screen_survivors:
        return "FAST106_LONG_HORIZON_FAILED_SCREEN", reasons + ["no_screen_survivor"]
    passing = [
        item
        for item in validations
        if item["historical"]["negative_sufficient_blocks"] == []
        and item["historical"]["net_total_bps"] > 0
        and item["extension"]["trade_count"] >= 3
        and item["extension"]["net_total_bps"] > 0
    ]
    if not passing:
        return "FAST106_LONG_HORIZON_FAILED_WALKFORWARD_OR_EXTENSION", reasons + [
            "no_screen_survivor_passed_walkforward_and_extension"
        ]
    return "FAST106_LONG_HORIZON_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION", reasons + [
        "requires_fresh_preregistration_before_prospective_use"
    ]


def write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


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
        "screen_result_count": report["screen_result_count"],
        "screen_survivor_count": report["screen_survivor_count"],
        "screen_survivors": report["screen_survivors"],
        "validations": report["validations"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
