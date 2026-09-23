"""Replay a frozen book rule with executable top-of-book prices.

The previous book-rule confirmations use trade-last path labels. This diagnostic
keeps the same frozen base+gate selection, but prices hypothetical marketable
taker entry/exit against bid/ask snapshots from the WAL. It is research-only and
never submits orders.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.event_replay import OrderSide  # noqa: E402
from app.services.research.research_dataset import (  # noqa: E402
    ResearchDatasetConfig,
    load_book_interval,
)
from scripts.research.run_book_regime_gate_scout import (  # noqa: E402
    apply_gate_threshold,
    fit_single_gate_threshold,
)
from scripts.research.run_book_rule_confirmation import (  # noqa: E402
    _load_rows,
    _parse_conditions,
    _parse_float_list,
    apply_frozen_rule,
    fit_rule_thresholds,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-development-wal-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "prospective-wal-fast16-18m",
    )
    parser.add_argument("--gate-development-wal-roots", required=True)
    parser.add_argument("--confirmation-wal-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-executable-replay",
    )
    parser.add_argument("--horizon-seconds", type=int, default=120)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    parser.add_argument("--base-condition-fraction", type=float, default=0.20)
    parser.add_argument(
        "--base-conditions-json",
        default='[{"feature":"directed_spot_perp_basis_bps","direction":"high"}]',
    )
    parser.add_argument("--gate-feature", required=True)
    parser.add_argument("--gate-direction", choices=("high", "low"), required=True)
    parser.add_argument("--gate-fraction", type=float, required=True)
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--minimum-executable-count", type=int, default=10)
    parser.add_argument("--cost-levels-bps", default="0,0.25,0.5,1,2")
    arguments = parser.parse_args()
    report = run_book_executable_replay(
        base_development_wal_root=arguments.base_development_wal_root,
        gate_development_wal_roots=_parse_paths(arguments.gate_development_wal_roots),
        confirmation_wal_root=arguments.confirmation_wal_root,
        output_root=arguments.output_root,
        horizon_seconds=arguments.horizon_seconds,
        threshold_bps=arguments.threshold_bps,
        base_condition_fraction=arguments.base_condition_fraction,
        base_conditions=_parse_conditions(arguments.base_conditions_json),
        gate_feature=arguments.gate_feature,
        gate_direction=arguments.gate_direction,
        gate_fraction=arguments.gate_fraction,
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        latency_ms=arguments.latency_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        minimum_executable_count=arguments.minimum_executable_count,
        cost_levels_bps=_parse_float_list(arguments.cost_levels_bps, allow_zero=True),
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_executable_replay(
    *,
    base_development_wal_root: Path,
    gate_development_wal_roots: tuple[Path, ...],
    confirmation_wal_root: Path,
    output_root: Path,
    horizon_seconds: int = 120,
    threshold_bps: float = 5.0,
    base_condition_fraction: float = 0.20,
    base_conditions: tuple[dict[str, str], ...] = (
        {"feature": "directed_spot_perp_basis_bps", "direction": "high"},
    ),
    gate_feature: str,
    gate_direction: str,
    gate_fraction: float,
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    latency_ms: int = 100,
    initial_stop_bps: float = 20.0,
    minimum_executable_count: int = 10,
    cost_levels_bps: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> dict[str, object]:
    _validate_parameters(
        gate_development_wal_roots=gate_development_wal_roots,
        horizon_seconds=horizon_seconds,
        threshold_bps=threshold_bps,
        base_condition_fraction=base_condition_fraction,
        base_conditions=base_conditions,
        gate_feature=gate_feature,
        gate_direction=gate_direction,
        gate_fraction=gate_fraction,
        max_book_staleness_ms=max_book_staleness_ms,
        latency_ms=latency_ms,
        initial_stop_bps=initial_stop_bps,
        minimum_executable_count=minimum_executable_count,
        cost_levels_bps=cost_levels_bps,
    )
    config = ResearchDatasetConfig(
        decision_stride_seconds=decision_stride_seconds,
        horizons_seconds=(horizon_seconds,),
        expected_cost_bps=threshold_bps,
        stress_cost_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
        feature_windows_seconds=(1, 5, 30),
        require_book_features=True,
        max_book_staleness_ms=max_book_staleness_ms,
    )
    base_report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "experiment_stage": "book_executable_replay",
        "approved_strategy": False,
        "approval_reasons": [
            "diagnostic_only",
            "top_of_book_taker_replay_not_exchange_execution",
            "no_queue_or_partial_fill_model",
            "no_order_execution",
        ],
        "base_development_wal_root": str(base_development_wal_root),
        "gate_development_wal_roots": [str(path) for path in gate_development_wal_roots],
        "confirmation_wal_root": str(confirmation_wal_root),
        "config": asdict(config),
        "base_rule": {
            "condition_fraction": base_condition_fraction,
            "conditions": list(base_conditions),
        },
        "gate_rule": {
            "feature": gate_feature,
            "direction": gate_direction,
            "fraction": gate_fraction,
        },
        "execution_replay": {
            "entry_exit_model": "marketable_taker_top_of_book",
            "latency_ms": latency_ms,
            "explicit_cost_levels_bps": list(cost_levels_bps),
            "bid_ask_spread_included_in_gross": True,
            "one_position_constraint_reported": True,
        },
        "minimum_executable_count": minimum_executable_count,
    }

    base_development = _load_rows(
        wal_root=base_development_wal_root,
        config=config,
        threshold_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
    )
    if base_development["status"] != "READY":
        return _write_report(
            output_root,
            {
                **base_report,
                "status": f"BASE_DEVELOPMENT_{base_development['status']}",
                "status_reasons": base_development["status_reasons"],
                "modeling_ready": False,
            },
        )
    base_thresholds = fit_rule_thresholds(
        base_development["rows"],
        conditions=base_conditions,
        condition_fraction=base_condition_fraction,
    )

    gate_frames: list[pd.DataFrame] = []
    gate_streams: dict[str, object] = {}
    for wal_root in gate_development_wal_roots:
        loaded = _load_rows(
            wal_root=wal_root,
            config=config,
            threshold_bps=threshold_bps,
            initial_stop_bps=initial_stop_bps,
        )
        key = str(wal_root)
        gate_streams[key] = {
            "status": loaded["status"],
            "status_reasons": loaded["status_reasons"],
        }
        if loaded["status"] != "READY":
            return _write_report(
                output_root,
                {
                    **base_report,
                    "base_thresholds": base_thresholds,
                    "gate_development_status": gate_streams,
                    "status": f"GATE_DEVELOPMENT_{loaded['status']}",
                    "status_reasons": loaded["status_reasons"],
                    "modeling_ready": False,
                },
            )
        gate_frames.append(loaded["rows"])
    gate_development_rows = pd.concat(gate_frames, ignore_index=True)
    gate_development_base = apply_frozen_rule(gate_development_rows, base_thresholds)
    if gate_feature not in gate_development_base.columns:
        return _write_report(
            output_root,
            {
                **base_report,
                "base_thresholds": base_thresholds,
                "gate_development_status": gate_streams,
                "status": "GATE_FEATURE_MISSING",
                "status_reasons": [f"gate_feature_missing:{gate_feature}"],
                "modeling_ready": False,
            },
        )
    frozen_gate = fit_single_gate_threshold(
        gate_development_base,
        feature=gate_feature,
        direction=gate_direction,
        fraction=gate_fraction,
    )

    confirmation = _load_rows(
        wal_root=confirmation_wal_root,
        config=config,
        threshold_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
    )
    report = {
        **base_report,
        "base_thresholds": base_thresholds,
        "gate_development_status": gate_streams,
        "frozen_gate": frozen_gate,
    }
    if confirmation["status"] != "READY":
        return _write_report(
            output_root,
            {
                **report,
                "status": f"CONFIRMATION_{confirmation['status']}",
                "status_reasons": confirmation["status_reasons"],
                "modeling_ready": False,
            },
        )

    confirmation_base = apply_frozen_rule(confirmation["rows"], base_thresholds)
    confirmation_selected = apply_gate_threshold(confirmation_base, frozen_gate)
    if confirmation_selected.empty:
        return _write_report(
            output_root,
            {
                **report,
                "status": "EXECUTABLE_REPLAY_NO_SELECTED_SIGNALS",
                "status_reasons": ["fixed_base_gate_selected_no_confirmation_rows"],
                "modeling_ready": False,
                "selected_signal_count": 0,
            },
        )
    book_events = _load_confirmation_book_events(
        confirmation_selected,
        confirmation_wal_root=confirmation_wal_root,
        horizon_seconds=horizon_seconds,
        latency_ms=latency_ms,
    )
    replay_rows = replay_selected_rows(
        confirmation_selected,
        book_events=book_events,
        horizon_seconds=horizon_seconds,
        target_bps=threshold_bps,
        initial_stop_bps=initial_stop_bps,
        latency_ms=latency_ms,
        max_book_staleness_ms=max_book_staleness_ms,
    )
    trades = pd.DataFrame([row for row in replay_rows if row["would_enter"]])
    portfolio = _apply_one_position_constraint(trades)
    independent_summary = summarize_executable_trades(
        trades,
        selected_signal_count=len(confirmation_selected),
        cost_levels_bps=cost_levels_bps,
    )
    portfolio_summary = summarize_executable_trades(
        portfolio,
        selected_signal_count=len(confirmation_selected),
        cost_levels_bps=cost_levels_bps,
    )
    status_reasons = []
    if independent_summary["executable_count"] < minimum_executable_count:
        status_reasons.append(
            "independent_executable_count_below_minimum:"
            f"{independent_summary['executable_count']}<{minimum_executable_count}"
        )
    if portfolio_summary["executable_count"] < minimum_executable_count:
        status_reasons.append(
            "one_position_executable_count_below_minimum:"
            f"{portfolio_summary['executable_count']}<{minimum_executable_count}"
        )
    if not _positive_at_cost(independent_summary, 1.0):
        status_reasons.append("independent_net_at_1bps_non_positive")
    if not _positive_at_cost(portfolio_summary, 1.0):
        status_reasons.append("one_position_net_at_1bps_non_positive")
    modeling_ready = not status_reasons
    if modeling_ready:
        status_reasons.append("shadow_replay_required_before_strategy_approval")

    output_root.mkdir(parents=True, exist_ok=True)
    trades_path = output_root / "executable_trades.parquet"
    trades.to_parquet(trades_path, index=False, compression="zstd")
    portfolio_path = output_root / "one_position_trades.parquet"
    portfolio.to_parquet(portfolio_path, index=False, compression="zstd")
    return _write_report(
        output_root,
        {
            **report,
            "status": (
                "EXECUTABLE_REPLAY_READY_FOR_SHADOW_REPLAY"
                if modeling_ready
                else "EXECUTABLE_REPLAY_FAILED"
            ),
            "status_reasons": status_reasons,
            "modeling_ready": modeling_ready,
            "selected_signal_count": int(len(confirmation_selected)),
            "replay_row_count": int(len(replay_rows)),
            "independent_summary": independent_summary,
            "one_position_summary": portfolio_summary,
            "trades_path": str(trades_path),
            "one_position_trades_path": str(portfolio_path),
        },
    )


def replay_selected_rows(
    selected_rows: pd.DataFrame,
    *,
    book_events: pd.DataFrame,
    horizon_seconds: int,
    target_bps: float,
    initial_stop_bps: float,
    latency_ms: int,
    max_book_staleness_ms: int,
) -> list[dict[str, object]]:
    book = _book_arrays(book_events)
    results: list[dict[str, object]] = []
    ordered = selected_rows.sort_values(["decision_time_ms", "side"], kind="stable")
    for row in ordered.itertuples(index=False):
        results.append(
            simulate_executable_trade(
                book=book,
                decision_time_ms=int(row.decision_time_ms),
                side=OrderSide(str(row.side)),
                horizon_seconds=horizon_seconds,
                target_bps=target_bps,
                initial_stop_bps=initial_stop_bps,
                latency_ms=latency_ms,
                max_book_staleness_ms=max_book_staleness_ms,
            )
        )
    return results


def simulate_executable_trade(
    *,
    book: dict[str, np.ndarray],
    decision_time_ms: int,
    side: OrderSide,
    horizon_seconds: int,
    target_bps: float,
    initial_stop_bps: float,
    latency_ms: int,
    max_book_staleness_ms: int,
) -> dict[str, object]:
    times = book["event_time_ms"]
    execution_time_ms = decision_time_ms + latency_ms
    entry_index = int(np.searchsorted(times, execution_time_ms, side="right") - 1)
    deadline_ms = decision_time_ms + horizon_seconds * 1000
    if entry_index < 0:
        return _skipped_trade(decision_time_ms, side, "no_entry_book")
    entry_book_age_ms = execution_time_ms - int(times[entry_index])
    if entry_book_age_ms > max_book_staleness_ms:
        return _skipped_trade(decision_time_ms, side, "stale_entry_book")
    exit_right = int(np.searchsorted(times, deadline_ms, side="right"))
    if entry_index + 1 >= exit_right:
        return _skipped_trade(decision_time_ms, side, "no_exit_path")

    if side == OrderSide.BUY:
        entry_price = float(book["ask_price"][entry_index])
        path_prices = book["bid_price"][entry_index + 1 : exit_right]
        target_price = entry_price * math.exp(target_bps / 10_000)
        stop_price = entry_price * math.exp(-initial_stop_bps / 10_000)
        target_mask = path_prices >= target_price
        stop_mask = path_prices <= stop_price
    else:
        entry_price = float(book["bid_price"][entry_index])
        path_prices = book["ask_price"][entry_index + 1 : exit_right]
        target_price = entry_price * math.exp(-target_bps / 10_000)
        stop_price = entry_price * math.exp(initial_stop_bps / 10_000)
        target_mask = path_prices <= target_price
        stop_mask = path_prices >= stop_price

    relative_target = _first_true_index(target_mask)
    relative_stop = _first_true_index(stop_mask)
    if relative_target is not None and (
        relative_stop is None or relative_target < relative_stop
    ):
        exit_index = entry_index + 1 + relative_target
        exit_price = float(target_price)
        exit_reason = "TARGET"
    elif relative_stop is not None:
        exit_index = entry_index + 1 + relative_stop
        exit_price = float(stop_price)
        exit_reason = "STOP"
    else:
        exit_index = exit_right - 1
        exit_price = float(path_prices[-1])
        exit_reason = "TIMEOUT"

    exit_book_age_ms = deadline_ms - int(times[exit_index]) if exit_reason == "TIMEOUT" else 0
    if exit_reason == "TIMEOUT" and exit_book_age_ms > max_book_staleness_ms:
        return _skipped_trade(decision_time_ms, side, "stale_timeout_book")
    gross_bps = _gross_bps(side, entry_price, exit_price)
    path_gross = np.array([_gross_bps(side, entry_price, price) for price in path_prices])
    return {
        "would_enter": True,
        "skip_reason": None,
        "decision_time_ms": decision_time_ms,
        "entry_time_ms": execution_time_ms,
        "entry_book_time_ms": int(times[entry_index]),
        "entry_book_age_ms": int(entry_book_age_ms),
        "exit_time_ms": int(times[exit_index]),
        "side": side.value,
        "horizon_seconds": horizon_seconds,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_executable_bps": gross_bps,
        "maximum_favorable_executable_bps": float(path_gross.max()),
        "maximum_adverse_executable_bps": float(path_gross.min()),
    }


def summarize_executable_trades(
    trades: pd.DataFrame,
    *,
    selected_signal_count: int,
    cost_levels_bps: tuple[float, ...],
) -> dict[str, object]:
    executable_count = int(len(trades))
    if trades.empty:
        gross_mean = 0.0
        gross_sum = 0.0
        win_rate = 0.0
        maximum_drawdown = 0.0
        exit_reasons: dict[str, int] = {}
    else:
        gross = trades["gross_executable_bps"].to_numpy(dtype=np.float64)
        gross_mean = float(gross.mean())
        gross_sum = float(gross.sum())
        win_rate = float((gross > 0).mean())
        cumulative = np.cumsum(gross)
        peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
        drawdowns = np.concatenate(([0.0], cumulative)) - peaks
        maximum_drawdown = float(-drawdowns.min())
        exit_reasons = (
            trades["exit_reason"].value_counts().sort_index().astype(int).to_dict()
        )
    return {
        "selected_signal_count": int(selected_signal_count),
        "executable_count": executable_count,
        "skipped_count": int(selected_signal_count - executable_count),
        "gross_executable_mean_bps": gross_mean,
        "gross_executable_total_bps": gross_sum,
        "break_even_explicit_cost_bps": gross_mean,
        "gross_win_rate": win_rate,
        "maximum_drawdown_bps": maximum_drawdown,
        "exit_reasons": exit_reasons,
        "cost_sensitivity": [
            {
                "explicit_cost_bps": float(cost_bps),
                "net_executable_mean_bps": float(gross_mean - cost_bps),
                "net_executable_total_bps": float(gross_sum - executable_count * cost_bps),
                "net_positive": bool(gross_mean - cost_bps > 0),
            }
            for cost_bps in cost_levels_bps
        ],
    }


def _load_confirmation_book_events(
    selected_rows: pd.DataFrame,
    *,
    confirmation_wal_root: Path,
    horizon_seconds: int,
    latency_ms: int,
) -> pd.DataFrame:
    if selected_rows.empty:
        raise ValueError("selected_rows cannot be empty")
    start_ms = int(selected_rows["decision_time_ms"].min()) + latency_ms - 2_000
    end_ms = int(selected_rows["decision_time_ms"].max()) + horizon_seconds * 1000 + 2_000
    interval_start = datetime.fromtimestamp(start_ms / 1000, tz=UTC)
    interval_end = datetime.fromtimestamp(end_ms / 1000, tz=UTC)
    return load_book_interval(confirmation_wal_root / "depth", interval_start, interval_end)


def _book_arrays(book_events: pd.DataFrame) -> dict[str, np.ndarray]:
    ordered = book_events.sort_values(
        ["event_time_ms", "sequence_id"], kind="stable"
    )
    return {
        "event_time_ms": ordered["event_time_ms"].to_numpy(dtype=np.int64),
        "bid_price": ordered["bid_price"].to_numpy(dtype=np.float64),
        "ask_price": ordered["ask_price"].to_numpy(dtype=np.float64),
    }


def _apply_one_position_constraint(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    ordered = trades.sort_values(
        ["decision_time_ms", "gross_executable_bps"],
        ascending=[True, False],
        kind="stable",
    )
    kept = []
    position_exit_ms = -1
    for row in ordered.itertuples(index=False):
        decision_time_ms = int(row.decision_time_ms)
        if decision_time_ms <= position_exit_ms:
            continue
        kept.append(row._asdict())
        position_exit_ms = int(row.exit_time_ms)
    return pd.DataFrame(kept)


def _positive_at_cost(summary: dict[str, object], cost_bps: float) -> bool:
    for item in summary["cost_sensitivity"]:
        if float(item["explicit_cost_bps"]) == cost_bps:
            return bool(item["net_positive"])
    return False


def _gross_bps(side: OrderSide, entry_price: float, exit_price: float) -> float:
    if side == OrderSide.BUY:
        return math.log(exit_price / entry_price) * 10_000
    return math.log(entry_price / exit_price) * 10_000


def _first_true_index(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return None if len(indices) == 0 else int(indices[0])


def _skipped_trade(decision_time_ms: int, side: OrderSide, reason: str) -> dict[str, object]:
    return {
        "would_enter": False,
        "skip_reason": reason,
        "decision_time_ms": decision_time_ms,
        "side": side.value,
    }


def _validate_parameters(
    *,
    gate_development_wal_roots: tuple[Path, ...],
    horizon_seconds: int,
    threshold_bps: float,
    base_condition_fraction: float,
    base_conditions: tuple[dict[str, str], ...],
    gate_feature: str,
    gate_direction: str,
    gate_fraction: float,
    max_book_staleness_ms: int,
    latency_ms: int,
    initial_stop_bps: float,
    minimum_executable_count: int,
    cost_levels_bps: tuple[float, ...],
) -> None:
    if not gate_development_wal_roots:
        raise ValueError("gate_development_wal_roots cannot be empty")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    if threshold_bps <= 0:
        raise ValueError("threshold_bps must be positive")
    if not 0 < base_condition_fraction < 0.5:
        raise ValueError("base_condition_fraction must be between 0 and 0.5")
    if not base_conditions:
        raise ValueError("base_conditions cannot be empty")
    for condition in base_conditions:
        if condition["direction"] not in {"high", "low"}:
            raise ValueError("base condition direction must be high or low")
    if not gate_feature:
        raise ValueError("gate_feature cannot be empty")
    if gate_direction not in {"high", "low"}:
        raise ValueError("gate_direction must be high or low")
    if not 0 < gate_fraction <= 0.5:
        raise ValueError("gate_fraction must be between 0 and 0.5")
    if max_book_staleness_ms < 0:
        raise ValueError("max_book_staleness_ms cannot be negative")
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if minimum_executable_count < 1:
        raise ValueError("minimum_executable_count must be positive")
    if not cost_levels_bps or any(cost_bps < 0 for cost_bps in cost_levels_bps):
        raise ValueError("cost_levels_bps must contain non-negative values")


def _parse_paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(part.strip()) for part in value.split(",") if part.strip())
    if not paths:
        raise ValueError("path list cannot be empty")
    return paths


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "approved_strategy": report["approved_strategy"],
        "selected_signal_count": report.get("selected_signal_count"),
        "independent_summary": report.get("independent_summary"),
        "one_position_summary": report.get("one_position_summary"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
