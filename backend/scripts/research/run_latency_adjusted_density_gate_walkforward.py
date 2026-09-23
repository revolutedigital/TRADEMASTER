"""Walk-forward density gate for a frozen latency-adjusted top-p candidate.

The frozen candidate failed the extended holdout when selected signal density
exploded. This diagnostic tests a causal no-trade gate: before taking a signal,
count how many frozen candidate signals occurred in a trailing time window and
allow entry only below a fixed threshold.

Gate parameters are selected only from prior blocks, then applied to the next
block. It is research-only and never approves a strategy or sends orders.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.trailing_portfolio import (  # noqa: E402
    HistoricalTrailingSimulator,
    replay_one_position_portfolio,
    summarize_portfolio,
)
from app.services.research.research_dataset import load_trade_interval  # noqa: E402
from scripts.research.run_latency_adjusted_frozen_top_p_holdout import (  # noqa: E402
    _predict_artifact_probabilities,
)
from scripts.research.run_latency_adjusted_top_p_scout import _policies  # noqa: E402


@dataclass(frozen=True)
class BlockSpec:
    label: str
    dataset_root: Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-report", type=Path, required=True)
    parser.add_argument(
        "--blocks",
        nargs="+",
        required=True,
        help="Space-separated LABEL=dataset_root entries in chronological order.",
    )
    parser.add_argument(
        "--trade-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "normalized"
        / "aggTrades",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "latency-adjusted-density-gate-walkforward",
    )
    parser.add_argument("--policy-set", choices=("v1", "low_cost"), default="low_cost")
    parser.add_argument("--policy-name", default="micro_8_3")
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--horizon-seconds", type=int, default=300)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    parser.add_argument("--minimum-training-blocks", type=int, default=3)
    parser.add_argument("--minimum-training-trades", type=int, default=50)
    parser.add_argument("--lookbacks-seconds", default="300,900,1800,3600,7200,21600,86400")
    parser.add_argument("--limits", default="0,1,2,3,5,8,13,21,34,55,89,144,233,377,610")
    arguments = parser.parse_args()

    report = run_density_gate_walkforward(
        artifact_report=arguments.artifact_report,
        block_specs=_parse_block_specs(arguments.blocks),
        trade_root=arguments.trade_root,
        output_root=arguments.output_root,
        policy_set=arguments.policy_set,
        policy_name=arguments.policy_name,
        tail_fraction=arguments.tail_fraction,
        horizon_seconds=arguments.horizon_seconds,
        latency_ms=arguments.latency_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
        minimum_training_blocks=arguments.minimum_training_blocks,
        minimum_training_trades=arguments.minimum_training_trades,
        lookbacks_seconds=_parse_ints(arguments.lookbacks_seconds),
        limits=_parse_ints(arguments.limits, allow_zero=True),
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_density_gate_walkforward(
    *,
    artifact_report: Path,
    block_specs: tuple[BlockSpec, ...],
    trade_root: Path,
    output_root: Path,
    policy_set: str,
    policy_name: str,
    tail_fraction: float,
    horizon_seconds: int,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
    minimum_training_blocks: int,
    minimum_training_trades: int,
    lookbacks_seconds: tuple[int, ...],
    limits: tuple[int, ...],
) -> dict[str, object]:
    if len(block_specs) <= minimum_training_blocks:
        raise ValueError("block count must exceed minimum_training_blocks")
    artifact_report_payload = json.loads(artifact_report.read_text(encoding="utf-8"))
    artifact = artifact_report_payload["frozen_artifact"]
    model_sha256 = str(artifact_report_payload["model_sha256"])
    threshold = float(artifact["probability_threshold"])
    selected_signals = _load_selected_signals(
        artifact=artifact,
        block_specs=block_specs,
        threshold=threshold,
        horizon_seconds=horizon_seconds,
        tail_fraction=tail_fraction,
    )
    simulator = _build_simulator(
        selected_signals,
        trade_root=trade_root,
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    policy = _policy_by_name(policy_set, policy_name)
    candidate_results = _evaluate_gate_candidates(
        selected_signals,
        simulator=simulator,
        policy=policy,
        tail_fraction=tail_fraction,
        lookbacks_seconds=lookbacks_seconds,
        limits=limits,
    )
    baseline_result = _evaluate_single_candidate(
        selected_signals,
        simulator=simulator,
        policy=policy,
        tail_fraction=tail_fraction,
        lookback_seconds=0,
        limit=math.inf,
        gate_name="no_density_gate",
    )
    block_labels = [block.label for block in block_specs]
    fold_results = []
    fold_trade_frames = []
    for index in range(minimum_training_blocks, len(block_labels)):
        training_blocks = tuple(block_labels[:index])
        test_block = block_labels[index]
        selected_gate = _select_gate(
            candidate_results,
            training_blocks=training_blocks,
            minimum_training_trades=minimum_training_trades,
        )
        test_trades = selected_gate["trades"][
            selected_gate["trades"]["block"] == test_block
        ].copy()
        if not test_trades.empty:
            test_trades["fold_test_block"] = test_block
            test_trades["gate_name"] = selected_gate["gate_name"]
            fold_trade_frames.append(test_trades)
        train_summary = _summarize_block_subset(selected_gate["trades"], training_blocks)
        test_summary = summarize_portfolio(test_trades.drop(columns=["block", "gate_name"], errors="ignore"))
        baseline_test_trades = baseline_result["trades"][
            baseline_result["trades"]["block"] == test_block
        ].copy()
        baseline_test_summary = summarize_portfolio(
            baseline_test_trades.drop(columns=["block"], errors="ignore")
        )
        fold_results.append(
            {
                "fold": len(fold_results) + 1,
                "training_blocks": list(training_blocks),
                "test_block": test_block,
                "selected_gate": {
                    "gate_name": selected_gate["gate_name"],
                    "lookback_seconds": selected_gate["lookback_seconds"],
                    "limit": selected_gate["limit"],
                    "training_score": selected_gate["training_score"],
                },
                "training_summary": train_summary,
                "test_summary": test_summary,
                "baseline_test_summary": baseline_test_summary,
            }
        )
    fold_trades = (
        pd.concat(fold_trade_frames, ignore_index=True) if fold_trade_frames else pd.DataFrame()
    )
    aggregate = summarize_portfolio(
        fold_trades.drop(columns=["block", "gate_name", "fold_test_block"], errors="ignore")
    )
    baseline_fold_trades = pd.concat(
        [
            baseline_result["trades"][baseline_result["trades"]["block"] == block]
            for block in block_labels[minimum_training_blocks:]
        ],
        ignore_index=True,
    )
    baseline_aggregate = summarize_portfolio(
        baseline_fold_trades.drop(columns=["block"], errors="ignore")
    )
    status, status_reasons = _status(aggregate, baseline_aggregate, fold_results)
    report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "approval_reasons": [
            "density_gate_walkforward_research_only",
            "offline_aggtrades_replay_not_exchange_execution",
            "future_book_or_shadow_validation_required",
        ],
        "experiment_stage": "latency_adjusted_density_gate_walkforward",
        "status": status,
        "status_reasons": status_reasons,
        "model_sha256": model_sha256,
        "threshold": threshold,
        "tail_fraction": tail_fraction,
        "horizon_seconds": horizon_seconds,
        "policy_set": policy_set,
        "policy_name": policy_name,
        "latency_ms": latency_ms,
        "expected_cost_bps": expected_cost_bps,
        "stress_cost_bps": stress_cost_bps,
        "blocks": [
            {"label": block.label, "dataset_root": str(block.dataset_root)}
            for block in block_specs
        ],
        "grid": {
            "lookbacks_seconds": list(lookbacks_seconds),
            "limits": list(limits),
            "candidate_count": len(candidate_results),
        },
        "selected_signal_count": int(len(selected_signals)),
        "fold_results": fold_results,
        "walkforward_summary": aggregate,
        "baseline_walkforward_summary": baseline_aggregate,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    trades_path = output_root / "density_gate_walkforward_trades.parquet"
    fold_trades.to_parquet(trades_path, index=False)
    report["trades_path"] = str(trades_path)
    report_path = output_root / "report.json"
    payload = {
        **report,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _load_selected_signals(
    *,
    artifact: dict[str, Any],
    block_specs: tuple[BlockSpec, ...],
    threshold: float,
    horizon_seconds: int,
    tail_fraction: float,
) -> pd.DataFrame:
    frames = []
    threshold_column = f"threshold_top_{tail_fraction:.0%}"
    for block in block_specs:
        for path in sorted(block.dataset_root.glob("date=*/research_rows.parquet")):
            rows = pd.read_parquet(path)
            rows = rows[rows["horizon_seconds"] == horizon_seconds].copy()
            if rows.empty:
                continue
            probabilities = _predict_artifact_probabilities(artifact, rows)
            rows["probability"] = probabilities
            selected = rows[rows["probability"] >= threshold].copy()
            if selected.empty:
                continue
            selected["block"] = block.label
            selected["date"] = path.parent.name.removeprefix("date=")
            selected[threshold_column] = threshold
            frames.append(
                selected[
                    [
                        "decision_time_ms",
                        "side",
                        "horizon_seconds",
                        "probability",
                        threshold_column,
                        "block",
                        "date",
                    ]
                ]
            )
    if not frames:
        raise ValueError("frozen artifact selected no signals in the supplied blocks")
    return pd.concat(frames, ignore_index=True).sort_values(
        ["decision_time_ms", "probability"],
        ascending=[True, False],
        kind="stable",
    )


def _build_simulator(
    selected_signals: pd.DataFrame,
    *,
    trade_root: Path,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
) -> HistoricalTrailingSimulator:
    start = datetime.fromtimestamp(
        selected_signals["decision_time_ms"].min() / 1000,
        tz=UTC,
    ) - timedelta(seconds=1)
    end = datetime.fromtimestamp(
        (
            selected_signals["decision_time_ms"]
            + selected_signals["horizon_seconds"] * 1000
        ).max()
        / 1000,
        tz=UTC,
    ) + timedelta(seconds=1)
    trades = load_trade_interval(trade_root, start, end, expected_product="usdm_perpetual")
    return HistoricalTrailingSimulator(
        trades["event_time_ms"].to_numpy(),
        trades["price"].to_numpy(),
        latency_ms=latency_ms,
        expected_round_trip_bps=expected_cost_bps,
        stress_round_trip_bps=stress_cost_bps,
    )


def _evaluate_gate_candidates(
    selected_signals: pd.DataFrame,
    *,
    simulator: HistoricalTrailingSimulator,
    policy: Any,
    tail_fraction: float,
    lookbacks_seconds: tuple[int, ...],
    limits: tuple[int, ...],
) -> list[dict[str, Any]]:
    results = []
    for lookback_seconds in lookbacks_seconds:
        counted = _add_recent_signal_count(selected_signals, lookback_seconds)
        for limit in limits:
            gated = counted[counted["recent_signal_count"] <= limit].copy()
            results.append(
                _evaluate_single_candidate(
                    gated,
                    simulator=simulator,
                    policy=policy,
                    tail_fraction=tail_fraction,
                    lookback_seconds=lookback_seconds,
                    limit=limit,
                    gate_name=f"recent_{lookback_seconds}s_lte_{limit}",
                )
            )
    return results


def _evaluate_single_candidate(
    selected_signals: pd.DataFrame,
    *,
    simulator: HistoricalTrailingSimulator,
    policy: Any,
    tail_fraction: float,
    lookback_seconds: int,
    limit: int | float,
    gate_name: str,
) -> dict[str, Any]:
    if selected_signals.empty:
        trades = pd.DataFrame()
    else:
        trades = replay_one_position_portfolio(
            selected_signals.drop(columns=["block", "date"], errors="ignore"),
            simulator,
            tail_fraction=tail_fraction,
            policy=policy,
        )
        if not trades.empty:
            block_map = selected_signals.drop_duplicates("decision_time_ms").set_index(
                "decision_time_ms"
            )["block"]
            trades["block"] = trades["decision_time_ms"].map(block_map)
    return {
        "gate_name": gate_name,
        "lookback_seconds": lookback_seconds,
        "limit": limit,
        "trades": trades,
    }


def _add_recent_signal_count(frame: pd.DataFrame, lookback_seconds: int) -> pd.DataFrame:
    ordered = frame.sort_values("decision_time_ms", kind="stable").copy()
    times = ordered["decision_time_ms"].to_numpy(dtype=np.int64)
    left = np.searchsorted(times, times - lookback_seconds * 1000, side="left")
    ordered["recent_signal_count"] = np.arange(len(times)) - left
    return ordered


def _select_gate(
    candidates: list[dict[str, Any]],
    *,
    training_blocks: tuple[str, ...],
    minimum_training_trades: int,
) -> dict[str, Any]:
    ranked = []
    for candidate in candidates:
        summary = _summarize_block_subset(candidate["trades"], training_blocks)
        trade_count = int(summary["trade_count"])
        if trade_count < minimum_training_trades:
            continue
        stress_total = float(summary["stress_total_log_return_bps"])
        stress_mean = float(summary["stress_mean_bps"])
        expected_total = float(summary["expected_total_log_return_bps"])
        ranked.append(
            (
                (stress_total > 0, stress_total, stress_mean, expected_total, -trade_count),
                {
                    **candidate,
                    "training_score": {
                        "training_blocks": list(training_blocks),
                        "minimum_training_trades": minimum_training_trades,
                        **summary,
                    },
                },
            )
        )
    if not ranked:
        raise ValueError("no density gate candidate met the training trade minimum")
    return max(ranked, key=lambda item: item[0])[1]


def _summarize_block_subset(trades: pd.DataFrame, blocks: tuple[str, ...]) -> dict[str, float | int]:
    if trades.empty:
        return summarize_portfolio(pd.DataFrame())
    subset = trades[trades["block"].isin(blocks)].copy()
    return summarize_portfolio(subset.drop(columns=["block"], errors="ignore"))


def _status(
    walkforward_summary: dict[str, float | int],
    baseline_summary: dict[str, float | int],
    fold_results: list[dict[str, object]],
) -> tuple[str, list[str]]:
    stress_mean = float(walkforward_summary["stress_mean_bps"])
    baseline_stress_mean = float(baseline_summary["stress_mean_bps"])
    positive_folds = sum(
        float(fold["test_summary"]["stress_total_log_return_bps"]) > 0
        for fold in fold_results
    )
    if stress_mean > 0 and positive_folds == len(fold_results):
        return (
            "DENSITY_GATE_WALKFORWARD_STRESS_POSITIVE_ALL_FOLDS_RESEARCH_ONLY",
            ["not_approved_strategy", "requires_book_or_shadow_confirmation"],
        )
    if stress_mean > 0 and stress_mean > baseline_stress_mean:
        return (
            "DENSITY_GATE_WALKFORWARD_IMPROVED_BUT_NOT_ALL_FOLDS",
            ["stress_positive_after_gate", "some_folds_failed_or_fragile"],
        )
    return (
        "DENSITY_GATE_WALKFORWARD_FAILED",
        ["no_robust_walkforward_density_gate", "do_not_promote_to_strategy"],
    )


def _policy_by_name(policy_set: str, policy_name: str) -> Any:
    for policy in _policies(policy_set):
        if policy.name == policy_name:
            return policy
    raise ValueError(f"policy {policy_name!r} not found in policy set {policy_set!r}")


def _parse_block_specs(raw_blocks: list[str]) -> tuple[BlockSpec, ...]:
    blocks = []
    for raw in raw_blocks:
        label, separator, path = raw.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"invalid block spec: {raw!r}")
        blocks.append(BlockSpec(label=label, dataset_root=Path(path)))
    return tuple(blocks)


def _parse_ints(raw: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or (value == 0 and not allow_zero):
            raise ValueError("values must be positive" if not allow_zero else "values cannot be negative")
        values.append(value)
    if not values:
        raise ValueError("at least one value is required")
    return tuple(values)


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "walkforward_summary": report["walkforward_summary"],
        "baseline_walkforward_summary": report["baseline_walkforward_summary"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
