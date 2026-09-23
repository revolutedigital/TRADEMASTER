"""Fixed density-gate holdout for a frozen latency-adjusted top-p candidate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.trailing_portfolio import (  # noqa: E402
    replay_one_position_portfolio,
    summarize_portfolio,
)
from scripts.research.run_latency_adjusted_density_gate_walkforward import (  # noqa: E402
    _add_recent_signal_count,
    _build_simulator,
    _load_selected_signals,
    _parse_block_specs,
    _policy_by_name,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-report", type=Path, required=True)
    parser.add_argument("--blocks", nargs="+", required=True)
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
        / "latency-adjusted-fixed-density-gate-holdout",
    )
    parser.add_argument("--lookback-seconds", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--policy-set", choices=("v1", "low_cost"), default="low_cost")
    parser.add_argument("--policy-name", default="micro_8_3")
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--horizon-seconds", type=int, default=300)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps", type=float, default=2.0)
    arguments = parser.parse_args()
    if arguments.lookback_seconds <= 0:
        parser.error("--lookback-seconds must be positive")
    if arguments.limit < 0:
        parser.error("--limit cannot be negative")

    report = run_fixed_density_gate_holdout(
        artifact_report=arguments.artifact_report,
        block_specs=_parse_block_specs(arguments.blocks),
        trade_root=arguments.trade_root,
        output_root=arguments.output_root,
        lookback_seconds=arguments.lookback_seconds,
        limit=arguments.limit,
        policy_set=arguments.policy_set,
        policy_name=arguments.policy_name,
        tail_fraction=arguments.tail_fraction,
        horizon_seconds=arguments.horizon_seconds,
        latency_ms=arguments.latency_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_cost_bps=arguments.stress_cost_bps,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fixed_density_gate_holdout(
    *,
    artifact_report: Path,
    block_specs: tuple[object, ...],
    trade_root: Path,
    output_root: Path,
    lookback_seconds: int,
    limit: int,
    policy_set: str,
    policy_name: str,
    tail_fraction: float,
    horizon_seconds: int,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
) -> dict[str, object]:
    artifact_payload = json.loads(artifact_report.read_text(encoding="utf-8"))
    artifact = artifact_payload["frozen_artifact"]
    threshold = float(artifact["probability_threshold"])
    selected_signals = _load_selected_signals(
        artifact=artifact,
        block_specs=block_specs,
        threshold=threshold,
        horizon_seconds=horizon_seconds,
        tail_fraction=tail_fraction,
    )
    counted = _add_recent_signal_count(selected_signals, lookback_seconds)
    gated = counted[counted["recent_signal_count"] <= limit].copy()
    simulator = _build_simulator(
        selected_signals,
        trade_root=trade_root,
        latency_ms=latency_ms,
        expected_cost_bps=expected_cost_bps,
        stress_cost_bps=stress_cost_bps,
    )
    policy = _policy_by_name(policy_set, policy_name)
    trades = replay_one_position_portfolio(
        gated.drop(columns=["block", "date", "recent_signal_count"], errors="ignore"),
        simulator,
        tail_fraction=tail_fraction,
        policy=policy,
    )
    if not trades.empty:
        block_map = gated.drop_duplicates("decision_time_ms").set_index("decision_time_ms")[
            "block"
        ]
        trades["block"] = trades["decision_time_ms"].map(block_map)
    summary = summarize_portfolio(trades.drop(columns=["block"], errors="ignore"))
    status, status_reasons = _status(summary)
    output_root.mkdir(parents=True, exist_ok=True)
    trades_path = output_root / "fixed_density_gate_trades.parquet"
    trades.to_parquet(trades_path, index=False)
    report_path = output_root / "report.json"
    report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "approval_reasons": [
            "fixed_density_gate_holdout_research_only",
            "offline_aggtrades_replay_not_exchange_execution",
            "book_or_shadow_confirmation_required",
        ],
        "experiment_stage": "latency_adjusted_fixed_density_gate_holdout",
        "status": status,
        "status_reasons": status_reasons,
        "model_sha256": artifact_payload["model_sha256"],
        "threshold": threshold,
        "tail_fraction": tail_fraction,
        "horizon_seconds": horizon_seconds,
        "policy_set": policy_set,
        "policy_name": policy_name,
        "latency_ms": latency_ms,
        "expected_cost_bps": expected_cost_bps,
        "stress_cost_bps": stress_cost_bps,
        "density_gate": {
            "lookback_seconds": lookback_seconds,
            "limit": limit,
            "selected_signal_count_before_gate": int(len(selected_signals)),
            "selected_signal_count_after_gate": int(len(gated)),
        },
        "blocks": [
            {"label": block.label, "dataset_root": str(block.dataset_root)}
            for block in block_specs
        ],
        "summary": summary,
        "trades_path": str(trades_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _status(summary: dict[str, float | int]) -> tuple[str, list[str]]:
    stress_mean = float(summary["stress_mean_bps"])
    expected_mean = float(summary["expected_mean_bps"])
    if stress_mean > 0:
        return (
            "FIXED_DENSITY_GATE_HOLDOUT_STRESS_POSITIVE_RESEARCH_ONLY",
            ["not_approved_strategy", "requires_additional_confirmation"],
        )
    if expected_mean > 0:
        return (
            "FIXED_DENSITY_GATE_HOLDOUT_EXPECTED_POSITIVE_STRESS_NEGATIVE",
            ["stress_cost_negative", "do_not_promote_to_strategy"],
        )
    return (
        "FIXED_DENSITY_GATE_HOLDOUT_FAILED",
        ["expected_and_stress_negative", "do_not_promote_to_strategy"],
    )


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "density_gate": report["density_gate"],
        "summary": report["summary"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
