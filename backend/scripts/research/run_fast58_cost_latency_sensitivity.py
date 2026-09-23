"""Fast58 cost/latency sensitivity audit for the frozen Fast57 candidate.

Fast58 does not search for a better rule. It replays the already frozen
Fast57 candidate under harsher execution assumptions so the research can tell
whether the current edge is economically robust or fragile.

It is research-only, never sends orders, and never approves a strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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
from scripts.research.run_fast57_prospective_update import (  # noqa: E402
    DEFAULT_ARTIFACT_REPORT,
    DEFAULT_SEED_RESEARCH_ROOT,
    DEFAULT_TRADE_ROOT,
)
from scripts.research.run_latency_adjusted_density_gate_walkforward import (  # noqa: E402
    _add_recent_signal_count,
    _load_selected_signals,
    _parse_block_specs,
    _policy_by_name,
)


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_v1"
    / "reports"
    / "fast58-cost-latency-sensitivity"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-report", type=Path, default=DEFAULT_ARTIFACT_REPORT)
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=[f"seed={DEFAULT_SEED_RESEARCH_ROOT}"],
        help="Space-separated LABEL=dataset_root blocks.",
    )
    parser.add_argument("--trade-root", type=Path, default=DEFAULT_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lookback-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--policy-set", choices=("v1", "low_cost"), default="low_cost")
    parser.add_argument("--policy-name", default="micro_8_3")
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--horizon-seconds", type=int, default=300)
    parser.add_argument("--latency-ms-values", default="100,250,500,1000")
    parser.add_argument("--expected-cost-bps", type=float, default=1.0)
    parser.add_argument("--stress-cost-bps-values", default="2,3,4,5,8")
    arguments = parser.parse_args()
    if arguments.lookback_seconds <= 0:
        parser.error("--lookback-seconds must be positive")
    if arguments.limit < 0:
        parser.error("--limit cannot be negative")
    if arguments.expected_cost_bps < 0:
        parser.error("--expected-cost-bps cannot be negative")

    latencies_ms = parse_int_grid(arguments.latency_ms_values, name="latency-ms-values")
    stress_costs_bps = parse_float_grid(
        arguments.stress_cost_bps_values,
        name="stress-cost-bps-values",
    )
    below_expected = [
        value for value in stress_costs_bps if value < arguments.expected_cost_bps
    ]
    if below_expected:
        parser.error("--stress-cost-bps-values cannot be below --expected-cost-bps")

    report = run_fast58_cost_latency_sensitivity(
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
        latencies_ms=latencies_ms,
        expected_cost_bps=arguments.expected_cost_bps,
        stress_costs_bps=stress_costs_bps,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast58_cost_latency_sensitivity(
    *,
    artifact_report: Path,
    block_specs: tuple[Any, ...],
    trade_root: Path,
    output_root: Path,
    lookback_seconds: int,
    limit: int,
    policy_set: str,
    policy_name: str,
    tail_fraction: float,
    horizon_seconds: int,
    latencies_ms: tuple[int, ...],
    expected_cost_bps: float,
    stress_costs_bps: tuple[float, ...],
) -> dict[str, Any]:
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
    trade_events = load_trade_interval(
        trade_root,
        datetime.fromtimestamp(selected_signals["decision_time_ms"].min() / 1000, tz=UTC)
        - timedelta(seconds=1),
        datetime.fromtimestamp(
            (
                selected_signals["decision_time_ms"]
                + selected_signals["horizon_seconds"] * 1000
            ).max()
            / 1000,
            tz=UTC,
        )
        + timedelta(seconds=1),
        expected_product="usdm_perpetual",
    )
    policy = _policy_by_name(policy_set, policy_name)
    scenarios: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for latency_ms in latencies_ms:
        for stress_cost_bps in stress_costs_bps:
            simulator = HistoricalTrailingSimulator(
                trade_events["event_time_ms"].to_numpy(),
                trade_events["price"].to_numpy(),
                latency_ms=latency_ms,
                expected_round_trip_bps=expected_cost_bps,
                stress_round_trip_bps=stress_cost_bps,
            )
            gated_trades = replay_one_position_portfolio(
                _prediction_frame(gated),
                simulator,
                tail_fraction=tail_fraction,
                policy=policy,
            )
            ungated_trades = replay_one_position_portfolio(
                _prediction_frame(counted),
                simulator,
                tail_fraction=tail_fraction,
                policy=policy,
            )
            scenario = build_sensitivity_scenario(
                latency_ms=latency_ms,
                expected_cost_bps=expected_cost_bps,
                stress_cost_bps=stress_cost_bps,
                gated_trades=gated_trades,
                ungated_trades=ungated_trades,
            )
            scenarios.append(scenario)
            scenario_root = (
                output_root
                / f"latency_{latency_ms}ms_stress_{format_cost_label(stress_cost_bps)}bps"
            )
            scenario_root.mkdir(parents=True, exist_ok=True)
            gated_trades.to_parquet(scenario_root / "gated_trades.parquet", index=False)
            ungated_trades.to_parquet(scenario_root / "ungated_trades.parquet", index=False)

    sensitivity_summary = summarize_sensitivity_scenarios(scenarios)
    status, status_reasons = fast58_status(sensitivity_summary)
    report_path = output_root / "report.json"
    report: dict[str, Any] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "approval_reasons": [
            "fast58_sensitivity_research_only",
            "offline_aggtrades_replay_not_exchange_execution",
            "sensitivity_audit_cannot_approve_strategy",
        ],
        "experiment_stage": "fast58_cost_latency_sensitivity",
        "status": status,
        "status_reasons": status_reasons,
        "candidate": {
            "model_sha256": artifact_payload["model_sha256"],
            "threshold": threshold,
            "tail_fraction": tail_fraction,
            "horizon_seconds": horizon_seconds,
            "policy_set": policy_set,
            "policy_name": policy_name,
            "density_gate": {
                "lookback_seconds": lookback_seconds,
                "limit": limit,
                "selected_signal_count_before_gate": int(len(selected_signals)),
                "selected_signal_count_after_gate": int(len(gated)),
            },
        },
        "blocks": [
            {"label": block.label, "dataset_root": str(block.dataset_root)}
            for block in block_specs
        ],
        "grid": {
            "latencies_ms": list(latencies_ms),
            "expected_cost_bps": expected_cost_bps,
            "stress_costs_bps": list(stress_costs_bps),
            "scenario_count": len(scenarios),
        },
        "sensitivity_summary": sensitivity_summary,
        "scenarios": scenarios,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_int_grid(raw: str, *, name: str) -> tuple[int, ...]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError(f"{name} cannot contain negative values")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(dict.fromkeys(values))


def parse_float_grid(raw: str, *, name: str) -> tuple[float, ...]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError(f"{name} cannot contain negative values")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(dict.fromkeys(values))


def build_sensitivity_scenario(
    *,
    latency_ms: int,
    expected_cost_bps: float,
    stress_cost_bps: float,
    gated_trades: pd.DataFrame,
    ungated_trades: pd.DataFrame,
) -> dict[str, Any]:
    gated_summary = summarize_portfolio(gated_trades)
    ungated_summary = summarize_portfolio(ungated_trades)
    gated_gross = gross_trade_summary(gated_trades)
    ungated_gross = gross_trade_summary(ungated_trades)
    return {
        "latency_ms": latency_ms,
        "expected_cost_bps": expected_cost_bps,
        "stress_cost_bps": stress_cost_bps,
        "gated_summary": gated_summary,
        "ungated_summary": ungated_summary,
        "gated_gross_summary": gated_gross,
        "ungated_gross_summary": ungated_gross,
        "gated_break_even_roundtrip_cost_bps": gated_gross["gross_mean_bps"],
        "ungated_break_even_roundtrip_cost_bps": ungated_gross["gross_mean_bps"],
        "gated_stress_positive": float(gated_summary["stress_mean_bps"]) > 0,
        "ungated_stress_positive": float(ungated_summary["stress_mean_bps"]) > 0,
        "gated_beats_ungated": float(gated_summary["stress_total_log_return_bps"])
        > float(ungated_summary["stress_total_log_return_bps"]),
    }


def gross_trade_summary(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "gross_mean_bps": 0.0,
            "gross_total_log_return_bps": 0.0,
            "gross_win_rate": 0.0,
        }
    gross = trades["gross_bps"].to_numpy(dtype="float64")
    return {
        "gross_mean_bps": float(gross.mean()),
        "gross_total_log_return_bps": float(gross.sum()),
        "gross_win_rate": float((gross > 0).mean()),
    }


def summarize_sensitivity_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [scenario for scenario in scenarios if scenario["gated_stress_positive"]]
    beats_ungated = [scenario for scenario in scenarios if scenario["gated_beats_ungated"]]
    by_latency: list[dict[str, Any]] = []
    for latency_ms in sorted({int(scenario["latency_ms"]) for scenario in scenarios}):
        rows = [scenario for scenario in scenarios if int(scenario["latency_ms"]) == latency_ms]
        positive_costs = [
            float(scenario["stress_cost_bps"])
            for scenario in rows
            if scenario["gated_stress_positive"]
        ]
        first_row = min(rows, key=lambda scenario: float(scenario["stress_cost_bps"]))
        by_latency.append(
            {
                "latency_ms": latency_ms,
                "break_even_roundtrip_cost_bps": first_row[
                    "gated_break_even_roundtrip_cost_bps"
                ],
                "max_tested_positive_stress_cost_bps": max(positive_costs)
                if positive_costs
                else None,
                "positive_scenario_count": len(positive_costs),
                "scenario_count": len(rows),
            }
        )
    return {
        "scenario_count": len(scenarios),
        "positive_stress_scenario_count": len(positive),
        "gated_beats_ungated_scenario_count": len(beats_ungated),
        "all_scenarios_stress_positive": len(positive) == len(scenarios) and bool(scenarios),
        "all_scenarios_beat_ungated": len(beats_ungated) == len(scenarios) and bool(scenarios),
        "by_latency": by_latency,
    }


def fast58_status(sensitivity_summary: dict[str, Any]) -> tuple[str, list[str]]:
    scenario_count = int(sensitivity_summary["scenario_count"])
    positive_count = int(sensitivity_summary["positive_stress_scenario_count"])
    beats_count = int(sensitivity_summary["gated_beats_ungated_scenario_count"])
    if scenario_count == 0:
        return "FAST58_NO_SCENARIOS", ["do_not_promote_to_strategy"]
    if (
        sensitivity_summary["all_scenarios_stress_positive"]
        and sensitivity_summary["all_scenarios_beat_ungated"]
    ):
        return (
            "FAST58_SENSITIVITY_ALL_POSITIVE_RESEARCH_ONLY",
            ["not_approved_strategy", "prospective_sample_still_required"],
        )
    if positive_count > 0 and beats_count > 0:
        return (
            "FAST58_SENSITIVITY_FRAGILE_RESEARCH_ONLY",
            [
                f"positive_scenarios:{positive_count}/{scenario_count}",
                f"beats_ungated_scenarios:{beats_count}/{scenario_count}",
                "do_not_promote_to_strategy",
            ],
        )
    return (
        "FAST58_SENSITIVITY_FAILED",
        [
            f"positive_scenarios:{positive_count}/{scenario_count}",
            f"beats_ungated_scenarios:{beats_count}/{scenario_count}",
            "do_not_promote_to_strategy",
        ],
    )


def format_cost_label(cost_bps: float) -> str:
    return ("%g" % cost_bps).replace(".", "p")


def _prediction_frame(signals: pd.DataFrame) -> pd.DataFrame:
    return signals.drop(columns=["block", "date", "recent_signal_count"], errors="ignore")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "candidate": report["candidate"],
        "grid": report["grid"],
        "sensitivity_summary": report["sensitivity_summary"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
