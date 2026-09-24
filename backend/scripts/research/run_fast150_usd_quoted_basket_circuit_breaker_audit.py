"""Fast150: circuit-breaker audit for the Fast149 basket candidate.

Fast149 selected a simple exposure overlay for the USD-quoted major basket:
keep the frozen SHORT Donchian-fade entry, but allow at most one open position
per pair. Fast150 keeps that entry and exposure overlay fixed, then tests a
small causal closed-equity drawdown circuit-breaker grid.

The breaker only uses outcomes from trades that have already closed before a
new entry timestamp. It is historical research only and never touches the
trading engine, credentials, ledger, Demo, Testnet or LIVE accounts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import sha256_file  # noqa: E402
from scripts.research.run_fast148_usd_quoted_basket_portfolio_risk_audit import (  # noqa: E402
    CURRENT_SAMPLE_LABEL,
    DEFAULT_CURRENT_INPUT_PATH,
    DEFAULT_PRE_INPUT_PATH,
    ENTRY_POLICY_ID,
    GROUP_ID,
    MANAGEMENT,
    PRE_SAMPLE_LABEL,
    RISK_POLICY_ID,
    SIDE_FILTER,
    TARGET_PAIRS,
    TARGET_POLICY_ID,
    equity_curve,
    filtered_sample,
    summarize_sample,
)
from scripts.research.run_fast149_usd_quoted_basket_position_cap_audit import (  # noqa: E402
    PositionOverlay,
    apply_position_overlay,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast150"
    / "usd-quoted-basket-circuit-breaker-audit"
)
SELECTED_POSITION_OVERLAY = PositionOverlay(
    "max_per_pair_1",
    max_total_open=None,
    max_open_per_pair=1,
)
MIN_PRE_TRADES = 50
MIN_CURRENT_TRADES = 50
MIN_POSITIVE_YEAR_SHARE = 0.65
MAX_PRE_DRAWDOWN_PIPS = 1000.0
MAX_CURRENT_DRAWDOWN_PIPS = 1000.0
MAX_CONCURRENT_TRADES = 4
SAFETY_FLAGS = {
    "research_only": True,
    "order_submission_allowed": False,
    "execution_authorization": "none",
    "orders_sent": False,
    "committed": False,
    "dry_run": True,
    "approved_strategy": False,
}


@dataclass(frozen=True)
class CircuitBreakerPolicy:
    identifier: str
    closed_equity_drawdown_threshold_pips: float | None
    cooldown_days: int


@dataclass(frozen=True)
class CloseEvent:
    exit_timestamp: pd.Timestamp
    sequence: int
    pair: str
    net_stress_pips: float


def circuit_breaker_grid() -> tuple[CircuitBreakerPolicy, ...]:
    policies = [CircuitBreakerPolicy("no_breaker", None, 0)]
    for threshold in (350.0, 500.0, 650.0, 800.0, 1000.0):
        for cooldown_days in (60, 120, 180, 365):
            policies.append(
                CircuitBreakerPolicy(
                    identifier=f"dd{int(threshold)}_cooldown{cooldown_days}d",
                    closed_equity_drawdown_threshold_pips=threshold,
                    cooldown_days=cooldown_days,
                )
            )
    return tuple(policies)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-input-path", type=Path, default=DEFAULT_PRE_INPUT_PATH)
    parser.add_argument("--current-input-path", type=Path, default=DEFAULT_CURRENT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(
        pre_input_path=arguments.pre_input_path,
        current_input_path=arguments.current_input_path,
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def apply_drawdown_circuit_breaker(
    trades: pd.DataFrame,
    policy: CircuitBreakerPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = trades.sort_values(["entry_timestamp", "pair", "candidate_id"]).reset_index(drop=True)
    if policy.closed_equity_drawdown_threshold_pips is None:
        return ordered, empty_skipped_frame(), empty_trigger_frame()

    accepted_indices: list[int] = []
    skipped_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    close_events: list[CloseEvent] = []
    next_close_index = 0
    closed_equity_pips = 0.0
    closed_peak_pips = 0.0
    pause_until = pd.Timestamp("1900-01-01T00:00:00Z")

    def process_closes_until(timestamp: pd.Timestamp | None) -> None:
        nonlocal closed_equity_pips
        nonlocal closed_peak_pips
        nonlocal next_close_index
        nonlocal pause_until

        close_events.sort(key=lambda event: (event.exit_timestamp, event.sequence))
        while next_close_index < len(close_events):
            event = close_events[next_close_index]
            if timestamp is not None and event.exit_timestamp > timestamp:
                break
            closed_equity_pips += event.net_stress_pips
            closed_peak_pips = max(closed_peak_pips, closed_equity_pips)
            closed_drawdown_pips = closed_equity_pips - closed_peak_pips
            if closed_drawdown_pips <= -policy.closed_equity_drawdown_threshold_pips:
                candidate_pause_until = event.exit_timestamp + pd.Timedelta(days=policy.cooldown_days)
                pause_until = max(pause_until, candidate_pause_until)
                trigger_rows.append(
                    {
                        "breaker_id": policy.identifier,
                        "trigger_timestamp": event.exit_timestamp,
                        "closed_trade_pair": event.pair,
                        "closed_trade_stress_pips": event.net_stress_pips,
                        "closed_equity_pips": closed_equity_pips,
                        "closed_peak_pips": closed_peak_pips,
                        "closed_drawdown_pips": closed_drawdown_pips,
                        "pause_until": pause_until,
                    }
                )
            next_close_index += 1

    for row in ordered.itertuples():
        entry_timestamp = pd.Timestamp(row.entry_timestamp)
        process_closes_until(entry_timestamp)
        if entry_timestamp < pause_until:
            skipped_rows.append(
                {
                    "breaker_id": policy.identifier,
                    "skip_reason": "circuit_breaker_pause_active",
                    "pair": row.pair,
                    "entry_timestamp": entry_timestamp,
                    "exit_stress_timestamp": pd.Timestamp(row.exit_stress_timestamp),
                    "net_stress_pips": float(row.net_stress_pips),
                    "pause_until": pause_until,
                }
            )
            continue
        accepted_indices.append(row.Index)
        close_events.append(
            CloseEvent(
                exit_timestamp=pd.Timestamp(row.exit_stress_timestamp),
                sequence=int(row.Index),
                pair=str(row.pair),
                net_stress_pips=float(row.net_stress_pips),
            )
        )

    process_closes_until(None)
    accepted = ordered.loc[accepted_indices].sort_values(["entry_timestamp", "pair"]).reset_index(drop=True)
    skipped = pd.DataFrame(skipped_rows)
    triggers = pd.DataFrame(trigger_rows)
    return accepted, normalize_skipped_frame(skipped), normalize_trigger_frame(triggers)


def empty_skipped_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "breaker_id",
            "skip_reason",
            "pair",
            "entry_timestamp",
            "exit_stress_timestamp",
            "net_stress_pips",
            "pause_until",
        ]
    )


def empty_trigger_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "breaker_id",
            "trigger_timestamp",
            "closed_trade_pair",
            "closed_trade_stress_pips",
            "closed_equity_pips",
            "closed_peak_pips",
            "closed_drawdown_pips",
            "pause_until",
        ]
    )


def normalize_skipped_frame(skipped: pd.DataFrame) -> pd.DataFrame:
    if skipped.empty:
        return empty_skipped_frame()
    for column in ("entry_timestamp", "exit_stress_timestamp", "pause_until"):
        skipped[column] = pd.to_datetime(skipped[column], utc=True)
    return skipped


def normalize_trigger_frame(triggers: pd.DataFrame) -> pd.DataFrame:
    if triggers.empty:
        return empty_trigger_frame()
    for column in ("trigger_timestamp", "pause_until"):
        triggers[column] = pd.to_datetime(triggers[column], utc=True)
    return triggers


def passes_breaker_gate(summary: dict[str, Any], *, min_trades: int, max_drawdown_pips: float) -> bool:
    return bool(
        summary["trade_count"] >= min_trades
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["lower_stress_pips"] is not None
        and summary["lower_stress_pips"] > 0
        and summary["max_drawdown_pips"] is not None
        and abs(summary["max_drawdown_pips"]) <= max_drawdown_pips
        and summary["max_concurrent_trades"] <= MAX_CONCURRENT_TRADES
        and not summary["two_consecutive_negative_years"]
        and summary["positive_year_share"] is not None
        and summary["positive_year_share"] >= MIN_POSITIVE_YEAR_SHARE
    )


def evaluate_breaker(
    pre_overlay_trades: pd.DataFrame,
    current_overlay_trades: pd.DataFrame,
    policy: CircuitBreakerPolicy,
) -> dict[str, Any]:
    pre_accepted, pre_skipped, pre_triggers = apply_drawdown_circuit_breaker(pre_overlay_trades, policy)
    current_accepted, current_skipped, current_triggers = apply_drawdown_circuit_breaker(
        current_overlay_trades,
        policy,
    )
    pre_summary = summarize_sample(pre_accepted)
    current_summary = summarize_sample(current_accepted)
    pre_passed = passes_breaker_gate(
        pre_summary,
        min_trades=MIN_PRE_TRADES,
        max_drawdown_pips=MAX_PRE_DRAWDOWN_PIPS,
    )
    current_passed = passes_breaker_gate(
        current_summary,
        min_trades=MIN_CURRENT_TRADES,
        max_drawdown_pips=MAX_CURRENT_DRAWDOWN_PIPS,
    )
    return {
        "breaker_id": policy.identifier,
        "closed_equity_drawdown_threshold_pips": policy.closed_equity_drawdown_threshold_pips,
        "cooldown_days": policy.cooldown_days,
        "pre": pre_summary,
        "current": current_summary,
        "pre_skipped_count": int(len(pre_skipped)),
        "current_skipped_count": int(len(current_skipped)),
        "pre_trigger_count": int(len(pre_triggers)),
        "current_trigger_count": int(len(current_triggers)),
        "pre_passed_breaker_gate": pre_passed,
        "current_passed_breaker_gate": current_passed,
        "both_passed_breaker_gate": pre_passed and current_passed,
    }


def select_breaker(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed_pre = [evaluation for evaluation in evaluations if evaluation["pre_passed_breaker_gate"]]
    if not passed_pre:
        return None
    return sorted(
        passed_pre,
        key=lambda item: (
            abs(item["pre"]["max_drawdown_pips"]),
            -item["pre"]["lower_stress_pips"],
            item["pre_skipped_count"],
            item["cooldown_days"],
        ),
    )[0]


def flat_metric_rows(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        row = {
            key: value
            for key, value in evaluation.items()
            if key not in {"pre", "current"}
        }
        for prefix in ("pre", "current"):
            summary = evaluation[prefix]
            for key, value in summary.items():
                if key in {"yearly", "pair_summary", "exit_reason_summary"}:
                    continue
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def fast150_status(selected: dict[str, Any] | None) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "circuit_breaker_audit"]
    if selected is None:
        return (
            "FAST150_CIRCUIT_BREAKER_FAILED_PRE_SELECTION",
            reasons + ["no_breaker_passed_pre_gate"],
        )
    if selected["current_passed_breaker_gate"]:
        return (
            "FAST150_CIRCUIT_BREAKER_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"selected_breaker:{selected['breaker_id']}",
                "pre_and_current_breaker_gates_passed",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    return (
        "FAST150_CIRCUIT_BREAKER_FAILED_CURRENT_AUDIT",
        reasons
        + [
            f"selected_breaker_failed_current:{selected['breaker_id']}",
            "fresh_preregistered_prospective_observation_not_allowed_until_current_audit_passes",
        ],
    )


def run(
    *,
    pre_input_path: Path = DEFAULT_PRE_INPUT_PATH,
    current_input_path: Path = DEFAULT_CURRENT_INPUT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    pre_source = pd.read_parquet(pre_input_path)
    current_source = pd.read_parquet(current_input_path)
    pre_trades = filtered_sample(
        pre_source,
        sample_column="sample",
        sample_value=PRE_SAMPLE_LABEL,
    )
    current_trades = filtered_sample(
        current_source,
        sample_column=None,
        sample_value=None,
    )
    pre_overlay_trades, pre_position_skipped = apply_position_overlay(pre_trades, SELECTED_POSITION_OVERLAY)
    current_overlay_trades, current_position_skipped = apply_position_overlay(current_trades, SELECTED_POSITION_OVERLAY)
    evaluations = [
        evaluate_breaker(pre_overlay_trades, current_overlay_trades, policy) for policy in circuit_breaker_grid()
    ]
    selected = select_breaker(evaluations)
    status, status_reasons = fast150_status(selected)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "circuit-breaker-metrics.csv"
    summary_path = output_root / "circuit-breaker-summary.json"
    selected_pre_trades_path = output_root / "selected-breaker-pre-trades.csv"
    selected_current_trades_path = output_root / "selected-breaker-current-trades.csv"
    selected_pre_skipped_path = output_root / "selected-breaker-pre-skipped.csv"
    selected_current_skipped_path = output_root / "selected-breaker-current-skipped.csv"
    selected_pre_triggers_path = output_root / "selected-breaker-pre-triggers.csv"
    selected_current_triggers_path = output_root / "selected-breaker-current-triggers.csv"
    selected_pre_curve_path = output_root / "selected-breaker-pre-equity-curve.csv"
    selected_current_curve_path = output_root / "selected-breaker-current-equity-curve.csv"

    pd.DataFrame(flat_metric_rows(evaluations)).to_csv(metrics_path, index=False)
    selected_payload = None
    if selected is not None:
        selected_policy = next(policy for policy in circuit_breaker_grid() if policy.identifier == selected["breaker_id"])
        selected_pre, selected_pre_skipped, selected_pre_triggers = apply_drawdown_circuit_breaker(
            pre_overlay_trades,
            selected_policy,
        )
        selected_current, selected_current_skipped, selected_current_triggers = apply_drawdown_circuit_breaker(
            current_overlay_trades,
            selected_policy,
        )
        selected_pre.to_csv(selected_pre_trades_path, index=False)
        selected_current.to_csv(selected_current_trades_path, index=False)
        selected_pre_skipped.to_csv(selected_pre_skipped_path, index=False)
        selected_current_skipped.to_csv(selected_current_skipped_path, index=False)
        selected_pre_triggers.to_csv(selected_pre_triggers_path, index=False)
        selected_current_triggers.to_csv(selected_current_triggers_path, index=False)
        equity_curve(selected_pre).to_csv(selected_pre_curve_path, index=False)
        equity_curve(selected_current).to_csv(selected_current_curve_path, index=False)
        selected_payload = selected
    else:
        pd.DataFrame().to_csv(selected_pre_trades_path, index=False)
        pd.DataFrame().to_csv(selected_current_trades_path, index=False)
        empty_skipped_frame().to_csv(selected_pre_skipped_path, index=False)
        empty_skipped_frame().to_csv(selected_current_skipped_path, index=False)
        empty_trigger_frame().to_csv(selected_pre_triggers_path, index=False)
        empty_trigger_frame().to_csv(selected_current_triggers_path, index=False)
        pd.DataFrame().to_csv(selected_pre_curve_path, index=False)
        pd.DataFrame().to_csv(selected_current_curve_path, index=False)

    summary = {
        "selected_position_overlay": {
            "overlay_id": SELECTED_POSITION_OVERLAY.identifier,
            "max_total_open": SELECTED_POSITION_OVERLAY.max_total_open,
            "max_open_per_pair": SELECTED_POSITION_OVERLAY.max_open_per_pair,
            "pre_position_skipped_count": int(len(pre_position_skipped)),
            "current_position_skipped_count": int(len(current_position_skipped)),
        },
        "selected_breaker": selected_payload,
        "evaluations": evaluations,
        "breaker_thresholds": {
            "min_pre_trades": MIN_PRE_TRADES,
            "min_current_trades": MIN_CURRENT_TRADES,
            "min_positive_year_share": MIN_POSITIVE_YEAR_SHARE,
            "max_pre_drawdown_pips": MAX_PRE_DRAWDOWN_PIPS,
            "max_current_drawdown_pips": MAX_CURRENT_DRAWDOWN_PIPS,
            "max_concurrent_trades": MAX_CONCURRENT_TRADES,
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast150_is_historical_circuit_breaker_audit_not_strategy_approval",
            "breaker_was_selected_on_historical_pre_sample",
            "fresh_preregistered_prospective_observation_required_before_shadow_or_testnet",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast150_usd_quoted_basket_circuit_breaker_audit",
        "status": status,
        "status_reasons": status_reasons,
        "target_policy_id": TARGET_POLICY_ID,
        "pre_input_path": str(pre_input_path),
        "pre_input_sha256": sha256_file(pre_input_path),
        "current_input_path": str(current_input_path),
        "current_input_sha256": sha256_file(current_input_path),
        "pre_sample_label": PRE_SAMPLE_LABEL,
        "current_sample_label": CURRENT_SAMPLE_LABEL,
        "policy": {
            "group_id": GROUP_ID,
            "target_pairs": TARGET_PAIRS,
            "entry_policy_id": ENTRY_POLICY_ID,
            "risk_policy_id": RISK_POLICY_ID,
            "side_filter": SIDE_FILTER,
            "management": MANAGEMENT,
            "position_overlay": {
                "overlay_id": SELECTED_POSITION_OVERLAY.identifier,
                "max_total_open": SELECTED_POSITION_OVERLAY.max_total_open,
                "max_open_per_pair": SELECTED_POSITION_OVERLAY.max_open_per_pair,
            },
            "circuit_breaker": {
                "selected_breaker_id": selected["breaker_id"] if selected else None,
                "closed_equity_drawdown_threshold_pips": selected["closed_equity_drawdown_threshold_pips"]
                if selected
                else None,
                "cooldown_days": selected["cooldown_days"] if selected else None,
            },
        },
        "summary": summary,
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "selected_pre_trades": {"path": str(selected_pre_trades_path), "sha256": sha256_file(selected_pre_trades_path)},
            "selected_current_trades": {
                "path": str(selected_current_trades_path),
                "sha256": sha256_file(selected_current_trades_path),
            },
            "selected_pre_skipped": {
                "path": str(selected_pre_skipped_path),
                "sha256": sha256_file(selected_pre_skipped_path),
            },
            "selected_current_skipped": {
                "path": str(selected_current_skipped_path),
                "sha256": sha256_file(selected_current_skipped_path),
            },
            "selected_pre_triggers": {
                "path": str(selected_pre_triggers_path),
                "sha256": sha256_file(selected_pre_triggers_path),
            },
            "selected_current_triggers": {
                "path": str(selected_current_triggers_path),
                "sha256": sha256_file(selected_current_triggers_path),
            },
            "selected_pre_equity_curve": {
                "path": str(selected_pre_curve_path),
                "sha256": sha256_file(selected_pre_curve_path),
            },
            "selected_current_equity_curve": {
                "path": str(selected_current_curve_path),
                "sha256": sha256_file(selected_current_curve_path),
            },
            "circuit_breaker_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    selected = report["summary"]["selected_breaker"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "target_policy_id": report["target_policy_id"],
        "selected_position_overlay_id": report["summary"]["selected_position_overlay"]["overlay_id"],
        "selected_breaker_id": selected["breaker_id"] if selected else None,
        "selected_pre": {
            key: selected["pre"][key]
            for key in (
                "trade_count",
                "total_stress_pips",
                "mean_stress_pips",
                "lower_stress_pips",
                "max_drawdown_pips",
                "max_concurrent_trades",
                "two_consecutive_negative_years",
                "positive_year_share",
            )
        }
        if selected
        else None,
        "selected_current": {
            key: selected["current"][key]
            for key in (
                "trade_count",
                "total_stress_pips",
                "mean_stress_pips",
                "lower_stress_pips",
                "max_drawdown_pips",
                "max_concurrent_trades",
                "two_consecutive_negative_years",
                "positive_year_share",
            )
        }
        if selected
        else None,
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
