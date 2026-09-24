"""Fast149: position-cap overlay audit for the Fast147 basket candidate.

Fast148 showed the USD-quoted basket has a real historical edge but needs
portfolio controls. Fast149 tests small causal exposure overlays that do not
change the frozen entry rule: skip a new trade if a cap on open positions is
already reached.

This is historical research only. It writes offline artifacts and never touches
the trading engine, credentials, ledger, Demo, Testnet or LIVE accounts.
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
    DEFAULT_CURRENT_INPUT_PATH,
    DEFAULT_PRE_INPUT_PATH,
    CURRENT_SAMPLE_LABEL,
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

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast149"
    / "usd-quoted-basket-position-cap-audit"
)
MIN_PRE_TRADES = 50
MIN_CURRENT_TRADES = 50
MAX_PRE_DRAWDOWN_PIPS = 1500.0
MAX_CURRENT_DRAWDOWN_PIPS = 1500.0
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
class PositionOverlay:
    identifier: str
    max_total_open: int | None
    max_open_per_pair: int | None


def overlay_grid() -> tuple[PositionOverlay, ...]:
    return (
        PositionOverlay("baseline_no_cap", max_total_open=None, max_open_per_pair=None),
        PositionOverlay("max_per_pair_1", max_total_open=None, max_open_per_pair=1),
        PositionOverlay("max_total_2", max_total_open=2, max_open_per_pair=None),
        PositionOverlay("max_total_2_per_pair_1", max_total_open=2, max_open_per_pair=1),
        PositionOverlay("max_total_3", max_total_open=3, max_open_per_pair=None),
        PositionOverlay("max_total_3_per_pair_1", max_total_open=3, max_open_per_pair=1),
        PositionOverlay("max_total_4", max_total_open=4, max_open_per_pair=None),
        PositionOverlay("max_total_4_per_pair_1", max_total_open=4, max_open_per_pair=1),
        PositionOverlay("max_total_6", max_total_open=6, max_open_per_pair=None),
        PositionOverlay("max_total_6_per_pair_1", max_total_open=6, max_open_per_pair=1),
    )


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


def apply_position_overlay(
    trades: pd.DataFrame,
    overlay: PositionOverlay,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_indices: list[int] = []
    skipped_rows: list[dict[str, Any]] = []
    open_positions: list[dict[str, Any]] = []
    ordered = trades.sort_values(["entry_timestamp", "pair", "candidate_id"]).copy()
    for row in ordered.itertuples():
        entry_timestamp = pd.Timestamp(row.entry_timestamp)
        open_positions = [
            position for position in open_positions if pd.Timestamp(position["exit_timestamp"]) > entry_timestamp
        ]
        skip_reason: str | None = None
        if overlay.max_total_open is not None and len(open_positions) >= overlay.max_total_open:
            skip_reason = "max_total_open_reached"
        if (
            skip_reason is None
            and overlay.max_open_per_pair is not None
            and sum(position["pair"] == row.pair for position in open_positions) >= overlay.max_open_per_pair
        ):
            skip_reason = "max_open_per_pair_reached"
        if skip_reason:
            skipped_rows.append(
                {
                    "overlay_id": overlay.identifier,
                    "skip_reason": skip_reason,
                    "pair": row.pair,
                    "entry_timestamp": entry_timestamp,
                    "exit_stress_timestamp": pd.Timestamp(row.exit_stress_timestamp),
                    "net_stress_pips": float(row.net_stress_pips),
                }
            )
            continue
        accepted_indices.append(row.Index)
        open_positions.append({"pair": row.pair, "exit_timestamp": pd.Timestamp(row.exit_stress_timestamp)})

    accepted = ordered.loc[accepted_indices].sort_values(["entry_timestamp", "pair"]).reset_index(drop=True)
    skipped = pd.DataFrame(skipped_rows)
    if not skipped.empty:
        skipped["entry_timestamp"] = pd.to_datetime(skipped["entry_timestamp"], utc=True)
        skipped["exit_stress_timestamp"] = pd.to_datetime(skipped["exit_stress_timestamp"], utc=True)
    return accepted, skipped


def passes_operability_gate(summary: dict[str, Any], *, min_trades: int) -> bool:
    return bool(
        summary["trade_count"] >= min_trades
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["lower_stress_pips"] is not None
        and summary["lower_stress_pips"] > 0
        and summary["max_drawdown_pips"] is not None
        and abs(summary["max_drawdown_pips"]) <= MAX_PRE_DRAWDOWN_PIPS
        and summary["max_concurrent_trades"] <= MAX_CONCURRENT_TRADES
    )


def evaluate_overlay(
    pre_trades: pd.DataFrame,
    current_trades: pd.DataFrame,
    overlay: PositionOverlay,
) -> dict[str, Any]:
    pre_accepted, pre_skipped = apply_position_overlay(pre_trades, overlay)
    current_accepted, current_skipped = apply_position_overlay(current_trades, overlay)
    pre_summary = summarize_sample(pre_accepted)
    current_summary = summarize_sample(current_accepted)
    pre_passed = passes_operability_gate(pre_summary, min_trades=MIN_PRE_TRADES)
    current_passed = (
        current_summary["trade_count"] >= MIN_CURRENT_TRADES
        and current_summary["mean_stress_pips"] is not None
        and current_summary["mean_stress_pips"] > 0
        and current_summary["lower_stress_pips"] is not None
        and current_summary["lower_stress_pips"] > 0
        and current_summary["max_drawdown_pips"] is not None
        and abs(current_summary["max_drawdown_pips"]) <= MAX_CURRENT_DRAWDOWN_PIPS
        and current_summary["max_concurrent_trades"] <= MAX_CONCURRENT_TRADES
    )
    return {
        "overlay_id": overlay.identifier,
        "max_total_open": overlay.max_total_open,
        "max_open_per_pair": overlay.max_open_per_pair,
        "pre": pre_summary,
        "current": current_summary,
        "pre_skipped_count": int(len(pre_skipped)),
        "current_skipped_count": int(len(current_skipped)),
        "pre_passed_operability_gate": pre_passed,
        "current_passed_operability_gate": current_passed,
        "both_passed_operability_gate": pre_passed and current_passed,
    }


def select_overlay(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed_pre = [evaluation for evaluation in evaluations if evaluation["pre_passed_operability_gate"]]
    if not passed_pre:
        return None
    return sorted(
        passed_pre,
        key=lambda item: (
            abs(item["pre"]["max_drawdown_pips"]),
            -item["pre"]["lower_stress_pips"],
            item["pre"]["max_concurrent_trades"],
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


def fast149_status(selected: dict[str, Any] | None) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "position_cap_overlay_audit"]
    if selected is None:
        return (
            "FAST149_POSITION_CAP_OVERLAY_FAILED_PRE_SELECTION",
            reasons + ["no_overlay_passed_pre_operability_gate"],
        )
    if selected["current_passed_operability_gate"] and selected["pre"]["two_consecutive_negative_years"]:
        return (
            "FAST149_POSITION_CAP_REDUCES_DRAWDOWN_BUT_REQUIRES_CIRCUIT_BREAKER",
            reasons
            + [
                f"selected_overlay:{selected['overlay_id']}",
                "pre_and_current_operability_gates_passed",
                "pre_still_has_two_consecutive_negative_years",
                "fresh_preregistered_prospective_observation_remains_required",
            ],
        )
    if selected["current_passed_operability_gate"]:
        return (
            "FAST149_POSITION_CAP_OVERLAY_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"selected_overlay:{selected['overlay_id']}",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    return (
        "FAST149_POSITION_CAP_OVERLAY_FAILED_CURRENT_AUDIT",
        reasons + [f"selected_overlay_failed_current:{selected['overlay_id']}"],
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
    evaluations = [evaluate_overlay(pre_trades, current_trades, overlay) for overlay in overlay_grid()]
    selected = select_overlay(evaluations)
    status, status_reasons = fast149_status(selected)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "position-cap-metrics.csv"
    summary_path = output_root / "position-cap-summary.json"
    selected_pre_trades_path = output_root / "selected-overlay-pre-trades.csv"
    selected_current_trades_path = output_root / "selected-overlay-current-trades.csv"
    selected_pre_curve_path = output_root / "selected-overlay-pre-equity-curve.csv"
    selected_current_curve_path = output_root / "selected-overlay-current-equity-curve.csv"

    pd.DataFrame(flat_metric_rows(evaluations)).to_csv(metrics_path, index=False)
    selected_payload = None
    if selected is not None:
        selected_overlay = next(overlay for overlay in overlay_grid() if overlay.identifier == selected["overlay_id"])
        selected_pre, _pre_skipped = apply_position_overlay(pre_trades, selected_overlay)
        selected_current, _current_skipped = apply_position_overlay(current_trades, selected_overlay)
        selected_pre.to_csv(selected_pre_trades_path, index=False)
        selected_current.to_csv(selected_current_trades_path, index=False)
        equity_curve(selected_pre).to_csv(selected_pre_curve_path, index=False)
        equity_curve(selected_current).to_csv(selected_current_curve_path, index=False)
        selected_payload = selected
    else:
        pd.DataFrame().to_csv(selected_pre_trades_path, index=False)
        pd.DataFrame().to_csv(selected_current_trades_path, index=False)
        pd.DataFrame().to_csv(selected_pre_curve_path, index=False)
        pd.DataFrame().to_csv(selected_current_curve_path, index=False)

    summary = {
        "selected_overlay": selected_payload,
        "evaluations": evaluations,
        "operability_thresholds": {
            "min_pre_trades": MIN_PRE_TRADES,
            "min_current_trades": MIN_CURRENT_TRADES,
            "max_pre_drawdown_pips": MAX_PRE_DRAWDOWN_PIPS,
            "max_current_drawdown_pips": MAX_CURRENT_DRAWDOWN_PIPS,
            "max_concurrent_trades": MAX_CONCURRENT_TRADES,
        },
    }
    write_json(summary_path, summary)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast149_is_historical_position_cap_audit_not_strategy_approval",
            "overlay_was_selected_on_historical_pre_sample",
            "circuit_breaker_still_required_for_bad_year_clusters",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast149_usd_quoted_basket_position_cap_audit",
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
        },
        "summary": summary,
        "artifacts": {
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "selected_pre_trades": {"path": str(selected_pre_trades_path), "sha256": sha256_file(selected_pre_trades_path)},
            "selected_current_trades": {
                "path": str(selected_current_trades_path),
                "sha256": sha256_file(selected_current_trades_path),
            },
            "selected_pre_equity_curve": {
                "path": str(selected_pre_curve_path),
                "sha256": sha256_file(selected_pre_curve_path),
            },
            "selected_current_equity_curve": {
                "path": str(selected_current_curve_path),
                "sha256": sha256_file(selected_current_curve_path),
            },
            "position_cap_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    selected = report["summary"]["selected_overlay"]
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "target_policy_id": report["target_policy_id"],
        "selected_overlay_id": selected["overlay_id"] if selected else None,
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
            )
        }
        if selected
        else None,
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
