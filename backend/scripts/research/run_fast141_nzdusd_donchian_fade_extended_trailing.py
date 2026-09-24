"""Fast141: extended trailing audit for the best NZDUSD D1 Donchian-fade clue.

Fast133 found the strongest current FX candidate with fixed 1R management:
NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r. Fast141 asks the user's
next practical question without touching execution: does a richer breakeven /
trailing grid improve this larger-timeframe candidate after costs and a
2024-2026 audit?

This is historical research only. It writes offline artifacts and never touches
the trading engine, credentials, ledger, Demo, Testnet or LIVE accounts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import (  # noqa: E402
    Management,
    write_json,
)
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    DEFAULT_M1_DIR,
    sha256_file,
)
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    FROZEN_ENTRY_POLICY,
    PAIR,
    RiskPolicy,
    audit_finalists,
    evaluate_risk_policy,
    frozen_entry_signals,
    load_pair_data,
    select_finalists,
    summarize_candidates,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast141"
    / "nzdusd-donchian-fade-extended-trailing"
)
BENCHMARK_RISK_POLICY_ID = "stop2_fixed_1r"


def r_label(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(m1_dir=arguments.m1_dir, output_root=arguments.output_root)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def extended_risk_policy_grid() -> tuple[RiskPolicy, ...]:
    """Return deterministic fixed/trailing policies for the best NZDUSD clue."""
    managements = [
        Management("fixed_1r", target_r=1.0, activation_r=None, trail_r=None),
        Management("fixed_2r", target_r=2.0, activation_r=None, trail_r=None),
        Management("stop_only", target_r=None, activation_r=None, trail_r=None),
    ]
    for activation_r in (0.5, 0.75, 1.0, 1.5):
        for trail_r in (0.5, 0.75, 1.0):
            managements.append(
                Management(
                    f"trail_{r_label(activation_r)}r_{r_label(trail_r)}r",
                    target_r=None,
                    activation_r=activation_r,
                    trail_r=trail_r,
                )
            )

    policies: list[RiskPolicy] = []
    for stop_multiple in (1.0, 1.5, 2.0):
        for management in managements:
            policies.append(
                RiskPolicy(
                    identifier=f"stop{stop_multiple:g}_{management.name}",
                    stop_atr_multiple=stop_multiple,
                    management=management,
                )
            )
    return tuple(policies)


def fast141_status(
    *,
    passed_count: int,
    finalist_count: int,
    benchmark_passed: bool,
    best_trailing_beats_benchmark: bool,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "extended_trailing_audit"]
    if best_trailing_beats_benchmark:
        return (
            "FAST141_EXTENDED_TRAILING_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"historical_audit_candidates:{passed_count}",
                "best_trailing_passed_and_beat_fixed_1r_benchmark",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    if benchmark_passed:
        return (
            "FAST141_EXTENDED_TRAILING_DID_NOT_BEAT_FIXED_1R_BENCHMARK",
            reasons
            + [
                "fixed_1r_benchmark_remains_preferred",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    if passed_count:
        return (
            "FAST141_EXTENDED_TRAILING_HISTORICAL_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"historical_audit_candidates:{passed_count}",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    if finalist_count:
        return (
            "FAST141_EXTENDED_TRAILING_FAILED_AUDIT",
            reasons + ["trailing_variants_failed_2024_2026_audit"],
        )
    return (
        "FAST141_EXTENDED_TRAILING_FAILED_SELECTION",
        reasons + ["no_extended_trailing_variant_positive_after_costs"],
    )


def row_or_none(frame: pd.DataFrame, predicate: pd.Series) -> dict[str, Any] | None:
    sample = frame.loc[predicate]
    if sample.empty:
        return None
    return sample.iloc[0].to_dict()


def best_trailing_row(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    sample = frame.loc[frame["management"].astype(str).str.startswith("trail_")].copy()
    if sample.empty:
        return None
    sample = sample.sort_values(
        [
            "pilot_gate",
            "audit_mean_stress_pips",
            "audit_lower_stress_pips",
            "audit_positive_stress_year_share",
            "audit_trades",
        ],
        ascending=[False, False, False, False, False],
    )
    return sample.iloc[0].to_dict()


def run(*, m1_dir: Path = DEFAULT_M1_DIR, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    pair_data = load_pair_data(m1_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_path = output_root / "pair-quality.json"
    write_json(quality_path, pair_data.quality)

    signals = frozen_entry_signals(pair_data)
    signals_path = output_root / "entry-signals.parquet"
    signals.to_parquet(signals_path, compression="zstd", index=False)

    trade_rows: list[dict[str, Any]] = []
    policies = extended_risk_policy_grid()
    for policy in policies:
        trade_rows.extend(evaluate_risk_policy(pair_data, signals, policy))
        sys.stdout.write(f"{policy.identifier}: accumulated trades={len(trade_rows)}\n")
        sys.stdout.flush()
    if not trade_rows:
        raise ValueError("no extended trailing trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "candidate_id"])
    trades_path = output_root / "extended-trailing-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)

    selection = summarize_candidates(trades)
    finalists = select_finalists(selection)
    audit = audit_finalists(trades, finalists)
    selection_path = output_root / "selection-metrics.csv"
    audit_path = output_root / "audited-finalists.csv"
    selection.to_csv(selection_path, index=False)
    audit.to_csv(audit_path, index=False)

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    benchmark = row_or_none(audit, audit["risk_policy_id"].eq(BENCHMARK_RISK_POLICY_ID)) if not audit.empty else None
    best_trailing = best_trailing_row(audit)
    benchmark_passed = bool(benchmark and benchmark.get("pilot_gate"))
    best_trailing_beats_benchmark = bool(
        best_trailing
        and best_trailing.get("pilot_gate")
        and benchmark
        and float(best_trailing["audit_mean_stress_pips"]) > float(benchmark["audit_mean_stress_pips"])
        and float(best_trailing["audit_lower_stress_pips"]) > 0
    )
    status, status_reasons = fast141_status(
        passed_count=len(passed),
        finalist_count=len(finalists),
        benchmark_passed=benchmark_passed,
        best_trailing_beats_benchmark=best_trailing_beats_benchmark,
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
            "fast141_is_historical_extended_trailing_audit_not_strategy_approval",
            "risk_policies_are_selected_on_historical_selection_sample",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast141_nzdusd_donchian_fade_extended_trailing",
        "status": status,
        "status_reasons": status_reasons,
        "pair": PAIR,
        "entry_policy": asdict(FROZEN_ENTRY_POLICY),
        "benchmark_risk_policy_id": BENCHMARK_RISK_POLICY_ID,
        "risk_policy_count": len(policies),
        "signal_count": int(len(signals)),
        "trade_rows": int(len(trades)),
        "selection_candidates": int(len(selection)),
        "eligible_finalists": int(len(finalists)),
        "audited_finalists": int(len(audit)),
        "passed_candidates": int(len(passed)),
        "benchmark_passed": benchmark_passed,
        "best_trailing_beats_benchmark": best_trailing_beats_benchmark,
        "benchmark_audit": benchmark,
        "best_trailing_audit": best_trailing,
        "artifacts": {
            "signals": {"path": str(signals_path.relative_to(REPO_ROOT)), "sha256": sha256_file(signals_path)},
            "trades": {"path": str(trades_path.relative_to(REPO_ROOT)), "sha256": sha256_file(trades_path)},
            "pair_quality": {"path": str(quality_path.relative_to(REPO_ROOT)), "sha256": sha256_file(quality_path)},
            "selection": {"path": str(selection_path.relative_to(REPO_ROOT)), "sha256": sha256_file(selection_path)},
            "audit": {"path": str(audit_path.relative_to(REPO_ROOT)), "sha256": sha256_file(audit_path)},
        },
    }
    if not finalists.empty:
        report["top_finalists"] = finalists.head(15).to_dict("records")
    if not audit.empty:
        report["top_audit"] = audit.head(15).to_dict("records")
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path.relative_to(REPO_ROOT))})
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "risk_policy_count": report["risk_policy_count"],
        "signal_count": report["signal_count"],
        "trade_rows": report["trade_rows"],
        "eligible_finalists": report["eligible_finalists"],
        "passed_candidates": report["passed_candidates"],
        "benchmark_passed": report["benchmark_passed"],
        "best_trailing_beats_benchmark": report["best_trailing_beats_benchmark"],
        "benchmark_audit": report["benchmark_audit"],
        "best_trailing_audit": report["best_trailing_audit"],
        "top_audit": report.get("top_audit", []),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
