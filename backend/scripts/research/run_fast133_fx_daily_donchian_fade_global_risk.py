"""Fast133: global risk audit for FX daily Donchian-fade policies.

Fast130 audited hard-stop risk only for the single Fast129 time-exit finalist.
Fast133 asks a wider question: can stop/target management rescue any daily
Donchian-fade policy across the full pair/channel/hold grid? This is historical
research only: it writes offline artifacts and never touches the trading engine,
credentials, ledger, Demo, Testnet or LIVE accounts.
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

from app.fx.sim.costs import FUSION_ZERO, STRESS  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import (  # noqa: E402
    plain_json,
    write_json,
)
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    AUDIT_END,
    DEFAULT_M1_DIR,
    PAIRS,
    PairData,
    load_market,
    sha256_file,
)
from scripts.research.run_fast129_fx_daily_donchian_fade_scout import (  # noqa: E402
    Policy as FadePolicy,
    fade_signals,
    policy_grid as entry_policy_grid,
)
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    RiskPolicy,
    daily_atr_pips,
    lower_confidence_bound,
    positive_year_share,
    risk_policy_grid,
    simulate_gappy_trade,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast133" / "fx-daily-donchian-fade-global-risk"
)

SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
FINALISTS = 100
MIN_SELECTION_TRADES = 30
MIN_AUDIT_TRADES = 12
MIN_AUDIT_YEARS = 2
MIN_AUDIT_POSITIVE_YEAR_SHARE = 2 / 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(m1_dir=arguments.m1_dir, output_root=arguments.output_root)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def enrich_signals_with_daily_atr(pair: str, pair_data: PairData, entry_policy: FadePolicy) -> pd.DataFrame:
    signals = fade_signals(pair_data.bars, pair=pair, policy=entry_policy)
    if signals.empty:
        return signals
    atr = daily_atr_pips(pair_data.bars, pair).rename("daily_atr_pips")
    enriched = signals.merge(
        atr,
        left_on="signal_timestamp",
        right_index=True,
        how="left",
    )
    enriched = enriched.loc[enriched["daily_atr_pips"].notna()].copy()
    return enriched.sort_values("entry_timestamp", ignore_index=True)


def evaluate_pair_policy(
    pair: str,
    pair_data: PairData,
    entry_policy: FadePolicy,
    risk_policy: RiskPolicy,
    signals: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hold_minutes = entry_policy.hold_days * 24 * 60
    for signal in signals.to_dict("records"):
        risk_pips = risk_policy.stop_atr_multiple * float(signal["daily_atr_pips"])
        side = int(signal["side_num"])
        base = simulate_gappy_trade(
            pair_data.bars,
            pair=pair,
            entry_timestamp=pd.Timestamp(signal["entry_timestamp"]),
            side=side,
            risk_pips=risk_pips,
            hold_minutes=hold_minutes,
            management=risk_policy.management,
            commission=pair_data.base_commission_pips,
            slippage_per_side_pips=FUSION_ZERO.slippage_pips,
        )
        if base is None:
            continue
        stress = simulate_gappy_trade(
            pair_data.stress_bars,
            pair=pair,
            entry_timestamp=pd.Timestamp(signal["entry_timestamp"]),
            side=side,
            risk_pips=risk_pips,
            hold_minutes=hold_minutes,
            management=risk_policy.management,
            commission=pair_data.stress_commission_pips,
            slippage_per_side_pips=STRESS.slippage_pips,
        )
        if stress is None:
            continue
        rows.append(
            {
                "candidate_id": f"{pair}:{entry_policy.identifier}:{risk_policy.identifier}",
                "pair": pair,
                "entry_policy_id": entry_policy.identifier,
                "risk_policy_id": risk_policy.identifier,
                "channel_days": entry_policy.channel_days,
                "hold_days": entry_policy.hold_days,
                "stop_atr_multiple": risk_policy.stop_atr_multiple,
                "management": risk_policy.management.name,
                "signal_timestamp": signal["signal_timestamp"],
                "entry_timestamp": signal["entry_timestamp"],
                "side": signal["side"],
                "breakout_direction": signal["breakout_direction"],
                "daily_atr_pips": float(signal["daily_atr_pips"]),
                "risk_pips": risk_pips,
                "breakout_distance_pips": float(signal["breakout_distance_pips"]),
                "raw_base_pips": base.raw_pips,
                "net_base_pips": base.net_pips,
                "exit_base_timestamp": base.exit_timestamp,
                "exit_base_reason": base.exit_reason,
                "raw_stress_pips": stress.raw_pips,
                "net_stress_pips": stress.net_pips,
                "exit_stress_timestamp": stress.exit_timestamp,
                "exit_stress_reason": stress.exit_reason,
            }
        )
    return rows


def summarize_candidates(trades: pd.DataFrame) -> pd.DataFrame:
    selection = trades.loc[trades["entry_timestamp"] < SELECTION_END].copy()
    grouped = selection.groupby("candidate_id", sort=True)
    summary = grouped.agg(
        pair=("pair", "first"),
        entry_policy_id=("entry_policy_id", "first"),
        risk_policy_id=("risk_policy_id", "first"),
        channel_days=("channel_days", "first"),
        hold_days=("hold_days", "first"),
        stop_atr_multiple=("stop_atr_multiple", "first"),
        management=("management", "first"),
        selection_trades=("candidate_id", "size"),
        selection_mean_base_pips=("net_base_pips", "mean"),
        selection_mean_stress_pips=("net_stress_pips", "mean"),
        selection_total_stress_pips=("net_stress_pips", "sum"),
        selection_median_stress_pips=("net_stress_pips", "median"),
        selection_mean_risk_pips=("risk_pips", "mean"),
        selection_mean_breakout_distance_pips=("breakout_distance_pips", "mean"),
    )
    rows: list[dict[str, Any]] = []
    for candidate_id, sample in selection.groupby("candidate_id", sort=True):
        share, positive, years = positive_year_share(sample, "net_stress_pips")
        rows.append(
            {
                "candidate_id": candidate_id,
                "selection_lower_stress_pips": lower_confidence_bound(sample["net_stress_pips"], 0.05),
                "selection_positive_stress_year_share": share,
                "selection_positive_stress_years": positive,
                "selection_years": years,
            }
        )
    return summary.reset_index().merge(pd.DataFrame(rows), on="candidate_id", how="left")


def select_finalists(selection: pd.DataFrame) -> pd.DataFrame:
    eligible = selection.loc[
        (selection["selection_trades"] >= MIN_SELECTION_TRADES)
        & (selection["selection_mean_base_pips"] > 0)
        & (selection["selection_mean_stress_pips"] > 0)
    ].copy()
    return eligible.sort_values(
        [
            "selection_mean_stress_pips",
            "selection_total_stress_pips",
            "selection_positive_stress_year_share",
            "selection_trades",
        ],
        ascending=[False, False, False, False],
    ).head(FINALISTS)


def audit_finalists(trades: pd.DataFrame, finalists: pd.DataFrame) -> pd.DataFrame:
    audit = trades.loc[
        (trades["entry_timestamp"] >= SELECTION_END) & (trades["entry_timestamp"] < AUDIT_END)
    ].copy()
    alpha = 0.05 / max(len(finalists), 1)
    rows: list[dict[str, Any]] = []
    for rank, finalist in enumerate(finalists.itertuples(index=False), start=1):
        sample = audit.loc[audit["candidate_id"] == finalist.candidate_id]
        stress_share, stress_positive_years, stress_years = positive_year_share(sample, "net_stress_pips")
        mean_base = float(sample["net_base_pips"].mean()) if len(sample) else float("nan")
        mean_stress = float(sample["net_stress_pips"].mean()) if len(sample) else float("nan")
        stress_lcb = lower_confidence_bound(sample["net_stress_pips"], alpha)
        passed = (
            len(sample) >= MIN_AUDIT_TRADES
            and stress_years >= MIN_AUDIT_YEARS
            and mean_base > 0
            and mean_stress > 0
            and stress_lcb > 0
            and stress_share >= MIN_AUDIT_POSITIVE_YEAR_SHARE
        )
        rows.append(
            {
                "rank": rank,
                **finalist._asdict(),
                "audit_trades": int(len(sample)),
                "audit_mean_base_pips": mean_base,
                "audit_mean_stress_pips": mean_stress,
                "audit_lower_stress_pips": stress_lcb,
                "audit_positive_stress_year_share": stress_share,
                "audit_positive_stress_years": stress_positive_years,
                "audit_years": stress_years,
                "bonferroni_alpha": alpha,
                "pilot_gate": bool(passed),
            }
        )
    return pd.DataFrame(rows)


def fast133_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if passed_count:
        return (
            "FAST133_FX_DAILY_DONCHIAN_FADE_GLOBAL_RISK_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"historical_audit_candidates:{passed_count}",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    if finalists_count:
        return (
            "FAST133_FX_DAILY_DONCHIAN_FADE_GLOBAL_RISK_FAILED_AUDIT",
            reasons + ["selection_positive_candidates_failed_2024_2026_audit"],
        )
    return (
        "FAST133_FX_DAILY_DONCHIAN_FADE_GLOBAL_RISK_FAILED_SELECTION",
        reasons + ["no_selection_candidate_positive_after_costs"],
    )


def run(*, m1_dir: Path = DEFAULT_M1_DIR, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    entry_policies = entry_policy_grid()
    risk_policies = risk_policy_grid()
    market = load_market(m1_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_path = output_root / "pair-quality.json"
    write_json(quality_path, [market[pair].quality for pair in PAIRS])

    trade_rows: list[dict[str, Any]] = []
    signal_counts: dict[str, int] = {}
    for pair in PAIRS:
        pair_data = market[pair]
        signal_cache: dict[str, pd.DataFrame] = {}
        for entry_policy in entry_policies:
            signals = enrich_signals_with_daily_atr(pair, pair_data, entry_policy)
            signal_cache[entry_policy.identifier] = signals
            signal_counts[f"{pair}:{entry_policy.identifier}"] = int(len(signals))
            for risk_policy in risk_policies:
                trade_rows.extend(evaluate_pair_policy(pair, pair_data, entry_policy, risk_policy, signals))
        sys.stdout.write(f"{pair}: accumulated trades={len(trade_rows)}\n")
        sys.stdout.flush()
    if not trade_rows:
        raise ValueError("no global daily Donchian fade risk trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "candidate_id"])
    trades_path = output_root / "daily-donchian-fade-global-risk-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)
    selection = summarize_candidates(trades)
    finalists = select_finalists(selection)
    audit = audit_finalists(trades, finalists)
    selection_path = output_root / "selection-metrics.csv"
    audit_path = output_root / "audited-finalists.csv"
    signal_counts_path = output_root / "signal-counts.json"
    write_json(signal_counts_path, signal_counts)
    selection.to_csv(selection_path, index=False)
    audit.to_csv(audit_path, index=False)

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    status, status_reasons = fast133_status(len(passed), len(finalists))
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast133_is_historical_global_risk_audit_not_strategy_approval",
            "daily_donchian_fade_risk_policies_are_selected_on_historical_selection_sample",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast133_fx_daily_donchian_fade_global_risk",
        "status": status,
        "status_reasons": status_reasons,
        "m1_dir": str(m1_dir.relative_to(REPO_ROOT)) if m1_dir.is_relative_to(REPO_ROOT) else str(m1_dir),
        "pairs": list(PAIRS),
        "entry_policy_count_per_pair": len(entry_policies),
        "risk_policy_count": len(risk_policies),
        "candidate_count": len(PAIRS) * len(entry_policies) * len(risk_policies),
        "selection_start": trades["entry_timestamp"].min().isoformat(),
        "selection_end": SELECTION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "trade_rows": int(len(trades)),
        "signal_count": int(sum(signal_counts.values())),
        "selection_candidates": int(len(selection)),
        "eligible_finalists": int(len(finalists)),
        "audited_finalists": int(len(audit)),
        "passed_candidates": int(len(passed)),
        "pair_trade_counts": trades.groupby("pair").size().to_dict(),
        "eligible_pair_counts": finalists.groupby("pair").size().to_dict() if not finalists.empty else {},
        "passed_pair_counts": passed.groupby("pair").size().to_dict() if not passed.empty else {},
        "costs": {
            "base": asdict(FUSION_ZERO),
            "stress": asdict(STRESS),
            "risk_basis": "daily ATR(14) shifted one D1 bar before signal",
        },
        "artifacts": {
            "trades": {"path": str(trades_path.relative_to(REPO_ROOT)), "sha256": sha256_file(trades_path)},
            "pair_quality": {"path": str(quality_path.relative_to(REPO_ROOT)), "sha256": sha256_file(quality_path)},
            "signal_counts": {"path": str(signal_counts_path.relative_to(REPO_ROOT)), "sha256": sha256_file(signal_counts_path)},
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
    return {**plain_json(report), "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "signal_count": report["signal_count"],
        "trade_rows": report["trade_rows"],
        "candidate_count": report["candidate_count"],
        "eligible_finalists": report["eligible_finalists"],
        "passed_candidates": report["passed_candidates"],
        "eligible_pair_counts": report["eligible_pair_counts"],
        "passed_pair_counts": report["passed_pair_counts"],
        "top_audit": report.get("top_audit", []),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
