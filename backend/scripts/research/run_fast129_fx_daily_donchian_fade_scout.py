"""Fast129: FX daily Donchian breakout fade scout on executable M1 bid/ask bars.

Fast128 tested following daily Donchian breakouts. Fast129 tests the opposite
family: fade the same daily breakouts and exit by time after several days. This
is research-only, writes offline artifacts, and never touches trading engine,
credentials, ledger, Demo, Testnet or LIVE accounts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.fx.instruments import Instrument  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    AUDIT_END,
    DEFAULT_M1_DIR,
    PAIRS,
    PairData,
    load_market,
    sha256_file,
)
from scripts.research.run_fast128_fx_daily_donchian_scout import (  # noqa: E402
    FINALISTS,
    MIN_AUDIT_TRADES,
    MIN_AUDIT_YEARS,
    MIN_AUDIT_POSITIVE_YEAR_SHARE,
    MIN_SELECTION_TRADES,
    SELECTION_END,
    audit_finalists,
    daily_mid_bars,
    lower_confidence_bound,
    positive_year_share,
    select_finalists,
    time_exit_trades,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast129" / "fx-daily-donchian-fade-scout"

CHANNEL_DAYS = (20, 55, 100)
HOLD_DAYS = (5, 10, 20)


@dataclass(frozen=True)
class Policy:
    identifier: str
    channel_days: int
    hold_days: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(m1_dir=arguments.m1_dir, output_root=arguments.output_root)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def policy_grid() -> tuple[Policy, ...]:
    return tuple(
        Policy(
            identifier=f"d1_donchian_fade_ch{channel}_hold{hold}",
            channel_days=channel,
            hold_days=hold,
        )
        for channel in CHANNEL_DAYS
        for hold in HOLD_DAYS
    )


def fade_signals(frame: pd.DataFrame, *, pair: str, policy: Policy) -> pd.DataFrame:
    """Generate contrarian trades after D1 channel breakouts."""
    instrument = Instrument.from_symbol(pair)
    bars = daily_mid_bars(frame)
    if len(bars) < policy.channel_days + 50:
        return pd.DataFrame()
    channel_high = bars["high"].rolling(policy.channel_days, min_periods=policy.channel_days).max().shift(1)
    channel_low = bars["low"].rolling(policy.channel_days, min_periods=policy.channel_days).min().shift(1)
    breakout_up = bars["close"] > channel_high
    breakout_down = bars["close"] < channel_low
    rows = []
    for timestamp in bars.index[breakout_up | breakout_down]:
        breakout_side = 1 if bool(breakout_up.loc[timestamp]) else -1
        fade_side = -breakout_side
        if breakout_side == 1:
            distance_pips = (float(bars.loc[timestamp, "close"]) - float(channel_high.loc[timestamp])) / instrument.pip_size
        else:
            distance_pips = (float(channel_low.loc[timestamp]) - float(bars.loc[timestamp, "close"])) / instrument.pip_size
        if not np.isfinite(distance_pips) or distance_pips <= 0:
            continue
        rows.append(
            {
                "signal_timestamp": pd.Timestamp(timestamp),
                "entry_timestamp": pd.Timestamp(timestamp) + pd.Timedelta(minutes=1),
                "side": "LONG" if fade_side == 1 else "SHORT",
                "side_num": fade_side,
                "breakout_direction": "UP" if breakout_side == 1 else "DOWN",
                "breakout_distance_pips": float(distance_pips),
            }
        )
    return pd.DataFrame(rows)


def evaluate_pair_policy(pair: str, pair_data: PairData, policy: Policy) -> list[dict[str, Any]]:
    signals = fade_signals(pair_data.bars, pair=pair, policy=policy)
    base = time_exit_trades(
        pair_data.bars,
        pair=pair,
        policy=policy,
        signals=signals,
        commission_pips=pair_data.base_commission_pips,
        slippage_per_side_pips=0.0,
    )
    if base.empty:
        return []
    stress = time_exit_trades(
        pair_data.stress_bars,
        pair=pair,
        policy=policy,
        signals=signals,
        commission_pips=pair_data.stress_commission_pips,
        slippage_per_side_pips=0.2,
    )
    if stress.empty:
        return []
    joined = base.merge(
        stress,
        on=["signal_timestamp", "entry_timestamp", "exit_timestamp", "side", "breakout_distance_pips"],
        suffixes=("_base", "_stress"),
    )
    rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        rows.append(
            {
                "candidate_id": f"{pair}:{policy.identifier}",
                "pair": pair,
                "policy_id": policy.identifier,
                "channel_days": policy.channel_days,
                "hold_days": policy.hold_days,
                "signal_timestamp": row.signal_timestamp,
                "entry_timestamp": row.entry_timestamp,
                "exit_timestamp": row.exit_timestamp,
                "side": row.side,
                "breakout_distance_pips": float(row.breakout_distance_pips),
                "raw_base_pips": float(row.raw_pips_base),
                "net_base_pips": float(row.net_pips_base),
                "raw_stress_pips": float(row.raw_pips_stress),
                "net_stress_pips": float(row.net_pips_stress),
                "base_commission_pips": pair_data.base_commission_pips,
                "stress_commission_pips": pair_data.stress_commission_pips,
            }
        )
    return rows


def summarize_candidates(trades: pd.DataFrame) -> pd.DataFrame:
    selection = trades.loc[trades["entry_timestamp"] < SELECTION_END].copy()
    grouped = selection.groupby("candidate_id", sort=True)
    summary = grouped.agg(
        pair=("pair", "first"),
        policy_id=("policy_id", "first"),
        channel_days=("channel_days", "first"),
        hold_days=("hold_days", "first"),
        selection_trades=("candidate_id", "size"),
        selection_mean_base_pips=("net_base_pips", "mean"),
        selection_mean_stress_pips=("net_stress_pips", "mean"),
        selection_total_stress_pips=("net_stress_pips", "sum"),
        selection_median_stress_pips=("net_stress_pips", "median"),
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


def fast129_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if passed_count:
        return (
            "FAST129_FX_DAILY_DONCHIAN_FADE_RESEARCH_CANDIDATE_REQUIRES_RISK_AUDIT",
            reasons
            + [
                f"historical_audit_candidates:{passed_count}",
                "time_exit_only_requires_stop_and_trailing_risk_audit",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    if finalists_count:
        return (
            "FAST129_FX_DAILY_DONCHIAN_FADE_FAILED_AUDIT",
            reasons + ["selection_positive_candidates_failed_2024_2026_audit"],
        )
    return (
        "FAST129_FX_DAILY_DONCHIAN_FADE_FAILED_SELECTION",
        reasons + ["no_selection_candidate_positive_after_costs"],
    )


def run(*, m1_dir: Path = DEFAULT_M1_DIR, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    policies = policy_grid()
    market = load_market(m1_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_path = output_root / "pair-quality.json"
    write_json(quality_path, [market[pair].quality for pair in PAIRS])

    trade_rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        pair_data = market[pair]
        for policy in policies:
            trade_rows.extend(evaluate_pair_policy(pair, pair_data, policy))
        sys.stdout.write(f"{pair}: accumulated trades={len(trade_rows)}\n")
        sys.stdout.flush()
    if not trade_rows:
        raise ValueError("no daily Donchian fade trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "candidate_id"])
    trades_path = output_root / "daily-donchian-fade-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)
    selection = summarize_candidates(trades)
    finalists = select_finalists(selection)
    audit = audit_finalists(trades, finalists)
    selection_path = output_root / "selection-metrics.csv"
    audit_path = output_root / "audited-finalists.csv"
    selection.to_csv(selection_path, index=False)
    audit.to_csv(audit_path, index=False)

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    status, status_reasons = fast129_status(len(passed), len(finalists))
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast129_is_historical_scout_not_strategy_approval",
            "donchian_fade_policies_are_selected_on_historical_selection_sample",
            "time_exit_only_without_stop_is_not_execution_risk_approval",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast129_fx_daily_donchian_fade_scout",
        "status": status,
        "status_reasons": status_reasons,
        "m1_dir": str(m1_dir.relative_to(REPO_ROOT)),
        "pairs": list(PAIRS),
        "policy_count_per_pair": len(policies),
        "candidate_count": len(PAIRS) * len(policies),
        "selection_start": trades["entry_timestamp"].min().isoformat(),
        "selection_end": SELECTION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "trade_rows": int(len(trades)),
        "selection_candidates": int(len(selection)),
        "eligible_finalists": int(len(finalists)),
        "audited_finalists": int(len(audit)),
        "passed_candidates": int(len(passed)),
        "costs": {
            "base_slippage_per_side_pips": 0.0,
            "stress_slippage_per_side_pips": 0.2,
            "note": "time-exit first pass; stop/trailing risk audit required for any positive candidate",
        },
        "artifacts": {
            "trades": {"path": str(trades_path.relative_to(REPO_ROOT)), "sha256": sha256_file(trades_path)},
            "pair_quality": {"path": str(quality_path.relative_to(REPO_ROOT)), "sha256": sha256_file(quality_path)},
            "selection": {"path": str(selection_path.relative_to(REPO_ROOT)), "sha256": sha256_file(selection_path)},
            "audit": {"path": str(audit_path.relative_to(REPO_ROOT)), "sha256": sha256_file(audit_path)},
        },
    }
    if not finalists.empty:
        report["top_finalists"] = finalists.head(10).to_dict("records")
    if not audit.empty:
        report["top_audit"] = audit.head(10).to_dict("records")
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
        "trade_rows": report["trade_rows"],
        "candidate_count": report["candidate_count"],
        "eligible_finalists": report["eligible_finalists"],
        "passed_candidates": report["passed_candidates"],
        "top_audit": report.get("top_audit", []),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
