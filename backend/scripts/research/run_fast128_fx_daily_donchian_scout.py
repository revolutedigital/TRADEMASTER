"""Fast128: FX daily Donchian breakout time-exit scout on executable M1 bid/ask bars.

This is a larger-move family: D1 channel breakouts with exits after several
days, intended to test whether costs become less destructive on bigger moves.
The first pass deliberately uses time exits only; any positive candidate would
require a separate stop/trailing risk audit before preregistration.

Research-only: writes offline artifacts and never touches the trading engine,
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
from scipy.stats import t

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
    mid_ohlc,
    sha256_file,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast128" / "fx-daily-donchian-scout"

SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
CHANNEL_DAYS = (20, 55, 100)
HOLD_DAYS = (5, 10, 20)
MIN_SELECTION_TRADES = 30
MIN_AUDIT_TRADES = 12
MIN_AUDIT_YEARS = 2
MIN_AUDIT_POSITIVE_YEAR_SHARE = 2 / 3
FINALISTS = 50


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
            identifier=f"d1_donchian_ch{channel}_hold{hold}",
            channel_days=channel,
            hold_days=hold,
        )
        for channel in CHANNEL_DAYS
        for hold in HOLD_DAYS
    )


def daily_mid_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate M1 midpoint bars into full UTC daily bars labelled by close minute."""
    mids = mid_ohlc(frame)
    bars = mids.resample("1D", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    counts = mids["close"].resample("1D", label="left", closed="left").count()
    bars = bars.loc[counts >= 20 * 60].dropna()
    bars.index = bars.index + pd.Timedelta(hours=23, minutes=59)
    return bars


def breakout_signals(frame: pd.DataFrame, *, pair: str, policy: Policy) -> pd.DataFrame:
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
        side = 1 if bool(breakout_up.loc[timestamp]) else -1
        if side == 1:
            distance_pips = (float(bars.loc[timestamp, "close"]) - float(channel_high.loc[timestamp])) / instrument.pip_size
        else:
            distance_pips = (float(channel_low.loc[timestamp]) - float(bars.loc[timestamp, "close"])) / instrument.pip_size
        if not np.isfinite(distance_pips) or distance_pips <= 0:
            continue
        rows.append(
            {
                "signal_timestamp": pd.Timestamp(timestamp),
                "entry_timestamp": pd.Timestamp(timestamp) + pd.Timedelta(minutes=1),
                "side": "LONG" if side == 1 else "SHORT",
                "side_num": side,
                "breakout_distance_pips": float(distance_pips),
            }
        )
    return pd.DataFrame(rows)


def time_exit_trades(
    frame: pd.DataFrame,
    *,
    pair: str,
    policy: Policy,
    signals: pd.DataFrame,
    commission_pips: float,
    slippage_per_side_pips: float,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    entries = pd.DatetimeIndex(signals["entry_timestamp"])
    exits = entries + pd.Timedelta(days=policy.hold_days)
    entry_positions = frame.index.get_indexer(entries)
    exit_positions = frame.index.get_indexer(exits)
    available = (entry_positions >= 0) & (exit_positions >= 0)
    if not available.any():
        return pd.DataFrame()
    selected = signals.loc[available].reset_index(drop=True)
    entry_bars = frame.iloc[entry_positions[available]]
    exit_bars = frame.iloc[exit_positions[available]]
    sides = selected["side_num"].to_numpy(dtype=int)
    pip = Instrument.from_symbol(pair).pip_size
    entry_price = np.where(
        sides == 1,
        entry_bars["ask_open"].to_numpy(dtype=float),
        entry_bars["bid_open"].to_numpy(dtype=float),
    )
    exit_price = np.where(
        sides == 1,
        exit_bars["bid_open"].to_numpy(dtype=float),
        exit_bars["ask_open"].to_numpy(dtype=float),
    )
    raw_pips = (exit_price - entry_price) * sides / pip
    net_pips = raw_pips - commission_pips - 2.0 * slippage_per_side_pips
    return pd.DataFrame(
        {
            "signal_timestamp": selected["signal_timestamp"],
            "entry_timestamp": selected["entry_timestamp"],
            "exit_timestamp": exits[available],
            "side": selected["side"],
            "breakout_distance_pips": selected["breakout_distance_pips"],
            "raw_pips": raw_pips,
            "net_pips": net_pips,
        }
    )


def evaluate_pair_policy(pair: str, pair_data: PairData, policy: Policy) -> list[dict[str, Any]]:
    signals = breakout_signals(pair_data.bars, pair=pair, policy=policy)
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


def lower_confidence_bound(values: pd.Series, alpha: float) -> float:
    sample = values.astype("float64").dropna().to_numpy()
    if len(sample) < 2:
        return float("nan")
    standard_error = float(sample.std(ddof=1) / np.sqrt(len(sample)))
    critical = float(t.ppf(1 - alpha, df=len(sample) - 1))
    return float(sample.mean() - critical * standard_error)


def positive_year_share(sample: pd.DataFrame, column: str) -> tuple[float, int, int]:
    if sample.empty:
        return float("nan"), 0, 0
    by_year = sample.assign(year=sample["entry_timestamp"].dt.year).groupby("year")[column].mean()
    positive = int((by_year > 0).sum())
    total = int(len(by_year))
    return float(positive / total) if total else float("nan"), positive, total


def fast128_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if passed_count:
        return (
            "FAST128_FX_DAILY_DONCHIAN_RESEARCH_CANDIDATE_REQUIRES_RISK_AUDIT",
            reasons
            + [
                f"historical_audit_candidates:{passed_count}",
                "time_exit_only_requires_stop_and_trailing_risk_audit",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    if finalists_count:
        return (
            "FAST128_FX_DAILY_DONCHIAN_FAILED_AUDIT",
            reasons + ["selection_positive_candidates_failed_2024_2026_audit"],
        )
    return (
        "FAST128_FX_DAILY_DONCHIAN_FAILED_SELECTION",
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
        raise ValueError("no daily Donchian trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "candidate_id"])
    trades_path = output_root / "daily-donchian-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)
    selection = summarize_candidates(trades)
    finalists = select_finalists(selection)
    audit = audit_finalists(trades, finalists)
    selection_path = output_root / "selection-metrics.csv"
    audit_path = output_root / "audited-finalists.csv"
    selection.to_csv(selection_path, index=False)
    audit.to_csv(audit_path, index=False)

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    status, status_reasons = fast128_status(len(passed), len(finalists))
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast128_is_historical_scout_not_strategy_approval",
            "donchian_policies_are_selected_on_historical_selection_sample",
            "time_exit_only_without_stop_is_not_execution_risk_approval",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast128_fx_daily_donchian_scout",
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
