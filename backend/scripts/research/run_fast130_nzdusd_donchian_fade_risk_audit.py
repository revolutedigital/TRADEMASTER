"""Fast130: risk audit for the Fast129 NZDUSD daily Donchian fade clue.

Fast129 found one historical audit candidate, but it used a time exit without a
hard stop. Fast130 freezes that entry family and audits basic stop/target/trail
management with daily ATR risk. Research-only: no trading engine, credentials,
ledger, Demo, Testnet or LIVE account is touched.
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

from app.fx.instruments import ConversionRates, Instrument  # noqa: E402
from app.fx.sim.costs import FUSION_ZERO, STRESS  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import (  # noqa: E402
    Management,
    ScenarioResult,
    write_json,
)
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    AUDIT_END,
    DEFAULT_M1_DIR,
    MAJOR_RATE_PAIRS,
    PairData,
    commission_pips,
    load_pair_bars,
    mid_ohlc,
    sha256_file,
    widen_quotes,
)
from scripts.research.run_fast129_fx_daily_donchian_fade_scout import (  # noqa: E402
    Policy as FadePolicy,
    fade_signals,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast130" / "nzdusd-donchian-fade-risk-audit"
)
PAIR = "NZDUSD"
FROZEN_ENTRY_POLICY = FadePolicy("d1_donchian_fade_ch100_hold20", channel_days=100, hold_days=20)
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
MIN_SELECTION_TRADES = 30
MIN_AUDIT_TRADES = 12
MIN_AUDIT_YEARS = 2
MIN_AUDIT_POSITIVE_YEAR_SHARE = 2 / 3


@dataclass(frozen=True)
class RiskPolicy:
    identifier: str
    stop_atr_multiple: float
    management: Management


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(m1_dir=arguments.m1_dir, output_root=arguments.output_root)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def risk_policy_grid() -> tuple[RiskPolicy, ...]:
    managements = (
        Management("stop_only", target_r=None, activation_r=None, trail_r=None),
        Management("fixed_1r", target_r=1.0, activation_r=None, trail_r=None),
        Management("fixed_2r", target_r=2.0, activation_r=None, trail_r=None),
        Management("trail_1r_1r", target_r=None, activation_r=1.0, trail_r=1.0),
    )
    policies: list[RiskPolicy] = []
    for stop_multiple in (1.0, 1.5, 2.0):
        for management in managements:
            identifier = f"stop{stop_multiple:g}_{management.name}"
            policies.append(RiskPolicy(identifier, stop_multiple, management))
    return tuple(policies)


def median_rates_from_dir(m1_dir: Path) -> ConversionRates:
    frames = {pair: load_pair_bars(m1_dir, pair) for pair in MAJOR_RATE_PAIRS}
    mids = {pair: float(np.median(mid_ohlc(frame)["close"].to_numpy())) for pair, frame in frames.items()}
    return ConversionRates(mids)


def load_pair_data(m1_dir: Path) -> PairData:
    frame = load_pair_bars(m1_dir, PAIR)
    median_price = float(np.median(mid_ohlc(frame)["close"].to_numpy()))
    rates = median_rates_from_dir(m1_dir)
    return PairData(
        bars=frame,
        stress_bars=widen_quotes(frame, STRESS.spread_multiplier),
        median_price=median_price,
        base_commission_pips=commission_pips(PAIR, median_price, rates, FUSION_ZERO),
        stress_commission_pips=commission_pips(PAIR, median_price, rates, STRESS),
        quality={
            "pair": PAIR,
            "raw_bars": int(frame.attrs.get("raw_bars", len(frame))),
            "usable_bars": int(len(frame)),
            "dropped_crossed_quote_bars": int(frame.attrs.get("dropped_crossed_quote_bars", 0)),
            "first_bar": frame.index.min().isoformat() if len(frame) else None,
            "last_bar": frame.index.max().isoformat() if len(frame) else None,
        },
    )


def daily_atr_pips(frame: pd.DataFrame, pair: str) -> pd.Series:
    instrument = Instrument.from_symbol(pair)
    bars = mid_ohlc(frame).resample("1D", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    counts = mid_ohlc(frame)["close"].resample("1D", label="left", closed="left").count()
    bars = bars.loc[counts >= 20 * 60].dropna()
    bars.index = bars.index + pd.Timedelta(hours=23, minutes=59)
    previous_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return (ranges / instrument.pip_size).rolling(14, min_periods=14).mean().shift(1)


def frozen_entry_signals(pair_data: PairData) -> pd.DataFrame:
    signals = fade_signals(pair_data.bars, pair=PAIR, policy=FROZEN_ENTRY_POLICY)
    if signals.empty:
        return signals
    atr = daily_atr_pips(pair_data.bars, PAIR).rename("daily_atr_pips")
    enriched = signals.merge(
        atr,
        left_on="signal_timestamp",
        right_index=True,
        how="left",
    )
    enriched = enriched.loc[enriched["daily_atr_pips"].notna()].copy()
    return enriched.sort_values("entry_timestamp", ignore_index=True)


def evaluate_risk_policy(
    pair_data: PairData,
    signals: pd.DataFrame,
    risk_policy: RiskPolicy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hold_minutes = FROZEN_ENTRY_POLICY.hold_days * 24 * 60
    for signal in signals.to_dict("records"):
        risk_pips = risk_policy.stop_atr_multiple * float(signal["daily_atr_pips"])
        side = int(signal["side_num"])
        base = simulate_gappy_trade(
            pair_data.bars,
            pair=PAIR,
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
            pair=PAIR,
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
                "candidate_id": f"{PAIR}:{FROZEN_ENTRY_POLICY.identifier}:{risk_policy.identifier}",
                "pair": PAIR,
                "entry_policy_id": FROZEN_ENTRY_POLICY.identifier,
                "risk_policy_id": risk_policy.identifier,
                "channel_days": FROZEN_ENTRY_POLICY.channel_days,
                "hold_days": FROZEN_ENTRY_POLICY.hold_days,
                "stop_atr_multiple": risk_policy.stop_atr_multiple,
                "management": risk_policy.management.name,
                "signal_timestamp": signal["signal_timestamp"],
                "entry_timestamp": signal["entry_timestamp"],
                "side": signal["side"],
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


def simulate_gappy_trade(
    frame: pd.DataFrame,
    *,
    pair: str,
    entry_timestamp: pd.Timestamp,
    side: int,
    risk_pips: float,
    hold_minutes: int,
    management: Management,
    commission: float,
    slippage_per_side_pips: float,
) -> ScenarioResult | None:
    """Simulate a multi-day trade over available M1 bars, preserving weekend gaps."""
    if not np.isfinite(risk_pips) or risk_pips <= 0:
        return None
    exit_deadline = entry_timestamp + pd.Timedelta(minutes=hold_minutes)
    if entry_timestamp not in frame.index or exit_deadline not in frame.index:
        return None
    window = frame.loc[entry_timestamp:exit_deadline]
    if len(window) < 2:
        return None

    pip = Instrument.from_symbol(pair).pip_size
    risk_price = risk_pips * pip
    first = window.iloc[0]
    entry_price = float(first["ask_open"] if side == 1 else first["bid_open"])
    if side == 1:
        stop = entry_price - risk_price
        target = entry_price + management.target_r * risk_price if management.target_r is not None else None
    else:
        stop = entry_price + risk_price
        target = entry_price - management.target_r * risk_price if management.target_r is not None else None

    best_favorable_pips = 0.0
    close_exit = entry_price
    for bar in window.itertuples():
        timestamp = pd.Timestamp(bar.Index)
        if side == 1:
            open_exit = float(bar.bid_open)
            adverse = float(bar.bid_low)
            favorable = float(bar.bid_high)
            close_exit = float(bar.bid_close)
            if open_exit <= stop:
                return scenario_result(
                    side, entry_price, open_exit, pip, commission, slippage_per_side_pips, timestamp, "stop"
                )
            if adverse <= stop:
                return scenario_result(
                    side, entry_price, stop, pip, commission, slippage_per_side_pips, timestamp, "stop"
                )
            if target is not None and favorable >= target:
                return scenario_result(
                    side, entry_price, target, pip, commission, slippage_per_side_pips, timestamp, "target"
                )
            best_favorable_pips = max(best_favorable_pips, (favorable - entry_price) / pip)
            if trail_is_active(management, best_favorable_pips, risk_pips):
                stop = max(stop, entry_price, favorable - float(management.trail_r) * risk_price)
        else:
            open_exit = float(bar.ask_open)
            adverse = float(bar.ask_high)
            favorable = float(bar.ask_low)
            close_exit = float(bar.ask_close)
            if open_exit >= stop:
                return scenario_result(
                    side, entry_price, open_exit, pip, commission, slippage_per_side_pips, timestamp, "stop"
                )
            if adverse >= stop:
                return scenario_result(
                    side, entry_price, stop, pip, commission, slippage_per_side_pips, timestamp, "stop"
                )
            if target is not None and favorable <= target:
                return scenario_result(
                    side, entry_price, target, pip, commission, slippage_per_side_pips, timestamp, "target"
                )
            best_favorable_pips = max(best_favorable_pips, (entry_price - favorable) / pip)
            if trail_is_active(management, best_favorable_pips, risk_pips):
                stop = min(stop, entry_price, favorable + float(management.trail_r) * risk_price)

    return scenario_result(
        side,
        entry_price,
        close_exit,
        pip,
        commission,
        slippage_per_side_pips,
        pd.Timestamp(window.index[-1]),
        "timeout",
    )


def trail_is_active(management: Management, favorable_pips: float, risk_pips: float) -> bool:
    return (
        management.activation_r is not None
        and management.trail_r is not None
        and favorable_pips >= management.activation_r * risk_pips
    )


def scenario_result(
    side: int,
    entry_price: float,
    exit_price: float,
    pip: float,
    commission: float,
    slippage_per_side_pips: float,
    exit_timestamp: pd.Timestamp,
    exit_reason: str,
) -> ScenarioResult:
    raw = (exit_price - entry_price) * side / pip
    net = raw - commission - 2.0 * slippage_per_side_pips
    return ScenarioResult(
        raw_pips=float(raw),
        net_pips=float(net),
        exit_timestamp=exit_timestamp,
        exit_reason=exit_reason,
    )


def summarize_candidates(trades: pd.DataFrame) -> pd.DataFrame:
    selection = trades.loc[trades["entry_timestamp"] < SELECTION_END].copy()
    grouped = selection.groupby("candidate_id", sort=True)
    summary = grouped.agg(
        risk_policy_id=("risk_policy_id", "first"),
        stop_atr_multiple=("stop_atr_multiple", "first"),
        management=("management", "first"),
        selection_trades=("candidate_id", "size"),
        selection_mean_base_pips=("net_base_pips", "mean"),
        selection_mean_stress_pips=("net_stress_pips", "mean"),
        selection_total_stress_pips=("net_stress_pips", "sum"),
        selection_median_stress_pips=("net_stress_pips", "median"),
        selection_mean_risk_pips=("risk_pips", "mean"),
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
    return selection.loc[
        (selection["selection_trades"] >= MIN_SELECTION_TRADES)
        & (selection["selection_mean_base_pips"] > 0)
        & (selection["selection_mean_stress_pips"] > 0)
    ].sort_values(
        ["selection_mean_stress_pips", "selection_total_stress_pips"],
        ascending=[False, False],
    )


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


def fast130_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if passed_count:
        return (
            "FAST130_NZDUSD_DONCHIAN_FADE_RISK_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"risk_audit_candidates:{passed_count}",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    if finalists_count:
        return (
            "FAST130_NZDUSD_DONCHIAN_FADE_RISK_FAILED_AUDIT",
            reasons + ["risk_management_variants_failed_2024_2026_audit"],
        )
    return (
        "FAST130_NZDUSD_DONCHIAN_FADE_RISK_FAILED_SELECTION",
        reasons + ["no_risk_management_variant_positive_after_costs"],
    )


def run(*, m1_dir: Path = DEFAULT_M1_DIR, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    pair_data = load_pair_data(m1_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_path = output_root / "pair-quality.json"
    write_json(quality_path, pair_data.quality)
    signals = frozen_entry_signals(pair_data)
    signals_path = output_root / "entry-signals.parquet"
    signals.to_parquet(signals_path, compression="zstd", index=False)

    trade_rows: list[dict[str, Any]] = []
    for policy in risk_policy_grid():
        trade_rows.extend(evaluate_risk_policy(pair_data, signals, policy))
        sys.stdout.write(f"{policy.identifier}: accumulated trades={len(trade_rows)}\n")
        sys.stdout.flush()
    if not trade_rows:
        raise ValueError("no risk-audit trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "candidate_id"])
    trades_path = output_root / "risk-audit-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)
    selection = summarize_candidates(trades)
    finalists = select_finalists(selection)
    audit = audit_finalists(trades, finalists)
    selection_path = output_root / "selection-metrics.csv"
    audit_path = output_root / "audited-finalists.csv"
    selection.to_csv(selection_path, index=False)
    audit.to_csv(audit_path, index=False)

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    status, status_reasons = fast130_status(len(passed), len(finalists))
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast130_is_risk_audit_not_strategy_approval",
            "risk_policies_are_selected_on_historical_selection_sample",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast130_nzdusd_donchian_fade_risk_audit",
        "status": status,
        "status_reasons": status_reasons,
        "pair": PAIR,
        "entry_policy": asdict(FROZEN_ENTRY_POLICY),
        "risk_policy_count": len(risk_policy_grid()),
        "signal_count": int(len(signals)),
        "selection_end": SELECTION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "trade_rows": int(len(trades)),
        "selection_candidates": int(len(selection)),
        "eligible_finalists": int(len(finalists)),
        "audited_finalists": int(len(audit)),
        "passed_candidates": int(len(passed)),
        "costs": {
            "base": asdict(FUSION_ZERO),
            "stress": asdict(STRESS),
            "risk_basis": "daily ATR(14) shifted one D1 bar before signal",
        },
        "artifacts": {
            "signals": {"path": str(signals_path.relative_to(REPO_ROOT)), "sha256": sha256_file(signals_path)},
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
        "signal_count": report["signal_count"],
        "trade_rows": report["trade_rows"],
        "eligible_finalists": report["eligible_finalists"],
        "passed_candidates": report["passed_candidates"],
        "top_audit": report.get("top_audit", []),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
