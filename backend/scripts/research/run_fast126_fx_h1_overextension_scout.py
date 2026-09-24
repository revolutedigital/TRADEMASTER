"""Fast126: FX H1 overextension continuation/reversion scout on executable M1 bars.

This is a different family from the Fast121 compression breakout clue. It asks
whether rare H1 overextensions across the FX M1 universe should be followed or
faded with a non-overlapping intraday trade. The runner is research-only: it
writes offline artifacts and never touches trading engine, credentials, ledger,
Demo, Testnet, or LIVE accounts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import t

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.fx.instruments import Instrument  # noqa: E402
from app.fx.sim.costs import FUSION_ZERO, STRESS  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import (  # noqa: E402
    Management,
    simulate_trade,
)
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    AUDIT_END,
    DEFAULT_M1_DIR,
    PAIRS,
    PairData,
    load_market,
    mid_ohlc,
    sha256_file,
)
from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast126" / "fx-h1-overextension-scout"

SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
FINALISTS = 40
MIN_SELECTION_TRADES = 40
MIN_AUDIT_TRADES = 15
MIN_AUDIT_YEARS = 2
MIN_AUDIT_POSITIVE_YEAR_SHARE = 2 / 3

Direction = Literal["continuation", "reversion"]


@dataclass(frozen=True)
class Policy:
    identifier: str
    direction: Direction
    lookback_bars: int
    threshold_atr: float
    stop_atr_multiple: float
    max_hold_minutes: int
    management: Management


@dataclass(frozen=True)
class Signal:
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    continuation_side: int
    move_pips: float
    atr_pips: float
    overextension_atr: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(m1_dir=arguments.m1_dir, output_root=arguments.output_root)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def policy_grid() -> tuple[Policy, ...]:
    managements = (
        Management("fixed_1r", target_r=1.0, activation_r=None, trail_r=None),
        Management("trail_0_5r_0_5r", target_r=None, activation_r=0.5, trail_r=0.5),
    )
    policies: list[Policy] = []
    for direction in ("continuation", "reversion"):
        for lookback in (4, 8):
            for threshold_atr in (3.0, 3.5):
                for hold in (240, 480):
                    for management in managements:
                        identifier = (
                            f"h1_{direction}_nonoverlap_lb{lookback}_thr{threshold_atr:g}"
                            f"_atr1.5_hold{hold}_{management.name}"
                        )
                        policies.append(
                            Policy(
                                identifier=identifier,
                                direction=direction,
                                lookback_bars=lookback,
                                threshold_atr=threshold_atr,
                                stop_atr_multiple=1.5,
                                max_hold_minutes=hold,
                                management=management,
                            )
                        )
    return tuple(policies)


def aggregate_h1_mid_bars(frame: pd.DataFrame) -> pd.DataFrame:
    mids = mid_ohlc(frame)
    bars = mids.resample("60min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    counts = mids["close"].resample("60min", label="left", closed="left").count()
    bars = bars.loc[counts >= 60].dropna()
    bars.index = bars.index + pd.Timedelta(minutes=59)
    return bars


def true_range_pips(bars: pd.DataFrame, pip: float) -> pd.Series:
    previous_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return ranges / pip


def raw_signals_for_key(
    frame: pd.DataFrame,
    *,
    pair: str,
    lookback_bars: int,
    threshold_atr: float,
) -> tuple[Signal, ...]:
    instrument = Instrument.from_symbol(pair)
    bars = aggregate_h1_mid_bars(frame)
    if len(bars) < lookback_bars + 200:
        return ()
    atr = true_range_pips(bars, instrument.pip_size).rolling(14, min_periods=14).mean().shift(1)
    move_pips = (bars["close"] - bars["close"].shift(lookback_bars)) / instrument.pip_size
    overextension = (move_pips.abs() / atr).replace([np.inf, -np.inf], np.nan)
    mask = (overextension >= threshold_atr) & atr.notna()
    signals: list[Signal] = []
    for timestamp in bars.index[mask]:
        move = float(move_pips.loc[timestamp])
        atr_pips = float(atr.loc[timestamp])
        ratio = float(overextension.loc[timestamp])
        if not np.isfinite(move) or not np.isfinite(atr_pips) or atr_pips <= 0:
            continue
        continuation_side = 1 if move > 0 else -1
        signals.append(
            Signal(
                signal_timestamp=pd.Timestamp(timestamp),
                entry_timestamp=pd.Timestamp(timestamp) + pd.Timedelta(minutes=1),
                continuation_side=continuation_side,
                move_pips=move,
                atr_pips=atr_pips,
                overextension_atr=ratio,
            )
        )
    return tuple(signals)


def non_overlapping_signals(signals: tuple[Signal, ...], *, hold_minutes: int) -> tuple[Signal, ...]:
    accepted: list[Signal] = []
    next_free = pd.Timestamp.min.tz_localize("UTC")
    for signal in sorted(signals, key=lambda item: item.entry_timestamp):
        if signal.entry_timestamp <= next_free:
            continue
        accepted.append(signal)
        next_free = signal.entry_timestamp + pd.Timedelta(minutes=hold_minutes)
    return tuple(accepted)


def side_for_direction(signal: Signal, direction: Direction) -> int:
    if direction == "continuation":
        return signal.continuation_side
    return -signal.continuation_side


def evaluate_pair_policy(
    pair: str,
    pair_data: PairData,
    policy: Policy,
    signals: tuple[Signal, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    accepted_signals = non_overlapping_signals(signals, hold_minutes=policy.max_hold_minutes)
    for signal in accepted_signals:
        side = side_for_direction(signal, policy.direction)
        risk_pips = policy.stop_atr_multiple * signal.atr_pips
        base = simulate_trade(
            pair_data.bars,
            pair=pair,
            entry_timestamp=signal.entry_timestamp,
            side=side,
            risk_pips=risk_pips,
            hold_minutes=policy.max_hold_minutes,
            management=policy.management,
            commission=pair_data.base_commission_pips,
            slippage_per_side_pips=FUSION_ZERO.slippage_pips,
        )
        if base is None:
            continue
        stress = simulate_trade(
            pair_data.stress_bars,
            pair=pair,
            entry_timestamp=signal.entry_timestamp,
            side=side,
            risk_pips=risk_pips,
            hold_minutes=policy.max_hold_minutes,
            management=policy.management,
            commission=pair_data.stress_commission_pips,
            slippage_per_side_pips=STRESS.slippage_pips,
        )
        if stress is None:
            continue
        rows.append(
            {
                "candidate_id": f"{pair}:{policy.identifier}",
                "pair": pair,
                "policy_id": policy.identifier,
                "direction": policy.direction,
                "timeframe": "1h",
                "lookback_bars": policy.lookback_bars,
                "threshold_atr": policy.threshold_atr,
                "stop_atr_multiple": policy.stop_atr_multiple,
                "max_hold_minutes": policy.max_hold_minutes,
                "management": policy.management.name,
                "signal_timestamp": signal.signal_timestamp,
                "entry_timestamp": signal.entry_timestamp,
                "side": "LONG" if side == 1 else "SHORT",
                "move_pips": signal.move_pips,
                "atr_pips": signal.atr_pips,
                "overextension_atr": signal.overextension_atr,
                "risk_pips": risk_pips,
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
        policy_id=("policy_id", "first"),
        direction=("direction", "first"),
        lookback_bars=("lookback_bars", "first"),
        threshold_atr=("threshold_atr", "first"),
        stop_atr_multiple=("stop_atr_multiple", "first"),
        max_hold_minutes=("max_hold_minutes", "first"),
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
        share, positive, total = positive_year_share(sample, "net_stress_pips")
        rows.append(
            {
                "candidate_id": candidate_id,
                "selection_lower_stress_pips": lower_confidence_bound(
                    sample["net_stress_pips"],
                    0.05,
                ),
                "selection_positive_stress_year_share": share,
                "selection_positive_stress_years": positive,
                "selection_years": total,
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


def fast126_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if passed_count:
        return (
            "FAST126_FX_H1_OVEREXTENSION_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons + [f"historical_audit_candidates:{passed_count}", "fresh_preregistered_prospective_observation_required"],
        )
    if finalists_count:
        return (
            "FAST126_FX_H1_OVEREXTENSION_FAILED_AUDIT",
            reasons + ["selection_positive_candidates_failed_2024_2026_audit"],
        )
    return (
        "FAST126_FX_H1_OVEREXTENSION_FAILED_SELECTION",
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
        signal_cache: dict[tuple[int, float], tuple[Signal, ...]] = {}
        for policy in policies:
            key = (policy.lookback_bars, policy.threshold_atr)
            if key not in signal_cache:
                signal_cache[key] = raw_signals_for_key(
                    pair_data.bars,
                    pair=pair,
                    lookback_bars=policy.lookback_bars,
                    threshold_atr=policy.threshold_atr,
                )
            trade_rows.extend(evaluate_pair_policy(pair, pair_data, policy, signal_cache[key]))
        sys.stdout.write(f"{pair}: accumulated trades={len(trade_rows)}\n")
        sys.stdout.flush()
    if not trade_rows:
        raise ValueError("no H1 overextension trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "candidate_id"])
    trades_path = output_root / "overextension-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)
    selection = summarize_candidates(trades)
    finalists = select_finalists(selection)
    audit = audit_finalists(trades, finalists)
    selection_path = output_root / "selection-metrics.csv"
    audit_path = output_root / "audited-finalists.csv"
    selection.to_csv(selection_path, index=False)
    audit.to_csv(audit_path, index=False)

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    status, status_reasons = fast126_status(len(passed), len(finalists))
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast126_is_historical_scout_not_strategy_approval",
            "overextension_policies_are_selected_on_historical_selection_sample",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast126_fx_h1_overextension_scout",
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
        "direction_counts": trades.groupby("direction").size().to_dict(),
        "eligible_direction_counts": finalists.groupby("direction").size().to_dict()
        if not finalists.empty
        else {},
        "costs": {
            "base": asdict(FUSION_ZERO),
            "stress": asdict(STRESS),
            "slippage_model_used": "fixed per side from scenario; stress spread multiplier applied",
        },
        "artifacts": {
            "trades": {"path": str(trades_path.relative_to(REPO_ROOT)), "sha256": sha256_file(trades_path)},
            "pair_quality": {"path": str(quality_path.relative_to(REPO_ROOT)), "sha256": sha256_file(quality_path)},
            "selection": {"path": str(selection_path.relative_to(REPO_ROOT)), "sha256": sha256_file(selection_path)},
            "audit": {"path": str(audit_path.relative_to(REPO_ROOT)), "sha256": sha256_file(audit_path)},
        },
    }
    if not finalists.empty:
        report["top_finalist"] = finalists.head(1).to_dict("records")[0]
    if not audit.empty:
        report["top_audit"] = audit.head(5).to_dict("records")
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
        "direction_counts": report["direction_counts"],
        "eligible_direction_counts": report["eligible_direction_counts"],
        "top_audit": report.get("top_audit", []),
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
