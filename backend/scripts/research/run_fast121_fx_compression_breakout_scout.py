"""Fast121: FX H1/H4 volatility-compression breakout scout, executed on M1 bid/ask.

Research-only script. Classic H4/D1 technical families already failed G0; this
tests a narrower structural variation: only trade range breakouts after a causal
low-volatility compression filter, with larger stops/targets so transaction
costs are smaller relative to the intended move. It writes offline artifacts
only and never touches the trading engine, database, credentials, ledger, Demo,
Testnet or LIVE accounts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.fx.instruments import ConversionRates, Instrument
from app.fx.sim.costs import FUSION_ZERO, STRESS, CostScenario, commission_round_trip_pips
from scripts.research.run_fast120_fx_currency_strength_scout import (
    AUDIT_END,
    DATA_START,
    DEFAULT_M1_DIR,
    lower_confidence_bound,
    plain_json,
    positive_year_share,
    simulate_trade,
    widen_quotes,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast121" / "fx-compression-breakout-scout"
)

PAIRS = (
    "AUDUSD",
    "EURGBP",
    "EURJPY",
    "EURUSD",
    "GBPJPY",
    "GBPUSD",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
)
MAJOR_RATE_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY")
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
MIN_SELECTION_TRADES = 60
MIN_AUDIT_TRADES = 25
MIN_AUDIT_YEARS = 2
MIN_AUDIT_POSITIVE_YEAR_SHARE = 2 / 3
FINALISTS = 50

Timeframe = Literal["1h", "4h"]


@dataclass(frozen=True)
class Management:
    """One exit-management rule expressed in multiples of initial risk R."""

    name: str
    target_r: float | None
    activation_r: float | None
    trail_r: float | None


@dataclass(frozen=True)
class Policy:
    """One compression-breakout hypothesis."""

    identifier: str
    timeframe: Timeframe
    channel_bars: int
    compression_quantile: float
    stop_atr_multiple: float
    max_hold_minutes: int
    management: Management


@dataclass(frozen=True)
class Signal:
    """A causal breakout signal emitted at an aggregated bar close."""

    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp
    side: int
    breakout_distance_pips: float
    atr_pips: float
    compression_value_pips: float
    compression_threshold_pips: float


@dataclass(frozen=True)
class PairData:
    """Loaded M1 data and cost facts for one pair."""

    bars: pd.DataFrame
    stress_bars: pd.DataFrame
    median_price: float
    base_commission_pips: float
    stress_commission_pips: float
    quality: dict[str, Any]


def policy_grid() -> tuple[Policy, ...]:
    """Return a deterministic H1/H4 compression-breakout grid."""
    managements = (
        Management("fixed_2r", target_r=2.0, activation_r=None, trail_r=None),
        Management("fixed_3r", target_r=3.0, activation_r=None, trail_r=None),
        Management("trail_1r_1r", target_r=None, activation_r=1.0, trail_r=1.0),
    )
    policies: list[Policy] = []
    for timeframe in ("1h", "4h"):
        hold_values = (12 * 60, 24 * 60) if timeframe == "1h" else (24 * 60, 72 * 60)
        for channel in (20, 55):
            for compression_quantile in (0.25, 0.40):
                for stop_atr_multiple in (1.0, 1.5):
                    for hold in hold_values:
                        for management in managements:
                            identifier = (
                                f"tf{timeframe}_ch{channel}_cq{compression_quantile:g}"
                                f"_atr{stop_atr_multiple:g}_hold{hold}_{management.name}"
                            )
                            policies.append(
                                Policy(
                                    identifier=identifier,
                                    timeframe=timeframe,
                                    channel_bars=channel,
                                    compression_quantile=compression_quantile,
                                    stop_atr_multiple=stop_atr_multiple,
                                    max_hold_minutes=hold,
                                    management=management,
                                )
                            )
    return tuple(policies)


def load_pair_bars(m1_dir: Path, pair: str) -> pd.DataFrame:
    """Load one pair's executable M1 bid/ask bars for the validated 2019-2026 window."""
    paths = sorted(m1_dir.glob(f"{pair}_M1_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no M1 parquet files for {pair} in {m1_dir}")
    columns = [
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]
    frame = pd.concat(pd.read_parquet(path, columns=columns) for path in paths).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.index.tz is None:
        raise ValueError(f"{pair} bars must be timezone-aware")
    raw_bars = len(frame)
    frame = frame.loc[(frame.index >= DATA_START) & (frame.index < AUDIT_END)].copy()
    crossed = (
        frame[["ask_open", "ask_high", "ask_low", "ask_close"]].to_numpy()
        < frame[["bid_open", "bid_high", "bid_low", "bid_close"]].to_numpy()
    ).any(axis=1)
    if crossed.any():
        frame = frame.loc[~crossed].copy()
    frame.attrs["raw_bars"] = raw_bars
    frame.attrs["dropped_crossed_quote_bars"] = int(crossed.sum())
    return frame


def mid_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert M1 bid/ask OHLC to midpoint OHLC."""
    return pd.DataFrame(
        {
            "open": 0.5 * (frame["bid_open"] + frame["ask_open"]),
            "high": 0.5 * (frame["bid_high"] + frame["ask_high"]),
            "low": 0.5 * (frame["bid_low"] + frame["ask_low"]),
            "close": 0.5 * (frame["bid_close"] + frame["ask_close"]),
        },
        index=frame.index,
    )


def aggregate_mid_bars(frame: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Aggregate M1 midpoint bars and label each bar by its close timestamp."""
    rule = "1h" if timeframe == "1h" else "4h"
    minutes = 60 if timeframe == "1h" else 240
    mids = mid_ohlc(frame)
    bars = mids.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    counts = mids["close"].resample(rule, label="left", closed="left").count()
    bars = bars.loc[counts >= minutes].dropna()
    bars.index = bars.index + pd.Timedelta(minutes=minutes - 1)
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


def signals_for_policy(frame: pd.DataFrame, pair: str, policy: Policy) -> tuple[Signal, ...]:
    """Generate causal compression-breakout signals for one pair/policy."""
    instrument = Instrument.from_symbol(pair)
    bars = aggregate_mid_bars(frame, policy.timeframe)
    if len(bars) < policy.channel_bars + 600:
        return ()
    pip = instrument.pip_size
    atr = true_range_pips(bars, pip).rolling(14, min_periods=14).mean().shift(1)
    channel_high = bars["high"].rolling(policy.channel_bars, min_periods=policy.channel_bars).max().shift(1)
    channel_low = bars["low"].rolling(policy.channel_bars, min_periods=policy.channel_bars).min().shift(1)
    compression = true_range_pips(bars, pip).rolling(20, min_periods=20).mean().shift(1)
    compression_threshold = (
        compression.rolling(500, min_periods=250).quantile(policy.compression_quantile).shift(1)
    )
    breakout_up = (bars["close"] > channel_high) & (compression <= compression_threshold)
    breakout_down = (bars["close"] < channel_low) & (compression <= compression_threshold)

    signals: list[Signal] = []
    for timestamp in bars.index[breakout_up | breakout_down]:
        atr_pips = float(atr.loc[timestamp])
        threshold = float(compression_threshold.loc[timestamp])
        value = float(compression.loc[timestamp])
        if not np.isfinite(atr_pips) or atr_pips <= 0 or not np.isfinite(value) or not np.isfinite(threshold):
            continue
        if bool(breakout_up.loc[timestamp]):
            side = 1
            distance = (float(bars.loc[timestamp, "close"]) - float(channel_high.loc[timestamp])) / pip
        else:
            side = -1
            distance = (float(channel_low.loc[timestamp]) - float(bars.loc[timestamp, "close"])) / pip
        if distance <= 0:
            continue
        signals.append(
            Signal(
                signal_timestamp=pd.Timestamp(timestamp),
                entry_timestamp=pd.Timestamp(timestamp) + pd.Timedelta(minutes=1),
                side=side,
                breakout_distance_pips=float(distance),
                atr_pips=atr_pips,
                compression_value_pips=value,
                compression_threshold_pips=threshold,
            )
        )
    return tuple(signals)


def median_rates_from_frames(frames: dict[str, pd.DataFrame]) -> ConversionRates:
    mids: dict[str, float] = {}
    for pair in MAJOR_RATE_PAIRS:
        if pair not in frames:
            raise FileNotFoundError(f"missing M1 rate pair {pair}")
        mids[pair] = float(np.median(mid_ohlc(frames[pair])["close"].to_numpy()))
    return ConversionRates(mids)


def commission_pips(pair: str, price: float, rates: ConversionRates, scenario: CostScenario) -> float:
    return commission_round_trip_pips(
        Instrument.from_symbol(pair),
        price,
        rates,
        base_currency_per_lot_per_side=scenario.commission_base_per_lot_per_side,
    )


def load_market(m1_dir: Path) -> dict[str, PairData]:
    raw_frames = {pair: load_pair_bars(m1_dir, pair) for pair in PAIRS}
    rates = median_rates_from_frames(raw_frames)
    market: dict[str, PairData] = {}
    for pair, frame in raw_frames.items():
        median_price = float(np.median(mid_ohlc(frame)["close"].to_numpy()))
        market[pair] = PairData(
            bars=frame,
            stress_bars=widen_quotes(frame, STRESS.spread_multiplier),
            median_price=median_price,
            base_commission_pips=commission_pips(pair, median_price, rates, FUSION_ZERO),
            stress_commission_pips=commission_pips(pair, median_price, rates, STRESS),
            quality={
                "pair": pair,
                "raw_bars": int(frame.attrs.get("raw_bars", len(frame))),
                "usable_bars": int(len(frame)),
                "dropped_crossed_quote_bars": int(frame.attrs.get("dropped_crossed_quote_bars", 0)),
                "first_bar": frame.index.min().isoformat() if len(frame) else None,
                "last_bar": frame.index.max().isoformat() if len(frame) else None,
            },
        )
        sys.stdout.write(
            f"{pair}: loaded {len(frame)} bars"
            f" (dropped_crossed={market[pair].quality['dropped_crossed_quote_bars']})\n"
        )
        sys.stdout.flush()
    return market


def evaluate_pair_policy(pair: str, pair_data: PairData, policy: Policy) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signals = signals_for_policy(pair_data.bars, pair, policy)
    for signal in signals:
        risk_pips = policy.stop_atr_multiple * signal.atr_pips
        base = simulate_trade(
            pair_data.bars,
            pair=pair,
            entry_timestamp=signal.entry_timestamp,
            side=signal.side,
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
            side=signal.side,
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
                "policy_id": policy.identifier,
                "pair": pair,
                "timeframe": policy.timeframe,
                "channel_bars": policy.channel_bars,
                "compression_quantile": policy.compression_quantile,
                "stop_atr_multiple": policy.stop_atr_multiple,
                "max_hold_minutes": policy.max_hold_minutes,
                "management": policy.management.name,
                "signal_timestamp": signal.signal_timestamp,
                "entry_timestamp": signal.entry_timestamp,
                "side": "LONG" if signal.side == 1 else "SHORT",
                "breakout_distance_pips": signal.breakout_distance_pips,
                "atr_pips": signal.atr_pips,
                "risk_pips": risk_pips,
                "compression_value_pips": signal.compression_value_pips,
                "compression_threshold_pips": signal.compression_threshold_pips,
                "raw_base_pips": base.raw_pips,
                "net_base_pips": base.net_pips,
                "exit_base_timestamp": base.exit_timestamp,
                "exit_base_reason": base.exit_reason,
                "raw_stress_pips": stress.raw_pips,
                "net_stress_pips": stress.net_pips,
                "exit_stress_timestamp": stress.exit_timestamp,
                "exit_stress_reason": stress.exit_reason,
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
        timeframe=("timeframe", "first"),
        channel_bars=("channel_bars", "first"),
        compression_quantile=("compression_quantile", "first"),
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
    alpha = 0.05 / max(len(finalists), 1)
    audit = trades.loc[
        (trades["entry_timestamp"] >= SELECTION_END) & (trades["entry_timestamp"] < AUDIT_END)
    ].copy()
    rows: list[dict[str, Any]] = []
    for rank, finalist in enumerate(finalists.itertuples(index=False), start=1):
        sample = audit.loc[audit["candidate_id"] == finalist.candidate_id]
        base_lcb = lower_confidence_bound(sample["net_base_pips"], alpha)
        stress_lcb = lower_confidence_bound(sample["net_stress_pips"], alpha)
        stress_share, stress_positive_years, stress_years = positive_year_share(sample, "net_stress_pips")
        mean_base = float(sample["net_base_pips"].mean()) if len(sample) else float("nan")
        mean_stress = float(sample["net_stress_pips"].mean()) if len(sample) else float("nan")
        passed = (
            len(sample) >= MIN_AUDIT_TRADES
            and stress_years >= MIN_AUDIT_YEARS
            and mean_base > 0
            and mean_stress > 0
            and base_lcb > 0
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
                "audit_lower_base_pips": base_lcb,
                "audit_lower_stress_pips": stress_lcb,
                "audit_positive_stress_year_share": stress_share,
                "audit_positive_stress_years": stress_positive_years,
                "audit_years": stress_years,
                "bonferroni_alpha": alpha,
                "pilot_gate": bool(passed),
            }
        )
    return pd.DataFrame(rows)


def fast121_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if passed_count:
        return (
            "FAST121_FX_COMPRESSION_BREAKOUT_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons + [f"historical_audit_candidates:{passed_count}", "fresh_preregistered_prospective_observation_required"],
        )
    if finalists_count:
        return (
            "FAST121_FX_COMPRESSION_BREAKOUT_FAILED_AUDIT",
            reasons + ["selection_positive_candidates_failed_2024_2026_audit"],
        )
    return (
        "FAST121_FX_COMPRESSION_BREAKOUT_FAILED_SELECTION",
        reasons + ["no_selection_candidate_positive_after_costs"],
    )


def run(
    *,
    m1_dir: Path = DEFAULT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    policies = policy_grid()
    market = load_market(m1_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_path = output_root / "pair-quality.json"
    write_json(quality_path, [market[pair].quality for pair in PAIRS])

    trade_rows: list[dict[str, Any]] = []
    total_jobs = len(PAIRS) * len(policies)
    done = 0
    for pair in PAIRS:
        pair_data = market[pair]
        for policy in policies:
            trade_rows.extend(evaluate_pair_policy(pair, pair_data, policy))
            done += 1
            if done % 100 == 0 or done == total_jobs:
                sys.stdout.write(f"jobs {done}/{total_jobs}; accumulated trades={len(trade_rows)}\n")
                sys.stdout.flush()
    if not trade_rows:
        raise ValueError("no compression-breakout trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "candidate_id"])
    trades_path = output_root / "compression-breakout-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)
    selection = summarize_candidates(trades)
    finalists = select_finalists(selection)
    audit = audit_finalists(trades, finalists)
    selection_path = output_root / "selection-metrics.csv"
    audit_path = output_root / "audited-finalists.csv"
    selection.to_csv(selection_path, index=False)
    audit.to_csv(audit_path, index=False)

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    winner = (
        passed.sort_values("audit_mean_stress_pips", ascending=False).head(1).to_dict("records")
        if not passed.empty
        else []
    )
    status, status_reasons = fast121_status(len(passed), len(finalists))
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast121_is_historical_scout_not_strategy_approval",
            "compression_breakout_policies_are_selected_on_historical_selection_sample",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast121_fx_compression_breakout_scout",
        "status": status,
        "status_reasons": status_reasons,
        "m1_dir": str(m1_dir.relative_to(REPO_ROOT)),
        "pairs": list(PAIRS),
        "policy_count_per_pair": len(policies),
        "candidate_count": len(PAIRS) * len(policies),
        "selection_start": DATA_START.isoformat(),
        "selection_end": SELECTION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "trade_rows": int(len(trades)),
        "selection_candidates": int(len(selection)),
        "eligible_finalists": int(len(finalists)),
        "audited_finalists": int(len(audit)),
        "passed_candidates": int(len(passed)),
        "winner": winner[0] if winner else None,
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
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path.relative_to(REPO_ROOT))})
    console = {
        "status": status,
        "status_reasons": status_reasons,
        "approved_strategy": report["approved_strategy"],
        "orders_sent": report["orders_sent"],
        "trade_rows": report["trade_rows"],
        "eligible_finalists": report["eligible_finalists"],
        "passed_candidates": report["passed_candidates"],
        "report_path": str(report_path.relative_to(REPO_ROOT)),
        "report_sha256": sha256_file(report_path),
    }
    sys.stdout.write(json.dumps(console, indent=2, sort_keys=True) + "\n")
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    run(m1_dir=arguments.m1_dir, output_root=arguments.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
