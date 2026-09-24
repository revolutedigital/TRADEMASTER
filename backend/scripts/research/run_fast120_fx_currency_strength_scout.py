"""Fast120: cross-sectional FX currency-strength scout on executable M1 bars.

Research-only script. It asks whether a currency basket view is better than
looking at each pair in isolation: at fixed session hours, rank currencies by
recent strength, trade the pair with the largest tradable divergence, and audit
the result after bid/ask spread, commission and slippage. It never touches the
trading engine, database, credentials, ledger, Demo, Testnet or LIVE accounts.
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
from scipy.stats import t

from app.fx.instruments import ConversionRates, Instrument
from app.fx.sim.costs import FUSION_ZERO, STRESS, CostScenario, commission_round_trip_pips

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_M1_DIR = REPO_ROOT / "backend" / "data" / "raw" / "fx_m1_hd"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast120" / "fx-currency-strength-scout"
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
CURRENCIES = ("AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "USD")
MAJOR_RATE_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY")
DATA_START = pd.Timestamp("2019-01-01T00:00:00Z")
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
AUDIT_END = pd.Timestamp("2026-09-01T00:00:00Z")
MIN_RISK_PIPS = 4.0
RISK_RANGE_MULTIPLE = 10.0
RISK_RANGE_WINDOW = 60
FINALISTS = 40
MIN_SELECTION_TRADES = 200
MIN_AUDIT_TRADES = 80
MIN_AUDIT_YEARS = 2
MIN_AUDIT_POSITIVE_YEAR_SHARE = 2 / 3

Mode = Literal["momentum", "reversal"]
ExitReason = Literal["stop", "target", "timeout"]


@dataclass(frozen=True)
class Management:
    """One exit-management rule expressed in multiples of initial risk R."""

    name: str
    target_r: float | None
    activation_r: float | None
    trail_r: float | None


@dataclass(frozen=True)
class Policy:
    """One cross-sectional currency-strength hypothesis."""

    identifier: str
    mode: Mode
    lookback_minutes: int
    decision_hour_utc: int
    threshold_bps: float
    max_hold_minutes: int
    management: Management


@dataclass(frozen=True)
class ScenarioResult:
    """Executable trade outcome in one cost scenario."""

    raw_pips: float
    net_pips: float
    exit_timestamp: pd.Timestamp
    exit_reason: ExitReason


@dataclass(frozen=True)
class PairEvaluationData:
    """Loaded pair data used by the signal generator and simulator."""

    bars: pd.DataFrame
    stress_bars: pd.DataFrame
    risk_pips: pd.Series
    median_price: float
    base_commission_pips: float
    stress_commission_pips: float
    quality: dict[str, Any]


def policy_grid() -> tuple[Policy, ...]:
    """Return the deterministic exploratory grid for Fast120."""
    managements = (
        Management("fixed_1r", target_r=1.0, activation_r=None, trail_r=None),
        Management("fixed_2r", target_r=2.0, activation_r=None, trail_r=None),
        Management("trail_1r_0_75r", target_r=None, activation_r=1.0, trail_r=0.75),
    )
    policies: list[Policy] = []
    for mode in ("momentum", "reversal"):
        for lookback in (60, 240, 1440):
            for hour in (0, 7, 13):
                for threshold in (5.0, 10.0, 20.0):
                    for hold in (60, 240):
                        for management in managements:
                            identifier = (
                                f"{mode}_lb{lookback}_h{hour:02d}_thr{threshold:g}"
                                f"_hold{hold}_{management.name}"
                            )
                            policies.append(
                                Policy(
                                    identifier=identifier,
                                    mode=mode,
                                    lookback_minutes=lookback,
                                    decision_hour_utc=hour,
                                    threshold_bps=threshold,
                                    max_hold_minutes=hold,
                                    management=management,
                                )
                            )
    return tuple(policies)


def split_pair(pair: str) -> tuple[str, str]:
    instrument = Instrument.from_symbol(pair)
    return instrument.base, instrument.quote


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


def widen_quotes(frame: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Widen each bid/ask column symmetrically around its midpoint."""
    if multiplier < 1.0:
        raise ValueError("spread multiplier cannot be below one")
    widened = frame.copy()
    for name in ("open", "high", "low", "close"):
        bid_column = f"bid_{name}"
        ask_column = f"ask_{name}"
        mid = 0.5 * (frame[bid_column] + frame[ask_column])
        half = 0.5 * (frame[ask_column] - frame[bid_column]) * multiplier
        widened[bid_column] = mid - half
        widened[ask_column] = mid + half
    return widened


def mid_close(frame: pd.DataFrame) -> pd.Series:
    return 0.5 * (frame["bid_close"] + frame["ask_close"])


def average_range_pips(frame: pd.DataFrame, pair: str) -> pd.Series:
    """Rolling causal average M1 high-low range, expressed in pips."""
    pip = Instrument.from_symbol(pair).pip_size
    mid_high = 0.5 * (frame["bid_high"] + frame["ask_high"])
    mid_low = 0.5 * (frame["bid_low"] + frame["ask_low"])
    return ((mid_high - mid_low) / pip).rolling(RISK_RANGE_WINDOW, min_periods=20).mean().shift(1)


def median_rates_from_frames(frames: dict[str, pd.DataFrame]) -> ConversionRates:
    mids: dict[str, float] = {}
    for pair in MAJOR_RATE_PAIRS:
        if pair not in frames:
            raise FileNotFoundError(f"missing M1 rate pair {pair}")
        mids[pair] = float(np.median(mid_close(frames[pair]).to_numpy()))
    return ConversionRates(mids)


def commission_pips(pair: str, price: float, rates: ConversionRates, scenario: CostScenario) -> float:
    return commission_round_trip_pips(
        Instrument.from_symbol(pair),
        price,
        rates,
        base_currency_per_lot_per_side=scenario.commission_base_per_lot_per_side,
    )


def load_market(m1_dir: Path) -> tuple[dict[str, PairEvaluationData], pd.DataFrame]:
    """Load all pair frames and the aligned mid-close matrix."""
    raw_frames = {pair: load_pair_bars(m1_dir, pair) for pair in PAIRS}
    rates = median_rates_from_frames(raw_frames)
    data: dict[str, PairEvaluationData] = {}
    mid_columns: dict[str, pd.Series] = {}
    for pair, frame in raw_frames.items():
        median_price = float(np.median(mid_close(frame).to_numpy()))
        risk = (RISK_RANGE_MULTIPLE * average_range_pips(frame, pair)).clip(lower=MIN_RISK_PIPS)
        data[pair] = PairEvaluationData(
            bars=frame,
            stress_bars=widen_quotes(frame, STRESS.spread_multiplier),
            risk_pips=risk,
            median_price=median_price,
            base_commission_pips=commission_pips(pair, median_price, rates, FUSION_ZERO),
            stress_commission_pips=commission_pips(pair, median_price, rates, STRESS),
            quality={
                "pair": pair,
                "raw_bars": int(frame.attrs.get("raw_bars", len(frame))),
                "usable_bars": int(len(frame)),
                "dropped_crossed_quote_bars": int(
                    frame.attrs.get("dropped_crossed_quote_bars", 0)
                ),
                "first_bar": frame.index.min().isoformat() if len(frame) else None,
                "last_bar": frame.index.max().isoformat() if len(frame) else None,
            },
        )
        mid_columns[pair] = mid_close(frame)
        sys.stdout.write(
            f"{pair}: loaded {len(frame)} bars"
            f" (dropped_crossed={data[pair].quality['dropped_crossed_quote_bars']})\n"
        )
        sys.stdout.flush()
    mids = pd.concat(mid_columns, axis=1, join="inner").sort_index()
    return data, mids


def currency_scores(pair_returns_bps: pd.DataFrame) -> pd.DataFrame:
    """Convert pair log returns into equal-weight currency strength scores."""
    scores = pd.DataFrame(0.0, index=pair_returns_bps.index, columns=CURRENCIES)
    counts = {currency: 0 for currency in CURRENCIES}
    for pair in PAIRS:
        base, quote = split_pair(pair)
        scores[base] = scores[base] + pair_returns_bps[pair]
        scores[quote] = scores[quote] - pair_returns_bps[pair]
        counts[base] += 1
        counts[quote] += 1
    for currency, count in counts.items():
        if count == 0:
            raise ValueError(f"currency {currency} has no pair")
        scores[currency] = scores[currency] / count
    return scores


def pair_score_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    """Score each tradable pair as base-currency strength minus quote-currency strength."""
    values: dict[str, pd.Series] = {}
    for pair in PAIRS:
        base, quote = split_pair(pair)
        values[pair] = scores[base] - scores[quote]
    return pd.DataFrame(values, index=scores.index)


def signals_for_policy(mids: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    """Generate one best-divergence signal per selected session timestamp."""
    returns = np.log(mids / mids.shift(policy.lookback_minutes)) * 10_000.0
    scores = pair_score_matrix(currency_scores(returns)).dropna(how="any")
    index = scores.index
    session_mask = (
        (index.hour == policy.decision_hour_utc)
        & (index.minute == 0)
        & (index.dayofweek < 5)
    )
    session_scores = scores.loc[session_mask]
    if session_scores.empty:
        return pd.DataFrame()
    absolute = session_scores.abs()
    chosen_pair = absolute.idxmax(axis=1)
    chosen_columns = session_scores.columns.get_indexer(chosen_pair)
    divergence = session_scores.to_numpy()[np.arange(len(session_scores)), chosen_columns]
    signal = pd.DataFrame(
        {
            "decision_timestamp": session_scores.index,
            "pair": chosen_pair.to_numpy(),
            "divergence_bps": divergence,
        }
    )
    signal = signal.loc[signal["divergence_bps"].abs() >= policy.threshold_bps].copy()
    if signal.empty:
        return signal
    signal["side"] = np.where(signal["divergence_bps"] > 0, 1, -1)
    if policy.mode == "reversal":
        signal["side"] = -signal["side"]
    signal["entry_timestamp"] = signal["decision_timestamp"] + pd.Timedelta(minutes=1)
    return signal


def simulate_trade(
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
    """Simulate one managed trade with executable bid/ask and stop-first ordering."""
    if not np.isfinite(risk_pips) or risk_pips <= 0:
        return None
    exit_deadline = entry_timestamp + pd.Timedelta(minutes=hold_minutes)
    if entry_timestamp not in frame.index or exit_deadline not in frame.index:
        return None
    entry_position = int(frame.index.get_loc(entry_timestamp))
    exit_position = int(frame.index.get_loc(exit_deadline))
    if exit_position <= entry_position:
        return None
    window = frame.iloc[entry_position : exit_position + 1]
    if len(window) != hold_minutes + 1:
        return None

    pip = Instrument.from_symbol(pair).pip_size
    risk_price = risk_pips * pip
    first = window.iloc[0]
    entry_price = float(first["ask_open"] if side == 1 else first["bid_open"])
    if side == 1:
        stop = entry_price - risk_price
        target = (
            entry_price + management.target_r * risk_price
            if management.target_r is not None
            else None
        )
    else:
        stop = entry_price + risk_price
        target = (
            entry_price - management.target_r * risk_price
            if management.target_r is not None
            else None
        )

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
        side, entry_price, close_exit, pip, commission, slippage_per_side_pips, pd.Timestamp(window.index[-1]), "timeout"
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
    exit_reason: ExitReason,
) -> ScenarioResult:
    raw = (exit_price - entry_price) * side / pip
    net = raw - commission - 2.0 * slippage_per_side_pips
    return ScenarioResult(
        raw_pips=float(raw),
        net_pips=float(net),
        exit_timestamp=exit_timestamp,
        exit_reason=exit_reason,
    )


def evaluate_policy(
    policy: Policy,
    market: dict[str, PairEvaluationData],
    mids: pd.DataFrame,
) -> list[dict[str, Any]]:
    signal = signals_for_policy(mids, policy)
    rows: list[dict[str, Any]] = []
    if signal.empty:
        return rows
    for row in signal.itertuples(index=False):
        pair = str(row.pair)
        pair_data = market[pair]
        entry_timestamp = pd.Timestamp(row.entry_timestamp)
        risk_pips = float(pair_data.risk_pips.get(entry_timestamp, np.nan))
        side = int(row.side)
        base = simulate_trade(
            pair_data.bars,
            pair=pair,
            entry_timestamp=entry_timestamp,
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
            entry_timestamp=entry_timestamp,
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
                "policy_id": policy.identifier,
                "mode": policy.mode,
                "lookback_minutes": policy.lookback_minutes,
                "decision_hour_utc": policy.decision_hour_utc,
                "threshold_bps": policy.threshold_bps,
                "max_hold_minutes": policy.max_hold_minutes,
                "management": policy.management.name,
                "decision_timestamp": pd.Timestamp(row.decision_timestamp),
                "entry_timestamp": entry_timestamp,
                "pair": pair,
                "side": "LONG" if side == 1 else "SHORT",
                "divergence_bps": float(row.divergence_bps),
                "risk_pips": risk_pips,
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


def summarize_candidates(trades: pd.DataFrame) -> pd.DataFrame:
    selection = trades.loc[trades["entry_timestamp"] < SELECTION_END].copy()
    grouped = selection.groupby("policy_id", sort=True)
    summary = grouped.agg(
        mode=("mode", "first"),
        lookback_minutes=("lookback_minutes", "first"),
        decision_hour_utc=("decision_hour_utc", "first"),
        threshold_bps=("threshold_bps", "first"),
        max_hold_minutes=("max_hold_minutes", "first"),
        management=("management", "first"),
        selection_trades=("policy_id", "size"),
        selection_pairs=("pair", "nunique"),
        selection_mean_base_pips=("net_base_pips", "mean"),
        selection_mean_stress_pips=("net_stress_pips", "mean"),
        selection_total_stress_pips=("net_stress_pips", "sum"),
        selection_median_stress_pips=("net_stress_pips", "median"),
    )
    rows: list[dict[str, Any]] = []
    for policy_id, sample in selection.groupby("policy_id", sort=True):
        share, positive, total = positive_year_share(sample, "net_stress_pips")
        rows.append(
            {
                "policy_id": policy_id,
                "selection_positive_stress_year_share": share,
                "selection_positive_stress_years": positive,
                "selection_years": total,
                "selection_top_pair": sample["pair"].value_counts().index[0],
            }
        )
    return summary.reset_index().merge(pd.DataFrame(rows), on="policy_id", how="left")


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
        sample = audit.loc[audit["policy_id"] == finalist.policy_id]
        base_lcb = lower_confidence_bound(sample["net_base_pips"], alpha)
        stress_lcb = lower_confidence_bound(sample["net_stress_pips"], alpha)
        stress_share, stress_positive_years, stress_years = positive_year_share(
            sample, "net_stress_pips"
        )
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
        top_pair = sample["pair"].value_counts().index[0] if len(sample) else None
        rows.append(
            {
                "rank": rank,
                **finalist._asdict(),
                "audit_trades": int(len(sample)),
                "audit_pairs": int(sample["pair"].nunique()) if len(sample) else 0,
                "audit_top_pair": top_pair,
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


def fast120_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if passed_count:
        return (
            "FAST120_FX_CURRENCY_STRENGTH_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"historical_audit_candidates:{passed_count}",
                "fresh_preregistered_prospective_observation_required",
            ],
        )
    if finalists_count:
        return (
            "FAST120_FX_CURRENCY_STRENGTH_FAILED_AUDIT",
            reasons + ["selection_positive_candidates_failed_2024_2026_audit"],
        )
    return (
        "FAST120_FX_CURRENCY_STRENGTH_FAILED_SELECTION",
        reasons + ["no_selection_candidate_positive_after_costs"],
    )


def run(
    *,
    m1_dir: Path = DEFAULT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    policies = policy_grid()
    market, mids = load_market(m1_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_path = output_root / "pair-quality.json"
    write_json(quality_path, [market[pair].quality for pair in PAIRS])

    trade_rows: list[dict[str, Any]] = []
    for position, policy in enumerate(policies, start=1):
        rows = evaluate_policy(policy, market, mids)
        trade_rows.extend(rows)
        if position % 25 == 0 or position == len(policies):
            sys.stdout.write(
                f"policies {position}/{len(policies)}; accumulated trades={len(trade_rows)}\n"
            )
            sys.stdout.flush()
    if not trade_rows:
        raise ValueError("no currency-strength trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "policy_id"])
    trades_path = output_root / "currency-strength-trades.parquet"
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
    status, status_reasons = fast120_status(len(passed), len(finalists))
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast120_is_historical_scout_not_strategy_approval",
            "currency_strength_policies_are_selected_on_historical_selection_sample",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast120_fx_currency_strength_scout",
        "status": status,
        "status_reasons": status_reasons,
        "m1_dir": str(m1_dir.relative_to(REPO_ROOT)),
        "pairs": list(PAIRS),
        "currencies": list(CURRENCIES),
        "policy_count": len(policies),
        "selection_start": DATA_START.isoformat(),
        "selection_end": SELECTION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "trade_rows": int(len(trades)),
        "selection_candidates": int(len(selection)),
        "eligible_finalists": int(len(finalists)),
        "audited_finalists": int(len(audit)),
        "passed_candidates": int(len(passed)),
        "winner": winner[0] if winner else None,
        "risk_model": {
            "minimum_pips": MIN_RISK_PIPS,
            "range_multiple": RISK_RANGE_MULTIPLE,
            "range_window_minutes": RISK_RANGE_WINDOW,
        },
        "costs": {
            "base": asdict(FUSION_ZERO),
            "stress": asdict(STRESS),
            "slippage_model_used": "fixed per side from scenario; stress spread multiplier applied",
        },
        "artifacts": {
            "trades": {
                "path": str(trades_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(trades_path),
            },
            "pair_quality": {
                "path": str(quality_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(quality_path),
            },
            "selection": {
                "path": str(selection_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(selection_path),
            },
            "audit": {
                "path": str(audit_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(audit_path),
            },
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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(plain_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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
