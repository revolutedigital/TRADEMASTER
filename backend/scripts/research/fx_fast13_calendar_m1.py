"""Round 13: restricted official macro-calendar drift/reversal on executable M1 FX bars.

Research-only script. It reads local official event times and local bid/ask M1 bars, writes offline
reports, and never touches the trading engine, the database, credentials, or an exchange account.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import t

from app.fx.instruments import ConversionRates, Instrument
from app.fx.sim.costs import FUSION_ZERO, STRESS, commission_round_trip_pips
from scripts.research.fx_fast4_materialize import _sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALENDAR = REPO_ROOT / "backend" / "data" / "raw" / "calendar" / "events.parquet"
DEFAULT_M1_DIR = REPO_ROOT / "backend" / "data" / "raw" / "fx_m1_hd"
DEFAULT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast13"

ALLOWED_EVENTS = frozenset(
    {
        "NFP",
        "CPI US",
        "FOMC rate decision",
        "ECB rate decision",
        "BoE rate decision",
    }
)
PAIRS_BY_CURRENCY = {
    "USD": ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"),
    "EUR": ("EURUSD", "EURGBP", "EURJPY"),
    "GBP": ("GBPUSD", "EURGBP", "GBPJPY"),
}
MAJOR_RATE_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY")
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
AUDIT_END = pd.Timestamp("2026-09-01T00:00:00Z")
IMPULSE_MINUTES = 5
BASELINE_MINUTES = 240
PRE_EVENT_GAP_MINUTES = 10
MIN_SELECTION_TRADES = 120
MIN_AUDIT_TRADES = 40
MIN_AUDIT_EVENTS = 25
FINALISTS = 25
MIN_RISK_PIPS = 0.5

Direction = Literal["drift", "reversal"]
ManagementName = Literal["fixed_1_5r", "trail_1r_0_75r", "trail_0_75r_0_5r"]
ExitReason = Literal["stop", "target", "timeout"]


@dataclass(frozen=True)
class Management:
    """One frozen exit-management rule, expressed in multiples of initial risk R."""

    name: ManagementName
    target_r: float | None
    activation_r: float | None
    trail_r: float | None


@dataclass(frozen=True)
class Policy:
    """One frozen calendar strategy hypothesis."""

    identifier: str
    direction: Direction
    threshold: float
    entry_delay_minutes: int
    max_hold_minutes: int
    management: Management


@dataclass(frozen=True)
class PreparedEvent:
    """A tradable event for one pair after causal impulse and baseline checks."""

    event_timestamp: pd.Timestamp
    event_name: str
    currency: str
    pair: str
    impulse_pips: float
    baseline_range_pips: float
    risk_pips: float


@dataclass(frozen=True)
class ScenarioResult:
    """Net pips for one executable scenario."""

    raw_pips: float
    net_pips: float
    exit_timestamp: pd.Timestamp
    exit_reason: ExitReason


@dataclass(frozen=True)
class PairEvaluation:
    """Trade rows plus the data-quality facts needed to audit one pair."""

    rows: list[dict[str, object]]
    quality: dict[str, object]


def policy_grid() -> tuple[Policy, ...]:
    """Return the preregistered 72-hypothesis grid in deterministic order."""
    managements = (
        Management("fixed_1_5r", target_r=1.5, activation_r=None, trail_r=None),
        Management("trail_1r_0_75r", target_r=None, activation_r=1.0, trail_r=0.75),
        Management("trail_0_75r_0_5r", target_r=None, activation_r=0.75, trail_r=0.5),
    )
    policies: list[Policy] = []
    for direction in ("drift", "reversal"):
        for threshold in (1.5, 2.0):
            for delay in (5, 10, 15):
                for hold in (30, 60, 120):
                    for management in managements:
                        identifier = (
                            f"{direction}_thr{threshold:g}_d{delay}_h{hold}_{management.name}"
                        )
                        policies.append(
                            Policy(
                                identifier=identifier,
                                direction=direction,
                                threshold=threshold,
                                entry_delay_minutes=delay,
                                max_hold_minutes=hold,
                                management=management,
                            )
                        )
    return tuple(policies)


def load_calendar(path: Path) -> pd.DataFrame:
    """Load the closed event universe from the official-source calendar artifact."""
    frame = pd.read_parquet(path)
    expected = {"timestamp_utc", "currency", "event"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"calendar missing columns: {sorted(missing)}")
    filtered = frame.loc[
        frame["event"].isin(ALLOWED_EVENTS) & frame["currency"].isin(PAIRS_BY_CURRENCY)
    ].copy()
    filtered["timestamp_utc"] = pd.to_datetime(filtered["timestamp_utc"], utc=True)
    filtered = filtered.loc[
        (filtered["timestamp_utc"] < AUDIT_END) & (filtered["timestamp_utc"] >= "2019-01-01")
    ]
    return filtered.sort_values("timestamp_utc").reset_index(drop=True)


def load_pair_bars(m1_dir: Path, pair: str) -> pd.DataFrame:
    """Read one pair's executable M1 bid/ask bars."""
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
    """Double or otherwise widen each bid/ask column symmetrically around its mid."""
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


def median_rates_from_m1(m1_dir: Path) -> ConversionRates:
    """Approximate conversion rates from the same M1 source used by the simulation."""
    mids: dict[str, float] = {}
    for pair in MAJOR_RATE_PAIRS:
        paths = sorted(m1_dir.glob(f"{pair}_M1_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"missing M1 rate pair {pair}")
        values: list[np.ndarray] = []
        for path in paths:
            frame = pd.read_parquet(path, columns=["bid_close", "ask_close"])
            values.append((0.5 * (frame["bid_close"] + frame["ask_close"])).to_numpy())
        mids[pair] = float(np.median(np.concatenate(values)))
    return ConversionRates(mids)


def commission_pips(pair: str, price: float, rates: ConversionRates, base_per_side: float) -> float:
    """Round-trip commission in pips for one standard lot."""
    return commission_round_trip_pips(
        Instrument.from_symbol(pair),
        price,
        rates,
        base_currency_per_lot_per_side=base_per_side,
    )


def _mid_open(row: pd.Series) -> float:
    return float(0.5 * (row["bid_open"] + row["ask_open"]))


def _mid_close(row: pd.Series) -> float:
    return float(0.5 * (row["bid_close"] + row["ask_close"]))


def _baseline_range_pips(frame: pd.DataFrame, event_timestamp: pd.Timestamp, pip: float) -> float | None:
    start = event_timestamp - pd.Timedelta(minutes=BASELINE_MINUTES + PRE_EVENT_GAP_MINUTES)
    end = event_timestamp - pd.Timedelta(minutes=PRE_EVENT_GAP_MINUTES + 1)
    window = frame.loc[start:end]
    if len(window) < BASELINE_MINUTES // 2:
        return None
    ranges = (
        (0.5 * (window["ask_high"] + window["bid_high"]))
        - (0.5 * (window["ask_low"] + window["bid_low"]))
    ) / pip
    ranges = ranges[np.isfinite(ranges) & (ranges > 0)]
    if ranges.empty:
        return None
    return float(ranges.median())


def prepare_events_for_pair(
    calendar: pd.DataFrame,
    frame: pd.DataFrame,
    pair: str,
) -> tuple[PreparedEvent, ...]:
    """Build causal event descriptors for the currencies that can move `pair`."""
    instrument = Instrument.from_symbol(pair)
    event_currencies = {instrument.base, instrument.quote} & set(PAIRS_BY_CURRENCY)
    rows = calendar.loc[calendar["currency"].isin(event_currencies)]
    prepared: list[PreparedEvent] = []
    for row in rows.itertuples(index=False):
        event_timestamp = pd.Timestamp(row.timestamp_utc).tz_convert("UTC")
        impulse_end = event_timestamp + pd.Timedelta(minutes=IMPULSE_MINUTES - 1)
        if event_timestamp not in frame.index or impulse_end not in frame.index:
            continue
        baseline = _baseline_range_pips(frame, event_timestamp, instrument.pip_size)
        if baseline is None:
            continue
        start = _mid_open(frame.loc[event_timestamp])
        end = _mid_close(frame.loc[impulse_end])
        impulse = (end - start) / instrument.pip_size
        if not np.isfinite(impulse) or impulse == 0:
            continue
        risk = max(MIN_RISK_PIPS, baseline, 0.5 * abs(float(impulse)))
        prepared.append(
            PreparedEvent(
                event_timestamp=event_timestamp,
                event_name=str(row.event),
                currency=str(row.currency),
                pair=pair,
                impulse_pips=float(impulse),
                baseline_range_pips=float(baseline),
                risk_pips=float(risk),
            )
        )
    return tuple(prepared)


def side_for_event(event: PreparedEvent, policy: Policy) -> int:
    """Return 1 for long and -1 for short."""
    impulse_side = 1 if event.impulse_pips > 0 else -1
    return impulse_side if policy.direction == "drift" else -impulse_side


def simulate_scenario(
    frame: pd.DataFrame,
    event: PreparedEvent,
    policy: Policy,
    *,
    side: int,
    commission: float,
    slippage_per_side_pips: float,
) -> ScenarioResult | None:
    """Simulate one event with executable bid/ask and conservative intrabar ordering."""
    if abs(event.impulse_pips) < policy.threshold * event.baseline_range_pips:
        return None
    instrument = Instrument.from_symbol(event.pair)
    pip = instrument.pip_size
    entry_timestamp = event.event_timestamp + pd.Timedelta(minutes=policy.entry_delay_minutes)
    exit_deadline = entry_timestamp + pd.Timedelta(minutes=policy.max_hold_minutes)
    if entry_timestamp not in frame.index or exit_deadline not in frame.index:
        return None
    entry_position = int(frame.index.get_loc(entry_timestamp))
    exit_position = int(frame.index.get_loc(exit_deadline))
    if exit_position <= entry_position:
        return None
    window = frame.iloc[entry_position : exit_position + 1]
    if len(window) != policy.max_hold_minutes + 1:
        return None

    first = window.iloc[0]
    entry_price = float(first["ask_open"] if side == 1 else first["bid_open"])
    risk_price = event.risk_pips * pip
    if side == 1:
        stop = entry_price - risk_price
        target = (
            entry_price + policy.management.target_r * risk_price
            if policy.management.target_r is not None
            else None
        )
    else:
        stop = entry_price + risk_price
        target = (
            entry_price - policy.management.target_r * risk_price
            if policy.management.target_r is not None
            else None
        )

    best_favorable_pips = 0.0
    for bar in window.itertuples():
        timestamp = pd.Timestamp(bar.Index)
        if side == 1:
            open_exit = float(bar.bid_open)
            adverse = float(bar.bid_low)
            favorable = float(bar.bid_high)
            close_exit = float(bar.bid_close)
            if open_exit <= stop:
                return _scenario_result(
                    side, entry_price, open_exit, pip, commission, slippage_per_side_pips,
                    timestamp, "stop",
                )
            if adverse <= stop:
                return _scenario_result(
                    side, entry_price, stop, pip, commission, slippage_per_side_pips,
                    timestamp, "stop",
                )
            if target is not None and favorable >= target:
                return _scenario_result(
                    side, entry_price, target, pip, commission, slippage_per_side_pips,
                    timestamp, "target",
                )
            best_favorable_pips = max(best_favorable_pips, (favorable - entry_price) / pip)
            if _trail_is_active(policy.management, best_favorable_pips, event.risk_pips):
                trail = favorable - float(policy.management.trail_r) * risk_price
                stop = max(stop, entry_price, trail)
        else:
            open_exit = float(bar.ask_open)
            adverse = float(bar.ask_high)
            favorable = float(bar.ask_low)
            close_exit = float(bar.ask_close)
            if open_exit >= stop:
                return _scenario_result(
                    side, entry_price, open_exit, pip, commission, slippage_per_side_pips,
                    timestamp, "stop",
                )
            if adverse >= stop:
                return _scenario_result(
                    side, entry_price, stop, pip, commission, slippage_per_side_pips,
                    timestamp, "stop",
                )
            if target is not None and favorable <= target:
                return _scenario_result(
                    side, entry_price, target, pip, commission, slippage_per_side_pips,
                    timestamp, "target",
                )
            best_favorable_pips = max(best_favorable_pips, (entry_price - favorable) / pip)
            if _trail_is_active(policy.management, best_favorable_pips, event.risk_pips):
                trail = favorable + float(policy.management.trail_r) * risk_price
                stop = min(stop, entry_price, trail)

    return _scenario_result(
        side,
        entry_price,
        close_exit,
        pip,
        commission,
        slippage_per_side_pips,
        pd.Timestamp(window.index[-1]),
        "timeout",
    )


def _trail_is_active(management: Management, favorable_pips: float, risk_pips: float) -> bool:
    return (
        management.activation_r is not None
        and management.trail_r is not None
        and favorable_pips >= management.activation_r * risk_pips
    )


def _scenario_result(
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


def evaluate_pair(
    calendar: pd.DataFrame,
    pair: str,
    m1_dir: Path,
    policies: tuple[Policy, ...],
    rates: ConversionRates,
) -> PairEvaluation:
    """Evaluate every frozen policy for one pair and return trade-level rows."""
    frame = load_pair_bars(m1_dir, pair)
    stress_frame = widen_quotes(frame, STRESS.spread_multiplier)
    instrument = Instrument.from_symbol(pair)
    median_price = float(np.median(0.5 * (frame["bid_close"] + frame["ask_close"])))
    base_commission = commission_pips(
        pair, median_price, rates, FUSION_ZERO.commission_base_per_lot_per_side
    )
    stress_commission = commission_pips(
        pair, median_price, rates, STRESS.commission_base_per_lot_per_side
    )
    events = prepare_events_for_pair(calendar, frame, pair)
    rows: list[dict[str, object]] = []
    for event in events:
        for policy in policies:
            side = side_for_event(event, policy)
            base = simulate_scenario(
                frame,
                event,
                policy,
                side=side,
                commission=base_commission,
                slippage_per_side_pips=FUSION_ZERO.slippage_pips,
            )
            if base is None:
                continue
            stress = simulate_scenario(
                stress_frame,
                event,
                policy,
                side=side,
                commission=stress_commission,
                slippage_per_side_pips=STRESS.slippage_pips,
            )
            if stress is None:
                continue
            rows.append(
                {
                    "policy_id": policy.identifier,
                    "pair": pair,
                    "event_timestamp": event.event_timestamp,
                    "event_name": event.event_name,
                    "currency": event.currency,
                    "side": "LONG" if side == 1 else "SHORT",
                    "impulse_pips": event.impulse_pips,
                    "baseline_range_pips": event.baseline_range_pips,
                    "risk_pips": event.risk_pips,
                    "raw_base_pips": base.raw_pips,
                    "net_base_pips": base.net_pips,
                    "exit_base_timestamp": base.exit_timestamp,
                    "exit_base_reason": base.exit_reason,
                    "raw_stress_pips": stress.raw_pips,
                    "net_stress_pips": stress.net_pips,
                    "exit_stress_timestamp": stress.exit_timestamp,
                    "exit_stress_reason": stress.exit_reason,
                    "base_commission_pips": base_commission,
                    "stress_commission_pips": stress_commission,
                    "pip_size": instrument.pip_size,
                }
            )
    quality = {
        "pair": pair,
        "raw_bars": int(frame.attrs.get("raw_bars", len(frame))),
        "usable_bars": int(len(frame)),
        "dropped_crossed_quote_bars": int(frame.attrs.get("dropped_crossed_quote_bars", 0)),
        "prepared_events": int(len(events)),
        "trades": int(len(rows)),
    }
    sys.stdout.write(
        f"{pair}: {len(events)} prepared events -> {len(rows)} trades"
        f" (dropped_crossed={quality['dropped_crossed_quote_bars']})\n"
    )
    sys.stdout.flush()
    return PairEvaluation(rows=rows, quality=quality)


def lower_bound_by_event(values: pd.Series, clusters: pd.Series, alpha: float) -> tuple[float, int]:
    """One-sided clustered lower bound, clustering multiple pair trades by official event."""
    realized = values.astype("float64").to_numpy()
    if len(realized) == 0:
        return float("nan"), 0
    identifiers, inverse = np.unique(clusters.astype("int64").to_numpy(), return_inverse=True)
    cluster_count = len(identifiers)
    mean = float(realized.mean())
    if cluster_count < 2:
        return float("nan"), cluster_count
    scores = np.bincount(inverse, weights=realized - mean)
    variance = cluster_count / (cluster_count - 1) * float(np.square(scores).sum()) / len(realized) ** 2
    critical = float(t.ppf(1 - alpha, df=cluster_count - 1))
    return mean - critical * np.sqrt(max(variance, 0.0)), cluster_count


def summarize_candidates(trades: pd.DataFrame) -> pd.DataFrame:
    """Selection metrics for all policies."""
    selection = trades.loc[trades["event_timestamp"] < SELECTION_END].copy()
    grouped = selection.groupby("policy_id", sort=True)
    summary = grouped.agg(
        selection_trades=("policy_id", "size"),
        selection_events=("event_timestamp", "nunique"),
        selection_pairs=("pair", "nunique"),
        selection_mean_base_pips=("net_base_pips", "mean"),
        selection_mean_stress_pips=("net_stress_pips", "mean"),
        selection_median_stress_pips=("net_stress_pips", "median"),
    )
    return summary.reset_index()


def audit_finalists(trades: pd.DataFrame, finalists: pd.DataFrame) -> pd.DataFrame:
    """Audit the frozen top policies on 2024-01-01..2026-08-31."""
    alpha = 0.05 / max(len(finalists), 1)
    rows: list[dict[str, object]] = []
    audit = trades.loc[
        (trades["event_timestamp"] >= SELECTION_END) & (trades["event_timestamp"] < AUDIT_END)
    ].copy()
    audit["event_cluster"] = audit["event_timestamp"].astype("int64")
    for rank, policy in enumerate(finalists.itertuples(index=False), start=1):
        sample = audit.loc[audit["policy_id"] == policy.policy_id]
        base_lcb, base_clusters = lower_bound_by_event(
            sample["net_base_pips"], sample["event_cluster"], alpha
        )
        stress_lcb, stress_clusters = lower_bound_by_event(
            sample["net_stress_pips"], sample["event_cluster"], alpha
        )
        mean_base = float(sample["net_base_pips"].mean()) if len(sample) else float("nan")
        mean_stress = float(sample["net_stress_pips"].mean()) if len(sample) else float("nan")
        clusters = min(base_clusters, stress_clusters)
        passed = (
            len(sample) >= MIN_AUDIT_TRADES
            and clusters >= MIN_AUDIT_EVENTS
            and mean_base > 0
            and mean_stress > 0
            and base_lcb > 0
            and stress_lcb > 0
        )
        rows.append(
            {
                "rank": rank,
                **policy._asdict(),
                "audit_trades": int(len(sample)),
                "audit_events": int(clusters),
                "audit_pairs": int(sample["pair"].nunique()) if len(sample) else 0,
                "audit_mean_base_pips": mean_base,
                "audit_mean_stress_pips": mean_stress,
                "audit_lower_base_pips": base_lcb,
                "audit_lower_stress_pips": stress_lcb,
                "bonferroni_alpha": alpha,
                "pilot_gate": bool(passed),
            }
        )
    return pd.DataFrame(rows)


def run(
    *,
    calendar_path: Path = DEFAULT_CALENDAR,
    m1_dir: Path = DEFAULT_M1_DIR,
    root: Path = DEFAULT_ROOT,
) -> dict[str, object]:
    """Run the preregistered round and write auditable artifacts."""
    policies = policy_grid()
    calendar = load_calendar(calendar_path)
    rates = median_rates_from_m1(m1_dir)
    pairs = sorted({pair for pairs in PAIRS_BY_CURRENCY.values() for pair in pairs})
    trade_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for pair in pairs:
        evaluation = evaluate_pair(calendar, pair, m1_dir, policies, rates)
        trade_rows.extend(evaluation.rows)
        quality_rows.append(evaluation.quality)
    if not trade_rows:
        raise ValueError("no trades generated")

    root.mkdir(parents=True, exist_ok=True)
    trades = pd.DataFrame(trade_rows).sort_values(["event_timestamp", "pair", "policy_id"])
    trades_path = root / "calendar-m1-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)
    quality_path = root / "pair-quality.json"
    quality_path.write_text(json.dumps(quality_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selection = summarize_candidates(trades)
    eligible = selection.loc[selection["selection_trades"] >= MIN_SELECTION_TRADES].copy()
    ranked = eligible.sort_values(
        ["selection_mean_stress_pips", "selection_mean_base_pips", "selection_trades"],
        ascending=[False, False, False],
    ).head(FINALISTS)
    audit = audit_finalists(trades, ranked)
    selection_path = root / "selection-metrics.csv"
    audit_path = root / "audited-finalists.csv"
    selection.to_csv(selection_path, index=False)
    audit.to_csv(audit_path, index=False)

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    winner = (
        passed.sort_values("audit_mean_stress_pips", ascending=False).head(1).to_dict("records")
        if not passed.empty
        else []
    )
    report = {
        "kind": "round13_official_calendar_m1_fx",
        "orders_sent": False,
        "calendar_path": str(calendar_path.relative_to(REPO_ROOT)),
        "m1_dir": str(m1_dir.relative_to(REPO_ROOT)),
        "events": int(len(calendar)),
        "policies": len(policies),
        "pairs": pairs,
        "selection_end": SELECTION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "trades": int(len(trades)),
        "eligible_policies": int(len(eligible)),
        "audited_policies": int(len(audit)),
        "passed_policies": int(audit["pilot_gate"].sum()) if not audit.empty else 0,
        "winner": winner[0] if winner else None,
        "status": "p0_pass" if winner else "calendar_m1_rejected",
        "artifacts": {
            "trades": {
                "path": str(trades_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(trades_path),
            },
            "selection": {
                "path": str(selection_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(selection_path),
            },
            "audit": {
                "path": str(audit_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(audit_path),
            },
            "pair_quality": {
                "path": str(quality_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(quality_path),
            },
        },
        "costs": {
            "base": asdict(FUSION_ZERO),
            "stress": asdict(STRESS),
            "slippage_model_used": "fixed per side from scenario; stress spread multiplier applied",
        },
    }
    report_path = root / "calendar-m1-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        f"round13: {len(trades)} trades, {len(eligible)} eligible, "
        f"{report['passed_policies']} passed, status={report['status']}, "
        f"sha256={_sha256(report_path)}\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    arguments = parser.parse_args(argv)
    run(calendar_path=arguments.calendar, m1_dir=arguments.m1_dir, root=arguments.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
