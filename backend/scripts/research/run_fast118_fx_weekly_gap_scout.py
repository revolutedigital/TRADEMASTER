"""Fast118: FX weekly-open gap reversion/continuation scout on executable M1 bars.

Research-only script. It tests whether the first tradable bars after a weekend
gap contain a pair-specific edge after bid/ask spread, commission and slippage.
It writes offline artifacts only and never touches the trading engine, database,
credentials, ledger, Demo, Testnet or LIVE accounts.
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast118" / "fx-weekly-gap-scout"

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
DATA_START = pd.Timestamp("2019-01-01T00:00:00Z")
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
AUDIT_END = pd.Timestamp("2026-09-01T00:00:00Z")
WEEKEND_GAP_MIN_HOURS = 36
FINALISTS = 50
MIN_SELECTION_TRADES = 60
MIN_AUDIT_TRADES = 30
MIN_AUDIT_YEARS = 2
MIN_AUDIT_POSITIVE_YEAR_SHARE = 2 / 3

Direction = Literal["reversion", "continuation"]
ExitReason = Literal["stop", "target", "timeout"]


@dataclass(frozen=True)
class Management:
    """One exit-management rule expressed in multiples of the weekend gap."""

    name: str
    stop_gap_multiple: float
    target_gap_fraction: float | None
    activation_gap_fraction: float | None
    trail_gap_fraction: float | None


@dataclass(frozen=True)
class Policy:
    """One pair-specific weekly-gap hypothesis."""

    identifier: str
    direction: Direction
    minimum_gap_pips: float
    entry_delay_minutes: int
    max_hold_minutes: int
    management: Management


@dataclass(frozen=True)
class WeeklyGap:
    """The first M1 bar after a weekend/market-closure gap."""

    pair: str
    previous_timestamp: pd.Timestamp
    reopen_timestamp: pd.Timestamp
    previous_mid_close: float
    reopen_mid_open: float
    gap_pips: float

    @property
    def absolute_gap_pips(self) -> float:
        return abs(self.gap_pips)


@dataclass(frozen=True)
class ScenarioResult:
    """Executable trade outcome in one cost scenario."""

    raw_pips: float
    net_pips: float
    exit_timestamp: pd.Timestamp
    exit_reason: ExitReason


@dataclass(frozen=True)
class PairEvaluation:
    """Trade rows and quality facts for one pair."""

    rows: list[dict[str, Any]]
    quality: dict[str, Any]


def policy_grid() -> tuple[Policy, ...]:
    """Return the deterministic exploratory grid for Fast118."""
    managements = (
        Management(
            "fixed_half_gap",
            stop_gap_multiple=1.0,
            target_gap_fraction=0.5,
            activation_gap_fraction=None,
            trail_gap_fraction=None,
        ),
        Management(
            "fixed_full_gap",
            stop_gap_multiple=1.0,
            target_gap_fraction=1.0,
            activation_gap_fraction=None,
            trail_gap_fraction=None,
        ),
        Management(
            "trail_after_half_gap",
            stop_gap_multiple=1.0,
            target_gap_fraction=None,
            activation_gap_fraction=0.5,
            trail_gap_fraction=0.35,
        ),
    )
    policies: list[Policy] = []
    for direction in ("reversion", "continuation"):
        for minimum_gap_pips in (2.0, 5.0, 10.0):
            for delay in (5, 15, 30):
                for hold_hours in (4, 12):
                    for management in managements:
                        identifier = (
                            f"{direction}_gap{minimum_gap_pips:g}"
                            f"_d{delay}_h{hold_hours}_{management.name}"
                        )
                        policies.append(
                            Policy(
                                identifier=identifier,
                                direction=direction,
                                minimum_gap_pips=minimum_gap_pips,
                                entry_delay_minutes=delay,
                                max_hold_minutes=hold_hours * 60,
                                management=management,
                            )
                        )
    return tuple(policies)


def load_pair_bars(m1_dir: Path, pair: str) -> pd.DataFrame:
    """Load one pair's local bid/ask M1 bars, excluding unvalidated Sep/2026 by default."""
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
    """Widen each OHLC bid/ask spread symmetrically around its midpoint."""
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
    """Approximate conversion rates from the same M1 source used by the scout."""
    mids: dict[str, float] = {}
    for pair in MAJOR_RATE_PAIRS:
        paths = sorted(m1_dir.glob(f"{pair}_M1_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"missing M1 rate pair {pair}")
        values: list[np.ndarray] = []
        for path in paths:
            frame = pd.read_parquet(path, columns=["bid_close", "ask_close"])
            frame = frame.loc[(frame.index >= DATA_START) & (frame.index < AUDIT_END)]
            values.append((0.5 * (frame["bid_close"] + frame["ask_close"])).to_numpy())
        mids[pair] = float(np.median(np.concatenate(values)))
    return ConversionRates(mids)


def commission_pips(pair: str, price: float, rates: ConversionRates, scenario: CostScenario) -> float:
    """Round-trip commission in pips for one standard lot."""
    return commission_round_trip_pips(
        Instrument.from_symbol(pair),
        price,
        rates,
        base_currency_per_lot_per_side=scenario.commission_base_per_lot_per_side,
    )


def find_weekly_gaps(frame: pd.DataFrame, pair: str) -> tuple[WeeklyGap, ...]:
    """Find first bars after a weekend/closure gap using only timestamp spacing and OHLC."""
    if not frame.index.is_monotonic_increasing:
        raise ValueError("M1 frame index must be sorted")
    instrument = Instrument.from_symbol(pair)
    deltas = frame.index.to_series().diff()
    reopen_positions = np.flatnonzero(deltas > pd.Timedelta(hours=WEEKEND_GAP_MIN_HOURS))
    gaps: list[WeeklyGap] = []
    for position in reopen_positions:
        if position <= 0:
            continue
        previous = frame.iloc[position - 1]
        reopen = frame.iloc[position]
        previous_mid_close = float(0.5 * (previous["bid_close"] + previous["ask_close"]))
        reopen_mid_open = float(0.5 * (reopen["bid_open"] + reopen["ask_open"]))
        gap_pips = (reopen_mid_open - previous_mid_close) / instrument.pip_size
        if not np.isfinite(gap_pips) or gap_pips == 0.0:
            continue
        gaps.append(
            WeeklyGap(
                pair=pair,
                previous_timestamp=pd.Timestamp(frame.index[position - 1]),
                reopen_timestamp=pd.Timestamp(frame.index[position]),
                previous_mid_close=previous_mid_close,
                reopen_mid_open=reopen_mid_open,
                gap_pips=float(gap_pips),
            )
        )
    return tuple(gaps)


def side_for_gap(gap: WeeklyGap, policy: Policy) -> int:
    """Return 1 for long and -1 for short."""
    gap_direction = 1 if gap.gap_pips > 0 else -1
    return -gap_direction if policy.direction == "reversion" else gap_direction


def simulate_scenario(
    frame: pd.DataFrame,
    gap: WeeklyGap,
    policy: Policy,
    *,
    side: int,
    commission: float,
    slippage_per_side_pips: float,
) -> ScenarioResult | None:
    """Simulate one weekly-gap trade with executable bid/ask and stop-first ordering."""
    if gap.absolute_gap_pips < policy.minimum_gap_pips:
        return None
    instrument = Instrument.from_symbol(gap.pair)
    pip = instrument.pip_size
    entry_timestamp = gap.reopen_timestamp + pd.Timedelta(minutes=policy.entry_delay_minutes)
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
    gap_price = gap.absolute_gap_pips * pip
    stop_distance = policy.management.stop_gap_multiple * gap_price
    target_distance = (
        policy.management.target_gap_fraction * gap_price
        if policy.management.target_gap_fraction is not None
        else None
    )
    if side == 1:
        stop = entry_price - stop_distance
        target = entry_price + target_distance if target_distance is not None else None
    else:
        stop = entry_price + stop_distance
        target = entry_price - target_distance if target_distance is not None else None

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
                    side,
                    entry_price,
                    open_exit,
                    pip,
                    commission,
                    slippage_per_side_pips,
                    timestamp,
                    "stop",
                )
            if adverse <= stop:
                return scenario_result(
                    side,
                    entry_price,
                    stop,
                    pip,
                    commission,
                    slippage_per_side_pips,
                    timestamp,
                    "stop",
                )
            if target is not None and favorable >= target:
                return scenario_result(
                    side,
                    entry_price,
                    target,
                    pip,
                    commission,
                    slippage_per_side_pips,
                    timestamp,
                    "target",
                )
            best_favorable_pips = max(best_favorable_pips, (favorable - entry_price) / pip)
            if trail_is_active(policy.management, best_favorable_pips, gap.absolute_gap_pips):
                trail = favorable - float(policy.management.trail_gap_fraction) * gap_price
                stop = max(stop, entry_price, trail)
        else:
            open_exit = float(bar.ask_open)
            adverse = float(bar.ask_high)
            favorable = float(bar.ask_low)
            close_exit = float(bar.ask_close)
            if open_exit >= stop:
                return scenario_result(
                    side,
                    entry_price,
                    open_exit,
                    pip,
                    commission,
                    slippage_per_side_pips,
                    timestamp,
                    "stop",
                )
            if adverse >= stop:
                return scenario_result(
                    side,
                    entry_price,
                    stop,
                    pip,
                    commission,
                    slippage_per_side_pips,
                    timestamp,
                    "stop",
                )
            if target is not None and favorable <= target:
                return scenario_result(
                    side,
                    entry_price,
                    target,
                    pip,
                    commission,
                    slippage_per_side_pips,
                    timestamp,
                    "target",
                )
            best_favorable_pips = max(best_favorable_pips, (entry_price - favorable) / pip)
            if trail_is_active(policy.management, best_favorable_pips, gap.absolute_gap_pips):
                trail = favorable + float(policy.management.trail_gap_fraction) * gap_price
                stop = min(stop, entry_price, trail)

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


def trail_is_active(management: Management, favorable_pips: float, gap_pips: float) -> bool:
    return (
        management.activation_gap_fraction is not None
        and management.trail_gap_fraction is not None
        and favorable_pips >= management.activation_gap_fraction * gap_pips
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


def evaluate_pair(
    pair: str,
    m1_dir: Path,
    policies: tuple[Policy, ...],
    rates: ConversionRates,
) -> PairEvaluation:
    """Evaluate every policy for one pair and return trade-level rows."""
    frame = load_pair_bars(m1_dir, pair)
    stress_frame = widen_quotes(frame, STRESS.spread_multiplier)
    median_price = float(np.median(0.5 * (frame["bid_close"] + frame["ask_close"])))
    base_commission = commission_pips(pair, median_price, rates, FUSION_ZERO)
    stress_commission = commission_pips(pair, median_price, rates, STRESS)
    gaps = find_weekly_gaps(frame, pair)
    rows: list[dict[str, Any]] = []
    for gap in gaps:
        for policy in policies:
            side = side_for_gap(gap, policy)
            base = simulate_scenario(
                frame,
                gap,
                policy,
                side=side,
                commission=base_commission,
                slippage_per_side_pips=FUSION_ZERO.slippage_pips,
            )
            if base is None:
                continue
            stress = simulate_scenario(
                stress_frame,
                gap,
                policy,
                side=side,
                commission=stress_commission,
                slippage_per_side_pips=STRESS.slippage_pips,
            )
            if stress is None:
                continue
            rows.append(
                {
                    "candidate_id": f"{pair}:{policy.identifier}",
                    "policy_id": policy.identifier,
                    "pair": pair,
                    "gap_reopen_timestamp": gap.reopen_timestamp,
                    "previous_timestamp": gap.previous_timestamp,
                    "side": "LONG" if side == 1 else "SHORT",
                    "direction": policy.direction,
                    "minimum_gap_pips": policy.minimum_gap_pips,
                    "entry_delay_minutes": policy.entry_delay_minutes,
                    "max_hold_minutes": policy.max_hold_minutes,
                    "management": policy.management.name,
                    "gap_pips": gap.gap_pips,
                    "absolute_gap_pips": gap.absolute_gap_pips,
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
                }
            )
    quality = {
        "pair": pair,
        "raw_bars": int(frame.attrs.get("raw_bars", len(frame))),
        "usable_bars": int(len(frame)),
        "dropped_crossed_quote_bars": int(frame.attrs.get("dropped_crossed_quote_bars", 0)),
        "weekly_gaps": int(len(gaps)),
        "trade_rows": int(len(rows)),
        "first_bar": frame.index.min().isoformat() if len(frame) else None,
        "last_bar": frame.index.max().isoformat() if len(frame) else None,
    }
    sys.stdout.write(
        f"{pair}: {len(gaps)} weekly gaps -> {len(rows)} policy trades"
        f" (dropped_crossed={quality['dropped_crossed_quote_bars']})\n"
    )
    sys.stdout.flush()
    return PairEvaluation(rows=rows, quality=quality)


def lower_confidence_bound(values: pd.Series, alpha: float) -> float:
    """One-sided t lower bound over weekly observations."""
    sample = values.astype("float64").dropna().to_numpy()
    if len(sample) < 2:
        return float("nan")
    standard_error = float(sample.std(ddof=1) / np.sqrt(len(sample)))
    critical = float(t.ppf(1 - alpha, df=len(sample) - 1))
    return float(sample.mean() - critical * standard_error)


def positive_year_share(sample: pd.DataFrame, column: str) -> tuple[float, int, int]:
    """Share of calendar years with positive mean in a candidate sample."""
    if sample.empty:
        return float("nan"), 0, 0
    by_year = sample.assign(year=sample["gap_reopen_timestamp"].dt.year).groupby("year")[column].mean()
    positive = int((by_year > 0).sum())
    total = int(len(by_year))
    return float(positive / total) if total else float("nan"), positive, total


def summarize_candidates(trades: pd.DataFrame) -> pd.DataFrame:
    """Selection metrics for all pair-specific policies before 2024."""
    selection = trades.loc[trades["gap_reopen_timestamp"] < SELECTION_END].copy()
    grouped = selection.groupby("candidate_id", sort=True)
    summary = grouped.agg(
        pair=("pair", "first"),
        policy_id=("policy_id", "first"),
        direction=("direction", "first"),
        minimum_gap_pips=("minimum_gap_pips", "first"),
        entry_delay_minutes=("entry_delay_minutes", "first"),
        max_hold_minutes=("max_hold_minutes", "first"),
        management=("management", "first"),
        selection_trades=("candidate_id", "size"),
        selection_mean_base_pips=("net_base_pips", "mean"),
        selection_mean_stress_pips=("net_stress_pips", "mean"),
        selection_total_stress_pips=("net_stress_pips", "sum"),
        selection_median_stress_pips=("net_stress_pips", "median"),
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
    year_summary = pd.DataFrame(rows)
    return summary.reset_index().merge(year_summary, on="candidate_id", how="left")


def select_finalists(selection: pd.DataFrame) -> pd.DataFrame:
    """Pick finalists using selection data only."""
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
    """Audit selection finalists on 2024-01-01..2026-08-31."""
    alpha = 0.05 / max(len(finalists), 1)
    audit = trades.loc[
        (trades["gap_reopen_timestamp"] >= SELECTION_END)
        & (trades["gap_reopen_timestamp"] < AUDIT_END)
    ].copy()
    rows: list[dict[str, Any]] = []
    for rank, finalist in enumerate(finalists.itertuples(index=False), start=1):
        sample = audit.loc[audit["candidate_id"] == finalist.candidate_id]
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


def fast118_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    """Status is deliberately never an execution approval."""
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if passed_count:
        return (
            "FAST118_FX_WEEKLY_GAP_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"historical_audit_candidates:{passed_count}",
                "fresh_preregistered_prospective_observation_required",
            ],
        )
    if finalists_count:
        return (
            "FAST118_FX_WEEKLY_GAP_FAILED_AUDIT",
            reasons + ["selection_positive_candidates_failed_2024_2026_audit"],
        )
    return (
        "FAST118_FX_WEEKLY_GAP_FAILED_SELECTION",
        reasons + ["no_selection_candidate_positive_after_costs"],
    )


def run(
    *,
    m1_dir: Path = DEFAULT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Run Fast118 and write auditable artifacts."""
    policies = policy_grid()
    rates = median_rates_from_m1(m1_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    trade_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        evaluation = evaluate_pair(pair, m1_dir, policies, rates)
        trade_rows.extend(evaluation.rows)
        quality_rows.append(evaluation.quality)
    if not trade_rows:
        raise ValueError("no weekly-gap trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(
        ["gap_reopen_timestamp", "pair", "policy_id"]
    )
    trades_path = output_root / "weekly-gap-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)

    quality_path = output_root / "pair-quality.json"
    write_json(quality_path, quality_rows)

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
    status, status_reasons = fast118_status(len(passed), len(finalists))
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast118_is_historical_scout_not_strategy_approval",
            "pair_specific_policies_are_selected_on_historical_selection_sample",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast118_fx_weekly_gap_scout",
        "status": status,
        "status_reasons": status_reasons,
        "m1_dir": str(m1_dir.relative_to(REPO_ROOT)),
        "pairs": list(PAIRS),
        "policy_count_per_pair": len(policies),
        "candidate_count": len(PAIRS) * len(policies),
        "selection_start": DATA_START.isoformat(),
        "selection_end": SELECTION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "weekly_gap_min_hours": WEEKEND_GAP_MIN_HOURS,
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
            "pair_quality": {
                "path": str(quality_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(quality_path),
            },
            "selection": {
                "path": str(selection_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(selection_path),
            },
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
