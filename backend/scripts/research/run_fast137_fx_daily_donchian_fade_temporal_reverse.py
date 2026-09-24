"""Fast137: temporal-reverse audit for FX daily Donchian-fade policies.

Fast133 selected daily Donchian-fade risk candidates from the 2019-2024 slice
and audited them in 2024-2026. Fast137 reverses the temporal pressure: select
the same pair/entry/risk family only on the local pre-2019 dataset, then audit
the selected finalists on the current 2019-2026 dataset. This is historical
research only and never touches trading engine, credentials, ledger, Demo,
Testnet or LIVE accounts.
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

from app.fx.instruments import ConversionRates, Instrument  # noqa: E402
from app.fx.sim.costs import FUSION_ZERO, STRESS  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import (  # noqa: E402
    plain_json,
    write_json,
)
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    AUDIT_END,
    DEFAULT_M1_DIR,
    MAJOR_RATE_PAIRS,
    PAIRS,
    PairData,
    commission_pips,
    load_market,
    mid_ohlc,
    sha256_file,
    widen_quotes,
)
from scripts.research.run_fast129_fx_daily_donchian_fade_scout import (  # noqa: E402
    policy_grid as entry_policy_grid,
)
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    lower_confidence_bound,
    positive_year_share,
    risk_policy_grid,
)
from scripts.research.run_fast133_fx_daily_donchian_fade_global_risk import (  # noqa: E402
    enrich_signals_with_daily_atr,
)
from scripts.research.run_fast135_nzdusd_donchian_fade_fixed1r_temporal_audit import (  # noqa: E402
    DEFAULT_PRE_M1_DIR,
    raw_pair_frame,
)

DEFAULT_CURRENT_M1_DIR = DEFAULT_M1_DIR
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast137" / "fx-daily-donchian-fade-temporal-reverse"
)

PRE_SAMPLE_LABEL = "pre_2011_2018_selection"
CURRENT_SAMPLE_LABEL = "current_2019_2026_audit"
FINALISTS = 100
MIN_SELECTION_TRADES = 30
MIN_SELECTION_YEARS = 3
MIN_SELECTION_POSITIVE_YEAR_SHARE = 0.50
MIN_AUDIT_TRADES = 12
MIN_AUDIT_YEARS = 2
MIN_AUDIT_POSITIVE_YEAR_SHARE = 2 / 3
SAFETY_FLAGS = {
    "research_only": True,
    "order_submission_allowed": False,
    "execution_authorization": "none",
    "orders_sent": False,
    "committed": False,
    "dry_run": True,
    "approved_strategy": False,
}


@dataclass(frozen=True)
class ScenarioResult:
    raw_pips: float
    net_pips: float
    exit_timestamp: pd.Timestamp
    exit_reason: str


@dataclass(frozen=True)
class ExecutableFrame:
    index: pd.DatetimeIndex
    bid_open: np.ndarray
    bid_high: np.ndarray
    bid_low: np.ndarray
    bid_close: np.ndarray
    ask_open: np.ndarray
    ask_high: np.ndarray
    ask_low: np.ndarray
    ask_close: np.ndarray

    @classmethod
    def from_bars(cls, bars: pd.DataFrame) -> ExecutableFrame:
        return cls(
            index=bars.index,
            bid_open=bars["bid_open"].to_numpy(dtype="float64"),
            bid_high=bars["bid_high"].to_numpy(dtype="float64"),
            bid_low=bars["bid_low"].to_numpy(dtype="float64"),
            bid_close=bars["bid_close"].to_numpy(dtype="float64"),
            ask_open=bars["ask_open"].to_numpy(dtype="float64"),
            ask_high=bars["ask_high"].to_numpy(dtype="float64"),
            ask_low=bars["ask_low"].to_numpy(dtype="float64"),
            ask_close=bars["ask_close"].to_numpy(dtype="float64"),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-m1-dir", type=Path, default=DEFAULT_PRE_M1_DIR)
    parser.add_argument("--current-m1-dir", type=Path, default=DEFAULT_CURRENT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(
        pre_m1_dir=arguments.pre_m1_dir,
        current_m1_dir=arguments.current_m1_dir,
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def load_pre_market(m1_dir: Path) -> dict[str, PairData]:
    """Load all pairs from a raw M1 directory without applying 2019+ truncation."""
    raw_frames = {pair: raw_pair_frame(m1_dir, pair) for pair in PAIRS}
    rate_medians = {
        pair: float(np.median(mid_ohlc(raw_frames[pair])["close"].to_numpy()))
        for pair in MAJOR_RATE_PAIRS
    }
    rates = ConversionRates(rate_medians)
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
            f"{pair} pre: loaded {len(frame)} bars"
            f" (dropped_crossed={market[pair].quality['dropped_crossed_quote_bars']})\n"
        )
        sys.stdout.flush()
    return market


def evaluate_market(
    label: str,
    market: dict[str, PairData],
    *,
    candidate_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    entry_policies = entry_policy_grid()
    risk_policies = risk_policy_grid()
    trade_rows: list[dict[str, Any]] = []
    signal_counts: dict[str, int] = {}
    for pair in PAIRS:
        pair_data = market[pair]
        base_frame = ExecutableFrame.from_bars(pair_data.bars)
        stress_frame = ExecutableFrame.from_bars(pair_data.stress_bars)
        for entry_policy in entry_policies:
            allowed_risk_policies = tuple(
                risk_policy
                for risk_policy in risk_policies
                if candidate_filter is None or f"{pair}:{entry_policy.identifier}:{risk_policy.identifier}" in candidate_filter
            )
            if not allowed_risk_policies:
                continue
            signals = enrich_signals_with_daily_atr(pair, pair_data, entry_policy)
            signal_counts[f"{label}:{pair}:{entry_policy.identifier}"] = int(len(signals))
            trade_rows.extend(
                {"sample": label, **row}
                for row in evaluate_pair_entry_risks(
                    pair=pair,
                    pair_data=pair_data,
                    base_frame=base_frame,
                    stress_frame=stress_frame,
                    entry_policy=entry_policy,
                    risk_policies=allowed_risk_policies,
                    signals=signals,
                )
            )
        sys.stdout.write(f"{label} {pair}: accumulated trades={len(trade_rows)}\n")
        sys.stdout.flush()
    if not trade_rows:
        return pd.DataFrame(), signal_counts
    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "candidate_id"], ignore_index=True)
    for column in ("signal_timestamp", "entry_timestamp", "exit_base_timestamp", "exit_stress_timestamp"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    return trades, signal_counts


def evaluate_pair_entry_risks(
    *,
    pair: str,
    pair_data: PairData,
    base_frame: ExecutableFrame,
    stress_frame: ExecutableFrame,
    entry_policy: Any,
    risk_policies: tuple[Any, ...],
    signals: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pip = Instrument.from_symbol(pair).pip_size
    hold_minutes = entry_policy.hold_days * 24 * 60
    for signal in signals.to_dict("records"):
        side = int(signal["side_num"])
        entry_timestamp = pd.Timestamp(signal["entry_timestamp"])
        for risk_policy in risk_policies:
            risk_pips = risk_policy.stop_atr_multiple * float(signal["daily_atr_pips"])
            base = simulate_gappy_trade_fast(
                base_frame,
                pip=pip,
                entry_timestamp=entry_timestamp,
                side=side,
                risk_pips=risk_pips,
                hold_minutes=hold_minutes,
                management=risk_policy.management,
                commission=pair_data.base_commission_pips,
                slippage_per_side_pips=FUSION_ZERO.slippage_pips,
            )
            if base is None:
                continue
            stress = simulate_gappy_trade_fast(
                stress_frame,
                pip=pip,
                entry_timestamp=entry_timestamp,
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
                    "entry_timestamp": entry_timestamp,
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


def simulate_gappy_trade_fast(
    frame: ExecutableFrame,
    *,
    pip: float,
    entry_timestamp: pd.Timestamp,
    side: int,
    risk_pips: float,
    hold_minutes: int,
    management: Any,
    commission: float,
    slippage_per_side_pips: float,
) -> ScenarioResult | None:
    if not np.isfinite(risk_pips) or risk_pips <= 0:
        return None
    exit_deadline = entry_timestamp + pd.Timedelta(minutes=hold_minutes)
    positions = frame.index.get_indexer([entry_timestamp, exit_deadline])
    entry_pos, exit_pos = int(positions[0]), int(positions[1])
    if entry_pos < 0 or exit_pos < 0 or exit_pos <= entry_pos:
        return None
    risk_price = risk_pips * pip
    entry_price = float(frame.ask_open[entry_pos] if side == 1 else frame.bid_open[entry_pos])
    if side == 1:
        stop = entry_price - risk_price
        target = entry_price + management.target_r * risk_price if management.target_r is not None else None
    else:
        stop = entry_price + risk_price
        target = entry_price - management.target_r * risk_price if management.target_r is not None else None

    if management.activation_r is None or management.trail_r is None:
        return simulate_fixed_exit(
            frame,
            pip=pip,
            side=side,
            entry_price=entry_price,
            entry_pos=entry_pos,
            exit_pos=exit_pos,
            stop=stop,
            target=target,
            commission=commission,
            slippage_per_side_pips=slippage_per_side_pips,
        )
    return simulate_trailing_exit(
        frame,
        pip=pip,
        side=side,
        entry_price=entry_price,
        entry_pos=entry_pos,
        exit_pos=exit_pos,
        stop=stop,
        target=target,
        risk_pips=risk_pips,
        risk_price=risk_price,
        management=management,
        commission=commission,
        slippage_per_side_pips=slippage_per_side_pips,
    )


def simulate_fixed_exit(
    frame: ExecutableFrame,
    *,
    pip: float,
    side: int,
    entry_price: float,
    entry_pos: int,
    exit_pos: int,
    stop: float,
    target: float | None,
    commission: float,
    slippage_per_side_pips: float,
) -> ScenarioResult:
    bar_slice = slice(entry_pos, exit_pos + 1)
    if side == 1:
        open_stop_offset = first_true_offset(frame.bid_open[bar_slice] <= stop)
        adverse_stop_offset = first_true_offset(frame.bid_low[bar_slice] <= stop)
        target_offset = first_true_offset(frame.bid_high[bar_slice] >= target) if target is not None else None
        first_stop_offset = first_existing_offset(open_stop_offset, adverse_stop_offset)
        if target_offset is not None and (first_stop_offset is None or target_offset < first_stop_offset):
            return scenario_result_fast(
                side, entry_price, float(target), pip, commission, slippage_per_side_pips, frame.index[entry_pos + target_offset], "target"
            )
        if first_stop_offset is not None:
            exit_price = float(frame.bid_open[entry_pos + first_stop_offset]) if open_stop_offset == first_stop_offset else stop
            return scenario_result_fast(
                side, entry_price, exit_price, pip, commission, slippage_per_side_pips, frame.index[entry_pos + first_stop_offset], "stop"
            )
        return scenario_result_fast(
            side,
            entry_price,
            float(frame.bid_close[exit_pos]),
            pip,
            commission,
            slippage_per_side_pips,
            frame.index[exit_pos],
            "timeout",
        )
    open_stop_offset = first_true_offset(frame.ask_open[bar_slice] >= stop)
    adverse_stop_offset = first_true_offset(frame.ask_high[bar_slice] >= stop)
    target_offset = first_true_offset(frame.ask_low[bar_slice] <= target) if target is not None else None
    first_stop_offset = first_existing_offset(open_stop_offset, adverse_stop_offset)
    if target_offset is not None and (first_stop_offset is None or target_offset < first_stop_offset):
        return scenario_result_fast(
            side, entry_price, float(target), pip, commission, slippage_per_side_pips, frame.index[entry_pos + target_offset], "target"
        )
    if first_stop_offset is not None:
        exit_price = float(frame.ask_open[entry_pos + first_stop_offset]) if open_stop_offset == first_stop_offset else stop
        return scenario_result_fast(
            side, entry_price, exit_price, pip, commission, slippage_per_side_pips, frame.index[entry_pos + first_stop_offset], "stop"
        )
    return scenario_result_fast(
        side,
        entry_price,
        float(frame.ask_close[exit_pos]),
        pip,
        commission,
        slippage_per_side_pips,
        frame.index[exit_pos],
        "timeout",
    )


def simulate_trailing_exit(
    frame: ExecutableFrame,
    *,
    pip: float,
    side: int,
    entry_price: float,
    entry_pos: int,
    exit_pos: int,
    stop: float,
    target: float | None,
    risk_pips: float,
    risk_price: float,
    management: Any,
    commission: float,
    slippage_per_side_pips: float,
) -> ScenarioResult:
    bar_slice = slice(entry_pos, exit_pos + 1)
    activation_pips = float(management.activation_r) * risk_pips
    trail_price = float(management.trail_r) * risk_price
    if side == 1:
        favorable = frame.bid_high[bar_slice]
        prior_best = shifted_cumulative_max(favorable)
        active = ((prior_best - entry_price) / pip) >= activation_pips
        dynamic_stop = np.full(len(favorable), stop, dtype="float64")
        trailed_stop = np.maximum(entry_price, prior_best - trail_price)
        dynamic_stop = np.where(active, np.maximum(dynamic_stop, trailed_stop), dynamic_stop)

        open_stop_offset = first_true_offset(frame.bid_open[bar_slice] <= dynamic_stop)
        adverse_stop_offset = first_true_offset(frame.bid_low[bar_slice] <= dynamic_stop)
        target_offset = first_true_offset(favorable >= target) if target is not None else None
        first_stop_offset = first_existing_offset(open_stop_offset, adverse_stop_offset)
        if target_offset is not None and (first_stop_offset is None or target_offset < first_stop_offset):
            return scenario_result_fast(
                side, entry_price, float(target), pip, commission, slippage_per_side_pips, frame.index[entry_pos + target_offset], "target"
            )
        if first_stop_offset is not None:
            absolute_pos = entry_pos + first_stop_offset
            exit_price = float(frame.bid_open[absolute_pos]) if open_stop_offset == first_stop_offset else float(dynamic_stop[first_stop_offset])
            return scenario_result_fast(
                side, entry_price, exit_price, pip, commission, slippage_per_side_pips, frame.index[absolute_pos], "stop"
            )
        return scenario_result_fast(
            side,
            entry_price,
            float(frame.bid_close[exit_pos]),
            pip,
            commission,
            slippage_per_side_pips,
            frame.index[exit_pos],
            "timeout",
        )

    favorable = frame.ask_low[bar_slice]
    prior_best = shifted_cumulative_min(favorable)
    active = ((entry_price - prior_best) / pip) >= activation_pips
    dynamic_stop = np.full(len(favorable), stop, dtype="float64")
    trailed_stop = np.minimum(entry_price, prior_best + trail_price)
    dynamic_stop = np.where(active, np.minimum(dynamic_stop, trailed_stop), dynamic_stop)

    open_stop_offset = first_true_offset(frame.ask_open[bar_slice] >= dynamic_stop)
    adverse_stop_offset = first_true_offset(frame.ask_high[bar_slice] >= dynamic_stop)
    target_offset = first_true_offset(favorable <= target) if target is not None else None
    first_stop_offset = first_existing_offset(open_stop_offset, adverse_stop_offset)
    if target_offset is not None and (first_stop_offset is None or target_offset < first_stop_offset):
        return scenario_result_fast(
            side, entry_price, float(target), pip, commission, slippage_per_side_pips, frame.index[entry_pos + target_offset], "target"
        )
    if first_stop_offset is not None:
        absolute_pos = entry_pos + first_stop_offset
        exit_price = float(frame.ask_open[absolute_pos]) if open_stop_offset == first_stop_offset else float(dynamic_stop[first_stop_offset])
        return scenario_result_fast(
            side, entry_price, exit_price, pip, commission, slippage_per_side_pips, frame.index[absolute_pos], "stop"
        )
    return scenario_result_fast(
        side,
        entry_price,
        float(frame.ask_close[exit_pos]),
        pip,
        commission,
        slippage_per_side_pips,
        frame.index[exit_pos],
        "timeout",
    )


def first_true_offset(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def first_existing_offset(*offsets: int | None) -> int | None:
    existing = [offset for offset in offsets if offset is not None]
    return min(existing) if existing else None


def shifted_cumulative_max(values: np.ndarray) -> np.ndarray:
    shifted = np.empty_like(values, dtype="float64")
    shifted[0] = np.nan
    if len(values) > 1:
        shifted[1:] = np.maximum.accumulate(values[:-1])
    return shifted


def shifted_cumulative_min(values: np.ndarray) -> np.ndarray:
    shifted = np.empty_like(values, dtype="float64")
    shifted[0] = np.nan
    if len(values) > 1:
        shifted[1:] = np.minimum.accumulate(values[:-1])
    return shifted


def scenario_result_fast(
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
    return ScenarioResult(float(raw), float(net), pd.Timestamp(exit_timestamp), exit_reason)


def summarize_pre_selection(pre_trades: pd.DataFrame) -> pd.DataFrame:
    if pre_trades.empty:
        return empty_selection_frame()
    grouped = pre_trades.groupby("candidate_id", sort=True)
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
    for candidate_id, sample in pre_trades.groupby("candidate_id", sort=True):
        share, positive, years = positive_year_share(sample, "net_stress_pips")
        rows.append(
            {
                "candidate_id": candidate_id,
                "selection_lower_base_pips": lower_confidence_bound(sample["net_base_pips"], 0.05),
                "selection_lower_stress_pips": lower_confidence_bound(sample["net_stress_pips"], 0.05),
                "selection_positive_stress_year_share": share,
                "selection_positive_stress_years": positive,
                "selection_years": years,
            }
        )
    return summary.reset_index().merge(pd.DataFrame(rows), on="candidate_id", how="left")


def empty_selection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate_id",
            "pair",
            "entry_policy_id",
            "risk_policy_id",
            "channel_days",
            "hold_days",
            "stop_atr_multiple",
            "management",
            "selection_trades",
            "selection_mean_base_pips",
            "selection_mean_stress_pips",
            "selection_total_stress_pips",
            "selection_median_stress_pips",
            "selection_mean_risk_pips",
            "selection_mean_breakout_distance_pips",
            "selection_lower_base_pips",
            "selection_lower_stress_pips",
            "selection_positive_stress_year_share",
            "selection_positive_stress_years",
            "selection_years",
        ]
    )


def select_pre_finalists(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty:
        return selection
    eligible = selection.loc[
        (selection["selection_trades"] >= MIN_SELECTION_TRADES)
        & (selection["selection_years"] >= MIN_SELECTION_YEARS)
        & (selection["selection_mean_base_pips"] > 0)
        & (selection["selection_mean_stress_pips"] > 0)
        & (selection["selection_positive_stress_year_share"] >= MIN_SELECTION_POSITIVE_YEAR_SHARE)
    ].copy()
    return eligible.sort_values(
        [
            "selection_lower_stress_pips",
            "selection_mean_stress_pips",
            "selection_total_stress_pips",
            "selection_positive_stress_year_share",
            "selection_trades",
        ],
        ascending=[False, False, False, False, False],
    ).head(FINALISTS)


def audit_current_sample(current_trades: pd.DataFrame, finalists: pd.DataFrame) -> pd.DataFrame:
    alpha = 0.05 / max(len(finalists), 1)
    rows: list[dict[str, Any]] = []
    for rank, finalist in enumerate(finalists.itertuples(index=False), start=1):
        sample = current_trades.loc[current_trades["candidate_id"] == finalist.candidate_id]
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


def fast137_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "temporal_reverse_audit"]
    if passed_count:
        return (
            "FAST137_FX_DAILY_DONCHIAN_FADE_TEMPORAL_REVERSE_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"pre2019_selected_current_audit_candidates:{passed_count}",
                "fresh_preregistered_prospective_observation_required_before_any_execution",
            ],
        )
    if finalists_count:
        return (
            "FAST137_FX_DAILY_DONCHIAN_FADE_TEMPORAL_REVERSE_FAILED_CURRENT_AUDIT",
            reasons + ["pre2019_positive_candidates_failed_2019_2026_audit"],
        )
    return (
        "FAST137_FX_DAILY_DONCHIAN_FADE_TEMPORAL_REVERSE_FAILED_PRE2019_SELECTION",
        reasons + ["no_pre2019_candidate_positive_after_costs"],
    )


def run(
    *,
    pre_m1_dir: Path = DEFAULT_PRE_M1_DIR,
    current_m1_dir: Path = DEFAULT_CURRENT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    pre_market = load_pre_market(pre_m1_dir)
    pre_trades, pre_signal_counts = evaluate_market(PRE_SAMPLE_LABEL, pre_market)
    selection = summarize_pre_selection(pre_trades)
    finalists = select_pre_finalists(selection)
    if finalists.empty:
        current_market: dict[str, PairData] = {}
        current_trades = pd.DataFrame()
        current_signal_counts: dict[str, int] = {}
    else:
        current_market = load_market(current_m1_dir)
        current_trades, current_signal_counts = evaluate_market(
            CURRENT_SAMPLE_LABEL,
            current_market,
            candidate_filter=set(finalists["candidate_id"].astype(str)),
        )
    audit = audit_current_sample(current_trades, finalists)

    if pre_trades.empty and current_trades.empty:
        raise ValueError("no temporal-reverse daily Donchian fade trades generated")

    trades = pd.concat([pre_trades, current_trades], ignore_index=True).sort_values(
        ["sample", "entry_timestamp", "candidate_id"], ignore_index=True
    )
    trades_path = output_root / "temporal-reverse-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)

    quality_path = output_root / "pair-quality.json"
    write_json(
        quality_path,
        {
            PRE_SAMPLE_LABEL: [pre_market[pair].quality for pair in PAIRS],
            CURRENT_SAMPLE_LABEL: [current_market[pair].quality for pair in PAIRS] if current_market else [],
        },
    )

    selection_path = output_root / "pre2019-selection-metrics.csv"
    finalists_path = output_root / "pre2019-selected-finalists.csv"
    audit_path = output_root / "current-audited-finalists.csv"
    signal_counts_path = output_root / "signal-counts.json"
    selection.to_csv(selection_path, index=False)
    finalists.to_csv(finalists_path, index=False)
    audit.to_csv(audit_path, index=False)
    write_json(signal_counts_path, {**pre_signal_counts, **current_signal_counts})

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    status, status_reasons = fast137_status(len(passed), len(finalists))
    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast137_is_historical_temporal_reverse_audit_not_strategy_approval",
            "candidates_are_selected_on_pre2019_historical_data",
            "current_2019_2026_audit_is_still_known_history_not_prospective_observation",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast137_fx_daily_donchian_fade_temporal_reverse",
        "materialization_source": "raw_m1",
        "status": status,
        "status_reasons": status_reasons,
        "pre_m1_dir": str(pre_m1_dir.relative_to(REPO_ROOT)) if pre_m1_dir.is_relative_to(REPO_ROOT) else str(pre_m1_dir),
        "current_m1_dir": (
            str(current_m1_dir.relative_to(REPO_ROOT)) if current_m1_dir.is_relative_to(REPO_ROOT) else str(current_m1_dir)
        ),
        "pairs": list(PAIRS),
        "entry_policy_count_per_pair": len(entry_policy_grid()),
        "risk_policy_count": len(risk_policy_grid()),
        "candidate_count": len(PAIRS) * len(entry_policy_grid()) * len(risk_policy_grid()),
        "pre_sample_label": PRE_SAMPLE_LABEL,
        "current_sample_label": CURRENT_SAMPLE_LABEL,
        "current_audit_end": AUDIT_END.isoformat(),
        "pre_trade_rows": int(len(pre_trades)),
        "current_trade_rows": int(len(current_trades)),
        "trade_rows": int(len(trades)),
        "signal_count": int(sum(pre_signal_counts.values()) + sum(current_signal_counts.values())),
        "pre_selection_candidates": int(len(selection)),
        "eligible_finalists": int(len(finalists)),
        "audited_finalists": int(len(audit)),
        "passed_candidates": int(len(passed)),
        "pre_pair_trade_counts": pre_trades.groupby("pair").size().to_dict() if not pre_trades.empty else {},
        "current_pair_trade_counts": current_trades.groupby("pair").size().to_dict() if not current_trades.empty else {},
        "eligible_pair_counts": finalists.groupby("pair").size().to_dict() if not finalists.empty else {},
        "passed_pair_counts": passed.groupby("pair").size().to_dict() if not passed.empty else {},
        "selection_gate": {
            "min_trades": MIN_SELECTION_TRADES,
            "min_years": MIN_SELECTION_YEARS,
            "min_positive_stress_year_share": MIN_SELECTION_POSITIVE_YEAR_SHARE,
            "requires_positive_base_mean": True,
            "requires_positive_stress_mean": True,
        },
        "audit_gate": {
            "min_trades": MIN_AUDIT_TRADES,
            "min_years": MIN_AUDIT_YEARS,
            "min_positive_stress_year_share": MIN_AUDIT_POSITIVE_YEAR_SHARE,
            "requires_positive_base_mean": True,
            "requires_positive_stress_mean": True,
            "requires_positive_base_lcb": True,
            "requires_positive_stress_lcb": True,
            "alpha_familywise": 0.05,
        },
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
            "finalists": {"path": str(finalists_path.relative_to(REPO_ROOT)), "sha256": sha256_file(finalists_path)},
            "audit": {"path": str(audit_path.relative_to(REPO_ROOT)), "sha256": sha256_file(audit_path)},
        },
    }
    if not finalists.empty:
        report["top_finalists"] = finalists.head(15).to_dict("records")
    if not audit.empty:
        report["top_audit"] = audit.head(15).to_dict("records")
    if not passed.empty:
        report["passed_candidate_rows"] = passed.sort_values("audit_mean_stress_pips", ascending=False).to_dict("records")
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
        "candidate_count": report["candidate_count"],
        "pre_trade_rows": report["pre_trade_rows"],
        "current_trade_rows": report["current_trade_rows"],
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
