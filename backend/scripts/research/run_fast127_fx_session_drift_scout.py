"""Fast127: FX intraday/session drift scout on executable M1 bid/ask bars.

This tests a different family from breakouts, top-p gates and H1 overextension:
fixed pair/hour/weekday directional drift with a fixed time exit. It is
research-only, writes offline artifacts, and never touches the trading engine,
credentials, ledger, Demo, Testnet or LIVE accounts.
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
from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    AUDIT_END,
    DEFAULT_M1_DIR,
    PAIRS,
    PairData,
    load_market,
    sha256_file,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast127" / "fx-session-drift-scout"

SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
ENTRY_HOURS_UTC = (0, 7, 8, 13, 14, 16, 20)
HOLD_MINUTES = (240, 480, 1440)
WEEKDAY_FILTERS = ("all",)
FINALISTS = 80
MIN_SELECTION_TRADES = 80
MIN_AUDIT_TRADES = 30
MIN_AUDIT_YEARS = 2
MIN_AUDIT_POSITIVE_YEAR_SHARE = 2 / 3

WeekdayFilter = Literal["all", "mon", "tue", "wed", "thu", "fri"]
SideName = Literal["long", "short"]

WEEKDAY_TO_NUMBER = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
}


@dataclass(frozen=True)
class Policy:
    identifier: str
    entry_hour_utc: int
    hold_minutes: int
    side: SideName
    weekday_filter: WeekdayFilter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=DEFAULT_M1_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    report = run(m1_dir=arguments.m1_dir, output_root=arguments.output_root)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def policy_grid() -> tuple[Policy, ...]:
    policies: list[Policy] = []
    for weekday in WEEKDAY_FILTERS:
        for hour in ENTRY_HOURS_UTC:
            for hold in HOLD_MINUTES:
                for side in ("long", "short"):
                    identifier = f"utc{hour:02d}_wd{weekday}_hold{hold}_{side}"
                    policies.append(
                        Policy(
                            identifier=identifier,
                            entry_hour_utc=hour,
                            hold_minutes=hold,
                            side=side,
                            weekday_filter=weekday,
                        )
                    )
    return tuple(policies)


def policy_entry_index(frame: pd.DataFrame, policy: Policy) -> pd.DatetimeIndex:
    index = frame.index
    mask = (index.minute == 0) & (index.hour == policy.entry_hour_utc)
    if policy.weekday_filter == "all":
        mask &= index.dayofweek <= 4
    else:
        mask &= index.dayofweek == WEEKDAY_TO_NUMBER[policy.weekday_filter]
    return pd.DatetimeIndex(index[mask])


def timebox_trade_rows(
    frame: pd.DataFrame,
    *,
    pair: str,
    policy: Policy,
    commission_pips: float,
    slippage_per_side_pips: float,
) -> pd.DataFrame:
    """Vectorized executable time-box trades for one pair/policy/cost scenario."""
    entries = policy_entry_index(frame, policy)
    if entries.empty:
        return empty_trade_frame()
    exits = entries + pd.Timedelta(minutes=policy.hold_minutes)
    exit_positions = frame.index.get_indexer(exits)
    available = exit_positions >= 0
    entries = entries[available]
    exits = exits[available]
    exit_positions = exit_positions[available]
    if entries.empty:
        return empty_trade_frame()

    entry_bars = frame.loc[entries]
    exit_bars = frame.iloc[exit_positions]
    pip = Instrument.from_symbol(pair).pip_size
    side = 1 if policy.side == "long" else -1
    if side == 1:
        entry_price = entry_bars["ask_open"].to_numpy(dtype=float)
        exit_price = exit_bars["bid_open"].to_numpy(dtype=float)
    else:
        entry_price = entry_bars["bid_open"].to_numpy(dtype=float)
        exit_price = exit_bars["ask_open"].to_numpy(dtype=float)
    raw_pips = (exit_price - entry_price) * side / pip
    net_pips = raw_pips - commission_pips - 2.0 * slippage_per_side_pips
    return pd.DataFrame(
        {
            "entry_timestamp": entries,
            "exit_timestamp": exits,
            "side": "LONG" if side == 1 else "SHORT",
            "raw_pips": raw_pips,
            "net_pips": net_pips,
        }
    )


def empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_timestamp": pd.DatetimeIndex([], tz="UTC"),
            "exit_timestamp": pd.DatetimeIndex([], tz="UTC"),
            "side": pd.Series(dtype="object"),
            "raw_pips": pd.Series(dtype="float64"),
            "net_pips": pd.Series(dtype="float64"),
        }
    )


def evaluate_pair_policy(pair: str, pair_data: PairData, policy: Policy) -> list[dict[str, Any]]:
    base = timebox_trade_rows(
        pair_data.bars,
        pair=pair,
        policy=policy,
        commission_pips=pair_data.base_commission_pips,
        slippage_per_side_pips=FUSION_ZERO.slippage_pips,
    )
    if base.empty:
        return []
    stress = timebox_trade_rows(
        pair_data.stress_bars,
        pair=pair,
        policy=policy,
        commission_pips=pair_data.stress_commission_pips,
        slippage_per_side_pips=STRESS.slippage_pips,
    )
    if stress.empty:
        return []
    joined = base.merge(
        stress,
        on=["entry_timestamp", "exit_timestamp", "side"],
        suffixes=("_base", "_stress"),
    )
    joined["candidate_id"] = f"{pair}:{policy.identifier}"
    joined["pair"] = pair
    joined["policy_id"] = policy.identifier
    joined["entry_hour_utc"] = policy.entry_hour_utc
    joined["weekday_filter"] = policy.weekday_filter
    joined["hold_minutes"] = policy.hold_minutes
    joined["raw_base_pips"] = joined["raw_pips_base"]
    joined["net_base_pips"] = joined["net_pips_base"]
    joined["raw_stress_pips"] = joined["raw_pips_stress"]
    joined["net_stress_pips"] = joined["net_pips_stress"]
    joined["base_commission_pips"] = pair_data.base_commission_pips
    joined["stress_commission_pips"] = pair_data.stress_commission_pips
    columns = [
        "candidate_id",
        "pair",
        "policy_id",
        "entry_hour_utc",
        "weekday_filter",
        "hold_minutes",
        "side",
        "entry_timestamp",
        "exit_timestamp",
        "raw_base_pips",
        "net_base_pips",
        "raw_stress_pips",
        "net_stress_pips",
        "base_commission_pips",
        "stress_commission_pips",
    ]
    return joined[columns].to_dict("records")


def summarize_candidates(trades: pd.DataFrame) -> pd.DataFrame:
    selection = trades.loc[trades["entry_timestamp"] < SELECTION_END].copy()
    grouped = selection.groupby("candidate_id", sort=True)
    summary = grouped.agg(
        pair=("pair", "first"),
        policy_id=("policy_id", "first"),
        entry_hour_utc=("entry_hour_utc", "first"),
        weekday_filter=("weekday_filter", "first"),
        hold_minutes=("hold_minutes", "first"),
        side=("side", "first"),
        selection_trades=("candidate_id", "size"),
        selection_mean_base_pips=("net_base_pips", "mean"),
        selection_mean_stress_pips=("net_stress_pips", "mean"),
        selection_total_stress_pips=("net_stress_pips", "sum"),
        selection_median_stress_pips=("net_stress_pips", "median"),
    )
    rows: list[dict[str, Any]] = []
    for candidate_id, sample in selection.groupby("candidate_id", sort=True):
        share, positive, years = positive_year_share(sample, "net_stress_pips")
        rows.append(
            {
                "candidate_id": candidate_id,
                "selection_lower_stress_pips": lower_confidence_bound(
                    sample["net_stress_pips"],
                    0.05,
                ),
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


def fast127_status(passed_count: int, finalists_count: int) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if passed_count:
        return (
            "FAST127_FX_SESSION_DRIFT_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"historical_audit_candidates:{passed_count}",
                "fresh_preregistered_prospective_observation_required",
                "execution_risk_requires_separate_stop_and_position_sizing_review",
            ],
        )
    if finalists_count:
        return (
            "FAST127_FX_SESSION_DRIFT_FAILED_AUDIT",
            reasons + ["selection_positive_candidates_failed_2024_2026_audit"],
        )
    return (
        "FAST127_FX_SESSION_DRIFT_FAILED_SELECTION",
        reasons + ["no_selection_candidate_positive_after_costs"],
    )


def run(*, m1_dir: Path = DEFAULT_M1_DIR, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
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
            if done % 500 == 0 or done == total_jobs:
                sys.stdout.write(f"jobs {done}/{total_jobs}; accumulated trades={len(trade_rows)}\n")
                sys.stdout.flush()
    if not trade_rows:
        raise ValueError("no session-drift trades generated")

    trades = pd.DataFrame(trade_rows).sort_values(["entry_timestamp", "candidate_id"])
    trades_path = output_root / "session-drift-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)
    selection = summarize_candidates(trades)
    finalists = select_finalists(selection)
    audit = audit_finalists(trades, finalists)
    selection_path = output_root / "selection-metrics.csv"
    audit_path = output_root / "audited-finalists.csv"
    selection.to_csv(selection_path, index=False)
    audit.to_csv(audit_path, index=False)

    passed = audit.loc[audit["pilot_gate"]] if not audit.empty else audit
    status, status_reasons = fast127_status(len(passed), len(finalists))
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast127_is_historical_scout_not_strategy_approval",
            "session_drift_policies_are_selected_on_historical_selection_sample",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "time_exit_without_hard_stop_requires_separate_risk_review",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast127_fx_session_drift_scout",
        "status": status,
        "status_reasons": status_reasons,
        "m1_dir": str(m1_dir.relative_to(REPO_ROOT)),
        "pairs": list(PAIRS),
        "policy_count_per_pair": len(policies),
        "candidate_count": len(PAIRS) * len(policies),
        "entry_hours_utc": list(ENTRY_HOURS_UTC),
        "hold_minutes": list(HOLD_MINUTES),
        "weekday_filters": list(WEEKDAY_FILTERS),
        "scope_note": "first_pass_all_weekdays_only_to_avoid_weekday_combination_overfit",
        "selection_start": trades["entry_timestamp"].min().isoformat(),
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
