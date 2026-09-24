"""Fast135: temporal robustness audit for the Fast134 NZDUSD fixed-1R clue.

Fast134 is the best preregistered FX candidate so far, but it was selected from
2019-2026 research. Fast135 checks the same frozen rule against the local
pre-2019 M1 dataset as a diagnostic temporal stress test. This does not approve
execution: pre-2019 feed quality differs and the result is historical evidence
only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.fx.instruments import ConversionRates  # noqa: E402
from app.fx.sim.costs import FUSION_ZERO, STRESS  # noqa: E402
from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json  # noqa: E402
from scripts.research.run_fast121_fx_compression_breakout_scout import (  # noqa: E402
    MAJOR_RATE_PAIRS,
    PairData,
    commission_pips,
    mid_ohlc,
    sha256_file,
    widen_quotes,
)
from scripts.research.run_fast130_nzdusd_donchian_fade_risk_audit import (  # noqa: E402
    PAIR,
    evaluate_risk_policy,
    frozen_entry_signals,
    lower_confidence_bound,
    positive_year_share,
)
from scripts.research.run_fast134_nzdusd_donchian_fade_fixed1r_observation import (  # noqa: E402
    FROZEN_ENTRY_POLICY,
    FROZEN_POLICY_ID,
    FROZEN_RISK_POLICY,
)

DEFAULT_PRE_M1_DIR = REPO_ROOT / "backend" / "data" / "raw" / "fx_m1_hd_pre"
DEFAULT_CURRENT_M1_DIR = REPO_ROOT / "backend" / "data" / "raw" / "fx_m1_hd"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast135" / "nzdusd-donchian-fade-fixed1r-temporal-audit"
)
TIMESTAMP_COLUMNS = (
    "signal_timestamp",
    "entry_timestamp",
    "exit_base_timestamp",
    "exit_stress_timestamp",
)
BAR_COLUMNS = [
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
]
SAFETY_FLAGS = {
    "research_only": True,
    "order_submission_allowed": False,
    "execution_authorization": "none",
    "orders_sent": False,
    "committed": False,
    "dry_run": True,
    "approved_strategy": False,
}


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


def raw_pair_frame(m1_dir: Path, pair: str) -> pd.DataFrame:
    paths = sorted(m1_dir.glob(f"{pair}_M1_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no M1 parquet files for {pair} in {m1_dir}")
    frame = pd.concat(pd.read_parquet(path, columns=BAR_COLUMNS) for path in paths).sort_index()
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


def pair_data_from_dir(m1_dir: Path, pair: str = PAIR) -> PairData:
    frames = {rate_pair: raw_pair_frame(m1_dir, rate_pair) for rate_pair in MAJOR_RATE_PAIRS}
    rates = ConversionRates(
        {
            rate_pair: float(np.median(mid_ohlc(frame)["close"].to_numpy()))
            for rate_pair, frame in frames.items()
        }
    )
    frame = frames[pair]
    median_price = float(np.median(mid_ohlc(frame)["close"].to_numpy()))
    return PairData(
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


def evaluate_slice(label: str, m1_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    pair_data = pair_data_from_dir(m1_dir)
    signals = frozen_entry_signals(pair_data)
    rows = evaluate_risk_policy(pair_data, signals, FROZEN_RISK_POLICY)
    trades = pd.DataFrame(rows)
    if not trades.empty:
        for column in TIMESTAMP_COLUMNS:
            trades[column] = pd.to_datetime(trades[column], utc=True)
        trades.insert(0, "slice", label)
    summary = summarize_slice(label=label, pair_data=pair_data, signals=signals, trades=trades)
    return summary, trades


def summarize_slice(
    *,
    label: str,
    pair_data: PairData,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, Any]:
    base = metric_summary(trades, "net_base_pips")
    stress = metric_summary(trades, "net_stress_pips")
    return {
        "slice": label,
        "pair_quality": pair_data.quality,
        "signal_count": int(len(signals)),
        "trade_count": int(len(trades)),
        "entry_start": trades["entry_timestamp"].min().isoformat() if not trades.empty else None,
        "entry_end": trades["entry_timestamp"].max().isoformat() if not trades.empty else None,
        "base": base,
        "stress": stress,
    }


def metric_summary(trades: pd.DataFrame, column: str) -> dict[str, Any]:
    if trades.empty:
        return {
            "column": column,
            "trade_count": 0,
            "total_pips": 0.0,
            "mean_pips": None,
            "lower_confidence_bound_pips": None,
            "positive_year_count": 0,
            "year_count": 0,
            "positive_year_share": None,
            "yearly": [],
        }
    values = trades[column].astype("float64")
    share, positive, years = positive_year_share(trades, column)
    return {
        "column": column,
        "trade_count": int(len(trades)),
        "total_pips": float(values.sum()),
        "mean_pips": float(values.mean()),
        "lower_confidence_bound_pips": lower_confidence_bound(values, 0.05),
        "positive_year_count": int(positive),
        "year_count": int(years),
        "positive_year_share": share,
        "yearly": yearly_summary(trades, column),
    }


def yearly_summary(trades: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    grouped = (
        trades.assign(year=trades["entry_timestamp"].dt.year)
        .groupby("year", sort=True)[column]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    return [
        {
            "year": int(row["year"]),
            "trade_count": int(row["size"]),
            "total_pips": float(row["sum"]),
            "mean_pips": float(row["mean"]),
        }
        for row in grouped.to_dict("records")
    ]


def fast135_status(summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "diagnostic_temporal_audit"]
    by_slice = {summary["slice"]: summary for summary in summaries}
    current = by_slice.get("current_2019_2026")
    pre = by_slice.get("pre_2011_2018")
    if current is None or pre is None:
        return "FAST135_TEMPORAL_AUDIT_INCOMPLETE", reasons + ["missing_required_slice"]
    current_lcb = current["stress"]["lower_confidence_bound_pips"]
    pre_lcb = pre["stress"]["lower_confidence_bound_pips"]
    if current_lcb is None or current_lcb <= 0:
        return "FAST135_CURRENT_SAMPLE_FAILED_TEMPORAL_AUDIT", reasons + ["current_lcb_not_positive"]
    if pre_lcb is not None and pre_lcb > 0:
        return (
            "FAST135_TEMPORAL_AUDIT_BOTH_SAMPLES_POSITIVE_REQUIRES_PROSPECTIVE_CONFIRMATION",
            reasons + ["pre_2019_lcb_positive", "prospective_confirmation_still_required"],
        )
    return (
        "FAST135_TEMPORAL_AUDIT_MIXED_STABILITY_REQUIRES_PROSPECTIVE_CONFIRMATION",
        reasons + ["current_lcb_positive", "pre_2019_lcb_not_positive", "prospective_confirmation_required"],
    )


def run(
    *,
    pre_m1_dir: Path = DEFAULT_PRE_M1_DIR,
    current_m1_dir: Path = DEFAULT_CURRENT_M1_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    pre_summary, pre_trades = evaluate_slice("pre_2011_2018", pre_m1_dir)
    current_summary, current_trades = evaluate_slice("current_2019_2026", current_m1_dir)
    trades = pd.concat([pre_trades, current_trades], ignore_index=True)
    trades_path = output_root / "temporal-audit-trades.parquet"
    trades.to_parquet(trades_path, compression="zstd", index=False)

    summaries = [pre_summary, current_summary]
    summary_rows = flatten_summary_rows(summaries)
    summary_path = output_root / "slice-summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    status, status_reasons = fast135_status(summaries)

    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast135_is_temporal_diagnostic_not_strategy_approval",
            "pre_2019_feed_quality_differs_from_current_research_feed",
            "fast134_still_requires_fresh_prospective_observation",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast135_nzdusd_donchian_fade_fixed1r_temporal_audit",
        "materialization_source": "raw_m1",
        "status": status,
        "status_reasons": status_reasons,
        "frozen_policy_id": FROZEN_POLICY_ID,
        "frozen_entry_policy": asdict(FROZEN_ENTRY_POLICY),
        "frozen_risk_policy": {
            **asdict(FROZEN_RISK_POLICY),
            "management": asdict(FROZEN_RISK_POLICY.management),
        },
        "slices": summaries,
        "trades_path": str(trades_path),
        "trades_sha256": sha256_file(trades_path),
        "slice_summary_path": str(summary_path),
        "slice_summary_sha256": sha256_file(summary_path),
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path)})
    return plain_json({**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)})


def flatten_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.append(
            {
                "slice": summary["slice"],
                "signal_count": summary["signal_count"],
                "trade_count": summary["trade_count"],
                "entry_start": summary["entry_start"],
                "entry_end": summary["entry_end"],
                "base_mean_pips": summary["base"]["mean_pips"],
                "stress_mean_pips": summary["stress"]["mean_pips"],
                "stress_total_pips": summary["stress"]["total_pips"],
                "stress_lower_confidence_bound_pips": summary["stress"]["lower_confidence_bound_pips"],
                "stress_positive_year_count": summary["stress"]["positive_year_count"],
                "stress_year_count": summary["stress"]["year_count"],
                "stress_positive_year_share": summary["stress"]["positive_year_share"],
            }
        )
    return rows


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "frozen_policy_id": report["frozen_policy_id"],
        "slices": [
            {
                "slice": item["slice"],
                "signal_count": item["signal_count"],
                "trade_count": item["trade_count"],
                "stress_mean_pips": item["stress"]["mean_pips"],
                "stress_lower_confidence_bound_pips": item["stress"]["lower_confidence_bound_pips"],
                "positive_years": f"{item['stress']['positive_year_count']}/{item['stress']['year_count']}",
            }
            for item in report["slices"]
        ],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
