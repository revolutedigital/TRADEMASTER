"""Evaluate every development portfolio against the pre-registered gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.services.research.statistical_gate import (
    evaluate_statistical_gate,
    probability_of_backtest_overfitting,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--portfolio-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "trailing-pilot-stress-target",
    )
    parser.add_argument("--attempted-hypotheses", type=int, default=44)
    arguments = parser.parse_args()
    paths = sorted(arguments.portfolio_root.glob("policy=*.parquet"))
    if not paths:
        parser.error("no portfolio trade files found")
    daily_series = {}
    trades_by_strategy = {}
    for path in paths:
        trades = pd.read_parquet(path)
        strategy = path.stem
        trades_by_strategy[strategy] = trades
        dates = pd.to_datetime(trades["entry_time_ms"], unit="ms", utc=True).dt.date
        daily_series[strategy] = (
            trades.assign(utc_date=dates).groupby("utc_date")["expected_net_bps"].sum()
        )
    daily_matrix = pd.DataFrame(daily_series).fillna(0)
    pbo = probability_of_backtest_overfitting(daily_matrix)
    results = []
    for strategy, trades in trades_by_strategy.items():
        gate = evaluate_statistical_gate(
            trades,
            attempted_hypotheses=arguments.attempted_hypotheses,
            temporal_fold_count=3,
            top_p_monotonic=True,
            prospective_positive=False,
            pbo=pbo,
        )
        results.append({"strategy": strategy, **gate.to_dict()})
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "attempted_hypotheses": arguments.attempted_hypotheses,
        "decision_counts": pd.Series([row["decision"] for row in results]).value_counts().to_dict(),
        "results": results,
    }
    output = arguments.portfolio_root / "statistical-gate.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["decision_counts"], indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
