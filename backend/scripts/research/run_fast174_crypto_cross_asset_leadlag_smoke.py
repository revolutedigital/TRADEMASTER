"""Fast174: crypto cross-asset lead-lag smoke.

Fast172/Fast173 showed that reusing the same low-cost top-p idea on ETHUSDT
and SOLUSDT does not solve the temporal stability problem. Fast174 changes the
information set: instead of scoring one symbol in isolation, it uses closed
minute features from BTCUSDT, ETHUSDT and SOLUSDT together to predict short
future moves in ETHUSDT or SOLUSDT.

This is a bounded smoke test only. It writes offline artifacts and never sends
orders, touches ledgers, or authorizes Demo/Testnet/LIVE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

DEFAULT_BTC_TRADE_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1" / "normalized" / "aggTrades"
DEFAULT_ETH_TRADE_ROOT = (
    REPO_ROOT / "backend" / "data" / "microstructure_ethusdt_v1" / "normalized" / "aggTrades"
)
DEFAULT_SOL_TRADE_ROOT = (
    REPO_ROOT / "backend" / "data" / "microstructure_solusdt_v1" / "normalized" / "aggTrades"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "microstructure_cross_asset_v1" / "reports" / "fast174-crypto-cross-asset-leadlag-smoke"
)

BLOCK_DATES = {
    "fast50": ("2026-08-30", "2026-08-31"),
    "fast51": ("2026-09-06", "2026-09-07"),
    "fast52": ("2026-09-13", "2026-09-14"),
    "fast53": ("2026-09-19", "2026-09-20"),
    "fast56": ("2026-09-21", "2026-09-22"),
}
TRAIN_BLOCKS = ("fast50", "fast51", "fast52")
VALIDATION_BLOCK = "fast53"
EXTENSION_BLOCK = "fast56"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
TARGET_SYMBOLS = ("ETHUSDT", "SOLUSDT")
FEATURE_WINDOWS_MINUTES = (1, 5, 15, 30)
HORIZONS_SECONDS = (900, 1800)
TOP_FRACTIONS = (0.005, 0.01, 0.02, 0.05)
DECISION_STRIDE_MINUTES = 5
EXPLICIT_COST_BPS = 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    report = run_fast174_crypto_cross_asset_leadlag_smoke(
        trade_roots={
            "BTCUSDT": arguments.btc_trade_root,
            "ETHUSDT": arguments.eth_trade_root,
            "SOLUSDT": arguments.sol_trade_root,
        },
        output_root=arguments.output_root,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast174_crypto_cross_asset_leadlag_smoke(
    *,
    trade_roots: dict[str, Path],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    validate_trade_roots(trade_roots)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = build_all_feature_rows(trade_roots)
    feature_columns = [
        column
        for column in rows.columns
        if any(token in column for token in ("_ret_", "_vol_", "_flow_", "_qv_"))
    ] + ["hour_sin", "hour_cos"]

    evaluations = evaluate_all_specs(rows, feature_columns)
    screen_survivors = [row for row in evaluations if row["train"]["passed"]]
    validation_survivors = [
        row for row in evaluations if row["train"]["passed"] and row["validation"]["passed"]
    ]
    strict_survivors = [
        row
        for row in evaluations
        if row["train"]["passed"] and row["validation"]["passed"] and row["extension"]["passed"]
    ]
    status, status_reasons = fast174_status(
        screen_survivor_count=len(screen_survivors),
        validation_survivor_count=len(validation_survivors),
        strict_survivor_count=len(strict_survivors),
    )

    evaluations_path = output_root / "spec-evaluations.json"
    write_json(evaluations_path, evaluations)
    summary_path = output_root / "summary.json"
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "symbols": list(SYMBOLS),
        "target_symbols": list(TARGET_SYMBOLS),
        "block_dates": BLOCK_DATES,
        "train_blocks": list(TRAIN_BLOCKS),
        "validation_block": VALIDATION_BLOCK,
        "extension_block": EXTENSION_BLOCK,
        "horizons_seconds": list(HORIZONS_SECONDS),
        "top_fractions": list(TOP_FRACTIONS),
        "decision_stride_minutes": DECISION_STRIDE_MINUTES,
        "explicit_cost_bps": EXPLICIT_COST_BPS,
        "feature_windows_minutes": list(FEATURE_WINDOWS_MINUTES),
        "feature_count": len(feature_columns),
        "row_count": int(len(rows)),
        "spec_count": len(evaluations),
        "screen_survivor_count": len(screen_survivors),
        "validation_survivor_count": len(validation_survivors),
        "strict_survivor_count": len(strict_survivors),
        "top_extension_candidates": top_candidates(evaluations, key_sample="extension", limit=10),
        "top_validation_candidates": top_candidates(evaluations, key_sample="validation", limit=10),
        "best_strict_survivors": top_candidates(strict_survivors, key_sample="extension", limit=10),
    }
    write_json(summary_path, summary)

    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast174_is_cross_asset_leadlag_smoke_not_strategy_approval",
            "uses_offline_public_archive_aggtrades_only",
            "strict_train_validation_extension_survival_required_before_any_preregistration",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast174_crypto_cross_asset_leadlag_smoke",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "artifacts": {
            "spec_evaluations": {"path": str(evaluations_path), "sha256": sha256_file(evaluations_path)},
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
        "trade_roots": {symbol: str(path) for symbol, path in sorted(trade_roots.items())},
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def validate_trade_roots(trade_roots: dict[str, Path]) -> None:
    missing = [symbol for symbol in SYMBOLS if symbol not in trade_roots]
    if missing:
        raise ValueError(f"missing trade roots for symbols: {missing}")
    for symbol, root in trade_roots.items():
        if symbol not in SYMBOLS:
            raise ValueError(f"unexpected symbol {symbol}")
        for dates in BLOCK_DATES.values():
            for utc_date in dates:
                path = root / f"date={utc_date}" / "events.parquet"
                if not path.exists():
                    raise FileNotFoundError(f"missing {symbol} partition: {path}")


def build_all_feature_rows(trade_roots: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for block, dates in BLOCK_DATES.items():
        for utc_date in dates:
            frame = build_date_features(trade_roots, utc_date)
            frame["block"] = block
            frame["date"] = utc_date
            frames.append(frame)
    rows = pd.concat(frames, ignore_index=True).sort_values("decision_time")
    return rows.loc[rows["minute_sequence"].mod(DECISION_STRIDE_MINUTES).eq(0)].reset_index(drop=True)


def build_date_features(trade_roots: dict[str, Path], utc_date: str) -> pd.DataFrame:
    parts = [
        minute_features(symbol=symbol, root=trade_roots[symbol], utc_date=utc_date)
        for symbol in SYMBOLS
    ]
    frame = pd.concat(parts, axis=1).sort_index()
    frame["minute_sequence"] = np.arange(len(frame), dtype=int)
    frame["minute_of_day"] = frame.index.hour * 60 + frame.index.minute
    frame["hour_sin"] = np.sin(2 * np.pi * frame["minute_of_day"] / 1440)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["minute_of_day"] / 1440)
    return frame.reset_index(names="decision_time")


def minute_features(*, symbol: str, root: Path, utc_date: str) -> pd.DataFrame:
    events = pd.read_parquet(
        root / f"date={utc_date}" / "events.parquet",
        columns=["event_time", "price", "quote_quantity", "is_buyer_maker"],
    )
    events = events.set_index("event_time").sort_index()
    taker_buy_quote = events["quote_quantity"].where(~events["is_buyer_maker"], 0.0).resample("1min").sum()
    taker_sell_quote = events["quote_quantity"].where(events["is_buyer_maker"], 0.0).resample("1min").sum()
    price = events["price"].resample("1min").last().ffill()

    frame = pd.DataFrame(index=price.index)
    frame[f"{symbol}_price"] = price
    frame[f"{symbol}_quote_volume_1m"] = taker_buy_quote + taker_sell_quote
    denominator = (taker_buy_quote + taker_sell_quote).replace(0, np.nan)
    frame[f"{symbol}_flow_1m"] = ((taker_buy_quote - taker_sell_quote) / denominator).fillna(0.0)
    frame[f"{symbol}_ret_1m_bps"] = (price / price.shift(1) - 1.0) * 10000.0
    for window in FEATURE_WINDOWS_MINUTES:
        frame[f"{symbol}_ret_{window}m_bps"] = (price / price.shift(window) - 1.0) * 10000.0
        if window > 1:
            frame[f"{symbol}_vol_{window}m_bps"] = frame[f"{symbol}_ret_1m_bps"].rolling(
                window,
                min_periods=max(2, window // 2),
            ).std()
        frame[f"{symbol}_flow_{window}m"] = frame[f"{symbol}_flow_1m"].rolling(
            window,
            min_periods=max(1, window // 2),
        ).mean()
        frame[f"{symbol}_qv_{window}m"] = frame[f"{symbol}_quote_volume_1m"].rolling(
            window,
            min_periods=max(1, window // 2),
        ).sum()
    return frame


def evaluate_all_specs(rows: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for target_symbol in TARGET_SYMBOLS:
        for horizon_seconds in HORIZONS_SECONDS:
            spec_rows = rows.copy()
            shift_rows = int(horizon_seconds / 60 / DECISION_STRIDE_MINUTES)
            spec_rows["future_price"] = spec_rows.groupby("date")[f"{target_symbol}_price"].shift(-shift_rows)
            spec_rows["actual_bps"] = (
                spec_rows["future_price"] / spec_rows[f"{target_symbol}_price"] - 1.0
            ) * 10000.0
            spec_rows = spec_rows.dropna(subset=["actual_bps"])
            train_rows = spec_rows.loc[spec_rows["block"].isin(TRAIN_BLOCKS)]
            validation_rows = spec_rows.loc[spec_rows["block"].eq(VALIDATION_BLOCK)]
            extension_rows = spec_rows.loc[spec_rows["block"].eq(EXTENSION_BLOCK)]
            if train_rows.empty or validation_rows.empty or extension_rows.empty:
                continue
            model = leadlag_model()
            model.fit(train_rows[feature_columns], train_rows["actual_bps"])
            for top_fraction in TOP_FRACTIONS:
                evaluation = {
                    "target_symbol": target_symbol,
                    "horizon_seconds": horizon_seconds,
                    "top_fraction": top_fraction,
                    "train": select_and_summarize(model, train_rows, feature_columns, top_fraction),
                    "validation": select_and_summarize(model, validation_rows, feature_columns, top_fraction),
                    "extension": select_and_summarize(model, extension_rows, feature_columns, top_fraction),
                }
                evaluation["strict_passed"] = (
                    evaluation["train"]["passed"]
                    and evaluation["validation"]["passed"]
                    and evaluation["extension"]["passed"]
                )
                evaluations.append(evaluation)
    return evaluations


def leadlag_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def select_and_summarize(
    model: Pipeline,
    sample: pd.DataFrame,
    feature_columns: list[str],
    top_fraction: float,
) -> dict[str, Any]:
    predictions = model.predict(sample[feature_columns])
    ranked = sample.assign(prediction=predictions, edge=np.abs(predictions), side=np.sign(predictions))
    ranked = ranked.loc[ranked["side"].ne(0)]
    selection_size = max(1, int(len(ranked) * top_fraction))
    selected = ranked.nlargest(selection_size, "edge")
    net_bps = selected["actual_bps"] * selected["side"] - EXPLICIT_COST_BPS
    block_summaries = []
    for block, block_rows in selected.groupby("block", sort=True):
        block_net = block_rows["actual_bps"] * block_rows["side"] - EXPLICIT_COST_BPS
        block_summaries.append(
            {
                "block": block,
                "trade_count": int(len(block_rows)),
                "net_total_bps": float(block_net.sum()),
                "net_mean_bps": float(block_net.mean()) if len(block_net) else 0.0,
                "win_rate": float((block_net > 0).mean()) if len(block_net) else 0.0,
            }
        )
    return {
        "passed": bool(len(net_bps) >= 3 and net_bps.sum() > 0),
        "trade_count": int(len(selected)),
        "net_total_bps": float(net_bps.sum()) if len(net_bps) else 0.0,
        "net_mean_bps": float(net_bps.mean()) if len(net_bps) else 0.0,
        "win_rate": float((net_bps > 0).mean()) if len(net_bps) else 0.0,
        "blocks": block_summaries,
    }


def fast174_status(
    *,
    screen_survivor_count: int,
    validation_survivor_count: int,
    strict_survivor_count: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "cross_asset_leadlag_smoke"]
    if strict_survivor_count:
        return (
            "FAST174_CROSS_ASSET_LEADLAG_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons + [f"strict_survivors:{strict_survivor_count}", "fresh_preregistered_observation_required"],
        )
    if validation_survivor_count:
        return (
            "FAST174_CROSS_ASSET_LEADLAG_FAILED_EXTENSION",
            reasons
            + [
                f"screen_survivors:{screen_survivor_count}",
                f"validation_survivors:{validation_survivor_count}",
                "validation_survivors_failed_extension",
            ],
        )
    if screen_survivor_count:
        return (
            "FAST174_CROSS_ASSET_LEADLAG_FAILED_VALIDATION",
            reasons + [f"screen_survivors:{screen_survivor_count}", "screen_survivors_failed_validation"],
        )
    return (
        "FAST174_CROSS_ASSET_LEADLAG_FAILED_SCREEN",
        reasons + ["no_train_screen_survivors"],
    )


def top_candidates(evaluations: list[dict[str, Any]], *, key_sample: str, limit: int) -> list[dict[str, Any]]:
    return [
        compact_candidate(row)
        for row in sorted(
            evaluations,
            key=lambda item: (
                item[key_sample]["net_total_bps"],
                item[key_sample]["trade_count"],
                item["train"]["net_total_bps"],
            ),
            reverse=True,
        )[:limit]
    ]


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_symbol": row["target_symbol"],
        "horizon_seconds": row["horizon_seconds"],
        "top_fraction": row["top_fraction"],
        "strict_passed": row["strict_passed"],
        "train": row["train"],
        "validation": row["validation"],
        "extension": row["extension"],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "research_only": report["research_only"],
        "approved_strategy": report["approved_strategy"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "spec_count": summary["spec_count"],
        "screen_survivor_count": summary["screen_survivor_count"],
        "validation_survivor_count": summary["validation_survivor_count"],
        "strict_survivor_count": summary["strict_survivor_count"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
