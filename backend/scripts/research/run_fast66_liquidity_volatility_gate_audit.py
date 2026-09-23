"""Fast66 constrained liquidity/volatility gate audit.

Fast65 found that causal microstructure gates improve the frozen top-p stream
but remain unstable. Fast66 narrows the family to the recurring signal:
high recent quote volume AND high realized volatility. Thresholds are computed
on training blocks only and then applied to the held-out block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast65_causal_feature_gate_walkforward import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    discover_historical_reports,
    load_selected_trades_with_causal_features,
    one_position_summary,
)


DEFAULT_FAST65_MERGED = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "fast65-causal-feature-gate-walkforward"
    / "selected_trades_with_causal_features.parquet"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast66-liquidity-volatility-gate-audit"
DEFAULT_VOLUME_QUANTILES = (0.80, 0.90, 0.95)
DEFAULT_VOLATILITY_QUANTILES = (0.75, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class LiquidityVolatilityGate:
    name: str
    quote_volume_quantile: float
    realized_vol_quantile: float
    quote_volume_threshold: float
    realized_vol_threshold: float


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fast65-merged", type=Path, default=DEFAULT_FAST65_MERGED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--historical-fast-start", type=int, default=43)
    parser.add_argument("--historical-fast-end", type=int, default=53)
    parser.add_argument("--minimum-holdout-trades", type=int, default=3)
    arguments = parser.parse_args()

    report = run_fast66_liquidity_volatility_gate_audit(
        data_root=arguments.data_root,
        fast65_merged_path=arguments.fast65_merged,
        output_root=arguments.output_root,
        historical_fast_start=arguments.historical_fast_start,
        historical_fast_end=arguments.historical_fast_end,
        minimum_holdout_trades=arguments.minimum_holdout_trades,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast66_liquidity_volatility_gate_audit(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    output_root: Path,
    historical_fast_start: int = 43,
    historical_fast_end: int = 53,
    minimum_holdout_trades: int = 3,
) -> dict[str, Any]:
    if historical_fast_start > historical_fast_end:
        raise ValueError("historical_fast_start cannot exceed historical_fast_end")
    if minimum_holdout_trades < 1:
        raise ValueError("minimum_holdout_trades must be positive")

    selected_trades = load_or_build_selected_trades(
        data_root=data_root,
        fast65_merged_path=fast65_merged_path,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if selected_trades.empty:
        return _write_report(
            output_root,
            _base_report(
                data_root=data_root,
                fast65_merged_path=fast65_merged_path,
                historical_fast_start=historical_fast_start,
                historical_fast_end=historical_fast_end,
                minimum_holdout_trades=minimum_holdout_trades,
            )
            | {
                "status": "FAST66_NO_SELECTED_TRADES",
                "status_reasons": ["fast65_selected_trades_unavailable"],
            },
        )

    fixed_holdouts = run_fixed_gate_walkforward(
        selected_trades,
        quote_volume_quantile=0.80,
        realized_vol_quantile=0.75,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    selected_holdouts = run_selected_gate_walkforward(
        selected_trades,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    fixed_aggregate = aggregate_holdouts(
        fixed_holdouts,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    selected_aggregate = aggregate_holdouts(
        selected_holdouts,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    status, status_reasons = fast66_status(
        fixed_aggregate=fixed_aggregate,
        selected_aggregate=selected_aggregate,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    fixed_path = output_root / "fixed_q80_q75_holdouts.json"
    fixed_path.write_text(
        json.dumps(fixed_holdouts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_path = output_root / "selected_strict_and_holdouts.json"
    selected_path.write_text(
        json.dumps(selected_holdouts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = _base_report(
        data_root=data_root,
        fast65_merged_path=fast65_merged_path,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        minimum_holdout_trades=minimum_holdout_trades,
    ) | {
        "status": status,
        "status_reasons": status_reasons,
        "trade_count": int(len(selected_trades)),
        "block_count": int(selected_trades["block"].nunique()),
        "fixed_gate": {
            "quote_volume_quantile": 0.80,
            "realized_vol_quantile": 0.75,
        },
        "fixed_aggregate": fixed_aggregate,
        "selected_strict_and_aggregate": selected_aggregate,
        "fixed_holdouts_path": str(fixed_path),
        "fixed_holdouts_sha256": _sha256_file(fixed_path),
        "selected_strict_and_holdouts_path": str(selected_path),
        "selected_strict_and_holdouts_sha256": _sha256_file(selected_path),
        "fixed_holdouts": fixed_holdouts,
        "selected_strict_and_holdouts": selected_holdouts,
    }
    return _write_report(output_root, report)


def load_or_build_selected_trades(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    historical_fast_start: int,
    historical_fast_end: int,
) -> pd.DataFrame:
    if fast65_merged_path.exists():
        return pd.read_parquet(fast65_merged_path)
    report_paths = discover_historical_reports(
        data_root,
        fast_start=historical_fast_start,
        fast_end=historical_fast_end,
    )
    return load_selected_trades_with_causal_features(report_paths, data_root=data_root)


def run_fixed_gate_walkforward(
    selected_trades: pd.DataFrame,
    *,
    quote_volume_quantile: float,
    realized_vol_quantile: float,
    minimum_holdout_trades: int,
) -> list[dict[str, Any]]:
    holdouts = []
    for holdout_block in sorted(selected_trades["block"].unique()):
        training = selected_trades[selected_trades["block"] != holdout_block]
        holdout = selected_trades[selected_trades["block"] == holdout_block]
        gate = make_gate(
            training,
            quote_volume_quantile=quote_volume_quantile,
            realized_vol_quantile=realized_vol_quantile,
        )
        holdouts.append(evaluate_holdout(holdout_block, holdout, gate, minimum_holdout_trades))
    return holdouts


def run_selected_gate_walkforward(
    selected_trades: pd.DataFrame,
    *,
    minimum_holdout_trades: int,
) -> list[dict[str, Any]]:
    holdouts = []
    for holdout_block in sorted(selected_trades["block"].unique()):
        training = selected_trades[selected_trades["block"] != holdout_block]
        holdout = selected_trades[selected_trades["block"] == holdout_block]
        training_candidates = [
            evaluate_training_gate(
                training,
                make_gate(
                    training,
                    quote_volume_quantile=quote_volume_quantile,
                    realized_vol_quantile=realized_vol_quantile,
                ),
                minimum_holdout_trades=minimum_holdout_trades,
            )
            for quote_volume_quantile in DEFAULT_VOLUME_QUANTILES
            for realized_vol_quantile in DEFAULT_VOLATILITY_QUANTILES
        ]
        training_candidates.sort(
            key=lambda item: (
                int(item["passed_stress_count"]),
                float(item["min_holdout_stress_total_log_return_bps"]),
                float(item["stress_total_log_return_bps"]),
                int(item["minimum_trade_holdout_count"]),
            ),
            reverse=True,
        )
        gate = gate_from_dict(training_candidates[0]["gate"])
        row = evaluate_holdout(holdout_block, holdout, gate, minimum_holdout_trades)
        row["training_best"] = training_candidates[0]
        row["training_candidate_count"] = len(training_candidates)
        holdouts.append(row)
    return holdouts


def make_gate(
    frame: pd.DataFrame,
    *,
    quote_volume_quantile: float,
    realized_vol_quantile: float,
) -> LiquidityVolatilityGate:
    quote_volume_threshold = float(frame["quote_volume_30s"].quantile(quote_volume_quantile))
    realized_vol_threshold = float(frame["realized_vol_30s_bps"].quantile(realized_vol_quantile))
    return LiquidityVolatilityGate(
        name=(
            f"quote_volume_30s>=q{quote_volume_quantile:.2f}"
            f"&realized_vol_30s_bps>=q{realized_vol_quantile:.2f}"
        ),
        quote_volume_quantile=quote_volume_quantile,
        realized_vol_quantile=realized_vol_quantile,
        quote_volume_threshold=quote_volume_threshold,
        realized_vol_threshold=realized_vol_threshold,
    )


def apply_gate(frame: pd.DataFrame, gate: LiquidityVolatilityGate) -> pd.DataFrame:
    return frame[
        (frame["quote_volume_30s"] >= gate.quote_volume_threshold)
        & (frame["realized_vol_30s_bps"] >= gate.realized_vol_threshold)
    ].copy()


def evaluate_holdout(
    holdout_block: str,
    holdout: pd.DataFrame,
    gate: LiquidityVolatilityGate,
    minimum_holdout_trades: int,
) -> dict[str, Any]:
    filtered = apply_gate(holdout, gate)
    summary = one_position_summary(filtered)
    return {
        "holdout_block": str(holdout_block),
        "gate": asdict(gate),
        "holdout_selected_signal_count": int(len(filtered)),
        "holdout_summary": summary,
        "passed_stress": (
            int(summary["trade_count"]) >= minimum_holdout_trades
            and float(summary["stress_total_log_return_bps"]) > 0
            and float(summary["stress_mean_bps"]) > 0
        ),
    }


def evaluate_training_gate(
    training: pd.DataFrame,
    gate: LiquidityVolatilityGate,
    *,
    minimum_holdout_trades: int,
) -> dict[str, Any]:
    holdouts = [
        evaluate_holdout(block, block_frame, gate, minimum_holdout_trades)
        for block, block_frame in training.groupby("block", sort=True)
    ]
    aggregate = aggregate_holdouts(holdouts, minimum_holdout_trades=minimum_holdout_trades)
    return {"gate": asdict(gate), **aggregate}


def aggregate_holdouts(
    holdouts: list[dict[str, Any]],
    *,
    minimum_holdout_trades: int,
) -> dict[str, Any]:
    summaries = [holdout["holdout_summary"] for holdout in holdouts]
    return {
        "holdout_count": len(holdouts),
        "passed_stress_count": sum(bool(holdout["passed_stress"]) for holdout in holdouts),
        "minimum_trade_holdout_count": sum(
            int(summary["trade_count"]) >= minimum_holdout_trades for summary in summaries
        ),
        "stress_total_log_return_bps": float(
            sum(float(summary["stress_total_log_return_bps"]) for summary in summaries)
        ),
        "expected_total_log_return_bps": float(
            sum(float(summary["expected_total_log_return_bps"]) for summary in summaries)
        ),
        "trade_count": int(sum(int(summary["trade_count"]) for summary in summaries)),
        "min_holdout_stress_total_log_return_bps": min(
            (float(summary["stress_total_log_return_bps"]) for summary in summaries),
            default=0.0,
        ),
        "selected_gates": [holdout["gate"]["name"] for holdout in holdouts],
    }


def fast66_status(
    *,
    fixed_aggregate: dict[str, Any],
    selected_aggregate: dict[str, Any],
    minimum_holdout_trades: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "diagnostic_only", "no_order_execution"]
    best = (
        fixed_aggregate
        if int(fixed_aggregate["passed_stress_count"])
        >= int(selected_aggregate["passed_stress_count"])
        else selected_aggregate
    )
    holdout_count = int(best["holdout_count"])
    if int(best["passed_stress_count"]) < holdout_count:
        return "FAST66_LIQUIDITY_VOLATILITY_GATE_PROMISING_BUT_FAILED_ROBUSTNESS", reasons + [
            f"best_passed_stress_holdouts:{best['passed_stress_count']}<{holdout_count}",
            f"minimum_holdout_trades:{minimum_holdout_trades}",
        ]
    if float(best["stress_total_log_return_bps"]) <= 0:
        return "FAST66_LIQUIDITY_VOLATILITY_GATE_FAILED", reasons + [
            "aggregate_stress_total_non_positive"
        ]
    return "FAST66_LIQUIDITY_VOLATILITY_GATE_CANDIDATE_RESEARCH_ONLY", reasons + [
        "requires_preregistration_and_prospective_shadow_before_testnet"
    ]


def gate_from_dict(payload: dict[str, Any]) -> LiquidityVolatilityGate:
    return LiquidityVolatilityGate(
        name=str(payload["name"]),
        quote_volume_quantile=float(payload["quote_volume_quantile"]),
        realized_vol_quantile=float(payload["realized_vol_quantile"]),
        quote_volume_threshold=float(payload["quote_volume_threshold"]),
        realized_vol_threshold=float(payload["realized_vol_threshold"]),
    )


def _base_report(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    historical_fast_start: int,
    historical_fast_end: int,
    minimum_holdout_trades: int,
) -> dict[str, Any]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast66_is_diagnostic_walkforward_only",
            "liquidity_volatility_gate_selected_from_training_blocks_only",
            "post_entry_outcome_fields_excluded_from_gate_features",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast66_liquidity_volatility_gate_audit",
        "data_root": str(data_root),
        "fast65_merged_path": str(fast65_merged_path),
        "historical_fast_start": historical_fast_start,
        "historical_fast_end": historical_fast_end,
        "minimum_holdout_trades": minimum_holdout_trades,
        "candidate_quantiles": {
            "quote_volume": list(DEFAULT_VOLUME_QUANTILES),
            "realized_volatility": list(DEFAULT_VOLATILITY_QUANTILES),
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report.json"
    payload = {**report, "report_path": str(report_path)}
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "trade_count": report.get("trade_count"),
        "block_count": report.get("block_count"),
        "fixed_aggregate": report.get("fixed_aggregate"),
        "selected_strict_and_aggregate": report.get("selected_strict_and_aggregate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
