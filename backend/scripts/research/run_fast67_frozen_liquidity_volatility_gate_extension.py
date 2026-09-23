"""Fast67 frozen liquidity/volatility gate extension audit.

Fast66 found the strongest family so far: frozen top-p signals gated by high
recent quote volume and high realized volatility. Fast67 removes the last
moving part before any shadow test: it freezes numeric thresholds from the
historical development set, applies them back to the historical blocks, and
then applies the exact same thresholds to any later extension blocks available.

This is still research-only. A failed extension kills the candidate instead of
becoming a reason to tune the rule in place.
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
    load_selected_trades_with_causal_features,
    one_position_summary,
)
from scripts.research.run_fast66_liquidity_volatility_gate_audit import (  # noqa: E402
    DEFAULT_FAST65_MERGED,
    load_or_build_selected_trades,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast67-frozen-liquidity-volatility-extension"
DEFAULT_FAST56_REPORT = (
    DEFAULT_DATA_ROOT
    / "reports"
    / "latency-adjusted-frozen-top-p-fast56-new-holdout-sep21-sep22-100ms"
    / "report.json"
)


@dataclass(frozen=True)
class FrozenLiquidityVolatilityGate:
    name: str
    calibration_scope: str
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
    parser.add_argument("--quote-volume-quantile", type=float, default=0.80)
    parser.add_argument("--realized-vol-quantile", type=float, default=0.75)
    parser.add_argument("--minimum-block-trades", type=int, default=3)
    parser.add_argument(
        "--extension-report",
        type=Path,
        action="append",
        default=None,
        help=(
            "Later frozen top-p report to audit with the frozen gate. "
            "Defaults to Fast56 if present."
        ),
    )
    arguments = parser.parse_args()

    extension_reports = (
        tuple(arguments.extension_report)
        if arguments.extension_report is not None
        else ((DEFAULT_FAST56_REPORT,) if DEFAULT_FAST56_REPORT.exists() else tuple())
    )
    report = run_fast67_frozen_liquidity_volatility_gate_extension(
        data_root=arguments.data_root,
        fast65_merged_path=arguments.fast65_merged,
        output_root=arguments.output_root,
        historical_fast_start=arguments.historical_fast_start,
        historical_fast_end=arguments.historical_fast_end,
        quote_volume_quantile=arguments.quote_volume_quantile,
        realized_vol_quantile=arguments.realized_vol_quantile,
        minimum_block_trades=arguments.minimum_block_trades,
        extension_report_paths=extension_reports,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast67_frozen_liquidity_volatility_gate_extension(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    output_root: Path,
    historical_fast_start: int = 43,
    historical_fast_end: int = 53,
    quote_volume_quantile: float = 0.80,
    realized_vol_quantile: float = 0.75,
    minimum_block_trades: int = 3,
    extension_report_paths: tuple[Path, ...] = tuple(),
) -> dict[str, Any]:
    _validate_parameters(
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        quote_volume_quantile=quote_volume_quantile,
        realized_vol_quantile=realized_vol_quantile,
        minimum_block_trades=minimum_block_trades,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    historical_trades = load_or_build_selected_trades(
        data_root=data_root,
        fast65_merged_path=fast65_merged_path,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
    )
    if historical_trades.empty:
        return _write_report(
            output_root,
            _base_report(
                data_root=data_root,
                fast65_merged_path=fast65_merged_path,
                historical_fast_start=historical_fast_start,
                historical_fast_end=historical_fast_end,
                quote_volume_quantile=quote_volume_quantile,
                realized_vol_quantile=realized_vol_quantile,
                minimum_block_trades=minimum_block_trades,
                extension_report_paths=extension_report_paths,
            )
            | {
                "status": "FAST67_NO_HISTORICAL_TRADES",
                "status_reasons": ["fast65_selected_trades_unavailable"],
            },
        )

    gate = freeze_gate_from_historical(
        historical_trades,
        quote_volume_quantile=quote_volume_quantile,
        realized_vol_quantile=realized_vol_quantile,
        calibration_scope=f"fast{historical_fast_start}_fast{historical_fast_end}",
    )
    historical_filtered = apply_frozen_gate(historical_trades, gate)
    historical_blocks = summarize_blocks(
        historical_filtered,
        all_block_names=sorted(str(value) for value in historical_trades["block"].unique()),
        minimum_block_trades=minimum_block_trades,
    )
    historical_aggregate = aggregate_block_summaries(
        historical_blocks,
        minimum_block_trades=minimum_block_trades,
    )

    extension_trades = load_extension_trades(extension_report_paths, data_root=data_root)
    extension_filtered = apply_frozen_gate(extension_trades, gate)
    extension_blocks = summarize_blocks(
        extension_filtered,
        all_block_names=tuple(
            path.parent.name for path in extension_report_paths if path.exists()
        ),
        minimum_block_trades=minimum_block_trades,
    )
    extension_aggregate = aggregate_block_summaries(
        extension_blocks,
        minimum_block_trades=minimum_block_trades,
    )
    status, status_reasons = fast67_status(
        historical_aggregate=historical_aggregate,
        extension_aggregate=extension_aggregate,
        minimum_block_trades=minimum_block_trades,
    )

    gate_path = output_root / "frozen_gate.json"
    gate_path.write_text(json.dumps(asdict(gate), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    historical_path = output_root / "historical_blocks.json"
    historical_path.write_text(
        json.dumps(historical_blocks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    extension_path = output_root / "extension_blocks.json"
    extension_path.write_text(
        json.dumps(extension_blocks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = _base_report(
        data_root=data_root,
        fast65_merged_path=fast65_merged_path,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        quote_volume_quantile=quote_volume_quantile,
        realized_vol_quantile=realized_vol_quantile,
        minimum_block_trades=minimum_block_trades,
        extension_report_paths=extension_report_paths,
    ) | {
        "status": status,
        "status_reasons": status_reasons,
        "frozen_gate": asdict(gate),
        "frozen_gate_path": str(gate_path),
        "frozen_gate_sha256": _sha256_file(gate_path),
        "historical_trade_count": int(len(historical_trades)),
        "historical_filtered_signal_count": int(len(historical_filtered)),
        "historical_blocks_path": str(historical_path),
        "historical_blocks_sha256": _sha256_file(historical_path),
        "historical_aggregate": historical_aggregate,
        "historical_blocks": historical_blocks,
        "extension_trade_count": int(len(extension_trades)),
        "extension_filtered_signal_count": int(len(extension_filtered)),
        "extension_blocks_path": str(extension_path),
        "extension_blocks_sha256": _sha256_file(extension_path),
        "extension_aggregate": extension_aggregate,
        "extension_blocks": extension_blocks,
    }
    return _write_report(output_root, report)


def freeze_gate_from_historical(
    historical_trades: pd.DataFrame,
    *,
    quote_volume_quantile: float,
    realized_vol_quantile: float,
    calibration_scope: str,
) -> FrozenLiquidityVolatilityGate:
    quote_volume_threshold = float(
        historical_trades["quote_volume_30s"].quantile(quote_volume_quantile)
    )
    realized_vol_threshold = float(
        historical_trades["realized_vol_30s_bps"].quantile(realized_vol_quantile)
    )
    return FrozenLiquidityVolatilityGate(
        name=(
            f"quote_volume_30s>={quote_volume_threshold:.10g}"
            f"&realized_vol_30s_bps>={realized_vol_threshold:.10g}"
        ),
        calibration_scope=calibration_scope,
        quote_volume_quantile=quote_volume_quantile,
        realized_vol_quantile=realized_vol_quantile,
        quote_volume_threshold=quote_volume_threshold,
        realized_vol_threshold=realized_vol_threshold,
    )


def apply_frozen_gate(
    frame: pd.DataFrame,
    gate: FrozenLiquidityVolatilityGate,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[
        (frame["quote_volume_30s"] >= gate.quote_volume_threshold)
        & (frame["realized_vol_30s_bps"] >= gate.realized_vol_threshold)
    ].copy()


def load_extension_trades(
    extension_report_paths: tuple[Path, ...],
    *,
    data_root: Path,
) -> pd.DataFrame:
    existing_paths = tuple(path for path in extension_report_paths if path.exists())
    if not existing_paths:
        return pd.DataFrame()
    return load_selected_trades_with_causal_features(existing_paths, data_root=data_root)


def summarize_blocks(
    frame: pd.DataFrame,
    *,
    all_block_names: tuple[str, ...] | list[str],
    minimum_block_trades: int,
) -> list[dict[str, Any]]:
    grouped = {
        str(block): block_frame
        for block, block_frame in frame.groupby("block", sort=True)
    } if not frame.empty else {}
    rows = []
    for block in all_block_names:
        summary = one_position_summary(grouped.get(str(block), pd.DataFrame()))
        rows.append(
            {
                "block": str(block),
                "minimum_trade_sample": int(summary["trade_count"]) >= minimum_block_trades,
                "passed_stress": (
                    int(summary["trade_count"]) >= minimum_block_trades
                    and float(summary["stress_total_log_return_bps"]) > 0
                    and float(summary["stress_mean_bps"]) > 0
                ),
                **summary,
            }
        )
    return rows


def aggregate_block_summaries(
    blocks: list[dict[str, Any]],
    *,
    minimum_block_trades: int,
) -> dict[str, Any]:
    return {
        "block_count": len(blocks),
        "minimum_trade_block_count": sum(
            int(row["trade_count"]) >= minimum_block_trades for row in blocks
        ),
        "passed_stress_count": sum(bool(row["passed_stress"]) for row in blocks),
        "stress_total_log_return_bps": float(
            sum(float(row["stress_total_log_return_bps"]) for row in blocks)
        ),
        "expected_total_log_return_bps": float(
            sum(float(row["expected_total_log_return_bps"]) for row in blocks)
        ),
        "trade_count": int(sum(int(row["trade_count"]) for row in blocks)),
        "min_block_stress_total_log_return_bps": min(
            (float(row["stress_total_log_return_bps"]) for row in blocks),
            default=0.0,
        ),
        "negative_sufficient_blocks": [
            row["block"]
            for row in blocks
            if int(row["trade_count"]) >= minimum_block_trades
            and float(row["stress_total_log_return_bps"]) <= 0
        ],
    }


def fast67_status(
    *,
    historical_aggregate: dict[str, Any],
    extension_aggregate: dict[str, Any],
    minimum_block_trades: int,
) -> tuple[str, list[str]]:
    reasons = [
        "not_approved_strategy",
        "research_only",
        "no_order_execution",
        f"minimum_block_trades:{minimum_block_trades}",
    ]
    historical_negative = historical_aggregate["negative_sufficient_blocks"]
    extension_negative = extension_aggregate["negative_sufficient_blocks"]
    if extension_negative:
        return "FAST67_FROZEN_GATE_FAILED_EXTENSION", reasons + [
            f"extension_negative_sufficient_blocks:{len(extension_negative)}",
            f"historical_negative_sufficient_blocks:{len(historical_negative)}",
        ]
    if historical_negative:
        return "FAST67_FROZEN_GATE_FAILED_HISTORICAL_ROBUSTNESS", reasons + [
            f"historical_negative_sufficient_blocks:{len(historical_negative)}"
        ]
    if int(extension_aggregate["block_count"]) == 0:
        return "FAST67_FROZEN_GATE_PREREGISTERED_PENDING_EXTENSION", reasons + [
            "no_extension_blocks_available"
        ]
    return "FAST67_FROZEN_GATE_RESEARCH_CANDIDATE_SHADOW_ONLY", reasons + [
        "requires_clean_prospective_shadow_before_testnet"
    ]


def _base_report(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    historical_fast_start: int,
    historical_fast_end: int,
    quote_volume_quantile: float,
    realized_vol_quantile: float,
    minimum_block_trades: int,
    extension_report_paths: tuple[Path, ...],
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
            "fast67_is_frozen_gate_research_audit_only",
            "thresholds_are_calibrated_on_historical_fast43_fast53",
            "extension_failure_kills_candidate",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast67_frozen_liquidity_volatility_extension",
        "data_root": str(data_root),
        "fast65_merged_path": str(fast65_merged_path),
        "historical_fast_start": historical_fast_start,
        "historical_fast_end": historical_fast_end,
        "quote_volume_quantile": quote_volume_quantile,
        "realized_vol_quantile": realized_vol_quantile,
        "minimum_block_trades": minimum_block_trades,
        "extension_report_paths": [str(path) for path in extension_report_paths],
    }


def _validate_parameters(
    *,
    historical_fast_start: int,
    historical_fast_end: int,
    quote_volume_quantile: float,
    realized_vol_quantile: float,
    minimum_block_trades: int,
) -> None:
    if historical_fast_start > historical_fast_end:
        raise ValueError("historical_fast_start cannot exceed historical_fast_end")
    for name, value in {
        "quote_volume_quantile": quote_volume_quantile,
        "realized_vol_quantile": realized_vol_quantile,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if minimum_block_trades < 1:
        raise ValueError("minimum_block_trades must be positive")


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
        "frozen_gate": report.get("frozen_gate"),
        "historical_aggregate": report.get("historical_aggregate"),
        "extension_aggregate": report.get("extension_aggregate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
