"""Fast73 anti-stop quote-volume density audit.

The current top-p target frequently predicts that the stress touch happens
before the stop, but many of those trades still lose after trailing replay.
Fast73 audits a different hypothesis over the frozen top-p stream:

1. keep only high 30s quote-volume signals, because this historically reduces
   initial STOP outcomes;
2. cap local signal density, because clustered bursts caused the large Fast49
   drawdown.

The rule is evaluated on historical Fast43-Fast53, then on the later Fast56
extension and optional Sep23 WAL settlements. It remains research-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from scripts.research.run_fast67_frozen_liquidity_volatility_gate_extension import (  # noqa: E402
    DEFAULT_FAST56_REPORT,
)
from scripts.research.run_fast61_fast60_shadow_settlement import (  # noqa: E402
    _apply_one_position_constraint,
    _summary_with_labels,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast73-anti-stop-density-audit"
DEFAULT_FAST69_RAW_DECISIONS = (
    DEFAULT_DATA_ROOT / "reports" / "fast69-directional-shadow-dry-run" / "raw_top_p_decisions.jsonl"
)
DEFAULT_FAST73_SETTLEMENT_CACHE = (
    DEFAULT_DATA_ROOT / "reports" / "fast73-top-p-settlement-cache" / "raw_top_p_settlements.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fast65-merged", type=Path, default=DEFAULT_FAST65_MERGED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--historical-fast-start", type=int, default=43)
    parser.add_argument("--historical-fast-end", type=int, default=53)
    parser.add_argument("--quote-volume-quantile", type=float, default=0.80)
    parser.add_argument("--density-lookback-seconds", type=int, default=300)
    parser.add_argument("--density-limit", type=int, default=2)
    parser.add_argument("--extension-report", type=Path, default=DEFAULT_FAST56_REPORT)
    parser.add_argument("--sep23-raw-decisions", type=Path, default=DEFAULT_FAST69_RAW_DECISIONS)
    parser.add_argument("--sep23-settlements", type=Path, default=DEFAULT_FAST73_SETTLEMENT_CACHE)
    arguments = parser.parse_args()

    report = run_fast73_anti_stop_density_audit(
        data_root=arguments.data_root,
        fast65_merged_path=arguments.fast65_merged,
        output_root=arguments.output_root,
        historical_fast_start=arguments.historical_fast_start,
        historical_fast_end=arguments.historical_fast_end,
        quote_volume_quantile=arguments.quote_volume_quantile,
        density_lookback_seconds=arguments.density_lookback_seconds,
        density_limit=arguments.density_limit,
        extension_report_path=arguments.extension_report,
        sep23_raw_decisions_path=arguments.sep23_raw_decisions,
        sep23_settlements_path=arguments.sep23_settlements,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast73_anti_stop_density_audit(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    output_root: Path,
    historical_fast_start: int = 43,
    historical_fast_end: int = 53,
    quote_volume_quantile: float = 0.80,
    density_lookback_seconds: int = 300,
    density_limit: int = 2,
    extension_report_path: Path = DEFAULT_FAST56_REPORT,
    sep23_raw_decisions_path: Path = DEFAULT_FAST69_RAW_DECISIONS,
    sep23_settlements_path: Path = DEFAULT_FAST73_SETTLEMENT_CACHE,
) -> dict[str, Any]:
    _validate_parameters(
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        quote_volume_quantile=quote_volume_quantile,
        density_lookback_seconds=density_lookback_seconds,
        density_limit=density_limit,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    historical = load_or_build_selected_trades(
        data_root=data_root,
        fast65_merged_path=fast65_merged_path,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
    )
    quote_volume_threshold = float(historical["quote_volume_30s"].quantile(quote_volume_quantile))
    historical_filtered = apply_quote_volume_density_gate(
        historical,
        quote_volume_threshold=quote_volume_threshold,
        lookback_seconds=density_lookback_seconds,
        limit=density_limit,
    )
    historical_blocks = summarize_dataframe_blocks(
        historical_filtered,
        all_block_names=tuple(sorted(str(value) for value in historical["block"].unique())),
    )
    historical_aggregate = aggregate_blocks(historical_blocks)
    target_alignment = target_alignment_report(historical)

    extension = (
        load_selected_trades_with_causal_features((extension_report_path,), data_root=data_root)
        if extension_report_path.exists()
        else pd.DataFrame()
    )
    extension_filtered = apply_quote_volume_density_gate(
        extension,
        quote_volume_threshold=quote_volume_threshold,
        lookback_seconds=density_lookback_seconds,
        limit=density_limit,
    )
    extension_blocks = summarize_dataframe_blocks(
        extension_filtered,
        all_block_names=(extension_report_path.parent.name,) if extension_report_path.exists() else tuple(),
    )
    extension_aggregate = aggregate_blocks(extension_blocks)

    sep23 = evaluate_sep23_settlements(
        raw_decisions_path=sep23_raw_decisions_path,
        settlements_path=sep23_settlements_path,
        quote_volume_threshold=quote_volume_threshold,
        lookback_seconds=density_lookback_seconds,
        limit=density_limit,
    )
    status, status_reasons = fast73_status(
        historical_aggregate=historical_aggregate,
        extension_aggregate=extension_aggregate,
        sep23=sep23,
    )

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
    sep23_path = output_root / "sep23_summary.json"
    sep23_path.write_text(json.dumps(sep23, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast73_is_post_failure_diagnostic",
            "extension_and_sep23_must_pass_before_shadow_candidate",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast73_anti_stop_density_audit",
        "status": status,
        "status_reasons": status_reasons,
        "data_root": str(data_root),
        "fast65_merged_path": str(fast65_merged_path),
        "historical_fast_start": historical_fast_start,
        "historical_fast_end": historical_fast_end,
        "quote_volume_quantile": quote_volume_quantile,
        "quote_volume_threshold": quote_volume_threshold,
        "density_lookback_seconds": density_lookback_seconds,
        "density_limit": density_limit,
        "target_alignment": target_alignment,
        "historical_aggregate": historical_aggregate,
        "historical_blocks_path": str(historical_path),
        "historical_blocks_sha256": _sha256_file(historical_path),
        "extension_aggregate": extension_aggregate,
        "extension_blocks_path": str(extension_path),
        "extension_blocks_sha256": _sha256_file(extension_path),
        "sep23": sep23,
        "sep23_summary_path": str(sep23_path),
        "sep23_summary_sha256": _sha256_file(sep23_path),
    }
    return _write_report(output_root, report)


def apply_quote_volume_density_gate(
    frame: pd.DataFrame,
    *,
    quote_volume_threshold: float,
    lookback_seconds: int,
    limit: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    candidates = frame[frame["quote_volume_30s"] >= quote_volume_threshold].copy()
    rows = density_filter_records(
        candidates.to_dict("records"),
        lookback_seconds=lookback_seconds,
        limit=limit,
    )
    return pd.DataFrame(rows)


def density_filter_records(
    records: list[dict[str, Any]],
    *,
    lookback_seconds: int,
    limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: (int(row["decision_time_ms"]), -float(row.get("probability", 0.0))),
    )
    times: list[int] = []
    lookback_ms = lookback_seconds * 1000
    kept = []
    for row in ordered:
        decision_time_ms = int(row["decision_time_ms"])
        recent_count = sum(time_ms >= decision_time_ms - lookback_ms for time_ms in times)
        enriched = {**row, "recent_filtered_signal_count": recent_count}
        if recent_count <= limit:
            kept.append(enriched)
        times.append(decision_time_ms)
    return kept


def summarize_dataframe_blocks(
    frame: pd.DataFrame,
    *,
    all_block_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped = {str(block): block_frame for block, block_frame in frame.groupby("block", sort=True)} if not frame.empty else {}
    blocks = []
    for block_name in all_block_names:
        summary = one_position_summary(grouped.get(block_name, pd.DataFrame()))
        blocks.append({"block": block_name, **summary})
    return blocks


def aggregate_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    sufficient = [row for row in blocks if int(row["trade_count"]) >= 3]
    return {
        "block_count": len(blocks),
        "minimum_trade_block_count": len(sufficient),
        "passed_stress_count": sum(
            float(row["stress_total_log_return_bps"]) > 0
            and float(row["stress_mean_bps"]) > 0
            for row in sufficient
        ),
        "trade_count": int(sum(int(row["trade_count"]) for row in blocks)),
        "expected_total_log_return_bps": float(
            sum(float(row["expected_total_log_return_bps"]) for row in blocks)
        ),
        "stress_total_log_return_bps": float(
            sum(float(row["stress_total_log_return_bps"]) for row in blocks)
        ),
        "min_sufficient_block_stress_total_log_return_bps": min(
            (float(row["stress_total_log_return_bps"]) for row in sufficient),
            default=0.0,
        ),
        "negative_sufficient_blocks": [
            row["block"] for row in sufficient if float(row["stress_total_log_return_bps"]) <= 0
        ],
    }


def target_alignment_report(frame: pd.DataFrame) -> dict[str, Any]:
    target_hit = frame["paid_stress_before_stop"].astype(bool)
    stress_positive = frame["stress_net_bps"] > 0
    return {
        "trade_count": int(len(frame)),
        "target_hit_rate": float(target_hit.mean()),
        "stress_positive_rate": float(stress_positive.mean()),
        "target_hit_stress_loss_count": int((target_hit & ~stress_positive).sum()),
        "target_miss_stress_win_count": int((~target_hit & stress_positive).sum()),
        "stop_count": int(frame["first_touch"].eq("STOP").sum()),
        "stop_stress_total_log_return_bps": float(
            frame.loc[frame["first_touch"].eq("STOP"), "stress_net_bps"].sum()
        ),
        "non_stop_stress_total_log_return_bps": float(
            frame.loc[~frame["first_touch"].eq("STOP"), "stress_net_bps"].sum()
        ),
    }


def evaluate_sep23_settlements(
    *,
    raw_decisions_path: Path,
    settlements_path: Path,
    quote_volume_threshold: float,
    lookback_seconds: int,
    limit: int,
) -> dict[str, Any]:
    if not raw_decisions_path.exists() or not settlements_path.exists():
        return {
            "available": False,
            "reasons": ["sep23_raw_decisions_or_settlements_missing"],
        }
    raw_rows = _load_jsonl(raw_decisions_path)
    raw_by_key = {
        (row["label"], int(row["decision_time_ms"]), row["side"], row["feature_vector_sha256"]): row
        for row in raw_rows
    }
    complete_rows = []
    for settlement in _load_jsonl(settlements_path):
        key = (
            settlement["label"],
            int(settlement["decision_time_ms"]),
            settlement["side"],
            settlement["feature_vector_sha256"],
        )
        raw = raw_by_key.get(key)
        if raw is None:
            continue
        values = raw["directional_gate_values"]
        if float(values["quote_volume_30s"]) < quote_volume_threshold:
            continue
        complete_rows.append(
            {
                **settlement,
                "quote_volume_30s": float(values["quote_volume_30s"]),
            }
        )
    density_rows = density_filter_records(
        complete_rows,
        lookback_seconds=lookback_seconds,
        limit=limit,
    )
    one_position_rows = _apply_one_position_constraint(density_rows)
    return {
        "available": True,
        "raw_complete_top_p_count": len(_load_jsonl(settlements_path)),
        "quote_volume_filtered_count": len(complete_rows),
        "density_filtered_count": len(density_rows),
        "independent_summary": _summary_with_labels(density_rows),
        "one_position_summary": _summary_with_labels(one_position_rows),
    }


def fast73_status(
    *,
    historical_aggregate: dict[str, Any],
    extension_aggregate: dict[str, Any],
    sep23: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if extension_aggregate["negative_sufficient_blocks"]:
        return "FAST73_ANTI_STOP_DENSITY_FAILED_EXTENSION", reasons + [
            f"extension_negative_blocks:{len(extension_aggregate['negative_sufficient_blocks'])}"
        ]
    if sep23.get("available"):
        one_position = sep23["one_position_summary"]
        if float(one_position["stress_total_log_return_bps"]) <= 0:
            return "FAST73_ANTI_STOP_DENSITY_FAILED_SEP23", reasons + [
                "sep23_one_position_stress_non_positive"
            ]
    if historical_aggregate["negative_sufficient_blocks"]:
        return "FAST73_ANTI_STOP_DENSITY_FAILED_HISTORICAL", reasons + [
            f"historical_negative_blocks:{len(historical_aggregate['negative_sufficient_blocks'])}"
        ]
    return "FAST73_ANTI_STOP_DENSITY_RESEARCH_CANDIDATE_REQUIRES_SHADOW", reasons + [
        "requires_clean_future_shadow"
    ]


def _validate_parameters(
    *,
    historical_fast_start: int,
    historical_fast_end: int,
    quote_volume_quantile: float,
    density_lookback_seconds: int,
    density_limit: int,
) -> None:
    if historical_fast_start > historical_fast_end:
        raise ValueError("historical_fast_start cannot exceed historical_fast_end")
    if not 0 <= quote_volume_quantile <= 1:
        raise ValueError("quote_volume_quantile must be between 0 and 1")
    if density_lookback_seconds <= 0:
        raise ValueError("density_lookback_seconds must be positive")
    if density_limit < 0:
        raise ValueError("density_limit cannot be negative")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(output_root: Path, report: dict[str, Any]) -> dict[str, Any]:
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
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "quote_volume_threshold": report["quote_volume_threshold"],
        "density_limit": report["density_limit"],
        "target_alignment": report["target_alignment"],
        "historical_aggregate": report["historical_aggregate"],
        "extension_aggregate": report["extension_aggregate"],
        "sep23": report["sep23"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
