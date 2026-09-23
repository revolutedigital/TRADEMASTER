"""Fast84 diagnostic for silent frozen top-p windows.

Fast84 scores all BUY/SELL polling rows, including non-entries, to explain why
a post-preregistration WAL produced no Fast60 top-p entries. It is diagnostic
only: it never changes the frozen threshold, never writes ledger state, and
never approves a strategy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.online_shadow_policy import (  # noqa: E402
    score_online_frozen_top_p_shadow_events,
)
from scripts.research.run_fast57_prospective_update import DEFAULT_ARTIFACT_REPORT  # noqa: E402
from scripts.research.run_fast60_fast57_shadow_dry_run import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    WalRootRef,
    align_datetime_to_stride,
    discover_default_wal_roots,
    infer_wal_trade_interval,
    parse_dates,
    parse_wal_root_refs,
    shadow_decision_to_row,
    write_jsonl,
)
from scripts.research.run_fast80_rare_vol_flow_observation import (  # noqa: E402
    FLOW_IMBALANCE_5S_THRESHOLD,
    MODEL_SHA256,
    REALIZED_VOL_30S_THRESHOLD,
)
from scripts.research.run_online_shadow_policy import (  # noqa: E402
    EventRoots,
    decision_times_between,
    load_online_microstructure_events,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast84-top-p-silence-diagnostic"
NEAR_THRESHOLD_MARGINS = (0.01, 0.02, 0.05, 0.10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-report", type=Path, default=DEFAULT_ARTIFACT_REPORT)
    parser.add_argument(
        "--wal-roots",
        nargs="+",
        help="Optional LABEL=path WAL roots. Defaults to prospective-wal-fast* roots.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dates", help="Optional comma-separated UTC dates to score.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--warmup-seconds", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    report = run_fast84_top_p_silence_diagnostic(
        artifact_report=arguments.artifact_report,
        wal_roots=(
            parse_wal_root_refs(arguments.wal_roots)
            if arguments.wal_roots
            else discover_default_wal_roots(arguments.data_root)
        ),
        dates=parse_dates(arguments.dates) if arguments.dates else None,
        symbol=arguments.symbol.upper(),
        decision_stride_seconds=arguments.decision_stride_seconds,
        warmup_seconds=arguments.warmup_seconds,
        top_k=arguments.top_k,
        output_root=arguments.output_root,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast84_top_p_silence_diagnostic(
    *,
    artifact_report: Path = DEFAULT_ARTIFACT_REPORT,
    wal_roots: tuple[WalRootRef, ...],
    dates: tuple[date, ...] | None = None,
    symbol: str = "BTCUSDT",
    decision_stride_seconds: int = 5,
    warmup_seconds: int = 60,
    top_k: int = 50,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    if decision_stride_seconds <= 0:
        raise ValueError("decision_stride_seconds must be positive")
    if warmup_seconds < 0:
        raise ValueError("warmup_seconds cannot be negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    artifact_payload = json.loads(artifact_report.read_text(encoding="utf-8"))
    artifact = artifact_payload["frozen_artifact"]
    threshold = float(artifact["probability_threshold"])
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for wal_root in wal_roots:
        interval = infer_wal_trade_interval(wal_root.path, dates=dates)
        if interval is None:
            runs.append(
                {
                    "label": wal_root.label,
                    "wal_root": str(wal_root.path),
                    "status": "FAST84_WAL_HAS_NO_TRADE_EVENTS",
                    "status_reasons": ["trade_wal_missing_or_empty"],
                    "scored_rows": 0,
                }
            )
            continue
        start = align_datetime_to_stride(
            interval.start + timedelta(seconds=warmup_seconds),
            stride_seconds=decision_stride_seconds,
        )
        end = interval.end
        if end <= start:
            runs.append(
                {
                    "label": wal_root.label,
                    "wal_root": str(wal_root.path),
                    "status": "FAST84_WAL_INTERVAL_TOO_SHORT",
                    "status_reasons": ["warmup_consumes_interval"],
                    "scored_rows": 0,
                }
            )
            continue
        loaded = load_online_microstructure_events(
            artifact=artifact,
            symbol=symbol,
            start=start,
            end=end,
            history_seconds=warmup_seconds,
            roots=EventRoots(
                trade=wal_root.path / "trade",
                book=wal_root.path / "depth",
                mark=wal_root.path / "mark_price",
                liquidation=wal_root.path / "liquidation",
                spot=wal_root.path / "spot_trade",
            ),
        )
        decision_times_ms = decision_times_between(
            start,
            end,
            stride_seconds=decision_stride_seconds,
        )
        selection = score_online_frozen_top_p_shadow_events(
            loaded.events,
            artifact=artifact,
            decision_times_ms=decision_times_ms,
            sides=("BUY", "SELL"),
            include_non_entries=True,
        )
        rows = [
            enrich_diagnostic_row(shadow_decision_to_row(decision, label=wal_root.label))
            for decision in selection.decisions
        ]
        all_rows.extend(rows)
        run_summary = diagnostic_summary(
            rows,
            threshold=threshold,
            near_threshold_margins=NEAR_THRESHOLD_MARGINS,
        )
        run_status, run_reasons = fast84_status(run_summary)
        runs.append(
            {
                "label": wal_root.label,
                "wal_root": str(wal_root.path),
                "status": run_status,
                "status_reasons": run_reasons,
                "dates": list(interval.dates),
                "interval": {
                    "first_trade": interval.start.isoformat(),
                    "last_trade": interval.end.isoformat(),
                    "decision_start": start.isoformat(),
                    "decision_end": end.isoformat(),
                },
                "trade_event_count": interval.trade_event_count,
                "event_counts": loaded.counts,
                "decision_count": len(decision_times_ms),
                "scored_rows": selection.scored_rows,
                "summary": run_summary,
            }
        )

    scored_path = output_root / "all_scored_decisions.jsonl"
    top_path = output_root / "top_scored_decisions.jsonl"
    top_rows = sorted(all_rows, key=lambda row: float(row["probability"]), reverse=True)[:top_k]
    write_jsonl(scored_path, all_rows)
    write_jsonl(top_path, top_rows)
    aggregate = diagnostic_summary(
        all_rows,
        threshold=threshold,
        near_threshold_margins=NEAR_THRESHOLD_MARGINS,
    )
    status, status_reasons = fast84_status(aggregate)
    report_path = output_root / "report.json"
    report: dict[str, Any] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast84_is_probability_diagnostic_only",
            "no_threshold_change",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast84_top_p_silence_diagnostic",
        "status": status,
        "status_reasons": status_reasons,
        "candidate": {
            "model_sha256": artifact_payload["model_sha256"],
            "threshold": threshold,
            "tail_fraction": float(artifact["tail_fraction"]),
            "horizon_seconds": int(artifact["horizon_seconds"]),
        },
        "wal_root_count": len(wal_roots),
        "run_count": len(runs),
        "runs": runs,
        "aggregate": aggregate,
        "all_scored_decisions_path": str(scored_path),
        "all_scored_decisions_sha256": _sha256_file(scored_path),
        "top_scored_decisions_path": str(top_path),
        "top_scored_decisions_sha256": _sha256_file(top_path),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def enrich_diagnostic_row(row: dict[str, Any]) -> dict[str, Any]:
    realized_vol = row.get("realized_vol_30s_bps")
    flow_imbalance = row.get("flow_imbalance_5s")
    has_fast80_features = realized_vol is not None and flow_imbalance is not None
    fast80_rule_match = (
        has_fast80_features
        and float(realized_vol) >= REALIZED_VOL_30S_THRESHOLD
        and float(flow_imbalance) <= FLOW_IMBALANCE_5S_THRESHOLD
        and row.get("model_sha256") == MODEL_SHA256
    )
    return {
        **row,
        "fast80_required_features_present": bool(has_fast80_features),
        "fast80_rule_match_diagnostic": bool(fast80_rule_match),
    }


def diagnostic_summary(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    near_threshold_margins: tuple[float, ...],
) -> dict[str, Any]:
    probabilities = [float(row["probability"]) for row in rows]
    selected = [row for row in rows if bool(row.get("would_enter"))]
    fast80_rows = [row for row in rows if bool(row.get("fast80_rule_match_diagnostic"))]
    summary: dict[str, Any] = {
        "scored_rows": len(rows),
        "threshold": threshold,
        "would_enter_count": len(selected),
        "would_enter_rate": safe_ratio(len(selected), len(rows)),
        "probability": probability_distribution(probabilities),
        "near_threshold_counts": {
            f"within_{margin:g}": sum(
                probability >= threshold - margin
                for probability in probabilities
            )
            for margin in near_threshold_margins
        },
        "side_summary": side_summary(rows, threshold=threshold),
        "fast80_rule_diagnostic": {
            "feature_scoreable_rows": sum(
                bool(row.get("fast80_required_features_present")) for row in rows
            ),
            "rule_match_rows": len(fast80_rows),
            "rule_match_would_enter_rows": sum(
                bool(row.get("would_enter")) for row in fast80_rows
            ),
            "max_probability_among_rule_matches": (
                max(float(row["probability"]) for row in fast80_rows)
                if fast80_rows
                else None
            ),
        },
    }
    max_probability = summary["probability"]["max"]
    summary["distance_from_threshold_at_max_probability"] = (
        threshold - max_probability if max_probability is not None else None
    )
    return summary


def probability_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "p25": nearest_rank_quantile(ordered, 0.25),
        "p50": nearest_rank_quantile(ordered, 0.50),
        "p75": nearest_rank_quantile(ordered, 0.75),
        "p90": nearest_rank_quantile(ordered, 0.90),
        "p95": nearest_rank_quantile(ordered, 0.95),
        "p99": nearest_rank_quantile(ordered, 0.99),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def nearest_rank_quantile(ordered_values: list[float], quantile: float) -> float:
    if not ordered_values:
        raise ValueError("ordered_values cannot be empty")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    index = max(0, math.ceil(quantile * len(ordered_values)) - 1)
    return float(ordered_values[index])


def side_summary(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    sides: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sides.setdefault(str(row["side"]), []).append(row)
    return {
        side: {
            "scored_rows": len(side_rows),
            "would_enter_count": sum(bool(row.get("would_enter")) for row in side_rows),
            "probability": probability_distribution(
                [float(row["probability"]) for row in side_rows]
            ),
            "within_0.05": sum(float(row["probability"]) >= threshold - 0.05 for row in side_rows),
        }
        for side, side_rows in sorted(sides.items())
    }


def fast84_status(summary: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if int(summary["scored_rows"]) <= 0:
        return "FAST84_NO_SCORED_ROWS", reasons + ["no_decisions_scored"]
    if int(summary["would_enter_count"]) > 0:
        return "FAST84_TOP_P_ENTRIES_PRESENT_RESEARCH_ONLY", reasons + [
            f"would_enter_count:{summary['would_enter_count']}"
        ]
    distance = summary["distance_from_threshold_at_max_probability"]
    if distance is not None and float(distance) <= 0.05:
        return "FAST84_TOP_P_NEAR_THRESHOLD_RESEARCH_ONLY", reasons + [
            f"threshold_minus_max_probability:{float(distance):.8f}"
        ]
    return "FAST84_TOP_P_SILENT_BELOW_THRESHOLD", reasons + [
        f"threshold_minus_max_probability:{float(distance):.8f}"
        if distance is not None
        else "no_probability_rows"
    ]


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "candidate": report["candidate"],
        "aggregate": report["aggregate"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
