"""Fast65 causal feature-gate walk-forward for frozen top-p trades.

Fast64 showed that session/side/probability filters do not rescue Fast43-Fast53.
Fast65 tests a stronger but still interpretable family: one causal research
feature threshold, selected only on training blocks, then frozen on the held-out
block. No post-entry fields are allowed in gate selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast65-causal-feature-gate-walkforward"
FAST_NUMBER_PATTERN = re.compile(r"fast(?P<number>\d+)")
GATE_QUANTILES = (0.20, 0.33, 0.50, 0.67, 0.80)
FORBIDDEN_FEATURE_COLUMNS = {
    "decision_time_ms",
    "entry_sequence_id",
    "entry_time_ms",
    "exit_time_ms",
    "entry_price",
    "entry_price_feature",
    "exit_price",
    "side",
    "horizon_seconds",
    "event_count",
    "mfe_bps",
    "mae_bps",
    "terminal_bps",
    "first_touch",
    "first_touch_time_ms",
    "first_touch_sequence_id",
    "expected_touch_time_ms",
    "expected_touch_sequence_id",
    "stress_touch_time_ms",
    "stress_touch_sequence_id",
    "stop_touch_time_ms",
    "stop_touch_sequence_id",
    "paid_expected_before_stop",
    "paid_stress_before_stop",
    "target",
    "probability",
    "gross_bps",
    "expected_net_bps",
    "stress_net_bps",
    "maximum_favorable_bps",
    "maximum_adverse_bps",
    "partial_take_fraction",
    "exit_reason",
    "block",
    "utc_date",
}


@dataclass(frozen=True)
class FeatureGate:
    name: str
    feature: str | None = None
    direction: str = "none"
    threshold: float | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--historical-fast-start", type=int, default=43)
    parser.add_argument("--historical-fast-end", type=int, default=53)
    parser.add_argument("--minimum-training-total-trades", type=int, default=30)
    parser.add_argument("--minimum-training-block-trades", type=int, default=3)
    parser.add_argument("--minimum-holdout-trades", type=int, default=3)
    arguments = parser.parse_args()

    report = run_fast65_causal_feature_gate_walkforward(
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        historical_fast_start=arguments.historical_fast_start,
        historical_fast_end=arguments.historical_fast_end,
        minimum_training_total_trades=arguments.minimum_training_total_trades,
        minimum_training_block_trades=arguments.minimum_training_block_trades,
        minimum_holdout_trades=arguments.minimum_holdout_trades,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast65_causal_feature_gate_walkforward(
    *,
    data_root: Path,
    output_root: Path,
    historical_fast_start: int = 43,
    historical_fast_end: int = 53,
    minimum_training_total_trades: int = 30,
    minimum_training_block_trades: int = 3,
    minimum_holdout_trades: int = 3,
) -> dict[str, Any]:
    _validate_parameters(
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        minimum_training_total_trades=minimum_training_total_trades,
        minimum_training_block_trades=minimum_training_block_trades,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_paths = discover_historical_reports(
        data_root,
        fast_start=historical_fast_start,
        fast_end=historical_fast_end,
    )
    if not report_paths:
        return _write_report(
            output_root,
            _base_report(
                data_root=data_root,
                historical_fast_start=historical_fast_start,
                historical_fast_end=historical_fast_end,
                minimum_training_total_trades=minimum_training_total_trades,
                minimum_training_block_trades=minimum_training_block_trades,
                minimum_holdout_trades=minimum_holdout_trades,
            )
            | {
                "status": "FAST65_MISSING_HISTORICAL_REPORTS",
                "status_reasons": ["historical_fast43_fast53_reports_not_found"],
            },
        )

    selected_trades = load_selected_trades_with_causal_features(report_paths, data_root=data_root)
    if selected_trades.empty:
        return _write_report(
            output_root,
            _base_report(
                data_root=data_root,
                historical_fast_start=historical_fast_start,
                historical_fast_end=historical_fast_end,
                minimum_training_total_trades=minimum_training_total_trades,
                minimum_training_block_trades=minimum_training_block_trades,
                minimum_holdout_trades=minimum_holdout_trades,
            )
            | {
                "status": "FAST65_NO_MERGED_SELECTED_TRADES",
                "status_reasons": ["selected_trades_could_not_merge_to_research_rows"],
            },
        )

    feature_columns = causal_feature_columns(selected_trades)
    merged_path = output_root / "selected_trades_with_causal_features.parquet"
    selected_trades.to_parquet(merged_path, index=False, compression="zstd")
    holdouts = []
    for holdout_block in sorted(selected_trades["block"].unique()):
        training = selected_trades[selected_trades["block"] != holdout_block].copy()
        holdout = selected_trades[selected_trades["block"] == holdout_block].copy()
        gate, training_evaluations = select_training_gate(
            training,
            feature_columns=feature_columns,
            minimum_training_total_trades=minimum_training_total_trades,
            minimum_training_block_trades=minimum_training_block_trades,
        )
        holdout_filtered = apply_gate(holdout, gate)
        holdout_summary = one_position_summary(holdout_filtered)
        holdouts.append(
            {
                "holdout_block": str(holdout_block),
                "selected_gate": asdict(gate),
                "training_candidate_count": len(training_evaluations),
                "training_best": training_evaluations[0] if training_evaluations else None,
                "holdout_selected_signal_count": int(len(holdout_filtered)),
                "holdout_summary": holdout_summary,
                "passed_stress": (
                    int(holdout_summary["trade_count"]) >= minimum_holdout_trades
                    and float(holdout_summary["stress_total_log_return_bps"]) > 0
                    and float(holdout_summary["stress_mean_bps"]) > 0
                ),
            }
        )

    aggregate = aggregate_holdouts(holdouts, minimum_holdout_trades=minimum_holdout_trades)
    status, status_reasons = fast65_status(
        aggregate=aggregate,
        minimum_holdout_trades=minimum_holdout_trades,
    )
    holdouts_path = output_root / "holdouts.json"
    holdouts_path.write_text(json.dumps(holdouts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _base_report(
        data_root=data_root,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        minimum_training_total_trades=minimum_training_total_trades,
        minimum_training_block_trades=minimum_training_block_trades,
        minimum_holdout_trades=minimum_holdout_trades,
    ) | {
        "status": status,
        "status_reasons": status_reasons,
        "historical_report_count": len(report_paths),
        "historical_reports": [str(path) for path in report_paths],
        "trade_count": int(len(selected_trades)),
        "block_count": int(selected_trades["block"].nunique()),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "merged_selected_trades_path": str(merged_path),
        "merged_selected_trades_sha256": _sha256_file(merged_path),
        "holdouts_path": str(holdouts_path),
        "holdouts_sha256": _sha256_file(holdouts_path),
        "aggregate": aggregate,
        "holdouts": holdouts,
    }
    return _write_report(output_root, report)


def discover_historical_reports(
    data_root: Path,
    *,
    fast_start: int,
    fast_end: int,
) -> tuple[Path, ...]:
    reports = []
    for path in sorted(
        (data_root / "reports").glob(
            "latency-adjusted-frozen-top-p-fast*-lowcost-micro-holdout-*-100ms/report.json"
        )
    ):
        match = FAST_NUMBER_PATTERN.search(str(path))
        if match is None:
            continue
        fast_number = int(match.group("number"))
        if fast_start <= fast_number <= fast_end:
            reports.append(path)
    return tuple(reports)


def load_selected_trades_with_causal_features(
    report_paths: tuple[Path, ...],
    *,
    data_root: Path,
) -> pd.DataFrame:
    research_roots = research_roots_by_dates(data_root)
    frames = []
    for report_path in report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        holdout_dates = tuple(str(item) for item in report["holdout_dates"])
        research_root = research_roots.get(holdout_dates)
        if research_root is None:
            continue
        trades = pd.read_parquet(report_path.parent / "frozen_holdout_trades.parquet")
        if trades.empty:
            continue
        research_rows = pd.concat(
            [
                pd.read_parquet(research_root / f"date={utc_date}" / "research_rows.parquet")
                for utc_date in holdout_dates
            ],
            ignore_index=True,
        )
        research_rows = research_rows[research_rows["horizon_seconds"].eq(300)].copy()
        merged = trades.merge(
            research_rows,
            on=["decision_time_ms", "side", "horizon_seconds"],
            how="left",
            suffixes=("", "_feature"),
        )
        merged = merged[merged["target"].notna()].copy()
        merged["block"] = report_path.parent.name
        merged["utc_date"] = pd.to_datetime(
            merged["decision_time_ms"],
            unit="ms",
            utc=True,
        ).dt.date.astype(str)
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["block", "decision_time_ms"],
        kind="stable",
    )


def research_roots_by_dates(data_root: Path) -> dict[tuple[str, ...], Path]:
    mapping = {}
    for root in sorted(data_root.glob("research-lowcost-fast*")):
        dates = tuple(path.name.removeprefix("date=") for path in sorted(root.glob("date=*")))
        if dates:
            mapping[dates] = root
    return mapping


def causal_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if column in FORBIDDEN_FEATURE_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < 10 or values.nunique(dropna=True) < 5:
            continue
        columns.append(column)
    return columns


def select_training_gate(
    training: pd.DataFrame,
    *,
    feature_columns: list[str],
    minimum_training_total_trades: int,
    minimum_training_block_trades: int,
) -> tuple[FeatureGate, list[dict[str, Any]]]:
    evaluations = [
        evaluate_gate(
            training,
            gate,
            minimum_training_block_trades=minimum_training_block_trades,
        )
        for gate in gate_candidates(training, feature_columns=feature_columns)
    ]
    evaluations = [
        item
        for item in evaluations
        if int(item["total_trade_count"]) >= minimum_training_total_trades
    ]
    evaluations.sort(
        key=lambda item: (
            int(item["positive_block_count"]),
            int(item["minimum_trade_block_count"]),
            float(item["min_block_stress_total_log_return_bps"]),
            float(item["stress_total_log_return_bps"]),
            float(item["stress_mean_bps"]),
        ),
        reverse=True,
    )
    if not evaluations:
        return FeatureGate(name="all"), []
    return gate_from_dict(evaluations[0]["gate"]), evaluations


def gate_candidates(training: pd.DataFrame, *, feature_columns: list[str]) -> list[FeatureGate]:
    gates = [FeatureGate(name="all")]
    for feature in feature_columns:
        values = pd.to_numeric(training[feature], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        values = values.dropna()
        if values.nunique() < 5:
            continue
        for quantile in GATE_QUANTILES:
            threshold = float(values.quantile(quantile))
            gates.append(
                FeatureGate(
                    name=f"{feature}<=q{quantile:.2f}:{threshold:.10g}",
                    feature=feature,
                    direction="low",
                    threshold=threshold,
                )
            )
            gates.append(
                FeatureGate(
                    name=f"{feature}>=q{quantile:.2f}:{threshold:.10g}",
                    feature=feature,
                    direction="high",
                    threshold=threshold,
                )
            )
    return gates


def evaluate_gate(
    frame: pd.DataFrame,
    gate: FeatureGate,
    *,
    minimum_training_block_trades: int,
) -> dict[str, Any]:
    block_summaries = []
    filtered = apply_gate(frame, gate)
    filtered_by_block = {
        str(block): block_frame
        for block, block_frame in filtered.groupby("block", sort=True)
    }
    for block in sorted(str(value) for value in frame["block"].unique()):
        summary = one_position_summary(filtered_by_block.get(block, pd.DataFrame()))
        block_summaries.append({"block": str(block), **summary})
    total_summary = one_position_summary(filtered)
    return {
        "gate": asdict(gate),
        "block_count": len(block_summaries),
        "positive_block_count": sum(
            float(row["stress_total_log_return_bps"]) > 0
            and float(row["stress_mean_bps"]) > 0
            for row in block_summaries
        ),
        "minimum_trade_block_count": sum(
            int(row["trade_count"]) >= minimum_training_block_trades for row in block_summaries
        ),
        "min_block_stress_total_log_return_bps": min(
            (float(row["stress_total_log_return_bps"]) for row in block_summaries),
            default=0.0,
        ),
        "total_trade_count": int(total_summary["trade_count"]),
        "stress_total_log_return_bps": float(total_summary["stress_total_log_return_bps"]),
        "stress_mean_bps": float(total_summary["stress_mean_bps"]),
        "expected_win_rate": float(total_summary["expected_win_rate"]),
    }


def apply_gate(frame: pd.DataFrame, gate: FeatureGate) -> pd.DataFrame:
    if frame.empty or gate.feature is None:
        return frame.copy()
    values = pd.to_numeric(frame[gate.feature], errors="coerce")
    if gate.direction == "low":
        return frame[values <= float(gate.threshold)].copy()
    if gate.direction == "high":
        return frame[values >= float(gate.threshold)].copy()
    raise ValueError(f"unknown gate direction: {gate.direction}")


def one_position_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return _empty_summary()
    position_exit_ms = -1
    required = [
        "decision_time_ms",
        "probability",
        "exit_time_ms",
        "expected_net_bps",
        "stress_net_bps",
        "entry_time_ms",
    ]
    ordered = frame[required].sort_values(
        ["decision_time_ms", "probability"],
        ascending=[True, False],
        kind="stable",
    )
    expected_values = []
    stress_values = []
    entry_times = []
    for row in ordered.itertuples():
        decision_time_ms = int(row.decision_time_ms)
        if decision_time_ms <= position_exit_ms:
            continue
        position_exit_ms = int(row.exit_time_ms)
        expected_values.append(float(row.expected_net_bps))
        stress_values.append(float(row.stress_net_bps))
        entry_times.append(int(row.entry_time_ms))
    if not expected_values:
        return _empty_summary()
    expected = np.asarray(expected_values, dtype=np.float64)
    stress = np.asarray(stress_values, dtype=np.float64)
    cumulative = np.cumsum(expected)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdowns = np.concatenate(([0.0], cumulative)) - peaks
    days = pd.to_datetime(entry_times, unit="ms", utc=True).date
    return {
        "trade_count": int(len(expected_values)),
        "distinct_days": int(len(set(days))),
        "expected_mean_bps": float(expected.mean()),
        "stress_mean_bps": float(stress.mean()),
        "expected_total_log_return_bps": float(expected.sum()),
        "stress_total_log_return_bps": float(stress.sum()),
        "expected_win_rate": float((expected > 0).mean()),
        "maximum_drawdown_bps": float(-drawdowns.min()),
    }


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
            int(row["trade_count"]) >= minimum_holdout_trades for row in summaries
        ),
        "stress_total_log_return_bps": float(
            sum(float(row["stress_total_log_return_bps"]) for row in summaries)
        ),
        "expected_total_log_return_bps": float(
            sum(float(row["expected_total_log_return_bps"]) for row in summaries)
        ),
        "trade_count": int(sum(int(row["trade_count"]) for row in summaries)),
        "min_holdout_stress_total_log_return_bps": min(
            (float(row["stress_total_log_return_bps"]) for row in summaries),
            default=0.0,
        ),
        "selected_gates": [holdout["selected_gate"]["name"] for holdout in holdouts],
    }


def fast65_status(
    *,
    aggregate: dict[str, Any],
    minimum_holdout_trades: int,
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "diagnostic_only", "no_order_execution"]
    holdout_count = int(aggregate["holdout_count"])
    if holdout_count <= 0:
        return "FAST65_NO_HOLDOUTS", reasons + ["no_holdout_blocks_evaluated"]
    if int(aggregate["passed_stress_count"]) < holdout_count:
        return "FAST65_CAUSAL_FEATURE_GATE_FAILED_WALKFORWARD", reasons + [
            f"passed_stress_holdouts:{aggregate['passed_stress_count']}<{holdout_count}",
            f"minimum_holdout_trades:{minimum_holdout_trades}",
        ]
    if float(aggregate["stress_total_log_return_bps"]) <= 0:
        return "FAST65_CAUSAL_FEATURE_GATE_FAILED_WALKFORWARD", reasons + [
            "aggregate_stress_total_non_positive"
        ]
    return "FAST65_CAUSAL_FEATURE_GATE_CANDIDATE_RESEARCH_ONLY", reasons + [
        "requires_preregistration_and_prospective_shadow_before_testnet"
    ]


def gate_from_dict(payload: dict[str, Any]) -> FeatureGate:
    return FeatureGate(
        name=str(payload["name"]),
        feature=payload.get("feature"),
        direction=str(payload.get("direction", "none")),
        threshold=payload.get("threshold"),
    )


def _empty_summary() -> dict[str, float | int]:
    return {
        "trade_count": 0,
        "distinct_days": 0,
        "expected_mean_bps": 0.0,
        "stress_mean_bps": 0.0,
        "expected_total_log_return_bps": 0.0,
        "stress_total_log_return_bps": 0.0,
        "expected_win_rate": 0.0,
        "maximum_drawdown_bps": 0.0,
    }


def _base_report(
    *,
    data_root: Path,
    historical_fast_start: int,
    historical_fast_end: int,
    minimum_training_total_trades: int,
    minimum_training_block_trades: int,
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
            "fast65_is_diagnostic_walkforward_only",
            "single_causal_feature_gate_selected_on_training_blocks_only",
            "post_entry_outcome_fields_excluded_from_gate_features",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast65_causal_feature_gate_walkforward",
        "data_root": str(data_root),
        "historical_fast_start": historical_fast_start,
        "historical_fast_end": historical_fast_end,
        "minimum_training_total_trades": minimum_training_total_trades,
        "minimum_training_block_trades": minimum_training_block_trades,
        "minimum_holdout_trades": minimum_holdout_trades,
        "gate_quantiles": list(GATE_QUANTILES),
    }


def _validate_parameters(
    *,
    historical_fast_start: int,
    historical_fast_end: int,
    minimum_training_total_trades: int,
    minimum_training_block_trades: int,
    minimum_holdout_trades: int,
) -> None:
    if historical_fast_start > historical_fast_end:
        raise ValueError("historical_fast_start cannot exceed historical_fast_end")
    if minimum_training_total_trades < 1:
        raise ValueError("minimum_training_total_trades must be positive")
    if minimum_training_block_trades < 1:
        raise ValueError("minimum_training_block_trades must be positive")
    if minimum_holdout_trades < 1:
        raise ValueError("minimum_holdout_trades must be positive")


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
        "block_count": report.get("block_count"),
        "trade_count": report.get("trade_count"),
        "feature_count": report.get("feature_count"),
        "aggregate": report.get("aggregate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
