"""Causal rule/intersection scout over prospective book WAL captures.

This script searches simple, auditable signal rules such as:

- directed microprice displacement is in the top quantile;
- directed flow imbalance is low;
- both a book signal and a spot/perp signal fire together.

Each rule's thresholds are fit only on prior temporal blocks and applied to the
next block. The output is a research diagnostic, not a strategy approval.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.research_dataset import (  # noqa: E402
    ResearchDatasetConfig,
    build_research_rows,
    load_book_interval,
    load_mark_interval,
    load_trade_interval,
)
from scripts.research.run_book_feature_model_scout import (  # noqa: E402
    LABEL_AND_FUTURE_COLUMNS,
    select_causal_feature_columns,
)
from scripts.research.run_book_wal_scout import (  # noqa: E402
    decision_times_for_wal,
    summarize_stream,
)

REGIME_FEATURE_PREFIXES = (
    "trade_count_",
    "quote_volume_",
    "realized_vol_",
    "mean_interarrival_ms_",
    "book_event_count_",
    "book_spread_",
    "spot_trade_count_",
    "spot_quote_volume_",
    "spot_realized_vol_",
    "spot_mean_interarrival_ms_",
    "spot_perp_quote_volume_ratio_",
)
REGIME_FEATURES = {
    "book_update_age_ms",
    "spread_bps",
    "mark_update_age_ms",
    "funding_rate",
    "spot_update_age_ms",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wal-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "prospective-wal-fast16-18m",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-rule-scout",
    )
    parser.add_argument("--horizons-seconds", default="30,60,120")
    parser.add_argument("--thresholds-bps", default="0.5,1,2,3,5")
    parser.add_argument("--condition-fractions", default="0.10,0.20,0.33")
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--max-book-staleness-ms", type=int, default=1_000)
    parser.add_argument("--initial-stop-bps", type=float, default=20.0)
    parser.add_argument("--temporal-blocks", type=int, default=4)
    parser.add_argument("--minimum-train-blocks", type=int, default=2)
    parser.add_argument("--minimum-selected-count", type=int, default=10)
    parser.add_argument("--max-pair-features", type=int, default=14)
    parser.add_argument("--max-report-candidates", type=int, default=60)
    arguments = parser.parse_args()
    report = run_book_rule_scout(
        wal_root=arguments.wal_root,
        output_root=arguments.output_root,
        horizons_seconds=_parse_int_list(arguments.horizons_seconds),
        thresholds_bps=_parse_float_list(arguments.thresholds_bps),
        condition_fractions=_parse_float_list(arguments.condition_fractions),
        decision_stride_seconds=arguments.decision_stride_seconds,
        max_book_staleness_ms=arguments.max_book_staleness_ms,
        initial_stop_bps=arguments.initial_stop_bps,
        temporal_blocks=arguments.temporal_blocks,
        minimum_train_blocks=arguments.minimum_train_blocks,
        minimum_selected_count=arguments.minimum_selected_count,
        max_pair_features=arguments.max_pair_features,
        max_report_candidates=arguments.max_report_candidates,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_rule_scout(
    *,
    wal_root: Path,
    output_root: Path,
    horizons_seconds: tuple[int, ...] = (30, 60, 120),
    thresholds_bps: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0),
    condition_fractions: tuple[float, ...] = (0.10, 0.20, 0.33),
    decision_stride_seconds: int = 5,
    max_book_staleness_ms: int = 1_000,
    initial_stop_bps: float = 20.0,
    temporal_blocks: int = 4,
    minimum_train_blocks: int = 2,
    minimum_selected_count: int = 10,
    max_pair_features: int = 24,
    max_report_candidates: int = 60,
) -> dict[str, object]:
    _validate_parameters(
        thresholds_bps=thresholds_bps,
        condition_fractions=condition_fractions,
        initial_stop_bps=initial_stop_bps,
        temporal_blocks=temporal_blocks,
        minimum_train_blocks=minimum_train_blocks,
        minimum_selected_count=minimum_selected_count,
        max_pair_features=max_pair_features,
        max_report_candidates=max_report_candidates,
    )
    base_config = ResearchDatasetConfig(
        decision_stride_seconds=decision_stride_seconds,
        horizons_seconds=horizons_seconds,
        expected_cost_bps=thresholds_bps[0],
        stress_cost_bps=thresholds_bps[0],
        initial_stop_bps=initial_stop_bps,
        feature_windows_seconds=(1, 5, 30),
        require_book_features=True,
        max_book_staleness_ms=max_book_staleness_ms,
    )
    streams = {
        "trade": summarize_stream(wal_root / "trade"),
        "depth": summarize_stream(wal_root / "depth"),
        "mark_price": summarize_stream(wal_root / "mark_price"),
        "spot_trade": summarize_stream(wal_root / "spot_trade"),
    }
    base_report: dict[str, object] = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "experiment_stage": "book_rule_scout",
        "wal_root": str(wal_root),
        "config": {
            **asdict(base_config),
            "thresholds_bps": list(thresholds_bps),
            "condition_fractions": list(condition_fractions),
            "temporal_blocks": temporal_blocks,
            "minimum_train_blocks": minimum_train_blocks,
            "minimum_selected_count": minimum_selected_count,
            "max_pair_features": max_pair_features,
            "max_report_candidates": max_report_candidates,
        },
        "leakage_guard": {
            "excluded_columns": sorted(LABEL_AND_FUTURE_COLUMNS),
            "thresholds_fit_on_train_only": True,
        },
        "streams": {name: asdict(summary) for name, summary in streams.items()},
        "approved_strategy": False,
        "approval_reasons": [
            "diagnostic_only",
            "short_single_capture",
            "no_statistical_gate",
            "no_transaction_cost_confirmation",
        ],
    }
    missing_required = [
        name for name in ("trade", "depth") if not streams[name].present
    ]
    if missing_required:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "MISSING_REQUIRED_STREAMS",
                "status_reasons": [f"{name}_missing" for name in missing_required],
                "modeling_ready": False,
                "top_candidates": [],
            },
        )

    decisions = decision_times_for_wal(
        trade=streams["trade"],
        book=streams["depth"],
        config=base_config,
    )
    if len(decisions) < temporal_blocks:
        return _write_report(
            output_root,
            {
                **base_report,
                "status": "INSUFFICIENT_INTERVAL",
                "status_reasons": [
                    f"decision_count_below_temporal_blocks:{len(decisions)}<{temporal_blocks}"
                ],
                "decision_count": int(len(decisions)),
                "modeling_ready": False,
                "top_candidates": [],
            },
        )

    interval_start = _dt(
        min(_not_none(streams["trade"].first_event_time_ms), int(decisions[0]))
    )
    interval_end = _dt(_not_none(streams["trade"].last_event_time_ms))
    trades = load_trade_interval(
        wal_root / "trade",
        interval_start,
        interval_end,
        expected_product="usdm_perpetual",
    )
    book_events = load_book_interval(wal_root / "depth", interval_start, interval_end)
    mark_events = (
        load_mark_interval(wal_root / "mark_price", interval_start, interval_end)
        if streams["mark_price"].present
        else None
    )
    spot_events = (
        load_trade_interval(
            wal_root / "spot_trade",
            interval_start,
            interval_end,
            expected_product="spot",
        )
        if streams["spot_trade"].present
        else None
    )

    all_candidates: list[dict[str, object]] = []
    candidates_evaluated = 0
    feature_count = 0
    for threshold_bps in thresholds_bps:
        config = ResearchDatasetConfig(
            decision_stride_seconds=decision_stride_seconds,
            horizons_seconds=horizons_seconds,
            expected_cost_bps=threshold_bps,
            stress_cost_bps=threshold_bps,
            initial_stop_bps=initial_stop_bps,
            feature_windows_seconds=(1, 5, 30),
            require_book_features=True,
            max_book_staleness_ms=max_book_staleness_ms,
        )
        rows = build_research_rows(
            trades,
            decisions,
            config,
            book_events=book_events,
            mark_events=mark_events,
            spot_trade_events=spot_events,
        )
        rows = _add_gross_path_payoff(rows, threshold_bps, initial_stop_bps)
        features = select_rule_feature_columns(rows)
        feature_count = max(feature_count, len(features))
        rule_specs = build_rule_specs(
            features,
            max_pair_features=max_pair_features,
            condition_fractions=condition_fractions,
        )
        candidates_evaluated += len(rule_specs) * len(horizons_seconds)
        for horizon_seconds in horizons_seconds:
            horizon_rows = rows[rows["horizon_seconds"] == horizon_seconds]
            all_candidates.extend(
                evaluate_rule_specs(
                    rows=horizon_rows,
                    threshold_bps=threshold_bps,
                    horizon_seconds=horizon_seconds,
                    rule_specs=rule_specs,
                    temporal_blocks=temporal_blocks,
                    minimum_train_blocks=minimum_train_blocks,
                    minimum_selected_count=minimum_selected_count,
                )
            )

    ranked = sorted(
        all_candidates,
        key=lambda item: (
            item["selected_gross_path_payoff_mean_bps"],
            item["selected_count"],
        ),
        reverse=True,
    )
    top_candidates = ranked[:max_report_candidates]
    best_candidate = top_candidates[0] if top_candidates else None
    return _write_report(
        output_root,
        {
            **base_report,
            "status": "READY_FOR_RULE_SCOUT_REVIEW"
            if top_candidates
            else "NO_RULE_CANDIDATES",
            "status_reasons": [] if top_candidates else ["no_rule_met_minimum_selected_count"],
            "modeling_ready": bool(top_candidates),
            "decision_count": int(len(decisions)),
            "feature_count": feature_count,
            "candidates_evaluated": candidates_evaluated,
            "top_candidates": top_candidates,
            "best_candidate": best_candidate,
            "interval_start": _iso(int(decisions.min())),
            "interval_end": _iso(int(decisions.max())),
        },
    )


def select_rule_feature_columns(rows: pd.DataFrame) -> list[str]:
    causal = select_causal_feature_columns(rows)
    selected: list[str] = []
    for column in causal:
        if not _is_rule_feature(column):
            continue
        values = rows[column].to_numpy(dtype=np.float64)
        if np.nanmax(values) == np.nanmin(values):
            continue
        selected.append(column)
    return selected


def build_rule_specs(
    features: list[str],
    *,
    max_pair_features: int,
    condition_fractions: tuple[float, ...],
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for fraction in condition_fractions:
        for feature in features:
            for direction in ("high", "low"):
                specs.append(
                    {
                        "conditions": [
                            {"feature": feature, "direction": direction},
                        ],
                        "condition_fraction": fraction,
                    }
                )
        pair_features = features[:max_pair_features]
        for first, second in itertools.combinations(pair_features, 2):
            for first_direction, second_direction in itertools.product(
                ("high", "low"),
                repeat=2,
            ):
                specs.append(
                    {
                        "conditions": [
                            {"feature": first, "direction": first_direction},
                            {"feature": second, "direction": second_direction},
                        ],
                        "condition_fraction": fraction,
                    }
                )
    return specs


def evaluate_rule_specs(
    *,
    rows: pd.DataFrame,
    threshold_bps: float,
    horizon_seconds: int,
    rule_specs: list[dict[str, object]],
    temporal_blocks: int,
    minimum_train_blocks: int,
    minimum_selected_count: int,
) -> list[dict[str, object]]:
    unique_decisions = np.array(sorted(rows["decision_time_ms"].unique()), dtype=np.int64)
    decision_blocks = np.array_split(unique_decisions, temporal_blocks)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for test_block_index in range(minimum_train_blocks, temporal_blocks):
        train_times = np.concatenate(decision_blocks[:test_block_index])
        test_times = decision_blocks[test_block_index]
        train = rows[rows["decision_time_ms"].isin(train_times)]
        test = rows[rows["decision_time_ms"].isin(test_times)]
        folds.append((train, test))
    candidates: list[dict[str, object]] = []
    for spec in rule_specs:
        aggregate = _empty_rule_aggregate(
            threshold_bps=threshold_bps,
            horizon_seconds=horizon_seconds,
            spec=spec,
        )
        for train, test in folds:
            selected = apply_train_fit_rule(train, test, spec)
            _accumulate_rule(aggregate, test, selected)
        if aggregate["selected_count"] >= minimum_selected_count:
            candidates.append(_finalize_rule(aggregate))
    return candidates


def apply_train_fit_rule(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: dict[str, object],
) -> pd.DataFrame:
    fraction = float(spec["condition_fraction"])
    mask = np.ones(len(test), dtype=bool)
    for raw_condition in spec["conditions"]:
        condition = dict(raw_condition)
        feature = str(condition["feature"])
        direction = str(condition["direction"])
        quantile = 1 - fraction if direction == "high" else fraction
        threshold = float(train[feature].quantile(quantile))
        values = test[feature].to_numpy(dtype=np.float64)
        if direction == "high":
            mask &= values >= threshold
        elif direction == "low":
            mask &= values <= threshold
        else:
            raise ValueError(f"unknown rule direction: {direction}")
    return test.loc[mask]


def _empty_rule_aggregate(
    *,
    threshold_bps: float,
    horizon_seconds: int,
    spec: dict[str, object],
) -> dict[str, Any]:
    return {
        "threshold_bps": threshold_bps,
        "horizon_seconds": horizon_seconds,
        "conditions": spec["conditions"],
        "condition_fraction": spec["condition_fraction"],
        "fold_count": 0,
        "test_count": 0,
        "test_target_hits": 0,
        "test_gross_path_payoff_sum_bps": 0.0,
        "selected_count": 0,
        "selected_target_hits": 0,
        "selected_gross_path_payoff_sum_bps": 0.0,
    }


def _accumulate_rule(
    aggregate: dict[str, Any],
    test: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    aggregate["fold_count"] += 1
    aggregate["test_count"] += int(len(test))
    aggregate["test_target_hits"] += int(test["target"].sum())
    aggregate["test_gross_path_payoff_sum_bps"] += float(
        test["gross_path_payoff_bps"].sum()
    )
    aggregate["selected_count"] += int(len(selected))
    aggregate["selected_target_hits"] += int(selected["target"].sum())
    aggregate["selected_gross_path_payoff_sum_bps"] += float(
        selected["gross_path_payoff_bps"].sum()
    )


def _finalize_rule(aggregate: dict[str, Any]) -> dict[str, object]:
    test_count = int(aggregate["test_count"])
    selected_count = int(aggregate["selected_count"])
    base_target_rate = aggregate["test_target_hits"] / test_count
    selected_target_rate = aggregate["selected_target_hits"] / selected_count
    base_payoff = aggregate["test_gross_path_payoff_sum_bps"] / test_count
    selected_payoff = aggregate["selected_gross_path_payoff_sum_bps"] / selected_count
    return {
        "threshold_bps": float(aggregate["threshold_bps"]),
        "horizon_seconds": int(aggregate["horizon_seconds"]),
        "conditions": aggregate["conditions"],
        "condition_fraction": float(aggregate["condition_fraction"]),
        "fold_count": int(aggregate["fold_count"]),
        "test_count": test_count,
        "selected_count": selected_count,
        "base_target_rate": float(base_target_rate),
        "selected_target_rate": float(selected_target_rate),
        "target_rate_lift_pp": float(selected_target_rate - base_target_rate),
        "base_gross_path_payoff_mean_bps": float(base_payoff),
        "selected_gross_path_payoff_mean_bps": float(selected_payoff),
        "gross_path_payoff_lift_bps": float(selected_payoff - base_payoff),
    }


def _is_rule_feature(column: str) -> bool:
    return (
        column.startswith("directed_")
        or column.startswith(REGIME_FEATURE_PREFIXES)
        or column in REGIME_FEATURES
    )


def _add_gross_path_payoff(
    rows: pd.DataFrame,
    threshold_bps: float,
    initial_stop_bps: float,
) -> pd.DataFrame:
    enriched = rows.copy()
    enriched["gross_path_payoff_bps"] = np.where(
        enriched["paid_expected_before_stop"],
        threshold_bps,
        np.where(
            enriched["first_touch"] == "STOP",
            -initial_stop_bps,
            enriched["terminal_bps"],
        ),
    )
    return enriched


def _validate_parameters(
    *,
    thresholds_bps: tuple[float, ...],
    condition_fractions: tuple[float, ...],
    initial_stop_bps: float,
    temporal_blocks: int,
    minimum_train_blocks: int,
    minimum_selected_count: int,
    max_pair_features: int,
    max_report_candidates: int,
) -> None:
    if not thresholds_bps or any(threshold <= 0 for threshold in thresholds_bps):
        raise ValueError("thresholds_bps must contain positive values")
    if not condition_fractions or any(
        not 0 < fraction < 0.5 for fraction in condition_fractions
    ):
        raise ValueError("condition_fractions must contain values between 0 and 0.5")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")
    if temporal_blocks < 3:
        raise ValueError("temporal_blocks must be at least 3")
    if minimum_train_blocks < 1 or minimum_train_blocks >= temporal_blocks:
        raise ValueError("minimum_train_blocks must be between 1 and temporal_blocks - 1")
    if minimum_selected_count < 1:
        raise ValueError("minimum_selected_count must be positive")
    if max_pair_features < 2:
        raise ValueError("max_pair_features must be at least 2")
    if max_report_candidates < 1:
        raise ValueError("max_report_candidates must be positive")


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("list must contain positive integers")
    return values


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("list must contain positive floats")
    return values


def _not_none(value: int | None) -> int:
    if value is None:
        raise ValueError("required stream has no event time bounds")
    return value


def _dt(event_time_ms: int) -> datetime:
    return datetime.fromtimestamp(event_time_ms / 1000, tz=UTC)


def _iso(event_time_ms: int | None) -> str | None:
    if event_time_ms is None:
        return None
    return _dt(event_time_ms).isoformat().replace("+00:00", "Z")


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "status": report["status"],
        "approved_strategy": report["approved_strategy"],
        "decision_count": report.get("decision_count", 0),
        "feature_count": report.get("feature_count", 0),
        "candidates_evaluated": report.get("candidates_evaluated", 0),
        "best_candidate": report.get("best_candidate"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
