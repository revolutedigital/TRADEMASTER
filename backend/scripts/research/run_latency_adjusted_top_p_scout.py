"""Scout top-p entries with latency-adjusted path labels.

This is a development-only experiment. It keeps the original causal features at
the signal timestamp, but rebuilds the binary label from the executable entry
path after a fixed latency. The point is to test whether the previous top-p
lift was an artifact of labeling at the decision price while the replay entered
later.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backtest.event_replay import OrderSide
from app.services.backtest.trailing_portfolio import (
    V1_TRAILING_POLICIES,
    HistoricalTrailingSimulator,
    replay_one_position_portfolio,
    summarize_portfolio,
)
from app.services.research.path_labels import EventPathLabeler, PathLabelConfig
from app.services.research.research_dataset import ResearchDatasetConfig, load_trade_interval
from app.services.research.top_p_model import (
    TOP_P_TAILS,
    run_calibrated_walk_forward,
    run_calibrated_walk_forward_with_predictions,
    summarize_walk_forward,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "microstructure_v1" / "research-v1",
    )
    parser.add_argument(
        "--trade-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "normalized"
        / "aggTrades",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "latency-adjusted-scout-stress",
    )
    parser.add_argument("--target", choices=("expected", "stress"), default="stress")
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--minimum-selected", type=int, default=50)
    arguments = parser.parse_args()
    if arguments.latency_ms < 0:
        parser.error("--latency-ms cannot be negative")
    if arguments.minimum_selected < 1:
        parser.error("--minimum-selected must be positive")

    dataset_frame, opened_dates, label_config, label_config_source = load_research_dataset(
        arguments.dataset_root
    )
    latency_frame = apply_latency_adjusted_labels(
        dataset_frame,
        trade_root=arguments.trade_root,
        latency_ms=arguments.latency_ms,
        label_config=label_config,
    )
    target_column = f"latency_paid_{arguments.target}_before_stop"
    latency_frame["target"] = latency_frame[target_column].astype("int8")
    feature_sets = available_feature_sets(latency_frame)

    top_p_results: list[dict[str, object]] = []
    for horizon_seconds in label_config.horizons_seconds:
        for feature_set in feature_sets:
            result = run_calibrated_walk_forward(
                latency_frame,
                horizon_seconds=horizon_seconds,
                feature_set=feature_set,
            )
            top_p_results.append(
                {
                    "summary": summarize_walk_forward(result),
                    "detail": result.to_dict(),
                }
            )

    selection = select_portfolio_candidate(
        top_p_results,
        minimum_selected=arguments.minimum_selected,
    )
    selected_summary = top_p_results[selection["result_index"]]["summary"]
    selected_horizon = int(selected_summary["horizon_seconds"])
    selected_feature_set = str(selected_summary["feature_set"])
    _, predictions = run_calibrated_walk_forward_with_predictions(
        latency_frame,
        horizon_seconds=selected_horizon,
        feature_set=selected_feature_set,
    )
    portfolio_results = run_portfolio_replay(
        predictions,
        trade_root=arguments.trade_root,
        latency_ms=arguments.latency_ms,
        output_root=arguments.output_root,
    )

    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "experiment_stage": "development_latency_adjusted_scout",
        "warning": (
            "Development scout on a short local dataset. This cannot approve Testnet, Demo, "
            "or LIVE trading."
        ),
        "opened_dates": opened_dates,
        "latency_ms": arguments.latency_ms,
        "target_column": target_column,
        "label_config_source": label_config_source,
        "label_config": {
            "horizons_seconds": list(label_config.horizons_seconds),
            "expected_cost_bps": label_config.expected_cost_bps,
            "stress_cost_bps": label_config.stress_cost_bps,
            "initial_stop_bps": label_config.initial_stop_bps,
        },
        "feature_sets": feature_sets,
        "hypothesis_attempt_count": len(top_p_results),
        "candidate_selection": selection,
        "selected_candidate_summary": selected_summary,
        "top_p_results": top_p_results,
        "portfolio_hypothesis_attempt_count": len(portfolio_results),
        "portfolio_results": portfolio_results,
    }
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    output = arguments.output_root / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_console_summary(report, output), indent=2, sort_keys=True))  # noqa: T201
    return 0


def load_research_dataset(
    dataset_root: Path,
) -> tuple[pd.DataFrame, list[str], PathLabelConfig, str]:
    paths = sorted(dataset_root.glob("date=*/research_rows.parquet"))
    if not paths:
        raise ValueError(f"no research dataset partitions found in {dataset_root}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    manifests = [path.with_name("research_rows.manifest.json") for path in paths]
    config, config_source = _consistent_label_config(manifests)
    opened_dates = [path.parent.name.removeprefix("date=") for path in paths]
    return frame, opened_dates, config, config_source


def apply_latency_adjusted_labels(
    frame: pd.DataFrame,
    *,
    trade_root: Path,
    latency_ms: int,
    label_config: PathLabelConfig,
) -> pd.DataFrame:
    required = {"decision_time_ms", "side", "horizon_seconds"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"research frame is missing columns: {sorted(missing)}")
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")

    labelled = frame.copy().reset_index(drop=True)
    decision_dates = pd.to_datetime(
        labelled["decision_time_ms"], unit="ms", utc=True
    ).dt.date.astype(str)
    labelled["_utc_date"] = decision_dates
    expected_values = [0] * len(labelled)
    stress_values = [0] * len(labelled)
    terminal_values = [0.0] * len(labelled)
    mfe_values = [0.0] * len(labelled)
    mae_values = [0.0] * len(labelled)
    max_horizon = max(label_config.horizons_seconds)
    latency_padding = math.ceil(latency_ms / 1000)

    for utc_date, group in labelled.groupby("_utc_date", sort=True):
        day = date.fromisoformat(str(utc_date))
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        interval_start = day_start - timedelta(seconds=1)
        interval_end = day_start + timedelta(
            days=1,
            seconds=max_horizon + latency_padding + 1,
        )
        trades = load_trade_interval(
            trade_root,
            interval_start,
            interval_end,
            expected_product="usdm_perpetual",
        )
        ordered = trades.sort_values(["event_time_ms", "sequence_id"], kind="stable")
        labeler = EventPathLabeler(
            ordered["event_time_ms"].to_numpy(),
            ordered["sequence_id"].to_numpy(),
            ordered["price"].to_numpy(),
            config=label_config,
        )
        for row in group.itertuples():
            latency_decision_time_ms = int(row.decision_time_ms) + latency_ms
            label = labeler.label(
                latency_decision_time_ms,
                OrderSide(str(row.side)),
                int(row.horizon_seconds),
            )
            expected_values[row.Index] = int(label.paid_expected_before_stop)
            stress_values[row.Index] = int(label.paid_stress_before_stop)
            terminal_values[row.Index] = label.terminal_bps
            mfe_values[row.Index] = label.mfe_bps
            mae_values[row.Index] = label.mae_bps

    labelled["latency_paid_expected_before_stop"] = expected_values
    labelled["latency_paid_stress_before_stop"] = stress_values
    labelled["latency_terminal_bps"] = terminal_values
    labelled["latency_mfe_bps"] = mfe_values
    labelled["latency_mae_bps"] = mae_values
    return labelled.drop(columns=["_utc_date"])


def available_feature_sets(frame: pd.DataFrame) -> list[str]:
    feature_sets = ["flow", "flow_price", "flow_price_session"]
    if _has_auxiliary_features(frame):
        feature_sets.extend(["flow_aux", "flow_price_aux", "flow_price_aux_session"])
    if _has_book_features(frame):
        feature_sets.extend(["flow_book", "flow_price_book", "flow_price_book_session"])
        if _has_auxiliary_features(frame):
            feature_sets.extend(
                ["flow_book_aux", "flow_price_book_aux", "flow_price_book_aux_session"]
            )
    return feature_sets


def select_portfolio_candidate(
    top_p_results: list[dict[str, object]],
    *,
    minimum_selected: int,
) -> dict[str, object]:
    if not top_p_results:
        raise ValueError("top_p_results cannot be empty")
    ranked: list[tuple[tuple[float, float, float, float], dict[str, object]]] = []
    for index, result in enumerate(top_p_results):
        summary = _dict(result["summary"])
        tails = _dict(summary["tails"])
        top_1 = _dict(tails["top_1%"])
        selected_count = int(top_1["selected_count"])
        target_rate = float(top_1["target_rate"])
        monotonic = summary.get("top_p_monotonic") is True
        enough_rows = selected_count >= minimum_selected
        tier = 2 if monotonic and enough_rows else 1 if enough_rows else 0
        key = (float(tier), target_rate, float(summary["mean_roc_auc"]), -float(index))
        ranked.append(
            (
                key,
                {
                    "result_index": index,
                    "selection_tier": int(tier),
                    "selection_tier_reason": (
                        "monotonic_and_minimum_selected"
                        if tier == 2
                        else "minimum_selected_only"
                        if tier == 1
                        else "fallback_below_minimum_selected"
                    ),
                    "minimum_selected": minimum_selected,
                    "selected_count_top_1pct": selected_count,
                    "target_rate_top_1pct": target_rate,
                    "top_p_monotonic": monotonic,
                    "horizon_seconds": int(summary["horizon_seconds"]),
                    "feature_set": str(summary["feature_set"]),
                    "mean_roc_auc": float(summary["mean_roc_auc"]),
                },
            )
        )
    return max(ranked, key=lambda row: row[0])[1]


def run_portfolio_replay(
    predictions: pd.DataFrame,
    *,
    trade_root: Path,
    latency_ms: int,
    output_root: Path,
) -> list[dict[str, object]]:
    if predictions.empty:
        raise ValueError("prediction frame cannot be empty")
    start = datetime.fromtimestamp(
        predictions["decision_time_ms"].min() / 1000,
        tz=UTC,
    ) - timedelta(seconds=1)
    end = datetime.fromtimestamp(
        (predictions["decision_time_ms"] + predictions["horizon_seconds"] * 1000).max()
        / 1000,
        tz=UTC,
    ) + timedelta(seconds=1)
    trades = load_trade_interval(trade_root, start, end, expected_product="usdm_perpetual")
    simulator = HistoricalTrailingSimulator(
        trades["event_time_ms"].to_numpy(),
        trades["price"].to_numpy(),
        latency_ms=latency_ms,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for policy in V1_TRAILING_POLICIES:
        for tail_fraction in TOP_P_TAILS:
            managed = replay_one_position_portfolio(
                predictions,
                simulator,
                tail_fraction=tail_fraction,
                policy=policy,
            )
            filename = f"latency_policy={policy.name}-top={tail_fraction:.0%}.parquet"
            managed.to_parquet(output_root / filename, index=False)
            results.append(
                {
                    "policy": policy.name,
                    "tail_fraction": tail_fraction,
                    "policy_parameters": {
                        "name": policy.name,
                        "initial_stop_bps": policy.initial_stop_bps,
                        "breakeven_activation_bps": policy.breakeven_activation_bps,
                        "breakeven_lock_bps": policy.breakeven_lock_bps,
                        "trailing_activation_bps": policy.trailing_activation_bps,
                        "trailing_distance_bps": policy.trailing_distance_bps,
                        "partial_take_profit_bps": policy.partial_take_profit_bps,
                        "partial_take_fraction": policy.partial_take_fraction,
                    },
                    "summary": summarize_portfolio(managed),
                    "trades_path": filename,
                }
            )
    return results


def _consistent_label_config(manifests: list[Path]) -> tuple[PathLabelConfig, str]:
    configs: list[dict[str, Any]] = []
    legacy_config_hashes: set[str] = set()
    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(f"missing research partition manifest: {manifest}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        config = payload.get("config")
        if isinstance(config, dict):
            configs.append(config)
            continue
        config_sha256 = payload.get("config_sha256")
        if isinstance(config_sha256, str):
            legacy_config_hashes.add(config_sha256)
            continue
        raise ValueError(f"manifest is missing config: {manifest}")
    if not configs:
        default_config = ResearchDatasetConfig()
        config_source = (
            "legacy_manifest_default_config_sha256"
            if legacy_config_hashes in (set(), {default_config.sha256})
            else "legacy_manifest_inferred_path_label_config"
        )
        return default_config.path_label_config(), config_source
    first = configs[0]
    comparable_keys = (
        "horizons_seconds",
        "expected_cost_bps",
        "stress_cost_bps",
        "initial_stop_bps",
    )
    for config in configs[1:]:
        for key in comparable_keys:
            if config.get(key) != first.get(key):
                raise ValueError(f"research partition config mismatch for {key}")
    return (
        PathLabelConfig(
            horizons_seconds=tuple(int(value) for value in first["horizons_seconds"]),
            expected_cost_bps=float(first["expected_cost_bps"]),
            stress_cost_bps=float(first["stress_cost_bps"]),
            initial_stop_bps=float(first["initial_stop_bps"]),
        ),
        "partition_manifest_config",
    )


def _has_book_features(frame: pd.DataFrame) -> bool:
    return any(
        column in frame.columns
        for column in (
            "book_available",
            "spread_bps",
            "depth_imbalance",
            "microprice_displacement_bps",
        )
    )


def _has_auxiliary_features(frame: pd.DataFrame) -> bool:
    return any(
        any(column.startswith(prefix) for column in frame.columns)
        for prefix in (
            "mark_available",
            "mark_index_basis_bps",
            "funding_rate",
            "liquidation_net_qty_1s",
            "spot_available",
            "spot_perp_basis_bps",
            "spot_perp_return_gap_",
            "spot_perp_flow_gap_",
        )
    )


def _console_summary(report: dict[str, Any], output: Path) -> dict[str, Any]:
    portfolio_rows = [
        {
            "policy": result["policy"],
            "tail_fraction": result["tail_fraction"],
            **_dict(result["summary"]),
        }
        for result in report["portfolio_results"]
    ]
    best_expected = max(
        portfolio_rows,
        key=lambda row: float(row["expected_mean_bps"]),
    )
    best_stress = max(
        portfolio_rows,
        key=lambda row: float(row["stress_mean_bps"]),
    )
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "opened_dates": report["opened_dates"],
        "target_column": report["target_column"],
        "latency_ms": report["latency_ms"],
        "candidate_selection": report["candidate_selection"],
        "best_expected_portfolio": best_expected,
        "best_stress_portfolio": best_stress,
        "report_path": str(output),
    }


def _dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected dict")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
