"""Fast75 low-cost full-candidate universe payoff scout.

Fast74 killed the "filter the old top-p stream" path. Fast75 widens the
search to the low-cost research row universe instead of inheriting the old top-p
selector. It remains a scout, not a strategy: the payoff label is derived from
path MFE/MAE/terminal columns and is optimistic for targets above the dataset's
native expected-touch threshold because exact touch ordering is unavailable.

The script first runs a cheap frozen historical -> Fast56 extension screen. It
then validates any screen survivor with leave-one-block-out walk-forward at a
denser sample. No orders, no ledger writes, no approval.
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


DEFAULT_DATA_ROOT = REPO_ROOT / "backend" / "data" / "microstructure_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast75-lowcost-candidate-universe-scout"
DEFAULT_HORIZONS = (30, 120, 300)
DEFAULT_TARGET_BPS_VALUES = (1.0, 2.0, 3.0, 5.0, 8.0)
DEFAULT_TOP_FRACTIONS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
DEFAULT_SCREEN_SAMPLE_SECONDS = 300
DEFAULT_VALIDATION_SAMPLE_SECONDS = 60
DEFAULT_EXPLICIT_COST_BPS = 1.0
DEFAULT_INITIAL_STOP_BPS = 20.0
FEATURE_COLUMNS = (
    "trade_count_1s",
    "quote_volume_1s",
    "flow_imbalance_1s",
    "return_1s_bps",
    "realized_vol_1s_bps",
    "mean_interarrival_ms_1s",
    "trade_count_5s",
    "quote_volume_5s",
    "flow_imbalance_5s",
    "return_5s_bps",
    "realized_vol_5s_bps",
    "mean_interarrival_ms_5s",
    "trade_count_30s",
    "quote_volume_30s",
    "flow_imbalance_30s",
    "return_30s_bps",
    "realized_vol_30s_bps",
    "mean_interarrival_ms_30s",
    "trade_count_60s",
    "quote_volume_60s",
    "flow_imbalance_60s",
    "return_60s_bps",
    "realized_vol_60s_bps",
    "mean_interarrival_ms_60s",
    "hour_sin",
    "hour_cos",
    "side_sign",
    "directed_flow_imbalance_1s",
    "directed_return_1s_bps",
    "directed_flow_imbalance_5s",
    "directed_return_5s_bps",
    "directed_flow_imbalance_30s",
    "directed_return_30s_bps",
    "directed_flow_imbalance_60s",
    "directed_return_60s_bps",
)
LOAD_COLUMNS = (
    "decision_time_ms",
    "side",
    "horizon_seconds",
    "mfe_bps",
    "mae_bps",
    "terminal_bps",
    "first_touch",
    *FEATURE_COLUMNS,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--screen-sample-seconds", type=int, default=DEFAULT_SCREEN_SAMPLE_SECONDS)
    parser.add_argument(
        "--validation-sample-seconds",
        type=int,
        default=DEFAULT_VALIDATION_SAMPLE_SECONDS,
    )
    parser.add_argument("--explicit-cost-bps", type=float, default=DEFAULT_EXPLICIT_COST_BPS)
    parser.add_argument("--initial-stop-bps", type=float, default=DEFAULT_INITIAL_STOP_BPS)
    arguments = parser.parse_args()
    report = run_fast75_lowcost_candidate_universe_scout(
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        screen_sample_seconds=arguments.screen_sample_seconds,
        validation_sample_seconds=arguments.validation_sample_seconds,
        explicit_cost_bps=arguments.explicit_cost_bps,
        initial_stop_bps=arguments.initial_stop_bps,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast75_lowcost_candidate_universe_scout(
    *,
    data_root: Path,
    output_root: Path,
    screen_sample_seconds: int = DEFAULT_SCREEN_SAMPLE_SECONDS,
    validation_sample_seconds: int = DEFAULT_VALIDATION_SAMPLE_SECONDS,
    explicit_cost_bps: float = DEFAULT_EXPLICIT_COST_BPS,
    initial_stop_bps: float = DEFAULT_INITIAL_STOP_BPS,
) -> dict[str, Any]:
    validate_parameters(
        screen_sample_seconds=screen_sample_seconds,
        validation_sample_seconds=validation_sample_seconds,
        explicit_cost_bps=explicit_cost_bps,
        initial_stop_bps=initial_stop_bps,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    historical_roots, extension_root = discover_lowcost_roots(data_root)
    screen_rows = run_screen(
        historical_roots=historical_roots,
        extension_root=extension_root,
        sample_seconds=screen_sample_seconds,
        explicit_cost_bps=explicit_cost_bps,
        initial_stop_bps=initial_stop_bps,
    )
    screen_survivors = [
        row
        for row in screen_rows
        if row["extension"]["trade_count"] >= 3
        and row["extension"]["net_total_bps"] > 0
        and row["historical"]["net_total_bps"] > 0
    ]
    validations = [
        validate_candidate(
            candidate,
            historical_roots=historical_roots,
            extension_root=extension_root,
            sample_seconds=validation_sample_seconds,
            explicit_cost_bps=explicit_cost_bps,
            initial_stop_bps=initial_stop_bps,
        )
        for candidate in screen_survivors
    ]
    status, status_reasons = fast75_status(screen_survivors=screen_survivors, validations=validations)
    screen_path = output_root / "screen_results.json"
    screen_path.write_text(json.dumps(screen_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validations_path = output_root / "validations.json"
    validations_path.write_text(json.dumps(validations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast75_is_lowcost_scout",
            "path_payoff_label_is_optimistic_for_targets_above_native_touch_threshold",
            "validation_and_extension_must_pass_before_shadow_candidate",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast75_lowcost_candidate_universe_scout",
        "status": status,
        "status_reasons": status_reasons,
        "data_root": str(data_root),
        "historical_roots": [str(root) for root in historical_roots],
        "extension_root": str(extension_root),
        "screen_sample_seconds": screen_sample_seconds,
        "validation_sample_seconds": validation_sample_seconds,
        "explicit_cost_bps": explicit_cost_bps,
        "initial_stop_bps": initial_stop_bps,
        "feature_count": len(FEATURE_COLUMNS),
        "horizons": list(DEFAULT_HORIZONS),
        "target_bps_values": list(DEFAULT_TARGET_BPS_VALUES),
        "top_fractions": list(DEFAULT_TOP_FRACTIONS),
        "screen_result_count": len(screen_rows),
        "screen_survivor_count": len(screen_survivors),
        "screen_survivors": screen_survivors,
        "validation_count": len(validations),
        "validations": validations,
        "screen_results_path": str(screen_path),
        "screen_results_sha256": _sha256_file(screen_path),
        "validations_path": str(validations_path),
        "validations_sha256": _sha256_file(validations_path),
    }
    return _write_report(output_root, report)


def validate_parameters(
    *,
    screen_sample_seconds: int,
    validation_sample_seconds: int,
    explicit_cost_bps: float,
    initial_stop_bps: float,
) -> None:
    if screen_sample_seconds <= 0:
        raise ValueError("screen_sample_seconds must be positive")
    if validation_sample_seconds <= 0:
        raise ValueError("validation_sample_seconds must be positive")
    if explicit_cost_bps < 0:
        raise ValueError("explicit_cost_bps cannot be negative")
    if initial_stop_bps <= 0:
        raise ValueError("initial_stop_bps must be positive")


def discover_lowcost_roots(data_root: Path) -> tuple[tuple[Path, ...], Path]:
    roots = sorted(data_root.glob("research-lowcost-fast*"))
    historical = tuple(root for root in roots if "fast41" not in root.name and "fast56" not in root.name)
    extension = data_root / "research-lowcost-fast56-holdout-new"
    if not historical:
        raise FileNotFoundError("no historical research-lowcost roots found")
    if not extension.exists():
        raise FileNotFoundError(f"extension root missing: {extension}")
    return historical, extension


def run_screen(
    *,
    historical_roots: tuple[Path, ...],
    extension_root: Path,
    sample_seconds: int,
    explicit_cost_bps: float,
    initial_stop_bps: float,
) -> list[dict[str, Any]]:
    rows = []
    for horizon_seconds in DEFAULT_HORIZONS:
        historical = pd.concat(
            [load_lowcost_root(root, horizon_seconds=horizon_seconds, sample_seconds=sample_seconds) for root in historical_roots],
            ignore_index=True,
        )
        extension = load_lowcost_root(extension_root, horizon_seconds=horizon_seconds, sample_seconds=sample_seconds)
        for target_bps in DEFAULT_TARGET_BPS_VALUES:
            historical_target = add_approx_path_net_bps(
                historical,
                target_bps=target_bps,
                explicit_cost_bps=explicit_cost_bps,
                initial_stop_bps=initial_stop_bps,
            )
            model = train_payoff_model(historical_target)
            historical_scored = score_rows(model, historical_target)
            extension_target = add_approx_path_net_bps(
                extension,
                target_bps=target_bps,
                explicit_cost_bps=explicit_cost_bps,
                initial_stop_bps=initial_stop_bps,
            )
            extension_scored = score_rows(model, extension_target)
            for top_fraction in DEFAULT_TOP_FRACTIONS:
                threshold = score_threshold(historical_scored, top_fraction=top_fraction)
                historical_selected = select_top_one_position(
                    historical_scored,
                    score_threshold_value=threshold,
                    horizon_seconds=horizon_seconds,
                )
                extension_selected = select_top_one_position(
                    extension_scored,
                    score_threshold_value=threshold,
                    horizon_seconds=horizon_seconds,
                )
                rows.append(
                    {
                        "horizon_seconds": horizon_seconds,
                        "target_bps": target_bps,
                        "top_fraction": top_fraction,
                        "score_threshold": threshold,
                        "historical": summarize_net(historical_selected),
                        "extension": summarize_net(extension_selected),
                    }
                )
    return rows


def validate_candidate(
    candidate: dict[str, Any],
    *,
    historical_roots: tuple[Path, ...],
    extension_root: Path,
    sample_seconds: int,
    explicit_cost_bps: float,
    initial_stop_bps: float,
) -> dict[str, Any]:
    horizon_seconds = int(candidate["horizon_seconds"])
    target_bps = float(candidate["target_bps"])
    top_fraction = float(candidate["top_fraction"])
    by_block = {
        root.name: add_approx_path_net_bps(
            load_lowcost_root(root, horizon_seconds=horizon_seconds, sample_seconds=sample_seconds),
            target_bps=target_bps,
            explicit_cost_bps=explicit_cost_bps,
            initial_stop_bps=initial_stop_bps,
        )
        for root in historical_roots
    }
    historical = pd.concat(by_block.values(), ignore_index=True)
    blocks = []
    selected_parts = []
    for block_name, holdout in by_block.items():
        training = historical[historical["block"].ne(block_name)]
        model = train_payoff_model(training)
        training_scored = score_rows(model, training)
        threshold = score_threshold(training_scored, top_fraction=top_fraction)
        holdout_scored = score_rows(model, holdout)
        selected = select_top_one_position(
            holdout_scored,
            score_threshold_value=threshold,
            horizon_seconds=horizon_seconds,
        )
        selected_parts.append(selected)
        blocks.append({"block": block_name, **summarize_net(selected)})
    all_selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    sufficient = [block for block in blocks if block["trade_count"] >= 3]
    historical_summary = summarize_net(all_selected) | {
        "sufficient_block_count": len(sufficient),
        "positive_sufficient_block_count": sum(block["net_total_bps"] > 0 for block in sufficient),
        "negative_sufficient_blocks": [
            block["block"] for block in sufficient if block["net_total_bps"] <= 0
        ],
        "min_sufficient_block_net_total_bps": min(
            (block["net_total_bps"] for block in sufficient),
            default=0.0,
        ),
    }
    extension = add_approx_path_net_bps(
        load_lowcost_root(extension_root, horizon_seconds=horizon_seconds, sample_seconds=sample_seconds),
        target_bps=target_bps,
        explicit_cost_bps=explicit_cost_bps,
        initial_stop_bps=initial_stop_bps,
    )
    model = train_payoff_model(historical)
    historical_scored = score_rows(model, historical)
    threshold = score_threshold(historical_scored, top_fraction=top_fraction)
    extension_selected = select_top_one_position(
        score_rows(model, extension),
        score_threshold_value=threshold,
        horizon_seconds=horizon_seconds,
    )
    return {
        "horizon_seconds": horizon_seconds,
        "target_bps": target_bps,
        "top_fraction": top_fraction,
        "sample_seconds": sample_seconds,
        "historical": historical_summary,
        "blocks": blocks,
        "extension": summarize_net(extension_selected),
    }


def load_lowcost_root(root: Path, *, horizon_seconds: int, sample_seconds: int) -> pd.DataFrame:
    parts = []
    for path in sorted(root.glob("date=*/research_rows.parquet")):
        frame = pd.read_parquet(path, columns=list(LOAD_COLUMNS))
        frame = frame[frame["horizon_seconds"].eq(horizon_seconds)].copy()
        frame = frame[((frame["decision_time_ms"] // 1000) % sample_seconds).eq(5)]
        if not frame.empty:
            parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=list(LOAD_COLUMNS) + ["block"])
    loaded = pd.concat(parts, ignore_index=True)
    loaded["block"] = root.name
    return loaded


def add_approx_path_net_bps(
    frame: pd.DataFrame,
    *,
    target_bps: float,
    explicit_cost_bps: float,
    initial_stop_bps: float,
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["net_bps"] = approx_path_net_bps(
        first_touch=enriched["first_touch"],
        mfe_bps=enriched["mfe_bps"],
        terminal_bps=enriched["terminal_bps"],
        target_bps=target_bps,
        explicit_cost_bps=explicit_cost_bps,
        initial_stop_bps=initial_stop_bps,
    )
    return enriched


def approx_path_net_bps(
    *,
    first_touch: pd.Series,
    mfe_bps: pd.Series,
    terminal_bps: pd.Series,
    target_bps: float,
    explicit_cost_bps: float,
    initial_stop_bps: float,
) -> np.ndarray:
    gross = np.where(
        first_touch.eq("STOP"),
        -initial_stop_bps,
        np.where(mfe_bps >= target_bps, target_bps, terminal_bps),
    )
    return gross - explicit_cost_bps


def train_payoff_model(frame: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    model.fit(frame[list(FEATURE_COLUMNS)], frame["net_bps"])
    return model


def score_rows(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = model.predict(scored[list(FEATURE_COLUMNS)]) if not scored.empty else []
    return scored


def score_threshold(frame: pd.DataFrame, *, top_fraction: float) -> float:
    return float(pd.to_numeric(frame["score"], errors="coerce").quantile(1.0 - top_fraction))


def select_top_one_position(
    frame: pd.DataFrame,
    *,
    score_threshold_value: float,
    horizon_seconds: int,
) -> pd.DataFrame:
    filtered = frame[frame["score"] >= score_threshold_value].copy()
    if filtered.empty:
        return filtered
    ordered = filtered.sort_values(
        ["decision_time_ms", "score"],
        ascending=[True, False],
        kind="stable",
    )
    keep = []
    position_exit_ms = -1
    for index, row in ordered.iterrows():
        decision_time_ms = int(row["decision_time_ms"])
        if decision_time_ms <= position_exit_ms:
            continue
        keep.append(index)
        position_exit_ms = decision_time_ms + horizon_seconds * 1000
    return ordered.loc[keep]


def summarize_net(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "trade_count": 0,
            "distinct_days": 0,
            "net_total_bps": 0.0,
            "net_mean_bps": 0.0,
            "win_rate": 0.0,
        }
    dates = pd.to_datetime(frame["decision_time_ms"], unit="ms", utc=True).dt.date
    return {
        "trade_count": int(len(frame)),
        "distinct_days": int(dates.nunique()),
        "net_total_bps": float(frame["net_bps"].sum()),
        "net_mean_bps": float(frame["net_bps"].mean()),
        "win_rate": float((frame["net_bps"] > 0).mean()),
    }


def fast75_status(
    *,
    screen_survivors: list[dict[str, Any]],
    validations: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if not screen_survivors:
        return "FAST75_LOWCOST_UNIVERSE_FAILED_SCREEN", reasons + ["no_screen_survivor"]
    passing = [
        item
        for item in validations
        if item["historical"]["negative_sufficient_blocks"] == []
        and item["historical"]["net_total_bps"] > 0
        and item["extension"]["trade_count"] >= 3
        and item["extension"]["net_total_bps"] > 0
    ]
    if not passing:
        return "FAST75_LOWCOST_UNIVERSE_FAILED_VALIDATION", reasons + [
            "no_screen_survivor_passed_denser_walkforward_and_extension"
        ]
    return "FAST75_LOWCOST_UNIVERSE_RESEARCH_CANDIDATE_REQUIRES_EXECUTABLE_SHADOW", reasons + [
        "requires_executable_shadow_replay"
    ]


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
        "screen_result_count": report["screen_result_count"],
        "screen_survivor_count": report["screen_survivor_count"],
        "screen_survivors": report["screen_survivors"],
        "validations": report["validations"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
