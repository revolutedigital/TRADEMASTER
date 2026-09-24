"""Fast125: entry-known top-p model gate for the USDJPY compression clue.

Fast124 froze the USDJPY H1 compression-breakout clue for prospective
observation. Fast125 asks the user's top-p question on historical data without
leakage: can a simple entry-known probability/expectancy model select better
entries year by year than the frozen rule itself?

Each test year trains only on earlier years, chooses the model/top-p threshold
only on that training sample, and then evaluates the threshold on the next
year. This is still historical research, not strategy approval.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast120_fx_currency_strength_scout import plain_json, write_json  # noqa: E402
from scripts.research.run_fast124_usdjpy_compression_observation import (  # noqa: E402
    FROZEN_POLICY,
    OBSERVATION_PAIR,
    SAFETY_FLAGS,
    SOURCE_FAST121_REPORT_SHA256,
    SOURCE_FAST122_REPORT_SHA256,
    SOURCE_FAST123_REPORT_SHA256,
    sha256_file,
)

DEFAULT_TRADE_ROWS_PATH = (
    REPO_ROOT
    / "backend"
    / "data"
    / "lab_fast122"
    / "usdjpy-compression-breakout-temporal-audit"
    / "frozen-policy-trades.parquet"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "data" / "lab_fast125" / "usdjpy-compression-topp-model"
)
MODEL_TEST_YEARS = tuple(range(2016, 2027))
TOP_P_GRID = (0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
MINIMUM_TRAIN_TRADES = 250
MINIMUM_TRAIN_SELECTED_TRADES = 80
MINIMUM_TEST_YEAR_TRADES = 20
DEFAULT_ALPHA = 0.05
SHADOW_CANDIDATE_MINIMUM_OOS_TRADES = 300
SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS = 1.0
SHADOW_CANDIDATE_MINIMUM_STRESS_LIFT_PIPS = 0.5
SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_SHARE = 2 / 3
SHADOW_CANDIDATE_MINIMUM_YEARLY_TRADES = 10
SHADOW_CANDIDATE_MINIMUM_WORST_YEAR_MEAN_PIPS = -5.0

ModelName = Literal["ridge_expected_pips", "logistic_positive"]


FEATURE_COLUMNS = (
    "side_num",
    "breakout_distance_pips",
    "atr_pips",
    "risk_pips",
    "compression_value_pips",
    "compression_threshold_pips",
    "compression_ratio",
    "breakout_over_atr",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-rows-path", type=Path, default=DEFAULT_TRADE_ROWS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args(argv)
    report = run_fast125_usdjpy_compression_topp_model(
        trade_rows_path=arguments.trade_rows_path,
        output_root=arguments.output_root,
        alpha=arguments.alpha,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast125_usdjpy_compression_topp_model(
    *,
    trade_rows_path: Path = DEFAULT_TRADE_ROWS_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    validate_parameters(alpha=alpha)
    trade_rows_path = absolute_path(trade_rows_path)
    output_root = absolute_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    trades = load_trade_rows(trade_rows_path)
    featured = add_entry_features(trades)
    baseline = featured.loc[featured["year"].isin(MODEL_TEST_YEARS)].copy()
    fold_candidates, selected_rows = walk_forward_top_p(featured, alpha=alpha)
    if fold_candidates.empty:
        raise ValueError("no fold candidates generated")

    fold_candidates_path = output_root / "fold-candidates.csv"
    selected_rows_path = output_root / "selected-oos-trades.parquet"
    fold_candidates.to_csv(fold_candidates_path, index=False)
    selected_rows.to_parquet(selected_rows_path, compression="zstd", index=False)

    selected_summary = metric_summary(selected_rows, alpha=alpha)
    baseline_summary = metric_summary(baseline, alpha=alpha)
    gates = evaluate_gates(selected_summary=selected_summary, baseline_summary=baseline_summary)
    status, status_reasons = fast125_status(gates)
    report = {
        **SAFETY_FLAGS,
        "approval_reasons": [
            "fast125_is_historical_walk_forward_research_not_strategy_approval",
            "top_p_thresholds_are_selected_on_prior_years_only",
            "fresh_preregistered_prospective_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast125_usdjpy_compression_topp_model",
        "status": status,
        "status_reasons": status_reasons,
        "pair": OBSERVATION_PAIR,
        "frozen_policy": {
            **asdict(FROZEN_POLICY),
            "management": asdict(FROZEN_POLICY.management),
        },
        "trade_rows_path": str(trade_rows_path),
        "trade_rows_sha256": sha256_file(trade_rows_path),
        "source_fast121_report_sha256": SOURCE_FAST121_REPORT_SHA256,
        "source_fast122_report_sha256": SOURCE_FAST122_REPORT_SHA256,
        "source_fast123_report_sha256": SOURCE_FAST123_REPORT_SHA256,
        "model_test_years": list(MODEL_TEST_YEARS),
        "top_p_grid": list(TOP_P_GRID),
        "feature_columns": list(FEATURE_COLUMNS),
        "selection_protocol": {
            "train_window": "all years strictly before test_year",
            "threshold": "train score quantile at 1 - top_p",
            "test_rule": "include test entries with score >= train threshold",
            "model_choice": "best train lower confidence bound among eligible model/top_p candidates",
            "target_for_logistic_positive": "net_stress_pips > 0",
            "target_for_ridge_expected_pips": "net_stress_pips",
        },
        "fold_count": int(fold_candidates["test_year"].nunique()),
        "fold_candidates": int(len(fold_candidates)),
        "selected_oos_summary": selected_summary,
        "baseline_oos_summary": baseline_summary,
        "gates": gates,
        "artifacts": {
            "fold_candidates": {
                "path": str(fold_candidates_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(fold_candidates_path),
            },
            "selected_oos_trades": {
                "path": str(selected_rows_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(selected_rows_path),
            },
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, {**report, "report_path": str(report_path.relative_to(REPO_ROOT))})
    return {
        **plain_json(report),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
    }


def absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def validate_parameters(*, alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def load_trade_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pd.read_parquet(path)
    required = {
        "pair",
        "policy_id",
        "entry_timestamp",
        "side",
        "breakout_distance_pips",
        "atr_pips",
        "risk_pips",
        "compression_value_pips",
        "compression_threshold_pips",
        "net_base_pips",
        "net_stress_pips",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"trade rows missing columns: {sorted(missing)}")
    rows = rows.copy()
    rows["entry_timestamp"] = pd.to_datetime(rows["entry_timestamp"], utc=True)
    rows = rows.loc[
        rows["pair"].eq(OBSERVATION_PAIR) & rows["policy_id"].eq(FROZEN_POLICY.identifier)
    ].copy()
    if rows.empty:
        raise ValueError("no rows for Fast125 frozen policy")
    return rows.sort_values("entry_timestamp", ignore_index=True)


def add_entry_features(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    frame["year"] = frame["entry_timestamp"].dt.year
    frame["hour"] = frame["entry_timestamp"].dt.hour
    frame["weekday"] = frame["entry_timestamp"].dt.weekday
    frame["month"] = frame["entry_timestamp"].dt.month
    frame["side_num"] = frame["side"].eq("LONG").astype(int)
    frame["compression_ratio"] = safe_ratio(
        frame["compression_value_pips"],
        frame["compression_threshold_pips"],
    )
    frame["breakout_over_atr"] = safe_ratio(frame["breakout_distance_pips"], frame["atr_pips"])
    add_cyclical_features(frame, column="hour", period=24)
    add_cyclical_features(frame, column="weekday", period=7)
    add_cyclical_features(frame, column="month", period=12)
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(FEATURE_COLUMNS))
    return clean.sort_values("entry_timestamp", ignore_index=True)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype("float64") / denominator.astype("float64").replace(0, np.nan)


def add_cyclical_features(frame: pd.DataFrame, *, column: str, period: int) -> None:
    radians = 2 * np.pi * frame[column].astype("float64") / period
    frame[f"{column}_sin"] = np.sin(radians)
    frame[f"{column}_cos"] = np.cos(radians)


def walk_forward_top_p(trades: pd.DataFrame, *, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for test_year in MODEL_TEST_YEARS:
        train = trades.loc[trades["year"] < test_year].copy()
        test = trades.loc[trades["year"].eq(test_year)].copy()
        if len(train) < MINIMUM_TRAIN_TRADES or len(test) < MINIMUM_TEST_YEAR_TRADES:
            continue

        best_key: tuple[float, float, int] | None = None
        best_row: dict[str, Any] | None = None
        best_test_sample = pd.DataFrame(columns=trades.columns)
        for model_name in ("ridge_expected_pips", "logistic_positive"):
            model = fit_model(model_name, train)
            if model is None:
                continue
            train_score = score_model(model_name, model, train)
            test_score = score_model(model_name, model, test)
            for top_p in TOP_P_GRID:
                threshold = float(np.quantile(train_score, 1 - top_p))
                train_sample = threshold_sample(train, train_score, threshold=threshold)
                test_sample = threshold_sample(test, test_score, threshold=threshold)
                train_summary = metric_summary(train_sample, alpha=alpha)
                test_summary = metric_summary(test_sample, alpha=alpha)
                row = fold_candidate_row(
                    test_year=test_year,
                    model_name=model_name,
                    top_p=top_p,
                    threshold=threshold,
                    train_summary=train_summary,
                    test_summary=test_summary,
                )
                candidate_rows.append(row)
                if not train_candidate_is_eligible(train_summary):
                    continue
                key = (
                    train_summary["lower_confidence_bound_pips"]
                    if train_summary["lower_confidence_bound_pips"] is not None
                    else -9999.0,
                    train_summary["mean_stress_pips"]
                    if train_summary["mean_stress_pips"] is not None
                    else -9999.0,
                    train_summary["trade_count"],
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_row = row
                    best_test_sample = test_sample.copy()

        if best_row is not None:
            best_test_sample["selected_model"] = best_row["model"]
            best_test_sample["selected_top_p"] = best_row["top_p"]
            best_test_sample["selected_threshold"] = best_row["threshold"]
            best_test_sample["test_year"] = test_year
            selected_frames.append(best_test_sample)

    candidates = pd.DataFrame(candidate_rows)
    selected = (
        pd.concat(selected_frames, ignore_index=True).sort_values("entry_timestamp", ignore_index=True)
        if selected_frames
        else pd.DataFrame(columns=list(trades.columns) + ["selected_model", "selected_top_p", "selected_threshold"])
    )
    return candidates, selected


def fit_model(model_name: ModelName, train: pd.DataFrame) -> Pipeline | None:
    if model_name == "ridge_expected_pips":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(train[list(FEATURE_COLUMNS)], train["net_stress_pips"].astype("float64"))
        return model
    target = train["net_stress_pips"].gt(0).astype(int)
    if target.nunique() < 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.25, max_iter=1000, class_weight="balanced"),
    )
    model.fit(train[list(FEATURE_COLUMNS)], target)
    return model


def score_model(model_name: ModelName, model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    if model_name == "logistic_positive":
        return model.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1]
    return model.predict(frame[list(FEATURE_COLUMNS)])


def threshold_sample(frame: pd.DataFrame, scores: np.ndarray, *, threshold: float) -> pd.DataFrame:
    scored = frame.copy()
    scored["model_score"] = scores
    return scored.loc[scored["model_score"] >= threshold].copy()


def train_candidate_is_eligible(summary: dict[str, Any]) -> bool:
    return (
        summary["trade_count"] >= MINIMUM_TRAIN_SELECTED_TRADES
        and summary["mean_base_pips"] is not None
        and summary["mean_base_pips"] > 0
        and summary["mean_stress_pips"] is not None
        and summary["mean_stress_pips"] > 0
        and summary["positive_year_share"] is not None
        and summary["positive_year_share"] >= 0.5
    )


def fold_candidate_row(
    *,
    test_year: int,
    model_name: str,
    top_p: float,
    threshold: float,
    train_summary: dict[str, Any],
    test_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "test_year": test_year,
        "model": model_name,
        "top_p": top_p,
        "threshold": threshold,
        "train_trades": train_summary["trade_count"],
        "train_mean_base_pips": train_summary["mean_base_pips"],
        "train_mean_stress_pips": train_summary["mean_stress_pips"],
        "train_lower_stress_pips": train_summary["lower_confidence_bound_pips"],
        "train_positive_year_share": train_summary["positive_year_share"],
        "test_trades": test_summary["trade_count"],
        "test_mean_base_pips": test_summary["mean_base_pips"],
        "test_mean_stress_pips": test_summary["mean_stress_pips"],
        "test_lower_stress_pips": test_summary["lower_confidence_bound_pips"],
        "test_total_stress_pips": test_summary["total_stress_pips"],
        "test_positive_year_share": test_summary["positive_year_share"],
    }


def metric_summary(sample: pd.DataFrame, *, alpha: float) -> dict[str, Any]:
    if sample.empty:
        return {
            "trade_count": 0,
            "mean_base_pips": None,
            "mean_stress_pips": None,
            "median_stress_pips": None,
            "total_stress_pips": 0.0,
            "lower_confidence_bound_pips": None,
            "positive_year_share": None,
            "positive_year_count": 0,
            "year_count": 0,
            "minimum_yearly_trades": 0,
            "worst_year_mean_stress_pips": None,
            "yearly": [],
        }
    frame = sample.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    stress = frame["net_stress_pips"].astype("float64")
    yearly = yearly_summary(frame)
    positive_year_count = sum(row["mean_stress_pips"] > 0 for row in yearly)
    return {
        "trade_count": int(len(frame)),
        "mean_base_pips": float(frame["net_base_pips"].mean()),
        "mean_stress_pips": float(stress.mean()),
        "median_stress_pips": float(stress.median()),
        "total_stress_pips": float(stress.sum()),
        "lower_confidence_bound_pips": lower_confidence_bound(stress, alpha=alpha),
        "positive_year_share": positive_year_count / len(yearly) if yearly else None,
        "positive_year_count": int(positive_year_count),
        "year_count": int(len(yearly)),
        "minimum_yearly_trades": min((row["trade_count"] for row in yearly), default=0),
        "worst_year_mean_stress_pips": min(
            (row["mean_stress_pips"] for row in yearly),
            default=None,
        ),
        "yearly": yearly,
    }


def yearly_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        frame.assign(year=frame["entry_timestamp"].dt.year)
        .groupby("year", sort=True)
        .agg(
            trade_count=("net_stress_pips", "size"),
            mean_base_pips=("net_base_pips", "mean"),
            mean_stress_pips=("net_stress_pips", "mean"),
            total_stress_pips=("net_stress_pips", "sum"),
        )
        .reset_index()
    )
    return [
        {
            "year": int(row["year"]),
            "trade_count": int(row["trade_count"]),
            "mean_base_pips": float(row["mean_base_pips"]),
            "mean_stress_pips": float(row["mean_stress_pips"]),
            "total_stress_pips": float(row["total_stress_pips"]),
        }
        for row in grouped.to_dict("records")
    ]


def lower_confidence_bound(values: pd.Series, *, alpha: float) -> float | None:
    sample = values.astype("float64").dropna().to_numpy()
    if len(sample) < 2:
        return None
    standard_error = float(sample.std(ddof=1) / np.sqrt(len(sample)))
    critical = float(t.ppf(1 - alpha, df=len(sample) - 1))
    return float(sample.mean() - critical * standard_error)


def evaluate_gates(
    *,
    selected_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    selected_mean = selected_summary["mean_stress_pips"]
    baseline_mean = baseline_summary["mean_stress_pips"]
    selected_lcb = selected_summary["lower_confidence_bound_pips"]
    stress_lift = (
        selected_mean - baseline_mean
        if selected_mean is not None and baseline_mean is not None
        else None
    )
    if selected_summary["trade_count"] < SHADOW_CANDIDATE_MINIMUM_OOS_TRADES:
        reasons.append("insufficient_oos_trades")
    if selected_mean is None or selected_mean <= SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS:
        reasons.append("selected_stress_mean_below_threshold")
    if selected_lcb is None or selected_lcb <= 0:
        reasons.append("selected_stress_lower_confidence_bound_not_positive")
    if stress_lift is None or stress_lift < SHADOW_CANDIDATE_MINIMUM_STRESS_LIFT_PIPS:
        reasons.append("stress_lift_vs_unfiltered_baseline_too_small")
    if (
        selected_summary["positive_year_share"] is None
        or selected_summary["positive_year_share"] < SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_SHARE
    ):
        reasons.append("positive_year_share_below_threshold")
    if selected_summary["minimum_yearly_trades"] < SHADOW_CANDIDATE_MINIMUM_YEARLY_TRADES:
        reasons.append("yearly_trade_count_too_sparse")
    if (
        selected_summary["worst_year_mean_stress_pips"] is None
        or selected_summary["worst_year_mean_stress_pips"]
        <= SHADOW_CANDIDATE_MINIMUM_WORST_YEAR_MEAN_PIPS
    ):
        reasons.append("worst_year_drawdown_too_large")
    return {
        "shadow_candidate": not reasons,
        "shadow_candidate_reasons": reasons,
        "stress_lift_vs_unfiltered_baseline_pips": stress_lift,
        "candidate_thresholds": {
            "minimum_oos_trades": SHADOW_CANDIDATE_MINIMUM_OOS_TRADES,
            "minimum_stress_mean_pips": SHADOW_CANDIDATE_MINIMUM_STRESS_MEAN_PIPS,
            "minimum_stress_lift_pips": SHADOW_CANDIDATE_MINIMUM_STRESS_LIFT_PIPS,
            "minimum_positive_year_share": SHADOW_CANDIDATE_MINIMUM_POSITIVE_YEAR_SHARE,
            "minimum_yearly_trades": SHADOW_CANDIDATE_MINIMUM_YEARLY_TRADES,
            "minimum_worst_year_mean_pips": SHADOW_CANDIDATE_MINIMUM_WORST_YEAR_MEAN_PIPS,
        },
    }


def fast125_status(gates: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if gates["shadow_candidate"]:
        return (
            "FAST125_USDJPY_COMPRESSION_TOPP_MODEL_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION",
            reasons + ["walk_forward_top_p_gate_passed_but_no_prospective_evidence"],
        )
    return (
        "FAST125_USDJPY_COMPRESSION_TOPP_MODEL_FAILED_OOS",
        reasons + gates["shadow_candidate_reasons"],
    )


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "approved_strategy": report["approved_strategy"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "selected_oos_summary": report["selected_oos_summary"],
        "baseline_oos_summary": report["baseline_oos_summary"],
        "gates": report["gates"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
