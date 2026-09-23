"""Fast74 non-STOP target model audit.

Fast73 showed that the original top-p target is misaligned with trailing
settlement PnL: many trades touch the stress target before the stop and still
lose after execution replay. Fast74 tests the next natural target over the
same frozen top-p stream: predict whether the trade avoids an initial STOP.

The audit separates a diagnostic rank view from an executable frozen-threshold
view. Only the frozen-threshold view can be promoted to shadow, and it is
checked against the later Fast56 extension block. No orders or ledger writes
are allowed.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast65_causal_feature_gate_walkforward import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    causal_feature_columns,
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
from scripts.research.run_fast73_anti_stop_density_audit import (  # noqa: E402
    aggregate_blocks,
    density_filter_records,
    summarize_dataframe_blocks,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "reports" / "fast74-non-stop-model-audit"
DEFAULT_FAST69_RAW_DECISIONS = (
    DEFAULT_DATA_ROOT / "reports" / "fast69-directional-shadow-dry-run" / "raw_top_p_decisions.jsonl"
)
DEFAULT_FAST73_SETTLEMENT_CACHE = (
    DEFAULT_DATA_ROOT / "reports" / "fast73-top-p-settlement-cache" / "raw_top_p_settlements.jsonl"
)
NON_STOP_SCORE_COLUMN = "non_stop_probability"
MODEL_RANDOM_STATE = 17


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--fast65-merged", type=Path, default=DEFAULT_FAST65_MERGED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--historical-fast-start", type=int, default=43)
    parser.add_argument("--historical-fast-end", type=int, default=53)
    parser.add_argument("--quote-volume-quantile", type=float, default=0.80)
    parser.add_argument("--score-tail", type=float, default=0.10)
    parser.add_argument("--density-lookback-seconds", type=int, default=300)
    parser.add_argument("--density-limit", type=int, default=3)
    parser.add_argument("--extension-report", type=Path, default=DEFAULT_FAST56_REPORT)
    parser.add_argument("--sep23-raw-decisions", type=Path, default=DEFAULT_FAST69_RAW_DECISIONS)
    parser.add_argument("--sep23-settlements", type=Path, default=DEFAULT_FAST73_SETTLEMENT_CACHE)
    arguments = parser.parse_args()

    report = run_fast74_non_stop_model_audit(
        data_root=arguments.data_root,
        fast65_merged_path=arguments.fast65_merged,
        output_root=arguments.output_root,
        historical_fast_start=arguments.historical_fast_start,
        historical_fast_end=arguments.historical_fast_end,
        quote_volume_quantile=arguments.quote_volume_quantile,
        score_tail=arguments.score_tail,
        density_lookback_seconds=arguments.density_lookback_seconds,
        density_limit=arguments.density_limit,
        extension_report_path=arguments.extension_report,
        sep23_raw_decisions_path=arguments.sep23_raw_decisions,
        sep23_settlements_path=arguments.sep23_settlements,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast74_non_stop_model_audit(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    output_root: Path,
    historical_fast_start: int = 43,
    historical_fast_end: int = 53,
    quote_volume_quantile: float = 0.80,
    score_tail: float = 0.10,
    density_lookback_seconds: int = 300,
    density_limit: int = 3,
    extension_report_path: Path = DEFAULT_FAST56_REPORT,
    sep23_raw_decisions_path: Path = DEFAULT_FAST69_RAW_DECISIONS,
    sep23_settlements_path: Path = DEFAULT_FAST73_SETTLEMENT_CACHE,
) -> dict[str, Any]:
    validate_parameters(
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        quote_volume_quantile=quote_volume_quantile,
        score_tail=score_tail,
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
    if historical.empty:
        return _write_report(
            output_root,
            _base_report(
                data_root=data_root,
                fast65_merged_path=fast65_merged_path,
                historical_fast_start=historical_fast_start,
                historical_fast_end=historical_fast_end,
                quote_volume_quantile=quote_volume_quantile,
                score_tail=score_tail,
                density_lookback_seconds=density_lookback_seconds,
                density_limit=density_limit,
                extension_report_path=extension_report_path,
                sep23_raw_decisions_path=sep23_raw_decisions_path,
                sep23_settlements_path=sep23_settlements_path,
            )
            | {
                "status": "FAST74_NO_HISTORICAL_TRADES",
                "status_reasons": ["fast43_fast53_selected_trades_unavailable"],
            },
        )

    feature_columns = causal_feature_columns(historical)
    honest_historical = walk_forward_non_stop_model(
        historical,
        feature_columns=feature_columns,
        quote_volume_quantile=quote_volume_quantile,
        score_tail=score_tail,
        density_lookback_seconds=density_lookback_seconds,
        density_limit=density_limit,
        score_threshold_source="training",
    )
    diagnostic_ranked_historical = walk_forward_non_stop_model(
        historical,
        feature_columns=feature_columns,
        quote_volume_quantile=quote_volume_quantile,
        score_tail=score_tail,
        density_lookback_seconds=density_lookback_seconds,
        density_limit=density_limit,
        score_threshold_source="holdout_rank",
    )
    frozen_policy = fit_frozen_policy(
        historical,
        feature_columns=feature_columns,
        quote_volume_quantile=quote_volume_quantile,
        score_tail=score_tail,
    )
    extension = (
        load_selected_trades_with_causal_features((extension_report_path,), data_root=data_root)
        if extension_report_path.exists()
        else pd.DataFrame()
    )
    extension_blocks = evaluate_frozen_policy_blocks(
        extension,
        frozen_policy=frozen_policy,
        all_block_names=(extension_report_path.parent.name,) if extension_report_path.exists() else tuple(),
        density_lookback_seconds=density_lookback_seconds,
        density_limit=density_limit,
    )
    extension_aggregate = aggregate_blocks(extension_blocks)
    sep23 = evaluate_sep23_scoreability(
        raw_decisions_path=sep23_raw_decisions_path,
        settlements_path=sep23_settlements_path,
        required_feature_columns=feature_columns,
    )
    status, status_reasons = fast74_status(
        honest_historical_aggregate=honest_historical["aggregate"],
        extension_aggregate=extension_aggregate,
        sep23=sep23,
    )

    honest_path = output_root / "honest_historical_blocks.json"
    honest_path.write_text(
        json.dumps(honest_historical["blocks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostic_path = output_root / "diagnostic_ranked_historical_blocks.json"
    diagnostic_path.write_text(
        json.dumps(diagnostic_ranked_historical["blocks"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    extension_path = output_root / "extension_blocks.json"
    extension_path.write_text(
        json.dumps(extension_blocks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sep23_path = output_root / "sep23_scoreability.json"
    sep23_path.write_text(json.dumps(sep23, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = _base_report(
        data_root=data_root,
        fast65_merged_path=fast65_merged_path,
        historical_fast_start=historical_fast_start,
        historical_fast_end=historical_fast_end,
        quote_volume_quantile=quote_volume_quantile,
        score_tail=score_tail,
        density_lookback_seconds=density_lookback_seconds,
        density_limit=density_limit,
        extension_report_path=extension_report_path,
        sep23_raw_decisions_path=sep23_raw_decisions_path,
        sep23_settlements_path=sep23_settlements_path,
    ) | {
        "status": status,
        "status_reasons": status_reasons,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "historical_trade_count": int(len(historical)),
        "historical_block_count": int(historical["block"].nunique()),
        "target": "first_touch != STOP",
        "model": frozen_policy["model_summary"],
        "frozen_quote_volume_threshold": frozen_policy["quote_volume_threshold"],
        "frozen_score_threshold": frozen_policy["score_threshold"],
        "honest_historical": honest_historical["aggregate"],
        "honest_historical_blocks_path": str(honest_path),
        "honest_historical_blocks_sha256": _sha256_file(honest_path),
        "diagnostic_ranked_historical": diagnostic_ranked_historical["aggregate"]
        | {
            "executable": False,
            "warning": "uses_holdout_score_distribution_for_top_tail_ranking",
        },
        "diagnostic_ranked_historical_blocks_path": str(diagnostic_path),
        "diagnostic_ranked_historical_blocks_sha256": _sha256_file(diagnostic_path),
        "extension_aggregate": extension_aggregate,
        "extension_blocks_path": str(extension_path),
        "extension_blocks_sha256": _sha256_file(extension_path),
        "sep23": sep23,
        "sep23_scoreability_path": str(sep23_path),
        "sep23_scoreability_sha256": _sha256_file(sep23_path),
    }
    return _write_report(output_root, report)


def validate_parameters(
    *,
    historical_fast_start: int,
    historical_fast_end: int,
    quote_volume_quantile: float,
    score_tail: float,
    density_lookback_seconds: int,
    density_limit: int,
) -> None:
    if historical_fast_start > historical_fast_end:
        raise ValueError("historical_fast_start cannot exceed historical_fast_end")
    if not 0 <= quote_volume_quantile <= 1:
        raise ValueError("quote_volume_quantile must be between 0 and 1")
    if not 0 < score_tail < 1:
        raise ValueError("score_tail must be between 0 and 1")
    if density_lookback_seconds <= 0:
        raise ValueError("density_lookback_seconds must be positive")
    if density_limit < 0:
        raise ValueError("density_limit cannot be negative")


def walk_forward_non_stop_model(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    quote_volume_quantile: float,
    score_tail: float,
    density_lookback_seconds: int,
    density_limit: int,
    score_threshold_source: str,
) -> dict[str, Any]:
    if score_threshold_source not in {"training", "holdout_rank"}:
        raise ValueError(f"unknown score_threshold_source: {score_threshold_source}")
    selected_parts = []
    diagnostics = []
    for holdout_block in sorted(str(value) for value in frame["block"].unique()):
        training = frame[frame["block"].ne(holdout_block)].copy()
        holdout = frame[frame["block"].eq(holdout_block)].copy()
        model = train_non_stop_model(training, feature_columns=feature_columns)
        scored_training = score_frame(model, training, feature_columns=feature_columns)
        scored_holdout = score_frame(model, holdout, feature_columns=feature_columns)
        quote_volume_threshold = float(scored_training["quote_volume_30s"].quantile(quote_volume_quantile))
        if score_threshold_source == "training":
            score_threshold = score_threshold_from_training_scores(
                scored_training[NON_STOP_SCORE_COLUMN],
                score_tail=score_tail,
            )
        else:
            score_threshold = score_threshold_from_training_scores(
                scored_holdout[NON_STOP_SCORE_COLUMN],
                score_tail=score_tail,
            )
        selected = apply_score_volume_density_gate(
            scored_holdout,
            quote_volume_threshold=quote_volume_threshold,
            score_threshold=score_threshold,
            density_lookback_seconds=density_lookback_seconds,
            density_limit=density_limit,
        )
        selected_parts.append(selected)
        diagnostics.append(
            {
                "holdout_block": holdout_block,
                "quote_volume_threshold": quote_volume_threshold,
                "score_threshold": score_threshold,
                "selected_count": int(len(selected)),
            }
        )
    selected_frame = pd.concat(selected_parts, ignore_index=True) if selected_parts else frame.iloc[:0].copy()
    blocks = summarize_dataframe_blocks(
        selected_frame,
        all_block_names=tuple(sorted(str(value) for value in frame["block"].unique())),
    )
    return {
        "aggregate": aggregate_blocks(blocks),
        "blocks": blocks,
        "fold_diagnostics": diagnostics,
    }


def fit_frozen_policy(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    quote_volume_quantile: float,
    score_tail: float,
) -> dict[str, Any]:
    model = train_non_stop_model(frame, feature_columns=feature_columns)
    scored = score_frame(model, frame, feature_columns=feature_columns)
    return {
        "model": model,
        "feature_columns": feature_columns,
        "quote_volume_threshold": float(scored["quote_volume_30s"].quantile(quote_volume_quantile)),
        "score_threshold": score_threshold_from_training_scores(
            scored[NON_STOP_SCORE_COLUMN],
            score_tail=score_tail,
        ),
        "model_summary": {
            "type": "LogisticRegression",
            "target": "first_touch != STOP",
            "feature_count": len(feature_columns),
            "class_weight": "balanced",
            "random_state": MODEL_RANDOM_STATE,
            "score_threshold_source": "historical_training_distribution",
        },
    }


def train_non_stop_model(frame: pd.DataFrame, *, feature_columns: list[str]) -> Pipeline:
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")
    target = (~frame["first_touch"].eq("STOP")).astype(int)
    if target.nunique() < 2:
        raise ValueError("non_stop target needs both classes")
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=MODEL_RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(frame[feature_columns], target)
    return model


def score_frame(model: Pipeline, frame: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        scored = frame.copy()
        scored[NON_STOP_SCORE_COLUMN] = pd.Series(dtype=float)
        return scored
    scored = frame.copy()
    scored[NON_STOP_SCORE_COLUMN] = model.predict_proba(scored[feature_columns])[:, 1]
    return scored


def score_threshold_from_training_scores(scores: pd.Series, *, score_tail: float) -> float:
    finite_scores = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite_scores.empty:
        raise ValueError("scores cannot be empty")
    return float(finite_scores.quantile(1.0 - score_tail))


def apply_score_volume_density_gate(
    frame: pd.DataFrame,
    *,
    quote_volume_threshold: float,
    score_threshold: float,
    density_lookback_seconds: int,
    density_limit: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    filtered = frame[
        (frame["quote_volume_30s"] >= quote_volume_threshold)
        & (frame[NON_STOP_SCORE_COLUMN] >= score_threshold)
    ].copy()
    if filtered.empty:
        return filtered
    return pd.DataFrame(
        density_filter_records(
            filtered.to_dict("records"),
            lookback_seconds=density_lookback_seconds,
            limit=density_limit,
        )
    )


def evaluate_frozen_policy_blocks(
    frame: pd.DataFrame,
    *,
    frozen_policy: dict[str, Any],
    all_block_names: tuple[str, ...],
    density_lookback_seconds: int,
    density_limit: int,
) -> list[dict[str, Any]]:
    if frame.empty:
        return summarize_dataframe_blocks(frame, all_block_names=all_block_names)
    scored = score_frame(
        frozen_policy["model"],
        frame,
        feature_columns=frozen_policy["feature_columns"],
    )
    selected = apply_score_volume_density_gate(
        scored,
        quote_volume_threshold=float(frozen_policy["quote_volume_threshold"]),
        score_threshold=float(frozen_policy["score_threshold"]),
        density_lookback_seconds=density_lookback_seconds,
        density_limit=density_limit,
    )
    return summarize_dataframe_blocks(selected, all_block_names=all_block_names)


def evaluate_sep23_scoreability(
    *,
    raw_decisions_path: Path,
    settlements_path: Path,
    required_feature_columns: list[str],
) -> dict[str, Any]:
    if not raw_decisions_path.exists() or not settlements_path.exists():
        return {
            "available": False,
            "scoreable": False,
            "reasons": ["sep23_raw_decisions_or_settlements_missing"],
        }
    raw_rows = _load_jsonl(raw_decisions_path)
    if not raw_rows:
        return {
            "available": True,
            "scoreable": False,
            "raw_decision_count": 0,
            "reasons": ["sep23_raw_decisions_empty"],
        }
    stored_feature_names = set(raw_rows[0].get("directional_gate_values", {}).keys())
    missing = [feature for feature in required_feature_columns if feature not in stored_feature_names]
    return {
        "available": True,
        "scoreable": not missing,
        "raw_decision_count": len(raw_rows),
        "settlement_count": len(_load_jsonl(settlements_path)),
        "stored_feature_count": len(stored_feature_names),
        "stored_feature_names": sorted(stored_feature_names),
        "required_feature_count": len(required_feature_columns),
        "missing_feature_count": len(missing),
        "missing_feature_examples": missing[:20],
        "reasons": []
        if not missing
        else ["sep23_shadow_decisions_missing_full_causal_feature_vector"],
    }


def fast74_status(
    *,
    honest_historical_aggregate: dict[str, Any],
    extension_aggregate: dict[str, Any],
    sep23: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    if extension_aggregate["negative_sufficient_blocks"]:
        return "FAST74_NON_STOP_MODEL_FAILED_EXTENSION", reasons + [
            f"extension_negative_blocks:{len(extension_aggregate['negative_sufficient_blocks'])}"
        ]
    if float(extension_aggregate["stress_total_log_return_bps"]) <= 0:
        return "FAST74_NON_STOP_MODEL_FAILED_EXTENSION", reasons + [
            "extension_stress_non_positive"
        ]
    if honest_historical_aggregate["negative_sufficient_blocks"]:
        return "FAST74_NON_STOP_MODEL_FAILED_HONEST_HISTORICAL", reasons + [
            f"historical_negative_blocks:{len(honest_historical_aggregate['negative_sufficient_blocks'])}"
        ]
    if sep23.get("available") and not sep23.get("scoreable"):
        return "FAST74_NON_STOP_MODEL_NOT_EXECUTABLE_ON_CURRENT_SHADOW_WAL", reasons + [
            "shadow_wal_missing_full_causal_feature_vector"
        ]
    return "FAST74_NON_STOP_MODEL_RESEARCH_CANDIDATE_REQUIRES_CLEAN_SHADOW", reasons + [
        "requires_clean_future_shadow"
    ]


def _base_report(
    *,
    data_root: Path,
    fast65_merged_path: Path,
    historical_fast_start: int,
    historical_fast_end: int,
    quote_volume_quantile: float,
    score_tail: float,
    density_lookback_seconds: int,
    density_limit: int,
    extension_report_path: Path,
    sep23_raw_decisions_path: Path,
    sep23_settlements_path: Path,
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
            "fast74_is_research_audit",
            "extension_must_pass_before_shadow_candidate",
            "current_wal_must_capture_full_feature_vector_before_executable_shadow",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast74_non_stop_model_audit",
        "data_root": str(data_root),
        "fast65_merged_path": str(fast65_merged_path),
        "historical_fast_start": historical_fast_start,
        "historical_fast_end": historical_fast_end,
        "quote_volume_quantile": quote_volume_quantile,
        "score_tail": score_tail,
        "density_lookback_seconds": density_lookback_seconds,
        "density_limit": density_limit,
        "extension_report_path": str(extension_report_path),
        "sep23_raw_decisions_path": str(sep23_raw_decisions_path),
        "sep23_settlements_path": str(sep23_settlements_path),
    }


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
        "target": report.get("target"),
        "feature_count": report.get("feature_count"),
        "frozen_quote_volume_threshold": report.get("frozen_quote_volume_threshold"),
        "frozen_score_threshold": report.get("frozen_score_threshold"),
        "honest_historical": report.get("honest_historical"),
        "diagnostic_ranked_historical": report.get("diagnostic_ranked_historical"),
        "extension_aggregate": report.get("extension_aggregate"),
        "sep23": report.get("sep23"),
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
