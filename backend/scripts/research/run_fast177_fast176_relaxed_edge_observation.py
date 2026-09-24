"""Fast177 prospective observer for the relaxed Fast176 diagnostic gate.

Fast177 freezes one relaxed edge/regime gate derived from Fast176 and only
counts decisions strictly after its own preregistration timestamp. It is offline
research: no orders, credentials, ledgers, Demo, Testnet or LIVE.
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
from scipy.stats import t

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.run_fast174_crypto_cross_asset_leadlag_smoke import (  # noqa: E402
    DECISION_STRIDE_MINUTES,
    DEFAULT_BTC_TRADE_ROOT,
    DEFAULT_ETH_TRADE_ROOT,
    DEFAULT_SOL_TRADE_ROOT,
    EXPLICIT_COST_BPS,
    SYMBOLS,
    TRAIN_BLOCKS,
    build_all_feature_rows,
    build_date_features,
    leadlag_model,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "backend"
    / "data"
    / "microstructure_cross_asset_v1"
    / "reports"
    / "fast177-fast176-relaxed-edge-observation"
)
DEFAULT_PREREGISTRATION_PATH = (
    REPO_ROOT / "docs" / "forex" / "fast177-fast176-relaxed-edge-prospective-preregistration.md"
)
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "forex" / "fast177-registry.jsonl"

PREREGISTRATION_TIMESTAMP = pd.Timestamp("2026-09-24T10:05:09.742000-03:00")
PREREGISTRATION_SHA256 = "72f78d9b9eda9d3f743a8027fa81460476d4bd8d2f8313a7e2f1f19aa9bd7c47"
TARGET_SYMBOL = "ETHUSDT"
HORIZON_SECONDS = 900
EDGE_THRESHOLD_BPS = 7.5
REGIME_GATE_FEATURE = "ETHUSDT_qv_30m"
REGIME_GATE_THRESHOLD = 100000000.0
MIN_EARLY_FAIL_DECISIONS = 20
EARLY_FAIL_TOTAL_BPS = -100.0
EARLY_FAIL_DAY_BPS = -60.0
MIN_EARLY_FAIL_DAY_DECISIONS = 5
SHADOW_MIN_DECISIONS = 60
SHADOW_MIN_DAYS = 5
SHADOW_MIN_MEAN_BPS = 3.0
SHADOW_MIN_POSITIVE_DAY_SHARE = 0.60
DEFAULT_ALPHA = 0.05


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--btc-trade-root", type=Path, default=DEFAULT_BTC_TRADE_ROOT)
    parser.add_argument("--eth-trade-root", type=Path, default=DEFAULT_ETH_TRADE_ROOT)
    parser.add_argument("--sol-trade-root", type=Path, default=DEFAULT_SOL_TRADE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION_PATH)
    parser.add_argument("--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args(argv)
    report = run_fast177_fast176_relaxed_edge_observation(
        trade_roots={
            "BTCUSDT": arguments.btc_trade_root,
            "ETHUSDT": arguments.eth_trade_root,
            "SOLUSDT": arguments.sol_trade_root,
        },
        output_root=arguments.output_root,
        preregistration_path=arguments.preregistration_path,
        registry_path=arguments.registry_path,
        alpha=arguments.alpha,
    )
    print(json.dumps(console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast177_fast176_relaxed_edge_observation(
    *,
    trade_roots: dict[str, Path],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION_PATH,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    validate_parameters(alpha)
    validate_preregistration(preregistration_path=preregistration_path, registry_path=registry_path)
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows = build_all_feature_rows(trade_roots)
    feature_columns = feature_columns_from_rows(train_rows)
    model_training_rows = build_training_label_rows(train_rows)
    model = leadlag_model()
    model.fit(model_training_rows[feature_columns], model_training_rows["actual_bps"])

    prospective_rows, source_summary = build_prospective_rows(trade_roots, feature_columns)
    all_scored_rows = score_rows(model, prospective_rows, feature_columns)
    scored_rows = filter_eligible_rows(all_scored_rows)
    observation_records = [observation_record(row) for _, row in scored_rows.iterrows()]
    prospective_diagnostic_records = [
        prospective_diagnostic_record(row) for _, row in all_scored_rows.iterrows()
    ]

    observation_rows_path = output_root / "observation-decision-rows.parquet"
    observations_path = output_root / "observations.jsonl"
    prospective_diagnostic_rows_path = output_root / "prospective-diagnostic-rows.jsonl"
    prospective_diagnostics_path = output_root / "prospective-diagnostics.json"
    summary_path = output_root / "summary.json"
    scored_rows.to_parquet(observation_rows_path, compression="zstd", index=False)
    write_jsonl(observations_path, observation_records)
    write_jsonl(prospective_diagnostic_rows_path, prospective_diagnostic_records)

    prospective_diagnostics = summarize_prospective_diagnostics(all_scored_rows)
    write_json(prospective_diagnostics_path, prospective_diagnostics)
    observation_summary = summarize_observations(scored_rows, alpha=alpha)
    gates = evaluate_gates(scored_rows, observation_summary)
    status, status_reasons = fast177_status(
        prospective_rows=prospective_rows,
        observations=scored_rows,
        gates=gates,
    )
    summary = {
        "status": status,
        "status_reasons": status_reasons,
        "preregistration_timestamp": PREREGISTRATION_TIMESTAMP.isoformat(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "target_symbol": TARGET_SYMBOL,
        "horizon_seconds": HORIZON_SECONDS,
        "decision_stride_minutes": DECISION_STRIDE_MINUTES,
        "edge_threshold_bps": EDGE_THRESHOLD_BPS,
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
        "explicit_cost_bps": EXPLICIT_COST_BPS,
        "source_summary": source_summary,
        "feature_count": len(feature_columns),
        "training_row_count": int(len(model_training_rows)),
        "prospective_row_count": int(len(prospective_rows)),
        "eligible_observation_count": int(len(scored_rows)),
        "prospective_diagnostics": prospective_diagnostics,
        "observation_summary": observation_summary,
        "gates": gates,
    }
    write_json(summary_path, summary)
    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast177_is_preregistered_prospective_observation_only",
            "relaxed_fast176_gate_can_only_count_post_preregistration_decisions",
            "relaxed_edge_threshold_and_regime_gate_are_frozen",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast177_fast176_relaxed_edge_observation",
        "status": status,
        "status_reasons": status_reasons,
        "summary": summary,
        "artifacts": {
            "observation_decision_rows": {
                "path": str(observation_rows_path),
                "sha256": sha256_file(observation_rows_path),
            },
            "observations": {"path": str(observations_path), "sha256": sha256_file(observations_path)},
            "prospective_diagnostic_rows": {
                "path": str(prospective_diagnostic_rows_path),
                "sha256": sha256_file(prospective_diagnostic_rows_path),
            },
            "prospective_diagnostics": {
                "path": str(prospective_diagnostics_path),
                "sha256": sha256_file(prospective_diagnostics_path),
            },
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        },
    }
    report_path = output_root / "report.json"
    write_json(report_path, report)
    return {**report, "report_path": str(report_path), "report_sha256": sha256_file(report_path)}


def validate_parameters(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")


def validate_preregistration(*, preregistration_path: Path, registry_path: Path) -> None:
    actual_sha = sha256_file(preregistration_path)
    if actual_sha != PREREGISTRATION_SHA256:
        raise ValueError(
            f"Fast177 preregistration hash mismatch: expected {PREREGISTRATION_SHA256}, got {actual_sha}"
        )
    if not registry_path.exists():
        raise FileNotFoundError(f"missing Fast177 registry: {registry_path}")
    events = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    preregistered = [
        event
        for event in events
        if event.get("event") == "fast177_fast176_relaxed_edge_prospective_preregistered"
    ]
    if not preregistered:
        raise ValueError("Fast177 preregistration event missing from registry")
    if preregistered[-1].get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ValueError("Fast177 registry preregistration hash mismatch")


def feature_columns_from_rows(rows: pd.DataFrame) -> list[str]:
    return [
        column
        for column in rows.columns
        if any(token in column for token in ("_ret_", "_vol_", "_flow_", "_qv_"))
    ] + ["hour_sin", "hour_cos"]


def build_training_label_rows(rows: pd.DataFrame) -> pd.DataFrame:
    shift_rows = int(HORIZON_SECONDS / 60 / DECISION_STRIDE_MINUTES)
    training = rows.loc[rows["block"].isin(TRAIN_BLOCKS)].copy()
    training["future_price"] = training.groupby("date")[f"{TARGET_SYMBOL}_price"].shift(-shift_rows)
    training["actual_bps"] = (training["future_price"] / training[f"{TARGET_SYMBOL}_price"] - 1.0) * 10000.0
    return training.dropna(subset=["actual_bps"])


def build_prospective_rows(
    trade_roots: dict[str, Path],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    available_dates = discover_available_dates(trade_roots)
    frames = []
    date_summaries = []
    for utc_date in available_dates:
        frame = build_date_features(trade_roots, utc_date)
        frame["date"] = utc_date
        frame["block"] = "prospective"
        frames.append(frame)
        date_summaries.append(
            {
                "date": utc_date,
                "row_count": int(len(frame)),
                "first_decision_time": None if frame.empty else frame["decision_time"].min().isoformat(),
                "last_decision_time": None if frame.empty else frame["decision_time"].max().isoformat(),
            }
        )
    if not frames:
        empty = pd.DataFrame(columns=["decision_time", "date", *feature_columns])
        return empty, {"available_dates": [], "date_summaries": []}
    rows = pd.concat(frames, ignore_index=True).sort_values("decision_time")
    rows = rows.loc[rows["minute_sequence"].mod(DECISION_STRIDE_MINUTES).eq(0)].copy()
    rows["future_price"] = rows.groupby("date")[f"{TARGET_SYMBOL}_price"].shift(
        -int(HORIZON_SECONDS / 60 / DECISION_STRIDE_MINUTES)
    )
    rows["actual_bps"] = (rows["future_price"] / rows[f"{TARGET_SYMBOL}_price"] - 1.0) * 10000.0
    rows = rows.loc[rows["decision_time"] > PREREGISTRATION_TIMESTAMP].dropna(subset=["actual_bps"])
    return rows.reset_index(drop=True), {
        "available_dates": available_dates,
        "date_summaries": date_summaries,
        "latest_available_date": None if not available_dates else max(available_dates),
    }


def discover_available_dates(trade_roots: dict[str, Path]) -> list[str]:
    date_sets = []
    for symbol in SYMBOLS:
        root = trade_roots[symbol]
        date_sets.append({path.name.removeprefix("date=") for path in root.glob("date=*") if path.is_dir()})
    if not date_sets:
        return []
    return sorted(set.intersection(*date_sets))


def score_rows(model: Any, rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if rows.empty:
        return rows.assign(
            prediction=pd.Series(dtype=float),
            edge=pd.Series(dtype=float),
            side=pd.Series(dtype=float),
            side_label=pd.Series(dtype=str),
            net_bps=pd.Series(dtype=float),
            edge_gate_passed=pd.Series(dtype=bool),
            regime_gate_passed=pd.Series(dtype=bool),
            eligible=pd.Series(dtype=bool),
        )
    scored = rows.copy()
    scored["prediction"] = model.predict(scored[feature_columns])
    scored["edge"] = np.abs(scored["prediction"])
    scored["side"] = np.sign(scored["prediction"])
    scored["side_label"] = np.select(
        [scored["side"] > 0, scored["side"] < 0],
        ["LONG", "SHORT"],
        default="FLAT",
    )
    scored["net_bps"] = np.where(
        scored["side"].ne(0),
        scored["actual_bps"] * scored["side"] - EXPLICIT_COST_BPS,
        0.0,
    )
    scored["edge_gate_passed"] = scored["side"].ne(0) & (scored["edge"] >= EDGE_THRESHOLD_BPS)
    scored["regime_gate_passed"] = scored[REGIME_GATE_FEATURE] >= REGIME_GATE_THRESHOLD
    scored["eligible"] = scored["edge_gate_passed"] & scored["regime_gate_passed"]
    return scored.reset_index(drop=True)


def filter_eligible_rows(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    return scored.loc[scored["eligible"]].copy().reset_index(drop=True)


def score_and_filter_rows(model: Any, rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return filter_eligible_rows(score_rows(model, rows, feature_columns))


def summarize_prospective_diagnostics(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty:
        return {
            "row_count": 0,
            "edge_gate_pass_count": 0,
            "regime_gate_pass_count": 0,
            "eligible_count": 0,
            "max_edge_bps": 0.0,
            "max_regime_gate_value": 0.0,
            "latest_decision_time": None,
            "latest_prediction_bps": None,
            "latest_edge_bps": None,
            "latest_regime_gate_value": None,
            "latest_edge_gate_passed": False,
            "latest_regime_gate_passed": False,
            "latest_eligible": False,
        }
    latest = scored.sort_values("decision_time").iloc[-1]
    return {
        "row_count": int(len(scored)),
        "edge_gate_pass_count": int(scored["edge_gate_passed"].sum()),
        "regime_gate_pass_count": int(scored["regime_gate_passed"].sum()),
        "eligible_count": int(scored["eligible"].sum()),
        "max_edge_bps": float(scored["edge"].max()),
        "max_regime_gate_value": float(scored[REGIME_GATE_FEATURE].max()),
        "latest_decision_time": latest["decision_time"].isoformat(),
        "latest_prediction_bps": float(latest["prediction"]),
        "latest_edge_bps": float(latest["edge"]),
        "latest_regime_gate_value": float(latest[REGIME_GATE_FEATURE]),
        "latest_edge_gate_passed": bool(latest["edge_gate_passed"]),
        "latest_regime_gate_passed": bool(latest["regime_gate_passed"]),
        "latest_eligible": bool(latest["eligible"]),
    }


def summarize_observations(observations: pd.DataFrame, *, alpha: float) -> dict[str, Any]:
    if observations.empty:
        return {
            "decision_count": 0,
            "day_count": 0,
            "net_total_bps": 0.0,
            "net_mean_bps": 0.0,
            "net_lcb_bps": 0.0,
            "positive_day_share": 0.0,
            "sufficient_day_count": 0,
            "worst_sufficient_day_net_bps": 0.0,
        }
    net = observations["net_bps"].astype(float)
    day_summary = observations.groupby("date", sort=True).agg(
        trade_count=("net_bps", "size"),
        net_total_bps=("net_bps", "sum"),
    )
    sufficient_days = day_summary.loc[day_summary["trade_count"] >= MIN_EARLY_FAIL_DAY_DECISIONS]
    return {
        "decision_count": int(len(observations)),
        "day_count": int(observations["date"].nunique()),
        "net_total_bps": float(net.sum()),
        "net_mean_bps": float(net.mean()),
        "net_lcb_bps": lower_confidence_bound(net, alpha=alpha),
        "positive_day_share": (
            0.0 if sufficient_days.empty else float((sufficient_days["net_total_bps"] > 0).mean())
        ),
        "sufficient_day_count": int(len(sufficient_days)),
        "worst_sufficient_day_net_bps": (
            0.0 if sufficient_days.empty else float(sufficient_days["net_total_bps"].min())
        ),
    }


def evaluate_gates(observations: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    early_fail_reasons = []
    if summary["decision_count"] >= MIN_EARLY_FAIL_DECISIONS and summary["net_total_bps"] <= EARLY_FAIL_TOTAL_BPS:
        early_fail_reasons.append("net_total_below_early_fail_threshold")
    if (
        summary["sufficient_day_count"] > 0
        and summary["worst_sufficient_day_net_bps"] <= EARLY_FAIL_DAY_BPS
    ):
        early_fail_reasons.append("sufficient_day_net_below_early_fail_threshold")
    shadow_reasons = []
    if summary["decision_count"] < SHADOW_MIN_DECISIONS:
        shadow_reasons.append("insufficient_decisions")
    if summary["day_count"] < SHADOW_MIN_DAYS:
        shadow_reasons.append("insufficient_days")
    if summary["net_mean_bps"] <= SHADOW_MIN_MEAN_BPS:
        shadow_reasons.append("mean_bps_below_threshold")
    if summary["net_total_bps"] <= 0:
        shadow_reasons.append("net_total_not_positive")
    if summary["positive_day_share"] < SHADOW_MIN_POSITIVE_DAY_SHARE:
        shadow_reasons.append("positive_day_share_below_threshold")
    if summary["net_lcb_bps"] <= 0:
        shadow_reasons.append("lcb_not_positive")
    if summary["worst_sufficient_day_net_bps"] <= EARLY_FAIL_DAY_BPS and summary["sufficient_day_count"] > 0:
        shadow_reasons.append("sufficient_day_breached_loss_limit")
    return {
        "early_fail": bool(early_fail_reasons),
        "early_fail_reasons": early_fail_reasons,
        "shadow_candidate": not shadow_reasons,
        "shadow_reasons": shadow_reasons,
    }


def fast177_status(
    *,
    prospective_rows: pd.DataFrame,
    observations: pd.DataFrame,
    gates: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution", "prospective_observation"]
    if prospective_rows.empty:
        return (
            "FAST177_WAITING_FOR_POST_PREREGISTRATION_DATA",
            reasons + ["no_post_preregistration_decision_rows"],
        )
    if observations.empty:
        return (
            "FAST177_DATA_FRESH_NO_ELIGIBLE_DECISIONS_YET",
            reasons + ["edge_or_regime_gate_not_met"],
        )
    if gates["early_fail"]:
        return (
            "FAST177_OBSERVATION_FAILED_EARLY",
            reasons + gates["early_fail_reasons"],
        )
    if gates["shadow_candidate"]:
        return (
            "FAST177_OBSERVATION_SHADOW_OFFLINE_CANDIDATE_REQUIRES_REVIEW",
            reasons + ["shadow_candidate_gate_passed", "still_not_approved_strategy"],
        )
    return (
        "FAST177_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE",
        reasons + gates["shadow_reasons"],
    )


def lower_confidence_bound(values: pd.Series, *, alpha: float) -> float:
    sample = values.dropna().astype(float)
    if len(sample) == 0:
        return 0.0
    if len(sample) == 1:
        return float(sample.iloc[0])
    std = float(sample.std(ddof=1))
    if std == 0:
        return float(sample.mean())
    critical = float(t.ppf(1 - alpha, df=len(sample) - 1))
    return float(sample.mean() - critical * std / np.sqrt(len(sample)))


def observation_record(row: pd.Series) -> dict[str, Any]:
    return {
        "decision_time": row["decision_time"].isoformat(),
        "date": row["date"],
        "target_symbol": TARGET_SYMBOL,
        "side": row["side_label"],
        "prediction": float(row["prediction"]),
        "edge": float(row["edge"]),
        "actual_bps": float(row["actual_bps"]),
        "net_bps": float(row["net_bps"]),
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_value": float(row[REGIME_GATE_FEATURE]),
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
    }


def prospective_diagnostic_record(row: pd.Series) -> dict[str, Any]:
    return {
        "decision_time": row["decision_time"].isoformat(),
        "date": row["date"],
        "target_symbol": TARGET_SYMBOL,
        "side": row["side_label"],
        "prediction": float(row["prediction"]),
        "edge": float(row["edge"]),
        "edge_threshold": EDGE_THRESHOLD_BPS,
        "edge_gate_passed": bool(row["edge_gate_passed"]),
        "actual_bps": float(row["actual_bps"]),
        "net_bps_if_eligible": float(row["net_bps"]),
        "regime_gate_feature": REGIME_GATE_FEATURE,
        "regime_gate_value": float(row[REGIME_GATE_FEATURE]),
        "regime_gate_threshold": REGIME_GATE_THRESHOLD,
        "regime_gate_passed": bool(row["regime_gate_passed"]),
        "eligible": bool(row["eligible"]),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "research_only": report["research_only"],
        "approved_strategy": report["approved_strategy"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "orders_sent": report["orders_sent"],
        "status": report["status"],
        "status_reasons": report["status_reasons"],
        "latest_available_date": summary["source_summary"].get("latest_available_date"),
        "prospective_row_count": summary["prospective_row_count"],
        "eligible_observation_count": summary["eligible_observation_count"],
        "observation_summary": summary["observation_summary"],
        "report_path": report["report_path"],
        "report_sha256": report["report_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
