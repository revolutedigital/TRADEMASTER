"""Fast77 temporal stability audit for Fast76 weak calendar survivors.

Fast76 found pair-specific weak survivors in the official macro-calendar FX
artifact, but none cleared a corrected lower confidence bound. Fast77 asks a
stricter operational question before any preregistration: did those weak
survivors stay positive year-by-year in the 2024-2026 audit window, or was the
aggregate result carried by one transient regime?

This remains research-only. A temporally stable weak survivor is not an
approved strategy; it can only justify a frozen, preregistered observation
stream.
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


FAST76_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast13" / "fast76-calendar-pair-specific-audit"
DEFAULT_TRADES_PATH = REPO_ROOT / "backend" / "data" / "lab_fast13" / "calendar-m1-trades.parquet"
DEFAULT_PAIR_POLICY_CANDIDATES_PATH = FAST76_OUTPUT_ROOT / "pair_policy_candidates.json"
DEFAULT_PAIR_EVENT_POLICY_CANDIDATES_PATH = FAST76_OUTPUT_ROOT / "pair_event_policy_candidates.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast13" / "fast77-calendar-temporal-stability-audit"
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
AUDIT_END = pd.Timestamp("2026-09-01T00:00:00Z")
DEFAULT_ALPHA = 0.05
DEFAULT_MINIMUM_AUDIT_YEAR_TRADES = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument(
        "--pair-policy-candidates-path",
        type=Path,
        default=DEFAULT_PAIR_POLICY_CANDIDATES_PATH,
    )
    parser.add_argument(
        "--pair-event-policy-candidates-path",
        type=Path,
        default=DEFAULT_PAIR_EVENT_POLICY_CANDIDATES_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--minimum-audit-year-trades",
        type=int,
        default=DEFAULT_MINIMUM_AUDIT_YEAR_TRADES,
    )
    arguments = parser.parse_args()
    report = run_fast77_calendar_temporal_stability_audit(
        trades_path=arguments.trades_path,
        pair_policy_candidates_path=arguments.pair_policy_candidates_path,
        pair_event_policy_candidates_path=arguments.pair_event_policy_candidates_path,
        output_root=arguments.output_root,
        alpha=arguments.alpha,
        minimum_audit_year_trades=arguments.minimum_audit_year_trades,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast77_calendar_temporal_stability_audit(
    *,
    trades_path: Path = DEFAULT_TRADES_PATH,
    pair_policy_candidates_path: Path = DEFAULT_PAIR_POLICY_CANDIDATES_PATH,
    pair_event_policy_candidates_path: Path = DEFAULT_PAIR_EVENT_POLICY_CANDIDATES_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    alpha: float = DEFAULT_ALPHA,
    minimum_audit_year_trades: int = DEFAULT_MINIMUM_AUDIT_YEAR_TRADES,
) -> dict[str, Any]:
    validate_parameters(alpha=alpha, minimum_audit_year_trades=minimum_audit_year_trades)
    output_root.mkdir(parents=True, exist_ok=True)
    trades = load_trades(trades_path)
    pair_policy_payload = load_candidate_payload(pair_policy_candidates_path)
    pair_event_policy_payload = load_candidate_payload(pair_event_policy_candidates_path)

    pair_policy = audit_scope(
        trades,
        candidates=pair_policy_payload["weak_survivors"],
        scope_name="pair_policy",
        group_columns=("pair", "policy_id"),
        alpha=alpha,
        minimum_audit_year_trades=minimum_audit_year_trades,
    )
    pair_event_policy = audit_scope(
        trades,
        candidates=pair_event_policy_payload["weak_survivors"],
        scope_name="pair_event_policy",
        group_columns=("pair", "event_name", "policy_id"),
        alpha=alpha,
        minimum_audit_year_trades=minimum_audit_year_trades,
    )
    status, status_reasons = fast77_status(
        pair_policy=pair_policy,
        pair_event_policy=pair_event_policy,
    )

    pair_policy_path = output_root / "pair_policy_temporal_candidates.json"
    pair_policy_path.write_text(json.dumps(pair_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pair_event_policy_path = output_root / "pair_event_policy_temporal_candidates.json"
    pair_event_policy_path.write_text(
        json.dumps(pair_event_policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    temporal_survivors = sorted(
        [
            *pair_policy["temporal_weak_survivors"],
            *pair_event_policy["temporal_weak_survivors"],
        ],
        key=lambda candidate: (
            candidate["audit"]["mean_stress_pips"],
            candidate["audit"]["total_stress_pips"],
            candidate["audit"]["trades"],
        ),
        reverse=True,
    )
    confirmed_candidates = sorted(
        [
            *pair_policy["confirmed_temporal_candidates"],
            *pair_event_policy["confirmed_temporal_candidates"],
        ],
        key=lambda candidate: (
            candidate["audit"]["lower_stress_pips_uncorrected"],
            candidate["audit"]["mean_stress_pips"],
            candidate["audit"]["trades"],
        ),
        reverse=True,
    )

    report = {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "committed": False,
        "dry_run": True,
        "approved_strategy": False,
        "approval_reasons": [
            "fast77_audits_fast76_weak_survivors_only",
            "temporal_weak_survivor_is_not_strategy",
            "lower_confidence_bounds_remain_required_before_shadow",
            "future_preregistered_observation_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast77_calendar_temporal_stability_audit",
        "status": status,
        "status_reasons": status_reasons,
        "trades_path": str(trades_path),
        "trades_sha256": _sha256_file(trades_path),
        "pair_policy_candidates_path": str(pair_policy_candidates_path),
        "pair_policy_candidates_sha256": _sha256_file(pair_policy_candidates_path),
        "pair_event_policy_candidates_path": str(pair_event_policy_candidates_path),
        "pair_event_policy_candidates_sha256": _sha256_file(pair_event_policy_candidates_path),
        "trade_count": int(len(trades)),
        "selection_end": SELECTION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "alpha": alpha,
        "minimum_audit_year_trades": minimum_audit_year_trades,
        "scopes": {
            "pair_policy": pair_policy["summary"],
            "pair_event_policy": pair_event_policy["summary"],
        },
        "pair_policy_temporal_candidates_path": str(pair_policy_path),
        "pair_policy_temporal_candidates_sha256": _sha256_file(pair_policy_path),
        "pair_event_policy_temporal_candidates_path": str(pair_event_policy_path),
        "pair_event_policy_temporal_candidates_sha256": _sha256_file(pair_event_policy_path),
        "top_temporal_weak_survivors": temporal_survivors[:10],
        "top_confirmed_temporal_candidates": confirmed_candidates[:10],
    }
    return _write_report(output_root, report)


def validate_parameters(*, alpha: float, minimum_audit_year_trades: int) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if minimum_audit_year_trades < 1:
        raise ValueError("minimum_audit_year_trades must be positive")


def load_candidate_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "weak_survivors" not in payload:
        raise ValueError(f"{path} missing weak_survivors")
    return payload


def load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    trades = pd.read_parquet(path)
    required = {
        "event_timestamp",
        "pair",
        "event_name",
        "policy_id",
        "net_base_pips",
        "net_stress_pips",
    }
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing columns: {sorted(missing)}")
    trades = trades.copy()
    trades["event_timestamp"] = pd.to_datetime(trades["event_timestamp"], utc=True)
    return trades.loc[trades["event_timestamp"] < AUDIT_END].copy()


def audit_scope(
    trades: pd.DataFrame,
    *,
    candidates: list[dict[str, Any]],
    scope_name: str,
    group_columns: tuple[str, ...],
    alpha: float,
    minimum_audit_year_trades: int,
) -> dict[str, Any]:
    corrected_alpha = alpha / max(1, len(candidates))
    audited_candidates = [
        audit_candidate(
            trades,
            candidate=candidate,
            scope_name=scope_name,
            group_columns=group_columns,
            alpha=alpha,
            corrected_alpha=corrected_alpha,
            minimum_audit_year_trades=minimum_audit_year_trades,
        )
        for candidate in candidates
    ]
    temporal_weak_survivors = [
        candidate
        for candidate in audited_candidates
        if candidate["temporal_flags"]["temporal_weak_survivor"]
    ]
    temporal_weak_survivors.sort(
        key=lambda candidate: (
            candidate["audit"]["mean_stress_pips"],
            candidate["audit"]["total_stress_pips"],
            candidate["audit"]["trades"],
        ),
        reverse=True,
    )
    confirmed_temporal_candidates = [
        candidate
        for candidate in temporal_weak_survivors
        if candidate["temporal_flags"]["confirmed_temporal_candidate"]
    ]
    confirmed_temporal_candidates.sort(
        key=lambda candidate: (
            candidate["audit"]["lower_stress_pips_uncorrected"],
            candidate["audit"]["mean_stress_pips"],
            candidate["audit"]["trades"],
        ),
        reverse=True,
    )
    return {
        "summary": {
            "scope_name": scope_name,
            "group_columns": list(group_columns),
            "fast76_weak_survivor_input_count": len(candidates),
            "temporal_weak_survivor_count": len(temporal_weak_survivors),
            "confirmed_temporal_candidate_count": len(confirmed_temporal_candidates),
            "minimum_audit_year_trades": minimum_audit_year_trades,
            "alpha": alpha,
            "bonferroni_alpha": corrected_alpha,
            "recent_year_failed_count": sum(
                not candidate["temporal_flags"]["recent_audit_year_positive_stress"]
                for candidate in audited_candidates
            ),
            "all_audit_years_positive_stress_count": sum(
                candidate["temporal_flags"]["all_audit_years_positive_stress"]
                for candidate in audited_candidates
            ),
        },
        "audited_candidates": audited_candidates,
        "temporal_weak_survivors": temporal_weak_survivors,
        "confirmed_temporal_candidates": confirmed_temporal_candidates,
    }


def audit_candidate(
    trades: pd.DataFrame,
    *,
    candidate: dict[str, Any],
    scope_name: str,
    group_columns: tuple[str, ...],
    alpha: float,
    corrected_alpha: float,
    minimum_audit_year_trades: int,
) -> dict[str, Any]:
    key = tuple(candidate[column] for column in group_columns)
    candidate_trades = filter_group(trades, group_columns=group_columns, key=key)
    selection = candidate_trades.loc[candidate_trades["event_timestamp"] < SELECTION_END]
    audit = candidate_trades.loc[
        (candidate_trades["event_timestamp"] >= SELECTION_END)
        & (candidate_trades["event_timestamp"] < AUDIT_END)
    ]
    public_candidate = {
        "scope": scope_name,
        **dict(zip(group_columns, key, strict=True)),
        "selection": summarize_period(selection, alpha=alpha, corrected_alpha=corrected_alpha),
        "audit": summarize_period(audit, alpha=alpha, corrected_alpha=corrected_alpha),
        "selection_years": yearly_summaries(selection, alpha=alpha, corrected_alpha=corrected_alpha),
        "audit_years": yearly_summaries(audit, alpha=alpha, corrected_alpha=corrected_alpha),
    }
    public_candidate["temporal_flags"] = temporal_flags(
        public_candidate,
        minimum_audit_year_trades=minimum_audit_year_trades,
    )
    return public_candidate


def filter_group(frame: pd.DataFrame, *, group_columns: tuple[str, ...], key: tuple[Any, ...]) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in zip(group_columns, key, strict=True):
        mask &= frame[column].eq(value).to_numpy()
    return frame.loc[mask]


def yearly_summaries(
    frame: pd.DataFrame,
    *,
    alpha: float,
    corrected_alpha: float,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    yearly = frame.assign(year=frame["event_timestamp"].dt.year)
    return [
        {"year": int(year), **summarize_period(group, alpha=alpha, corrected_alpha=corrected_alpha)}
        for year, group in yearly.groupby("year", sort=True)
    ]


def summarize_period(
    frame: pd.DataFrame,
    *,
    alpha: float,
    corrected_alpha: float,
) -> dict[str, Any]:
    base_values = frame["net_base_pips"].astype(float)
    stress_values = frame["net_stress_pips"].astype(float)
    return {
        "trades": int(len(frame)),
        "events": int(frame["event_timestamp"].nunique()) if "event_timestamp" in frame else 0,
        "mean_base_pips": safe_mean(base_values),
        "mean_stress_pips": safe_mean(stress_values),
        "total_stress_pips": safe_sum(stress_values),
        "median_stress_pips": safe_median(stress_values),
        "lower_base_pips_uncorrected": lower_confidence_bound(base_values, alpha=alpha),
        "lower_stress_pips_uncorrected": lower_confidence_bound(stress_values, alpha=alpha),
        "lower_base_pips_bonferroni": lower_confidence_bound(base_values, alpha=corrected_alpha),
        "lower_stress_pips_bonferroni": lower_confidence_bound(stress_values, alpha=corrected_alpha),
        "event_name_counts": event_name_counts(frame),
    }


def temporal_flags(
    candidate: dict[str, Any],
    *,
    minimum_audit_year_trades: int,
) -> dict[str, Any]:
    audit_years = candidate["audit_years"]
    selection_years = candidate["selection_years"]
    recent_audit_year = max((year["year"] for year in audit_years), default=None)
    recent_summary = next(
        (year for year in audit_years if year["year"] == recent_audit_year),
        None,
    )
    audit_year_count = len(audit_years)
    all_audit_years_have_minimum_trades = bool(audit_years) and all(
        year["trades"] >= minimum_audit_year_trades for year in audit_years
    )
    all_audit_years_positive_stress = bool(audit_years) and all(
        year["mean_stress_pips"] > 0 for year in audit_years
    )
    all_audit_years_positive_base = bool(audit_years) and all(
        year["mean_base_pips"] > 0 for year in audit_years
    )
    recent_audit_year_positive_stress = bool(
        recent_summary and recent_summary["mean_stress_pips"] > 0
    )
    recent_audit_year_positive_base = bool(recent_summary and recent_summary["mean_base_pips"] > 0)
    selection_positive_stress_year_count = sum(
        year["mean_stress_pips"] > 0 for year in selection_years
    )
    selection_year_count = len(selection_years)
    selection_stress_positive_year_rate = (
        selection_positive_stress_year_count / selection_year_count if selection_year_count else 0.0
    )
    aggregate_positive = (
        candidate["selection"]["mean_base_pips"] > 0
        and candidate["selection"]["mean_stress_pips"] > 0
        and candidate["audit"]["mean_base_pips"] > 0
        and candidate["audit"]["mean_stress_pips"] > 0
    )
    temporal_weak_survivor = (
        audit_year_count >= 3
        and all_audit_years_have_minimum_trades
        and all_audit_years_positive_stress
        and all_audit_years_positive_base
        and recent_audit_year_positive_stress
        and recent_audit_year_positive_base
        and aggregate_positive
    )
    confirmed_temporal_candidate = (
        temporal_weak_survivor
        and candidate["audit"]["lower_base_pips_uncorrected"] > 0
        and candidate["audit"]["lower_stress_pips_uncorrected"] > 0
        and candidate["audit"]["lower_base_pips_bonferroni"] > 0
        and candidate["audit"]["lower_stress_pips_bonferroni"] > 0
    )
    return {
        "audit_year_count": audit_year_count,
        "recent_audit_year": recent_audit_year,
        "all_audit_years_have_minimum_trades": all_audit_years_have_minimum_trades,
        "all_audit_years_positive_base": all_audit_years_positive_base,
        "all_audit_years_positive_stress": all_audit_years_positive_stress,
        "recent_audit_year_positive_base": recent_audit_year_positive_base,
        "recent_audit_year_positive_stress": recent_audit_year_positive_stress,
        "selection_positive_stress_year_count": selection_positive_stress_year_count,
        "selection_year_count": selection_year_count,
        "selection_stress_positive_year_rate": selection_stress_positive_year_rate,
        "aggregate_positive": aggregate_positive,
        "audit_lower_base_uncorrected_positive": candidate["audit"]["lower_base_pips_uncorrected"] > 0,
        "audit_lower_stress_uncorrected_positive": (
            candidate["audit"]["lower_stress_pips_uncorrected"] > 0
        ),
        "audit_lower_base_bonferroni_positive": candidate["audit"]["lower_base_pips_bonferroni"] > 0,
        "audit_lower_stress_bonferroni_positive": (
            candidate["audit"]["lower_stress_pips_bonferroni"] > 0
        ),
        "temporal_weak_survivor": temporal_weak_survivor,
        "confirmed_temporal_candidate": confirmed_temporal_candidate,
    }


def event_name_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "event_name" not in frame:
        return {}
    return {str(key): int(value) for key, value in frame["event_name"].value_counts().to_dict().items()}


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def safe_sum(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.sum())


def safe_median(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.median())


def lower_confidence_bound(values: list[float] | pd.Series | np.ndarray, *, alpha: float) -> float:
    series = pd.Series(values, dtype="float64").dropna()
    sample_size = len(series)
    if sample_size < 2:
        return float("nan")
    standard_error = float(series.std(ddof=1) / np.sqrt(sample_size))
    return float(series.mean() - t.ppf(1.0 - alpha, sample_size - 1) * standard_error)


def fast77_status(
    *,
    pair_policy: dict[str, Any],
    pair_event_policy: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    confirmed = (
        pair_policy["summary"]["confirmed_temporal_candidate_count"]
        + pair_event_policy["summary"]["confirmed_temporal_candidate_count"]
    )
    temporal_weak = (
        pair_policy["summary"]["temporal_weak_survivor_count"]
        + pair_event_policy["summary"]["temporal_weak_survivor_count"]
    )
    if confirmed:
        return "FAST77_CALENDAR_TEMPORAL_STABILITY_CANDIDATE_REQUIRES_SHADOW", reasons + [
            f"confirmed_temporal_candidates:{confirmed}",
            "future_preregistered_shadow_required",
        ]
    if temporal_weak:
        return (
            "FAST77_CALENDAR_TEMPORAL_STABILITY_WEAK_SURVIVOR_REQUIRES_PREREGISTRATION",
            reasons
            + [
                f"temporal_weak_survivors:{temporal_weak}",
                "confidence_bounds_non_positive",
                "preregistered_observation_only",
            ],
        )
    return "FAST77_CALENDAR_TEMPORAL_STABILITY_FAILED_RECENT_STABILITY", reasons + [
        "no_temporal_weak_survivor",
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
        "scopes": report["scopes"],
        "top_temporal_weak_survivors": report["top_temporal_weak_survivors"][:5],
        "top_confirmed_temporal_candidates": report["top_confirmed_temporal_candidates"][:5],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
