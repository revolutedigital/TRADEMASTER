"""Fast76 pair-specific official-calendar FX audit.

Fast13 rejected the official macro-calendar family when policies were aggregated
across ten FX pairs. Fast76 asks a narrower question: did Fast13 hide a
pair-specific or pair/event-specific survivor that was positive in the original
selection period and replicated in the 2024-2026 audit?

This remains research-only. It reuses the committed Fast13 trade artifact,
performs no market access, and never approves a strategy from a positive mean
alone. A candidate must clear a Bonferroni-corrected lower confidence bound.
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


DEFAULT_TRADES_PATH = REPO_ROOT / "backend" / "data" / "lab_fast13" / "calendar-m1-trades.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "backend" / "data" / "lab_fast13" / "fast76-calendar-pair-specific-audit"
SELECTION_END = pd.Timestamp("2024-01-01T00:00:00Z")
AUDIT_END = pd.Timestamp("2026-09-01T00:00:00Z")
DEFAULT_ALPHA = 0.05


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    arguments = parser.parse_args()
    report = run_fast76_calendar_pair_specific_audit(
        trades_path=arguments.trades_path,
        output_root=arguments.output_root,
        alpha=arguments.alpha,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_fast76_calendar_pair_specific_audit(
    *,
    trades_path: Path = DEFAULT_TRADES_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    validate_parameters(alpha=alpha)
    output_root.mkdir(parents=True, exist_ok=True)
    trades = load_trades(trades_path)
    pair_policy = evaluate_scope(
        trades,
        group_columns=("pair", "policy_id"),
        minimum_selection_trades=30,
        minimum_audit_trades=15,
        alpha=alpha,
    )
    pair_event_policy = evaluate_scope(
        trades,
        group_columns=("pair", "event_name", "policy_id"),
        minimum_selection_trades=15,
        minimum_audit_trades=8,
        alpha=alpha,
    )
    status, status_reasons = fast76_status(pair_policy=pair_policy, pair_event_policy=pair_event_policy)
    pair_policy_path = output_root / "pair_policy_candidates.json"
    pair_policy_path.write_text(json.dumps(pair_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pair_event_policy_path = output_root / "pair_event_policy_candidates.json"
    pair_event_policy_path.write_text(
        json.dumps(pair_event_policy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
            "fast76_is_post_fast13_diagnostic",
            "positive_mean_without_corrected_lower_bound_is_not_strategy",
            "future_preregistered_shadow_required_before_any_execution",
            "no_order_execution",
            "no_ledger_commit",
        ],
        "experiment_stage": "fast76_calendar_pair_specific_audit",
        "status": status,
        "status_reasons": status_reasons,
        "trades_path": str(trades_path),
        "trades_sha256": _sha256_file(trades_path),
        "trade_count": int(len(trades)),
        "selection_end": SELECTION_END.isoformat(),
        "audit_end": AUDIT_END.isoformat(),
        "alpha": alpha,
        "scopes": {
            "pair_policy": pair_policy["summary"],
            "pair_event_policy": pair_event_policy["summary"],
        },
        "pair_policy_candidates_path": str(pair_policy_path),
        "pair_policy_candidates_sha256": _sha256_file(pair_policy_path),
        "pair_event_policy_candidates_path": str(pair_event_policy_path),
        "pair_event_policy_candidates_sha256": _sha256_file(pair_event_policy_path),
        "top_pair_policy_weak_survivors": pair_policy["weak_survivors"][:10],
        "top_pair_event_policy_weak_survivors": pair_event_policy["weak_survivors"][:10],
    }
    return _write_report(output_root, report)


def validate_parameters(*, alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


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


def evaluate_scope(
    trades: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    minimum_selection_trades: int,
    minimum_audit_trades: int,
    alpha: float,
) -> dict[str, Any]:
    selection = trades.loc[trades["event_timestamp"] < SELECTION_END]
    audit = trades.loc[
        (trades["event_timestamp"] >= SELECTION_END) & (trades["event_timestamp"] < AUDIT_END)
    ]
    candidates = []
    for raw_key, selection_group in selection.groupby(list(group_columns), sort=False):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        audit_group = filter_group(audit, group_columns=group_columns, key=key)
        if len(selection_group) < minimum_selection_trades or len(audit_group) < minimum_audit_trades:
            continue
        candidate = {
            **dict(zip(group_columns, key, strict=True)),
            "selection": summarize_period(selection_group),
            "audit": summarize_period(audit_group),
            "_audit_values": {
                "base": audit_group["net_base_pips"].astype(float).tolist(),
                "stress": audit_group["net_stress_pips"].astype(float).tolist(),
            },
        }
        candidates.append(candidate)
    screen_candidates = [
        candidate
        for candidate in candidates
        if candidate["selection"]["mean_base_pips"] > 0
        and candidate["selection"]["mean_stress_pips"] > 0
    ]
    corrected_alpha = alpha / max(1, len(screen_candidates))
    enriched_screen = [
        enrich_audit_candidate(candidate, alpha=alpha, corrected_alpha=corrected_alpha)
        for candidate in screen_candidates
    ]
    weak_survivors = [
        candidate
        for candidate in enriched_screen
        if candidate["audit"]["mean_base_pips"] > 0 and candidate["audit"]["mean_stress_pips"] > 0
    ]
    weak_survivors.sort(
        key=lambda candidate: (
            candidate["audit"]["mean_stress_pips"],
            candidate["audit"]["trades"],
        ),
        reverse=True,
    )
    passed = [
        candidate
        for candidate in weak_survivors
        if candidate["audit"]["lower_stress_pips_bonferroni"] > 0
        and candidate["audit"]["lower_base_pips_bonferroni"] > 0
    ]
    passed.sort(
        key=lambda candidate: (
            candidate["audit"]["lower_stress_pips_bonferroni"],
            candidate["audit"]["mean_stress_pips"],
        ),
        reverse=True,
    )
    return {
        "summary": {
            "group_columns": list(group_columns),
            "eligible_group_count": len(candidates),
            "screen_candidate_count": len(screen_candidates),
            "weak_survivor_count": len(weak_survivors),
            "passed_candidate_count": len(passed),
            "minimum_selection_trades": minimum_selection_trades,
            "minimum_audit_trades": minimum_audit_trades,
            "alpha": alpha,
            "bonferroni_alpha": corrected_alpha,
        },
        "screen_candidates": enriched_screen,
        "weak_survivors": weak_survivors,
        "passed_candidates": passed,
    }


def filter_group(frame: pd.DataFrame, *, group_columns: tuple[str, ...], key: tuple[Any, ...]) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in zip(group_columns, key, strict=True):
        mask &= frame[column].eq(value).to_numpy()
    return frame.loc[mask]


def summarize_period(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "trades": int(len(frame)),
        "events": int(frame["event_timestamp"].nunique()),
        "mean_base_pips": float(frame["net_base_pips"].mean()),
        "mean_stress_pips": float(frame["net_stress_pips"].mean()),
        "total_stress_pips": float(frame["net_stress_pips"].sum()),
        "median_stress_pips": float(frame["net_stress_pips"].median()),
        "event_name_counts": {
            str(key): int(value) for key, value in frame["event_name"].value_counts().to_dict().items()
        },
    }


def enrich_audit_candidate(
    candidate: dict[str, Any],
    *,
    alpha: float,
    corrected_alpha: float,
) -> dict[str, Any]:
    audit_values = filter_audit_values(candidate)
    public_candidate = {key: value for key, value in candidate.items() if key != "_audit_values"}
    enriched = json.loads(json.dumps(public_candidate))
    enriched["audit"]["lower_base_pips_uncorrected"] = lower_confidence_bound(
        audit_values["base"],
        alpha=alpha,
    )
    enriched["audit"]["lower_stress_pips_uncorrected"] = lower_confidence_bound(
        audit_values["stress"],
        alpha=alpha,
    )
    enriched["audit"]["lower_base_pips_bonferroni"] = lower_confidence_bound(
        audit_values["base"],
        alpha=corrected_alpha,
    )
    enriched["audit"]["lower_stress_pips_bonferroni"] = lower_confidence_bound(
        audit_values["stress"],
        alpha=corrected_alpha,
    )
    return enriched


def filter_audit_values(candidate: dict[str, Any]) -> dict[str, list[float]]:
    raw = candidate.get("_audit_values")
    if raw is None:
        raise ValueError("candidate missing raw audit values")
    return raw


def lower_confidence_bound(values: list[float] | pd.Series | np.ndarray, *, alpha: float) -> float:
    series = pd.Series(values, dtype="float64").dropna()
    sample_size = len(series)
    if sample_size < 2:
        return float("nan")
    standard_error = float(series.std(ddof=1) / np.sqrt(sample_size))
    return float(series.mean() - t.ppf(1.0 - alpha, sample_size - 1) * standard_error)


def fast76_status(
    *,
    pair_policy: dict[str, Any],
    pair_event_policy: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons = ["not_approved_strategy", "research_only", "no_order_execution"]
    passed = (
        pair_policy["summary"]["passed_candidate_count"]
        + pair_event_policy["summary"]["passed_candidate_count"]
    )
    weak = (
        pair_policy["summary"]["weak_survivor_count"]
        + pair_event_policy["summary"]["weak_survivor_count"]
    )
    if passed:
        return "FAST76_CALENDAR_PAIR_SPECIFIC_RESEARCH_CANDIDATE_REQUIRES_SHADOW", reasons + [
            f"passed_candidates:{passed}",
            "future_preregistered_shadow_required",
        ]
    if weak:
        return "FAST76_CALENDAR_PAIR_SPECIFIC_WEAK_SURVIVORS_FAILED_SIGNIFICANCE", reasons + [
            f"weak_survivors:{weak}",
            "corrected_lower_bounds_non_positive",
        ]
    return "FAST76_CALENDAR_PAIR_SPECIFIC_FAILED_SCREEN", reasons + ["no_weak_survivor"]


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
        "top_pair_policy_weak_survivors": report["top_pair_policy_weak_survivors"][:5],
        "top_pair_event_policy_weak_survivors": report["top_pair_event_policy_weak_survivors"][:5],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
